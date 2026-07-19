import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state
from quantize.faithfulness import (
    build_faithfulness_prompt_batches,
    load_faithfulness_training_records,
)
from eaquant.losses import masked_token_kl, option_set_kl, row_target_log_scores
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")



def is_llama_like(net, model=None):
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    if model_type:
        return model_type in {"llama", "mistral"}
    net = (net or "").lower()
    return "llama" in net or "openbiollm" in net or "mistral" in net


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     



def _faithfulness_enabled(args):
    return bool(
        getattr(args, "explanation_loss_enabled", False)
        and getattr(args, "faithfulness_cache_path", "")
    )


def _slice_optional_tensor(tensor, indices, dev):
    if tensor is None:
        return None
    if tensor.shape[0] == 1:
        return tensor.to(dev)
    return tensor.index_select(0, indices).to(dev)


def _tensor_to_device(tensor, dev):
    if tensor is None:
        return None
    return tensor.to(dev)


def _faithfulness_tail_device(layer_idx, start_layer, dev, args):
    base_dev = torch.device(dev)
    if not bool(getattr(args, "faithfulness_tail_multigpu", True)):
        return base_dev
    if base_dev.type != "cuda" or not torch.cuda.is_available():
        return base_dev
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return base_dev
    return torch.device(f"cuda:{(int(layer_idx) - int(start_layer)) % device_count}")


def _branch_indices(total, batch_size, step):
    start = (int(step) * int(batch_size)) % int(total)
    return torch.arange(start, start + int(batch_size), dtype=torch.long) % int(total)


def _masked_hidden_mse(fp_hidden, quant_hidden, target_mask, token_weights=None):
    valid_mask = target_mask.bool()
    if int(valid_mask.sum().item()) == 0:
        return quant_hidden.sum() * 0.0

    per_token_loss = (fp_hidden.float() - quant_hidden.float()).pow(2).mean(dim=-1)
    if token_weights is None:
        return per_token_loss[valid_mask].mean()

    weights = token_weights.to(per_token_loss.device, dtype=torch.float32).clamp_min(0.0)
    weights = weights * valid_mask.to(weights.dtype)
    total_weight = weights.sum()
    if float(total_weight.item()) <= 0:
        return per_token_loss[valid_mask].mean()
    return (per_token_loss * weights).sum() / total_weight


def _select_branch_batch(branch, indices, dev):
    batch = {
        "quant_inps": branch["quant_inps"].index_select(0, indices).to(dev),
        "fp_inps": branch["fp_inps"].index_select(0, indices).to(dev),
        "attention_mask": _slice_optional_tensor(branch.get("attention_mask"), indices, dev),
        "position_ids": _slice_optional_tensor(branch.get("position_ids"), indices, dev),
        "target_mask": branch["target_mask"].index_select(0, indices).to(dev),
        "token_weights": branch["token_weights"].index_select(0, indices).to(dev),
    }
    if branch.get("input_ids") is not None:
        batch["input_ids"] = branch["input_ids"].index_select(0, indices).to(dev)
    return batch


def _group_slices(group_sizes):
    if group_sizes is None:
        return []
    sizes = [int(size) for size in group_sizes.tolist()]
    slices = []
    start = 0
    for size in sizes:
        end = start + max(size, 0)
        if size > 0:
            slices.append((start, end))
        start = end
    return slices


def _select_branch_group_batch(branch, args, step, dev):
    group_slices = _group_slices(branch.get("group_sizes"))
    if not group_slices:
        return None
    group_batch_size = min(max(int(args.faithfulness_batch_size), 1), len(group_slices))
    first_group = (int(step) * group_batch_size) % len(group_slices)
    selected_group_ids = [
        (first_group + offset) % len(group_slices)
        for offset in range(group_batch_size)
    ]
    row_ranges = []
    selected_sizes = []
    for group_id in selected_group_ids:
        start, end = group_slices[group_id]
        if end <= start:
            continue
        row_ranges.append(torch.arange(start, end, dtype=torch.long))
        selected_sizes.append(end - start)
    if not row_ranges:
        return None
    indices = torch.cat(row_ranges, dim=0)
    batch = _select_branch_batch(branch, indices, dev)
    batch["group_sizes"] = torch.tensor(selected_sizes, dtype=torch.long, device=dev)
    return batch


def _move_output_modules_to_device(model, args, dev):
    net = (getattr(args, "net", "") or "").lower()
    if is_llama_like(net, model) or "mixtral" in net:
        model.model.norm = model.model.norm.to(dev)
        model.lm_head = model.lm_head.to(dev)
        model.model.norm.requires_grad_(False)
        model.lm_head.requires_grad_(False)
    elif "opt" in net:
        decoder = model.model.decoder
        if getattr(decoder, "final_layer_norm", None) is not None:
            decoder.final_layer_norm = decoder.final_layer_norm.to(dev)
            decoder.final_layer_norm.requires_grad_(False)
        if getattr(decoder, "project_out", None) is not None:
            decoder.project_out = decoder.project_out.to(dev)
            decoder.project_out.requires_grad_(False)
        model.lm_head = model.lm_head.to(dev)
        model.lm_head.requires_grad_(False)
    elif "falcon" in net:
        model.transformer.ln_f = model.transformer.ln_f.to(dev)
        model.lm_head = model.lm_head.to(dev)
        model.transformer.ln_f.requires_grad_(False)
        model.lm_head.requires_grad_(False)
    else:
        raise ValueError(f"Unsupported net for logit faithfulness loss: {args.net}")


def _apply_output_head(model, args, hidden_states):
    net = (getattr(args, "net", "") or "").lower()
    if is_llama_like(net, model) or "mixtral" in net:
        hidden_states = model.model.norm(hidden_states)
    elif "opt" in net:
        decoder = model.model.decoder
        if getattr(decoder, "final_layer_norm", None) is not None:
            hidden_states = decoder.final_layer_norm(hidden_states)
        if getattr(decoder, "project_out", None) is not None:
            hidden_states = decoder.project_out(hidden_states)
    elif "falcon" in net:
        hidden_states = model.transformer.ln_f(hidden_states)
    else:
        raise ValueError(f"Unsupported net for logit faithfulness loss: {args.net}")
    return model.lm_head(hidden_states)


def _tail_logits_from_hidden(model, layers, start_layer, hidden_states, attention_mask, position_ids, is_llama, args, dev):
    output_dev = torch.device(dev)
    if start_layer < len(layers):
        output_dev = _faithfulness_tail_device(len(layers) - 1, start_layer, dev, args)
    _move_output_modules_to_device(model, args, output_dev)
    net = (getattr(args, "net", "") or "").lower()
    use_checkpoint = bool(
        getattr(args, "faithfulness_tail_checkpoint", True)
        and torch.is_grad_enabled()
        and hidden_states.requires_grad
    )
    for layer_idx in range(start_layer, len(layers)):
        layer_dev = _faithfulness_tail_device(layer_idx, start_layer, dev, args)
        if hidden_states.device != layer_dev:
            hidden_states = hidden_states.to(layer_dev)
        layer_attention_mask = _tensor_to_device(attention_mask, layer_dev)
        layer_position_ids = _tensor_to_device(position_ids, layer_dev)
        layer = layers[layer_idx].to(layer_dev)
        layer.requires_grad_(False)

        def layer_forward(
            layer_hidden_states,
            layer=layer,
            layer_attention_mask=layer_attention_mask,
            layer_position_ids=layer_position_ids,
            is_llama=is_llama,
            net=net,
        ):
            if is_llama:
                return layer(
                    layer_hidden_states,
                    attention_mask=layer_attention_mask,
                    position_ids=layer_position_ids,
                    use_cache=False,
                )[0]
            if "falcon" in net:
                return layer(
                    layer_hidden_states,
                    attention_mask=layer_attention_mask,
                    use_cache=False,
                )[0]
            return layer(layer_hidden_states, attention_mask=layer_attention_mask)[0]

        if use_checkpoint:
            hidden_states = checkpoint(layer_forward, hidden_states, use_reentrant=False)
        else:
            hidden_states = layer_forward(hidden_states)
    if hidden_states.device != output_dev:
        hidden_states = hidden_states.to(output_dev)
    return _apply_output_head(model, args, hidden_states)


def _masked_logit_kl(student_logits, teacher_logits, target_mask, token_weights=None):
    return masked_token_kl(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        target_mask=target_mask,
        token_weights=token_weights,
    )


def _row_target_log_scores(logits, input_ids, target_mask, normalize_by_length=False):
    return row_target_log_scores(
        logits=logits,
        input_ids=input_ids,
        target_mask=target_mask,
        normalize_by_length=normalize_by_length,
    )


def _option_set_kl(student_logits, teacher_logits, input_ids, target_mask, group_sizes, args):
    normalize_by_length = (
        getattr(args, "faithfulness_answer_scoring_mode", "single_letter")
        == "letter_option_mean_logprob"
    )
    return option_set_kl(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        input_ids=input_ids,
        target_mask=target_mask,
        group_sizes=group_sizes,
        normalize_by_length=normalize_by_length,
    )


def _compute_faithfulness_hidden_mse_loss(qlayer, branch, args, step, dev):
    total = int(branch["quant_inps"].shape[0])
    if total <= 0:
        return None
    batch_size = min(max(int(args.faithfulness_batch_size), 1), total)
    indices = _branch_indices(total, batch_size, step)
    batch = _select_branch_batch(branch, indices, dev)
    quant_out = qlayer(
        batch["quant_inps"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
    )[0]
    return _masked_hidden_mse(
        fp_hidden=batch["fp_inps"],
        quant_hidden=quant_out,
        target_mask=batch["target_mask"],
        token_weights=batch["token_weights"],
    )


def _compute_faithfulness_token_loss(model, layers, layer_idx, qlayer, branch, args, step, dev, is_llama):
    if getattr(args, "faithfulness_loss_type", "logit_kl") == "hidden_mse":
        return _compute_faithfulness_hidden_mse_loss(qlayer, branch, args, step, dev)
    total = int(branch["quant_inps"].shape[0])
    if total <= 0:
        return None
    batch_size = min(max(int(args.faithfulness_batch_size), 1), total)
    indices = _branch_indices(total, batch_size, step)
    batch = _select_branch_batch(branch, indices, dev)
    quant_out = qlayer(
        batch["quant_inps"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
    )[0]
    with torch.no_grad():
        teacher_logits = _tail_logits_from_hidden(
            model=model,
            layers=layers,
            start_layer=layer_idx + 1,
            hidden_states=batch["fp_inps"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            is_llama=is_llama,
            args=args,
            dev=dev,
        )
    student_logits = _tail_logits_from_hidden(
        model=model,
        layers=layers,
        start_layer=layer_idx + 1,
        hidden_states=quant_out,
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        is_llama=is_llama,
        args=args,
        dev=dev,
    )
    loss = _masked_logit_kl(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        target_mask=batch["target_mask"],
        token_weights=batch["token_weights"],
    )
    return loss.to(dev)


def _compute_faithfulness_recovery_loss(model, layers, layer_idx, qlayer, branch, args, step, dev, is_llama):
    if getattr(args, "faithfulness_loss_type", "logit_kl") == "hidden_mse":
        return _compute_faithfulness_hidden_mse_loss(qlayer, branch, args, step, dev)
    batch = _select_branch_group_batch(branch, args, step, dev)
    if batch is None:
        return None
    quant_out = qlayer(
        batch["quant_inps"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
    )[0]
    with torch.no_grad():
        teacher_logits = _tail_logits_from_hidden(
            model=model,
            layers=layers,
            start_layer=layer_idx + 1,
            hidden_states=batch["fp_inps"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            is_llama=is_llama,
            args=args,
            dev=dev,
        )
    student_logits = _tail_logits_from_hidden(
        model=model,
        layers=layers,
        start_layer=layer_idx + 1,
        hidden_states=quant_out,
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        is_llama=is_llama,
        args=args,
        dev=dev,
    )
    loss = _option_set_kl(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        input_ids=batch["input_ids"],
        target_mask=batch["target_mask"],
        group_sizes=batch["group_sizes"],
        args=args,
    )
    return loss.to(dev)


def _run_branch_layer(qlayer, branch, input_key, output_key, batch_size, dev, traincast):
    total = int(branch[input_key].shape[0])
    outputs = []
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            hidden = branch[input_key][start:end].to(dev)
            attention_mask = branch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask[start:end].to(dev)
            position_ids = branch.get("position_ids")
            if position_ids is not None:
                if position_ids.shape[0] == 1:
                    position_ids = position_ids.to(dev)
                else:
                    position_ids = position_ids[start:end].to(dev)
            with traincast():
                out = qlayer(
                    hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
            outputs.append(out.detach().cpu())
    branch[output_key] = torch.cat(outputs, dim=0)


def _update_faithfulness_fp_inputs(qlayer, faith_branches, args, dev, traincast):
    if not faith_branches:
        return
    batch_size = max(int(args.faithfulness_batch_size), 1)
    for branch in faith_branches.values():
        _run_branch_layer(
            qlayer=qlayer,
            branch=branch,
            input_key="fp_inps",
            output_key="fp_inps",
            batch_size=batch_size,
            dev=dev,
            traincast=traincast,
        )


def _update_faithfulness_quant_inputs(qlayer, faith_branches, args, dev, traincast):
    if not faith_branches:
        return
    batch_size = max(int(args.faithfulness_batch_size), 1)
    for branch in faith_branches.values():
        _run_branch_layer(
            qlayer=qlayer,
            branch=branch,
            input_key="quant_inps",
            output_key="quant_inps",
            batch_size=batch_size,
            dev=dev,
            traincast=traincast,
        )


def _capture_faithfulness_first_layer_inputs(
    model,
    layers,
    prompt_batches,
    dev,
    is_llama,
    batch_size,
    logger,
):
    if not prompt_batches:
        return {}

    capture = {}

    class FaithCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            capture["hidden"] = inp.detach().cpu()
            attention_mask = kwargs.get("attention_mask", None)
            capture["attention_mask"] = (
                attention_mask.detach().cpu() if attention_mask is not None else None
            )
            position_ids = kwargs.get("position_ids", None)
            capture["position_ids"] = (
                position_ids.detach().cpu()
                if is_llama and position_ids is not None
                else None
            )
            raise ValueError

    original_layer0 = layers[0]
    layers[0] = FaithCatcher(original_layer0)
    faith_branches = {}
    try:
        with torch.no_grad():
            for branch_name, batch in prompt_batches.items():
                hidden_chunks = []
                attention_chunks = []
                position_chunks = []
                total = int(batch["input_ids"].shape[0])
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    capture.clear()
                    input_ids = batch["input_ids"][start:end].to(dev)
                    attention_mask = batch["attention_mask_2d"][start:end].to(dev)
                    try:
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    except ValueError:
                        pass
                    if "hidden" not in capture:
                        raise RuntimeError(
                            f"Failed to capture first-layer inputs for {branch_name}."
                        )
                    hidden_chunks.append(capture["hidden"])
                    if capture.get("attention_mask") is not None:
                        attention_chunks.append(capture["attention_mask"])
                    if capture.get("position_ids") is not None:
                        position_chunks.append(capture["position_ids"])

                position_ids = None
                if position_chunks:
                    if all(chunk.shape[0] == 1 for chunk in position_chunks):
                        position_ids = position_chunks[0]
                    else:
                        position_ids = torch.cat(position_chunks, dim=0)

                quant_inps = torch.cat(hidden_chunks, dim=0)
                branch_record = {
                    "quant_inps": quant_inps,
                    "fp_inps": quant_inps.clone(),
                    "attention_mask": (
                        torch.cat(attention_chunks, dim=0)
                        if attention_chunks
                        else None
                    ),
                    "position_ids": position_ids,
                    "target_mask": batch["target_mask"].cpu(),
                    "token_weights": batch["token_weights"].cpu(),
                    "input_ids": batch["input_ids"].cpu(),
                }
                if batch.get("group_sizes") is not None:
                    branch_record["group_sizes"] = batch["group_sizes"].cpu()
                faith_branches[branch_name] = branch_record
                selected_positions = int(batch["target_mask"].sum().item())
                logger.info(
                    "faithfulness branch %s: %d rows, seq_len %d, selected positions %d",
                    branch_name,
                    total,
                    int(batch["input_ids"].shape[1]),
                    selected_positions,
                )
    finally:
        layers[0] = layers[0].module

    return faith_branches

def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if is_llama_like(args.net, model):
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/mistral/openbiollm/falcon/mixtral now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    faith_branches = {}
    if _faithfulness_enabled(args):
        faith_records = load_faithfulness_training_records(
            args.faithfulness_cache_path,
            limit=None if args.faithfulness_cache_limit <= 0 else args.faithfulness_cache_limit,
            weight_source=args.faithfulness_weight_source,
        )
        logger.info(
            f"Loaded {len(faith_records)} faithfulness records from {args.faithfulness_cache_path}"
        )
        faith_max_length = int(args.faithfulness_max_seq_len or lm.seqlen)
        prompt_batches = build_faithfulness_prompt_batches(
            records=faith_records,
            tokenizer=lm.tokenizer,
            max_length=faith_max_length,
            answer_scoring_mode=args.faithfulness_answer_scoring_mode,
            use_recovery_unit_prompts=args.faithfulness_use_recovery_unit_prompts,
        )
        faith_branches = _capture_faithfulness_first_layer_inputs(
            model=model,
            layers=layers,
            prompt_batches=prompt_batches,
            dev=dev,
            is_llama=is_llama,
            batch_size=max(int(args.faithfulness_batch_size), 1),
            logger=logger,
        )
        del prompt_batches
        del faith_records
    layers[0] = layers[0].cpu()
    if is_llama_like(args.net, model) or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/mistral/openbiollm/falcon/mixtral now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None



    if args.resume:
        omni_parameters = torch.load(args.resume, map_location="cpu")
        if not isinstance(omni_parameters, dict):
            raise ValueError(f"Invalid OmniQuant checkpoint format: {args.resume}")
        missing_layers = [idx for idx in range(len(layers)) if idx not in omni_parameters]
        if missing_layers:
            raise ValueError(
                f"Incomplete OmniQuant checkpoint {args.resume}; "
                f"missing layer entries: {missing_layers}"
            )
    else:
        omni_parameters = {}

    
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                _update_faithfulness_fp_inputs(
                    qlayer=qlayer,
                    faith_branches=faith_branches,
                    args=args,
                    dev=dev,
                    traincast=traincast,
                )
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            saved_layer_state = omni_parameters[i]
            expected_omni_keys = set(omni_state_dict(qlayer).keys())
            saved_omni_keys = set(saved_layer_state.keys())
            missing_omni_keys = sorted(expected_omni_keys - saved_omni_keys)
            unexpected_omni_keys = sorted(saved_omni_keys - expected_omni_keys)
            if missing_omni_keys or unexpected_omni_keys:
                raise ValueError(
                    f"Incompatible OmniQuant checkpoint at layer {i}: "
                    f"missing={missing_omni_keys}, unexpected={unexpected_omni_keys}"
                )
            qlayer.load_state_dict(saved_layer_state, strict=False)
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                faith_token_loss_list = []
                faith_recovery_loss_list = []
                steps_per_epoch = args.nsamples // args.batch_size
                for j in range(steps_per_epoch):
                    index = j * args.batch_size
                    global_step = epochs * steps_per_epoch + j
                    # obtain output of quantization model
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama)
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                        if faith_branches and global_step % max(int(args.faithfulness_every_n_steps), 1) == 0:
                            token_branch = faith_branches.get("token")
                            if token_branch is not None and args.explanation_token_loss_weight != 0:
                                token_loss = _compute_faithfulness_token_loss(
                                    model=model,
                                    layers=layers,
                                    layer_idx=i,
                                    qlayer=qlayer,
                                    branch=token_branch,
                                    args=args,
                                    step=global_step,
                                    dev=dev,
                                    is_llama=is_llama,
                                )
                                if token_loss is not None:
                                    loss = loss + args.explanation_token_loss_weight * token_loss
                                    faith_token_loss_list.append(token_loss.detach().cpu())
                            recovery_branch = faith_branches.get("recovery")
                            if recovery_branch is not None and args.explanation_recovery_loss_weight != 0:
                                recovery_loss = _compute_faithfulness_recovery_loss(
                                    model=model,
                                    layers=layers,
                                    layer_idx=i,
                                    qlayer=qlayer,
                                    branch=recovery_branch,
                                    args=args,
                                    step=global_step,
                                    dev=dev,
                                    is_llama=is_llama,
                                )
                                if recovery_loss is not None:
                                    loss = loss + args.explanation_recovery_loss_weight * recovery_loss
                                    faith_recovery_loss_list.append(recovery_loss.detach().cpu())
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                faith_msg = ""
                if faith_token_loss_list:
                    faith_msg += f" faith_token:{torch.stack(faith_token_loss_list).mean()}"
                if faith_recovery_loss_list:
                    faith_msg += f" faith_recovery:{torch.stack(faith_recovery_loss_list).mean()}"
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean}{faith_msg} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            clear_temp_variable(qlayer)
            del optimizer
        qlayer.half() 
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                _update_faithfulness_quant_inputs(
                    qlayer=qlayer,
                    faith_branches=faith_branches,
                    args=args,
                    dev=dev,
                    traincast=traincast,
                )
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            checkpoint_path = os.path.join(args.output_dir, "omni_parameters.pth")
            checkpoint_tmp_path = f"{checkpoint_path}.tmp"
            torch.save(omni_parameters, checkpoint_tmp_path)
            os.replace(checkpoint_tmp_path, checkpoint_path)
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    del faith_branches
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model
