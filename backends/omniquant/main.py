import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.omniquant import omniquant
from tqdm import tqdm
import utils
from pathlib import Path

from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear

import pdb


torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "Meta-Llama-3-8B",
    "Meta-Llama-3-8B-Instruct",
    "Llama3-OpenBioLLM-8B",
    "llama3-8b-instruct",
    "llama3_8b_instruct",
    "openbiollm_8b",
    "openbiollm-8b",
    "Mistral-7B-Instruct-v0.3",
    "BioMistral-7B",
    "mistral_7b_instruct",
    "biomistral_7b",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b"
]


def is_llama_like(net, model=None):
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    if model_type:
        return model_type in {"llama", "mistral"}
    net = (net or "").lower()
    return "llama" in net or "openbiollm" in net or "mistral" in net


@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}
    if args.multigpu:
        if "opt" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif is_llama_like(args.net, lm.model) or "mixtral" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
        elif "falcon" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.transformer.h)
            input_device = lm.model.transformer.h[0].device
            output_device = lm.model.transformer.h[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.transformer.word_embeddings.to(input_device)
            lm.model.transformer.ln_f.to(output_device)
            lm.model.lm_head.to(output_device)
    else:
        if "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif is_llama_like(args.net, lm.model) or "mixtral" in args.net.lower():
            lm.model = lm.model.to(lm.device)
        elif "falcon" in args.net.lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)


    if args.eval_ppl:
        # for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
        for dataset in ["wikitext2", "c4"]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all_seed{args.seed}_seq{lm.seqlen}.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(lm.device)
                if "opt" in args.net.lower():
                    outputs = lm.model.model.decoder(batch)
                elif is_llama_like(args.net, lm.model) or "mixtral" in args.net.lower():
                    outputs = lm.model.model(batch)
                elif "falcon" in args.model:
                    outputs = lm.model.transformer(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                    :, 1:
                ].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--real_quant", default=False, action="store_true", help="real quantization, which can see memory reduce. Note that due to the limitations of AutoGPTQ kernels, the real quantization of weight-only quantization can only lead memory reduction, but with slower inference speed.")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix", "pile", "medmix"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length used for calibration and PPL evaluation.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--let",default=False, action="store_true",help="activate learnable equivalent transformation")
    parser.add_argument("--lwc",default=False, action="store_true",help="activate learnable weight clipping")
    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--disable_zero_point",default=False, action="store_true", help="quantization without zero_point")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--explanation_loss_enabled", default=False, action="store_true", help="Add explanation-aware faithfulness losses during OmniQuant layer optimization.")
    parser.add_argument("--faithfulness_cache_path", type=str, default="", help="Path to a Qwen/faithfulness cache jsonl file.")
    parser.add_argument("--faithfulness_cache_limit", type=int, default=128, help="Maximum usable faithfulness records to load; <=0 means all.")
    parser.add_argument("--faithfulness_max_seq_len", type=int, default=512, help="Max tokenized length for faithfulness prompts.")
    parser.add_argument("--faithfulness_batch_size", type=int, default=1, help="Mini-batch size for faithfulness branch forwards.")
    parser.add_argument("--faithfulness_every_n_steps", type=int, default=1, help="Apply faithfulness losses every N OmniQuant optimization steps.")
    parser.add_argument("--faithfulness_loss_type", type=str, default="logit_kl", choices=["logit_kl", "hidden_mse"], help="Faithfulness auxiliary objective. logit_kl matches the OSTQuant token/recovery KL objective; hidden_mse keeps the older OmniQuant layer-reconstruction version.")
    parser.add_argument("--faithfulness_answer_scoring_mode", type=str, default="letter_option_mean_logprob", choices=["single_letter", "letter_option_mean_logprob"])
    parser.add_argument("--faithfulness_weight_source", type=str, default="teacher", choices=["teacher", "omni_quant", "uniform", "cache_quant"], help="Which cache fields provide selected-evidence loss weights. cache_quant is the old cache quant fields; omni_quant expects newly rebuilt OmniQuant fields.")
    parser.add_argument("--faithfulness_use_recovery_unit_prompts", default=False, action="store_true", help="Use per-evidence recovery prompts instead of only the selected combined recovery prompt.")
    parser.add_argument("--explanation_token_loss_weight", type=float, default=1.0)
    parser.add_argument("--explanation_recovery_loss_weight", type=float, default=1.0)
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)

    args = parser.parse_args()
    if args.explanation_loss_enabled and not args.faithfulness_cache_path:
        raise ValueError("--explanation_loss_enabled requires --faithfulness_cache_path.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # check
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.replace("/", "_")
    lm = LMClass(args)
    lm.seqlen = args.seqlen
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "disable_zero_point": args.disable_zero_point
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'

    # quantization
    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}_seed{args.seed}_seq{lm.seqlen}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)    
        act_scales = None
        act_shifts = None
        if args.let:
            act_scales = torch.load(args.act_scales)
            act_shifts = torch.load(args.act_shifts)
        omniquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(time.time() - tick)
    if args.save_dir:
        # delete omni parameters
        for name, module in lm.model.named_modules():
            if isinstance(module, QuantLinear):
                del module.weight_quantizer.lowbound_factor
                del module.weight_quantizer.upbound_factor
            if isinstance(module,QuantLlamaDecoderLayer) or isinstance(module,QuantOPTDecoderLayer):
                if args.let:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.out_smooth_scale
                    del module.out_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift           
        lm.model.save_pretrained(args.save_dir)  
        lm.tokenizer.save_pretrained(args.save_dir) 
    evaluate(lm, args,logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
