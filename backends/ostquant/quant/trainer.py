import transformers, torch, os, datasets, random
import torch.nn.functional as F, torch, torch.nn as nn
import geoopt
from quant.cayley_opt import SGDG
from quant.ost_model_utils import SmoothModule, RotateModule
from transformers.utils import logging as hf_logging
from eaquant.losses import masked_token_kl, option_distribution_kl, row_target_log_scores

import torch.distributed.fsdp as fsdp

fsdp.FullyShardedDataParallel

logger = hf_logging.get_logger(__name__)


class MyTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latest_aux_train_metrics = None
        if (
            hasattr(self.accelerator.state, "fsdp_plugin")
            and self.accelerator.state.fsdp_plugin is not None
        ):
            model: nn.Module = self.model
            ignored_modules = list()
            for m in model.modules():
                if isinstance(m, (RotateModule, SmoothModule)):
                    ignored_modules.append(m)
            self.accelerator.state.fsdp_plugin.ignored_modules = ignored_modules
            self.accelerator.state.fsdp_plugin.use_orig_params = True

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = dict(inputs)
        faith_token = self._pop_branch_inputs(inputs, "faith_token")
        faith_recovery_selected = self._pop_branch_inputs(
            inputs,
            "faith_recovery_selected",
        )

        base_loss, outputs = self._compute_base_loss(model, inputs)
        total_loss = base_loss
        token_loss = None
        token_stats = self._empty_token_stats()
        recovery_selected_loss = None
        recovery_stats = self._empty_recovery_stats()

        if self.args.explanation_loss_enabled:
            if faith_token is not None:
                token_loss, token_stats = self._compute_masked_kl_loss(model, faith_token)
                total_loss = (
                    total_loss
                    + self.args.explanation_token_loss_weight * token_loss
                )
            if faith_recovery_selected is not None:
                recovery_selected_loss, recovery_stats = self._compute_option_set_kl_loss(
                    model,
                    faith_recovery_selected,
                )
                total_loss = (
                    total_loss
                    + self.args.explanation_recovery_loss_weight
                    * recovery_selected_loss
                )

        if model.training:
            self._latest_aux_train_metrics = {
                "base_loss": self._loss_to_float(base_loss),
                "faith_token_loss_raw": self._loss_to_float(token_loss),
                "faith_token_loss_weighted": self._loss_to_float(
                    None
                    if token_loss is None
                    else self.args.explanation_token_loss_weight * token_loss
                ),
                "faith_recovery_selected_loss_raw": self._loss_to_float(
                    recovery_selected_loss
                ),
                "faith_recovery_selected_loss_weighted": self._loss_to_float(
                    None
                    if recovery_selected_loss is None
                    else self.args.explanation_recovery_loss_weight
                    * recovery_selected_loss
                ),
                "num_selected_token_positions": float(
                    token_stats["num_selected_token_positions"]
                ),
                "num_recovery_groups": float(recovery_stats["num_recovery_groups"]),
                "num_skipped_recovery_groups": float(
                    recovery_stats["num_skipped_recovery_groups"]
                ),
            }

        if outputs is not None and hasattr(outputs, "loss"):
            outputs.loss = total_loss
        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs):
        logs = dict(logs)
        if "loss" in logs and self._latest_aux_train_metrics is not None:
            logs.update(self._latest_aux_train_metrics)
        return super().log(logs)

    def _loss_to_float(self, loss):
        if loss is None:
            return 0.0
        return float(loss.detach().float().item())

    def _empty_token_stats(self):
        return {
            "num_selected_token_positions": 0,
        }

    def _empty_recovery_stats(self):
        return {
            "num_recovery_groups": 0,
            "num_skipped_recovery_groups": 0,
        }

    def _distributed_max_int(self, value: int, device):
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            return int(value)
        value_tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
        torch.distributed.all_reduce(
            value_tensor,
            op=torch.distributed.ReduceOp.MAX,
        )
        return int(value_tensor.item())

    def _compute_base_loss(self, model, inputs):
        args = self.args
        loss_type = args.loss_type

        if loss_type == "origin":
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            return loss, outputs

        teacher_inputs = dict(inputs)
        teacher_inputs.pop("labels", None)
        ori_logits = self.get_ori_outputs(model, teacher_inputs).logits
        outputs = model(**inputs)
        logits = outputs.logits

        if loss_type == "rkl":
            loss = F.kl_div(
                F.log_softmax(ori_logits.flatten(0, -2), dim=-1),
                F.softmax(logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            return loss, outputs

        if loss_type == "kl":
            loss = F.kl_div(
                F.log_softmax(logits.flatten(0, -2), dim=-1),
                F.softmax(ori_logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            return loss, outputs

        if loss_type.startswith("r_kl_top"):
            k = 1000 if loss_type == "r_kl_top" else int(loss_type.split("_")[-1])
            top_logits, indices = logits.topk(k, dim=-1, sorted=False)
            top_ori_logits = ori_logits.gather(-1, indices)
            loss = F.kl_div(
                F.log_softmax(top_ori_logits.flatten(0, -2), dim=-1),
                F.softmax(top_logits.flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
            return loss, outputs

        if loss_type.startswith("kl_top"):
            k = 1000 if loss_type == "kl_top" else int(loss_type.split("_")[-1])
            top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)
            if args.post_attn:
                ref = F.softmax(ori_logits, dim=-1).gather(-1, indices).flatten(0, -2)
                can = F.log_softmax(logits, dim=-1).gather(-1, indices).flatten(0, -2)
                loss = F.kl_div(can, ref, reduction="batchmean")
            else:
                top_logits = logits.gather(-1, indices)
                loss = F.kl_div(
                    F.log_softmax(top_logits, dim=-1).flatten(0, -2),
                    F.softmax(top_ori_logits, dim=-1).flatten(0, -2),
                    reduction="batchmean",
                )
            return loss, outputs

        if loss_type == "mse":
            loss = F.mse_loss(logits, ori_logits)
            return loss, outputs

        if loss_type == "kd":
            temperature = getattr(self, "temperature", 1.0)
            alpha = getattr(self, "loss_alpha", 0.5)
            logits_flat = logits.view(-1, logits.size(-1))
            ori_logits_flat = ori_logits.view(-1, ori_logits.size(-1))
            distill_loss = F.kl_div(
                F.log_softmax(logits_flat / temperature, dim=-1),
                F.softmax(ori_logits_flat / temperature, dim=-1),
                reduction="batchmean",
            )
            student_ce_loss = outputs.get("loss", logits.sum() * 0.0)
            loss = student_ce_loss * (1 - alpha) + distill_loss * (
                alpha * temperature * temperature
            )
            return loss, outputs

        raise ValueError(f"Unsupported loss_type: {loss_type}")

    def _pop_branch_inputs(self, inputs, prefix: str):
        input_ids = inputs.pop(f"{prefix}_input_ids", None)
        attention_mask = inputs.pop(f"{prefix}_attention_mask", None)
        target_mask = inputs.pop(f"{prefix}_target_mask", None)
        token_weights = inputs.pop(f"{prefix}_token_weights", None)
        seq_weights = inputs.pop(f"{prefix}_seq_weights", None)
        group_sizes = inputs.pop(f"{prefix}_group_sizes", None)
        if input_ids is None or attention_mask is None or target_mask is None:
            return None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
            "token_weights": token_weights,
            "seq_weights": seq_weights,
            "group_sizes": group_sizes,
        }

    def _compute_masked_kl_loss(self, model, branch_inputs):
        model_inputs = {
            "input_ids": branch_inputs["input_ids"],
            "attention_mask": branch_inputs["attention_mask"],
        }
        target_mask = branch_inputs["target_mask"].to(model_inputs["input_ids"].device)
        token_weights = branch_inputs.get("token_weights", None)
        seq_weights = branch_inputs.get("seq_weights", None)

        ori_logits = self.get_ori_outputs(model, model_inputs).logits
        outputs = model(**model_inputs)
        logits = outputs.logits
        shift_mask = target_mask[..., 1:].bool()
        selected_token_positions = int(shift_mask.sum().item())
        stats = {
            "num_selected_token_positions": selected_token_positions,
        }
        return masked_token_kl(
            student_logits=logits,
            teacher_logits=ori_logits,
            target_mask=target_mask,
            token_weights=token_weights,
            sequence_weights=seq_weights,
        ), stats

    def _compute_option_set_kl_loss(self, model, branch_inputs):
        model_inputs = {
            "input_ids": branch_inputs["input_ids"],
            "attention_mask": branch_inputs["attention_mask"],
        }
        target_mask = branch_inputs["target_mask"].to(model_inputs["input_ids"].device)
        group_sizes = branch_inputs.get("group_sizes", None)
        if group_sizes is None:
            return model_inputs["input_ids"].sum() * 0.0, self._empty_recovery_stats()

        normalize_by_length = (
            getattr(self.args, "faithfulness_answer_scoring_mode", "single_letter")
            == "letter_option_mean_logprob"
        )

        sizes = [int(size) for size in group_sizes.tolist()]
        group_slices = []
        start_idx = 0
        for group_size in sizes:
            end_idx = start_idx + max(group_size, 0)
            if group_size > 0:
                group_slices.append((start_idx, end_idx))
            start_idx = end_idx
        max_group_count = self._distributed_max_int(
            len(group_slices),
            device=model_inputs["input_ids"].device,
        )
        total_loss = model_inputs["input_ids"].sum() * 0.0
        total_groups = 0
        skipped_groups = 0
        for group_idx in range(max_group_count):
            is_dummy_group = group_idx >= len(group_slices)
            if is_dummy_group:
                group_inputs = {
                    "input_ids": model_inputs["input_ids"][:1],
                    "attention_mask": model_inputs["attention_mask"][:1],
                }
                group_target_mask = torch.zeros_like(target_mask[:1])
            else:
                start_idx, end_idx = group_slices[group_idx]
                group_inputs = {
                    "input_ids": model_inputs["input_ids"][start_idx:end_idx],
                    "attention_mask": model_inputs["attention_mask"][start_idx:end_idx],
                }
                group_target_mask = target_mask[start_idx:end_idx]

            ori_logits = self.get_ori_outputs(model, group_inputs).logits
            outputs = model(**group_inputs)
            logits = outputs.logits
            zero_loss = logits.sum() * 0.0
            if is_dummy_group:
                total_loss = total_loss + zero_loss
                continue
            student_group = self._compute_row_target_log_scores(
                logits=logits,
                input_ids=group_inputs["input_ids"],
                target_mask=group_target_mask,
                normalize_by_length=normalize_by_length,
            )
            teacher_group = self._compute_row_target_log_scores(
                logits=ori_logits,
                input_ids=group_inputs["input_ids"],
                target_mask=group_target_mask,
                normalize_by_length=normalize_by_length,
            )

            if student_group.numel() == 0 or teacher_group.numel() == 0:
                total_loss = total_loss + zero_loss
                continue
            if not torch.all(torch.isfinite(student_group)) or not torch.all(
                torch.isfinite(teacher_group)
            ):
                skipped_groups += 1
                total_loss = total_loss + zero_loss
                continue

            # Treat each option set as one categorical distribution. Without the
            # extra batch axis, reduction="batchmean" divides by num_options.
            group_loss = option_distribution_kl(student_group, teacher_group)
            total_loss = total_loss + group_loss
            total_groups += 1

        if skipped_groups > 0:
            self._recovery_skipped_groups_total = (
                getattr(self, "_recovery_skipped_groups_total", 0) + skipped_groups
            )
            warned_count = getattr(self, "_recovery_skipped_groups_warn_count", 0)
            if warned_count < 5:
                logger.warning(
                    "Skipped %d recovery option groups because at least one option "
                    "score was non-finite, likely due to truncation removing all "
                    "target tokens for that option.",
                    skipped_groups,
                )
                self._recovery_skipped_groups_warn_count = warned_count + 1

        if total_groups <= 0:
            return logits.sum() * 0.0, {
                "num_recovery_groups": 0,
                "num_skipped_recovery_groups": skipped_groups,
            }
        return total_loss / float(total_groups), {
            "num_recovery_groups": total_groups,
            "num_skipped_recovery_groups": skipped_groups,
        }

    def _compute_row_target_log_scores(
        self,
        logits,
        input_ids,
        target_mask,
        normalize_by_length: bool = False,
    ):
        return row_target_log_scores(
            logits=logits,
            input_ids=input_ids,
            target_mask=target_mask,
            normalize_by_length=normalize_by_length,
        )

    @torch.no_grad()
    def get_ori_outputs(self, model, inputs):
        args = self.args
        inputs = dict(inputs)
        inputs.pop("labels", None)
        acc = self.accelerator

        
        def set_temporary(model, temporary=True):
            model.temporary = temporary
            model.model.temporary = temporary
            model.model.embed_tokens.temporary = temporary
            model.lm_head.temporary = temporary
            for layer in model.model.layers:
                layer.set_temporary(temporary)

        def set_quant_state(
            model,
            use_weight_quant: bool = False,
            use_act_quant: bool = False,
            use_fully_quant: bool = False,
        ):
            model.model.norm.use_act_quant = (
                use_fully_quant
            )
            model.model.embed_tokens.use_act_quant = (
                use_fully_quant
            )
            for layer in model.model.layers:
                layer.set_quant_state(use_weight_quant, use_act_quant, use_fully_quant)

        set_temporary(acc.unwrap_model(model), False)
        set_quant_state(acc.unwrap_model(model), False, False, False)
        outputs = model(**inputs, use_cache=False)
        set_temporary(acc.unwrap_model(model), True)
        set_quant_state(
            acc.unwrap_model(model), args.train_enable_wquant, True, args.fully_quant
        )
        return outputs

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        
        
        args = self.args
        params_rotate = []
        params_smooth = []
        for param in self.model.parameters():
            param: torch.nn.Parameter
            if param.requires_grad:
                if len(param.size()) == 2:
                    params_rotate.append(param)
                else:
                    params_smooth.append(param)
        dict_rotate = {
            "params": params_rotate,
            "lr": args.rotate_lr,
            "momentum": args.rotate_momentom,
            "stiefel": True,
            "grassmann": True,
            "omega": 0.1,
        }
        dict_smooth = {
            "params": params_smooth,
            "lr": args.smooth_lr,
            "momentum": args.smooth_momentom,
            "stiefel": False,
            "nesterov": False,
        }
        if args.opt_type == "SGDG":
            optimizer = SGDG(
                [dict_rotate, dict_smooth], weight_decay=0
            )  
        elif args.opt_type == "RSGD":
            optimizer = geoopt.optim.RiemannianSGD(
                [dict_rotate, dict_smooth], weight_decay=0, lr=args.rotate_lr,stabilize=10,
            )
        elif args.opt_type == "RAdam":
            optimizer = geoopt.optim.RiemannianAdam(
                [dict_rotate, dict_smooth], weight_decay=0, lr=args.rotate_lr,stabilize=10
            )
        self.optimizer = optimizer
        
        self.create_scheduler(
            num_training_steps=num_training_steps,
            optimizer=optimizer,
        )
