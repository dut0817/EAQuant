#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import torch

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parents[1]
for import_root in (PROJECT_ROOT / "src", BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from datautils import get_loaders
from models.LMClass import LMClass
from quantize.faithfulness import (
    build_messages,
    build_model_input_text,
    build_option_target_text,
    build_recovery_prompt,
    get_option_labels,
    mask_rationale_span,
    normalize_answer_scoring_mode,
)
from quantize.omniquant import omniquant


class PrintLogger:
    def info(self, msg, *args):
        if args:
            msg = msg % args
        print(msg, flush=True)


def _safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def uses_option_text_mean_logprob(answer_scoring_mode: str) -> bool:
    return normalize_answer_scoring_mode(answer_scoring_mode) == "letter_option_mean_logprob"


def _softmax_map(score_map: Dict[str, float], labels: Sequence[str]) -> Dict[str, float]:
    score_tensor = torch.tensor(
        [float(score_map.get(label, float("-inf"))) for label in labels],
        dtype=torch.float32,
    )
    if not torch.any(torch.isfinite(score_tensor)):
        return {label: float("nan") for label in labels}
    probs = torch.softmax(score_tensor, dim=-1).tolist()
    return {label: float(prob) for label, prob in zip(labels, probs)}


def _compute_answer_margin(score_map: Dict[str, float], target_label: str) -> float:
    target = str(target_label).strip().upper()
    target_score = float(score_map.get(target, float("-inf")))
    if not math.isfinite(target_score):
        return float("-inf")
    other_scores = [
        float(score)
        for label, score in score_map.items()
        if str(label).strip().upper() != target and math.isfinite(float(score))
    ]
    if not other_scores:
        return target_score
    return target_score - max(other_scores)


def _positive_delta(reference: float, comparison: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(comparison):
        return 0.0
    return max(float(reference) - float(comparison), 0.0)


def _positive_gain(value: float, baseline: float) -> float:
    if not math.isfinite(value) or not math.isfinite(baseline):
        return 0.0
    return max(float(value) - float(baseline), 0.0)


@torch.no_grad()
def score_target_texts(
    model,
    tokenizer,
    user_prompts: Sequence[str],
    target_texts: Sequence[str],
    device: torch.device,
    batch_size: int = 1,
    length_normalize: bool = False,
) -> List[float]:
    if len(user_prompts) != len(target_texts):
        raise ValueError("user_prompts and target_texts must have the same length.")
    if not user_prompts:
        return []
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    scores: List[float] = []
    effective_batch_size = max(int(batch_size), 1)
    model.eval()
    for start_idx in range(0, len(user_prompts), effective_batch_size):
        chunk_user_prompts = user_prompts[start_idx : start_idx + effective_batch_size]
        chunk_target_texts = target_texts[start_idx : start_idx + effective_batch_size]
        prefix_texts = [
            build_model_input_text(build_messages(prompt), tokenizer)
            for prompt in chunk_user_prompts
        ]
        candidate_texts = [
            prefix_text + target_text
            for prefix_text, target_text in zip(prefix_texts, chunk_target_texts)
        ]
        batch = tokenizer(candidate_texts, return_tensors="pt", padding=True)
        prefix_id_list = [
            tokenizer(prefix_text, return_tensors="pt")["input_ids"][0]
            for prefix_text in prefix_texts
        ]
        moved_inputs = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**moved_inputs, use_cache=False)
        logits = outputs.logits
        input_ids = moved_inputs["input_ids"]
        attention_mask = moved_inputs["attention_mask"]

        for row_idx, prefix_ids in enumerate(prefix_id_list):
            candidate_ids = input_ids[row_idx]
            full_len = int(attention_mask[row_idx].sum().item())
            prefix_len = int(prefix_ids.shape[0])
            if not torch.equal(candidate_ids[:prefix_len].cpu(), prefix_ids):
                common_prefix = 0
                max_prefix = min(prefix_len, full_len)
                while (
                    common_prefix < max_prefix
                    and int(candidate_ids[common_prefix].item())
                    == int(prefix_ids[common_prefix].item())
                ):
                    common_prefix += 1
                prefix_len = common_prefix

            if prefix_len <= 0 or prefix_len >= full_len:
                scores.append(float("-inf"))
                continue
            suffix_ids = candidate_ids[prefix_len:full_len]
            suffix_logits = logits[row_idx, prefix_len - 1 : full_len - 1, :]
            suffix_log_probs = torch.log_softmax(suffix_logits.float(), dim=-1)
            token_scores = suffix_log_probs.gather(
                dim=-1,
                index=suffix_ids.unsqueeze(-1),
            ).squeeze(-1)
            row_score = token_scores.sum()
            if length_normalize:
                row_score = row_score / float(max(int(suffix_ids.numel()), 1))
            scores.append(float(row_score.item()))

        del outputs, logits, moved_inputs, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return scores


def score_option_distributions_for_prompts(
    model,
    tokenizer,
    example: Dict,
    user_prompts: Sequence[str],
    device: torch.device,
    batch_size: int = 1,
    answer_scoring_mode: str = "letter_option_mean_logprob",
) -> List[Tuple[Dict[str, float], Dict[str, float], str]]:
    labels = get_option_labels(example)
    target_texts: List[str] = []
    prompt_batch: List[str] = []
    for user_prompt in user_prompts:
        prompt_batch.extend([user_prompt] * len(labels))
        target_texts.extend(
            [
                build_option_target_text(
                    example=example,
                    label=label,
                    answer_scoring_mode=answer_scoring_mode,
                )
                for label in labels
            ]
        )
    score_values = score_target_texts(
        model=model,
        tokenizer=tokenizer,
        user_prompts=prompt_batch,
        target_texts=target_texts,
        device=device,
        batch_size=batch_size,
        length_normalize=uses_option_text_mean_logprob(answer_scoring_mode),
    )
    distributions = []
    option_count = len(labels)
    for prompt_idx in range(len(user_prompts)):
        start = prompt_idx * option_count
        end = start + option_count
        prompt_scores = score_values[start:end]
        score_map = {label: float(score) for label, score in zip(labels, prompt_scores)}
        prob_map = _softmax_map(score_map, labels)
        finite_scores = [float(score_map[label]) for label in labels]
        if any(math.isfinite(score) for score in finite_scores):
            pred_answer = labels[int(torch.tensor(finite_scores).argmax().item())]
        else:
            pred_answer = ""
        distributions.append((score_map, prob_map, pred_answer))
    return distributions


def _normalize_loss_weights(spans: Sequence[Dict]) -> List[float]:
    raw = [max(float(span.get("omni_quant_impact_score", 0.0)), 0.0) for span in spans]
    total = sum(raw)
    if total <= 0:
        raw = [max(float(span.get("teacher_importance", 0.0)), 0.0) for span in spans]
        total = sum(raw)
    if total <= 0:
        return [1.0 for _ in spans]
    scale = float(len(spans)) / total
    return [float(weight * scale) for weight in raw]


def _apply_char_span_weights(record: Dict, selected_spans: Sequence[Dict]) -> None:
    weight_by_index = {
        int(span.get("index", idx)): float(span.get("omni_loss_weight", 1.0))
        for idx, span in enumerate(selected_spans)
    }
    char_spans = []
    for idx, char_span in enumerate(record.get("faith_token_selected_char_spans", [])):
        char_span = dict(char_span)
        span_index = int(char_span.get("index", idx))
        char_span["loss_weight"] = weight_by_index.get(span_index, 1.0)
        char_spans.append(char_span)
    record["faith_token_selected_char_spans"] = char_spans


def _build_masked_and_keep_prompts(
    record: Dict,
    selected_spans: Sequence[Dict],
    answer_scoring_mode: str,
) -> Tuple[List[str], List[str]]:
    rationale = _safe_text(record.get("pred_explanation_stripped"))
    if not rationale:
        rationale = _safe_text(record.get("faith_token_target_text"))
    keep_prompts = record.get("faith_recovery_unit_user_prompts") or []
    if len(keep_prompts) < len(selected_spans):
        keep_prompts = [
            build_recovery_prompt(
                record,
                _safe_text(span.get("text", "")),
                answer_scoring_mode=answer_scoring_mode,
            )
            for span in selected_spans
        ]
    else:
        keep_prompts = list(keep_prompts[: len(selected_spans)])
    masked_prompts = [
        build_recovery_prompt(
            record,
            mask_rationale_span(rationale, span),
            answer_scoring_mode=answer_scoring_mode,
        )
        for span in selected_spans
    ]
    return masked_prompts, keep_prompts


def score_record_with_omniquant(
    model,
    tokenizer,
    record: Dict,
    device: torch.device,
    score_batch_size: int,
    answer_scoring_mode: str,
    keep_margin_gap_beta: float,
) -> Dict:
    output = dict(record)
    selected_spans = [dict(span) for span in record.get("faith_selected_spans", [])]
    if not selected_spans or not record.get("faith_recovery_selected_user_prompt"):
        output["omni_cache_usable"] = False
        output["omni_cache_failure_reason"] = "missing selected spans or recovery prompt"
        return output

    target_label = _safe_text(record.get("pred_answer")) or _safe_text(record.get("gold_answer"))
    target_label = target_label.upper()
    empty_prompt = build_recovery_prompt(
        record,
        "Relevant rationale omitted.",
        answer_scoring_mode=answer_scoring_mode,
    )
    masked_prompts, keep_prompts = _build_masked_and_keep_prompts(
        record,
        selected_spans,
        answer_scoring_mode=answer_scoring_mode,
    )
    all_prompts = [record["faith_recovery_selected_user_prompt"], empty_prompt]
    all_prompts.extend(masked_prompts)
    all_prompts.extend(keep_prompts)
    distributions = score_option_distributions_for_prompts(
        model=model,
        tokenizer=tokenizer,
        example=record,
        user_prompts=all_prompts,
        device=device,
        batch_size=score_batch_size,
        answer_scoring_mode=answer_scoring_mode,
    )
    if len(distributions) != len(all_prompts):
        output["omni_cache_usable"] = False
        output["omni_cache_failure_reason"] = "scoring returned an unexpected number of distributions"
        return output

    full_scores, full_probs, full_pred = distributions[0]
    empty_scores, empty_probs, empty_pred = distributions[1]
    masked_distributions = distributions[2 : 2 + len(selected_spans)]
    keep_distributions = distributions[2 + len(selected_spans) :]
    full_margin = _compute_answer_margin(full_scores, target_label)
    empty_margin = _compute_answer_margin(empty_scores, target_label)

    output.update(
        {
            "omni_cache_usable": True,
            "omni_cache_failure_reason": "",
            "omni_quant_full_option_scores": full_scores,
            "omni_quant_full_option_probs": full_probs,
            "omni_quant_full_pred_answer": full_pred,
            "omni_quant_full_margin": full_margin,
            "omni_quant_empty_option_scores": empty_scores,
            "omni_quant_empty_option_probs": empty_probs,
            "omni_quant_empty_pred_answer": empty_pred,
            "omni_quant_empty_margin": empty_margin,
        }
    )

    for span, masked_info, keep_info in zip(
        selected_spans,
        masked_distributions,
        keep_distributions,
    ):
        masked_scores, masked_probs, masked_pred = masked_info
        keep_scores, keep_probs, keep_pred = keep_info
        masked_margin = _compute_answer_margin(masked_scores, target_label)
        keep_margin = _compute_answer_margin(keep_scores, target_label)
        remove_margin_drop = _positive_delta(full_margin, masked_margin)
        keep_margin_gain = _positive_gain(keep_margin, empty_margin)
        omni_importance = remove_margin_drop + keep_margin_gain
        teacher_importance = max(float(span.get("teacher_importance", 0.0)), 0.0)
        importance_gap = max(teacher_importance - omni_importance, 0.0)
        teacher_keep_margin = float(span.get("teacher_keep_margin", float("nan")))
        keep_margin_gap = _positive_delta(teacher_keep_margin, keep_margin)
        keep_margin_gap_weighted = keep_margin_gap * max(float(keep_margin_gap_beta), 0.0)
        impact_score = max(importance_gap, keep_margin_gap_weighted)
        span.update(
            {
                "omni_quant_masked_option_scores": masked_scores,
                "omni_quant_masked_option_probs": masked_probs,
                "omni_quant_masked_pred_answer": masked_pred,
                "omni_quant_masked_margin": masked_margin,
                "omni_quant_keep_option_scores": keep_scores,
                "omni_quant_keep_option_probs": keep_probs,
                "omni_quant_keep_pred_answer": keep_pred,
                "omni_quant_keep_margin": keep_margin,
                "omni_quant_remove_margin_drop": remove_margin_drop,
                "omni_quant_keep_margin_gain": keep_margin_gain,
                "omni_quant_importance": omni_importance,
                "omni_quant_importance_gap": importance_gap,
                "omni_quant_keep_margin_gap": keep_margin_gap,
                "omni_quant_keep_margin_gap_beta": float(keep_margin_gap_beta),
                "omni_quant_keep_margin_gap_weighted": keep_margin_gap_weighted,
                "omni_quant_impact_score": impact_score,
                "omni_quant_impact_source": (
                    "importance_gap"
                    if importance_gap >= keep_margin_gap_weighted
                    else "keep_margin_gap"
                ),
                "omni_quant_affected_evidence": importance_gap > 0,
            }
        )

    for span, loss_weight in zip(selected_spans, _normalize_loss_weights(selected_spans)):
        span["omni_loss_weight"] = float(loss_weight)
        span["loss_weight"] = float(loss_weight)

    output["faith_selected_spans"] = selected_spans
    output["faith_q_evidence_units"] = selected_spans
    output["omni_faith_selected_span_count"] = len(selected_spans)
    _apply_char_span_weights(output, selected_spans)
    return output


def _read_records(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _make_quant_args(parsed) -> SimpleNamespace:
    args = SimpleNamespace(**vars(parsed))
    args.epochs = 0
    args.batch_size = max(int(parsed.batch_size), 1)
    args.output_dir = str(parsed.work_dir)
    args.save_dir = None
    args.real_quant = False
    args.multigpu = False
    args.eval_ppl = False
    args.tasks = ""
    args.num_fewshot = 0
    args.limit = -1
    args.deactive_amp = False
    if (args.wbits < 16 and args.wbits >= 8) or (args.abits < 16 and args.abits >= 8):
        args.deactive_amp = True
    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc": args.lwc,
        "disable_zero_point": args.disable_zero_point,
    }
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = dict(args.act_quant_params)
    args.k_quant_params = dict(args.act_quant_params)
    args.v_quant_params = dict(args.act_quant_params)
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    args.explanation_loss_enabled = False
    args.faithfulness_cache_path = ""
    return args


def load_fake_quantized_model(parsed):
    quant_args = _make_quant_args(parsed)
    lm = LMClass(quant_args)
    lm.seqlen = parsed.seqlen
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False
    dataloader, _ = get_loaders(
        parsed.calib_dataset,
        nsamples=max(int(parsed.nsamples), 1),
        seed=parsed.seed,
        model=parsed.model,
        seqlen=parsed.seqlen,
    )
    act_scales = torch.load(parsed.act_scales) if parsed.let else None
    act_shifts = torch.load(parsed.act_shifts) if parsed.let else None
    omniquant(
        lm=lm,
        args=quant_args,
        dataloader=dataloader,
        act_scales=act_scales,
        act_shifts=act_shifts,
        logger=PrintLogger(),
    )
    lm.model = lm.model.to(lm.device)
    lm.model.eval()
    return lm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--net", required=True)
    parser.add_argument("--resume", required=True, help="Path to OmniQuant omni_parameters.pth")
    parser.add_argument("--source_cache_path", required=True)
    parser.add_argument("--output_cache_path", required=True)
    parser.add_argument("--work_dir", default=str(BACKEND_ROOT / "cache" / "omni_effect_tmp"))
    parser.add_argument("--cache_dir", default=str(BACKEND_ROOT / "cache"))
    parser.add_argument("--calib_dataset", default="medmix")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--score_batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--answer_scoring_mode", default="letter_option_mean_logprob", choices=["single_letter", "letter_option_mean_logprob"])
    parser.add_argument("--keep_margin_gap_beta", type=float, default=0.01)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--let", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--lwc", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--aug_loss", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--symmetric", default=False, action="store_true")
    parser.add_argument("--disable_zero_point", default=False, action="store_true")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--act-scales", dest="act_scales", default=None)
    parser.add_argument("--act-shifts", dest="act_shifts", default=None)
    args = parser.parse_args()
    if args.act_scales is None:
        args.act_scales = str(BACKEND_ROOT / "act_scales" / "medmix_seq512" / f"{args.net}.pt")
    if args.act_shifts is None:
        args.act_shifts = str(BACKEND_ROOT / "act_shifts" / "medmix_seq512" / f"{args.net}.pt")
    return args


def main():
    args = parse_args()
    records = _read_records(Path(args.source_cache_path))
    if args.limit > 0:
        records = records[: args.limit]
    print(f"Loaded {len(records)} source cache records", flush=True)
    lm = load_fake_quantized_model(args)
    output_records = []
    processed = 0
    for idx, record in enumerate(records, start=1):
        should_score = bool(
            record.get("is_correct", False)
            and (record.get("faith_cache_usable") or record.get("faith_recovery_selected_success"))
            and record.get("faith_selected_spans")
            and record.get("faith_recovery_selected_user_prompt")
        )
        if should_score:
            record = score_record_with_omniquant(
                model=lm.model,
                tokenizer=lm.tokenizer,
                record=record,
                device=lm.device,
                score_batch_size=args.score_batch_size,
                answer_scoring_mode=args.answer_scoring_mode,
                keep_margin_gap_beta=args.keep_margin_gap_beta,
            )
            processed += 1
        output_records.append(record)
        if idx % 10 == 0:
            print(f"scored {processed} usable / {idx} total", flush=True)
    _write_jsonl(Path(args.output_cache_path), output_records)
    print(f"Wrote {len(output_records)} records to {args.output_cache_path}", flush=True)
    print(f"Scored {processed} usable records with OmniQuant fake-quant model", flush=True)


if __name__ == "__main__":
    main()
