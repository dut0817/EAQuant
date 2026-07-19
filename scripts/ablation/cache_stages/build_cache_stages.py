#!/usr/bin/env python3
"""Build Qwen-span component ablation caches for faithfulness training.

The input caches already contain Qwen evidence spans plus fp16-teacher and
quantized-model diagnostics. This script rewrites only the training fields used
by FaithfulnessDataCollator so each ablation variant can be run by changing
FAITH_CACHE_PATH.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]


DEFAULT_MODELS = (
    "llama3_8b_instruct",
    "mistral_7b_instruct",
    "openbiollm_8b",
    "biomistral_7b",
)

VARIANTS = (
    "v1_qwen_all",
    "v2_teacher_faithful",
    "v3_teacher_faithful_qdamage",
    "v4_teacher_faithful_qdamage_keepmargin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ablation cache variants under ablation/cache."
    )
    parser.add_argument(
        "--repo_dir",
        type=Path,
        default=PROJECT_ROOT,
        help="EAQuant repository root.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "cache" / "ablation" / "cache_stages",
        help="Where ablation caches are written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model tags to process.",
    )
    parser.add_argument(
        "--source_tag",
        default="imp0p02_q0p02_beta0p01",
        help="Qwen evidence source tag.",
    )
    parser.add_argument(
        "--min_importance",
        type=float,
        default=0.0,
        help="Minimum fp16 teacher importance for teacher-faithful variants.",
    )
    parser.add_argument(
        "--min_quant_impact",
        type=float,
        default=0.0,
        help="Minimum quant impact/gap for quant-damage variants.",
    )
    parser.add_argument(
        "--keep_margin_beta",
        type=float,
        default=0.01,
        help="Multiplier for keep-margin gap in V4.",
    )
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_example_block(example: Dict[str, Any]) -> str:
    lines = [f"Question:\n{_safe_text(example.get('question'))}"]
    for idx, context in enumerate(example.get("contexts", []) or [], start=1):
        context = _safe_text(context)
        if context:
            lines.append(f"Context {idx}:\n{context}")
    lines.append("Options:")
    for label, text in (example.get("options") or {}).items():
        lines.append(f"{label}. {_safe_text(text)}")
    return "\n\n".join(lines)


def _build_explanation_training_prompt(
    example: Dict[str, Any],
    pred_answer: str,
    pred_answer_text: str,
) -> str:
    return (
        "Answer the medical multiple-choice question.\n"
        "The final answer has already been selected.\n"
        "Write a concise medical rationale that supports this selected answer.\n\n"
        "Output exactly in this format:\n"
        "Rationale: <2-3 concise medical sentences explaining the key evidence>\n\n"
        f"{_format_example_block(example)}\n\n"
        f"Selected answer:\n{pred_answer}. {_safe_text(pred_answer_text)}\n"
        "Rationale:"
    )


def _build_recovery_prompt(
    example: Dict[str, Any],
    rationale_text: str,
) -> str:
    rationale_text = _safe_text(rationale_text) or "Relevant rationale omitted."
    return (
        "Answer the medical multiple-choice question.\n"
        "Use the provided rationale to infer the selected answer.\n"
        "Return the option label followed by the option text.\n\n"
        f"{_format_example_block(example)}\n\n"
        f"Rationale:\n{rationale_text}\n\n"
        "Final:"
    )


def _teacher_importance(unit: Dict[str, Any]) -> float:
    return max(
        _safe_float(unit.get("teacher_importance")),
        _safe_float(unit.get("teacher_comprehensiveness")),
        _safe_float(unit.get("teacher_sufficiency")),
    )


def _quant_importance_gap(unit: Dict[str, Any]) -> float:
    if unit.get("quant_importance_gap") is not None:
        return max(_safe_float(unit.get("quant_importance_gap")), 0.0)
    if unit.get("quant_importance_gap_raw") is not None:
        return max(_safe_float(unit.get("quant_importance_gap_raw")), 0.0)
    return max(_teacher_importance(unit) - _safe_float(unit.get("quant_importance")), 0.0)


def _keep_margin_gap(unit: Dict[str, Any]) -> float:
    if unit.get("quant_keep_margin_gap_pos") is not None:
        return max(_safe_float(unit.get("quant_keep_margin_gap_pos")), 0.0)
    if unit.get("quant_keep_margin_gap_raw") is not None:
        return max(_safe_float(unit.get("quant_keep_margin_gap_raw")), 0.0)
    teacher_margin = _safe_float(unit.get("teacher_keep_margin"), float("nan"))
    quant_margin = _safe_float(unit.get("quant_keep_margin"), float("nan"))
    if not math.isfinite(teacher_margin) or not math.isfinite(quant_margin):
        return 0.0
    return max(teacher_margin - quant_margin, 0.0)


def _combined_keepmargin_impact(unit: Dict[str, Any], beta: float) -> float:
    return _quant_importance_gap(unit) + max(beta, 0.0) * _keep_margin_gap(unit)


def _normalize_weights(units: Sequence[Dict[str, Any]], raw_weights: Sequence[float]) -> List[float]:
    if not units:
        return []
    weights = [max(float(weight), 0.0) for weight in raw_weights]
    total = sum(weights)
    if total <= 0:
        return [1.0 for _ in units]
    scale = len(units) / total
    return [weight * scale for weight in weights]


def _target_offset(record: Dict[str, Any]) -> int:
    target = _safe_text(record.get("faith_token_target_text"))
    stripped = _safe_text(
        record.get("pred_explanation_stripped") or record.get("pred_explanation")
    )
    if not target or not stripped:
        return 1
    pos = target.find(stripped)
    return pos if pos >= 0 else 1


def _selected_char_spans(record: Dict[str, Any], units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    offset = _target_offset(record)
    spans: List[Dict[str, Any]] = []
    for idx, unit in enumerate(units):
        start = unit.get("start_char")
        end = unit.get("end_char")
        if start is None or end is None:
            continue
        start_char = int(start) + offset
        end_char = int(end) + offset
        if end_char <= start_char:
            continue
        spans.append(
            {
                "start_char": start_char,
                "end_char": end_char,
                "index": int(unit.get("index", idx)),
                "loss_weight": max(_safe_float(unit.get("loss_weight"), 1.0), 0.0),
            }
        )
    return spans


def _select_units(
    record: Dict[str, Any],
    variant: str,
    *,
    min_importance: float,
    min_quant_impact: float,
    keep_margin_beta: float,
) -> List[Dict[str, Any]]:
    if variant == "v1_qwen_all":
        return [copy.deepcopy(unit) for unit in record.get("qwen_evidence_units") or []]

    f_units = [
        copy.deepcopy(unit)
        for unit in record.get("faith_f_evidence_units") or []
        if _teacher_importance(unit) > min_importance
    ]
    if variant == "v2_teacher_faithful":
        return f_units

    if variant == "v3_teacher_faithful_qdamage":
        return [
            unit
            for unit in f_units
            if _quant_importance_gap(unit) > min_quant_impact
        ]

    if variant == "v4_teacher_faithful_qdamage_keepmargin":
        return [
            unit
            for unit in f_units
            if _combined_keepmargin_impact(unit, keep_margin_beta) > min_quant_impact
        ]

    raise ValueError(f"Unsupported variant: {variant}")


def _assign_weights(
    units: List[Dict[str, Any]],
    variant: str,
    *,
    keep_margin_beta: float,
) -> List[Dict[str, Any]]:
    if not units:
        return []

    if variant in {
        "v1_qwen_all",
        "v2_teacher_faithful",
        "v3_teacher_faithful_qdamage",
    }:
        weights = [1.0 for _ in units]
    elif variant == "v4_teacher_faithful_qdamage_keepmargin":
        weights = _normalize_weights(
            units,
            [
                _teacher_importance(unit)
                * _combined_keepmargin_impact(unit, keep_margin_beta)
                for unit in units
            ],
        )
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    weighted: List[Dict[str, Any]] = []
    for unit, weight in zip(units, weights):
        updated = copy.deepcopy(unit)
        updated["teacher_importance"] = _teacher_importance(unit)
        updated["quant_importance_gap"] = _quant_importance_gap(unit)
        updated["quant_keep_margin_gap_pos"] = _keep_margin_gap(unit)
        updated["ablation_keepmargin_impact"] = _combined_keepmargin_impact(
            unit,
            keep_margin_beta,
        )
        updated["loss_weight"] = float(weight)
        weighted.append(updated)
    return weighted


def transform_record(
    record: Dict[str, Any],
    variant: str,
    *,
    min_importance: float,
    min_quant_impact: float,
    keep_margin_beta: float,
) -> Dict[str, Any]:
    rec = copy.deepcopy(record)
    pred_answer = _safe_text(rec.get("pred_answer") or rec.get("gold_answer"))
    pred_answer_text = _safe_text(
        rec.get("pred_answer_text")
        or (rec.get("options") or {}).get(pred_answer, "")
    )
    rationale = _safe_text(rec.get("pred_explanation_stripped") or rec.get("pred_explanation"))

    units = _select_units(
        rec,
        variant,
        min_importance=min_importance,
        min_quant_impact=min_quant_impact,
        keep_margin_beta=keep_margin_beta,
    )
    units = _assign_weights(units, variant, keep_margin_beta=keep_margin_beta)
    selected_rationale = " ".join(_safe_text(unit.get("text")) for unit in units).strip()

    rec["faith_token_user_prompt"] = _build_explanation_training_prompt(
        example=rec,
        pred_answer=pred_answer,
        pred_answer_text=pred_answer_text,
    )
    rec["faith_token_target_text"] = " " + rationale
    rec["faith_token_selected_char_spans"] = _selected_char_spans(rec, units)
    rec["faith_recovery_selected_user_prompt"] = _build_recovery_prompt(
        rec,
        selected_rationale,
    )
    rec["faith_recovery_selected_rationale"] = selected_rationale
    rec["faith_recovery_unit_user_prompts"] = [
        _build_recovery_prompt(
            rec,
            _safe_text(unit.get("text")),
        )
        for unit in units
    ]
    rec["faith_selected_spans"] = units
    rec["faith_selected_span_count"] = len(units)
    rec["faith_q_evidence_units"] = units
    rec["faith_q_selected_span_count"] = len(units)
    rec["faith_ablation_variant"] = variant
    rec["faith_ablation_min_importance"] = float(min_importance)
    rec["faith_ablation_min_quant_impact"] = float(min_quant_impact)
    rec["faith_ablation_keep_margin_beta"] = float(keep_margin_beta)
    rec["faith_span_granularity"] = variant

    usable = bool(
        rec.get("is_correct", False)
        and units
        and rec["faith_token_selected_char_spans"]
        and rec["faith_recovery_selected_user_prompt"]
    )
    rec["selected_evidence_available"] = usable
    rec["faith_cache_usable"] = usable
    rec["faith_recovery_selected_success"] = usable
    rec["faith_cache_failure_reason"] = "" if usable else f"no_selected_spans_for_{variant}"
    return rec


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_summary(
    *,
    model_tag: str,
    variant: str,
    source_path: Path,
    output_path: Path,
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    usable = [record for record in records if record.get("faith_cache_usable")]
    span_counts = [len(record.get("faith_selected_spans") or []) for record in usable]
    token_span_counts = [
        len(record.get("faith_token_selected_char_spans") or []) for record in usable
    ]
    return {
        "model_tag": model_tag,
        "variant": variant,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "num_rows": len(records),
        "num_usable_rows": len(usable),
        "total_selected_spans": sum(span_counts),
        "mean_selected_spans_per_usable_row": (
            sum(span_counts) / len(span_counts) if span_counts else 0.0
        ),
        "total_token_spans": sum(token_span_counts),
        "first_usable_example_ids": [
            record.get("example_id") for record in usable[:5]
        ],
    }


def main() -> None:
    args = parse_args()
    source_root = args.repo_dir / "cache" / "med_faithfulness"
    all_summaries: List[Dict[str, Any]] = []

    for model_tag in args.models:
        source_path = (
            source_root
            / model_tag
            / f"medmix_train_teacher_predictions_qwen_evidence_pos_{args.source_tag}.jsonl"
        )
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        source_records = list(iter_jsonl(source_path))
        for variant in VARIANTS:
            records = [
                transform_record(
                    record,
                    variant,
                    min_importance=args.min_importance,
                    min_quant_impact=args.min_quant_impact,
                    keep_margin_beta=args.keep_margin_beta,
                )
                for record in source_records
            ]
            output_path = (
                args.output_root
                / model_tag
                / f"medmix_train_teacher_predictions_{variant}_{args.source_tag}.jsonl"
            )
            write_jsonl(output_path, records)
            summary = build_summary(
                model_tag=model_tag,
                variant=variant,
                source_path=source_path,
                output_path=output_path,
                records=records,
            )
            summary_path = output_path.with_name(output_path.stem + "_summary.json")
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            all_summaries.append(summary)
            print(
                f"{model_tag:22s} {variant:40s} "
                f"usable={summary['num_usable_rows']:3d} "
                f"spans={summary['total_selected_spans']:4d} "
                f"-> {output_path}"
            )

    aggregate_path = args.output_root / f"qwen_component_ablation_{args.source_tag}_summary.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] wrote aggregate summary: {aggregate_path}")


if __name__ == "__main__":
    main()
