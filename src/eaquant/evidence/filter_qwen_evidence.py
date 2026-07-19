"""Post-process a Qwen evidence cache with the thresholds used in the paper."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute selected Q evidence spans from stored per-unit scores."
    )
    parser.add_argument("--input", required=True, help="Input JSONL cache.")
    parser.add_argument("--output", required=True, help="Output JSONL cache.")
    parser.add_argument("--summary", default="", help="Optional summary JSON path.")
    parser.add_argument("--min_importance", type=float, default=0.0)
    parser.add_argument("--min_quant_impact", type=float, default=0.0)
    parser.add_argument("--quant_keep_margin_beta", type=float, default=0.0)
    parser.add_argument("--tag", default="", help="Optional tag recorded in each row.")
    return parser.parse_args()


def _recovery_unit_prompt(record: Dict[str, Any], rationale: str) -> str:
    options = record.get("options") or {}
    option_text = "\n\n".join(f"{label}. {value}" for label, value in options.items())
    return (
        "Answer the medical multiple-choice question.\n"
        "Use the provided rationale to infer the selected answer.\n"
        "Return the option label followed by the option text.\n\n"
        f"Question:\n{record.get('question', '')}\n\n"
        f"Options:\n\n{option_text}\n\n"
        f"Rationale:\n{rationale}\n\n"
        "Final:"
    )


def _target_offset(record: Dict[str, Any]) -> int:
    target = record.get("faith_token_target_text") or ""
    stripped = record.get("pred_explanation_stripped") or record.get("pred_explanation") or ""
    pos = target.find(stripped)
    return pos if pos >= 0 else 0


def transform_record(
    record: Dict[str, Any],
    *,
    min_importance: float,
    min_quant_impact: float,
    quant_keep_margin_beta: float,
    tag: str,
) -> Dict[str, Any]:
    rec = copy.deepcopy(record)
    f_units: List[Dict[str, Any]] = []

    for unit in rec.get("faith_f_evidence_units") or []:
        teacher_importance = float(unit.get("teacher_importance", 0.0) or 0.0)
        if teacher_importance <= min_importance:
            continue

        updated = copy.deepcopy(unit)
        importance_gap = max(
            float(
                updated.get(
                    "quant_importance_gap_raw",
                    updated.get("quant_importance_gap", 0.0),
                )
                or 0.0
            ),
            0.0,
        )
        keep_margin_gap = max(
            float(
                updated.get(
                    "quant_keep_margin_gap_raw",
                    updated.get("quant_keep_margin_gap_pos", 0.0),
                )
                or 0.0
            ),
            0.0,
        )
        weighted_keep_gap = max(float(quant_keep_margin_beta), 0.0) * keep_margin_gap
        impact = importance_gap + weighted_keep_gap

        updated["quant_importance_gap"] = float(importance_gap)
        updated["quant_keep_margin_gap_pos"] = float(keep_margin_gap)
        updated["quant_keep_margin_gap_beta"] = float(quant_keep_margin_beta)
        updated["quant_keep_margin_gap_weighted"] = float(weighted_keep_gap)
        updated["quant_impact_score"] = float(impact)
        updated["quant_impact_source"] = "importance_gap_plus_keep_margin_gap"
        updated["quant_affected_evidence"] = bool(impact > min_quant_impact)
        f_units.append(updated)

    q_units = [
        copy.deepcopy(unit)
        for unit in f_units
        if bool(unit.get("quant_affected_evidence"))
    ]
    total_impact = sum(
        max(float(unit.get("quant_impact_score", 0.0) or 0.0), 0.0)
        for unit in q_units
    )
    offset = _target_offset(rec)
    selected_char_spans = []
    for unit in q_units:
        impact = max(float(unit.get("quant_impact_score", 0.0) or 0.0), 0.0)
        loss_weight = impact / total_impact * len(q_units) if total_impact > 0 else 1.0
        unit["loss_weight"] = float(loss_weight)
        selected_char_spans.append(
            {
                "start_char": int(unit.get("start_char", 0)) + offset,
                "end_char": int(unit.get("end_char", 0)) + offset,
                "index": int(unit.get("index", len(selected_char_spans))),
                "loss_weight": float(loss_weight),
            }
        )

    rec["faith_f_evidence_units"] = f_units
    rec["faith_q_evidence_units"] = q_units
    rec["faith_f_selected_span_count"] = len(f_units)
    rec["faith_q_selected_span_count"] = len(q_units)
    rec["faith_selected_span_count"] = len(q_units)
    rec["faith_selected_spans"] = q_units
    rec["faith_token_selected_char_spans"] = selected_char_spans
    rec["faith_recovery_unit_user_prompts"] = [
        _recovery_unit_prompt(rec, unit.get("text", "")) for unit in f_units
    ]
    rec["faith_min_importance"] = float(min_importance)
    rec["faith_min_quant_impact"] = float(min_quant_impact)
    rec["faith_quant_keep_margin_beta"] = float(quant_keep_margin_beta)
    rec["faith_posthoc_filter_tag"] = tag
    rec["selected_evidence_available"] = bool(selected_char_spans)

    usable = (
        bool(rec.get("is_correct", False))
        and bool(
            rec.get(
                "candidate_evidence_recovery_valid",
                rec.get("faith_recovery_selected_success", False),
            )
        )
        and bool(selected_char_spans)
    )
    rec["faith_cache_usable"] = bool(usable)
    if usable:
        rec["faith_cache_failure_reason"] = ""
    elif not f_units:
        rec["faith_cache_failure_reason"] = "no_full_precision_f_evidence_posthoc"
    elif not q_units:
        rec["faith_cache_failure_reason"] = "no_quant_affected_q_evidence_posthoc"
    else:
        rec["faith_cache_failure_reason"] = (
            rec.get("faith_cache_failure_reason") or "not_usable_posthoc"
        )
    return rec


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_name(
        output_path.stem + "_summary.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "source_path": str(input_path),
        "output_path": str(output_path),
        "min_importance": float(args.min_importance),
        "min_quant_impact": float(args.min_quant_impact),
        "quant_keep_margin_beta": float(args.quant_keep_margin_beta),
        "tag": args.tag,
        "total_records": 0,
        "usable_records": 0,
        "records_with_f_evidence": 0,
        "records_with_q_evidence": 0,
        "total_f_units_all_records": 0,
        "total_q_units_all_records": 0,
        "failure_reasons": {},
    }

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            record = transform_record(
                json.loads(line),
                min_importance=args.min_importance,
                min_quant_impact=args.min_quant_impact,
                quant_keep_margin_beta=args.quant_keep_margin_beta,
                tag=args.tag,
            )
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["total_records"] += 1
            stats["usable_records"] += int(bool(record.get("faith_cache_usable")))
            f_count = int(record.get("faith_f_selected_span_count", 0) or 0)
            q_count = int(record.get("faith_q_selected_span_count", 0) or 0)
            stats["records_with_f_evidence"] += int(f_count > 0)
            stats["records_with_q_evidence"] += int(q_count > 0)
            stats["total_f_units_all_records"] += f_count
            stats["total_q_units_all_records"] += q_count
            reason = record.get("faith_cache_failure_reason") or ""
            if reason:
                stats["failure_reasons"][reason] = (
                    stats["failure_reasons"].get(reason, 0) + 1
                )

    summary_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
