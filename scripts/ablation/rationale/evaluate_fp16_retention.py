#!/usr/bin/env python3
"""Evaluate five-way FP16 evidence retention for the rationale ablation.

This is the public, path-parameterized version of the runner used for the
paper's Llama-3 rationale ablation.  It evaluates only examples for which all
five systems predict the same answer label:

* FP16 teacher
* standard MedMix PTQ (shown to the paper judge as ``OSTQuant``)
* EAQuant (shown to the paper judge as ``Ours``)
* the full-rationale ablation
* the matched-random-rationale ablation

The FP16 claim cache is filtered to that common-label cohort.  Previously
judged standard-PTQ and EAQuant support records may be supplied explicitly;
otherwise all four candidate systems are judged in this run.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eaquant.evaluation import fp16_retention as fec  # noqa: E402
from eaquant.evaluation.judge_common import (  # noqa: E402
    DEFAULT_API_KEY_FILE,
    OPENAI_CHAT_COMPLETIONS_URL,
    append_jsonl,
    extract_gold_answer,
    extract_pred_answer,
    format_answer,
    infer_correct,
    key_to_str,
    load_api_key_from_file,
    normalize_text,
    read_jsonl,
    record_key,
    slugify,
    write_json,
)


CANDIDATE_SYSTEMS = (
    "medmix_baseline",
    "eaquant",
    "full_noanswer",
    "random_noanswer",
)
SYSTEM_DISPLAY = {
    "fp16": "FP16 teacher",
    # Preserve the labels used in the paper's judge prompts.
    "medmix_baseline": "OSTQuant",
    "eaquant": "Ours",
    "full_noanswer": "Full rationale",
    "random_noanswer": "Random matched",
}
SYSTEM_ORDER = ("fp16",) + CANDIDATE_SYSTEMS
REUSABLE_SYSTEMS = ("medmix_baseline", "eaquant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Five-way FP16 evidence retention on the cohort where FP16, standard "
            "MedMix PTQ, EAQuant, full-rationale, and random-rationale predictions "
            "have the same answer label."
        )
    )
    parser.add_argument("--fp16_jsonl", type=Path, required=True)
    parser.add_argument("--medmix_baseline_jsonl", type=Path, required=True)
    parser.add_argument("--eaquant_jsonl", type=Path, required=True)
    parser.add_argument("--full_rationale_jsonl", type=Path, required=True)
    parser.add_argument("--random_rationale_jsonl", type=Path, required=True)
    parser.add_argument(
        "--claim_cache",
        type=Path,
        required=True,
        help="FP16 atomic-claim JSONL produced by the main retention evaluator.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--medmix_baseline_reuse_jsonl",
        type=Path,
        default=None,
        help=(
            "Optional existing standard-PTQ support judgments to filter to the "
            "five-way cohort."
        ),
    )
    parser.add_argument(
        "--eaquant_reuse_jsonl",
        type=Path,
        default=None,
        help=(
            "Optional existing EAQuant support judgments to filter to the "
            "five-way cohort."
        ),
    )
    parser.add_argument("--dataset", default="medexpqa")
    parser.add_argument("--dataset_display", default="MedExpQA")
    parser.add_argument("--model_key", default="llama3_instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--judge_model", default="gpt-5.4")
    parser.add_argument("--max_claims", type=int, default=8)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--api_key_file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument(
        "--api_url",
        default=os.environ.get(
            "OPENAI_CHAT_COMPLETIONS_URL", OPENAI_CHAT_COMPLETIONS_URL
        ),
    )
    parser.add_argument("--max_completion_tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--omit_temperature", action="store_true")
    parser.add_argument(
        "--reasoning_effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    parser.add_argument("--request_timeout", type=float, default=90.0)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--dry_run_prompts", action="store_true")
    parser.add_argument("--verbose_prompts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--force_rejudge_reusable",
        action="store_true",
        help="Ignore supplied reusable support files and call the judge again.",
    )
    args = parser.parse_args()
    if args.max_claims <= 0:
        parser.error("--max_claims must be positive")
    return args


def prediction_paths(args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "fp16": args.fp16_jsonl,
        "medmix_baseline": args.medmix_baseline_jsonl,
        "eaquant": args.eaquant_jsonl,
        "full_noanswer": args.full_rationale_jsonl,
        "random_noanswer": args.random_rationale_jsonl,
    }


def reuse_paths(args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    return {
        "medmix_baseline": args.medmix_baseline_reuse_jsonl,
        "eaquant": args.eaquant_reuse_jsonl,
    }


def by_pair_key(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for record in records:
        key = record_key(record)
        if key is None:
            continue
        pair_key = key_to_str(key)
        if pair_key not in out:
            out[pair_key] = record
    return out


def read_completed_pair_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        pair_key
        for record in read_jsonl(path)
        if (pair_key := normalize_text(record.get("pair_key")))
    }


def load_api_key(args: argparse.Namespace) -> str:
    if args.dry_run_prompts:
        return ""
    api_key = normalize_text(os.environ.get(args.api_key_env))
    if not api_key:
        api_key = load_api_key_from_file(args.api_key_file)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {args.api_key_env}=..., put it in "
            f"{args.api_key_file}, or use --dry_run_prompts."
        )
    return api_key


def coverage_args(args: argparse.Namespace) -> SimpleNamespace:
    """Build the namespace expected by the shared OpenAI-call helper."""
    return SimpleNamespace(
        dataset=args.dataset,
        dataset_display=args.dataset_display,
        judge_model=args.judge_model,
        api_url=args.api_url,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        omit_temperature=args.omit_temperature,
        reasoning_effort=args.reasoning_effort,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        dry_run_prompts=args.dry_run_prompts,
        verbose_prompts=args.verbose_prompts,
    )


def support_output_paths(
    args: argparse.Namespace, system: str
) -> Tuple[Path, Path]:
    stem = (
        f"fec_support_{slugify(args.dataset)}_{slugify(args.model_key)}_"
        f"{system}_seed{args.seed}_by_{slugify(args.judge_model)}"
    )
    return args.output_dir / f"{stem}.jsonl", args.output_dir / f"{stem}_summary.json"


def claim_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    stem = (
        f"fp16_claims_{slugify(args.dataset)}_{slugify(args.model_key)}_"
        f"by_{slugify(args.judge_model)}"
    )
    return args.output_dir / f"{stem}.jsonl", args.output_dir / f"{stem}_summary.json"


def claim_summary_from_records(
    records: Sequence[Dict[str, Any]],
    output_jsonl: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    total_claims = sum(len(record.get("claims") or []) for record in records)
    nonempty = sum(1 for record in records if record.get("claims"))
    parse_failures = sum(
        1 for record in records if record.get("judge_parse_success") is False
    )
    return {
        "task": "claims",
        "dataset": args.dataset,
        "dataset_display": args.dataset_display,
        "model_key": args.model_key,
        "seed": args.seed,
        "judge_model": args.judge_model,
        "num_records": len(records),
        "num_model_calls": 0,
        "num_parse_failures": parse_failures,
        "num_records_with_claims": nonempty,
        "num_records_without_claims": len(records) - nonempty,
        "total_claims": total_claims,
        "avg_claims_per_record": total_claims / len(records) if records else None,
        "output_jsonl": str(output_jsonl),
    }


def fp16_row_from_claim_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    total_claims = int(summary.get("total_claims") or 0)
    return {
        "dataset": summary.get("dataset"),
        "dataset_display": summary.get("dataset_display"),
        "model_key": summary.get("model_key"),
        "system": "fp16",
        "system_display": SYSTEM_DISPLAY["fp16"],
        "seed": summary.get("seed"),
        "num_runs": 1,
        "num_records": summary.get("num_records"),
        "num_scored_records": summary.get("num_records_with_claims"),
        "total_claims_scored": total_claims,
        "supported_count": total_claims,
        "unsupported_count": 0,
        "contradicted_count": 0,
        "fec_macro_rate": 1.0 if total_claims > 0 else None,
        "fec_micro_rate": 1.0 if total_claims > 0 else None,
        "num_parse_failures": summary.get("num_parse_failures"),
        "output_jsonl": summary.get("output_jsonl"),
    }


def row_from_support_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": summary.get("dataset"),
        "dataset_display": summary.get("dataset_display"),
        "model_key": summary.get("model_key"),
        "system": summary.get("system"),
        "system_display": summary.get("system_display"),
        "seed": summary.get("seed"),
        "num_runs": 1,
        "num_records": summary.get("num_records"),
        "num_scored_records": summary.get("num_scored_records"),
        "total_claims_scored": summary.get("total_claims_scored"),
        "supported_count": summary.get("supported_count"),
        "unsupported_count": summary.get("unsupported_count"),
        "contradicted_count": summary.get("contradicted_count"),
        "fec_macro_rate": summary.get("fec_macro_rate"),
        "fec_micro_rate": summary.get("fec_micro_rate"),
        "num_parse_failures": summary.get("num_parse_failures"),
        "output_jsonl": summary.get("output_jsonl"),
    }


def pct(value: Optional[float]) -> str:
    return "" if value is None else f"{value * 100:.2f}"


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def filter_reusable_support(
    *,
    source_path: Path,
    output_path: Path,
    active_pair_keys: set[str],
    system: str,
    candidate_file: Path,
    overwrite: bool,
) -> List[Dict[str, Any]]:
    if output_path.exists() and not overwrite:
        records = [
            record
            for record in read_jsonl(output_path)
            if normalize_text(record.get("pair_key")) in active_pair_keys
        ]
        if len(records) == len(active_pair_keys):
            return records

    source_records = [
        record
        for record in read_jsonl(source_path)
        if normalize_text(record.get("pair_key")) in active_pair_keys
    ]
    by_key = {normalize_text(record.get("pair_key")): record for record in source_records}
    missing = sorted(active_pair_keys - set(by_key))
    if missing:
        raise RuntimeError(
            f"Reusable support file is missing {len(missing)} pair keys: {source_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    for pair_key in sorted(active_pair_keys):
        record = dict(by_key[pair_key])
        record["system"] = system
        record["system_display"] = SYSTEM_DISPLAY[system]
        record["candidate_file"] = str(candidate_file)
        record["reused_from_jsonl"] = str(source_path)
        append_jsonl(output_path, record)
    return read_jsonl(output_path)


def run_new_support(
    *,
    args: argparse.Namespace,
    api_key: str,
    fec_args: SimpleNamespace,
    system: str,
    fp16_by_key: Dict[str, Dict[str, Any]],
    candidate_by_key: Dict[str, Dict[str, Any]],
    claims_by_key: Dict[str, Dict[str, Any]],
    fp16_file: Path,
    candidate_file: Path,
    active_pair_keys: List[str],
    output_jsonl: Path,
) -> List[Dict[str, Any]]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_jsonl.exists():
        output_jsonl.unlink()

    completed = read_completed_pair_keys(output_jsonl)
    active_set = set(active_pair_keys)
    existing = (
        [
            record
            for record in read_jsonl(output_jsonl)
            if normalize_text(record.get("pair_key")) in active_set
        ]
        if output_jsonl.exists()
        else []
    )

    total = len(active_pair_keys)
    new_records: List[Dict[str, Any]] = []
    for index, pair_key in enumerate(active_pair_keys, start=1):
        if pair_key in completed and not args.overwrite:
            continue

        fp16_record = fp16_by_key[pair_key]
        candidate_record = candidate_by_key[pair_key]
        claim_record = claims_by_key[pair_key]
        claims = [
            claim for claim in (claim_record.get("claims") or []) if isinstance(claim, dict)
        ]
        user_prompt = fec.build_coverage_prompt(
            fp16_record=fp16_record,
            candidate_record=candidate_record,
            candidate_system=system,
            claims=claims,
        )
        messages = fec.build_messages(user_prompt)
        record = {
            "task": "coverage",
            "dataset": args.dataset,
            "dataset_display": args.dataset_display,
            "model_key": args.model_key,
            "seed": args.seed,
            "system": system,
            "system_display": SYSTEM_DISPLAY[system],
            "judge_model": args.judge_model,
            "judge_api_url": args.api_url,
            "judge_model_call_skipped": bool(args.dry_run_prompts),
            "fp16_file": str(fp16_file),
            "candidate_file": str(candidate_file),
            "pair_key": pair_key,
            "example_id": fp16_record.get("example_id")
            or candidate_record.get("example_id"),
            "split": fp16_record.get("split") or candidate_record.get("split"),
            "question_type": fp16_record.get("question_type")
            or candidate_record.get("question_type"),
            "source_file": fp16_record.get("source_file")
            or candidate_record.get("source_file"),
            "row_idx": fp16_record.get("row_idx")
            if fp16_record.get("row_idx") is not None
            else candidate_record.get("row_idx"),
            "question": normalize_text(
                fp16_record.get("question") or candidate_record.get("question")
            ),
            "options": fp16_record.get("options")
            or candidate_record.get("options")
            or {},
            "gold_answer": extract_gold_answer(fp16_record)
            or extract_gold_answer(candidate_record),
            "fp16_pred_answer": extract_pred_answer(fp16_record),
            "fp16_pred_answer_text": format_answer(fp16_record),
            "fp16_is_correct": infer_correct(fp16_record),
            "fp16_explanation": fec.extract_explanation(fp16_record),
            "candidate_pred_answer": extract_pred_answer(candidate_record),
            "candidate_pred_answer_text": format_answer(candidate_record),
            "candidate_is_correct": infer_correct(candidate_record),
            "candidate_explanation": fec.extract_explanation(candidate_record),
            "claims": claims,
        }
        if args.verbose_prompts or args.dry_run_prompts:
            record["system_prompt"] = fec.SYSTEM_PROMPT
            record["user_prompt"] = user_prompt

        if args.dry_run_prompts:
            claim_results = fec.unjudged_claim_results(claims)
            coverage_reason = ""
            call_result = {
                "judge_parse_success": None,
                "judge_attempt_count": 0,
                "judge_error": "",
                "raw_judge_response": "",
                "openai_response_id": "",
                "openai_usage": {},
            }
        elif not claims:
            claim_results = []
            coverage_reason = "No FP16 clinical evidence claims were extracted."
            call_result = {
                "judge_parse_success": True,
                "judge_attempt_count": 0,
                "judge_error": "",
                "raw_judge_response": "",
                "openai_response_id": "",
                "openai_usage": {},
            }
        else:
            call_result = fec.run_openai_json_call(
                args=fec_args,
                api_key=api_key,
                task_name=f"five_way_coverage_{system}",
                messages=messages,
                response_format=fec.COVERAGE_SCHEMA,
            )
            if call_result.get("judge_parse_success") is True:
                claim_results, coverage_reason = fec.normalize_coverage(
                    call_result.pop("parsed", {}), claims
                )
            else:
                call_result.pop("parsed", None)
                claim_results = fec.unjudged_claim_results(claims)
                coverage_reason = "Coverage judge call failed."

        if any(item.get("support") == "unjudged" for item in claim_results):
            score = {
                "num_claims": len(claims),
                "supported_count": 0,
                "unsupported_count": 0,
                "contradicted_count": 0,
                "fec_score": None,
            }
        else:
            score = fec.score_claim_results(claim_results)

        record.update(call_result)
        record.update(score)
        record.update(
            {"claim_results": claim_results, "coverage_reason": coverage_reason}
        )
        append_jsonl(output_jsonl, record)
        new_records.append(record)
        print(
            f"[{index}/{total}] {system} {pair_key}: "
            f"FEC={pct(record.get('fec_score')) or 'NA'}"
        )

    return new_records if args.overwrite else existing + new_records


def summarize_support_records(
    *,
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    system: str,
    candidate_file: Path,
    output_jsonl: Path,
    summary_json: Path,
    reused_from: Optional[Path],
    num_active_pairs: int,
) -> Dict[str, Any]:
    summary = fec.summarize_coverage_records(
        records=records, dry_run_prompts=args.dry_run_prompts
    )
    summary.update(
        {
            "task": "coverage",
            "dataset": args.dataset,
            "dataset_display": args.dataset_display,
            "model_key": args.model_key,
            "seed": args.seed,
            "system": system,
            "system_display": SYSTEM_DISPLAY[system],
            "judge_model": args.judge_model,
            "five_way_common_label_filter_enabled": True,
            "required_same_predicted_answer_labels": list(SYSTEM_ORDER),
            "num_five_way_common_label_pairs": num_active_pairs,
            "candidate_file": str(candidate_file),
            "output_jsonl": str(output_jsonl),
            "summary_json": str(summary_json),
            "reused_from_jsonl": str(reused_from) if reused_from else None,
        }
    )
    write_json(summary_json, summary)
    return summary


def build_common_label_cohort(
    records: Dict[str, List[Dict[str, Any]]],
    by_key: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[List[str], Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "num_rows_by_system": {
            system: len(system_records) for system, system_records in records.items()
        },
        "num_common_keys": 0,
        "num_missing_answer": 0,
        "num_different_answer": 0,
        "label_counts": {},
    }
    fp16_order = [
        key_to_str(record_key(record))
        for record in records["fp16"]
        if record_key(record) is not None
    ]
    all_keys = set.intersection(*(set(mapping) for mapping in by_key.values()))
    stats["num_common_keys"] = len(all_keys)

    active_pair_keys: List[str] = []
    label_counts: Dict[str, int] = {}
    for pair_key in fp16_order:
        if pair_key not in all_keys:
            continue
        answers = {
            system: normalize_text(extract_pred_answer(by_key[system][pair_key])).upper()
            for system in SYSTEM_ORDER
        }
        if not all(answers.values()):
            stats["num_missing_answer"] += 1
            continue
        if len(set(answers.values())) != 1:
            stats["num_different_answer"] += 1
            continue
        active_pair_keys.append(pair_key)
        label = answers["fp16"]
        label_counts[label] = label_counts.get(label, 0) + 1

    stats["num_five_way_common_label_pairs"] = len(active_pair_keys)
    stats["label_counts"] = dict(sorted(label_counts.items()))
    return active_pair_keys, stats


def make_mean_std_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "model_key": row["model_key"],
                "system": row["system"],
                "system_display": row["system_display"],
                "num_runs": 1,
                "seeds": "" if row["seed"] is None else str(row["seed"]),
                "num_scored_records": row.get("num_scored_records"),
                "total_claims_scored": row.get("total_claims_scored"),
                "supported_count": row.get("supported_count"),
                "num_parse_failures": row.get("num_parse_failures"),
                "fec_macro_mean_rate": row.get("fec_macro_rate"),
                "fec_macro_std_rate": 0.0,
                "fec_macro_mean_pct": pct(row.get("fec_macro_rate")),
                "fec_macro_std_pct": pct(0.0),
                "fec_macro_mean_std_pct": pct(row.get("fec_macro_rate")),
                "fec_macro_num_values": 1
                if row.get("fec_macro_rate") is not None
                else 0,
                "fec_micro_mean_rate": row.get("fec_micro_rate"),
                "fec_micro_std_rate": 0.0,
                "fec_micro_mean_pct": pct(row.get("fec_micro_rate")),
                "fec_micro_std_pct": pct(0.0),
                "fec_micro_mean_std_pct": pct(row.get("fec_micro_rate")),
                "fec_micro_num_values": 1
                if row.get("fec_micro_rate") is not None
                else 0,
            }
        )
    return output


def write_aggregate_outputs(
    *,
    args: argparse.Namespace,
    paths: Dict[str, Path],
    common_label_stats: Dict[str, Any],
    claim_summary: Dict[str, Any],
    support_summaries: List[Dict[str, Any]],
) -> Tuple[Path, Path, Path]:
    rows = [fp16_row_from_claim_summary(claim_summary)] + [
        row_from_support_summary(summary) for summary in support_summaries
    ]
    for row in rows:
        row["fec_macro_pct"] = pct(row.get("fec_macro_rate"))
        row["fec_micro_pct"] = pct(row.get("fec_micro_rate"))
    by_model_mean_std = make_mean_std_rows(rows)

    rows_csv = args.output_dir / "fp16_evidence_coverage_rows.csv"
    mean_std_csv = args.output_dir / "fp16_evidence_coverage_by_model_mean_std.csv"
    summary_json = args.output_dir / "fp16_evidence_coverage_summary.json"
    write_csv(
        rows_csv,
        rows,
        [
            "dataset",
            "model_key",
            "system",
            "system_display",
            "seed",
            "num_records",
            "num_scored_records",
            "total_claims_scored",
            "supported_count",
            "unsupported_count",
            "contradicted_count",
            "fec_macro_rate",
            "fec_macro_pct",
            "fec_micro_rate",
            "fec_micro_pct",
            "num_parse_failures",
            "output_jsonl",
        ],
    )
    write_csv(
        mean_std_csv,
        by_model_mean_std,
        [
            "model_key",
            "system",
            "system_display",
            "num_runs",
            "seeds",
            "num_scored_records",
            "total_claims_scored",
            "supported_count",
            "fec_macro_mean_rate",
            "fec_macro_std_rate",
            "fec_macro_mean_pct",
            "fec_macro_std_pct",
            "fec_macro_mean_std_pct",
            "fec_macro_num_values",
            "fec_micro_mean_rate",
            "fec_micro_std_rate",
            "fec_micro_mean_pct",
            "fec_micro_std_pct",
            "fec_micro_mean_std_pct",
            "fec_micro_num_values",
            "num_parse_failures",
        ],
    )
    write_json(
        summary_json,
        {
            "dataset": args.dataset,
            "dataset_display": args.dataset_display,
            "model_key": args.model_key,
            "seed": args.seed,
            "judge_model": args.judge_model,
            "max_claims": args.max_claims,
            "output_dir": str(args.output_dir),
            "prediction_files": {system: str(path) for system, path in paths.items()},
            "claim_cache": str(args.claim_cache),
            "common_label_filter": common_label_stats,
            "claim_summary": claim_summary,
            "support_summaries": support_summaries,
            "rows": rows,
            "by_model_mean_std": by_model_mean_std,
        },
    )
    return summary_json, rows_csv, mean_std_csv


def main() -> None:
    args = parse_args()
    # The shared prompt builder looks up these labels at call time.
    fec.SYSTEM_DISPLAY.update(SYSTEM_DISPLAY)
    fec.PAPER_JUDGE_SYSTEM_DISPLAY.update(SYSTEM_DISPLAY)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key(args)
    fec_args = coverage_args(args)
    paths = prediction_paths(args)
    reusable_paths = reuse_paths(args)

    required_paths = list(paths.values()) + [args.claim_cache]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
    supplied_reuse = [path for path in reusable_paths.values() if path is not None]
    missing_reuse = [str(path) for path in supplied_reuse if not path.is_file()]
    if missing_reuse:
        raise FileNotFoundError(
            "Missing explicitly supplied reusable support files:\n"
            + "\n".join(missing_reuse)
        )

    records = {system: read_jsonl(path) for system, path in paths.items()}
    by_key = {system: by_pair_key(items) for system, items in records.items()}
    active_pair_keys, common_label_stats = build_common_label_cohort(records, by_key)

    claim_records = read_jsonl(args.claim_cache)
    claims_by_key = {
        pair_key: record
        for record in claim_records
        if (pair_key := normalize_text(record.get("pair_key")))
    }
    missing_claims = [
        pair_key for pair_key in active_pair_keys if pair_key not in claims_by_key
    ]
    if missing_claims:
        raise RuntimeError(
            f"Missing {len(missing_claims)} claim records in {args.claim_cache}"
        )

    claim_jsonl, claim_summary_json = claim_output_paths(args)
    if args.overwrite and claim_jsonl.exists():
        claim_jsonl.unlink()
    filtered_claims = [claims_by_key[pair_key] for pair_key in active_pair_keys]
    if args.overwrite or not claim_jsonl.exists():
        for record in filtered_claims:
            append_jsonl(claim_jsonl, record)
    claim_summary = claim_summary_from_records(filtered_claims, claim_jsonl, args)
    claim_summary["source_claim_cache"] = str(args.claim_cache)
    claim_summary["max_claims"] = args.max_claims
    claim_summary["five_way_common_label_filter"] = common_label_stats
    write_json(claim_summary_json, claim_summary)

    support_summaries: List[Dict[str, Any]] = []
    for system in CANDIDATE_SYSTEMS:
        output_jsonl, summary_json = support_output_paths(args, system)
        reuse_path = reusable_paths.get(system)
        reused_from = None
        if (
            system in REUSABLE_SYSTEMS
            and reuse_path is not None
            and not args.force_rejudge_reusable
            and not args.dry_run_prompts
        ):
            support_records = filter_reusable_support(
                source_path=reuse_path,
                output_path=output_jsonl,
                active_pair_keys=set(active_pair_keys),
                system=system,
                candidate_file=paths[system],
                overwrite=args.overwrite,
            )
            reused_from = reuse_path
            print(
                f"[REUSE] {system}: {len(support_records)} records from {reuse_path}"
            )
        else:
            support_records = run_new_support(
                args=args,
                api_key=api_key,
                fec_args=fec_args,
                system=system,
                fp16_by_key=by_key["fp16"],
                candidate_by_key=by_key[system],
                claims_by_key=claims_by_key,
                fp16_file=paths["fp16"],
                candidate_file=paths[system],
                active_pair_keys=active_pair_keys,
                output_jsonl=output_jsonl,
            )
        support_summaries.append(
            summarize_support_records(
                args=args,
                records=support_records,
                system=system,
                candidate_file=paths[system],
                output_jsonl=output_jsonl,
                summary_json=summary_json,
                reused_from=reused_from,
                num_active_pairs=len(active_pair_keys),
            )
        )

    summary_json, rows_csv, mean_std_csv = write_aggregate_outputs(
        args=args,
        paths=paths,
        common_label_stats=common_label_stats,
        claim_summary=claim_summary,
        support_summaries=support_summaries,
    )
    print(f"[DONE] wrote {summary_json}")
    print(f"[DONE] wrote {rows_csv}")
    print(f"[DONE] wrote {mean_std_csv}")


if __name__ == "__main__":
    main()
