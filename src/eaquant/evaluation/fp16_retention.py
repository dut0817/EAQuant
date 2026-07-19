import argparse
import csv
import json
import os
import sys
import time
import urllib.error
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .judge_common import (
    DEFAULT_API_KEY_FILE,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_SEEDS,
    MODEL_KEYS,
    OPENAI_CHAT_COMPLETIONS_URL,
    append_jsonl,
    build_index,
    call_openai_chat_completion,
    extract_explanation,
    extract_gold_answer,
    extract_pred_answer,
    format_answer,
    format_options,
    infer_correct,
    key_to_str,
    load_api_key_from_file,
    normalize_text,
    parse_json_object,
    read_error_body,
    read_jsonl,
    record_key,
    should_retry_status,
    slugify,
    write_json,
)
from .fp16_agreement import (
    DATASET_CONFIGS,
    SYSTEM_DISPLAY,
    SYSTEM_ORDER,
    dataset_data_dir,
    fp16_prediction_path,
    medmix_baseline_prediction_path,
    eaquant_prediction_path,
)


SYSTEM_PROMPT = (
    "You are an expert medical evaluator for clinical multiple-choice QA. "
    "Follow the rubric exactly. Output only the requested JSON object."
)

COVERAGE_SYSTEMS = ("medmix_baseline", "eaquant")
DEFAULT_MAX_CLAIMS = 8
PAPER_JUDGE_SYSTEM_DISPLAY = {
    "medmix_baseline": "OSTQuant",
    "eaquant": "Ours",
}

CLAIM_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fp16_evidence_claim_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "claim_id": {"type": "string"},
                            "claim": {"type": "string"},
                            "evidence_type": {
                                "type": "string",
                                "enum": [
                                    "clinical_finding",
                                    "lab_finding",
                                    "diagnosis",
                                    "treatment",
                                    "risk_factor",
                                    "pathophysiology",
                                    "other",
                                ],
                            },
                        },
                        "required": ["claim_id", "claim", "evidence_type"],
                    },
                },
                "reason": {"type": "string"},
            },
            "required": ["claims", "reason"],
        },
    },
}

COVERAGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fp16_evidence_coverage_judgment",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "claim_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "claim_id": {"type": "string"},
                            "support": {
                                "type": "string",
                                "enum": ["supported", "unsupported", "contradicted"],
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["claim_id", "support", "reason"],
                    },
                },
                "reason": {"type": "string"},
            },
            "required": ["claim_results", "reason"],
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use an OpenAI judge to compute FP16 Evidence Coverage (FEC): "
            "how much answer-supporting clinical evidence from the FP16 rationale "
            "is preserved in quantized rationales."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        default="medexpqa",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--mode",
        choices=["claims", "coverage", "both"],
        default="both",
        help=(
            "claims: extract FP16 atomic clinical claims only. "
            "coverage: judge MedMix PTQ/EAQuant coverage using existing claims. "
            "both: run claim extraction then coverage."
        ),
    )
    parser.add_argument(
        "--fp16_dir",
        type=Path,
        default=None,
        help="Directory containing original train_baseline prediction JSONL files.",
    )
    parser.add_argument(
        "--medmix_baseline_dir",
        type=Path,
        default=None,
        help="Directory containing standard MedMix PTQ prediction files.",
    )
    parser.add_argument(
        "--eaquant_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing EAQuant prediction files. Defaults to files without "
            "tok/w ablation tags, except MedExpQA llama3 uses the v4 qwen_component files."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory where FEC JSONL, summaries, CSV, and LaTeX files are written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KEYS,
        default=list(MODEL_KEYS),
        help="Model keys to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Quantized seeds to evaluate. Baseline seed 0 maps to the no-suffix file.",
    )
    parser.add_argument(
        "--judge_model",
        default=DEFAULT_JUDGE_MODEL,
        help=(
            "OpenAI model for judging. Defaults to OPENAI_JUDGE_MODEL if set, "
            f"otherwise {DEFAULT_JUDGE_MODEL}."
        ),
    )
    parser.add_argument(
        "--api_key_env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the OpenAI API key.",
    )
    parser.add_argument(
        "--api_key_file",
        type=Path,
        default=DEFAULT_API_KEY_FILE,
        help=(
            "Optional file containing the OpenAI API key. The environment variable "
            "takes precedence. Supports either a raw key or OPENAI_API_KEY=... format."
        ),
    )
    parser.add_argument(
        "--api_url",
        default=os.environ.get("OPENAI_CHAT_COMPLETIONS_URL", OPENAI_CHAT_COMPLETIONS_URL),
        help="Chat Completions endpoint URL.",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=700,
        help="Maximum output tokens per judge call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the judge call. Use --omit_temperature to leave it unset.",
    )
    parser.add_argument(
        "--omit_temperature",
        action="store_true",
        help="Do not send a temperature field in the OpenAI request.",
    )
    parser.add_argument(
        "--reasoning_effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Optional reasoning_effort value for models that support it.",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=90.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=4,
        help="Retries for rate limits, server errors, transport errors, or invalid JSON.",
    )
    parser.add_argument(
        "--retry_sleep",
        type=float,
        default=2.0,
        help="Base sleep in seconds before exponential-backoff retries.",
    )
    parser.add_argument(
        "--max_claims",
        type=int,
        default=DEFAULT_MAX_CLAIMS,
        help="Maximum number of atomic clinical evidence claims to extract per FP16 example.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of examples per model/seed/system after matching and slicing.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="0-based inclusive start index after matching rows.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="0-based exclusive end index after matching rows.",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip missing prediction files instead of raising an error.",
    )
    parser.add_argument(
        "--include_different_answers",
        action="store_true",
        help=(
            "Evaluate all matched examples. By default, coverage is computed only on "
            "examples where FP16, MedMix PTQ, and EAQuant predict the same answer label."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-example outputs instead of resuming/skipping done rows.",
    )
    parser.add_argument(
        "--dry_run_prompts",
        action="store_true",
        help="Write prompt JSONL files and summaries without calling the OpenAI API.",
    )
    parser.add_argument(
        "--verbose_prompts",
        action="store_true",
        help="Store full system/user prompts in normal judge JSONL outputs.",
    )
    args = parser.parse_args()

    data_dir = dataset_data_dir(args.dataset)
    if args.fp16_dir is None:
        args.fp16_dir = data_dir / "train_baseline"
    if args.medmix_baseline_dir is None:
        args.medmix_baseline_dir = data_dir / "train_quantized_medmix"
    if args.eaquant_dir is None:
        args.eaquant_dir = data_dir / "llm"
    if args.output_dir is None:
        args.output_dir = data_dir / "analysis" / "fp16_retention"
    args.dataset_display = str(DATASET_CONFIGS[args.dataset]["display"])
    return args


def safe_rate(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def pct(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}"


def format_mean_std_text(
    mean_value: Optional[float],
    std_value: Optional[float],
    num_runs: int,
) -> str:
    if mean_value is None:
        return ""
    if num_runs <= 1:
        return pct(mean_value)
    return f"{pct(mean_value)} +/- {pct(std_value)}"


def mean_std(values: Iterable[Any]) -> Tuple[Optional[float], Optional[float], int]:
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, (float, int)) and value is not None
    ]
    count = len(numeric_values)
    if count == 0:
        return None, None, 0
    mean_value = sum(numeric_values) / count
    if count == 1:
        return mean_value, 0.0, count
    variance = sum((value - mean_value) ** 2 for value in numeric_values) / (count - 1)
    return mean_value, variance**0.5, count


def build_messages(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def format_explanation_text(explanation: str) -> str:
    explanation = normalize_text(explanation)
    if explanation:
        return explanation
    return "[No explanation provided]"


def build_claim_extraction_prompt(
    *,
    fp16_record: Dict[str, Any],
    max_claims: int,
) -> str:
    question = normalize_text(fp16_record.get("question"))
    question_type = normalize_text(fp16_record.get("question_type"))
    return (
        "Extract answer-supporting clinical evidence claims from the FP16 rationale.\n\n"
        "Definition:\n"
        "- A claim is one atomic clinical evidence item or reasoning statement used by "
        "the FP16 rationale to support the selected answer.\n"
        "- Extract claims from the FP16 rationale itself. Use the question/options only "
        "for disambiguation.\n"
        "- Do not extract the answer letter itself as a claim.\n"
        "- Do not invent facts that are not stated or clearly implied by the FP16 rationale.\n"
        "- Prefer short, atomic claims. Split compound statements when needed.\n"
        f"- Return at most {max_claims} claims. If there is no clinical evidence, return an empty list.\n\n"
        f"Question type:\n{question_type or '[unknown]'}\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options(fp16_record.get('options'))}\n\n"
        "FP16 selected answer:\n"
        f"{format_answer(fp16_record)}\n\n"
        "FP16 rationale:\n"
        f"{format_explanation_text(extract_explanation(fp16_record))}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "claims": [\n'
        '    {"claim_id": "c1", "claim": "<atomic clinical evidence claim>", "evidence_type": "clinical_finding"}\n'
        "  ],\n"
        '  "reason": "<brief note on extraction>"\n'
        "}"
    )


def claims_for_prompt(claims: List[Dict[str, Any]]) -> str:
    lines = []
    for claim in claims:
        claim_id = normalize_text(claim.get("claim_id"))
        claim_text = normalize_text(claim.get("claim"))
        evidence_type = normalize_text(claim.get("evidence_type")) or "other"
        if claim_id and claim_text:
            lines.append(f"- {claim_id} [{evidence_type}]: {claim_text}")
    return "\n".join(lines) if lines else "[No FP16 clinical evidence claims extracted]"


def build_coverage_prompt(
    *,
    fp16_record: Dict[str, Any],
    candidate_record: Dict[str, Any],
    candidate_system: str,
    claims: List[Dict[str, Any]],
) -> str:
    question = normalize_text(fp16_record.get("question") or candidate_record.get("question"))
    options = fp16_record.get("options") or candidate_record.get("options") or {}
    question_type = normalize_text(
        fp16_record.get("question_type") or candidate_record.get("question_type")
    )
    return (
        "Judge whether a quantized rationale preserves each FP16 clinical evidence claim.\n\n"
        "Rubric:\n"
        '- Mark "supported" only if the candidate rationale explicitly states the same '
        "claim or a clear paraphrase that supports the same clinical point.\n"
        '- Mark "unsupported" if the claim is absent, too vague, or only recoverable by '
        "outside medical knowledge rather than the candidate rationale.\n"
        '- Mark "contradicted" if the candidate rationale says the opposite or uses '
        "incompatible evidence.\n"
        "- Matching the FP16 answer alone is not enough. The clinical evidence must be preserved.\n"
        "- Use the question/options only for context; do not credit a claim that is not in the candidate rationale.\n\n"
        f"Question type:\n{question_type or '[unknown]'}\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options(options)}\n\n"
        "FP16 selected answer:\n"
        f"{format_answer(fp16_record)}\n\n"
        "FP16 rationale:\n"
        f"{format_explanation_text(extract_explanation(fp16_record))}\n\n"
        "FP16 clinical evidence claims to judge:\n"
        f"{claims_for_prompt(claims)}\n\n"
        f"{PAPER_JUDGE_SYSTEM_DISPLAY.get(candidate_system, candidate_system)} selected answer:\n"
        f"{format_answer(candidate_record)}\n\n"
        f"{PAPER_JUDGE_SYSTEM_DISPLAY.get(candidate_system, candidate_system)} rationale:\n"
        f"{format_explanation_text(extract_explanation(candidate_record))}\n\n"
        "Return JSON only, with one result for every claim_id:\n"
        "{\n"
        '  "claim_results": [\n'
        '    {"claim_id": "c1", "support": "supported", "reason": "<short reason>"}\n'
        "  ],\n"
        '  "reason": "<brief overall note>"\n'
        "}"
    )


def normalize_claims(parsed: Dict[str, Any], max_claims: int) -> Tuple[List[Dict[str, Any]], str]:
    raw_claims = parsed.get("claims")
    if not isinstance(raw_claims, list):
        return [], normalize_text(parsed.get("reason"))

    claims: List[Dict[str, Any]] = []
    seen_ids = set()
    for idx, item in enumerate(raw_claims, start=1):
        if not isinstance(item, dict):
            continue
        claim = normalize_text(item.get("claim"))
        if not claim:
            continue
        claim_id = normalize_text(item.get("claim_id")) or f"c{idx}"
        if claim_id in seen_ids:
            claim_id = f"c{idx}"
        seen_ids.add(claim_id)
        evidence_type = normalize_text(item.get("evidence_type")) or "other"
        if evidence_type not in {
            "clinical_finding",
            "lab_finding",
            "diagnosis",
            "treatment",
            "risk_factor",
            "pathophysiology",
            "other",
        }:
            evidence_type = "other"
        claims.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "evidence_type": evidence_type,
            }
        )
        if len(claims) >= max_claims:
            break
    return claims, normalize_text(parsed.get("reason"))


def normalize_coverage(
    parsed: Dict[str, Any],
    claims: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    raw_results = parsed.get("claim_results")
    if not isinstance(raw_results, list):
        raw_results = []

    by_id: Dict[str, Dict[str, Any]] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        claim_id = normalize_text(item.get("claim_id"))
        if claim_id and claim_id not in by_id:
            by_id[claim_id] = item

    normalized_results: List[Dict[str, Any]] = []
    for claim in claims:
        claim_id = normalize_text(claim.get("claim_id"))
        item = by_id.get(claim_id, {})
        support = normalize_text(item.get("support")).lower()
        if support not in {"supported", "unsupported", "contradicted"}:
            support = "unsupported"
        normalized_results.append(
            {
                "claim_id": claim_id,
                "claim": normalize_text(claim.get("claim")),
                "evidence_type": normalize_text(claim.get("evidence_type")),
                "support": support,
                "reason": normalize_text(item.get("reason")),
            }
        )
    return normalized_results, normalize_text(parsed.get("reason"))


def run_openai_json_call(
    *,
    args: argparse.Namespace,
    api_key: str,
    task_name: str,
    messages: List[Dict[str, str]],
    response_format: Dict[str, Any],
) -> Dict[str, Any]:
    temperature = None if args.omit_temperature else args.temperature
    last_error = ""
    raw_response = ""

    for attempt in range(args.max_retries + 1):
        try:
            response_payload, raw_response = call_openai_chat_completion(
                api_key=api_key,
                api_url=args.api_url,
                judge_model=args.judge_model,
                messages=messages,
                response_format=response_format,
                max_completion_tokens=args.max_completion_tokens,
                temperature=temperature,
                reasoning_effort=args.reasoning_effort,
                request_timeout=args.request_timeout,
            )
            parsed = parse_json_object(raw_response)
            if parsed is None:
                last_error = "Could not parse JSON object from model response"
                raise ValueError(last_error)
            return {
                "parsed": parsed,
                "judge_parse_success": True,
                "judge_attempt_count": attempt + 1,
                "judge_error": "",
                "raw_judge_response": raw_response,
                "openai_response_id": normalize_text(response_payload.get("id")),
                "openai_usage": response_payload.get("usage") or {},
            }
        except urllib.error.HTTPError as exc:
            body = read_error_body(exc)
            last_error = f"HTTP {exc.code}: {body[:1000]}"
            if not should_retry_status(exc.code) or attempt >= args.max_retries:
                break
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt >= args.max_retries:
                break

        sleep_s = args.retry_sleep * (2**attempt)
        print(
            f"[WARN] {task_name} judge attempt {attempt + 1} failed: {last_error}",
            file=sys.stderr,
        )
        print(f"[WARN] sleeping {sleep_s:.1f}s before retry", file=sys.stderr)
        time.sleep(sleep_s)

    return {
        "parsed": {},
        "judge_parse_success": False,
        "judge_attempt_count": args.max_retries + 1,
        "judge_error": last_error,
        "raw_judge_response": raw_response,
        "openai_response_id": "",
        "openai_usage": {},
    }


def common_example_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pair_key": key_to_str(record_key(record)),
        "example_id": record.get("example_id"),
        "split": record.get("split"),
        "question_type": record.get("question_type"),
        "source_file": record.get("source_file"),
        "row_idx": record.get("row_idx"),
        "question": normalize_text(record.get("question")),
        "options": record.get("options") or {},
        "gold_answer": extract_gold_answer(record),
        "fp16_pred_answer": extract_pred_answer(record),
        "fp16_pred_answer_text": format_answer(record),
        "fp16_is_correct": infer_correct(record),
        "fp16_explanation": extract_explanation(record),
    }


def claim_output_paths(
    *,
    output_dir: Path,
    dataset: str,
    model_key: str,
    judge_model: str,
    dry_run_prompts: bool,
) -> Tuple[Path, Path]:
    stem = f"fp16_claims_{dataset}_{model_key}_by_{slugify(judge_model)}"
    if dry_run_prompts:
        stem = f"{stem}_prompts"
    return output_dir / f"{stem}.jsonl", output_dir / f"{stem}_summary.json"


def coverage_output_paths(
    *,
    output_dir: Path,
    dataset: str,
    model_key: str,
    seed: int,
    system: str,
    judge_model: str,
    dry_run_prompts: bool,
) -> Tuple[Path, Path]:
    stem = f"fec_support_{dataset}_{model_key}_{system}_seed{seed}_by_{slugify(judge_model)}"
    if dry_run_prompts:
        stem = f"{stem}_prompts"
    return output_dir / f"{stem}.jsonl", output_dir / f"{stem}_summary.json"


def read_completed_pair_keys(path: Path) -> set:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            pair_key = normalize_text(parsed.get("pair_key"))
            if pair_key:
                completed.add(pair_key)
    return completed


def load_existing_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def apply_slice_and_limit(items: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    sliced = items[args.start_idx : args.end_idx]
    if args.limit is not None:
        sliced = sliced[: args.limit]
    return sliced


def all_system_same_answer_pair_keys_for_claims(
    *,
    args: argparse.Namespace,
    fp16_records: List[Dict[str, Any]],
    model_key: str,
) -> Tuple[Optional[set], Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "enabled": not bool(args.include_different_answers),
        "required_same_predicted_answer_labels": ["fp16", "medmix_baseline", "eaquant"],
        "num_fp16_rows_before_answer_filter": len(fp16_records),
    }
    if args.include_different_answers:
        stats["num_fp16_rows_after_answer_filter"] = len(fp16_records)
        return None, stats

    fp16_by_key = {
        key_to_str(record_key(record)): record
        for record in fp16_records
        if record_key(record) is not None
    }
    union_pair_keys = set()
    per_seed: Dict[str, Any] = {}

    for seed in args.seeds:
        baseline_file = medmix_baseline_prediction_path(args.medmix_baseline_dir, args.dataset, model_key, seed)
        eaquant_file = eaquant_prediction_path(args.eaquant_dir, args.dataset, model_key, seed)
        seed_stats: Dict[str, Any] = {
            "medmix_baseline_file": str(baseline_file),
            "eaquant_file": str(eaquant_file),
        }
        missing_files = [str(path) for path in (baseline_file, eaquant_file) if not path.exists()]
        if missing_files:
            if args.skip_missing:
                seed_stats["skipped_missing_files"] = missing_files
                seed_stats["num_all_system_same_answer_pairs"] = 0
                per_seed[str(seed)] = seed_stats
                continue
            raise FileNotFoundError(missing_files[0])

        baseline_index, baseline_index_stats = build_index(read_jsonl(baseline_file))
        eaquant_index, eaquant_index_stats = build_index(read_jsonl(eaquant_file))
        baseline_by_key = {key_to_str(key): record for key, record in baseline_index.items()}
        eaquant_by_key = {key_to_str(key): record for key, record in eaquant_index.items()}

        seed_pair_keys = set()
        num_missing_candidate_match = 0
        num_missing_answer = 0
        num_different_answer_triplets = 0

        for pair_key, fp16_record in fp16_by_key.items():
            baseline_record = baseline_by_key.get(pair_key)
            eaquant_record = eaquant_by_key.get(pair_key)
            if baseline_record is None or eaquant_record is None:
                num_missing_candidate_match += 1
                continue

            answers = [
                normalize_text(extract_pred_answer(fp16_record)),
                normalize_text(extract_pred_answer(baseline_record)),
                normalize_text(extract_pred_answer(eaquant_record)),
            ]
            if not all(answers):
                num_missing_answer += 1
                continue
            if len(set(answers)) == 1:
                seed_pair_keys.add(pair_key)
            else:
                num_different_answer_triplets += 1

        union_pair_keys.update(seed_pair_keys)
        seed_stats.update(
            {
                "medmix_baseline_index": baseline_index_stats,
                "eaquant_index": eaquant_index_stats,
                "num_missing_candidate_match": num_missing_candidate_match,
                "num_missing_answer": num_missing_answer,
                "num_different_answer_triplets": num_different_answer_triplets,
                "num_all_system_same_answer_pairs": len(seed_pair_keys),
            }
        )
        per_seed[str(seed)] = seed_stats

    stats["per_seed"] = per_seed
    stats["num_fp16_rows_after_answer_filter"] = len(union_pair_keys)
    return union_pair_keys, stats


def run_claim_extraction_for_model(
    *,
    args: argparse.Namespace,
    api_key: str,
    model_key: str,
) -> Dict[str, Any]:
    fp16_file = fp16_prediction_path(args.fp16_dir, args.dataset, model_key)
    if not fp16_file.exists():
        if args.skip_missing:
            return {
                "task": "claims",
                "dataset": args.dataset,
                "model_key": model_key,
                "skipped_missing_file": str(fp16_file),
            }
        raise FileNotFoundError(fp16_file)

    output_jsonl, summary_json = claim_output_paths(
        output_dir=args.output_dir,
        dataset=args.dataset,
        model_key=model_key,
        judge_model=args.judge_model,
        dry_run_prompts=args.dry_run_prompts,
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_jsonl.exists():
        output_jsonl.unlink()

    fp16_records = read_jsonl(fp16_file)
    fp16_records = [record for record in fp16_records if record_key(record) is not None]
    answer_filter_pair_keys, answer_filter_stats = all_system_same_answer_pair_keys_for_claims(
        args=args,
        fp16_records=fp16_records,
        model_key=model_key,
    )
    if answer_filter_pair_keys is not None:
        fp16_records = [
            record
            for record in fp16_records
            if key_to_str(record_key(record)) in answer_filter_pair_keys
        ]
    selected_records = apply_slice_and_limit(fp16_records, args)
    active_pair_keys = {key_to_str(record_key(record)) for record in selected_records}
    completed_pair_keys = read_completed_pair_keys(output_jsonl) & active_pair_keys
    existing_records = [
        record
        for record in load_existing_records(output_jsonl)
        if normalize_text(record.get("pair_key")) in active_pair_keys
    ]
    new_records: List[Dict[str, Any]] = []

    print(
        f"[INFO] claims {args.dataset} {model_key}: "
        f"{len(selected_records)} FP16 rows, {len(completed_pair_keys)} already done"
    )

    for idx, fp16_record in enumerate(selected_records, start=1):
        pair_key = key_to_str(record_key(fp16_record))
        if pair_key in completed_pair_keys and not args.overwrite:
            continue

        user_prompt = build_claim_extraction_prompt(
            fp16_record=fp16_record,
            max_claims=args.max_claims,
        )
        messages = build_messages(user_prompt)
        record = {
            "task": "claims",
            "dataset": args.dataset,
            "dataset_display": args.dataset_display,
            "model_key": model_key,
            "judge_model": args.judge_model,
            "judge_api_url": args.api_url,
            "judge_model_call_skipped": bool(args.dry_run_prompts),
            "fp16_file": str(fp16_file),
            "all_system_same_answer_filter_enabled": not bool(args.include_different_answers),
            **common_example_fields(fp16_record),
        }
        if args.verbose_prompts or args.dry_run_prompts:
            record["system_prompt"] = SYSTEM_PROMPT
            record["user_prompt"] = user_prompt

        if args.dry_run_prompts:
            claims: List[Dict[str, Any]] = []
            extraction_reason = ""
            call_result = {
                "judge_parse_success": None,
                "judge_attempt_count": 0,
                "judge_error": "",
                "raw_judge_response": "",
                "openai_response_id": "",
                "openai_usage": {},
            }
        else:
            call_result = run_openai_json_call(
                args=args,
                api_key=api_key,
                task_name="claim_extraction",
                messages=messages,
                response_format=CLAIM_EXTRACTION_SCHEMA,
            )
            claims, extraction_reason = normalize_claims(
                call_result.pop("parsed", {}),
                args.max_claims,
            )

        record.update(call_result)
        record.update(
            {
                "claims": claims,
                "num_claims": len(claims),
                "claim_extraction_reason": extraction_reason,
            }
        )
        append_jsonl(output_jsonl, record)
        new_records.append(record)

        if args.dry_run_prompts:
            print(f"[{idx}/{len(selected_records)}] wrote claim prompt {model_key} {pair_key}")
        else:
            print(
                f"[{idx}/{len(selected_records)}] claims {model_key} {pair_key}: "
                f"{len(claims)} claims"
            )

    all_records = existing_records + new_records if not args.overwrite else new_records
    summary = summarize_claim_records(
        records=all_records,
        dry_run_prompts=args.dry_run_prompts,
    )
    summary.update(
        {
            "task": "claims",
            "dataset": args.dataset,
            "dataset_display": args.dataset_display,
            "model_key": model_key,
            "judge_model": args.judge_model,
            "dry_run_prompts": bool(args.dry_run_prompts),
            "fp16_file": str(fp16_file),
            "output_jsonl": str(output_jsonl),
            "summary_json": str(summary_json),
            "all_system_same_answer_filter_enabled": not bool(args.include_different_answers),
            "answer_filter": answer_filter_stats,
            "num_fp16_rows": len(fp16_records),
            "num_selected_rows": len(selected_records),
            "num_existing_records_before_run": len(existing_records),
            "num_new_records_this_run": len(new_records),
            "start_idx": args.start_idx,
            "end_idx": args.end_idx,
            "limit": args.limit,
        }
    )
    write_json(summary_json, summary)
    print(f"[DONE] wrote {output_jsonl}")
    print(f"[DONE] wrote {summary_json}")
    return summary


def summarize_claim_records(
    *,
    records: List[Dict[str, Any]],
    dry_run_prompts: bool,
) -> Dict[str, Any]:
    parse_failures = sum(1 for record in records if record.get("judge_parse_success") is False)
    model_calls = sum(1 for record in records if record.get("judge_model_call_skipped") is not True)
    records_with_claims = [
        record for record in records if int(record.get("num_claims") or 0) > 0
    ]
    total_claims = sum(int(record.get("num_claims") or 0) for record in records)
    evidence_types = Counter()
    for record in records:
        claims = record.get("claims")
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if isinstance(claim, dict):
                evidence_types[normalize_text(claim.get("evidence_type")) or "other"] += 1
    return {
        "num_records": len(records),
        "dry_run_prompts": bool(dry_run_prompts),
        "num_model_calls": model_calls,
        "num_parse_failures": parse_failures,
        "num_records_with_claims": len(records_with_claims),
        "num_records_without_claims": len(records) - len(records_with_claims),
        "total_claims": total_claims,
        "avg_claims_per_record": (
            total_claims / len(records) if records else None
        ),
        "avg_claims_per_claimed_record": (
            total_claims / len(records_with_claims) if records_with_claims else None
        ),
        "evidence_type_counts": dict(sorted(evidence_types.items())),
    }


def load_claim_records_for_model(
    *,
    args: argparse.Namespace,
    model_key: str,
) -> Dict[str, Dict[str, Any]]:
    candidate_paths = []
    if args.dry_run_prompts:
        candidate_paths.append(
            claim_output_paths(
                output_dir=args.output_dir,
                dataset=args.dataset,
                model_key=model_key,
                judge_model=args.judge_model,
                dry_run_prompts=False,
            )[0]
        )
    candidate_paths.append(
        claim_output_paths(
            output_dir=args.output_dir,
            dataset=args.dataset,
            model_key=model_key,
            judge_model=args.judge_model,
            dry_run_prompts=args.dry_run_prompts,
        )[0]
    )

    claims_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
    if not claims_path.exists():
        raise FileNotFoundError(
            f"Missing FP16 claim cache for {model_key}: {claims_path}. "
            "Run --mode claims or --mode both first."
        )
    records = read_jsonl(claims_path)
    by_pair_key: Dict[str, Dict[str, Any]] = {}
    for record in records:
        pair_key = normalize_text(record.get("pair_key"))
        if pair_key and pair_key not in by_pair_key:
            by_pair_key[pair_key] = record
    return by_pair_key


def load_claim_summary_for_model(
    *,
    args: argparse.Namespace,
    model_key: str,
) -> Optional[Dict[str, Any]]:
    candidate_paths = []
    if args.dry_run_prompts:
        candidate_paths.append(
            claim_output_paths(
                output_dir=args.output_dir,
                dataset=args.dataset,
                model_key=model_key,
                judge_model=args.judge_model,
                dry_run_prompts=False,
            )[1]
        )
    candidate_paths.append(
        claim_output_paths(
            output_dir=args.output_dir,
            dataset=args.dataset,
            model_key=model_key,
            judge_model=args.judge_model,
            dry_run_prompts=args.dry_run_prompts,
        )[1]
    )

    for path in candidate_paths:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                parsed = json.load(f)
            return parsed if isinstance(parsed, dict) else None
    return None


def matched_pairs_for_system(
    *,
    fp16_records: List[Dict[str, Any]],
    candidate_records: List[Dict[str, Any]],
    claim_records_by_pair_key: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_index, candidate_index_stats = build_index(candidate_records)
    matched: List[Dict[str, Any]] = []
    stats = {
        "num_fp16_rows": len(fp16_records),
        "num_candidate_rows": len(candidate_records),
        "candidate_index": candidate_index_stats,
        "num_fp16_rows_missing_key": 0,
        "num_missing_candidate_match": 0,
        "num_missing_claim_cache": 0,
    }

    for fp16_record in fp16_records:
        key = record_key(fp16_record)
        if key is None:
            stats["num_fp16_rows_missing_key"] += 1
            continue
        pair_key = key_to_str(key)
        candidate_record = candidate_index.get(key)
        if candidate_record is None:
            stats["num_missing_candidate_match"] += 1
            continue
        claim_record = claim_records_by_pair_key.get(pair_key)
        if claim_record is None:
            stats["num_missing_claim_cache"] += 1
            continue
        matched.append(
            {
                "pair_key": pair_key,
                "fp16_record": fp16_record,
                "candidate_record": candidate_record,
                "claim_record": claim_record,
            }
        )

    stats["num_matched_pairs"] = len(matched)
    return matched, stats


def coverage_prediction_path(
    *,
    args: argparse.Namespace,
    model_key: str,
    seed: int,
    system: str,
) -> Path:
    if system == "medmix_baseline":
        return medmix_baseline_prediction_path(args.medmix_baseline_dir, args.dataset, model_key, seed)
    if system == "eaquant":
        return eaquant_prediction_path(args.eaquant_dir, args.dataset, model_key, seed)
    raise ValueError(f"Unsupported coverage system: {system}")


def records_by_pair_key(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    indexed, index_stats = build_index(records)
    return {key_to_str(key): record for key, record in indexed.items()}, index_stats


def maybe_filter_all_system_same_answer_pairs(
    *,
    args: argparse.Namespace,
    matched_pairs: List[Dict[str, Any]],
    match_stats: Dict[str, Any],
    model_key: str,
    seed: int,
    system: str,
) -> List[Dict[str, Any]]:
    match_stats["num_matched_pairs_before_answer_filter"] = len(matched_pairs)
    match_stats["all_system_same_answer_filter_enabled"] = not bool(
        args.include_different_answers
    )
    if args.include_different_answers:
        match_stats["num_matched_pairs_after_answer_filter"] = len(matched_pairs)
        return matched_pairs

    other_system = "eaquant" if system == "medmix_baseline" else "medmix_baseline"
    other_file = coverage_prediction_path(
        args=args,
        model_key=model_key,
        seed=seed,
        system=other_system,
    )
    if not other_file.exists():
        if args.skip_missing:
            match_stats["answer_filter_skipped_missing_other_file"] = str(other_file)
            match_stats["num_matched_pairs_after_answer_filter"] = 0
            return []
        raise FileNotFoundError(other_file)

    other_records = read_jsonl(other_file)
    other_by_key, other_index_stats = records_by_pair_key(other_records)
    filtered: List[Dict[str, Any]] = []
    num_missing_other_match = 0
    num_missing_answer = 0
    num_different_answer_triplets = 0

    for pair in matched_pairs:
        pair_key = normalize_text(pair.get("pair_key"))
        fp16_record = pair.get("fp16_record") or {}
        candidate_record = pair.get("candidate_record") or {}
        other_record = other_by_key.get(pair_key)
        if other_record is None:
            num_missing_other_match += 1
            continue

        answers = [
            normalize_text(extract_pred_answer(fp16_record)),
            normalize_text(extract_pred_answer(candidate_record)),
            normalize_text(extract_pred_answer(other_record)),
        ]
        if not all(answers):
            num_missing_answer += 1
            continue
        if len(set(answers)) == 1:
            filtered.append(pair)
        else:
            num_different_answer_triplets += 1

    match_stats["answer_filter"] = {
        "required_same_predicted_answer_labels": ["fp16", "medmix_baseline", "eaquant"],
        "other_system": other_system,
        "other_file": str(other_file),
        "other_index": other_index_stats,
        "num_missing_other_match": num_missing_other_match,
        "num_missing_answer": num_missing_answer,
        "num_different_answer_triplets": num_different_answer_triplets,
    }
    match_stats["num_all_system_same_answer_pairs"] = len(filtered)
    match_stats["num_matched_pairs_after_answer_filter"] = len(filtered)
    return filtered


def score_claim_results(claim_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    num_claims = len(claim_results)
    supported = sum(1 for item in claim_results if item.get("support") == "supported")
    unsupported = sum(1 for item in claim_results if item.get("support") == "unsupported")
    contradicted = sum(1 for item in claim_results if item.get("support") == "contradicted")
    return {
        "num_claims": num_claims,
        "supported_count": supported,
        "unsupported_count": unsupported,
        "contradicted_count": contradicted,
        "fec_score": safe_rate(supported, num_claims),
    }


def unjudged_claim_results(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "claim_id": normalize_text(claim.get("claim_id")),
            "claim": normalize_text(claim.get("claim")),
            "evidence_type": normalize_text(claim.get("evidence_type")),
            "support": "unjudged",
            "reason": "Judge call was skipped or failed.",
        }
        for claim in claims
    ]


def run_coverage_for_model_seed_system(
    *,
    args: argparse.Namespace,
    api_key: str,
    model_key: str,
    seed: int,
    system: str,
) -> Dict[str, Any]:
    fp16_file = fp16_prediction_path(args.fp16_dir, args.dataset, model_key)
    candidate_file = coverage_prediction_path(
        args=args,
        model_key=model_key,
        seed=seed,
        system=system,
    )

    for path in (fp16_file, candidate_file):
        if not path.exists():
            if args.skip_missing:
                return {
                    "task": "coverage",
                    "dataset": args.dataset,
                    "model_key": model_key,
                    "seed": seed,
                    "system": system,
                    "skipped_missing_file": str(path),
                }
            raise FileNotFoundError(path)

    claim_records_by_pair_key = load_claim_records_for_model(args=args, model_key=model_key)
    output_jsonl, summary_json = coverage_output_paths(
        output_dir=args.output_dir,
        dataset=args.dataset,
        model_key=model_key,
        seed=seed,
        system=system,
        judge_model=args.judge_model,
        dry_run_prompts=args.dry_run_prompts,
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_jsonl.exists():
        output_jsonl.unlink()

    fp16_records = read_jsonl(fp16_file)
    candidate_records = read_jsonl(candidate_file)
    matched_pairs, match_stats = matched_pairs_for_system(
        fp16_records=fp16_records,
        candidate_records=candidate_records,
        claim_records_by_pair_key=claim_records_by_pair_key,
    )
    matched_pairs = maybe_filter_all_system_same_answer_pairs(
        args=args,
        matched_pairs=matched_pairs,
        match_stats=match_stats,
        model_key=model_key,
        seed=seed,
        system=system,
    )
    selected_pairs = apply_slice_and_limit(matched_pairs, args)
    active_pair_keys = {normalize_text(item.get("pair_key")) for item in selected_pairs}
    completed_pair_keys = read_completed_pair_keys(output_jsonl) & active_pair_keys
    existing_records = [
        record
        for record in load_existing_records(output_jsonl)
        if normalize_text(record.get("pair_key")) in active_pair_keys
    ]
    new_records: List[Dict[str, Any]] = []

    print(
        f"[INFO] coverage {args.dataset} {model_key} seed{seed} {system}: "
        f"{len(selected_pairs)} pairs, {len(completed_pair_keys)} already done"
    )

    for idx, item in enumerate(selected_pairs, start=1):
        pair_key = normalize_text(item.get("pair_key"))
        if pair_key in completed_pair_keys and not args.overwrite:
            continue

        fp16_record = item["fp16_record"]
        candidate_record = item["candidate_record"]
        claim_record = item["claim_record"]
        claims = claim_record.get("claims")
        if not isinstance(claims, list):
            claims = []
        claims = [claim for claim in claims if isinstance(claim, dict)]

        user_prompt = build_coverage_prompt(
            fp16_record=fp16_record,
            candidate_record=candidate_record,
            candidate_system=system,
            claims=claims,
        )
        messages = build_messages(user_prompt)
        record = {
            "task": "coverage",
            "dataset": args.dataset,
            "dataset_display": args.dataset_display,
            "model_key": model_key,
            "seed": seed,
            "system": system,
            "system_display": SYSTEM_DISPLAY.get(system, system),
            "judge_model": args.judge_model,
            "judge_api_url": args.api_url,
            "judge_model_call_skipped": bool(args.dry_run_prompts),
            "fp16_file": str(fp16_file),
            "candidate_file": str(candidate_file),
            "pair_key": pair_key,
            "example_id": fp16_record.get("example_id") or candidate_record.get("example_id"),
            "split": fp16_record.get("split") or candidate_record.get("split"),
            "question_type": fp16_record.get("question_type") or candidate_record.get("question_type"),
            "source_file": fp16_record.get("source_file") or candidate_record.get("source_file"),
            "row_idx": fp16_record.get("row_idx")
            if fp16_record.get("row_idx") is not None
            else candidate_record.get("row_idx"),
            "question": normalize_text(fp16_record.get("question") or candidate_record.get("question")),
            "options": fp16_record.get("options") or candidate_record.get("options") or {},
            "gold_answer": extract_gold_answer(fp16_record) or extract_gold_answer(candidate_record),
            "fp16_pred_answer": extract_pred_answer(fp16_record),
            "fp16_pred_answer_text": format_answer(fp16_record),
            "fp16_is_correct": infer_correct(fp16_record),
            "fp16_explanation": extract_explanation(fp16_record),
            "candidate_pred_answer": extract_pred_answer(candidate_record),
            "candidate_pred_answer_text": format_answer(candidate_record),
            "candidate_is_correct": infer_correct(candidate_record),
            "candidate_explanation": extract_explanation(candidate_record),
            "claims": claims,
        }
        if args.verbose_prompts or args.dry_run_prompts:
            record["system_prompt"] = SYSTEM_PROMPT
            record["user_prompt"] = user_prompt

        if args.dry_run_prompts:
            claim_results = unjudged_claim_results(claims)
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
            call_result = run_openai_json_call(
                args=args,
                api_key=api_key,
                task_name="coverage",
                messages=messages,
                response_format=COVERAGE_SCHEMA,
            )
            if call_result.get("judge_parse_success") is True:
                claim_results, coverage_reason = normalize_coverage(
                    call_result.pop("parsed", {}),
                    claims,
                )
            else:
                call_result.pop("parsed", None)
                claim_results = unjudged_claim_results(claims)
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
            score = score_claim_results(claim_results)

        record.update(call_result)
        record.update(score)
        record.update(
            {
                "claim_results": claim_results,
                "coverage_reason": coverage_reason,
            }
        )
        append_jsonl(output_jsonl, record)
        new_records.append(record)

        if args.dry_run_prompts:
            print(
                f"[{idx}/{len(selected_pairs)}] wrote coverage prompt "
                f"{model_key} seed{seed} {system} {pair_key}"
            )
        else:
            score_text = pct(record.get("fec_score")) if record.get("fec_score") is not None else "NA"
            print(
                f"[{idx}/{len(selected_pairs)}] coverage {model_key} seed{seed} "
                f"{system} {pair_key}: FEC={score_text}"
            )

    all_records = existing_records + new_records if not args.overwrite else new_records
    summary = summarize_coverage_records(
        records=all_records,
        dry_run_prompts=args.dry_run_prompts,
    )
    summary.update(
        {
            "task": "coverage",
            "dataset": args.dataset,
            "dataset_display": args.dataset_display,
            "model_key": model_key,
            "seed": seed,
            "system": system,
            "system_display": SYSTEM_DISPLAY.get(system, system),
            "judge_model": args.judge_model,
            "all_system_same_answer_filter_enabled": not bool(args.include_different_answers),
            "dry_run_prompts": bool(args.dry_run_prompts),
            "fp16_file": str(fp16_file),
            "candidate_file": str(candidate_file),
            "output_jsonl": str(output_jsonl),
            "summary_json": str(summary_json),
            "match_stats": match_stats,
            "num_selected_pairs": len(selected_pairs),
            "num_existing_records_before_run": len(existing_records),
            "num_new_records_this_run": len(new_records),
            "start_idx": args.start_idx,
            "end_idx": args.end_idx,
            "limit": args.limit,
        }
    )
    write_json(summary_json, summary)
    print(f"[DONE] wrote {output_jsonl}")
    print(f"[DONE] wrote {summary_json}")
    return summary


def summarize_coverage_records(
    *,
    records: List[Dict[str, Any]],
    dry_run_prompts: bool,
) -> Dict[str, Any]:
    scored_records = [
        record
        for record in records
        if isinstance(record.get("fec_score"), (float, int))
        and int(record.get("num_claims") or 0) > 0
    ]
    all_claim_records = [
        record for record in records if int(record.get("num_claims") or 0) > 0
    ]
    supported = sum(int(record.get("supported_count") or 0) for record in scored_records)
    unsupported = sum(int(record.get("unsupported_count") or 0) for record in scored_records)
    contradicted = sum(int(record.get("contradicted_count") or 0) for record in scored_records)
    total_claims = sum(int(record.get("num_claims") or 0) for record in scored_records)
    parse_failures = sum(1 for record in records if record.get("judge_parse_success") is False)
    model_calls = sum(1 for record in records if record.get("judge_model_call_skipped") is not True)
    support_counts = Counter()
    for record in records:
        claim_results = record.get("claim_results")
        if not isinstance(claim_results, list):
            continue
        for item in claim_results:
            if isinstance(item, dict):
                support_counts[normalize_text(item.get("support")) or "missing"] += 1

    macro_fec = (
        sum(float(record["fec_score"]) for record in scored_records) / len(scored_records)
        if scored_records
        else None
    )
    fp16_correct_records = [
        record for record in scored_records if record.get("fp16_is_correct") is True
    ]
    fp16_correct_claims = sum(int(record.get("num_claims") or 0) for record in fp16_correct_records)
    fp16_correct_supported = sum(
        int(record.get("supported_count") or 0) for record in fp16_correct_records
    )
    return {
        "num_records": len(records),
        "dry_run_prompts": bool(dry_run_prompts),
        "num_model_calls": model_calls,
        "num_parse_failures": parse_failures,
        "num_records_with_claims": len(all_claim_records),
        "num_scored_records": len(scored_records),
        "num_unscored_records": len(records) - len(scored_records),
        "total_claims_scored": total_claims,
        "supported_count": supported,
        "unsupported_count": unsupported,
        "contradicted_count": contradicted,
        "support_counts": dict(sorted(support_counts.items())),
        "fec_macro_rate": macro_fec,
        "fec_micro_rate": safe_rate(supported, total_claims),
        "fp16_correct_only": {
            "num_scored_records": len(fp16_correct_records),
            "total_claims_scored": fp16_correct_claims,
            "supported_count": fp16_correct_supported,
            "fec_macro_rate": (
                sum(float(record["fec_score"]) for record in fp16_correct_records)
                / len(fp16_correct_records)
                if fp16_correct_records
                else None
            ),
            "fec_micro_rate": safe_rate(fp16_correct_supported, fp16_correct_claims),
        },
    }


def fp16_baseline_row_from_claim_summary(claim_summary: Dict[str, Any]) -> Dict[str, Any]:
    total_claims = int(claim_summary.get("total_claims") or 0)
    scored_records = int(claim_summary.get("num_records_with_claims") or 0)
    rate = 1.0 if total_claims > 0 else None
    return {
        "dataset": claim_summary.get("dataset"),
        "dataset_display": claim_summary.get("dataset_display"),
        "model_key": claim_summary.get("model_key"),
        "system": "fp16",
        "system_display": SYSTEM_DISPLAY.get("fp16", "FP16"),
        "seed": None,
        "num_runs": 1,
        "num_records": claim_summary.get("num_records"),
        "num_scored_records": scored_records,
        "total_claims_scored": total_claims,
        "supported_count": total_claims,
        "unsupported_count": 0,
        "contradicted_count": 0,
        "fec_macro_rate": rate,
        "fec_micro_rate": rate,
        "num_parse_failures": claim_summary.get("num_parse_failures"),
        "output_jsonl": claim_summary.get("output_jsonl"),
    }


def row_from_coverage_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
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


def aggregate_mean_std(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("model_key")), str(row.get("system")))].append(row)

    stats_rows: List[Dict[str, Any]] = []
    for (model_key, system), group in grouped.items():
        seeds = sorted(
            int(row["seed"]) for row in group if row.get("seed") is not None
        )
        stats_row: Dict[str, Any] = {
            "model_key": model_key,
            "system": system,
            "system_display": SYSTEM_DISPLAY.get(system, system),
            "num_runs": len(group),
            "seeds": ",".join(str(seed) for seed in seeds),
            "num_scored_records": sum(int(row.get("num_scored_records") or 0) for row in group),
            "total_claims_scored": sum(int(row.get("total_claims_scored") or 0) for row in group),
            "supported_count": sum(int(row.get("supported_count") or 0) for row in group),
            "num_parse_failures": sum(int(row.get("num_parse_failures") or 0) for row in group),
        }
        for metric_key, output_prefix in (
            ("fec_macro_rate", "fec_macro"),
            ("fec_micro_rate", "fec_micro"),
        ):
            mean_value, std_value, count = mean_std(row.get(metric_key) for row in group)
            stats_row[f"{output_prefix}_mean_rate"] = mean_value
            stats_row[f"{output_prefix}_std_rate"] = std_value
            stats_row[f"{output_prefix}_num_values"] = count
        stats_rows.append(stats_row)

    return sorted(
        stats_rows,
        key=lambda row: (
            row["model_key"],
            SYSTEM_ORDER.index(row["system"]) if row.get("system") in SYSTEM_ORDER else 99,
        ),
    )


def add_pct_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for row in rows:
        copied = dict(row)
        copied["fec_macro_pct"] = pct(copied.get("fec_macro_rate"))
        copied["fec_micro_pct"] = pct(copied.get("fec_micro_rate"))
        enriched.append(copied)
    return enriched


def add_mean_std_pct_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for row in rows:
        copied = dict(row)
        num_runs = int(copied.get("num_runs") or 0)
        for prefix in ("fec_macro", "fec_micro"):
            mean_key = f"{prefix}_mean_rate"
            std_key = f"{prefix}_std_rate"
            copied[f"{prefix}_mean_pct"] = pct(copied.get(mean_key))
            copied[f"{prefix}_std_pct"] = pct(copied.get(std_key))
            copied[f"{prefix}_mean_std_pct"] = format_mean_std_text(
                copied.get(mean_key),
                copied.get(std_key),
                num_runs,
            )
        enriched.append(copied)
    return enriched


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latex_mean_std_metric(row: Dict[str, Any], prefix: str) -> str:
    mean_value = row.get(f"{prefix}_mean_rate")
    num_runs = int(row.get("num_runs") or 0)
    if mean_value is None:
        return "--"
    if num_runs <= 1:
        return pct(mean_value)
    return f"${pct(mean_value)} \\pm {pct(row.get(f'{prefix}_std_rate'))}$"


def latex_by_model_table(rows: List[Dict[str, Any]], caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Model & System & FP16 EvCov & Claims \\",
        r"\midrule",
    ]
    current_model = None
    for row in rows:
        model_key = str(row.get("model_key"))
        if current_model is not None and model_key != current_model:
            lines.append(r"\midrule")
        current_model = model_key
        lines.append(
            f"{model_key} & "
            f"{SYSTEM_DISPLAY.get(str(row.get('system')), str(row.get('system')))} & "
            f"{latex_mean_std_metric(row, 'fec_macro')} & "
            f"{int(row.get('total_claims_scored') or 0)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def print_by_model_mean_std(rows: List[Dict[str, Any]]) -> None:
    print()
    print("Model, System, FP16 EvCov, Claims")
    for row in rows:
        num_runs = int(row.get("num_runs") or 0)
        print(
            f"{row['model_key']}, "
            f"{row['system_display']}, "
            f"{format_mean_std_text(row.get('fec_macro_mean_rate'), row.get('fec_macro_std_rate'), num_runs)}, "
            f"{int(row.get('total_claims_scored') or 0)}"
        )


def iter_modes(mode: str) -> Iterable[str]:
    if mode == "both":
        yield "claims"
        yield "coverage"
    else:
        yield mode


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key(args)

    claim_summaries: List[Dict[str, Any]] = []
    coverage_summaries: List[Dict[str, Any]] = []

    for mode in iter_modes(args.mode):
        if mode == "claims":
            for model_key in args.models:
                claim_summaries.append(
                    run_claim_extraction_for_model(
                        args=args,
                        api_key=api_key,
                        model_key=model_key,
                    )
                )
        elif mode == "coverage":
            for model_key in args.models:
                for seed in args.seeds:
                    for system in COVERAGE_SYSTEMS:
                        coverage_summaries.append(
                            run_coverage_for_model_seed_system(
                                args=args,
                                api_key=api_key,
                                model_key=model_key,
                                seed=seed,
                                system=system,
                            )
                        )

    rows: List[Dict[str, Any]] = []
    if args.mode in {"claims", "both"}:
        rows.extend(
            fp16_baseline_row_from_claim_summary(summary)
            for summary in claim_summaries
            if not summary.get("skipped_missing_file")
        )
    elif args.mode == "coverage":
        for model_key in args.models:
            claim_summary = load_claim_summary_for_model(args=args, model_key=model_key)
            if claim_summary is not None:
                rows.append(fp16_baseline_row_from_claim_summary(claim_summary))
    if args.mode in {"coverage", "both"}:
        rows.extend(
            row_from_coverage_summary(summary)
            for summary in coverage_summaries
            if not summary.get("skipped_missing_file")
        )

    by_model_mean_std = aggregate_mean_std(rows)
    rows_csv = add_pct_columns(rows)
    by_model_mean_std_csv = add_mean_std_pct_columns(by_model_mean_std)

    summary_path = args.output_dir / "fp16_evidence_coverage_summary.json"
    rows_csv_path = args.output_dir / "fp16_evidence_coverage_rows.csv"
    by_model_csv_path = args.output_dir / "fp16_evidence_coverage_by_model_mean_std.csv"
    latex_path = args.output_dir / "fp16_evidence_coverage_by_model_table.tex"

    write_json(
        summary_path,
        {
            "dataset": args.dataset,
            "dataset_display": args.dataset_display,
            "mode": args.mode,
            "fp16_dir": str(args.fp16_dir),
            "medmix_baseline_dir": str(args.medmix_baseline_dir),
            "eaquant_dir": str(args.eaquant_dir),
            "output_dir": str(args.output_dir),
            "models": args.models,
            "seeds": args.seeds,
            "judge_model": args.judge_model,
            "all_system_same_answer_filter_enabled": not bool(args.include_different_answers),
            "dry_run_prompts": bool(args.dry_run_prompts),
            "metric_definitions": {
                "fp16_evidence_coverage": (
                    "For each example, extract atomic answer-supporting clinical claims "
                    "from the FP16 rationale. FEC is the fraction of those claims that "
                    "the candidate rationale supports. By default, examples are included "
                    "only when FP16, MedMix PTQ, and EAQuant predict the same answer label. "
                    "Table values use macro average over examples with at least one "
                    "extracted FP16 claim."
                ),
                "fec_macro_rate": "Mean of per-example supported_claims / fp16_claims.",
                "fec_micro_rate": "Total supported claims divided by total FP16 claims.",
                "fp16_row": "FP16 is 100% by definition when at least one claim is extracted.",
            },
            "claim_summaries": claim_summaries,
            "coverage_summaries": coverage_summaries,
            "rows": rows,
            "by_model_mean_std": by_model_mean_std,
        },
    )
    write_csv(
        rows_csv_path,
        rows_csv,
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
        by_model_csv_path,
        by_model_mean_std_csv,
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
    latex_path.write_text(
        latex_by_model_table(
            by_model_mean_std,
            (
                f"{args.dataset_display} FP16 evidence coverage. FP16 EvCov measures "
                "whether each quantized rationale preserves the answer-supporting "
                "clinical evidence claims extracted from the full-precision rationale "
                "on examples where FP16, MedMix PTQ, and EAQuant predict the same answer label. "
                "Quantized rows report mean and standard deviation over seeds."
            ),
            f"tab:{args.dataset}_fp16_evidence_coverage_by_model",
        ),
        encoding="utf-8",
    )

    print_by_model_mean_std(by_model_mean_std)
    print(f"[DONE] wrote {summary_path}")
    print(f"[DONE] wrote {rows_csv_path}")
    print(f"[DONE] wrote {by_model_csv_path}")
    print(f"[DONE] wrote {latex_path}")


if __name__ == "__main__":
    main()
