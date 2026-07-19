import argparse
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
    extract_explanation,
    extract_gold_answer,
    extract_pred_answer,
    format_answer,
    format_options,
    key_to_str,
    normalize_text,
    read_jsonl,
    record_key,
    slugify,
    write_json,
)
from .fp16_agreement import (
    DATASET_CONFIGS,
    dataset_data_dir,
    fp16_prediction_path,
    medmix_baseline_prediction_path,
    eaquant_prediction_path,
)
from .fp16_retention import (
    build_messages,
    format_mean_std_text,
    load_api_key,
    mean_std,
    pct,
    run_openai_json_call,
    safe_rate,
    write_csv,
)


DATASET = "medexpqa"
DATASET_DISPLAY = str(DATASET_CONFIGS[DATASET]["display"])
DATA_DIR = dataset_data_dir(DATASET)
DEFAULT_FP16_DIR = DATA_DIR / "train_baseline"
DEFAULT_MEDMIX_BASELINE_DIR = DATA_DIR / "train_quantized_medmix"
DEFAULT_EAQUANT_DIR = DATA_DIR / "llm"
DEFAULT_OUTPUT_DIR = DATA_DIR / "analysis" / "unsupported_claims"
DEFAULT_MAX_CLAIMS = 20
SCHEMA_VERSION = "uacr_v4_fp16only_alllabelsame"
SYSTEM_ORDER = ("medmix_baseline", "eaquant")
SYSTEM_DISPLAY = {
    "medmix_baseline": "MedMix PTQ",
    "eaquant": "EAQuant",
}

SYSTEM_PROMPT = (
    "You are an expert medical evaluator for clinical multiple-choice QA. "
    "Follow the rubric exactly. Output only the requested JSON object."
)

CLAIM_TYPES = (
    "clinical_finding",
    "lab_finding",
    "diagnosis",
    "treatment",
    "risk_factor",
    "pathophysiology",
    "general_medical_fact",
    "answer_mapping",
    "other",
)
SUPPORT_LABELS = (
    "source_supported",
    "valid_background",
    "unsupported_added",
    "contradicted",
    "not_a_claim",
)
SUPPORT_STATUSES = SUPPORT_LABELS + ("unjudged",)
API_SUPPORT_STATUSES = SUPPORT_LABELS
FP16_OVERLAPS = ("present_in_fp16", "not_in_fp16", "unclear")
SUPPORT_SOURCES = (
    "fp16_rationale",
    "standard_medical_knowledge",
    "none",
    "not_applicable",
)
LEGACY_SUPPORT_STATUS_MAP = {
    "supported": "source_supported",
    "unsupported": "unsupported_added",
    "contradicted": "contradicted",
    "not_a_claim": "not_a_claim",
    "unjudged": "unjudged",
}

CLAIM_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "medexpqa_candidate_claim_extraction",
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
                            "claim_type": {"type": "string", "enum": list(CLAIM_TYPES)},
                        },
                        "required": ["claim_id", "claim", "claim_type"],
                    },
                },
                "reason": {"type": "string"},
                "truncated": {"type": "boolean"},
            },
            "required": ["claims", "reason", "truncated"],
        },
    },
}

SUPPORT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "medexpqa_uacr_judgment",
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
                            "fp16_overlap": {"type": "string", "enum": list(FP16_OVERLAPS)},
                            "teacher_support_status": {
                                "type": "string",
                                "enum": list(API_SUPPORT_STATUSES),
                            },
                            "teacher_support_source": {
                                "type": "string",
                                "enum": list(SUPPORT_SOURCES),
                            },
                            "clinical_support_status": {
                                "type": "string",
                                "enum": list(API_SUPPORT_STATUSES),
                            },
                            "clinical_support_source": {
                                "type": "string",
                                "enum": list(SUPPORT_SOURCES),
                            },
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "claim_id",
                            "fp16_overlap",
                            "teacher_support_status",
                            "teacher_support_source",
                            "clinical_support_status",
                            "clinical_support_source",
                            "reason",
                        ],
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
            "Compute MedExpQA unsupported added claim rate for MedMix PTQ and EAQuant outputs. "
            "This extracts claims from candidate rationales, compares them to the FP16 teacher "
            "rationale, and by default evaluates only examples where FP16 and all selected "
            "candidate systems predicted the same answer label."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["claims", "support", "both"],
        default="both",
        help=(
            "claims: extract candidate rationale claims only. "
            "support: judge existing candidate claims. "
            "both: run extraction then support judgment."
        ),
    )
    parser.add_argument(
        "--fp16_dir",
        type=Path,
        default=DEFAULT_FP16_DIR,
        help="Directory containing original train_baseline prediction JSONL files.",
    )
    parser.add_argument(
        "--medmix_baseline_dir",
        type=Path,
        default=DEFAULT_MEDMIX_BASELINE_DIR,
        help="Directory containing standard MedMix PTQ prediction files.",
    )
    parser.add_argument(
        "--eaquant_dir",
        type=Path,
        default=DEFAULT_EAQUANT_DIR,
        help=(
            "Directory containing EAQuant prediction files. Defaults to the same "
            "MedExpQA path logic used by the FP16 agreement evaluator."
        ),
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=SYSTEM_ORDER,
        default=list(SYSTEM_ORDER),
        help="Candidate systems to evaluate: medmix_baseline and/or eaquant.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-example JSONL and aggregate summaries are written.",
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
        help="Quantized seeds to evaluate for each selected system.",
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
        default=OPENAI_CHAT_COMPLETIONS_URL,
        help="Chat Completions endpoint URL.",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=900,
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
        "--incomplete_judgment_retries",
        type=int,
        default=2,
        help=(
            "Extra full judge-call retries when a parsed support judgment omits claim_ids "
            "or returns invalid labels that normalize to unjudged."
        ),
    )
    parser.add_argument(
        "--max_claims",
        type=int,
        default=DEFAULT_MAX_CLAIMS,
        help="Maximum candidate claims to extract per example.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of matched examples per model/seed after slicing.",
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
        "--include_different_fp16_answers",
        action="store_true",
        help=(
            "Evaluate all matched examples. By default, only examples where FP16 and "
            "all selected candidate systems predict the same answer label are evaluated."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-example outputs instead of resuming/skipping rows.",
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
    return parser.parse_args()


def format_explanation_text(record: Dict[str, Any]) -> str:
    explanation = normalize_text(extract_explanation(record))
    return explanation if explanation else "[No explanation provided]"


def build_claim_extraction_prompt(
    *,
    candidate_record: Dict[str, Any],
    max_claims: int,
) -> str:
    return (
        "Extract atomic factual claims from the candidate rationale.\n\n"
        "Definition:\n"
        "- A claim is one medically or clinically meaningful factual statement, "
        "diagnostic assertion, mechanism, treatment statement, risk factor, or "
        "answer-supporting reasoning step.\n"
        "- Extract claims only from the candidate rationale. Use the question/options "
        "only for disambiguation.\n"
        "- Do not extract the answer letter itself as a claim.\n"
        "- Exclude generic filler, uncertainty hedges, and purely stylistic text.\n"
        "- Prefer short, atomic claims. Split compound statements when needed.\n"
        f"- Return at most {max_claims} claims. Set truncated=true if additional factual "
        "claims were omitted because of this cap. If there is no factual medical claim, "
        "return an empty list and truncated=false.\n\n"
        f"Question type:\n{normalize_text(candidate_record.get('question_type')) or '[unknown]'}\n\n"
        f"Question:\n{normalize_text(candidate_record.get('question'))}\n\n"
        f"Options:\n{format_options(candidate_record.get('options'))}\n\n"
        "Candidate selected answer:\n"
        f"{format_answer(candidate_record)}\n\n"
        "Candidate rationale:\n"
        f"{format_explanation_text(candidate_record)}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "claims": [\n'
        '    {"claim_id": "c1", "claim": "<atomic claim>", "claim_type": "diagnosis"}\n'
        "  ],\n"
        '  "reason": "<brief note on extraction>",\n'
        '  "truncated": false\n'
        "}"
    )


def claims_for_prompt(claims: List[Dict[str, Any]]) -> str:
    lines = []
    for claim in claims:
        claim_id = normalize_text(claim.get("claim_id"))
        claim_text = normalize_text(claim.get("claim"))
        claim_type = normalize_text(claim.get("claim_type")) or "other"
        if claim_id and claim_text:
            lines.append(f"- {claim_id} [{claim_type}]: {claim_text}")
    return "\n".join(lines) if lines else "[No candidate claims extracted]"


def build_support_prompt(
    *,
    fp16_record: Dict[str, Any],
    candidate_record: Dict[str, Any],
    claims: List[Dict[str, Any]],
) -> str:
    question = normalize_text(fp16_record.get("question") or candidate_record.get("question"))
    options = fp16_record.get("options") or candidate_record.get("options") or {}
    question_type = normalize_text(
        fp16_record.get("question_type") or candidate_record.get("question_type")
    )
    return (
        "Judge each candidate rationale claim under two reference settings.\n\n"
        "Primary teacher-source setting, used for teacher-source UACR:\n"
        "- The active source is only the FP16 teacher rationale.\n"
        "- The question/options are shown only to disambiguate claim meaning; do not use "
        "them to mark a claim as source_supported unless the same claim is stated or clearly "
        "entailed by the FP16 rationale.\n"
        "- Do not use the gold answer, gold explanation, or standard medical knowledge to mark "
        "a claim as source_supported in this primary setting.\n"
        "- The question is whether the quantized rationale introduced a new evidence claim "
        "that is not in the FP16 teacher rationale.\n\n"
        "Secondary lenient clinical setting:\n"
        "- The active source is the FP16 teacher rationale plus clearly valid standard medical "
        "knowledge. Gold answer and gold explanation are intentionally excluded.\n"
        "- This setting asks whether a new claim can still be clinically justified without "
        "using the gold explanation as a reference.\n\n"
        "Labels for both settings:\n"
        '- "source_supported": the claim is stated or clearly entailed by the active source.\n'
        '- "valid_background": the claim is not stated in the active source, but is a generally '
        "valid medical background fact and does not introduce a new patient-specific finding, "
        "diagnosis, treatment effect, causal claim, or answer-specific evidence.\n"
        '- "unsupported_added": the claim introduces a new patient-specific condition, finding, '
        "diagnosis, treatment effect, causal/risk claim, or answer-supporting evidence that is "
        "not supported by the active source.\n"
        '- "contradicted": the claim contradicts the active source or standard medical knowledge.\n'
        '- "not_a_claim": the item is only an answer letter, formatting, generic filler, or not a '
        "factual medical/clinical claim.\n"
        '- fp16_overlap should be "present_in_fp16" only when the FP16 rationale states the same '
        'claim or a clear paraphrase. Use "not_in_fp16" for claims newly introduced relative '
        'to the FP16 rationale.\n\n'
        "Important: standard medical knowledge can make a claim valid_background in the primary "
        "teacher-source setting, but it must not make that claim source_supported unless it is "
        "also stated or clearly entailed by the FP16 rationale.\n\n"
        f"Question type:\n{question_type or '[unknown]'}\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options(options)}\n\n"
        "FP16 teacher selected answer:\n"
        f"{format_answer(fp16_record)}\n\n"
        "FP16 teacher rationale:\n"
        f"{format_explanation_text(fp16_record)}\n\n"
        "Candidate selected answer:\n"
        f"{format_answer(candidate_record)}\n\n"
        "Candidate rationale:\n"
        f"{format_explanation_text(candidate_record)}\n\n"
        "Candidate claims to judge:\n"
        f"{claims_for_prompt(claims)}\n\n"
        "Return JSON only, with one result for every claim_id:\n"
        "{\n"
        '  "claim_results": [\n'
        "    {\n"
        '      "claim_id": "c1",\n'
        '      "fp16_overlap": "not_in_fp16",\n'
        '      "teacher_support_status": "unsupported_added",\n'
        '      "teacher_support_source": "none",\n'
        '      "clinical_support_status": "valid_background",\n'
        '      "clinical_support_source": "standard_medical_knowledge",\n'
        '      "reason": "<short reason>"\n'
        "    }\n"
        "  ],\n"
        '  "reason": "<brief overall note>"\n'
        "}"
    )


def normalize_claims(parsed: Dict[str, Any], max_claims: int) -> Tuple[List[Dict[str, Any]], str, bool]:
    raw_claims = parsed.get("claims")
    if not isinstance(raw_claims, list):
        return [], normalize_text(parsed.get("reason")), bool(parsed.get("truncated"))

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
        claim_type = normalize_text(item.get("claim_type")) or "other"
        if claim_type not in CLAIM_TYPES:
            claim_type = "other"
        claims.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "claim_type": claim_type,
            }
        )
        if len(claims) >= max_claims:
            break
    truncated = bool(parsed.get("truncated")) or len(raw_claims) > max_claims
    return claims, normalize_text(parsed.get("reason")), truncated


def normalize_support_status(value: Any) -> str:
    status = normalize_text(value).lower()
    if status in SUPPORT_STATUSES:
        return status
    return LEGACY_SUPPORT_STATUS_MAP.get(status, "unjudged")


def normalize_support_source(value: Any) -> str:
    source = normalize_text(value).lower()
    return source if source in SUPPORT_SOURCES else "none"


def normalize_support_results(
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

    results: List[Dict[str, Any]] = []
    for claim in claims:
        claim_id = normalize_text(claim.get("claim_id"))
        item = by_id.get(claim_id, {})
        teacher_status = normalize_support_status(
            item.get("teacher_support_status", item.get("support_status"))
        )
        clinical_status = normalize_support_status(
            item.get("clinical_support_status", item.get("support_status"))
        )
        fp16_overlap = normalize_text(item.get("fp16_overlap")).lower()
        if fp16_overlap not in FP16_OVERLAPS:
            fp16_overlap = "unclear"
        teacher_source = normalize_support_source(
            item.get("teacher_support_source", item.get("support_source"))
        )
        clinical_source = normalize_support_source(
            item.get("clinical_support_source", item.get("support_source"))
        )
        results.append(
            {
                "claim_id": claim_id,
                "claim": normalize_text(claim.get("claim")),
                "claim_type": normalize_text(claim.get("claim_type")),
                "fp16_overlap": fp16_overlap,
                "teacher_support_status": teacher_status,
                "teacher_support_source": teacher_source,
                "clinical_support_status": clinical_status,
                "clinical_support_source": clinical_source,
                "support_status": teacher_status,
                "support_source": teacher_source,
                "reason": normalize_text(item.get("reason")),
            }
        )
    return results, normalize_text(parsed.get("reason"))


def unjudged_support_results(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "claim_id": normalize_text(claim.get("claim_id")),
            "claim": normalize_text(claim.get("claim")),
            "claim_type": normalize_text(claim.get("claim_type")),
            "fp16_overlap": "unclear",
            "teacher_support_status": "unjudged",
            "teacher_support_source": "none",
            "clinical_support_status": "unjudged",
            "clinical_support_source": "none",
            "support_status": "unjudged",
            "support_source": "none",
            "reason": "Judge call was skipped, failed, or omitted this claim_id.",
        }
        for claim in claims
    ]


def score_support_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def status(item: Dict[str, Any], key: str) -> str:
        return normalize_support_status(item.get(key))

    teacher_judged = [
        item
        for item in results
        if status(item, "teacher_support_status") not in {"not_a_claim", "unjudged"}
    ]
    clinical_judged = [
        item
        for item in results
        if status(item, "clinical_support_status") not in {"not_a_claim", "unjudged"}
    ]
    num_judged = len(teacher_judged)
    clinical_num_judged = len(clinical_judged)

    teacher_source_supported = sum(
        1 for item in teacher_judged if status(item, "teacher_support_status") == "source_supported"
    )
    teacher_valid_background = sum(
        1 for item in teacher_judged if status(item, "teacher_support_status") == "valid_background"
    )
    teacher_unsupported_added = sum(
        1 for item in teacher_judged if status(item, "teacher_support_status") == "unsupported_added"
    )
    teacher_contradicted = sum(
        1 for item in teacher_judged if status(item, "teacher_support_status") == "contradicted"
    )
    teacher_added = teacher_valid_background + teacher_unsupported_added + teacher_contradicted
    teacher_bad_added = teacher_unsupported_added + teacher_contradicted

    clinical_source_supported = sum(
        1 for item in clinical_judged if status(item, "clinical_support_status") == "source_supported"
    )
    clinical_valid_background = sum(
        1 for item in clinical_judged if status(item, "clinical_support_status") == "valid_background"
    )
    clinical_unsupported_added = sum(
        1 for item in clinical_judged if status(item, "clinical_support_status") == "unsupported_added"
    )
    clinical_contradicted = sum(
        1 for item in clinical_judged if status(item, "clinical_support_status") == "contradicted"
    )
    clinical_added = clinical_valid_background + clinical_unsupported_added + clinical_contradicted
    clinical_bad_added = clinical_unsupported_added + clinical_contradicted

    teacher_not_a_claim = sum(
        1 for item in results if status(item, "teacher_support_status") == "not_a_claim"
    )
    teacher_unjudged = sum(
        1 for item in results if status(item, "teacher_support_status") == "unjudged"
    )
    clinical_not_a_claim = sum(
        1 for item in results if status(item, "clinical_support_status") == "not_a_claim"
    )
    clinical_unjudged = sum(
        1 for item in results if status(item, "clinical_support_status") == "unjudged"
    )

    new_claims = [item for item in teacher_judged if item.get("fp16_overlap") == "not_in_fp16"]
    present_in_fp16 = sum(1 for item in teacher_judged if item.get("fp16_overlap") == "present_in_fp16")
    new_unsupported = sum(
        1 for item in new_claims if status(item, "teacher_support_status") == "unsupported_added"
    )
    new_contradicted = sum(
        1 for item in new_claims if status(item, "teacher_support_status") == "contradicted"
    )

    teacher_uacr = safe_rate(teacher_bad_added, num_judged)
    clinical_uacr = safe_rate(clinical_bad_added, clinical_num_judged)
    return {
        "candidate_claim_count": len(results),
        "judged_claim_count": num_judged,
        "clinical_judged_claim_count": clinical_num_judged,
        "source_supported_count": teacher_source_supported,
        "valid_background_count": teacher_valid_background,
        "unsupported_added_count": teacher_unsupported_added,
        "contradicted_count": teacher_contradicted,
        "added_claim_count": teacher_added,
        "added_unsupported_or_contradicted_count": teacher_bad_added,
        "not_a_claim_count": teacher_not_a_claim,
        "unjudged_count": teacher_unjudged,
        "clinical_source_supported_count": clinical_source_supported,
        "clinical_valid_background_count": clinical_valid_background,
        "clinical_unsupported_added_count": clinical_unsupported_added,
        "clinical_contradicted_count": clinical_contradicted,
        "clinical_added_claim_count": clinical_added,
        "clinical_added_unsupported_or_contradicted_count": clinical_bad_added,
        "clinical_not_a_claim_count": clinical_not_a_claim,
        "clinical_unjudged_count": clinical_unjudged,
        "present_in_fp16_count": present_in_fp16,
        "new_claim_count": len(new_claims),
        "new_unsupported_count": new_unsupported,
        "new_contradicted_count": new_contradicted,
        "teacher_source_uacr_rate": teacher_uacr,
        "added_unsupported_or_contradicted_over_all_claims": teacher_uacr,
        "any_added_unsupported_or_contradicted": (
            int(teacher_bad_added > 0) if num_judged > 0 else None
        ),
        "teacher_source_ccr_rate": safe_rate(teacher_contradicted, num_judged),
        "clinical_uacr_rate": clinical_uacr,
        "clinical_added_unsupported_or_contradicted_over_all_claims": clinical_uacr,
        "clinical_any_added_unsupported_or_contradicted": (
            int(clinical_bad_added > 0) if clinical_num_judged > 0 else None
        ),
        "clinical_ccr_rate": safe_rate(clinical_contradicted, clinical_num_judged),
        "supported_count": teacher_source_supported,
        "unsupported_count": teacher_unsupported_added,
        "supported_claim_rate": safe_rate(teacher_source_supported, num_judged),
        "valid_background_claim_rate": safe_rate(teacher_valid_background, num_judged),
        "unsupported_claim_rate": safe_rate(teacher_unsupported_added, num_judged),
        "contradicted_claim_rate": safe_rate(teacher_contradicted, num_judged),
        "hallucinated_claim_rate": teacher_uacr,
        "new_claim_rate": safe_rate(len(new_claims), num_judged),
        "new_unsupported_claim_rate": safe_rate(new_unsupported, len(new_claims)),
        "new_hallucinated_claim_rate": safe_rate(
            new_unsupported + new_contradicted,
            len(new_claims),
        ),
    }


def read_completed_pair_keys(path: Path) -> set:
    completed = set()
    if not path.exists():
        return completed
    for record in read_jsonl(path):
        pair_key = normalize_text(record.get("pair_key"))
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


def pair_has_same_fp16_answer(pair: Dict[str, Any]) -> bool:
    fp16_record = pair.get("fp16_record") or {}
    candidate_record = pair.get("candidate_record") or {}
    fp16_answer = normalize_text(extract_pred_answer(fp16_record))
    candidate_answer = normalize_text(extract_pred_answer(candidate_record))
    return bool(fp16_answer and candidate_answer and fp16_answer == candidate_answer)


def all_selected_systems_same_answer_pair_keys(
    *,
    args: argparse.Namespace,
    model_key: str,
    seed: int,
) -> Tuple[Optional[set], Dict[str, Any]]:
    if args.include_different_fp16_answers:
        return None, {"answer_filter": "none"}

    cache = getattr(args, "_all_system_same_answer_pair_key_cache", None)
    if cache is None:
        cache = {}
        setattr(args, "_all_system_same_answer_pair_key_cache", cache)
    cache_key = (
        model_key,
        seed,
        tuple(args.systems),
        str(args.fp16_dir),
        str(args.medmix_baseline_dir),
        str(args.eaquant_dir),
    )
    if cache_key in cache:
        return cache[cache_key]

    fp16_file = fp16_prediction_path(args.fp16_dir, DATASET, model_key)
    if not fp16_file.exists():
        if args.skip_missing:
            result = (set(), {"answer_filter": "fp16_and_selected_systems_same", "skipped_missing_file": str(fp16_file)})
            cache[cache_key] = result
            return result
        raise FileNotFoundError(fp16_file)

    fp16_records = read_jsonl(fp16_file)
    fp16_index, fp16_index_stats = build_index(fp16_records)
    candidate_indexes: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    candidate_index_stats: Dict[str, Any] = {}
    candidate_files: Dict[str, str] = {}
    for system in args.systems:
        candidate_file = candidate_prediction_path(
            args=args,
            system=system,
            model_key=model_key,
            seed=seed,
        )
        candidate_files[system] = str(candidate_file)
        if not candidate_file.exists():
            if args.skip_missing:
                result = (
                    set(),
                    {
                        "answer_filter": "fp16_and_selected_systems_same",
                        "skipped_missing_file": str(candidate_file),
                    },
                )
                cache[cache_key] = result
                return result
            raise FileNotFoundError(candidate_file)
        candidate_index, stats = build_index(read_jsonl(candidate_file))
        candidate_indexes[system] = candidate_index
        candidate_index_stats[system] = stats

    allowed_pair_keys = set()
    missing_candidate_counts = Counter()
    missing_answer_count = 0
    label_mismatch_count = 0
    for key, fp16_record in fp16_index.items():
        fp16_answer = normalize_text(extract_pred_answer(fp16_record))
        if not fp16_answer:
            missing_answer_count += 1
            continue
        answers = [fp16_answer]
        missing_candidate = False
        for system, candidate_index in candidate_indexes.items():
            candidate_record = candidate_index.get(key)
            if candidate_record is None:
                missing_candidate_counts[system] += 1
                missing_candidate = True
                break
            candidate_answer = normalize_text(extract_pred_answer(candidate_record))
            if not candidate_answer:
                missing_answer_count += 1
                missing_candidate = True
                break
            answers.append(candidate_answer)
        if missing_candidate:
            continue
        if len(set(answers)) == 1:
            allowed_pair_keys.add(key_to_str(key))
        else:
            label_mismatch_count += 1

    stats = {
        "answer_filter": "fp16_and_selected_systems_same",
        "answer_filter_systems": list(args.systems),
        "fp16_file_for_answer_filter": str(fp16_file),
        "candidate_files_for_answer_filter": candidate_files,
        "fp16_index_for_answer_filter": fp16_index_stats,
        "candidate_index_for_answer_filter": candidate_index_stats,
        "num_all_label_same_pair_keys": len(allowed_pair_keys),
        "num_label_mismatch_pair_keys": label_mismatch_count,
        "num_missing_answer_pair_keys": missing_answer_count,
        "num_missing_candidate_pair_keys_by_system": dict(sorted(missing_candidate_counts.items())),
    }
    result = (allowed_pair_keys, stats)
    cache[cache_key] = result
    return result


def maybe_filter_same_fp16_answer_pairs(
    matched_pairs: List[Dict[str, Any]],
    args: argparse.Namespace,
    match_stats: Dict[str, Any],
    allowed_pair_keys: Optional[set] = None,
    answer_filter_stats: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    match_stats["num_matched_pairs_before_answer_filter"] = len(matched_pairs)
    if answer_filter_stats:
        match_stats.update(answer_filter_stats)
    if args.include_different_fp16_answers:
        match_stats["same_fp16_answer_only"] = False
        match_stats["all_selected_systems_same_answer_only"] = False
        match_stats["num_matched_pairs_after_answer_filter"] = len(matched_pairs)
        return matched_pairs

    if allowed_pair_keys is not None:
        filtered_pairs = [
            pair
            for pair in matched_pairs
            if normalize_text(pair.get("pair_key")) in allowed_pair_keys
        ]
        match_stats["same_fp16_answer_only"] = True
        match_stats["all_selected_systems_same_answer_only"] = True
        match_stats["num_matched_pairs_after_answer_filter"] = len(filtered_pairs)
        return filtered_pairs

    same_answer_pairs = [pair for pair in matched_pairs if pair_has_same_fp16_answer(pair)]
    match_stats["num_same_fp16_answer_pairs"] = len(same_answer_pairs)
    match_stats["same_fp16_answer_only"] = True
    match_stats["all_selected_systems_same_answer_only"] = False
    match_stats["num_matched_pairs_after_answer_filter"] = len(same_answer_pairs)
    return same_answer_pairs


def candidate_prediction_path(
    *,
    args: argparse.Namespace,
    system: str,
    model_key: str,
    seed: int,
) -> Path:
    if system == "medmix_baseline":
        return medmix_baseline_prediction_path(args.medmix_baseline_dir, DATASET, model_key, seed)
    if system == "eaquant":
        return eaquant_prediction_path(args.eaquant_dir, DATASET, model_key, seed)
    raise ValueError(f"Unsupported system: {system}")


def system_stem_prefix(system: str) -> str:
    # Preserve the EAQuant filenames so interrupted runs can resume.
    return "" if system == "eaquant" else f"{system}_"


def matched_pairs_for_model_seed(
    *,
    fp16_records: List[Dict[str, Any]],
    candidate_records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_index, candidate_index_stats = build_index(candidate_records)
    matched: List[Dict[str, Any]] = []
    stats = {
        "num_fp16_rows": len(fp16_records),
        "num_candidate_rows": len(candidate_records),
        "candidate_index": candidate_index_stats,
        "num_fp16_rows_missing_key": 0,
        "num_missing_candidate_match": 0,
    }
    for fp16_record in fp16_records:
        key = record_key(fp16_record)
        if key is None:
            stats["num_fp16_rows_missing_key"] += 1
            continue
        candidate_record = candidate_index.get(key)
        if candidate_record is None:
            stats["num_missing_candidate_match"] += 1
            continue
        matched.append(
            {
                "pair_key": key_to_str(key),
                "fp16_record": fp16_record,
                "candidate_record": candidate_record,
            }
        )
    stats["num_matched_pairs"] = len(matched)
    return matched, stats


def input_files_for_model_seed(
    *,
    args: argparse.Namespace,
    system: str,
    model_key: str,
    seed: int,
) -> Tuple[Path, Path]:
    return (
        fp16_prediction_path(args.fp16_dir, DATASET, model_key),
        candidate_prediction_path(args=args, system=system, model_key=model_key, seed=seed),
    )


def claim_output_paths(
    *,
    output_dir: Path,
    system: str,
    model_key: str,
    seed: int,
    judge_model: str,
    dry_run_prompts: bool,
) -> Tuple[Path, Path]:
    stem = f"candidate_claims_{DATASET}_{SCHEMA_VERSION}_{system_stem_prefix(system)}{model_key}_seed{seed}_by_{slugify(judge_model)}"
    if dry_run_prompts:
        stem = f"{stem}_prompts"
    return output_dir / f"{stem}.jsonl", output_dir / f"{stem}_summary.json"


def support_output_paths(
    *,
    output_dir: Path,
    system: str,
    model_key: str,
    seed: int,
    judge_model: str,
    dry_run_prompts: bool,
) -> Tuple[Path, Path]:
    stem = f"unsupported_claims_{DATASET}_{SCHEMA_VERSION}_{system_stem_prefix(system)}{model_key}_seed{seed}_by_{slugify(judge_model)}"
    if dry_run_prompts:
        stem = f"{stem}_prompts"
    return output_dir / f"{stem}.jsonl", output_dir / f"{stem}_summary.json"


def run_claim_extraction_for_model_seed(
    *,
    args: argparse.Namespace,
    api_key: str,
    system: str,
    model_key: str,
    seed: int,
) -> Dict[str, Any]:
    fp16_file, candidate_file = input_files_for_model_seed(
        args=args,
        system=system,
        model_key=model_key,
        seed=seed,
    )
    for path in (fp16_file, candidate_file):
        if not path.exists():
            if args.skip_missing:
                return {
                    "task": "claims",
                    "dataset": DATASET,
                    "system": system,
                    "system_display": SYSTEM_DISPLAY[system],
                    "model_key": model_key,
                    "seed": seed,
                    "skipped_missing_file": str(path),
                }
            raise FileNotFoundError(path)

    output_jsonl, summary_json = claim_output_paths(
        output_dir=args.output_dir,
        system=system,
        model_key=model_key,
        seed=seed,
        judge_model=args.judge_model,
        dry_run_prompts=args.dry_run_prompts,
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_jsonl.exists():
        output_jsonl.unlink()

    fp16_records = read_jsonl(fp16_file)
    candidate_records = read_jsonl(candidate_file)
    matched_pairs, match_stats = matched_pairs_for_model_seed(
        fp16_records=fp16_records,
        candidate_records=candidate_records,
    )
    allowed_pair_keys, answer_filter_stats = all_selected_systems_same_answer_pair_keys(
        args=args,
        model_key=model_key,
        seed=seed,
    )
    matched_pairs = maybe_filter_same_fp16_answer_pairs(
        matched_pairs,
        args,
        match_stats,
        allowed_pair_keys=allowed_pair_keys,
        answer_filter_stats=answer_filter_stats,
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
        f"[INFO] claims {DATASET} {system} {model_key} seed{seed}: "
        f"{len(selected_pairs)} pairs, {len(completed_pair_keys)} already done"
    )

    for idx, item in enumerate(selected_pairs, start=1):
        pair_key = normalize_text(item.get("pair_key"))
        if pair_key in completed_pair_keys and not args.overwrite:
            continue

        fp16_record = item["fp16_record"]
        candidate_record = item["candidate_record"]
        user_prompt = build_claim_extraction_prompt(
            candidate_record=candidate_record,
            max_claims=args.max_claims,
        )
        messages = build_messages(user_prompt)
        record = {
            "task": "claims",
            "dataset": DATASET,
            "dataset_display": DATASET_DISPLAY,
            "system": system,
            "system_display": SYSTEM_DISPLAY[system],
            "model_key": model_key,
            "seed": seed,
            "judge_model": args.judge_model,
            "schema_version": SCHEMA_VERSION,
            "max_claims": args.max_claims,
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
            "fp16_is_correct": fp16_record.get("is_correct"),
            "candidate_pred_answer": extract_pred_answer(candidate_record),
            "candidate_is_correct": candidate_record.get("is_correct"),
            "candidate_explanation": extract_explanation(candidate_record),
        }
        if args.verbose_prompts or args.dry_run_prompts:
            record["system_prompt"] = SYSTEM_PROMPT
            record["user_prompt"] = user_prompt

        if args.dry_run_prompts:
            claims: List[Dict[str, Any]] = []
            extraction_reason = ""
            claims_truncated = False
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
                task_name="candidate_claim_extraction",
                messages=messages,
                response_format=CLAIM_EXTRACTION_SCHEMA,
            )
            claims, extraction_reason, claims_truncated = normalize_claims(
                call_result.pop("parsed", {}),
                args.max_claims,
            )

        record.update(call_result)
        record.update(
            {
                "claims": claims,
                "num_claims": len(claims),
                "claims_truncated": bool(claims_truncated),
                "claim_extraction_reason": extraction_reason,
            }
        )
        append_jsonl(output_jsonl, record)
        new_records.append(record)

        if args.dry_run_prompts:
            print(f"[{idx}/{len(selected_pairs)}] wrote claim prompt {system} {model_key} seed{seed} {pair_key}")
        else:
            print(
                f"[{idx}/{len(selected_pairs)}] claims {system} {model_key} seed{seed} "
                f"{pair_key}: {len(claims)} claims"
            )

    all_records = existing_records + new_records if not args.overwrite else new_records
    summary = summarize_claim_records(
        records=all_records,
        dry_run_prompts=args.dry_run_prompts,
    )
    summary.update(
        {
            "task": "claims",
            "dataset": DATASET,
            "dataset_display": DATASET_DISPLAY,
            "system": system,
            "system_display": SYSTEM_DISPLAY[system],
            "model_key": model_key,
            "seed": seed,
            "judge_model": args.judge_model,
            "schema_version": SCHEMA_VERSION,
            "same_fp16_answer_filter_enabled": not bool(args.include_different_fp16_answers),
            "all_selected_systems_same_answer_filter_enabled": not bool(args.include_different_fp16_answers),
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


def summarize_claim_records(
    *,
    records: List[Dict[str, Any]],
    dry_run_prompts: bool,
) -> Dict[str, Any]:
    parse_failures = sum(1 for record in records if record.get("judge_parse_success") is False)
    model_calls = sum(1 for record in records if record.get("judge_model_call_skipped") is not True)
    total_claims = sum(int(record.get("num_claims") or 0) for record in records)
    max_claims_values = [
        int(record.get("max_claims"))
        for record in records
        if record.get("max_claims") is not None
    ]
    effective_max_claims = max(max_claims_values) if max_claims_values else DEFAULT_MAX_CLAIMS
    records_with_claims = [
        record for record in records if int(record.get("num_claims") or 0) > 0
    ]
    truncated_records = [record for record in records if record.get("claims_truncated") is True]
    records_at_cap = [
        record
        for record in records
        if int(record.get("num_claims") or 0) >= int(record.get("max_claims") or DEFAULT_MAX_CLAIMS)
    ]
    claim_type_counts = Counter()
    for record in records:
        claims = record.get("claims")
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if isinstance(claim, dict):
                claim_type_counts[normalize_text(claim.get("claim_type")) or "other"] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "max_claims": effective_max_claims,
        "num_records": len(records),
        "dry_run_prompts": bool(dry_run_prompts),
        "num_model_calls": model_calls,
        "num_parse_failures": parse_failures,
        "num_records_with_claims": len(records_with_claims),
        "num_records_without_claims": len(records) - len(records_with_claims),
        "num_truncated_records": len(truncated_records),
        "truncated_record_rate": safe_rate(len(truncated_records), len(records)),
        "num_records_at_claim_cap": len(records_at_cap),
        "record_at_claim_cap_rate": safe_rate(len(records_at_cap), len(records)),
        "total_claims": total_claims,
        "avg_claims_per_record": total_claims / len(records) if records else None,
        "avg_claims_per_claimed_record": (
            total_claims / len(records_with_claims) if records_with_claims else None
        ),
        "claim_type_counts": dict(sorted(claim_type_counts.items())),
    }


def load_claim_records_for_model_seed(
    *,
    args: argparse.Namespace,
    system: str,
    model_key: str,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    candidate_paths = []
    if args.dry_run_prompts:
        candidate_paths.append(
            claim_output_paths(
                output_dir=args.output_dir,
                system=system,
                model_key=model_key,
                seed=seed,
                judge_model=args.judge_model,
                dry_run_prompts=False,
            )[0]
        )
    candidate_paths.append(
        claim_output_paths(
            output_dir=args.output_dir,
            system=system,
            model_key=model_key,
            seed=seed,
            judge_model=args.judge_model,
            dry_run_prompts=args.dry_run_prompts,
        )[0]
    )

    claims_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
    if not claims_path.exists():
        raise FileNotFoundError(
            f"Missing candidate claim cache for {system} {model_key} seed{seed}: {claims_path}. "
            "Run --mode claims or --mode both first."
        )
    records = read_jsonl(claims_path)
    by_pair_key: Dict[str, Dict[str, Any]] = {}
    for record in records:
        pair_key = normalize_text(record.get("pair_key"))
        if pair_key and pair_key not in by_pair_key:
            by_pair_key[pair_key] = record
    return by_pair_key


def run_support_judgment_for_model_seed(
    *,
    args: argparse.Namespace,
    api_key: str,
    system: str,
    model_key: str,
    seed: int,
) -> Dict[str, Any]:
    fp16_file, candidate_file = input_files_for_model_seed(
        args=args,
        system=system,
        model_key=model_key,
        seed=seed,
    )
    for path in (fp16_file, candidate_file):
        if not path.exists():
            if args.skip_missing:
                return {
                    "task": "support",
                    "dataset": DATASET,
                    "system": system,
                    "system_display": SYSTEM_DISPLAY[system],
                    "model_key": model_key,
                    "seed": seed,
                    "skipped_missing_file": str(path),
                }
            raise FileNotFoundError(path)

    claim_records_by_pair_key = load_claim_records_for_model_seed(
        args=args,
        system=system,
        model_key=model_key,
        seed=seed,
    )
    output_jsonl, summary_json = support_output_paths(
        output_dir=args.output_dir,
        system=system,
        model_key=model_key,
        seed=seed,
        judge_model=args.judge_model,
        dry_run_prompts=args.dry_run_prompts,
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_jsonl.exists():
        output_jsonl.unlink()

    fp16_records = read_jsonl(fp16_file)
    candidate_records = read_jsonl(candidate_file)
    matched_pairs, match_stats = matched_pairs_for_model_seed(
        fp16_records=fp16_records,
        candidate_records=candidate_records,
    )
    allowed_pair_keys, answer_filter_stats = all_selected_systems_same_answer_pair_keys(
        args=args,
        model_key=model_key,
        seed=seed,
    )
    matched_pairs = maybe_filter_same_fp16_answer_pairs(
        matched_pairs,
        args,
        match_stats,
        allowed_pair_keys=allowed_pair_keys,
        answer_filter_stats=answer_filter_stats,
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
        f"[INFO] support {DATASET} {system} {model_key} seed{seed}: "
        f"{len(selected_pairs)} pairs, {len(completed_pair_keys)} already done"
    )

    for idx, item in enumerate(selected_pairs, start=1):
        pair_key = normalize_text(item.get("pair_key"))
        if pair_key in completed_pair_keys and not args.overwrite:
            continue

        fp16_record = item["fp16_record"]
        candidate_record = item["candidate_record"]
        claim_record = claim_records_by_pair_key.get(pair_key, {})
        claims = claim_record.get("claims")
        if not isinstance(claims, list):
            claims = []
        claims = [claim for claim in claims if isinstance(claim, dict)]

        user_prompt = build_support_prompt(
            fp16_record=fp16_record,
            candidate_record=candidate_record,
            claims=claims,
        )
        messages = build_messages(user_prompt)
        record = {
            "task": "support",
            "dataset": DATASET,
            "dataset_display": DATASET_DISPLAY,
            "system": system,
            "system_display": SYSTEM_DISPLAY[system],
            "model_key": model_key,
            "seed": seed,
            "judge_model": args.judge_model,
            "schema_version": SCHEMA_VERSION,
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
            "fp16_is_correct": fp16_record.get("is_correct"),
            "fp16_explanation": extract_explanation(fp16_record),
            "candidate_pred_answer": extract_pred_answer(candidate_record),
            "candidate_pred_answer_text": format_answer(candidate_record),
            "candidate_is_correct": candidate_record.get("is_correct"),
            "candidate_explanation": extract_explanation(candidate_record),
            "claims": claims,
        }
        if args.verbose_prompts or args.dry_run_prompts:
            record["system_prompt"] = SYSTEM_PROMPT
            record["user_prompt"] = user_prompt

        if args.dry_run_prompts:
            claim_results = unjudged_support_results(claims)
            support_reason = ""
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
            support_reason = "No candidate claims were extracted."
            call_result = {
                "judge_parse_success": True,
                "judge_attempt_count": 0,
                "judge_error": "",
                "raw_judge_response": "",
                "openai_response_id": "",
                "openai_usage": {},
            }
        else:
            best_call_result: Dict[str, Any] = {}
            best_claim_results = unjudged_support_results(claims)
            best_support_reason = "Support judge call failed."
            best_unjudged_count = len(claims)
            incomplete_retries_used = 0
            max_incomplete_retries = max(0, int(args.incomplete_judgment_retries or 0))
            for incomplete_attempt in range(max_incomplete_retries + 1):
                attempt_call_result = run_openai_json_call(
                    args=args,
                    api_key=api_key,
                    task_name="unsupported_claim_support",
                    messages=messages,
                    response_format=SUPPORT_SCHEMA,
                )
                parsed = attempt_call_result.pop("parsed", {})
                if attempt_call_result.get("judge_parse_success") is True:
                    attempt_claim_results, attempt_support_reason = normalize_support_results(
                        parsed,
                        claims,
                    )
                else:
                    attempt_claim_results = unjudged_support_results(claims)
                    attempt_support_reason = "Support judge call failed."

                attempt_score = score_support_results(attempt_claim_results)
                attempt_unjudged_count = int(attempt_score.get("unjudged_count") or 0)
                if not best_call_result or attempt_unjudged_count < best_unjudged_count:
                    best_call_result = attempt_call_result
                    best_claim_results = attempt_claim_results
                    best_support_reason = attempt_support_reason
                    best_unjudged_count = attempt_unjudged_count
                    incomplete_retries_used = incomplete_attempt
                if attempt_unjudged_count == 0:
                    break
                if incomplete_attempt < max_incomplete_retries:
                    print(
                        f"[WARN] support {system} {model_key} seed{seed} {pair_key}: "
                        f"{attempt_unjudged_count} unjudged claims; retrying incomplete judgment "
                        f"({incomplete_attempt + 1}/{max_incomplete_retries})"
                    )

            call_result = best_call_result
            claim_results = best_claim_results
            support_reason = best_support_reason
            call_result["incomplete_judgment_retry_count"] = incomplete_retries_used
            call_result["final_unjudged_count_before_scoring"] = best_unjudged_count

        score = score_support_results(claim_results)
        record.update(call_result)
        record.update(score)
        record.update(
            {
                "claim_results": claim_results,
                "support_reason": support_reason,
            }
        )
        append_jsonl(output_jsonl, record)
        new_records.append(record)

        if args.dry_run_prompts:
            print(f"[{idx}/{len(selected_pairs)}] wrote support prompt {system} {model_key} seed{seed} {pair_key}")
        else:
            rate = record.get("teacher_source_uacr_rate")
            rate_text = pct(rate) if rate is not None else "NA"
            print(
                f"[{idx}/{len(selected_pairs)}] support {system} {model_key} seed{seed} "
                f"{pair_key}: teacher_source_uacr={rate_text}"
            )

    all_records = existing_records + new_records if not args.overwrite else new_records
    summary = summarize_support_records(
        records=all_records,
        dry_run_prompts=args.dry_run_prompts,
    )
    summary.update(
        {
            "task": "support",
            "dataset": DATASET,
            "dataset_display": DATASET_DISPLAY,
            "system": system,
            "system_display": SYSTEM_DISPLAY[system],
            "model_key": model_key,
            "seed": seed,
            "judge_model": args.judge_model,
            "schema_version": SCHEMA_VERSION,
            "same_fp16_answer_filter_enabled": not bool(args.include_different_fp16_answers),
            "all_selected_systems_same_answer_filter_enabled": not bool(args.include_different_fp16_answers),
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


def mean_record_rate(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [float(record[key]) for record in records if isinstance(record.get(key), (float, int))]
    return sum(values) / len(values) if values else None


def _support_summary_for_subset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored_records = [record for record in records if int(record.get("judged_claim_count") or 0) > 0]
    clinical_scored_records = [
        record for record in records if int(record.get("clinical_judged_claim_count") or 0) > 0
    ]
    records_with_new_claims = [
        record
        for record in scored_records
        if int(record.get("new_claim_count") or 0) > 0
    ]
    sum_keys = (
        "candidate_claim_count",
        "judged_claim_count",
        "clinical_judged_claim_count",
        "source_supported_count",
        "valid_background_count",
        "unsupported_added_count",
        "supported_count",
        "unsupported_count",
        "contradicted_count",
        "added_claim_count",
        "added_unsupported_or_contradicted_count",
        "not_a_claim_count",
        "unjudged_count",
        "clinical_source_supported_count",
        "clinical_valid_background_count",
        "clinical_unsupported_added_count",
        "clinical_contradicted_count",
        "clinical_added_claim_count",
        "clinical_added_unsupported_or_contradicted_count",
        "clinical_not_a_claim_count",
        "clinical_unjudged_count",
        "present_in_fp16_count",
        "new_claim_count",
        "new_unsupported_count",
        "new_contradicted_count",
    )
    totals = {
        key: sum(int(record.get(key) or 0) for record in records)
        for key in sum_keys
    }
    unsupported_plus_contradicted = totals["unsupported_count"] + totals["contradicted_count"]
    new_unsupported_plus_contradicted = (
        totals["new_unsupported_count"] + totals["new_contradicted_count"]
    )
    return {
        "num_records": len(records),
        "num_scored_records": len(scored_records),
        "num_clinical_scored_records": len(clinical_scored_records),
        "num_records_with_new_claims": len(records_with_new_claims),
        **totals,
        "teacher_source_uacr_macro_rate": mean_record_rate(
            scored_records,
            "teacher_source_uacr_rate",
        ),
        "teacher_source_uacr_micro_rate": safe_rate(
            totals["added_unsupported_or_contradicted_count"],
            totals["judged_claim_count"],
        ),
        "teacher_source_any_uac_rate": mean_record_rate(
            scored_records,
            "any_added_unsupported_or_contradicted",
        ),
        "teacher_source_ccr_macro_rate": mean_record_rate(scored_records, "teacher_source_ccr_rate"),
        "teacher_source_ccr_micro_rate": safe_rate(
            totals["contradicted_count"],
            totals["judged_claim_count"],
        ),
        "clinical_uacr_macro_rate": mean_record_rate(clinical_scored_records, "clinical_uacr_rate"),
        "clinical_uacr_micro_rate": safe_rate(
            totals["clinical_added_unsupported_or_contradicted_count"],
            totals["clinical_judged_claim_count"],
        ),
        "clinical_any_uac_rate": mean_record_rate(
            clinical_scored_records,
            "clinical_any_added_unsupported_or_contradicted",
        ),
        "clinical_ccr_macro_rate": mean_record_rate(clinical_scored_records, "clinical_ccr_rate"),
        "clinical_ccr_micro_rate": safe_rate(
            totals["clinical_contradicted_count"],
            totals["clinical_judged_claim_count"],
        ),
        "unsupported_claim_macro_rate": mean_record_rate(scored_records, "unsupported_claim_rate"),
        "unsupported_claim_micro_rate": safe_rate(
            totals["unsupported_count"],
            totals["judged_claim_count"],
        ),
        "hallucinated_claim_macro_rate": mean_record_rate(scored_records, "hallucinated_claim_rate"),
        "hallucinated_claim_micro_rate": safe_rate(
            unsupported_plus_contradicted,
            totals["judged_claim_count"],
        ),
        "new_claim_macro_rate": mean_record_rate(scored_records, "new_claim_rate"),
        "new_claim_micro_rate": safe_rate(
            totals["new_claim_count"],
            totals["judged_claim_count"],
        ),
        "new_unsupported_claim_macro_rate": mean_record_rate(
            records_with_new_claims,
            "new_unsupported_claim_rate",
        ),
        "new_hallucinated_claim_macro_rate": mean_record_rate(
            records_with_new_claims,
            "new_hallucinated_claim_rate",
        ),
        "new_hallucinated_claim_micro_rate": safe_rate(
            new_unsupported_plus_contradicted,
            totals["new_claim_count"],
        ),
    }


def same_answer(record: Dict[str, Any]) -> bool:
    candidate_answer = normalize_text(record.get("candidate_pred_answer"))
    fp16_answer = normalize_text(record.get("fp16_pred_answer"))
    return bool(candidate_answer and fp16_answer and candidate_answer == fp16_answer)


def summarize_support_records(
    *,
    records: List[Dict[str, Any]],
    dry_run_prompts: bool,
) -> Dict[str, Any]:
    parse_failures = sum(1 for record in records if record.get("judge_parse_success") is False)
    model_calls = sum(1 for record in records if record.get("judge_model_call_skipped") is not True)
    teacher_status_counts = Counter()
    clinical_status_counts = Counter()
    teacher_source_counts = Counter()
    clinical_source_counts = Counter()
    fp16_overlap_counts = Counter()
    for record in records:
        claim_results = record.get("claim_results")
        if not isinstance(claim_results, list):
            continue
        for item in claim_results:
            if not isinstance(item, dict):
                continue
            teacher_status_counts[normalize_text(item.get("teacher_support_status")) or "missing"] += 1
            clinical_status_counts[normalize_text(item.get("clinical_support_status")) or "missing"] += 1
            teacher_source_counts[normalize_text(item.get("teacher_support_source")) or "missing"] += 1
            clinical_source_counts[normalize_text(item.get("clinical_support_source")) or "missing"] += 1
            fp16_overlap_counts[normalize_text(item.get("fp16_overlap")) or "missing"] += 1

    all_summary = _support_summary_for_subset(records)
    correct_records = [record for record in records if record.get("candidate_is_correct") is True]
    incorrect_records = [record for record in records if record.get("candidate_is_correct") is False]
    same_fp16_answer_records = [record for record in records if same_answer(record)]
    different_fp16_answer_records = [
        record
        for record in records
        if normalize_text(record.get("candidate_pred_answer"))
        and normalize_text(record.get("fp16_pred_answer"))
        and not same_answer(record)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "dry_run_prompts": bool(dry_run_prompts),
        "num_model_calls": model_calls,
        "num_parse_failures": parse_failures,
        **all_summary,
        "candidate_correct_only": _support_summary_for_subset(correct_records),
        "candidate_incorrect_only": _support_summary_for_subset(incorrect_records),
        "same_fp16_answer_only": _support_summary_for_subset(same_fp16_answer_records),
        "different_fp16_answer_only": _support_summary_for_subset(different_fp16_answer_records),
        "support_status_counts": dict(sorted(teacher_status_counts.items())),
        "teacher_support_status_counts": dict(sorted(teacher_status_counts.items())),
        "clinical_support_status_counts": dict(sorted(clinical_status_counts.items())),
        "support_source_counts": dict(sorted(teacher_source_counts.items())),
        "teacher_support_source_counts": dict(sorted(teacher_source_counts.items())),
        "clinical_support_source_counts": dict(sorted(clinical_source_counts.items())),
        "fp16_overlap_counts": dict(sorted(fp16_overlap_counts.items())),
    }


def row_from_support_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    same_answer_summary = summary.get("same_fp16_answer_only") or {}
    correct_summary = summary.get("candidate_correct_only") or {}
    return {
        "dataset": summary.get("dataset"),
        "dataset_display": summary.get("dataset_display"),
        "system": summary.get("system"),
        "system_display": summary.get("system_display"),
        "model_key": summary.get("model_key"),
        "seed": summary.get("seed"),
        "schema_version": summary.get("schema_version"),
        "num_runs": 1,
        "num_records": summary.get("num_records"),
        "num_scored_records": summary.get("num_scored_records"),
        "num_clinical_scored_records": summary.get("num_clinical_scored_records"),
        "candidate_claim_count": summary.get("candidate_claim_count"),
        "judged_claim_count": summary.get("judged_claim_count"),
        "clinical_judged_claim_count": summary.get("clinical_judged_claim_count"),
        "source_supported_count": summary.get("source_supported_count"),
        "valid_background_count": summary.get("valid_background_count"),
        "unsupported_added_count": summary.get("unsupported_added_count"),
        "supported_count": summary.get("supported_count"),
        "unsupported_count": summary.get("unsupported_count"),
        "contradicted_count": summary.get("contradicted_count"),
        "added_claim_count": summary.get("added_claim_count"),
        "added_unsupported_or_contradicted_count": summary.get(
            "added_unsupported_or_contradicted_count"
        ),
        "clinical_added_unsupported_or_contradicted_count": summary.get(
            "clinical_added_unsupported_or_contradicted_count"
        ),
        "not_a_claim_count": summary.get("not_a_claim_count"),
        "unjudged_count": summary.get("unjudged_count"),
        "new_claim_count": summary.get("new_claim_count"),
        "new_unsupported_count": summary.get("new_unsupported_count"),
        "new_contradicted_count": summary.get("new_contradicted_count"),
        "teacher_source_uacr_macro_rate": summary.get("teacher_source_uacr_macro_rate"),
        "teacher_source_uacr_micro_rate": summary.get("teacher_source_uacr_micro_rate"),
        "teacher_source_any_uac_rate": summary.get("teacher_source_any_uac_rate"),
        "teacher_source_ccr_macro_rate": summary.get("teacher_source_ccr_macro_rate"),
        "teacher_source_ccr_micro_rate": summary.get("teacher_source_ccr_micro_rate"),
        "clinical_uacr_macro_rate": summary.get("clinical_uacr_macro_rate"),
        "clinical_uacr_micro_rate": summary.get("clinical_uacr_micro_rate"),
        "clinical_any_uac_rate": summary.get("clinical_any_uac_rate"),
        "clinical_ccr_macro_rate": summary.get("clinical_ccr_macro_rate"),
        "clinical_ccr_micro_rate": summary.get("clinical_ccr_micro_rate"),
        "same_fp16_answer_num_records": same_answer_summary.get("num_records"),
        "same_fp16_answer_teacher_source_uacr_macro_rate": same_answer_summary.get(
            "teacher_source_uacr_macro_rate"
        ),
        "same_fp16_answer_teacher_source_any_uac_rate": same_answer_summary.get(
            "teacher_source_any_uac_rate"
        ),
        "same_fp16_answer_clinical_uacr_macro_rate": same_answer_summary.get(
            "clinical_uacr_macro_rate"
        ),
        "candidate_correct_num_records": correct_summary.get("num_records"),
        "candidate_correct_teacher_source_uacr_macro_rate": correct_summary.get(
            "teacher_source_uacr_macro_rate"
        ),
        "unsupported_claim_macro_rate": summary.get("unsupported_claim_macro_rate"),
        "unsupported_claim_micro_rate": summary.get("unsupported_claim_micro_rate"),
        "hallucinated_claim_macro_rate": summary.get("hallucinated_claim_macro_rate"),
        "hallucinated_claim_micro_rate": summary.get("hallucinated_claim_micro_rate"),
        "new_claim_macro_rate": summary.get("new_claim_macro_rate"),
        "new_claim_micro_rate": summary.get("new_claim_micro_rate"),
        "new_hallucinated_claim_macro_rate": summary.get("new_hallucinated_claim_macro_rate"),
        "new_hallucinated_claim_micro_rate": summary.get("new_hallucinated_claim_micro_rate"),
        "num_parse_failures": summary.get("num_parse_failures"),
        "output_jsonl": summary.get("output_jsonl"),
        "candidate_file": summary.get("candidate_file"),
    }


def aggregate_mean_std(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("system")), str(row.get("model_key")))].append(row)

    stats_rows: List[Dict[str, Any]] = []
    total_keys = (
        "num_records",
        "num_scored_records",
        "num_clinical_scored_records",
        "candidate_claim_count",
        "judged_claim_count",
        "clinical_judged_claim_count",
        "unsupported_added_count",
        "contradicted_count",
        "added_claim_count",
        "added_unsupported_or_contradicted_count",
        "clinical_added_unsupported_or_contradicted_count",
        "new_claim_count",
        "new_unsupported_count",
        "new_contradicted_count",
        "unjudged_count",
        "num_parse_failures",
    )
    metric_pairs = (
        ("teacher_source_uacr_macro_rate", "teacher_source_uacr_macro"),
        ("teacher_source_uacr_micro_rate", "teacher_source_uacr_micro"),
        ("teacher_source_any_uac_rate", "teacher_source_any_uac"),
        ("teacher_source_ccr_macro_rate", "teacher_source_ccr_macro"),
        ("teacher_source_ccr_micro_rate", "teacher_source_ccr_micro"),
        ("clinical_uacr_macro_rate", "clinical_uacr_macro"),
        ("clinical_uacr_micro_rate", "clinical_uacr_micro"),
        ("clinical_any_uac_rate", "clinical_any_uac"),
        ("clinical_ccr_macro_rate", "clinical_ccr_macro"),
        ("clinical_ccr_micro_rate", "clinical_ccr_micro"),
        ("same_fp16_answer_teacher_source_uacr_macro_rate", "same_fp16_answer_teacher_source_uacr_macro"),
        ("unsupported_claim_macro_rate", "unsupported_claim_macro"),
        ("unsupported_claim_micro_rate", "unsupported_claim_micro"),
        ("hallucinated_claim_macro_rate", "hallucinated_claim_macro"),
        ("hallucinated_claim_micro_rate", "hallucinated_claim_micro"),
        ("new_claim_macro_rate", "new_claim_macro"),
        ("new_claim_micro_rate", "new_claim_micro"),
        ("new_hallucinated_claim_macro_rate", "new_hallucinated_claim_macro"),
        ("new_hallucinated_claim_micro_rate", "new_hallucinated_claim_micro"),
    )
    for (system, model_key), group in grouped.items():
        seeds = sorted(int(row["seed"]) for row in group if row.get("seed") is not None)
        stats_row: Dict[str, Any] = {
            "system": system,
            "system_display": SYSTEM_DISPLAY.get(system, system),
            "model_key": model_key,
            "schema_version": SCHEMA_VERSION,
            "num_runs": len(group),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for key in total_keys:
            stats_row[key] = sum(int(row.get(key) or 0) for row in group)
        for metric_key, output_prefix in metric_pairs:
            mean_value, std_value, count = mean_std(row.get(metric_key) for row in group)
            stats_row[f"{output_prefix}_mean_rate"] = mean_value
            stats_row[f"{output_prefix}_std_rate"] = std_value
            stats_row[f"{output_prefix}_num_values"] = count
        stats_rows.append(stats_row)
    system_rank = {system: idx for idx, system in enumerate(SYSTEM_ORDER)}
    return sorted(
        stats_rows,
        key=lambda row: (row["model_key"], system_rank.get(str(row.get("system")), 99)),
    )


def add_pct_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    rate_keys = (
        "teacher_source_uacr_macro_rate",
        "teacher_source_uacr_micro_rate",
        "teacher_source_any_uac_rate",
        "teacher_source_ccr_macro_rate",
        "teacher_source_ccr_micro_rate",
        "clinical_uacr_macro_rate",
        "clinical_uacr_micro_rate",
        "clinical_any_uac_rate",
        "clinical_ccr_macro_rate",
        "clinical_ccr_micro_rate",
        "same_fp16_answer_teacher_source_uacr_macro_rate",
        "same_fp16_answer_teacher_source_any_uac_rate",
        "same_fp16_answer_clinical_uacr_macro_rate",
        "candidate_correct_teacher_source_uacr_macro_rate",
        "unsupported_claim_macro_rate",
        "unsupported_claim_micro_rate",
        "hallucinated_claim_macro_rate",
        "hallucinated_claim_micro_rate",
        "new_claim_macro_rate",
        "new_claim_micro_rate",
        "new_hallucinated_claim_macro_rate",
        "new_hallucinated_claim_micro_rate",
    )
    for row in rows:
        copied = dict(row)
        for key in rate_keys:
            copied[key.replace("_rate", "_pct")] = pct(copied.get(key))
        enriched.append(copied)
    return enriched


def add_mean_std_pct_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    prefixes = (
        "teacher_source_uacr_macro",
        "teacher_source_uacr_micro",
        "teacher_source_any_uac",
        "teacher_source_ccr_macro",
        "teacher_source_ccr_micro",
        "clinical_uacr_macro",
        "clinical_uacr_micro",
        "clinical_any_uac",
        "clinical_ccr_macro",
        "clinical_ccr_micro",
        "same_fp16_answer_teacher_source_uacr_macro",
        "unsupported_claim_macro",
        "unsupported_claim_micro",
        "hallucinated_claim_macro",
        "hallucinated_claim_micro",
        "new_claim_macro",
        "new_claim_micro",
        "new_hallucinated_claim_macro",
        "new_hallucinated_claim_micro",
    )
    for row in rows:
        copied = dict(row)
        num_runs = int(copied.get("num_runs") or 0)
        for prefix in prefixes:
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


def latex_mean_std_metric(row: Dict[str, Any], prefix: str) -> str:
    mean_value = row.get(f"{prefix}_mean_rate")
    num_runs = int(row.get("num_runs") or 0)
    if mean_value is None:
        return "--"
    if num_runs <= 1:
        return pct(mean_value)
    return f"${pct(mean_value)} \pm {pct(row.get(f'{prefix}_std_rate'))}$"


def pct_signed(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value * 100:+.2f}"


def build_seed_comparison_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        model_key = normalize_text(row.get("model_key"))
        system = normalize_text(row.get("system"))
        seed_value = row.get("seed")
        if not model_key or system not in SYSTEM_ORDER or seed_value is None:
            continue
        grouped[(model_key, int(seed_value))][system] = row

    comparison_rows: List[Dict[str, Any]] = []
    for (model_key, seed), system_rows in grouped.items():
        baseline = system_rows.get("medmix_baseline")
        eaquant = system_rows.get("eaquant")
        baseline_rate = baseline.get("teacher_source_uacr_macro_rate") if baseline else None
        eaquant_rate = eaquant.get("teacher_source_uacr_macro_rate") if eaquant else None
        delta_rate = (
            float(eaquant_rate) - float(baseline_rate)
            if isinstance(baseline_rate, (float, int)) and isinstance(eaquant_rate, (float, int))
            else None
        )
        comparison_rows.append(
            {
                "dataset": DATASET,
                "model_key": model_key,
                "seed": seed,
                "baseline_teacher_source_uacr_rate": baseline_rate,
                "eaquant_teacher_source_uacr_rate": eaquant_rate,
                "delta_teacher_source_uacr_rate": delta_rate,
                "baseline_teacher_source_any_uac_rate": baseline.get("teacher_source_any_uac_rate") if baseline else None,
                "eaquant_teacher_source_any_uac_rate": eaquant.get("teacher_source_any_uac_rate") if eaquant else None,
                "baseline_teacher_source_ccr_rate": baseline.get("teacher_source_ccr_macro_rate") if baseline else None,
                "eaquant_teacher_source_ccr_rate": eaquant.get("teacher_source_ccr_macro_rate") if eaquant else None,
                "baseline_clinical_uacr_rate": baseline.get("clinical_uacr_macro_rate") if baseline else None,
                "eaquant_clinical_uacr_rate": eaquant.get("clinical_uacr_macro_rate") if eaquant else None,
                "baseline_same_answer_teacher_source_uacr_rate": (
                    baseline.get("same_fp16_answer_teacher_source_uacr_macro_rate") if baseline else None
                ),
                "eaquant_same_answer_teacher_source_uacr_rate": (
                    eaquant.get("same_fp16_answer_teacher_source_uacr_macro_rate") if eaquant else None
                ),
                "baseline_judged_claim_count": baseline.get("judged_claim_count") if baseline else None,
                "eaquant_judged_claim_count": eaquant.get("judged_claim_count") if eaquant else None,
                "baseline_unjudged_count": baseline.get("unjudged_count") if baseline else None,
                "eaquant_unjudged_count": eaquant.get("unjudged_count") if eaquant else None,
                "baseline_output_jsonl": baseline.get("output_jsonl") if baseline else "",
                "eaquant_output_jsonl": eaquant.get("output_jsonl") if eaquant else "",
            }
        )
    return sorted(comparison_rows, key=lambda row: (row["model_key"], int(row["seed"])))


def add_seed_comparison_pct_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for row in rows:
        copied = dict(row)
        for key in (
            "baseline_teacher_source_uacr_rate",
            "eaquant_teacher_source_uacr_rate",
            "baseline_teacher_source_any_uac_rate",
            "eaquant_teacher_source_any_uac_rate",
            "baseline_teacher_source_ccr_rate",
            "eaquant_teacher_source_ccr_rate",
            "baseline_clinical_uacr_rate",
            "eaquant_clinical_uacr_rate",
            "baseline_same_answer_teacher_source_uacr_rate",
            "eaquant_same_answer_teacher_source_uacr_rate",
        ):
            copied[key.replace("_rate", "_pct")] = pct(copied.get(key))
        copied["delta_teacher_source_uacr_pct"] = pct_signed(
            copied.get("delta_teacher_source_uacr_rate")
        )
        enriched.append(copied)
    return enriched


def latex_rate(value: Optional[float]) -> str:
    return "--" if value is None else pct(value)


def latex_delta(value: Optional[float]) -> str:
    return "--" if value is None else pct_signed(value)


def latex_by_seed_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model & Seed & MedMix PTQ & EAQuant & $\Delta$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model_key']} & {row['seed']} & "
            f"{latex_rate(row.get('baseline_teacher_source_uacr_rate'))} & "
            f"{latex_rate(row.get('eaquant_teacher_source_uacr_rate'))} & "
            f"{latex_delta(row.get('delta_teacher_source_uacr_rate'))} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{MedExpQA teacher-source unsupported added claim rate (UACR) by seed. "
                r"The baseline uses standard MedMix PTQ; EAQuant adds the evidence objectives. "
                r"Rates are macro percentages over judged clinical candidate claims; lower is better.}"
            ),
            r"\label{tab:medexpqa_teacher_source_uacr_by_seed}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def print_by_seed_table(rows: List[Dict[str, Any]]) -> None:
    print()
    print("Model, Seed, MedMix PTQ teacher-source UACR, EAQuant teacher-source UACR, Delta")
    for row in rows:
        print(
            f"{row['model_key']}, {row['seed']}, "
            f"{latex_rate(row.get('baseline_teacher_source_uacr_rate'))}, "
            f"{latex_rate(row.get('eaquant_teacher_source_uacr_rate'))}, "
            f"{latex_delta(row.get('delta_teacher_source_uacr_rate'))}"
        )


def iter_modes(mode: str) -> Iterable[str]:
    if mode == "both":
        yield "claims"
        yield "support"
    else:
        yield mode


def write_aggregate_outputs(
    *,
    args: argparse.Namespace,
    claim_summaries: List[Dict[str, Any]],
    support_summaries: List[Dict[str, Any]],
) -> None:
    rows = [
        row_from_support_summary(summary)
        for summary in support_summaries
        if not summary.get("skipped_missing_file")
    ]
    by_model_mean_std = aggregate_mean_std(rows)
    seed_comparison_rows = build_seed_comparison_rows(rows)
    rows_csv = add_pct_columns(rows)
    by_model_mean_std_csv = add_mean_std_pct_columns(by_model_mean_std)
    seed_comparison_rows_csv = add_seed_comparison_pct_columns(seed_comparison_rows)

    summary_path = args.output_dir / "unsupported_claim_rate_summary.json"
    rows_csv_path = args.output_dir / "unsupported_claim_rate_rows.csv"
    by_model_csv_path = args.output_dir / "unsupported_claim_rate_by_model_system_mean_std.csv"
    by_seed_csv_path = args.output_dir / "unsupported_claim_rate_by_seed_baseline_vs_eaquant.csv"
    latex_path = args.output_dir / "unsupported_claim_rate_by_seed_table.tex"

    write_json(
        summary_path,
        {
            "dataset": DATASET,
            "dataset_display": DATASET_DISPLAY,
            "schema_version": SCHEMA_VERSION,
            "mode": args.mode,
            "fp16_dir": str(args.fp16_dir),
            "medmix_baseline_dir": str(args.medmix_baseline_dir),
            "eaquant_dir": str(args.eaquant_dir),
            "output_dir": str(args.output_dir),
            "systems": args.systems,
            "models": args.models,
            "seeds": args.seeds,
            "judge_model": args.judge_model,
            "max_claims": args.max_claims,
            "incomplete_judgment_retries": args.incomplete_judgment_retries,
            "same_fp16_answer_filter_enabled": not bool(args.include_different_fp16_answers),
            "all_selected_systems_same_answer_filter_enabled": not bool(args.include_different_fp16_answers),
            "dry_run_prompts": bool(args.dry_run_prompts),
            "metric_definitions": {
                "teacher_source_uacr": (
                    "Primary metric. For each candidate rationale, extract atomic clinical claims. "
                    "By default, evaluate only examples where FP16 and all selected candidate "
                    "systems predict the same answer label. Judge each claim against the FP16 "
                    "teacher rationale only. "
                    "UACR is (unsupported_added + contradicted) / all judged clinical claims."
                ),
                "teacher_source_any_uac": (
                    "Fraction of examples with at least one unsupported_added or contradicted claim "
                    "under the primary teacher-source setting."
                ),
                "teacher_source_ccr": (
                    "Contradicted claim rate under the primary teacher-source setting."
                ),
                "clinical_uacr": (
                    "Secondary lenient metric using the FP16 teacher rationale plus clearly valid "
                    "standard medical knowledge as the reference. Gold answer and gold explanation "
                    "are excluded."
                ),
                "new_hallucinated_claim_rate": (
                    "Secondary diagnostic metric: among claims not present in FP16, fraction judged "
                    "unsupported_added or contradicted."
                ),
                "same_fp16_answer_only": (
                    "Subset where candidate_pred_answer equals fp16_pred_answer. With the default "
                    "all-system label filter this should match the evaluated set."
                ),
                "unjudged_count": (
                    "Claims omitted by the judge or returned with invalid labels. These are excluded "
                    "from denominators and retried up to incomplete_judgment_retries times."
                ),
            },
            "claim_summaries": claim_summaries,
            "support_summaries": support_summaries,
            "rows": rows,
            "by_model_system_mean_std": by_model_mean_std,
            "by_seed_baseline_vs_eaquant": seed_comparison_rows,
        },
    )
    row_fields = [
        "dataset",
        "system",
        "system_display",
        "model_key",
        "seed",
        "schema_version",
        "num_records",
        "num_scored_records",
        "num_clinical_scored_records",
        "candidate_claim_count",
        "judged_claim_count",
        "clinical_judged_claim_count",
        "source_supported_count",
        "valid_background_count",
        "unsupported_added_count",
        "contradicted_count",
        "added_claim_count",
        "added_unsupported_or_contradicted_count",
        "clinical_added_unsupported_or_contradicted_count",
        "not_a_claim_count",
        "unjudged_count",
        "new_claim_count",
        "new_unsupported_count",
        "new_contradicted_count",
        "teacher_source_uacr_macro_rate",
        "teacher_source_uacr_macro_pct",
        "teacher_source_uacr_micro_rate",
        "teacher_source_uacr_micro_pct",
        "teacher_source_any_uac_rate",
        "teacher_source_any_uac_pct",
        "teacher_source_ccr_macro_rate",
        "teacher_source_ccr_macro_pct",
        "teacher_source_ccr_micro_rate",
        "teacher_source_ccr_micro_pct",
        "clinical_uacr_macro_rate",
        "clinical_uacr_macro_pct",
        "clinical_uacr_micro_rate",
        "clinical_uacr_micro_pct",
        "clinical_any_uac_rate",
        "clinical_any_uac_pct",
        "clinical_ccr_macro_rate",
        "clinical_ccr_macro_pct",
        "same_fp16_answer_num_records",
        "same_fp16_answer_teacher_source_uacr_macro_rate",
        "same_fp16_answer_teacher_source_uacr_macro_pct",
        "same_fp16_answer_teacher_source_any_uac_rate",
        "same_fp16_answer_teacher_source_any_uac_pct",
        "same_fp16_answer_clinical_uacr_macro_rate",
        "same_fp16_answer_clinical_uacr_macro_pct",
        "candidate_correct_num_records",
        "candidate_correct_teacher_source_uacr_macro_rate",
        "candidate_correct_teacher_source_uacr_macro_pct",
        "unsupported_claim_macro_rate",
        "unsupported_claim_macro_pct",
        "hallucinated_claim_macro_rate",
        "hallucinated_claim_macro_pct",
        "new_hallucinated_claim_macro_rate",
        "new_hallucinated_claim_macro_pct",
        "num_parse_failures",
        "candidate_file",
        "output_jsonl",
    ]
    write_csv(rows_csv_path, rows_csv, row_fields)
    write_csv(
        by_model_csv_path,
        by_model_mean_std_csv,
        [
            "system",
            "system_display",
            "model_key",
            "schema_version",
            "num_runs",
            "seeds",
            "num_records",
            "num_scored_records",
            "candidate_claim_count",
            "judged_claim_count",
            "added_unsupported_or_contradicted_count",
            "clinical_added_unsupported_or_contradicted_count",
            "unjudged_count",
            "teacher_source_uacr_macro_mean_rate",
            "teacher_source_uacr_macro_std_rate",
            "teacher_source_uacr_macro_mean_pct",
            "teacher_source_uacr_macro_std_pct",
            "teacher_source_uacr_macro_mean_std_pct",
            "teacher_source_uacr_macro_num_values",
            "teacher_source_any_uac_mean_rate",
            "teacher_source_any_uac_mean_pct",
            "teacher_source_ccr_macro_mean_rate",
            "teacher_source_ccr_macro_mean_pct",
            "clinical_uacr_macro_mean_rate",
            "clinical_uacr_macro_mean_pct",
            "clinical_any_uac_mean_rate",
            "clinical_any_uac_mean_pct",
            "same_fp16_answer_teacher_source_uacr_macro_mean_rate",
            "same_fp16_answer_teacher_source_uacr_macro_mean_pct",
            "unsupported_claim_macro_mean_rate",
            "unsupported_claim_macro_mean_pct",
            "new_hallucinated_claim_macro_mean_rate",
            "new_hallucinated_claim_macro_mean_pct",
            "num_parse_failures",
        ],
    )
    write_csv(
        by_seed_csv_path,
        seed_comparison_rows_csv,
        [
            "dataset",
            "model_key",
            "seed",
            "baseline_teacher_source_uacr_rate",
            "baseline_teacher_source_uacr_pct",
            "eaquant_teacher_source_uacr_rate",
            "eaquant_teacher_source_uacr_pct",
            "delta_teacher_source_uacr_rate",
            "delta_teacher_source_uacr_pct",
            "baseline_teacher_source_any_uac_rate",
            "baseline_teacher_source_any_uac_pct",
            "eaquant_teacher_source_any_uac_rate",
            "eaquant_teacher_source_any_uac_pct",
            "baseline_teacher_source_ccr_rate",
            "baseline_teacher_source_ccr_pct",
            "eaquant_teacher_source_ccr_rate",
            "eaquant_teacher_source_ccr_pct",
            "baseline_clinical_uacr_rate",
            "baseline_clinical_uacr_pct",
            "eaquant_clinical_uacr_rate",
            "eaquant_clinical_uacr_pct",
            "baseline_same_answer_teacher_source_uacr_rate",
            "baseline_same_answer_teacher_source_uacr_pct",
            "eaquant_same_answer_teacher_source_uacr_rate",
            "eaquant_same_answer_teacher_source_uacr_pct",
            "baseline_judged_claim_count",
            "eaquant_judged_claim_count",
            "baseline_unjudged_count",
            "eaquant_unjudged_count",
            "baseline_output_jsonl",
            "eaquant_output_jsonl",
        ],
    )
    latex_path.write_text(latex_by_seed_table(seed_comparison_rows), encoding="utf-8")

    print_by_seed_table(seed_comparison_rows)
    print(f"[DONE] wrote {summary_path}")
    print(f"[DONE] wrote {rows_csv_path}")
    print(f"[DONE] wrote {by_model_csv_path}")
    print(f"[DONE] wrote {by_seed_csv_path}")
    print(f"[DONE] wrote {latex_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key(args)

    claim_summaries: List[Dict[str, Any]] = []
    support_summaries: List[Dict[str, Any]] = []

    for mode in iter_modes(args.mode):
        if mode == "claims":
            for system in args.systems:
                for model_key in args.models:
                    for seed in args.seeds:
                        claim_summaries.append(
                            run_claim_extraction_for_model_seed(
                                args=args,
                                api_key=api_key,
                                system=system,
                                model_key=model_key,
                                seed=seed,
                            )
                        )
        elif mode == "support":
            for system in args.systems:
                for model_key in args.models:
                    for seed in args.seeds:
                        support_summaries.append(
                            run_support_judgment_for_model_seed(
                                args=args,
                                api_key=api_key,
                                system=system,
                                model_key=model_key,
                                seed=seed,
                            )
                        )

    write_aggregate_outputs(
        args=args,
        claim_summaries=claim_summaries,
        support_summaries=support_summaries,
    )


if __name__ == "__main__":
    main()
