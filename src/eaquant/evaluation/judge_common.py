"""Shared judge utilities and the pairwise LLM-as-a-judge entry point."""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_ROOT = Path(os.environ.get("EAQUANT_DATA_ROOT", PROJECT_ROOT / "data"))
DATA_DIR = DATA_ROOT / "medexpqa"

DEFAULT_FP16_DIR = DATA_DIR / "train_baseline"
DEFAULT_MEDMIX_BASELINE_DIR = DATA_DIR / "train_quantized_medmix"
DEFAULT_EAQUANT_DIR = DATA_DIR / "llm"
DEFAULT_OUTPUT_DIR = DATA_DIR / "analysis" / "medmix_baseline_vs_eaquant_judge"
_API_KEY_FILE = os.environ.get("OPENAI_API_KEY_FILE")
DEFAULT_API_KEY_FILE = Path(_API_KEY_FILE) if _API_KEY_FILE else None

DEFAULT_JUDGE_MODEL = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5.4")
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

MODEL_KEYS = (
    "biomistral_7b",
    "llama3_instruct",
    "mistral_7b_instruct",
    "openbiollm",
)
DEFAULT_SEEDS = (0, 1, 2)
AB_CHOICES = ("A", "B", "Tie")
METHOD_CHOICES = ("medmix_baseline", "eaquant", "tie")

SYSTEM_PROMPT = (
    "You are an expert medical evaluator for multiple-choice clinical QA. "
    "Follow the rubric exactly. Output only the requested JSON object."
)

PAIRWISE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "medexpqa_pairwise_explanation_judgment",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "winner": {"type": "string", "enum": list(AB_CHOICES)},
                "reason": {"type": "string"},
            },
            "required": [
                "winner",
                "reason",
            ],
        },
    },
}

FP16_CLOSENESS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "medexpqa_fp16_closeness_judgment",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "winner": {"type": "string", "enum": list(AB_CHOICES)},
                "evidence_alignment_winner": {
                    "type": "string",
                    "enum": list(AB_CHOICES),
                },
                "reasoning_alignment_winner": {
                    "type": "string",
                    "enum": list(AB_CHOICES),
                },
                "drift_reduction_winner": {
                    "type": "string",
                    "enum": list(AB_CHOICES),
                },
                "reason": {"type": "string"},
            },
            "required": [
                "winner",
                "evidence_alignment_winner",
                "reasoning_alignment_winner",
                "drift_reduction_winner",
                "reason",
            ],
        },
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the OpenAI Chat Completions API to judge MedExpQA explanations. "
            "The default matrix compares FP16, standard MedMix PTQ, and EAQuant "
            "outputs for 4 model families "
            "and seeds 0/1/2."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["pairwise", "fp16_closeness", "both"],
        default="both",
        help=(
            "pairwise: choose the better explanation between MedMix PTQ and EAQuant. "
            "fp16_closeness: choose which quantized system is closer to FP16. "
            "both: run both evaluations."
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
        help="Directory containing train_quantized_medmix prediction JSONL files.",
    )
    parser.add_argument(
        "--eaquant_dir",
        type=Path,
        default=DEFAULT_EAQUANT_DIR,
        help=(
            "Directory containing EAQuant prediction JSONL files. The default "
            "uses files without tok/w ablation tags, i.e. the default 0.1 setting."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for judge JSONL and summary JSON outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_KEYS),
        choices=MODEL_KEYS,
        help="Model keys to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Quantized seeds to evaluate. Seed 0 maps to files without a seed suffix.",
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
        default=220,
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
        "--limit",
        type=int,
        default=None,
        help="Optional number of matched examples per model/seed/mode.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="0-based inclusive start index after matching triples.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="0-based exclusive end index after matching triples.",
    )
    parser.add_argument(
        "--only_correct_all",
        action="store_true",
        help="Only judge rows where FP16, MedMix PTQ, and EAQuant are all correct.",
    )
    parser.add_argument(
        "--only_same_answer",
        action="store_true",
        help="Only judge rows where FP16, MedMix PTQ, and EAQuant predicted the same answer.",
    )
    parser.add_argument(
        "--judge1_all_samples",
        action="store_true",
        help=(
            "For pairwise/Judge1, use all matched rows instead of the default "
            "MedMix PTQ/EAQuant same-predicted-answer subset."
        ),
    )
    parser.add_argument(
        "--judge2_all_samples",
        action="store_true",
        help=(
            "For fp16_closeness/Judge2, use all matched rows instead of the "
            "default all-three-same-predicted-answer subset."
        ),
    )
    parser.add_argument(
        "--swap_ab_order",
        action="store_true",
        help=(
            "Flip the deterministic A/B assignment for each row. Use this as a "
            "position-bias sanity check against a previous run without the flag."
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
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            records.append(parsed)
    return records


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_api_key_from_file(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" in line:
            _, line = line.split("=", 1)
            line = line.strip()
        line = line.strip().strip('"').strip("'")
        if line:
            return line
    return ""


def normalize_answer(value: Any) -> str:
    text = normalize_text(value).upper()
    if not text:
        return ""
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)
    if text[:1] in {"A", "B", "C", "D", "E"}:
        return text[:1]
    return text


def bool_or_none(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def extract_explanation(record: Dict[str, Any]) -> str:
    for key in ("pred_explanation", "used_explanation", "original_explanation"):
        value = normalize_text(record.get(key))
        if value:
            return value
    return ""


def extract_pred_answer(record: Dict[str, Any]) -> str:
    for key in ("pred_answer", "source_pred_answer"):
        value = normalize_answer(record.get(key))
        if value:
            return value
    return ""


def extract_gold_answer(record: Dict[str, Any]) -> str:
    return normalize_answer(record.get("gold_answer"))


def infer_correct(record: Dict[str, Any]) -> Optional[bool]:
    explicit = bool_or_none(record.get("is_correct"))
    if explicit is not None:
        return explicit
    pred = extract_pred_answer(record)
    gold = extract_gold_answer(record)
    if pred and gold:
        return pred == gold
    return None


def record_key(record: Dict[str, Any]) -> Optional[Tuple[str, ...]]:
    row_idx = normalize_text(record.get("row_idx"))
    split = normalize_text(record.get("split"))
    source_file = normalize_text(record.get("source_file"))
    if row_idx:
        return ("row_idx", split, source_file, row_idx)

    question_id_specific = normalize_text(record.get("question_id_specific"))
    if question_id_specific:
        return ("question_id_specific", question_id_specific)

    example_id = normalize_text(record.get("example_id"))
    question = normalize_text(record.get("question"))
    if example_id and question:
        return ("example_id_question", example_id, question)
    if example_id:
        return ("example_id", example_id)
    if question:
        return ("question", question)
    return None


def key_to_str(key: Optional[Tuple[str, ...]]) -> str:
    if key is None:
        return ""
    return "::".join(key)


def build_index(records: List[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, ...], Dict[str, Any]], Dict[str, int]]:
    index: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    stats = {"missing_key": 0, "duplicate_key": 0}
    for record in records:
        key = record_key(record)
        if key is None:
            stats["missing_key"] += 1
            continue
        if key in index:
            stats["duplicate_key"] += 1
            continue
        index[key] = record
    return index, stats


def format_options(options: Any) -> str:
    if not isinstance(options, dict):
        return "[No options provided]"
    lines = []
    for key in sorted(options):
        value = normalize_text(options.get(key))
        lines.append(f"{key}. {value}")
    return "\n".join(lines) if lines else "[No options provided]"


def format_answer(record: Dict[str, Any]) -> str:
    pred = extract_pred_answer(record)
    options = record.get("options")
    pred_text = ""
    if pred and isinstance(options, dict):
        pred_text = normalize_text(options.get(pred))
    if pred and pred_text:
        return f"{pred}. {pred_text}"
    if pred:
        return pred
    return "[No predicted answer]"


def format_explanation(explanation: str) -> str:
    explanation = normalize_text(explanation)
    if explanation:
        return explanation
    return "[No explanation provided]"


def get_seed_suffix(seed: int) -> str:
    return "" if seed == 0 else f"_seed{seed}"


def fp16_path(fp16_dir: Path, model_key: str) -> Path:
    return fp16_dir / f"test_{model_key}_original_train_baseline_predictions.jsonl"


def medmix_baseline_path(medmix_baseline_dir: Path, model_key: str, seed: int) -> Path:
    suffix = get_seed_suffix(seed)
    return (
        medmix_baseline_dir
        / f"test_{model_key}_ostquant_w4a4kv4_train_quantized_medmix{suffix}_predictions.jsonl"
    )


def eaquant_path(eaquant_dir: Path, model_key: str, seed: int) -> Path:
    llm_model_key = "openbiollm_8b" if model_key == "openbiollm" else model_key
    return (
        eaquant_dir
        / f"test_{llm_model_key}_ostquant_w4a4kv4_llm_qwen_imp0p02_q0p02_beta0p01_seed{seed}_predictions.jsonl"
    )


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


def build_pairwise_prompt(
    question: str,
    options: Any,
    response_a_record: Dict[str, Any],
    response_b_record: Dict[str, Any],
    question_type: str,
) -> str:
    return (
        "You are evaluating medical explanations for a multiple-choice clinical QA task.\n\n"
        "Choose which explanation is better as a clinical rationale.\n\n"
        "A better clinical rationale should:\n"
        "- be medically correct,\n"
        "- use relevant evidence from the question/options,\n"
        "- clearly support its selected answer,\n"
        "- avoid unsupported or hallucinated medical claims,\n"
        "- be concise but sufficiently specific.\n\n"
        "Do NOT reward an explanation just because it is longer or more polished.\n"
        "Reward details only when they directly improve the clinical reasoning.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options(options)}\n\n"
        "Response A selected answer:\n"
        f"{format_answer(response_a_record)}\n\n"
        "Response A explanation:\n"
        f"{format_explanation(extract_explanation(response_a_record))}\n\n"
        "Response B selected answer:\n"
        f"{format_answer(response_b_record)}\n\n"
        "Response B explanation:\n"
        f"{format_explanation(extract_explanation(response_b_record))}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "winner": "A" or "B" or "Tie",\n'
        '  "reason": "<1-3 concise sentences>"\n'
        "}"
    )


def build_fp16_closeness_prompt(
    question: str,
    options: Any,
    fp16_record: Dict[str, Any],
    response_a_record: Dict[str, Any],
    response_b_record: Dict[str, Any],
    question_type: str,
) -> str:
    gold_answer = extract_gold_answer(fp16_record)
    return (
        "You are evaluating which quantized explanation better preserves the reasoning "
        "of the original FP16 explanation.\n\n"
        "Your goal is NOT to judge which explanation is generally better.\n"
        "Instead, determine which explanation more closely matches the FP16 explanation in:\n"
        "- clinical evidence used\n"
        "- reasoning structure\n"
        "- diagnostic logic\n"
        "- answer-supporting rationale\n"
        "- avoidance of reasoning drift\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options(options)}\n\n"
        f"Gold answer:\n{gold_answer}\n\n"
        "Selected answer:\n"
        f"{format_answer(fp16_record)}\n\n"
        "Reference FP16 explanation:\n"
        f"{format_explanation(extract_explanation(fp16_record))}\n\n"
        "Quantized explanation A:\n"
        f"{format_explanation(extract_explanation(response_a_record))}\n\n"
        "Quantized explanation B:\n"
        f"{format_explanation(extract_explanation(response_b_record))}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "winner": "A" or "B" or "Tie",\n'
        '  "evidence_alignment_winner": "A" or "B" or "Tie",\n'
        '  "reasoning_alignment_winner": "A" or "B" or "Tie",\n'
        '  "drift_reduction_winner": "A" or "B" or "Tie",\n'
        '  "reason": "<1-3 concise sentences>"\n'
        "}"
    )

def build_messages(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def response_format_for_mode(mode: str) -> Dict[str, Any]:
    if mode == "pairwise":
        return PAIRWISE_SCHEMA
    if mode == "fp16_closeness":
        return FP16_CLOSENESS_SCHEMA
    raise ValueError(f"Unsupported mode: {mode}")


def expected_result_keys(mode: str) -> Tuple[str, ...]:
    if mode == "pairwise":
        return (
            "winner",
            "reason",
        )
    if mode == "fp16_closeness":
        return (
            "winner",
            "evidence_alignment_winner",
            "reasoning_alignment_winner",
            "drift_reduction_winner",
            "reason",
        )
    raise ValueError(f"Unsupported mode: {mode}")


def coerce_choice(value: Any) -> Optional[str]:
    text = normalize_text(value)
    upper = text.upper()
    lowered = text.lower()
    if upper in {"A", "RESPONSE A", "EXPLANATION A"}:
        return "A"
    if upper in {"B", "RESPONSE B", "EXPLANATION B"}:
        return "B"
    if lowered in {"tie", "tied", "same", "equal", "neither", "both"}:
        return "tie"
    return None

def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_judge_result(mode: str, parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    normalized: Dict[str, Any] = {}
    for key in expected_result_keys(mode):
        if key == "confidence":
            try:
                confidence = int(parsed.get(key))
            except (TypeError, ValueError):
                return None
            normalized[key] = max(1, min(5, confidence))
        elif key in {"rationale", "reason"}:
            normalized[key] = normalize_text(parsed.get(key))
        else:
            choice = coerce_choice(parsed.get(key))
            if choice is None:
                return None
            normalized[key] = choice
    return normalized


def ab_assignment(
    *,
    mode: str,
    model_key: str,
    seed: int,
    pair_key: str,
    medmix_baseline_record: Dict[str, Any],
    eaquant_record: Dict[str, Any],
    swap_ab_order: bool,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    seed_text = f"{mode}|{model_key}|{seed}|{pair_key}"
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    medmix_baseline_is_a = digest[0] % 2 == 0
    if medmix_baseline_is_a:
        records_by_ab = {"A": medmix_baseline_record, "B": eaquant_record}
        source_by_ab = {"A": "medmix_baseline", "B": "eaquant"}
    else:
        records_by_ab = {"A": eaquant_record, "B": medmix_baseline_record}
        source_by_ab = {"A": "eaquant", "B": "medmix_baseline"}
    if swap_ab_order:
        records_by_ab = {"A": records_by_ab["B"], "B": records_by_ab["A"]}
        source_by_ab = {"A": source_by_ab["B"], "B": source_by_ab["A"]}
    ab_by_source = {source: label for label, source in source_by_ab.items()}
    return records_by_ab, source_by_ab, ab_by_source


def map_ab_choice_to_method(choice: Any, source_by_ab: Dict[str, str]) -> str:
    normalized = coerce_choice(choice)
    if normalized == "tie" or normalized is None:
        return "tie"
    return source_by_ab.get(normalized, "tie")


def map_judge_result_to_methods(
    mode: str,
    judge_result: Dict[str, Any],
    source_by_ab: Dict[str, str],
) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {
        "candidate_a_source": source_by_ab["A"],
        "candidate_b_source": source_by_ab["B"],
    }
    for key in expected_result_keys(mode):
        if key in {"confidence", "rationale", "reason"}:
            continue
        ab_value = judge_result.get(key)
        mapped[f"{key}_ab"] = ab_value
        mapped[key] = map_ab_choice_to_method(ab_value, source_by_ab)
    return mapped


def extract_chat_content(response_payload: Dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ValueError("OpenAI response missing choices[0].message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(normalize_text(part.get("text")))
        return "\n".join(part for part in parts if part)
    raise ValueError("OpenAI response missing text content")


def read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def should_retry_status(status: int) -> bool:
    return status in {408, 409, 429, 500, 502, 503, 504}


def call_openai_chat_completion(
    *,
    api_key: str,
    api_url: str,
    judge_model: str,
    messages: List[Dict[str, str]],
    response_format: Dict[str, Any],
    max_completion_tokens: int,
    temperature: Optional[float],
    reasoning_effort: Optional[str],
    request_timeout: float,
) -> Tuple[Dict[str, Any], str]:
    payload: Dict[str, Any] = {
        "model": judge_model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "response_format": response_format,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=request_timeout) as response:
        raw = response.read().decode("utf-8")
    response_payload = json.loads(raw)
    return response_payload, extract_chat_content(response_payload)


def run_judge_call(
    *,
    args: argparse.Namespace,
    api_key: str,
    mode: str,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    response_format = response_format_for_mode(mode)
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
            normalized = normalize_judge_result(mode, parsed)
            if normalized is None:
                last_error = f"Parsed JSON did not match expected {mode} schema: {parsed}"
                raise ValueError(last_error)
            return {
                **normalized,
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
        print(f"[WARN] {mode} judge attempt {attempt + 1} failed: {last_error}", file=sys.stderr)
        print(f"[WARN] sleeping {sleep_s:.1f}s before retry", file=sys.stderr)
        time.sleep(sleep_s)

    fallback: Dict[str, Any]
    if mode == "pairwise":
        fallback = {
            "winner": "tie",
            "reason": "Judge call failed; defaulted to tie.",
        }
    else:
        fallback = {
            "winner": "tie",
            "evidence_alignment_winner": "tie",
            "reasoning_alignment_winner": "tie",
            "drift_reduction_winner": "tie",
            "reason": "Judge call failed; defaulted to tie.",
        }
    return {
        **fallback,
        "judge_parse_success": False,
        "judge_attempt_count": args.max_retries + 1,
        "judge_error": last_error,
        "raw_judge_response": raw_response,
        "openai_response_id": "",
        "openai_usage": {},
    }


def make_matched_triples(
    fp16_records: List[Dict[str, Any]],
    medmix_baseline_records: List[Dict[str, Any]],
    eaquant_records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    baseline_index, baseline_index_stats = build_index(medmix_baseline_records)
    eaquant_index, eaquant_index_stats = build_index(eaquant_records)
    triples: List[Dict[str, Any]] = []
    stats = {
        "num_fp16_rows": len(fp16_records),
        "num_medmix_baseline_rows": len(medmix_baseline_records),
        "num_eaquant_rows": len(eaquant_records),
        "medmix_baseline_index": baseline_index_stats,
        "eaquant_index": eaquant_index_stats,
        "num_fp16_rows_missing_key": 0,
        "num_missing_medmix_baseline_match": 0,
        "num_missing_eaquant_match": 0,
    }

    for fp16_record in fp16_records:
        key = record_key(fp16_record)
        if key is None:
            stats["num_fp16_rows_missing_key"] += 1
            continue
        baseline_record = baseline_index.get(key)
        eaquant_record = eaquant_index.get(key)
        if baseline_record is None:
            stats["num_missing_medmix_baseline_match"] += 1
            continue
        if eaquant_record is None:
            stats["num_missing_eaquant_match"] += 1
            continue
        triples.append(
            {
                "pair_key": key_to_str(key),
                "fp16_record": fp16_record,
                "medmix_baseline_record": baseline_record,
                "eaquant_record": eaquant_record,
            }
        )

    stats["num_matched_triples"] = len(triples)
    return triples, stats


def apply_filters(
    triples: List[Dict[str, Any]],
    args: argparse.Namespace,
    mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    judge1_same_medmix_baseline_eaquant_answer_default = (
        mode == "pairwise" and not args.judge1_all_samples and not args.only_same_answer
    )
    judge2_same_answer_default = mode == "fp16_closeness" and not args.judge2_all_samples
    stats = {
        "num_before_filters": len(triples),
        "only_correct_all": bool(args.only_correct_all),
        "only_same_answer": bool(args.only_same_answer),
        "judge1_all_samples": bool(args.judge1_all_samples),
        "judge1_same_medmix_baseline_eaquant_answer_default": bool(
            judge1_same_medmix_baseline_eaquant_answer_default
        ),
        "judge2_same_answer_default": bool(judge2_same_answer_default),
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "limit": args.limit,
    }
    filtered = triples

    if args.only_correct_all:
        filtered = [
            item
            for item in filtered
            if infer_correct(item["fp16_record"]) is True
            and infer_correct(item["medmix_baseline_record"]) is True
            and infer_correct(item["eaquant_record"]) is True
        ]
    stats["num_after_only_correct_all"] = len(filtered)

    if judge1_same_medmix_baseline_eaquant_answer_default:
        filtered = [
            item
            for item in filtered
            if extract_pred_answer(item["medmix_baseline_record"])
            and extract_pred_answer(item["medmix_baseline_record"])
            == extract_pred_answer(item["eaquant_record"])
        ]
    stats["num_after_judge1_same_medmix_baseline_eaquant_answer"] = len(filtered)

    same_answer_filter = args.only_same_answer or judge2_same_answer_default

    if same_answer_filter:
        filtered = [
            item
            for item in filtered
            if extract_pred_answer(item["fp16_record"])
            and extract_pred_answer(item["fp16_record"])
            == extract_pred_answer(item["medmix_baseline_record"])
            == extract_pred_answer(item["eaquant_record"])
        ]
    stats["num_after_only_same_answer"] = len(filtered)

    filtered = filtered[args.start_idx : args.end_idx]
    if args.limit is not None:
        filtered = filtered[: args.limit]
    stats["num_after_slice_and_limit"] = len(filtered)
    return filtered, stats


def prompt_for_mode(
    mode: str,
    model_key: str,
    seed: int,
    triple: Dict[str, Any],
    swap_ab_order: bool,
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    fp16_record = triple["fp16_record"]
    medmix_baseline_record = triple["medmix_baseline_record"]
    eaquant_record = triple["eaquant_record"]
    question = normalize_text(
        fp16_record.get("question")
        or medmix_baseline_record.get("question")
        or eaquant_record.get("question")
    )
    options = fp16_record.get("options") or medmix_baseline_record.get("options") or eaquant_record.get("options") or {}
    question_type = normalize_text(
        fp16_record.get("question_type")
        or medmix_baseline_record.get("question_type")
        or eaquant_record.get("question_type")
    )
    records_by_ab, source_by_ab, ab_by_source = ab_assignment(
        mode=mode,
        model_key=model_key,
        seed=seed,
        pair_key=normalize_text(triple.get("pair_key")),
        medmix_baseline_record=medmix_baseline_record,
        eaquant_record=eaquant_record,
        swap_ab_order=swap_ab_order,
    )
    if mode == "pairwise":
        prompt = build_pairwise_prompt(
            question=question,
            options=options,
            response_a_record=records_by_ab["A"],
            response_b_record=records_by_ab["B"],
            question_type=question_type,
        )
        return prompt, source_by_ab, ab_by_source
    if mode == "fp16_closeness":
        prompt = build_fp16_closeness_prompt(
            question=question,
            options=options,
            fp16_record=fp16_record,
            response_a_record=records_by_ab["A"],
            response_b_record=records_by_ab["B"],
            question_type=question_type,
        )
        return prompt, source_by_ab, ab_by_source
    raise ValueError(f"Unsupported mode: {mode}")

def base_output_record(
    *,
    mode: str,
    model_key: str,
    seed: int,
    triple: Dict[str, Any],
    fp16_file: Path,
    medmix_baseline_file: Path,
    eaquant_file: Path,
) -> Dict[str, Any]:
    fp16_record = triple["fp16_record"]
    medmix_baseline_record = triple["medmix_baseline_record"]
    eaquant_record = triple["eaquant_record"]
    question = normalize_text(
        fp16_record.get("question")
        or medmix_baseline_record.get("question")
        or eaquant_record.get("question")
    )
    options = fp16_record.get("options") or medmix_baseline_record.get("options") or eaquant_record.get("options") or {}
    return {
        "mode": mode,
        "model_key": model_key,
        "seed": seed,
        "pair_key": triple["pair_key"],
        "example_id": fp16_record.get("example_id")
        or medmix_baseline_record.get("example_id")
        or eaquant_record.get("example_id"),
        "split": fp16_record.get("split") or medmix_baseline_record.get("split") or eaquant_record.get("split"),
        "question_type": fp16_record.get("question_type")
        or medmix_baseline_record.get("question_type")
        or eaquant_record.get("question_type"),
        "source_file": fp16_record.get("source_file")
        or medmix_baseline_record.get("source_file")
        or eaquant_record.get("source_file"),
        "row_idx": fp16_record.get("row_idx")
        if fp16_record.get("row_idx") is not None
        else medmix_baseline_record.get("row_idx")
        if medmix_baseline_record.get("row_idx") is not None
        else eaquant_record.get("row_idx"),
        "question": question,
        "options": options,
        "gold_answer": extract_gold_answer(fp16_record)
        or extract_gold_answer(medmix_baseline_record)
        or extract_gold_answer(eaquant_record),
        "fp16_pred_answer": extract_pred_answer(fp16_record),
        "fp16_pred_answer_text": format_answer(fp16_record),
        "fp16_is_correct": infer_correct(fp16_record),
        "fp16_explanation": extract_explanation(fp16_record),
        "medmix_baseline_pred_answer": extract_pred_answer(medmix_baseline_record),
        "medmix_baseline_pred_answer_text": format_answer(medmix_baseline_record),
        "medmix_baseline_is_correct": infer_correct(medmix_baseline_record),
        "medmix_baseline_explanation": extract_explanation(medmix_baseline_record),
        "eaquant_pred_answer": extract_pred_answer(eaquant_record),
        "eaquant_pred_answer_text": format_answer(eaquant_record),
        "eaquant_is_correct": infer_correct(eaquant_record),
        "eaquant_explanation": extract_explanation(eaquant_record),
        "fp16_file": str(fp16_file),
        "medmix_baseline_file": str(medmix_baseline_file),
        "eaquant_file": str(eaquant_file),
    }


def output_paths(
    output_dir: Path,
    mode: str,
    model_key: str,
    seed: int,
    judge_model: str,
    dry_run_prompts: bool,
    swap_ab_order: bool,
) -> Tuple[Path, Path]:
    model_slug = slugify(judge_model)
    stem = f"chatgpt_abblind_{mode}_{model_key}_seed{seed}_by_{model_slug}"
    if swap_ab_order:
        stem = f"{stem}_swapab"
    if dry_run_prompts:
        stem = f"{stem}_prompts"
    jsonl_path = output_dir / f"{stem}.jsonl"
    summary_path = output_dir / f"{stem}_summary.json"
    return jsonl_path, summary_path


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


def summarize_records(
    *,
    mode: str,
    records: List[Dict[str, Any]],
    dry_run_prompts: bool,
) -> Dict[str, Any]:
    if dry_run_prompts:
        return {"num_records": len(records), "dry_run_prompts": True}

    main_key = "winner"
    result_keys = [
        key
        for key in expected_result_keys(mode)
        if key not in {"confidence", "rationale", "reason"}
    ]
    result_key_summaries = {}
    for result_key in result_keys:
        result_ab_key = f"{result_key}_ab"
        result_counts = Counter(
            normalize_text(record.get(result_key)).lower() for record in records
        )
        result_ab_counts = Counter(
            normalize_text(record.get(result_ab_key)) for record in records
        )
        result_decisive = int(
            result_counts.get("eaquant", 0) + result_counts.get("medmix_baseline", 0)
        )
        result_key_summaries[result_key] = {
            "result_key": result_key,
            "ab_result_key": result_ab_key,
            "winner_counts": {
                "medmix_baseline": int(result_counts.get("medmix_baseline", 0)),
                "eaquant": int(result_counts.get("eaquant", 0)),
                "tie": int(result_counts.get("tie", 0)),
            },
            "winner_rates": {
                "medmix_baseline": safe_rate(int(result_counts.get("medmix_baseline", 0)), len(records)),
                "eaquant": safe_rate(int(result_counts.get("eaquant", 0)), len(records)),
                "tie": safe_rate(int(result_counts.get("tie", 0)), len(records)),
            },
            "ab_winner_counts": {
                "A": int(result_ab_counts.get("A", 0)),
                "B": int(result_ab_counts.get("B", 0)),
                "tie": int(result_ab_counts.get("tie", 0)),
            },
            "eaquant_win_rate_excluding_ties": safe_rate(
                int(result_counts.get("eaquant", 0)), result_decisive
            ),
            "medmix_baseline_win_rate_excluding_ties": safe_rate(
                int(result_counts.get("medmix_baseline", 0)), result_decisive
            ),
        }
    main_ab_key = f"{main_key}_ab"
    counts = Counter(normalize_text(record.get(main_key)).lower() for record in records)
    ab_counts = Counter(normalize_text(record.get(main_ab_key)) for record in records)
    candidate_a_sources = Counter(
        normalize_text(record.get("candidate_a_source")).lower() for record in records
    )
    candidate_b_sources = Counter(
        normalize_text(record.get("candidate_b_source")).lower() for record in records
    )
    parse_failures = sum(1 for record in records if record.get("judge_parse_success") is False)
    model_calls = sum(1 for record in records if record.get("judge_model_call_skipped") is not True)
    confidence_values = [
        int(record.get("confidence"))
        for record in records
        if isinstance(record.get("confidence"), int)
    ]
    total = len(records)
    decisive = int(counts.get("eaquant", 0) + counts.get("medmix_baseline", 0))
    return {
        "num_records": total,
        "main_result_key": main_key,
        "main_ab_result_key": main_ab_key,
        "winner_counts": {
            "medmix_baseline": int(counts.get("medmix_baseline", 0)),
            "eaquant": int(counts.get("eaquant", 0)),
            "tie": int(counts.get("tie", 0)),
        },
        "winner_rates": {
            "medmix_baseline": safe_rate(int(counts.get("medmix_baseline", 0)), total),
            "eaquant": safe_rate(int(counts.get("eaquant", 0)), total),
            "tie": safe_rate(int(counts.get("tie", 0)), total),
        },
        "ab_winner_counts": {
            "A": int(ab_counts.get("A", 0)),
            "B": int(ab_counts.get("B", 0)),
            "tie": int(ab_counts.get("tie", 0)),
        },
        "result_key_summaries": result_key_summaries,
        "candidate_a_source_counts": {
            "medmix_baseline": int(candidate_a_sources.get("medmix_baseline", 0)),
            "eaquant": int(candidate_a_sources.get("eaquant", 0)),
        },
        "candidate_b_source_counts": {
            "medmix_baseline": int(candidate_b_sources.get("medmix_baseline", 0)),
            "eaquant": int(candidate_b_sources.get("eaquant", 0)),
        },
        "eaquant_win_rate_excluding_ties": safe_rate(int(counts.get("eaquant", 0)), decisive),
        "medmix_baseline_win_rate_excluding_ties": safe_rate(int(counts.get("medmix_baseline", 0)), decisive),
        "num_parse_failures_defaulted_to_tie": parse_failures,
        "num_model_calls": model_calls,
        "avg_confidence": (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else None
        ),
    }


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def load_existing_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def run_one_mode_for_model_seed(
    *,
    args: argparse.Namespace,
    api_key: str,
    mode: str,
    model_key: str,
    seed: int,
) -> Dict[str, Any]:
    fp16_file = fp16_path(args.fp16_dir, model_key)
    medmix_baseline_file = medmix_baseline_path(args.medmix_baseline_dir, model_key, seed)
    eaquant_file = eaquant_path(args.eaquant_dir, model_key, seed)
    for path in (fp16_file, medmix_baseline_file, eaquant_file):
        if not path.exists():
            raise FileNotFoundError(path)

    output_jsonl, summary_json = output_paths(
        output_dir=args.output_dir,
        mode=mode,
        model_key=model_key,
        seed=seed,
        judge_model=args.judge_model,
        dry_run_prompts=args.dry_run_prompts,
        swap_ab_order=args.swap_ab_order,
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_jsonl.exists():
        output_jsonl.unlink()

    fp16_records = read_jsonl(fp16_file)
    medmix_baseline_records = read_jsonl(medmix_baseline_file)
    eaquant_records = read_jsonl(eaquant_file)
    triples, match_stats = make_matched_triples(fp16_records, medmix_baseline_records, eaquant_records)
    triples, filter_stats = apply_filters(triples, args, mode)
    if not triples:
        raise ValueError(f"No matched triples left after filters for {model_key} seed{seed}")

    active_pair_keys = {normalize_text(item.get("pair_key")) for item in triples}
    completed_pair_keys = read_completed_pair_keys(output_jsonl) & active_pair_keys
    existing_records = [
        record
        for record in load_existing_records(output_jsonl)
        if normalize_text(record.get("pair_key")) in active_pair_keys
    ]
    new_records: List[Dict[str, Any]] = []

    print(
        f"[INFO] {mode} {model_key} seed{seed}: "
        f"{len(triples)} matched triples, {len(completed_pair_keys)} already done"
    )

    for idx, triple in enumerate(triples, start=1):
        pair_key = normalize_text(triple.get("pair_key"))
        if pair_key in completed_pair_keys and not args.overwrite:
            continue

        user_prompt, source_by_ab, ab_by_source = prompt_for_mode(
            mode,
            model_key,
            seed,
            triple,
            args.swap_ab_order,
        )
        messages = build_messages(user_prompt)
        record = base_output_record(
            mode=mode,
            model_key=model_key,
            seed=seed,
            triple=triple,
            fp16_file=fp16_file,
            medmix_baseline_file=medmix_baseline_file,
            eaquant_file=eaquant_file,
        )
        record.update(
            {
                "judge_model": args.judge_model,
                "judge_api_url": args.api_url,
                "judge_model_call_skipped": bool(args.dry_run_prompts),
                "swap_ab_order": bool(args.swap_ab_order),
                "candidate_a_source": source_by_ab["A"],
                "candidate_b_source": source_by_ab["B"],
                "medmix_baseline_ab_label": ab_by_source["medmix_baseline"],
                "eaquant_ab_label": ab_by_source["eaquant"],
            }
        )

        if args.verbose_prompts or args.dry_run_prompts:
            record["system_prompt"] = SYSTEM_PROMPT
            record["user_prompt"] = user_prompt

        if args.dry_run_prompts:
            if mode == "pairwise":
                judge_result = {
                    "winner": "",
                    "winner_ab": "",
                    "reason": "",
                    "judge_parse_success": None,
                    "judge_attempt_count": 0,
                    "judge_error": "",
                    "raw_judge_response": "",
                    "openai_response_id": "",
                    "openai_usage": {},
                }
            else:
                judge_result = {
                    "winner": "",
                    "winner_ab": "",
                    "evidence_alignment_winner": "",
                    "evidence_alignment_winner_ab": "",
                    "reasoning_alignment_winner": "",
                    "reasoning_alignment_winner_ab": "",
                    "drift_reduction_winner": "",
                    "drift_reduction_winner_ab": "",
                    "reason": "",
                    "judge_parse_success": None,
                    "judge_attempt_count": 0,
                    "judge_error": "",
                    "raw_judge_response": "",
                    "openai_response_id": "",
                    "openai_usage": {},
                }
        else:
            judge_result = run_judge_call(
                args=args,
                api_key=api_key,
                mode=mode,
                messages=messages,
            )
            judge_result.update(
                map_judge_result_to_methods(
                    mode=mode,
                    judge_result=judge_result,
                    source_by_ab=source_by_ab,
                )
            )

        record.update(judge_result)
        append_jsonl(output_jsonl, record)
        new_records.append(record)

        if args.dry_run_prompts:
            print(f"[{idx}/{len(triples)}] wrote prompt {model_key} seed{seed} {pair_key}")
        else:
            main_result = record.get("winner")
            print(
                f"[{idx}/{len(triples)}] {model_key} seed{seed} {pair_key} "
                f"{mode}={main_result} "
                f"attempts={record.get('judge_attempt_count')}"
            )

    all_records = existing_records + new_records if not args.overwrite else new_records
    summary = {
        "mode": mode,
        "model_key": model_key,
        "seed": seed,
        "judge_model": args.judge_model,
        "dry_run_prompts": bool(args.dry_run_prompts),
        "swap_ab_order": bool(args.swap_ab_order),
        "fp16_file": str(fp16_file),
        "medmix_baseline_file": str(medmix_baseline_file),
        "eaquant_file": str(eaquant_file),
        "output_jsonl": str(output_jsonl),
        "summary_json": str(summary_json),
        "match_stats": match_stats,
        "filter_stats": filter_stats,
        "num_existing_records_before_run": len(existing_records),
        "num_new_records_this_run": len(new_records),
        **summarize_records(
            mode=mode,
            records=all_records,
            dry_run_prompts=args.dry_run_prompts,
        ),
    }
    write_json(summary_json, summary)
    print(f"[DONE] wrote {output_jsonl}")
    print(f"[DONE] wrote {summary_json}")
    return summary


def iter_modes(mode: str) -> Iterable[str]:
    if mode == "both":
        yield "pairwise"
        yield "fp16_closeness"
    else:
        yield mode


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api_key = ""
    if not args.dry_run_prompts:
        api_key = normalize_text(os.environ.get(args.api_key_env))
        if not api_key:
            api_key = load_api_key_from_file(args.api_key_file)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set {args.api_key_env}=..., put it in "
                f"{args.api_key_file}, or use --dry_run_prompts."
            )

    summaries: List[Dict[str, Any]] = []
    for mode in iter_modes(args.mode):
        for model_key in args.models:
            for seed in args.seeds:
                summaries.append(
                    run_one_mode_for_model_seed(
                        args=args,
                        api_key=api_key,
                        mode=mode,
                        model_key=model_key,
                        seed=seed,
                    )
                )

    aggregate_path = args.output_dir / (
        f"chatgpt_abblind_{args.mode}_aggregate_by_{slugify(args.judge_model)}"
        f"{'_swapab' if args.swap_ab_order else ''}"
        f"{'_prompts' if args.dry_run_prompts else ''}_summary.json"
    )
    write_json(
        aggregate_path,
        {
            "mode": args.mode,
            "judge_model": args.judge_model,
            "models": args.models,
            "seeds": args.seeds,
            "dry_run_prompts": bool(args.dry_run_prompts),
            "swap_ab_order": bool(args.swap_ab_order),
            "summaries": summaries,
        },
    )
    print(f"[DONE] wrote aggregate summary {aggregate_path}")


if __name__ == "__main__":
    main()
