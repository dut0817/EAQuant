#!/usr/bin/env python3
"""Derive answer-sanitized full and matched-random faithfulness caches.

This script never runs Qwen or either teacher model.  It freezes the ordered
usable cohort from an existing Qwen-evidence cache and rewrites only the fields
consumed by ``FaithfulnessDataCollator``.

The two branches of the current method are deliberately kept distinct:

* token KL uses Q evidence and its loss weights;
* answer-recovery KL uses all F evidence units.

For ``random_noanswer_matched`` every F unit receives a random replacement from
the original rationale with the same word count and, when a tokenizer is
available, the closest token budget.  Replacements corresponding to source Q
units become the random Q mask and inherit the source Q loss weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
MODEL_ROOT = Path(os.environ.get("EAQUANT_MODEL_ROOT", PROJECT_ROOT.parent / "models"))

DEFAULT_MODELS = (
    "llama3_8b_instruct",
    "mistral_7b_instruct",
    "openbiollm_8b",
    "biomistral_7b",
)

MODEL_PATHS = {
    "llama3_8b_instruct": MODEL_ROOT / "Meta-Llama-3-8B-Instruct",
    "mistral_7b_instruct": MODEL_ROOT / "Mistral-7B-Instruct-v0.3",
    "openbiollm_8b": MODEL_ROOT / "Llama3-OpenBioLLM-8B",
    "biomistral_7b": MODEL_ROOT / "BioMistral-7B",
}

VARIANTS = ("full_noanswer", "random_noanswer_matched")

ANSWER_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "when", "which", "with",
}

FORBIDDEN_PHRASES = (
    "the correct answer is",
    "therefore",
    "thus",
    "the selected answer",
    "the most appropriate choice",
    "end of response",
    "please note",
    "hypothetical scenario",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full-no-answer and matched-random baseline caches."
    )
    parser.add_argument("--repo_dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "cache" / "ablation" / "rationale",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument(
        "--source_tag", default="imp0p02_q0p02_beta0p01"
    )
    parser.add_argument("--selection_seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--tokenizer_mode",
        choices=("required", "auto", "off"),
        default="auto",
        help="Use each target-model tokenizer to match random span budgets.",
    )
    parser.add_argument(
        "--answer_leakage_mode",
        choices=("strict", "relaxed", "off"),
        default="relaxed",
    )
    parser.add_argument("--short_answer_symbol_max_chars", type=int, default=6)
    parser.add_argument("--short_answer_symbol_max_words", type=int, default=2)
    parser.add_argument("--answer_overlap_reject_ratio", type=float, default=0.6)
    parser.add_argument("--answer_overlap_reject_min_tokens", type=int, default=2)
    parser.add_argument(
        "--random_word_tolerance",
        type=int,
        default=6,
        help=(
            "Allow this many words of window-width slack when it improves exact "
            "target-model token-budget matching."
        ),
    )
    parser.add_argument("--max_backtrack_nodes", type=int, default=100000)
    parser.add_argument("--max_candidates_per_unit", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", safe_text(text)))


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", safe_text(text)).casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_values(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def source_record_is_usable(record: Dict[str, Any]) -> bool:
    cache_usable = record.get("faith_cache_usable")
    if cache_usable is None:
        cache_usable = record.get("faith_recovery_selected_success", False)
    return bool(
        record.get("is_correct", False)
        and cache_usable
        and record.get("faith_token_user_prompt")
        and record.get("faith_token_target_text")
        and record.get("faith_token_selected_char_spans")
        and record.get("faith_recovery_selected_user_prompt")
        and record.get("faith_selected_spans")
    )


def target_label(record: Dict[str, Any]) -> str:
    return safe_text(record.get("pred_answer") or record.get("gold_answer")).upper()


def selected_answer_texts(record: Dict[str, Any]) -> List[str]:
    label = target_label(record)
    options = record.get("options") if isinstance(record.get("options"), dict) else {}
    texts = [
        safe_text(record.get("pred_answer_text")),
        safe_text(record.get("gold_option_text")),
        safe_text(options.get(label)),
    ]
    result: List[str] = []
    seen = set()
    for text in texts:
        key = normalized(text)
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def answer_tokens(text: str) -> List[str]:
    tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+(?:[+\-][A-Za-z0-9]+)?", safe_text(text))
    ]
    return [token for token in tokens if token not in ANSWER_STOPWORDS]


def is_short_answer_symbol(text: str, args: argparse.Namespace) -> bool:
    value = safe_text(text)
    compact = re.sub(r"\s+", "", value)
    if not compact:
        return False
    if len(compact) > max(args.short_answer_symbol_max_chars, 1):
        return False
    if word_count(value) > max(args.short_answer_symbol_max_words, 1):
        return False
    if re.search(r"[0-9+\-/]", compact):
        return True
    alnum = re.sub(r"[^A-Za-z0-9]", "", compact)
    return bool(alnum and not alnum.islower())


def answer_overlap_features(
    text: str, record: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    value_norm = normalized(text)
    span_tokens = set(answer_tokens(text))
    label = re.escape(target_label(record))
    has_label = bool(re.search(rf"(^|\s){label}\s*[\.)]", safe_text(text), re.I))
    best: Dict[str, Any] = {
        "has_option_label": has_label,
        "has_selected_answer_text": False,
        "selected_answer_is_short_symbol": False,
        "overlap_count": 0,
        "overlap_ratio": 0.0,
        "overlap_tokens": [],
    }
    for answer_text in selected_answer_texts(record):
        answer_norm = normalized(answer_text)
        tokens = set(answer_tokens(answer_text))
        overlap = sorted(span_tokens & tokens)
        ratio = len(overlap) / len(tokens) if tokens else 0.0
        exact = bool(answer_norm and answer_norm in value_norm)
        if (
            exact
            or len(overlap) > int(best["overlap_count"])
            or ratio > float(best["overlap_ratio"])
        ):
            best = {
                "has_option_label": has_label,
                "has_selected_answer_text": exact,
                "selected_answer_is_short_symbol": is_short_answer_symbol(
                    answer_text, args
                ),
                "overlap_count": len(overlap),
                "overlap_ratio": float(ratio),
                "overlap_tokens": overlap,
            }
    return best


def candidate_rejection_reason(
    text: str, record: Dict[str, Any], args: argparse.Namespace
) -> str:
    features = answer_overlap_features(text, record, args)
    mode = args.answer_leakage_mode
    if mode != "off" and features["has_option_label"]:
        return "option_label_overlap"
    if mode == "strict":
        if features["has_selected_answer_text"]:
            return "selected_answer_text_overlap"
        if features["overlap_count"] > 0:
            return "selected_answer_token_overlap"
    elif mode == "relaxed" and not features["selected_answer_is_short_symbol"]:
        if features["has_selected_answer_text"]:
            return "selected_answer_text_overlap"
        if (
            features["overlap_count"] >= max(args.answer_overlap_reject_min_tokens, 1)
            and features["overlap_ratio"] >= max(args.answer_overlap_reject_ratio, 0.0)
        ):
            return "selected_answer_token_overlap"
    value_norm = normalized(text)
    for phrase in FORBIDDEN_PHRASES:
        if phrase in value_norm:
            return "generic_or_meta_phrase"
    return ""


def add_detection(
    detections: List[Dict[str, Any]], start: int, end: int, reason: str, text: str
) -> None:
    if end > start:
        detections.append(
            {"start_char": int(start), "end_char": int(end), "reason": reason, "text": text[start:end]}
        )


def flexible_text_pattern(text: str) -> Optional[re.Pattern[str]]:
    pieces = [re.escape(piece) for piece in re.split(r"\s+", safe_text(text)) if piece]
    if not pieces:
        return None
    return re.compile(r"\s+".join(pieces), re.I)


def detect_leak_spans(
    rationale: str, record: Dict[str, Any], args: argparse.Namespace
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    detections: List[Dict[str, Any]] = []
    if args.answer_leakage_mode == "off":
        return detections, []

    # Exact answer surface forms.  Relaxed mode keeps short biomedical symbols,
    # matching the production Qwen evidence filter.
    for answer_text in selected_answer_texts(record):
        if args.answer_leakage_mode == "relaxed" and is_short_answer_symbol(answer_text, args):
            continue
        pattern = flexible_text_pattern(answer_text)
        if pattern is not None:
            for match in pattern.finditer(rationale):
                add_detection(detections, match.start(), match.end(), "selected_answer_text", rationale)

    label = re.escape(target_label(record))
    label_patterns = (
        rf"(?i)\b(?:option|choice|answer)\s*[:\-]?\s*{label}\b",
        rf"(?i)(?<!\w)\({label}\)",
        rf"(?i)(^|\s){label}\s*[\.)](?=\s|$)",
    )
    for pattern_text in label_patterns:
        for match in re.finditer(pattern_text, rationale):
            start, end = match.span()
            while start < end and rationale[start].isspace():
                start += 1
            add_detection(detections, start, end, "selected_option_label", rationale)

    for phrase in FORBIDDEN_PHRASES:
        for match in re.finditer(re.escape(phrase), rationale, re.I):
            add_detection(detections, match.start(), match.end(), "generic_or_meta_phrase", rationale)

    # Mirror the relaxed partial-token rule at sentence scope, but remove only
    # answer-derived surface tokens rather than discarding the whole sentence.
    if args.answer_leakage_mode in {"strict", "relaxed"}:
        token_matches = list(re.finditer(r"[A-Za-z0-9]+(?:[+\-][A-Za-z0-9]+)?", rationale))
        sentence_matches = list(re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", rationale))
        for answer_text in selected_answer_texts(record):
            if args.answer_leakage_mode == "relaxed" and is_short_answer_symbol(answer_text, args):
                continue
            answer_set = set(answer_tokens(answer_text))
            if not answer_set:
                continue
            for sentence in sentence_matches:
                local = [
                    match
                    for match in token_matches
                    if sentence.start() <= match.start() < sentence.end()
                    and match.group(0).casefold() in answer_set
                ]
                overlap = {match.group(0).casefold() for match in local}
                ratio = len(overlap) / len(answer_set)
                reject = bool(overlap) if args.answer_leakage_mode == "strict" else (
                    len(overlap) >= max(args.answer_overlap_reject_min_tokens, 1)
                    and ratio >= max(args.answer_overlap_reject_ratio, 0.0)
                )
                if reject:
                    for match in local:
                        add_detection(
                            detections,
                            match.start(),
                            match.end(),
                            "selected_answer_token_overlap",
                            rationale,
                        )

    intervals = merge_intervals(
        [(int(item["start_char"]), int(item["end_char"])) for item in detections]
    )
    return detections, intervals


def merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[List[int]] = []
    for start, end in sorted((max(int(s), 0), max(int(e), 0)) for s, e in intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def complement_intervals(text: str, excluded: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    cursor = 0
    for start, end in merge_intervals(excluded):
        start = min(max(start, 0), len(text))
        end = min(max(end, start), len(text))
        if start > cursor and re.search(r"\w", text[cursor:start]):
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < len(text) and re.search(r"\w", text[cursor:]):
        result.append((cursor, len(text)))
    return result


def remove_intervals(text: str, excluded: Sequence[Tuple[int, int]]) -> str:
    pieces: List[str] = []
    cursor = 0
    for start, end in merge_intervals(excluded):
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    cleaned = "".join(pieces)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\n\s+", "\n", cleaned)
    return cleaned.strip()


def target_offset(record: Dict[str, Any], rationale: str) -> int:
    target = str(record.get("faith_token_target_text") or "")
    position = target.find(rationale)
    if position < 0:
        raise ValueError(
            f"Rationale is not in faith_token_target_text for {record.get('example_id')}"
        )
    return position


def format_example_block(record: Dict[str, Any]) -> str:
    blocks = [f"Question:\n{safe_text(record.get('question'))}"]
    for index, context in enumerate(record.get("contexts", []) or [], start=1):
        context = safe_text(context)
        if context:
            blocks.append(f"Context {index}:\n{context}")
    blocks.append("Options:")
    for label, text in (record.get("options") or {}).items():
        blocks.append(f"{label}. {safe_text(text)}")
    return "\n\n".join(blocks)


def build_recovery_prompt(record: Dict[str, Any], rationale_text: str) -> str:
    rationale_text = safe_text(rationale_text) or "Relevant rationale omitted."
    return (
        "Answer the medical multiple-choice question.\n"
        "Use the provided rationale to infer the selected answer.\n"
        "Return the option label followed by the option text.\n\n"
        f"{format_example_block(record)}\n\n"
        f"Rationale:\n{rationale_text}\n\n"
        "Final:"
    )


def load_tokenizer(model_tag: str, mode: str):
    if mode == "off":
        return None
    model_path = MODEL_PATHS.get(model_tag)
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            str(model_path), use_fast=False, local_files_only=True
        )
    except Exception as exc:  # dependency availability is environment-specific
        if mode == "required":
            raise RuntimeError(f"Could not load tokenizer for {model_tag}: {exc}") from exc
        print(f"[WARN] tokenizer unavailable for {model_tag}; using word budgets: {exc}")
        return None


def token_count(tokenizer, text: str) -> int:
    if tokenizer is None:
        return word_count(text)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def target_span_token_count(
    tokenizer, target_text: str, start_char: int, end_char: int
) -> int:
    if tokenizer is None:
        return word_count(target_text[start_char:end_char])
    start_ids = tokenizer(target_text[:start_char], add_special_tokens=False)["input_ids"]
    end_ids = tokenizer(target_text[:end_char], add_special_tokens=False)["input_ids"]
    return max(len(end_ids) - len(start_ids), 0)


def unit_key(unit: Dict[str, Any]) -> Tuple[Any, ...]:
    if unit.get("index") is not None:
        return ("index", int(unit["index"]))
    return (
        "span",
        int(unit.get("start_char", -1)),
        int(unit.get("end_char", -1)),
        safe_text(unit.get("text")),
    )


def make_random_templates(
    record: Dict[str, Any], rationale: str, tokenizer
) -> List[Dict[str, Any]]:
    f_units = [copy.deepcopy(unit) for unit in record.get("faith_f_evidence_units") or []]
    q_units = [copy.deepcopy(unit) for unit in record.get("faith_q_evidence_units") or []]
    q_by_key = {unit_key(unit): unit for unit in q_units}
    target = str(record["faith_token_target_text"])
    offset = target_offset(record, rationale)
    templates: List[Dict[str, Any]] = []
    for position, unit in enumerate(f_units):
        key = unit_key(unit)
        q_unit = q_by_key.get(key)
        if q_unit is None:
            for candidate_q in q_units:
                if (
                    int(candidate_q.get("start_char", -1)) == int(unit.get("start_char", -2))
                    and int(candidate_q.get("end_char", -1)) == int(unit.get("end_char", -2))
                ):
                    q_unit = candidate_q
                    break
        template = {
            "position": position,
            "source_index": int(unit.get("index", position)),
            "source_text": safe_text(unit.get("text")),
            "word_count": int(unit.get("word_count") or word_count(unit.get("text", ""))),
            "text_token_count": token_count(tokenizer, safe_text(unit.get("text"))),
            "is_q": q_unit is not None,
            "loss_weight": float((q_unit or {}).get("loss_weight", 1.0)),
            "source_q_text": safe_text((q_unit or {}).get("text")),
        }
        if q_unit is not None:
            start = offset + int(q_unit["start_char"])
            end = offset + int(q_unit["end_char"])
            template["target_token_count"] = target_span_token_count(
                tokenizer, target, start, end
            )
        else:
            template["target_token_count"] = None
        templates.append(template)
    if len(q_by_key) and sum(bool(item["is_q"]) for item in templates) != len(q_units):
        raise ValueError(f"Could not map every Q unit to F for {record.get('example_id')}")
    return templates


def build_candidate_windows(
    record: Dict[str, Any],
    rationale: str,
    target: str,
    offset: int,
    template: Dict[str, Any],
    tokenizer,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    words = list(re.finditer(r"\S+", rationale))
    source_width = int(template["word_count"])
    candidates: List[Dict[str, Any]] = []
    if source_width <= 0 or source_width > len(words):
        return candidates
    tolerance = max(int(args.random_word_tolerance), 0)
    widths = range(
        max(source_width - tolerance, 1),
        min(source_width + tolerance, len(words)) + 1,
    )
    for width in widths:
        for start_index in range(0, len(words) - width + 1):
            start = words[start_index].start()
            end = words[start_index + width - 1].end()
            text = rationale[start:end]
            rejection = candidate_rejection_reason(text, record, args)
            if rejection:
                continue
            text_tokens = token_count(tokenizer, text)
            target_tokens = target_span_token_count(
                tokenizer, target, offset + start, offset + end
            )
            text_delta = abs(text_tokens - int(template["text_token_count"]))
            expected_target = template.get("target_token_count")
            target_delta = (
                abs(target_tokens - int(expected_target)) if expected_target is not None else 0
            )
            word_delta = abs(width - source_width)
            cost = (
                4 * target_delta + 2 * text_delta + word_delta
                if expected_target is not None
                else 3 * text_delta + word_delta
            )
            candidates.append(
                {
                    "start_char": start,
                    "end_char": end,
                    "text": text,
                    "word_count": width,
                    "word_count_delta": word_delta,
                    "text_token_count": text_tokens,
                    "target_token_count": target_tokens,
                    "text_token_delta": text_delta,
                    "target_token_delta": target_delta,
                    "cost": cost,
                }
            )
    return candidates


def intervals_overlap(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return int(left["start_char"]) < int(right["end_char"]) and int(right["start_char"]) < int(left["end_char"])


def select_random_windows(
    record: Dict[str, Any],
    rationale: str,
    templates: Sequence[Dict[str, Any]],
    tokenizer,
    model_tag: str,
    selection_seed: int,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    seed_material = f"{selection_seed}\0{model_tag}\0{record.get('example_id')}"
    per_example_seed = int.from_bytes(
        hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big"
    )
    rng = random.Random(per_example_seed)
    target = str(record["faith_token_target_text"])
    offset = target_offset(record, rationale)
    pools: List[List[Dict[str, Any]]] = []
    for template in templates:
        pool = build_candidate_windows(
            record, rationale, target, offset, template, tokenizer, args
        )
        if not pool:
            raise ValueError(
                f"No leakage-safe random window for {record.get('example_id')} "
                f"F position {template['position']}"
            )
        rng.shuffle(pool)
        pool.sort(key=lambda candidate: int(candidate["cost"]))
        pools.append(pool[: max(args.max_candidates_per_unit, 1)])

    order = sorted(range(len(templates)), key=lambda index: len(pools[index]))
    assignment: Dict[int, Dict[str, Any]] = {}
    node_count = 0

    def backtrack(depth: int) -> bool:
        nonlocal node_count
        if depth >= len(order):
            return True
        template_index = order[depth]
        for candidate in pools[template_index]:
            node_count += 1
            if node_count > max(args.max_backtrack_nodes, 1):
                return False
            if any(intervals_overlap(candidate, chosen) for chosen in assignment.values()):
                continue
            assignment[template_index] = candidate
            if backtrack(depth + 1):
                return True
            assignment.pop(template_index, None)
        return False

    nonoverlap_success = backtrack(0)
    nonoverlap_assignment = dict(assignment) if nonoverlap_success else None

    # Also construct the best budget-matched assignment without a non-overlap
    # constraint.  Some source F/Q spans overlap, and exact selected-token
    # matching is the primary control.  Exact duplicate windows stay forbidden.
    overlap_assignment: Dict[int, Dict[str, Any]] = {}
    used_bounds = set()
    for template_index in order:
        chosen = next(
            (
                candidate
                for candidate in pools[template_index]
                if (candidate["start_char"], candidate["end_char"]) not in used_bounds
            ),
            pools[template_index][0],
        )
        overlap_assignment[template_index] = chosen
        used_bounds.add((chosen["start_char"], chosen["end_char"]))

    def assignment_quality(current: Dict[int, Dict[str, Any]]) -> Tuple[int, int, int, int]:
        ordered = [current[index] for index in range(len(templates))]
        q_delta = sum(
            int(item["target_token_delta"])
            for item, template in zip(ordered, templates)
            if template["is_q"]
        )
        text_delta = sum(int(item["text_token_delta"]) for item in ordered)
        word_delta = sum(int(item["word_count_delta"]) for item in ordered)
        overlap_pairs = sum(
            intervals_overlap(ordered[left], ordered[right])
            for left in range(len(ordered))
            for right in range(left + 1, len(ordered))
        )
        return q_delta, text_delta, word_delta, overlap_pairs

    if nonoverlap_assignment is None or assignment_quality(overlap_assignment) < assignment_quality(nonoverlap_assignment):
        assignment = overlap_assignment
        selected_is_nonoverlap = assignment_quality(overlap_assignment)[3] == 0
    else:
        assignment = nonoverlap_assignment
        selected_is_nonoverlap = True

    selected = [copy.deepcopy(assignment[index]) for index in range(len(templates))]
    source_bounds = {
        (int(unit.get("start_char", -1)), int(unit.get("end_char", -1)))
        for unit in (record.get("faith_f_evidence_units") or [])
    }
    diagnostics = {
        "selection_seed": int(selection_seed),
        "per_example_seed_sha256": hashlib.sha256(seed_material.encode("utf-8")).hexdigest(),
        "nonoverlap_solution_found": bool(nonoverlap_success),
        "selected_assignment_is_nonoverlap": bool(selected_is_nonoverlap),
        "backtrack_nodes": int(node_count),
        "exact_source_span_matches": sum(
            (int(item["start_char"]), int(item["end_char"])) in source_bounds
            for item in selected
        ),
        "total_text_token_delta": sum(int(item["text_token_delta"]) for item in selected),
        "total_word_count_delta": sum(int(item["word_count_delta"]) for item in selected),
        "total_q_target_token_delta": sum(
            int(item["target_token_delta"])
            for item, template in zip(selected, templates)
            if template["is_q"]
        ),
    }
    return selected, diagnostics


def reset_unscored_recovery_fields(record: Dict[str, Any]) -> None:
    record["faith_recovery_selected_success"] = None
    record["candidate_evidence_recovery_correct"] = None
    record["candidate_evidence_recovery_valid"] = None
    record["faith_recovery_selected_pred_answer"] = None
    record["faith_recovery_selected_option_scores"] = {}
    record["faith_recovery_selected_option_probs"] = {}
    record["faith_recovery_selected_margin"] = None
    record["faith_baseline_recovery_not_rescored"] = True


def common_baseline_fields(
    record: Dict[str, Any],
    variant: str,
    model_tag: str,
    source_path: Path,
    detections: Sequence[Dict[str, Any]],
    leak_intervals: Sequence[Tuple[int, int]],
) -> None:
    record["faith_baseline_variant"] = variant
    record["faith_baseline_model_tag"] = model_tag
    record["faith_baseline_source_cache"] = str(source_path)
    record["faith_baseline_source_example_id"] = record.get("example_id")
    record["faith_baseline_answer_leak_detections"] = list(detections)
    record["faith_baseline_answer_leak_intervals"] = [
        {"start_char": start, "end_char": end}
        for start, end in leak_intervals
    ]
    record["faith_cache_usable"] = True
    record["selected_evidence_available"] = True
    record["faith_cache_failure_reason"] = ""
    reset_unscored_recovery_fields(record)


def transform_full_noanswer(
    source: Dict[str, Any],
    model_tag: str,
    source_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    record = copy.deepcopy(source)
    rationale = safe_text(record.get("pred_explanation_stripped"))
    detections, leak_intervals = detect_leak_spans(rationale, record, args)
    eligible = complement_intervals(rationale, leak_intervals)
    sanitized = remove_intervals(rationale, leak_intervals)
    if not eligible or not sanitized:
        raise ValueError(f"No full-noanswer content for {record.get('example_id')}")
    offset = target_offset(record, rationale)
    units = [
        {
            "start_char": start,
            "end_char": end,
            "text": rationale[start:end],
            "word_count": word_count(rationale[start:end]),
            "index": index,
            "loss_weight": 1.0,
            "baseline_unit_type": "full_noanswer_eligible_segment",
        }
        for index, (start, end) in enumerate(eligible)
    ]
    record["faith_token_selected_char_spans"] = [
        {
            "start_char": offset + unit["start_char"],
            "end_char": offset + unit["end_char"],
            "index": unit["index"],
            "loss_weight": 1.0,
        }
        for unit in units
    ]
    recovery_prompt = build_recovery_prompt(record, sanitized)
    record["faith_recovery_selected_user_prompt"] = recovery_prompt
    record["faith_recovery_unit_user_prompts"] = []
    record["faith_recovery_selected_rationale"] = sanitized
    record["faith_recovery_selected_k"] = 1
    record["faith_recovery_selected_word_count"] = word_count(sanitized)
    record["faith_f_evidence_units"] = [
        {
            "text": sanitized,
            "word_count": word_count(sanitized),
            "index": 0,
            "baseline_pseudo_unit": True,
        }
    ]
    record["faith_f_selected_span_count"] = 1
    record["faith_q_evidence_units"] = units
    record["faith_q_selected_span_count"] = len(units)
    record["faith_selected_spans"] = units
    record["faith_selected_span_count"] = len(units)
    record["faith_span_granularity"] = "full_noanswer"
    record["faith_span_score_mode"] = "all_nonleak_rationale_positions_uniform"
    common_baseline_fields(
        record, "full_noanswer", model_tag, source_path, detections, leak_intervals
    )
    return record


def transform_random_noanswer(
    source: Dict[str, Any],
    model_tag: str,
    source_path: Path,
    selection_seed: int,
    tokenizer,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    record = copy.deepcopy(source)
    rationale = safe_text(record.get("pred_explanation_stripped"))
    detections, leak_intervals = detect_leak_spans(rationale, record, args)
    templates = make_random_templates(record, rationale, tokenizer)
    selected, diagnostics = select_random_windows(
        record,
        rationale,
        templates,
        tokenizer,
        model_tag,
        selection_seed,
        args,
    )
    offset = target_offset(record, rationale)
    f_units: List[Dict[str, Any]] = []
    q_units: List[Dict[str, Any]] = []
    for candidate, template in zip(selected, templates):
        unit = {
            "start_char": int(candidate["start_char"]),
            "end_char": int(candidate["end_char"]),
            "text": candidate["text"],
            "word_count": int(candidate["word_count"]),
            "index": int(template["source_index"]),
            "baseline_random_f_position": int(template["position"]),
            "baseline_source_text": template["source_text"],
            "baseline_source_text_token_count": int(template["text_token_count"]),
            "baseline_random_text_token_count": int(candidate["text_token_count"]),
            "baseline_text_token_delta": int(candidate["text_token_delta"]),
            "baseline_source_word_count": int(template["word_count"]),
            "baseline_random_word_count": int(candidate["word_count"]),
            "baseline_word_count_delta": int(candidate["word_count_delta"]),
            "baseline_is_q": bool(template["is_q"]),
        }
        f_units.append(unit)
        if template["is_q"]:
            q_unit = copy.deepcopy(unit)
            q_unit["loss_weight"] = float(template["loss_weight"])
            q_unit["baseline_source_q_text"] = template["source_q_text"]
            q_unit["baseline_source_target_token_count"] = int(
                template["target_token_count"]
            )
            q_unit["baseline_random_target_token_count"] = int(
                candidate["target_token_count"]
            )
            q_unit["baseline_target_token_delta"] = int(candidate["target_token_delta"])
            q_units.append(q_unit)
    if len(f_units) != len(templates) or len(q_units) != sum(
        bool(template["is_q"]) for template in templates
    ):
        raise AssertionError(f"F/Q count mismatch for {record.get('example_id')}")

    selected_rationale = " ".join(unit["text"] for unit in f_units).strip()
    record["faith_token_selected_char_spans"] = [
        {
            "start_char": offset + int(unit["start_char"]),
            "end_char": offset + int(unit["end_char"]),
            "index": int(unit["index"]),
            "loss_weight": float(unit["loss_weight"]),
        }
        for unit in q_units
    ]
    record["faith_recovery_selected_user_prompt"] = build_recovery_prompt(
        record, selected_rationale
    )
    record["faith_recovery_unit_user_prompts"] = [
        build_recovery_prompt(record, unit["text"]) for unit in f_units
    ]
    record["faith_recovery_selected_rationale"] = selected_rationale
    record["faith_recovery_selected_k"] = len(f_units)
    record["faith_recovery_selected_word_count"] = word_count(selected_rationale)
    record["faith_f_evidence_units"] = f_units
    record["faith_f_selected_span_count"] = len(f_units)
    record["faith_q_evidence_units"] = q_units
    record["faith_q_selected_span_count"] = len(q_units)
    record["faith_selected_spans"] = q_units
    record["faith_selected_span_count"] = len(q_units)
    record["faith_span_granularity"] = "random_noanswer_matched"
    record["faith_span_score_mode"] = "random_matched_to_source_f_q"
    record["faith_baseline_random_diagnostics"] = diagnostics
    record["faith_baseline_selection_seed"] = int(selection_seed)
    common_baseline_fields(
        record,
        "random_noanswer_matched",
        model_tag,
        source_path,
        detections,
        leak_intervals,
    )
    return record


def build_summary(
    records: Sequence[Dict[str, Any]],
    *,
    model_tag: str,
    variant: str,
    source_path: Path,
    output_path: Path,
    source_hash: str,
    tokenizer_used: bool,
    selection_seed: Optional[int],
) -> Dict[str, Any]:
    ids = [safe_text(record.get("example_id")) for record in records]
    random_diagnostics = [
        record.get("faith_baseline_random_diagnostics") or {} for record in records
    ]
    return {
        "model_tag": model_tag,
        "variant": variant,
        "selection_seed": selection_seed,
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "output_path": str(output_path),
        "ordered_example_ids_sha256": sha256_values(ids),
        "num_records": len(records),
        "first_example_ids": ids[:5],
        "tokenizer_used": bool(tokenizer_used),
        "records_with_detected_leakage": sum(
            bool(record.get("faith_baseline_answer_leak_intervals")) for record in records
        ),
        "total_leakage_characters": sum(
            sum(
                int(span["end_char"]) - int(span["start_char"])
                for span in record.get("faith_baseline_answer_leak_intervals", [])
            )
            for record in records
        ),
        "total_f_units": sum(len(record.get("faith_f_evidence_units") or []) for record in records),
        "total_q_units": sum(len(record.get("faith_q_evidence_units") or []) for record in records),
        "random_nonoverlap_fallback_records": sum(
            bool(diagnostic)
            and not diagnostic.get("selected_assignment_is_nonoverlap", False)
            for diagnostic in random_diagnostics
        ),
        "random_exact_source_span_matches": sum(
            int(diagnostic.get("exact_source_span_matches", 0))
            for diagnostic in random_diagnostics
        ),
        "random_total_text_token_delta": sum(
            int(diagnostic.get("total_text_token_delta", 0))
            for diagnostic in random_diagnostics
        ),
        "random_total_word_count_delta": sum(
            int(diagnostic.get("total_word_count_delta", 0))
            for diagnostic in random_diagnostics
        ),
        "random_total_q_target_token_delta": sum(
            int(diagnostic.get("total_q_target_token_delta", 0))
            for diagnostic in random_diagnostics
        ),
    }


def main() -> None:
    args = parse_args()
    source_root = args.repo_dir / "cache" / "med_faithfulness"
    for model_tag in args.models:
        if model_tag not in MODEL_PATHS:
            raise ValueError(f"Unsupported model tag: {model_tag}")
        source_path = (
            source_root
            / model_tag
            / f"medmix_train_teacher_predictions_qwen_evidence_pos_{args.source_tag}.jsonl"
        )
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_records = [
            record for record in iter_jsonl(source_path) if source_record_is_usable(record)
        ]
        if not source_records:
            raise ValueError(f"No source-loader-usable records in {source_path}")
        source_hash = sha256_file(source_path)
        tokenizer = (
            load_tokenizer(model_tag, args.tokenizer_mode)
            if "random_noanswer_matched" in args.variants
            else None
        )
        output_dir = args.output_root / model_tag

        if "full_noanswer" in args.variants:
            records = [
                transform_full_noanswer(record, model_tag, source_path, args)
                for record in source_records
            ]
            output_path = output_dir / (
                f"medmix_train_teacher_predictions_full_noanswer_{args.source_tag}.jsonl"
            )
            write_jsonl(output_path, records, args.overwrite)
            summary = build_summary(
                records,
                model_tag=model_tag,
                variant="full_noanswer",
                source_path=source_path,
                output_path=output_path,
                source_hash=source_hash,
                tokenizer_used=tokenizer is not None,
                selection_seed=None,
            )
            summary_path = output_path.with_name(output_path.stem + "_summary.json")
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[DONE] {model_tag} full_noanswer: {len(records)} -> {output_path}")

        if "random_noanswer_matched" in args.variants:
            for selection_seed in args.selection_seeds:
                records = [
                    transform_random_noanswer(
                        record,
                        model_tag,
                        source_path,
                        selection_seed,
                        tokenizer,
                        args,
                    )
                    for record in source_records
                ]
                output_path = output_dir / (
                    "medmix_train_teacher_predictions_random_noanswer_matched_"
                    f"r{selection_seed}_{args.source_tag}.jsonl"
                )
                write_jsonl(output_path, records, args.overwrite)
                summary = build_summary(
                    records,
                    model_tag=model_tag,
                    variant="random_noanswer_matched",
                    source_path=source_path,
                    output_path=output_path,
                    source_hash=source_hash,
                    tokenizer_used=tokenizer is not None,
                    selection_seed=selection_seed,
                )
                summary_path = output_path.with_name(output_path.stem + "_summary.json")
                summary_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"[DONE] {model_tag} random r{selection_seed}: "
                    f"{len(records)} -> {output_path}"
                )


if __name__ == "__main__":
    main()
