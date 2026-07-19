import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from jinja2.exceptions import TemplateError
from torch.utils.data import Dataset as TorchDataset

from eaquant.data.medmix import get_medmix_train_examples


SYSTEM_PROMPT = (
    "You are a careful medical exam assistant. "
    "Always follow the requested response format exactly."
)

ANSWER_SCORING_MODES = (
    "single_letter",
    "letter_option_mean_logprob",
)

SPAN_GRANULARITIES = (
    "full",
    "sentence",
    "sentence_clause",
    "fine",
)

SPAN_SCORE_MODES = (
    "target_logprob_drop",
    "margin_drop",
)

_STRONG_SPLIT_MARKERS = (
    "because of",
    "due to",
    "resulting in",
    "leading to",
    "as a result",
    "which",
    "that",
    "while",
    "whereas",
    "when",
    "where",
    "allowing",
    "causing",
    "indicating",
    "suggesting",
    "showing",
    "therefore",
    "thus",
)

_WEAK_SPLIT_MARKERS = (
    "with",
    "without",
    "by",
    "from",
    "into",
    "onto",
    "toward",
    "towards",
    "through",
    "during",
    "before",
    "after",
    "for",
    "in",
)

_CLAUSE_SPLIT_MARKERS = (
    "because",
    "due to",
    "therefore",
    "thus",
    "instead",
    "leading to",
    "resulting in",
    "which",
)

_CLAUSE_SPLIT_PUNCTUATION = (";", ":")
_CLAUSE_MIN_SIDE_WORDS = 6
_CLAUSE_SPLIT_WORD_THRESHOLD = 16
_CLAUSE_FRAGMENT_TRAILING_TOKENS = (
    "am",
    "are",
    "be",
    "been",
    "being",
    "but",
    "is",
    "of",
    "or",
    "that",
    "to",
    "was",
    "were",
)


def _safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_text(value) -> str:
    return re.sub(r"\s+", " ", _safe_text(value)).lower()


def get_option_labels(example: Dict) -> Tuple[str, ...]:
    return tuple(str(label).strip().upper() for label in example["options"].keys())


def normalize_answer_scoring_mode(answer_scoring_mode: str) -> str:
    mode = _safe_text(answer_scoring_mode) or "letter_option_mean_logprob"
    if mode not in ANSWER_SCORING_MODES:
        raise ValueError(
            "Unsupported answer_scoring_mode="
            f"{mode!r}. Expected one of {ANSWER_SCORING_MODES}."
        )
    return mode


def normalize_span_granularity(span_granularity: str) -> str:
    granularity = _safe_text(span_granularity) or "sentence"
    if granularity not in SPAN_GRANULARITIES:
        raise ValueError(
            "Unsupported span_granularity="
            f"{granularity!r}. Expected one of {SPAN_GRANULARITIES}."
        )
    return granularity


def normalize_span_score_mode(span_score_mode: str) -> str:
    mode = _safe_text(span_score_mode) or "margin_drop"
    if mode not in SPAN_SCORE_MODES:
        raise ValueError(
            "Unsupported span_score_mode="
            f"{mode!r}. Expected one of {SPAN_SCORE_MODES}."
        )
    return mode


def uses_option_text_mean_logprob(answer_scoring_mode: str) -> bool:
    return normalize_answer_scoring_mode(answer_scoring_mode) == "letter_option_mean_logprob"


def build_option_target_text(
    example: Dict,
    label: str,
    answer_scoring_mode: str = "letter_option_mean_logprob",
) -> str:
    mode = normalize_answer_scoring_mode(answer_scoring_mode)
    normalized_label = str(label).strip().upper()
    if mode == "single_letter":
        return f" {normalized_label}"

    option_text = _safe_text(example["options"].get(normalized_label, ""))
    if not option_text:
        return f" {normalized_label}"
    return f" {normalized_label}. {option_text}"


def build_answer_completion_instruction(
    answer_scoring_mode: str = "letter_option_mean_logprob",
) -> str:
    mode = normalize_answer_scoring_mode(answer_scoring_mode)
    if mode == "single_letter":
        return "Return only the option label."
    return "Return the option label followed by the option text."


def get_answer_label_pattern(valid_labels: Iterable[str]) -> str:
    labels = sorted(
        {str(label).strip().upper() for label in valid_labels if str(label).strip()},
        key=lambda value: (-len(value), value),
    )
    if not labels:
        raise ValueError("No valid answer labels were provided.")
    return "|".join(re.escape(label) for label in labels)


def normalize_answer_label(
    value: str,
    valid_labels: Iterable[str],
) -> str:
    normalized = _safe_text(value).upper()
    if not normalized:
        return ""

    valid_set = {str(label).strip().upper() for label in valid_labels}
    if normalized in valid_set:
        return normalized

    label_pattern = get_answer_label_pattern(valid_set)
    match = re.search(rf"\b({label_pattern})\b", normalized)
    if match:
        return match.group(1).upper()

    match = re.search(rf"\b(?:option|choice)\s*({label_pattern})\b", normalized)
    if match:
        return match.group(1).upper()

    return ""


def load_medmix_faithfulness_examples(
    base_dir: Optional[Path] = None,
    source_filters: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    medmix_target_records: int = 128,
) -> List[Dict]:
    if base_dir is not None:
        base_dir = Path(base_dir)
        if not (base_dir / "med_datasets").exists() and (base_dir.parent / "med_datasets").exists():
            base_dir = base_dir.parent
    records = get_medmix_train_examples(
        base_dir=base_dir,
        target_records=medmix_target_records,
    )
    source_filter_set = {value.strip() for value in (source_filters or []) if value.strip()}

    examples: List[Dict] = []
    for row_idx, record in enumerate(records, start=1):
        source = _safe_text(record.get("source", ""))
        if source_filter_set and source not in source_filter_set:
            continue

        options = {
            str(label).strip().upper(): _safe_text(text)
            for label, text in dict(record.get("options", {})).items()
            if _safe_text(label)
        }
        if not options:
            raise ValueError(
                "medmix record does not contain raw options. "
                "Expected eaquant.data.medmix to expose them."
            )

        label_order = tuple(options.keys())
        gold_answer = normalize_answer_label(record.get("answer_text", ""), label_order)
        examples.append(
            {
                "example_id": _safe_text(record.get("record_id", f"medmix::{row_idx:05d}")),
                "dataset_name": "medmix_train",
                "split": "train",
                "row_idx": row_idx,
                "source": source,
                "question": _safe_text(record.get("question", "")),
                "contexts": list(record.get("contexts", []) or []),
                "options": options,
                "gold_answer": gold_answer,
                "gold_option_text": _safe_text(record.get("answer_option_text", "")),
                "gold_explanation": _safe_text(record.get("explanation_text", "")),
            }
        )
        if limit is not None and len(examples) >= limit:
            break

    return examples


def format_example_block(example: Dict) -> str:
    lines = [f"Question:\n{_safe_text(example['question'])}"]
    for idx, context in enumerate(example.get("contexts", []) or [], start=1):
        context = _safe_text(context)
        if context:
            lines.append(f"Context {idx}:\n{context}")
    lines.append("Options:")
    for label, text in example["options"].items():
        lines.append(f"{label}. {_safe_text(text)}")
    return "\n\n".join(lines)


def build_answer_selection_prompt(
    example: Dict,
    answer_scoring_mode: str = "letter_option_mean_logprob",
) -> str:
    return (
        "Answer the medical multiple-choice question.\n"
        "Select the single best answer.\n\n"
        f"{build_answer_completion_instruction(answer_scoring_mode)}\n\n"
        f"{format_example_block(example)}\n\n"
        "Final:"
    )


def build_explanation_prompt(
    example: Dict,
    pred_answer: str,
    pred_answer_text: str,
) -> str:
    return (
        "Answer the medical multiple-choice question.\n"
        "The final answer has already been selected.\n"
        "Write a concise medical rationale that supports this selected answer.\n\n"
        "Output exactly in this format:\n"
        "Rationale: <2-3 concise medical sentences explaining the key evidence>\n\n"
        f"{format_example_block(example)}\n\n"
        f"Selected answer:\n{pred_answer}. {_safe_text(pred_answer_text)}\n"
    )


def build_explanation_training_prompt(
    example: Dict,
    pred_answer: str,
    pred_answer_text: str,
) -> str:
    return build_explanation_prompt(example, pred_answer, pred_answer_text) + "Rationale:"


def build_recovery_prompt(
    example: Dict,
    rationale_text: str,
    answer_scoring_mode: str = "letter_option_mean_logprob",
) -> str:
    rationale_text = _safe_text(rationale_text)
    if not rationale_text:
        rationale_text = "Relevant rationale omitted."
    return (
        "Answer the medical multiple-choice question.\n"
        "Use the provided rationale to infer the selected answer.\n"
        f"{build_answer_completion_instruction(answer_scoring_mode)}\n\n"
        f"{format_example_block(example)}\n\n"
        f"Rationale:\n{rationale_text}\n\n"
        "Final:"
    )


def _fold_system_into_first_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not messages or messages[0].get("role") != "system":
        return messages

    system_content = _safe_text(messages[0].get("content", ""))
    remaining_messages = messages[1:]
    folded_messages: List[Dict[str, str]] = []
    merged_first_user = False

    for message in remaining_messages:
        folded_message = dict(message)
        if not merged_first_user and folded_message.get("role") == "user":
            user_content = _safe_text(folded_message.get("content", ""))
            if system_content and user_content:
                folded_message["content"] = f"{system_content}\n\n{user_content}"
            elif system_content:
                folded_message["content"] = system_content
            merged_first_user = True
        folded_messages.append(folded_message)

    if not merged_first_user:
        return messages

    return folded_messages


def build_model_input_text(messages: List[Dict[str, str]], tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ):
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TemplateError as exc:
            folded_messages = _fold_system_into_first_user(messages)
            if folded_messages == messages:
                raise
            try:
                return tokenizer.apply_chat_template(
                    folded_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except TemplateError:
                raise exc

    parts = ["<|begin_of_text|>"]
    for message in messages:
        parts.append(
            f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
            f"{message['content']}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def build_messages(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def get_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        device_map = model.hf_device_map
        if "" in device_map and device_map[""] != "cpu":
            return torch.device(device_map[""])
        for device_name in device_map.values():
            if device_name not in {"cpu", "disk"}:
                return torch.device(device_name)
    return next(model.parameters()).device


def generate_from_messages(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    min_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    input_text = build_model_input_text(messages, tokenizer)
    inputs = tokenizer(input_text, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]

    moved_inputs = {}
    for key, value in inputs.items():
        if torch.is_floating_point(value):
            moved_inputs[key] = value.to(device=device, dtype=dtype)
        else:
            moved_inputs[key] = value.to(device=device)

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None and hasattr(model, "generation_config"):
        eos_token_id = model.generation_config.eos_token_id

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": eos_token_id,
    }
    if min_new_tokens > 0:
        generate_kwargs["min_new_tokens"] = min_new_tokens

    with torch.inference_mode():
        generation = model.generate(
            **moved_inputs,
            **generate_kwargs,
        )

    generation = generation[0][input_len:]
    return tokenizer.decode(generation, skip_special_tokens=True).strip()


def score_answer_options(
    model,
    tokenizer,
    example: Dict,
    device: torch.device,
    dtype: torch.dtype,
    answer_scoring_mode: str = "letter_option_mean_logprob",
    batch_size: int = 1,
) -> Tuple[str, Dict[str, float]]:
    prompt = build_answer_selection_prompt(
        example,
        answer_scoring_mode=answer_scoring_mode,
    )
    candidate_labels = get_option_labels(example)
    target_texts = [
        build_option_target_text(
            example=example,
            label=label,
            answer_scoring_mode=answer_scoring_mode,
        )
        for label in candidate_labels
    ]
    score_values = score_target_texts(
        model=model,
        tokenizer=tokenizer,
        user_prompts=[prompt] * len(candidate_labels),
        target_texts=target_texts,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        length_normalize=uses_option_text_mean_logprob(answer_scoring_mode),
    )
    scores = {
        label: float(score)
        for label, score in zip(candidate_labels, score_values)
    }
    pred_answer = max(scores, key=scores.get)
    return pred_answer, scores


def clean_explanation_text(text: str) -> str:
    cleaned = _safe_text(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"(?im)^\s*(?:rationale|explanation)\s*:?\s*", "", cleaned).strip()
    return cleaned


def strip_answer_cues(text: str, valid_labels: Iterable[str]) -> str:
    cleaned = clean_explanation_text(text)
    if not cleaned:
        return ""

    label_pattern = get_answer_label_pattern(valid_labels)
    patterns = [
        rf"(?is)^\s*(?:the\s+)?(?:correct\s+)?answer\s+is\s+(?:option\s+)?({label_pattern})(?:[\)\.\:]?\s*[^.?!\n]{{0,120}})?[.?!]?\s*",
        rf"(?is)^\s*(?:option|choice)\s+({label_pattern})(?:[\)\.\:]?\s*[^.?!\n]{{0,120}})?[.?!]?\s*",
        rf"(?is)\s+(?:final|answer)\s*:?\s*({label_pattern})(?:[\.\):]\s*.*)?$",
        rf"(?is)(?:[\s,;:-]*(?:therefore|thus|hence|so))?[\s,;:-]*(?:the\s+)?(?:correct\s+)?answer\s+is\s+(?:option\s+)?({label_pattern})(?:[\)\.\:]?\s*.*)?$",
        rf"(?is)(?:[\s,;:-]*(?:therefore|thus|hence|so))?[\s,;:-]*(?:option|choice)\s+({label_pattern})(?:[\)\.\:]?\s*.*)?$",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned).strip()
    return cleaned


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", _safe_text(text)))


def _trim_text_span(text: str, start_char: int, end_char: int) -> Tuple[int, int, str]:
    raw = text[start_char:end_char]
    left_trim = len(raw) - len(raw.lstrip())
    right_trim = len(raw.rstrip())
    start_char += left_trim
    end_char = start_char + max(right_trim - left_trim, 0)
    trimmed = text[start_char:end_char]
    return start_char, end_char, trimmed


def _build_span(text: str, start_char: int, end_char: int) -> Optional[Dict]:
    start_char, end_char, trimmed = _trim_text_span(text, start_char, end_char)
    if not trimmed:
        return None
    return {
        "start_char": start_char,
        "end_char": end_char,
        "text": trimmed,
        "word_count": _word_count(trimmed),
    }


def _chunk_long_span(
    text: str,
    start_char: int,
    end_char: int,
    max_words: int,
) -> List[Dict]:
    chunk_text = text[start_char:end_char]
    word_matches = list(re.finditer(r"\S+", chunk_text))
    if not word_matches:
        return []
    if len(word_matches) <= max_words:
        start_char, end_char, trimmed = _trim_text_span(text, start_char, end_char)
        return [
            {
                "start_char": start_char,
                "end_char": end_char,
                "text": trimmed,
                "word_count": _word_count(trimmed),
            }
        ]

    chunks: List[Dict] = []
    chunk_start = 0
    while chunk_start < len(word_matches):
        chunk_end = min(chunk_start + max_words, len(word_matches))
        local_start = word_matches[chunk_start].start()
        local_end = word_matches[chunk_end - 1].end()
        abs_start = start_char + local_start
        abs_end = start_char + local_end
        abs_start, abs_end, trimmed = _trim_text_span(text, abs_start, abs_end)
        if trimmed:
            chunks.append(
                {
                    "start_char": abs_start,
                    "end_char": abs_end,
                    "text": trimmed,
                    "word_count": _word_count(trimmed),
                }
            )
        chunk_start = chunk_end
    return chunks


def _find_marker_split_offset(
    chunk_text: str,
    min_words: int,
    markers: Sequence[str],
) -> Optional[int]:
    midpoint = len(chunk_text) / 2.0
    candidates: List[Tuple[float, int, int]] = []
    for marker in markers:
        pattern = rf"(?i)\b{re.escape(marker)}\b"
        for match in re.finditer(pattern, chunk_text):
            left_text = chunk_text[: match.start()]
            right_text = chunk_text[match.start() :]
            left_words = _word_count(left_text)
            right_words = _word_count(right_text)
            if left_words < min_words or right_words < min_words:
                continue
            candidates.append(
                (
                    abs(match.start() - midpoint),
                    -min(left_words, right_words),
                    match.start(),
                )
            )
    if not candidates:
        return None
    candidates.sort()
    return int(candidates[0][2])


def _normalize_clause_split_offsets(
    sentence_text: str,
    left_end: int,
    right_start: int,
) -> Tuple[int, int]:
    while left_end > 0 and sentence_text[left_end - 1].isspace():
        left_end -= 1
    while left_end > 0 and sentence_text[left_end - 1] in ",;:":
        left_end -= 1
        while left_end > 0 and sentence_text[left_end - 1].isspace():
            left_end -= 1
    while right_start < len(sentence_text) and sentence_text[right_start] in " ,;:":
        right_start += 1
    return left_end, right_start


def _is_fragment_like_clause(text: str, min_side_words: int) -> bool:
    tokens = re.findall(r"[A-Za-z0-9'-]+", _safe_text(text).lower())
    if not tokens:
        return True
    if len(tokens) > max(min_side_words + 1, 7):
        return False
    return tokens[-1] in _CLAUSE_FRAGMENT_TRAILING_TOKENS


def _maybe_add_clause_split_candidate(
    candidates: List[Tuple[float, int, int, int, int]],
    sentence_text: str,
    left_end: int,
    right_start: int,
    midpoint: float,
    min_side_words: int,
) -> None:
    left_end, right_start = _normalize_clause_split_offsets(
        sentence_text,
        left_end,
        right_start,
    )
    if left_end <= 0 or right_start >= len(sentence_text):
        return

    left_text = sentence_text[:left_end]
    right_text = sentence_text[right_start:]
    left_words = _word_count(left_text)
    right_words = _word_count(right_text)
    if left_words < min_side_words or right_words < min_side_words:
        return
    if _is_fragment_like_clause(left_text, min_side_words=min_side_words):
        return

    candidates.append(
        (
            abs(((left_end + right_start) / 2.0) - midpoint),
            -min(left_words, right_words),
            -max(left_words, right_words),
            left_end,
            right_start,
        )
    )


def _find_clause_split_offsets(
    sentence_text: str,
    min_side_words: int,
) -> Optional[Tuple[int, int]]:
    midpoint = len(sentence_text) / 2.0
    candidates: List[Tuple[float, int, int, int, int]] = []

    for match in re.finditer(r"[;:]", sentence_text):
        _maybe_add_clause_split_candidate(
            candidates,
            sentence_text,
            left_end=match.start(),
            right_start=match.end(),
            midpoint=midpoint,
            min_side_words=min_side_words,
        )

    for marker in _CLAUSE_SPLIT_MARKERS:
        pattern = rf"(?i)\b{re.escape(marker)}\b"
        for match in re.finditer(pattern, sentence_text):
            _maybe_add_clause_split_candidate(
                candidates,
                sentence_text,
                left_end=match.start(),
                right_start=match.start(),
                midpoint=midpoint,
                min_side_words=min_side_words,
            )

    if not candidates:
        return None
    candidates.sort()
    _, _, _, left_end, right_start = candidates[0]
    return int(left_end), int(right_start)


def _split_long_span(
    text: str,
    start_char: int,
    end_char: int,
    min_words: int,
    max_words: int,
) -> List[Dict]:
    start_char, end_char, trimmed = _trim_text_span(text, start_char, end_char)
    if not trimmed:
        return []

    if _word_count(trimmed) <= max_words:
        return [
            {
                "start_char": start_char,
                "end_char": end_char,
                "text": trimmed,
                "word_count": _word_count(trimmed),
            }
        ]

    for markers in (_STRONG_SPLIT_MARKERS, _WEAK_SPLIT_MARKERS):
        split_offset = _find_marker_split_offset(trimmed, min_words=min_words, markers=markers)
        if split_offset is None:
            continue
        left_end = start_char + split_offset
        right_start = start_char + split_offset
        left_spans = _split_long_span(
            text=text,
            start_char=start_char,
            end_char=left_end,
            min_words=min_words,
            max_words=max_words,
        )
        right_spans = _split_long_span(
            text=text,
            start_char=right_start,
            end_char=end_char,
            min_words=min_words,
            max_words=max_words,
        )
        if left_spans and right_spans:
            return left_spans + right_spans

    return _chunk_long_span(
        text=text,
        start_char=start_char,
        end_char=end_char,
        max_words=max_words,
    )


def _split_sentence_clause_span(
    text: str,
    start_char: int,
    end_char: int,
    min_words: int,
    max_words: int,
) -> List[Dict]:
    sentence_span = _build_span(text, start_char, end_char)
    if sentence_span is None:
        return []

    clause_trigger_words = min(max_words, _CLAUSE_SPLIT_WORD_THRESHOLD)
    min_side_words = max(min_words, _CLAUSE_MIN_SIDE_WORDS)
    sentence_word_count = int(sentence_span["word_count"])

    if sentence_word_count >= max(clause_trigger_words, min_side_words * 2):
        split_offsets = _find_clause_split_offsets(
            sentence_span["text"],
            min_side_words=min_side_words,
        )
        if split_offsets is not None:
            left_end, right_start = split_offsets
            left_span = _build_span(
                text,
                int(sentence_span["start_char"]),
                int(sentence_span["start_char"]) + left_end,
            )
            right_span = _build_span(
                text,
                int(sentence_span["start_char"]) + right_start,
                int(sentence_span["end_char"]),
            )
            if left_span is not None and right_span is not None:
                clause_spans = [left_span, right_span]
                if all(int(span["word_count"]) <= max_words for span in clause_spans):
                    return clause_spans
                return [sentence_span]

    return [sentence_span]


def _merge_short_neighbor_spans(
    text: str,
    spans: Sequence[Dict],
    min_words: int,
) -> List[Dict]:
    if not spans:
        return []

    merged_spans: List[Dict] = []
    for span in spans:
        if not merged_spans:
            merged_spans.append(dict(span))
            continue
        if span["word_count"] < min_words:
            prev = merged_spans.pop()
            start_char = prev["start_char"]
            end_char = span["end_char"]
            start_char, end_char, merged_text = _trim_text_span(text, start_char, end_char)
            merged_spans.append(
                {
                    "start_char": start_char,
                    "end_char": end_char,
                    "text": merged_text,
                    "word_count": _word_count(merged_text),
                }
            )
        else:
            merged_spans.append(dict(span))

    if len(merged_spans) >= 2 and merged_spans[-1]["word_count"] < min_words:
        tail = merged_spans.pop()
        prev = merged_spans.pop()
        start_char = prev["start_char"]
        end_char = tail["end_char"]
        start_char, end_char, merged_text = _trim_text_span(text, start_char, end_char)
        merged_spans.append(
            {
                "start_char": start_char,
                "end_char": end_char,
                "text": merged_text,
                "word_count": _word_count(merged_text),
            }
        )

    return merged_spans


def _split_explanation_into_sentence_spans(
    cleaned: str,
    min_words: int,
    max_words: int,
) -> List[Dict]:
    # Prefer sentence-level supervision and only split when a sentence is
    # genuinely too long, so the cache keeps semantically meaningful units.
    sentence_matches = list(re.finditer(r"[^.!?;\n]+(?:[.!?;]+|\n+|$)", cleaned))
    raw_spans: List[Dict] = []
    for sentence_match in sentence_matches:
        sentence_start = sentence_match.start()
        sentence_end = sentence_match.end()
        sentence_start, sentence_end, sentence_text = _trim_text_span(
            cleaned,
            sentence_start,
            sentence_end,
        )
        if not sentence_text:
            continue
        raw_spans.extend(
            _split_long_span(
                text=cleaned,
                start_char=sentence_start,
                end_char=sentence_end,
                min_words=min_words,
                max_words=max_words,
            )
        )
    return raw_spans


def _split_explanation_into_sentence_clause_spans(
    cleaned: str,
    min_words: int,
    max_words: int,
) -> List[Dict]:
    sentence_matches = list(re.finditer(r"[^.!?;\n]+(?:[.!?;]+|\n+|$)", cleaned))
    raw_spans: List[Dict] = []
    for sentence_match in sentence_matches:
        sentence_start = sentence_match.start()
        sentence_end = sentence_match.end()
        raw_spans.extend(
            _split_sentence_clause_span(
                text=cleaned,
                start_char=sentence_start,
                end_char=sentence_end,
                min_words=min_words,
                max_words=max_words,
            )
        )
    return raw_spans


def _split_explanation_into_fine_spans(
    cleaned: str,
    min_words: int,
    max_words: int,
) -> List[Dict]:
    sentence_matches = list(re.finditer(r"[^.!?;\n]+(?:[.!?;]+|\n+|$)", cleaned))
    raw_spans: List[Dict] = []
    for sentence_match in sentence_matches:
        sentence_start = sentence_match.start()
        sentence_end = sentence_match.end()
        sentence_text = cleaned[sentence_start:sentence_end]
        clause_matches = list(re.finditer(r"[^,]+(?:,|$)", sentence_text))
        if not clause_matches:
            clause_matches = [re.match(r".*", sentence_text)]
        for clause_match in clause_matches:
            if clause_match is None:
                continue
            clause_start = sentence_start + clause_match.start()
            clause_end = sentence_start + clause_match.end()
            clause_start, clause_end, clause_text = _trim_text_span(
                cleaned,
                clause_start,
                clause_end,
            )
            if not clause_text:
                continue
            raw_spans.extend(
                _split_long_span(
                    text=cleaned,
                    start_char=clause_start,
                    end_char=clause_end,
                    min_words=min_words,
                    max_words=max_words,
                )
            )
    return raw_spans


def _build_full_explanation_span(cleaned: str) -> List[Dict]:
    if not cleaned:
        return []
    start_char, end_char, trimmed = _trim_text_span(cleaned, 0, len(cleaned))
    if not trimmed:
        return []
    return [
        {
            "start_char": start_char,
            "end_char": end_char,
            "text": trimmed,
            "word_count": _word_count(trimmed),
            "index": 0,
        }
    ]


def split_explanation_into_spans(
    text: str,
    min_words: int = 6,
    max_words: int = 24,
    span_granularity: str = "sentence",
) -> List[Dict]:
    cleaned = _safe_text(text)
    if not cleaned:
        return []

    granularity = normalize_span_granularity(span_granularity)
    if granularity == "full":
        return _build_full_explanation_span(cleaned)

    if granularity == "sentence":
        raw_spans = _split_explanation_into_sentence_spans(
            cleaned=cleaned,
            min_words=min_words,
            max_words=max_words,
        )
    elif granularity == "sentence_clause":
        raw_spans = _split_explanation_into_sentence_clause_spans(
            cleaned=cleaned,
            min_words=min_words,
            max_words=max_words,
        )
    else:
        raw_spans = _split_explanation_into_fine_spans(
            cleaned=cleaned,
            min_words=min_words,
            max_words=max_words,
        )

    if not raw_spans:
        return []

    merged_spans = _merge_short_neighbor_spans(
        text=cleaned,
        spans=raw_spans,
        min_words=min_words,
    )
    for index, span in enumerate(merged_spans):
        span["index"] = index
    return merged_spans


def mask_rationale_span(
    rationale_text: str,
    span: Dict,
    replacement_text: str = "[omitted rationale span]",
) -> str:
    replacement_text = _safe_text(replacement_text) or "[omitted rationale span]"
    start_char = int(span["start_char"])
    end_char = int(span["end_char"])
    masked = f"{rationale_text[:start_char].rstrip()} {replacement_text} {rationale_text[end_char:].lstrip()}"
    return _safe_text(masked)


def score_target_texts(
    model,
    tokenizer,
    user_prompts: Sequence[str],
    target_texts: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 1,
    length_normalize: bool = False,
) -> List[float]:
    if len(user_prompts) != len(target_texts):
        raise ValueError("user_prompts and target_texts must have the same length.")
    if not user_prompts:
        return []
    scores: List[float] = []
    effective_batch_size = max(int(batch_size), 1)

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

        moved_inputs = {}
        for key, value in batch.items():
            if torch.is_floating_point(value):
                moved_inputs[key] = value.to(device=device, dtype=dtype)
            else:
                moved_inputs[key] = value.to(device=device)

        with torch.inference_mode():
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
                target_len = max(int(suffix_ids.numel()), 1)
                row_score = row_score / float(target_len)
            scores.append(float(row_score.item()))

        del outputs
        del logits
        del input_ids
        del attention_mask
        del moved_inputs
        del batch
        del prefix_id_list
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return scores


def score_option_distribution_for_prompt(
    model,
    tokenizer,
    example: Dict,
    user_prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 1,
    answer_scoring_mode: str = "letter_option_mean_logprob",
) -> Tuple[Dict[str, float], Dict[str, float], str]:
    option_labels = list(get_option_labels(example))
    target_texts = [
        build_option_target_text(
            example=example,
            label=label,
            answer_scoring_mode=answer_scoring_mode,
        )
        for label in option_labels
    ]
    prompt_batch = [user_prompt] * len(option_labels)
    score_values = score_target_texts(
        model=model,
        tokenizer=tokenizer,
        user_prompts=prompt_batch,
        target_texts=target_texts,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        length_normalize=uses_option_text_mean_logprob(answer_scoring_mode),
    )
    score_map = {
        label: float(score)
        for label, score in zip(option_labels, score_values)
    }
    score_tensor = torch.tensor(score_values, dtype=torch.float32)
    if not torch.any(torch.isfinite(score_tensor)):
        prob_map = {label: float("nan") for label in option_labels}
        return score_map, prob_map, ""
    prob_tensor = torch.softmax(score_tensor, dim=-1)
    prob_map = {
        label: float(prob)
        for label, prob in zip(option_labels, prob_tensor.tolist())
    }
    pred_answer = option_labels[int(torch.argmax(score_tensor).item())]
    return score_map, prob_map, pred_answer


def score_option_distributions_for_prompts(
    model,
    tokenizer,
    example: Dict,
    user_prompts: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 1,
    answer_scoring_mode: str = "letter_option_mean_logprob",
) -> List[Tuple[Dict[str, float], Dict[str, float], str]]:
    option_labels = list(get_option_labels(example))
    target_texts: List[str] = []
    prompt_batch: List[str] = []
    for user_prompt in user_prompts:
        prompt_batch.extend([user_prompt] * len(option_labels))
        target_texts.extend(
            [
                build_option_target_text(
                    example=example,
                    label=label,
                    answer_scoring_mode=answer_scoring_mode,
                )
                for label in option_labels
            ]
        )

    score_values = score_target_texts(
        model=model,
        tokenizer=tokenizer,
        user_prompts=prompt_batch,
        target_texts=target_texts,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        length_normalize=uses_option_text_mean_logprob(answer_scoring_mode),
    )
    if len(score_values) != len(prompt_batch):
        return []

    distributions: List[Tuple[Dict[str, float], Dict[str, float], str]] = []
    option_count = len(option_labels)
    for prompt_idx in range(len(user_prompts)):
        start_idx = prompt_idx * option_count
        end_idx = start_idx + option_count
        prompt_scores = score_values[start_idx:end_idx]
        score_map = {
            label: float(score)
            for label, score in zip(option_labels, prompt_scores)
        }
        score_tensor = torch.tensor(prompt_scores, dtype=torch.float32)
        if not torch.any(torch.isfinite(score_tensor)):
            prob_map = {label: float("nan") for label in option_labels}
            distributions.append((score_map, prob_map, ""))
            continue
        prob_tensor = torch.softmax(score_tensor, dim=-1)
        prob_map = {
            label: float(prob)
            for label, prob in zip(option_labels, prob_tensor.tolist())
        }
        pred_answer = option_labels[int(torch.argmax(score_tensor).item())]
        distributions.append((score_map, prob_map, pred_answer))

    return distributions


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


def _positive_metric_delta(reference: float, comparison: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(comparison):
        return 0.0
    return max(float(reference) - float(comparison), 0.0)


def build_faithfulness_training_fields(
    model,
    tokenizer,
    example: Dict,
    pred_answer: str,
    pred_answer_text: str,
    pred_explanation_stripped: str,
    device: torch.device,
    dtype: torch.dtype,
    top_k_spans: int = 4,
    min_span_words: int = 6,
    max_span_words: int = 24,
    min_selected_words: int = 12,
    span_granularity: str = "sentence",
    mask_replacement_text: str = "[omitted rationale span]",
    sufficiency_weight: float = 0.25,
    length_norm_alpha: float = 0.5,
    score_batch_size: int = 1,
    answer_scoring_mode: str = "letter_option_mean_logprob",
    span_score_mode: str = "margin_drop",
) -> Dict:
    rationale_text = _safe_text(pred_explanation_stripped)
    if not rationale_text:
        return {}

    spans = split_explanation_into_spans(
        rationale_text,
        min_words=min_span_words,
        max_words=max_span_words,
        span_granularity=span_granularity,
    )
    if not spans:
        return {}

    span_score_mode = normalize_span_score_mode(span_score_mode)
    full_prompt = build_recovery_prompt(
        example,
        rationale_text,
        answer_scoring_mode=answer_scoring_mode,
    )
    baseline_prompt = build_recovery_prompt(
        example,
        "Relevant rationale omitted.",
        answer_scoring_mode=answer_scoring_mode,
    )
    masked_prompts = [
        build_recovery_prompt(
            example,
            mask_rationale_span(
                rationale_text,
                span,
                replacement_text=mask_replacement_text,
            ),
            answer_scoring_mode=answer_scoring_mode,
        )
        for span in spans
    ]
    span_only_prompts = [
        build_recovery_prompt(
            example,
            span["text"],
            answer_scoring_mode=answer_scoring_mode,
        )
        for span in spans
    ]

    all_prompts = [full_prompt, baseline_prompt] + masked_prompts + span_only_prompts
    all_distributions = score_option_distributions_for_prompts(
        model=model,
        tokenizer=tokenizer,
        example=example,
        user_prompts=all_prompts,
        device=device,
        dtype=dtype,
        batch_size=score_batch_size,
        answer_scoring_mode=answer_scoring_mode,
    )
    if len(all_distributions) != len(all_prompts):
        return {}

    full_option_scores, full_option_probs, full_pred_answer = all_distributions[0]
    baseline_option_scores, baseline_option_probs, baseline_pred_answer = all_distributions[1]
    masked_distributions = all_distributions[2 : 2 + len(spans)]
    span_only_distributions = all_distributions[2 + len(spans) :]

    target_label = str(pred_answer).strip().upper()
    full_score = float(full_option_scores.get(target_label, float("-inf")))
    baseline_score = float(baseline_option_scores.get(target_label, float("-inf")))
    full_margin = _compute_answer_margin(full_option_scores, target_label)
    baseline_margin = _compute_answer_margin(baseline_option_scores, target_label)

    ranked_spans: List[Dict] = []
    for span, masked_info, span_only_info in zip(
        spans,
        masked_distributions,
        span_only_distributions,
    ):
        masked_option_scores, masked_option_probs, masked_pred_answer = masked_info
        span_only_option_scores, span_only_option_probs, span_only_pred_answer = span_only_info
        masked_score = float(masked_option_scores.get(target_label, float("-inf")))
        span_only_score = float(span_only_option_scores.get(target_label, float("-inf")))
        masked_margin = _compute_answer_margin(masked_option_scores, target_label)
        span_only_margin = _compute_answer_margin(span_only_option_scores, target_label)
        word_count = max(int(span["word_count"]), 1)
        norm = float(word_count) ** max(length_norm_alpha, 0.0)
        if span_score_mode == "margin_drop":
            necessity_score = _positive_metric_delta(full_margin, masked_margin)
            sufficiency_score = _positive_metric_delta(span_only_margin, baseline_margin)
        else:
            necessity_score = _positive_metric_delta(full_score, masked_score)
            sufficiency_score = _positive_metric_delta(span_only_score, baseline_score)
        faithful_score = (necessity_score / norm) + (
            sufficiency_weight * sufficiency_score / norm
        )
        span_record = dict(span)
        span_record.update(
            {
                "masked_rationale": mask_rationale_span(
                    rationale_text,
                    span,
                    replacement_text=mask_replacement_text,
                ),
                "full_recovery_score": full_score,
                "masked_recovery_score": masked_score,
                "span_only_recovery_score": span_only_score,
                "baseline_recovery_score": baseline_score,
                "full_recovery_margin": full_margin,
                "masked_recovery_margin": masked_margin,
                "span_only_recovery_margin": span_only_margin,
                "baseline_recovery_margin": baseline_margin,
                "masked_recovery_pred_answer": masked_pred_answer,
                "span_only_recovery_pred_answer": span_only_pred_answer,
                "span_score_mode": span_score_mode,
                "necessity_score": necessity_score,
                "sufficiency_score": sufficiency_score,
                "faithful_score": faithful_score,
            }
        )
        ranked_spans.append(span_record)

    ranked_spans.sort(
        key=lambda span: (
            float(span["faithful_score"]),
            float(span["necessity_score"]),
            -int(span["index"]),
        ),
        reverse=True,
    )
    max_selected_spans = max(int(top_k_spans), 1)
    min_selected_words = max(int(min_selected_words), 0)
    selected_ranked: List[Dict] = []
    selected_in_order: List[Dict] = []
    selected_text = ""
    selected_recovery_prompt = ""
    selected_option_scores: Dict[str, float] = {}
    selected_option_probs: Dict[str, float] = {}
    selected_pred_answer = ""
    selected_k = 0
    selected_word_count = 0

    last_attempt_text = ""
    last_attempt_scores: Dict[str, float] = {}
    last_attempt_probs: Dict[str, float] = {}
    last_attempt_pred_answer = ""
    last_attempt_k = 0
    last_attempt_word_count = 0

    for k in range(1, min(len(ranked_spans), max_selected_spans) + 1):
        candidate_ranked = ranked_spans[:k]
        candidate_in_order = sorted(candidate_ranked, key=lambda span: int(span["index"]))
        candidate_text = " ".join(
            _safe_text(span["text"]) for span in candidate_in_order
        ).strip()
        if not candidate_text:
            continue
        candidate_word_count = _word_count(candidate_text)

        candidate_prompt = build_recovery_prompt(
            example,
            candidate_text,
            answer_scoring_mode=answer_scoring_mode,
        )
        option_scores, option_probs, recovery_pred_answer = score_option_distribution_for_prompt(
            model=model,
            tokenizer=tokenizer,
            example=example,
            user_prompt=candidate_prompt,
            device=device,
            dtype=dtype,
            batch_size=score_batch_size,
            answer_scoring_mode=answer_scoring_mode,
        )

        last_attempt_text = candidate_text
        last_attempt_scores = option_scores
        last_attempt_probs = option_probs
        last_attempt_pred_answer = recovery_pred_answer
        last_attempt_k = k
        last_attempt_word_count = candidate_word_count

        if (
            recovery_pred_answer == str(example["gold_answer"]).strip().upper()
            and candidate_word_count >= min_selected_words
        ):
            selected_ranked = candidate_ranked
            selected_in_order = candidate_in_order
            selected_text = candidate_text
            selected_recovery_prompt = candidate_prompt
            selected_option_scores = option_scores
            selected_option_probs = option_probs
            selected_pred_answer = recovery_pred_answer
            selected_k = k
            selected_word_count = candidate_word_count
            break

    if not selected_in_order or not selected_text:
        return {
            "faith_recovery_selected_success": False,
            "faith_recovery_selected_k": 0,
            "faith_recovery_selected_rationale": last_attempt_text,
            "faith_recovery_selected_pred_answer": last_attempt_pred_answer,
            "faith_recovery_selected_option_scores": last_attempt_scores,
            "faith_recovery_selected_option_probs": last_attempt_probs,
            "faith_recovery_selected_last_attempt_k": last_attempt_k,
            "faith_recovery_selected_last_attempt_word_count": last_attempt_word_count,
            "faith_min_selected_words": min_selected_words,
            "faith_span_granularity": normalize_span_granularity(span_granularity),
            "faith_selected_span_count": 0,
            "faith_selected_spans": [],
            "faith_full_recovery_score": full_score,
            "faith_baseline_recovery_score": baseline_score,
            "faith_full_recovery_margin": full_margin,
            "faith_baseline_recovery_margin": baseline_margin,
            "faith_full_recovery_pred_answer": full_pred_answer,
            "faith_baseline_recovery_pred_answer": baseline_pred_answer,
            "faith_full_recovery_option_scores": full_option_scores,
            "faith_baseline_recovery_option_scores": baseline_option_scores,
            "faith_full_recovery_option_probs": full_option_probs,
            "faith_baseline_recovery_option_probs": baseline_option_probs,
            "faith_span_score_mode": span_score_mode,
            "faith_answer_scoring_mode": normalize_answer_scoring_mode(answer_scoring_mode),
        }

    selected_char_spans = [
        {
            "start_char": int(span["start_char"]) + 1,
            "end_char": int(span["end_char"]) + 1,
            "index": int(span["index"]),
        }
        for span in selected_in_order
    ]

    return {
        "faith_token_user_prompt": build_explanation_training_prompt(
            example=example,
            pred_answer=pred_answer,
            pred_answer_text=pred_answer_text,
        ),
        "faith_token_target_text": " " + rationale_text,
        "faith_token_selected_char_spans": selected_char_spans,
        "faith_recovery_selected_user_prompt": selected_recovery_prompt,
        "faith_recovery_selected_rationale": selected_text,
        "faith_recovery_selected_success": True,
        "faith_recovery_selected_k": selected_k,
        "faith_recovery_selected_word_count": selected_word_count,
        "faith_recovery_selected_pred_answer": selected_pred_answer,
        "faith_recovery_selected_option_scores": selected_option_scores,
        "faith_recovery_selected_option_probs": selected_option_probs,
        "faith_min_selected_words": min_selected_words,
        "faith_span_granularity": normalize_span_granularity(span_granularity),
        "faith_selected_span_count": len(selected_in_order),
        "faith_selected_spans": selected_in_order,
        "faith_full_recovery_score": full_score,
        "faith_baseline_recovery_score": baseline_score,
        "faith_full_recovery_margin": full_margin,
        "faith_baseline_recovery_margin": baseline_margin,
        "faith_full_recovery_pred_answer": full_pred_answer,
        "faith_baseline_recovery_pred_answer": baseline_pred_answer,
        "faith_full_recovery_option_scores": full_option_scores,
        "faith_baseline_recovery_option_scores": baseline_option_scores,
        "faith_full_recovery_option_probs": full_option_probs,
        "faith_baseline_recovery_option_probs": baseline_option_probs,
        "faith_span_score_mode": span_score_mode,
        "faith_answer_scoring_mode": normalize_answer_scoring_mode(answer_scoring_mode),
    }


def build_teacher_prediction_record(
    example: Dict,
    pred_answer: str,
    answer_scores: Dict[str, float],
    raw_generation: str,
    model_path: str,
) -> Dict:
    pred_answer_text = _safe_text(example["options"].get(pred_answer, ""))
    pred_explanation = clean_explanation_text(raw_generation)
    pred_explanation_stripped = strip_answer_cues(
        pred_explanation,
        valid_labels=get_option_labels(example),
    )
    return {
        "example_id": example["example_id"],
        "dataset_name": example["dataset_name"],
        "split": example["split"],
        "row_idx": example["row_idx"],
        "source": example["source"],
        "question": example["question"],
        "contexts": example.get("contexts", []) or [],
        "options": example["options"],
        "gold_answer": example["gold_answer"],
        "gold_option_text": example.get("gold_option_text", ""),
        "gold_explanation": example.get("gold_explanation", ""),
        "pred_answer": pred_answer,
        "pred_answer_text": pred_answer_text,
        "pred_explanation": pred_explanation,
        "pred_explanation_stripped": pred_explanation_stripped,
        "raw_generation": raw_generation,
        "answer_scores": answer_scores,
        "is_correct": pred_answer == example["gold_answer"],
        "model_path": model_path,
    }


def load_faithfulness_training_records(
    cache_path: str,
    limit: Optional[int] = None,
) -> List[Dict]:
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Faithfulness cache not found: {cache_path}")

    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("is_correct", False):
                continue
            cache_usable = record.get("faith_cache_usable")
            if cache_usable is None:
                cache_usable = record.get("faith_recovery_selected_success", False)
            if not cache_usable:
                continue
            if not record.get("faith_token_user_prompt") or not record.get(
                "faith_token_target_text"
            ):
                continue
            if not record.get("faith_token_selected_char_spans"):
                continue
            if not record.get("faith_recovery_selected_user_prompt"):
                continue
            if not record.get("faith_selected_spans"):
                continue
            records.append(record)
            if limit is not None and len(records) >= limit:
                break

    if not records:
        raise ValueError(
            f"No usable faithfulness training records were found in {cache_path}."
        )
    return records


class FaithfulnessAugmentedDataset(TorchDataset):
    def __init__(self, base_dataset, faith_records: Sequence[Dict]):
        self.base_dataset = base_dataset
        self.faith_records = list(faith_records)
        if not self.faith_records:
            raise ValueError("faith_records must not be empty.")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict:
        item = dict(self.base_dataset[index])
        item["_faith_record"] = self.faith_records[index % len(self.faith_records)]
        return item


class FaithfulnessDataCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        answer_scoring_mode: str = "letter_option_mean_logprob",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_scoring_mode = normalize_answer_scoring_mode(answer_scoring_mode)
        self._warned_on_missing_faith_records = False

    def _build_prompt_target_batch(
        self,
        user_prompts: Sequence[str],
        target_texts: Sequence[str],
        selected_char_spans_per_row: Optional[Sequence[Sequence[Dict]]] = None,
    ) -> Dict[str, torch.Tensor]:
        prefix_texts = [
            build_model_input_text(build_messages(prompt), self.tokenizer)
            for prompt in user_prompts
        ]
        full_texts = [
            prefix_text + target_text
            for prefix_text, target_text in zip(prefix_texts, target_texts)
        ]
        batch = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        target_mask = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
        target_weights = torch.zeros_like(batch["input_ids"], dtype=torch.float32)

        for row_idx, (prefix_text, target_text) in enumerate(zip(prefix_texts, target_texts)):
            prefix_ids = self.tokenizer(
                prefix_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )["input_ids"][0]
            full_ids = batch["input_ids"][row_idx]
            full_len = int(batch["attention_mask"][row_idx].sum().item())
            prefix_len = min(int(prefix_ids.shape[0]), full_len)
            if not torch.equal(full_ids[:prefix_len], prefix_ids[:prefix_len]):
                common_prefix = 0
                max_prefix = min(prefix_len, full_len)
                while (
                    common_prefix < max_prefix
                    and int(full_ids[common_prefix].item())
                    == int(prefix_ids[common_prefix].item())
                ):
                    common_prefix += 1
                prefix_len = common_prefix

            if selected_char_spans_per_row is None:
                if prefix_len < full_len:
                    target_mask[row_idx, prefix_len:full_len] = True
                    target_weights[row_idx, prefix_len:full_len] = 1.0
                continue

            if prefix_len >= full_len:
                continue

            target_token_spans = self._char_spans_to_token_spans(
                target_text=target_text,
                selected_char_spans=selected_char_spans_per_row[row_idx],
            )
            target_token_len = full_len - prefix_len
            for start_tok, end_tok, loss_weight in target_token_spans:
                clipped_start = min(max(int(start_tok), 0), target_token_len)
                clipped_end = min(max(int(end_tok), 0), target_token_len)
                if clipped_start >= clipped_end:
                    continue
                target_mask[
                    row_idx,
                    prefix_len + clipped_start : prefix_len + clipped_end,
                ] = True
                target_weights[
                    row_idx,
                    prefix_len + clipped_start : prefix_len + clipped_end,
                ] = float(loss_weight)

        batch["target_mask"] = target_mask
        batch["target_weights"] = target_weights
        return batch

    def _char_spans_to_token_spans(
        self,
        target_text: str,
        selected_char_spans: Sequence[Dict],
    ) -> List[Tuple[int, int, float]]:
        token_spans: List[Tuple[int, int, float]] = []
        for span in selected_char_spans:
            start_char = max(int(span.get("start_char", 0)), 0)
            end_char = max(int(span.get("end_char", 0)), start_char)
            loss_weight = max(float(span.get("loss_weight", 1.0)), 0.0)
            prefix_ids = self.tokenizer(
                target_text[:start_char],
                add_special_tokens=False,
            )["input_ids"]
            full_ids = self.tokenizer(
                target_text[:end_char],
                add_special_tokens=False,
            )["input_ids"]
            start_tok = len(prefix_ids)
            end_tok = len(full_ids)
            if end_tok > start_tok:
                token_spans.append((start_tok, end_tok, loss_weight))
        return token_spans

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        from transformers import default_data_collator

        base_features: List[Dict] = []
        faith_records: List[Dict] = []
        for feature in features:
            current = dict(feature)
            faith_records.append(current.pop("_faith_record", None))
            base_features.append(current)

        batch = default_data_collator(base_features)
        usable_records = [record for record in faith_records if record is not None]
        if not usable_records:
            if not self._warned_on_missing_faith_records:
                warnings.warn(
                    "FaithfulnessDataCollator received a batch without any "
                    "_faith_record values. Falling back to the base batch only, "
                    "which is expected for plain evaluation datasets and disables "
                    "explanation-specific losses for that batch.",
                    stacklevel=2,
                )
                self._warned_on_missing_faith_records = True
            return batch

        token_prompts = [record["faith_token_user_prompt"] for record in usable_records]
        token_targets = [record["faith_token_target_text"] for record in usable_records]
        token_selected_char_spans = [
            record["faith_token_selected_char_spans"] for record in usable_records
        ]
        token_batch = self._build_prompt_target_batch(
            token_prompts,
            token_targets,
            selected_char_spans_per_row=token_selected_char_spans,
        )
        batch["faith_token_input_ids"] = token_batch["input_ids"]
        batch["faith_token_attention_mask"] = token_batch["attention_mask"]
        batch["faith_token_target_mask"] = token_batch["target_mask"]
        batch["faith_token_token_weights"] = token_batch["target_weights"]

        selected_prompts: List[str] = []
        selected_targets: List[str] = []
        selected_group_sizes: List[int] = []
        for record in usable_records:
            option_labels = list(get_option_labels(record))
            unit_prompts = record.get("faith_recovery_unit_user_prompts") or [
                record["faith_recovery_selected_user_prompt"]
            ]
            for unit_prompt in unit_prompts:
                selected_group_sizes.append(len(option_labels))
                selected_prompts.extend([unit_prompt] * len(option_labels))
                selected_targets.extend(
                    [
                        build_option_target_text(
                            example=record,
                            label=label,
                            answer_scoring_mode=self.answer_scoring_mode,
                        )
                        for label in option_labels
                    ]
                )

        if selected_prompts:
            recovery_selected_batch = self._build_prompt_target_batch(
                selected_prompts,
                selected_targets,
            )
            batch["faith_recovery_selected_input_ids"] = recovery_selected_batch["input_ids"]
            batch["faith_recovery_selected_attention_mask"] = recovery_selected_batch[
                "attention_mask"
            ]
            batch["faith_recovery_selected_target_mask"] = recovery_selected_batch[
                "target_mask"
            ]
            batch["faith_recovery_selected_group_sizes"] = torch.tensor(
                selected_group_sizes,
                dtype=torch.long,
            )

        return batch
