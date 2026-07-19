import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from jinja2.exceptions import TemplateError


SYSTEM_PROMPT = (
    "You are a careful medical exam assistant. "
    "Always follow the requested response format exactly."
)

ANSWER_SCORING_MODES = (
    "single_letter",
    "letter_option_mean_logprob",
)


def _safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_answer_scoring_mode(answer_scoring_mode: str) -> str:
    mode = _safe_text(answer_scoring_mode) or "letter_option_mean_logprob"
    if mode not in ANSWER_SCORING_MODES:
        raise ValueError(
            "Unsupported answer_scoring_mode="
            f"{mode!r}. Expected one of {ANSWER_SCORING_MODES}."
        )
    return mode


def get_option_labels(example: Dict) -> List[str]:
    return [
        str(label).strip().upper()
        for label in dict(example.get("options", {})).keys()
        if str(label).strip()
    ]


def build_option_target_text(
    example: Dict,
    label: str,
    answer_scoring_mode: str = "letter_option_mean_logprob",
) -> str:
    normalized_label = str(label).strip().upper()
    if normalize_answer_scoring_mode(answer_scoring_mode) == "single_letter":
        return f" {normalized_label}"

    option_text = _safe_text(dict(example.get("options", {})).get(normalized_label, ""))
    if not option_text:
        return f" {normalized_label}"
    return f" {normalized_label}. {option_text}"


def build_answer_completion_instruction(
    answer_scoring_mode: str = "letter_option_mean_logprob",
) -> str:
    if normalize_answer_scoring_mode(answer_scoring_mode) == "single_letter":
        return "Return only the option label."
    return "Return the option label followed by the option text."


def format_example_block(example: Dict) -> str:
    lines = [f"Question:\n{_safe_text(example.get('question', ''))}"]
    for idx, context in enumerate(example.get("contexts", []) or [], start=1):
        context = _safe_text(context)
        if context:
            lines.append(f"Context {idx}:\n{context}")
    lines.append("Options:")
    for label, text in dict(example.get("options", {})).items():
        lines.append(f"{str(label).strip().upper()}. {_safe_text(text)}")
    return "\n\n".join(lines)


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


def build_messages(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


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


def mask_rationale_span(
    rationale_text: str,
    span: Dict,
    replacement_text: str = "[omitted rationale span]",
) -> str:
    rationale_text = _safe_text(rationale_text)
    start_char = max(int(span.get("start_char", 0)), 0)
    end_char = max(int(span.get("end_char", 0)), start_char)
    return _safe_text(
        f"{rationale_text[:start_char].rstrip()} "
        f"{replacement_text} "
        f"{rationale_text[end_char:].lstrip()}"
    )


def _score_from_span(span: Dict, weight_source: str) -> float:
    if weight_source == "uniform":
        return 1.0
    if weight_source == "teacher":
        keys = ("teacher_importance", "faithful_score", "necessity_score")
    elif weight_source == "omni_quant":
        keys = ("omni_loss_weight", "omni_quant_impact_score")
    elif weight_source == "cache_quant":
        keys = ("loss_weight", "quant_impact_score")
    else:
        raise ValueError(f"Unsupported faithfulness weight source: {weight_source}")

    for key in keys:
        value = span.get(key)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value) and value > 0:
            return value
    return 0.0


def _normalize_positive_weights(weights: Sequence[float]) -> List[float]:
    clean = [float(weight) if math.isfinite(float(weight)) else 0.0 for weight in weights]
    clean = [max(weight, 0.0) for weight in clean]
    total = sum(clean)
    if total <= 0:
        return [1.0 for _ in clean]
    scale = float(len(clean)) / total
    return [max(weight * scale, 0.0) for weight in clean]


def _prepare_record_weights(record: Dict, weight_source: str) -> Dict:
    prepared = dict(record)
    selected_spans = [dict(span) for span in record.get("faith_selected_spans", [])]
    if not selected_spans:
        return prepared

    weights = _normalize_positive_weights(
        [_score_from_span(span, weight_source) for span in selected_spans]
    )
    span_weight_by_index = {}
    for span, weight in zip(selected_spans, weights):
        span["loss_weight"] = float(weight)
        span_weight_by_index[int(span.get("index", len(span_weight_by_index)))] = float(weight)

    selected_char_spans = []
    for char_span in record.get("faith_token_selected_char_spans", []):
        char_span = dict(char_span)
        span_index = int(char_span.get("index", len(selected_char_spans)))
        char_span["loss_weight"] = span_weight_by_index.get(span_index, 1.0)
        selected_char_spans.append(char_span)

    prepared["faith_selected_spans"] = selected_spans
    prepared["faith_token_selected_char_spans"] = selected_char_spans
    return prepared


def load_faithfulness_training_records(
    cache_path: str,
    limit: Optional[int] = None,
    weight_source: str = "teacher",
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
            if not record.get("faith_token_user_prompt"):
                continue
            if not record.get("faith_token_target_text"):
                continue
            if not record.get("faith_token_selected_char_spans"):
                continue
            if not record.get("faith_recovery_selected_user_prompt"):
                continue
            if not record.get("faith_selected_spans"):
                continue
            records.append(_prepare_record_weights(record, weight_source))
            if limit is not None and len(records) >= limit:
                break

    if not records:
        raise ValueError(
            f"No usable faithfulness training records were found in {cache_path}."
        )
    return records


class FaithfulnessPromptCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        answer_scoring_mode: str = "letter_option_mean_logprob",
        use_recovery_unit_prompts: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.answer_scoring_mode = normalize_answer_scoring_mode(answer_scoring_mode)
        self.use_recovery_unit_prompts = bool(use_recovery_unit_prompts)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

            if prefix_len >= full_len:
                continue
            if selected_char_spans_per_row is None:
                target_mask[row_idx, prefix_len:full_len] = True
                target_weights[row_idx, prefix_len:full_len] = 1.0
                continue

            target_token_len = full_len - prefix_len
            for start_tok, end_tok, loss_weight in self._char_spans_to_token_spans(
                target_text=target_text,
                selected_char_spans=selected_char_spans_per_row[row_idx],
            ):
                clipped_start = min(max(int(start_tok), 0), target_token_len)
                clipped_end = min(max(int(end_tok), 0), target_token_len)
                if clipped_start >= clipped_end:
                    continue
                target_slice = slice(prefix_len + clipped_start, prefix_len + clipped_end)
                target_mask[row_idx, target_slice] = True
                target_weights[row_idx, target_slice] = float(loss_weight)

        batch["target_mask"] = target_mask
        batch["target_weights"] = target_weights
        return batch

    def _char_spans_to_token_spans(
        self,
        target_text: str,
        selected_char_spans: Sequence[Dict],
    ) -> List:
        token_spans = []
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

    def build(self, records: Sequence[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        token_prompts = [record["faith_token_user_prompt"] for record in records]
        token_targets = [record["faith_token_target_text"] for record in records]
        token_spans = [
            record["faith_token_selected_char_spans"] for record in records
        ]
        token_batch = self._build_prompt_target_batch(
            token_prompts,
            token_targets,
            selected_char_spans_per_row=token_spans,
        )

        recovery_prompts: List[str] = []
        recovery_targets: List[str] = []
        recovery_weights: List[float] = []
        recovery_group_sizes: List[int] = []
        for record in records:
            option_labels = get_option_labels(record)
            if not option_labels:
                continue
            if self.use_recovery_unit_prompts:
                unit_prompts = record.get("faith_recovery_unit_user_prompts") or [
                    record["faith_recovery_selected_user_prompt"]
                ]
                unit_weights = [
                    float(span.get("loss_weight", 1.0))
                    for span in record.get("faith_selected_spans", [])
                ]
                if len(unit_weights) < len(unit_prompts):
                    unit_weights.extend([1.0] * (len(unit_prompts) - len(unit_weights)))
            else:
                unit_prompts = [record["faith_recovery_selected_user_prompt"]]
                unit_weights = [1.0]

            for unit_prompt, unit_weight in zip(unit_prompts, unit_weights):
                recovery_group_sizes.append(len(option_labels))
                for label in option_labels:
                    recovery_prompts.append(unit_prompt)
                    recovery_targets.append(
                        build_option_target_text(
                            example=record,
                            label=label,
                            answer_scoring_mode=self.answer_scoring_mode,
                        )
                    )
                    recovery_weights.append(float(max(unit_weight, 0.0)))

        batches = {
            "token": {
                "input_ids": token_batch["input_ids"],
                "attention_mask_2d": token_batch["attention_mask"],
                "target_mask": token_batch["target_mask"],
                "token_weights": token_batch["target_weights"],
            }
        }
        if recovery_prompts:
            recovery_batch = self._build_prompt_target_batch(
                recovery_prompts,
                recovery_targets,
            )
            row_weights = torch.tensor(recovery_weights, dtype=torch.float32)
            recovery_token_weights = (
                recovery_batch["target_weights"] * row_weights.unsqueeze(-1)
            )
            batches["recovery"] = {
                "input_ids": recovery_batch["input_ids"],
                "attention_mask_2d": recovery_batch["attention_mask"],
                "target_mask": recovery_batch["target_mask"],
                "token_weights": recovery_token_weights,
                "group_sizes": torch.tensor(recovery_group_sizes, dtype=torch.long),
            }
        return batches


def build_faithfulness_prompt_batches(
    records: Sequence[Dict],
    tokenizer,
    max_length: int = 512,
    answer_scoring_mode: str = "letter_option_mean_logprob",
    use_recovery_unit_prompts: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    collator = FaithfulnessPromptCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        answer_scoring_mode=answer_scoring_mode,
        use_recovery_unit_prompts=use_recovery_unit_prompts,
    )
    return collator.build(records)
