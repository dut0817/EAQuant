import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from jinja2.exceptions import TemplateError
from torch import nn


VALID_ANSWERS = {"A", "B", "C", "D", "E"}
ANSWER_LABEL_ORDER = ("A", "B", "C", "D", "E")
ANSWER_KEY_MAP = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
OPTION_KEY_MAP = {value: key for key, value in ANSWER_KEY_MAP.items()}
SPLIT_FILE_MAP = {"train": "train.json", "val": "val.json", "test": "test.json"}
ANSWER_SCORING_MODES = ("single_letter", "letter_option_mean_logprob")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
SYSTEM_PROMPT = (
    "You are a careful medical exam assistant. "
    "Always follow the requested response format exactly."
)


def _build_answer_completion_instruction(answer_scoring_mode: str) -> str:
    if answer_scoring_mode == "single_letter":
        return "Return only the option label."
    if answer_scoring_mode == "letter_option_mean_logprob":
        return "Return the option label followed by the option text."
    raise ValueError(f"Unsupported answer_scoring_mode: {answer_scoring_mode}")


def _safe_text(value: str) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_reference_text(value: str) -> str:
    text = _safe_text(value).replace("[HIDDEN]", " ")
    return re.sub(r"\s+", " ", text).strip()


def _format_options_block(options: Dict[str, str]) -> str:
    lines: List[str] = []
    for letter in ANSWER_LABEL_ORDER:
        if letter in options:
            lines.append(f"{letter}. {_safe_text(options.get(letter, ''))}")
    return "\n".join(lines)


def _get_candidate_letters(options: Dict[str, str]) -> Tuple[str, ...]:
    return tuple(letter for letter in ANSWER_LABEL_ORDER if letter in options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a causal LM on MedExpQA and save predictions plus explanations as jsonl."
        )
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local Hugging Face model path.",
    )
    parser.add_argument(
        "--system_name",
        choices=("fp16", "medmix_baseline", "eaquant"),
        default="fp16",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="MedExpQA dataset root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which MedExpQA split to run.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Optional output path. Defaults under outputs/medexpqa.",
    )
    parser.add_argument(
        "--artifact_subdir",
        type=str,
        default=None,
        help="Optional output subdirectory overriding the default system name.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="MedExpQA language/config directory under data/, e.g. en.",
    )
    parser.add_argument(
        "--question_types",
        nargs="*",
        default=None,
        help="Optional MedExpQA type filter.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=220,
        help="Maximum generated tokens per example.",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=32,
        help="Minimum generated tokens for the main rationale-generation pass.",
    )
    parser.add_argument(
        "--answer_retry_max_new_tokens",
        type=int,
        default=8,
        help="Deprecated compatibility flag. Answers are selected by conditional log-prob scoring instead of an answer-only retry.",
    )
    parser.add_argument(
        "--answer_scoring_mode",
        type=str,
        default="letter_option_mean_logprob",
        choices=ANSWER_SCORING_MODES,
        help="How to score answer candidates for the final choice.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional example limit after loading.",
    )
    parser.add_argument(
        "--load_quant",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Compute dtype for inference.",
    )
    return parser.parse_args()


def _ensure_rmsnorm_compat() -> None:
    if hasattr(nn, "RMSNorm"):
        return

    class _CompatRMSNorm(nn.Module):
        def __init__(
            self,
            normalized_shape,
            eps=1e-6,
            elementwise_affine=True,
            device=None,
            dtype=None,
        ):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            if elementwise_affine:
                self.weight = nn.Parameter(
                    torch.ones(self.normalized_shape, device=device, dtype=dtype)
                )
            else:
                self.register_parameter("weight", None)

        def forward(self, x):
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight is not None:
                x = x * self.weight
            return x

    nn.RMSNorm = _CompatRMSNorm


def _maybe_allow_unsafe_torch_load() -> None:
    flag = os.environ.get("AWQ_ALLOW_UNSAFE_TORCH_LOAD", "").strip().lower()
    if flag not in {"1", "true", "yes", "y", "on"}:
        return

    import transformers.modeling_utils as modeling_utils
    import transformers.utils.import_utils as import_utils

    def _skip_safety_check():
        return

    import_utils.check_torch_load_is_safe = _skip_safety_check
    modeling_utils.check_torch_load_is_safe = _skip_safety_check
    print(
        "[WARN] AWQ_ALLOW_UNSAFE_TORCH_LOAD is enabled. "
        "Only use this with trusted local checkpoints."
    )


def _resolve_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.compute_dtype == "fp16":
        return torch.float16
    if args.compute_dtype == "bf16":
        return torch.bfloat16
    if args.compute_dtype == "fp32":
        return torch.float32

    if not torch.cuda.is_available():
        return torch.float32
    if args.load_quant:
        return torch.float16
    return torch.bfloat16


def _get_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        dm = model.hf_device_map
        if "" in dm and dm[""] != "cpu":
            return torch.device(dm[""])
        for device_name in dm.values():
            if device_name not in {"cpu", "disk"}:
                return torch.device(device_name)
    return next(model.parameters()).device


def _load_tokenizer_for_model(model_path: str):
    from transformers import AutoTokenizer

    tokenizer_kwargs = {
        "use_fast": False,
        "trust_remote_code": True,
    }
    if "mistral" in str(model_path).lower():
        tokenizer_kwargs["legacy"] = True

    try:
        return AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    except ImportError as exc:
        if "protobuf" not in str(exc).lower() or tokenizer_kwargs.get("legacy"):
            raise
        tokenizer_kwargs["legacy"] = True
        return AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)


def _load_model_and_tokenizer(args: argparse.Namespace, dtype: torch.dtype):
    from transformers import AutoTokenizer

    if args.load_quant:
        from accelerate import (
            init_empty_weights,
            infer_auto_device_map,
            load_checkpoint_in_model,
        )
        from awq.quantize.quantizer import real_quantize_model_weight
        from awq.utils.utils import simple_dispatch_model
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        config.use_cache = False

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=dtype, trust_remote_code=True
            )

        q_config = {"zero_point": True, "q_group_size": args.q_group_size}
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True
        )
        model.tie_weights()

        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
        )
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        model = simple_dispatch_model(model, device_map=device_map)
        model.eval()
        tokenizer = _load_tokenizer_for_model(args.model_path)
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    tokenizer = _load_tokenizer_for_model(args.model_path)
    return model, tokenizer


def _normalize_answer_letter(value: str) -> str:
    if not value:
        return ""
    normalized = str(value).strip().upper()
    if normalized in ANSWER_KEY_MAP:
        return ANSWER_KEY_MAP[normalized]
    if normalized in VALID_ANSWERS:
        return normalized

    match = re.search(r"\b([1-5]|[ABCDE])\b", normalized)
    if match:
        matched = match.group(1)
        return ANSWER_KEY_MAP.get(matched, matched)

    match = re.search(r"OPTION\s*([1-5]|[ABCDE])", normalized)
    if match:
        matched = match.group(1)
        return ANSWER_KEY_MAP.get(matched, matched)

    return ""


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def _try_parse_json_dict(candidate: str) -> Dict[str, str]:
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _decode_json_fragment(value: str) -> str:
    try:
        return str(json.loads(f'"{value}"'))
    except json.JSONDecodeError:
        return str(value)


def _extract_partial_json_dict(text: str) -> Dict[str, str]:
    recovered: Dict[str, str] = {}
    for key in ("answer", "final", "rationale", "explanation", "reasoning", "analysis"):
        if key in {"answer", "final"}:
            match = re.search(rf'(?is)"{key}"\s*:\s*"?([ABCDE])\b', text)
            if match:
                recovered[key] = match.group(1).upper()
                continue

        # Allow truncated quoted values like {"answer":"C" or {"rationale":"... without final }.
        match = re.search(
            rf'(?is)"{key}"\s*:\s*"((?:\\.|[^"\\])*)(?:"|$)',
            text,
        )
        if match:
            recovered[key] = _decode_json_fragment(match.group(1)).strip()

    return recovered


def _extract_json_from_text(text: str) -> Dict[str, str]:
    parsed = _try_parse_json_dict(text)
    if parsed:
        return parsed

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        parsed = _try_parse_json_dict(match.group(0))
        if parsed:
            return parsed

    first_brace = text.find("{")
    if first_brace != -1:
        candidate = text[first_brace:].strip()
        repair_candidates = []
        base = candidate.rstrip()
        for variant in (
            base,
            base.rstrip(", \n\r\t"),
            base.rstrip(", \n\r\t") + "}",
            base.rstrip(", \n\r\t") + '"}',
        ):
            if variant not in repair_candidates:
                repair_candidates.append(variant)

        for variant in repair_candidates:
            parsed = _try_parse_json_dict(variant)
            if parsed:
                return parsed

    partial = _extract_partial_json_dict(text)
    if partial:
        return partial

    return {}


def _build_output_path(args: argparse.Namespace) -> str:
    if args.output_jsonl:
        return args.output_jsonl

    model_name = Path(args.model_path).name.lower().replace("/", "_")
    output_dir = PROJECT_ROOT / "outputs" / "medexpqa"
    output_subdir = args.artifact_subdir or args.system_name
    if output_subdir:
        output_dir = output_dir / output_subdir
    return str(output_dir / f"{args.split}_{model_name}_original_predictions.jsonl")


def load_medexpqa_examples(
    data_root: str,
    split: str,
    lang: str = "en",
    question_types: List[str] | None = None,
) -> List[Dict]:
    split_name = SPLIT_FILE_MAP[split]
    data_root_path = Path(data_root)
    candidate_paths = (
        data_root_path / "data" / lang / split_name,
        data_root_path / lang / split_name,
        data_root_path / split_name,
    )
    split_path = next((path for path in candidate_paths if path.is_file()), None)
    if split_path is None:
        raise FileNotFoundError(
            "Could not find MedExpQA split file. Checked: "
            + ", ".join(str(path) for path in candidate_paths)
        )

    type_filter = {value.strip() for value in (question_types or []) if value.strip()}
    examples: List[Dict] = []

    with split_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {split_path}")

    for row_idx, ex in enumerate(data, start=1):
        if not isinstance(ex, dict):
            continue

        question_type = _safe_text(ex.get("type")) or "unknown"
        if type_filter and question_type not in type_filter:
            continue

        raw_options = ex.get("options") or {}
        options = {
            label: _safe_text(raw_options.get(option_key, ""))
            for label, option_key in OPTION_KEY_MAP.items()
        }
        options = {label: text for label, text in options.items() if text}
        if not options:
            continue

        correct_option_key = _safe_text(ex.get("correct_option"))
        gold_answer = ANSWER_KEY_MAP.get(
            correct_option_key,
            _normalize_answer_letter(correct_option_key),
        )
        explanation_map = ex.get("explanations") or {}
        gold_explanation_1 = _clean_reference_text(
            (explanation_map.get(correct_option_key) or {}).get("text", "")
        )
        gold_explanation_2 = _clean_reference_text(
            ex.get("full_answer_no_ref") or ex.get("full_answer") or gold_explanation_1
        )
        if not gold_explanation_1 and gold_explanation_2:
            gold_explanation_1 = gold_explanation_2
        if not gold_explanation_2 and gold_explanation_1:
            gold_explanation_2 = gold_explanation_1

        question_id_specific = _safe_text(ex.get("question_id_specific"))
        dataset_id = _safe_text(ex.get("id"))
        example_id = (
            question_id_specific
            or dataset_id
            or f"medexpqa_{split}_{row_idx:05d}"
        )
        examples.append(
            {
                "example_id": example_id,
                "split": split,
                "question_type": question_type,
                "source_lang": _safe_text(ex.get("lang")) or lang,
                "source_file": str(split_path),
                "row_idx": row_idx,
                "question": _safe_text(ex.get("full_question")),
                "options": options,
                "gold_explanation_1": gold_explanation_1,
                "gold_explanation_2": gold_explanation_2,
                "gold_answer": gold_answer,
                "dataset_id": dataset_id,
                "question_id_specific": question_id_specific,
                "year": _safe_text(ex.get("year")),
            }
        )

    return examples


def build_answer_selection_prompt(
    example: Dict,
    answer_scoring_mode: str = "letter_option_mean_logprob",
) -> str:
    options = example["options"]
    return (
        "Answer the medical multiple-choice question.\n"
        "Select the single best answer.\n\n"
        f"{_build_answer_completion_instruction(answer_scoring_mode)}\n\n"
        f"Question:\n{example['question']}\n\n"
        f"Options:\n{_format_options_block(options)}\n\n"
        "Final:"
    )


def build_prompt(example: Dict, pred_answer: str, pred_answer_text: str) -> str:
    options = example["options"]
    return (
        "Answer the medical multiple-choice question.\n"
        "The final answer has already been selected.\n"
        "Write a concise medical rationale that supports this selected answer.\n\n"
        "Output exactly in this format:\n"
        "Rationale: <2-3 concise medical sentences explaining the key evidence>\n\n"
        f"Question:\n{example['question']}\n\n"
        f"Options:\n{_format_options_block(options)}\n\n"
        f"Selected answer:\n{pred_answer}. {pred_answer_text}\n"
    )


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
            # Some Mistral-family templates reject an explicit `system` role and
            # require the instruction to be folded into the first user turn.
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


def _fold_system_into_first_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not messages or messages[0].get("role") != "system":
        return messages

    system_content = str(messages[0].get("content", "")).strip()
    remaining_messages = messages[1:]
    folded_messages: List[Dict[str, str]] = []
    merged_first_user = False

    for message in remaining_messages:
        folded_message = dict(message)
        if not merged_first_user and folded_message.get("role") == "user":
            user_content = str(folded_message.get("content", "")).strip()
            if system_content and user_content:
                folded_message["content"] = f"{system_content}\n\n{user_content}"
            elif system_content:
                folded_message["content"] = system_content
            merged_first_user = True
        folded_messages.append(folded_message)

    if not merged_first_user:
        return messages

    return folded_messages


def _extract_answer_from_text(text: str, options: Dict[str, str]) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for line in reversed(lines):
        if "|" in line or "/" in line:
            continue

        match = re.search(r"(?i)\bfinal\s*:?\s*([ABCDE])(?:[\.\):]\s*.*)?$", line)
        if match:
            return match.group(1).upper()

        match = re.match(r"(?i)^final\s*:?\s*([ABCDE])(?:[\.\):]\s*.*)?$", line)
        if match:
            return match.group(1).upper()

        match = re.search(r"(?i)\banswer\s*:?\s*([ABCDE])(?:[\.\):]\s*.*)?$", line)
        if match:
            return match.group(1).upper()

        match = re.match(r"(?i)^answer\s*:?\s*([ABCDE])(?:[\.\):]\s*.*)?$", line)
        if match:
            return match.group(1).upper()

        match = re.match(
            r"(?i)^(?:the\s+)?(?:correct\s+)?answer\s+is\s+([ABCDE])(?:[\.\):]?\s*.*)?$",
            line,
        )
        if match:
            return match.group(1).upper()

        match = re.match(r"(?i)^([ABCDE])[\.\):]\s+.+$", line)
        if match:
            return match.group(1).upper()

        if re.fullmatch(r"(?i)[ABCDE]", line):
            return line.upper()

    normalized_text = _normalize_text(text)
    if len(normalized_text) > 80:
        return ""

    for letter, option_text in options.items():
        normalized_option = _normalize_text(option_text)
        if normalized_text == normalized_option:
            return letter
        if normalized_option and normalized_option in normalized_text:
            extra_chars = max(len(normalized_text) - len(normalized_option), 0)
            if extra_chars <= 12:
                return letter

    if re.fullmatch(r"[ABCDE]", normalized_text.upper()):
        return normalized_text.upper()

    return ""


def _extract_json_answer_field(parsed: Dict[str, str]) -> str:
    for key in ("final", "answer"):
        answer_value = str(parsed.get(key, "")).strip()
        if not answer_value or "|" in answer_value or "/" in answer_value:
            continue
        answer = _normalize_answer_letter(answer_value)
        if answer:
            return answer

    return ""


def _extract_structured_response(text: str) -> Tuple[str, str]:
    answer = ""
    explanation = ""

    answer_match = re.search(
        r"(?im)^\s*final\s*:?\s*([ABCDE])(?:[\.\):]\s*.*)?\s*$",
        text,
    )
    if not answer_match:
        answer_match = re.search(
        r"(?im)^\s*answer\s*:?\s*([ABCDE])(?:[\.\):]\s*.*)?\s*$",
        text,
        )
    if not answer_match:
        answer_match = re.search(
            r"(?is)\bfinal\s*:?\s*([ABCDE])(?:[\.\):]\s*.*)?$",
            text,
        )
    if not answer_match:
        answer_match = re.search(
            r"(?is)\banswer\s*:?\s*([ABCDE])(?:[\.\):]\s*.*)?$",
            text,
        )
    if answer_match:
        answer = answer_match.group(1).upper()

    explanation_match = re.search(
        r"(?is)\brationale\s*:?\s*(.*?)(?:\n+\s*(?:final|answer)\s*:|\s+(?:final|answer)\s*:|\Z)",
        text,
    )
    if not explanation_match:
        explanation_match = re.search(
        r"(?is)\bexplanation\s*:?\s*(.*?)(?:\n+\s*(?:final|answer)\s*:|\s+(?:final|answer)\s*:|\Z)",
        text,
        )
    if explanation_match:
        explanation = explanation_match.group(1).strip()
    elif answer_match:
        prefix = text[: answer_match.start()].strip()
        if prefix:
            explanation = re.sub(
                r"(?im)^\s*(?:rationale|explanation)\s*:?\s*",
                "",
                prefix,
            ).strip()

    return answer, explanation


def _clean_explanation_text(text: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"(?is)\[answer_retry\].*$", "", cleaned).strip()
    answer_first_then_rationale = re.match(
        r"(?is)^\s*(?:the\s+)?(?:correct\s+)?answer\s+is\b.*?\b(?:rationale|explanation)\s*:?\s*(.*)$",
        cleaned,
    )
    if answer_first_then_rationale:
        cleaned = answer_first_then_rationale.group(1).strip()
    cleaned = re.sub(r"(?im)^\s*(?:rationale|explanation)\s*:?\s*", "", cleaned).strip()
    cleaned = re.sub(
        r"(?is)^\s*(?:the\s+)?(?:correct\s+)?answer\s+is\s+(?:option\s+)?[ABCDE](?:[\)\.\:]?\s*[^.?!\n]{0,120})?[.?!]?\s*",
        "",
        cleaned,
    ).strip()
    cleaned = re.sub(
        r"(?is)^\s*(?:option|choice)\s+[ABCDE](?:[\)\.\:]?\s*[^.?!\n]{0,120})?[.?!]?\s*",
        "",
        cleaned,
    ).strip()
    cleaned = re.sub(
        r"(?is)\s+(?:final|answer)\s*:?\s*[ABCDE](?:[\.\):]\s*.*)?$",
        "",
        cleaned,
    ).strip()
    cleaned = re.sub(
        r"(?is)(?:[\s,;:-]*(?:therefore|thus|hence|so))?[\s,;:-]*(?:the\s+)?(?:correct\s+)?answer\s+is\s+(?:option\s+)?[ABCDE](?:[\)\.\:]?\s*.*)?$",
        "",
        cleaned,
    ).strip()
    cleaned = re.sub(
        r"(?is)(?:[\s,;:-]*(?:therefore|thus|hence|so))?[\s,;:-]*(?:option|choice)\s+[ABCDE](?:[\)\.\:]?\s*.*)?$",
        "",
        cleaned,
    ).strip()
    cleaned_lower = cleaned.lower()
    if re.fullmatch(r"(?is)(?:answer|final)\s*:?\s*[ABCDE]\b", cleaned):
        return ""
    if cleaned in {"{", "}", '""'}:
        return ""
    if cleaned_lower in {
        "rationale",
        "explanation",
        "medical rationale",
        "medical explanation",
        "the rationale",
        "the explanation",
    }:
        return ""
    return cleaned


def _parse_prediction_output(decoded: str, example: Dict) -> Tuple[str, str]:
    pred_answer, pred_explanation = _extract_structured_response(decoded)
    pred_explanation = _clean_explanation_text(pred_explanation)
    answer_line_match = re.search(
        r"(?im)(?:^\s*(?:final|answer)\s*:.*$|\b(?:final|answer)\s*:\s*[ABCDE])",
        decoded,
    )

    if not pred_answer or not pred_explanation:
        parsed = _extract_json_from_text(decoded)
        if not pred_answer:
            pred_answer = _extract_json_answer_field(parsed)
        if not pred_explanation:
            pred_explanation = _clean_explanation_text(
                str(parsed.get("rationale", parsed.get("explanation", ""))).strip()
            )

    if not pred_answer:
        pred_answer = _extract_answer_from_text(decoded, example["options"])

    if not pred_explanation:
        if answer_line_match:
            prefix = decoded[: answer_line_match.start()].strip()
            if prefix:
                pred_explanation = _clean_explanation_text(
                    re.sub(
                        r"(?im)^\s*(?:rationale|explanation)\s*:?\s*",
                        "",
                        prefix,
                    ).strip()
                )

    if not pred_explanation and not answer_line_match:
        pred_explanation = _clean_explanation_text(decoded.strip())

    return pred_answer, pred_explanation


def _build_messages(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _generate_from_messages(
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


def _build_answer_score_suffixes(
    example: Dict, answer_scoring_mode: str
) -> Dict[str, str]:
    options = example["options"]
    candidate_letters = _get_candidate_letters(options)
    if answer_scoring_mode == "single_letter":
        return {letter: f" {letter}" for letter in candidate_letters}
    if answer_scoring_mode == "letter_option_mean_logprob":
        return {
            letter: f" {letter}. {options[letter]}"
            for letter in candidate_letters
        }
    raise ValueError(
        f"Unsupported answer_scoring_mode: {answer_scoring_mode}"
    )


def _score_answer_options(
    model,
    tokenizer,
    example: Dict,
    device: torch.device,
    dtype: torch.dtype,
    answer_scoring_mode: str,
) -> str:
    prompt = build_answer_selection_prompt(
        example,
        answer_scoring_mode=answer_scoring_mode,
    )
    messages = _build_messages(prompt)
    input_text = build_model_input_text(messages, tokenizer)
    candidate_letters = _get_candidate_letters(example["options"])
    if not candidate_letters:
        raise ValueError("Example is missing answer options.")
    candidate_suffixes = _build_answer_score_suffixes(
        example, answer_scoring_mode
    )
    candidate_texts = [
        input_text + candidate_suffixes[letter] for letter in candidate_letters
    ]
    batch = tokenizer(candidate_texts, return_tensors="pt", padding=True)
    prefix_ids = tokenizer(input_text, return_tensors="pt")["input_ids"][0]

    moved_inputs = {}
    for key, value in batch.items():
        if torch.is_floating_point(value):
            moved_inputs[key] = value.to(device=device, dtype=dtype)
        else:
            moved_inputs[key] = value.to(device=device)

    with torch.inference_mode():
        outputs = model(**moved_inputs)

    logits = outputs.logits
    input_ids = moved_inputs["input_ids"]
    attention_mask = moved_inputs["attention_mask"]
    scores: Dict[str, float] = {}

    for row_idx, letter in enumerate(candidate_letters):
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
        if prefix_len <= 0:
            scores[letter] = float("-inf")
            continue

        suffix_ids = candidate_ids[prefix_len:full_len]
        if suffix_ids.numel() == 0:
            scores[letter] = float("-inf")
            continue

        suffix_logits = logits[row_idx, prefix_len - 1 : full_len - 1, :]
        suffix_log_probs = torch.log_softmax(suffix_logits.float(), dim=-1)
        token_scores = suffix_log_probs.gather(
            dim=-1,
            index=suffix_ids.unsqueeze(-1),
        ).squeeze(-1)
        score = float(token_scores.sum().item())
        # Length-normalize the full-answer mode so option text content matters
        # more than raw token count.
        if answer_scoring_mode == "letter_option_mean_logprob":
            score /= max(int(token_scores.numel()), 1)
        scores[letter] = score

    return max(scores, key=scores.get)


def run_one(
    model,
    tokenizer,
    example: Dict,
    max_new_tokens: int,
    min_new_tokens: int,
    answer_retry_max_new_tokens: int,
    answer_scoring_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[str, str, str]:
    pred_answer = _score_answer_options(
        model=model,
        tokenizer=tokenizer,
        example=example,
        device=device,
        dtype=dtype,
        answer_scoring_mode=answer_scoring_mode,
    )
    pred_answer_text = example["options"].get(pred_answer, "")
    decoded = _generate_from_messages(
        model=model,
        tokenizer=tokenizer,
        messages=_build_messages(build_prompt(example, pred_answer, pred_answer_text)),
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        device=device,
        dtype=dtype,
    )
    _, pred_explanation = _parse_prediction_output(decoded, example)
    return pred_answer, pred_explanation, decoded


def main() -> None:
    args = parse_args()
    _ensure_rmsnorm_compat()
    _maybe_allow_unsafe_torch_load()

    dtype = _resolve_dtype(args)
    output_jsonl = _build_output_path(args)
    output_dir = os.path.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    examples = load_medexpqa_examples(
        args.data_root,
        args.split,
        lang=args.lang,
        question_types=args.question_types,
    )
    if args.limit is not None:
        examples = examples[: args.limit]

    if not examples:
        raise ValueError("No MedExpQA examples matched the requested split/filter.")

    model, tokenizer = _load_model_and_tokenizer(args, dtype)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = _get_input_device(model)

    print(f"[INFO] examples: {len(examples)}")
    print(f"[INFO] output: {output_jsonl}")
    print(f"[INFO] dtype: {dtype}")
    print(f"[INFO] model: {args.model_path}")
    print(f"[INFO] lang: {args.lang}")
    print(f"[INFO] answer_scoring_mode: {args.answer_scoring_mode}")
    if args.load_quant:
        print(f"[INFO] load_quant: {args.load_quant}")

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for idx, example in enumerate(examples, start=1):
            pred_answer, pred_explanation, raw_generation = run_one(
                model=model,
                tokenizer=tokenizer,
                example=example,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                answer_retry_max_new_tokens=args.answer_retry_max_new_tokens,
                answer_scoring_mode=args.answer_scoring_mode,
                device=device,
                dtype=dtype,
            )

            record = {
                "example_id": example["example_id"],
                "split": example["split"],
                "question_type": example["question_type"],
                "source_lang": example["source_lang"],
                "source_file": example["source_file"],
                "row_idx": example["row_idx"],
                "question": example["question"],
                "options": example["options"],
                "gold_answer": example["gold_answer"],
                "gold_explanation_1": example["gold_explanation_1"],
                "gold_explanation_2": example["gold_explanation_2"],
                "dataset_id": example["dataset_id"],
                "question_id_specific": example["question_id_specific"],
                "year": example["year"],
                "pred_answer": pred_answer,
                "pred_explanation": pred_explanation,
                "raw_generation": raw_generation,
                "is_correct": pred_answer == example["gold_answer"],
                "system": args.system_name,
                "model_path": args.model_path,
                "load_quant": args.load_quant,
                "answer_scoring_mode": args.answer_scoring_mode,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"[{idx}/{len(examples)}] {example['example_id']} "
                f"pred={pred_answer or '?'} gold={example['gold_answer']}"
            )

    print(f"[DONE] Saved predictions to {output_jsonl}")


if __name__ == "__main__":
    main()
