#!/usr/bin/env python3

import argparse
import gc
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
OSTQUANT_REPO_DIR = PROJECT_ROOT / "backends" / "ostquant"

for path in (PROJECT_ROOT / "src", OSTQUANT_REPO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eaquant.evidence.schema import (  # noqa: E402
    ANSWER_SCORING_MODES,
    build_explanation_training_prompt,
    build_messages,
    build_recovery_prompt,
    generate_from_messages,
    get_input_device,
    mask_rationale_span,
    normalize_answer_scoring_mode,
    score_option_distribution_for_prompt,
    score_option_distributions_for_prompts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a medmix faithfulness cache using Qwen-extracted minimal "
            "evidence units. Recovery uses all full-precision faithful evidence "
            "units (F) without per-unit weights. Token KL uses quantization-"
            "affected evidence units (Q) with importance weights."
        )
    )
    parser.add_argument("--teacher_cache_path", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        required=True,
    )
    parser.add_argument("--quant_checkpoint", type=Path, required=True)
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default="fp16",
        choices=["auto", "fp16", "bf16", "fp32"],
    )
    parser.add_argument(
        "--qwen_compute_dtype",
        type=str,
        default="bf16",
        choices=["auto", "fp16", "bf16", "fp32"],
    )
    parser.add_argument(
        "--answer_scoring_mode",
        type=str,
        default="letter_option_mean_logprob",
        choices=list(ANSWER_SCORING_MODES),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--extract_max_new_tokens", type=int, default=256)
    parser.add_argument("--extract_min_new_tokens", type=int, default=0)
    parser.add_argument("--max_evidence_units", type=int, default=6)
    parser.add_argument("--min_evidence_words", type=int, default=3)
    parser.add_argument("--max_evidence_words", type=int, default=20)
    parser.add_argument(
        "--answer_leakage_mode",
        type=str,
        default="relaxed",
        choices=["strict", "relaxed", "off"],
        help=(
            "How aggressively to reject evidence spans that overlap the selected "
            "answer. strict rejects any answer text/token overlap, relaxed rejects "
            "option-label and full-answer leakage while keeping short biomedical "
            "symbols, and off only records overlap features."
        ),
    )
    parser.add_argument("--short_answer_symbol_max_chars", type=int, default=6)
    parser.add_argument("--short_answer_symbol_max_words", type=int, default=2)
    parser.add_argument("--answer_overlap_reject_ratio", type=float, default=0.6)
    parser.add_argument("--answer_overlap_reject_min_tokens", type=int, default=2)
    parser.add_argument(
        "--importance_sufficiency_alpha",
        type=float,
        default=1.0,
        help=(
            "Weight for sufficiency in importance=max(comprehensiveness, "
            "alpha*sufficiency)."
        ),
    )
    parser.add_argument(
        "--min_importance",
        type=float,
        default=0.0,
        help=(
            "Minimum positive teacher importance required for F. Importance is "
            "max(comprehensiveness, alpha*sufficiency). The comparison is strict, "
            "so 0.0 means teacher_importance > 0."
        ),
    )
    parser.add_argument(
        "--min_quant_impact",
        type=float,
        default=0.0,
        help=(
            "Minimum quant impact required for an F evidence unit to enter Q. "
            "By default, quant impact is max(teacher_importance - "
            "quant_importance, 0). If quant_keep_margin_beta > 0, the weighted "
            "evidence-only keep-margin gap is also considered. The comparison "
            "is strict, so 0.0 means quant_impact > 0. This controls inclusion "
            "only; loss weights use teacher importance."
        ),
    )
    parser.add_argument(
        "--quant_keep_margin_beta",
        type=float,
        default=0.0,
        help=(
            "Optional weight for direct evidence-only margin degradation in Q "
            "selection. The default 0.0 records keep-margin gaps without using "
            "them for Q selection. If set > 0, quant_impact=max(importance_gap, "
            "beta*max(teacher_keep_margin - quant_keep_margin, 0))."
        ),
    )
    parser.add_argument("--min_recovery_words", type=int, default=1)
    parser.add_argument(
        "--require_recovery_correct",
        type=lambda x: str(x).lower() == "true",
        default=True,
    )
    parser.add_argument("--token_weight_power", type=float, default=1.0)
    parser.add_argument("--token_weight_min", type=float, default=0.25)
    parser.add_argument("--token_weight_max", type=float, default=4.0)
    parser.add_argument(
        "--mask_replacement_text",
        type=str,
        default="[omitted rationale span]",
    )
    parser.add_argument("--score_batch_size", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--v_bits", type=int, default=4)
    parser.add_argument("--k_bits", type=int, default=4)
    parser.add_argument("--down_bits", type=int, default=4)
    parser.add_argument(
        "--train_enable_wquant",
        type=lambda x: str(x).lower() == "true",
        default=False,
    )
    parser.add_argument(
        "--allow_state_key_mismatch",
        type=lambda x: str(x).lower() == "true",
        default=False,
    )
    parser.add_argument("--rotate_ov", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument(
        "--rotate_post_rope",
        type=lambda x: str(x).lower() == "true",
        default=False,
    )
    parser.add_argument(
        "--online_qk_hadamard",
        type=lambda x: str(x).lower() == "true",
        default=True,
    )
    parser.add_argument("--smooth_qk", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--smooth_ov", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument(
        "--smooth_up_down",
        type=lambda x: str(x).lower() == "true",
        default=True,
    )
    parser.add_argument(
        "--smooth_norm_linear",
        type=lambda x: str(x).lower() == "true",
        default=True,
    )
    parser.add_argument("--rotate_down_dim", type=int, default=1)
    parser.add_argument("--sub_mean", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--use_klt", type=lambda x: str(x).lower() == "true", default=False)
    return parser.parse_args()


def _safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _load_tokenizer_for_model(model_path: str):
    from transformers import AutoTokenizer

    tokenizer_kwargs = {
        "use_fast": False,
        "trust_remote_code": True,
    }
    if "mistral" in str(model_path).lower():
        tokenizer_kwargs["legacy"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    except ImportError as exc:
        if "protobuf" not in str(exc).lower() or tokenizer_kwargs.get("legacy"):
            raise
        tokenizer_kwargs["legacy"] = True
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_causal_lm_and_tokenizer(model_path: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    tokenizer = _load_tokenizer_for_model(model_path)
    return model, tokenizer


def _build_quant_args(args: argparse.Namespace) -> SimpleNamespace:
    quant_args = SimpleNamespace()
    quant_args.model = args.model_path
    quant_args.seqlen = args.seqlen
    compute_dtype = args.compute_dtype
    if compute_dtype == "auto":
        compute_dtype = (
            "bf16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "fp16"
        )
    quant_args.bf16 = compute_dtype == "bf16"
    quant_args.fp16 = compute_dtype == "fp16"
    quant_args.use_sdpa = True
    quant_args.online_hadamard = "down"
    quant_args.rotate_down_dim = args.rotate_down_dim
    quant_args.qwen2_downfill = False
    quant_args.fsdp = ""
    quant_args.local_rank = -1
    quant_args.sub_mean = args.sub_mean
    quant_args.use_klt = args.use_klt
    quant_args.smooth_up_down = args.smooth_up_down
    quant_args.smooth_up_gate = False
    quant_args.smooth_qk = args.smooth_qk
    quant_args.smooth_ov = args.smooth_ov
    quant_args.smooth_norm_linear = args.smooth_norm_linear
    quant_args.fp32_had = True
    quant_args.online_qk_hadamard = args.online_qk_hadamard
    quant_args.rotate_post_rope = args.rotate_post_rope
    quant_args.rotate_pre_rope = False
    quant_args.rotate_ov = args.rotate_ov
    quant_args.force_rdtype_inplace = False
    quant_args.train_enable_wquant = args.train_enable_wquant

    quant_args.w_bits = args.w_bits
    quant_args.a_bits = args.a_bits
    quant_args.v_bits = args.v_bits
    quant_args.k_bits = args.k_bits
    quant_args.down_bits = args.down_bits
    quant_args.residual_bits = 16
    quant_args.attn_bits = 16
    quant_args.act_bits = 16

    quant_args.a_dynamic_method = "pertoken"
    quant_args.a_groupsize = -1
    quant_args.a_asym = True
    quant_args.a_clip_ratio = 1.0
    quant_args.w_groupsize = -1
    quant_args.w_asym = False
    quant_args.w_clip = True
    quant_args.v_groupsize = 128
    quant_args.v_asym = True
    quant_args.v_clip_ratio = 1.0
    quant_args.k_groupsize = 128
    quant_args.k_asym = True
    quant_args.k_clip_ratio = 1.0

    quant_args.embed_quant_params = dict(
        bits=quant_args.residual_bits,
        sym=not quant_args.a_asym,
        dynamic=True,
        dynamic_method="perchannel",
    )
    quant_args.weight_quant_params = dict(
        bits=quant_args.w_bits,
        sym=not quant_args.w_asym,
        groupsize=quant_args.w_groupsize,
        dynamic=True,
        dynamic_method="pertoken",
        mse=quant_args.w_clip,
    )
    quant_args.norm_quant_params = dict(
        bits=quant_args.a_bits,
        sym=not quant_args.a_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.a_clip_ratio,
        dynamic=True,
        dynamic_method=quant_args.a_dynamic_method,
    )
    quant_args.ropek_quant_params = dict(
        bits=quant_args.k_bits,
        sym=not quant_args.k_asym,
        groupsize=quant_args.k_groupsize,
        clip_ratio=quant_args.k_clip_ratio,
        dynamic=True,
        dynamic_method=quant_args.a_dynamic_method,
    )
    quant_args.v_proj_quant_params = dict(
        bits=quant_args.v_bits,
        sym=not quant_args.v_asym,
        groupsize=quant_args.v_groupsize,
        clip_ratio=quant_args.v_clip_ratio,
        dynamic=True,
        dynamic_method=quant_args.a_dynamic_method,
    )
    quant_args.pv_matmul_quant_params = dict(
        bits=quant_args.a_bits,
        sym=not quant_args.a_asym,
        groupsize=128,
        clip_ratio=quant_args.k_clip_ratio,
        dynamic=True,
        dynamic_method=quant_args.a_dynamic_method,
    )
    quant_args.mul_quant_params = dict(
        bits=quant_args.down_bits,
        sym=not quant_args.a_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.k_clip_ratio,
        dynamic=True,
        dynamic_method=quant_args.a_dynamic_method,
    )
    quant_args.q_proj_quant_params = dict(
        bits=quant_args.a_bits,
        sym=not quant_args.a_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.a_clip_ratio,
        dynamic=True,
        dynamic_method=quant_args.a_dynamic_method,
    )
    quant_args.ropeq_quant_params = dict(
        bits=quant_args.a_bits,
        sym=not quant_args.a_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.a_clip_ratio,
        dynamic=True,
        dynamic_method=quant_args.a_dynamic_method,
    )
    quant_args.k_proj_quant_params = dict(
        bits=quant_args.a_bits,
        sym=not quant_args.k_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.a_clip_ratio,
        dynamic=True,
        dynamic_method=quant_args.a_dynamic_method,
    )
    quant_args.qk_matmul_quant_params = dict(
        bits=quant_args.attn_bits,
        sym=not quant_args.a_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.a_clip_ratio,
        dynamic=True,
        dynamic_method=quant_args.a_dynamic_method,
    )
    quant_args.softmax_quant_params = dict(
        bits=quant_args.attn_bits,
        sym=not quant_args.a_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.a_clip_ratio,
        dynamic=True,
        dynamic_method=quant_args.a_dynamic_method,
    )
    quant_args.o_proj_quant_params = dict(
        bits=quant_args.residual_bits,
        sym=not quant_args.a_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.a_clip_ratio,
        dynamic=True,
        dynamic_method="perchannel",
    )
    quant_args.resadd1_quant_params = dict(
        bits=quant_args.residual_bits,
        sym=not quant_args.a_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.a_clip_ratio,
        dynamic=True,
        dynamic_method="perchannel",
    )
    quant_args.up_proj_quant_params = dict(
        bits=quant_args.a_bits,
        sym=not quant_args.a_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.k_clip_ratio,
        dynamic=True,
        dynamic_method=quant_args.a_dynamic_method,
    )
    quant_args.gate_proj_quant_params = dict(
        bits=quant_args.act_bits,
        sym=not quant_args.a_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.k_clip_ratio,
        dynamic=True,
        dynamic_method=quant_args.a_dynamic_method,
    )
    quant_args.silu_quant_params = dict(
        bits=quant_args.act_bits,
        sym=not quant_args.a_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.k_clip_ratio,
        dynamic=True,
        dynamic_method=quant_args.a_dynamic_method,
    )
    quant_args.down_proj_quant_params = dict(
        bits=quant_args.residual_bits,
        sym=not quant_args.a_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.k_clip_ratio,
        dynamic=True,
        dynamic_method="perchannel",
    )
    quant_args.resadd2_quant_params = dict(
        bits=quant_args.residual_bits,
        sym=not quant_args.a_asym,
        groupsize=quant_args.a_groupsize,
        clip_ratio=quant_args.k_clip_ratio,
        dynamic=True,
        dynamic_method="perchannel",
    )
    return quant_args


def _load_quantized_model_and_tokenizer(args: argparse.Namespace):
    from quant import ost_model_utils

    quant_args = _build_quant_args(args)
    lm = ost_model_utils.LM(quant_args)
    lm.model.eval()
    lm.fuse_layer_norms()
    lm.generate_rotate_parameters()
    lm.rotate_smooth_model_inplace()
    lm.set_quant_state(use_weight_quant=False, use_act_quant=False, use_fully_quant=False)

    checkpoint_obj = torch.load(args.quant_checkpoint, map_location="cpu")
    state_dict = checkpoint_obj.get("model", checkpoint_obj)
    missing_keys, unexpected_keys = lm.model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[WARN] Missing quantized state keys: {len(missing_keys)}")
        print(f"[WARN] First missing keys: {missing_keys[:20]}")
    if unexpected_keys:
        print(f"[WARN] Unexpected quantized state keys: {len(unexpected_keys)}")
        print(f"[WARN] First unexpected keys: {unexpected_keys[:20]}")
    if (missing_keys or unexpected_keys) and not args.allow_state_key_mismatch:
        raise RuntimeError(
            "Quantized checkpoint state_dict did not match the reconstructed model."
        )

    lm.set_quant_state(
        use_weight_quant=args.train_enable_wquant,
        use_act_quant=True,
        use_fully_quant=False,
    )
    if torch.cuda.is_available():
        target_device = torch.device("cuda:0")
        lm.model.to(target_device)
        lm.model.hf_device_map = {"": str(target_device)}
    lm.model.eval()
    if lm.tokenizer.pad_token_id is None and lm.tokenizer.eos_token_id is not None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    return lm.model, lm.tokenizer


def _release_model(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_source_records(args: argparse.Namespace) -> List[Dict]:
    if not args.teacher_cache_path.is_file():
        raise FileNotFoundError(f"Teacher cache not found: {args.teacher_cache_path}")

    records: List[Dict] = []
    with args.teacher_cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("is_correct", False):
                continue
            if not _safe_text(record.get("pred_explanation_stripped")):
                continue
            records.append(record)
            if args.limit is not None and len(records) >= args.limit:
                break
    if not records:
        raise ValueError("No usable correct records with explanations were found.")
    return records


def _target_label(record: Dict) -> str:
    return (_safe_text(record.get("pred_answer")) or _safe_text(record.get("gold_answer"))).upper()


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", _safe_text(text)))


def _compute_answer_margin(score_map: Dict[str, float], target_label: str) -> float:
    target = _safe_text(target_label).upper()
    target_score = float(score_map.get(target, float("-inf")))
    if not math.isfinite(target_score):
        return float("-inf")
    other_scores = [
        float(score)
        for label, score in score_map.items()
        if _safe_text(label).upper() != target and math.isfinite(float(score))
    ]
    if not other_scores:
        return target_score
    return target_score - max(other_scores)


def _metric_value(score_map: Dict[str, float], target_label: str) -> float:
    return _compute_answer_margin(score_map, target_label)


def _build_evidence_extraction_prompt(
    record: Dict,
    args: argparse.Namespace,
) -> str:
    options_text = "\n".join(
        f"{_safe_text(label).upper()}. {_safe_text(text)}"
        for label, text in record.get("options", {}).items()
    )
    selected_label = _target_label(record)
    selected_text = _safe_text(record.get("pred_answer_text"))
    rationale = _safe_text(record.get("pred_explanation_stripped"))
    return (
        "You are a key-evidence span extractor.\n\n"
        "Given a question, options, selected answer, and rationale, extract "
        "minimal evidence spans.\n\n"
        "Definition:\n"
        "An evidence span is a short phrase or clause that supports the selected "
        "answer without directly naming the selected answer.\n\n"
        "Rules:\n"
        f"1. Extract up to {int(args.max_evidence_units)} candidate evidence spans.\n"
        "2. Each evidence span must be an exact substring of the rationale.\n"
        f"3. Do NOT include the selected answer option text: {selected_text!r}.\n"
        f"4. Do NOT include the option label: {selected_label!r}.\n"
        "5. Do NOT include phrases such as \"the correct answer is\", "
        "\"therefore\", \"thus\", \"the most appropriate choice\", or "
        "\"the selected answer\".\n"
        "6. If a sentence contains the selected answer term, extract only the "
        "clue portion that does not name it.\n"
        "7. Prefer clinical/statistical/mechanistic clues, contrastive facts, "
        "symptoms, signs, lab or imaging findings, and causal links.\n"
        "8. Avoid generic conclusions and meta-text such as \"End of response\".\n"
        "9. Each span should usually be 3-15 words.\n"
        f"10. If a candidate is longer than {int(args.max_evidence_words)} words, "
        "split it into smaller evidence spans.\n"
        "11. Return JSON only, exactly with this schema:\n"
        "{\"evidence_units\":[{\"text\":\"exact substring\"}]}\n\n"
        f"Question:\n{_safe_text(record.get('question'))}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Selected answer:\n{selected_label}. {selected_text}\n\n"
        f"Rationale:\n{rationale}"
    )


def _extract_json_payload(text: str):
    cleaned = _safe_text(text)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    starts = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx >= 0]
    if not starts:
        return None
    start = min(starts)
    for end in range(len(cleaned), start, -1):
        candidate = cleaned[start:end].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _regex_find_span(rationale: str, needle: str) -> Tuple[int, int]:
    if not needle:
        return -1, -1
    direct = rationale.find(needle)
    if direct >= 0:
        return direct, direct + len(needle)
    lowered = rationale.lower()
    lowered_needle = needle.lower()
    direct = lowered.find(lowered_needle)
    if direct >= 0:
        return direct, direct + len(needle)

    pieces = [re.escape(part) for part in re.split(r"\s+", needle.strip()) if part]
    if not pieces:
        return -1, -1
    pattern = r"\s+".join(pieces)
    match = re.search(pattern, rationale, flags=re.IGNORECASE)
    if match is None:
        return -1, -1
    return int(match.start()), int(match.end())


def _normalize_for_filter(text: str) -> str:
    return re.sub(r"\s+", " ", _safe_text(text)).casefold()


_ANSWER_OVERLAP_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "when",
    "which",
    "with",
}


def _answer_overlap_tokens(text: str) -> List[str]:
    tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+(?:[+\-][A-Za-z0-9]+)?", _safe_text(text))
    ]
    return [token for token in tokens if token and token not in _ANSWER_OVERLAP_STOPWORDS]


def _is_short_answer_symbol(answer_text: str, args: argparse.Namespace) -> bool:
    text = _safe_text(answer_text)
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    if len(compact) > max(int(args.short_answer_symbol_max_chars), 1):
        return False
    if _word_count(text) > max(int(args.short_answer_symbol_max_words), 1):
        return False
    if re.search(r"[0-9+\-/]", compact):
        return True
    alnum = re.sub(r"[^A-Za-z0-9]", "", compact)
    if not alnum:
        return False
    if len(alnum) <= max(int(args.short_answer_symbol_max_chars), 1):
        return not alnum.islower()
    return False


def _selected_answer_texts(record: Dict) -> List[str]:
    target_label = _target_label(record)
    options = record.get("options", {}) if isinstance(record.get("options"), dict) else {}
    texts = [
        _safe_text(record.get("pred_answer_text")),
        _safe_text(record.get("gold_option_text")),
        _safe_text(options.get(target_label)),
    ]
    unique: List[str] = []
    seen = set()
    for text in texts:
        norm = _normalize_for_filter(text)
        if len(norm) < 2 or norm in seen:
            continue
        seen.add(norm)
        unique.append(text)
    return unique


def _answer_overlap_features(
    text: str,
    record: Dict,
    args: argparse.Namespace,
) -> Dict:
    normalized = _normalize_for_filter(text)
    span_tokens = set(_answer_overlap_tokens(text))
    selected_label = re.escape(_target_label(record))
    has_option_label = bool(
        re.search(rf"(^|\s){selected_label}\s*[\.\)]", _safe_text(text), flags=re.IGNORECASE)
    )
    best = {
        "answer_leakage_mode": args.answer_leakage_mode,
        "answer_overlap_has_option_label": has_option_label,
        "answer_overlap_has_selected_answer_text": False,
        "answer_overlap_selected_answer_is_short_symbol": False,
        "answer_overlap_token_count": 0,
        "answer_overlap_token_ratio": 0.0,
        "answer_overlap_tokens": [],
        "answer_overlap_answer_token_count": 0,
    }
    for answer_text in _selected_answer_texts(record):
        answer_norm = _normalize_for_filter(answer_text)
        answer_tokens = set(_answer_overlap_tokens(answer_text))
        overlap_tokens = sorted(span_tokens & answer_tokens)
        token_ratio = (
            len(overlap_tokens) / len(answer_tokens) if answer_tokens else 0.0
        )
        exact_text_overlap = bool(answer_norm and answer_norm in normalized)
        if (
            exact_text_overlap
            or len(overlap_tokens) > int(best["answer_overlap_token_count"])
            or token_ratio > float(best["answer_overlap_token_ratio"])
        ):
            best = {
                "answer_leakage_mode": args.answer_leakage_mode,
                "answer_overlap_has_option_label": has_option_label,
                "answer_overlap_has_selected_answer_text": exact_text_overlap,
                "answer_overlap_selected_answer_is_short_symbol": _is_short_answer_symbol(
                    answer_text,
                    args,
                ),
                "answer_overlap_token_count": len(overlap_tokens),
                "answer_overlap_token_ratio": float(token_ratio),
                "answer_overlap_tokens": overlap_tokens,
                "answer_overlap_answer_token_count": len(answer_tokens),
            }
    return best


def _evidence_span_rejection(
    text: str,
    record: Dict,
    args: argparse.Namespace,
) -> Tuple[bool, str, Dict]:
    cleaned = _safe_text(text)
    if not cleaned:
        return True, "empty_span", _answer_overlap_features(cleaned, record, args)
    overlap = _answer_overlap_features(cleaned, record, args)
    word_count = _word_count(cleaned)
    if word_count < max(int(args.min_evidence_words), 1):
        return True, "too_few_words", overlap
    if word_count > max(int(args.max_evidence_words), 1):
        return True, "too_many_words", overlap

    mode = args.answer_leakage_mode
    if mode != "off" and overlap["answer_overlap_has_option_label"]:
        return True, "option_label_overlap", overlap
    if mode == "strict":
        if overlap["answer_overlap_has_selected_answer_text"]:
            return True, "selected_answer_text_overlap", overlap
        if overlap["answer_overlap_token_count"] > 0:
            return True, "selected_answer_token_overlap", overlap
    elif mode == "relaxed" and not overlap["answer_overlap_selected_answer_is_short_symbol"]:
        if overlap["answer_overlap_has_selected_answer_text"]:
            return True, "selected_answer_text_overlap", overlap
        if (
            overlap["answer_overlap_token_count"]
            >= max(int(args.answer_overlap_reject_min_tokens), 1)
            and overlap["answer_overlap_token_ratio"]
            >= max(float(args.answer_overlap_reject_ratio), 0.0)
        ):
            return True, "selected_answer_token_overlap", overlap

    normalized = _normalize_for_filter(cleaned)
    forbidden_phrases = (
        "the correct answer is",
        "therefore",
        "thus",
        "the selected answer",
        "the most appropriate choice",
        "end of response",
        "please note",
        "hypothetical scenario",
    )
    for phrase in forbidden_phrases:
        if phrase in normalized:
            return True, "generic_or_meta_phrase", overlap
    return False, "", overlap


def _split_long_evidence_text(text: str, args: argparse.Namespace) -> List[str]:
    cleaned = _safe_text(text)
    max_words = max(int(args.max_evidence_words), 1)
    if _word_count(cleaned) <= max_words:
        return [cleaned]

    pieces = [
        piece.strip(" \t\n\r,;:.")
        for piece in re.split(r"(?:\n+|[.;]\s+|,\s+)", cleaned)
    ]
    split_pieces: List[str] = []
    for piece in pieces:
        if not piece:
            continue
        if _word_count(piece) <= max_words:
            split_pieces.append(piece)
            continue
        split_pieces.extend(
            part.strip(" \t\n\r,;:.")
            for part in re.split(
                r"\s+(?:and|but|while|whereas|because|which|that|with)\s+",
                piece,
                flags=re.IGNORECASE,
            )
            if part.strip(" \t\n\r,;:.")
        )
    return split_pieces or [cleaned]


def _parse_evidence_units(
    raw_generation: str,
    rationale: str,
    record: Dict,
    args: argparse.Namespace,
) -> Tuple[List[Dict], List[Dict]]:
    payload = _extract_json_payload(raw_generation)
    if isinstance(payload, dict):
        raw_units = payload.get("evidence_units", [])
    elif isinstance(payload, list):
        raw_units = payload
    else:
        raw_units = []

    units: List[Dict] = []
    rejected_units: List[Dict] = []
    seen_bounds = set()
    for raw_unit in raw_units:
        if isinstance(raw_unit, dict):
            raw_text = _safe_text(raw_unit.get("text") or raw_unit.get("evidence"))
        else:
            raw_text = _safe_text(raw_unit)
        if not raw_text:
            continue
        for text in _split_long_evidence_text(raw_text, args):
            start_char, end_char = _regex_find_span(rationale, text)
            if start_char < 0 or end_char <= start_char:
                rejected_units.append(
                    {
                        "start_char": -1,
                        "end_char": -1,
                        "text": text,
                        "word_count": _word_count(text),
                        "rejection_reason": "not_exact_substring",
                        **_answer_overlap_features(text, record, args),
                    }
                )
                continue
            span_text = rationale[start_char:end_char].strip()
            rejected, rejection_reason, overlap = _evidence_span_rejection(
                span_text,
                record,
                args,
            )
            if rejected:
                rejected_units.append(
                    {
                        "start_char": int(start_char),
                        "end_char": int(end_char),
                        "text": span_text,
                        "word_count": _word_count(span_text),
                        "rejection_reason": rejection_reason,
                        **overlap,
                    }
                )
                continue
            bounds = (start_char, end_char)
            if bounds in seen_bounds:
                continue
            seen_bounds.add(bounds)
            units.append(
                {
                    "start_char": int(start_char),
                    "end_char": int(end_char),
                    "text": span_text,
                    "word_count": _word_count(span_text),
                    "index": len(units),
                    **overlap,
                }
            )
            if len(units) >= max(int(args.max_evidence_units), 1):
                break
        if len(units) >= max(int(args.max_evidence_units), 1):
            break
    return units, rejected_units


def _extract_evidence_units(records: Sequence[Dict], args: argparse.Namespace) -> None:
    dtype = _resolve_dtype(args.qwen_compute_dtype)
    model, tokenizer = _load_causal_lm_and_tokenizer(args.qwen_model_path, dtype)
    device = get_input_device(model)
    print(f"[INFO] qwen extractor: {args.qwen_model_path}")

    for idx, record in enumerate(records, start=1):
        rationale = _safe_text(record.get("pred_explanation_stripped"))
        prompt = _build_evidence_extraction_prompt(record, args)
        raw_generation = generate_from_messages(
            model=model,
            tokenizer=tokenizer,
            messages=build_messages(prompt),
            max_new_tokens=args.extract_max_new_tokens,
            min_new_tokens=args.extract_min_new_tokens,
            device=device,
            dtype=dtype,
        )
        units, rejected_units = _parse_evidence_units(
            raw_generation,
            rationale,
            record,
            args,
        )
        record["qwen_evidence_raw_generation"] = raw_generation
        record["qwen_evidence_units"] = units
        record["qwen_evidence_rejected_units"] = rejected_units
        print(
            f"[EXTRACT] {idx}/{len(records)} {record.get('example_id')} "
            f"units={len(units)} rejected={len(rejected_units)}",
            flush=True,
        )

    _release_model(model, tokenizer)


def _score_full_and_masked(
    record: Dict,
    units: Sequence[Dict],
    *,
    model,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    prefix: str,
) -> None:
    rationale = _safe_text(record.get("pred_explanation_stripped"))
    target_label = _target_label(record)
    answer_scoring_mode = normalize_answer_scoring_mode(
        _safe_text(record.get("answer_scoring_mode", args.answer_scoring_mode))
        or args.answer_scoring_mode
    )
    prompts = [
        build_recovery_prompt(
            record,
            rationale,
            answer_scoring_mode=answer_scoring_mode,
        ),
        build_recovery_prompt(
            record,
            "",
            answer_scoring_mode=answer_scoring_mode,
        ),
    ]
    prompts.extend(
        build_recovery_prompt(
            record,
            mask_rationale_span(
                rationale,
                unit,
                replacement_text=args.mask_replacement_text,
            ),
            answer_scoring_mode=answer_scoring_mode,
        )
        for unit in units
    )
    prompts.extend(
        build_recovery_prompt(
            record,
            _safe_text(unit.get("text")),
            answer_scoring_mode=answer_scoring_mode,
        )
        for unit in units
    )
    distributions = score_option_distributions_for_prompts(
        model=model,
        tokenizer=tokenizer,
        example=record,
        user_prompts=prompts,
        device=device,
        dtype=dtype,
        batch_size=args.score_batch_size,
        answer_scoring_mode=answer_scoring_mode,
    )
    if len(distributions) != len(prompts):
        raise RuntimeError(
            f"{prefix} returned {len(distributions)} distributions but expected "
            f"{len(prompts)} for {record.get('example_id')}."
        )

    full_scores, full_probs, full_pred = distributions[0]
    empty_scores, empty_probs, empty_pred = distributions[1]
    full_metric = _metric_value(full_scores, target_label)
    empty_metric = _metric_value(empty_scores, target_label)
    record[f"{prefix}_full_option_scores"] = full_scores
    record[f"{prefix}_full_option_probs"] = full_probs
    record[f"{prefix}_full_pred_answer"] = full_pred
    record[f"{prefix}_full_margin"] = full_metric
    record[f"{prefix}_empty_option_scores"] = empty_scores
    record[f"{prefix}_empty_option_probs"] = empty_probs
    record[f"{prefix}_empty_pred_answer"] = empty_pred
    record[f"{prefix}_empty_margin"] = empty_metric

    remove_distributions = distributions[2 : 2 + len(units)]
    keep_distributions = distributions[2 + len(units) :]
    alpha = max(float(args.importance_sufficiency_alpha), 0.0)
    for unit, masked_info, keep_info in zip(
        units,
        remove_distributions,
        keep_distributions,
    ):
        masked_scores, masked_probs, masked_pred = masked_info
        keep_scores, keep_probs, keep_pred = keep_info
        masked_metric = _metric_value(masked_scores, target_label)
        keep_metric = _metric_value(keep_scores, target_label)
        raw_contribution = (
            full_metric - masked_metric
            if math.isfinite(float(full_metric)) and math.isfinite(float(masked_metric))
            else 0.0
        )
        raw_sufficiency = (
            keep_metric - empty_metric
            if math.isfinite(float(keep_metric)) and math.isfinite(float(empty_metric))
            else 0.0
        )
        comprehensiveness = max(raw_contribution, 0.0)
        sufficiency = max(raw_sufficiency, 0.0)
        importance = max(comprehensiveness, alpha * sufficiency)
        unit[f"{prefix}_masked_option_scores"] = masked_scores
        unit[f"{prefix}_masked_option_probs"] = masked_probs
        unit[f"{prefix}_masked_pred_answer"] = masked_pred
        unit[f"{prefix}_masked_margin"] = masked_metric
        unit[f"{prefix}_keep_option_scores"] = keep_scores
        unit[f"{prefix}_keep_option_probs"] = keep_probs
        unit[f"{prefix}_keep_pred_answer"] = keep_pred
        unit[f"{prefix}_keep_margin"] = keep_metric
        unit[f"{prefix}_contribution_raw"] = float(raw_contribution)
        unit[f"{prefix}_contribution_pos"] = float(comprehensiveness)
        unit[f"{prefix}_remove_margin_drop"] = float(raw_contribution)
        unit[f"{prefix}_remove_margin_drop_pos"] = float(comprehensiveness)
        unit[f"{prefix}_keep_margin_drop"] = float(raw_sufficiency)
        unit[f"{prefix}_keep_margin_drop_pos"] = float(sufficiency)
        unit[f"{prefix}_sufficiency_raw"] = float(raw_sufficiency)
        unit[f"{prefix}_sufficiency_pos"] = float(sufficiency)
        unit[f"{prefix}_comprehensiveness"] = float(comprehensiveness)
        unit[f"{prefix}_sufficiency"] = float(sufficiency)
        unit[f"{prefix}_importance"] = float(importance)


def _score_recovery(
    record: Dict,
    rationale_text: str,
    *,
    model,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> Tuple[str, Dict[str, float], Dict[str, float], str, float]:
    answer_scoring_mode = normalize_answer_scoring_mode(
        _safe_text(record.get("answer_scoring_mode", args.answer_scoring_mode))
        or args.answer_scoring_mode
    )
    prompt = build_recovery_prompt(
        record,
        rationale_text,
        answer_scoring_mode=answer_scoring_mode,
    )
    option_scores, option_probs, pred_answer = score_option_distribution_for_prompt(
        model=model,
        tokenizer=tokenizer,
        example=record,
        user_prompt=prompt,
        device=device,
        dtype=dtype,
        batch_size=args.score_batch_size,
        answer_scoring_mode=answer_scoring_mode,
    )
    margin = _metric_value(option_scores, _target_label(record))
    return prompt, option_scores, option_probs, pred_answer, margin


def _normalize_importance_weights(
    units: Sequence[Dict],
    args: argparse.Namespace,
) -> List[float]:
    power = max(float(args.token_weight_power), 0.0)
    raw_weights = [
        max(float(unit.get("teacher_importance", 0.0)), 0.0) ** power for unit in units
    ]
    mean_weight = sum(raw_weights) / len(raw_weights) if raw_weights else 0.0
    if mean_weight <= 0.0:
        normalized = [1.0 for _ in raw_weights]
    else:
        normalized = [weight / mean_weight for weight in raw_weights]
    min_weight = max(float(args.token_weight_min), 0.0)
    max_weight = max(float(args.token_weight_max), min_weight)
    return [float(min(max(weight, min_weight), max_weight)) for weight in normalized]


def _copy_full_rationale_recovery_fields(record: Dict) -> None:
    target_label = _target_label(record)
    full_scores = record.get("teacher_full_option_scores", {})
    empty_scores = record.get("teacher_empty_option_scores", {})
    record["faith_full_recovery_score"] = full_scores.get(target_label)
    record["faith_baseline_recovery_score"] = empty_scores.get(target_label)
    record["faith_full_recovery_margin"] = record.get("teacher_full_margin")
    record["faith_baseline_recovery_margin"] = record.get("teacher_empty_margin")
    record["faith_full_recovery_pred_answer"] = record.get("teacher_full_pred_answer")
    record["faith_baseline_recovery_pred_answer"] = record.get("teacher_empty_pred_answer")
    record["faith_full_recovery_option_scores"] = record.get("teacher_full_option_scores")
    record["faith_baseline_recovery_option_scores"] = record.get(
        "teacher_empty_option_scores"
    )
    record["faith_full_recovery_option_probs"] = record.get("teacher_full_option_probs")
    record["faith_baseline_recovery_option_probs"] = record.get(
        "teacher_empty_option_probs"
    )
    record["full_rationale_recovery_correct"] = (
        record.get("teacher_full_pred_answer") == target_label
    )


def _mark_no_selected_evidence(record: Dict, reason: str) -> None:
    record["selected_evidence_available"] = False
    record["faith_cache_usable"] = False
    record["faith_cache_failure_reason"] = reason
    record["faith_span_granularity"] = "none"
    record["faith_selected_span_count"] = 0
    record["faith_selected_spans"] = []
    record["faith_token_selected_char_spans"] = []
    record["faith_q_selected_span_count"] = 0
    record["faith_q_evidence_units"] = []


def _build_training_payload(
    record: Dict,
    *,
    teacher_model,
    teacher_tokenizer,
    teacher_device: torch.device,
    teacher_dtype: torch.dtype,
    args: argparse.Namespace,
) -> Dict:
    updated = dict(record)
    units = [dict(unit) for unit in record.get("qwen_evidence_units", [])]
    updated["candidate_evidence_recovery_correct"] = False
    updated["candidate_evidence_recovery_valid"] = False
    updated["selected_evidence_available"] = False
    updated["faith_cache_usable"] = False
    updated["faith_cache_failure_reason"] = ""
    updated["full_rationale_recovery_correct"] = False
    updated["faith_f_evidence_units"] = []
    updated["faith_q_evidence_units"] = []
    updated["faith_f_selected_span_count"] = 0
    updated["faith_q_selected_span_count"] = 0
    updated["faith_recovery_unit_user_prompts"] = []
    updated["faith_recovery_selected_user_prompt"] = ""
    updated["faith_recovery_selected_rationale"] = ""
    updated["faith_recovery_selected_k"] = 0
    updated["faith_recovery_selected_word_count"] = 0
    updated["faith_recovery_selected_pred_answer"] = ""
    updated["faith_recovery_selected_option_scores"] = {}
    updated["faith_recovery_selected_option_probs"] = {}
    updated["faith_recovery_selected_margin"] = None
    updated["faith_selected_span_count"] = 0
    updated["faith_selected_spans"] = []
    updated["faith_token_selected_char_spans"] = []
    updated["faith_span_granularity"] = "none"
    updated["faith_span_score_mode"] = "qwen_fp_mask_then_quant_impact"
    updated["faith_min_selected_words"] = int(args.min_recovery_words)
    updated["faith_answer_scoring_mode"] = normalize_answer_scoring_mode(
        args.answer_scoring_mode
    )
    if not units:
        _score_full_and_masked(
            updated,
            [],
            model=teacher_model,
            tokenizer=teacher_tokenizer,
            device=teacher_device,
            dtype=teacher_dtype,
            args=args,
            prefix="teacher",
        )
        _copy_full_rationale_recovery_fields(updated)
        updated.update(
            {
                "faith_recovery_selected_success": False,
                "faith_recovery_selected_failure_reason": "no_qwen_evidence_units",
                "qwen_evidence_units": [],
                "faith_f_evidence_units": [],
            }
        )
        _mark_no_selected_evidence(updated, "no_qwen_evidence_units")
        return updated

    _score_full_and_masked(
        updated,
        units,
        model=teacher_model,
        tokenizer=teacher_tokenizer,
        device=teacher_device,
        dtype=teacher_dtype,
        args=args,
        prefix="teacher",
    )
    _copy_full_rationale_recovery_fields(updated)
    f_units: List[Dict] = []
    for unit in units:
        importance = float(unit.get("teacher_importance", 0.0))
        unit["teacher_importance"] = importance
        if importance > float(args.min_importance):
            f_units.append(unit)

    if not f_units:
        updated.update(
            {
                "faith_recovery_selected_success": False,
                "faith_recovery_selected_failure_reason": "no_full_precision_f_evidence",
                "qwen_evidence_units": units,
                "faith_f_evidence_units": [],
                "faith_f_selected_span_count": 0,
            }
        )
        _mark_no_selected_evidence(updated, "no_full_precision_f_evidence")
        return updated

    selected_rationale = " ".join(_safe_text(unit.get("text")) for unit in f_units).strip()
    selected_word_count = _word_count(selected_rationale)
    (
        selected_prompt,
        selected_scores,
        selected_probs,
        selected_pred,
        selected_margin,
    ) = _score_recovery(
        updated,
        selected_rationale,
        model=teacher_model,
        tokenizer=teacher_tokenizer,
        device=teacher_device,
        dtype=teacher_dtype,
        args=args,
    )
    candidate_correct = selected_pred == _target_label(updated)
    if selected_word_count < int(args.min_recovery_words):
        valid = False
        failure_reason = "too_few_f_recovery_words"
    elif args.require_recovery_correct and not candidate_correct:
        valid = False
        failure_reason = "f_recovery_wrong_answer"
    else:
        valid = True
        failure_reason = ""

    updated["qwen_evidence_units"] = units
    updated["faith_f_evidence_units"] = f_units
    updated["faith_f_selected_span_count"] = len(f_units)
    updated["faith_recovery_selected_user_prompt"] = selected_prompt
    updated["faith_recovery_selected_rationale"] = selected_rationale
    updated["faith_recovery_selected_success"] = bool(candidate_correct)
    updated["candidate_evidence_recovery_correct"] = bool(candidate_correct)
    updated["candidate_evidence_recovery_valid"] = bool(valid)
    updated["faith_recovery_selected_failure_reason"] = failure_reason
    updated["faith_recovery_selected_k"] = len(f_units)
    updated["faith_recovery_selected_word_count"] = selected_word_count
    updated["faith_recovery_selected_pred_answer"] = selected_pred
    updated["faith_recovery_selected_option_scores"] = selected_scores
    updated["faith_recovery_selected_option_probs"] = selected_probs
    updated["faith_recovery_selected_margin"] = selected_margin
    updated["faith_recovery_unit_user_prompts"] = [
        build_recovery_prompt(
            updated,
            _safe_text(unit.get("text")),
            answer_scoring_mode=normalize_answer_scoring_mode(args.answer_scoring_mode),
        )
        for unit in f_units
    ]
    updated["faith_min_selected_words"] = int(args.min_recovery_words)
    updated["faith_span_granularity"] = "none"
    updated["faith_span_score_mode"] = "qwen_fp_mask_then_quant_impact"
    updated["faith_selected_span_count"] = 0
    updated["faith_selected_spans"] = []
    updated["faith_answer_scoring_mode"] = normalize_answer_scoring_mode(
        args.answer_scoring_mode
    )
    if not valid:
        _mark_no_selected_evidence(updated, failure_reason)
    return updated


def _mark_q_units(
    record: Dict,
    *,
    quant_model,
    quant_tokenizer,
    quant_device: torch.device,
    quant_dtype: torch.dtype,
    args: argparse.Namespace,
) -> Dict:
    updated = dict(record)
    f_units = [dict(unit) for unit in updated.get("faith_f_evidence_units", [])]
    candidate_valid = bool(
        updated.get(
            "candidate_evidence_recovery_valid",
            updated.get("faith_recovery_selected_success", False),
        )
    )
    if not candidate_valid or not f_units:
        updated["faith_q_evidence_units"] = []
        reason = _safe_text(
            updated.get("faith_cache_failure_reason")
            or updated.get("faith_recovery_selected_failure_reason")
            or "no_recoverable_f_evidence"
        )
        _mark_no_selected_evidence(updated, reason)
        return updated

    _score_full_and_masked(
        updated,
        f_units,
        model=quant_model,
        tokenizer=quant_tokenizer,
        device=quant_device,
        dtype=quant_dtype,
        args=args,
        prefix="quant",
    )

    q_units: List[Dict] = []
    keep_margin_beta = max(float(args.quant_keep_margin_beta), 0.0)
    for unit in f_units:
        teacher_importance = float(unit.get("teacher_importance", 0.0))
        quant_importance = float(unit.get("quant_importance", 0.0))
        importance_gap_raw = teacher_importance - quant_importance
        importance_gap = max(importance_gap_raw, 0.0)
        teacher_keep_margin = float(unit.get("teacher_keep_margin", float("nan")))
        quant_keep_margin = float(unit.get("quant_keep_margin", float("nan")))
        keep_margin_gap_raw = (
            teacher_keep_margin - quant_keep_margin
            if math.isfinite(teacher_keep_margin) and math.isfinite(quant_keep_margin)
            else 0.0
        )
        keep_margin_gap = max(keep_margin_gap_raw, 0.0)
        weighted_keep_margin_gap = keep_margin_beta * keep_margin_gap
        quant_impact = max(importance_gap, weighted_keep_margin_gap)
        unit["quant_contribution"] = quant_importance
        unit["quant_importance_gap_raw"] = float(importance_gap_raw)
        unit["quant_importance_gap"] = float(importance_gap)
        unit["quant_keep_margin_gap_raw"] = float(keep_margin_gap_raw)
        unit["quant_keep_margin_gap_pos"] = float(keep_margin_gap)
        unit["quant_keep_margin_gap_beta"] = float(keep_margin_beta)
        unit["quant_keep_margin_gap_weighted"] = float(weighted_keep_margin_gap)
        unit["quant_impact_score"] = quant_impact
        unit["quant_impact_source"] = (
            "keep_margin_gap" if weighted_keep_margin_gap > importance_gap else "importance_gap"
        )
        unit["quant_affected_evidence"] = bool(
            quant_impact > float(args.min_quant_impact)
        )
        if unit["quant_affected_evidence"]:
            q_units.append(unit)

    q_weights = _normalize_importance_weights(q_units, args)
    for unit, loss_weight in zip(q_units, q_weights):
        unit["loss_weight"] = float(loss_weight)

    selected_char_spans = [
        {
            "start_char": int(unit["start_char"]) + 1,
            "end_char": int(unit["end_char"]) + 1,
            "index": int(unit.get("index", 0)),
            "loss_weight": float(unit.get("loss_weight", 1.0)),
        }
        for unit in q_units
    ]

    updated["faith_f_evidence_units"] = f_units
    updated["faith_q_evidence_units"] = q_units
    updated["faith_q_selected_span_count"] = len(q_units)
    updated["faith_token_user_prompt"] = build_explanation_training_prompt(
        example=updated,
        pred_answer=_target_label(updated),
        pred_answer_text=_safe_text(updated.get("pred_answer_text")),
    )
    updated["faith_token_target_text"] = " " + _safe_text(
        updated.get("pred_explanation_stripped")
    )
    updated["faith_token_selected_char_spans"] = selected_char_spans
    if not q_units:
        _mark_no_selected_evidence(updated, "no_quant_affected_q_evidence")
    else:
        updated["selected_evidence_available"] = bool(selected_char_spans)
        updated["faith_cache_usable"] = bool(selected_char_spans)
        updated["faith_cache_failure_reason"] = "" if selected_char_spans else "no_token_char_spans"
        updated["faith_span_granularity"] = "qwen_minimal_evidence"
        updated["faith_selected_span_count"] = len(selected_char_spans)
        updated["faith_selected_spans"] = q_units
    return updated


def _build_summary(records: Sequence[Dict], args: argparse.Namespace) -> Dict:
    usable = [
        record
        for record in records
        if record.get(
            "faith_cache_usable",
            record.get("faith_recovery_selected_success", False)
            and bool(record.get("faith_token_selected_char_spans")),
        )
    ]
    usable_ids = {id(record) for record in usable}
    failure_reasons: Dict[str, int] = {}
    for record in records:
        if id(record) in usable_ids:
            continue
        reason = _safe_text(
            record.get("faith_cache_failure_reason")
            or record.get("faith_recovery_selected_failure_reason")
            or "unknown"
        ) or "unknown"
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    return {
        "teacher_cache_path": str(args.teacher_cache_path),
        "output_jsonl": str(args.output_jsonl),
        "model_path": args.model_path,
        "qwen_model_path": args.qwen_model_path,
        "quant_checkpoint": str(args.quant_checkpoint),
        "answer_scoring_mode": args.answer_scoring_mode,
        "max_evidence_units": int(args.max_evidence_units),
        "min_evidence_words": int(args.min_evidence_words),
        "max_evidence_words": int(args.max_evidence_words),
        "answer_leakage_mode": args.answer_leakage_mode,
        "short_answer_symbol_max_chars": int(args.short_answer_symbol_max_chars),
        "short_answer_symbol_max_words": int(args.short_answer_symbol_max_words),
        "answer_overlap_reject_ratio": float(args.answer_overlap_reject_ratio),
        "answer_overlap_reject_min_tokens": int(args.answer_overlap_reject_min_tokens),
        "importance_sufficiency_alpha": float(args.importance_sufficiency_alpha),
        "min_importance": float(args.min_importance),
        "min_quant_impact": float(args.min_quant_impact),
        "quant_keep_margin_beta": float(args.quant_keep_margin_beta),
        "token_weight_power": float(args.token_weight_power),
        "token_weight_min": float(args.token_weight_min),
        "token_weight_max": float(args.token_weight_max),
        "total_records": len(records),
        "usable_records": len(usable),
        "full_rationale_recovery_correct_records": sum(
            1 for record in records if record.get("full_rationale_recovery_correct", False)
        ),
        "candidate_evidence_recovery_correct_records": sum(
            1
            for record in records
            if record.get("candidate_evidence_recovery_correct", False)
        ),
        "candidate_evidence_recovery_valid_records": sum(
            1 for record in records if record.get("candidate_evidence_recovery_valid", False)
        ),
        "selected_evidence_available_records": sum(
            1 for record in records if record.get("selected_evidence_available", False)
        ),
        "records_with_f_evidence": sum(
            1 for record in records if record.get("faith_f_evidence_units")
        ),
        "records_with_q_evidence": sum(
            1 for record in records if record.get("faith_q_evidence_units")
        ),
        "failure_reasons": failure_reasons,
        "total_f_units": sum(len(record.get("faith_f_evidence_units", [])) for record in usable),
        "total_q_units": sum(len(record.get("faith_q_evidence_units", [])) for record in usable),
        "total_f_units_all_records": sum(
            len(record.get("faith_f_evidence_units", [])) for record in records
        ),
        "total_q_units_all_records": sum(
            len(record.get("faith_q_evidence_units", [])) for record in records
        ),
    }


def main() -> None:
    args = parse_args()
    records = _load_source_records(args)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] source records: {len(records)}")
    print(f"[INFO] output: {args.output_jsonl}")
    _extract_evidence_units(records, args)

    teacher_dtype = _resolve_dtype(args.compute_dtype)
    teacher_model, teacher_tokenizer = _load_causal_lm_and_tokenizer(
        args.model_path,
        teacher_dtype,
    )
    teacher_device = get_input_device(teacher_model)
    teacher_scored: List[Dict] = []
    for idx, record in enumerate(records, start=1):
        payload = _build_training_payload(
            record,
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
            teacher_device=teacher_device,
            teacher_dtype=teacher_dtype,
            args=args,
        )
        teacher_scored.append(payload)
        print(
            f"[TEACHER] {idx}/{len(records)} {record.get('example_id')} "
            f"F={len(payload.get('faith_f_evidence_units', []))} "
            f"recovery_correct={payload.get('candidate_evidence_recovery_correct', False)} "
            f"recovery_valid={payload.get('candidate_evidence_recovery_valid', False)}",
            flush=True,
        )
    _release_model(teacher_model, teacher_tokenizer)
    teacher_model = None
    teacher_tokenizer = None

    quant_dtype = _resolve_dtype(args.compute_dtype)
    quant_model, quant_tokenizer = _load_quantized_model_and_tokenizer(args)
    quant_device = get_input_device(quant_model)
    output_records: List[Dict] = []
    with args.output_jsonl.open("w", encoding="utf-8") as out_f:
        for idx, record in enumerate(teacher_scored, start=1):
            payload = _mark_q_units(
                record,
                quant_model=quant_model,
                quant_tokenizer=quant_tokenizer,
                quant_device=quant_device,
                quant_dtype=quant_dtype,
                args=args,
            )
            output_records.append(payload)
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(
                f"[QUANT] {idx}/{len(teacher_scored)} {record.get('example_id')} "
                f"Q={len(payload.get('faith_q_evidence_units', []))} "
                f"usable={payload.get('faith_cache_usable', False)}",
                flush=True,
            )
    _release_model(quant_model, quant_tokenizer)
    quant_model = None
    quant_tokenizer = None

    summary = _build_summary(output_records, args)
    summary_path = args.output_jsonl.with_name(args.output_jsonl.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved Qwen evidence cache to {args.output_jsonl}")
    print(f"[DONE] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
