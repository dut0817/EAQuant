import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
OSTQUANT_REPO_DIR = SCRIPT_DIR
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

for path in (PROJECT_ROOT / "src", OSTQUANT_REPO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eaquant.evaluation.inference import (  # noqa: E402
    _ensure_rmsnorm_compat,
    load_medexpqa_examples,
    run_one,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MedExpQA inference with an OSTQuant quantized checkpoint."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Base local Hugging Face model path used to build the OSTQuant model.",
    )
    parser.add_argument(
        "--system_name",
        choices=("medmix_baseline", "eaquant"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quant_checkpoint",
        type=str,
        required=True,
        help="Path to the OSTQuant quantized checkpoint (*.pt).",
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
        help="Optional output path. Defaults under outputs/medexpqa/ostquant.",
    )
    parser.add_argument(
        "--artifact_subdir",
        type=str,
        default=None,
        help="Optional runtime/output subdirectory overriding the default system name.",
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
        default="single_letter",
        choices=["single_letter", "letter_option_mean_logprob"],
        help="How to score answer candidates for the final choice.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional example limit after loading.",
    )
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Compute dtype for inference.",
    )
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--v_bits", type=int, default=4)
    parser.add_argument("--k_bits", type=int, default=4)
    parser.add_argument("--down_bits", type=int, default=4)
    parser.add_argument("--rotate_ov", type=str, default="True")
    parser.add_argument("--rotate_post_rope", type=str, default="False")
    parser.add_argument("--online_qk_hadamard", type=str, default="True")
    parser.add_argument("--smooth_qk", type=str, default="True")
    parser.add_argument("--smooth_ov", type=str, default="True")
    parser.add_argument("--smooth_up_down", type=str, default="True")
    parser.add_argument("--smooth_norm_linear", type=str, default="True")
    parser.add_argument("--train_enable_wquant", type=str, default="False")
    parser.add_argument("--sub_mean", type=str, default="False")
    parser.add_argument("--use_klt", type=str, default="False")
    parser.add_argument("--rotate_down_dim", type=int, default=1)
    return parser.parse_args()


def _bool_string(value: str) -> str:
    return "True" if str(value).strip().lower() in {"1", "true", "yes", "y", "on"} else "False"


def _resolve_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.compute_dtype == "fp16":
        return torch.float16
    if args.compute_dtype == "bf16":
        return torch.bfloat16
    if args.compute_dtype == "fp32":
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16


def _build_output_path(args: argparse.Namespace) -> str:
    if args.output_jsonl:
        return args.output_jsonl

    ckpt_name = Path(args.quant_checkpoint).stem.lower()
    output_dir = OUTPUT_ROOT / "medexpqa" / "ostquant"
    output_subdir = args.artifact_subdir or args.system_name
    if output_subdir:
        output_dir = output_dir / output_subdir
    return str(output_dir / f"{args.split}_{ckpt_name}_predictions.jsonl")


def _build_runtime_dir(args: argparse.Namespace) -> Path:
    model_name = Path(args.model_path).name
    ckpt_name = Path(args.quant_checkpoint).stem
    runtime_root = OUTPUT_ROOT / "runtime" / "ostquant"
    runtime_subdir = args.artifact_subdir or args.system_name
    if runtime_subdir:
        runtime_root = runtime_root / runtime_subdir
    return runtime_root / "runtime" / f"{model_name}_{ckpt_name}"


def _build_ostquant_args(
    args: argparse.Namespace,
    runtime_dir: Path,
    dtype: torch.dtype,
):
    import quant._get_args as ost_get_args

    cli_args = [
        "ostquant_inference",
        "--output_dir",
        str(runtime_dir),
        "--model",
        args.model_path,
        "--seqlen",
        str(args.seqlen),
        "--w_bits",
        str(args.w_bits),
        "--a_bits",
        str(args.a_bits),
        "--v_bits",
        str(args.v_bits),
        "--k_bits",
        str(args.k_bits),
        "--down_bits",
        str(args.down_bits),
        "--rotate_ov",
        _bool_string(args.rotate_ov),
        "--rotate_post_rope",
        _bool_string(args.rotate_post_rope),
        "--online_qk_hadamard",
        _bool_string(args.online_qk_hadamard),
        "--smooth_qk",
        _bool_string(args.smooth_qk),
        "--smooth_ov",
        _bool_string(args.smooth_ov),
        "--smooth_up_down",
        _bool_string(args.smooth_up_down),
        "--smooth_norm_linear",
        _bool_string(args.smooth_norm_linear),
        "--train_enable_wquant",
        _bool_string(args.train_enable_wquant),
        "--sub_mean",
        _bool_string(args.sub_mean),
        "--use_klt",
        _bool_string(args.use_klt),
        "--rotate_down_dim",
        str(args.rotate_down_dim),
        "--eval_strategy",
        "no",
        "--save_strategy",
        "no",
        "--report_to",
        "none",
        "--logging_steps",
        "1",
        "--per_device_train_batch_size",
        "1",
        "--max_steps",
        "1",
        "--lm_eval",
        "False",
        "--distribute",
        "False",
    ]

    if dtype == torch.bfloat16:
        cli_args += ["--bf16", "True", "--fp16", "False"]
    elif dtype == torch.float16:
        cli_args += ["--bf16", "False", "--fp16", "True"]
    else:
        cli_args += ["--bf16", "False", "--fp16", "False"]

    old_argv = sys.argv[:]
    try:
        sys.argv = cli_args
        return ost_get_args.parse_args()
    finally:
        sys.argv = old_argv


def _load_ostquant_model_and_tokenizer(
    args: argparse.Namespace, dtype: torch.dtype
):
    _ensure_rmsnorm_compat()

    checkpoint_path = Path(args.quant_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"OSTQuant checkpoint not found: {checkpoint_path}"
        )

    runtime_dir = _build_runtime_dir(args)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    import quant.ost_model_utils as ost_model_utils

    ost_args = _build_ostquant_args(args, runtime_dir, dtype)
    lm = ost_model_utils.LM(ost_args)
    lm.model.eval()

    if ost_args.rotate:
        lm.fuse_layer_norms()
        lm.generate_rotate_parameters()
        lm.rotate_smooth_model_inplace()

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain a 'model' key: {args.quant_checkpoint}")

    lm.model.load_state_dict(checkpoint["model"])
    # Follow OSTQuant's load_qmodel_path evaluation path:
    # the saved checkpoint already contains quantized weights, so we should
    # enable activation quantization but not re-quantize weights again.
    lm.set_quant_state(
        use_weight_quant=False,
        use_act_quant=True,
        use_fully_quant=ost_args.fully_quant,
    )
    lm.set_dynamic(True)
    lm.model.config.use_cache = True
    if hasattr(lm.model, "generation_config"):
        lm.model.generation_config.use_cache = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm.model.to(device)
    lm.model.eval()

    tokenizer = lm.tokenizer
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return lm.model, tokenizer, device


def main() -> None:
    args = parse_args()
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

    model, tokenizer, device = _load_ostquant_model_and_tokenizer(args, dtype)

    print(f"[INFO] examples: {len(examples)}")
    print(f"[INFO] output: {output_jsonl}")
    print(f"[INFO] dtype: {dtype}")
    print(f"[INFO] model_path: {args.model_path}")
    print(f"[INFO] quant_checkpoint: {args.quant_checkpoint}")
    print(f"[INFO] lang: {args.lang}")
    print(f"[INFO] answer_scoring_mode: {args.answer_scoring_mode}")
    print(f"[INFO] device: {device}")

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
                "quant_checkpoint": args.quant_checkpoint,
                "answer_scoring_mode": args.answer_scoring_mode,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            print(
                f"[{idx}/{len(examples)}] {example['example_id']} "
                f"pred={pred_answer or '?'} gold={example['gold_answer']}"
            )

    print(f"[DONE] Saved predictions to {output_jsonl}")


if __name__ == "__main__":
    main()
