import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
OMNIQUANT_REPO_DIR = SCRIPT_DIR
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

for path in (PROJECT_ROOT / "src", OMNIQUANT_REPO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eaquant.evaluation.inference import (  # noqa: E402
    _clean_reference_text,
    _ensure_rmsnorm_compat,
    _normalize_answer_letter,
    _safe_text,
    load_medexpqa_examples,
    run_one,
)


DEFAULT_CHALLENGECLINICAL_FILE = None


class PrintLogger:
    def info(self, msg, *args):
        if args:
            msg = msg % args
        print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MedExpQA inference with an OmniQuant fake-quantized checkpoint."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--system_name",
        choices=("medmix_baseline", "eaquant"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument(
        "--omni_parameters",
        "--quant_checkpoint",
        dest="omni_parameters",
        type=str,
        required=True,
        help="Path to OmniQuant omni_parameters.pth.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="medexpqa",
        choices=["medexpqa", "medexqa", "challengeclinical"],
        help="Dataset loader to use.",
    )
    parser.add_argument(
        "--challengeclinical_file",
        type=str,
        default=DEFAULT_CHALLENGECLINICAL_FILE,
        help="ChallengeClinical/MedBullets JSON file used when dataset_name=challengeclinical.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "op4", "op5", "all"],
    )
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument(
        "--artifact_subdir",
        type=str,
        default=None,
        help="Optional output subdirectory overriding the default system name.",
    )
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--question_types", nargs="*", default=None)
    parser.add_argument("--subsets", nargs="*", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--min_new_tokens", type=int, default=32)
    parser.add_argument("--answer_retry_max_new_tokens", type=int, default=8)
    parser.add_argument(
        "--answer_scoring_mode",
        type=str,
        default="letter_option_mean_logprob",
        choices=["single_letter", "letter_option_mean_logprob"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compute_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--cache_dir", type=str, default=str(OMNIQUANT_REPO_DIR / "cache"))
    parser.add_argument("--calib_dataset", type=str, default="medmix")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--let", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--lwc", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--aug_loss", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--symmetric", default=False, action="store_true")
    parser.add_argument("--disable_zero_point", default=False, action="store_true")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--act-scales", dest="act_scales", default=None)
    parser.add_argument("--act-shifts", dest="act_shifts", default=None)
    args = parser.parse_args()
    if args.act_scales is None:
        args.act_scales = str(
            OMNIQUANT_REPO_DIR / "act_scales" / f"medmix_seq512_seed{args.seed}" / f"{args.net}.pt"
        )
    if args.act_shifts is None:
        args.act_shifts = str(
            OMNIQUANT_REPO_DIR / "act_shifts" / f"medmix_seq512_seed{args.seed}" / f"{args.net}.pt"
        )
    return args


def _resolve_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.compute_dtype == "fp16":
        return torch.float16
    if args.compute_dtype == "bf16":
        return torch.bfloat16
    if args.compute_dtype == "fp32":
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    return torch.float16


def _build_output_path(args: argparse.Namespace) -> str:
    if args.output_jsonl:
        return args.output_jsonl
    ckpt_name = Path(args.omni_parameters).parent.name.lower()
    dataset_dir = args.dataset_name
    output_dir = OUTPUT_ROOT / dataset_dir / "omniquant"
    output_subdir = args.artifact_subdir or args.system_name
    if output_subdir:
        output_dir = output_dir / output_subdir
    return str(output_dir / f"{args.split}_{ckpt_name}_predictions.jsonl")


def load_medexqa_examples(
    data_root: str,
    split: str = "test",
    specialties: list[str] | None = None,
) -> list[dict]:
    data_root_path = Path(data_root)
    split_dir = data_root_path / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"MedExQA split directory not found: {split_dir}")

    specialty_filter = {value.strip() for value in (specialties or []) if value.strip()}
    examples: list[dict] = []
    for path in sorted(split_dir.glob(f"*_{split}.tsv")):
        specialty = path.name[: -len(f"_{split}.tsv")]
        if specialty_filter and specialty not in specialty_filter:
            continue

        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="	")
            for row_idx, row in enumerate(reader, start=1):
                if len(row) < 8:
                    continue
                question, option_a, option_b, option_c, option_d, exp1, exp2, gold = row[:8]
                options = {
                    "A": _safe_text(option_a),
                    "B": _safe_text(option_b),
                    "C": _safe_text(option_c),
                    "D": _safe_text(option_d),
                }
                options = {label: text for label, text in options.items() if text}
                gold_explanation_1 = _clean_reference_text(exp1)
                gold_explanation_2 = _clean_reference_text(exp2 or exp1)
                if not gold_explanation_1 and gold_explanation_2:
                    gold_explanation_1 = gold_explanation_2
                if not gold_explanation_2 and gold_explanation_1:
                    gold_explanation_2 = gold_explanation_1
                example_id = f"{specialty}:{row_idx:04d}"
                gold_answer = _normalize_answer_letter(gold)
                examples.append(
                    {
                        "example_id": example_id,
                        "split": split,
                        "question_type": specialty,
                        "specialty": specialty,
                        "subset": specialty,
                        "source_lang": "en",
                        "source_file": str(path),
                        "row_idx": row_idx,
                        "question": _safe_text(question),
                        "options": options,
                        "gold_explanation_1": gold_explanation_1,
                        "gold_explanation_2": gold_explanation_2,
                        "gold_answer": gold_answer,
                        "gold_answer_text": options.get(gold_answer, ""),
                        "dataset_id": example_id,
                        "question_id_specific": "",
                        "year": "",
                        "num_options": len(options),
                        "source_dataset": "MedExQA",
                        "link": "",
                    }
                )

    return examples


def load_challengeclinical_examples(
    json_path: str,
    split: str = "op5",
    subsets: list[str] | None = None,
) -> list[dict]:
    data_path = Path(json_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"ChallengeClinical file not found: {data_path}")

    subset_filter = {value.strip() for value in (subsets or []) if value.strip()}
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {data_path}")

    examples: list[dict] = []
    for row_idx, ex in enumerate(data, start=1):
        if not isinstance(ex, dict):
            continue

        subset = _safe_text(ex.get("subset")) or _safe_text(ex.get("question_type")) or split
        if subset_filter and subset not in subset_filter:
            continue

        raw_options = ex.get("options") or {}
        if isinstance(raw_options, list):
            options = {
                _safe_text(option.get("label", "")): _safe_text(option.get("text", ""))
                for option in raw_options
                if isinstance(option, dict)
            }
        else:
            options = {
                _safe_text(label): _safe_text(text)
                for label, text in raw_options.items()
            }
        options = {label: text for label, text in options.items() if label and text}
        if not options:
            continue

        gold_explanation_1 = _clean_reference_text(
            ex.get("gold_explanation_1") or ex.get("explanation") or ""
        )
        gold_explanation_2 = _clean_reference_text(
            ex.get("gold_explanation_2") or gold_explanation_1
        )
        if not gold_explanation_1 and gold_explanation_2:
            gold_explanation_1 = gold_explanation_2
        if not gold_explanation_2 and gold_explanation_1:
            gold_explanation_2 = gold_explanation_1

        dataset_id = _safe_text(ex.get("example_id")) or f"challengeclinical_{split}_{row_idx:05d}"
        examples.append(
            {
                "example_id": dataset_id,
                "split": split,
                "question_type": subset,
                "subset": subset,
                "source_lang": "en",
                "source_file": str(data_path),
                "row_idx": row_idx,
                "question": _safe_text(ex.get("question")),
                "options": options,
                "gold_explanation_1": gold_explanation_1,
                "gold_explanation_2": gold_explanation_2,
                "gold_answer": _normalize_answer_letter(
                    ex.get("gold_answer") or ex.get("answer_idx") or ex.get("correct_option")
                ),
                "gold_answer_text": _safe_text(ex.get("gold_answer_text")),
                "dataset_id": dataset_id,
                "question_id_specific": "",
                "year": "",
                "num_options": int(ex.get("num_options", len(options))),
                "source_dataset": _safe_text(ex.get("source_dataset")) or "medbullets",
                "link": _safe_text(ex.get("link")),
            }
        )

    return examples


def _make_omniquant_args(args: argparse.Namespace) -> SimpleNamespace:
    quant_args = SimpleNamespace(**vars(args))
    quant_args.model = args.model_path
    quant_args.resume = args.omni_parameters
    quant_args.output_dir = str(
        OUTPUT_ROOT / "runtime" / "omniquant" / Path(args.omni_parameters).parent.name
    )
    quant_args.save_dir = None
    quant_args.real_quant = False
    quant_args.epochs = 0
    quant_args.deactive_amp = False
    quant_args.multigpu = False
    quant_args.eval_ppl = False
    quant_args.tasks = ""
    quant_args.num_fewshot = 0
    quant_args.limit = -1
    quant_args.explanation_loss_enabled = False
    quant_args.faithfulness_cache_path = ""
    quant_args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc": args.lwc,
        "disable_zero_point": args.disable_zero_point,
    }
    quant_args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    quant_args.q_quant_params = dict(quant_args.act_quant_params)
    quant_args.k_quant_params = dict(quant_args.act_quant_params)
    quant_args.v_quant_params = dict(quant_args.act_quant_params)
    quant_args.p_quant_params = {"n_bits": 16, "metric": "fix0to1"}
    return quant_args


def _load_omniquant_model_and_tokenizer(args: argparse.Namespace, dtype: torch.dtype):
    _ensure_rmsnorm_compat()
    checkpoint_path = Path(args.omni_parameters)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"OmniQuant parameter file not found: {checkpoint_path}")
    if args.let:
        if not Path(args.act_scales).is_file():
            raise FileNotFoundError(f"Activation scales not found: {args.act_scales}")
        if not Path(args.act_shifts).is_file():
            raise FileNotFoundError(f"Activation shifts not found: {args.act_shifts}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    from datautils import get_loaders
    from models.LMClass import LMClass
    from quantize.omniquant import omniquant

    quant_args = _make_omniquant_args(args)
    Path(quant_args.output_dir).mkdir(parents=True, exist_ok=True)
    lm = LMClass(quant_args)
    lm.seqlen = args.seqlen
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    dataloader, _ = get_loaders(
        args.calib_dataset,
        nsamples=max(int(args.nsamples), 1),
        seed=args.seed,
        model=args.model_path,
        seqlen=args.seqlen,
    )
    act_scales = torch.load(args.act_scales) if args.let else None
    act_shifts = torch.load(args.act_shifts) if args.let else None
    omniquant(
        lm=lm,
        args=quant_args,
        dataloader=dataloader,
        act_scales=act_scales,
        act_shifts=act_shifts,
        logger=PrintLogger(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm.model.to(device)
    lm.model.eval()
    lm.model.config.use_cache = True
    if hasattr(lm.model, "generation_config"):
        lm.model.generation_config.use_cache = True
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

    if args.dataset_name == "challengeclinical":
        examples = load_challengeclinical_examples(
            args.challengeclinical_file,
            split=args.split,
            subsets=args.subsets,
        )
        empty_message = "No ChallengeClinical examples matched the requested split/filter."
    elif args.dataset_name == "medexqa":
        examples = load_medexqa_examples(
            args.data_root,
            split=args.split,
            specialties=args.subsets,
        )
        empty_message = "No MedExQA examples matched the requested split/filter."
    else:
        examples = load_medexpqa_examples(
            args.data_root,
            args.split,
            lang=args.lang,
            question_types=args.question_types,
        )
        empty_message = "No MedExpQA examples matched the requested split/filter."
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError(empty_message)

    model, tokenizer, device = _load_omniquant_model_and_tokenizer(args, dtype)

    print(f"[INFO] examples: {len(examples)}")
    print(f"[INFO] output: {output_jsonl}")
    print(f"[INFO] dtype: {dtype}")
    print(f"[INFO] model_path: {args.model_path}")
    print(f"[INFO] net: {args.net}")
    print(f"[INFO] omni_parameters: {args.omni_parameters}")
    print(f"[INFO] dataset_name: {args.dataset_name}")
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
                "gold_answer_text": example.get("gold_answer_text", ""),
                "dataset_id": example.get("dataset_id", ""),
                "question_id_specific": example.get("question_id_specific", ""),
                "year": example.get("year", ""),
                "subset": example.get("subset", ""),
                "num_options": example.get("num_options", len(example["options"])),
                "source_dataset": example.get("source_dataset", ""),
                "link": example.get("link", ""),
                "specialty": example.get("specialty", ""),
                "pred_answer": pred_answer,
                "pred_explanation": pred_explanation,
                "raw_generation": raw_generation,
                "is_correct": pred_answer == example["gold_answer"],
                "system": args.system_name,
                "dataset_name": args.dataset_name,
                "model_path": args.model_path,
                "net": args.net,
                "omni_parameters": args.omni_parameters,
                "answer_scoring_mode": args.answer_scoring_mode,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            print(
                f"[{idx}/{len(examples)}] {example['example_id']} "
                f"pred={pred_answer or '?'} gold={example['gold_answer']}",
                flush=True,
            )

    print(f"[DONE] Saved predictions to {output_jsonl}")


if __name__ == "__main__":
    main()
