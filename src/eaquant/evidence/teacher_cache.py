import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
REPO_DIR = PROJECT_ROOT
DATA_BASE_DIR = Path(os.environ.get("EAQUANT_DATA_ROOT", PROJECT_ROOT))

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eaquant.evidence.schema import (  # noqa: E402
    ANSWER_SCORING_MODES,
    SPAN_GRANULARITIES,
    SPAN_SCORE_MODES,
    build_faithfulness_training_fields,
    build_explanation_prompt,
    build_messages,
    build_teacher_prediction_record,
    generate_from_messages,
    get_input_device,
    load_medmix_faithfulness_examples,
    score_answer_options,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fp16 teacher inference over medmix train examples and save a "
            "faithfulness cache for later span mining."
        )
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local Hugging Face model path for the original fp16 teacher.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DATA_BASE_DIR,
        help="Directory containing the med_datasets/ folder.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Optional output jsonl path. Defaults under cache/med_faithfulness/.",
    )
    parser.add_argument(
        "--source_filters",
        nargs="*",
        default=None,
        help="Optional medmix source filter, e.g. 'MedExpQA train' 'ChallengeClinical op4'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional example limit after loading.",
    )
    parser.add_argument(
        "--target_written",
        type=int,
        default=None,
        help=(
            "Optional number of records to write before stopping. This is useful "
            "with --correct_only when the desired cache size is the number of "
            "teacher-correct rows rather than the number of examples seen."
        ),
    )
    parser.add_argument(
        "--medmix_target_records",
        type=int,
        default=128,
        help="Number of medmix source examples to select before optional --limit.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=220,
        help="Maximum generated tokens for teacher rationales.",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=32,
        help="Minimum generated tokens for teacher rationales.",
    )
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default="fp16",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Compute dtype for teacher inference.",
    )
    parser.add_argument(
        "--answer_scoring_mode",
        type=str,
        default="letter_option_mean_logprob",
        choices=list(ANSWER_SCORING_MODES),
        help=(
            "How answer options are scored for teacher answer selection and "
            "selected-span recovery."
        ),
    )
    parser.add_argument(
        "--correct_only",
        action="store_true",
        help="If set, only write teacher-correct examples to the output jsonl.",
    )
    parser.add_argument(
        "--top_k_spans",
        type=int,
        default=4,
        help="Maximum number of cumulative ranked spans to try for selected-span recovery.",
    )
    parser.add_argument(
        "--span_granularity",
        type=str,
        default="sentence",
        choices=list(SPAN_GRANULARITIES),
        help="Span granularity used to mine faithful rationale supervision.",
    )
    parser.add_argument(
        "--span_score_mode",
        type=str,
        default="margin_drop",
        choices=list(SPAN_SCORE_MODES),
        help="How faithful spans are ranked: target-answer logprob drop or option-margin drop.",
    )
    parser.add_argument(
        "--min_span_words",
        type=int,
        default=6,
        help="Minimum target size for each mined explanation span.",
    )
    parser.add_argument(
        "--max_span_words",
        type=int,
        default=24,
        help="Maximum target size for each mined explanation span before long-sentence chunking.",
    )
    parser.add_argument(
        "--min_selected_words",
        type=int,
        default=12,
        help="Minimum total word count required before a selected-rationale set can be accepted.",
    )
    parser.add_argument(
        "--mask_replacement_text",
        type=str,
        default="[omitted rationale span]",
        help="Neutral replacement text used when ablating a selected span.",
    )
    parser.add_argument(
        "--sufficiency_weight",
        type=float,
        default=0.25,
        help="Relative weight for span-only sufficiency when ranking faithful spans.",
    )
    parser.add_argument(
        "--length_norm_alpha",
        type=float,
        default=0.5,
        help="Length normalization exponent used when ranking faithful spans.",
    )
    parser.add_argument(
        "--score_batch_size",
        type=int,
        default=1,
        help="Micro-batch size used for rationale-recovery scoring prompts.",
    )
    return parser.parse_args()


def _resolve_dtype(args: argparse.Namespace) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if args.compute_dtype == "fp16":
        return torch.float16
    if args.compute_dtype == "bf16":
        return torch.bfloat16
    if args.compute_dtype == "fp32":
        return torch.float32
    return torch.float16


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


def _load_model_and_tokenizer(
    model_path: str,
    dtype: torch.dtype,
) -> Tuple[torch.nn.Module, object]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    tokenizer = _load_tokenizer_for_model(model_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _default_output_path(model_path: str, span_granularity: str) -> Path:
    model_name = Path(model_path).name.lower().replace("/", "_")
    output_dir = REPO_DIR / "cache" / "med_faithfulness" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"medmix_train_teacher_predictions_{span_granularity}.jsonl"


def _summary_from_records(
    total_seen: int,
    total_written: int,
    total_correct: int,
    total_with_spans: int,
    source_totals: Dict[str, Dict[str, int]],
    args: argparse.Namespace,
) -> Dict:
    return {
        "model_path": args.model_path,
        "compute_dtype": args.compute_dtype,
        "answer_scoring_mode": args.answer_scoring_mode,
        "span_granularity": args.span_granularity,
        "span_score_mode": args.span_score_mode,
        "correct_only": args.correct_only,
        "top_k_spans": args.top_k_spans,
        "min_span_words": args.min_span_words,
        "max_span_words": args.max_span_words,
        "min_selected_words": args.min_selected_words,
        "source_filters": list(args.source_filters or []),
        "medmix_target_records": args.medmix_target_records,
        "target_written": args.target_written,
        "total_examples_seen": total_seen,
        "total_examples_written": total_written,
        "total_correct": total_correct,
        "total_with_selected_spans": total_with_spans,
        "accuracy_over_seen": (float(total_correct) / float(total_seen)) if total_seen else 0.0,
        "per_source": source_totals,
    }


def main() -> None:
    args = parse_args()
    dtype = _resolve_dtype(args)
    output_jsonl = (
        Path(args.output_jsonl)
        if args.output_jsonl
        else _default_output_path(args.model_path, args.span_granularity)
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    examples = load_medmix_faithfulness_examples(
        base_dir=args.data_root,
        source_filters=args.source_filters,
        limit=args.limit,
        medmix_target_records=args.medmix_target_records,
    )
    if not examples:
        raise ValueError("No medmix examples matched the requested filters.")

    model, tokenizer = _load_model_and_tokenizer(args.model_path, dtype)
    device = get_input_device(model)

    print(f"[INFO] examples: {len(examples)}")
    print(f"[INFO] output: {output_jsonl}")
    print(f"[INFO] dtype: {dtype}")
    print(f"[INFO] model: {args.model_path}")
    print(f"[INFO] data_base_dir: {args.data_root}")

    total_written = 0
    total_correct = 0
    total_with_spans = 0
    total_seen = 0
    source_totals: Dict[str, Dict[str, int]] = {}

    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for idx, example in enumerate(examples, start=1):
            total_seen = idx
            pred_answer, answer_scores = score_answer_options(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                dtype=dtype,
                answer_scoring_mode=args.answer_scoring_mode,
                batch_size=args.score_batch_size,
            )
            pred_answer_text = example["options"].get(pred_answer, "")
            raw_generation = generate_from_messages(
                model=model,
                tokenizer=tokenizer,
                messages=build_messages(
                    build_explanation_prompt(example, pred_answer, pred_answer_text)
                ),
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                device=device,
                dtype=dtype,
            )
            record = build_teacher_prediction_record(
                example=example,
                pred_answer=pred_answer,
                answer_scores=answer_scores,
                raw_generation=raw_generation,
                model_path=args.model_path,
            )
            record["answer_scoring_mode"] = args.answer_scoring_mode
            record["span_granularity"] = args.span_granularity
            record["span_score_mode"] = args.span_score_mode
            if record["is_correct"]:
                record.update(
                    build_faithfulness_training_fields(
                        model=model,
                        tokenizer=tokenizer,
                        example=example,
                        pred_answer=pred_answer,
                        pred_answer_text=pred_answer_text,
                        pred_explanation_stripped=record["pred_explanation_stripped"],
                        device=device,
                        dtype=dtype,
                        top_k_spans=args.top_k_spans,
                        span_granularity=args.span_granularity,
                        min_span_words=args.min_span_words,
                        max_span_words=args.max_span_words,
                        min_selected_words=args.min_selected_words,
                        mask_replacement_text=args.mask_replacement_text,
                        sufficiency_weight=args.sufficiency_weight,
                        length_norm_alpha=args.length_norm_alpha,
                        score_batch_size=args.score_batch_size,
                        answer_scoring_mode=args.answer_scoring_mode,
                        span_score_mode=args.span_score_mode,
                    )
                )

            source_name = str(record["source"])
            source_stats = source_totals.setdefault(
                source_name,
                {"seen": 0, "written": 0, "correct": 0},
            )
            source_stats["seen"] += 1

            if record["is_correct"]:
                total_correct += 1
                source_stats["correct"] += 1
            if record.get("faith_selected_span_count", 0) > 0:
                total_with_spans += 1

            should_write = bool(record["is_correct"]) if args.correct_only else True
            if should_write:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1
                source_stats["written"] += 1

            print(
                f"[{idx}/{len(examples)}] {record['example_id']} "
                f"source={source_name} pred={record['pred_answer']} gold={record['gold_answer']} "
                f"correct={record['is_correct']}"
            )
            if args.target_written is not None and total_written >= args.target_written:
                print(
                    f"[INFO] target_written reached: {total_written}/{args.target_written}",
                    flush=True,
                )
                break

    summary = _summary_from_records(
        total_seen=total_seen,
        total_written=total_written,
        total_correct=total_correct,
        total_with_spans=total_with_spans,
        source_totals=source_totals,
        args=args,
    )
    summary_path = output_jsonl.with_name(output_jsonl.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved teacher cache to {output_jsonl}")
    print(f"[DONE] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
