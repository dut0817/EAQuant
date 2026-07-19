import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


VALID_ANSWERS = {"A", "B", "C", "D", "E"}
ANSWER_KEY_MAP = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MedExpQA prediction jsonl files with answer accuracy."
    )
    parser.add_argument(
        "input_jsonls",
        nargs="+",
        help="One or more prediction jsonl files produced by run_inference.py",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save evaluation summary as JSON.",
    )
    return parser.parse_args()


def normalize_answer_letter(value: str) -> str:
    if value is None:
        return ""

    text = str(value).strip().upper()
    if text in ANSWER_KEY_MAP:
        return ANSWER_KEY_MAP[text]
    if text in VALID_ANSWERS:
        return text

    match = re.search(r"\b([1-5]|[ABCDE])\b", text)
    if match:
        matched = match.group(1)
        return ANSWER_KEY_MAP.get(matched, matched)

    return ""


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_idx} of {path}: {exc}") from exc
    return rows


def evaluate_file(path: str) -> Dict:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")

    total = 0
    correct = 0
    missing_pred_answer = 0
    invalid_pred_answer = 0

    question_type_total = defaultdict(int)
    question_type_correct = defaultdict(int)

    for row in rows:
        gold = normalize_answer_letter(row.get("gold_answer", ""))
        pred = normalize_answer_letter(row.get("pred_answer", ""))
        question_type = row.get("question_type", "unknown")

        total += 1
        question_type_total[question_type] += 1

        if not pred:
            missing_pred_answer += 1
        elif pred not in VALID_ANSWERS:
            invalid_pred_answer += 1

        is_correct = pred == gold and bool(gold)
        if is_correct:
            correct += 1
            question_type_correct[question_type] += 1

    question_type_accuracy = {}
    for question_type in sorted(question_type_total):
        sp_total = question_type_total[question_type]
        sp_correct = question_type_correct[question_type]
        question_type_accuracy[question_type] = {
            "num_examples": sp_total,
            "num_correct": sp_correct,
            "accuracy": sp_correct / sp_total if sp_total else 0.0,
        }

    first = rows[0]
    summary = {
        "prediction_file": path,
        "file_name": Path(path).name,
        "num_examples": total,
        "num_correct": correct,
        "accuracy": correct / total if total else 0.0,
        "missing_pred_answer": missing_pred_answer,
        "invalid_pred_answer": invalid_pred_answer,
        "system": first.get("system"),
        "model_path": first.get("model_path"),
        "load_quant": first.get("quant_checkpoint") or first.get("load_quant"),
        "split": first.get("split"),
        "question_type_accuracy": question_type_accuracy,
    }
    return summary


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_summary(summary: Dict) -> None:
    print(f"File: {summary['prediction_file']}")
    print(f"System: {summary.get('system')}")
    print(f"Model: {summary.get('model_path')}")
    print(f"Quantized checkpoint: {summary.get('load_quant')}")
    print(
        "Overall: "
        f"{summary['num_correct']}/{summary['num_examples']} "
        f"({format_pct(summary['accuracy'])})"
    )
    print(
        "Pred answer issues: "
        f"missing={summary['missing_pred_answer']}, "
        f"invalid={summary['invalid_pred_answer']}"
    )
    print("Per question_type:")
    for question_type, stats in summary["question_type_accuracy"].items():
        print(
            f"  {question_type}: {stats['num_correct']}/{stats['num_examples']} "
            f"({format_pct(stats['accuracy'])})"
        )
    print()


def main() -> None:
    args = parse_args()
    summaries = [evaluate_file(path) for path in args.input_jsonls]

    for summary in summaries:
        print_summary(summary)

    if len(summaries) > 1:
        print("Macro comparison:")
        for summary in summaries:
            print(
                f"  {summary['file_name']}: "
                f"{format_pct(summary['accuracy'])} "
                f"({summary['num_correct']}/{summary['num_examples']})"
            )
        print()

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        payload = summaries[0] if len(summaries) == 1 else summaries
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved summary JSON to {args.output_json}")


if __name__ == "__main__":
    main()
