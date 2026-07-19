import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .judge_common import (
    DATA_DIR as MEDEXPQA_DATA_DIR,
    DEFAULT_SEEDS,
    MODEL_KEYS,
    build_index,
    extract_gold_answer,
    extract_pred_answer,
    get_seed_suffix,
    normalize_text,
    read_jsonl,
    record_key,
    write_json,
)


DATA_ROOT = MEDEXPQA_DATA_DIR.parent
DATASET_CONFIGS = {
    "medexpqa": {
        "display": "MedExpQA",
        "data_subdir": "medexpqa",
        "prediction_prefix": "test",
        "llama3_eaquant_v4": True,
    },
    "medexqa": {
        "display": "MedExQA",
        "data_subdir": "medexqa",
        "prediction_prefix": "test",
        "llama3_eaquant_v4": False,
    },
    "challengeclinical": {
        "display": "ChallengeClinicalQA",
        "data_subdir": "challengeclinical",
        "prediction_prefix": "op5",
        "llama3_eaquant_v4": False,
    },
}
VALID_ANSWERS = {"A", "B", "C", "D", "E"}
SYSTEM_ORDER = ("fp16", "medmix_baseline", "eaquant")
SYSTEM_DISPLAY = {
    "fp16": "FP16",
    "medmix_baseline": "MedMix PTQ",
    "eaquant": "EAQuant",
}


def dataset_data_dir(dataset: str) -> Path:
    return DATA_ROOT / str(DATASET_CONFIGS[dataset]["data_subdir"])


def dataset_output_dir(dataset: str) -> Path:
    return dataset_data_dir(dataset) / "analysis" / "fp16_answer_agreement"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute answer accuracy and FP16 answer agreement for FP16, "
            "standard MedMix PTQ, and EAQuant outputs."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        default="medexpqa",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--fp16_dir",
        type=Path,
        default=None,
        help="Directory containing original train_baseline prediction JSONL files.",
    )
    parser.add_argument(
        "--medmix_baseline_dir",
        type=Path,
        default=None,
        help="Directory containing standard MedMix PTQ prediction files.",
    )
    parser.add_argument(
        "--eaquant_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing EAQuant prediction files. Defaults to files without "
            "tok/w ablation tags, i.e. qwen_imp0p02_q0p02_beta0p01_seed*."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory where JSON/CSV/LaTeX summaries will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KEYS,
        default=list(MODEL_KEYS),
        help="Model keys to include.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Quantized seeds to include. Baseline seed 0 maps to the no-suffix file.",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip missing prediction files instead of raising an error.",
    )
    args = parser.parse_args()
    data_dir = dataset_data_dir(args.dataset)
    if args.fp16_dir is None:
        args.fp16_dir = data_dir / "train_baseline"
    if args.medmix_baseline_dir is None:
        args.medmix_baseline_dir = data_dir / "train_quantized_medmix"
    if args.eaquant_dir is None:
        args.eaquant_dir = data_dir / "llm"
    if args.output_dir is None:
        args.output_dir = dataset_output_dir(args.dataset)
    args.dataset_display = str(DATASET_CONFIGS[args.dataset]["display"])
    return args


def safe_rate(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def pct(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}"


def is_valid_answer(answer: str) -> bool:
    return bool(answer) and answer in VALID_ANSWERS


LLAMA3_EAQUANT_V4_DIR = Path("ablation") / "qwen_component"
LLAMA3_EAQUANT_V4_TAG = (
    "ostquant_w4a4kv4_llm_ab_v4_teacher_faithful_qdamage_keepmargin_"
    "imp0p02_q0p02_beta0p01_tok01_rec01"
)
LLAMA3_LEGACY_PAPER_V5_TAG = (
    "ostquant_w4a4kv4_llm_ab_v5_teacher_faithful_qdamage_keepmargin_"
    "imp0p02_q0p02_beta0p01_tok01_rec01"
)


def prediction_prefix(dataset: str) -> str:
    return str(DATASET_CONFIGS[dataset]["prediction_prefix"])


def fp16_prediction_path(fp16_dir: Path, dataset: str, model_key: str) -> Path:
    prefix = prediction_prefix(dataset)
    return fp16_dir / f"{prefix}_{model_key}_original_train_baseline_predictions.jsonl"


def medmix_baseline_prediction_path(
    medmix_baseline_dir: Path, dataset: str, model_key: str, seed: int
) -> Path:
    prefix = prediction_prefix(dataset)
    suffix = get_seed_suffix(seed)
    return (
        medmix_baseline_dir
        / f"{prefix}_{model_key}_ostquant_w4a4kv4_train_quantized_medmix{suffix}_predictions.jsonl"
    )


def uses_llama3_eaquant_v4(dataset: str, model_key: str) -> bool:
    return bool(DATASET_CONFIGS[dataset].get("llama3_eaquant_v4")) and model_key == "llama3_instruct"


def prefer_existing_path(primary: Path, legacy: Path) -> Path:
    """Prefer the public V4 name, while reading retained paper V5 artifacts."""
    if primary.exists() or not legacy.exists():
        return primary
    return legacy


def eaquant_prediction_path(
    eaquant_dir: Path, dataset: str, model_key: str, seed: int
) -> Path:
    prefix = prediction_prefix(dataset)
    if uses_llama3_eaquant_v4(dataset, model_key):
        public_v4 = (
            eaquant_dir
            / LLAMA3_EAQUANT_V4_DIR
            / f"{prefix}_llama3_instruct_{LLAMA3_EAQUANT_V4_TAG}_seed{seed}_predictions.jsonl"
        )
        legacy_v5 = (
            eaquant_dir
            / LLAMA3_EAQUANT_V4_DIR
            / f"{prefix}_llama3_instruct_{LLAMA3_LEGACY_PAPER_V5_TAG}_seed{seed}_predictions.jsonl"
        )
        return prefer_existing_path(public_v4, legacy_v5)
    llm_model_key = "openbiollm_8b" if model_key == "openbiollm" else model_key
    return (
        eaquant_dir
        / f"{prefix}_{llm_model_key}_ostquant_w4a4kv4_llm_qwen_imp0p02_q0p02_beta0p01_seed{seed}_predictions.jsonl"
    )


def eaquant_label_recovery_summary_path(
    eaquant_dir: Path, dataset: str, model_key: str, seed: int
) -> Path:
    if uses_llama3_eaquant_v4(dataset, model_key):
        public_v4 = (
            eaquant_dir
            / LLAMA3_EAQUANT_V4_DIR
            / f"label_recovery_llama3_instruct_{LLAMA3_EAQUANT_V4_TAG}_seed{seed}_selfmodel_summary.json"
        )
        legacy_v5 = (
            eaquant_dir
            / LLAMA3_EAQUANT_V4_DIR
            / f"label_recovery_llama3_instruct_{LLAMA3_LEGACY_PAPER_V5_TAG}_seed{seed}_selfmodel_summary.json"
        )
        return prefer_existing_path(public_v4, legacy_v5)
    llm_model_key = "openbiollm_8b" if model_key == "openbiollm" else model_key
    return (
        eaquant_dir
        / f"label_recovery_{llm_model_key}_ostquant_w4a4kv4_llm_qwen_imp0p02_q0p02_beta0p01_seed{seed}_selfmodel_summary.json"
    )


def label_recovery_summary_path(
    *,
    fp16_dir: Path,
    medmix_baseline_dir: Path,
    eaquant_dir: Path,
    system: str,
    dataset: str,
    model_key: str,
    seed: Optional[int],
) -> Path:
    if system == "fp16":
        return fp16_dir / f"label_recovery_{model_key}_original_train_baseline_selfmodel_summary.json"
    if system == "medmix_baseline":
        suffix = get_seed_suffix(int(seed or 0))
        return (
            medmix_baseline_dir
            / f"label_recovery_{model_key}_ostquant_w4a4kv4_train_quantized_medmix{suffix}_selfmodel_summary.json"
        )
    if system == "eaquant":
        return eaquant_label_recovery_summary_path(eaquant_dir, dataset, model_key, int(seed or 0))
    raise ValueError(f"Unsupported system: {system}")


def load_predrec(summary_path: Path) -> Dict[str, Any]:
    if not summary_path.exists():
        return {
            "predrec_summary_file": str(summary_path),
            "predrec_summary_exists": False,
            "predrec_count": None,
            "predrec_denominator": None,
            "predrec_rate": None,
        }

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    views = summary.get("summary_views")
    if isinstance(views, dict):
        view = views.get("source_pred_present_only") or views.get("all_rows") or summary
    else:
        view = summary

    count = view.get("original_prediction_match_count")
    rate = view.get("original_prediction_match_rate")
    denominator = view.get("final_parse_success_count") or view.get("num_rows_evaluated")

    if isinstance(count, int) and isinstance(rate, (float, int)) and rate > 0:
        inferred = round(count / float(rate))
        if inferred > 0:
            denominator = inferred

    if not isinstance(count, int):
        count = None
    if not isinstance(denominator, int):
        denominator = None
    if not isinstance(rate, (float, int)):
        rate = safe_rate(count, denominator)

    return {
        "predrec_summary_file": str(summary_path),
        "predrec_summary_exists": True,
        "predrec_count": count,
        "predrec_denominator": denominator,
        "predrec_rate": float(rate) if isinstance(rate, (float, int)) else None,
    }


def evaluate_fp16(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = 0
    missing_pred = 0
    invalid_pred = 0
    missing_gold = 0
    missing_key = 0

    for record in records:
        if record_key(record) is None:
            missing_key += 1
        pred = extract_pred_answer(record)
        gold = extract_gold_answer(record)
        if not pred:
            missing_pred += 1
        elif pred not in VALID_ANSWERS:
            invalid_pred += 1
        if not gold:
            missing_gold += 1
        if is_valid_answer(pred) and gold and pred == gold:
            correct += 1

    return {
        "num_fp16_rows": total,
        "num_candidate_rows": total,
        "num_matched": total,
        "num_missing_match": 0,
        "num_fp16_rows_missing_key": missing_key,
        "candidate_index_missing_key": 0,
        "candidate_index_duplicate_key": 0,
        "num_missing_pred_answer": missing_pred,
        "num_invalid_pred_answer": invalid_pred,
        "num_missing_gold_answer": missing_gold,
        "num_missing_fp16_pred_answer": missing_pred,
        "num_correct": correct,
        "accuracy_rate": safe_rate(correct, total),
        "fp16_agree_count": total,
        "fp16_agree_rate": 1.0 if total else None,
    }


def evaluate_against_fp16(
    fp16_records: List[Dict[str, Any]],
    candidate_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_index, index_stats = build_index(candidate_records)
    matched = 0
    missing_match = 0
    fp16_missing_key = 0
    missing_pred = 0
    invalid_pred = 0
    missing_gold = 0
    missing_fp16_pred = 0
    correct = 0
    agree = 0

    for fp16_record in fp16_records:
        key = record_key(fp16_record)
        if key is None:
            fp16_missing_key += 1
            continue

        candidate_record = candidate_index.get(key)
        if candidate_record is None:
            missing_match += 1
            continue

        matched += 1
        pred = extract_pred_answer(candidate_record)
        fp16_pred = extract_pred_answer(fp16_record)
        gold = extract_gold_answer(candidate_record) or extract_gold_answer(fp16_record)

        if not pred:
            missing_pred += 1
        elif pred not in VALID_ANSWERS:
            invalid_pred += 1
        if not gold:
            missing_gold += 1
        if not fp16_pred:
            missing_fp16_pred += 1

        if is_valid_answer(pred) and gold and pred == gold:
            correct += 1
        if is_valid_answer(pred) and fp16_pred and pred == fp16_pred:
            agree += 1

    return {
        "num_fp16_rows": len(fp16_records),
        "num_candidate_rows": len(candidate_records),
        "num_matched": matched,
        "num_missing_match": missing_match,
        "num_fp16_rows_missing_key": fp16_missing_key,
        "candidate_index_missing_key": index_stats["missing_key"],
        "candidate_index_duplicate_key": index_stats["duplicate_key"],
        "num_missing_pred_answer": missing_pred,
        "num_invalid_pred_answer": invalid_pred,
        "num_missing_gold_answer": missing_gold,
        "num_missing_fp16_pred_answer": missing_fp16_pred,
        "num_correct": correct,
        "accuracy_rate": safe_rate(correct, matched),
        "fp16_agree_count": agree,
        "fp16_agree_rate": safe_rate(agree, matched),
    }


def make_row(
    *,
    system: str,
    dataset: str,
    model_key: str,
    seed: Optional[int],
    prediction_file: Path,
    fp16_file: Path,
    metrics: Dict[str, Any],
    predrec: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "dataset": dataset,
        "model_key": model_key,
        "system": system,
        "system_display": SYSTEM_DISPLAY[system],
        "seed": seed,
        "prediction_file": str(prediction_file),
        "fp16_file": str(fp16_file),
    }
    row.update(metrics)
    row.update(predrec)
    return row


def iter_prediction_jobs(args: argparse.Namespace) -> Iterable[Tuple[str, str, Optional[int], Path]]:
    for model_key in args.models:
        yield model_key, "fp16", None, fp16_prediction_path(args.fp16_dir, args.dataset, model_key)
        for seed in args.seeds:
            yield model_key, "medmix_baseline", seed, medmix_baseline_prediction_path(args.medmix_baseline_dir, args.dataset, model_key, seed)
            yield model_key, "eaquant", seed, eaquant_prediction_path(args.eaquant_dir, args.dataset, model_key, seed)


def collect_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    fp16_records_by_model: Dict[str, List[Dict[str, Any]]] = {}
    rows: List[Dict[str, Any]] = []

    for model_key, system, seed, prediction_file in iter_prediction_jobs(args):
        if not prediction_file.exists():
            if args.skip_missing:
                print(f"[WARN] skipping missing file: {prediction_file}")
                continue
            raise FileNotFoundError(prediction_file)

        fp16_file = fp16_prediction_path(args.fp16_dir, args.dataset, model_key)
        if model_key not in fp16_records_by_model:
            if not fp16_file.exists():
                if args.skip_missing:
                    print(f"[WARN] skipping model with missing FP16 file: {fp16_file}")
                    continue
                raise FileNotFoundError(fp16_file)
            fp16_records_by_model[model_key] = read_jsonl(fp16_file)

        prediction_records = (
            fp16_records_by_model[model_key]
            if system == "fp16"
            else read_jsonl(prediction_file)
        )
        metrics = (
            evaluate_fp16(prediction_records)
            if system == "fp16"
            else evaluate_against_fp16(fp16_records_by_model[model_key], prediction_records)
        )
        predrec = load_predrec(
            label_recovery_summary_path(
                fp16_dir=args.fp16_dir,
                medmix_baseline_dir=args.medmix_baseline_dir,
                eaquant_dir=args.eaquant_dir,
                system=system,
                dataset=args.dataset,
                model_key=model_key,
                seed=seed,
            )
        )
        rows.append(
            make_row(
                system=system,
                dataset=args.dataset,
                model_key=model_key,
                seed=seed,
                prediction_file=prediction_file,
                fp16_file=fp16_file,
                metrics=metrics,
                predrec=predrec,
            )
        )

    return rows


def aggregate_rows(rows: List[Dict[str, Any]], group_keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)

    aggregates: List[Dict[str, Any]] = []
    for key_values, group in grouped.items():
        aggregate = {key: value for key, value in zip(group_keys, key_values)}
        aggregate["system_display"] = SYSTEM_DISPLAY.get(str(aggregate.get("system")), "")
        aggregate["num_runs"] = len(group)

        matched = sum(int(row.get("num_matched") or 0) for row in group)
        correct = sum(int(row.get("num_correct") or 0) for row in group)
        agree = sum(int(row.get("fp16_agree_count") or 0) for row in group)
        predrec_rows = [
            row
            for row in group
            if row.get("predrec_count") is not None
            and row.get("predrec_denominator") is not None
        ]
        predrec_count = sum(int(row["predrec_count"]) for row in predrec_rows)
        predrec_denominator = sum(int(row["predrec_denominator"]) for row in predrec_rows)

        aggregate.update(
            {
                "num_matched": matched,
                "num_correct": correct,
                "accuracy_rate": safe_rate(correct, matched),
                "fp16_agree_count": agree,
                "fp16_agree_rate": safe_rate(agree, matched),
                "predrec_count": predrec_count if predrec_rows else None,
                "predrec_denominator": predrec_denominator if predrec_rows else None,
                "predrec_rate": safe_rate(predrec_count, predrec_denominator)
                if predrec_rows
                else None,
                "num_missing_match": sum(int(row.get("num_missing_match") or 0) for row in group),
                "num_missing_pred_answer": sum(
                    int(row.get("num_missing_pred_answer") or 0) for row in group
                ),
                "num_missing_fp16_pred_answer": sum(
                    int(row.get("num_missing_fp16_pred_answer") or 0) for row in group
                ),
            }
        )
        aggregates.append(aggregate)

    return sorted(
        aggregates,
        key=lambda row: (
            normalize_text(row.get("model_key")),
            SYSTEM_ORDER.index(row["system"]) if row.get("system") in SYSTEM_ORDER else 99,
            int(row.get("seed")) if row.get("seed") is not None else -1,
        ),
    )


def mean_std(values: Iterable[Any]) -> Tuple[Optional[float], Optional[float], int]:
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, (float, int)) and value is not None
    ]
    count = len(numeric_values)
    if count == 0:
        return None, None, 0
    mean_value = sum(numeric_values) / count
    if count == 1:
        return mean_value, 0.0, count
    variance = sum((value - mean_value) ** 2 for value in numeric_values) / (count - 1)
    return mean_value, variance**0.5, count


def mean_std_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("model_key")), str(row.get("system")))].append(row)

    stats_rows: List[Dict[str, Any]] = []
    for (model_key, system), group in grouped.items():
        seeds = sorted(
            int(row["seed"]) for row in group if row.get("seed") is not None
        )
        stats_row: Dict[str, Any] = {
            "model_key": model_key,
            "system": system,
            "system_display": SYSTEM_DISPLAY.get(system, system),
            "num_runs": len(group),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for metric_key, output_prefix in (
            ("accuracy_rate", "accuracy"),
            ("fp16_agree_rate", "fp16_agree"),
            ("predrec_rate", "predrec"),
        ):
            mean_value, std_value, count = mean_std(row.get(metric_key) for row in group)
            stats_row[f"{output_prefix}_mean_rate"] = mean_value
            stats_row[f"{output_prefix}_std_rate"] = std_value
            stats_row[f"{output_prefix}_num_values"] = count
        stats_rows.append(stats_row)

    return sorted(
        stats_rows,
        key=lambda row: (
            row["model_key"],
            SYSTEM_ORDER.index(row["system"]) if row.get("system") in SYSTEM_ORDER else 99,
        ),
    )


def add_pct_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for row in rows:
        copied = dict(row)
        copied["accuracy_pct"] = pct(copied.get("accuracy_rate"))
        copied["fp16_agree_pct"] = pct(copied.get("fp16_agree_rate"))
        copied["predrec_pct"] = pct(copied.get("predrec_rate"))
        enriched.append(copied)
    return enriched


def format_mean_std_text(
    mean_value: Optional[float], std_value: Optional[float], num_runs: int
) -> str:
    if mean_value is None:
        return ""
    if num_runs <= 1:
        return pct(mean_value)
    return f"{pct(mean_value)} +/- {pct(std_value)}"


def add_mean_std_pct_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for row in rows:
        copied = dict(row)
        num_runs = int(copied.get("num_runs") or 0)
        for prefix in ("accuracy", "fp16_agree", "predrec"):
            mean_key = f"{prefix}_mean_rate"
            std_key = f"{prefix}_std_rate"
            copied[f"{prefix}_mean_pct"] = pct(copied.get(mean_key))
            copied[f"{prefix}_std_pct"] = pct(copied.get(std_key))
            copied[f"{prefix}_mean_std_pct"] = format_mean_std_text(
                copied.get(mean_key),
                copied.get(std_key),
                num_runs,
            )
        enriched.append(copied)
    return enriched


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latex_metric(value: Optional[float]) -> str:
    return pct(value) if value is not None else "--"


def latex_mean_std_metric(row: Dict[str, Any], prefix: str) -> str:
    mean_value = row.get(f"{prefix}_mean_rate")
    num_runs = int(row.get("num_runs") or 0)
    if mean_value is None:
        return "--"
    if num_runs <= 1:
        return pct(mean_value)
    return f"${pct(mean_value)} \\pm {pct(row.get(f'{prefix}_std_rate'))}$"


def latex_table(rows: List[Dict[str, Any]], caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"System & Acc. & FP16 Agree & PredRec \\",
        r"\midrule",
    ]
    for system in SYSTEM_ORDER:
        row = next((item for item in rows if item.get("system") == system), None)
        if row is None:
            continue
        lines.append(
            f"{SYSTEM_DISPLAY[system]} & "
            f"{latex_metric(row.get('accuracy_rate'))} & "
            f"{latex_metric(row.get('fp16_agree_rate'))} & "
            f"{latex_metric(row.get('predrec_rate'))} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def latex_by_model_mean_std_table(
    rows: List[Dict[str, Any]], caption: str, label: str
) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model & System & Acc. & FP16 Agree & PredRec \\",
        r"\midrule",
    ]
    current_model = None
    for row in rows:
        model_key = str(row.get("model_key"))
        if current_model is not None and model_key != current_model:
            lines.append(r"\midrule")
        current_model = model_key
        lines.append(
            f"{model_key} & "
            f"{SYSTEM_DISPLAY.get(str(row.get('system')), str(row.get('system')))} & "
            f"{latex_mean_std_metric(row, 'accuracy')} & "
            f"{latex_mean_std_metric(row, 'fp16_agree')} & "
            f"{latex_mean_std_metric(row, 'predrec')} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def print_overall(rows: List[Dict[str, Any]]) -> None:
    print("System, Acc., FP16 Agree, PredRec")
    for system in SYSTEM_ORDER:
        row = next((item for item in rows if item.get("system") == system), None)
        if row is None:
            continue
        print(
            f"{SYSTEM_DISPLAY[system]}, "
            f"{latex_metric(row.get('accuracy_rate'))}, "
            f"{latex_metric(row.get('fp16_agree_rate'))}, "
            f"{latex_metric(row.get('predrec_rate'))}"
        )


def print_by_model_mean_std(rows: List[Dict[str, Any]]) -> None:
    print()
    print("Model, System, Acc., FP16 Agree, PredRec")
    for row in rows:
        num_runs = int(row.get("num_runs") or 0)
        print(
            f"{row['model_key']}, "
            f"{row['system_display']}, "
            f"{format_mean_std_text(row.get('accuracy_mean_rate'), row.get('accuracy_std_rate'), num_runs)}, "
            f"{format_mean_std_text(row.get('fp16_agree_mean_rate'), row.get('fp16_agree_std_rate'), num_runs)}, "
            f"{format_mean_std_text(row.get('predrec_mean_rate'), row.get('predrec_std_rate'), num_runs)}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(args)
    overall = aggregate_rows(rows, ("system",))
    by_model_system = aggregate_rows(rows, ("model_key", "system"))
    by_model_mean_std = mean_std_rows(rows)

    rows_csv = add_pct_columns(rows)
    overall_csv = add_pct_columns(overall)
    by_model_system_csv = add_pct_columns(by_model_system)
    by_model_mean_std_csv = add_mean_std_pct_columns(by_model_mean_std)

    row_fields = [
        "dataset",
        "model_key",
        "system",
        "system_display",
        "seed",
        "num_matched",
        "num_correct",
        "accuracy_rate",
        "accuracy_pct",
        "fp16_agree_count",
        "fp16_agree_rate",
        "fp16_agree_pct",
        "predrec_count",
        "predrec_denominator",
        "predrec_rate",
        "predrec_pct",
        "num_missing_match",
        "num_missing_pred_answer",
        "num_invalid_pred_answer",
        "num_missing_fp16_pred_answer",
        "num_fp16_rows",
        "num_candidate_rows",
        "prediction_file",
        "fp16_file",
        "predrec_summary_file",
        "predrec_summary_exists",
    ]
    aggregate_fields = [
        "model_key",
        "system",
        "system_display",
        "num_runs",
        "num_matched",
        "num_correct",
        "accuracy_rate",
        "accuracy_pct",
        "fp16_agree_count",
        "fp16_agree_rate",
        "fp16_agree_pct",
        "predrec_count",
        "predrec_denominator",
        "predrec_rate",
        "predrec_pct",
        "num_missing_match",
        "num_missing_pred_answer",
        "num_missing_fp16_pred_answer",
    ]

    mean_std_fields = [
        "model_key",
        "system",
        "system_display",
        "num_runs",
        "seeds",
        "accuracy_mean_rate",
        "accuracy_std_rate",
        "accuracy_mean_pct",
        "accuracy_std_pct",
        "accuracy_mean_std_pct",
        "accuracy_num_values",
        "fp16_agree_mean_rate",
        "fp16_agree_std_rate",
        "fp16_agree_mean_pct",
        "fp16_agree_std_pct",
        "fp16_agree_mean_std_pct",
        "fp16_agree_num_values",
        "predrec_mean_rate",
        "predrec_std_rate",
        "predrec_mean_pct",
        "predrec_std_pct",
        "predrec_mean_std_pct",
        "predrec_num_values",
    ]

    summary_json = args.output_dir / "fp16_answer_agreement_summary.json"
    rows_csv_path = args.output_dir / "fp16_answer_agreement_rows.csv"
    overall_csv_path = args.output_dir / "fp16_answer_agreement_overall.csv"
    by_model_csv_path = args.output_dir / "fp16_answer_agreement_by_model.csv"
    by_model_mean_std_csv_path = (
        args.output_dir / "fp16_answer_agreement_by_model_mean_std.csv"
    )
    latex_path = args.output_dir / "fp16_answer_agreement_table.tex"
    by_model_latex_path = args.output_dir / "fp16_answer_agreement_by_model_table.tex"

    write_json(
        summary_json,
        {
            "dataset": args.dataset,
            "dataset_display": args.dataset_display,
            "fp16_dir": str(args.fp16_dir),
            "medmix_baseline_dir": str(args.medmix_baseline_dir),
            "eaquant_dir": str(args.eaquant_dir),
            "output_dir": str(args.output_dir),
            "models": args.models,
            "seeds": args.seeds,
            "metric_definitions": {
                "accuracy_rate": "pred_answer == gold_answer on matched rows",
                "fp16_agree_rate": "pred_answer == FP16 pred_answer on matched rows",
                "predrec_rate": "original_prediction_match_rate from label_recovery summaries",
            },
            "rows": rows,
            "overall_by_system": overall,
            "by_model_system": by_model_system,
            "by_model_mean_std": by_model_mean_std,
        },
    )
    write_csv(rows_csv_path, rows_csv, row_fields)
    write_csv(overall_csv_path, overall_csv, aggregate_fields)
    write_csv(by_model_csv_path, by_model_system_csv, aggregate_fields)
    write_csv(by_model_mean_std_csv_path, by_model_mean_std_csv, mean_std_fields)

    latex_path.write_text(
        latex_table(
            overall,
            (
                f"{args.dataset_display} accuracy, full-precision answer agreement, and "
                "predicted-answer reconstruction. FP16 Agree measures whether "
                "quantization preserves the full-precision model's selected answer."
            ),
            f"tab:{args.dataset}_fp16_answer_agreement",
        ),
        encoding="utf-8",
    )
    by_model_latex_path.write_text(
        latex_by_model_mean_std_table(
            by_model_mean_std,
            (
                f"{args.dataset_display} per-model accuracy, full-precision answer agreement, "
                "and predicted-answer reconstruction. Quantized rows report "
                "mean and standard deviation over seeds 0, 1, and 2."
            ),
            f"tab:{args.dataset}_fp16_answer_agreement_by_model",
        ),
        encoding="utf-8",
    )

    print_overall(overall)
    print_by_model_mean_std(by_model_mean_std)
    print(f"[DONE] wrote {summary_json}")
    print(f"[DONE] wrote {rows_csv_path}")
    print(f"[DONE] wrote {overall_csv_path}")
    print(f"[DONE] wrote {by_model_csv_path}")
    print(f"[DONE] wrote {by_model_mean_std_csv_path}")
    print(f"[DONE] wrote {latex_path}")
    print(f"[DONE] wrote {by_model_latex_path}")


if __name__ == "__main__":
    main()
