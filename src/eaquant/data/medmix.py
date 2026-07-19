import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import transformers


def _repo_root() -> Path:
    return Path(
        os.environ.get("EAQUANT_DATA_ROOT", Path(__file__).resolve().parents[3])
    )


def _safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _build_tokenizer(model: str, hf_token=None):
    tokenizer_kwargs = {"use_fast": False}
    if hf_token is not None:
        tokenizer_kwargs["use_auth_token"] = hf_token
    if "mistral" in model.lower():
        tokenizer_kwargs["legacy"] = True
    return transformers.AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)


def _combine_explanations(exp1, exp2) -> str:
    exp1 = _safe_text(exp1)
    exp2 = _safe_text(exp2)
    if exp1 and exp2:
        if exp1 == exp2:
            return exp1
        return f"{exp1}\n\nAdditional rationale: {exp2}"
    return exp1 or exp2


def _normalize_option_letter(value) -> str:
    normalized = _safe_text(value).upper()
    answer_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    return answer_map.get(normalized, normalized)


def _build_medmix_prompt(
    question: str,
    option_items: Sequence[Tuple[str, str]],
    contexts: Optional[Sequence[str]] = None,
) -> str:
    lines = [f"Question: {_safe_text(question)}"]
    for idx, context in enumerate(contexts or []):
        context = _safe_text(context)
        if context:
            lines.append(f"Context {idx + 1}: {context}")
    if option_items:
        lines.append("Options:")
        for label, text in option_items:
            lines.append(f"{_safe_text(label)}. {_safe_text(text)}")
    return "\n".join(lines)


def _build_medmix_train_record(
    record_id: str,
    source: str,
    question: str,
    option_items: Sequence[Tuple[str, str]],
    gold_answer: str,
    gold_option_text: str,
    explanation: str,
    contexts: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    prompt = _build_medmix_prompt(question, option_items=option_items, contexts=contexts)
    normalized_option_items = [
        (_safe_text(label), _safe_text(text)) for label, text in option_items
    ]
    normalized_contexts = [
        context for context in (_safe_text(context) for context in (contexts or [])) if context
    ]
    answer_text = f"{_safe_text(gold_answer)}. {_safe_text(gold_option_text)}"
    answer_prefix = f"{prompt}\nAnswer: "
    explanation_prefix = f"{prompt}\nAnswer: {answer_text}\nRationale: "
    explanation_text = _safe_text(explanation)
    full_text = f"{explanation_prefix}{explanation_text}"
    return {
        "record_id": record_id,
        "source": source,
        "question": _safe_text(question),
        "options": {label: text for label, text in normalized_option_items},
        "contexts": normalized_contexts,
        "prompt": prompt,
        "option_labels": [label for label, _ in normalized_option_items],
        "answer_prefix": answer_prefix,
        "answer_text": _safe_text(gold_answer),
        "answer_option_text": _safe_text(gold_option_text),
        "answer_display_text": answer_text,
        "explanation_prefix": explanation_prefix,
        "explanation_text": explanation_text,
        "full_text": full_text,
    }


def _iter_medexqa_dev_train_records(base_dir: Path) -> Iterable[Dict[str, str]]:
    dev_dir = base_dir / "med_datasets" / "MedExQA" / "dev"
    for path in sorted(dev_dir.glob("*_dev.tsv")):
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            specialty = path.name[: -len("_dev.tsv")]
            for row_idx, row in enumerate(reader, start=1):
                if len(row) < 8:
                    continue
                question, a, b, c, d, exp1, exp2, gold = row[:8]
                options = {
                    "A": _safe_text(a),
                    "B": _safe_text(b),
                    "C": _safe_text(c),
                    "D": _safe_text(d),
                }
                gold_label = _safe_text(gold)
                gold_option_text = options.get(gold_label, "")
                yield _build_medmix_train_record(
                    record_id=f"medexqa_dev::{specialty}::{row_idx:04d}",
                    source="MedExQA dev",
                    question=question,
                    option_items=tuple(options.items()),
                    gold_answer=gold_label,
                    gold_option_text=gold_option_text,
                    explanation=_combine_explanations(exp1, exp2),
                )


def _iter_medexpqa_train_records(base_dir: Path) -> Iterable[Dict[str, str]]:
    train_path = base_dir / "med_datasets" / "MedExpQA" / "data" / "en" / "train.json"
    with train_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    answer_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    option_key_map = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}
    for row_idx, ex in enumerate(data, start=1):
        correct_option = str(ex.get("correct_option", ""))
        options = ex.get("options", {})
        gold_label = answer_map.get(correct_option, correct_option)
        gold_option_text = _safe_text(options.get(option_key_map.get(gold_label, ""), ""))
        explanation = ex.get("explanations", {}).get(correct_option, {}).get("text", "")
        yield _build_medmix_train_record(
            record_id=f"medexpqa_train::{row_idx:05d}",
            source="MedExpQA train",
            question=ex.get("full_question", ""),
            option_items=(
                ("A", options.get("1", "")),
                ("B", options.get("2", "")),
                ("C", options.get("3", "")),
                ("D", options.get("4", "")),
                ("E", options.get("5", "")),
            ),
            gold_answer=gold_label,
            gold_option_text=gold_option_text,
            explanation=explanation,
        )


def _iter_challengeclinical_op4_records(base_dir: Path) -> Iterable[Dict[str, str]]:
    train_path = base_dir / "med_datasets" / "challengeclinicalQA" / "medbullets_op4.json"
    with train_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {train_path}")

    for row_idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue

        raw_options = item.get("options") or {}
        if isinstance(raw_options, list):
            options = {
                _safe_text(option.get("label", "")): _safe_text(option.get("text", ""))
                for option in raw_options
                if isinstance(option, dict)
            }
        else:
            options = {
                _safe_text(label): _safe_text(text) for label, text in raw_options.items()
            }
        options = {label: text for label, text in options.items() if label and text}
        if not options:
            continue

        gold_label = _normalize_option_letter(
            item.get("gold_answer") or item.get("answer_idx") or item.get("correct_option")
        )
        gold_option_text = options.get(gold_label, _safe_text(item.get("gold_answer_text", "")))
        explanation = _combine_explanations(
            item.get("gold_explanation_1") or item.get("explanation", ""),
            item.get("gold_explanation_2")
            or item.get("gold_explanation_1")
            or item.get("explanation", ""),
        )
        example_id = _safe_text(item.get("example_id")) or f"challengeclinical_op4::{row_idx:05d}"
        yield _build_medmix_train_record(
            record_id=example_id,
            source="ChallengeClinical op4",
            question=item.get("question", ""),
            option_items=tuple(options.items()),
            gold_answer=gold_label,
            gold_option_text=gold_option_text,
            explanation=explanation,
        )


def _select_medmix_records(
    records: List[Dict[str, str]],
    max_records: int,
    seed: int,
    shuffle: bool,
) -> List[Dict[str, str]]:
    if len(records) <= max_records:
        return records

    selected = list(records)
    if shuffle:
        random.Random(seed).shuffle(selected)
    return selected[:max_records]


def _medmix_source_limits(raw_counts: Sequence[int], target_records: int) -> Tuple[int, int, int]:
    target_records = max(int(target_records), 0)
    medexqa_limit = min(raw_counts[0], 25, target_records)
    remaining = max(target_records - medexqa_limit, 0)

    medexpqa_limit = min(raw_counts[1], (remaining + 1) // 2)
    challenge_limit = min(raw_counts[2], remaining - medexpqa_limit)

    extra = remaining - medexpqa_limit - challenge_limit
    if extra > 0:
        add = min(extra, raw_counts[1] - medexpqa_limit)
        medexpqa_limit += add
        extra -= add
    if extra > 0:
        add = min(extra, raw_counts[2] - challenge_limit)
        challenge_limit += add

    return medexqa_limit, medexpqa_limit, challenge_limit


def _load_medmix_source_groups(
    base_dir: Path,
    target_records: int = 128,
) -> List[Tuple[str, List[Dict[str, str]], bool]]:
    raw_records = (
        ("MedExQA dev", list(_iter_medexqa_dev_train_records(base_dir)), False),
        ("MedExpQA train", list(_iter_medexpqa_train_records(base_dir)), True),
        ("ChallengeClinical op4", list(_iter_challengeclinical_op4_records(base_dir)), True),
    )
    limits = _medmix_source_limits(
        tuple(len(records) for _, records, _ in raw_records),
        target_records=target_records,
    )

    source_groups: List[Tuple[str, List[Dict[str, str]], bool]] = []
    for idx, ((dataset_name, records, shuffle), max_records) in enumerate(
        zip(raw_records, limits)
    ):
        selected = _select_medmix_records(
            records,
            max_records=max_records,
            seed=idx,
            shuffle=shuffle,
        )
        print(
            f"[medmix-train] {dataset_name}: "
            f"raw_examples={len(records)}, selected_examples={len(selected)}"
        )
        source_groups.append((dataset_name, selected, shuffle))

    return source_groups


def get_medmix_train_examples(
    base_dir: Optional[Path] = None,
    target_records: int = 128,
) -> List[Dict[str, str]]:
    if base_dir is None:
        base_dir = _repo_root()

    records: List[Dict[str, str]] = []
    for _, group, _ in _load_medmix_source_groups(
        base_dir,
        target_records=target_records,
    ):
        records.extend(group)

    print(f"[medmix-train] final selected_examples={len(records)}")
    return records


def get_medmix_train_texts(base_dir: Optional[Path] = None) -> List[str]:
    records = get_medmix_train_examples(base_dir=base_dir)
    return [record["full_text"] for record in records]


def _build_source_corpus(
    texts: Iterable[str], tokenizer, seed: int, shuffle: bool, dataset_name: str
) -> torch.Tensor:
    encoded_texts: List[List[int]] = []
    for text in texts:
        text = _safe_text(text)
        if not text:
            continue
        token_ids = tokenizer.encode(text)
        if len(token_ids) > 0:
            encoded_texts.append(token_ids)

    if not encoded_texts:
        raise ValueError(f"{dataset_name}: no usable texts found")

    if shuffle:
        random.Random(seed).shuffle(encoded_texts)

    sep_ids = []
    if tokenizer.eos_token_id is not None:
        sep_ids = [tokenizer.eos_token_id]

    flat_ids: List[int] = []
    for token_ids in encoded_texts:
        flat_ids.extend(token_ids)
        flat_ids.extend(sep_ids)

    if len(flat_ids) == 0:
        raise ValueError(f"{dataset_name}: empty token stream")

    print(
        f"[medmix] {dataset_name}: raw_texts={len(encoded_texts)}, total_tokens={len(flat_ids)}"
    )
    return torch.tensor(flat_ids, dtype=torch.long).unsqueeze(0)


def _sample_windows(
    input_ids: torch.Tensor, n_samples: int, block_size: int, seed: int, dataset_name: str
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    total_tokens = input_ids.shape[1]
    max_start = total_tokens - block_size
    if max_start < 0:
        raise ValueError(
            f"{dataset_name}: token stream too short ({total_tokens}) for block_size={block_size}"
        )

    rng = random.Random(seed)
    samples = []
    for _ in range(n_samples):
        start = rng.randint(0, max_start)
        end = start + block_size
        inp = input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        samples.append((inp, tar))
    return samples


def get_medmix(nsamples=128, seed=0, seqlen=512, model="", hf_token=None, eval_mode=False):
    if eval_mode:
        raise ValueError("medmix is currently supported only for calibration (eval_mode=False)")
    if nsamples != 128:
        raise ValueError(f"medmix currently expects nsamples=128, but got {nsamples}")

    tokenizer = _build_tokenizer(model, hf_token=hf_token)
    base_dir = _repo_root()

    trainloader: List[Tuple[torch.Tensor, torch.Tensor]] = []
    source_groups = _load_medmix_source_groups(base_dir, target_records=nsamples)
    for idx, (dataset_name, records, shuffle) in enumerate(source_groups):
        corpus = _build_source_corpus(
            (record["full_text"] for record in records),
            tokenizer=tokenizer,
            seed=seed + idx,
            shuffle=shuffle,
            dataset_name=dataset_name,
        )
        trainloader.extend(
            _sample_windows(
                corpus,
                n_samples=len(records),
                block_size=seqlen,
                seed=seed + idx,
                dataset_name=dataset_name,
            )
        )

    if len(trainloader) != nsamples:
        raise ValueError(f"medmix produced {len(trainloader)} samples, expected {nsamples}")

    print(f"[medmix] final samples={len(trainloader)}, block_size={seqlen}")
    return trainloader
