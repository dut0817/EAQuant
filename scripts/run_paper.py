#!/usr/bin/env python3
"""One-line entry points for the retained EAQuant paper workflow."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SOURCE_TAG = "imp0p02_q0p02_beta0p01"


@dataclass(frozen=True)
class ModelProfile:
    key: str
    directory: str
    evaluation_key: str
    ea_prediction_key: str
    faith_max_seq_len: int
    uses_posthoc_filter: bool


MODEL_PROFILES = {
    "mistral_7b_instruct": ModelProfile(
        "mistral_7b_instruct",
        "Mistral-7B-Instruct-v0.3",
        "mistral_7b_instruct",
        "mistral_7b_instruct",
        640,
        True,
    ),
    "llama3_8b_instruct": ModelProfile(
        "llama3_8b_instruct",
        "Meta-Llama-3-8B-Instruct",
        "llama3_instruct",
        "llama3_instruct",
        1024,
        True,
    ),
    "biomistral_7b": ModelProfile(
        "biomistral_7b",
        "BioMistral-7B",
        "biomistral_7b",
        "biomistral_7b",
        640,
        False,
    ),
    "openbiollm_8b": ModelProfile(
        "openbiollm_8b",
        "Llama3-OpenBioLLM-8B",
        "openbiollm",
        "openbiollm_8b",
        640,
        False,
    ),
}

OMNIQUANT_MODELS = frozenset(
    {
        "mistral_7b_instruct",
        "llama3_8b_instruct",
    }
)

BACKEND_AWARE_STAGES = frozenset(
    {
        "quantize",
        "infer-baseline",
        "infer-eaquant",
        "accuracy",
        "agreement",
        "fec",
        "uacr",
        "pairwise",
    }
)

PUBLIC_STAGES = (
    "cache",
    "quantize",
    "infer-fp16",
    "infer-baseline",
    "infer-eaquant",
    "accuracy",
    "agreement",
    "fec",
    "uacr",
    "pairwise",
    "rationale-caches",
    "cache-stage-caches",
    "ablation-train",
    "ablation-infer",
    "rationale-eval",
)

INTERNAL_STAGES = (
    "teacher",
    "ostquant-baseline",
    "evidence",
    "filter",
    "ostquant-eaquant",
    "omniquant-stats",
    "omniquant-baseline",
    "omniquant-cache",
    "omniquant-eaquant",
)

STAGES = PUBLIC_STAGES + INTERNAL_STAGES

ABLATION_VARIANTS = (
    "no-ea",
    "token",
    "recovery",
    "both",
    "full",
    "random",
    "cache-v1",
    "cache-v2",
    "cache-v3",
    "cache-v4",
)


class PaperPaths:
    def __init__(self, profile: ModelProfile, seed: int):
        self.profile = profile
        self.seed = seed
        self.data_root = Path(
            os.environ.get("EAQUANT_DATA_ROOT", PROJECT_ROOT / "data")
        ).resolve()
        self.output_root = Path(
            os.environ.get("EAQUANT_OUTPUT_ROOT", PROJECT_ROOT / "outputs")
        ).resolve()
        self.model_root = Path(
            os.environ.get("EAQUANT_MODEL_ROOT", self.data_root / "models")
        ).resolve()
        self.model = self.model_root / profile.directory
        self.qwen_model = self.model_root / "Qwen2.5-72B-Instruct"
        self.medexpqa = self.data_root / "med_datasets" / "MedExpQA"

        self.cache_dir = (
            self.output_root / "cache" / "med_faithfulness" / profile.key
        )
        self.teacher_cache = self.cache_dir / "teacher.jsonl"
        self.raw_qwen_cache = self.cache_dir / "qwen-evidence-raw.jsonl"
        self.paper_cache = self.cache_dir / (
            f"medmix_train_teacher_predictions_qwen_evidence_pos_{SOURCE_TAG}.jsonl"
        )

        quant_root = self.output_root / "quantized"
        self.ost_baseline_dir = (
            quant_root / "ostquant" / profile.key / "medmix" / f"seed{seed}"
        )
        self.ost_eaquant_dir = (
            quant_root / "ostquant" / profile.key / "eaquant" / f"seed{seed}"
        )
        self.omni_baseline_dir = (
            quant_root / "omniquant" / profile.key / "medmix" / f"seed{seed}"
        )
        self.omni_eaquant_dir = (
            quant_root / "omniquant" / profile.key / "eaquant" / f"seed{seed}"
        )
        self.act_scale_dir = (
            quant_root / "omniquant" / profile.key / "act_scales" / f"seed{seed}"
        )
        self.act_shift_dir = (
            quant_root / "omniquant" / profile.key / "act_shifts" / f"seed{seed}"
        )
        self.act_scales = self.act_scale_dir / f"{profile.key}.pt"
        self.act_shifts = self.act_shift_dir / f"{profile.key}.pt"
        self.omni_cache = self.cache_dir / f"omni-effect-seed{seed}.jsonl"

        self.prediction_root = self.output_root / "predictions" / "medexpqa"
        self.fp16_dir = self.prediction_root / "train_baseline"
        self.medmix_dir = self.prediction_root / "train_quantized_medmix"
        self.eaquant_dir = self.prediction_root / "llm"
        self.evaluation_root = self.output_root / "evaluation" / "medexpqa"

    @property
    def ost_baseline_checkpoint(self) -> Path:
        return self.ost_baseline_dir / "model.pt"

    @property
    def ost_eaquant_checkpoint(self) -> Path:
        return self.ost_eaquant_dir / "model.pt"

    @property
    def omni_baseline_checkpoint(self) -> Path:
        return self.omni_baseline_dir / "omni_parameters.pth"

    @property
    def omni_eaquant_checkpoint(self) -> Path:
        return self.omni_eaquant_dir / "omni_parameters.pth"

    @property
    def fp16_json(self) -> Path:
        return self.fp16_dir / (
            f"test_{self.profile.evaluation_key}_"
            "original_train_baseline_predictions.jsonl"
        )

    @property
    def medmix_json(self) -> Path:
        suffix = "" if self.seed == 0 else f"_seed{self.seed}"
        return self.medmix_dir / (
            f"test_{self.profile.evaluation_key}_ostquant_w4a4kv4_"
            f"train_quantized_medmix{suffix}_predictions.jsonl"
        )

    @property
    def eaquant_json(self) -> Path:
        if self.profile.key == "llama3_8b_instruct":
            tag = (
                "ostquant_w4a4kv4_llm_ab_v4_teacher_faithful_"
                "qdamage_keepmargin_imp0p02_q0p02_beta0p01_tok01_rec01"
            )
            return (
                self.eaquant_dir
                / "ablation"
                / "qwen_component"
                / f"test_llama3_instruct_{tag}_seed{self.seed}_predictions.jsonl"
            )
        return self.eaquant_dir / (
            f"test_{self.profile.ea_prediction_key}_ostquant_w4a4kv4_llm_"
            f"qwen_imp0p02_q0p02_beta0p01_seed{self.seed}_predictions.jsonl"
        )

    def prediction_dirs(self, backend: str) -> tuple[Path, Path, Path]:
        if backend == "ostquant":
            return self.fp16_dir, self.medmix_dir, self.eaquant_dir
        root = self.prediction_root / "omniquant"
        return (
            self.fp16_dir,
            root / "train_quantized_medmix",
            root / "llm",
        )

    def prediction_json(self, system: str, backend: str) -> Path:
        fp16_dir, medmix_dir, eaquant_dir = self.prediction_dirs(backend)
        if system == "fp16":
            return fp16_dir / self.fp16_json.name
        if system == "medmix_baseline":
            return medmix_dir / self.medmix_json.name
        return eaquant_dir / self.eaquant_json.name

    def selected_evidence_cache(self) -> Path:
        return self.paper_cache

    def rationale_cache(self, variant: str) -> Path:
        root = self.output_root / "cache" / "ablation" / "rationale" / self.profile.key
        if variant == "full":
            name = f"medmix_train_teacher_predictions_full_noanswer_{SOURCE_TAG}.jsonl"
        else:
            name = (
                "medmix_train_teacher_predictions_random_noanswer_matched_"
                f"r{self.seed}_{SOURCE_TAG}.jsonl"
            )
        return root / name

    def cache_stage_path(self, variant: str) -> Path:
        stage_names = {
            "cache-v1": "v1_qwen_all",
            "cache-v2": "v2_teacher_faithful",
            "cache-v3": "v3_teacher_faithful_qdamage",
            "cache-v4": "v4_teacher_faithful_qdamage_keepmargin",
        }
        return (
            self.output_root
            / "cache"
            / "ablation"
            / "cache_stages"
            / self.profile.key
            / f"medmix_train_teacher_predictions_{stage_names[variant]}_{SOURCE_TAG}.jsonl"
        )

    def ablation_checkpoint(self, variant: str) -> Path:
        return (
            self.output_root
            / "quantized"
            / "ostquant"
            / self.profile.key
            / "ablation"
            / variant
            / f"seed{self.seed}"
            / "model.pt"
        )

    def ablation_prediction(self, variant: str) -> Path:
        return (
            self.prediction_root
            / "ablation"
            / f"{self.profile.key}-{variant}-seed{self.seed}.jsonl"
        )

    def prepare_output_directories(self) -> None:
        """Create the local output tree hidden behind the one-line commands."""
        _, omni_medmix_dir, omni_eaquant_dir = self.prediction_dirs("omniquant")
        directories = (
            self.cache_dir,
            self.ost_baseline_dir,
            self.ost_eaquant_dir,
            self.omni_baseline_dir,
            self.omni_eaquant_dir,
            self.act_scale_dir,
            self.act_shift_dir,
            self.fp16_dir,
            self.medmix_dir,
            self.eaquant_dir,
            omni_medmix_dir,
            omni_eaquant_dir,
            self.prediction_root / "ablation",
            self.evaluation_root,
            self.output_root / "ablation",
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one retained paper stage with repository-relative defaults. "
            "Defaults: Mistral-7B-Instruct-v0.3, seed 0, MedExpQA."
        )
    )
    parser.add_argument(
        "stage",
        metavar="STAGE",
        help=(
            "Public stages: cache, quantize, inference, evaluation, and "
            "ablation entry points."
        ),
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_PROFILES),
        default="mistral_7b_instruct",
        help="Paper model key.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=("ostquant", "omniquant"),
        default="ostquant",
        help="Backend for quantization, inference, and evaluation.",
    )
    parser.add_argument(
        "--variant",
        choices=ABLATION_VARIANTS,
        default="both",
        help="Variant for ablation-train and ablation-infer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and paths without running it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed steps in the cache or quantize pipeline.",
    )
    args, extra = parser.parse_known_args()
    if args.stage not in STAGES:
        parser.error(
            f"unknown stage {args.stage!r}; public stages: "
            f"{', '.join(PUBLIC_STAGES)}"
        )
    if args.seed not in (0, 1, 2):
        parser.error("--seed must be 0, 1, or 2 for the paper profile.")
    uses_omniquant = args.stage.startswith("omniquant-") or (
        args.stage in BACKEND_AWARE_STAGES and args.backend == "omniquant"
    )
    if uses_omniquant and args.model not in OMNIQUANT_MODELS:
        parser.error(
            "OmniQuant supports --model mistral_7b_instruct or "
            "llama3_8b_instruct in this repository."
        )
    if extra and extra[0] == "--":
        extra = extra[1:]
    args.extra = extra
    return args


def execute(
    command: Sequence[str | Path],
    *,
    args: argparse.Namespace,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
) -> None:
    resolved = [str(value) for value in command]
    print(f"[EAQuant] cwd: {cwd}")
    if env is not None:
        for key in sorted(key for key in env if key.startswith("EAQUANT_")):
            print(f"[EAQuant] {key}={env[key]}")
    print(f"[EAQuant] command: {shlex.join(resolved)}")
    if args.dry_run:
        return
    subprocess.run(resolved, cwd=cwd, env=env, check=True)


def ostquant_environment(
    paths: PaperPaths,
    *,
    system: str,
    output_dir: Path,
    checkpoint: Path,
    cache_path: Path | None = None,
    loss_variant: str = "both",
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "EAQUANT_SYSTEM": system,
            "EAQUANT_MODEL_PATH": str(paths.model),
            "EAQUANT_OUTPUT_DIR": str(output_dir),
            "EAQUANT_SAVE_QMODEL_PATH": str(checkpoint),
            "EAQUANT_SEED": str(paths.seed),
            "EAQUANT_FAITH_MAX_SEQ_LEN": str(
                paths.profile.faith_max_seq_len
            ),
            "EAQUANT_LOSS_VARIANT": loss_variant,
        }
    )
    if cache_path is not None:
        env["EAQUANT_CACHE_PATH"] = str(cache_path)
    return env


def omniquant_environment(
    paths: PaperPaths,
    *,
    system: str,
    output_dir: Path,
    cache_path: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "EAQUANT_SYSTEM": system,
            "EAQUANT_MODEL_PATH": str(paths.model),
            "EAQUANT_MODEL_KEY": paths.profile.key,
            "EAQUANT_OUTPUT_DIR": str(output_dir),
            "EAQUANT_ACT_SCALES": str(paths.act_scales),
            "EAQUANT_ACT_SHIFTS": str(paths.act_shifts),
            "EAQUANT_SEED": str(paths.seed),
        }
    )
    if cache_path is not None:
        env["EAQUANT_CACHE_PATH"] = str(cache_path)
    return env


def cache_command(stage: str, paths: PaperPaths) -> list[str | Path]:
    entry = PROJECT_ROOT / "scripts" / "build_evidence_cache.py"
    if stage == "teacher":
        return [
            PYTHON,
            entry,
            "teacher",
            "--model_path",
            paths.model,
            "--data_root",
            paths.data_root,
            "--output_jsonl",
            paths.teacher_cache,
            "--correct_only",
        ]
    if stage == "evidence":
        output = (
            paths.raw_qwen_cache
            if paths.profile.uses_posthoc_filter
            else paths.paper_cache
        )
        return [
            PYTHON,
            entry,
            "evidence",
            "--teacher_cache_path",
            paths.teacher_cache,
            "--output_jsonl",
            output,
            "--model_path",
            paths.model,
            "--qwen_model_path",
            paths.qwen_model,
            "--quant_checkpoint",
            paths.ost_baseline_checkpoint,
        ]
    return [
        PYTHON,
        entry,
        "filter",
        "--input",
        paths.raw_qwen_cache,
        "--output",
        paths.paper_cache,
        "--min_importance",
        "0.02",
        "--min_quant_impact",
        "0.02",
        "--quant_keep_margin_beta",
        "0.01",
        "--tag",
        SOURCE_TAG,
    ]


def inference_command(
    paths: PaperPaths,
    *,
    system: str,
    backend: str | None,
    checkpoint: Path | None,
    output: Path,
) -> list[str | Path]:
    command: list[str | Path] = [
        PYTHON,
        PROJECT_ROOT / "scripts" / "evaluate" / "run_inference.py",
        "--system",
        system,
    ]
    if backend is not None:
        command += ["--backend", backend]
    command += [
        "--model_path",
        paths.model,
        "--data_root",
        paths.medexpqa,
        "--output_jsonl",
        output,
    ]
    if checkpoint is not None:
        command += ["--quant_checkpoint", checkpoint]
    if backend == "omniquant":
        command += [
            "--net",
            paths.profile.key,
            "--act-scales",
            paths.act_scales,
            "--act-shifts",
            paths.act_shifts,
        ]
    return command


def evaluation_prediction_inputs(
    paths: PaperPaths,
    backend: str,
) -> dict[str, Path]:
    inputs = {
        "fp16": paths.prediction_json("fp16", backend),
        "medmix_baseline": paths.prediction_json(
            "medmix_baseline",
            backend,
        ),
        "eaquant": paths.prediction_json("eaquant", backend),
    }

    # Retained Llama-3 experiments may use the legacy V5 artifact name. The
    # public evaluator treats it as an alias of the final V4 cache stage.
    eaquant_path = inputs["eaquant"]
    legacy_v5 = Path(str(eaquant_path).replace("_ab_v4_", "_ab_v5_"))
    if (
        legacy_v5 != eaquant_path
        and not eaquant_path.exists()
        and legacy_v5.exists()
    ):
        inputs["eaquant"] = legacy_v5
    return inputs


def evaluation_command(
    stage: str,
    paths: PaperPaths,
    backend: str,
    prediction_inputs: dict[str, Path],
) -> list[str | Path]:
    scripts = PROJECT_ROOT / "scripts" / "evaluate"
    fp16_dir, medmix_dir, eaquant_dir = paths.prediction_dirs(backend)
    common_dirs: list[str | Path] = [
        "--fp16_dir",
        fp16_dir,
        "--medmix_baseline_dir",
        medmix_dir,
        "--eaquant_dir",
        eaquant_dir,
    ]
    if stage == "accuracy":
        return [
            PYTHON,
            scripts / "evaluate_accuracy.py",
            prediction_inputs["fp16"],
            prediction_inputs["medmix_baseline"],
            prediction_inputs["eaquant"],
            "--output_json",
            paths.evaluation_root / backend / "accuracy.json",
        ]
    if stage == "agreement":
        return [
            PYTHON,
            scripts / "evaluate_fp16_agreement.py",
            "--dataset",
            "medexpqa",
            *common_dirs,
            "--models",
            paths.profile.evaluation_key,
            "--seeds",
            str(paths.seed),
            "--output_dir",
            paths.evaluation_root / backend / "fp16-agreement",
        ]
    metric_script = {
        "fec": "evaluate_fp16_retention.py",
        "uacr": "evaluate_unsupported_claims.py",
        "pairwise": "evaluate_pairwise_llm_judge.py",
    }[stage]
    command: list[str | Path] = [
        PYTHON,
        scripts / metric_script,
        *common_dirs,
        "--models",
        paths.profile.evaluation_key,
        "--seeds",
        str(paths.seed),
        "--output_dir",
        paths.evaluation_root / backend / stage,
        "--judge_model",
        "gpt-5.4",
    ]
    if stage == "fec":
        command += ["--max_claims", "8"]
    elif stage == "uacr":
        command += ["--max_claims", "20"]
    else:
        command += ["--mode", "pairwise", "--judge1_all_samples"]
    return command


def ablation_cache_path(paths: PaperPaths, variant: str) -> Path | None:
    if variant == "no-ea":
        return None
    if variant in {"token", "recovery", "both"}:
        return paths.selected_evidence_cache()
    if variant in {"full", "random"}:
        return paths.rationale_cache(variant)
    return paths.cache_stage_path(variant)


def run_pipeline_step(
    args: argparse.Namespace,
    stage: str,
    artifacts: Sequence[Path],
) -> None:
    if (
        not args.force
        and not args.dry_run
        and artifacts
        and all(path.exists() for path in artifacts)
    ):
        print(f"[EAQuant] reuse completed step: {stage}")
        return

    print(f"[EAQuant] pipeline step: {stage}")
    nested = argparse.Namespace(**vars(args))
    nested.stage = stage
    nested.extra = []
    run_stage(nested)


def run_stage(args: argparse.Namespace) -> None:
    profile = MODEL_PROFILES[args.model]
    paths = PaperPaths(profile, args.seed)
    if not args.dry_run:
        paths.prepare_output_directories()
    extra = list(args.extra)

    if args.stage in {"cache", "quantize"} and extra:
        raise SystemExit(
            "Extra backend arguments are only supported by individual "
            "advanced stages."
        )

    if args.stage == "cache":
        cache_args = argparse.Namespace(**vars(args))
        cache_args.seed = 0
        cache_paths = PaperPaths(profile, seed=0)
        if args.seed != 0:
            print("[EAQuant] the shared evidence cache uses source seed 0")
        if not args.dry_run:
            cache_paths.prepare_output_directories()
        if (
            not args.force
            and not args.dry_run
            and cache_paths.paper_cache.exists()
        ):
            print(
                f"[EAQuant] reuse completed cache: {cache_paths.paper_cache}"
            )
            return
        run_pipeline_step(
            cache_args,
            "teacher",
            [cache_paths.teacher_cache],
        )
        run_pipeline_step(
            cache_args,
            "ostquant-baseline",
            [cache_paths.ost_baseline_checkpoint],
        )
        if profile.uses_posthoc_filter:
            run_pipeline_step(
                cache_args,
                "evidence",
                [cache_paths.raw_qwen_cache],
            )
            run_pipeline_step(
                cache_args,
                "filter",
                [cache_paths.paper_cache],
            )
        else:
            run_pipeline_step(
                cache_args,
                "evidence",
                [cache_paths.paper_cache],
            )
        return

    if args.stage == "quantize":
        cache_will_run = (
            args.force or args.dry_run or not paths.paper_cache.exists()
        )
        run_pipeline_step(args, "cache", [paths.paper_cache])
        if args.backend == "ostquant":
            # Seed 0 is already produced when a new shared cache is built.
            if args.seed != 0 or not cache_will_run:
                run_pipeline_step(
                    args,
                    "ostquant-baseline",
                    [paths.ost_baseline_checkpoint],
                )
            run_pipeline_step(
                args,
                "ostquant-eaquant",
                [paths.ost_eaquant_checkpoint],
            )
        else:
            run_pipeline_step(
                args,
                "omniquant-stats",
                [paths.act_scales, paths.act_shifts],
            )
            run_pipeline_step(
                args,
                "omniquant-baseline",
                [paths.omni_baseline_checkpoint],
            )
            run_pipeline_step(args, "omniquant-cache", [paths.omni_cache])
            run_pipeline_step(
                args,
                "omniquant-eaquant",
                [paths.omni_eaquant_checkpoint],
            )
        return

    if args.stage in {"teacher", "evidence", "filter"}:
        if args.stage == "filter" and not profile.uses_posthoc_filter:
            raise SystemExit(
                f"{profile.key} uses the direct evidence cache; filter is not required."
            )
        execute([*cache_command(args.stage, paths), *extra], args=args)
        return

    ost_launcher = PROJECT_ROOT / "scripts" / "quantize" / "run_ostquant.sh"
    omni_launcher = PROJECT_ROOT / "scripts" / "quantize" / "run_omniquant.sh"

    if args.stage == "ostquant-baseline":
        env = ostquant_environment(
            paths,
            system="medmix_baseline",
            output_dir=paths.ost_baseline_dir,
            checkpoint=paths.ost_baseline_checkpoint,
            loss_variant="none",
        )
        execute([ost_launcher, *extra], args=args, env=env)
        return

    if args.stage == "ostquant-eaquant":
        env = ostquant_environment(
            paths,
            system="eaquant",
            output_dir=paths.ost_eaquant_dir,
            checkpoint=paths.ost_eaquant_checkpoint,
            cache_path=paths.selected_evidence_cache(),
        )
        execute([ost_launcher, *extra], args=args, env=env)
        return

    if args.stage == "omniquant-stats":
        command = [
            PYTHON,
            PROJECT_ROOT / "backends" / "omniquant" / "generate_act_scale_shift.py",
            "--model",
            paths.model,
            "--net",
            profile.key,
            "--calib_dataset",
            "medmix",
            "--num-samples",
            "128",
            "--seq-len",
            "512",
            "--seed",
            str(args.seed),
            "--scales-output-path",
            paths.act_scale_dir,
            "--shifts-output-path",
            paths.act_shift_dir,
            *extra,
        ]
        omni_env = os.environ.copy()
        backend = str(PROJECT_ROOT / "backends" / "omniquant")
        omni_env["PYTHONPATH"] = (
            backend
            if not omni_env.get("PYTHONPATH")
            else f"{backend}:{omni_env['PYTHONPATH']}"
        )
        execute(command, args=args, env=omni_env)
        return

    if args.stage == "omniquant-baseline":
        env = omniquant_environment(
            paths,
            system="medmix_baseline",
            output_dir=paths.omni_baseline_dir,
        )
        execute([omni_launcher, *extra], args=args, env=env)
        return

    if args.stage == "omniquant-cache":
        execute(
            [
                PYTHON,
                PROJECT_ROOT
                / "backends"
                / "omniquant"
                / "build_omniquant_faithfulness_cache.py",
                "--model",
                paths.model,
                "--net",
                profile.key,
                "--resume",
                paths.omni_baseline_checkpoint,
                "--source_cache_path",
                paths.paper_cache,
                "--output_cache_path",
                paths.omni_cache,
                "--seed",
                str(args.seed),
                "--seqlen",
                "512",
                "--nsamples",
                "128",
                "--act-scales",
                paths.act_scales,
                "--act-shifts",
                paths.act_shifts,
                *extra,
            ],
            args=args,
        )
        return

    if args.stage == "omniquant-eaquant":
        env = omniquant_environment(
            paths,
            system="eaquant",
            output_dir=paths.omni_eaquant_dir,
            cache_path=paths.omni_cache,
        )
        execute([omni_launcher, *extra], args=args, env=env)
        return

    if args.stage == "infer-fp16":
        command = inference_command(
            paths,
            system="fp16",
            backend=None,
            checkpoint=None,
            output=paths.fp16_json,
        )
        execute([*command, *extra], args=args)
        return

    if args.stage in {"infer-baseline", "infer-eaquant"}:
        is_baseline = args.stage == "infer-baseline"
        system = "medmix_baseline" if is_baseline else "eaquant"
        if args.backend == "ostquant":
            checkpoint = (
                paths.ost_baseline_checkpoint
                if is_baseline
                else paths.ost_eaquant_checkpoint
            )
        else:
            checkpoint = (
                paths.omni_baseline_checkpoint
                if is_baseline
                else paths.omni_eaquant_checkpoint
            )
        output = paths.prediction_json(system, args.backend)
        command = inference_command(
            paths,
            system=system,
            backend=args.backend,
            checkpoint=checkpoint,
            output=output,
        )
        execute([*command, *extra], args=args)
        return

    if args.stage in {"accuracy", "agreement", "fec", "uacr", "pairwise"}:
        prediction_inputs = evaluation_prediction_inputs(paths, args.backend)
        print("[EAQuant] evaluation prediction inputs:")
        for system, prediction_path in prediction_inputs.items():
            print(f"  - {system}: {prediction_path}")
        if not args.dry_run:
            missing = [
                (system, prediction_path)
                for system, prediction_path in prediction_inputs.items()
                if not prediction_path.is_file()
            ]
            if missing:
                details = "\n".join(
                    f"  - {system}: {prediction_path}"
                    for system, prediction_path in missing
                )
                raise SystemExit(
                    "Missing inference prediction files. Run the matching "
                    f"inference commands first:\n{details}"
                )
        execute(
            [
                *evaluation_command(
                    args.stage,
                    paths,
                    args.backend,
                    prediction_inputs,
                ),
                *extra,
            ],
            args=args,
        )
        return

    if args.stage == "rationale-caches":
        env = os.environ.copy()
        env["EAQUANT_MODEL_ROOT"] = str(paths.model_root)
        execute(
            [
                PYTHON,
                PROJECT_ROOT
                / "scripts"
                / "ablation"
                / "rationale"
                / "build_rationale_caches.py",
                "--repo_dir",
                paths.output_root,
                "--output_root",
                paths.output_root / "cache" / "ablation" / "rationale",
                "--models",
                profile.key,
                "--selection_seeds",
                str(args.seed),
                *extra,
            ],
            args=args,
            env=env,
        )
        return

    if args.stage == "cache-stage-caches":
        execute(
            [
                PYTHON,
                PROJECT_ROOT
                / "scripts"
                / "ablation"
                / "cache_stages"
                / "build_cache_stages.py",
                "--repo_dir",
                paths.output_root,
                "--output_root",
                paths.output_root / "cache" / "ablation" / "cache_stages",
                "--models",
                profile.key,
                *extra,
            ],
            args=args,
        )
        return

    if args.stage == "ablation-train":
        output_dir = paths.ablation_checkpoint(args.variant).parent
        loss_variant = {
            "no-ea": "none",
            "token": "token",
            "recovery": "recovery",
        }.get(args.variant, "both")
        env = ostquant_environment(
            paths,
            system="eaquant",
            output_dir=output_dir,
            checkpoint=paths.ablation_checkpoint(args.variant),
            cache_path=ablation_cache_path(paths, args.variant),
            loss_variant=loss_variant,
        )
        execute([ost_launcher, *extra], args=args, env=env)
        return

    if args.stage == "ablation-infer":
        command = inference_command(
            paths,
            system="eaquant",
            backend="ostquant",
            checkpoint=paths.ablation_checkpoint(args.variant),
            output=paths.ablation_prediction(args.variant),
        )
        execute([*command, *extra], args=args)
        return

    if args.stage == "rationale-eval":
        execute(
            [
                PYTHON,
                PROJECT_ROOT
                / "scripts"
                / "ablation"
                / "rationale"
                / "evaluate_fp16_retention.py",
                "--fp16_jsonl",
                paths.fp16_json,
                "--medmix_baseline_jsonl",
                paths.medmix_json,
                "--eaquant_jsonl",
                paths.eaquant_json,
                "--full_rationale_jsonl",
                paths.ablation_prediction("full"),
                "--random_rationale_jsonl",
                paths.ablation_prediction("random"),
                "--claim_cache",
                paths.evaluation_root
                / "ostquant"
                / "fec"
                / (
                    "fp16_claims_medexpqa_"
                    f"{paths.profile.evaluation_key}_by_gpt-5-4.jsonl"
                ),
                "--output_dir",
                paths.output_root / "ablation" / "rationale-fec",
                "--judge_model",
                "gpt-5.4",
                "--max_claims",
                "8",
                *extra,
            ],
            args=args,
        )
        return

    raise AssertionError(f"Unhandled stage: {args.stage}")


def main() -> None:
    run_stage(parse_args())


if __name__ == "__main__":
    main()
