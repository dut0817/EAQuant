#!/usr/bin/env python3
"""Run one of the three paper comparison systems on MedExpQA."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
SYSTEMS = ("fp16", "medmix_baseline", "eaquant")
QUANT_BACKENDS = ("omniquant", "ostquant")


def main() -> None:
    help_parser = argparse.ArgumentParser(
        description=(
            "Run FP16, standard MedMix quantization, or EAQuant inference. "
            "Quantized systems additionally require a backend."
        )
    )
    help_parser.add_argument("--system", choices=SYSTEMS, required=True)
    help_parser.add_argument("--backend", choices=QUANT_BACKENDS)
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        help_parser.print_help()
        return

    parser = argparse.ArgumentParser(
        description="Run one paper comparison system and write predictions.",
        add_help=False,
    )
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--backend", choices=QUANT_BACKENDS)
    args, remaining = parser.parse_known_args()

    if args.system == "fp16":
        if args.backend is not None:
            parser.error("--backend is only valid for quantized systems.")
    elif args.backend is None:
        parser.error("--backend is required for medmix_baseline and eaquant.")

    sys.argv = [sys.argv[0], "--system_name", args.system, *remaining]

    if args.system == "fp16":
        from eaquant.evaluation.inference import main as inference_main

        inference_main()
        return

    backend_script = (
        PROJECT_ROOT / "backends" / str(args.backend) / "medexpqa_inference.py"
    )
    runpy.run_path(str(backend_script), run_name="__main__")


if __name__ == "__main__":
    main()
