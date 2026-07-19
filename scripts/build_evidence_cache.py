#!/usr/bin/env python3
"""Build the teacher, evidence, and paper-filter cache stages used by EAQuant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    help_parser = argparse.ArgumentParser(
        description=(
            "Build teacher predictions, Qwen evidence, or the paper's "
            "post-hoc evidence filter for EAQuant."
        )
    )
    help_parser.add_argument(
        "stage", choices=("teacher", "evidence", "filter"), nargs="?"
    )
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        help_parser.print_help()
        return

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("stage", choices=("teacher", "evidence", "filter"))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    if args.stage == "teacher":
        from eaquant.evidence.teacher_cache import main as stage_main
    elif args.stage == "evidence":
        from eaquant.evidence.qwen_evidence import main as stage_main
    else:
        from eaquant.evidence.filter_qwen_evidence import main as stage_main
    stage_main()


if __name__ == "__main__":
    main()
