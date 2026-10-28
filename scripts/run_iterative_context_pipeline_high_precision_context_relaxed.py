#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_iterative_context_pipeline import main


if __name__ == "__main__":
    if "--precision-profile" not in sys.argv:
        sys.argv.extend(["--precision-profile", "high"])
    if "--context-similarity-threshold-override" not in sys.argv:
        sys.argv.extend(["--context-similarity-threshold-override", "0.50"])
    if "--fallback-keep-top-n" not in sys.argv:
        sys.argv.extend(["--fallback-keep-top-n", "5"])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "ontology_experiment_framework/stage_outputs/iterative_context_compare_high_precision_context_relaxed"])
    main()
