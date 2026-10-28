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
    if "--extractors" not in sys.argv:
        sys.argv.extend(["--extractors", "yake", "keybert_mpnet"])
    if "--candidate-intersection-methods" not in sys.argv:
        sys.argv.extend(["--candidate-intersection-methods", "yake", "keybert_mpnet"])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "ontology_experiment_framework/stage_outputs/iterative_context_compare_high_precision_yake_keybert_intersection"])
    main()
