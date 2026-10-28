
# Top-K Sweep for Term Extraction Benchmark

This script runs a **Top-K sweep** for the following methods:
- `yake`
- `ensemble_yake_spacy_union`
- `ensemble_yake_spacy_regex_acronym_union`
- `ensemble_yake_spacy_regex_acronym_pattern_union`

It evaluates each method at:
- 25
- 50
- 75
- 100
- 150
- 200

## Output files
- `data/processed/evaluation/metrics_topk_sweep.csv`
- `data/processed/evaluation/metrics_topk_best.csv`
- `data/processed/reports/topk_sweep_report.md`
- per-(method, k) extraction CSVs and error-analysis JSONs

## How to run
From the repo root, in the env where YAKE + spaCy work:

```bash
source .venv/bin/activate
python src/run_benchmark_topk_sweep.py
```
