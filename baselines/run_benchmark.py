from __future__ import annotations

import csv
from pathlib import Path
from preprocess import clean_text

from benchmark_config import (
    CHAPTER_FILES,
    GOLD_FILE,
    INTERIM_DIR,
    EXTRACTIONS_DIR,
    EVALUATION_DIR,
    REPORTS_DIR,
    DEFAULT_METHODS,
    TOP_K,
)
from gold_loader import load_gold_json, build_gold_term_set, build_gold_corpus_text
from normalization import normalize_term
from extractors import run_method
from evaluation import evaluate_predictions_flat, write_error_analysis


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_dirs():
    for p in [INTERIM_DIR, EXTRACTIONS_DIR, EVALUATION_DIR, REPORTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def build_combined_corpus(paths):
    parts = []
    for p in paths:
        raw_text = read_text(p)
        cleaned_text = clean_text(raw_text)
        parts.append(cleaned_text)

    text = "\n\n".join(parts).strip()

    out = INTERIM_DIR / "combined_corpus.txt"
    out.write_text(text, encoding="utf-8")
    return text, out

def build_raw_combined_corpus(paths):
    parts = []
    for p in paths:
        parts.append(read_text(p).strip())

    text = "\n\n".join(parts).strip()

    out = INTERIM_DIR / "combined_corpus_raw.txt"
    out.write_text(text, encoding="utf-8")
    return text, out


def main():
    ensure_dirs()

    # 1) combine both chapters into one corpus
    corpus_text, combined_path = build_combined_corpus(CHAPTER_FILES)
    raw_corpus_text, raw_combined_path = build_raw_combined_corpus(CHAPTER_FILES)
    print(f"Raw combined corpus written to: {raw_combined_path}")
    print(f"Cleaned combined corpus written to: {combined_path}")
    
    
    

    # 2) load combined gold JSON and flatten all terms + aliases
    gold_items = load_gold_json(GOLD_FILE)
    gold_term_set = build_gold_term_set(gold_items)
    print(f"Loaded {len(gold_term_set)} flattened gold terms from: {GOLD_FILE}")

    # optional: save flat gold corpus for inspection
    gold_corpus_text = build_gold_corpus_text(gold_items)
    gold_corpus_path = INTERIM_DIR / "gold_corpus.txt"
    gold_corpus_path.write_text(gold_corpus_text, encoding="utf-8")
    print(f"Gold corpus written to: {gold_corpus_path}")

    metrics_rows = []

    # 3) run each extractor and evaluate
    for method in DEFAULT_METHODS:
        print(f"\n=== Running method: {method} ===")
        pred_rows = run_method(method, corpus_text, normalize_term, top_k=TOP_K)

        # save extraction output
        extraction_path = EXTRACTIONS_DIR / f"{method}_terms.csv"
        write_csv(
            extraction_path,
            pred_rows,
            fieldnames=["term", "score", "method", "normalized_term"]
        )
        print(f"Extraction saved to: {extraction_path}")

        # evaluate against flat gold term set
        metrics, error_analysis = evaluate_predictions_flat(pred_rows, gold_term_set)
        metrics_row = {"method": method, "top_k": TOP_K, **metrics}
        metrics_rows.append(metrics_row)

        # save per-method error analysis
        error_path = EVALUATION_DIR / f"error_analysis_{method}.json"
        write_error_analysis(error_path, error_analysis)
        print(f"Error analysis saved to: {error_path}")
        print(
            f"Metrics for {method}: "
            f"P={metrics['precision']}, "
            f"R={metrics['recall']}, "
            f"F1={metrics['f1']}"
        )

    # 4) save combined metrics table
    metrics_path = EVALUATION_DIR / "metrics.csv"
    write_csv(
        metrics_path,
        metrics_rows,
        fieldnames=[
            "method",
            "top_k",
            "n_gold_terms",
            "n_pred_unique",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ],
    )
    print(f"\nMetrics table saved to: {metrics_path}")

    # 5) simple markdown report
    report_path = REPORTS_DIR / "comparison_report.md"
    lines = [
        "# Term Extraction Benchmark Report",
        "",
        f"- Combined corpus: `{combined_path}`",
        f"- Gold JSON: `{GOLD_FILE}`",
        f"- Flattened gold vocabulary size: `{len(gold_term_set)}`",
        f"- Top-K: `{TOP_K}`",
        "",
        "| Method | Precision | Recall | F1 | TP | FP | FN | Predicted | Gold Terms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in metrics_rows:
        lines.append(
            f"| {row['method']} | {row['precision']} | {row['recall']} | "
            f"{row['f1']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['n_pred_unique']} | {row['n_gold_terms']} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
