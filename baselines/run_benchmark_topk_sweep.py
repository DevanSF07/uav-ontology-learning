from __future__ import annotations

import csv
from pathlib import Path

from benchmark_config import (
    CHAPTER_FILES,
    GOLD_FILE,
    INTERIM_DIR,
    EXTRACTIONS_DIR,
    EVALUATION_DIR,
    REPORTS_DIR,
)
from gold_loader import load_gold_json, build_gold_term_set, build_gold_corpus_text
from normalization import normalize_term
from extractors import run_method
from evaluation import evaluate_predictions_flat, write_error_analysis


# --- Only one method ---
SWEEP_METHODS = [
    "ensemble_yake_spacy_regex_acronym_pattern_union",
]

# --- Fixed K values ---
SWEEP_TOP_KS = [25, 50, 75, 100, 150, 200]


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
        parts.append(read_text(p).strip())

    text = "\n\n".join(parts).strip()

    out = INTERIM_DIR / "combined_corpus.txt"
    out.write_text(text, encoding="utf-8")
    return text, out


def main():
    ensure_dirs()

    # 1) combine both chapters into one corpus
    corpus_text, combined_path = build_combined_corpus(CHAPTER_FILES)
    print(f"Combined corpus written to: {combined_path}")

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

    # 3) run sweep for fixed top-k values
    for method in SWEEP_METHODS:
        for top_k in SWEEP_TOP_KS:
            print(f"\n=== Running method: {method} | top_k={top_k} ===")
            pred_rows = run_method(method, corpus_text, normalize_term, top_k=top_k)

            extraction_path = EXTRACTIONS_DIR / f"{method}_topk_{top_k}_terms.csv"
            write_csv(
                extraction_path,
                pred_rows,
                fieldnames=["term", "score", "method", "normalized_term"],
            )
            print(f"Extraction saved to: {extraction_path}")

            metrics, error_analysis = evaluate_predictions_flat(pred_rows, gold_term_set)
            metrics_row = {"method": method, "top_k": top_k, **metrics}
            metrics_rows.append(metrics_row)

            error_path = EVALUATION_DIR / f"error_analysis_{method}_topk_{top_k}.json"
            write_error_analysis(error_path, error_analysis)
            print(f"Error analysis saved to: {error_path}")

            print(
                f"Metrics for {method} @ k={top_k}: "
                f"P={metrics['precision']}, R={metrics['recall']}, F1={metrics['f1']}"
            )

        # 4) extra run for ALL predictions
        print(f"\n=== Running method: {method} | top_k=ALL ===")
        pred_rows_all = run_method(method, corpus_text, normalize_term, top_k=None)

        top_k_all = len(pred_rows_all)

        extraction_path_all = EXTRACTIONS_DIR / f"{method}_topk_{top_k_all}_terms.csv"
        write_csv(
            extraction_path_all,
            pred_rows_all,
            fieldnames=["term", "score", "method", "normalized_term"],
        )
        print(f"Extraction saved to: {extraction_path_all}")

        metrics_all, error_analysis_all = evaluate_predictions_flat(pred_rows_all, gold_term_set)
        metrics_row_all = {"method": method, "top_k": top_k_all, **metrics_all}
        metrics_rows.append(metrics_row_all)

        error_path_all = EVALUATION_DIR / f"error_analysis_{method}_topk_{top_k_all}.json"
        write_error_analysis(error_path_all, error_analysis_all)
        print(f"Error analysis saved to: {error_path_all}")

        print(
            f"Metrics for {method} @ k={top_k_all} (ALL): "
            f"P={metrics_all['precision']}, "
            f"R={metrics_all['recall']}, "
            f"F1={metrics_all['f1']}"
        )

    # 5) save sweep metrics table
    metrics_path = EVALUATION_DIR / "metrics_topk_sweep_single_method.csv"
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
    print(f"\nSweep metrics table saved to: {metrics_path}")

    # 6) save best result
    best = sorted(
        metrics_rows,
        key=lambda r: (-r["f1"], -r["recall"], -r["precision"], r["top_k"])
    )[0]

    best_path = EVALUATION_DIR / "metrics_topk_best_single_method.csv"
    write_csv(
        best_path,
        [best],
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
    print(f"Best result saved to: {best_path}")

    # 7) markdown report
    report_path = REPORTS_DIR / "topk_sweep_single_method_report.md"
    lines = [
        "# Top-K Sweep Report (Single Method)",
        "",
        f"- Method: `{SWEEP_METHODS[0]}`",
        f"- Combined corpus: `{combined_path}`",
        f"- Gold JSON: `{GOLD_FILE}`",
        f"- Gold vocabulary size: `{len(gold_term_set)}`",
        "",
        "## Best result",
        "",
        "| Method | Best Top-K | Precision | Recall | F1 | TP | FP | FN | Predicted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {best['method']} | {best['top_k']} | {best['precision']} | {best['recall']} | {best['f1']} | {best['tp']} | {best['fp']} | {best['fn']} | {best['n_pred_unique']} |",
        "",
        "## Full sweep results",
        "",
        "| Method | Top-K | Precision | Recall | F1 | TP | FP | FN | Predicted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in sorted(metrics_rows, key=lambda r: r["top_k"]):
        lines.append(
            f"| {row['method']} | {row['top_k']} | {row['precision']} | {row['recall']} | "
            f"{row['f1']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['n_pred_unique']} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()