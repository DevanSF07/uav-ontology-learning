# src/run_benchmark_local_llm.py

from __future__ import annotations

import csv
from pathlib import Path

from benchmark_config import (
    INTERIM_DIR,
    EXTRACTIONS_DIR,
    EVALUATION_DIR,
    REPORTS_DIR,
    CHAPTER_FILES,
    GOLD_FILE,
)
from gold_loader import load_gold_json, build_gold_term_set, build_gold_corpus_text
from evaluation import evaluate_predictions_flat, write_error_analysis
from local_llm_extractors import call_ollama_extract_terms_plain
from performance import RuntimeMetrics, timed_call


# Edit this list to match the models you have in Ollama
LLM_MODELS = [
   "qwen2.5:7b",
   "llama3.1:8b",
    "mistral:7b",
]


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

    corpus_text, combined_path = build_combined_corpus(CHAPTER_FILES)
    print(f"Combined corpus written to: {combined_path}")

    gold_items = load_gold_json(GOLD_FILE)
    gold_term_set = build_gold_term_set(gold_items)
    print(f"Loaded {len(gold_term_set)} flattened gold terms from: {GOLD_FILE}")

    gold_corpus_path = INTERIM_DIR / "gold_corpus.txt"
    gold_corpus_path.write_text(build_gold_corpus_text(gold_items), encoding="utf-8")
    print(f"Gold corpus written to: {gold_corpus_path}")

    metrics_rows = []
    runtime_rows = []

    for model in LLM_MODELS:
        method_name = f"llm_{model.replace(':', '_').replace('/', '_')}"
        print(f"\n=== Running local LLM extractor: {model} ===")

        (pred_rows, usage, raw_text), elapsed = timed_call(
            call_ollama_extract_terms_plain, model, corpus_text
        )

        extraction_path = EXTRACTIONS_DIR / f"{method_name}_terms.csv"
        write_csv(extraction_path, pred_rows, ["term", "score", "method", "normalized_term"])
        print(f"Extraction saved to: {extraction_path}")

        metrics, error_analysis = evaluate_predictions_flat(pred_rows, gold_term_set)
        metrics_row = {"method": method_name, **metrics}
        metrics_rows.append(metrics_row)

        error_path = EVALUATION_DIR / f"error_analysis_{method_name}.json"
        write_error_analysis(error_path, error_analysis)
        print(f"Error analysis saved to: {error_path}")

        runtime = RuntimeMetrics(
            method=method_name,
            wall_clock_sec=round(elapsed, 4),
            n_pred_unique=len({r["normalized_term"] for r in pred_rows}),
            total_duration_ns=usage.get("total_duration_ns"),
            load_duration_ns=usage.get("load_duration_ns"),
            prompt_eval_count=usage.get("prompt_eval_count"),
            prompt_eval_duration_ns=usage.get("prompt_eval_duration_ns"),
            eval_count=usage.get("eval_count"),
            eval_duration_ns=usage.get("eval_duration_ns"),
            generated_tokens_per_sec=usage.get("generated_tokens_per_sec"),
        )
        runtime_rows.append(runtime.to_row())

        print(
            f"Metrics for {method_name}: "
            f"P={metrics['precision']}, R={metrics['recall']}, F1={metrics['f1']} | "
            f"wall_clock_sec={runtime.wall_clock_sec} | "
            f"tokens_per_sec={runtime.generated_tokens_per_sec}"
        )

    metrics_path = EVALUATION_DIR / "metrics_local_llm.csv"
    write_csv(
        metrics_path,
        metrics_rows,
        ["method", "n_gold_terms", "n_pred_unique", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    print(f"\nLocal LLM metrics table saved to: {metrics_path}")

    runtime_path = EVALUATION_DIR / "runtime_local_llm.csv"
    write_csv(
        runtime_path,
        runtime_rows,
        [
            "method",
            "wall_clock_sec",
            "n_pred_unique",
            "total_duration_ns",
            "load_duration_ns",
            "prompt_eval_count",
            "prompt_eval_duration_ns",
            "eval_count",
            "eval_duration_ns",
            "generated_tokens_per_sec",
        ],
    )
    print(f"Runtime table saved to: {runtime_path}")

    report_path = REPORTS_DIR / "local_llm_report.md"
    lines = [
        "# Local LLM Term Extraction Benchmark",
        "",
        f"- Combined corpus: `{combined_path}`",
        f"- Gold JSON: `{GOLD_FILE}`",
        f"- Models: `{', '.join(LLM_MODELS)}`",
        "",
        "## Accuracy Results",
        "",
        "| Method | Precision | Recall | F1 | TP | FP | FN | Predicted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in metrics_rows:
        lines.append(
            f"| {row['method']} | {row['precision']} | {row['recall']} | "
            f"{row['f1']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['n_pred_unique']} |"
        )

    lines += [
        "",
        "## Runtime Results",
        "",
        "| Method | Wall Clock (s) | Prompt Tokens | Output Tokens | Tokens/s | Total Duration (ns) | Load Duration (ns) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in runtime_rows:
        lines.append(
            f"| {row['method']} | {row['wall_clock_sec']} | {row['prompt_eval_count']} | "
            f"{row['eval_count']} | {row['generated_tokens_per_sec']} | "
            f"{row['total_duration_ns']} | {row['load_duration_ns']} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()