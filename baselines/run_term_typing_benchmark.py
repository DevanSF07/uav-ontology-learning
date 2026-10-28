from __future__ import annotations

import csv
from pathlib import Path
from itertools import islice

import pandas as pd

from context_retrieval import build_sentence_index, retrieve_context_for_term
from evaluate_term_typing import (
    load_gold_term_labels,
    evaluate_pure_typing,
    evaluate_end_to_end,
)
from local_llm_term_typing import call_ollama_term_typing_batch
from performance import RuntimeMetrics, timed_call

# Try to reuse your project config if available.
try:
    from benchmark_config import (
        CHAPTER_FILES,
        INTERIM_DIR,
        EXTRACTIONS_DIR,
        EVALUATION_DIR,
        REPORTS_DIR,
    )
except Exception:
    CHAPTER_FILES = [Path("data/raw/chapter1.txt"), Path("data/raw/chapter2.txt")]
    INTERIM_DIR = Path("data/processed/interim")
    EXTRACTIONS_DIR = Path("data/processed/extractions")
    EVALUATION_DIR = Path("data/processed/evaluation")
    REPORTS_DIR = Path("data/processed/reports")

try:
    from normalization import normalize_term
except Exception:
    import re

    def normalize_term(text: str) -> str:
        text = str(text).lower().strip()
        text = text.replace("_", " ").replace("-", " ").replace("/", " ")
        text = re.sub(r"[^a-z0-9\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


# -----------------------
# USER SETTINGS
# -----------------------
MODEL_NAME = "llama3.1:8b"
GOLD_LABELS_CSV = Path("/home/faujdar/onto_gen/data/gold/terms_labels_GT.csv")

# Use the extracted-term CSV you want to classify here:
EXTRACTED_TERMS_CSV = Path("data/processed/extractions/llm_llama3.1_8b_terms.csv")

# Example alternative:
# EXTRACTED_TERMS_CSV = Path(
#     "data/processed/extractions/ensemble_yake_spacy_regex_acronym_pattern_union_terms.csv"
# )

BATCH_SIZE = 20
MAX_CONTEXT_HITS = 3
SENTENCE_WINDOW = 1
TIMEOUT_SEC = 600
NUM_CTX = 8192


def ensure_dirs():
    for p in [INTERIM_DIR, EXTRACTIONS_DIR, EVALUATION_DIR, REPORTS_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def build_combined_corpus(paths):
    texts = []
    for p in paths:
        path = Path(p)
        if path.exists():
            texts.append(read_text(path).strip())

    text = "\n\n".join(t for t in texts if t).strip()

    out = Path(INTERIM_DIR) / "combined_corpus.txt"
    out.write_text(text, encoding="utf-8")
    return text, out


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def batched(iterable, n):
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, n))
        if not batch:
            break
        yield batch


def load_extracted_terms(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Extracted terms file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "term" not in df.columns:
        raise ValueError(f"Expected a 'term' column in {csv_path}")

    df["term"] = df["term"].astype(str).str.strip()

    if "normalized_term" not in df.columns:
        df["normalized_term"] = df["term"].map(normalize_term)

    df = df[["term", "normalized_term"]].drop_duplicates()
    return df


def main():
    ensure_dirs()

    corpus_text, combined_path = build_combined_corpus(CHAPTER_FILES)
    if not corpus_text:
        raise RuntimeError(
            "No combined corpus text found. Check CHAPTER_FILES or your input text files."
        )
    print(f"Combined corpus written to: {combined_path}")

    gold_df = load_gold_term_labels(GOLD_LABELS_CSV, normalize_term)
    print(f"Loaded {len(gold_df)} gold term-label rows from: {GOLD_LABELS_CSV}")

    extracted_df = load_extracted_terms(EXTRACTED_TERMS_CSV)
    print(f"Loaded {len(extracted_df)} extracted terms from: {EXTRACTED_TERMS_CSV}")

    sentence_index = build_sentence_index(corpus_text, normalize_term)
    print(f"Built sentence index with {len(sentence_index)} sentences")

    # Build term -> context rows
    term_rows = []
    for i, row in extracted_df.iterrows():
        ctx = retrieve_context_for_term(
            term=row["term"],
            term_norm=row["normalized_term"],
            sentence_index=sentence_index,
            window=SENTENCE_WINDOW,
            max_hits=MAX_CONTEXT_HITS,
        )

        term_rows.append(
            {
                "id": i,
                "term": row["term"],
                "normalized_term": row["normalized_term"],
                "context": ctx or "No direct context match found in the corpus.",
            }
        )

    all_preds = []
    runtime_rows = []
    raw_outputs_rows = []

    for batch_num, batch in enumerate(batched(term_rows, BATCH_SIZE), start=1):
        print(f"\n=== Typing batch {batch_num} | size={len(batch)} ===")

        (preds, usage, raw_text), elapsed = timed_call(
            call_ollama_term_typing_batch,
            model=MODEL_NAME,
            batch_rows=batch,
            timeout_sec=TIMEOUT_SEC,
            num_ctx=NUM_CTX,
        )

        all_preds.extend(preds)

        runtime_rows.append(
            RuntimeMetrics(
                method=f"typing_{MODEL_NAME.replace(':', '_')}",
                wall_clock_sec=round(elapsed, 4),
                n_pred_unique=len(preds),
                total_duration_ns=usage.get("total_duration_ns"),
                load_duration_ns=usage.get("load_duration_ns"),
                prompt_eval_count=usage.get("prompt_eval_count"),
                prompt_eval_duration_ns=usage.get("prompt_eval_duration_ns"),
                eval_count=usage.get("eval_count"),
                eval_duration_ns=usage.get("eval_duration_ns"),
                generated_tokens_per_sec=usage.get("generated_tokens_per_sec"),
            ).to_row()
        )

        raw_outputs_rows.append(
            {
                "batch": batch_num,
                "raw_output": raw_text,
            }
        )

    pred_df = pd.DataFrame(all_preds)
    pred_df["normalized_term"] = pred_df["term"].map(normalize_term)

    # Save predictions
    pred_path = Path(EVALUATION_DIR) / "predicted_term_labels.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predicted term labels saved to: {pred_path}")

    # Save batch runtimes
    runtime_path = Path(EVALUATION_DIR) / "term_typing_runtime.csv"
    pd.DataFrame(runtime_rows).to_csv(runtime_path, index=False)
    print(f"Typing runtime saved to: {runtime_path}")

    raw_path = Path(EVALUATION_DIR) / "term_typing_raw_outputs.csv"
    pd.DataFrame(raw_outputs_rows).to_csv(raw_path, index=False)
    print(f"Raw LLM outputs saved to: {raw_path}")

    # Evaluation 1: pure typing on overlap only
    pure_summary, pure_merged_df, cm_df, report_df = evaluate_pure_typing(
        pred_df, gold_df
    )

    pure_summary_path = Path(EVALUATION_DIR) / "term_typing_metrics_pure.csv"
    pd.DataFrame([pure_summary]).to_csv(pure_summary_path, index=False)

    pure_eval_path = Path(EVALUATION_DIR) / "term_typing_overlap_eval.csv"
    pure_merged_df.to_csv(pure_eval_path, index=False)

    cm_path = Path(EVALUATION_DIR) / "term_typing_confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    report_path = Path(EVALUATION_DIR) / "term_typing_classification_report.csv"
    report_df.to_csv(report_path, index=False)

    print(f"Pure typing metrics saved to: {pure_summary_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Per-class report saved to: {report_path}")

    # Evaluation 2: end-to-end label-pair evaluation
    e2e_summary, e2e_pair_df, e2e_summary_df = evaluate_end_to_end(pred_df, gold_df)

    e2e_summary_path = Path(EVALUATION_DIR) / "term_typing_metrics_end_to_end.csv"
    e2e_pairs_path = Path(EVALUATION_DIR) / "term_typing_end_to_end_pairs.csv"

    e2e_summary_df.to_csv(e2e_summary_path, index=False)
    e2e_pair_df.to_csv(e2e_pairs_path, index=False)

    print(f"End-to-end metrics saved to: {e2e_summary_path}")

    # Write report
    report_md = Path(REPORTS_DIR) / "term_typing_report.md"
    lines = [
        "# Term Typing Benchmark Report",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Gold labels CSV: `{GOLD_LABELS_CSV}`",
        f"- Extracted terms CSV: `{EXTRACTED_TERMS_CSV}`",
        f"- Combined corpus: `{combined_path}`",
        "",
        "## Pure typing (overlap-only) summary",
        "",
        f"- n_items: `{pure_summary['n_items']}`",
        f"- accuracy: `{pure_summary['accuracy']}`",
        f"- macro_precision: `{pure_summary['macro_precision']}`",
        f"- macro_recall: `{pure_summary['macro_recall']}`",
        f"- macro_f1: `{pure_summary['macro_f1']}`",
        f"- weighted_precision: `{pure_summary['weighted_precision']}`",
        f"- weighted_recall: `{pure_summary['weighted_recall']}`",
        f"- weighted_f1: `{pure_summary['weighted_f1']}`",
        "",
        "## End-to-end (term,label pair) summary",
        "",
        f"- n_gold_pairs: `{e2e_summary['n_gold_pairs']}`",
        f"- n_pred_pairs: `{e2e_summary['n_pred_pairs']}`",
        f"- tp: `{e2e_summary['tp']}`",
        f"- fp: `{e2e_summary['fp']}`",
        f"- fn: `{e2e_summary['fn']}`",
        f"- precision: `{e2e_summary['precision']}`",
        f"- recall: `{e2e_summary['recall']}`",
        f"- f1: `{e2e_summary['f1']}`",
        "",
        "## Output files",
        f"- Predictions: `{pred_path}`",
        f"- Confusion matrix: `{cm_path}`",
        f"- Per-class report: `{report_path}`",
        f"- Runtime: `{runtime_path}`",
    ]

    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to: {report_md}")


if __name__ == "__main__":
    main()