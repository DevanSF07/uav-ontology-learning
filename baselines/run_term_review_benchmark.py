from __future__ import annotations

from pathlib import Path
import csv
import pandas as pd

from context_retrieval_large import build_sentence_index, retrieve_large_context_for_term
from local_llm_term_review import call_ollama_term_review
from performance import RuntimeMetrics, timed_call

# Try to reuse your config if available
try:
    from benchmark_config import (
        CHAPTER_FILES,
        INTERIM_DIR,
        EVALUATION_DIR,
        REPORTS_DIR,
    )
except Exception:
    CHAPTER_FILES = [Path("data/raw/chapter1.txt"), Path("data/raw/chapter2.txt")]
    INTERIM_DIR = Path("data/processed/interim")
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


# ==================================================
# USER SETTINGS
# ==================================================
MODEL_NAME = "llama3.1:8b"

# Existing term-label prediction file from the first classification stage
PREDICTED_TERM_LABELS_CSV = Path("data/processed/evaluation/predicted_term_labels.csv")

# New output files
REVIEWED_OUTPUT_CSV = Path("data/processed/evaluation/reviewed_term_labels_with_confidence.csv")
REVIEW_RUNTIME_CSV = Path("data/processed/evaluation/reviewed_term_labels_runtime.csv")
REVIEW_RAW_OUTPUTS_CSV = Path("data/processed/evaluation/reviewed_term_labels_raw_outputs.csv")
REVIEW_REPORT_MD = Path("data/processed/reports/term_review_report.md")

# Large-context review settings
MAX_CONTEXT_SENTENCES = 10
MAX_CONTEXT_CHARS = 5000
TIMEOUT_SEC = 600
NUM_CTX = 8192


def ensure_dirs():
    for p in [INTERIM_DIR, EVALUATION_DIR, REPORTS_DIR]:
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


def load_predicted_term_labels(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Predicted term labels file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"term", "predicted_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    # ✅ Drop NaNs early
    df = df.dropna(subset=["term", "predicted_label"])

    df["term"] = df["term"].astype(str).str.strip()
    df["predicted_label"] = df["predicted_label"].astype(str).str.strip().str.lower()

    # ✅ Remove garbage terms
    df = df[df["term"] != ""]
    df = df[df["term"].str.lower() != "nan"]

    if "normalized_term" not in df.columns:
        df["normalized_term"] = df["term"].map(normalize_term)

    # ✅ Additional safety
    df = df[df["normalized_term"].astype(str).str.strip() != ""]

    df = df[["term", "normalized_term", "predicted_label"]].drop_duplicates()

    return df


def main():
    ensure_dirs()

    # --------------------------------------------------
    # 1) Load corpus and build sentence index
    # --------------------------------------------------
    corpus_text, combined_path = build_combined_corpus(CHAPTER_FILES)
    if not corpus_text:
        raise RuntimeError(
            "No combined corpus text found. Check CHAPTER_FILES or input text files."
        )

    print(f"Combined corpus written to: {combined_path}")

    sentence_index = build_sentence_index(corpus_text, normalize_term)
    print(f"Built sentence index with {len(sentence_index)} sentences")

    # --------------------------------------------------
    # 2) Load initial predicted term labels
    # --------------------------------------------------
    pred_df = load_predicted_term_labels(PREDICTED_TERM_LABELS_CSV)
    print(f"Loaded {len(pred_df)} initial predicted term-label rows from: {PREDICTED_TERM_LABELS_CSV}")

    reviewed_rows = []
    runtime_rows = []
    raw_rows = []

    # --------------------------------------------------
    # 3) Review each term one by one
    # --------------------------------------------------
    for idx, row in pred_df.iterrows():
        term = row["term"]
        normalized_term = row["normalized_term"]
        initial_label = row["predicted_label"]

        print(f"[{idx + 1}/{len(pred_df)}] Reviewing term: {term} | initial_label={initial_label}")

        context_text, context_sentence_count = retrieve_large_context_for_term(
            term=term,
            term_norm=normalized_term,
            sentence_index=sentence_index,
            max_sentences=MAX_CONTEXT_SENTENCES,
            max_chars=MAX_CONTEXT_CHARS,
        )

        if not context_text:
            context_text = "No direct sentence-level evidence found in the corpus."

        (review_row, usage, raw_text), elapsed = timed_call(
            call_ollama_term_review,
            model=MODEL_NAME,
            term=term,
            initial_label=initial_label,
            context_text=context_text,
            timeout_sec=TIMEOUT_SEC,
            num_ctx=NUM_CTX,
        )

        reviewed_label = review_row["reviewed_label"]
        confidence_score = review_row["confidence_score"]
        review_decision = (
            "confirmed" if reviewed_label == initial_label else "changed"
        )

        reviewed_rows.append(
            {
                "term": term,
                "normalized_term": normalized_term,
                "initial_label": initial_label,
                "reviewed_label": reviewed_label,
                "confidence_score": confidence_score,
                "review_decision": review_decision,
                "context_sentence_count": context_sentence_count,
                "context_char_count": len(context_text),
                "model": MODEL_NAME,
            }
        )

        runtime_rows.append(
            RuntimeMetrics(
                method=f"review_{MODEL_NAME.replace(':', '_')}",
                wall_clock_sec=round(elapsed, 4),
                n_pred_unique=1,
                total_duration_ns=usage.get("total_duration_ns"),
                load_duration_ns=usage.get("load_duration_ns"),
                prompt_eval_count=usage.get("prompt_eval_count"),
                prompt_eval_duration_ns=usage.get("prompt_eval_duration_ns"),
                eval_count=usage.get("eval_count"),
                eval_duration_ns=usage.get("eval_duration_ns"),
                generated_tokens_per_sec=usage.get("generated_tokens_per_sec"),
            ).to_row()
        )

        raw_rows.append(
            {
                "term": term,
                "initial_label": initial_label,
                "raw_output": raw_text,
            }
        )

    # --------------------------------------------------
    # 4) Save outputs
    # --------------------------------------------------
    reviewed_df = pd.DataFrame(reviewed_rows)
    reviewed_df.to_csv(REVIEWED_OUTPUT_CSV, index=False)
    print(f"Reviewed term-label CSV saved to: {REVIEWED_OUTPUT_CSV}")

    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(REVIEW_RUNTIME_CSV, index=False)
    print(f"Review runtime CSV saved to: {REVIEW_RUNTIME_CSV}")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(REVIEW_RAW_OUTPUTS_CSV, index=False)
    print(f"Raw review outputs CSV saved to: {REVIEW_RAW_OUTPUTS_CSV}")

    # --------------------------------------------------
    # 5) Write summary report
    # --------------------------------------------------
    n_changed = int((reviewed_df["review_decision"] == "changed").sum())
    n_confirmed = int((reviewed_df["review_decision"] == "confirmed").sum())

    avg_confidence = round(float(reviewed_df["confidence_score"].mean()), 4) if not reviewed_df.empty else 0.0
    avg_context_sentences = round(float(reviewed_df["context_sentence_count"].mean()), 4) if not reviewed_df.empty else 0.0
    avg_context_chars = round(float(reviewed_df["context_char_count"].mean()), 4) if not reviewed_df.empty else 0.0

    report_lines = [
        "# Term Review Benchmark Report",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Initial term-label file: `{PREDICTED_TERM_LABELS_CSV}`",
        f"- Combined corpus: `{combined_path}`",
        f"- Max review context sentences: `{MAX_CONTEXT_SENTENCES}`",
        f"- Max review context chars: `{MAX_CONTEXT_CHARS}`",
        "",
        "## Review summary",
        "",
        f"- Total reviewed rows: `{len(reviewed_df)}`",
        f"- Confirmed labels: `{n_confirmed}`",
        f"- Changed labels: `{n_changed}`",
        f"- Average confidence: `{avg_confidence}`",
        f"- Average context sentence count: `{avg_context_sentences}`",
        f"- Average context char count: `{avg_context_chars}`",
        "",
        "## Output files",
        "",
        f"- Reviewed CSV: `{REVIEWED_OUTPUT_CSV}`",
        f"- Runtime CSV: `{REVIEW_RUNTIME_CSV}`",
        f"- Raw outputs CSV: `{REVIEW_RAW_OUTPUTS_CSV}`",
    ]

    REVIEW_REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Review report saved to: {REVIEW_REPORT_MD}")


if __name__ == "__main__":
    main()