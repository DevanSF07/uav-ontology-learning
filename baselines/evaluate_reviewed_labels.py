from __future__ import annotations

from pathlib import Path
import pandas as pd
import re

from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# Normalization (same style as rest of pipeline)
# --------------------------------------------------
def normalize_term(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# Paths (update if needed)
# --------------------------------------------------
REVIEWED_CSV = Path("data/processed/evaluation/reviewed_term_labels_with_confidence.csv")
GOLD_CSV = Path("/home/faujdar/onto_gen/data/gold/terms_labels_GT.csv")

OUTDIR = Path("data/processed/evaluation")


# --------------------------------------------------
# Load GOLD
# --------------------------------------------------
def load_gold(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["term"] = df["term"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df["normalized_term"] = df["term"].map(normalize_term)

    df = df[["normalized_term", "label"]].drop_duplicates()
    return df


# --------------------------------------------------
# Load REVIEWED
# --------------------------------------------------
def load_reviewed(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"term", "reviewed_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in reviewed CSV: {missing}")

    df["term"] = df["term"].astype(str).str.strip()
    df["reviewed_label"] = df["reviewed_label"].astype(str).str.strip().str.lower()

    if "normalized_term" not in df.columns:
        df["normalized_term"] = df["term"].map(normalize_term)

    df = df[["term", "normalized_term", "reviewed_label"]].drop_duplicates()

    return df


# --------------------------------------------------
# TERM-ONLY METRICS (after removing unknown)
# --------------------------------------------------
def compute_term_only_metrics(filtered_df: pd.DataFrame, gold_df: pd.DataFrame):
    gold_terms = set(gold_df["normalized_term"])
    pred_terms = set(filtered_df["normalized_term"])

    tp = pred_terms & gold_terms
    fp = pred_terms - gold_terms
    fn = gold_terms - pred_terms

    precision = len(tp) / len(pred_terms) if pred_terms else 0.0
    recall = len(tp) / len(gold_terms) if gold_terms else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    summary = {
        "mode": "term_only_reviewed_no_unknown",
        "n_gold_terms": len(gold_terms),
        "n_pred_terms": len(pred_terms),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    return summary


# --------------------------------------------------
# CLASSIFICATION METRICS (on filtered terms only)
# --------------------------------------------------
def compute_classification_metrics(filtered_df: pd.DataFrame, gold_df: pd.DataFrame):
    merged = filtered_df.merge(
        gold_df,
        on="normalized_term",
        how="inner"
    )

    if merged.empty:
        return {"mode": "classification_reviewed_no_unknown", "n_items": 0}

    y_true = merged["label"]
    y_pred = merged["reviewed_label"]

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    summary = {
        "mode": "classification_reviewed_no_unknown",
        "n_items": len(merged),
        "accuracy": round(accuracy, 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_precision": round(report["weighted avg"]["precision"], 4),
        "weighted_recall": round(report["weighted avg"]["recall"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
    }

    return summary, merged, report


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    reviewed_df = load_reviewed(REVIEWED_CSV)
    gold_df = load_gold(GOLD_CSV)

    print(f"Loaded reviewed rows: {len(reviewed_df)}")
    print(f"Loaded gold rows: {len(gold_df)}")

    # --------------------------------------------------
    # Remove unknown
    # --------------------------------------------------
    filtered_df = reviewed_df[reviewed_df["reviewed_label"] != "unknown"].copy()

    print(f"\nAfter removing 'unknown': {len(filtered_df)} rows")

    # --------------------------------------------------
    # TERM-ONLY METRICS
    # --------------------------------------------------
    term_summary = compute_term_only_metrics(filtered_df, gold_df)

    print("\n=== Term-only metrics (reviewed, no unknown) ===")
    for k, v in term_summary.items():
        print(f"{k}: {v}")

    pd.DataFrame([term_summary]).to_csv(
        OUTDIR / "review_term_only_metrics.csv",
        index=False
    )

    # --------------------------------------------------
    # CLASSIFICATION METRICS
    # --------------------------------------------------
    class_summary, merged_df, report = compute_classification_metrics(filtered_df, gold_df)

    print("\n=== Classification metrics (reviewed, no unknown) ===")
    for k, v in class_summary.items():
        print(f"{k}: {v}")

    pd.DataFrame([class_summary]).to_csv(
        OUTDIR / "review_classification_metrics.csv",
        index=False
    )

    # Save detailed classification report
    report_rows = []
    for label, vals in report.items():
        if isinstance(vals, dict):
            report_rows.append({
                "label": label,
                "precision": round(vals.get("precision", 0), 4),
                "recall": round(vals.get("recall", 0), 4),
                "f1": round(vals.get("f1-score", 0), 4),
                "support": int(vals.get("support", 0)),
            })

    pd.DataFrame(report_rows).to_csv(
        OUTDIR / "review_classification_report.csv",
        index=False
    )

    # Save merged dataset for inspection
    merged_df.to_csv(
        OUTDIR / "review_classification_overlap.csv",
        index=False
    )


if __name__ == "__main__":
    main()