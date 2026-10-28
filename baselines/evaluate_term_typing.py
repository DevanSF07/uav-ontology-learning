from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = [
    "mission",
    "system",
    "environment",
    "capability",
    "part",
    "domain",
    "parameter",
    "unknown",
]


def load_gold_term_labels(csv_path: str | Path, normalize_fn) -> pd.DataFrame:
    """
    Load the gold term-label CSV and normalize terms.
    Expected columns:
      - term
      - label
    """
    df = pd.read_csv(csv_path)
    df["term"] = df["term"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["normalized_term"] = df["term"].map(normalize_fn)

    return df[["term", "normalized_term", "label"]].drop_duplicates()


def evaluate_pure_typing(
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pure typing evaluation on overlap only.

    Only evaluates terms that are present in both:
    - predicted term set
    - gold term set

    Returns:
      summary_metrics,
      merged_eval_df,
      confusion_df,
      class_report_df
    """
    merged = pred_df.merge(
        gold_df[["normalized_term", "label"]].rename(columns={"label": "gold_label"}),
        on="normalized_term",
        how="inner",
    )

    merged = merged.rename(columns={"predicted_label": "pred_label"})

    y_true = merged["gold_label"].tolist()
    y_pred = merged["pred_label"].tolist()

    summary = build_metrics_summary(y_true, y_pred, labels=LABELS)
    cm_df = build_confusion_df(y_true, y_pred, labels=LABELS)
    report_df = build_report_df(y_true, y_pred, labels=LABELS)

    return summary, merged, cm_df, report_df


def evaluate_end_to_end(
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end evaluation on (normalized_term, label) pairs.

    A predicted pair is correct only if BOTH:
    - term matches gold normalized term
    - predicted label matches gold label
    """
    gold_pairs = set(zip(gold_df["normalized_term"], gold_df["label"]))
    pred_pairs = set(zip(pred_df["normalized_term"], pred_df["predicted_label"]))

    tp_pairs = pred_pairs & gold_pairs
    fp_pairs = pred_pairs - gold_pairs
    fn_pairs = gold_pairs - pred_pairs

    precision = len(tp_pairs) / len(pred_pairs) if pred_pairs else 0.0
    recall = len(tp_pairs) / len(gold_pairs) if gold_pairs else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    summary = {
        "mode": "end_to_end",
        "n_gold_pairs": len(gold_pairs),
        "n_pred_pairs": len(pred_pairs),
        "tp": len(tp_pairs),
        "fp": len(fp_pairs),
        "fn": len(fn_pairs),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    pair_rows = []

    for term, label in sorted(tp_pairs):
        pair_rows.append(
            {
                "normalized_term": term,
                "label": label,
                "status": "tp",
            }
        )

    for term, label in sorted(fp_pairs):
        pair_rows.append(
            {
                "normalized_term": term,
                "label": label,
                "status": "fp",
            }
        )

    for term, label in sorted(fn_pairs):
        pair_rows.append(
            {
                "normalized_term": term,
                "label": label,
                "status": "fn",
            }
        )

    return summary, pd.DataFrame(pair_rows), pd.DataFrame([summary])


def build_metrics_summary(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict:
    """
    Build overall classification summary metrics.
    """
    if not y_true:
        return {
            "mode": "pure_typing",
            "n_items": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
        }

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "mode": "pure_typing",
        "n_items": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_precision": round(report["weighted avg"]["precision"], 4),
        "weighted_recall": round(report["weighted avg"]["recall"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
    }


def build_confusion_df(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> pd.DataFrame:
    """
    Build confusion matrix as a labeled DataFrame.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return pd.DataFrame(
        cm,
        index=[f"gold::{label}" for label in labels],
        columns=[f"pred::{label}" for label in labels],
    )


def build_report_df(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> pd.DataFrame:
    """
    Build per-class precision / recall / F1 / support table.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for label in labels:
        if label in report:
            rows.append(
                {
                    "label": label,
                    "precision": round(report[label]["precision"], 4),
                    "recall": round(report[label]["recall"], 4),
                    "f1": round(report[label]["f1-score"], 4),
                    "support": int(report[label]["support"]),
                }
            )

    return pd.DataFrame(rows)