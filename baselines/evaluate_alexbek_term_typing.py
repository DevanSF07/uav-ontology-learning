from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


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


def normalize_term(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = pd.Series([text]).str.replace(r"[^a-z0-9\s]+", " ", regex=True).iloc[0]
    text = pd.Series([text]).str.replace(r"\s+", " ", regex=True).iloc[0]
    return str(text).strip()


def load_gold(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"term", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df["term"] = df["term"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["normalized_term"] = df["term"].map(normalize_term)

    df = df[df["normalized_term"].astype(str).str.strip() != ""]
    return df[["term", "normalized_term", "label"]].drop_duplicates()


def load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "term" not in df.columns:
        raise ValueError(f"Expected a 'term' column in {csv_path}")

    if "predicted_label" in df.columns:
        label_col = "predicted_label"
    elif "label" in df.columns:
        label_col = "label"
    else:
        raise ValueError(f"Expected a 'label' or 'predicted_label' column in {csv_path}")

    df["term"] = df["term"].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df["normalized_term"] = df.get("normalized_term", df["term"].map(normalize_term))
    df["normalized_term"] = df["normalized_term"].astype(str).str.strip()

    return df[["term", "normalized_term", label_col]].drop_duplicates()


def compute_classification_metrics(
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    pred_label_col: str,
    drop_unknown: bool = False,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = pred_df.copy()
    unknown_removed = 0

    if drop_unknown:
        unknown_mask = working[pred_label_col] == "unknown"
        unknown_removed = int(unknown_mask.sum())
        working = working.loc[~unknown_mask].copy()

    merged = working.merge(
        gold_df[["normalized_term", "label"]].rename(columns={"label": "gold_label"}),
        on="normalized_term",
        how="inner",
    ).rename(columns={pred_label_col: "pred_label"})

    y_true = merged["gold_label"].tolist()
    y_pred = merged["pred_label"].tolist()

    if y_true:
        report = classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
        accuracy = round(accuracy_score(y_true, y_pred), 4)
    else:
        report = {}
        accuracy = 0.0

    summary = {
        "mode": "alexbek_typing_no_unknown" if drop_unknown else "alexbek_typing",
        "n_pred_rows_original": len(pred_df),
        "n_pred_rows_after_filter": len(working),
        "n_unknown_removed": unknown_removed,
        "n_gold_rows": len(gold_df),
        "n_overlap_rows": len(merged),
        "coverage": round(len(merged) / len(gold_df), 4) if len(gold_df) else 0.0,
        "accuracy": accuracy,
        "micro_precision": round(report.get("accuracy", 0.0), 4),
        "micro_recall": round(report.get("accuracy", 0.0), 4),
        "micro_f1": round(report.get("accuracy", 0.0), 4),
        "macro_precision": round(report.get("macro avg", {}).get("precision", 0.0), 4),
        "macro_recall": round(report.get("macro avg", {}).get("recall", 0.0), 4),
        "macro_f1": round(report.get("macro avg", {}).get("f1-score", 0.0), 4),
        "weighted_precision": round(report.get("weighted avg", {}).get("precision", 0.0), 4),
        "weighted_recall": round(report.get("weighted avg", {}).get("recall", 0.0), 4),
        "weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0.0), 4),
    }

    cm_df = pd.DataFrame()
    report_df = pd.DataFrame()
    overlap_df = merged.copy()

    if y_true:
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        cm_df = pd.DataFrame(
            cm,
            index=[f"gold::{label}" for label in LABELS],
            columns=[f"pred::{label}" for label in LABELS],
        )

        rows = []
        for label in LABELS:
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
        report_df = pd.DataFrame(rows)

    return summary, overlap_df, cm_df, report_df


def save_confusion_matrix_image(cm_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if cm_df.empty:
        return

    fig_width = max(8, 0.8 * len(cm_df.columns) + 2)
    fig_height = max(6, 0.6 * len(cm_df.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)

    image = ax.imshow(cm_df.values, cmap="Blues")
    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_xticklabels([label.replace("pred::", "") for label in cm_df.columns], rotation=45, ha="right")
    ax.set_yticklabels([label.replace("gold::", "") for label in cm_df.index])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_title(title)

    threshold = cm_df.values.max() / 2 if cm_df.values.size else 0
    for row_idx in range(cm_df.shape[0]):
        for col_idx in range(cm_df.shape[1]):
            value = int(cm_df.iat[row_idx, col_idx])
            color = "white" if value > threshold else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def discover_prediction_files(values: Iterable[str] | None) -> list[Path]:
    if values:
        return [Path(v) for v in values]

    candidates = []
    for path in [
        Path("uav_zero_shot_results.csv"),
        Path("uav_few_shot_results.csv"),
        Path("uav_hybrid_results.csv"),
    ]:
        if path.exists():
            candidates.append(path)

    if candidates:
        return candidates

    return sorted(Path(".").glob("uav*_results.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Alexbek term typing results against gold labels")
    parser.add_argument("--gold", default="data/gold/terms_labels_GT.csv", help="Gold CSV with term,label")
    parser.add_argument(
        "--pred",
        nargs="*",
        help="Prediction CSV files to evaluate. If omitted, the script auto-discovers Alexbek outputs.",
    )
    parser.add_argument("--outdir", default="data/processed/evaluation", help="Directory for evaluation outputs")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gold_df = load_gold(gold_path)
    pred_paths = discover_prediction_files(args.pred)

    if not pred_paths:
        raise FileNotFoundError(
            "No prediction CSV files found. Provide --pred or create uav_zero_shot_results.csv / uav_hybrid_results.csv."
        )

    summary_rows = []

    for pred_path in pred_paths:
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")

        pred_df = load_predictions(pred_path)
        label_col = "predicted_label" if "predicted_label" in pred_df.columns else "label"

        summary, overlap_df, cm_df, report_df = compute_classification_metrics(
            pred_df=pred_df,
            gold_df=gold_df,
            pred_label_col=label_col,
            drop_unknown=False,
        )

        summary["prediction_file"] = str(pred_path)
        summary_rows.append(summary)

        stem = pred_path.stem
        overlap_df.to_csv(outdir / f"{stem}_overlap_eval.csv", index=False)
        if not cm_df.empty:
            cm_df.to_csv(outdir / f"{stem}_confusion_matrix.csv")
            save_confusion_matrix_image(
                cm_df,
                outdir / f"{stem}_confusion_matrix.png",
                title=f"Alexbek confusion matrix: {stem}",
            )
        if not report_df.empty:
            report_df.to_csv(outdir / f"{stem}_classification_report.csv", index=False)

        # Optional no-unknown view for easier comparison with the existing benchmark scripts.
        summary_no_unknown, overlap_no_unknown_df, cm_no_unknown_df, report_no_unknown_df = compute_classification_metrics(
            pred_df=pred_df,
            gold_df=gold_df,
            pred_label_col=label_col,
            drop_unknown=True,
        )
        summary_no_unknown["prediction_file"] = str(pred_path)
        summary_rows.append(summary_no_unknown)

        overlap_no_unknown_df.to_csv(outdir / f"{stem}_overlap_eval_no_unknown.csv", index=False)
        if not cm_no_unknown_df.empty:
            cm_no_unknown_df.to_csv(outdir / f"{stem}_confusion_matrix_no_unknown.csv")
            save_confusion_matrix_image(
                cm_no_unknown_df,
                outdir / f"{stem}_confusion_matrix_no_unknown.png",
                title=f"Alexbek confusion matrix no unknown: {stem}",
            )
        if not report_no_unknown_df.empty:
            report_no_unknown_df.to_csv(outdir / f"{stem}_classification_report_no_unknown.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(outdir / "alexbek_classification_metrics.csv", index=False)
    print(f"Wrote evaluation summary to: {outdir / 'alexbek_classification_metrics.csv'}")


if __name__ == "__main__":
    main()