from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
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


def normalize_term(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_gold_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["term"] = df["term"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["normalized_term"] = df["term"].map(normalize_term)
    return df[["term", "normalized_term", "label"]].drop_duplicates()


def load_preds(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "predicted_label" not in df.columns:
        raise ValueError(f"Expected column 'predicted_label' in {csv_path}")
    if "term" not in df.columns:
        raise ValueError(f"Expected column 'term' in {csv_path}")

    df["term"] = df["term"].astype(str).str.strip()
    df["predicted_label"] = df["predicted_label"].astype(str).str.strip().str.lower()

    if "normalized_term" not in df.columns:
        df["normalized_term"] = df["term"].map(normalize_term)

    return df[["term", "normalized_term", "predicted_label"]].drop_duplicates()


def extract_terms_from_json(obj, collected: set[str]):
    """
    Recursively extract likely term strings from a JSON structure.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_l = str(key).lower()

            if key_l in {
                "term",
                "name",
                "preferred_term",
                "preferred label",
                "preferred_label",
                "surface_form",
            }:
                if isinstance(value, str):
                    collected.add(value.strip())

            elif key_l in {
                "alias",
                "aliases",
                "synonym",
                "synonyms",
                "surface_forms",
                "alt_labels",
                "alternative_labels",
            }:
                if isinstance(value, str):
                    collected.add(value.strip())
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            collected.add(item.strip())

            if isinstance(value, (dict, list)):
                extract_terms_from_json(value, collected)

    elif isinstance(obj, list):
        for item in obj:
            extract_terms_from_json(item, collected)


def load_gold_terms_from_json(json_path: Path) -> pd.DataFrame:
    if not json_path.exists():
        raise FileNotFoundError(f"Gold JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    collected = set()
    extract_terms_from_json(data, collected)

    rows = []
    for term in sorted(collected):
        if term:
            rows.append(
                {
                    "term": term,
                    "normalized_term": normalize_term(term),
                }
            )

    df = pd.DataFrame(rows).drop_duplicates()
    df = df[df["normalized_term"].astype(str).str.strip() != ""].copy()
    return df


def metrics_summary(y_true, y_pred):
    if not y_true:
        return {
            "mode": "pure_typing_no_unknown",
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
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "mode": "pure_typing_no_unknown",
        "n_items": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_precision": round(report["weighted avg"]["precision"], 4),
        "weighted_recall": round(report["weighted avg"]["recall"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
    }


def compute_term_only_metrics(pred_terms: set[str], gold_terms: set[str], mode_name: str):
    tp_terms = pred_terms & gold_terms
    fp_terms = pred_terms - gold_terms
    fn_terms = gold_terms - pred_terms

    precision = len(tp_terms) / len(pred_terms) if pred_terms else 0.0
    recall = len(tp_terms) / len(gold_terms) if gold_terms else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    summary = {
        "mode": mode_name,
        "n_gold_terms": len(gold_terms),
        "n_pred_terms_after_filter": len(pred_terms),
        "tp": len(tp_terms),
        "fp": len(fp_terms),
        "fn": len(fn_terms),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    details = []
    for t in sorted(tp_terms):
        details.append({"normalized_term": t, "status": "tp"})
    for t in sorted(fp_terms):
        details.append({"normalized_term": t, "status": "fp"})
    for t in sorted(fn_terms):
        details.append({"normalized_term": t, "status": "fn"})

    return summary, pd.DataFrame(details)


def main():
    parser = argparse.ArgumentParser(
        description="Recompute all metrics after excluding unknown predictions"
    )
    parser.add_argument(
        "--pred",
        default="data/processed/evaluation/predicted_term_labels.csv",
        help="Predicted term labels CSV",
    )
    parser.add_argument(
        "--gold_csv",
        default="/home/faujdar/onto_gen/data/gold/terms_labels_GT.csv",
        help="Gold CSV with term,label",
    )
    parser.add_argument(
        "--gold_json",
        default="/home/faujdar/onto_gen/data/gold/terms_aliases 1.json",
        help="Gold JSON with terms+aliases",
    )
    parser.add_argument(
        "--outdir",
        default="data/processed/evaluation",
        help="Output directory",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred)
    gold_csv_path = Path(args.gold_csv)
    gold_json_path = Path(args.gold_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred_df = load_preds(pred_path)
    gold_csv_df = load_gold_csv(gold_csv_path)
    gold_json_terms_df = load_gold_terms_from_json(gold_json_path)

    # --------------------------------------------------
    # Remove rows classified as unknown
    # --------------------------------------------------
    pred_df_no_unknown = pred_df[pred_df["predicted_label"] != "unknown"].copy()
    removed = len(pred_df) - len(pred_df_no_unknown)

    # ==================================================
    # 1) PURE TYPING (overlap only) AFTER REMOVING UNKNOWN
    # ==================================================
    merged = pred_df_no_unknown.merge(
        gold_csv_df[["normalized_term", "label"]].rename(columns={"label": "gold_label"}),
        on="normalized_term",
        how="inner",
    ).rename(columns={"predicted_label": "pred_label"})

    y_true = merged["gold_label"].tolist()
    y_pred = merged["pred_label"].tolist()

    pure_summary = metrics_summary(y_true, y_pred)
    pure_summary["n_pred_rows_original"] = len(pred_df)
    pure_summary["n_pred_rows_after_filter"] = len(pred_df_no_unknown)
    pure_summary["n_unknown_removed"] = removed

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"gold::{l}" for l in LABELS],
        columns=[f"pred::{l}" for l in LABELS],
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_rows = []
    for label in LABELS:
        if label in report:
            report_rows.append(
                {
                    "label": label,
                    "precision": round(report[label]["precision"], 4),
                    "recall": round(report[label]["recall"], 4),
                    "f1": round(report[label]["f1-score"], 4),
                    "support": int(report[label]["support"]),
                }
            )
    report_df = pd.DataFrame(report_rows)

    # ==================================================
    # 2) END-TO-END AFTER REMOVING UNKNOWN
    #    (INCLUDING UNMATCHED CASES)
    # ==================================================
    gold_pairs = set(zip(gold_csv_df["normalized_term"], gold_csv_df["label"]))
    pred_pairs = set(
        zip(pred_df_no_unknown["normalized_term"], pred_df_no_unknown["predicted_label"])
    )
    pred_pairs_original = set(zip(pred_df["normalized_term"], pred_df["predicted_label"]))

    tp_pairs = pred_pairs & gold_pairs
    fp_pairs = pred_pairs - gold_pairs
    fn_pairs = gold_pairs - pred_pairs

    precision = len(tp_pairs) / len(pred_pairs) if pred_pairs else 0.0
    recall = len(tp_pairs) / len(gold_pairs) if gold_pairs else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    e2e_summary = {
        "mode": "end_to_end_no_unknown",
        "n_gold_pairs": len(gold_pairs),
        "n_pred_pairs_original": len(pred_pairs_original),
        "n_pred_pairs_after_filter": len(pred_pairs),
        "n_unknown_removed": removed,
        "tp": len(tp_pairs),
        "fp": len(fp_pairs),
        "fn": len(fn_pairs),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    pair_rows = []
    for t, l in sorted(tp_pairs):
        pair_rows.append({"normalized_term": t, "label": l, "status": "tp"})
    for t, l in sorted(fp_pairs):
        pair_rows.append({"normalized_term": t, "label": l, "status": "fp"})
    for t, l in sorted(fn_pairs):
        pair_rows.append({"normalized_term": t, "label": l, "status": "fn"})

    # ==================================================
    # 3) TERM-ONLY METRICS AFTER REMOVING UNKNOWN
    #    USING GOLD CSV TERMS
    # ==================================================
    pred_terms_original = set(pred_df["normalized_term"])
    pred_terms_after_filter = set(pred_df_no_unknown["normalized_term"])

    gold_terms_csv = set(gold_csv_df["normalized_term"])
    term_only_csv_summary, term_only_csv_details = compute_term_only_metrics(
        pred_terms_after_filter,
        gold_terms_csv,
        mode_name="term_only_no_unknown_from_csv",
    )
    term_only_csv_summary["n_pred_terms_original"] = len(pred_terms_original)
    term_only_csv_summary["n_unknown_rows_removed"] = removed

    # ==================================================
    # 4) TERM-ONLY METRICS AFTER REMOVING UNKNOWN
    #    USING GOLD JSON TERMS + ALIASES
    # ==================================================
    gold_terms_json = set(gold_json_terms_df["normalized_term"])
    term_only_json_summary, term_only_json_details = compute_term_only_metrics(
        pred_terms_after_filter,
        gold_terms_json,
        mode_name="term_only_no_unknown_from_json",
    )
    term_only_json_summary["n_pred_terms_original"] = len(pred_terms_original)
    term_only_json_summary["n_unknown_rows_removed"] = removed

    # ==================================================
    # SAVE OUTPUTS
    # ==================================================
    merged.to_csv(outdir / "term_typing_overlap_eval_no_unknown.csv", index=False)
    pd.DataFrame([pure_summary]).to_csv(
        outdir / "term_typing_metrics_pure_no_unknown.csv",
        index=False,
    )
    cm_df.to_csv(outdir / "term_typing_confusion_matrix_no_unknown.csv")
    report_df.to_csv(
        outdir / "term_typing_classification_report_no_unknown.csv",
        index=False,
    )

    pd.DataFrame([e2e_summary]).to_csv(
        outdir / "term_typing_metrics_end_to_end_no_unknown.csv",
        index=False,
    )
    pd.DataFrame(pair_rows).to_csv(
        outdir / "term_typing_end_to_end_pairs_no_unknown.csv",
        index=False,
    )

    pd.DataFrame([term_only_csv_summary]).to_csv(
        outdir / "term_only_metrics_no_unknown_from_csv.csv",
        index=False,
    )
    term_only_csv_details.to_csv(
        outdir / "term_only_eval_no_unknown_from_csv.csv",
        index=False,
    )

    pd.DataFrame([term_only_json_summary]).to_csv(
        outdir / "term_only_metrics_no_unknown_from_json.csv",
        index=False,
    )
    term_only_json_details.to_csv(
        outdir / "term_only_eval_no_unknown_from_json.csv",
        index=False,
    )

    # ==================================================
    # PRINT RESULTS
    # ==================================================
    print("=== Pure typing (no unknown) ===")
    for k, v in pure_summary.items():
        print(f"{k}: {v}")

    print("\n=== End-to-end (no unknown; unmatched cases included) ===")
    for k, v in e2e_summary.items():
        print(f"{k}: {v}")

    print("\n=== Term-only metrics after removing unknown (gold CSV terms) ===")
    for k, v in term_only_csv_summary.items():
        print(f"{k}: {v}")

    print("\n=== Term-only metrics after removing unknown (gold JSON terms+aliases) ===")
    for k, v in term_only_json_summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()