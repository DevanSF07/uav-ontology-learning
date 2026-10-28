from pathlib import Path
import pandas as pd
import re
from sklearn.metrics import accuracy_score, classification_report


# ==================================================
# Normalize
# ==================================================
def normalize_term(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==================================================
# TERM METRICS (FLAT)
# ==================================================
def term_metrics(pred_list, gold_set):
    pred_list = [t for t in pred_list if t]
    pred_set = set(pred_list)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return round(precision, 4), round(recall, 4), round(f1, 4), tp, fp, fn


# ==================================================
# CLASSIFICATION METRICS (FULL)
# ==================================================
def class_metrics(df, gold_df, label_col):

    merged = df.merge(gold_df, on="normalized_term", how="inner")

    if merged.empty:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
        }

    acc = accuracy_score(merged["label"], merged[label_col])

    report = classification_report(
        merged["label"],
        merged[label_col],
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_precision": round(report["weighted avg"]["precision"], 4),
        "weighted_recall": round(report["weighted avg"]["recall"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
    }


# ==================================================
# LOAD GOLD
# ==================================================
def load_gold():
    path = Path("/home/faujdar/onto_gen/data/gold/terms_labels_GT.csv")
    df = pd.read_csv(path)
    df["normalized_term"] = df["term"].map(normalize_term)
    return df[["normalized_term", "label"]].drop_duplicates()


# ==================================================
# MAIN
# ==================================================
def main():

    gold_df = load_gold()
    gold_terms = set(gold_df["normalized_term"])

    term_results = []
    class_results = []

    # ==================================================
    # ✅ 1. ENSEMBLE (TERM ONLY)
    # ==================================================
    ens_path = Path(
        "/home/faujdar/onto_gen/data/processed/extractions/ensemble_yake_spacy_regex_acronym_pattern_union_topk_5453_terms.csv"
    )

    ens = pd.read_csv(ens_path)

    # detect column
    col = None
    for c in ["term", "phrase", "keyword"]:
        if c in ens.columns:
            col = c
            break

    if col is None:
        raise ValueError("No valid term column in ensemble file")

    ens["normalized_term"] = ens[col].map(normalize_term)
    ens = ens[ens["normalized_term"].str.strip() != ""]

    p, r, f, tp, fp, fn = term_metrics(
        ens["normalized_term"].tolist(),
        gold_terms
    )

    term_results.append({
        "method": "ensemble",
        "stage": "all",
        "precision": p,
        "recall": r,
        "f1": f,
        "tp": tp,
        "fp": fp,
        "fn": fn
    })


    # ==================================================
    # ✅ 2. LLAMA (INITIAL)
    # ==================================================
    pred = pd.read_csv("data/processed/evaluation/predicted_term_labels.csv")
    pred["normalized_term"] = pred["term"].map(normalize_term)

    # --- BEFORE UNKNOWN
    p, r, f, tp, fp, fn = term_metrics(
        pred["normalized_term"].tolist(),
        gold_terms
    )

    term_results.append({
        "method": "llama",
        "stage": "before_unknown",
        "precision": p,
        "recall": r,
        "f1": f
    })

    class_results.append({
        "method": "llama",
        "stage": "before_unknown",
        **class_metrics(pred, gold_df, "predicted_label")
    })

    # --- AFTER UNKNOWN
    pred_f = pred[pred["predicted_label"] != "unknown"]

    p, r, f, tp, fp, fn = term_metrics(
        pred_f["normalized_term"].tolist(),
        gold_terms
    )

    term_results.append({
        "method": "llama",
        "stage": "after_unknown",
        "precision": p,
        "recall": r,
        "f1": f
    })

    class_results.append({
        "method": "llama",
        "stage": "after_unknown",
        **class_metrics(pred_f, gold_df, "predicted_label")
    })


    # ==================================================
    # ✅ 3. REVIEW
    # ==================================================
    rev = pd.read_csv("data/processed/evaluation/reviewed_term_labels_with_confidence.csv")
    rev["normalized_term"] = rev["term"].map(normalize_term)

    # --- BEFORE UNKNOWN
    p, r, f, tp, fp, fn = term_metrics(
        rev["normalized_term"].tolist(),
        gold_terms
    )

    term_results.append({
        "method": "review",
        "stage": "before_unknown",
        "precision": p,
        "recall": r,
        "f1": f
    })

    class_results.append({
        "method": "review",
        "stage": "before_unknown",
        **class_metrics(rev, gold_df, "reviewed_label")
    })

    # --- AFTER UNKNOWN
    rev_f = rev[rev["reviewed_label"] != "unknown"]

    p, r, f, tp, fp, fn = term_metrics(
        rev_f["normalized_term"].tolist(),
        gold_terms
    )

    term_results.append({
        "method": "review",
        "stage": "after_unknown",
        "precision": p,
        "recall": r,
        "f1": f
    })

    class_results.append({
        "method": "review",
        "stage": "after_unknown",
        **class_metrics(rev_f, gold_df, "reviewed_label")
    })


    # ==================================================
    # SAVE
    # ==================================================
    outdir = Path("data/processed/results/final_all")
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(term_results).to_csv(outdir / "term_metrics_all.csv", index=False)
    pd.DataFrame(class_results).to_csv(outdir / "classification_metrics_all.csv", index=False)

    print("\n✅ ALL 3 METHODS EVALUATED SUCCESSFULLY")
    print(f"📁 Results saved in: {outdir}")


if __name__ == "__main__":
    main()
