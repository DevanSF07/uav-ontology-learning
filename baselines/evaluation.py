import json
from pathlib import Path
from typing import List, Set


def evaluate_predictions_flat(pred_rows: List[dict], gold_term_set: Set[str]):
    """
    Flat surface-form evaluation:
    - gold_term_set contains all canonical terms + aliases
    - predicted terms are compared directly against that set

    Metrics:
    - TP = predicted normalized terms that are in gold_term_set
    - FP = predicted normalized terms not in gold_term_set
    - FN = gold terms not predicted
    """
    pred_norms = []
    for row in pred_rows:
        norm = row.get("normalized_term", "").strip()
        if norm:
            pred_norms.append(norm)

    # Deduplicate predictions
    pred_norms = list(dict.fromkeys(pred_norms))

    pred_set = set(pred_norms)

    tp_terms = pred_set & gold_term_set
    fp_terms = pred_set - gold_term_set
    fn_terms = gold_term_set - pred_set

    tp = len(tp_terms)
    fp = len(fp_terms)
    fn = len(fn_terms)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "n_gold_terms": len(gold_term_set),
        "n_pred_unique": len(pred_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    error_analysis = {
        "matched_terms": sorted(tp_terms),
        "false_positives": sorted(fp_terms),
        "false_negatives": sorted(fn_terms),
    }

    return metrics, error_analysis


def write_error_analysis(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)