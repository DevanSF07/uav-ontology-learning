from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path("ontology_experiment_framework")
EXTRACTION_DIR = PROJECT_DIR / "stage_outputs/03_non_llm_extraction/by_method"
GROUND_TRUTH_FILE = PROJECT_DIR / "data/Ground_Truth.xlsx"
OUTPUT_DIR = PROJECT_DIR / "stage_outputs/03c_weighted_voting"

METHOD_FILES = {
    "yake": EXTRACTION_DIR / "yake/candidates_unique_ranked.csv",
    "tfidf": EXTRACTION_DIR / "tfidf/candidates_unique_ranked.csv",
    "textrank": EXTRACTION_DIR / "textrank/candidates_unique_ranked.csv",
    "noun_phrases": EXTRACTION_DIR / "noun_phrases/candidates_unique_ranked.csv",
    "keybert_minilm": EXTRACTION_DIR / "keybert_minilm/candidates_unique_ranked.csv",
    "keybert_mpnet": EXTRACTION_DIR / "keybert_mpnet/candidates_unique_ranked.csv",
}

# Performance-informed weights. MiniLM and MPNet are capped as one embedding family.
METHOD_WEIGHTS = {
    "yake": 5.0,
    "textrank": 3.0,
    "tfidf": 1.5,
    "noun_phrases": 1.0,
}
EMBEDDING_WEIGHTS = {
    "keybert_minilm": 2.0,
    "keybert_mpnet": 3.0,
}
PRIORITY_WEIGHTS = {"high": 3.0, "medium": 2.0, "low": 1.0}
MANUAL_REVIEW_STATUS = "needs_manual_review"


def normalize_term(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).casefold().strip()
    for dash in ("‐", "‑", "‒", "–", "—"):
        text = text.replace(dash, "-")
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"[^a-z0-9&/+ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_inputs() -> None:
    missing = [str(path) for path in METHOD_FILES.values() if not path.is_file()]
    if not GROUND_TRUTH_FILE.is_file():
        missing.append(str(GROUND_TRUTH_FILE))
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_ground_truth():
    gt = pd.read_excel(
        GROUND_TRUTH_FILE, sheet_name="S1_GT_TERMS", engine="openpyxl"
    )
    negatives = pd.read_excel(
        GROUND_TRUTH_FILE, sheet_name="S7_NEGATIVE_TERMS", engine="openpyxl"
    )
    gt = gt[
        gt["keep_for_tasks"].astype(str).str.contains(
            "term_extraction", case=False, na=False
        )
    ].copy()
    gt["normalized_gt"] = gt["term_canonical"].map(normalize_term)
    gt = gt[gt["normalized_gt"].ne("")].copy()
    status = gt["grounded_status_v2"].fillna("").astype(str).str.casefold()
    strict = gt[status.ne(MANUAL_REVIEW_STATUS)].copy()
    review = gt[status.eq(MANUAL_REVIEW_STATUS)].copy()
    negatives["normalized_negative"] = negatives["candidate"].map(normalize_term)
    return strict, review, negatives


def load_support_matrix() -> pd.DataFrame:
    method_frames = {}
    all_terms = set()

    for method, path in METHOD_FILES.items():
        frame = pd.read_csv(path, encoding="utf-8")
        required = {"candidate", "normalized_candidate", "score", "global_rank"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["normalized_candidate"] = frame["normalized_candidate"].map(
            normalize_term
        )
        frame = frame[frame["normalized_candidate"].ne("")]
        frame = frame.sort_values("global_rank").drop_duplicates(
            "normalized_candidate", keep="first"
        )
        method_frames[method] = frame.set_index("normalized_candidate")
        all_terms.update(frame["normalized_candidate"])

    support = pd.DataFrame({"normalized_candidate": sorted(all_terms)})

    for method, lookup in method_frames.items():
        support[f"found_by_{method}"] = support["normalized_candidate"].isin(
            lookup.index
        )
        support[f"{method}_rank"] = support["normalized_candidate"].map(
            lookup["global_rank"]
        )
        support[f"{method}_score"] = support["normalized_candidate"].map(
            lookup["score"]
        )

    surface_priority = [
        "yake", "keybert_mpnet", "textrank", "keybert_minilm", "tfidf",
        "noun_phrases",
    ]

    def surface(term):
        for method in surface_priority:
            lookup = method_frames[method]
            if term in lookup.index:
                return lookup.at[term, "candidate"]
        return term

    def source_value(term, column, default=""):
        for method in surface_priority:
            lookup = method_frames[method]
            if term in lookup.index and column in lookup.columns:
                value = lookup.at[term, column]
                if not pd.isna(value):
                    return value
        return default

    support["candidate"] = support["normalized_candidate"].map(surface)
    support["source_sentence"] = support["normalized_candidate"].map(
        lambda term: source_value(term, "source_sentence")
    )
    support["context_text"] = support["normalized_candidate"].map(
        lambda term: source_value(term, "context_text")
    )

    found_columns = [f"found_by_{method}" for method in METHOD_FILES]
    support["method_vote_count"] = support[found_columns].sum(axis=1)
    support["embedding_model_count"] = support[
        ["found_by_keybert_minilm", "found_by_keybert_mpnet"]
    ].sum(axis=1)

    support["embedding_score"] = support.apply(
        lambda row: max(
            EMBEDDING_WEIGHTS["keybert_mpnet"]
            if row["found_by_keybert_mpnet"] else 0.0,
            EMBEDDING_WEIGHTS["keybert_minilm"]
            if row["found_by_keybert_minilm"] else 0.0,
        ),
        axis=1,
    )

    support["weighted_vote_score"] = (
        METHOD_WEIGHTS["yake"] * support["found_by_yake"].astype(int)
        + METHOD_WEIGHTS["textrank"] * support["found_by_textrank"].astype(int)
        + METHOD_WEIGHTS["tfidf"] * support["found_by_tfidf"].astype(int)
        + METHOD_WEIGHTS["noun_phrases"]
        * support["found_by_noun_phrases"].astype(int)
        + support["embedding_score"]
    )

    support["family_statistical"] = (
        support["found_by_yake"] | support["found_by_tfidf"]
    )
    support["family_graph"] = support["found_by_textrank"]
    support["family_syntactic"] = support["found_by_noun_phrases"]
    support["family_embedding"] = (
        support["found_by_keybert_minilm"]
        | support["found_by_keybert_mpnet"]
    )
    support["method_family_count"] = support[
        [
            "family_statistical", "family_graph", "family_syntactic",
            "family_embedding",
        ]
    ].sum(axis=1)

    rank_columns = [f"{method}_rank" for method in METHOD_FILES]
    support["best_source_rank"] = support[rank_columns].min(axis=1, skipna=True)
    support["average_source_rank"] = support[rank_columns].mean(axis=1, skipna=True)
    return support


def build_rules(support: pd.DataFrame):
    yake = support["found_by_yake"]
    mpnet = support["found_by_keybert_mpnet"]
    minilm = support["found_by_keybert_minilm"]
    textrank = support["found_by_textrank"]

    rules = {
        "B00_yake_and_mpnet_baseline": yake & mpnet,
        "L01_yake_or_textrank_support": yake & (mpnet | textrank),
        "L02_yake_mpnet_textrank_strict": yake & mpnet & textrank,
        "L03_yake_any_embedding": yake & (mpnet | minilm),
        "F02_at_least_2_families": support["method_family_count"] >= 2,
        "F03_at_least_3_families": support["method_family_count"] >= 3,
        "FY02_yake_at_least_2_families": yake
        & (support["method_family_count"] >= 2),
        "FY03_yake_at_least_3_families": yake
        & (support["method_family_count"] >= 3),
    }

    for threshold in range(3, 11):
        rules[f"W{threshold:02d}_score_ge_{threshold}"] = (
            support["weighted_vote_score"] >= threshold
        )

    for threshold in range(6, 11):
        rules[f"WY{threshold:02d}_yake_score_ge_{threshold}"] = yake & (
            support["weighted_vote_score"] >= threshold
        )

    for votes in range(2, 7):
        rules[f"V{votes}_at_least_{votes}_of_6"] = (
            support["method_vote_count"] >= votes
        )

    return rules


def evaluate(rule_name, candidates, strict_gt, review_gt, negatives):
    strict_unique = strict_gt.drop_duplicates("normalized_gt").copy()
    strict_lookup = strict_unique.set_index("normalized_gt")
    review_lookup = review_gt.drop_duplicates("normalized_gt").set_index(
        "normalized_gt"
    )
    negative_set = set(negatives["normalized_negative"].dropna())

    details = candidates.copy()
    details["rule"] = rule_name
    details["match_type"] = details["normalized_candidate"].map(
        lambda term: "strict_exact" if term in strict_lookup.index else (
            "manual_review_exact" if term in review_lookup.index else "unmatched"
        )
    )
    details["evaluation_flag"] = details["match_type"].map(
        {"strict_exact": "TP", "manual_review_exact": "REVIEW", "unmatched": "FP"}
    )
    details["matched_gt_id"] = details["normalized_candidate"].map(
        lambda term: strict_lookup.at[term, "gt_id"]
        if term in strict_lookup.index else (
            review_lookup.at[term, "gt_id"] if term in review_lookup.index else ""
        )
    )
    details["negative_benchmark_hit"] = details["normalized_candidate"].isin(
        negative_set
    )

    scored = details[details["match_type"].ne("manual_review_exact")]
    predicted = set(scored["normalized_candidate"])
    gold = set(strict_unique["normalized_gt"])
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    weights = (
        strict_unique["priority"].fillna("low").astype(str).str.casefold()
        .map(PRIORITY_WEIGHTS).fillna(1.0)
    )
    matched = strict_unique["normalized_gt"].isin(predicted)
    weighted_recall = float(weights[matched].sum() / weights.sum()) if weights.sum() else 0.0

    metric = {
        "rule": rule_name,
        "candidate_count": len(details),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "priority_weighted_recall": weighted_recall,
        "manual_review_matches": int(details["match_type"].eq("manual_review_exact").sum()),
        "negative_term_hits": int(details["negative_benchmark_hit"].sum()),
        "average_vote_count": float(details["method_vote_count"].mean()) if len(details) else 0.0,
        "average_family_count": float(details["method_family_count"].mean()) if len(details) else 0.0,
        "average_weighted_score": float(details["weighted_vote_score"].mean()) if len(details) else 0.0,
    }
    return metric, details


def format_workbook(path: Path):
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    book = load_workbook(path)
    for sheet in book.worksheets:
        sheet.freeze_panes = "A2"
        sheet.sheet_view.showGridLines = False
        if sheet.max_row and sheet.max_column:
            sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for col in range(1, sheet.max_column + 1):
            lengths = [len(str(sheet.cell(row, col).value or "")) for row in range(1, min(sheet.max_row, 150) + 1)]
            sheet.column_dimensions[get_column_letter(col)].width = min(max(max(lengths, default=10) + 2, 11), 42)
        if sheet.title == "Metrics Summary":
            headers = {sheet.cell(1, col).value: col for col in range(1, sheet.max_column + 1)}
            for name in ("precision", "recall", "f1", "priority_weighted_recall"):
                if name in headers:
                    for row in range(2, sheet.max_row + 1):
                        sheet.cell(row, headers[name]).number_format = "0.00%"
    book.save(path)


def main():
    validate_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rules_dir = OUTPUT_DIR / "by_rule"
    rules_dir.mkdir(parents=True, exist_ok=True)

    strict_gt, review_gt, negatives = load_ground_truth()
    support = load_support_matrix()
    rules = build_rules(support)

    metrics = []
    all_details = []
    seen_sets = {}
    duplicate_rules = []

    for rule_name, mask in rules.items():
        selected = support[mask].copy()
        signature = tuple(sorted(selected["normalized_candidate"]))
        if signature in seen_sets:
            duplicate_rules.append({"rule": rule_name, "duplicate_of": seen_sets[signature]})
            continue
        seen_sets[signature] = rule_name

        selected = selected.sort_values(
            ["weighted_vote_score", "method_family_count", "best_source_rank"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        selected["rule_rank"] = range(1, len(selected) + 1)

        metric, details = evaluate(
            rule_name, selected, strict_gt, review_gt, negatives
        )
        metrics.append(metric)
        all_details.append(details)

        rule_dir = rules_dir / rule_name
        rule_dir.mkdir(parents=True, exist_ok=True)
        selected.to_csv(rule_dir / "candidates.csv", index=False)
        details.to_csv(rule_dir / "evaluation.csv", index=False)
        pd.DataFrame([metric]).to_csv(rule_dir / "metrics.csv", index=False)

    metrics_df = pd.DataFrame(metrics).sort_values(
        ["f1", "precision", "recall"], ascending=[False, False, False]
    ).reset_index(drop=True)
    details_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    duplicates_df = pd.DataFrame(duplicate_rules)

    support.to_csv(OUTPUT_DIR / "method_support_matrix.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "weighted_voting_metrics.csv", index=False)
    details_df.to_csv(OUTPUT_DIR / "all_rule_evaluations.csv", index=False)
    duplicates_df.to_csv(OUTPUT_DIR / "duplicate_rules.csv", index=False)

    best_rule = metrics_df.iloc[0]["rule"] if not metrics_df.empty else None
    if best_rule:
        best_source = rules_dir / best_rule / "candidates.csv"
        pd.read_csv(best_source).to_csv(OUTPUT_DIR / "best_f1_candidates.csv", index=False)

    report = OUTPUT_DIR / "weighted_voting_report.xlsx"
    readme = pd.DataFrame({
        "Field": ["Purpose", "Weights", "Embedding handling", "Duplicate rules", "Best rule"],
        "Value": [
            "Performance-informed weighted voting over six non-LLM extractors.",
            json.dumps(METHOD_WEIGHTS),
            "MiniLM and MPNet are one capped family; maximum embedding contribution is 3.0.",
            "Rules yielding identical candidate sets are evaluated once and logged separately.",
            best_rule or "No rule completed",
        ],
    })
    with pd.ExcelWriter(report, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics Summary", index=False)
        duplicates_df.to_excel(writer, sheet_name="Duplicate Rules", index=False)
        support.to_excel(writer, sheet_name="Method Support", index=False)
        details_df.to_excel(writer, sheet_name="All Evaluations", index=False)
    format_workbook(report)

    manifest = {
        "stage": "03c_weighted_voting",
        "method_weights": METHOD_WEIGHTS,
        "embedding_weights": EMBEDDING_WEIGHTS,
        "best_rule_by_f1": best_rule,
        "rules_tested": len(rules),
        "unique_candidate_sets_evaluated": len(metrics_df),
        "duplicate_rules": duplicate_rules,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Stage 03c weighted voting completed")
    print("=" * 80)
    print(metrics_df.head(15).to_string(index=False))
    print(f"\nBest rule: {best_rule}")
    print(f"Best candidate file: {OUTPUT_DIR / 'best_f1_candidates.csv'}")
    print(f"Excel report: {report}")


if __name__ == "__main__":
    main()
