#!/usr/bin/env python3
"""Evaluate retained ontology terms against Ground_Truth.xlsx/S3_GT_CONCEPTS.

Purpose
-------
This is a structural-readiness demo after Qwen filtering. It asks:

1. How many S3 concepts are represented by at least one retained term?
2. What are term-level precision, recall, and F1 under normalized exact matching?
3. Which concepts are covered, missing, or supported by multiple terms?
4. Is concept recall high enough to begin taxonomy construction?

Important interpretation
------------------------
S3 is concept-level ground truth. Multiple extracted terms can support the same
concept. Therefore, the primary readiness statistic is CONCEPT RECALL:

    covered S3 concepts / total eligible S3 concepts

Term-level precision and F1 are also reported, but they can be conservative if
S3 does not enumerate every valid lexical variant or alias.

Default input
-------------
The script automatically reads the best Qwen policy and threshold from:

    stage_outputs/07b_qwen_validation_yake_mpnet/qwen_threshold_metrics.csv

It then reconstructs the corresponding retained inventory from:

    qwen_validation_results_with_gt.csv

You can instead pass a retained CSV directly with --input.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path("/home/faujdar/onto_gen_product/ontology_experiment_framework")
DEFAULT_RUN_DIR = PROJECT_DIR / "stage_outputs/07b_qwen_validation_yake_mpnet"
DEFAULT_GT = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_OUTPUT = PROJECT_DIR / "stage_outputs/08_s3_concept_coverage_demo"
S3_SHEET = "S3_GT_CONCEPTS"

# Candidate names are tried in order. CLI overrides remain available.
CONCEPT_ID_CANDIDATES = (
    "concept_id", "gt_concept_id", "id", "concept_code", "canonical_id"
)
CONCEPT_LABEL_CANDIDATES = (
    "concept_canonical", "canonical_concept", "concept_name", "preferred_label",
    "concept_label", "canonical_label", "label", "concept", "term_canonical"
)
ALIAS_COLUMN_HINTS = (
    "aliases", "alias", "synonyms", "synonym", "surface_forms", "surface_form",
    "terms", "term_variants", "lexicalizations", "alternative_labels",
    "alternate_labels", "abbreviations", "acronyms"
)
KEEP_COLUMN_CANDIDATES = (
    "keep_for_tasks", "include", "is_active", "keep", "eligible"
)
STATUS_COLUMN_CANDIDATES = (
    "grounded_status_v2", "status", "review_status", "concept_status"
)
MANUAL_REVIEW_VALUES = {
    "needs_manual_review", "needs manual review", "manual_review", "review"
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_term(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).casefold().strip()
    for dash in ("‐", "‑", "‒", "–", "—"):
        text = text.replace(dash, "-")
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"[^a-z0-9&/+ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {str(column).casefold(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def split_multi_value(value: object) -> list[str]:
    """Parse list-like or delimiter-separated alias cells safely."""
    if value is None or pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [safe_text(item) for item in value if safe_text(item)]

    text = safe_text(value)
    if not text:
        return []

    # Handle JSON/Python-list strings first.
    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [safe_text(item) for item in parsed if safe_text(item)]
            except Exception:
                pass

    # Avoid splitting ordinary multiword labels on commas unless a delimiter exists.
    if any(delimiter in text for delimiter in ("|", ";", "\n")):
        return [part.strip() for part in re.split(r"[|;\n]+", text) if part.strip()]
    return [text]


def infer_alias_columns(frame: pd.DataFrame, explicit: list[str] | None) -> list[str]:
    if explicit:
        missing = [column for column in explicit if column not in frame.columns]
        if missing:
            raise ValueError(f"Requested alias columns are missing: {missing}")
        return explicit

    selected = []
    for column in frame.columns:
        lowered = str(column).casefold()
        if any(hint == lowered or hint in lowered for hint in ALIAS_COLUMN_HINTS):
            selected.append(str(column))
    return selected


def load_s3_concepts(
    gt_file: Path,
    concept_id_column: str | None,
    concept_label_column: str | None,
    alias_columns: list[str] | None,
    include_manual_review: bool,
) -> tuple[pd.DataFrame, str, str, list[str]]:
    if not gt_file.is_file():
        raise FileNotFoundError(gt_file)

    s3 = pd.read_excel(gt_file, sheet_name=S3_SHEET, engine="openpyxl")
    if s3.empty:
        raise ValueError(f"{S3_SHEET} is empty")

    id_column = concept_id_column or find_column(s3, CONCEPT_ID_CANDIDATES)
    label_column = concept_label_column or find_column(s3, CONCEPT_LABEL_CANDIDATES)
    alias_columns = infer_alias_columns(s3, alias_columns)

    if label_column is None:
        raise ValueError(
            "Could not infer the S3 canonical concept-label column. Available columns: "
            + ", ".join(map(str, s3.columns))
            + ". Pass --concept-label-column explicitly."
        )
    if id_column is None:
        id_column = "__generated_concept_id"
        s3[id_column] = [f"S3_CONCEPT_{index:04d}" for index in range(1, len(s3) + 1)]

    # Apply optional eligibility columns only when their meaning is clear.
    keep_column = find_column(s3, KEEP_COLUMN_CANDIDATES)
    if keep_column:
        values = s3[keep_column].fillna("").astype(str).str.casefold()
        if values.isin({"true", "false", "1", "0", "yes", "no"}).any():
            keep_mask = values.isin({"true", "1", "yes"})
            if keep_mask.any():
                s3 = s3[keep_mask].copy()
        elif values.str.contains("taxonomy|concept|ontology", regex=True).any():
            s3 = s3[values.str.contains("taxonomy|concept|ontology", regex=True)].copy()

    if not include_manual_review:
        status_column = find_column(s3, STATUS_COLUMN_CANDIDATES)
        if status_column:
            status = s3[status_column].fillna("").astype(str).str.casefold().str.strip()
            s3 = s3[~status.isin(MANUAL_REVIEW_VALUES)].copy()

    records: list[dict[str, Any]] = []
    for row_number, row in s3.iterrows():
        concept_id = safe_text(row[id_column]) or f"S3_CONCEPT_{row_number + 1:04d}"
        canonical = safe_text(row[label_column])
        labels = [canonical] if canonical else []
        alias_sources: dict[str, list[str]] = {}
        for column in alias_columns:
            parsed = split_multi_value(row[column])
            if parsed:
                alias_sources[column] = parsed
                labels.extend(parsed)

        normalized_labels = sorted({normalize_term(label) for label in labels if normalize_term(label)})
        if not normalized_labels:
            continue

        record = {
            "concept_id": concept_id,
            "canonical_concept": canonical or labels[0],
            "normalized_canonical_concept": normalize_term(canonical or labels[0]),
            "all_labels": labels,
            "normalized_labels": normalized_labels,
            "alias_sources": alias_sources,
        }
        # Preserve useful S3 metadata.
        for column in s3.columns:
            if column not in {id_column, label_column, *alias_columns}:
                record[f"s3_{column}"] = row[column]
        records.append(record)

    concepts = pd.DataFrame(records)
    if concepts.empty:
        raise ValueError("No usable concepts or labels were extracted from S3_GT_CONCEPTS")
    if concepts["concept_id"].duplicated().any():
        concepts["concept_id"] = [
            f"{concept_id}__{index}" for index, concept_id in enumerate(concepts["concept_id"], 1)
        ]
    return concepts.reset_index(drop=True), id_column, label_column, alias_columns


def load_best_qwen_inventory(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics_file = run_dir / "qwen_threshold_metrics.csv"
    results_file = run_dir / "qwen_validation_results_with_gt.csv"
    if not metrics_file.is_file():
        raise FileNotFoundError(metrics_file)
    if not results_file.is_file():
        raise FileNotFoundError(results_file)

    metrics = pd.read_csv(metrics_file)
    if metrics.empty:
        raise ValueError("qwen_threshold_metrics.csv is empty")
    best = metrics.sort_values(
        ["f1", "precision", "recall"], ascending=[False, False, False]
    ).iloc[0]
    policy = str(best["policy"])
    threshold = float(best["threshold"])

    results = pd.read_csv(results_file)
    required = {
        "candidate", "normalized_candidate", "protected_decision", "llm_score",
        "confidence", "genericity", "ontology_suitability"
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Qwen result file is missing columns: {sorted(missing)}")

    hard_reject = (
        results["protected_decision"].eq("reject")
        & results["confidence"].ge(0.75)
        & results["genericity"].ge(0.60)
        & results["ontology_suitability"].le(0.40)
    )

    if policy == "baseline_no_qwen_filter":
        retain = pd.Series(True, index=results.index)
    elif policy == "safe_hard_rejection":
        retain = ~hard_reject
    elif policy == "retain_only_gated":
        retain = (
            results["protected_decision"].eq("retain")
            & results.get("candidate_is_concept", pd.Series(True, index=results.index)).astype(bool)
            & ~results.get("candidate_depends_on_context", pd.Series(False, index=results.index)).astype(bool)
            & ~results.get("is_incidental_named_entity", pd.Series(False, index=results.index)).astype(bool)
            & results["llm_score"].ge(threshold)
        )
    elif policy == "retain_or_review_gated":
        retain = (
            results["protected_decision"].isin(["retain", "review"])
            & ~hard_reject
            & results["llm_score"].ge(threshold)
        )
    else:
        raise ValueError(f"Unsupported best Qwen policy: {policy!r}")

    retained = results[retain].copy()
    metadata = {
        "selection_source": str(metrics_file),
        "qwen_policy": policy,
        "qwen_threshold": threshold,
        "qwen_term_f1": float(best["f1"]),
        "qwen_term_precision": float(best["precision"]),
        "qwen_term_recall": float(best["recall"]),
    }
    return retained, metadata


def load_input_inventory(input_file: Path | None, run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if input_file is None:
        return load_best_qwen_inventory(run_dir)

    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    frame = pd.read_csv(input_file, encoding="utf-8")
    if "candidate" not in frame.columns:
        raise ValueError("Input retained-term CSV must contain a candidate column")
    return frame, {"selection_source": str(input_file), "qwen_policy": "preselected_inventory"}


def make_label_index(concepts: pd.DataFrame) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for row in concepts.itertuples(index=False):
        for label in row.normalized_labels:
            index[label].add(str(row.concept_id))
    return index


def evaluate_concept_coverage(
    retained: pd.DataFrame,
    concepts: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    retained = retained.copy()
    retained["normalized_candidate"] = retained.get(
        "normalized_candidate", retained["candidate"]
    ).map(normalize_term)
    retained = retained[retained["normalized_candidate"].ne("")]
    retained = retained.drop_duplicates("normalized_candidate", keep="first").reset_index(drop=True)

    label_index = make_label_index(concepts)
    concept_lookup = concepts.set_index("concept_id")
    matches: list[dict[str, Any]] = []
    unmatched_rows = []

    for row in retained.itertuples(index=False):
        term = str(row.normalized_candidate)
        concept_ids = sorted(label_index.get(term, set()))
        if not concept_ids:
            unmatched_rows.append(row._asdict())
            continue
        for concept_id in concept_ids:
            concept = concept_lookup.loc[concept_id]
            match_type = (
                "canonical_exact"
                if term == concept["normalized_canonical_concept"]
                else "alias_exact"
            )
            matches.append({
                "candidate": row.candidate,
                "normalized_candidate": term,
                "concept_id": concept_id,
                "canonical_concept": concept["canonical_concept"],
                "match_type": match_type,
            })

    match_frame = pd.DataFrame(matches)
    unmatched = pd.DataFrame(unmatched_rows)
    matched_term_set = set(match_frame["normalized_candidate"]) if not match_frame.empty else set()
    predicted_term_set = set(retained["normalized_candidate"])
    covered_concepts = set(match_frame["concept_id"]) if not match_frame.empty else set()
    all_concepts = set(concepts["concept_id"])

    tp = len(matched_term_set)
    fp = len(predicted_term_set - matched_term_set)
    fn = len(all_concepts - covered_concepts)
    precision = tp / (tp + fp) if tp + fp else 0.0
    concept_recall = len(covered_concepts) / len(all_concepts) if all_concepts else 0.0
    f1 = (
        2 * precision * concept_recall / (precision + concept_recall)
        if precision + concept_recall else 0.0
    )

    concept_rows = []
    grouped_matches = (
        match_frame.groupby("concept_id") if not match_frame.empty else {}
    )
    for concept in concepts.itertuples(index=False):
        if concept.concept_id in covered_concepts:
            group = grouped_matches.get_group(concept.concept_id)
            supporting_terms = sorted(set(group["candidate"]))
            match_types = sorted(set(group["match_type"]))
        else:
            supporting_terms = []
            match_types = []
        concept_rows.append({
            "concept_id": concept.concept_id,
            "canonical_concept": concept.canonical_concept,
            "covered": concept.concept_id in covered_concepts,
            "supporting_term_count": len(supporting_terms),
            "supporting_terms": " | ".join(supporting_terms),
            "match_types": " | ".join(match_types),
            "known_label_count": len(concept.normalized_labels),
            "known_labels": " | ".join(concept.all_labels),
        })
    concept_coverage = pd.DataFrame(concept_rows)

    metrics = {
        "retained_unique_term_count": len(predicted_term_set),
        "matched_unique_term_count": tp,
        "unmatched_unique_term_count": fp,
        "total_s3_concept_count": len(all_concepts),
        "covered_s3_concept_count": len(covered_concepts),
        "missing_s3_concept_count": fn,
        "term_precision_against_s3_labels": precision,
        "concept_recall": concept_recall,
        "concept_level_f1": f1,
        "canonical_match_count": int(
            match_frame["match_type"].eq("canonical_exact").sum()
        ) if not match_frame.empty else 0,
        "alias_match_count": int(
            match_frame["match_type"].eq("alias_exact").sum()
        ) if not match_frame.empty else 0,
    }
    return metrics, match_frame, unmatched, concept_coverage


def readiness_assessment(concept_recall: float, target: float) -> tuple[str, str]:
    if concept_recall >= target:
        return (
            "ready_for_taxonomy_pilot",
            f"Concept recall {concept_recall:.2%} meets the configured target {target:.2%}.",
        )
    if concept_recall >= max(0.0, target - 0.10):
        return (
            "near_ready_targeted_gap_review",
            f"Concept recall {concept_recall:.2%} is within 10 percentage points of the target; review missing core concepts before taxonomy induction.",
        )
    return (
        "not_ready_expand_concept_coverage",
        f"Concept recall {concept_recall:.2%} is below the target {target:.2%}; taxonomy construction would omit too many benchmark concepts.",
    )


def plot_summary(metrics: dict[str, Any], output: Path) -> None:
    labels = ["Term precision", "Concept recall", "Concept F1"]
    values = [
        metrics["term_precision_against_s3_labels"],
        metrics["concept_recall"],
        metrics["concept_level_f1"],
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("S3 concept-coverage readiness metrics")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}", ha="center")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retained terms against S3_GT_CONCEPTS before taxonomy construction."
    )
    parser.add_argument("--input", type=Path, default=None, help="Optional preselected retained-term CSV")
    parser.add_argument("--qwen-run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concept-id-column", default=None)
    parser.add_argument("--concept-label-column", default=None)
    parser.add_argument("--alias-columns", nargs="+", default=None)
    parser.add_argument("--include-manual-review", action="store_true")
    parser.add_argument("--readiness-recall-target", type=float, default=0.75)
    args = parser.parse_args()

    if not 0 <= args.readiness_recall_target <= 1:
        raise ValueError("--readiness-recall-target must be between 0 and 1")

    started = time.perf_counter()
    started_utc = utc_now()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    retained, source_metadata = load_input_inventory(args.input, args.qwen_run_dir)
    concepts, id_column, label_column, alias_columns = load_s3_concepts(
        args.ground_truth,
        args.concept_id_column,
        args.concept_label_column,
        args.alias_columns,
        args.include_manual_review,
    )

    metrics, matches, unmatched, coverage = evaluate_concept_coverage(retained, concepts)
    status, explanation = readiness_assessment(
        metrics["concept_recall"], args.readiness_recall_target
    )
    metrics.update({
        "readiness_status": status,
        "readiness_explanation": explanation,
        "readiness_recall_target": args.readiness_recall_target,
        **source_metadata,
    })

    retained.to_csv(args.output_dir / "evaluated_retained_terms.csv", index=False)
    concepts_export = concepts.copy()
    for column in ("all_labels", "normalized_labels", "alias_sources"):
        concepts_export[column] = concepts_export[column].map(
            lambda value: json.dumps(value, ensure_ascii=False)
        )
    concepts_export.to_csv(args.output_dir / "s3_concept_inventory.csv", index=False)
    matches.to_csv(args.output_dir / "term_to_s3_concept_matches.csv", index=False)
    unmatched.to_csv(args.output_dir / "retained_terms_unmatched_to_s3.csv", index=False)
    coverage.to_csv(args.output_dir / "s3_concept_coverage.csv", index=False)
    coverage[coverage["covered"]].to_csv(
        args.output_dir / "s3_concepts_covered.csv", index=False
    )
    coverage[~coverage["covered"]].to_csv(
        args.output_dir / "s3_concepts_missing.csv", index=False
    )
    pd.DataFrame([metrics]).to_csv(args.output_dir / "s3_concept_metrics.csv", index=False)
    plot_summary(metrics, args.output_dir / "s3_concept_metrics.png")

    runtime = time.perf_counter() - started
    manifest = {
        "run_started_utc": started_utc,
        "run_finished_utc": utc_now(),
        "runtime_seconds": runtime,
        "runtime_hhmmss": format_duration(runtime),
        "ground_truth_file": str(args.ground_truth),
        "ground_truth_sheet": S3_SHEET,
        "concept_id_column": id_column,
        "concept_label_column": label_column,
        "alias_columns": alias_columns,
        "include_manual_review": args.include_manual_review,
        "metrics": metrics,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("=" * 92)
    print("S3 concept-coverage demo completed")
    print("=" * 92)
    print(f"Retained unique terms:         {metrics['retained_unique_term_count']}")
    print(f"Terms matching S3 labels:      {metrics['matched_unique_term_count']}")
    print(f"Unmatched retained terms:      {metrics['unmatched_unique_term_count']}")
    print(f"Total eligible S3 concepts:    {metrics['total_s3_concept_count']}")
    print(f"Covered S3 concepts:           {metrics['covered_s3_concept_count']}")
    print(f"Missing S3 concepts:           {metrics['missing_s3_concept_count']}")
    print(f"Term precision vs S3 labels:   {metrics['term_precision_against_s3_labels']:.2%}")
    print(f"Concept recall:                {metrics['concept_recall']:.2%}")
    print(f"Concept-level F1:              {metrics['concept_level_f1']:.2%}")
    print(f"Readiness status:              {status}")
    print(f"Assessment:                    {explanation}")
    print(f"Runtime:                       {format_duration(runtime)}")
    print(f"\nMetrics: {args.output_dir / 's3_concept_metrics.csv'}")
    print(f"Missing concepts: {args.output_dir / 's3_concepts_missing.csv'}")
    print(f"Coverage map: {args.output_dir / 's3_concept_coverage.csv'}")


if __name__ == "__main__":
    main()
