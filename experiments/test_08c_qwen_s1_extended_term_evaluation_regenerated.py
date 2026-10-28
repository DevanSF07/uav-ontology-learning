#!/usr/bin/env python3
"""Evaluate a Qwen-retained term inventory against S1_GT_TERMS.

The evaluator reports cumulative levels:

1. strict
   Normalized exact matches only.

2. safe_extended
   Strict matches plus:
   - conservative singular/plural variants
   - explicit canonical/alias/acronym pairs from S2_GT_ALIASES
   - parenthetical plural normalization, such as station(s)

3. context_extended
   Safe matches plus:
   - controlled slash-compound components, such as VLA/BLOS
   - context-validated domain-head completion, such as Mini -> Mini UAV

No unrestricted substring matching or embedding similarity is used.
Ground truth is used only for evaluation and never changes the Qwen output.
"""

from __future__ import annotations

import argparse
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


PROJECT_DIR = Path("ontology_experiment_framework")
DEFAULT_INPUT = (
    PROJECT_DIR
    / "stage_outputs/03b_extraction_ensembles/by_ensemble/E04_yake_and_mpnet/candidates.csv"
)
DEFAULT_GT = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_OUTPUT = PROJECT_DIR / "stage_outputs/03b_extraction_ensembles/by_ensemble/E04_yake_and_mpnet"

S1_SHEET = "S1_GT_TERMS"
S2_SHEET = "S2_GT_ALIASES"
MANUAL_REVIEW_STATUS = "needs_manual_review"

DOMAIN_HEADS = {
    "uav", "uavs", "uas", "aircraft", "vehicle", "vehicles",
    "system", "systems", "station", "stations", "link", "links",
    "sensor", "sensors", "suite", "suites", "platform", "platforms",
    "application", "applications", "operation", "operations",
    "classification", "class", "classes", "radar", "terminal", "terminals",
    "communication", "communications", "control", "controller", "controllers",
}

# These short forms are usually related to longer terms but are not safely
# equivalent by head completion alone.
AMBIGUOUS_SHORT_FORMS = {
    "altitude", "autonomy", "classification", "ownership", "control",
    "risk", "safety", "monitoring", "integration", "metrics",
    "capabilities", "uav", "system", "aircraft", "data", "communication",
}

ACRONYM_EXCEPTIONS = {
    "uas", "uav", "uavs", "rpas", "rpa", "gps", "isr", "los", "blos",
    "s&a", "mtow", "hale", "male", "mav", "mavs", "muav", "muavs",
    "ucav", "ucavs", "suas", "gcs", "rvt", "rvts", "faa", "icao",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    text = re.sub(r"\(s\)", "s", text)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"[^a-z0-9&/+ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def singularize_head_token(token: str) -> str:
    if token in ACRONYM_EXCEPTIONS:
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 4 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def morphological_form(value: object) -> str:
    tokens = normalize_term(value).split()
    if not tokens:
        return ""
    tokens[-1] = singularize_head_token(tokens[-1])
    return " ".join(tokens)


def phrase_in_context(phrase: str, context: str) -> bool:
    phrase = normalize_term(phrase)
    context = normalize_term(context)
    if not phrase or not context:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return re.search(pattern, context) is not None


def parse_context_value(value: object) -> str:
    text = safe_text(value)
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return " ".join(safe_text(item) for item in parsed)
        except Exception:
            pass
    return text


def candidate_context(row: pd.Series) -> str:
    return " ".join(
        part for part in (
            safe_text(row.get("source_sentence", "")),
            safe_text(row.get("context_text", "")),
            parse_context_value(row.get("additional_contexts", "")),
        ) if part
    )


def split_compound(value: object) -> list[str]:
    raw = safe_text(value)
    if not any(separator in raw for separator in ("/", "|", ";")):
        return []
    return [normalize_term(part) for part in re.split(r"[/|;]+", raw) if normalize_term(part)]


def load_qwen_terms(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8", low_memory=False)
    if "candidate" not in frame.columns:
        raise ValueError("Qwen output must contain a candidate column")
    frame = frame.copy()
    frame["normalized_candidate"] = frame.get(
        "normalized_candidate", frame["candidate"]
    ).map(normalize_term)
    frame["candidate_morph"] = frame["candidate"].map(morphological_form)
    frame = frame[frame["normalized_candidate"].ne("")]
    return frame.drop_duplicates("normalized_candidate", keep="first").reset_index(drop=True)


def load_ground_truth(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(path)
    s1 = pd.read_excel(path, sheet_name=S1_SHEET, engine="openpyxl")
    s2 = pd.read_excel(path, sheet_name=S2_SHEET, engine="openpyxl")

    required_s1 = {"gt_id", "term_canonical", "grounded_status_v2"}
    required_s2 = {"canonical", "alias", "grounded_status_v2"}
    if missing := required_s1 - set(s1.columns):
        raise ValueError(f"S1 is missing columns: {sorted(missing)}")
    if missing := required_s2 - set(s2.columns):
        raise ValueError(f"S2 is missing columns: {sorted(missing)}")

    if "keep_for_tasks" in s1.columns:
        s1 = s1[
            s1["keep_for_tasks"].astype(str).str.contains(
                "term_extraction", case=False, na=False
            )
        ].copy()
    status = s1["grounded_status_v2"].fillna("").astype(str).str.casefold()
    s1 = s1[status.ne(MANUAL_REVIEW_STATUS)].copy()
    s1["normalized_gt"] = s1["term_canonical"].map(normalize_term)
    s1["gt_morph"] = s1["term_canonical"].map(morphological_form)
    s1 = s1[s1["normalized_gt"].ne("")].drop_duplicates("gt_id").reset_index(drop=True)

    status = s2["grounded_status_v2"].fillna("").astype(str).str.casefold()
    s2 = s2[status.ne(MANUAL_REVIEW_STATUS)].copy()
    s2["canonical_norm"] = s2["canonical"].map(normalize_term)
    s2["alias_norm"] = s2["alias"].map(normalize_term)
    s2 = s2[s2["canonical_norm"].ne("") & s2["alias_norm"].ne("")]
    return s1, s2


def build_alias_graph(s2: pd.DataFrame) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for row in s2.itertuples(index=False):
        graph[row.canonical_norm].add(row.alias_norm)
        graph[row.alias_norm].add(row.canonical_norm)
    return graph


def alias_labels_for_gt(gt_term: str, alias_graph: dict[str, set[str]]) -> set[str]:
    normalized = normalize_term(gt_term)
    labels = set(alias_graph.get(normalized, set()))
    tokens = normalized.split()

    # S1 may contain a domain head while S2 canonicalizes the lexical core.
    # Example: HALE UAVs can inherit HALE from high altitude long endurance.
    for known_label, neighbors in alias_graph.items():
        known_tokens = known_label.split()
        if len(tokens) > len(known_tokens) and tokens[: len(known_tokens)] == known_tokens:
            suffix = tokens[len(known_tokens):]
            if suffix and set(suffix).issubset(DOMAIN_HEADS):
                labels.update(neighbors)
        for neighbor in neighbors:
            neighbor_tokens = neighbor.split()
            if len(tokens) > len(neighbor_tokens) and tokens[: len(neighbor_tokens)] == neighbor_tokens:
                suffix = tokens[len(neighbor_tokens):]
                if suffix and set(suffix).issubset(DOMAIN_HEADS):
                    labels.add(neighbor)
    return labels


def completion_is_valid(candidate: str, gt_term: str, context: str) -> tuple[bool, str]:
    candidate = normalize_term(candidate)
    gt_term = normalize_term(gt_term)
    candidate_tokens = candidate.split()
    gt_tokens = gt_term.split()

    if not candidate_tokens or len(gt_tokens) <= len(candidate_tokens):
        return False, ""
    if candidate in AMBIGUOUS_SHORT_FORMS:
        return False, ""

    if gt_tokens[: len(candidate_tokens)] == candidate_tokens:
        missing = gt_tokens[len(candidate_tokens):]
    elif gt_tokens[-len(candidate_tokens):] == candidate_tokens:
        missing = gt_tokens[:-len(candidate_tokens)]
    else:
        return False, ""

    if not missing or not set(missing).issubset(DOMAIN_HEADS):
        return False, ""
    if phrase_in_context(gt_term, context):
        return True, "full_gt_phrase_in_context"
    if phrase_in_context(candidate, context) and any(
        phrase_in_context(head, context) for head in missing
    ):
        return True, "candidate_and_missing_head_in_context"
    return False, ""


def add_mapping(
    mappings: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    candidate_row: pd.Series,
    gt_row: Any,
    rule: str,
    tier: str,
    confidence: float,
    evidence: str = "",
) -> None:
    key = (candidate_row["normalized_candidate"], str(gt_row.gt_id), rule)
    if key in seen:
        return
    seen.add(key)
    mappings.append({
        "candidate": candidate_row["candidate"],
        "normalized_candidate": candidate_row["normalized_candidate"],
        "gt_id": gt_row.gt_id,
        "gt_term": gt_row.term_canonical,
        "gt_category": getattr(gt_row, "gt_category", ""),
        "priority": getattr(gt_row, "priority", ""),
        "match_rule": rule,
        "evaluation_tier": tier,
        "confidence": confidence,
        "rule_evidence": evidence,
    })


def evaluate(
    candidates: pd.DataFrame,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exact_index: dict[str, list[Any]] = defaultdict(list)
    morph_index: dict[str, list[Any]] = defaultdict(list)
    alias_index: dict[str, list[Any]] = defaultdict(list)
    alias_graph = build_alias_graph(s2)

    for gt_row in s1.itertuples(index=False):
        exact_index[gt_row.normalized_gt].append(gt_row)
        morph_index[gt_row.gt_morph].append(gt_row)
        for label in alias_labels_for_gt(gt_row.term_canonical, alias_graph):
            alias_index[label].append(gt_row)

    mappings: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for _, candidate_row in candidates.iterrows():
        candidate = candidate_row["normalized_candidate"]
        candidate_morph = candidate_row["candidate_morph"]
        context = candidate_context(candidate_row)

        for gt_row in exact_index.get(candidate, []):
            add_mapping(mappings, seen, candidate_row, gt_row, "exact", "strict", 1.0)

        for gt_row in morph_index.get(candidate_morph, []):
            if gt_row.normalized_gt != candidate:
                add_mapping(
                    mappings, seen, candidate_row, gt_row,
                    "morphological_variant", "safe_extended", 0.98,
                )

        for gt_row in alias_index.get(candidate, []):
            add_mapping(
                mappings, seen, candidate_row, gt_row,
                "explicit_s2_alias", "safe_extended", 1.0,
            )

        # Full form gets priority. Components add coverage only at context tier.
        for component in split_compound(candidate_row["candidate"]):
            if component == candidate:
                continue
            for gt_row in exact_index.get(component, []):
                add_mapping(
                    mappings, seen, candidate_row, gt_row,
                    "compound_component", "context_extended", 0.92, component,
                )
            for gt_row in alias_index.get(component, []):
                add_mapping(
                    mappings, seen, candidate_row, gt_row,
                    "compound_alias_component", "context_extended", 0.90, component,
                )

        for gt_row in s1.itertuples(index=False):
            valid, evidence = completion_is_valid(
                candidate, gt_row.normalized_gt, context
            )
            if valid:
                add_mapping(
                    mappings, seen, candidate_row, gt_row,
                    "contextual_head_completion", "context_extended", 0.90, evidence,
                )
            else:
                candidate_tokens = candidate.split()
                gt_tokens = gt_row.normalized_gt.split()
                if (
                    candidate not in AMBIGUOUS_SHORT_FORMS
                    and candidate_tokens
                    and len(gt_tokens) > len(candidate_tokens)
                    and gt_tokens[: len(candidate_tokens)] == candidate_tokens
                    and set(gt_tokens[len(candidate_tokens):]).issubset(DOMAIN_HEADS)
                ):
                    review.append({
                        "candidate": candidate_row["candidate"],
                        "normalized_candidate": candidate,
                        "gt_id": gt_row.gt_id,
                        "gt_term": gt_row.term_canonical,
                        "proposal": "unverified_head_completion",
                    })

    mapping_frame = pd.DataFrame(mappings)
    review_frame = (
        pd.DataFrame(review).drop_duplicates()
        if review
        else pd.DataFrame(columns=[
            "candidate", "normalized_candidate", "gt_id", "gt_term", "proposal"
        ])
    )

    tier_rank = {"strict": 0, "safe_extended": 1, "context_extended": 2}
    all_terms = set(candidates["normalized_candidate"])
    all_gt = set(s1["gt_id"])
    metric_rows = []

    for level in ("strict", "safe_extended", "context_extended"):
        allowed = {tier for tier, rank in tier_rank.items() if rank <= tier_rank[level]}
        subset = (
            mapping_frame[mapping_frame["evaluation_tier"].isin(allowed)]
            if not mapping_frame.empty else mapping_frame
        )
        matched_terms = set(subset["normalized_candidate"]) if not subset.empty else set()
        covered_gt = set(subset["gt_id"]) if not subset.empty else set()
        tp = len(matched_terms)
        fp = len(all_terms - matched_terms)
        fn = len(all_gt - covered_gt)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = len(covered_gt) / len(all_gt) if all_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metric_rows.append({
            "evaluation_level": level,
            "retained_unique_terms": len(all_terms),
            "matched_unique_terms": tp,
            "unmatched_unique_terms": fp,
            "eligible_s1_gt_terms": len(all_gt),
            "covered_s1_gt_terms": len(covered_gt),
            "missing_s1_gt_terms": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    # Prefer the safest mapping when duplicate rules map the same pair.
    rule_rank = {
        "exact": 0,
        "morphological_variant": 1,
        "explicit_s2_alias": 2,
        "compound_component": 3,
        "compound_alias_component": 4,
        "contextual_head_completion": 5,
    }
    if not mapping_frame.empty:
        deduplicated = (
            mapping_frame.assign(_rank=mapping_frame["match_rule"].map(rule_rank).fillna(99))
            .sort_values(["_rank", "confidence"], ascending=[True, False])
            .drop_duplicates(["normalized_candidate", "gt_id"])
            .drop(columns="_rank")
        )
    else:
        deduplicated = mapping_frame.copy()

    return pd.DataFrame(metric_rows), mapping_frame, deduplicated, review_frame


def plot_metrics(metrics: pd.DataFrame, output: Path) -> None:
    labels = metrics["evaluation_level"].str.replace("_", " ").str.title()
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, column, display in (
        (-width, "precision", "Precision"),
        (0, "recall", "Recall"),
        (width, "f1", "F1"),
    ):
        bars = ax.bar(x + offset, metrics[column], width, label=display)
        for bar, value in zip(bars, metrics[column]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.1%}",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(0.65, float(metrics[["precision", "recall", "f1"]].max().max()) + 0.10))
    ax.set_ylabel("Score")
    ax.set_title("YAKE and MPNet Terms against S1_GT_TERMS")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen output against S1_GT_TERMS using controlled variant rules."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.perf_counter()
    started_utc = utc_now()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_qwen_terms(args.input)
    s1, s2 = load_ground_truth(args.ground_truth)
    metrics, mappings, deduplicated, review = evaluate(candidates, s1, s2)

    metrics.to_csv(args.output_dir / "s1_extended_metrics.csv", index=False)
    mappings.to_csv(args.output_dir / "all_accepted_mappings.csv", index=False)
    deduplicated.to_csv(args.output_dir / "deduplicated_term_gt_mappings.csv", index=False)
    review.to_csv(args.output_dir / "mapping_review_queue.csv", index=False)

    covered = set(deduplicated["gt_id"]) if not deduplicated.empty else set()
    matched = set(deduplicated["normalized_candidate"]) if not deduplicated.empty else set()
    s1.assign(covered=s1["gt_id"].isin(covered)).to_csv(
        args.output_dir / "s1_gt_coverage.csv", index=False
    )
    s1[~s1["gt_id"].isin(covered)].to_csv(
        args.output_dir / "s1_gt_terms_still_missing.csv", index=False
    )
    candidates[candidates["normalized_candidate"].isin(matched)].to_csv(
        args.output_dir / "matched_qwen_terms.csv", index=False
    )
    candidates[~candidates["normalized_candidate"].isin(matched)].to_csv(
        args.output_dir / "unmatched_qwen_terms.csv", index=False
    )
    plot_metrics(metrics, args.output_dir / "s1_extended_metrics.png")

    runtime = time.perf_counter() - started
    rule_counts = (
        mappings["match_rule"].value_counts().to_dict()
        if not mappings.empty else {}
    )
    manifest = {
        "run_started_utc": started_utc,
        "run_finished_utc": utc_now(),
        "runtime_seconds": runtime,
        "input": str(args.input),
        "ground_truth": str(args.ground_truth),
        "input_unique_terms": len(candidates),
        "eligible_s1_terms": len(s1),
        "rule_counts": rule_counts,
        "domain_heads": sorted(DOMAIN_HEADS),
        "ambiguous_short_forms": sorted(AMBIGUOUS_SHORT_FORMS),
        "metrics": metrics.to_dict(orient="records"),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 105)
    print("Qwen S1 extended term evaluation completed")
    print("=" * 105)
    print(metrics.to_string(
        index=False,
        formatters={
            "precision": "{:.2%}".format,
            "recall": "{:.2%}".format,
            "f1": "{:.2%}".format,
        },
    ))
    print("\nAccepted mappings by rule:")
    print(mappings["match_rule"].value_counts().to_string() if not mappings.empty else "None")
    print(f"\nReview proposals: {len(review)}")
    print(f"Runtime: {runtime:.2f}s")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
