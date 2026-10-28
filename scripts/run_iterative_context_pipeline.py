from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from olbench.extractors import EXTRACTORS, acronym_extract, yake_extract, keybert_extract
from olbench.noise import filter_noise
from olbench.preprocess import chunk_text, clean_mmd
from olbench.utils import dump_json, file_hash, save_df


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = PROJECT_DIR / "data/combined_corpus_raw.txt"
DEFAULT_GROUND_TRUTH_FILE = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "stage_outputs/iterative_context_compare"

GROUND_TRUTH_SHEET = "S1_GT_TERMS"
NEGATIVE_TERMS_SHEET = "S7_NEGATIVE_TERMS"
MANUAL_REVIEW_STATUS = "needs_manual_review"

DEFAULT_EXTRACTORS = ["yake", "acronyms"]

ALIAS_PAIR_PATTERNS = [
    re.compile(r"\b([A-Z][A-Za-z][A-Za-z\-/ ]{4,80}?)\s*\(([A-Z][A-Z0-9\-/]{1,12})\)"),
    re.compile(r"\b([A-Z][A-Z0-9\-/]{1,12})\s*\(([A-Z][A-Za-z][A-Za-z\-/ ]{4,80}?)\)"),
]

BAYESIAN_PRIOR_ALPHA = 2.0
BAYESIAN_PRIOR_BETA = 8.0
RETAIN_THRESHOLD = 0.36
PROVISIONAL_THRESHOLD = 0.30
BACKLOG_THRESHOLD = 0.30

HIGH_PRECISION_PROFILE = {
    "retain_threshold": 0.48,
    "provisional_threshold": 0.40,
    "top_k_per_chunk": 15,
    "seed_similarity_threshold": 0.42,
    "context_similarity_threshold": 0.70,
    "cluster_count": 4,
    "fallback_keep_top_n": 0,
    "max_context_terms": 80,
}

DOMAIN_SEEDS = [
    "UAV",
    "unmanned aerial vehicle",
]

ACRONYM_TERMS = {
    "uav",
    "uas",
    "rpas",
    "rpa",
    "gps",
    "gcs",
    "imu",
    "isr",
    "los",
    "blos",
    "mtow",
}

WEAK_UAV_PREFIXES = {
    "be",
    "been",
    "being",
    "developing",
    "integrate",
    "integrating",
    "produced",
    "procured",
    "term",
    "using",
}

PLURAL_EXCEPTIONS = ACRONYM_TERMS | {
    "aircraft",
    "class",
    "gas",
    "gps",
    "is",
    "mass",
    "news",
    "series",
    "species",
    "status",
}

STAGE_ORDER = {
    "raw_candidates": 0,
    "after_gliner": 1,
    "after_seed_similarity": 2,
    "after_clustering": 3,
    "iterative_context": 4,
    "final_registry": 5,
}


def normalize_term(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).casefold().strip()
    for dash in ("‐", "‑", "‒", "–", "—"):
        text = text.replace(dash, "-")

    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"[^a-z0-9&/+ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return lemmatize_term_head(text)


def singularize_head_token(token: str) -> str:
    token = safe_text(token).casefold()
    if not token or token in PLURAL_EXCEPTIONS:
        return token
    if token.endswith("s") and token[:-1] in ACRONYM_TERMS:
        return token[:-1]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 4 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def lemmatize_term_head(term: str) -> str:
    tokens = safe_text(term).split()
    if not tokens:
        return ""
    tokens[-1] = singularize_head_token(tokens[-1])
    return " ".join(tokens)


def safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _bounded(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 0:
        return 0.0
    return 1.0 - math.exp(-numeric)


def _stage_sort_key(stage_name: str) -> int:
    return STAGE_ORDER.get(stage_name, 99)


def _alias_group_id(members: list[str]) -> str:
    key = "|".join(sorted(member for member in members if member))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"alias_{digest}"


def extract_alias_evidence(chunk: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    text = safe_text(chunk.get("text", ""))

    for pattern in ALIAS_PAIR_PATTERNS:
        for match in pattern.finditer(text):
            left, right = [safe_text(value) for value in match.groups()]
            if not left or not right:
                continue

            if left.isupper():
                acronym, expansion = left, right
            else:
                acronym, expansion = right, left

            acronym_norm = normalize_term(acronym)
            expansion_norm = normalize_term(expansion)
            if not acronym_norm or not expansion_norm or acronym_norm == expansion_norm:
                continue

            rows.append(
                {
                    "chunk_id": safe_text(chunk.get("chunk_id", "")),
                    "chunk_index": int(chunk.get("chunk_index", -1)),
                    "section": safe_text(chunk.get("section", "")),
                    "canonical_term": expansion,
                    "alias_term": acronym,
                    "canonical_norm": expansion_norm,
                    "alias_norm": acronym_norm,
                    "method": "parenthetical_acronym",
                    "score": 1.0,
                    "source_text": match.group(0),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "chunk_id",
                "chunk_index",
                "section",
                "canonical_term",
                "alias_term",
                "canonical_norm",
                "alias_norm",
                "method",
                "score",
                "source_text",
            ]
        )

    return pd.DataFrame(rows).drop_duplicates(["canonical_norm", "alias_norm"]).reset_index(drop=True)


def build_alias_metadata(alias_evidence: pd.DataFrame) -> dict[str, dict[str, object]]:
    if alias_evidence.empty:
        return {}

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for row in alias_evidence.itertuples(index=False):
        union(row.canonical_norm, row.alias_norm)

    groups: dict[str, list[str]] = {}
    for value in list(parent):
        groups.setdefault(find(value), []).append(value)

    evidence_counts: dict[str, int] = {}
    for row in alias_evidence.itertuples(index=False):
        evidence_counts[row.canonical_norm] = evidence_counts.get(row.canonical_norm, 0) + 1
        evidence_counts[row.alias_norm] = evidence_counts.get(row.alias_norm, 0) + 1

    metadata: dict[str, dict[str, object]] = {}
    for members in groups.values():
        representative = max(members, key=lambda value: (len(value.split()), len(value)))
        group_id = _alias_group_id(members)
        aliases = " | ".join(sorted(member for member in members if member != representative))
        for member in members:
            metadata[member] = {
                "alias_group_id": group_id,
                "alias_canonical_term": representative,
                "aliases": aliases,
                "alias_evidence_count": evidence_counts.get(member, 0),
            }
    return metadata


def apply_alias_metadata(frame: pd.DataFrame, alias_metadata: dict[str, dict[str, object]]) -> pd.DataFrame:
    enriched = frame.copy()
    if enriched.empty:
        for column in ["alias_group_id", "alias_canonical_term", "aliases", "alias_evidence_count"]:
            enriched[column] = []
        return enriched

    if "normalized_candidate" not in enriched.columns:
        enriched["normalized_candidate"] = enriched.get("candidate", "").map(normalize_term)

    enriched["alias_group_id"] = enriched["normalized_candidate"].map(lambda value: alias_metadata.get(value, {}).get("alias_group_id", ""))
    enriched["alias_canonical_term"] = enriched["normalized_candidate"].map(lambda value: alias_metadata.get(value, {}).get("alias_canonical_term", value))
    enriched["aliases"] = enriched["normalized_candidate"].map(lambda value: alias_metadata.get(value, {}).get("aliases", ""))
    enriched["alias_evidence_count"] = enriched["normalized_candidate"].map(lambda value: int(alias_metadata.get(value, {}).get("alias_evidence_count", 0)))
    return enriched


def is_acronym_like(candidate: object, normalized_candidate: object) -> bool:
    raw = safe_text(candidate)
    normalized = safe_text(normalized_candidate).casefold()
    if normalized in ACRONYM_TERMS:
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9/+&-]{1,11}s?", raw):
        return True
    return False


def apply_rule_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if enriched.empty:
        enriched["rule_support_count"] = []
        enriched["rule_support_reasons"] = []
        return enriched

    if "normalized_candidate" not in enriched.columns:
        enriched["normalized_candidate"] = enriched.get("candidate", "").map(normalize_term)

    rule_counts: list[int] = []
    rule_reasons: list[str] = []
    for row in enriched.to_dict(orient="records"):
        candidate = safe_text(row.get("candidate", ""))
        normalized = safe_text(row.get("normalized_candidate", ""))
        tokens = normalized.split()
        reasons: list[str] = []

        if tokens and tokens[-1] == "uav":
            reasons.append("head_uav")
            if tokens[0] not in WEAK_UAV_PREFIXES:
                reasons.append("strong_uav_head")

        if is_acronym_like(candidate, normalized):
            reasons.append("acronym_like")

        if safe_text(row.get("supporting_methods", "")).find("acronyms") >= 0:
            reasons.append("acronym_extractor")

        unique_reasons = sorted(set(reasons))
        rule_counts.append(len(unique_reasons))
        rule_reasons.append(" | ".join(unique_reasons))

    enriched["rule_support_count"] = rule_counts
    enriched["rule_support_reasons"] = rule_reasons
    return enriched


def _posterior_from_evidence(row: pd.Series) -> tuple[float, float, float, list[str]]:
    method_support = float(row.get("method_support_count", 0) or 0)
    occurrence_count = float(row.get("occurrence_count", 0) or 0)
    chunk_count = float(row.get("chunk_count", 0) or 0)
    score = float(row.get("score", 0) or 0)
    seed_similarity = float(row.get("max_seed_similarity", row.get("best_seed_similarity", 0.0)) or 0)
    context_similarity = float(row.get("max_context_similarity", row.get("best_context_similarity", 0.0)) or 0)
    alias_evidence_count = float(row.get("alias_evidence_count", 0) or 0)
    rule_support_count = float(row.get("rule_support_count", 0) or 0)
    cluster_representative = 1.0 if str(row.get("cluster_representative", "")).strip().casefold() in {"1", "true", "yes"} else 0.0
    cluster_size = float(row.get("cluster_size", 0) or 0)
    cluster_memory_support_count = float(row.get("cluster_memory_support_count", 0) or 0)
    removed_by = safe_text(row.get("removed_by", ""))

    alpha = BAYESIAN_PRIOR_ALPHA
    beta = BAYESIAN_PRIOR_BETA

    alpha += 1.15 * _bounded(method_support / 3.0)
    alpha += 0.95 * _bounded(occurrence_count / 4.0)
    alpha += 0.90 * _bounded(chunk_count / 3.0)
    alpha += 1.10 * _bounded(score)
    alpha += 1.20 * _bounded(seed_similarity)
    alpha += 1.25 * _bounded(context_similarity)
    alpha += 0.85 * _bounded(alias_evidence_count)
    alpha += 1.15 * _bounded(rule_support_count)
    alpha += 0.35 * cluster_representative
    alpha += 0.20 * _bounded(cluster_size / 4.0)
    alpha += 1.00 * _bounded(cluster_memory_support_count / 4.0)

    beta += 0.85 * (1.0 - _bounded(method_support / 3.0))
    beta += 0.60 * (1.0 - _bounded(occurrence_count / 4.0))
    beta += 0.55 * (1.0 - _bounded(chunk_count / 3.0))
    beta += 0.65 * (1.0 - _bounded(score))
    beta += 0.95 * (1.0 - _bounded(seed_similarity))
    beta += 1.00 * (1.0 - _bounded(context_similarity))

    if removed_by:
        beta += 4.0

    posterior = alpha / (alpha + beta)

    reasons = [
        f"method_support={method_support:.0f}",
        f"chunk_support={chunk_count:.0f}",
        f"occurrence_support={occurrence_count:.0f}",
        f"seed_similarity={seed_similarity:.3f}",
        f"context_similarity={context_similarity:.3f}",
        f"alias_evidence={alias_evidence_count:.0f}",
        f"rule_support={rule_support_count:.0f}",
        f"cluster_memory_support={cluster_memory_support_count:.0f}",
        f"score={score:.4f}",
    ]
    rule_reasons = safe_text(row.get("rule_support_reasons", ""))
    if rule_reasons:
        reasons.append(f"rule_support_reasons={rule_reasons}")
    if cluster_representative:
        reasons.append("cluster_representative=1")
    if removed_by:
        reasons.append(f"removed_by={removed_by}")

    return posterior, alpha, beta, reasons


def assign_state(row: pd.Series, stage_name: str, keep_survivor: bool = True) -> dict[str, object]:
    posterior, alpha, beta, reasons = _posterior_from_evidence(row)
    method_support = int(float(row.get("method_support_count", 0) or 0))
    chunk_count = int(float(row.get("chunk_count", 0) or 0))
    removed_by = safe_text(row.get("removed_by", ""))

    if removed_by:
        state = "backlog"
        keep_node = 0
        reasons.append("stage_filtered_to_backlog=1")
    elif posterior >= RETAIN_THRESHOLD and method_support >= 1 and chunk_count >= 1:
        state = "retained"
        keep_node = 1
        reasons.append("state=retained")
    elif posterior >= PROVISIONAL_THRESHOLD:
        state = "provisional"
        keep_node = 1 if keep_survivor else 0
        reasons.append("state=provisional")
    else:
        state = "backlog"
        keep_node = 0
        reasons.append("state=backlog")

    return {
        "stage": stage_name,
        "bayesian_alpha": round(alpha, 4),
        "bayesian_beta": round(beta, 4),
        "bayesian_posterior": round(posterior, 6),
        "state": state,
        "keep_node": int(keep_node),
        "confidence_reason": " | ".join(reasons),
    }


def annotate_state_frame(frame: pd.DataFrame, stage_name: str, chunk_row: pd.Series, keep_survivor: bool = True) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    annotated_rows: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        record = dict(row)
        record.update(assign_state(pd.Series(record), stage_name, keep_survivor=keep_survivor))
        record["chunk_id"] = safe_text(chunk_row.get("chunk_id", record.get("chunk_id", "")))
        record["chunk_index"] = int(chunk_row.get("chunk_index", record.get("chunk_index", -1)))
        record["section"] = safe_text(chunk_row.get("section", record.get("section", "")))
        record["state_order"] = _stage_sort_key(stage_name)
        annotated_rows.append(record)

    return pd.DataFrame(annotated_rows)


def append_state_history(history: list[dict[str, object]], frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    history.extend(frame.to_dict(orient="records"))


def survivor_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "keep_node" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[pd.to_numeric(frame["keep_node"], errors="coerce").fillna(0).astype(int).eq(1)].copy().reset_index(drop=True)


def build_final_registry(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    work = history.copy()
    if "normalized_candidate" not in work.columns and "candidate" in work.columns:
        work["normalized_candidate"] = work["candidate"].map(normalize_term)

    work = work.sort_values(
        by=["normalized_candidate", "state_order", "chunk_index", "bayesian_posterior"],
        ascending=[True, True, True, False],
    )

    trace_df = work.groupby("normalized_candidate").apply(
        lambda group: " -> ".join(
            f"{row.stage}:{row.state}"
            for row in group.sort_values(["state_order", "chunk_index"]).itertuples()
        )
    )

    final = work.drop_duplicates("normalized_candidate", keep="last").copy().reset_index(drop=True)
    final["state_trace"] = final["normalized_candidate"].map(trace_df.to_dict())
    final = final.rename(
        columns={
            "stage": "final_stage",
            "state": "final_state",
            "keep_node": "final_keep_node",
            "bayesian_posterior": "final_posterior",
            "confidence_reason": "final_confidence_reason",
        }
    )
    unresolved_mask = final["final_state"].fillna("").astype(str).eq("backlog")
    final.loc[unresolved_mask, "final_state"] = "rejected"
    final.loc[unresolved_mask, "final_keep_node"] = 0
    final.loc[unresolved_mask, "final_confidence_reason"] = (
        final.loc[unresolved_mask, "final_confidence_reason"].fillna("").astype(str)
        + " | final_rejection_after_all_chunks=1"
    )
    return final.sort_values(
        by=["final_posterior", "final_keep_node", "normalized_candidate"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def write_final_state_tables(final_registry: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    if final_registry.empty:
        return {
            "surviving_terms": 0,
            "accepted_terms": 0,
            "provisional_terms": 0,
            "backlog_terms": 0,
            "rejected_terms": 0,
        }

    final_states = final_registry["final_state"].fillna("").astype(str)
    surviving = final_registry[final_states.isin({"retained", "provisional"})].copy()
    accepted = final_registry[final_states.eq("retained")].copy()
    provisional = final_registry[final_states.eq("provisional")].copy()
    backlog = final_registry[final_states.eq("backlog")].copy()
    rejected = final_registry[final_states.eq("rejected")].copy()

    surviving.to_csv(output_dir / "surviving_terms.csv", index=False, encoding="utf-8")
    accepted.to_csv(output_dir / "accepted_terms.csv", index=False, encoding="utf-8")
    provisional.to_csv(output_dir / "provisional_terms.csv", index=False, encoding="utf-8")
    backlog.to_csv(output_dir / "backlog_terms.csv", index=False, encoding="utf-8")
    rejected.to_csv(output_dir / "rejected_terms.csv", index=False, encoding="utf-8")

    return {
        "surviving_terms": int(len(surviving)),
        "accepted_terms": int(len(accepted)),
        "provisional_terms": int(len(provisional)),
        "backlog_terms": int(len(backlog)),
        "rejected_terms": int(len(rejected)),
    }


def write_chunk_state_tables(frame: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    if frame.empty or "state" not in frame.columns:
        return

    state_values = frame["state"].fillna("").astype(str)
    state_splits = {
        "surviving": frame[state_values.isin({"retained", "provisional"})],
        "retained": frame[state_values.eq("retained")],
        "provisional": frame[state_values.eq("provisional")],
        "backlog": frame[state_values.eq("backlog")],
        "rejected": frame[state_values.eq("rejected")],
    }

    for state_name, state_frame in state_splits.items():
        if not state_frame.empty:
            state_frame.to_csv(
                output_dir / f"{prefix}_{state_name}.csv",
                index=False,
                encoding="utf-8",
            )


def run_extended_gt_evaluations(output_dir: Path, ground_truth_file: Path) -> list[dict[str, object]]:
    evaluator = PROJECT_DIR / "test_08c_qwen_s1_extended_term_evaluation_regenerated.py"
    inputs = {
        "surviving": output_dir / "surviving_terms.csv",
        "accepted": output_dir / "accepted_terms.csv",
        "provisional": output_dir / "provisional_terms.csv",
    }
    results: list[dict[str, object]] = []

    if not evaluator.is_file():
        return [{"scope": "all", "status": "skipped", "reason": f"missing evaluator: {evaluator}"}]

    for scope, input_file in inputs.items():
        eval_dir = output_dir / "gt_extended_eval" / scope
        eval_dir.mkdir(parents=True, exist_ok=True)
        if not input_file.is_file():
            results.append({"scope": scope, "status": "skipped", "reason": f"missing input: {input_file}"})
            continue

        command = [
            sys.executable,
            str(evaluator),
            "--input",
            str(input_file),
            "--ground-truth",
            str(ground_truth_file),
            "--output-dir",
            str(eval_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_DIR.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        (eval_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (eval_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        results.append(
            {
                "scope": scope,
                "status": "completed" if completed.returncode == 0 else "failed",
                "returncode": int(completed.returncode),
                "output_dir": str(eval_dir),
            }
        )

    return results


def load_ground_truth(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt = pd.read_excel(workbook, sheet_name=GROUND_TRUTH_SHEET, engine="openpyxl")
    negatives = pd.read_excel(workbook, sheet_name=NEGATIVE_TERMS_SHEET, engine="openpyxl")

    required_columns = {"term_canonical", "keep_for_tasks", "grounded_status_v2", "priority"}
    missing = required_columns - set(gt.columns)
    if missing:
        raise ValueError(f"{workbook} is missing GT columns: {sorted(missing)}")

    gt = gt[
        gt["keep_for_tasks"]
        .astype(str)
        .str.contains("term_extraction", case=False, na=False)
    ].copy()
    gt["normalized_gt"] = gt["term_canonical"].map(normalize_term)
    gt = gt[gt["normalized_gt"].ne("")].copy()

    status = gt["grounded_status_v2"].fillna("").astype(str).str.casefold()
    strict_gt = gt[status.ne(MANUAL_REVIEW_STATUS)].copy()
    review_gt = gt[status.eq(MANUAL_REVIEW_STATUS)].copy()

    if "candidate" not in negatives.columns:
        raise ValueError(f"{workbook} sheet {NEGATIVE_TERMS_SHEET} must contain a candidate column.")

    negatives["normalized_negative"] = negatives["candidate"].map(normalize_term)
    negatives = negatives[negatives["normalized_negative"].ne("")].copy()
    return strict_gt, review_gt, negatives


def evaluate_terms(
    candidates: pd.DataFrame,
    strict_gt: pd.DataFrame,
    review_gt: pd.DataFrame,
    negatives: pd.DataFrame,
    mode: str,
    chunk_id: str | None = None,
    chunk_index: int | None = None,
) -> dict[str, object]:
    strict_unique = strict_gt.drop_duplicates("normalized_gt").copy()
    review_unique = review_gt.drop_duplicates("normalized_gt").copy()
    negative_unique = negatives.drop_duplicates("normalized_negative").copy()

    predicted = set(candidates["normalized_candidate"].map(normalize_term))
    predicted.discard("")

    gold = set(strict_unique["normalized_gt"])
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    priority_weights = {"high": 3.0, "medium": 2.0, "low": 1.0}
    gt_weights = (
        strict_unique["priority"]
        .fillna("low")
        .astype(str)
        .str.casefold()
        .map(priority_weights)
        .fillna(1.0)
    )
    matched_mask = strict_unique["normalized_gt"].isin(predicted)
    weighted_recall = float(gt_weights[matched_mask].sum() / gt_weights.sum()) if gt_weights.sum() else 0.0

    details = candidates.copy()
    if not details.empty:
        details["strict_match"] = details["normalized_candidate"].isin(gold)
        details["manual_review_match"] = details["normalized_candidate"].isin(set(review_unique["normalized_gt"]))
        details["negative_benchmark_hit"] = details["normalized_candidate"].isin(set(negative_unique["normalized_negative"]))
    else:
        details["strict_match"] = []
        details["manual_review_match"] = []
        details["negative_benchmark_hit"] = []

    return {
        "mode": mode,
        "chunk_id": chunk_id or "",
        "chunk_index": -1 if chunk_index is None else int(chunk_index),
        "candidate_count": int(len(candidates)),
        "strict_gold_count": int(len(gold)),
        "manual_review_gold_count": int(len(review_unique)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "priority_weighted_recall": weighted_recall,
        "manual_review_matches": int(details["manual_review_match"].sum()) if not details.empty else 0,
        "negative_term_hits": int(details["negative_benchmark_hit"].sum()) if not details.empty else 0,
    }


def load_chunks_from_input(input_file: Path, max_words: int, overlap_sentences: int) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    cleaned_text, cleaning_report = clean_mmd(input_file)
    chunks = chunk_text(cleaned_text, max_words=max_words, overlap_sentences=overlap_sentences)
    if chunks.empty:
        raise RuntimeError("No chunks were produced from the cleaned corpus.")
    return cleaned_text, cleaning_report, chunks


def build_extractor_plan(extractors: list[str], top_k_per_chunk: int, keybert_model: str) -> list[tuple[str, dict[str, object]]]:
    plan: list[tuple[str, dict[str, object]]] = []
    for extractor_name in extractors:
        if extractor_name not in {"yake", "acronyms", "keybert_mpnet"}:
            raise ValueError(f"Unknown extractor: {extractor_name}")
        params: dict[str, object] = {"top_k": top_k_per_chunk}
        if extractor_name == "acronyms":
            params["top_k"] = None
        elif extractor_name == "keybert_mpnet":
            params["model"] = keybert_model
        plan.append((extractor_name, params))
    return plan


def extract_chunk_candidates(chunk: pd.Series, extractor_plan: list[tuple[str, dict[str, object]]]) -> pd.DataFrame:
    chunk_frame = pd.DataFrame([chunk.to_dict()])
    outputs: list[pd.DataFrame] = []

    for extractor_name, params in extractor_plan:
        if extractor_name == "yake":
            extractor = yake_extract
        elif extractor_name == "acronyms":
            extractor = acronym_extract
        else:
            extractor = keybert_extract
        extracted = extractor(chunk_frame, **params)
        if extracted.empty:
            continue
        extracted = extracted.copy()
        extracted["chunk_index"] = int(chunk["chunk_index"])
        extracted["section"] = safe_text(chunk.get("section", ""))
        outputs.append(extracted)

    if not outputs:
        return pd.DataFrame(columns=["term_id", "raw_term", "term_canonical", "score", "rank", "extractor", "chunk_id", "source_sentence", "context_text", "chunk_index", "section"])

    return pd.concat(outputs, ignore_index=True)


def aggregate_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "candidate",
        "normalized_candidate",
        "score",
        "global_rank",
        "occurrence_count",
        "chunk_count",
        "method_support_count",
        "supporting_methods",
        "best_chunk_id",
        "source_sentence",
        "context_text",
        "chunk_index",
        "section",
        "best_context_similarity",
        "best_context_term",
    ]

    if raw.empty:
        return pd.DataFrame(columns=output_columns)

    work = raw.copy()
    required = {"raw_term", "score", "rank", "extractor", "chunk_id", "source_sentence", "context_text", "chunk_index", "section"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"Extractor output is missing columns: {sorted(missing)}")

    work["normalized_candidate"] = work["raw_term"].map(normalize_term)
    work = work[work["normalized_candidate"].ne("")].copy()
    work["score"] = pd.to_numeric(work["score"], errors="coerce").fillna(0.0)
    work["rank"] = pd.to_numeric(work["rank"], errors="coerce").fillna(999999)
    work = work.sort_values(by=["score", "rank", "normalized_candidate"], ascending=[False, True, True])

    best = work.drop_duplicates(subset="normalized_candidate", keep="first")
    stats = work.groupby("normalized_candidate", as_index=False).agg(
        occurrence_count=("normalized_candidate", "size"),
        chunk_count=("chunk_id", "nunique"),
        method_support_count=("extractor", "nunique"),
        supporting_methods=("extractor", lambda values: " | ".join(sorted(set(map(str, values))))),
        best_score=("score", "max"),
    )

    result = stats.merge(
        best[
            [
                "normalized_candidate",
                "raw_term",
                "chunk_id",
                "source_sentence",
                "context_text",
                "chunk_index",
                "section",
            ]
        ],
        on="normalized_candidate",
        how="left",
    )

    result = result.rename(
        columns={
            "raw_term": "candidate",
            "best_score": "score",
            "chunk_id": "best_chunk_id",
        }
    )
    result = result.sort_values(
        by=["score", "occurrence_count", "normalized_candidate"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    result["global_rank"] = range(1, len(result) + 1)
    result["best_context_similarity"] = 0.0
    result["best_context_term"] = ""
    return result[output_columns]


def intersect_extraction_methods(raw: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    if raw.empty:
        return raw

    work = raw.copy()
    if "normalized_candidate" not in work.columns:
        if "raw_term" not in work.columns:
            raise ValueError("Extraction output must include raw_term or normalized_candidate before intersection.")
        work["normalized_candidate"] = work["raw_term"].map(normalize_term)

    method_sets = []
    for method in methods:
        method_terms = set(
            work.loc[work["extractor"] == method, "normalized_candidate"].tolist()
        )
        method_sets.append(method_terms)

    if not method_sets:
        return work.iloc[0:0].copy()

    shared_terms = set.intersection(*method_sets) if len(method_sets) > 1 else method_sets[0]
    if not shared_terms:
        shared_terms = set.union(*method_sets)
    return work[work["normalized_candidate"].isin(shared_terms)].copy().reset_index(drop=True)


def score_context_similarity(
    candidates: pd.DataFrame,
    context_state: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int,
) -> pd.DataFrame:
    scored = candidates.copy()
    if scored.empty:
        scored["max_context_similarity"] = []
        scored["best_context_term"] = []
        return scored

    if context_state.empty:
        scored["max_context_similarity"] = 0.0
        scored["best_context_term"] = ""
        return scored

    candidate_texts = scored["normalized_candidate"].tolist()
    context_texts = context_state["normalized_candidate"].tolist()
    candidate_embeddings = model.encode(
        candidate_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    context_embeddings = model.encode(
        context_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )

    similarity_matrix = np.asarray(candidate_embeddings) @ np.asarray(context_embeddings).T
    best_indices = similarity_matrix.argmax(axis=1)
    best_scores = similarity_matrix.max(axis=1)

    scored["max_context_similarity"] = best_scores
    scored["best_context_term"] = [context_state.iloc[index]["candidate"] for index in best_indices]
    return scored


def score_seed_similarity(candidates: pd.DataFrame, model: SentenceTransformer, batch_size: int) -> pd.DataFrame:
    scored = candidates.copy()
    if scored.empty:
        scored["max_seed_similarity"] = []
        scored["closest_seed"] = []
        return scored

    candidate_embeddings = model.encode(
        scored["normalized_candidate"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    seed_embeddings = model.encode(
        DOMAIN_SEEDS,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    similarity_matrix = np.asarray(candidate_embeddings) @ np.asarray(seed_embeddings).T
    scored["max_seed_similarity"] = similarity_matrix.max(axis=1)
    scored["closest_seed"] = [DOMAIN_SEEDS[index] for index in similarity_matrix.argmax(axis=1)]
    return scored


@dataclass
class ContextState:
    frame: pd.DataFrame

    @classmethod
    def empty(cls) -> "ContextState":
        return cls(frame=pd.DataFrame())


def update_context_state(
    state: ContextState,
    accepted: pd.DataFrame,
    chunk_row: pd.Series,
    max_context_terms: int,
) -> ContextState:
    if accepted.empty:
        return state

    additions = accepted.copy()
    for column, default in [
        ("score", 0.0),
        ("max_context_similarity", 0.0),
        ("bayesian_posterior", 0.0),
        ("keep_node", 0),
        ("state", "backlog"),
    ]:
        if column not in additions.columns:
            additions[column] = default

    additions = additions[
        [
            "candidate",
            "normalized_candidate",
            "score",
            "max_context_similarity",
            "bayesian_posterior",
            "keep_node",
            "state",
        ]
    ].copy()
    additions["occurrence_count"] = 1
    additions["first_chunk_id"] = safe_text(chunk_row["chunk_id"])
    additions["last_chunk_id"] = safe_text(chunk_row["chunk_id"])
    additions["first_chunk_index"] = int(chunk_row["chunk_index"])
    additions["last_chunk_index"] = int(chunk_row["chunk_index"])

    if state.frame.empty:
        combined = additions
    else:
        combined = pd.concat([state.frame, additions], ignore_index=True, sort=False)

    grouped = combined.groupby("normalized_candidate", as_index=False).agg(
        candidate=("candidate", "first"),
        occurrence_count=("occurrence_count", "sum"),
        best_score=("score", "max"),
        best_context_similarity=("max_context_similarity", "max"),
        best_posterior=("bayesian_posterior", "max"),
        keep_node=("keep_node", "max"),
        state=("state", "last"),
        first_chunk_id=("first_chunk_id", "first"),
        last_chunk_id=("last_chunk_id", "last"),
        first_chunk_index=("first_chunk_index", "min"),
        last_chunk_index=("last_chunk_index", "max"),
    )

    grouped["memory_weight"] = (
        grouped["best_score"].fillna(0.0).map(_bounded) * 0.25
        + grouped["best_context_similarity"].fillna(0.0) * 0.25
        + grouped["best_posterior"].fillna(0.0) * 1.0
        + grouped["keep_node"].fillna(0.0) * 0.50
        + np.log1p(grouped["occurrence_count"].astype(float))
    )
    grouped = grouped.sort_values(
        by=["memory_weight", "best_score", "occurrence_count", "normalized_candidate"],
        ascending=[False, False, False, True],
    ).head(max_context_terms).reset_index(drop=True)
    grouped["memory_rank"] = range(1, len(grouped) + 1)
    return ContextState(frame=grouped)


def filter_with_context(
    scored_candidates: pd.DataFrame,
    context_state: ContextState,
    model: SentenceTransformer,
    similarity_threshold: float,
    batch_size: int,
    fallback_keep_top_n: int,
    hard_drop: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = score_context_similarity(
        scored_candidates,
        context_state.frame,
        model,
        batch_size,
    )
    if context_state.frame.empty:
        scored["context_similarity_pass"] = True
    else:
        scored["context_similarity_pass"] = scored["max_context_similarity"] >= similarity_threshold

    if hard_drop:
        kept = scored[scored["context_similarity_pass"]].copy().reset_index(drop=True)
        dropped = scored[~scored["context_similarity_pass"]].copy().reset_index(drop=True)
        if not dropped.empty:
            dropped["removed_by"] = "context_similarity_filter"
    else:
        kept = scored.reset_index(drop=True)
        dropped = scored.iloc[0:0].copy()
        dropped["removed_by"] = []
    return scored.reset_index(drop=True), kept, dropped


def run_gliner_noise_filter(candidates: pd.DataFrame, allow_fallback: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return candidates.copy(), candidates.copy()

    work = candidates.copy()
    if "raw_term" not in work.columns:
        work["raw_term"] = work.get("candidate", "")
    if "term_canonical" not in work.columns:
        work["term_canonical"] = work.get("normalized_candidate", work.get("candidate", ""))

    kept, dropped = filter_noise(work, allow_fallback=allow_fallback)
    dropped["removed_by"] = dropped.get("noise_label", "gliner_noise")
    return kept, dropped


def seed_similarity_filter(candidates: pd.DataFrame, threshold: float, hard_drop: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        empty = candidates.copy()
        return candidates.copy(), candidates.copy(), empty
    scored = candidates.copy()
    scored["seed_similarity_pass"] = scored["max_seed_similarity"] >= threshold
    if hard_drop:
        kept = scored[scored["seed_similarity_pass"]].copy().reset_index(drop=True)
        dropped = scored[~scored["seed_similarity_pass"]].copy().reset_index(drop=True)
        if not dropped.empty:
            dropped["removed_by"] = "seed_similarity_filter"
    else:
        kept = scored.reset_index(drop=True)
        dropped = scored.iloc[0:0].copy()
        dropped["removed_by"] = []
    return scored.reset_index(drop=True), kept, dropped


def cluster_similarity_filter(
    candidates: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int,
    cluster_count: int,
    context_state: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = candidates.copy()
    if scored.empty:
        scored["cluster_id"] = []
        scored["cluster_size"] = []
        scored["cluster_memory_support_count"] = []
        scored["cluster_context_terms"] = []
        return scored, scored.copy(), scored.copy()

    memory = pd.DataFrame()
    if context_state is not None and not context_state.empty:
        memory = context_state.copy()
        if "normalized_candidate" in memory.columns:
            memory = memory[memory["normalized_candidate"].fillna("").astype(str).ne("")].copy()

    if len(scored) == 1 and memory.empty:
        scored["cluster_id"] = 0
        scored["cluster_size"] = 1
        scored["cluster_memory_support_count"] = 0
        scored["cluster_context_terms"] = ""
        empty = scored.iloc[0:0].copy()
        empty["removed_by"] = []
        return scored.reset_index(drop=True), scored.reset_index(drop=True), empty

    candidate_terms = scored["normalized_candidate"].fillna("").astype(str).tolist()
    memory_terms = memory["normalized_candidate"].fillna("").astype(str).tolist() if not memory.empty else []
    combined_terms = candidate_terms + memory_terms

    combined_embeddings = model.encode(
        combined_terms,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    n_clusters = max(1, min(int(cluster_count), len(combined_terms)))
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clusterer.fit_predict(np.asarray(combined_embeddings))

    candidate_labels = labels[: len(scored)]
    memory_labels = labels[len(scored) :]

    scored["cluster_id"] = candidate_labels
    scored["cluster_size"] = [int(np.sum(labels == label)) for label in candidate_labels]

    context_terms_by_cluster: dict[int, list[str]] = {}
    for label, term in zip(memory_labels, memory_terms):
        context_terms_by_cluster.setdefault(int(label), []).append(term)

    scored["cluster_memory_support_count"] = [
        len(context_terms_by_cluster.get(int(label), [])) for label in candidate_labels
    ]
    scored["cluster_context_terms"] = [
        " | ".join(sorted(set(context_terms_by_cluster.get(int(label), [])))[:10])
        for label in candidate_labels
    ]

    empty = scored.iloc[0:0].copy()
    empty["removed_by"] = []
    return scored.reset_index(drop=True), scored.reset_index(drop=True), empty


def stage_metrics_row(stage_name: str, metrics: dict[str, object], chunk: pd.Series) -> dict[str, object]:
    row = {
        "stage": stage_name,
        "chunk_index": int(chunk.chunk_index),
        "chunk_id": safe_text(chunk.chunk_id),
        "section": safe_text(chunk.section),
    }
    row.update(metrics)
    return row


def summarize_metrics(rows: list[dict[str, object]], label: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["label", "candidate_count", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1", "priority_weighted_recall"])
    frame = pd.DataFrame(rows)
    summary = {
        "label": label,
        "candidate_count": int(frame["candidate_count"].sum()),
        "true_positives": int(frame["true_positives"].sum()),
        "false_positives": int(frame["false_positives"].sum()),
        "false_negatives": int(frame["false_negatives"].sum()),
    }
    tp = summary["true_positives"]
    fp = summary["false_positives"]
    fn = summary["false_negatives"]
    summary["precision"] = tp / (tp + fp) if tp + fp else 0.0
    summary["recall"] = tp / (tp + fn) if tp + fn else 0.0
    summary["f1"] = 2 * summary["precision"] * summary["recall"] / (summary["precision"] + summary["recall"] ) if summary["precision"] + summary["recall"] else 0.0
    summary["priority_weighted_recall"] = float(frame["priority_weighted_recall"].mean()) if len(frame) else 0.0
    return pd.DataFrame([summary])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative chunk pipeline with term-memory context comparison.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--ground-truth-file", type=Path, default=DEFAULT_GROUND_TRUTH_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--precision-profile", choices=["standard", "high"], default="standard")
    parser.add_argument("--context-similarity-threshold-override", type=float, default=None)
    parser.add_argument("--candidate-intersection-methods", nargs="+", default=[])
    parser.add_argument("--extractors", nargs="+", default=DEFAULT_EXTRACTORS)
    parser.add_argument("--embedding-model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--top-k-per-chunk", type=int, default=30)
    parser.add_argument("--chunk-max-words", type=int, default=220)
    parser.add_argument("--chunk-overlap-sentences", type=int, default=1)
    parser.add_argument("--context-similarity-threshold", type=float, default=0.55)
    parser.add_argument("--seed-similarity-threshold", type=float, default=0.30)
    parser.add_argument("--cluster-count", type=int, default=6)
    parser.add_argument("--fallback-keep-top-n", type=int, default=10)
    parser.add_argument("--max-context-terms", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--keybert-model", type=str, default="all-mpnet-base-v2")
    return parser.parse_args()


def apply_precision_profile(args: argparse.Namespace) -> bool:
    high_precision = args.precision_profile == "high"
    if not high_precision:
        return False

    global RETAIN_THRESHOLD, PROVISIONAL_THRESHOLD
    RETAIN_THRESHOLD = HIGH_PRECISION_PROFILE["retain_threshold"]
    PROVISIONAL_THRESHOLD = HIGH_PRECISION_PROFILE["provisional_threshold"]

    args.top_k_per_chunk = min(args.top_k_per_chunk, HIGH_PRECISION_PROFILE["top_k_per_chunk"])
    args.seed_similarity_threshold = max(args.seed_similarity_threshold, HIGH_PRECISION_PROFILE["seed_similarity_threshold"])
    args.context_similarity_threshold = max(args.context_similarity_threshold, HIGH_PRECISION_PROFILE["context_similarity_threshold"])
    args.cluster_count = min(args.cluster_count, HIGH_PRECISION_PROFILE["cluster_count"])
    args.fallback_keep_top_n = min(args.fallback_keep_top_n, HIGH_PRECISION_PROFILE["fallback_keep_top_n"])
    args.max_context_terms = min(args.max_context_terms, HIGH_PRECISION_PROFILE["max_context_terms"])

    if args.extractors:
        args.extractors = [extractor for extractor in args.extractors if extractor in {"yake", "acronyms", "keybert_mpnet"}]
    if not args.extractors:
        args.extractors = ["yake", "acronyms"]

    return True


def apply_context_override(args: argparse.Namespace) -> None:
    if args.context_similarity_threshold_override is not None:
        args.context_similarity_threshold = float(args.context_similarity_threshold_override)


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    if not args.input_file.is_file():
        raise FileNotFoundError(f"Missing input file: {args.input_file}")
    if not args.ground_truth_file.is_file():
        raise FileNotFoundError(f"Missing ground truth workbook: {args.ground_truth_file}")

    high_precision = apply_precision_profile(args)
    apply_context_override(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_chunks"
    alias_dir = args.output_dir / "alias_evidence"
    baseline_dir = args.output_dir / "baseline_chunks"
    iterative_dir = args.output_dir / "iterative_chunks"
    memory_dir = args.output_dir / "memory"
    state_dir = args.output_dir / "state_history"
    for path in (raw_dir, alias_dir, baseline_dir, iterative_dir, memory_dir, state_dir):
        path.mkdir(parents=True, exist_ok=True)

    started = time.time()
    input_hash = file_hash(args.input_file)
    cleaned_text, cleaning_report, chunks = load_chunks_from_input(
        args.input_file,
        max_words=args.chunk_max_words,
        overlap_sentences=args.chunk_overlap_sentences,
    )
    strict_gt, review_gt, negatives = load_ground_truth(args.ground_truth_file)
    extractor_plan = build_extractor_plan(args.extractors, args.top_k_per_chunk, args.keybert_model)
    model = SentenceTransformer(args.embedding_model)

    baseline_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    iterative_rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    state_history_rows: list[dict[str, object]] = []
    alias_evidence_rows: list[dict[str, object]] = []
    all_baseline_candidates: list[pd.DataFrame] = []
    all_iterative_candidates: list[pd.DataFrame] = []

    iterative_state = ContextState.empty()

    for chunk in chunks.itertuples(index=False):
        chunk_row = pd.Series(chunk._asdict())
        raw = extract_chunk_candidates(chunk_row, extractor_plan)
        if args.candidate_intersection_methods:
            raw = intersect_extraction_methods(raw, args.candidate_intersection_methods)
        raw_unique = aggregate_candidates(raw)
        alias_evidence = extract_alias_evidence(chunk_row)
        alias_metadata = build_alias_metadata(alias_evidence)
        raw_unique = apply_alias_metadata(raw_unique, alias_metadata)
        raw_unique = apply_rule_evidence(raw_unique)
        alias_evidence.to_csv(alias_dir / f"chunk_{int(chunk.chunk_index):04d}_alias_evidence.csv", index=False, encoding="utf-8")
        alias_evidence_rows.extend(alias_evidence.to_dict(orient="records"))
        raw_state = annotate_state_frame(raw_unique, "raw_candidates", chunk_row, keep_survivor=True)
        raw_state.to_csv(raw_dir / f"chunk_{int(chunk.chunk_index):04d}.csv", index=False, encoding="utf-8")
        write_chunk_state_tables(raw_state, state_dir, f"chunk_{int(chunk.chunk_index):04d}_raw_candidates")
        append_state_history(state_history_rows, raw_state)

        baseline_metrics = evaluate_terms(
            raw_state,
            strict_gt,
            review_gt,
            negatives,
            mode="baseline_no_context",
            chunk_id=safe_text(chunk.chunk_id),
            chunk_index=int(chunk.chunk_index),
        )
        baseline_rows.append(baseline_metrics)
        stage_rows.append(stage_metrics_row("baseline_no_context", baseline_metrics, chunk))
        raw_state.to_csv(baseline_dir / f"chunk_{int(chunk.chunk_index):04d}.csv", index=False, encoding="utf-8")
        all_baseline_candidates.append(raw_state)

        gliner_kept, gliner_dropped = run_gliner_noise_filter(raw_unique, allow_fallback=not high_precision)
        gliner_kept_state = annotate_state_frame(gliner_kept, "after_gliner", chunk_row, keep_survivor=True)
        gliner_dropped_state = annotate_state_frame(gliner_dropped, "after_gliner", chunk_row, keep_survivor=False)
        gliner_survivors = survivor_rows(gliner_kept_state)
        append_state_history(state_history_rows, gliner_kept_state)
        append_state_history(state_history_rows, gliner_dropped_state)
        gliner_kept_state.to_csv(iterative_dir / f"chunk_{int(chunk.chunk_index):04d}_gliner_kept.csv", index=False, encoding="utf-8")
        gliner_dropped_state.to_csv(iterative_dir / f"chunk_{int(chunk.chunk_index):04d}_gliner_dropped.csv", index=False, encoding="utf-8")
        write_chunk_state_tables(pd.concat([gliner_kept_state, gliner_dropped_state], ignore_index=True), state_dir, f"chunk_{int(chunk.chunk_index):04d}_after_gliner")
        gliner_metrics = evaluate_terms(
            gliner_survivors,
            strict_gt,
            review_gt,
            negatives,
            mode="after_gliner",
            chunk_id=safe_text(chunk.chunk_id),
            chunk_index=int(chunk.chunk_index),
        )
        stage_rows.append(stage_metrics_row("after_gliner", gliner_metrics, chunk))

        seed_scored = score_seed_similarity(gliner_kept_state, model, args.batch_size)
        seed_scored, seed_kept, seed_dropped = seed_similarity_filter(
            seed_scored,
            threshold=args.seed_similarity_threshold,
            hard_drop=high_precision,
        )
        seed_kept_state = annotate_state_frame(seed_kept, "after_seed_similarity", chunk_row, keep_survivor=True)
        seed_dropped_state = annotate_state_frame(seed_dropped, "after_seed_similarity", chunk_row, keep_survivor=False)
        seed_survivors = survivor_rows(seed_kept_state)
        append_state_history(state_history_rows, seed_kept_state)
        append_state_history(state_history_rows, seed_dropped_state)
        seed_kept_state.to_csv(iterative_dir / f"chunk_{int(chunk.chunk_index):04d}_seed_kept.csv", index=False, encoding="utf-8")
        seed_dropped_state.to_csv(iterative_dir / f"chunk_{int(chunk.chunk_index):04d}_seed_dropped.csv", index=False, encoding="utf-8")
        write_chunk_state_tables(pd.concat([seed_kept_state, seed_dropped_state], ignore_index=True), state_dir, f"chunk_{int(chunk.chunk_index):04d}_after_seed_similarity")
        seed_metrics = evaluate_terms(
            seed_survivors,
            strict_gt,
            review_gt,
            negatives,
            mode="after_seed_similarity",
            chunk_id=safe_text(chunk.chunk_id),
            chunk_index=int(chunk.chunk_index),
        )
        stage_rows.append(stage_metrics_row("after_seed_similarity", seed_metrics, chunk))

        cluster_scored, cluster_kept, cluster_dropped = cluster_similarity_filter(
            seed_kept_state,
            model=model,
            batch_size=args.batch_size,
            cluster_count=args.cluster_count,
            context_state=iterative_state.frame,
        )
        cluster_scored_state = annotate_state_frame(cluster_scored, "after_clustering", chunk_row, keep_survivor=True)
        cluster_kept_state = annotate_state_frame(cluster_kept, "after_clustering", chunk_row, keep_survivor=True)
        cluster_dropped_state = annotate_state_frame(cluster_dropped, "after_clustering", chunk_row, keep_survivor=False)
        if high_precision and not cluster_kept_state.empty:
            cluster_gate = (
                cluster_kept_state["cluster_memory_support_count"].fillna(0).astype(float).gt(0)
                | cluster_kept_state["alias_evidence_count"].fillna(0).astype(float).gt(0)
                | cluster_kept_state["rule_support_count"].fillna(0).astype(float).gt(0)
            )
            extra_dropped = cluster_kept_state[~cluster_gate].copy()
            if not extra_dropped.empty:
                extra_dropped["removed_by"] = "high_precision_cluster_gate"
                extra_dropped_state = annotate_state_frame(extra_dropped, "after_clustering", chunk_row, keep_survivor=False)
                cluster_dropped_state = pd.concat([cluster_dropped_state, extra_dropped_state], ignore_index=True)
                cluster_kept_state = cluster_kept_state[cluster_gate].copy().reset_index(drop=True)
        cluster_survivors = survivor_rows(cluster_kept_state)
        append_state_history(state_history_rows, cluster_scored_state)
        append_state_history(state_history_rows, cluster_kept_state)
        append_state_history(state_history_rows, cluster_dropped_state)
        cluster_scored_state.to_csv(iterative_dir / f"chunk_{int(chunk.chunk_index):04d}_clustered_scored.csv", index=False, encoding="utf-8")
        cluster_kept_state.to_csv(iterative_dir / f"chunk_{int(chunk.chunk_index):04d}_clustered_kept.csv", index=False, encoding="utf-8")
        cluster_dropped_state.to_csv(iterative_dir / f"chunk_{int(chunk.chunk_index):04d}_clustered_dropped.csv", index=False, encoding="utf-8")
        write_chunk_state_tables(pd.concat([cluster_kept_state, cluster_dropped_state], ignore_index=True), state_dir, f"chunk_{int(chunk.chunk_index):04d}_after_clustering")
        cluster_metrics = evaluate_terms(
            cluster_survivors,
            strict_gt,
            review_gt,
            negatives,
            mode="after_clustering",
            chunk_id=safe_text(chunk.chunk_id),
            chunk_index=int(chunk.chunk_index),
        )
        stage_rows.append(stage_metrics_row("after_clustering", cluster_metrics, chunk))

        scored, kept, dropped = filter_with_context(
            cluster_kept_state,
            iterative_state,
            model=model,
            similarity_threshold=args.context_similarity_threshold,
            batch_size=args.batch_size,
            fallback_keep_top_n=args.fallback_keep_top_n,
            hard_drop=high_precision,
        )
        scored_state = annotate_state_frame(scored, "iterative_context", chunk_row, keep_survivor=True)
        kept_state = annotate_state_frame(kept, "iterative_context", chunk_row, keep_survivor=True)
        dropped_state = annotate_state_frame(dropped, "iterative_context", chunk_row, keep_survivor=False)
        iterative_survivors = survivor_rows(kept_state)
        append_state_history(state_history_rows, scored_state)
        append_state_history(state_history_rows, kept_state)
        append_state_history(state_history_rows, dropped_state)
        scored_state.to_csv(iterative_dir / f"chunk_{int(chunk.chunk_index):04d}_scored.csv", index=False, encoding="utf-8")
        kept_state.to_csv(iterative_dir / f"chunk_{int(chunk.chunk_index):04d}_kept.csv", index=False, encoding="utf-8")
        dropped_state.to_csv(iterative_dir / f"chunk_{int(chunk.chunk_index):04d}_dropped.csv", index=False, encoding="utf-8")
        write_chunk_state_tables(pd.concat([kept_state, dropped_state], ignore_index=True), state_dir, f"chunk_{int(chunk.chunk_index):04d}_iterative_context")

        iterative_metrics = evaluate_terms(
            iterative_survivors,
            strict_gt,
            review_gt,
            negatives,
            mode="iterative_context",
            chunk_id=safe_text(chunk.chunk_id),
            chunk_index=int(chunk.chunk_index),
        )
        iterative_rows.append(iterative_metrics)
        stage_rows.append(stage_metrics_row("iterative_context", iterative_metrics, chunk))
        all_iterative_candidates.append(iterative_survivors)

        before_memory = len(iterative_state.frame)
        iterative_state = update_context_state(
            iterative_state,
            kept_state,
            chunk_row,
            max_context_terms=args.max_context_terms,
        )
        after_memory = len(iterative_state.frame)

        chunk_rows.append(
            {
                "chunk_index": int(chunk.chunk_index),
                "chunk_id": safe_text(chunk.chunk_id),
                "section": safe_text(chunk.section),
                "raw_candidate_count": int(len(raw_unique)),
                "baseline_candidate_count": int(len(raw_unique)),
                "gliner_kept_candidate_count": int(len(gliner_kept)),
                "gliner_dropped_candidate_count": int(len(gliner_dropped)),
                "seed_kept_candidate_count": int(len(seed_kept)),
                "seed_dropped_candidate_count": int(len(seed_dropped)),
                "cluster_kept_candidate_count": int(len(cluster_kept)),
                "cluster_dropped_candidate_count": int(len(cluster_dropped)),
                "iterative_candidate_count": int(len(kept)),
                "dropped_candidate_count": int(len(dropped)),
                "baseline_f1": baseline_metrics["f1"],
                "gliner_f1": gliner_metrics["f1"],
                "seed_f1": seed_metrics["f1"],
                "cluster_f1": cluster_metrics["f1"],
                "iterative_f1": iterative_metrics["f1"],
                "baseline_precision": baseline_metrics["precision"],
                "gliner_precision": gliner_metrics["precision"],
                "seed_precision": seed_metrics["precision"],
                "cluster_precision": cluster_metrics["precision"],
                "iterative_precision": iterative_metrics["precision"],
                "baseline_recall": baseline_metrics["recall"],
                "gliner_recall": gliner_metrics["recall"],
                "seed_recall": seed_metrics["recall"],
                "cluster_recall": cluster_metrics["recall"],
                "iterative_recall": iterative_metrics["recall"],
                "memory_size_before": before_memory,
                "memory_size_after": after_memory,
                "mean_seed_similarity": float(seed_scored["max_seed_similarity"].mean()) if len(seed_scored) else 0.0,
                "max_seed_similarity": float(seed_scored["max_seed_similarity"].max()) if len(seed_scored) else 0.0,
                "mean_context_similarity": float(scored["max_context_similarity"].mean()) if len(scored) else 0.0,
                "max_context_similarity": float(scored["max_context_similarity"].max()) if len(scored) else 0.0,
            }
        )

    state_history_df = pd.DataFrame(state_history_rows)
    if not state_history_df.empty:
        state_history_df.to_csv(state_dir / "state_history.csv", index=False, encoding="utf-8")

    alias_evidence_df = pd.DataFrame(alias_evidence_rows)
    if not alias_evidence_df.empty:
        alias_evidence_df.to_csv(alias_dir / "alias_evidence_all.csv", index=False, encoding="utf-8")

    final_registry = build_final_registry(state_history_df)
    final_state_counts = {
        "surviving_terms": 0,
        "accepted_terms": 0,
        "provisional_terms": 0,
        "backlog_terms": 0,
        "rejected_terms": 0,
    }
    if not final_registry.empty:
        final_registry.to_csv(args.output_dir / "candidate_registry.csv", index=False, encoding="utf-8")
        final_state_counts = write_final_state_tables(final_registry, args.output_dir)

    extended_gt_evaluations = run_extended_gt_evaluations(args.output_dir, args.ground_truth_file)

    baseline_all = pd.concat(all_baseline_candidates, ignore_index=True) if all_baseline_candidates else pd.DataFrame()
    iterative_all = pd.concat(all_iterative_candidates, ignore_index=True) if all_iterative_candidates else pd.DataFrame()

    baseline_overall = evaluate_terms(
        baseline_all.drop_duplicates("normalized_candidate") if not baseline_all.empty else baseline_all,
        strict_gt,
        review_gt,
        negatives,
        mode="baseline_no_context_overall",
    )
    iterative_overall = evaluate_terms(
        iterative_all.drop_duplicates("normalized_candidate") if not iterative_all.empty else iterative_all,
        strict_gt,
        review_gt,
        negatives,
        mode="iterative_context_overall",
    )

    baseline_df = pd.DataFrame(baseline_rows)
    stage_df = pd.DataFrame(stage_rows)
    iterative_df = pd.DataFrame(iterative_rows)
    chunk_df = pd.DataFrame(chunk_rows)

    baseline_df.to_csv(args.output_dir / "baseline_metrics_by_chunk.csv", index=False, encoding="utf-8")
    stage_df.to_csv(args.output_dir / "stage_metrics_by_chunk.csv", index=False, encoding="utf-8")
    iterative_df.to_csv(args.output_dir / "iterative_metrics_by_chunk.csv", index=False, encoding="utf-8")
    chunk_df.to_csv(args.output_dir / "chunk_comparison.csv", index=False, encoding="utf-8")

    overall_metrics = pd.DataFrame([baseline_overall, iterative_overall])
    overall_metrics.to_csv(args.output_dir / "overall_metrics.csv", index=False, encoding="utf-8")

    stage_overall_rows = []
    for stage_name in ["baseline_no_context", "after_gliner", "after_seed_similarity", "after_clustering", "iterative_context"]:
        stage_rows_for_name = [row for row in stage_rows if row["stage"] == stage_name]
        if stage_rows_for_name:
            summary = summarize_metrics(stage_rows_for_name, stage_name).iloc[0].to_dict()
            summary["stage"] = stage_name
            stage_overall_rows.append(summary)

    if stage_overall_rows:
        pd.DataFrame(stage_overall_rows).to_csv(args.output_dir / "overall_stage_metrics.csv", index=False, encoding="utf-8")

    comparison_summary = pd.DataFrame(
        [
            {
                "metric": "candidate_count",
                "baseline": baseline_overall["candidate_count"],
                "iterative": iterative_overall["candidate_count"],
                "delta": iterative_overall["candidate_count"] - baseline_overall["candidate_count"],
            },
            {
                "metric": "precision",
                "baseline": baseline_overall["precision"],
                "iterative": iterative_overall["precision"],
                "delta": iterative_overall["precision"] - baseline_overall["precision"],
            },
            {
                "metric": "recall",
                "baseline": baseline_overall["recall"],
                "iterative": iterative_overall["recall"],
                "delta": iterative_overall["recall"] - baseline_overall["recall"],
            },
            {
                "metric": "f1",
                "baseline": baseline_overall["f1"],
                "iterative": iterative_overall["f1"],
                "delta": iterative_overall["f1"] - baseline_overall["f1"],
            },
            {
                "metric": "priority_weighted_recall",
                "baseline": baseline_overall["priority_weighted_recall"],
                "iterative": iterative_overall["priority_weighted_recall"],
                "delta": iterative_overall["priority_weighted_recall"] - baseline_overall["priority_weighted_recall"],
            },
        ]
    )
    comparison_summary.to_csv(args.output_dir / "comparison_summary.csv", index=False, encoding="utf-8")

    if not iterative_state.frame.empty:
        iterative_state.frame.to_csv(memory_dir / "iterative_context_memory.csv", index=False, encoding="utf-8")

    cleaned_path = args.output_dir / "cleaned_corpus.txt"
    cleaned_path.write_text(cleaned_text, encoding="utf-8")
    save_df(cleaning_report, args.output_dir / "preprocess_report.csv")
    save_df(chunks, args.output_dir / "chunks.csv")

    manifest = {
        "stage": "iterative_context_compare",
        "input_file": str(args.input_file.resolve()),
        "input_sha256": input_hash,
        "ground_truth_file": str(args.ground_truth_file.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "extractors": args.extractors,
        "candidate_intersection_methods": args.candidate_intersection_methods,
        "embedding_model": args.embedding_model,
        "chunk_count": int(len(chunks)),
        "candidate_count_baseline": int(len(baseline_all)),
        "candidate_count_iterative": int(len(iterative_all)),
        "tracked_state_rows": int(len(state_history_df)),
        "alias_evidence_rows": int(len(alias_evidence_df)),
        "final_registry_rows": int(len(final_registry)),
        "final_state_counts": final_state_counts,
        "extended_gt_evaluations": extended_gt_evaluations,
        "baseline_overall": baseline_overall,
        "iterative_overall": iterative_overall,
        "context_similarity_threshold": float(args.context_similarity_threshold),
        "seed_similarity_threshold": float(args.seed_similarity_threshold),
        "cluster_count": int(args.cluster_count),
        "fallback_keep_top_n": int(args.fallback_keep_top_n),
        "max_context_terms": int(args.max_context_terms),
        "duration_seconds": round(time.time() - started, 3),
    }
    dump_json(args.output_dir / "run_manifest.json", manifest)

    report_lines = [
        "# Iterative Context Pipeline",
        "",
        "This run compares two approaches on the same chunk sequence:",
        "- baseline_no_context: raw chunk extraction without prior-chunk term memory",
        "- iterative_context: raw chunk extraction plus a semantic filter built from prior extracted terms",
        "",
        f"Input: {args.input_file}",
        f"Ground truth: {args.ground_truth_file}",
        f"Chunks: {len(chunks)}",
        f"Extractors: {', '.join(args.extractors)}",
        f"Intersection methods: {', '.join(args.candidate_intersection_methods) if args.candidate_intersection_methods else 'none'}",
        f"Embedding model: {args.embedding_model}",
        f"Context threshold: {args.context_similarity_threshold}",
        f"Cluster count: {args.cluster_count}",
        f"Fallback top-n: {args.fallback_keep_top_n}",
        "",
        "Key outputs:",
        "1. comparison_summary.csv",
        "2. overall_metrics.csv",
        "3. overall_stage_metrics.csv",
        "4. stage_metrics_by_chunk.csv",
        "5. chunk_comparison.csv",
        "6. baseline_metrics_by_chunk.csv",
        "7. iterative_metrics_by_chunk.csv",
        "8. state_history/state_history.csv",
        "9. candidate_registry.csv",
        "10. alias_evidence/alias_evidence_all.csv",
        "11. surviving_terms.csv",
        "12. accepted_terms.csv",
        "13. provisional_terms.csv",
        "14. backlog_terms.csv",
        "15. rejected_terms.csv",
        "16. gt_extended_eval/*/s1_extended_metrics.csv",
        "17. memory/iterative_context_memory.csv",
    ]
    (args.output_dir / "comparison_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return manifest


def main() -> None:
    args = parse_args()
    try:
        manifest = run_pipeline(args)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    except Exception:
        error_path = args.output_dir / "ERROR.txt"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()