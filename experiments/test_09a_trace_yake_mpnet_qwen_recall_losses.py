#!/usr/bin/env python3
"""Trace S1 ground-truth terms through YAKE, MPNet selection, and Qwen.

For every eligible S1_GT_TERMS row, assign exactly one primary group:

1. yake_found_and_qwen_retained
   YAKE generated the GT term and it survived MPNet and Qwen.

2. yake_found_but_lost_by_mpnet
   YAKE generated the GT term, but the MPNet/weighted-selection inventory did
   not retain it.

3. survived_mpnet_but_lost_by_qwen
   YAKE generated the GT term and it survived MPNet, but Qwen removed it.

4. never_found_by_yake
   No YAKE candidate matched the GT term. This includes terms later found by
   another extractor, which are identified by separate stage flags.

Matching modes:
- strict: normalized exact labels
- safe_extended: strict + head singular/plural + explicit S2 aliases
- context_extended: safe + controlled slash components and context-supported
  domain-head completion

The script writes one CSV per group, a complete trace table, summary metrics,
and a Sankey-friendly transition table. It does not train or run MPNet.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_DIR = Path("/home/faujdar/onto_gen_product/ontology_experiment_framework")
DEFAULT_GT = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_MP = PROJECT_DIR / "stage_outputs/03c_weighted_voting/best_f1_candidates.csv"
DEFAULT_QWEN = PROJECT_DIR / "stage_outputs/07b_qwen_validation_yake_mpnet/best_qwen_retained.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "stage_outputs/09a_yake_mpnet_qwen_recall_trace"

S1_SHEET = "S1_GT_TERMS"
S2_SHEET = "S2_GT_ALIASES"
MANUAL_REVIEW = "needs_manual_review"

DOMAIN_HEADS = {
    "uav", "uavs", "uas", "aircraft", "vehicle", "vehicles", "system",
    "systems", "station", "stations", "link", "links", "sensor", "sensors",
    "suite", "suites", "platform", "platforms", "application", "applications",
    "operation", "operations", "classification", "class", "classes", "radar",
    "terminal", "terminals", "communication", "communications", "control",
}
AMBIGUOUS_SHORT_FORMS = {
    "altitude", "autonomy", "classification", "ownership", "control", "risk",
    "safety", "monitoring", "integration", "metrics", "capabilities", "uav",
    "system", "aircraft", "data", "communication",
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


def singularize_head(token: str) -> str:
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


def morph(value: object) -> str:
    tokens = normalize_term(value).split()
    if not tokens:
        return ""
    tokens[-1] = singularize_head(tokens[-1])
    return " ".join(tokens)


def parse_context(value: object) -> str:
    text = safe_text(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return " ".join(safe_text(item) for item in parsed)
        except Exception:
            pass
    return text


def row_context(row: pd.Series) -> str:
    return " ".join(
        part for part in (
            safe_text(row.get("source_sentence", "")),
            safe_text(row.get("context_text", "")),
            parse_context(row.get("additional_contexts", "")),
        ) if part
    )


def phrase_in_context(phrase: str, context: str) -> bool:
    phrase = normalize_term(phrase)
    context = normalize_term(context)
    if not phrase or not context:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return re.search(pattern, context) is not None


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def find_candidate_column(frame: pd.DataFrame) -> str:
    for column in ("candidate", "term", "keyword", "phrase", "normalized_candidate"):
        if column in frame.columns:
            return column
    raise ValueError(f"Could not find candidate column. Columns: {list(frame.columns)}")


def load_inventory(path: Path, stage: str, yake_only: bool = False) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8", low_memory=False)
    candidate_column = find_candidate_column(frame)
    frame = frame.copy()

    if yake_only and "found_by_yake" in frame.columns:
        frame = frame[bool_series(frame["found_by_yake"])].copy()
    elif yake_only and "method" in frame.columns:
        frame = frame[frame["method"].astype(str).str.contains("yake", case=False, na=False)].copy()

    frame["candidate_display"] = frame[candidate_column].map(safe_text)
    frame["candidate_norm"] = frame[candidate_column].map(normalize_term)
    frame["candidate_morph"] = frame[candidate_column].map(morph)
    frame["candidate_context"] = frame.apply(row_context, axis=1)
    frame["stage"] = stage
    return frame[frame["candidate_norm"].ne("")].drop_duplicates("candidate_norm").reset_index(drop=True)


def load_yake_inventory(yake_file: Path | None, mpnet_file: Path) -> tuple[pd.DataFrame, str]:
    if yake_file is not None:
        return load_inventory(yake_file, "yake", yake_only=True), str(yake_file)

    # The weighted-voting candidate file preserves found_by_yake for the full
    # candidate set in the usual project workflow. Use it if no standalone
    # YAKE file is supplied.
    frame = load_inventory(mpnet_file, "yake_source", yake_only=True)
    if frame.empty:
        raise ValueError(
            "No YAKE candidates found in the MPNet/weighted file. Pass a standalone "
            "YAKE output using --yake-file."
        )
    return frame, f"{mpnet_file} [found_by_yake=True]"


def load_gt(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(path)
    s1 = pd.read_excel(path, sheet_name=S1_SHEET, engine="openpyxl")
    s2 = pd.read_excel(path, sheet_name=S2_SHEET, engine="openpyxl")

    s1 = s1[
        s1["keep_for_tasks"].astype(str).str.contains(
            "term_extraction", case=False, na=False
        )
    ].copy()
    status = s1["grounded_status_v2"].fillna("").astype(str).str.casefold()
    s1 = s1[status.ne(MANUAL_REVIEW)].copy()
    s1["gt_norm"] = s1["term_canonical"].map(normalize_term)
    s1["gt_morph"] = s1["term_canonical"].map(morph)
    s1 = s1[s1["gt_norm"].ne("")].drop_duplicates("gt_id").reset_index(drop=True)

    status = s2["grounded_status_v2"].fillna("").astype(str).str.casefold()
    s2 = s2[status.ne(MANUAL_REVIEW)].copy()
    s2["canonical_norm"] = s2["canonical"].map(normalize_term)
    s2["alias_norm"] = s2["alias"].map(normalize_term)
    return s1, s2


def build_alias_graph(s2: pd.DataFrame) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for row in s2.itertuples(index=False):
        if row.canonical_norm and row.alias_norm:
            graph[row.canonical_norm].add(row.alias_norm)
            graph[row.alias_norm].add(row.canonical_norm)
    return graph


def aliases_for_gt(gt_term: str, alias_graph: dict[str, set[str]]) -> set[str]:
    gt = normalize_term(gt_term)
    labels = set(alias_graph.get(gt, set()))
    gt_tokens = gt.split()
    for label, neighbors in alias_graph.items():
        label_tokens = label.split()
        if len(gt_tokens) > len(label_tokens) and gt_tokens[:len(label_tokens)] == label_tokens:
            suffix = gt_tokens[len(label_tokens):]
            if suffix and set(suffix).issubset(DOMAIN_HEADS):
                labels.update(neighbors)
        for neighbor in neighbors:
            nt = neighbor.split()
            if len(gt_tokens) > len(nt) and gt_tokens[:len(nt)] == nt:
                suffix = gt_tokens[len(nt):]
                if suffix and set(suffix).issubset(DOMAIN_HEADS):
                    labels.add(neighbor)
    return labels


def split_components(candidate: str) -> set[str]:
    if not any(separator in candidate for separator in ("/", "|", ";")):
        return set()
    return {
        normalize_term(part)
        for part in re.split(r"[/|;]+", candidate)
        if normalize_term(part)
    }


def completion_matches(candidate: str, gt: str, context: str) -> bool:
    candidate = normalize_term(candidate)
    gt = normalize_term(gt)
    ct = candidate.split()
    gt_tokens = gt.split()
    if not ct or len(gt_tokens) <= len(ct) or candidate in AMBIGUOUS_SHORT_FORMS:
        return False
    if gt_tokens[:len(ct)] == ct:
        missing = gt_tokens[len(ct):]
    elif gt_tokens[-len(ct):] == ct:
        missing = gt_tokens[:-len(ct)]
    else:
        return False
    if not missing or not set(missing).issubset(DOMAIN_HEADS):
        return False
    return phrase_in_context(gt, context) or (
        phrase_in_context(candidate, context)
        and any(phrase_in_context(head, context) for head in missing)
    )


def match_gt_to_inventory(
    gt_row: pd.Series,
    inventory: pd.DataFrame,
    alias_graph: dict[str, set[str]],
    level: str,
) -> tuple[bool, str, str]:
    """Return found, matched candidate, and rule. Prefer safer rules."""
    exact = inventory[inventory["candidate_norm"].eq(gt_row["gt_norm"])]
    if not exact.empty:
        return True, exact.iloc[0]["candidate_display"], "exact"
    if level == "strict":
        return False, "", ""

    morphological = inventory[inventory["candidate_morph"].eq(gt_row["gt_morph"])]
    if not morphological.empty:
        return True, morphological.iloc[0]["candidate_display"], "morphological_variant"

    aliases = aliases_for_gt(gt_row["term_canonical"], alias_graph)
    alias_match = inventory[inventory["candidate_norm"].isin(aliases)]
    if not alias_match.empty:
        return True, alias_match.iloc[0]["candidate_display"], "explicit_s2_alias"
    if level == "safe_extended":
        return False, "", ""

    # Full candidate components, e.g. VLA/BLOS -> VLA and BLOS.
    for row in inventory.itertuples(index=False):
        if gt_row["gt_norm"] in split_components(row.candidate_display):
            return True, row.candidate_display, "compound_component"

    # Short candidate expanded with evidence, e.g. Mini -> Mini UAV.
    for row in inventory.itertuples(index=False):
        if completion_matches(row.candidate_norm, gt_row["gt_norm"], row.candidate_context):
            return True, row.candidate_display, "contextual_head_completion"
    return False, "", ""


def classify_group(yake: bool, mpnet: bool, qwen: bool) -> str:
    if not yake:
        return "4_never_found_by_yake"
    if not mpnet:
        return "2_yake_found_but_lost_by_mpnet"
    if not qwen:
        return "3_survived_mpnet_but_lost_by_qwen"
    return "1_yake_found_and_qwen_retained"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace eligible S1 terms through YAKE, MPNet selection, and Qwen."
    )
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument(
        "--yake-file", type=Path, default=None,
        help=(
            "Standalone/full YAKE candidate CSV. If omitted, the script uses "
            "--mpnet-file rows where found_by_yake=True."
        ),
    )
    parser.add_argument(
        "--mpnet-file", type=Path, default=DEFAULT_MP,
        help="Inventory retained by the YAKE+MPNet/weighted stage before Qwen.",
    )
    parser.add_argument(
        "--qwen-file", type=Path, default=DEFAULT_QWEN,
        help="Final best Qwen-retained inventory.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--matching-level",
        choices=("strict", "safe_extended", "context_extended"),
        default="context_extended",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    s1, s2 = load_gt(args.ground_truth)
    alias_graph = build_alias_graph(s2)
    yake, yake_source = load_yake_inventory(args.yake_file, args.mpnet_file)
    mpnet = load_inventory(args.mpnet_file, "mpnet_selected")
    qwen = load_inventory(args.qwen_file, "qwen_retained")

    rows = []
    for index, gt_row in s1.iterrows():
        y_found, y_candidate, y_rule = match_gt_to_inventory(
            gt_row, yake, alias_graph, args.matching_level
        )
        m_found, m_candidate, m_rule = match_gt_to_inventory(
            gt_row, mpnet, alias_graph, args.matching_level
        )
        q_found, q_candidate, q_rule = match_gt_to_inventory(
            gt_row, qwen, alias_graph, args.matching_level
        )
        group = classify_group(y_found, m_found, q_found)
        rows.append({
            "gt_id": gt_row["gt_id"],
            "gt_term": gt_row["term_canonical"],
            "gt_category": gt_row.get("gt_category", ""),
            "expected_type": gt_row.get("expected_type", ""),
            "priority": gt_row.get("priority", ""),
            "primary_group": group,
            "found_by_yake": y_found,
            "yake_matched_candidate": y_candidate,
            "yake_match_rule": y_rule,
            "survived_mpnet_selection": m_found,
            "mpnet_matched_candidate": m_candidate,
            "mpnet_match_rule": m_rule,
            "retained_by_qwen": q_found,
            "qwen_matched_candidate": q_candidate,
            "qwen_match_rule": q_rule,
            "found_after_yake_by_other_methods": (not y_found and (m_found or q_found)),
        })

    trace = pd.DataFrame(rows)
    trace.to_csv(args.output_dir / "all_s1_term_stage_trace.csv", index=False)

    group_order = [
        "1_yake_found_and_qwen_retained",
        "2_yake_found_but_lost_by_mpnet",
        "3_survived_mpnet_but_lost_by_qwen",
        "4_never_found_by_yake",
    ]
    for group in group_order:
        trace[trace["primary_group"].eq(group)].to_csv(
            args.output_dir / f"{group}.csv", index=False
        )

    total = len(trace)
    summary = []
    for group in group_order:
        subset = trace[trace["primary_group"].eq(group)]
        summary.append({
            "group": group,
            "gt_term_count": len(subset),
            "percentage_of_eligible_s1": len(subset) / total if total else 0.0,
            "high_priority_count": int(
                subset["priority"].fillna("").astype(str).str.casefold().eq("high").sum()
            ),
        })
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(args.output_dir / "four_group_summary.csv", index=False)

    # Stage coverage and transition information.
    stage_summary = pd.DataFrame([
        {
            "stage": "YAKE candidate generation",
            "covered_s1_terms": int(trace["found_by_yake"].sum()),
            "recall": float(trace["found_by_yake"].mean()),
        },
        {
            "stage": "MPNet/weighted selection",
            "covered_s1_terms": int(trace["survived_mpnet_selection"].sum()),
            "recall": float(trace["survived_mpnet_selection"].mean()),
        },
        {
            "stage": "Qwen retained",
            "covered_s1_terms": int(trace["retained_by_qwen"].sum()),
            "recall": float(trace["retained_by_qwen"].mean()),
        },
    ])
    stage_summary.to_csv(args.output_dir / "stage_recall_summary.csv", index=False)

    transitions = pd.DataFrame([
        {"source": "Eligible S1 GT", "target": "Found by YAKE", "count": int(trace.found_by_yake.sum())},
        {"source": "Eligible S1 GT", "target": "Never found by YAKE", "count": int((~trace.found_by_yake).sum())},
        {"source": "Found by YAKE", "target": "Survived MPNet", "count": int((trace.found_by_yake & trace.survived_mpnet_selection).sum())},
        {"source": "Found by YAKE", "target": "Lost by MPNet", "count": int((trace.found_by_yake & ~trace.survived_mpnet_selection).sum())},
        {"source": "Survived MPNet", "target": "Retained by Qwen", "count": int((trace.found_by_yake & trace.survived_mpnet_selection & trace.retained_by_qwen).sum())},
        {"source": "Survived MPNet", "target": "Lost by Qwen", "count": int((trace.found_by_yake & trace.survived_mpnet_selection & ~trace.retained_by_qwen).sum())},
    ])
    transitions.to_csv(args.output_dir / "stage_transitions.csv", index=False)

    runtime = time.perf_counter() - started
    manifest = {
        "run_started_utc": utc_now(),
        "runtime_seconds": runtime,
        "matching_level": args.matching_level,
        "ground_truth": str(args.ground_truth),
        "yake_source": yake_source,
        "mpnet_file": str(args.mpnet_file),
        "qwen_file": str(args.qwen_file),
        "eligible_s1_count": total,
        "inventory_counts": {
            "yake_candidates": len(yake),
            "mpnet_selected_candidates": len(mpnet),
            "qwen_retained_candidates": len(qwen),
        },
        "group_counts": dict(zip(summary_frame.group, summary_frame.gt_term_count)),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 105)
    print("S1 recall-loss trace completed")
    print("=" * 105)
    print(f"Matching level: {args.matching_level}")
    print(f"Eligible S1 terms: {total}")
    print("\nFour groups:")
    print(summary_frame.to_string(
        index=False,
        formatters={"percentage_of_eligible_s1": "{:.2%}".format},
    ))
    print("\nStage recall:")
    print(stage_summary.to_string(index=False, formatters={"recall": "{:.2%}".format}))
    print(f"\nRuntime: {runtime:.2f}s")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
