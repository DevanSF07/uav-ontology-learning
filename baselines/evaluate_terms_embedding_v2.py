"""
evaluate_terms.py

Run the chunk-confidence pipeline on chapter 5 and/or chapter 6,
aggregate node-confidence outputs, evaluate against the shared GT CSV,
and produce a combined evaluation report showing which terms are:

  RETAINED   - node kept AND matches GT  (True Positive)
  FALSE_POS  - node kept NOT in GT       (False Positive)
  FILTERED   - node removed by confidence threshold, but IS in GT (False Negative via filter)
  MISSED     - in GT but never extracted by any method at all
  NOISE_REMOVED - removed by confidence threshold, also not in GT (correct removal)

Usage:
  # Run both chapters then evaluate
  python evaluate_terms.py --output outputs/evaluation_run/

  # Re-use existing pipeline outputs (skip pipeline re-run)
  python evaluate_terms.py --skip_pipeline --output outputs/evaluation_run/

  # Only chapter 5
  python evaluate_terms.py --chapter5_only --output outputs/evaluation_run/

  # Only chapter 6
  python evaluate_terms.py --chapter6_only --output outputs/evaluation_run/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.chunk_confidence_pipeline_embedding_v2 import (
    normalize_term,
    run_chunk_confidence_pipeline,
)
from src.gold_loader import load_gold_term_catalog

CHAPTER_INPUTS = {
    "chapter5": REPO_ROOT / "data/raw_data/Handbook_of_Unmanned_Aerial_Vehicles_chapter5.mmd",
    "chapter6": REPO_ROOT / "data/raw_data/Handbook_of_Unmanned_Aerial_Vehicles_chapter6.mmd",
}

GOLD_CSV = REPO_ROOT / "data/gold/terms_labels_GT.csv"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

REPORT_FIELDS = [
    "normalized_term",
    "representative_term",
    "status",
    "gt_label",
    "document_id",
    "node_confidence",
    "method_support_count",
    "chunk_support_count",
    "dependency_support_count",
    "dependency_sentence_support_count",
    "dependency_rescue_score",
    "evidence_diversity_score",
    "ontology_shape_score",
    "supporting_methods",
    "keep_node",
    "confidence_reason",
]

EVAL_FIELDS = [
    "scope",
    "document_ids",
    "n_gold_terms",
    "n_extracted_all",
    "n_filtered_nodes",
    "tp_retained",
    "fp_retained",
    "fn_filtered_or_missed",
    "fn_filtered",
    "fn_missed",
    "precision",
    "recall",
    "f1",
]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_node_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Node CSV not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _run_one_chapter(
    chapter_key: str,
    input_path: Path,
    output_path: Path,
    config_path: Path,
    document_id: str | None = None,
    initial_node_states: dict[str, dict] | None = None,
    initial_ontology_belief_states: dict[str, dict] | None = None,
) -> Path:
    print(f"\n{'='*60}")
    print(f"Running pipeline: {chapter_key}")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    t0 = time.time()
    result = run_chunk_confidence_pipeline(
        input_path=input_path,
        output_dir=output_path,
        config_path=config_path,
        document_id=document_id,
        initial_node_states=initial_node_states,
        initial_ontology_belief_states=initial_ontology_belief_states,
    )
    elapsed = time.time() - t0
    print(f"Done: {chapter_key} | chunks={result['chunks_processed']}"
          f" | evidence={result['evidence_rows']}"
          f" | nodes={result['node_rows']}"
          f" | filtered_nodes={result['filtered_node_rows']}"
          f" | {elapsed:.1f}s")
    return output_path


def _build_term_report(
    chapter_outputs: dict[str, Path],
    gold_labels: dict[str, str],
    gold_terms: set[str],
    gold_match_index: dict[str, str],
) -> tuple[list[dict], dict[str, dict]]:
    matched_gold_rows: dict[str, dict] = {}
    unmatched_rows: dict[str, dict] = {}

    for chapter_key, chapter_dir in chapter_outputs.items():
        node_csv = chapter_dir / "node_confidence_table.csv"
        for row in _load_node_csv(node_csv):
            norm = row.get("normalized_term", "").strip()
            if not norm:
                continue
            try:
                new_conf = float(row.get("node_confidence", 0) or 0)
            except ValueError:
                new_conf = 0.0
            canonical_norm = gold_match_index.get(norm)
            row_copy = dict(row)
            row_copy["_chapter"] = chapter_key
            if canonical_norm:
                row_copy["normalized_term"] = canonical_norm
                existing = matched_gold_rows.get(canonical_norm)
                if existing is None or new_conf > float(existing.get("node_confidence") or 0):
                    matched_gold_rows[canonical_norm] = row_copy
            else:
                existing = unmatched_rows.get(norm)
                if existing is None or new_conf > float(existing.get("node_confidence") or 0):
                    unmatched_rows[norm] = row_copy

    all_node_rows: dict[str, dict] = {**matched_gold_rows, **unmatched_rows}

    report_rows: list[dict] = []

    for norm, row in sorted(all_node_rows.items()):
        try:
            kept = int(row.get("keep_node", 0)) == 1
        except (ValueError, TypeError):
            kept = False
        in_gold = norm in gold_terms

        if kept and in_gold:
            status = "RETAINED"
        elif kept and not in_gold:
            status = "FALSE_POS"
        elif not kept and in_gold:
            status = "FILTERED"
        else:
            status = "NOISE_REMOVED"

        report_rows.append({
            "normalized_term": norm,
            "representative_term": row.get("representative_term", ""),
            "status": status,
            "gt_label": gold_labels.get(norm, ""),
            "document_id": row.get("document_id", row.get("_chapter", "")),
            "node_confidence": row.get("node_confidence", ""),
            "method_support_count": row.get("method_support_count", ""),
            "chunk_support_count": row.get("chunk_support_count", ""),
            "dependency_support_count": row.get("dependency_support_count", ""),
            "dependency_sentence_support_count": row.get("dependency_sentence_support_count", ""),
            "dependency_rescue_score": row.get("dependency_rescue_score", ""),
            "evidence_diversity_score": row.get("evidence_diversity_score", ""),
            "ontology_shape_score": row.get("ontology_shape_score", ""),
            "supporting_methods": row.get("supporting_methods", ""),
            "keep_node": row.get("keep_node", ""),
            "confidence_reason": row.get("confidence_reason", ""),
        })

    for gold_term in sorted(gold_terms):
        if gold_term not in matched_gold_rows:
            report_rows.append({
                "normalized_term": gold_term,
                "representative_term": "",
                "status": "MISSED",
                "gt_label": gold_labels.get(gold_term, ""),
                "document_id": "",
                "node_confidence": "",
                "method_support_count": "",
                "chunk_support_count": "",
                "dependency_support_count": "",
                "dependency_sentence_support_count": "",
                "dependency_rescue_score": "",
                "evidence_diversity_score": "",
                "ontology_shape_score": "",
                "supporting_methods": "",
                "keep_node": "0",
                "confidence_reason": "never extracted by any method",
            })

    return report_rows, all_node_rows


def _build_diagnostic_summary(report_rows: list[dict]) -> list[dict]:
    summary_rows: list[dict] = []
    statuses = sorted({row.get("status", "") for row in report_rows if row.get("status")})

    for status in statuses:
        subset = [row for row in report_rows if row.get("status") == status]
        if not subset:
            continue

        def _safe_float(value: object) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        dependency_rescued = sum(1 for row in subset if "dependency_support=" in str(row.get("confidence_reason", "")))
        summary_rows.append({
            "status": status,
            "count": len(subset),
            "mean_node_confidence": round(sum(_safe_float(row.get("node_confidence")) for row in subset) / len(subset), 4),
            "mean_method_support_count": round(sum(_safe_float(row.get("method_support_count")) for row in subset) / len(subset), 4),
            "mean_chunk_support_count": round(sum(_safe_float(row.get("chunk_support_count")) for row in subset) / len(subset), 4),
            "mean_dependency_support_count": round(sum(_safe_float(row.get("dependency_support_count")) for row in subset) / len(subset), 4),
            "mean_dependency_rescue_score": round(sum(_safe_float(row.get("dependency_rescue_score")) for row in subset) / len(subset), 4),
            "mean_evidence_diversity_score": round(sum(_safe_float(row.get("evidence_diversity_score")) for row in subset) / len(subset), 4),
            "dependency_rescued_rows": dependency_rescued,
        })

    return summary_rows


def _compute_evaluation(
    report_rows: list[dict],
    gold_terms: set[str],
    chapter_outputs: dict[str, Path],
    all_node_rows: dict[str, dict],
) -> list[dict]:
    by_status: dict[str, list] = {}
    for row in report_rows:
        by_status.setdefault(row["status"], []).append(row["normalized_term"])

    tp = len(by_status.get("RETAINED", []))
    fp = len(by_status.get("FALSE_POS", []))
    fn_filtered = len(by_status.get("FILTERED", []))
    fn_missed = len(by_status.get("MISSED", []))
    fn = fn_filtered + fn_missed
    n_extracted_all = len(all_node_rows)
    n_filtered_nodes = sum(
        1 for row in all_node_rows.values()
        if str(row.get("keep_node", "0")).strip() == "1"
    )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return [{
        "scope": "all_chapters_combined",
        "document_ids": "|".join(chapter_outputs.keys()),
        "n_gold_terms": len(gold_terms),
        "n_extracted_all": n_extracted_all,
        "n_filtered_nodes": n_filtered_nodes,
        "tp_retained": tp,
        "fp_retained": fp,
        "fn_filtered_or_missed": fn,
        "fn_filtered": fn_filtered,
        "fn_missed": fn_missed,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }]


def _print_sample(rows: list[dict], status: str, n: int = 15) -> None:
    subset = [r for r in rows if r["status"] == status]
    print(f"\n--- {status} ({len(subset)} total) — showing first {min(n, len(subset))} ---")
    for r in subset[:n]:
        conf = r.get("node_confidence", "")
        conf_str = f" conf={float(conf):.3f}" if conf else ""
        label = f" [{r['gt_label']}]" if r.get("gt_label") else ""
        print(f"  {r['normalized_term']}{label}{conf_str}")


def _run_bertopic_postprocess(output_root: Path, config_path: Path, evaluate_gold: bool) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "run_bertopic_typing_cluster.py"),
        "--output-root",
        str(output_root),
        "--config",
        str(config_path),
        "--typing-labels",
        "",
        "--include-statuses",
        "candidate,promoted",
        "--min-topic-size",
        "2",
        "--min-cluster-score",
        "0.5",
        "--min-promoted-in-cluster",
        "2",
        "--promote",
        "--promote-probability",
        "0.25",
    ]
    if evaluate_gold:
        command.append("--evaluate-gold")

    print("\n[bertopic] Running post-processing clustering...")
    print(f"[bertopic] Command: {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/evaluation_run")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML pipeline config to use for both chapter runs.",
    )
    parser.add_argument("--skip_pipeline", action="store_true",
                        help="Skip pipeline run; use existing output dirs.")
    parser.add_argument("--chapter5_only", action="store_true")
    parser.add_argument("--chapter6_only", action="store_true")
    parser.add_argument("--run_bertopic", action="store_true", help="Run BERTopic clustering after the baseline evaluation.")
    parser.add_argument("--bertopic-evaluate-gold", action="store_true", help="Evaluate BERTopic promotion suggestions against gold.")
    args = parser.parse_args()

    output_root = Path(args.output)
    config_path = Path(args.config)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.chapter6_only:
        chapters_to_run = {"chapter6": CHAPTER_INPUTS["chapter6"]}
    elif args.chapter5_only:
        chapters_to_run = {"chapter5": CHAPTER_INPUTS["chapter5"]}
    else:
        chapters_to_run = dict(CHAPTER_INPUTS)

    chapter_outputs: dict[str, Path] = {}
    carry_node_states = None
    carry_ontology_states = None
    shared_document_id = None

    if not args.skip_pipeline:
        for chapter_key, input_path in chapters_to_run.items():
            chapter_dir = output_root / chapter_key
            if shared_document_id is None:
                shared_document_id = re.sub(r"_chapter\d+$", "", input_path.stem)
            result = run_chunk_confidence_pipeline(
                input_path=input_path,
                output_dir=chapter_dir,
                config_path=config_path,
                document_id=shared_document_id,
                initial_node_states=carry_node_states,
                initial_ontology_belief_states=carry_ontology_states,
            )
            elapsed_note = ""
            print(
                f"Done: {chapter_key} | chunks={result['chunks_processed']}"
                f" | evidence={result['evidence_rows']}"
                f" | nodes={result['node_rows']}"
                f" | filtered_nodes={result['filtered_node_rows']}"
                f" | carried_forward={'yes' if carry_node_states else 'no'}"
            )
            carry_node_states = result["node_states"]
            carry_ontology_states = result["ontology_belief_states"]
            chapter_outputs[chapter_key] = chapter_dir
    else:
        for chapter_key in chapters_to_run:
            chapter_dir = output_root / chapter_key
            if not (chapter_dir / "node_confidence_table.csv").exists():
                print(f"ERROR: {chapter_dir}/node_confidence_table.csv not found. "
                      f"Remove --skip_pipeline or run pipeline first.")
                sys.exit(1)
            chapter_outputs[chapter_key] = chapter_dir
            print(f"[skip_pipeline] Using: {chapter_dir}")

    gold_terms, gold_match_index, gold_labels = load_gold_term_catalog(GOLD_CSV)

    report_rows, all_node_rows = _build_term_report(chapter_outputs, gold_labels, gold_terms, gold_match_index)
    eval_rows = _compute_evaluation(report_rows, gold_terms, chapter_outputs, all_node_rows)
    diagnostic_rows = _build_diagnostic_summary(report_rows)

    report_path = output_root / "term_status_report.csv"
    eval_path = output_root / "evaluation_summary.csv"
    retained_path = output_root / "retained_terms.csv"
    filtered_path = output_root / "filtered_terms.csv"
    missed_path = output_root / "missed_terms.csv"
    noise_path = output_root / "noise_removed_terms.csv"
    diagnostic_path = output_root / "diagnostic_summary.csv"

    _write_csv(report_path, report_rows, REPORT_FIELDS)
    _write_csv(eval_path, eval_rows, EVAL_FIELDS)
    _write_csv(retained_path, [r for r in report_rows if r["status"] == "RETAINED"], REPORT_FIELDS)
    _write_csv(filtered_path, [r for r in report_rows if r["status"] == "FILTERED"], REPORT_FIELDS)
    _write_csv(missed_path, [r for r in report_rows if r["status"] == "MISSED"], REPORT_FIELDS)
    _write_csv(noise_path, [r for r in report_rows if r["status"] == "NOISE_REMOVED"], REPORT_FIELDS)
    _write_csv(diagnostic_path, diagnostic_rows, [
        "status",
        "count",
        "mean_node_confidence",
        "mean_method_support_count",
        "mean_chunk_support_count",
        "mean_dependency_support_count",
        "mean_dependency_rescue_score",
        "mean_evidence_diversity_score",
        "dependency_rescued_rows",
    ])

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    for row in eval_rows:
        print(f"  Gold terms (GT)           : {row['n_gold_terms']}")
        print(f"  Total extracted nodes     : {row['n_extracted_all']}")
        print(f"  Kept after filtering      : {row['n_filtered_nodes']}")
        print(f"  RETAINED (TP)             : {row['tp_retained']}")
        print(f"  FALSE_POS (FP)            : {row['fp_retained']}")
        print(f"  FILTERED (FN via threshold): {row['fn_filtered']}")
        print(f"  MISSED (FN never extracted): {row['fn_missed']}")
        print(f"  Precision : {row['precision']}")
        print(f"  Recall    : {row['recall']}")
        print(f"  F1        : {row['f1']}")

    _print_sample(report_rows, "RETAINED", 20)
    _print_sample(report_rows, "FILTERED", 20)
    _print_sample(report_rows, "MISSED", 20)

    if args.run_bertopic:
        _run_bertopic_postprocess(output_root, config_path, args.bertopic_evaluate_gold)

    print(f"\nOutputs written to: {output_root}")
    print(f"  term_status_report.csv   — all terms with status")
    print(f"  evaluation_summary.csv   — precision/recall/F1")
    print(f"  retained_terms.csv       — TP terms kept after filter")
    print(f"  filtered_terms.csv       — GT terms removed by threshold")
    print(f"  missed_terms.csv         — GT terms never extracted")
    print(f"  noise_removed_terms.csv  — non-GT terms correctly removed")
    print(f"  diagnostic_summary.csv   — status-level rescue/diversity diagnostics")


if __name__ == "__main__":
    main()
