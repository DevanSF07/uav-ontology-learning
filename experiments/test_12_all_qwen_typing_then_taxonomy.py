#!/usr/bin/env python3
"""Run entity typing and System/Subsystem taxonomy construction as one workflow.

Stage 1
-------
Runs test_10_entity_typing_all_qwen_terms.py on Qwen-retained terms.

Stage 2
-------
Reads Stage 1 outputs, selects terms predicted as System or Subsystem, and runs
the multi-agent taxonomy builder with these allowed graph semantics:

- System --is_a--> System
- Subsystem --is_a--> Subsystem
- System --has_subsystem--> Subsystem

This orchestrator executes both stages sequentially, checks their required output
artifacts, records a combined manifest, and supports running either stage alone.

Expected companion files in the same project directory:
- test_10_entity_typing_all_qwen_terms.py
- test_11_taxonomy_from_all_typed_terms.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_DIR = Path("/home/faujdar/onto_gen_product/ontology_experiment_framework")
DEFAULT_STAGE1_SCRIPT = PROJECT_DIR / "test_10_entity_typing_all_qwen_terms.py"
DEFAULT_STAGE2_SCRIPT = PROJECT_DIR / "test_11_taxonomy_from_all_typed_terms.py"
DEFAULT_TERMS = PROJECT_DIR / "stage_outputs/07b_qwen_validation_yake_mpnet/best_qwen_retained.csv"
DEFAULT_CORPUS = PROJECT_DIR / "data/combined_corpus_raw.txt"
DEFAULT_TYPING_GT = PROJECT_DIR / "data/Ground_Truth_S1_Term_Typing.xlsx"
DEFAULT_ALIAS_GT = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_STAGE1_OUTPUT = PROJECT_DIR / "stage_outputs/10_entity_typing_all_qwen_terms"
DEFAULT_STAGE2_OUTPUT = PROJECT_DIR / "stage_outputs/11_taxonomy_from_all_typed_terms"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def validate_ollama_url(value: str) -> str:
    """Return a clean Ollama base URL and reject copied HTML markup."""
    url = str(value).strip().rstrip("/")
    if any(token in url for token in ("<", ">", "href=", "&quot;", "&gt;", "&lt;")):
        raise ValueError(
            "Invalid --ollama-url: HTML markup was detected. "
            "Use exactly http://localhost:11434"
        )
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"Invalid --ollama-url {value!r}. "
            "Expected a base URL such as http://localhost:11434"
        )
    return url


def check_ollama(base_url: str, timeout: int = 10) -> None:
    """Fail early with a clear message when the Ollama service is unavailable."""
    request = urllib.request.Request(
        f"{base_url}/api/tags",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status >= 400:
                raise RuntimeError(f"Ollama returned HTTP {response.status}")
    except Exception as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {base_url}. "
            "Start it in another terminal with 'ollama serve', then verify with "
            f"'curl {base_url}/api/tags'. Original error: {exc}"
        ) from exc


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], stage_name: str, log_path: Path) -> float:
    """Run one stage while streaming output and retaining a complete stage log."""
    print("=" * 100)
    print(stage_name)
    print("=" * 100)
    print("Command:")
    print(" ".join(command))
    print()

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()

    runtime = time.perf_counter() - started
    if return_code != 0:
        raise RuntimeError(
            f"{stage_name} failed with exit code {return_code}. See log: {log_path}"
        )
    print(f"\n{stage_name} completed in {runtime:.2f} seconds.\n")
    return runtime


def require_files(paths: list[Path], description: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{description} did not produce the required files:\n- " + "\n- ".join(missing)
        )


def inspect_stage1(stage1_output: Path) -> dict[str, Any]:
    all_results_path = stage1_output / "08_all_qwen_terms_with_entity_types.csv"
    evaluation_path = stage1_output / "09_typing_results_with_gt.csv"
    contexts_path = stage1_output / "03_retrieved_contexts.csv"
    require_files([all_results_path, evaluation_path, contexts_path], "Stage 1")

    all_terms = pd.read_csv(all_results_path, low_memory=False)
    evaluated = read_csv_allow_empty(evaluation_path)
    required = {"canonical_term", "final_type", "final_confidence"}
    missing = required - set(all_terms.columns)
    if missing:
        raise ValueError(f"Stage 1 all-terms output is missing columns: {sorted(missing)}")

    selected = all_terms[all_terms["final_type"].isin(["System", "Subsystem"])].copy()
    stats = {
        "all_qwen_terms_typed": int(len(all_terms)),
        "gt_matched_terms_evaluated": int(len(evaluated)),
        "predicted_system_terms": int((all_terms["final_type"] == "System").sum()),
        "predicted_subsystem_terms": int((all_terms["final_type"] == "Subsystem").sum()),
        "taxonomy_input_terms": int(len(selected)),
        "stage1_all_results": str(all_results_path),
        "stage1_evaluation_results": str(evaluation_path),
        "stage1_contexts": str(contexts_path),
    }

    if "correct" in evaluated.columns and not evaluated.empty:
        values = evaluated["correct"].astype(str).str.casefold()
        known = values.isin(["true", "false"])
        if known.any():
            stats["mapped_typing_accuracy"] = float(values[known].eq("true").mean())

    selected.to_csv(stage1_output / "17_system_subsystem_inventory_from_all_terms.csv", index=False)
    return stats


def read_csv_allow_empty(path: Path) -> pd.DataFrame:
    """Read a CSV, returning an empty frame when the file has no rows or headers."""
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def inspect_stage2(stage2_output: Path) -> dict[str, Any]:
    required = [
        stage2_output / "21_taxonomy_nodes.csv",
        stage2_output / "22_taxonomy_edges.csv",
        stage2_output / "18_system_is_a_edges.csv",
        stage2_output / "19_subsystem_is_a_edges.csv",
        stage2_output / "20_has_subsystem_edges.csv",
    ]
    require_files(required, "Stage 2")

    nodes = read_csv_allow_empty(stage2_output / "21_taxonomy_nodes.csv")
    edges = read_csv_allow_empty(stage2_output / "22_taxonomy_edges.csv")
    system_edges = read_csv_allow_empty(stage2_output / "18_system_is_a_edges.csv")
    subsystem_edges = read_csv_allow_empty(stage2_output / "19_subsystem_is_a_edges.csv")
    composition_edges = read_csv_allow_empty(stage2_output / "20_has_subsystem_edges.csv")

    return {
        "taxonomy_nodes": int(len(nodes)),
        "taxonomy_edges": int(len(edges)),
        "system_is_a_edges": int(len(system_edges)),
        "subsystem_is_a_edges": int(len(subsystem_edges)),
        "has_subsystem_edges": int(len(composition_edges)),
        "stage2_nodes": str(stage2_output / "21_taxonomy_nodes.csv"),
        "stage2_edges": str(stage2_output / "22_taxonomy_edges.csv"),
    }


def build_stage1_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        str(args.stage1_script),
        "--terms", str(args.terms),
        "--corpus", str(args.corpus),
        "--ground-truth", str(args.typing_ground_truth),
        "--typing-sheet", "S1_TERM_TYPING",
        "--gt-term-column", "term_canonical",
        "--gt-type-column", "entity_type",
        "--definitions-sheet", "TYPE_DEFINITIONS",
        "--alias-workbook", str(args.alias_ground_truth),
        "--alias-sheet", "S2_GT_ALIASES",
        "--evaluation-mode", args.evaluation_mode,
        "--model", args.model,
        "--ollama-url", args.ollama_url,
        "--embedding-model", args.embedding_model,
        "--temperature", str(args.temperature),
        "--output-dir", str(args.stage1_output),
    ]
    if args.stage1_max_terms > 0:
        command += ["--max-terms", str(args.stage1_max_terms)]
    if args.rebuild_profiles:
        command.append("--rebuild-profiles")
    if args.skip_ollama_check:
        command.append("--skip-ollama-check")
    return command


def build_stage2_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        str(args.stage2_script),
        "--typing-results", str(args.stage1_output / "08_all_qwen_terms_with_entity_types.csv"),
        "--typing-contexts", str(args.stage1_output / "03_retrieved_contexts.csv"),
        "--alias-workbook", str(args.alias_ground_truth),
        "--alias-sheet", "S2_GT_ALIASES",
        "--relation-gt-sheet", "S5_GT_RELATIONS",
        "--corpus", str(args.corpus),
        "--model", args.model,
        "--ollama-url", args.ollama_url,
        "--embedding-model", args.embedding_model,
        "--temperature", str(args.temperature),
        "--top-k", str(args.top_k),
        "--typing-confidence-threshold", str(args.typing_confidence_threshold),
        "--evidence-quality-threshold", str(args.evidence_quality_threshold),
        "--relation-confidence-threshold", str(args.relation_confidence_threshold),
        "--max-is-a-parents", str(args.max_is_a_parents),
        "--output-dir", str(args.stage2_output),
    ]
    if args.stage2_max_nodes > 0:
        command += ["--max-nodes", str(args.stage2_max_nodes)]
    if args.stage2_max_pairs > 0:
        command += ["--max-pairs", str(args.stage2_max_pairs)]
    if args.disable_node_agent:
        command.append("--disable-node-agent")
    if args.critic_positive_only_when_uncertain:
        command.append("--critic-positive-only-when-uncertain")
    if args.skip_ollama_check:
        command.append("--skip-ollama-check")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--stage1-script", type=Path, default=DEFAULT_STAGE1_SCRIPT)
    parser.add_argument("--stage2-script", type=Path, default=DEFAULT_STAGE2_SCRIPT)
    parser.add_argument("--terms", type=Path, default=DEFAULT_TERMS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--typing-ground-truth", type=Path, default=DEFAULT_TYPING_GT)
    parser.add_argument("--alias-ground-truth", type=Path, default=DEFAULT_ALIAS_GT)
    parser.add_argument("--stage1-output", type=Path, default=DEFAULT_STAGE1_OUTPUT)
    parser.add_argument("--stage2-output", type=Path, default=DEFAULT_STAGE2_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--evaluation-mode", choices=["full", "heldout"], default="full")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--typing-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--evidence-quality-threshold", type=float, default=0.50)
    parser.add_argument("--relation-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--max-is-a-parents", type=int, default=1)
    parser.add_argument("--stage1-max-terms", type=int, default=0)
    parser.add_argument("--stage2-max-nodes", type=int, default=0)
    parser.add_argument("--stage2-max-pairs", type=int, default=0)
    parser.add_argument("--run-stage", choices=["both", "typing", "taxonomy"], default="both")
    parser.add_argument("--rebuild-profiles", action="store_true")
    parser.add_argument("--disable-node-agent", action="store_true")
    parser.add_argument("--critic-positive-only-when-uncertain", action="store_true")
    parser.add_argument("--skip-ollama-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    args.ollama_url = validate_ollama_url(args.ollama_url)
    if not args.skip_ollama_check:
        check_ollama(args.ollama_url)

    required_paths = [args.alias_ground_truth]
    if args.run_stage in {"both", "typing"}:
        required_paths.extend([
            args.stage1_script,
            args.terms,
            args.corpus,
            args.typing_ground_truth,
        ])
    if args.run_stage in {"both", "taxonomy"}:
        required_paths.append(args.stage2_script)
        if args.run_stage == "taxonomy":
            required_paths.extend([
                args.stage1_output / "08_all_qwen_terms_with_entity_types.csv",
                args.stage1_output / "09_typing_results_with_gt.csv",
                args.stage1_output / "03_retrieved_contexts.csv",
            ])
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    args.stage1_output.mkdir(parents=True, exist_ok=True)
    args.stage2_output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "created_utc": now_utc(),
        "run_stage": args.run_stage,
        "model": args.model,
        "embedding_model": args.embedding_model,
        "allowed_taxonomy_edges": [
            "System --is_a--> System",
            "Subsystem --is_a--> Subsystem",
            "System --has_subsystem--> Subsystem",
        ],
        "inputs": {
            "terms": str(args.terms),
            "corpus": str(args.corpus),
            "typing_ground_truth": str(args.typing_ground_truth),
            "alias_ground_truth": str(args.alias_ground_truth),
            "terms_sha256": sha256_file(args.terms),
            "typing_gt_sha256": sha256_file(args.typing_ground_truth),
            "alias_gt_sha256": sha256_file(args.alias_ground_truth),
        },
    }

    if args.run_stage in {"both", "typing"}:
        stage1_runtime = run_command(
            build_stage1_command(args),
            "STAGE 1: MULTI-AGENT ENTITY TYPING",
            args.stage1_output / "combined_stage1_console.log",
        )
        summary["stage1"] = inspect_stage1(args.stage1_output)
        summary["stage1"]["runtime_seconds"] = stage1_runtime
    else:
        summary["stage1"] = inspect_stage1(args.stage1_output)
        summary["stage1"]["runtime_seconds"] = 0.0

    if args.stage1_max_terms == 0 and args.run_stage in {"both", "typing"}:
        source_terms = pd.read_csv(args.terms, low_memory=False)
        expected_terms = int(len(source_terms))
        actual_terms = int(summary["stage1"]["all_qwen_terms_typed"])
        if actual_terms != expected_terms:
            raise RuntimeError(
                f"Stage 1 typed {actual_terms} terms, but the Qwen input contains "
                f"{expected_terms} rows. Taxonomy will not run on an incomplete typing output."
            )

    if summary["stage1"]["taxonomy_input_terms"] == 0:
        raise RuntimeError("Stage 1 produced no terms predicted as System or Subsystem")

    if args.run_stage in {"both", "taxonomy"}:
        stage2_runtime = run_command(
            build_stage2_command(args),
            "STAGE 2: MULTI-AGENT SYSTEM/SUBSYSTEM TAXONOMY",
            args.stage2_output / "combined_stage2_console.log",
        )
        summary["stage2"] = inspect_stage2(args.stage2_output)
        summary["stage2"]["runtime_seconds"] = stage2_runtime

    summary["total_runtime_seconds"] = time.perf_counter() - started
    manifest_root = args.stage2_output if args.run_stage in {"both", "taxonomy"} else args.stage1_output
    manifest_path = manifest_root / "two_stage_run_manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("TWO-STAGE PIPELINE COMPLETED")
    print("=" * 100)
    print(f"All Qwen terms typed: {summary['stage1']['all_qwen_terms_typed']}")
    print(f"GT-matched terms evaluated: {summary['stage1']['gt_matched_terms_evaluated']}")
    print(f"Predicted System terms: {summary['stage1']['predicted_system_terms']}")
    print(f"Predicted Subsystem terms: {summary['stage1']['predicted_subsystem_terms']}")
    if "stage2" in summary:
        print(f"Final taxonomy nodes: {summary['stage2']['taxonomy_nodes']}")
        print(f"System is_a edges: {summary['stage2']['system_is_a_edges']}")
        print(f"Subsystem is_a edges: {summary['stage2']['subsystem_is_a_edges']}")
        print(f"has_subsystem edges: {summary['stage2']['has_subsystem_edges']}")
    print(f"Total runtime: {summary['total_runtime_seconds']:.2f}s")
    print(f"Combined manifest: {manifest_path}")


if __name__ == "__main__":
    main()
