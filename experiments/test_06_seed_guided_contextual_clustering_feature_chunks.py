from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ontology_experiment_framework import test_06_seed_guided_contextual_clustering as base

PROJECT_DIR = base.PROJECT_DIR
DEFAULT_CONFIG_FILE = PROJECT_DIR / "configs/seed_guided_contextual_clustering_feature_chunks.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "stage_outputs/06_seed_guided_contextual_clustering_feature_chunks"
DEFAULT_FEATURE_CHUNKS_FILE = PROJECT_DIR / "stage_outputs/02a_sae_based_chunking/chunks.csv"


def load_feature_chunk_contexts(feature_chunks_file: Path | None) -> pd.DataFrame:
    if feature_chunks_file is None or not feature_chunks_file.is_file():
        return pd.DataFrame(columns=["chunk_id", "chunk_index", "section", "text"])

    chunks = pd.read_csv(feature_chunks_file, encoding="utf-8")
    if "text" not in chunks.columns:
        return pd.DataFrame(columns=["chunk_id", "chunk_index", "section", "text"])

    for column in ("chunk_id", "chunk_index", "section"):
        if column not in chunks.columns:
            chunks[column] = ""

    chunks = chunks.copy()
    chunks["text"] = chunks["text"].map(base.safe_text)
    return chunks[["chunk_id", "chunk_index", "section", "text"]]


def select_feature_chunk_snippet(
    term: str,
    feature_chunks: pd.DataFrame,
    chunk_sentence_limit: int,
) -> str:
    if feature_chunks.empty:
        return ""

    term_text = base.safe_text(term)
    if not term_text:
        return ""

    term_cf = term_text.casefold()
    matches = feature_chunks[feature_chunks["text"].str.casefold().str.contains(term_cf, na=False)].copy()
    if matches.empty:
        return ""

    matches = matches.sort_values(["chunk_index", "chunk_id"], ascending=[True, True])
    return base.safe_text(matches.iloc[0]["text"])


def load_candidates(
    input_file: Path,
    seed_terms: list[str],
    context_sentence_limit: int,
    feature_chunks_file: Path | None = None,
) -> pd.DataFrame:
    candidates = pd.read_csv(input_file, encoding="utf-8")
    required = {"candidate", "normalized_candidate"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"{input_file} is missing columns: {sorted(missing)}")

    for optional_column in ("source_sentence", "context_text"):
        if optional_column not in candidates.columns:
            candidates[optional_column] = ""

    candidates = candidates.copy()
    candidates["candidate"] = candidates["candidate"].map(base.safe_text)
    candidates["normalized_candidate"] = candidates["normalized_candidate"].map(base.normalize_term)
    candidates["source_sentence"] = candidates["source_sentence"].map(base.safe_text)
    candidates["context_text"] = candidates["context_text"].map(base.safe_text)

    sort_columns = [column for column in ("global_rank", "score") if column in candidates.columns]
    if sort_columns:
        ascending = [True if column == "global_rank" else False for column in sort_columns]
        candidates = candidates.sort_values(sort_columns, ascending=ascending)

    seed_norms = {base.normalize_term(term) for term in seed_terms}
    seed_rows = pd.DataFrame(
        {
            "candidate": seed_terms,
            "normalized_candidate": [base.normalize_term(term) for term in seed_terms],
            "score": [float("inf")] * len(seed_terms),
            "global_rank": [0] * len(seed_terms),
            "source_sentence": [""] * len(seed_terms),
            "context_text": [""] * len(seed_terms),
        }
    )

    pool = pd.concat([candidates, seed_rows], ignore_index=True, sort=False)
    pool = pool[pool["normalized_candidate"].ne("")].copy()
    pool = pool.drop_duplicates("normalized_candidate", keep="first").reset_index(drop=True)
    pool["is_explicit_seed"] = pool["normalized_candidate"].isin(seed_norms)

    feature_chunks = load_feature_chunk_contexts(feature_chunks_file)

    pool["selected_context_snippet"] = pool.apply(
        lambda row: base.select_context_snippet(
            base.safe_text(row.candidate),
            base.safe_text(row.source_sentence),
            base.safe_text(row.context_text),
            context_sentence_limit,
        ),
        axis=1,
    )
    pool["feature_chunk_context"] = pool.apply(
        lambda row: select_feature_chunk_snippet(
            base.safe_text(row.candidate),
            feature_chunks,
            context_sentence_limit,
        ),
        axis=1,
    )
    pool["representation_text"] = pool.apply(
        lambda row: " ".join(
            part
            for part in (
                base.safe_text(row.candidate),
                f"[FEATURE_CONTEXT] {base.safe_text(row.feature_chunk_context)}" if base.safe_text(row.feature_chunk_context) else "",
            )
            if part
        ),
        axis=1,
    )
    return pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_FILE))
    args = parser.parse_args()

    config_path = Path(args.config)
    config = base.load_config(config_path)
    base.validate_inputs(config)

    input_file = Path(config["input_file"])
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "by_method").mkdir(parents=True, exist_ok=True)

    seed_terms = config["seed_terms"]
    started = time.time()

    pool = load_candidates(
        input_file=input_file,
        seed_terms=seed_terms,
        context_sentence_limit=int(config.get("context_sentence_limit", 2)),
        feature_chunks_file=Path(config.get("feature_chunks_file", DEFAULT_FEATURE_CHUNKS_FILE)),
    )

    model = SentenceTransformer(config["embedding_model"])
    embeddings = base.encode_representations(pool, model)
    pool = pool.copy()
    pool["embedding_vector"] = embeddings.tolist()
    scored = base.score_seed_similarity(
        pool=pool,
        embeddings=embeddings,
        seed_terms=seed_terms,
        threshold=float(config["seed_similarity_threshold"]),
    )

    reference_start_indices = set(scored.index[scored["seed_related"]].tolist())
    for seed_term in seed_terms:
        seed_norm = base.normalize_term(seed_term)
        seed_index = int(scored.index[scored["normalized_candidate"].eq(seed_norm)][0])
        reference_start_indices.add(seed_index)

    graph = base.build_similarity_graph(
        embeddings=embeddings,
        neighbor_count=int(config.get("graph_neighbors", 10)),
        similarity_threshold=float(config["graph_similarity_threshold"]),
    )
    reference_depth_map = base.expand_reference_set(
        graph=graph,
        start_indices=reference_start_indices,
        max_depth=int(config["graph_depth"]),
    )
    scored["reference_depth"] = scored.index.map(reference_depth_map.get).fillna(-1).astype(int)
    scored["in_reference_set"] = scored["reference_depth"].ge(0)

    scored = base.cluster_pool(
        pool=scored,
        embeddings=embeddings,
        cluster_methods=config["cluster_methods"],
    )

    clusters_df, cluster_members_df, cluster_membership_map = base.collect_seed_clusters(
        scored=scored,
        cluster_methods=config["cluster_methods"],
        seed_terms=seed_terms,
        reference_depth_map=reference_depth_map,
        stability_jaccard_threshold=float(config["stability_jaccard_threshold"]),
        reference_coverage_threshold=float(config["reference_coverage_threshold"]),
    )

    reference_frame = base.build_reference_frame(scored, reference_depth_map)
    raw_final_candidates = base.build_final_candidates(scored, clusters_df, cluster_membership_map)
    final_candidates, lexical_noise_removed = base.apply_lexical_noise_filter(raw_final_candidates)
    concept_graph = base.build_concept_graph(scored, reference_depth_map)
    overlap_table, best_overlap_table = base.build_cluster_overlap_table(
        cluster_members_df=cluster_members_df,
        clusters_df=clusters_df,
    )
    cross_method_overlap_table = base.build_cross_method_overlap_table(
        scored=scored,
        cluster_membership_map=cluster_membership_map,
        clusters_df=clusters_df,
        overlap_threshold=float(config.get("cross_method_overlap_threshold", 0.5)),
    )

    scored.to_csv(output_dir / "seed_guided_scored_candidates.csv", index=False, encoding="utf-8")
    reference_frame.to_csv(output_dir / "reference_concept_set.csv", index=False, encoding="utf-8")
    clusters_df.to_csv(output_dir / "seed_cluster_summary.csv", index=False, encoding="utf-8")
    cluster_members_df.to_csv(output_dir / "seed_cluster_members.csv", index=False, encoding="utf-8")
    overlap_table.to_csv(output_dir / "cluster_concept_overlap_table.csv", index=False, encoding="utf-8")
    best_overlap_table.to_csv(output_dir / "best_cluster_concept_overlap_table.csv", index=False, encoding="utf-8")
    cross_method_overlap_table.to_csv(output_dir / "cross_method_cluster_overlap_table.csv", index=False, encoding="utf-8")
    raw_final_candidates.to_csv(output_dir / "final_seed_guided_candidates_raw.csv", index=False, encoding="utf-8")
    lexical_noise_removed.to_csv(output_dir / "final_seed_guided_candidates_filtered_lexical_noise.csv", index=False, encoding="utf-8")
    final_candidates.to_csv(output_dir / "final_seed_guided_candidates.csv", index=False, encoding="utf-8")
    gt_metrics = base.compute_gt_metrics(final_candidates)
    (output_dir / "gt_metrics.json").write_text(
        json.dumps(gt_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    graph_nodes = base.plot_concept_graph(
        graph=concept_graph,
        scored=scored,
        final_candidates=final_candidates,
        output_path=output_dir / "concept_graph.png",
    )

    for method in config["cluster_methods"]:
        base.save_method_outputs(output_dir, scored, clusters_df, method["name"])

    manifest = {
        "stage": "06_seed_guided_contextual_clustering_feature_chunks",
        "config_file": str(config_path),
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "feature_chunks_file": str(Path(config.get("feature_chunks_file", DEFAULT_FEATURE_CHUNKS_FILE))),
        "seed_terms": seed_terms,
        "seed_similarity_threshold": float(config["seed_similarity_threshold"]),
        "graph_similarity_threshold": float(config["graph_similarity_threshold"]),
        "graph_depth": int(config["graph_depth"]),
        "reference_coverage_threshold": float(config["reference_coverage_threshold"]),
        "stability_jaccard_threshold": float(config["stability_jaccard_threshold"]),
        "cross_method_overlap_threshold": float(config.get("cross_method_overlap_threshold", 0.5)),
        "cluster_methods": config["cluster_methods"],
        "candidate_count": int(len(scored)),
        "reference_concept_count": int(len(reference_frame)),
        "seed_cluster_count": int(len(clusters_df)),
        "retained_cluster_count": int(clusters_df["retained"].sum()),
        "raw_retained_candidate_count": int(len(raw_final_candidates)),
        "lexical_noise_removed_count": int(len(lexical_noise_removed)),
        "retained_candidate_count": int(len(final_candidates)),
        "best_cluster_overlap_file": str(output_dir / "best_cluster_concept_overlap_table.csv"),
        "cross_method_overlap_file": str(output_dir / "cross_method_cluster_overlap_table.csv"),
        "graph_node_count": int(len(graph_nodes)),
        "graph_edge_count": int(concept_graph.number_of_edges()),
        "duration_seconds": round(time.time() - started, 3),
        "final_candidate_file": str(output_dir / "final_seed_guided_candidates.csv"),
        "graph_plot_file": str(output_dir / "concept_graph.png"),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("Stage 06 seed-guided contextual clustering with feature chunks completed")
    print("=" * 80)
    print(
        clusters_df[
            [
                "method",
                "cluster_label",
                "cluster_size",
                "seed_count",
                "reference_coverage",
                "retained_by_coverage",
                "retained_by_stability",
                "retained",
            ]
        ].to_string(index=False)
    )
    print(f"Cluster overlap table: {output_dir / 'cluster_concept_overlap_table.csv'}")
    print(f"Best cluster overlap table: {output_dir / 'best_cluster_concept_overlap_table.csv'}")
    print(f"Cross-method cluster overlap table: {output_dir / 'cross_method_cluster_overlap_table.csv'}")
    print(f"Lexical-noise filtered candidates: {output_dir / 'final_seed_guided_candidates_filtered_lexical_noise.csv'}")
    print(f"\nFinal retained candidates: {output_dir / 'final_seed_guided_candidates.csv'}")
    print(f"GT metrics: {output_dir / 'gt_metrics.json'}")
    print(f"Concept graph plot: {output_dir / 'concept_graph.png'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_file = PROJECT_DIR / "stage_outputs/06_seed_guided_contextual_clustering_feature_chunks/ERROR_seed_guided_contextual_clustering_feature_chunks.txt"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        error_file.write_text(traceback.format_exc(), encoding="utf-8")
        raise
