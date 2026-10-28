from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path("ontology_experiment_framework")
DEFAULT_ASSIGNMENTS = (
    PROJECT_DIR
    / "stage_outputs/05_clustering_filtering/clustering/all_cluster_assignments.csv"
)
DEFAULT_SELECTED_RUNS = (
    PROJECT_DIR
    / "stage_outputs/05_clustering_filtering/clustering/selected_algorithm_runs.csv"
)
DEFAULT_CANDIDATES = (
    PROJECT_DIR
    / "stage_outputs/04_gliner_filtering/best_gliner_retained.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "stage_outputs/05_clustering_filtering/cluster_matching"
)

ALGORITHMS = ("hdbscan", "kmeans", "agglomerative")


def safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8")


def validate_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def load_selected_runs(
    selected_runs_file: Path,
    representation: str,
    embedding_space: str,
) -> dict[str, str]:
    selected = read_csv(selected_runs_file)
    validate_columns(
        selected,
        {"representation", "embedding_space", "algorithm", "run_id"},
        "selected_algorithm_runs.csv",
    )

    subset = selected[
        selected["representation"].astype(str).eq(representation)
        & selected["embedding_space"].astype(str).eq(embedding_space)
    ].copy()

    run_ids: dict[str, str] = {}
    for algorithm in ALGORITHMS:
        rows = subset[subset["algorithm"].astype(str).eq(algorithm)]
        if rows.empty:
            raise ValueError(
                f"No selected {algorithm!r} run exists for "
                f"representation={representation!r}, embedding_space={embedding_space!r}."
            )
        run_ids[algorithm] = str(rows.iloc[0]["run_id"])

    return run_ids


def load_run_assignments(
    assignments_file: Path,
    run_ids: dict[str, str],
    include_rejected: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load selected runs and return HDBSCAN native noise separately.

    HDBSCAN cluster label -1 is never included in cross-method cluster matching.
    It is returned as a separate dataframe for independent noise inspection.
    """
    assignments = read_csv(assignments_file)
    validate_columns(
        assignments,
        {
            "run_id",
            "algorithm",
            "candidate_row_id",
            "candidate",
            "normalized_candidate",
            "cluster_label",
            "retain",
        },
        "all_cluster_assignments.csv",
    )

    result: dict[str, pd.DataFrame] = {}
    hdbscan_noise = pd.DataFrame()

    for algorithm, run_id in run_ids.items():
        frame = assignments[assignments["run_id"].astype(str).eq(run_id)].copy()
        if frame.empty:
            raise ValueError(f"No assignments found for selected run: {run_id}")

        frame["candidate_row_id"] = pd.to_numeric(
            frame["candidate_row_id"], errors="raise"
        ).astype(int)
        frame["cluster_label"] = pd.to_numeric(
            frame["cluster_label"], errors="raise"
        ).astype(int)
        frame["retain"] = frame["retain"].astype(str).str.casefold().map(
            {"true": True, "false": False}
        ).fillna(frame["retain"].astype(bool))

        if frame["candidate_row_id"].duplicated().any():
            raise ValueError(f"Duplicate candidate rows found in {run_id}")

        if algorithm == "hdbscan":
            hdbscan_noise = frame[frame["cluster_label"].eq(-1)].copy()
            hdbscan_noise["noise_type"] = "hdbscan_native_noise"
            hdbscan_noise["selected_run_id"] = run_id

        # Native HDBSCAN noise is excluded from semantic cluster equivalence.
        semantic_clusters = frame[frame["cluster_label"].ge(0)].copy()
        if not include_rejected:
            semantic_clusters = semantic_clusters[semantic_clusters["retain"]]

        result[algorithm] = semantic_clusters.reset_index(drop=True)

    return result, hdbscan_noise.reset_index(drop=True)


def build_cluster_sets(frame: pd.DataFrame) -> dict[int, set[int]]:
    clusters: dict[int, set[int]] = {}
    for cluster_label, group in frame.groupby("cluster_label", sort=True):
        clusters[int(cluster_label)] = set(group["candidate_row_id"].astype(int))
    return clusters


def pair_overlap(left: set[int], right: set[int]) -> dict[str, float | int]:
    intersection = len(left & right)
    union = len(left | right)
    minimum = min(len(left), len(right))
    maximum = max(len(left), len(right))
    return {
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 0.0,
        "overlap_coefficient": intersection / minimum if minimum else 0.0,
        "left_coverage": intersection / len(left) if left else 0.0,
        "right_coverage": intersection / len(right) if right else 0.0,
        "size_ratio": minimum / maximum if maximum else 0.0,
    }


def triple_overlap(
    hdbscan_members: set[int],
    kmeans_members: set[int],
    agglomerative_members: set[int],
) -> dict[str, float | int]:
    triple_intersection = hdbscan_members & kmeans_members & agglomerative_members
    triple_union = hdbscan_members | kmeans_members | agglomerative_members
    sizes = [len(hdbscan_members), len(kmeans_members), len(agglomerative_members)]
    minimum_size = min(sizes)
    maximum_size = max(sizes)

    hk = pair_overlap(hdbscan_members, kmeans_members)
    ha = pair_overlap(hdbscan_members, agglomerative_members)
    ka = pair_overlap(kmeans_members, agglomerative_members)
    pair_jaccards = [hk["jaccard"], ha["jaccard"], ka["jaccard"]]
    pair_overlaps = [
        hk["overlap_coefficient"],
        ha["overlap_coefficient"],
        ka["overlap_coefficient"],
    ]

    return {
        "hdbscan_size": sizes[0],
        "kmeans_size": sizes[1],
        "agglomerative_size": sizes[2],
        "triple_intersection_count": len(triple_intersection),
        "triple_union_count": len(triple_union),
        "triple_jaccard": (
            len(triple_intersection) / len(triple_union) if triple_union else 0.0
        ),
        "triple_overlap_coefficient": (
            len(triple_intersection) / minimum_size if minimum_size else 0.0
        ),
        "size_ratio": minimum_size / maximum_size if maximum_size else 0.0,
        "hk_jaccard": hk["jaccard"],
        "ha_jaccard": ha["jaccard"],
        "ka_jaccard": ka["jaccard"],
        "minimum_pair_jaccard": min(pair_jaccards),
        "mean_pair_jaccard": float(np.mean(pair_jaccards)),
        "minimum_pair_overlap": min(pair_overlaps),
        "mean_pair_overlap": float(np.mean(pair_overlaps)),
    }


def generate_candidate_matches(
    cluster_sets: dict[str, dict[int, set[int]]],
    minimum_intersection: int,
    minimum_pair_jaccard: float,
    minimum_triple_overlap: float,
    minimum_size_ratio: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for h_label, k_label, a_label in itertools.product(
        cluster_sets["hdbscan"],
        cluster_sets["kmeans"],
        cluster_sets["agglomerative"],
    ):
        h_members = cluster_sets["hdbscan"][h_label]
        k_members = cluster_sets["kmeans"][k_label]
        a_members = cluster_sets["agglomerative"][a_label]

        metrics = triple_overlap(h_members, k_members, a_members)
        if metrics["triple_intersection_count"] < minimum_intersection:
            continue
        if metrics["minimum_pair_jaccard"] < minimum_pair_jaccard:
            continue
        if metrics["triple_overlap_coefficient"] < minimum_triple_overlap:
            continue
        if metrics["size_ratio"] < minimum_size_ratio:
            continue

        # Consensus score rewards pairwise agreement, common membership, and size balance.
        consensus_score = (
            0.40 * metrics["mean_pair_jaccard"]
            + 0.30 * metrics["triple_overlap_coefficient"]
            + 0.20 * metrics["minimum_pair_overlap"]
            + 0.10 * metrics["size_ratio"]
        )

        consensus_members = h_members & k_members & a_members
        union_members = h_members | k_members | a_members
        rows.append(
            {
                "hdbscan_cluster": h_label,
                "kmeans_cluster": k_label,
                "agglomerative_cluster": a_label,
                **metrics,
                "consensus_score": consensus_score,
                "consensus_member_ids": sorted(consensus_members),
                "union_member_ids": sorted(union_members),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        [
            "consensus_score",
            "triple_intersection_count",
            "minimum_pair_jaccard",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def greedy_one_to_one_selection(matches: pd.DataFrame) -> pd.DataFrame:
    """Select non-conflicting triplets in descending consensus-score order."""
    if matches.empty:
        return matches.copy()

    used_h: set[int] = set()
    used_k: set[int] = set()
    used_a: set[int] = set()
    selected_rows = []

    for row in matches.itertuples(index=False):
        if row.hdbscan_cluster in used_h:
            continue
        if row.kmeans_cluster in used_k:
            continue
        if row.agglomerative_cluster in used_a:
            continue
        selected_rows.append(row._asdict())
        used_h.add(row.hdbscan_cluster)
        used_k.add(row.kmeans_cluster)
        used_a.add(row.agglomerative_cluster)

    selected = pd.DataFrame(selected_rows)
    if not selected.empty:
        selected.insert(0, "matched_cluster_id", [
            f"MATCH_{index:03d}" for index in range(1, len(selected) + 1)
        ])
    return selected


def enrich_members(
    selected_matches: pd.DataFrame,
    candidates_file: Path,
    run_assignments: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected_matches.empty:
        return pd.DataFrame(), pd.DataFrame()

    candidates = read_csv(candidates_file)
    if "candidate_row_id" not in candidates.columns:
        candidates = candidates.copy()
        candidates.insert(0, "candidate_row_id", np.arange(len(candidates), dtype=int))
    validate_columns(
        candidates,
        {"candidate_row_id", "candidate", "normalized_candidate"},
        "candidate inventory",
    )
    candidates["candidate_row_id"] = pd.to_numeric(
        candidates["candidate_row_id"], errors="raise"
    ).astype(int)

    optional_columns = [
        "match_type",
        "evaluation_flag_before_filter",
        "matched_gt_id",
        "gt_category",
        "negative_benchmark_hit",
        "source_sentence",
    ]
    available = [column for column in optional_columns if column in candidates.columns]
    candidate_lookup = candidates.set_index("candidate_row_id")

    assignment_maps = {}
    for algorithm, frame in run_assignments.items():
        assignment_maps[algorithm] = frame.set_index("candidate_row_id")["cluster_label"].to_dict()

    member_rows = []
    summary_rows = []

    for match in selected_matches.itertuples(index=False):
        consensus_ids = set(match.consensus_member_ids)
        union_ids = set(match.union_member_ids)

        match_member_rows = []
        for candidate_id in sorted(union_ids):
            if candidate_id not in candidate_lookup.index:
                continue
            candidate = candidate_lookup.loc[candidate_id]
            h_cluster = assignment_maps["hdbscan"].get(candidate_id)
            k_cluster = assignment_maps["kmeans"].get(candidate_id)
            a_cluster = assignment_maps["agglomerative"].get(candidate_id)

            in_h = h_cluster == match.hdbscan_cluster
            in_k = k_cluster == match.kmeans_cluster
            in_a = a_cluster == match.agglomerative_cluster
            vote_count = int(in_h) + int(in_k) + int(in_a)

            row = {
                "matched_cluster_id": match.matched_cluster_id,
                "candidate_row_id": candidate_id,
                "candidate": candidate["candidate"],
                "normalized_candidate": candidate["normalized_candidate"],
                "hdbscan_cluster": match.hdbscan_cluster,
                "kmeans_cluster": match.kmeans_cluster,
                "agglomerative_cluster": match.agglomerative_cluster,
                "in_hdbscan_cluster": in_h,
                "in_kmeans_cluster": in_k,
                "in_agglomerative_cluster": in_a,
                "cluster_membership_vote_count": vote_count,
                "is_three_way_consensus_member": candidate_id in consensus_ids,
                "consensus_score": match.consensus_score,
            }
            for column in available:
                row[column] = candidate[column]
            match_member_rows.append(row)
            member_rows.append(row)

        member_frame = pd.DataFrame(match_member_rows)
        consensus_frame = member_frame[
            member_frame["is_three_way_consensus_member"]
        ] if not member_frame.empty else pd.DataFrame()

        summary = {
            "matched_cluster_id": match.matched_cluster_id,
            "hdbscan_cluster": match.hdbscan_cluster,
            "kmeans_cluster": match.kmeans_cluster,
            "agglomerative_cluster": match.agglomerative_cluster,
            "consensus_score": match.consensus_score,
            "three_way_consensus_size": len(consensus_frame),
            "union_size": len(member_frame),
            "consensus_terms": " | ".join(
                consensus_frame["candidate"].astype(str).tolist()
            ) if not consensus_frame.empty else "",
        }
        if "match_type" in consensus_frame.columns:
            summary["strict_gt_count"] = int(
                consensus_frame["match_type"].eq("strict_exact").sum()
            )
            summary["unmatched_count"] = int(
                consensus_frame["match_type"].eq("unmatched").sum()
            )
        summary_rows.append(summary)

    return pd.DataFrame(member_rows), pd.DataFrame(summary_rows)


def write_per_cluster_files(
    members: pd.DataFrame,
    output_dir: Path,
) -> None:
    cluster_dir = output_dir / "matched_clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    if members.empty:
        return
    for matched_cluster_id, group in members.groupby("matched_cluster_id", sort=True):
        group.sort_values(
            ["is_three_way_consensus_member", "cluster_membership_vote_count", "candidate"],
            ascending=[False, False, True],
        ).to_csv(
            cluster_dir / f"{matched_cluster_id}.csv",
            index=False,
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Match equivalent clusters produced by selected HDBSCAN, K-means, and "
            "Agglomerative runs using member overlap, then return three-way consensus clusters."
        )
    )
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--selected-runs", type=Path, default=DEFAULT_SELECTED_RUNS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--representation", default="term_sentence")
    parser.add_argument("--embedding-space", default="raw")
    parser.add_argument(
        "--minimum-intersection",
        type=int,
        default=6,
        help="Minimum number of candidates shared by all three clusters. Default 6 means more than 5 matching terms.",
    )
    parser.add_argument(
        "--minimum-pair-jaccard",
        type=float,
        default=0.20,
        help="Minimum Jaccard similarity required for every pair of clusters.",
    )
    parser.add_argument(
        "--minimum-triple-overlap",
        type=float,
        default=0.50,
        help=(
            "Minimum three-way overlap coefficient: size of three-way intersection "
            "divided by the smallest cluster size."
        ),
    )
    parser.add_argument(
        "--minimum-size-ratio",
        type=float,
        default=0.25,
        help="Minimum smallest-cluster/largest-cluster size ratio.",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include candidates rejected as centroid outliers. HDBSCAN label -1 remains excluded.",
    )
    parser.add_argument(
        "--allow-many-to-many",
        action="store_true",
        help="Return every qualifying triplet instead of greedy one-to-one cluster matches.",
    )
    args = parser.parse_args()

    for name, value in (
        ("minimum_pair_jaccard", args.minimum_pair_jaccard),
        ("minimum_triple_overlap", args.minimum_triple_overlap),
        ("minimum_size_ratio", args.minimum_size_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.minimum_intersection < 1:
        raise ValueError("--minimum-intersection must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_ids = load_selected_runs(
        args.selected_runs,
        args.representation,
        args.embedding_space,
    )
    assignments, hdbscan_noise = load_run_assignments(
        args.assignments,
        run_ids,
        include_rejected=args.include_rejected,
    )

    # HDBSCAN native noise is handled independently from cluster matching.
    hdbscan_noise.to_csv(
        args.output_dir / "hdbscan_noise_cluster.csv",
        index=False,
        encoding="utf-8",
    )
    hdbscan_noise_summary = pd.DataFrame([{
        "selected_hdbscan_run": run_ids["hdbscan"],
        "noise_cluster_label": -1,
        "noise_candidate_count": len(hdbscan_noise),
        "mean_membership_probability": (
            pd.to_numeric(hdbscan_noise.get("membership_probability"), errors="coerce").mean()
            if not hdbscan_noise.empty and "membership_probability" in hdbscan_noise.columns
            else np.nan
        ),
        "mean_outlier_score": (
            pd.to_numeric(hdbscan_noise.get("outlier_score"), errors="coerce").mean()
            if not hdbscan_noise.empty and "outlier_score" in hdbscan_noise.columns
            else np.nan
        ),
    }])
    hdbscan_noise_summary.to_csv(
        args.output_dir / "hdbscan_noise_summary.csv",
        index=False,
        encoding="utf-8",
    )
    cluster_sets = {
        algorithm: build_cluster_sets(frame)
        for algorithm, frame in assignments.items()
    }

    candidate_matches = generate_candidate_matches(
        cluster_sets,
        minimum_intersection=args.minimum_intersection,
        minimum_pair_jaccard=args.minimum_pair_jaccard,
        minimum_triple_overlap=args.minimum_triple_overlap,
        minimum_size_ratio=args.minimum_size_ratio,
    )

    if candidate_matches.empty:
        candidate_matches.to_csv(
            args.output_dir / "all_qualifying_cluster_triplets.csv", index=False
        )
        print("No equivalent three-way clusters met the configured thresholds.")
        print("Try lowering --minimum-pair-jaccard or --minimum-triple-overlap.")
        return

    # Preserve list columns for internal processing and serialize only for CSV output.
    selected_matches = (
        candidate_matches.copy()
        if args.allow_many_to_many
        else greedy_one_to_one_selection(candidate_matches)
    )
    if args.allow_many_to_many and not selected_matches.empty:
        selected_matches.insert(0, "matched_cluster_id", [
            f"MATCH_{index:03d}" for index in range(1, len(selected_matches) + 1)
        ])

    members, summaries = enrich_members(
        selected_matches,
        args.candidates,
        assignments,
    )

    def serializable(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        for column in ("consensus_member_ids", "union_member_ids"):
            if column in output.columns:
                output[column] = output[column].map(json.dumps)
        return output

    serializable(candidate_matches).to_csv(
        args.output_dir / "all_qualifying_cluster_triplets.csv",
        index=False,
        encoding="utf-8",
    )
    serializable(selected_matches).to_csv(
        args.output_dir / "matched_cluster_triplets.csv",
        index=False,
        encoding="utf-8",
    )
    summaries.to_csv(
        args.output_dir / "matched_cluster_summary.csv",
        index=False,
        encoding="utf-8",
    )
    members.to_csv(
        args.output_dir / "matched_cluster_members.csv",
        index=False,
        encoding="utf-8",
    )
    if not members.empty:
        members[members["is_three_way_consensus_member"]].to_csv(
            args.output_dir / "three_way_consensus_members.csv",
            index=False,
            encoding="utf-8",
        )
    write_per_cluster_files(members, args.output_dir)

    manifest = {
        "representation": args.representation,
        "embedding_space": args.embedding_space,
        "selected_run_ids": run_ids,
        "include_rejected": args.include_rejected,
        "allow_many_to_many": args.allow_many_to_many,
        "thresholds": {
            "minimum_intersection": args.minimum_intersection,
            "minimum_pair_jaccard": args.minimum_pair_jaccard,
            "minimum_triple_overlap": args.minimum_triple_overlap,
            "minimum_size_ratio": args.minimum_size_ratio,
        },
        "cluster_counts": {
            algorithm: len(cluster_sets[algorithm]) for algorithm in ALGORITHMS
        },
        "qualifying_triplet_count": len(candidate_matches),
        "returned_match_count": len(selected_matches),
        "three_way_consensus_member_count": int(
            members["is_three_way_consensus_member"].sum()
        ) if not members.empty else 0,
        "hdbscan_native_noise_count": len(hdbscan_noise),
    }
    (args.output_dir / "cluster_matching_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("=" * 88)
    print("Three-way cluster matching completed")
    print("=" * 88)
    print(f"Representation: {args.representation}")
    print(f"Embedding space: {args.embedding_space}")
    print(f"HDBSCAN run: {run_ids['hdbscan']}")
    print(f"K-means run: {run_ids['kmeans']}")
    print(f"Agglomerative run: {run_ids['agglomerative']}")
    print(f"Qualifying cluster triplets: {len(candidate_matches)}")
    print(f"Returned cluster matches: {len(selected_matches)}")
    print(f"HDBSCAN native noise candidates: {len(hdbscan_noise)}")
    print(f"Output: {args.output_dir}")
    if not summaries.empty:
        display_columns = [
            "matched_cluster_id",
            "hdbscan_cluster",
            "kmeans_cluster",
            "agglomerative_cluster",
            "consensus_score",
            "three_way_consensus_size",
            "union_size",
        ]
        print("\nTop matched clusters:")
        print(summaries[display_columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
