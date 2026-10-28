"""Sweep cosine thresholds for term-only and contextual MPNet seed graphs.

For each representation:
    1. Embed every candidate once with MPNet.
    2. Calculate one N x N cosine-similarity matrix.
    3. For every threshold, perform multi-source BFS from:
           UAV
           unmanned aerial vehicle
       up to the configured maximum layer.
    4. Evaluate the discovered graph nodes against strict ground truth.
    5. Plot threshold on the x-axis and F1 on the y-axis.

Artificially inserted seed nodes and manual-review GT terms are excluded from
strict evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path("/home/faujdar/onto_gen/ontology_experiment_framework")
DEFAULT_INPUT = (
    PROJECT_DIR
    / "stage_outputs/05_seed_similarity_after_gliner/best_seed_filtered_candidates.csv"
)
DEFAULT_GT = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_OUTPUT = PROJECT_DIR / "stage_outputs/06_seed_graph_threshold_sweep"
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_SEEDS = ("UAV", "unmanned aerial vehicle")
GT_SHEET = "S1_GT_TERMS"
NEGATIVE_SHEET = "S7_NEGATIVE_TERMS"
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


def safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_candidates(path: Path, seeds: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, encoding="utf-8")
    if "candidate" not in frame.columns:
        raise ValueError("Input CSV must contain a 'candidate' column")

    frame = frame.copy()
    if "normalized_candidate" not in frame.columns:
        frame["normalized_candidate"] = frame["candidate"].map(normalize_term)
    else:
        frame["normalized_candidate"] = frame["normalized_candidate"].map(normalize_term)

    frame["candidate"] = frame["candidate"].map(safe_text)
    for column in ("source_sentence", "context_text"):
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].map(safe_text)

    frame = frame[frame["normalized_candidate"].ne("")]
    frame = frame.drop_duplicates("normalized_candidate", keep="first").reset_index(drop=True)
    frame["was_added_seed"] = False

    existing = set(frame["normalized_candidate"])
    additions: list[dict[str, Any]] = []
    for seed in seeds:
        normalized = normalize_term(seed)
        if normalized and normalized not in existing:
            record = {column: "" for column in frame.columns}
            record["candidate"] = seed
            record["normalized_candidate"] = normalized
            record["source_sentence"] = ""
            record["context_text"] = ""
            record["was_added_seed"] = True
            additions.append(record)
            existing.add(normalized)

    if additions:
        frame = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)

    frame.insert(0, "matrix_row_id", np.arange(len(frame), dtype=int))
    return frame


def load_ground_truth(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(path)

    gt = pd.read_excel(path, sheet_name=GT_SHEET, engine="openpyxl")
    negatives = pd.read_excel(path, sheet_name=NEGATIVE_SHEET, engine="openpyxl")

    required = {"term_canonical", "keep_for_tasks", "grounded_status_v2"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"Ground-truth sheet is missing columns: {sorted(missing)}")

    gt = gt[
        gt["keep_for_tasks"].astype(str).str.contains(
            "term_extraction", case=False, na=False
        )
    ].copy()
    gt["normalized_gt"] = gt["term_canonical"].map(normalize_term)
    gt = gt[gt["normalized_gt"].ne("")]

    status = gt["grounded_status_v2"].fillna("").astype(str).str.casefold()
    strict_gt = gt[status.ne(MANUAL_REVIEW_STATUS)].drop_duplicates("normalized_gt").copy()
    review_gt = gt[status.eq(MANUAL_REVIEW_STATUS)].drop_duplicates("normalized_gt").copy()

    if "candidate" not in negatives.columns:
        raise ValueError("S7_NEGATIVE_TERMS must contain a 'candidate' column")
    negatives["normalized_negative"] = negatives["candidate"].map(normalize_term)
    negatives = negatives[negatives["normalized_negative"].ne("")]
    return strict_gt, review_gt, negatives


def annotate_candidates(
    candidates: pd.DataFrame,
    strict_gt: pd.DataFrame,
    review_gt: pd.DataFrame,
    negatives: pd.DataFrame,
) -> pd.DataFrame:
    strict_terms = set(strict_gt["normalized_gt"])
    review_terms = set(review_gt["normalized_gt"])
    negative_terms = set(negatives["normalized_negative"])

    result = candidates.copy()
    result["match_type"] = result["normalized_candidate"].map(
        lambda term: "strict_exact" if term in strict_terms else (
            "manual_review_exact" if term in review_terms else "unmatched"
        )
    )
    result["negative_benchmark_hit"] = result["normalized_candidate"].isin(negative_terms)
    return result


def embedding_text(row: Any, representation: str) -> str:
    term = safe_text(row.candidate)
    if representation == "term":
        return term
    if representation == "term_sentence":
        sentence = safe_text(row.source_sentence)
        return f"Term: {term}. Context: {sentence}" if sentence else f"Term: {term}"
    raise ValueError(f"Unknown representation: {representation}")


def encode_representation(
    candidates: pd.DataFrame,
    model: SentenceTransformer,
    representation: str,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    texts = [
        embedding_text(row, representation)
        for row in candidates.itertuples(index=False)
    ]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return embeddings, texts


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix.astype(np.float32)


def threshold_neighbors(
    node_index: int,
    similarity: np.ndarray,
    threshold: float,
    maximum_neighbors: int | None,
) -> list[tuple[int, float]]:
    scores = similarity[node_index]
    indices = np.where(scores >= threshold)[0]
    indices = indices[indices != node_index]
    neighbors = sorted(
        ((int(index), float(scores[index])) for index in indices),
        key=lambda item: (-item[1], item[0]),
    )
    return neighbors if maximum_neighbors is None else neighbors[:maximum_neighbors]


def layered_bfs(
    similarity: np.ndarray,
    seed_indices: list[int],
    threshold: float,
    maximum_layers: int,
    maximum_neighbors: int | None,
    maximum_nodes: int | None,
) -> tuple[dict[int, int], dict[int, int | None], dict[int, float]]:
    layers = {index: 0 for index in seed_indices}
    parents: dict[int, int | None] = {index: None for index in seed_indices}
    discovery_scores = {index: 1.0 for index in seed_indices}
    queue = deque(sorted(seed_indices))

    while queue:
        current = queue.popleft()
        current_layer = layers[current]
        if current_layer >= maximum_layers:
            continue

        for neighbor, score in threshold_neighbors(
            current, similarity, threshold, maximum_neighbors
        ):
            if neighbor in layers:
                continue
            if maximum_nodes is not None and len(layers) >= maximum_nodes:
                return layers, parents, discovery_scores
            layers[neighbor] = current_layer + 1
            parents[neighbor] = current
            discovery_scores[neighbor] = score
            queue.append(neighbor)

    return layers, parents, discovery_scores


def score_graph_terms(
    candidates: pd.DataFrame,
    discovered_indices: set[int],
    strict_gt: pd.DataFrame,
) -> dict[str, Any]:
    discovered = candidates.iloc[sorted(discovered_indices)].copy()

    # Do not give the graph free credit for seeds inserted by this script.
    scored = discovered[~discovered["was_added_seed"].fillna(False).astype(bool)].copy()
    scored = scored[scored["match_type"].ne("manual_review_exact")]

    predicted = set(scored["normalized_candidate"])
    gold = set(strict_gt["normalized_gt"])
    tp_terms = predicted & gold
    fp_terms = predicted - gold
    fn_terms = gold - predicted

    tp = len(tp_terms)
    fp = len(fp_terms)
    fn = len(fn_terms)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "graph_node_count": len(discovered),
        "scored_term_count": len(predicted),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "manual_review_terms_excluded": int(
            discovered["match_type"].eq("manual_review_exact").sum()
        ),
        "added_seed_terms_excluded": int(
            discovered["was_added_seed"].fillna(False).astype(bool).sum()
        ),
        "negative_benchmark_hits": int(
            discovered["negative_benchmark_hit"].fillna(False).astype(bool).sum()
        ),
    }


def make_thresholds(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--threshold-step must be positive")
    if start > stop:
        raise ValueError("--threshold-start cannot exceed --threshold-stop")
    count = int(round((stop - start) / step))
    values = [round(start + index * step, 10) for index in range(count + 1)]
    if values[-1] < stop - 1e-9:
        values.append(stop)
    return [round(value, 4) for value in values if value <= stop + 1e-9]


def build_threshold_node_rows(
    candidates: pd.DataFrame,
    representation: str,
    threshold: float,
    layers: dict[int, int],
    parents: dict[int, int | None],
    discovery_scores: dict[int, float],
) -> pd.DataFrame:
    terms = candidates["normalized_candidate"].tolist()
    rows = []
    for index in sorted(layers, key=lambda i: (layers[i], terms[i])):
        parent_index = parents[index]
        rows.append({
            "representation": representation,
            "threshold": threshold,
            "matrix_row_id": index,
            "candidate": candidates.iloc[index]["candidate"],
            "normalized_candidate": terms[index],
            "layer": layers[index],
            "discovery_parent": terms[parent_index] if parent_index is not None else "",
            "discovery_similarity": discovery_scores[index],
            "match_type": candidates.iloc[index]["match_type"],
            "was_added_seed": candidates.iloc[index]["was_added_seed"],
            "negative_benchmark_hit": candidates.iloc[index]["negative_benchmark_hit"],
        })
    return pd.DataFrame(rows)


def plot_f1_curve(metrics: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for representation, group in metrics.groupby("representation", sort=False):
        group = group.sort_values("threshold")
        label = "Term only" if representation == "term" else "Term + source sentence"
        ax.plot(
            group["threshold"],
            group["f1"],
            marker="o",
            linewidth=2,
            label=label,
        )

        best = group.loc[group["f1"].idxmax()]
        ax.scatter([best["threshold"]], [best["f1"]], s=70, zorder=5)
        ax.annotate(
            f"best: {best['f1']:.3f} at {best['threshold']:.2f}",
            (best["threshold"], best["f1"]),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Cosine similarity threshold")
    ax.set_ylabel("F1 score")
    ax.set_title("Seed-layer graph F1 by MPNet representation and threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for representation, group in metrics.groupby("representation", sort=False):
        group = group.sort_values("threshold")
        label = "Term only" if representation == "term" else "Term + source sentence"
        axes[0].plot(group["threshold"], group["precision"], marker="o", linewidth=2, label=label)
        axes[1].plot(group["threshold"], group["recall"], marker="o", linewidth=2, label=label)

    axes[0].set_xlabel("Cosine similarity threshold")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision by threshold")
    axes[1].set_xlabel("Cosine similarity threshold")
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Recall by threshold")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare term-only and term+sentence MPNet seed graphs across cosine thresholds, "
            "and plot F1 versus threshold."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seeds", nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--threshold-start", type=float, default=0.40)
    parser.add_argument("--threshold-stop", type=float, default=0.85)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-neighbors-per-node",
        type=int,
        default=0,
        help="0 retains every neighbor meeting the threshold.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="0 disables the graph node cap. Keep it at 0 for unbiased threshold comparison.",
    )
    parser.add_argument(
        "--save-node-inventories",
        action="store_true",
        help="Save graph-node CSVs for every representation and threshold.",
    )
    args = parser.parse_args()

    if not 0 <= args.threshold_start <= 1 or not 0 <= args.threshold_stop <= 1:
        raise ValueError("Thresholds must be between 0 and 1")
    if args.max_layers < 0:
        raise ValueError("--max-layers cannot be negative")

    thresholds = make_thresholds(
        args.threshold_start,
        args.threshold_stop,
        args.threshold_step,
    )
    maximum_neighbors = (
        None if args.max_neighbors_per_node <= 0 else args.max_neighbors_per_node
    )
    maximum_nodes = None if args.max_nodes <= 0 else args.max_nodes

    output = args.output_dir
    (output / "matrices").mkdir(parents=True, exist_ok=True)
    (output / "embeddings").mkdir(parents=True, exist_ok=True)
    if args.save_node_inventories:
        (output / "nodes_by_threshold").mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args.input, args.seeds)
    strict_gt, review_gt, negatives = load_ground_truth(args.ground_truth)
    candidates = annotate_candidates(candidates, strict_gt, review_gt, negatives)

    normalized_seeds = [normalize_term(seed) for seed in args.seeds]
    term_to_index = {
        term: index
        for index, term in enumerate(candidates["normalized_candidate"])
    }
    seed_indices = [term_to_index[seed] for seed in normalized_seeds]

    model = SentenceTransformer(args.model)
    metrics_rows: list[dict[str, Any]] = []
    best_node_frames: dict[str, tuple[float, pd.DataFrame]] = {}

    for representation in ("term", "term_sentence"):
        print("\n" + "=" * 78)
        print(f"Encoding representation: {representation}")
        print("=" * 78)

        embeddings, texts = encode_representation(
            candidates,
            model,
            representation,
            args.batch_size,
        )
        similarity = cosine_matrix(embeddings)

        np.save(output / f"embeddings/{representation}__mpnet.npy", embeddings)
        pd.DataFrame({
            "matrix_row_id": candidates["matrix_row_id"],
            "candidate": candidates["candidate"],
            "embedding_input": texts,
        }).to_csv(output / f"embeddings/{representation}__inputs.csv", index=False)

        terms = candidates["normalized_candidate"].tolist()
        similarity_frame = pd.DataFrame(similarity, index=terms, columns=terms)
        similarity_frame.index.name = "term"
        similarity_frame.to_csv(
            output / f"matrices/{representation}__cosine_similarity_matrix.csv"
        )

        best_f1 = -1.0
        for threshold in thresholds:
            layers, parents, discovery_scores = layered_bfs(
                similarity,
                seed_indices,
                threshold,
                args.max_layers,
                maximum_neighbors,
                maximum_nodes,
            )
            scoring = score_graph_terms(candidates, set(layers), strict_gt)
            layer_counts = {
                layer: sum(1 for value in layers.values() if value == layer)
                for layer in range(args.max_layers + 1)
            }
            row = {
                "representation": representation,
                "threshold": threshold,
                "max_layers": args.max_layers,
                **{f"layer_{layer}_node_count": count for layer, count in layer_counts.items()},
                **scoring,
            }
            metrics_rows.append(row)

            node_frame = build_threshold_node_rows(
                candidates,
                representation,
                threshold,
                layers,
                parents,
                discovery_scores,
            )
            if args.save_node_inventories:
                threshold_name = f"{threshold:.2f}".replace(".", "p")
                node_frame.to_csv(
                    output
                    / "nodes_by_threshold"
                    / f"{representation}__threshold_{threshold_name}.csv",
                    index=False,
                )

            if scoring["f1"] > best_f1:
                best_f1 = scoring["f1"]
                best_node_frames[representation] = (threshold, node_frame)

            print(
                f"{representation:14s} threshold={threshold:.2f} "
                f"nodes={scoring['graph_node_count']:4d} "
                f"P={scoring['precision']:.4f} "
                f"R={scoring['recall']:.4f} "
                f"F1={scoring['f1']:.4f}"
            )

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["representation", "threshold"]
    ).reset_index(drop=True)
    metrics.to_csv(output / "threshold_sweep_metrics.csv", index=False)

    best_rows = (
        metrics.loc[metrics.groupby("representation")["f1"].idxmax()]
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    best_rows.to_csv(output / "best_threshold_by_representation.csv", index=False)

    overall_best = metrics.loc[metrics["f1"].idxmax()].to_frame().T
    overall_best.to_csv(output / "overall_best_threshold.csv", index=False)

    for representation, (threshold, node_frame) in best_node_frames.items():
        node_frame.to_csv(
            output / f"{representation}__best_threshold_nodes.csv",
            index=False,
        )

    plot_f1_curve(metrics, output / "f1_vs_threshold.png")
    plot_precision_recall(metrics, output / "precision_recall_vs_threshold.png")

    manifest = {
        "input": str(args.input),
        "ground_truth": str(args.ground_truth),
        "model": args.model,
        "representations": ["term", "term_sentence"],
        "seeds": normalized_seeds,
        "thresholds": thresholds,
        "max_layers": args.max_layers,
        "maximum_neighbors_per_node": maximum_neighbors,
        "maximum_nodes": maximum_nodes,
        "matrix_term_count_including_added_seeds": len(candidates),
        "strict_gt_count": len(strict_gt),
        "overall_best": json.loads(overall_best.to_json(orient="records"))[0],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    display = [
        "representation",
        "threshold",
        "graph_node_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
    ]
    print("\n" + "=" * 100)
    print("Best threshold for each representation")
    print("=" * 100)
    print(best_rows[display].to_string(index=False))
    print(f"\nMetrics: {output / 'threshold_sweep_metrics.csv'}")
    print(f"F1 plot: {output / 'f1_vs_threshold.png'}")
    print(f"Precision/recall plot: {output / 'precision_recall_vs_threshold.png'}")


if __name__ == "__main__":
    main()
