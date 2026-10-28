"""Build a three-layer seed-centered semantic graph from extracted terms.

The script:
1. Loads the current best-F1 candidate inventory.
2. Adds configured seed terms if they are absent.
3. Embeds terms with MPNet, with optional source-sentence context.
4. Calculates one square N x N cosine-similarity matrix.
5. Performs multi-source breadth-first expansion from the seeds.
6. Adds terms connected by cosine similarity >= the threshold for 3 layers.
7. Saves nodes, edges, matrices, GraphML, and a radial PNG visualization.

Layer assignment is shortest threshold-graph distance from either seed:
    Layer 0: seed terms
    Layer 1: direct neighbors of a seed
    Layer 2: new neighbors of Layer 1
    Layer 3: new neighbors of Layer 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path("/home/faujdar/onto_gen/ontology_experiment_framework")
DEFAULT_INPUT = PROJECT_DIR / "stage_outputs/05_seed_similarity_after_gliner/best_seed_filtered_candidates.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "stage_outputs/06_seed_layer_semantic_graph"
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_SEEDS = ("UAV", "unmanned aerial vehicle")
RANDOM_SEED = 42
LAYER_COLORS = {
    0: "#F4B183",
    1: "#5B9BD5",
    2: "#70AD47",
    3: "#8064A2",
}


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
        raise ValueError("Input candidate CSV must contain a 'candidate' column")

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
    additions = []
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


def build_embedding_text(row: Any, representation: str) -> str:
    term = safe_text(row.candidate)
    if representation == "term":
        return term
    if representation == "term_sentence":
        context = safe_text(row.source_sentence)
    elif representation == "term_context":
        context = safe_text(row.context_text) or safe_text(row.source_sentence)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    return f"Term: {term}. Context: {context}" if context else f"Term: {term}"


def encode_terms(
    candidates: pd.DataFrame,
    model_name: str,
    representation: str,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    model = SentenceTransformer(model_name)
    texts = [
        build_embedding_text(row, representation)
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


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    similarity = embeddings @ embeddings.T
    similarity = np.clip(similarity, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def strongest_neighbors(
    node_index: int,
    similarity: np.ndarray,
    threshold: float,
    maximum_neighbors: int | None,
) -> list[tuple[int, float]]:
    scores = similarity[node_index]
    indices = np.where(scores >= threshold)[0]
    indices = indices[indices != node_index]
    ordered = sorted(
        ((int(index), float(scores[index])) for index in indices),
        key=lambda item: (-item[1], item[0]),
    )
    if maximum_neighbors is not None:
        return ordered[:maximum_neighbors]
    return ordered


def layered_expansion(
    similarity: np.ndarray,
    seed_indices: list[int],
    threshold: float,
    maximum_layers: int,
    maximum_neighbors_per_node: int | None,
    maximum_nodes: int | None,
) -> tuple[dict[int, int], dict[int, int | None], dict[int, float], list[tuple[int, int, float]]]:
    """Perform deterministic multi-source BFS over threshold-qualified similarities."""
    layer = {index: 0 for index in seed_indices}
    parent: dict[int, int | None] = {index: None for index in seed_indices}
    discovery_score = {index: 1.0 for index in seed_indices}
    queue = deque(sorted(seed_indices))
    discovery_edges: list[tuple[int, int, float]] = []

    while queue:
        current = queue.popleft()
        current_layer = layer[current]
        if current_layer >= maximum_layers:
            continue

        for neighbour, score in strongest_neighbors(
            current,
            similarity,
            threshold,
            maximum_neighbors_per_node,
        ):
            if neighbour in layer:
                continue
            if maximum_nodes is not None and len(layer) >= maximum_nodes:
                return layer, parent, discovery_score, discovery_edges

            layer[neighbour] = current_layer + 1
            parent[neighbour] = current
            discovery_score[neighbour] = score
            discovery_edges.append((current, neighbour, score))
            queue.append(neighbour)

    return layer, parent, discovery_score, discovery_edges


def classify_edge(source_layer: int, target_layer: int, is_discovery: bool) -> str:
    if is_discovery:
        return "discovery"
    if source_layer == target_layer:
        return "same_layer"
    if abs(source_layer - target_layer) == 1:
        return "adjacent_layer"
    return "cross_layer"


def build_graph_tables(
    candidates: pd.DataFrame,
    similarity: np.ndarray,
    layers: dict[int, int],
    parents: dict[int, int | None],
    discovery_scores: dict[int, float],
    discovery_edges: list[tuple[int, int, float]],
    seed_indices: list[int],
    threshold: float,
    include_all_edges: bool,
    maximum_plot_edges: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    terms = candidates["normalized_candidate"].tolist()
    seed_set = set(seed_indices)
    discovered = sorted(layers)
    discovered_set = set(discovered)
    discovery_pairs = {
        frozenset((source, target)) for source, target, _ in discovery_edges
    }

    seed_similarity = similarity[:, seed_indices]
    maximum_seed_similarity = seed_similarity.max(axis=1)
    maximum_seed_position = seed_similarity.argmax(axis=1)

    node_rows = []
    optional_columns = [
        "source_sentence",
        "context_text",
        "match_type",
        "gt_category",
        "negative_benchmark_hit",
        "was_added_seed",
    ]
    for index in discovered:
        parent_index = parents[index]
        row = {
            "matrix_row_id": index,
            "term": terms[index],
            "display_term": candidates.iloc[index]["candidate"],
            "layer": layers[index],
            "is_seed": index in seed_set,
            "discovery_parent": terms[parent_index] if parent_index is not None else "",
            "discovery_similarity": discovery_scores[index],
            "maximum_seed_similarity": float(maximum_seed_similarity[index]),
            "closest_seed": terms[seed_indices[int(maximum_seed_position[index])]],
        }
        for column in optional_columns:
            if column in candidates.columns:
                row[column] = candidates.iloc[index][column]
        node_rows.append(row)

    all_edges: list[tuple[int, int, float]] = []
    for left_position, source in enumerate(discovered):
        for target in discovered[left_position + 1 :]:
            score = float(similarity[source, target])
            if score >= threshold:
                all_edges.append((source, target, score))

    if not include_all_edges:
        all_edges = [
            edge for edge in all_edges
            if frozenset((edge[0], edge[1])) in discovery_pairs
        ]

    if maximum_plot_edges is not None and len(all_edges) > maximum_plot_edges:
        discovery = [
            edge for edge in all_edges
            if frozenset((edge[0], edge[1])) in discovery_pairs
        ]
        additional = [
            edge for edge in all_edges
            if frozenset((edge[0], edge[1])) not in discovery_pairs
        ]
        additional.sort(key=lambda item: item[2], reverse=True)
        room = max(0, maximum_plot_edges - len(discovery))
        all_edges = discovery + additional[:room]

    edge_rows = []
    for source, target, score in sorted(all_edges, key=lambda item: (-item[2], item[0], item[1])):
        is_discovery = frozenset((source, target)) in discovery_pairs
        edge_rows.append({
            "source": terms[source],
            "target": terms[target],
            "cosine_similarity": score,
            "source_layer": layers[source],
            "target_layer": layers[target],
            "edge_type": classify_edge(layers[source], layers[target], is_discovery),
            "is_discovery_edge": is_discovery,
        })

    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def radial_positions(nodes: pd.DataFrame) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    seeds = nodes[nodes["layer"].eq(0)].sort_values("term")

    if len(seeds) == 1:
        positions[str(seeds.iloc[0]["term"])] = (0.0, 0.0)
    else:
        seed_radius = 0.55
        for position, row in enumerate(seeds.itertuples(index=False)):
            angle = 2 * math.pi * position / len(seeds)
            positions[str(row.term)] = (
                seed_radius * math.cos(angle),
                seed_radius * math.sin(angle),
            )

    for layer_number in sorted(set(nodes["layer"]) - {0}):
        layer_nodes = nodes[nodes["layer"].eq(layer_number)].sort_values("term")
        radius = 3.0 * layer_number
        count = len(layer_nodes)
        for position, row in enumerate(layer_nodes.itertuples(index=False)):
            angle = 2 * math.pi * position / max(count, 1)
            positions[str(row.term)] = (
                radius * math.cos(angle),
                radius * math.sin(angle),
            )

    return positions


def wrap_label(text: str, width: int = 17) -> str:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def draw_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    threshold: float,
    maximum_layers: int,
    output_png: Path,
    show_edge_labels: bool,
    label_mode: str,
) -> None:
    positions = radial_positions(nodes)
    node_count = len(nodes)
    figure_width = max(16, min(32, 13 + math.sqrt(max(node_count, 1))))
    fig, ax = plt.subplots(figsize=(figure_width, figure_width * 0.75))

    for edge in edges.itertuples(index=False):
        source_position = positions[edge.source]
        target_position = positions[edge.target]
        relative = max(0.0, (edge.cosine_similarity - threshold) / max(1.0 - threshold, 1e-9))
        if edge.is_discovery_edge:
            color = "#355C7D"
            alpha = 0.70
            line_width = 1.1 + 3.5 * relative
            line_style = "-"
        else:
            color = "#A5A5A5"
            alpha = 0.24
            line_width = 0.4 + 1.6 * relative
            line_style = "-"
        ax.plot(
            [source_position[0], target_position[0]],
            [source_position[1], target_position[1]],
            color=color,
            alpha=alpha,
            linewidth=line_width,
            linestyle=line_style,
            zorder=1,
        )
        if show_edge_labels:
            midpoint = (
                (source_position[0] + target_position[0]) / 2,
                (source_position[1] + target_position[1]) / 2,
            )
            ax.text(
                midpoint[0],
                midpoint[1],
                f"{edge.cosine_similarity:.2f}",
                fontsize=5.5,
                color="#666666",
                zorder=2,
            )

    degrees = defaultdict(int)
    for edge in edges.itertuples(index=False):
        degrees[edge.source] += 1
        degrees[edge.target] += 1

    label_terms: set[str]
    if label_mode == "all":
        label_terms = set(nodes["term"])
    elif label_mode == "seeds":
        label_terms = set(nodes.loc[nodes["is_seed"], "term"])
    elif label_mode == "important":
        seeds = set(nodes.loc[nodes["is_seed"], "term"])
        strongest = {
            term for term, _ in sorted(degrees.items(), key=lambda item: (-item[1], item[0]))[:25]
        }
        label_terms = seeds | strongest
    else:
        label_terms = set()

    for row in nodes.itertuples(index=False):
        x, y = positions[row.term]
        size = 1900 if row.is_seed else 500 + min(degrees[row.term], 15) * 30
        ax.scatter(
            [x],
            [y],
            s=size,
            color=LAYER_COLORS.get(int(row.layer), "#BFBFBF"),
            edgecolor="#17365D",
            linewidth=2.1 if row.is_seed else 0.8,
            alpha=0.96,
            zorder=3,
        )
        if row.term in label_terms:
            ax.text(
                x,
                y,
                wrap_label(row.display_term),
                fontsize=8.5 if row.is_seed else 6.5,
                fontweight="bold" if row.is_seed else "normal",
                ha="center",
                va="center",
                zorder=4,
            )

    layer_counts = nodes.groupby("layer").size().to_dict()
    count_text = ", ".join(f"L{layer}: {count}" for layer, count in sorted(layer_counts.items()))
    ax.set_title(
        "Seed-centered semantic expansion graph\n"
        f"cosine similarity >= {threshold:.2f}; max layer = {maximum_layers}; "
        f"nodes = {len(nodes)}; edges = {len(edges)}\n{count_text}",
        fontsize=14,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def export_graphml(nodes: pd.DataFrame, edges: pd.DataFrame, output_file: Path) -> None:
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        attributes = {
            "display_term": str(row.display_term),
            "layer": int(row.layer),
            "is_seed": bool(row.is_seed),
            "discovery_parent": str(row.discovery_parent),
            "discovery_similarity": float(row.discovery_similarity),
            "maximum_seed_similarity": float(row.maximum_seed_similarity),
            "closest_seed": str(row.closest_seed),
        }
        if hasattr(row, "gt_category"):
            attributes["gt_category"] = str(row.gt_category)
        if hasattr(row, "match_type"):
            attributes["match_type"] = str(row.match_type)
        graph.add_node(str(row.term), **attributes)

    for row in edges.itertuples(index=False):
        graph.add_edge(
            str(row.source),
            str(row.target),
            cosine_similarity=float(row.cosine_similarity),
            edge_type=str(row.edge_type),
            is_discovery_edge=bool(row.is_discovery_edge),
        )
    nx.write_graphml(graph, output_file)


def add_layer_summary(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer_number, group in nodes.groupby("layer", sort=True):
        incident = edges[
            edges["source_layer"].eq(layer_number)
            | edges["target_layer"].eq(layer_number)
        ] if not edges.empty else pd.DataFrame()
        row = {
            "layer": int(layer_number),
            "node_count": len(group),
            "seed_count": int(group["is_seed"].sum()),
            "incident_edge_count": len(incident),
            "mean_discovery_similarity": float(group["discovery_similarity"].mean()),
            "minimum_discovery_similarity": float(group["discovery_similarity"].min()),
            "maximum_discovery_similarity": float(group["discovery_similarity"].max()),
        }
        if "match_type" in group.columns:
            row["strict_gt_count"] = int(group["match_type"].eq("strict_exact").sum())
            row["unmatched_count"] = int(group["match_type"].eq("unmatched").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a three-layer semantic graph from an N x N cosine-similarity matrix."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seeds", nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--representation",
        choices=("term", "term_sentence", "term_context"),
        default="term",
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-neighbors-per-node",
        type=int,
        default=0,
        help="0 keeps all threshold-qualified neighbors.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="0 disables the node cap.",
    )
    parser.add_argument(
        "--max-plot-edges",
        type=int,
        default=2500,
        help="Maximum edges rendered and exported in graph tables; 0 keeps all.",
    )
    parser.add_argument(
        "--discovery-edges-only",
        action="store_true",
        help="Render only BFS discovery edges instead of every threshold edge among discovered nodes.",
    )
    parser.add_argument("--show-edge-labels", action="store_true")
    parser.add_argument(
        "--label-mode",
        choices=("all", "important", "seeds", "none"),
        default="important",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.max_layers < 0:
        raise ValueError("--max-layers cannot be negative")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    max_neighbors = None if args.max_neighbors_per_node <= 0 else args.max_neighbors_per_node
    max_nodes = None if args.max_nodes <= 0 else args.max_nodes
    max_plot_edges = None if args.max_plot_edges <= 0 else args.max_plot_edges

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args.input, args.seeds)
    normalized_seeds = [normalize_term(seed) for seed in args.seeds]
    term_to_index = {
        term: int(index)
        for index, term in enumerate(candidates["normalized_candidate"])
    }
    missing = [seed for seed in normalized_seeds if seed not in term_to_index]
    if missing:
        raise ValueError(f"Seeds could not be added or found: {missing}")
    seed_indices = [term_to_index[seed] for seed in normalized_seeds]

    embeddings, embedding_inputs = encode_terms(
        candidates,
        args.model,
        args.representation,
        args.batch_size,
    )
    similarity = cosine_similarity_matrix(embeddings)

    terms = candidates["normalized_candidate"].tolist()
    matrix_frame = pd.DataFrame(similarity, index=terms, columns=terms)
    matrix_frame.index.name = "term"
    matrix_file = output / f"{args.representation}__cosine_similarity_matrix.csv"
    matrix_frame.to_csv(matrix_file, encoding="utf-8")
    np.save(output / f"{args.representation}__term_embeddings.npy", embeddings)
    candidates.assign(embedding_input=embedding_inputs).to_csv(
        output / f"{args.representation}__embedding_inputs.csv",
        index=False,
        encoding="utf-8",
    )

    layers, parents, discovery_scores, discovery_edges = layered_expansion(
        similarity,
        seed_indices,
        args.threshold,
        args.max_layers,
        max_neighbors,
        max_nodes,
    )

    nodes, edges = build_graph_tables(
        candidates,
        similarity,
        layers,
        parents,
        discovery_scores,
        discovery_edges,
        seed_indices,
        args.threshold,
        include_all_edges=not args.discovery_edges_only,
        maximum_plot_edges=max_plot_edges,
    )

    nodes_file = output / f"{args.representation}__graph_nodes.csv"
    edges_file = output / f"{args.representation}__graph_edges.csv"
    summary_file = output / f"{args.representation}__layer_summary.csv"
    png_file = output / f"{args.representation}__seed_expansion_graph.png"
    graphml_file = output / f"{args.representation}__seed_expansion_graph.graphml"

    nodes.to_csv(nodes_file, index=False, encoding="utf-8")
    edges.to_csv(edges_file, index=False, encoding="utf-8")
    layer_summary = add_layer_summary(nodes, edges)
    layer_summary.to_csv(summary_file, index=False, encoding="utf-8")
    draw_graph(
        nodes,
        edges,
        args.threshold,
        args.max_layers,
        png_file,
        args.show_edge_labels,
        args.label_mode,
    )
    export_graphml(nodes, edges, graphml_file)

    manifest = {
        "input_file": str(args.input),
        "model": args.model,
        "representation": args.representation,
        "seeds": normalized_seeds,
        "threshold": args.threshold,
        "maximum_layers": args.max_layers,
        "maximum_neighbors_per_node": max_neighbors,
        "maximum_nodes": max_nodes,
        "include_all_threshold_edges_among_discovered_nodes": not args.discovery_edges_only,
        "maximum_exported_plot_edges": max_plot_edges,
        "matrix_term_count": len(candidates),
        "similarity_matrix_shape": list(similarity.shape),
        "discovered_node_count": len(nodes),
        "exported_edge_count": len(edges),
        "layer_counts": {
            str(int(layer)): int(count)
            for layer, count in nodes.groupby("layer").size().items()
        },
        "files": {
            "similarity_matrix": str(matrix_file),
            "nodes": str(nodes_file),
            "edges": str(edges_file),
            "layer_summary": str(summary_file),
            "png": str(png_file),
            "graphml": str(graphml_file),
        },
    }
    (output / f"{args.representation}__run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 92)
    print("Seed-centered semantic graph completed")
    print("=" * 92)
    print(f"Input terms including added seeds: {len(candidates)}")
    print(f"Similarity matrix: {similarity.shape[0]} x {similarity.shape[1]}")
    print(f"Seeds: {', '.join(normalized_seeds)}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Maximum layers: {args.max_layers}")
    print(f"Discovered nodes: {len(nodes)}")
    print(f"Exported edges: {len(edges)}")
    print("\nLayer summary:")
    print(layer_summary.to_string(index=False))
    print(f"\nSimilarity matrix: {matrix_file}")
    print(f"Node table: {nodes_file}")
    print(f"Edge table: {edges_file}")
    print(f"Visualization: {png_file}")
    print(f"GraphML: {graphml_file}")


if __name__ == "__main__":
    main()
