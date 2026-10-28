from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from functools import lru_cache
from collections import defaultdict, deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import Birch, KMeans
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

from olbench.utils import text_sentences


PROJECT_DIR = Path("ontology_experiment_framework")
DEFAULT_CONFIG_FILE = PROJECT_DIR / "configs/seed_guided_contextual_clustering.json"
GROUND_TRUTH_FILE = PROJECT_DIR / "data/Ground_Truth.xlsx"
GROUND_TRUTH_SHEET = "S1_GT_TERMS"
MANUAL_REVIEW_STATUS = "needs_manual_review"

PRIORITY_WEIGHTS = {
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}

SPACY_MODEL_NAME = "en_core_web_sm"
MODIFIER_POS_TAGS = {"ADJ", "ADV", "VERB", "AUX", "PART"}
SHORT_PATTERN_NOISE = re.compile(r"^(?:[a-z]{1,3}(?: [a-z]{1,3})+|[a-z]{1,3} s)$")
UAV_SUFFIX_EXEMPTIONS = {
    "uav",
    "uas",
    "rpa",
    "rpv",
    "rvt",
    "mav",
    "muav",
    "tuav",
    "opv",
    "ucav",
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


def normalize_term_for_evaluation(value: object) -> str:
    text = normalize_term(value)
    if not text:
        return ""

    normalized_tokens: list[str] = []
    for token in text.split():
        singular = token
        if len(token) > 4 and token.endswith("ies"):
            singular = token[:-3] + "y"
        elif len(token) > 3 and token.endswith(("ses", "xes", "zes", "ches", "shes")):
            singular = token[:-2]
        elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            singular = token[:-1]
        normalized_tokens.append(singular)

    return " ".join(normalized_tokens)


def safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


@lru_cache(maxsize=1)
def load_spacy_nlp():
    try:
        return spacy.load(SPACY_MODEL_NAME, disable=["ner"])
    except OSError:
        return None


def has_uav_suffix_exemption(normalized: str) -> bool:
    tokens = normalized.split()
    if not tokens:
        return False
    if tokens[-1] == "uav" or normalized.endswith(" uav"):
        return True
    return any(token in UAV_SUFFIX_EXEMPTIONS for token in tokens)


def looks_like_short_pattern_noise(normalized: str) -> bool:
    if not normalized or has_uav_suffix_exemption(normalized):
        return False
    if SHORT_PATTERN_NOISE.fullmatch(normalized):
        return True
    tokens = normalized.split()
    if len(tokens) > 1 and all(len(token) <= 3 for token in tokens):
        return True
    return False


def looks_like_lexical_noise(term: object) -> bool:
    normalized = normalize_term(term)
    if not normalized:
        return False

    if looks_like_short_pattern_noise(normalized):
        return True

    nlp = load_spacy_nlp()
    if nlp is None:
        tokens = normalized.split()
        return bool(tokens) and tokens[-1].endswith(("ing", "ed", "ly", "ive", "ous", "able", "ible", "ary", "ic", "ical", "less", "ful"))

    doc = nlp(normalized)
    if not doc:
        return False

    tokens = [token for token in doc if not token.is_space]
    if not tokens:
        return False
    if has_uav_suffix_exemption(normalized):
        return False

    content_tokens = [token for token in tokens if not token.is_punct]
    if not content_tokens:
        return False

    pos_tags = [token.pos_ for token in content_tokens]
    has_noun = any(tag in {"NOUN", "PROPN"} for tag in pos_tags)
    has_modifier = any(tag in MODIFIER_POS_TAGS for tag in pos_tags)
    trailing_modifier = content_tokens[-1].pos_ in MODIFIER_POS_TAGS

    if len(content_tokens) == 1:
        token = content_tokens[0]
        return token.pos_ in MODIFIER_POS_TAGS or (token.pos_ == "X" and len(token.text) <= 3)

    if trailing_modifier and (not has_noun or len(content_tokens) <= 3):
        return True

    if has_modifier and not has_noun and len(content_tokens) <= 4:
        return True

    return False


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def validate_inputs(config: dict) -> None:
    input_file = Path(config["input_file"])
    if not input_file.is_file():
        raise FileNotFoundError(f"Missing required file: {input_file}")

    if not GROUND_TRUTH_FILE.is_file():
        raise FileNotFoundError(f"Missing required file: {GROUND_TRUTH_FILE}")


def load_ground_truth() -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = pd.read_excel(
        GROUND_TRUTH_FILE,
        sheet_name=GROUND_TRUTH_SHEET,
        engine="openpyxl",
    )

    required_columns = {"term_canonical", "keep_for_tasks", "grounded_status_v2", "priority"}
    missing = required_columns - set(gt.columns)
    if missing:
        raise ValueError(f"{GROUND_TRUTH_FILE} is missing columns: {sorted(missing)}")

    gt = gt[
        gt["keep_for_tasks"].astype(str).str.contains("term_extraction", case=False, na=False)
    ].copy()
    gt["normalized_gt"] = gt["term_canonical"].map(normalize_term_for_evaluation)
    gt = gt[gt["normalized_gt"].ne("")].copy()

    status = gt["grounded_status_v2"].fillna("").astype(str).str.casefold()
    strict_gt = gt[status.ne(MANUAL_REVIEW_STATUS)].copy()
    review_gt = gt[status.eq(MANUAL_REVIEW_STATUS)].copy()

    return strict_gt, review_gt


def compute_gt_metrics(final_candidates: pd.DataFrame) -> dict:
    strict_gt, review_gt = load_ground_truth()

    predicted_terms = set(final_candidates["normalized_candidate"].map(normalize_term_for_evaluation))
    predicted_terms.discard("")

    gold_terms = set(strict_gt["normalized_gt"])

    true_positive_terms = predicted_terms & gold_terms
    false_positive_terms = predicted_terms - gold_terms
    false_negative_terms = gold_terms - predicted_terms

    true_positives = len(true_positive_terms)
    false_positives = len(false_positive_terms)
    false_negatives = len(false_negative_terms)

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    gt_weights = (
        strict_gt["priority"]
        .fillna("low")
        .astype(str)
        .str.casefold()
        .map(PRIORITY_WEIGHTS)
        .fillna(1.0)
    )
    matched_gt_mask = strict_gt["normalized_gt"].isin(predicted_terms)
    total_weight = float(gt_weights.sum())
    matched_weight = float(gt_weights[matched_gt_mask].sum())

    return {
        "predicted_terms": len(predicted_terms),
        "gold_rows": int(len(strict_gt) + len(review_gt)),
        "gold_terms": int(len(gold_terms)),
        "strict_rows": int(len(strict_gt)),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "priority_weighted_recall": matched_weight / total_weight if total_weight else 0.0,
        "matched_gt_terms": sorted(true_positive_terms),
        "false_positive_terms": sorted(false_positive_terms),
        "false_negative_terms": sorted(false_negative_terms),
    }


def select_context_snippet(
    term: str,
    source_sentence: str,
    context_text: str,
    context_sentence_limit: int,
) -> str:
    source_sentence = safe_text(source_sentence)
    if source_sentence:
        return source_sentence

    context_text = safe_text(context_text)
    if not context_text:
        return ""

    sentences = [sentence.strip() for sentence in text_sentences(context_text)]
    if not sentences:
        return context_text

    term_cf = term.casefold()
    match_index = next(
        (
            index
            for index, sentence in enumerate(sentences)
            if term_cf in sentence.casefold()
        ),
        None,
    )
    if match_index is None:
        return " ".join(sentences[:context_sentence_limit])

    end_index = min(len(sentences), match_index + context_sentence_limit)
    return " ".join(sentences[match_index:end_index])


def load_candidates(
    input_file: Path,
    seed_terms: list[str],
    context_sentence_limit: int,
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
    candidates["candidate"] = candidates["candidate"].map(safe_text)
    candidates["normalized_candidate"] = candidates["normalized_candidate"].map(normalize_term)
    candidates["source_sentence"] = candidates["source_sentence"].map(safe_text)
    candidates["context_text"] = candidates["context_text"].map(safe_text)

    sort_columns = [column for column in ("global_rank", "score") if column in candidates.columns]
    if sort_columns:
        ascending = [True if column == "global_rank" else False for column in sort_columns]
        candidates = candidates.sort_values(sort_columns, ascending=ascending)

    seed_norms = {normalize_term(term) for term in seed_terms}
    seed_rows = pd.DataFrame(
        {
            "candidate": seed_terms,
            "normalized_candidate": [normalize_term(term) for term in seed_terms],
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

    pool["selected_context_snippet"] = pool.apply(
        lambda row: select_context_snippet(
            safe_text(row.candidate),
            safe_text(row.source_sentence),
            safe_text(row.context_text),
            context_sentence_limit,
        ),
        axis=1,
    )
    pool["representation_text"] = pool.apply(
        lambda row: (
            f"{safe_text(row.candidate)} [CONTEXT] {safe_text(row.selected_context_snippet)}"
            if safe_text(row.selected_context_snippet)
            else safe_text(row.candidate)
        ),
        axis=1,
    )
    return pool


def encode_representations(pool: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    embeddings = model.encode(
        pool["representation_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    return np.asarray(embeddings)


def score_seed_similarity(
    pool: pd.DataFrame,
    embeddings: np.ndarray,
    seed_terms: list[str],
    threshold: float,
) -> pd.DataFrame:
    seed_norms = [normalize_term(term) for term in seed_terms]
    seed_indices = []
    for seed_norm in seed_norms:
        match = pool.index[pool["normalized_candidate"].eq(seed_norm)]
        if match.empty:
            raise ValueError(f"Missing seed concept in candidate pool: {seed_norm}")
        seed_indices.append(int(match[0]))

    seed_embeddings = embeddings[seed_indices]
    similarity_matrix = embeddings @ seed_embeddings.T

    result = pool.copy()
    for seed_index, seed_term in enumerate(seed_terms):
        result[f"seed_similarity__{normalize_term(seed_term).replace(' ', '_')}"] = similarity_matrix[:, seed_index]

    result["max_seed_similarity"] = similarity_matrix.max(axis=1)
    result["closest_seed"] = [seed_terms[index] for index in similarity_matrix.argmax(axis=1)]
    result["seed_related"] = result["max_seed_similarity"] > threshold
    return result


def build_similarity_graph(
    embeddings: np.ndarray,
    neighbor_count: int,
    similarity_threshold: float,
) -> nx.Graph:
    graph = nx.Graph()
    total = embeddings.shape[0]
    for index in range(total):
        graph.add_node(index)

    if total <= 1:
        return graph

    neighbor_count = max(1, min(neighbor_count, total - 1))
    neighbors = NearestNeighbors(metric="cosine", n_neighbors=neighbor_count + 1)
    distances, indices = neighbors.fit(embeddings).kneighbors(embeddings)

    for row_index in range(total):
        for distance, neighbor_index in zip(distances[row_index][1:], indices[row_index][1:]):
            similarity = 1.0 - float(distance)
            if similarity >= similarity_threshold:
                graph.add_edge(row_index, int(neighbor_index), weight=similarity)

    return graph


def expand_reference_set(
    graph: nx.Graph,
    start_indices: set[int],
    max_depth: int,
) -> dict[int, int]:
    reference_depth = {index: 0 for index in sorted(start_indices)}
    frontier = deque(sorted(start_indices))

    while frontier:
        current = frontier.popleft()
        current_depth = reference_depth[current]
        if current_depth >= max_depth:
            continue

        for neighbor in graph.neighbors(current):
            if neighbor in reference_depth:
                continue
            reference_depth[neighbor] = current_depth + 1
            frontier.append(neighbor)

    return reference_depth


def cluster_pool(pool: pd.DataFrame, embeddings: np.ndarray, cluster_methods: list[dict]) -> pd.DataFrame:
    result = pool.copy()

    for method in cluster_methods:
        method_name = method["name"]
        cluster_count = max(1, min(int(method.get("n_clusters", 8)), len(pool)))

        if method_name == "kmeans":
            clusterer = KMeans(
                n_clusters=cluster_count,
                random_state=int(method.get("random_state", 42)),
                n_init=10,
            )
            labels = clusterer.fit_predict(embeddings)
        elif method_name == "birch":
            clusterer = Birch(n_clusters=cluster_count)
            labels = clusterer.fit_predict(embeddings)
        elif method_name == "hdbscan_umap":
            reducer = UMAP(
                n_neighbors=int(method.get("umap_n_neighbors", 15)),
                min_dist=float(method.get("umap_min_dist", 0.0)),
                n_components=int(method.get("umap_n_components", 5)),
                metric="cosine",
                random_state=int(method.get("random_state", 42)),
            )
            reduced = reducer.fit_transform(embeddings)
            clusterer = HDBSCAN(
                min_cluster_size=int(method.get("hdbscan_min_cluster_size", 8)),
                min_samples=int(method.get("hdbscan_min_samples", 1)),
            )
            labels = clusterer.fit_predict(reduced)
            result["umap_component_1"] = reduced[:, 0]
            if reduced.shape[1] > 1:
                result["umap_component_2"] = reduced[:, 1]
            result["umap_cluster_space"] = "umap"
        else:
            raise ValueError(f"Unknown clustering method: {method_name}")

        result[f"cluster__{method_name}"] = labels

    return result


def build_concept_graph(scored: pd.DataFrame, reference_depth_map: dict[int, int]) -> nx.Graph:
    concept_indices = sorted(reference_depth_map)
    concept_set = set(concept_indices)
    if not concept_indices:
        return nx.Graph()

    graph = nx.Graph()
    for index in concept_indices:
        graph.add_node(index)

    concept_embeddings = scored.loc[concept_indices, "embedding_vector"].tolist()
    concept_embeddings = np.asarray(concept_embeddings)
    neighbors = min(10, max(1, len(concept_indices) - 1))
    if len(concept_indices) > 1:
        knn = NearestNeighbors(metric="cosine", n_neighbors=neighbors + 1)
        distances, indices = knn.fit(concept_embeddings).kneighbors(concept_embeddings)
        for row_index, source_index in enumerate(concept_indices):
            for distance, neighbor_pos in zip(distances[row_index][1:], indices[row_index][1:]):
                target_index = concept_indices[int(neighbor_pos)]
                similarity = 1.0 - float(distance)
                if similarity >= 0.7 and target_index in concept_set:
                    graph.add_edge(source_index, target_index, weight=similarity)

    return graph


def plot_concept_graph(
    graph: nx.Graph,
    scored: pd.DataFrame,
    final_candidates: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        empty = pd.DataFrame(columns=["index", "candidate", "normalized_candidate", "survives", "reference_depth"])
        empty.to_csv(output_path.with_suffix(".nodes.csv"), index=False, encoding="utf-8")
        return empty

    node_ids = list(graph.nodes())
    node_rows = scored.loc[node_ids, ["candidate", "normalized_candidate", "reference_depth"]].copy()
    survived_set = set(final_candidates.index.tolist())
    node_rows["survives"] = node_rows.index.isin(survived_set)

    node_positions = None
    if len(node_ids) > 2:
        reducer = UMAP(
            n_neighbors=min(10, len(node_ids) - 1),
            min_dist=0.0,
            n_components=2,
            metric="cosine",
            random_state=42,
        )
        coordinates = reducer.fit_transform(np.asarray(scored.loc[node_ids, "embedding_vector"].tolist()))
        node_positions = {node_id: coordinates[position] for position, node_id in enumerate(node_ids)}
    else:
        node_positions = nx.spring_layout(graph, seed=42, weight="weight")

    plt.figure(figsize=(18, 13), dpi=200)
    alive_nodes = [node for node in node_ids if node in survived_set]
    dead_nodes = [node for node in node_ids if node not in survived_set]

    nx.draw_networkx_edges(
        graph,
        pos=node_positions,
        edge_color="#666666",
        width=[1.0 + 2.0 * float(data.get("weight", 0.0)) for _, _, data in graph.edges(data=True)],
        alpha=0.55,
    )
    nx.draw_networkx_nodes(
        graph,
        pos=node_positions,
        nodelist=dead_nodes,
        node_color="#B0B0B0",
        node_size=220,
        linewidths=0.8,
        edgecolors="#4A4A4A",
    )
    nx.draw_networkx_nodes(
        graph,
        pos=node_positions,
        nodelist=alive_nodes,
        node_color="#1B9E77",
        node_size=280,
        linewidths=1.1,
        edgecolors="#0B4F41",
    )

    labels = {node_id: scored.at[node_id, "candidate"] for node_id in node_ids}
    nx.draw_networkx_labels(
        graph,
        pos=node_positions,
        labels=labels,
        font_size=7,
        font_color="#111111",
    )
    edge_labels = {
        (source, target): f"{float(data.get('weight', 0.0)):.2f}"
        for source, target, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        graph,
        pos=node_positions,
        edge_labels=edge_labels,
        font_size=6,
        label_pos=0.55,
        rotate=False,
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    node_rows.insert(0, "index", node_rows.index)
    node_rows.to_csv(output_path.with_suffix(".nodes.csv"), index=False, encoding="utf-8")

    edge_rows = [
        {
            "source": scored.at[source, "candidate"],
            "target": scored.at[target, "candidate"],
            "source_index": source,
            "target_index": target,
            "similarity": float(data.get("weight", 0.0)),
        }
        for source, target, data in graph.edges(data=True)
    ]
    pd.DataFrame(edge_rows).to_csv(output_path.with_suffix(".edges.csv"), index=False, encoding="utf-8")
    return node_rows


def collect_seed_clusters(
    scored: pd.DataFrame,
    cluster_methods: list[dict],
    seed_terms: list[str],
    reference_depth_map: dict[int, int],
    stability_jaccard_threshold: float,
    reference_coverage_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[int]]]:
    seed_norms = {normalize_term(term) for term in seed_terms}
    reference_set = set(reference_depth_map)

    cluster_records = []
    cluster_members_rows = []
    cluster_membership_map: dict[str, set[int]] = {}

    for method in cluster_methods:
        method_name = method["name"]
        cluster_column = f"cluster__{method_name}"
        labels = sorted(scored[cluster_column].dropna().unique().tolist())
        labels = [label for label in labels if int(label) >= 0]

        for label in labels:
            member_mask = scored[cluster_column].eq(label)
            member_indices = set(scored.index[member_mask].tolist())
            cluster_key = f"{method_name}__{int(label)}"
            cluster_membership_map[cluster_key] = member_indices

            seed_members = scored.loc[member_mask & scored["normalized_candidate"].isin(seed_norms)]
            reference_overlap = member_indices & reference_set

            cluster_records.append(
                {
                    "method": method_name,
                    "cluster_label": int(label),
                    "cluster_key": cluster_key,
                    "cluster_size": len(member_indices),
                    "seed_count": int(len(seed_members)),
                    "seed_terms": " | ".join(seed_members["candidate"].tolist()),
                    "reference_overlap_count": int(len(reference_overlap)),
                    "reference_coverage": (
                        len(reference_overlap) / len(reference_set)
                        if reference_set
                        else 0.0
                    ),
                }
            )

            for member_index in member_indices:
                cluster_members_rows.append(
                    {
                        "method": method_name,
                        "cluster_label": int(label),
                        "cluster_key": cluster_key,
                        "candidate": scored.at[member_index, "candidate"],
                        "normalized_candidate": scored.at[member_index, "normalized_candidate"],
                        "is_seed": scored.at[member_index, "normalized_candidate"] in seed_norms,
                        "in_reference_set": member_index in reference_set,
                        "reference_depth": reference_depth_map.get(member_index, -1),
                    }
                )

    clusters_df = pd.DataFrame(cluster_records)
    members_df = pd.DataFrame(cluster_members_rows)
    if clusters_df.empty:
                raise RuntimeError("No clusters were discovered.")

    partner_map: dict[str, set[str]] = defaultdict(set)
    record_rows = clusters_df.to_dict("records")
    for left_index, left_record in enumerate(record_rows):
        left_members = cluster_membership_map[left_record["cluster_key"]]
        for right_index in range(left_index + 1, len(record_rows)):
            right_record = record_rows[right_index]
            if left_record["method"] == right_record["method"]:
                continue

            right_members = cluster_membership_map[right_record["cluster_key"]]
            union = len(left_members | right_members)
            jaccard = (len(left_members & right_members) / union) if union else 0.0
            if jaccard >= stability_jaccard_threshold:
                partner_map[left_record["cluster_key"]].add(right_record["cluster_key"])
                partner_map[right_record["cluster_key"]].add(left_record["cluster_key"])

    clusters_df["stable_partner_keys"] = clusters_df["cluster_key"].map(
        lambda key: " | ".join(sorted(partner_map.get(key, set())))
    )
    clusters_df["stable_partner_count"] = clusters_df["stable_partner_keys"].map(
        lambda value: 0 if not value else len(value.split(" | "))
    )
    clusters_df["retained_by_coverage"] = clusters_df["reference_coverage"] >= reference_coverage_threshold
    clusters_df["retained_by_stability"] = clusters_df["stable_partner_count"] >= 1
    clusters_df["retained"] = clusters_df["retained_by_coverage"] | clusters_df["retained_by_stability"]
    clusters_df["retained_reason"] = clusters_df.apply(
        lambda row: "; ".join(
            reason
            for reason, flag in (("coverage", row["retained_by_coverage"]), ("stability", row["retained_by_stability"]))
            if flag
        )
        or "not_retained",
        axis=1,
    )

    return clusters_df, members_df, cluster_membership_map


def build_reference_frame(scored: pd.DataFrame, reference_depth_map: dict[int, int]) -> pd.DataFrame:
    if not reference_depth_map:
        return scored.iloc[0:0].copy()
    reference_indices = sorted(reference_depth_map)
    frame = scored.loc[reference_indices].copy()
    frame["reference_depth"] = frame.index.map(reference_depth_map.get)
    return frame


def build_final_candidates(
    scored: pd.DataFrame,
    clusters_df: pd.DataFrame,
    cluster_membership_map: dict[str, set[int]],
) -> pd.DataFrame:
    retained_keys = clusters_df.loc[clusters_df["retained"], "cluster_key"].tolist()
    if not retained_keys:
        return scored.iloc[0:0].copy()

    retained_indices: set[int] = set()
    for cluster_key in retained_keys:
        retained_indices |= cluster_membership_map[cluster_key]

    final = scored.loc[sorted(retained_indices)].copy()
    cluster_lookup = clusters_df.set_index("cluster_key")

    def keys_for_index(index: int) -> list[str]:
        return [cluster_key for cluster_key in retained_keys if index in cluster_membership_map[cluster_key]]

    final["retained_cluster_keys"] = final.index.map(lambda index: " | ".join(sorted(keys_for_index(int(index)))))
    final["retained_cluster_count"] = final["retained_cluster_keys"].map(
        lambda value: 0 if not value else len(value.split(" | "))
    )
    final["retained_methods"] = final["retained_cluster_keys"].map(
        lambda value: " | ".join(
            sorted(
                {
                    cluster_lookup.at[key, "method"]
                    for key in value.split(" | ")
                    if key in cluster_lookup.index
                }
            )
        )
    )
    final["retained_by_coverage"] = final["retained_cluster_keys"].map(
        lambda value: any(
            cluster_lookup.at[key, "retained_by_coverage"]
            for key in value.split(" | ")
            if key in cluster_lookup.index
        )
    )
    final["retained_by_stability"] = final["retained_cluster_keys"].map(
        lambda value: any(
            cluster_lookup.at[key, "retained_by_stability"]
            for key in value.split(" | ")
            if key in cluster_lookup.index
        )
    )
    final["retained_reason"] = final["retained_cluster_keys"].map(
        lambda value: " | ".join(
            sorted(
                {
                    cluster_lookup.at[key, "retained_reason"]
                    for key in value.split(" | ")
                    if key in cluster_lookup.index
                }
            )
        )
    )
    return final


def apply_lexical_noise_filter(final_candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if final_candidates.empty:
        return final_candidates.copy(), final_candidates.copy()

    filtered_mask = final_candidates["normalized_candidate"].map(looks_like_lexical_noise)
    removed = final_candidates.loc[filtered_mask].copy()
    kept = final_candidates.loc[~filtered_mask].copy()

    kept["filtered_by_lexical_noise"] = False
    removed["filtered_by_lexical_noise"] = True
    return kept, removed


def build_cluster_overlap_table(
    cluster_members_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cluster_members_df.empty:
        empty = pd.DataFrame(
            columns=[
                "method",
                "cluster_key",
                "cluster_size",
                "concept_count",
                "additional_count",
                "concept_terms",
                "additional_terms",
                "reference_coverage",
                "retained",
            ]
        )
        return empty, empty

    rows = []
    for cluster_key, group in cluster_members_df.groupby("cluster_key", sort=False):
        method = group["method"].iloc[0]
        concept_terms = sorted(group.loc[group["in_reference_set"], "candidate"].dropna().map(str).unique().tolist())
        additional_terms = sorted(group.loc[~group["in_reference_set"], "candidate"].dropna().map(str).unique().tolist())
        cluster_info = clusters_df.loc[clusters_df["cluster_key"].eq(cluster_key)].iloc[0]
        rows.append(
            {
                "method": method,
                "cluster_key": cluster_key,
                "cluster_size": int(cluster_info["cluster_size"]),
                "concept_count": len(concept_terms),
                "additional_count": len(additional_terms),
                "concept_terms": " | ".join(concept_terms),
                "additional_terms": " | ".join(additional_terms),
                "reference_coverage": float(cluster_info["reference_coverage"]),
                "retained": bool(cluster_info["retained"]),
            }
        )

    overlap_df = pd.DataFrame(rows).sort_values(
        ["concept_count", "reference_coverage", "cluster_size"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best_rows = (
        overlap_df.sort_values(
            ["method", "concept_count", "reference_coverage", "cluster_size"],
            ascending=[True, False, False, False],
        )
        .groupby("method", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return overlap_df, best_rows


def build_cross_method_overlap_table(
    scored: pd.DataFrame,
    cluster_membership_map: dict[str, set[int]],
    clusters_df: pd.DataFrame,
    overlap_threshold: float,
) -> pd.DataFrame:
    if clusters_df.empty:
        return pd.DataFrame(
            columns=[
                "left_method",
                "left_cluster_key",
                "left_cluster_size",
                "right_method",
                "right_cluster_key",
                "right_cluster_size",
                "shared_count",
                "shared_fraction_left",
                "shared_fraction_right",
                "shared_fraction_smaller",
                "jaccard",
                "shared_terms",
            ]
        )

    cluster_lookup = clusters_df.set_index("cluster_key")
    cluster_keys = clusters_df["cluster_key"].tolist()
    rows = []

    for left_index, left_key in enumerate(cluster_keys):
        left_members = cluster_membership_map[left_key]
        left_info = cluster_lookup.loc[left_key]
        for right_key in cluster_keys[left_index + 1 :]:
            right_info = cluster_lookup.loc[right_key]
            if left_info["method"] == right_info["method"]:
                continue

            right_members = cluster_membership_map[right_key]
            shared_members = left_members & right_members
            shared_count = len(shared_members)
            if not shared_count:
                continue

            left_size = len(left_members)
            right_size = len(right_members)
            union_size = len(left_members | right_members)
            shared_fraction_left = shared_count / left_size if left_size else 0.0
            shared_fraction_right = shared_count / right_size if right_size else 0.0
            shared_fraction_smaller = shared_count / min(left_size, right_size) if min(left_size, right_size) else 0.0
            jaccard = shared_count / union_size if union_size else 0.0

            if shared_fraction_smaller < overlap_threshold:
                continue

            shared_terms = sorted(
                scored.loc[sorted(shared_members), "candidate"].dropna().map(str).unique().tolist()
            )
            rows.append(
                {
                    "left_method": left_info["method"],
                    "left_cluster_key": left_key,
                    "left_cluster_size": left_size,
                    "right_method": right_info["method"],
                    "right_cluster_key": right_key,
                    "right_cluster_size": right_size,
                    "shared_count": shared_count,
                    "shared_fraction_left": shared_fraction_left,
                    "shared_fraction_right": shared_fraction_right,
                    "shared_fraction_smaller": shared_fraction_smaller,
                    "jaccard": jaccard,
                    "shared_terms": " | ".join(shared_terms),
                }
            )

    overlap_df = pd.DataFrame(rows)
    if overlap_df.empty:
        return overlap_df

    return overlap_df.sort_values(
        ["shared_fraction_smaller", "shared_count", "jaccard"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def save_method_outputs(output_dir: Path, scored: pd.DataFrame, clusters_df: pd.DataFrame, method_name: str) -> None:
    method_dir = output_dir / "by_method"
    method_dir.mkdir(parents=True, exist_ok=True)

    method_keys = clusters_df.loc[clusters_df["method"].eq(method_name), "cluster_key"].tolist()
    if not method_keys:
        return

    cluster_labels = [int(key.split("__", 1)[1]) for key in method_keys]
    cluster_column = f"cluster__{method_name}"
    method_frame = scored[scored[cluster_column].isin(cluster_labels)].copy()
    method_frame.to_csv(
        method_dir / f"{method_name}__seed_cluster_members.csv",
        index=False,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_FILE))
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    validate_inputs(config)

    input_file = Path(config["input_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "by_method").mkdir(parents=True, exist_ok=True)

    seed_terms = config["seed_terms"]
    started = time.time()

    pool = load_candidates(
        input_file=input_file,
        seed_terms=seed_terms,
        context_sentence_limit=int(config.get("context_sentence_limit", 2)),
    )

    model = SentenceTransformer(config["embedding_model"])
    embeddings = encode_representations(pool, model)
    pool = pool.copy()
    pool["embedding_vector"] = embeddings.tolist()
    scored = score_seed_similarity(
        pool=pool,
        embeddings=embeddings,
        seed_terms=seed_terms,
        threshold=float(config["seed_similarity_threshold"]),
    )

    reference_start_indices = set(scored.index[scored["seed_related"]].tolist())
    for seed_term in seed_terms:
        seed_norm = normalize_term(seed_term)
        seed_index = int(scored.index[scored["normalized_candidate"].eq(seed_norm)][0])
        reference_start_indices.add(seed_index)

    graph = build_similarity_graph(
        embeddings=embeddings,
        neighbor_count=int(config.get("graph_neighbors", 10)),
        similarity_threshold=float(config["graph_similarity_threshold"]),
    )
    reference_depth_map = expand_reference_set(
        graph=graph,
        start_indices=reference_start_indices,
        max_depth=int(config["graph_depth"]),
    )
    scored["reference_depth"] = scored.index.map(reference_depth_map.get).fillna(-1).astype(int)
    scored["in_reference_set"] = scored["reference_depth"].ge(0)

    scored = cluster_pool(
        pool=scored,
        embeddings=embeddings,
        cluster_methods=config["cluster_methods"],
    )

    clusters_df, cluster_members_df, cluster_membership_map = collect_seed_clusters(
        scored=scored,
        cluster_methods=config["cluster_methods"],
        seed_terms=seed_terms,
        reference_depth_map=reference_depth_map,
        stability_jaccard_threshold=float(config["stability_jaccard_threshold"]),
        reference_coverage_threshold=float(config["reference_coverage_threshold"]),
    )

    reference_frame = build_reference_frame(scored, reference_depth_map)
    raw_final_candidates = build_final_candidates(scored, clusters_df, cluster_membership_map)
    final_candidates, lexical_noise_removed = apply_lexical_noise_filter(raw_final_candidates)
    concept_graph = build_concept_graph(scored, reference_depth_map)
    overlap_table, best_overlap_table = build_cluster_overlap_table(
        cluster_members_df=cluster_members_df,
        clusters_df=clusters_df,
    )
    cross_method_overlap_table = build_cross_method_overlap_table(
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
    gt_metrics = compute_gt_metrics(final_candidates)
    (output_dir / "gt_metrics.json").write_text(
        json.dumps(gt_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    graph_nodes = plot_concept_graph(
        graph=concept_graph,
        scored=scored,
        final_candidates=final_candidates,
        output_path=output_dir / "concept_graph.png",
    )

    for method in config["cluster_methods"]:
        save_method_outputs(output_dir, scored, clusters_df, method["name"])

    manifest = {
        "stage": "06_seed_guided_contextual_clustering",
        "config_file": str(config_path),
        "input_file": str(input_file),
        "output_dir": str(output_dir),
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
    print("Stage 06 seed-guided contextual clustering completed")
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
        error_file = PROJECT_DIR / "stage_outputs/06_seed_guided_contextual_clustering/ERROR_seed_guided_contextual_clustering.txt"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        error_file.write_text(traceback.format_exc(), encoding="utf-8")
        raise