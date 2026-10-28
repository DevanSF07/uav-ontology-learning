from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path("/home/faujdar/onto_gen/ontology_experiment_framework")
DEFAULT_INPUT = PROJECT_DIR / "stage_outputs/04_gliner_filtering/best_gliner_retained.csv"
DEFAULT_CHUNKS = PROJECT_DIR / "stage_outputs/02_chunking/chunks.csv"
DEFAULT_GT = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_OUTPUT = PROJECT_DIR / "stage_outputs/05_clustering_filtering"

GT_SHEET = "S1_GT_TERMS"
NEGATIVE_SHEET = "S7_NEGATIVE_TERMS"
MANUAL_REVIEW_STATUS = "needs_manual_review"
RANDOM_SEED = 42

MODEL_NAME = "all-mpnet-base-v2"
REPRESENTATIONS = ("term", "term_sentence", "term_tfidf3")
EMBEDDING_SPACES = ("raw", "pca50")

# Phase A parameter search. These grids are deliberately bounded for the first run.
HDBSCAN_CONFIGS = (
    {"min_cluster_size": 5, "min_samples": None, "probability_threshold": 0.0},
    {"min_cluster_size": 10, "min_samples": None, "probability_threshold": 0.0},
    {"min_cluster_size": 15, "min_samples": None, "probability_threshold": 0.0},
    {"min_cluster_size": 10, "min_samples": 5, "probability_threshold": 0.0},
    {"min_cluster_size": 15, "min_samples": 5, "probability_threshold": 0.0},
    {"min_cluster_size": 10, "min_samples": 5, "probability_threshold": 0.40},
)
K_VALUES = (15, 25, 40)
OUTLIER_QUANTILES = (0.90, 0.95, 0.975)
AGGLO_LINKAGES = ("average", "complete")

PRIORITY_WEIGHTS = {"high": 3.0, "medium": 2.0, "low": 1.0}


@dataclass(frozen=True)
class ClusterRun:
    run_id: str
    algorithm: str
    representation: str
    embedding_space: str
    parameters: dict[str, Any]
    assignments: pd.DataFrame
    quality: dict[str, Any]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_term(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).casefold().strip()
    for dash in ("‐", "‑", "‒", "–", "—"):
        text = text.replace(dash, "-")
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"[^a-z0-9&/+ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_search_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).casefold()
    for dash in ("‐", "‑", "‒", "–", "—"):
        text = text.replace(dash, "-")
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"[^a-z0-9&/+.!? ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def split_sentences(text: str) -> list[str]:
    text = safe_text(text)
    if not text:
        return []
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        if part.strip()
    ]


def find_text_column(frame: pd.DataFrame) -> str:
    preferred = (
        "source_sentence", "sentence", "chunk_text", "context_text",
        "text", "chunk", "content",
    )
    lowered = {str(column).casefold(): column for column in frame.columns}
    for name in preferred:
        if name in lowered:
            return lowered[name]
    object_columns = [column for column in frame.columns if frame[column].dtype == "object"]
    if not object_columns:
        raise ValueError("No text-like column was found in chunks.csv")
    return max(
        object_columns,
        key=lambda column: frame[column].fillna("").astype(str).str.len().mean(),
    )


def validate_paths(input_file: Path, chunks_file: Path, gt_file: Path) -> None:
    missing = [str(path) for path in (input_file, chunks_file, gt_file) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_candidates(input_file: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_file, encoding="utf-8")
    required = {"candidate", "normalized_candidate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Candidate file is missing columns: {sorted(missing)}")
    for column in ("source_sentence", "context_text"):
        if column not in frame.columns:
            frame[column] = ""
    frame = frame.copy()
    frame["candidate"] = frame["candidate"].map(safe_text)
    frame["normalized_candidate"] = frame["normalized_candidate"].map(normalize_term)
    frame["source_sentence"] = frame["source_sentence"].map(safe_text)
    frame["context_text"] = frame["context_text"].map(safe_text)
    frame = frame[frame["normalized_candidate"].ne("")]
    frame = frame.drop_duplicates("normalized_candidate", keep="first").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Candidate inventory is empty")
    frame.insert(0, "candidate_row_id", np.arange(len(frame), dtype=int))
    return frame


def load_ground_truth(gt_file: Path):
    gt = pd.read_excel(gt_file, sheet_name=GT_SHEET, engine="openpyxl")
    negatives = pd.read_excel(gt_file, sheet_name=NEGATIVE_SHEET, engine="openpyxl")
    gt = gt[
        gt["keep_for_tasks"].astype(str).str.contains("term_extraction", case=False, na=False)
    ].copy()
    gt["normalized_gt"] = gt["term_canonical"].map(normalize_term)
    gt = gt[gt["normalized_gt"].ne("")]
    status = gt["grounded_status_v2"].fillna("").astype(str).str.casefold()
    strict_gt = gt[status.ne(MANUAL_REVIEW_STATUS)].drop_duplicates("normalized_gt").copy()
    review_gt = gt[status.eq(MANUAL_REVIEW_STATUS)].drop_duplicates("normalized_gt").copy()
    negatives["normalized_negative"] = negatives["candidate"].map(normalize_term)
    negatives = negatives[negatives["normalized_negative"].ne("")]
    return strict_gt, review_gt, negatives


def annotate_evaluation(
    candidates: pd.DataFrame,
    strict_gt: pd.DataFrame,
    review_gt: pd.DataFrame,
    negatives: pd.DataFrame,
) -> pd.DataFrame:
    strict_lookup = strict_gt.set_index("normalized_gt")
    review_lookup = review_gt.set_index("normalized_gt")
    negative_set = set(negatives["normalized_negative"])
    result = candidates.copy()
    result["match_type"] = result["normalized_candidate"].map(
        lambda term: "strict_exact" if term in strict_lookup.index else (
            "manual_review_exact" if term in review_lookup.index else "unmatched"
        )
    )
    result["evaluation_flag_before_filter"] = result["match_type"].map(
        {"strict_exact": "TP", "manual_review_exact": "REVIEW", "unmatched": "FP"}
    )
    result["matched_gt_id"] = result["normalized_candidate"].map(
        lambda term: strict_lookup.at[term, "gt_id"] if term in strict_lookup.index else (
            review_lookup.at[term, "gt_id"] if term in review_lookup.index else ""
        )
    )
    result["gt_category"] = result["normalized_candidate"].map(
        lambda term: strict_lookup.at[term, "gt_category"] if term in strict_lookup.index else (
            review_lookup.at[term, "gt_category"] if term in review_lookup.index else ""
        )
    )
    result["negative_benchmark_hit"] = result["normalized_candidate"].isin(negative_set)
    return result


def load_corpus_sentences(chunks_file: Path) -> pd.DataFrame:
    chunks = pd.read_csv(chunks_file, encoding="utf-8")
    text_column = find_text_column(chunks)
    records = []
    for chunk_position, value in enumerate(chunks[text_column].fillna("")):
        chunk_id = chunks.iloc[chunk_position].get("chunk_id", chunk_position)
        for sentence_position, sentence in enumerate(split_sentences(str(value))):
            records.append({
                "chunk_id": chunk_id,
                "sentence_position": sentence_position,
                "sentence": sentence,
                "normalized_sentence": normalize_search_text(sentence),
            })
    result = pd.DataFrame(records).drop_duplicates("sentence").reset_index(drop=True)
    if result.empty:
        raise ValueError("No corpus sentences could be loaded")
    return result


def term_pattern(normalized_term: str) -> re.Pattern:
    body = r"[\s\-]+".join(re.escape(token) for token in normalized_term.split())
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)


def retrieve_tfidf_contexts(
    candidates: pd.DataFrame,
    sentences: pd.DataFrame,
    top_k: int = 3,
    max_occurrences: int = 100,
) -> pd.DataFrame:
    """Retrieve occurrence sentences and rank them by TF-IDF similarity to term + source sentence."""
    sentence_norm = sentences["normalized_sentence"].tolist()
    records = []

    for row_index, row in enumerate(candidates.itertuples(index=False), start=1):
        pattern = term_pattern(row.normalized_candidate)
        occurrence_indices = [
            index for index, sentence in enumerate(sentence_norm) if pattern.search(sentence)
        ][:max_occurrences]

        occurrence_sentences = sentences.iloc[occurrence_indices]["sentence"].tolist()
        source_sentence = safe_text(row.source_sentence)
        query = f"{row.candidate} {source_sentence}".strip()

        if not occurrence_sentences:
            selected = [source_sentence] if source_sentence else []
            scores = [1.0] if source_sentence else []
        elif len(occurrence_sentences) == 1:
            selected = occurrence_sentences
            scores = [1.0]
        else:
            documents = [query] + occurrence_sentences
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
            )
            matrix = vectorizer.fit_transform(documents)
            similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
            ranked_indices = np.argsort(-similarities)[:top_k]
            selected = [occurrence_sentences[index] for index in ranked_indices]
            scores = [float(similarities[index]) for index in ranked_indices]

        records.append({
            "candidate_row_id": row.candidate_row_id,
            "occurrence_sentence_count": len(occurrence_sentences),
            "tfidf_contexts": selected[:top_k],
            "tfidf_context_scores": scores[:top_k],
        })

        if row_index % 100 == 0 or row_index == len(candidates):
            print(f"Retrieved TF-IDF contexts for {row_index}/{len(candidates)} candidates")

    context_frame = pd.DataFrame(records)
    result = candidates.merge(context_frame, on="candidate_row_id", how="left", validate="one_to_one")
    result["tfidf_contexts"] = result["tfidf_contexts"].map(
        lambda value: value if isinstance(value, list) else []
    )
    result["tfidf_context_scores"] = result["tfidf_context_scores"].map(
        lambda value: value if isinstance(value, list) else []
    )
    return result


def build_representation(row, representation: str) -> str:
    term = safe_text(row.candidate)
    if representation == "term":
        return term
    if representation == "term_sentence":
        sentence = safe_text(row.source_sentence)
        return f"Term: {term}. Context: {sentence}" if sentence else f"Term: {term}"
    if representation == "term_tfidf3":
        contexts = row.tfidf_contexts if isinstance(row.tfidf_contexts, list) else []
        if not contexts:
            sentence = safe_text(row.source_sentence)
            contexts = [sentence] if sentence else []
        joined = " [CONTEXT] ".join(contexts[:3])
        return f"Term: {term}. Context: {joined}" if joined else f"Term: {term}"
    raise ValueError(f"Unknown representation: {representation}")


def encode_representations(
    candidates: pd.DataFrame,
    model: SentenceTransformer,
    representation: str,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    texts = [build_representation(row, representation) for row in candidates.itertuples(index=False)]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return embeddings, texts


def make_embedding_spaces(embeddings: np.ndarray) -> dict[str, np.ndarray]:
    spaces = {"raw": embeddings}
    component_count = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
    if component_count >= 2:
        pca = PCA(n_components=component_count, random_state=RANDOM_SEED)
        reduced = pca.fit_transform(embeddings)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        reduced = reduced / np.clip(norms, 1e-12, None)
        spaces["pca50"] = reduced.astype(np.float32)
    return spaces


def safe_cluster_quality(X: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    non_noise = labels >= 0
    valid_labels = labels[non_noise]
    unique = np.unique(valid_labels)
    result = {
        "sample_count": int(len(labels)),
        "noise_count": int((labels < 0).sum()),
        "noise_rate": float((labels < 0).mean()),
        "cluster_count": int(len(unique)),
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }
    if non_noise.sum() >= 3 and 2 <= len(unique) < non_noise.sum():
        try:
            result["silhouette"] = float(silhouette_score(X[non_noise], valid_labels, metric="cosine"))
        except Exception:
            pass
        try:
            result["calinski_harabasz"] = float(calinski_harabasz_score(X[non_noise], valid_labels))
        except Exception:
            pass
        try:
            result["davies_bouldin"] = float(davies_bouldin_score(X[non_noise], valid_labels))
        except Exception:
            pass
    return result


def fit_hdbscan(X: np.ndarray, parameters: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use external hdbscan when available; otherwise use sklearn.cluster.HDBSCAN."""
    minimum_probability = float(parameters["probability_threshold"])
    try:
        import hdbscan as hdbscan_package

        model = hdbscan_package.HDBSCAN(
            min_cluster_size=int(parameters["min_cluster_size"]),
            min_samples=parameters["min_samples"],
            metric="euclidean",
            prediction_data=False,
        )
        labels = model.fit_predict(X)
        probabilities = np.asarray(model.probabilities_, dtype=float)
        outlier_scores = np.asarray(model.outlier_scores_, dtype=float)
    except ImportError:
        from sklearn.cluster import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=int(parameters["min_cluster_size"]),
            min_samples=parameters["min_samples"],
            metric="euclidean",
            store_centers="centroid",
        )
        labels = model.fit_predict(X)
        probabilities = np.ones(len(labels), dtype=float)
        probabilities[labels < 0] = 0.0
        outlier_scores = np.where(labels < 0, 1.0, 0.0)

    reject = (labels < 0) | (probabilities < minimum_probability)
    return labels.astype(int), probabilities, reject.astype(bool), outlier_scores


def cluster_centroids(X: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    centers = {}
    for cluster_id in sorted(set(int(value) for value in labels if value >= 0)):
        center = X[labels == cluster_id].mean(axis=0)
        norm = np.linalg.norm(center)
        centers[cluster_id] = center / norm if norm > 0 else center
    return centers


def cosine_distances_to_centers(X: np.ndarray, labels: np.ndarray, centers: dict[int, np.ndarray]) -> np.ndarray:
    distances = np.zeros(len(X), dtype=float)
    for index, cluster_id in enumerate(labels):
        center = centers[int(cluster_id)]
        x = X[index]
        x_norm = np.linalg.norm(x)
        center_norm = np.linalg.norm(center)
        similarity = float(np.dot(x, center) / max(x_norm * center_norm, 1e-12))
        distances[index] = 1.0 - np.clip(similarity, -1.0, 1.0)
    return distances


def within_cluster_percentile(distances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    percentiles = np.zeros(len(distances), dtype=float)
    for cluster_id in sorted(set(int(value) for value in labels)):
        indices = np.where(labels == cluster_id)[0]
        values = distances[indices]
        if len(indices) == 1:
            percentiles[indices] = 1.0
            continue
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        percentiles[indices] = ranks / max(len(values) - 1, 1)
    return percentiles


def run_hdbscan(
    X: np.ndarray,
    representation: str,
    embedding_space: str,
    parameters: dict[str, Any],
) -> ClusterRun:
    labels, probabilities, reject, outlier_scores = fit_hdbscan(X, parameters)
    assignments = pd.DataFrame({
        "cluster_label": labels,
        "membership_probability": probabilities,
        "outlier_score": outlier_scores,
        "distance_to_centroid": np.nan,
        "distance_percentile": np.nan,
        "retain": ~reject,
    })
    parameter_text = "_".join(f"{key}-{value}" for key, value in parameters.items())
    run_id = f"{representation}__{embedding_space}__hdbscan__{parameter_text}"
    return ClusterRun(
        run_id=run_id,
        algorithm="hdbscan",
        representation=representation,
        embedding_space=embedding_space,
        parameters=parameters,
        assignments=assignments,
        quality=safe_cluster_quality(X, labels),
    )


def run_kmeans(
    X: np.ndarray,
    representation: str,
    embedding_space: str,
    cluster_count: int,
    outlier_quantile: float,
) -> ClusterRun:
    effective_k = min(cluster_count, max(2, len(X) - 1))
    model = KMeans(n_clusters=effective_k, random_state=RANDOM_SEED, n_init=20)
    labels = model.fit_predict(X)
    centers = cluster_centroids(X, labels)
    distances = cosine_distances_to_centers(X, labels, centers)
    percentiles = within_cluster_percentile(distances, labels)
    reject = percentiles >= outlier_quantile
    assignments = pd.DataFrame({
        "cluster_label": labels,
        "membership_probability": np.nan,
        "outlier_score": percentiles,
        "distance_to_centroid": distances,
        "distance_percentile": percentiles,
        "retain": ~reject,
    })
    parameters = {"n_clusters": effective_k, "outlier_quantile": outlier_quantile}
    run_id = f"{representation}__{embedding_space}__kmeans__k-{effective_k}_q-{outlier_quantile}"
    return ClusterRun(
        run_id=run_id,
        algorithm="kmeans",
        representation=representation,
        embedding_space=embedding_space,
        parameters=parameters,
        assignments=assignments,
        quality=safe_cluster_quality(X, labels),
    )


def make_agglomerative(cluster_count: int, linkage: str):
    kwargs = {"n_clusters": cluster_count, "linkage": linkage}
    try:
        return AgglomerativeClustering(metric="cosine", **kwargs)
    except TypeError:
        return AgglomerativeClustering(affinity="cosine", **kwargs)


def run_agglomerative(
    X: np.ndarray,
    representation: str,
    embedding_space: str,
    cluster_count: int,
    linkage: str,
    outlier_quantile: float,
) -> ClusterRun:
    effective_k = min(cluster_count, max(2, len(X) - 1))
    model = make_agglomerative(effective_k, linkage)
    labels = model.fit_predict(X)
    centers = cluster_centroids(X, labels)
    distances = cosine_distances_to_centers(X, labels, centers)
    percentiles = within_cluster_percentile(distances, labels)
    reject = percentiles >= outlier_quantile
    assignments = pd.DataFrame({
        "cluster_label": labels,
        "membership_probability": np.nan,
        "outlier_score": percentiles,
        "distance_to_centroid": distances,
        "distance_percentile": percentiles,
        "retain": ~reject,
    })
    parameters = {
        "n_clusters": effective_k,
        "linkage": linkage,
        "outlier_quantile": outlier_quantile,
    }
    run_id = (
        f"{representation}__{embedding_space}__agglomerative__"
        f"k-{effective_k}_link-{linkage}_q-{outlier_quantile}"
    )
    return ClusterRun(
        run_id=run_id,
        algorithm="agglomerative",
        representation=representation,
        embedding_space=embedding_space,
        parameters=parameters,
        assignments=assignments,
        quality=safe_cluster_quality(X, labels),
    )


def evaluate_mask(
    experiment: str,
    candidates: pd.DataFrame,
    retain_mask: np.ndarray | pd.Series,
    strict_gt: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    retain_mask = np.asarray(retain_mask, dtype=bool)
    retained = candidates.loc[retain_mask].copy()
    dropped = candidates.loc[~retain_mask].copy()
    strict_unique = strict_gt.drop_duplicates("normalized_gt")
    gold = set(strict_unique["normalized_gt"])
    scored = retained[retained["match_type"].ne("manual_review_exact")]
    predicted = set(scored["normalized_candidate"])
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    gt_weights = (
        strict_unique["priority"].fillna("low").astype(str).str.casefold()
        .map(PRIORITY_WEIGHTS).fillna(1.0)
    )
    matched = strict_unique["normalized_gt"].isin(predicted)
    weighted_recall = float(gt_weights[matched].sum() / gt_weights.sum()) if gt_weights.sum() else 0.0

    tp_dropped = int(dropped["match_type"].eq("strict_exact").sum())
    fp_dropped = int(dropped["match_type"].eq("unmatched").sum())
    metric = {
        "experiment": experiment,
        **metadata,
        "input_candidate_count": len(candidates),
        "retained_candidate_count": len(retained),
        "dropped_candidate_count": len(dropped),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "priority_weighted_recall": weighted_recall,
        "tp_retained": int(retained["match_type"].eq("strict_exact").sum()),
        "tp_dropped": tp_dropped,
        "fp_retained": int(retained["match_type"].eq("unmatched").sum()),
        "fp_dropped": fp_dropped,
        "manual_review_retained": int(retained["match_type"].eq("manual_review_exact").sum()),
        "negative_hits_retained": int(retained["negative_benchmark_hit"].sum()),
        "negative_hits_dropped": int(dropped["negative_benchmark_hit"].sum()),
        "filter_efficiency_fp_per_tp_removed": fp_dropped / (tp_dropped + 1),
    }
    retained["filter_experiment"] = experiment
    retained["filter_decision"] = "retained"
    dropped["filter_experiment"] = experiment
    dropped["filter_decision"] = "dropped"
    return metric, retained, dropped


def evaluate_cluster_run(
    run: ClusterRun,
    candidates: pd.DataFrame,
    strict_gt: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metadata = {
        "phase": "A_individual_parameter_search",
        "algorithm": run.algorithm,
        "representation": run.representation,
        "embedding_space": run.embedding_space,
        "decision_rule": "individual",
        "parameters": json.dumps(run.parameters, sort_keys=True),
        **run.quality,
    }
    metric, _, _ = evaluate_mask(
        run.run_id,
        candidates,
        run.assignments["retain"].to_numpy(),
        strict_gt,
        metadata,
    )
    assignments = pd.concat(
        [candidates[["candidate_row_id", "candidate", "normalized_candidate"]].reset_index(drop=True),
         run.assignments.reset_index(drop=True)],
        axis=1,
    )
    assignments.insert(0, "run_id", run.run_id)
    assignments.insert(1, "algorithm", run.algorithm)
    assignments.insert(2, "representation", run.representation)
    assignments.insert(3, "embedding_space", run.embedding_space)
    assignments["parameters"] = json.dumps(run.parameters, sort_keys=True)
    return metric, assignments


def choose_best_runs(metrics: pd.DataFrame, runs: dict[str, ClusterRun]) -> dict[tuple[str, str, str], ClusterRun]:
    selected = {}
    for key, group in metrics.groupby(["representation", "embedding_space", "algorithm"], sort=True):
        best = group.sort_values(
            ["f1", "precision", "recall", "silhouette"],
            ascending=[False, False, False, False],
            na_position="last",
        ).iloc[0]
        selected[key] = runs[best["experiment"]]
    return selected


def combination_rules(h: np.ndarray, k: np.ndarray, a: np.ndarray) -> dict[str, np.ndarray]:
    votes = h.astype(int) + k.astype(int) + a.astype(int)
    return {
        "hdbscan_only": h,
        "kmeans_only": k,
        "agglomerative_only": a,
        "hdbscan_or_kmeans": h | k,
        "hdbscan_and_kmeans": h & k,
        "hdbscan_or_agglomerative": h | a,
        "hdbscan_and_agglomerative": h & a,
        "kmeans_or_agglomerative": k | a,
        "kmeans_and_agglomerative": k & a,
        "at_least_1_of_3": votes >= 1,
        "at_least_2_of_3": votes >= 2,
        "all_3": votes == 3,
    }


def phase_b_combinations(
    selected_runs: dict[tuple[str, str, str], ClusterRun],
    candidates: pd.DataFrame,
    strict_gt: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    metrics = []
    outputs = {}
    vote_frames = []

    for representation in REPRESENTATIONS:
        for embedding_space in EMBEDDING_SPACES:
            key_h = (representation, embedding_space, "hdbscan")
            key_k = (representation, embedding_space, "kmeans")
            key_a = (representation, embedding_space, "agglomerative")
            if not all(key in selected_runs for key in (key_h, key_k, key_a)):
                continue
            h_run, k_run, a_run = selected_runs[key_h], selected_runs[key_k], selected_runs[key_a]
            h = h_run.assignments["retain"].to_numpy(dtype=bool)
            k = k_run.assignments["retain"].to_numpy(dtype=bool)
            a = a_run.assignments["retain"].to_numpy(dtype=bool)

            vote_frame = candidates[[
                "candidate_row_id", "candidate", "normalized_candidate", "match_type",
                "negative_benchmark_hit",
            ]].copy()
            vote_frame["representation"] = representation
            vote_frame["embedding_space"] = embedding_space
            vote_frame["hdbscan_run_id"] = h_run.run_id
            vote_frame["kmeans_run_id"] = k_run.run_id
            vote_frame["agglomerative_run_id"] = a_run.run_id
            vote_frame["hdbscan_retain"] = h
            vote_frame["kmeans_retain"] = k
            vote_frame["agglomerative_retain"] = a
            vote_frame["retain_vote_count"] = h.astype(int) + k.astype(int) + a.astype(int)
            vote_frames.append(vote_frame)

            for rule_name, mask in combination_rules(h, k, a).items():
                experiment = f"{representation}__{embedding_space}__combo__{rule_name}"
                metadata = {
                    "phase": "B_combination_testing",
                    "algorithm": "combination",
                    "representation": representation,
                    "embedding_space": embedding_space,
                    "decision_rule": rule_name,
                    "parameters": json.dumps({
                        "hdbscan_run": h_run.run_id,
                        "kmeans_run": k_run.run_id,
                        "agglomerative_run": a_run.run_id,
                    }, sort_keys=True),
                    "sample_count": len(candidates),
                    "noise_count": int((~mask).sum()),
                    "noise_rate": float((~mask).mean()),
                    "cluster_count": np.nan,
                    "silhouette": np.nan,
                    "calinski_harabasz": np.nan,
                    "davies_bouldin": np.nan,
                }
                metric, retained, dropped = evaluate_mask(
                    experiment, candidates, mask, strict_gt, metadata
                )
                metrics.append(metric)
                outputs[experiment] = (retained, dropped)

    return (
        pd.DataFrame(metrics),
        outputs,
        pd.concat(vote_frames, ignore_index=True) if vote_frames else pd.DataFrame(),
    )


def format_workbook(path: Path) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = load_workbook(path)
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.sheet_view.showGridLines = False
        if worksheet.max_row and worksheet.max_column:
            worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for column in range(1, worksheet.max_column + 1):
            lengths = [
                len(str(worksheet.cell(row, column).value or ""))
                for row in range(1, min(worksheet.max_row, 120) + 1)
            ]
            worksheet.column_dimensions[get_column_letter(column)].width = min(
                max(max(lengths, default=10) + 2, 11), 42
            )
        if worksheet.title in {"All Metrics", "Phase A Metrics", "Phase B Metrics"}:
            headers = {worksheet.cell(1, col).value: col for col in range(1, worksheet.max_column + 1)}
            for name in ("precision", "recall", "f1", "priority_weighted_recall", "noise_rate"):
                if name in headers:
                    for row in range(2, worksheet.max_row + 1):
                        worksheet.cell(row, headers[name]).number_format = "0.00%"
    workbook.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental clustering-based noise filtering using MPNet term/context embeddings."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller first-pass grid: one HDBSCAN config, K=25, q=0.95, average linkage.",
    )
    args = parser.parse_args()

    set_random_seed(args.seed)
    validate_paths(args.input, args.chunks, args.ground_truth)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ("context_retrieval", "embeddings", "clustering", "voting", "best"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    started = time.time()
    candidates = load_candidates(args.input)
    strict_gt, review_gt, negatives = load_ground_truth(args.ground_truth)
    candidates = annotate_evaluation(candidates, strict_gt, review_gt, negatives)
    sentences = load_corpus_sentences(args.chunks)
    candidates = retrieve_tfidf_contexts(candidates, sentences, top_k=3)

    context_export = candidates.copy()
    context_export["tfidf_contexts"] = context_export["tfidf_contexts"].map(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    context_export["tfidf_context_scores"] = context_export["tfidf_context_scores"].map(
        lambda value: json.dumps(value)
    )
    context_export.to_csv(
        output_dir / "context_retrieval/candidate_contexts.csv", index=False, encoding="utf-8"
    )

    model = SentenceTransformer(args.model)
    runs: dict[str, ClusterRun] = {}
    phase_a_metrics = []
    all_assignments = []
    embedding_manifest = []

    hdbscan_configs = HDBSCAN_CONFIGS[:1] if args.quick else HDBSCAN_CONFIGS
    k_values = (25,) if args.quick else K_VALUES
    quantiles = (0.95,) if args.quick else OUTLIER_QUANTILES
    linkages = ("average",) if args.quick else AGGLO_LINKAGES

    for representation in REPRESENTATIONS:
        print("\n" + "=" * 80)
        print(f"Encoding representation: {representation}")
        print("=" * 80)
        embeddings, embedding_inputs = encode_representations(
            candidates, model, representation, args.batch_size
        )
        np.save(output_dir / f"embeddings/{representation}__mpnet.npy", embeddings)
        pd.DataFrame({
            "candidate_row_id": candidates["candidate_row_id"],
            "candidate": candidates["candidate"],
            "representation": representation,
            "embedding_input": embedding_inputs,
        }).to_csv(
            output_dir / f"embeddings/{representation}__inputs.csv", index=False, encoding="utf-8"
        )

        spaces = make_embedding_spaces(embeddings)
        for embedding_space, X in spaces.items():
            np.save(output_dir / f"embeddings/{representation}__{embedding_space}.npy", X)
            embedding_manifest.append({
                "representation": representation,
                "embedding_space": embedding_space,
                "sample_count": X.shape[0],
                "dimension_count": X.shape[1],
            })

            for parameters in hdbscan_configs:
                run = run_hdbscan(X, representation, embedding_space, dict(parameters))
                runs[run.run_id] = run
                metric, assignments = evaluate_cluster_run(run, candidates, strict_gt)
                phase_a_metrics.append(metric)
                all_assignments.append(assignments)

            for cluster_count in k_values:
                for outlier_quantile in quantiles:
                    run = run_kmeans(
                        X, representation, embedding_space, cluster_count, outlier_quantile
                    )
                    runs[run.run_id] = run
                    metric, assignments = evaluate_cluster_run(run, candidates, strict_gt)
                    phase_a_metrics.append(metric)
                    all_assignments.append(assignments)

            for cluster_count in k_values:
                for linkage in linkages:
                    for outlier_quantile in quantiles:
                        run = run_agglomerative(
                            X, representation, embedding_space, cluster_count,
                            linkage, outlier_quantile,
                        )
                        runs[run.run_id] = run
                        metric, assignments = evaluate_cluster_run(run, candidates, strict_gt)
                        phase_a_metrics.append(metric)
                        all_assignments.append(assignments)

    phase_a_df = pd.DataFrame(phase_a_metrics).sort_values(
        ["f1", "precision", "recall"], ascending=[False, False, False]
    ).reset_index(drop=True)
    selected_runs = choose_best_runs(phase_a_df, runs)
    phase_b_df, phase_b_outputs, vote_matrix = phase_b_combinations(
        selected_runs, candidates, strict_gt
    )
    phase_b_df = phase_b_df.sort_values(
        ["f1", "precision", "recall"], ascending=[False, False, False]
    ).reset_index(drop=True)

    baseline_metadata = {
        "phase": "baseline",
        "algorithm": "none",
        "representation": "none",
        "embedding_space": "none",
        "decision_rule": "no_clustering_filter",
        "parameters": "{}",
        "sample_count": len(candidates),
        "noise_count": 0,
        "noise_rate": 0.0,
        "cluster_count": np.nan,
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }
    baseline_metric, baseline_retained, baseline_dropped = evaluate_mask(
        "baseline_no_clustering_filter",
        candidates,
        np.ones(len(candidates), dtype=bool),
        strict_gt,
        baseline_metadata,
    )

    all_metrics = pd.concat(
        [pd.DataFrame([baseline_metric]), phase_a_df, phase_b_df], ignore_index=True
    ).sort_values(["f1", "precision", "recall"], ascending=[False, False, False]).reset_index(drop=True)

    best_metric = all_metrics.iloc[0]
    best_experiment = best_metric["experiment"]
    if best_experiment in phase_b_outputs:
        best_retained, best_dropped = phase_b_outputs[best_experiment]
    elif best_experiment in runs:
        best_run = runs[best_experiment]
        _, best_retained, best_dropped = evaluate_mask(
            best_experiment,
            candidates,
            best_run.assignments["retain"].to_numpy(),
            strict_gt,
            {
                "phase": "A_individual_parameter_search",
                "algorithm": best_run.algorithm,
                "representation": best_run.representation,
                "embedding_space": best_run.embedding_space,
                "decision_rule": "individual",
                "parameters": json.dumps(best_run.parameters, sort_keys=True),
                **best_run.quality,
            },
        )
    else:
        best_retained, best_dropped = baseline_retained, baseline_dropped

    phase_a_df.to_csv(output_dir / "clustering/phase_a_individual_metrics.csv", index=False)
    phase_b_df.to_csv(output_dir / "voting/phase_b_combination_metrics.csv", index=False)
    all_metrics.to_csv(output_dir / "clustering_metrics.csv", index=False)
    pd.DataFrame(embedding_manifest).to_csv(output_dir / "embeddings/embedding_manifest.csv", index=False)
    vote_matrix.to_csv(output_dir / "voting/candidate_vote_matrix.csv", index=False)
    if all_assignments:
        pd.concat(all_assignments, ignore_index=True).to_csv(
            output_dir / "clustering/all_cluster_assignments.csv", index=False
        )

    selected_rows = []
    for key, run in selected_runs.items():
        selected_rows.append({
            "representation": key[0],
            "embedding_space": key[1],
            "algorithm": key[2],
            "run_id": run.run_id,
            "parameters": json.dumps(run.parameters, sort_keys=True),
        })
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(output_dir / "clustering/selected_algorithm_runs.csv", index=False)

    best_retained.to_csv(output_dir / "best_clustering_retained.csv", index=False)
    best_dropped.to_csv(output_dir / "best_clustering_dropped.csv", index=False)

    report_path = output_dir / "clustering_filtering_report.xlsx"
    readme = pd.DataFrame({
        "Field": [
            "Purpose", "Input", "Baseline", "Model", "Representations",
            "Embedding spaces", "Phase A", "Phase B", "Best experiment", "Caution",
        ],
        "Value": [
            "Test clustering-based noise filtering with HDBSCAN, K-means, and agglomerative clustering.",
            str(args.input),
            "No-clustering filter is included in all metrics.",
            args.model,
            ", ".join(REPRESENTATIONS),
            ", ".join(EMBEDDING_SPACES),
            "Parameter search for each individual clustering algorithm.",
            "Permutation and combination voting using the best development run per algorithm and representation.",
            best_experiment,
            "Best parameter selection and reporting use the same silver GT; validate the chosen rule on held-out data.",
        ],
    })
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        all_metrics.to_excel(writer, sheet_name="All Metrics", index=False)
        phase_a_df.to_excel(writer, sheet_name="Phase A Metrics", index=False)
        phase_b_df.to_excel(writer, sheet_name="Phase B Metrics", index=False)
        selected_df.to_excel(writer, sheet_name="Selected Runs", index=False)
        best_retained.to_excel(writer, sheet_name="Best Retained", index=False)
        best_dropped.to_excel(writer, sheet_name="Best Dropped", index=False)
    format_workbook(report_path)

    manifest = {
        "stage": "05_clustering_filtering",
        "input_file": str(args.input),
        "chunks_file": str(args.chunks),
        "ground_truth_file": str(args.ground_truth),
        "model": args.model,
        "random_seed": args.seed,
        "quick_mode": args.quick,
        "representations": list(REPRESENTATIONS),
        "embedding_spaces": list(EMBEDDING_SPACES),
        "hdbscan_configs": list(hdbscan_configs),
        "k_values": list(k_values),
        "outlier_quantiles": list(quantiles),
        "agglomerative_linkages": list(linkages),
        "input_candidate_count": len(candidates),
        "strict_gt_count": len(strict_gt),
        "phase_a_run_count": len(phase_a_df),
        "phase_b_rule_count": len(phase_b_df),
        "best_experiment": best_experiment,
        "best_f1": float(best_metric["f1"]),
        "duration_seconds": round(time.time() - started, 3),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 100)
    print("Stage 05 clustering-based noise filtering completed")
    print("=" * 100)
    display_columns = [
        "experiment", "phase", "algorithm", "representation", "embedding_space",
        "retained_candidate_count", "true_positives", "false_positives",
        "precision", "recall", "f1", "tp_dropped", "fp_dropped",
    ]
    print(all_metrics[display_columns].head(20).to_string(index=False))
    print(f"\nBest experiment: {best_experiment}")
    print(f"Best retained: {output_dir / 'best_clustering_retained.csv'}")
    print(f"Best dropped: {output_dir / 'best_clustering_dropped.csv'}")
    print(f"Metrics: {output_dir / 'clustering_metrics.csv'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
        raise
