from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.cluster import Birch, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_FILE = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_OUTPUT_FILE = (
    PROJECT_DIR / "stage_outputs/06_seed_guided_contextual_clustering/s1_domain_clustering_ground_truth.xlsx"
)
DEFAULT_EMBEDDING_MODEL = "models/all-mpnet-base-v2"
SOURCE_SHEET = "S1_GT_TERMS"


def split_gt_term(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    raw_parts = [part.strip() for part in re.split(r"\s*[\/&]\s*", text)]
    parts = [part for part in raw_parts if part]
    if len(parts) <= 1:
        return parts

    tail_tokens = parts[-1].split()
    if len(tail_tokens) > 1:
        shared_suffix = " ".join(tail_tokens[1:])
        if shared_suffix:
            expanded_parts = []
            for part in parts:
                part_tokens = part.split()
                if len(part_tokens) == 1:
                    expanded_parts.append(f"{part} {shared_suffix}")
                else:
                    expanded_parts.append(part)
            return expanded_parts

    return parts


def normalize_term(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).casefold().strip()
    for dash in ("‐", "‑", "‒", "–", "—"):
        text = text.replace(dash, "-")
    text = text.replace("-", " ")
    text = " ".join(text.split())
    return "".join(char for char in text if char.isalnum() or char.isspace())


def load_domain_terms(source_file: Path) -> pd.DataFrame:
    terms = pd.read_excel(source_file, sheet_name=SOURCE_SHEET, engine="openpyxl")

    required_columns = {"gt_id", "term_canonical", "keep_for_tasks", "grounded_status_v2", "priority"}
    missing = required_columns - set(terms.columns)
    if missing:
        raise ValueError(f"{source_file} is missing columns: {sorted(missing)}")

    terms = terms[
        terms["keep_for_tasks"].astype(str).str.contains("term_extraction", case=False, na=False)
    ].copy()
    expanded_rows = []
    split_map_rows = []
    for row in terms.to_dict("records"):
        split_terms = split_gt_term(row["term_canonical"])
        if not split_terms:
            split_terms = [row["term_canonical"]]
        for split_index, split_term in enumerate(split_terms, start=1):
            expanded_row = dict(row)
            expanded_row["term_canonical"] = split_term
            expanded_row["split_index"] = split_index
            expanded_row["split_count"] = len(split_terms)
            expanded_rows.append(expanded_row)
            split_map_rows.append(
                {
                    "gt_id": row["gt_id"],
                    "source_term_canonical": row["term_canonical"],
                    "split_term_canonical": split_term,
                    "split_index": split_index,
                    "split_count": len(split_terms),
                }
            )

    terms = pd.DataFrame(expanded_rows)
    terms.attrs["split_map"] = pd.DataFrame(split_map_rows)
    terms["normalized_term"] = terms["term_canonical"].map(normalize_term)
    terms = terms[terms["normalized_term"].ne("")].copy().reset_index(drop=True)
    return terms


def encode_terms(terms: pd.DataFrame, model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        terms["term_canonical"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    return np.asarray(embeddings)


def cluster_with_kmeans(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return clusterer.fit_predict(embeddings)


def cluster_with_birch(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    clusterer = Birch(n_clusters=n_clusters)
    return clusterer.fit_predict(embeddings)


def cluster_with_hdbscan(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.0,
        n_components=5,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
    return clusterer.fit_predict(reduced)


def add_cluster_metadata(frame: pd.DataFrame, embeddings: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    result["cluster_label"] = labels
    result["is_noise"] = result["cluster_label"].eq(-1)
    result["cluster_size"] = result.groupby("cluster_label")["cluster_label"].transform("size")
    result.loc[result["is_noise"], "cluster_size"] = 1

    representatives: dict[int, str] = {}
    for cluster_label in sorted(set(int(label) for label in labels if int(label) >= 0)):
        member_mask = result["cluster_label"].eq(cluster_label)
        member_indices = np.flatnonzero(member_mask.to_numpy())
        member_embeddings = embeddings[member_indices]
        centroid = member_embeddings.mean(axis=0, keepdims=True)
        similarities = cosine_similarity(member_embeddings, centroid).ravel()
        best_position = int(member_indices[int(similarities.argmax())])
        representatives[cluster_label] = str(result.at[best_position, "term_canonical"])

    result["cluster_representative"] = result["cluster_label"].map(representatives).fillna("")
    return result


def build_variant_frame(
    terms: pd.DataFrame,
    embeddings: np.ndarray,
    method: str,
    parameter: int,
) -> pd.DataFrame:
    if method == "kmeans":
        labels = cluster_with_kmeans(embeddings, parameter)
    elif method == "birch":
        labels = cluster_with_birch(embeddings, parameter)
    elif method == "hdbscan":
        labels = cluster_with_hdbscan(embeddings, parameter)
    else:
        raise ValueError(f"Unknown method: {method}")

    frame = add_cluster_metadata(terms, embeddings, labels)
    frame.insert(0, "method", method)
    frame.insert(1, "cluster_parameter", parameter)
    frame.insert(2, "gt_variant", f"{method}_{parameter:02d}")
    return frame[
        [
            "gt_variant",
            "method",
            "cluster_parameter",
            "gt_id",
            "split_index",
            "split_count",
            "term_canonical",
            "normalized_term",
            "cluster_label",
            "cluster_size",
            "cluster_representative",
            "is_noise",
            "grounded_status_v2",
            "priority",
            "keep_for_tasks",
            "gt_category",
            "expected_type",
            "evidence_chapter",
            "evidence_page_pdf",
            "why_expected",
        ]
    ].sort_values(["cluster_label", "term_canonical"], ascending=[True, True]).reset_index(drop=True)


def build_summary(variant_frames: dict[str, pd.DataFrame], source_file: Path, model_name: str) -> pd.DataFrame:
    rows = []
    for variant_name, frame in variant_frames.items():
        labels = frame["cluster_label"].astype(int)
        non_noise = frame.loc[~frame["is_noise"], "cluster_label"]
        rows.append(
            {
                "gt_variant": variant_name,
                "method": frame["method"].iloc[0],
                "cluster_parameter": int(frame["cluster_parameter"].iloc[0]),
                "term_count": int(len(frame)),
                "cluster_count": int(non_noise.nunique()),
                "noise_count": int(labels.eq(-1).sum()),
                "largest_cluster_size": int(frame.loc[~frame["is_noise"], "cluster_size"].max() if non_noise.nunique() else 0),
                "smallest_cluster_size": int(frame.loc[~frame["is_noise"], "cluster_size"].min() if non_noise.nunique() else 0),
                "source_file": str(source_file),
                "embedding_model": model_name,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a separate clustering ground-truth workbook from S1_GT_TERMS using kmeans, birch, and hdbscan."
    )
    parser.add_argument("--source-file", default=str(DEFAULT_SOURCE_FILE))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--cluster-sizes", default="8,9,10")
    args = parser.parse_args()

    source_file = Path(args.source_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cluster_sizes = [int(value.strip()) for value in args.cluster_sizes.split(",") if value.strip()]
    if not cluster_sizes:
        raise ValueError("At least one cluster size must be provided.")

    terms = load_domain_terms(source_file)
    embeddings = encode_terms(terms, args.embedding_model)

    variant_frames: dict[str, pd.DataFrame] = {}
    for method in ("kmeans", "birch", "hdbscan"):
        for parameter in cluster_sizes:
            variant_name = f"{method}_{parameter:02d}"
            variant_frames[variant_name] = build_variant_frame(terms, embeddings, method, parameter)

    summary = build_summary(variant_frames, source_file, args.embedding_model)
    split_map = terms.attrs.get("split_map", pd.DataFrame())
    readme = pd.DataFrame(
        [
            ("source_file", str(source_file)),
            ("source_sheet", SOURCE_SHEET),
            ("term_rows", str(len(terms))),
            ("embedding_model", args.embedding_model),
            ("cluster_sizes", ", ".join(str(size) for size in cluster_sizes)),
            ("methods", "kmeans, birch, hdbscan"),
            ("note", "hdbscan uses UMAP + HDBSCAN with min_cluster_size set to the requested cluster size."),
        ],
        columns=["field", "value"],
    )

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        if not split_map.empty:
            split_map.to_excel(writer, sheet_name="split_term_map", index=False)
        for variant_name, frame in variant_frames.items():
            frame.to_excel(writer, sheet_name=variant_name[:31], index=False)

    print(f"Wrote clustering ground truth workbook: {output_file}")
    print(summary[["gt_variant", "cluster_count", "noise_count", "term_count"]].to_string(index=False))


if __name__ == "__main__":
    main()