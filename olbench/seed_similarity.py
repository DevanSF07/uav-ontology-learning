from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


DOMAIN_SEEDS = [
    "UAV unmanned aerial vehicle",
    "unmanned aircraft",
    "unmanned aircraft system",
    "aircraft",
    "remotely piloted aircraft",
    "remotely piloted aircraft system",
    "aerial vehicle",
    "flight control system",
    "ground control station",
    "aircraft mission",
]


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


def build_representations(candidates: pd.DataFrame, representation_name: str) -> list[str]:
    representations: list[str] = []

    for row in candidates.itertuples():
        term = safe_text(getattr(row, "candidate", getattr(row, "term", "")))
        sentence = safe_text(getattr(row, "source_sentence", ""))
        context = safe_text(getattr(row, "context_text", ""))

        if representation_name == "term":
            value = term
        elif representation_name == "term_sentence":
            value = f"{term}. Context sentence: {sentence}" if sentence else term
        elif representation_name == "term_context":
            value = f"{term}. Context: {context}" if context else term
        else:
            raise ValueError(f"Unknown representation: {representation_name}")

        representations.append(value)

    return representations


def score_seed_similarity(
    candidates: pd.DataFrame,
    model,
    model_key: str,
    model_name: str,
    representation_name: str = "term_context",
    seed_terms: Iterable[str] = DOMAIN_SEEDS,
    batch_size: int = 32,
) -> pd.DataFrame:
    seed_terms = list(seed_terms)
    if candidates.empty:
        result = candidates.copy()
        result["model_key"] = model_key
        result["model_name"] = model_name
        result["representation"] = representation_name
        result["embedding_input"] = []
        result["max_seed_similarity"] = []
        result["mean_seed_similarity"] = []
        result["median_seed_similarity"] = []
        result["closest_seed"] = []
        for index in range(len(seed_terms)):
            result[f"seed_{index + 1:02d}_similarity"] = []
        for threshold in (0.30, 0.40, 0.50):
            result[f"seed_count_ge_{threshold:.2f}"] = []
        return result

    representations = build_representations(candidates, representation_name)

    candidate_embeddings = model.encode(
        representations,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    seed_embeddings = model.encode(
        seed_terms,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )

    similarity_matrix = np.asarray(candidate_embeddings) @ np.asarray(seed_embeddings).T

    result = candidates.copy()
    result["model_key"] = model_key
    result["model_name"] = model_name
    result["representation"] = representation_name
    result["embedding_input"] = representations

    for seed_index, _seed in enumerate(seed_terms):
        result[f"seed_{seed_index + 1:02d}_similarity"] = similarity_matrix[:, seed_index]

    closest_indices = similarity_matrix.argmax(axis=1)
    result["max_seed_similarity"] = similarity_matrix.max(axis=1)
    result["mean_seed_similarity"] = similarity_matrix.mean(axis=1)
    result["median_seed_similarity"] = np.median(similarity_matrix, axis=1)
    result["closest_seed"] = [seed_terms[index] for index in closest_indices]

    for threshold in (0.30, 0.40, 0.50):
        result[f"seed_count_ge_{threshold:.2f}"] = (similarity_matrix >= threshold).sum(axis=1)

    return result
