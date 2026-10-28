from __future__ import annotations

import re
from typing import List


def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter for technical text.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def build_sentence_index(text: str, normalize_fn):
    """
    Build a sentence index with normalized sentence text for matching.
    """
    sentences = split_sentences(text)
    sent_meta = []

    for idx, sent in enumerate(sentences):
        sent_meta.append({
            "idx": idx,
            "sentence": sent,
            "normalized": normalize_fn(sent),
        })

    return sent_meta


def retrieve_context_for_term(
    term: str,
    term_norm: str,
    sentence_index: list[dict],
    window: int = 1,
    max_hits: int = 3,
) -> str:
    """
    Returns compact context built from up to max_hits matching sentence windows.
    Matching is done on normalized forms.
    """
    if not term_norm:
        return ""

    hits = []
    pattern = re.compile(rf"\b{re.escape(term_norm)}\b")

    # exact normalized match first
    for row in sentence_index:
        if pattern.search(row["normalized"]):
            hits.append(row["idx"])
