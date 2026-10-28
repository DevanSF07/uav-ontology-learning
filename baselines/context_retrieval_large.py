from __future__ import annotations

import re
from typing import List


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter for technical prose."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def build_sentence_index(text: str, normalize_fn):
    sentences = split_sentences(text)
    sent_meta = []

    for idx, sent in enumerate(sentences):
        sent_meta.append(
            {
                "idx": idx,
                "sentence": sent,
                "normalized": normalize_fn(sent),
            }
        )

    return sent_meta


def retrieve_large_context_for_term(
    term: str,
    term_norm: str,
    sentence_index: list[dict],
    max_sentences: int = 10,
    max_chars: int = 5000,
) -> tuple[str, int]:
    """
    Retrieve up to max_sentences directly matched sentences for a term.

    Returns:
      - joined context string
      - number of sentences used

    Matching is done on normalized sentence text.
    """
    if not term_norm or not isinstance(term_norm, str):
        return "", 0

    hits = []
    seen = set()

    pattern = re.compile(rf"\b{re.escape(term_norm)}\b")

    # exact normalized match first
    for row in sentence_index:
        if pattern.search(row["normalized"]):
            sentence = row["sentence"].strip()
            if sentence and sentence not in seen:
                seen.add(sentence)
                hits.append(sentence)

        if len(hits) >= max_sentences:
            break

    # fallback substring match if needed
    if len(hits) < max_sentences:
        for row in sentence_index:
            if term_norm in row["normalized"]:
                sentence = row["sentence"].strip()
                if sentence and sentence not in seen:
                    seen.add(sentence)
                    hits.append(sentence)

            if len(hits) >= max_sentences:
                break

    # enforce max character budget
    kept = []
    total_chars = 0

    for sentence in hits:
        addition = len(sentence) + 5  # separator safety margin
        if kept and total_chars + addition > max_chars:
            break

        kept.append(sentence)
        total_chars += addition

    context_text = "\n---\n".join(kept)
    return context_text, len(kept)
