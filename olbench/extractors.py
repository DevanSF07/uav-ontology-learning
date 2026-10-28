from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .utils import canonical, ollama_json, stable_id, text_sentences


TERM_COLS = [
    "term_id",
    "raw_term",
    "term_canonical",
    "score",
    "rank",
    "extractor",
    "chunk_id",
    "source_sentence",
    "context_text",
]

COMPOUND_SPLIT_PATTERN = re.compile(r"\s+(?:and|or)\s+|[\/&]", re.IGNORECASE)


def _expand_compound_term(term: str) -> list[str]:
    cleaned = str(term).strip()
    if not cleaned:
        return []

    if not COMPOUND_SPLIT_PATTERN.search(cleaned):
        return [cleaned]

    parts = [part.strip() for part in COMPOUND_SPLIT_PATTERN.split(cleaned) if part.strip()]
    if len(parts) < 2:
        return [cleaned]

    suffix_tokens = parts[-1].split()
    suffix = suffix_tokens[-1] if len(suffix_tokens) > 1 else ""

    expanded: list[str] = []
    for part in parts:
        part_tokens = part.split()
        if suffix and len(part_tokens) == 1 and not part.isupper():
            expanded.append(f"{part} {suffix}".strip())
        else:
            expanded.append(part)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in expanded:
        normalized = candidate.casefold()
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(candidate)

    return deduped or [cleaned]


def _rows(candidates, chunk, name):
    """Convert ranked candidate tuples into the standard extraction schema."""

    rows = []

    for rank, (term, score) in enumerate(candidates, 1):
        for expanded_term in _expand_compound_term(term):
            expanded_term = str(expanded_term).strip()

            if not expanded_term:
                continue

            sentence = next(
                (
                    sentence
                    for sentence in text_sentences(chunk.text)
                    if expanded_term.casefold() in sentence.casefold()
                ),
                chunk.text,
            )

            rows.append(
                {
                    "term_id": stable_id("term", canonical(expanded_term)),
                    "raw_term": expanded_term,
                    "term_canonical": canonical(expanded_term),
                    "score": float(score),
                    "rank": rank,
                    "extractor": name,
                    "chunk_id": chunk.chunk_id,
                    "source_sentence": sentence,
                    "context_text": chunk.text,
                }
            )

    return rows


def _candidate_ngrams(text, min_n=1, max_n=3):
    """Return all usable 1-to-3-gram candidates from text without top-k."""

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", str(text))
    candidates = []

    for ngram_size in range(min_n, max_n + 1):
        for index in range(len(tokens) - ngram_size + 1):
            words = tokens[index : index + ngram_size]
            lowered = [word.casefold() for word in words]

            if lowered[0] in ENGLISH_STOP_WORDS:
                continue

            if lowered[-1] in ENGLISH_STOP_WORDS:
                continue

            if all(word in ENGLISH_STOP_WORDS for word in lowered):
                continue

            candidates.append(" ".join(words))

    return candidates


def _fallback_phrases(text, top_k=None):
    """Return frequency-ranked fallback phrases with no limit by default."""

    counts = Counter(_candidate_ngrams(text, min_n=1, max_n=3))
    ranked = counts.most_common()

    if top_k is not None:
        ranked = ranked[:top_k]

    return [(phrase, float(count)) for phrase, count in ranked]


def noun_phrases(chunks, top_k=None, **_):
    """Extract every noun-phrase-like candidate matched by the rule pattern."""

    rows = []
    pattern = re.compile(
        r"\b"
        r"(?:[A-Z][A-Za-z0-9-]*|[a-z]+(?:al|ic|ive|ary|ory|ed|ing))?"
        r"(?:\s+)?"
        r"(?:[A-Za-z][A-Za-z0-9-]*\s+){0,3}"
        r"(?:system|subsystem|component|interface|flow|parameter|mission|"
        r"capability|activity|environment|vehicle|aircraft|sensor|controller|"
        r"module|unit|network|class)\b",
        re.IGNORECASE,
    )

    for chunk in chunks.itertuples():
        values = Counter(
            canonical(match.group())
            for match in pattern.finditer(str(chunk.text))
            if len(canonical(match.group())) > 2
        )

        ranked_values = values.most_common()

        if top_k is not None:
            ranked_values = ranked_values[:top_k]

        candidates = [
            (term, float(count))
            for term, count in ranked_values
        ]

        rows += _rows(candidates, chunk, "noun_phrases")

    return pd.DataFrame(rows, columns=TERM_COLS)


def tfidf(chunks, top_k=None, **_):
    """Extract every nonzero TF-IDF 1-to-3-gram; no top-k by default."""

    documents = chunks.text.fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=1,
        max_features=None,
    )

    matrix = vectorizer.fit_transform(documents)
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    rows = []

    for index, chunk in enumerate(chunks.itertuples()):
        sparse_row = matrix.getrow(index)
        candidate_indices = sparse_row.indices
        candidate_scores = sparse_row.data

        order = np.argsort(candidate_scores)[::-1]
        candidate_indices = candidate_indices[order]
        candidate_scores = candidate_scores[order]

        if top_k is not None:
            candidate_indices = candidate_indices[:top_k]
            candidate_scores = candidate_scores[:top_k]

        candidates = [
            (feature_names[feature_index], score)
            for feature_index, score in zip(
                candidate_indices,
                candidate_scores,
            )
            if score > 0
        ]

        rows += _rows(candidates, chunk, "tfidf")

    return pd.DataFrame(rows, columns=TERM_COLS)


def yake_extract(chunks, top_k=None, **_):
    """Extract all YAKE candidates available in each chunk by default."""

    try:
        import yake
    except ImportError as error:
        raise ImportError(
            "YAKE is not installed. Run: python -m pip install yake"
        ) from error

    rows = []
    max_ngram_size = 4

    for chunk in chunks.itertuples():
        # YAKE requires a numeric `top` argument. The internal value below is
        # the complete count of usable 1-to-6-gram candidates in this chunk,
        # not a user-defined selection cutoff.
        candidate_space_size = len(
            set(_candidate_ngrams(chunk.text, min_n=1, max_n=max_ngram_size))
        )

        if candidate_space_size == 0:
            continue

        internal_limit = (
            min(top_k, candidate_space_size)
            if top_k is not None
            else candidate_space_size
        )

        extractor = yake.KeywordExtractor(
            lan="en",
            n=max_ngram_size,
            top=internal_limit,
        )

        extracted = extractor.extract_keywords(str(chunk.text))
        candidates = [
            (term, 1.0 / (score + 1e-9))
            for term, score in extracted
        ]

        rows += _rows(candidates, chunk, "yake")

    return pd.DataFrame(rows, columns=TERM_COLS)


ACRONYM_PATTERNS = [
    re.compile(r"\b([A-Z][A-Za-z][A-Za-z\-/ ]{4,80}?)\s*\(([A-Z][A-Z0-9\-/]{1,12})\)"),
    re.compile(r"\b([A-Z][A-Z0-9\-/]{1,12})\s*\(([A-Z][A-Za-z][A-Za-z\-/ ]{4,80}?)\)"),
]

STANDALONE_ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z0-9/+-]{1,11}\b")


def _normalize_acronym_term(term: str) -> str:
    cleaned = str(term).strip()
    if len(cleaned) > 2 and cleaned.endswith("s") and cleaned[:-1].isupper():
        cleaned = cleaned[:-1]
    return cleaned


def acronym_extract(chunks, top_k=None, **_):
    """Extract acronym-like technical terms and parenthetical acronym forms."""

    rows = []

    for chunk in chunks.itertuples():
        text = str(chunk.text)
        candidates = []

        for pattern in ACRONYM_PATTERNS:
            for match in pattern.finditer(text):
                left, right = match.groups()
                acronym = left if left.isupper() else right
                acronym = _normalize_acronym_term(acronym)
                if 1 < len(acronym) <= 12 and not acronym.isdigit():
                    candidates.append(acronym)

        for token in STANDALONE_ACRONYM_PATTERN.findall(text):
            acronym = _normalize_acronym_term(token)
            if 1 < len(acronym) <= 12 and not acronym.isdigit():
                candidates.append(acronym)

        ranked = Counter(candidates).most_common(top_k)
        rows += _rows([(term, float(count)) for term, count in ranked], chunk, "acronyms")

    return pd.DataFrame(rows, columns=TERM_COLS)


def textrank(chunks, top_k=None, **_):
    """Rank and return every TextRank token candidate by default."""

    rows = []

    for chunk in chunks.itertuples():
        tokens = [
            token.casefold()
            for token in re.findall(
                r"[A-Za-z][A-Za-z0-9-]{2,}",
                str(chunk.text),
            )
            if token.casefold() not in ENGLISH_STOP_WORDS
        ]

        vocabulary = sorted(set(tokens))
        token_index = {
            token: index
            for index, token in enumerate(vocabulary)
        }
        adjacency = np.zeros(
            (len(vocabulary), len(vocabulary)),
            dtype=float,
        )

        for index, token in enumerate(tokens):
            for neighbor in tokens[index + 1 : index + 5]:
                if token == neighbor:
                    continue

                first = token_index[token]
                second = token_index[neighbor]
                adjacency[first, second] += 1
                adjacency[second, first] += 1

        if vocabulary:
            row_sums = adjacency.sum(axis=1)
            transition = np.divide(
                adjacency,
                row_sums[:, None],
                out=np.zeros_like(adjacency),
                where=row_sums[:, None] != 0,
            )
            page_rank = np.ones(len(vocabulary)) / len(vocabulary)

            for _ in range(40):
                page_rank = (
                    0.15 / len(vocabulary)
                    + 0.85 * transition.T.dot(page_rank)
                )

            scores = dict(zip(vocabulary, page_rank))
        else:
            scores = {}

        candidates = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        if top_k is not None:
            candidates = candidates[:top_k]

        rows += _rows(candidates, chunk, "textrank")

    return pd.DataFrame(rows, columns=TERM_COLS)


def keybert_extract(
    chunks,
    top_k=None,
    model="all-MiniLM-L6-v2",
    **_,
):
    """Extract the complete KeyBERT candidate set for each chunk by default."""

    try:
        from keybert import KeyBERT
    except ImportError as error:
        raise ImportError(
            "KeyBERT is not installed. Run: "
            "python -m pip install keybert sentence-transformers"
        ) from error

    keyword_model = KeyBERT(model=model)
    rows = []

    for chunk in chunks.itertuples():
        # KeyBERT requires numeric top_n. The candidate-space count is used so
        # no arbitrary top-k filtering is imposed by this framework.
        candidate_space_size = len(
            set(_candidate_ngrams(chunk.text, min_n=1, max_n=3))
        )

        if candidate_space_size == 0:
            continue

        internal_limit = (
            min(top_k, candidate_space_size)
            if top_k is not None
            else candidate_space_size
        )

        extracted = keyword_model.extract_keywords(
            str(chunk.text),
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=internal_limit,
        )

        rows += _rows(
            extracted,
            chunk,
            f"keybert_{model}",
        )

    return pd.DataFrame(rows, columns=TERM_COLS)


def ollama_extract(
    chunks,
    top_k=None,
    model="llama3.1:8b",
    allow_fallback=True,
    **_,
):
    """Extract candidates with Ollama; top_k is optional for compatibility."""

    rows = []

    for chunk in chunks.itertuples():
        try:
            limit_instruction = (
                f" Return at most {top_k} terms."
                if top_k is not None
                else " Return every relevant candidate term you identify."
            )

            prompt = (
                "Extract ontology candidate terms from the text. "
                'Return JSON only: {"terms":[{"term":"...","score":0.0}]}. '
                "Prefer systems, components, interfaces, flows, parameters, "
                "missions, capabilities, activities, environments, and "
                "classifications."
                f"{limit_instruction}\nText:\n{chunk.text}"
            )

            data = ollama_json(prompt, model)
            candidates = [
                (item["term"], float(item.get("score", 1.0)))
                for item in data.get("terms", [])
                if item.get("term")
            ]

            if top_k is not None:
                candidates = candidates[:top_k]

            rows += _rows(
                candidates,
                chunk,
                f"ollama_{model}",
            )
        except Exception:
            if not allow_fallback:
                raise

            rows += _rows(
                _fallback_phrases(chunk.text, top_k=top_k),
                chunk,
                "ollama_fallback",
            )

    return pd.DataFrame(rows, columns=TERM_COLS)


EXTRACTORS = {
    "yake": yake_extract,
    "acronyms": acronym_extract,
    "tfidf": tfidf,
    "noun_phrases": noun_phrases,
    "textrank": textrank,
    "keybert": keybert_extract,
    "ollama_extract": ollama_extract,
}
