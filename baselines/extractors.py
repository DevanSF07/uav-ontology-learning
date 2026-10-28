from __future__ import annotations

from typing import List, Dict
import re

from benchmark_config import (
    TOP_K,
    NGRAM_MIN,
    NGRAM_MAX,
    KEYBERT_MINILM_MODEL,
    KEYBERT_MPNET_MODEL,
    KEYBERT_USE_MMR,
    KEYBERT_DIVERSITY,
    YAKE_LANGUAGE,
    YAKE_DEDUP_LIM,
    YAKE_WINDOW_SIZE,
)


def _dedupe_ranked(rows: List[Dict], normalize_fn):
    seen = set()
    out = []
    for row in rows:
        norm = normalize_fn(row["term"])
        if not norm or norm in seen:
            continue
        seen.add(norm)
        row["normalized_term"] = norm
        out.append(row)
    return out

def _load_scispacy_lg():
    import spacy
    return spacy.load("en_core_sci_lg")
def extract_with_scispacy_pos_patterns_lg(text: str, normalize_fn, top_k: int = TOP_K):
    """
    Extract candidate technical terms using scispaCy en_core_sci_lg
    with simple POS patterns:
    (ADJ|NOUN|PROPN)+ ending in NOUN/PROPN
    """
    nlp = _load_scispacy_lg()
    doc = nlp(text)

    rows = []
    current = []
    valid_pos = {"ADJ", "NOUN", "PROPN"}

    for token in doc:
        if token.pos_ in valid_pos and not token.is_stop and token.is_alpha:
            current.append(token.text)
        else:
            if current:
                phrase = " ".join(current).strip()
                if len(phrase) > 2:
                    rows.append({
                        "term": phrase,
                        "score": 1.0,
                        "method": "scispacy_pos_patterns_lg"
                    })
            current = []

    if current:
        phrase = " ".join(current).strip()
        if len(phrase) > 2:
            rows.append({
                "term": phrase,
                "score": 1.0,
                "method": "scispacy_pos_patterns_lg"
            })

    rows = _dedupe_ranked(rows, normalize_fn)
    return rows[:top_k]



def _load_spacy_model():
    import spacy
    return spacy.load("en_core_web_sm")


def extract_acronym_terms_regex(text: str, normalize_fn):
    """
    Extract acronym-like terms from text using regex.

    Examples:
    UAV, UAS, ISR, GCS, LOS, BLOS, MALE, HALE, EO/IR
    """
    pattern = r"\b[A-Z][A-Z0-9/]{1,}\b"
    matches = re.findall(pattern, text)

    rows = []
    seen = set()

    for term in matches:
        norm = normalize_fn(term)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        rows.append({
            "term": term,
            "score": 1.0,
            "method": "regex_acronym",
            "normalized_term": norm,
        })

    return rows

def extract_acronym_expansions_regex(text: str, normalize_fn, top_k: int = TOP_K):
    """
    Find phrase expansions whose initials match acronym terms found in the corpus.

    Example:
    ground control station -> GCS
    line of sight -> LOS
    medium altitude long endurance -> MALE
    """
    acronym_rows = extract_acronym_terms_regex(text, normalize_fn)
    acronym_set = set()

    for row in acronym_rows:
        # keep uppercase canonical form for matching initials
        acronym_set.add(row["term"].replace("/", "").upper())

    # use your existing POS-pattern terms as expansion candidates
    phrase_rows = extract_with_spacy_pos_patterns(text, normalize_fn, top_k=top_k)

    matched_rows = []
    seen = set()

    for row in phrase_rows:
        phrase = row["term"]
        tokens = re.findall(r"[A-Za-z]+", phrase)

        if len(tokens) < 2:
            continue

        initials = "".join(token[0].upper() for token in tokens if token)

        if initials in acronym_set:
            norm = row["normalized_term"]
            if norm and norm not in seen:
                seen.add(norm)
                matched_rows.append({
                    "term": phrase,
                    "score": 1.0,
                    "method": "regex_acronym_expansion",
                    "normalized_term": norm,
                })

    return matched_rows

def _unique_by_normalized(rows):
    """
    Keep only one row per normalized term.
    """
    seen = set()
    out = []

    for row in rows:
        norm = row.get("normalized_term", "").strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(row)

    return out

def extract_with_ensemble_yake_spacy_regex_acronym_union(text: str, normalize_fn, top_k: int = TOP_K):
    """
    Union ensemble of:
    - YAKE
    - spaCy POS patterns
    - regex acronym terms
    - regex acronym-expansion matches

    Keeps all unique normalized terms.
    """

    yake_rows = extract_with_yake(text, normalize_fn, top_k=top_k)
    spacy_rows = extract_with_spacy_pos_patterns(text, normalize_fn, top_k=top_k)
    acronym_rows = extract_acronym_terms_regex(text, normalize_fn)
    expansion_rows = extract_acronym_expansions_regex(text, normalize_fn, top_k=top_k)

    combined = []

    for row in yake_rows:
        combined.append({
            "term": row["term"],
            "score": row.get("score", 1.0),
            "method": "ensemble_yake_spacy_regex_acronym_union",
            "normalized_term": row["normalized_term"],
        })

    for row in spacy_rows:
        combined.append({
            "term": row["term"],
            "score": row.get("score", 1.0),
            "method": "ensemble_yake_spacy_regex_acronym_union",
            "normalized_term": row["normalized_term"],
        })

    for row in acronym_rows:
        combined.append({
            "term": row["term"],
            "score": row.get("score", 1.0),
            "method": "ensemble_yake_spacy_regex_acronym_union",
            "normalized_term": row["normalized_term"],
        })

    for row in expansion_rows:
        combined.append({
            "term": row["term"],
            "score": row.get("score", 1.0),
            "method": "ensemble_yake_spacy_regex_acronym_union",
            "normalized_term": row["normalized_term"],
        })

    combined = _unique_by_normalized(combined)
    combined.sort(key=lambda x: x["normalized_term"])

    return combined



def extract_with_spacy_noun_chunks(text: str, normalize_fn, top_k: int = TOP_K) -> List[Dict]:
    """
    Extract candidate terms using spaCy noun chunks.
    These are parser-derived base noun phrases.
    """
    nlp = _load_spacy_model()
    doc = nlp(text)

    rows = []
    for chunk in doc.noun_chunks:
        term = chunk.text.strip()
        if term:
            rows.append({
                "term": term,
                "score": 1.0,
                "method": "spacy_noun_chunks"
            })

    rows = _dedupe_ranked(rows, normalize_fn)
    return rows[:top_k]

def extract_with_spacy_pos_patterns(text: str, normalize_fn, top_k: int = TOP_K) -> List[Dict]:
    """
    Extract candidate terms using simple POS patterns:
    (ADJ|NOUN|PROPN)+ ending in NOUN/PROPN
    """
    nlp = _load_spacy_model()
    doc = nlp(text)

    rows = []
    current = []

    valid_pos = {"ADJ", "NOUN", "PROPN"}

    for token in doc:
        if token.pos_ in valid_pos and not token.is_stop and token.is_alpha:
            current.append(token.text)
        else:
            if current:
                phrase = " ".join(current).strip()
                if len(phrase) > 2:
                    rows.append({
                        "term": phrase,
                        "score": 1.0,
                        "method": "spacy_pos_patterns"
                    })
            current = []

    if current:
        phrase = " ".join(current).strip()
        if len(phrase) > 2:
            rows.append({
                "term": phrase,
                "score": 1.0,
                "method": "spacy_pos_patterns"
            })

    rows = _dedupe_ranked(rows, normalize_fn)
    return rows[:top_k]



def extract_with_yake(text: str, normalize_fn, top_k: int = TOP_K) -> List[Dict]:
    import yake

    kw_extractor = yake.KeywordExtractor(
        lan=YAKE_LANGUAGE,
        n=NGRAM_MAX,
        dedupLim=YAKE_DEDUP_LIM,
        windowsSize=YAKE_WINDOW_SIZE,
        top=top_k,
    )

    ranked = kw_extractor.extract_keywords(text)

    rows = [
        {"term": term, "score": float(score), "method": "yake"}
        for term, score in ranked
    ]
    return _dedupe_ranked(rows, normalize_fn)


def extract_with_rake(text: str, normalize_fn, top_k: int = TOP_K) -> List[Dict]:
    import nltk
    from rake_nltk import Rake

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)

    r = Rake(
        min_length=NGRAM_MIN,
        max_length=NGRAM_MAX,
        include_repeated_phrases=False
    )

    r.extract_keywords_from_text(text)
    ranked = r.get_ranked_phrases_with_scores()[:top_k]

    rows = [
        {"term": term, "score": float(score), "method": "rake"}
        for score, term in ranked
    ]
    return _dedupe_ranked(rows, normalize_fn)

def _unique_by_normalized(rows):
    """
    Keep only one row per normalized term.
    """
    seen = set()
    out = []

    for row in rows:
        norm = row.get("normalized_term", "").strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(row)

    return out

def extract_with_ensemble_yake_spacy_union(text: str, normalize_fn, top_k: int = TOP_K):
    """
    Simple union ensemble:
    - run YAKE
    - run spaCy POS patterns
    - normalize terms
    - take all unique terms from both methods
    - return the union as the final prediction set

    Note:
    This intentionally keeps ALL unique terms from both methods,
    so it may return more than top_k terms.
    """

    # Run both methods
    yake_rows = extract_with_yake(text, normalize_fn, top_k=top_k)
    spacy_rows = extract_with_spacy_pos_patterns(text, normalize_fn, top_k=top_k)

    # Merge all rows
    combined = []

    for row in yake_rows:
        combined.append({
            "term": row["term"],
            "score": row.get("score", 1.0),
            "method": "ensemble_yake_spacy_union",
            "normalized_term": row["normalized_term"],
        })

    for row in spacy_rows:
        combined.append({
            "term": row["term"],
            "score": row.get("score", 1.0),
            "method": "ensemble_yake_spacy_union",
            "normalized_term": row["normalized_term"],
        })

    # Deduplicate by normalized term
    combined = _unique_by_normalized(combined)

    # Optional: sort alphabetically for stable output
    combined.sort(key=lambda x: x["normalized_term"])

    return combined

def _acronym_core(acronym: str) -> str:
    """
    Normalize acronym for matching:
    remove slashes and non-alnum, uppercase result.
    """
    return re.sub(r"[^A-Za-z0-9]", "", acronym).upper()


def _initials_from_phrase_all_words(phrase: str) -> str:
    """
    Build initials from all alphabetic words in the phrase.
    Example:
      'Autonomy levels for unmanned systems' -> 'ALFUS'
      'Line of sight' -> 'LOS'
      'Ground control station' -> 'GCS'
    """
    tokens = re.findall(r"[A-Za-z]+", phrase)
    return "".join(tok[0].upper() for tok in tokens if tok)


def _clean_long_form(phrase: str) -> str:
    """
    Clean a long-form phrase captured by regex.
    """
    phrase = phrase.strip(" ,;:-")
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase

def extract_longform_acronym_pairs_regex(text: str, normalize_fn):
    """
    Extract explicit Long Form (ACRONYM) and ACRONYM (Long Form) pairs.

    Example:
      'Autonomy levels for unmanned systems (ALFUS)'
      -> extracts:
         - 'Autonomy levels for unmanned systems'
         - 'ALFUS'

    Returns rows with both forms when initials match.
    """

    rows = []
    seen = set()

    # ------------------------------------------------------------
    # Pattern 1: Long Form (ACRONYM)
    # Example: Autonomy levels for unmanned systems (ALFUS)
    # ------------------------------------------------------------
    pattern_long_first = re.compile(
        r"([A-Za-z][A-Za-z0-9\-/,& ]{3,120}?)\s*\(([A-Z][A-Z0-9/]{1,15})\)"
    )

    for match in pattern_long_first.finditer(text):
        long_form = _clean_long_form(match.group(1))
        acronym = match.group(2).strip()

        # keep phrase lengths reasonable
        word_count = len(re.findall(r"[A-Za-z]+", long_form))
        if word_count < 2 or word_count > 10:
            continue

        initials = _initials_from_phrase_all_words(long_form)
        acronym_core = _acronym_core(acronym)

        # accept only if initials line up with acronym
        if initials != acronym_core:
            continue

        # add long form
        long_norm = normalize_fn(long_form)
        if long_norm and long_norm not in seen:
            seen.add(long_norm)
            rows.append({
                "term": long_form,
                "score": 1.0,
                "method": "regex_longform_acronym_pattern",
                "normalized_term": long_norm,
            })

        # add acronym
        acr_norm = normalize_fn(acronym)
        if acr_norm and acr_norm not in seen:
            seen.add(acr_norm)
            rows.append({
                "term": acronym,
                "score": 1.0,
                "method": "regex_longform_acronym_pattern",
                "normalized_term": acr_norm,
            })

    # ------------------------------------------------------------
    # Pattern 2: ACRONYM (Long Form)
    # Example: ALFUS (Autonomy levels for unmanned systems)
    # ------------------------------------------------------------
    pattern_acr_first = re.compile(
        r"\b([A-Z][A-Z0-9/]{1,15})\s*\(([A-Za-z][A-Za-z0-9\-/,& ]{3,120}?)\)"
    )

    for match in pattern_acr_first.finditer(text):
        acronym = match.group(1).strip()
        long_form = _clean_long_form(match.group(2))

        word_count = len(re.findall(r"[A-Za-z]+", long_form))
        if word_count < 2 or word_count > 10:
            continue

        initials = _initials_from_phrase_all_words(long_form)
        acronym_core = _acronym_core(acronym)

        if initials != acronym_core:
            continue

        # add long form
        long_norm = normalize_fn(long_form)
        if long_norm and long_norm not in seen:
            seen.add(long_norm)
            rows.append({
                "term": long_form,
                "score": 1.0,
                "method": "regex_longform_acronym_pattern",
                "normalized_term": long_norm,
            })

        # add acronym
        acr_norm = normalize_fn(acronym)
        if acr_norm and acr_norm not in seen:
            seen.add(acr_norm)
            rows.append({
                "term": acronym,
                "score": 1.0,
                "method": "regex_longform_acronym_pattern",
                "normalized_term": acr_norm,
            })

    return rows

def extract_with_ensemble_yake_spacy_regex_acronym_pattern_union(text: str, normalize_fn, top_k: int |None  = TOP_K):
    """
    Union ensemble of:
    - YAKE
    - spaCy POS patterns
    - regex acronym terms
    - regex acronym-expansion matches
    - explicit Long Form (ACRONYM) pattern matches

    Keeps all unique normalized terms.
    """

    yake_rows = extract_with_yake(text, normalize_fn, top_k=top_k)
    spacy_rows = extract_with_spacy_pos_patterns(text, normalize_fn, top_k=top_k)
    acronym_rows = extract_acronym_terms_regex(text, normalize_fn)
    expansion_rows = extract_acronym_expansions_regex(text, normalize_fn, top_k=top_k)
    pattern_rows = extract_longform_acronym_pairs_regex(text, normalize_fn)

    combined = []

    for row in yake_rows:
        combined.append({
            "term": row["term"],
            "score": row.get("score", 1.0),
            "method": "ensemble_yake_spacy_regex_acronym_pattern_union",
            "normalized_term": row["normalized_term"],
        })

    for row in spacy_rows:
        combined.append({
            "term": row["term"],
            "score": row.get("score", 1.0),
            "method": "ensemble_yake_spacy_regex_acronym_pattern_union",
            "normalized_term": row["normalized_term"],
        })

    for row in acronym_rows:
        combined.append({
            "term": row["term"],
            "score": row.get("score", 1.0),
            "method": "ensemble_yake_spacy_regex_acronym_pattern_union",
            "normalized_term": row["normalized_term"],
        })

    for row in expansion_rows:
        combined.append({
            "term": row["term"],
            "score": row.get("score", 1.0),
            "method": "ensemble_yake_spacy_regex_acronym_pattern_union",
            "normalized_term": row["normalized_term"],
        })

    for row in pattern_rows:
        combined.append({
            "term": row["term"],
            "score": row.get("score", 1.0),
            "method": "ensemble_yake_spacy_regex_acronym_pattern_union",
            "normalized_term": row["normalized_term"],
        })

    combined = _unique_by_normalized(combined)
    combined.sort(key=lambda x: x["normalized_term"])

    return combined

def extract_with_keybert(text: str, model_name: str, method_name: str, normalize_fn, top_k: int = TOP_K) -> List[Dict]:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(model_name)
    kw_model = KeyBERT(model=embedding_model)

    ranked = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(NGRAM_MIN, NGRAM_MAX),
        stop_words="english",
        use_mmr=KEYBERT_USE_MMR,
        diversity=KEYBERT_DIVERSITY,
        top_n=top_k,
    )

    rows = [
        {"term": term, "score": float(score), "method": method_name}
        for term, score in ranked
    ]
    return _dedupe_ranked(rows, normalize_fn)


def run_method(method_name: str, text: str, normalize_fn, top_k: int | None = TOP_K) -> List[Dict]:
    method_name = method_name.lower().strip()

    if method_name == "yake":
        return extract_with_yake(text, normalize_fn, top_k=top_k)

    if method_name == "rake":
        return extract_with_rake(text, normalize_fn, top_k=top_k)

    if method_name == "keybert_minilm":
        return extract_with_keybert(
            text=text,
            model_name=KEYBERT_MINILM_MODEL,
            method_name="keybert_minilm",
            normalize_fn=normalize_fn,
            top_k=top_k,
        )

    if method_name == "keybert_mpnet":
        return extract_with_keybert(
            text=text,
            model_name=KEYBERT_MPNET_MODEL,
            method_name="keybert_mpnet",
            normalize_fn=normalize_fn,
            top_k=top_k,
        )
    if method_name == "spacy_noun_chunks":
        return extract_with_spacy_noun_chunks(text, normalize_fn, top_k=top_k)

    if method_name == "spacy_pos_patterns":
        return extract_with_spacy_pos_patterns(text, normalize_fn, top_k=top_k)
    
    if method_name == "scispacy_pos_patterns_lg":
        return extract_with_scispacy_pos_patterns_lg(text, normalize_fn, top_k=top_k)
    
    if method_name == "ensemble_yake_spacy_union":
        return extract_with_ensemble_yake_spacy_union(text, normalize_fn, top_k=top_k)
    
    
    if method_name == "ensemble_yake_spacy_regex_acronym_union":
        return extract_with_ensemble_yake_spacy_regex_acronym_union(text, normalize_fn, top_k=top_k)
    
    if method_name == "ensemble_yake_spacy_regex_acronym_pattern_union":
        return extract_with_ensemble_yake_spacy_regex_acronym_pattern_union(text, normalize_fn, top_k=top_k)



    raise ValueError(f"Unknown method: {method_name}")