# src/candidates.py

import re
from collections import Counter

from config import STOPWORDS
from preprocess import regex_word_tokenize

# Optional NLTK support
NLTK_AVAILABLE = True
try:
    from nltk import word_tokenize, sent_tokenize, pos_tag
except Exception:
    NLTK_AVAILABLE = False


def extract_acronyms(text: str):
    """
    Extract acronym-like tokens such as UAV, MTOW, LOS, GCS, MALE.
    """
    return Counter(re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", text))


def extract_ngram_candidates(text: str, min_n=2, max_n=4):
    """
    Extract 2- to 4-word n-gram candidates from text.
    Filters out phrases with bad stopword structure.
    """
    tokens = regex_word_tokenize(text)
    lowered = [t.lower() for t in tokens]
    cands = Counter()

    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            span = tokens[i:i+n]
            low = lowered[i:i+n]

            if low[0] in STOPWORDS or low[-1] in STOPWORDS:
                continue

            if sum(tok in STOPWORDS for tok in low) > 1:
                continue

            if not any(re.search(r"[A-Za-z]", tok) for tok in span):
                continue

            phrase = " ".join(span)
            cands[phrase] += 1

    return cands


def extract_pos_candidates(text: str):
    """
    Optional POS-based candidate extraction using NLTK.
    Looks for adjective/noun sequences likely to be technical terms.
    """
    if not NLTK_AVAILABLE:
        return Counter()

    try:
        sents = sent_tokenize(text)
        cands = Counter()

        for sent in sents:
            toks = word_tokenize(sent)
            tagged = pos_tag(toks)
            curr = []

            for tok, tag in tagged:
                ok = tag.startswith("JJ") or tag.startswith("NN")

                if ok:
                    curr.append(tok)
                else:
                    if len(curr) >= 1 and curr[-1][0].isalnum():
                        phrase = " ".join(curr)
                        if len(phrase) > 2:
                            cands[phrase] += 1
                    curr = []

            if len(curr) >= 1 and curr[-1][0].isalnum():
                phrase = " ".join(curr)
                if len(phrase) > 2:
                    cands[phrase] += 1

        filtered = Counter()
        for k, v in cands.items():
            toks = k.split()
            if len(toks) == 1 and len(k) < 4 and not k.isupper():
                continue
            filtered[k] = v

        return filtered

    except LookupError:
        # NLTK installed but punkt/tagger resources missing
        return Counter()