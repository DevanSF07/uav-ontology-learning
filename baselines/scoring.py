# scoring.py

import re
import math
from collections import defaultdict


def normalize_phrase(p):
    p = p.strip(" -_/")
    p = re.sub(r"\s+", " ", p)
    return p


def c_value_like(candidate_counts):
    phrases = list(candidate_counts.keys())
    contains = defaultdict(list)

    for p in phrases:
        lp = p.lower()
        for q in phrases:
            if p == q:
                continue
            lq = q.lower()
            if (" " in q) and (lp in lq) and (len(lq.split()) > len(lp.split())):
                contains[p].append(q)

    scores = {}

    for p, freq in candidate_counts.items():
        length = len(p.split())
        nested_in = contains[p]
        base = float(freq)

        if nested_in:
            avg_super = sum(candidate_counts[q] for q in nested_in) / len(nested_in)
            base = max(freq - avg_super, 0.0)

        scores[p] = math.log2(1 + length) * base

    return scores


def tf_idf_phrase_scores(candidates, docs):
    n_docs = max(len(docs), 1)
    low_docs = [d.lower() for d in docs]

    scores = {}
    dfs = {}

    for phrase, tf in candidates.items():
        lp = phrase.lower()
        df = sum(1 for d in low_docs if lp in d)
        idf = math.log((1 + n_docs) / (1 + df)) + 1.0
        scores[phrase] = tf * idf
        dfs[phrase] = df

