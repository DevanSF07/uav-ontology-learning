# src/pipeline.py

from pathlib import Path
from collections import Counter

from config import STOPWORDS
from io_utils import read_input, write_csv, write_acronyms
from preprocess import clean_text, split_into_docs
from candidates import extract_ngram_candidates, extract_pos_candidates, extract_acronyms
from scoring import normalize_phrase, c_value_like, tf_idf_phrase_scores


def rank_terms(text: str, top_k=200):
    """
    Extract and rank candidate terms from text.
    """
    docs = split_into_docs(text)
    ngram_cands = extract_ngram_candidates(text)
    pos_cands = extract_pos_candidates(text)
    acronyms = extract_acronyms(text)

    combined = Counter()
    combined.update(ngram_cands)

    pos_supported = set()
    for k, v in pos_cands.items():
        combined[k] += v * 2  # boost if also found by POS patterning
        pos_supported.add(normalize_phrase(k))

    filtered = Counter()
    for phrase, cnt in combined.items():
        norm = normalize_phrase(phrase)
        low = norm.lower()
        toks = low.split()

        if len(norm) < 3:
            continue

        if all(t in STOPWORDS for t in toks):
            continue

        if len(toks) == 1 and low in STOPWORDS:
            continue

        if any(ch.isdigit() for ch in norm) and len(toks) == 1:
            continue

        filtered[norm] += cnt

    cvals = c_value_like(filtered)
    tfidf, dfs, n_docs = tf_idf_phrase_scores(filtered, docs)

    rows = []
    for phrase, freq in filtered.items():
        toks = phrase.split()
        initials = "".join(t[0].upper() for t in toks if t and t[0].isalpha())

        acr_boost = 1.5 if initials in acronyms else 0.0
        pos_boost = 1.0 if phrase in pos_supported else 0.0
        length = len(toks)

        score = (
            0.40 * cvals.get(phrase, 0.0)
            + 0.40 * tfidf.get(phrase, 0.0)
            + 0.10 * freq
            + 0.05 * acr_boost
            + 0.05 * pos_boost
        )

        rows.append({
            "term": phrase,
            "length_tokens": length,
            "frequency": freq,
            "segment_df": dfs.get(phrase, 0),
            "num_segments": n_docs,
            "c_value_like": round(cvals.get(phrase, 0.0), 4),
            "tfidf_like": round(tfidf.get(phrase, 0.0), 4),
            "acronym_boost": acr_boost,
            "pos_boost": pos_boost,
            "score": round(score, 4),
        })

    rows.sort(key=lambda x: (-x["score"], -x["frequency"], x["term"].lower()))
    return rows[:top_k], docs, acronyms


def process_one_file(input_file: str, top_k: int = 200, outdir: str | None = None):
    """
    Process one cleaned .mmd file and write outputs.
    """
    in_path = Path(input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    target_dir = Path(outdir) if outdir else in_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_text = read_input(in_path)
    clean = clean_text(raw_text)

    term_rows, docs, acronyms = rank_terms(clean, top_k=top_k)

    stem = in_path.stem
    text_out = target_dir / f"{stem}.cleaned.txt"
    terms_out = target_dir / f"{stem}.terms.csv"
    acr_out = target_dir / f"{stem}.acronyms.csv"
    docs_out = target_dir / f"{stem}.segments.txt"

    text_out.write_text(clean, encoding="utf-8")
    docs_out.write_text("\n\n---SEGMENT---\n\n".join(docs), encoding="utf-8")

    write_csv(
        terms_out,
        term_rows,
        [
            "term",
            "length_tokens",
            "frequency",
            "segment_df",
            "num_segments",
            "c_value_like",
            "tfidf_like",
            "acronym_boost",
            "pos_boost",
            "score",
        ],
    )

    write_acronyms(acr_out, acronyms)

    print(f"\nProcessed: {in_path.name}")
    print(f"Wrote cleaned text to: {text_out}")
    print(f"Wrote segments to: {docs_out}")
    print(f"Wrote term ranking to: {terms_out}")
    print(f"Wrote acronym counts to: {acr_out}")
    print("Top 20 terms:")
    for r in term_rows[:20]:
        print(
            f"- {r['term']} | score={r['score']} | freq={r['frequency']} | "
            f"tfidf={r['tfidf_like']} | cval={r['c_value_like']}"
        )
