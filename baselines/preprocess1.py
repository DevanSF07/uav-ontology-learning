import re
import math
import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# Optional NLTK support for POS-based term candidates
NLTK_AVAILABLE = True
try:
    import nltk
    from nltk import word_tokenize, sent_tokenize, pos_tag
except Exception:
    NLTK_AVAILABLE = False

STOPWORDS = {
    'a','an','the','and','or','but','if','while','is','are','was','were','be','been','being',
    'to','of','in','on','for','with','by','as','at','from','into','through','during','including',
    'until','against','among','throughout','despite','towards','upon','about','above','below','up',
    'down','out','off','over','under','again','further','then','once','here','there','all','any',
    'both','each','few','more','most','other','some','such','no','nor','not','only','own','same',
    'so','than','too','very','can','will','just','should','now','this','that','these','those','it',
    'its','their','his','her','our','your','you','we','they','he','she','them','us','i','me','my',
    'mine','ours','yours','also','may','might','must','would','could','shall','do','does','did',
    'have','has','had'
}


def read_input(path: Path, start_page=160, end_page=179) -> str:
    if path.suffix.lower() == '.pdf':
        if PdfReader is None:
            raise RuntimeError("PyPDF2 not available to read PDF files.")

        reader = PdfReader(str(path))
        pages = []

        total_pages = len(reader.pages)

        if start_page is None:
            start_page = 1
        if end_page is None:
            end_page = 50

        # Convert to 0-based indexing
        start_idx = max(start_page - 1, 0)
        end_idx = min(end_page, total_pages)

        for i in range(start_idx, end_idx):
            try:
                pages.append(reader.pages[i].extract_text() or "")
            except Exception:
                pages.append("")

        return "\n".join(pages)

    return path.read_text(encoding='utf-8', errors='ignore')



def clean_text(text: str) -> str:
    text = text.replace('\u00ad', '')  # remove soft hyphen
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bPage\s+\d+\b', ' ', text, flags=re.I)
    return text.strip()


def split_into_docs(text: str):
    """
    Treat paragraphs as mini-documents for IDF.
    Fallback: group sentences into blocks if paragraphs are not available.
    """
    paras = [p.strip() for p in re.split(r'\n\s*\n|(?<=[.!?])\s{2,}', text) if p.strip()]

    if len(paras) < 3:
        sents = re.split(r'(?<=[.!?])\s+', text)
        paras = [
            ' '.join(sents[i:i+5]).strip()
            for i in range(0, len(sents), 5)
            if ' '.join(sents[i:i+5]).strip()
        ]

    return paras


def regex_sent_tokenize(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def regex_word_tokenize(text: str):
    return re.findall(r"[A-Za-z][A-Za-z0-9\-_\/]*", text)


def preprocess_for_tokens(text: str):
    tokens = [t.lower() for t in regex_word_tokenize(text)]
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def extract_acronyms(text: str):
    acronyms = Counter(re.findall(r'\b[A-Z][A-Z0-9]{1,}\b', text))
    return acronyms


def extract_ngram_candidates(text: str, min_n=2, max_n=4):
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

            if not any(re.search(r'[A-Za-z]', tok) for tok in span):
                continue

            phrase = ' '.join(span)
            cands[phrase] += 1

    return cands


def extract_pos_candidates(text: str):
    """
    Candidate patterns roughly matching technical terms:
    (ADJ|NOUN|PROPN)+ ending in NOUN/PROPN
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
                ok = tag.startswith('JJ') or tag.startswith('NN')
                if ok:
                    curr.append(tok)
                else:
                    if len(curr) >= 1:
                        if curr[-1][0].isalnum():
                            phrase = ' '.join(curr)
                            if len(phrase) > 2:
                                cands[phrase] += 1
                    curr = []

            if len(curr) >= 1 and curr[-1][0].isalnum():
                phrase = ' '.join(curr)
                if len(phrase) > 2:
                    cands[phrase] += 1

        # filter tiny single generic words unless acronym-like
        filtered = Counter()
        for k, v in cands.items():
            toks = k.split()
            if len(toks) == 1 and len(k) < 4 and not k.isupper():
                continue
            filtered[k] = v

        return filtered

    except LookupError:
        # NLTK installed but tokenizer/POS data may be missing
        return Counter()


def c_value_like(candidate_counts):
    """
    Simplified nested multiword term score inspired by C-value.
    """
    phrases = list(candidate_counts.keys())
    contains = defaultdict(list)

    for p in phrases:
        lp = p.lower()
        for q in phrases:
            if p == q:
                continue
            lq = q.lower()
            if (' ' in q) and (lp in lq) and (len(lq.split()) > len(lp.split())):
                contains[p].append(q)

    scores = {}
    for p, freq in candidate_counts.items():
        length = len(p.split())
        if length == 0:
            continue

        nested_in = contains[p]
        if nested_in:
            avg_super = sum(candidate_counts[q] for q in nested_in) / len(nested_in)
            base = max(freq - avg_super, 0.0)
        else:
            base = float(freq)

        scores[p] = math.log2(1 + length) * base

    return scores


def tf_idf_phrase_scores(candidates, docs):
    """
    Phrase-level IDF over mini-documents (paragraphs/segments).
    """
    N = max(len(docs), 1)
    low_docs = [d.lower() for d in docs]

    scores = {}
    for phrase, tf in candidates.items():
        lp = phrase.lower()
        df = sum(1 for d in low_docs if lp in d)
        idf = math.log((1 + N) / (1 + df)) + 1.0
        scores[phrase] = tf * idf

    return scores


def normalize_phrase(p):
    p = p.strip(' -_/')
    p = re.sub(r'\s+', ' ', p)
    return p


def rank_terms(text, top_k=100):
    docs = split_into_docs(text)

    ngram_cands = extract_ngram_candidates(text)
    pos_cands = extract_pos_candidates(text)

    combined = Counter()
    combined.update(ngram_cands)

    # boost POS-supported phrases
    for k, v in pos_cands.items():
        combined[k] += v * 2

    # Filter overly generic / short candidates
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
    tfidf = tf_idf_phrase_scores(filtered, docs)
    acronyms = extract_acronyms(text)

    rows = []
    for phrase, freq in filtered.items():
        toks = phrase.split()

        # acronym bonus if initials match a chapter acronym
        acr_boost = 0.0
        initials = ''.join(t[0].upper() for t in toks if t and t[0].isalpha())
        if initials in acronyms:
            acr_boost = 1.5

        score = (
            0.45 * cvals.get(phrase, 0.0)
            + 0.45 * tfidf.get(phrase, 0.0)
            + 0.10 * freq
            + acr_boost
        )

        rows.append({
            'term': phrase,
            'frequency': freq,
            'c_value_like': round(cvals.get(phrase, 0.0), 4),
            'tfidf_like': round(tfidf.get(phrase, 0.0), 4),
            'acronym_boost': acr_boost,
            'score': round(score, 4),
        })

    rows.sort(key=lambda x: (-x['score'], -x['frequency'], x['term'].lower()))
    return rows[:top_k], docs, acronyms


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_acronyms(path, acronyms: Counter):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['acronym', 'count'])
        for a, c in acronyms.most_common():
            w.writerow([a, c])


def main():
    # -------- SETTINGS --------
    input_file = "/home/faujdar/Project1/Handbook_of_Unmanned_Aerial_Vehicles.pdf"
    top_k = 100
    outdir = None
    # --------------------------

    in_path = Path(input_file)
    outdir = Path(outdir) if outdir else in_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    raw_text = read_input(in_path)
    clean = clean_text(raw_text)

    term_rows, docs, acronyms = rank_terms(clean, top_k=top_k)

    stem = in_path.stem
    text_out = outdir / f'{stem}.cleaned.txt'
    terms_out = outdir / f'{stem}.terms.csv'
    acr_out = outdir / f'{stem}.acronyms.csv'
    docs_out = outdir / f'{stem}.segments.txt'

    text_out.write_text(clean, encoding='utf-8')
    docs_out.write_text('\n\n---SEGMENT---\n\n'.join(docs), encoding='utf-8')
    write_csv(
        terms_out,
        term_rows,
        ['term', 'frequency', 'c_value_like', 'tfidf_like', 'acronym_boost', 'score']
    )
    write_acronyms(acr_out, acronyms)

    print(f'Wrote cleaned text to: {text_out}')
    print(f'Wrote term ranking to: {terms_out}')
    print(f'Wrote acronym counts to: {acr_out}')
    print('Top 20 terms:')
    for r in term_rows[:20]:
        print(f"- {r['term']} (score={r['score']}, freq={r['frequency']})")


if __name__ == '__main__':
    main()