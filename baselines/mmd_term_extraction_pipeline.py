
import re
import math
import csv
from pathlib import Path
from collections import Counter, defaultdict

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# Optional NLTK support for POS-based candidates
NLTK_AVAILABLE = True
try:
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

# =============== CONFIGURE THESE PATHS FOR YOUR REPO ===============
INPUT_FILES = [
    # Example:
    "Handbook_of_Unmanned_Aerial_Vehicles.mmd",
     "Handbook_of_Unmanned_Aerial_Vehicles_chapter6.mmd",
]
TOP_K = 200
OUTDIR = None   # None = save next to each input file; or set a folder path string
# ================================================================


def read_input(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == '.pdf':
        if PdfReader is None:
            raise RuntimeError('PyPDF2 not available. Install it in your virtual environment: pip install PyPDF2')
        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or '')
            except Exception:
                pages.append('')
        return '\n'.join(pages)
    # Treat .mmd, .md, .txt, and similar as plain text
    return path.read_text(encoding='utf-8', errors='ignore')


def clean_text(text: str) -> str:
    # Normalize Unicode-ish PDF/markdown artefacts
    text = text.replace('\u00ad', '')
    text = text.replace('\ufeff', '')

    # Remove fenced code blocks
    text = re.sub(r'```.*?```', ' ', text, flags=re.S)

    # Markdown headings / bullets / blockquotes
    text = re.sub(r'(?m)^\s{0,3}#{1,6}\s*', '', text)
    text = re.sub(r'(?m)^\s{0,3}[-*+]\s+', '', text)
    text = re.sub(r'(?m)^\s*>\s*', '', text)

    # Convert markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)

    # Remove raw URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove HTML tags if present
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove markdown tables separators and pipes but keep content
    text = text.replace('|', ' ')
    text = re.sub(r'(?m)^\s*[-:]{3,}\s*$', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_docs(text: str):
    # Use markdown-style section markers if they survived, otherwise sentence groups.
    paras = [p.strip() for p in re.split(r'\n\s*\n|(?<=[.!?])\s{2,}', text) if p.strip()]
    if len(paras) < 3:
        sents = re.split(r'(?<=[.!?])\s+', text)
        paras = [' '.join(sents[i:i+5]).strip() for i in range(0, len(sents), 5) if ' '.join(sents[i:i+5]).strip()]
    return paras


def regex_word_tokenize(text: str):
    # Keep things like UAV, MTOW, sense-and-avoid, C-band, EO/IR
    return re.findall(r"[A-Za-z][A-Za-z0-9\-_/]*", text)


def extract_acronyms(text: str):
    return Counter(re.findall(r'\b[A-Z][A-Z0-9]{1,}\b', text))


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
                    if len(curr) >= 1 and curr[-1][0].isalnum():
                        phrase = ' '.join(curr)
                        if len(phrase) > 2:
                            cands[phrase] += 1
                    curr = []
            if len(curr) >= 1 and curr[-1][0].isalnum():
                phrase = ' '.join(curr)
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
        # Missing punkt/tagger resources; fallback with no POS candidates
        return Counter()


def normalize_phrase(p):
    p = p.strip(' -_/')
    p = re.sub(r'\s+', ' ', p)
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
            if (' ' in q) and (lp in lq) and (len(lq.split()) > len(lp.split())):
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
    return scores, dfs, n_docs


def rank_terms(text: str, top_k=200):
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
        initials = ''.join(t[0].upper() for t in toks if t and t[0].isalpha())
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
            'term': phrase,
            'length_tokens': length,
            'frequency': freq,
            'segment_df': dfs.get(phrase, 0),
            'num_segments': n_docs,
            'c_value_like': round(cvals.get(phrase, 0.0), 4),
            'tfidf_like': round(tfidf.get(phrase, 0.0), 4),
            'acronym_boost': acr_boost,
            'pos_boost': pos_boost,
            'score': round(score, 4),
        })
    rows.sort(key=lambda x: (-x['score'], -x['frequency'], x['term'].lower()))
    return rows[:top_k], docs, acronyms


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_acronyms(path, acronyms: Counter):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['acronym', 'count'])
        for a, c in acronyms.most_common():
            writer.writerow([a, c])


def process_one_file(input_file: str, top_k: int = 200, outdir: str | None = None):
    in_path = Path(input_file)
    if not in_path.exists():
        raise FileNotFoundError(f'Input file not found: {in_path}')

    target_dir = Path(outdir) if outdir else in_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_text = read_input(in_path)
    clean = clean_text(raw_text)
    term_rows, docs, acronyms = rank_terms(clean, top_k=top_k)

    stem = in_path.stem
    text_out = target_dir / f'{stem}.cleaned.txt'
    terms_out = target_dir / f'{stem}.terms.csv'
    acr_out = target_dir / f'{stem}.acronyms.csv'
    docs_out = target_dir / f'{stem}.segments.txt'

    text_out.write_text(clean, encoding='utf-8')
    docs_out.write_text('\n\n---SEGMENT---\n\n'.join(docs), encoding='utf-8')
    write_csv(
        terms_out,
        term_rows,
        ['term', 'length_tokens', 'frequency', 'segment_df', 'num_segments',
         'c_value_like', 'tfidf_like', 'acronym_boost', 'pos_boost', 'score']
    )
    write_acronyms(acr_out, acronyms)

    print(f'\nProcessed: {in_path.name}')
    print(f'Wrote cleaned text to: {text_out}')
    print(f'Wrote segments to: {docs_out}')
    print(f'Wrote term ranking to: {terms_out}')
    print(f'Wrote acronym counts to: {acr_out}')
    print('Top 20 terms:')
    for r in term_rows[:20]:
        print(f"- {r['term']} | score={r['score']} | freq={r['frequency']} | tfidf={r['tfidf_like']} | cval={r['c_value_like']}")


def main():
    if not INPUT_FILES:
        print('No INPUT_FILES configured yet. Edit the script and put your .mmd file paths into the INPUT_FILES list at the top.')
        return
    for fp in INPUT_FILES:
        process_one_file(fp, top_k=TOP_K, outdir=OUTDIR)


if __name__ == '__main__':
    main()
