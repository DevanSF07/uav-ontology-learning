
import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\-/& ]+", flags=re.UNICODE)
_DASHES_RE = re.compile(r"[‐‑‒–—−]+")
_WS_RE = re.compile(r"\s+")


def normalize_term(text: str) -> str:
    if text is None:
        return ''
    text = unicodedata.normalize('NFKC', str(text))
    text = text.strip()
    text = _DASHES_RE.sub('-', text)
    text = text.replace('–', '-')
    text = text.replace('—', '-')
    text = text.replace('’', "'")
    text = text.replace('“', '"').replace('”', '"')
    # Keep /, -, &, alnum; drop other punctuation
    text = _PUNCT_RE.sub(' ', text)
    text = text.lower()
    text = _WS_RE.sub(' ', text).strip()
    return text
