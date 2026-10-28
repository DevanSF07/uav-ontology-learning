#!/usr/bin/env python3
"""
UAV term + supporting sentence -> MPNet embeddings -> NMF latent factor audit

Pipeline requested:
  1. Read PDF/TXT/MD or a directory of chapter files.
  2. Clean document noise: tables, references, LaTeX table fragments, captions, headers/footers.
  3. Chunk cleaned text.
  4. Extract candidate terms from every chunk using YAKE.
  5. Optionally filter/refine terms with local Ollama so only engineering terms remain.
  6. Resolve aliases/acronyms using regex patterns such as "Global Positioning System (GPS)".
  7. Build one record per canonical term using term + supporting sentence/context.
  8. Embed records with sentence-transformers/all-mpnet-base-v2.
  9. Run NMF on transformed non-negative embeddings.
  10. Export CSV/Markdown files for manual factor inspection.

Important interpretation:
  NMF factors are NOT automatically metamodel classes. They are latent factors over the MPNet embedding space.
  You inspect top activating terms per factor and decide whether a factor corresponds to structural, component,
  mission, parameter, environment, or noise/artifact signal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler


# -----------------------------
# Basic text IO
# -----------------------------

def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        from PyPDF2 import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(f"\n\n[PAGE {i+1}]\n{txt}")
    return "\n".join(pages)


def read_document_or_dir(input_path: str) -> pd.DataFrame:
    p = Path(input_path)
    files: List[Path] = []
    if p.is_dir():
        for ext in ("*.pdf", "*.txt", "*.mmd"):
            files.extend(sorted(p.glob(ext)))
    elif p.exists():
        files = [p]
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    rows = []
    for fp in files:
        suffix = fp.suffix.lower()
        if suffix == ".pdf":
            text = read_pdf(fp)
        elif suffix in {".txt", ".mmd"}:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        if text.strip():
            rows.append({"source_file": fp.name, "raw_text": text})
    if not rows:
        raise ValueError("No readable PDF/TXT/MD content found.")
    return pd.DataFrame(rows)


# -----------------------------
# Cleaning: tables, references, latex artifacts
# -----------------------------

REFERENCE_HEADINGS = re.compile(
    r"(?im)^\s*(references|bibliography|works cited|further reading)\s*$"
)

LATEX_TABLE_BLOCKS = [
    re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.S),
    re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.S),
    re.compile(r"\\begin\{longtable\}.*?\\end\{longtable\}", re.S),
    re.compile(r"\\begin\{array\}.*?\\end\{array\}", re.S),
]


def remove_references_section(text: str) -> str:
    """Remove reference section within a chapter/file. Conservative: removes after a clear references heading."""
    m = REFERENCE_HEADINGS.search(text)
    if not m:
        return text
    return text[: m.start()]


def looks_like_table_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # Common PDF table remnants: many numbers, separators, columns, or table captions.
    if re.match(r"(?i)^(table|fig\.|figure)\s+\d+(\.\d+)?", s):
        return True
    if s.count("|") >= 2 or s.count("&") >= 3:
        return True
    if re.search(r"\\hline|\\cline|\\toprule|\\midrule|\\bottomrule", s):
        return True
    # Lines dominated by numeric columns.
    tokens = s.split()
    if len(tokens) >= 5:
        numericish = sum(bool(re.search(r"^[\d\-+.,:%/()]+$", t)) for t in tokens)
        if numericish / len(tokens) > 0.55:
            return True
    # Page footer/header-like short lines.
    if re.match(r"^\d+\s*$", s):
        return True
    if re.match(r"(?i)^\s*(contents|index|list of figures|list of tables)\s*$", s):
        return True
    return False


def clean_document_text(text: str, remove_references: bool = True) -> str:
    # Remove LaTeX table blocks first.
    for pat in LATEX_TABLE_BLOCKS:
        text = pat.sub(" ", text)

    # Remove common LaTeX commands but keep content where possible.
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", text)

    # Remove DOI/URL/copyright-ish noise.
    text = re.sub(r"(?i)doi:\s*\S+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"(?i)©\s*springer.*?(?=\.|\n)", " ", text)
    text = re.sub(r"(?i)springer science\+business media.*?(?=\.|\n)", " ", text)

    # Remove references after heading per file/chapter.
    if remove_references:
        text = remove_references_section(text)

    cleaned_lines = []
    for line in text.splitlines():
        if looks_like_table_line(line):
            continue
        s = line.strip()
        # Drop very short isolated fragments that are likely headers or column labels.
        if len(s) < 3:
            continue
        cleaned_lines.append(s)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"-\s*\n\s*", "", text)  # dehyphenate line breaks
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


# -----------------------------
# Splitting
# -----------------------------

def split_sentences(text: str) -> List[str]:
    # Conservative splitting for technical text.
    candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    sents = []
    for s in candidates:
        s = s.strip()
        if 45 <= len(s) <= 900:
            # Avoid reference-like sentences dominated by citations.
            if len(re.findall(r"\b[A-Z][a-z]+\s*\(\d{4}\)", s)) >= 3:
                continue
            sents.append(s)
    return sents


def chunk_sentences(sentences: List[str], chunk_words: int = 700, overlap_sentences: int = 2) -> List[Dict]:
    chunks = []
    cur = []
    cur_words = 0
    cid = 0
    for s in sentences:
        w = len(s.split())
        if cur and cur_words + w > chunk_words:
            cid += 1
            chunks.append({"chunk_id": f"C{cid:06d}", "text": " ".join(cur), "sentences": list(cur)})
            cur = cur[-overlap_sentences:] if overlap_sentences > 0 else []
            cur_words = sum(len(x.split()) for x in cur)
        cur.append(s)
        cur_words += w
    if cur:
        cid += 1
        chunks.append({"chunk_id": f"C{cid:06d}", "text": " ".join(cur), "sentences": list(cur)})
    return chunks


# -----------------------------
# Acronym/alias extraction
# -----------------------------

ACRONYM_PATTERNS = [
    # Global Positioning System (GPS)
    re.compile(r"\b([A-Z][A-Za-z][A-Za-z\-/ ]{4,80}?)\s*\(([A-Z][A-Z0-9\-/]{1,12})\)"),
    # GPS (Global Positioning System)
    re.compile(r"\b([A-Z][A-Z0-9\-/]{1,12})\s*\(([A-Z][A-Za-z][A-Za-z\-/ ]{4,80}?)\)"),
]


def extract_acronym_aliases(text: str) -> Dict[str, str]:
    alias_to_canonical: Dict[str, str] = {}
    for pat in ACRONYM_PATTERNS:
        for m in pat.finditer(text):
            a, b = m.group(1).strip(), m.group(2).strip()
            if a.isupper():
                acronym, full = a, b
            else:
                full, acronym = a, b
            full_clean = normalize_space(full)
            acro_clean = acronym.strip()
            # Basic validity: acronym should match initials reasonably or be short all-caps.
            initials = "".join(w[0].upper() for w in re.findall(r"[A-Za-z]+", full_clean))
            if len(acro_clean) >= 2 and (acro_clean.replace("/", "") in initials or acro_clean.isupper()):
                canonical = full_clean.lower()
                alias_to_canonical[acro_clean.lower()] = canonical
                alias_to_canonical[full_clean.lower()] = canonical
    return alias_to_canonical


# -----------------------------
# Term extraction and filtering
# -----------------------------

BAD_TERM_PATTERNS = [
    re.compile(r"(?i)^(chapter|section|table|figure|fig|page|references|contents|abstract|introduction|conclusion|discussion)\b"),
    re.compile(r"(?i)\b(et al|copyright|springer|doi|press|journal|conference|university|department)\b"),
    re.compile(r"[.;:!?]"),
    re.compile(r"\b\d{4}\b"),
]

GENERIC_PHRASES = set(
    "current state state art future directions important area main objective more information different types several regional efforts".split()
)

STOPWORD_EDGE = set(
    "a an the and or of for to in on by with without into from as at be is are was were been being this that these those such each other more most many several different current future main important possible necessary relevant based using used use".split()
)

TECH_HINTS = set(
    "system subsystem component module unit assembly airframe payload sensor actuator controller control navigation guidance communication link station propulsion engine motor battery power autopilot imu gps camera radar lidar antenna vehicle aircraft wing rotor propeller servo processor interface bus receiver transmitter datalink flight avionics software hardware".split()
)


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def singularize_light(term: str) -> str:
    # Conservative singularization for common engineering plural nouns.
    words = term.split()
    out = []
    for w in words:
        if len(w) > 4 and w.endswith("ies"):
            out.append(w[:-3] + "y")
        elif len(w) > 4 and w.endswith("sses"):
            out.append(w[:-2])
        elif len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            out.append(w[:-1])
        else:
            out.append(w)
    return " ".join(out)


def canonicalize_term(term: str, alias_to_canonical: Dict[str, str]) -> str:
    t = normalize_space(term.lower())
    t = t.strip(" -_/.,;:()[]{}")
    t = singularize_light(t)
    if t in alias_to_canonical:
        return alias_to_canonical[t]
    return t


def is_good_term(term: str, max_words: int = 5) -> bool:
    t = normalize_space(term.lower())
    if not t or len(t) < 3:
        return False
    if len(t) > 80:
        return False
    words = t.split()
    if len(words) > max_words:
        return False
    if len(words) == 1 and len(words[0]) < 3:
        return False
    if words[0] in STOPWORD_EDGE or words[-1] in STOPWORD_EDGE:
        return False
    if any(p.search(t) for p in BAD_TERM_PATTERNS):
        return False
    # Reject phrases that look like clauses rather than terms.
    if re.search(r"\b(is|are|was|were|has|have|had|can|may|will|shall|should|would|could|includes|include|provide|provides|used|using)\b", t):
        return False
    # Reject too many generic words.
    if len(words) >= 3:
        generic_count = sum(w in GENERIC_PHRASES or w in STOPWORD_EDGE for w in words)
        if generic_count >= 2:
            return False
    # Prefer terms that contain at least one alpha token.
    if not re.search(r"[a-zA-Z]", t):
        return False
    return True


def yake_terms_for_chunk(text: str, max_terms: int, max_ngram: int, dedup_lim: float) -> List[Tuple[str, float]]:
    try:
        import yake
    except Exception as e:
        raise ImportError("Missing package 'yake'. Install with: pip install yake") from e
    extractor = yake.KeywordExtractor(
        lan="en",
        n=max_ngram,
        dedupLim=dedup_lim,
        dedupFunc="seqm",
        windowsSize=2,
        top=max_terms,
        features=None,
    )
    kws = extractor.extract_keywords(text)
    # YAKE lower score is better.
    return [(normalize_space(k), float(score)) for k, score in kws]


def ollama_filter_terms(
    chunk_text: str,
    candidate_terms: List[str],
    model: str,
    endpoint: str,
    timeout: int = 120,
) -> List[str]:
    """Use local Ollama to keep only engineering concepts/terms. Returns original or corrected terms."""
    import requests

    if not candidate_terms:
        return []

    prompt = f"""
You are filtering candidate terms extracted from a UAV engineering handbook chunk.

Keep ONLY concise engineering concepts/terms, not descriptive phrases.
Prefer:
- systems, subsystems, components, physical parts
- sensors, actuators, control/navigation/communication/power elements
- interfaces, links, stations, onboard/offboard equipment
- accepted technical acronyms and their expanded forms

Reject:
- full sentences, clauses, verbs/actions
- generic phrases such as current state, future direction, important area
- references, authors, organizations, chapter/table/figure text
- long phrases longer than 5 words unless it is a standard technical name
- pure parameters unless they name a device/sensor/component

Return STRICT JSON only in this format:
{{"terms": ["term 1", "term 2"]}}

Candidate terms:
{json.dumps(candidate_terms, ensure_ascii=False)}

Chunk excerpt:
{chunk_text[:3500]}
""".strip()

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    r = requests.post(endpoint.rstrip("/") + "/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data.get("message", {}).get("content", "")
    try:
        obj = json.loads(content)
    except Exception:
        # Extract first JSON object if the model wrapped output.
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            return []
        obj = json.loads(m.group(0))
    terms = obj.get("terms", [])
    if not isinstance(terms, list):
        return []
    return [normalize_space(str(t)) for t in terms if normalize_space(str(t))]


def extract_terms_from_chunks(
    chunks: List[Dict],
    alias_to_canonical: Dict[str, str],
    args,
) -> pd.DataFrame:
    rows = []
    for i, ch in enumerate(chunks, start=1):
        raw = yake_terms_for_chunk(ch["text"], args.yake_top, args.yake_max_ngram, args.yake_dedup)
        raw_terms = [t for t, score in raw if is_good_term(t, args.max_term_words)]
        raw_terms = raw_terms[: args.yake_keep_before_ollama]

        if args.use_ollama:
            try:
                filtered = ollama_filter_terms(
                    ch["text"], raw_terms, args.ollama_model, args.ollama_endpoint, args.ollama_timeout
                )
            except Exception as e:
                print(f"[WARN] Ollama failed on {ch['chunk_id']}: {e}. Falling back to YAKE terms.", file=sys.stderr)
                filtered = raw_terms
        else:
            filtered = raw_terms

        for rank, term in enumerate(filtered, start=1):
            if not is_good_term(term, args.max_term_words):
                continue
            canonical = canonicalize_term(term, alias_to_canonical)
            if not is_good_term(canonical, args.max_term_words):
                continue
            rows.append({
                "chunk_id": ch["chunk_id"],
                "term_raw": term,
                "term_canonical": canonical,
                "rank_in_chunk": rank,
            })

        if args.sleep_between_ollama_calls > 0 and args.use_ollama:
            time.sleep(args.sleep_between_ollama_calls)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(["chunk_id", "term_canonical"])
    return df


# -----------------------------
# Supporting sentence lookup
# -----------------------------

def term_regex(term: str) -> re.Pattern:
    # Match term words with flexible whitespace/hyphen.
    parts = [re.escape(w) for w in term.split()]
    pattern = r"\b" + r"[\s\-/]+".join(parts) + r"\b"
    return re.compile(pattern, flags=re.I)


def find_supporting_sentence(term: str, sentences: List[str]) -> str:
    pat = term_regex(term)
    for s in sentences:
        if pat.search(s):
            return s
    # If no exact canonical match, try any token overlap with technical words.
    term_words = [w for w in re.findall(r"[a-zA-Z0-9]+", term.lower()) if w not in STOPWORD_EDGE]
    best_s = ""
    best_score = 0
    for s in sentences:
        sl = s.lower()
        score = sum(1 for w in term_words if w in sl)
        if score > best_score:
            best_score = score
            best_s = s
    return best_s if best_score > 0 else (sentences[0] if sentences else "")


def build_concept_records(term_df: pd.DataFrame, chunks: List[Dict], alias_to_canonical: Dict[str, str], contexts_per_term: int) -> pd.DataFrame:
    chunk_map = {c["chunk_id"]: c for c in chunks}
    # reverse alias map canonical -> aliases
    canonical_aliases = defaultdict(set)
    for alias, canonical in alias_to_canonical.items():
        canonical_aliases[canonical].add(alias)

    grouped = defaultdict(list)
    raw_alias_seen = defaultdict(set)
    for _, r in term_df.iterrows():
        canonical = r.term_canonical
        raw_alias_seen[canonical].add(r.term_raw.lower())
        ch = chunk_map.get(r.chunk_id)
        if not ch:
            continue
        sent = find_supporting_sentence(canonical, ch["sentences"])
        grouped[canonical].append({"chunk_id": r.chunk_id, "supporting_sentence": sent})

    rows = []
    for idx, (canonical, occs) in enumerate(grouped.items(), start=1):
        # Keep unique supporting sentences.
        uniq = []
        seen = set()
        for o in occs:
            s = normalize_space(o["supporting_sentence"])
            if s and s not in seen:
                uniq.append(o)
                seen.add(s)
            if len(uniq) >= contexts_per_term:
                break
        aliases = sorted((canonical_aliases.get(canonical, set()) | raw_alias_seen.get(canonical, set())) - {canonical})
        context_text = " ".join([u["supporting_sentence"] for u in uniq])
        embed_text = f"Concept: {canonical}."
        if aliases:
            embed_text += f" Aliases/acronyms: {', '.join(aliases[:10])}."
        embed_text += f" Supporting context: {context_text}"
        rows.append({
            "concept_id": f"K{idx:07d}",
            "term_canonical": canonical,
            "aliases": " | ".join(aliases),
            "occurrence_count": len(occs),
            "context_count": len(uniq),
            "supporting_sentences": " ||| ".join([u["supporting_sentence"] for u in uniq]),
            "source_chunks": " | ".join([u["chunk_id"] for u in uniq]),
            "embedding_text": embed_text,
        })
    records = pd.DataFrame(rows)
    if not records.empty:
        records = records.sort_values(["occurrence_count", "term_canonical"], ascending=[False, True]).reset_index(drop=True)
    return records


# -----------------------------
# Embedding + NMF
# -----------------------------

def embed_records(records: pd.DataFrame, model_name: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(
        records["embedding_text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def summarize_factor_terms(texts: Iterable[str], topn: int = 16) -> str:
    stop = STOPWORD_EDGE | {"concept", "supporting", "context", "alias", "acronym"}
    words = []
    for t in texts:
        for w in re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-/]{2,}\b", t.lower()):
            if w not in stop:
                words.append(w)
    return ", ".join([w for w, _ in Counter(words).most_common(topn)])


def hint_counts(terms: Iterable[str]) -> Dict[str, int]:
    blob = " ".join(terms).lower()
    categories = {
        "structural_hint_count": ["system", "subsystem", "component", "module", "unit", "assembly", "airframe", "payload", "station"],
        "sensor_actuator_hint_count": ["sensor", "camera", "radar", "lidar", "imu", "gps", "actuator", "servo", "antenna", "receiver", "transmitter"],
        "control_navigation_hint_count": ["control", "controller", "navigation", "guidance", "autopilot", "flight", "avionics"],
        "communication_hint_count": ["communication", "link", "datalink", "telemetry", "command", "antenna", "radio"],
        "power_propulsion_hint_count": ["power", "battery", "propulsion", "motor", "engine", "fuel", "propeller"],
        "parameter_hint_count": ["altitude", "speed", "range", "endurance", "weight", "accuracy", "rate", "frequency", "bandwidth"],
        "mission_hint_count": ["mission", "surveillance", "reconnaissance", "tracking", "monitoring", "inspection", "target"],
        "environment_hint_count": ["airspace", "weather", "wind", "terrain", "obstacle", "traffic", "environment"],
    }
    return {k: sum(blob.count(h) for h in hints) for k, hints in categories.items()}


def run_nmf(records: pd.DataFrame, embeddings: np.ndarray, n_components: int, top_n: int, outdir: Path) -> None:
    # NMF needs non-negative input. MPNet embeddings can be negative, so we use min-max scaling.
    scaler = MinMaxScaler()
    X_nonneg = scaler.fit_transform(embeddings)

    nmf = NMF(n_components=n_components, init="nndsvda", max_iter=1200, random_state=42)
    W = nmf.fit_transform(X_nonneg)
    H = nmf.components_

    loading_df = pd.DataFrame(W, columns=[f"factor_{i:03d}" for i in range(n_components)])
    loading_df.insert(0, "concept_id", records["concept_id"].values)
    loading_df.insert(1, "term_canonical", records["term_canonical"].values)
    loading_df.to_csv(outdir / "nmf_term_factor_loadings.csv", index=False)

    top_rows = []
    summary_rows = []
    for k in range(n_components):
        order = np.argsort(W[:, k])[::-1][:top_n]
        top_terms = records.iloc[order]["term_canonical"].tolist()
        top_texts = records.iloc[order]["embedding_text"].tolist()
        hcounts = hint_counts(top_terms)
        summary_rows.append({
            "factor_id": f"factor_{k:03d}",
            "top_keywords_from_records": summarize_factor_terms(top_texts),
            "top_terms_preview": " | ".join(top_terms[:12]),
            **hcounts,
        })
        for rank, idx in enumerate(order, start=1):
            r = records.iloc[idx]
            top_rows.append({
                "factor_id": f"factor_{k:03d}",
                "rank": rank,
                "activation": float(W[idx, k]),
                "concept_id": r.concept_id,
                "term_canonical": r.term_canonical,
                "aliases": r.aliases,
                "occurrence_count": int(r.occurrence_count),
                "context_count": int(r.context_count),
                "supporting_sentences": r.supporting_sentences,
                "source_chunks": r.source_chunks,
            })

    pd.DataFrame(summary_rows).to_csv(outdir / "nmf_factor_summary.csv", index=False)
    pd.DataFrame(top_rows).to_csv(outdir / "nmf_top_terms_by_factor.csv", index=False)
    np.save(outdir / "nmf_W_term_factor_matrix.npy", W)
    np.save(outdir / "nmf_H_factor_embedding_matrix.npy", H)


def write_report(records: pd.DataFrame, args, outdir: Path) -> None:
    factor_summary = pd.read_csv(outdir / "nmf_factor_summary.csv")
    lines = []
    lines.append("# UAV Term + Supporting Sentence MPNet + NMF Audit")
    lines.append("")
    lines.append("## What was embedded")
    lines.append("Each record is a canonical term plus aliases/acronyms and one or more supporting sentences from the cleaned UAV handbook.")
    lines.append("")
    lines.append("Example embedding text format:")
    lines.append("```text")
    lines.append("Concept: payload system. Aliases/acronyms: mission payload. Supporting context: The payload system carries EO/IR sensors ...")
    lines.append("```")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Input: `{args.input}`")
    lines.append(f"- MPNet model: `{args.model}`")
    lines.append(f"- NMF factors: `{args.nmf_components}`")
    lines.append(f"- Concepts embedded: `{len(records)}`")
    lines.append(f"- Ollama filtering used: `{args.use_ollama}`")
    lines.append("")
    lines.append("## Important interpretation rule")
    lines.append("NMF does not decide ontology class names. NMF only gives latent factors. You inspect top activating terms and decide whether a factor is structural, sensor/component-like, mission-like, parameter-like, environment-like, or noise.")
    lines.append("")
    lines.append("## First 20 factor previews")
    preview_cols = ["factor_id", "top_terms_preview", "top_keywords_from_records", "structural_hint_count", "sensor_actuator_hint_count", "control_navigation_hint_count", "parameter_hint_count", "mission_hint_count"]
    lines.append(factor_summary[preview_cols].head(20).to_markdown(index=False))
    lines.append("")
    lines.append("## Files to inspect first")
    lines.append("1. `concept_records_for_embedding.csv` - final canonical terms and supporting sentences.")
    lines.append("2. `nmf_factor_summary.csv` - quick factor overview.")
    lines.append("3. `nmf_top_terms_by_factor.csv` - top activating concepts per factor; this is the main manual inspection file.")
    lines.append("4. `nmf_term_factor_loadings.csv` - factor weights for every term.")
    (outdir / "nmf_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="UAV term+context MPNet embedding + NMF latent-factor audit")
    ap.add_argument("--input", required=True, help="PDF/TXT/MD file or directory of chapter files")
    ap.add_argument("--outdir", default="uav_term_context_nmf_out")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--chunk-words", type=int, default=700)
    ap.add_argument("--chunk-overlap-sentences", type=int, default=2)
    ap.add_argument("--contexts-per-term", type=int, default=3)
    ap.add_argument("--yake-top", type=int, default=60, help="YAKE terms per chunk before filtering")
    ap.add_argument("--yake-keep-before-ollama", type=int, default=35, help="Max YAKE candidates sent to Ollama per chunk")
    ap.add_argument("--yake-max-ngram", type=int, default=4)
    ap.add_argument("--yake-dedup", type=float, default=0.85)
    ap.add_argument("--max-term-words", type=int, default=5, help="Reject long descriptive phrases")
    ap.add_argument("--use-ollama", action="store_true", help="Use local Ollama to filter YAKE candidates")
    ap.add_argument("--ollama-model", default="llama3.1:8b")
    ap.add_argument("--ollama-endpoint", default="http://localhost:11434")
    ap.add_argument("--ollama-timeout", type=int, default=120)
    ap.add_argument("--sleep-between-ollama-calls", type=float, default=0.0)
    ap.add_argument("--nmf-components", type=int, default=70)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-concepts", type=int, default=0, help="Optional cap after extraction; 0 means no cap")
    ap.add_argument("--no-remove-references", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/8] Reading corpus...")
    corpus = read_document_or_dir(args.input)

    print("[2/8] Cleaning text...")
    cleaned_parts = []
    all_alias_map: Dict[str, str] = {}
    for _, row in corpus.iterrows():
        cleaned = clean_document_text(row.raw_text, remove_references=not args.no_remove_references)
        cleaned_parts.append({"source_file": row.source_file, "clean_text": cleaned})
        all_alias_map.update(extract_acronym_aliases(cleaned))
    clean_df = pd.DataFrame(cleaned_parts)
    clean_df.to_csv(outdir / "cleaned_documents.csv", index=False)
    pd.DataFrame([{"alias": k, "canonical": v} for k, v in sorted(all_alias_map.items())]).to_csv(outdir / "regex_acronym_aliases.csv", index=False)

    print("[3/8] Splitting into sentences and chunks...")
    all_sentences = []
    for _, row in clean_df.iterrows():
        all_sentences.extend(split_sentences(row.clean_text))
    chunks = chunk_sentences(all_sentences, args.chunk_words, args.chunk_overlap_sentences)
    pd.DataFrame([{"chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks]).to_csv(outdir / "chunks.csv", index=False)

    print(f"[4/8] Extracting terms with YAKE{' + Ollama' if args.use_ollama else ''}...")
    term_df = extract_terms_from_chunks(chunks, all_alias_map, args)
    if term_df.empty:
        raise RuntimeError("No terms extracted after filtering. Relax max-term-words or disable Ollama filtering.")
    term_df.to_csv(outdir / "chunk_terms_yake_ollama_filtered.csv", index=False)

    print("[5/8] Building canonical term + supporting sentence records...")
    records = build_concept_records(term_df, chunks, all_alias_map, args.contexts_per_term)
    if args.max_concepts and len(records) > args.max_concepts:
        records = records.head(args.max_concepts).copy()
    if records.empty:
        raise RuntimeError("No concept records created.")
    records.to_csv(outdir / "concept_records_for_embedding.csv", index=False)

    print("[6/8] Embedding term + supporting sentence records with MPNet...")
    embeddings = embed_records(records, args.model, args.batch_size)
    np.save(outdir / "mpnet_term_context_embeddings.npy", embeddings)

    print("[7/8] Running NMF...")
    run_nmf(records, embeddings, args.nmf_components, args.top_n, outdir)

    print("[8/8] Writing report...")
    write_report(records, args, outdir)

    print("\nDone.")
    print(f"Output directory: {outdir.resolve()}")
    print("Open these first:")
    print("  - nmf_audit_report.md")
    print("  - nmf_factor_summary.csv")
    print("  - nmf_top_terms_by_factor.csv")
    print("  - concept_records_for_embedding.csv")


if __name__ == "__main__":
    main()
