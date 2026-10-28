#!/usr/bin/env python3
"""
UAV term typing pipeline using Ollama.

Features:
1) Reads terms from CSV: data/processed/extractions/llm_llama3.1_8b_terms.csv
2) Reads source context from MMD: data/raw_data/
3) Splits MMD into sentences and builds an embedding index
4) Retrieves at least top-k (default 10) most relevant sentences per term
5) Zero-shot term typing using label prototypes loaded from JSON
6) Optional few-shot retrieval + prompting using a labeled example CSV
7) Writes outputs to CSV/TSV

This is a practical Alexbek-style adaptation:
- Zero-shot: multi-embedding similarity + confidence-weighted score fusion
- Few-shot: embedding-based retrieval of labeled examples + LLM prompt classification
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
# import os
from typing import List
from sentence_transformers import SentenceTransformer

from context_retrieval_large import build_sentence_index, retrieve_large_context_for_term
from normalization import normalize_term

# ============================================================
# DEFAULT PATHS
# ============================================================
DEFAULT_TERMS_CSV = "data/processed/extractions/llm_llama3.1_8b_terms.csv"
# DEFAULT_CONTEXT_MMD = "data/raw_data/Handbook_of_Unmanned_Aerial_Vehicles_chapter5.mmd"
DEFAULT_CONTEXT_DIR = "data/raw_data/"
DEFAULT_LABELS_JSON = "data/uav_label_prototypes.json"
DEFAULT_FEWSHOT_CSV = "data/uav_fewshot_examples.csv"
DEFAULT_ZEROSHOT_OUT_CSV = "uav_zero_shot_results.csv"
DEFAULT_ZEROSHOT_OUT_TSV = "uav_zero_shot_results.tsv"
DEFAULT_FEWSHOT_OUT_CSV = "uav_few_shot_results.csv"
DEFAULT_FEWSHOT_OUT_TSV = "uav_few_shot_results.tsv"
DEFAULT_HYBRID_OUT_CSV = "uav_hybrid_results.csv"
DEFAULT_HYBRID_OUT_TSV = "uav_hybrid_results.tsv"


# ============================================================
# OLLAMA CONFIG
# ============================================================
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "llama3.1:8b"
DEFAULT_EMBED_MODELS = [
    "models/all-MiniLM-L6-v2",
    "models/all-mpnet-base-v2",
]
DEFAULT_CONTEXT_RETRIEVAL_MODEL = "models/all-MiniLM-L6-v2"
DEFAULT_TOP_K_CONTEXTS = 10
DEFAULT_TOP_K_FEWSHOT = 5
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.10
DEFAULT_CONTEXT_SENTENCES = 5
EMBED_MAX_SENTENCES = 5
EMBED_MAX_CHARS = 3500
DEFAULT_GENERATE_TIMEOUT = 600
DEFAULT_GENERATE_RETRIES = 2
DEFAULT_FEWSHOT_MAX_CONTEXT_CHARS = 2500
DEFAULT_FEWSHOT_MAX_EXAMPLE_CONTEXT_CHARS = 1000


# canonical set for validation
LABELS_CANONICAL = {
    "system", "part", "mission", "capability",
    "parameter", "environment", "domain", "unknown"
}


# ============================================================
# HELPERS
# ============================================================
def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def safe_mkdir_for(filepath: str) -> None:
    parent = os.path.dirname(os.path.abspath(filepath))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def split_text_for_embedding(text: str, max_sentences: int = EMBED_MAX_SENTENCES, max_chars: int = EMBED_MAX_CHARS) -> List[str]:
    normalized = normalize_space(text)
    if not normalized:
        return []

    sentence_parts = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: List[str] = []
    current: List[str] = []
    current_chars = 0

    def flush_current() -> None:
        nonlocal current, current_chars
        if current:
            chunks.append(" ".join(current))
            current = []
            current_chars = 0

    for part in sentence_parts:
        part = normalize_space(part)
        if not part:
            continue

        part_len = len(part)
        if current and (len(current) >= max_sentences or current_chars + part_len + 1 > max_chars):
            flush_current()

        if part_len > max_chars:
            flush_current()
            words = part.split()
            word_chunk: List[str] = []
            word_chars = 0
            for word in words:
                word_len = len(word)
                if word_chunk and word_chars + word_len + 1 > max_chars:
                    chunks.append(" ".join(word_chunk))
                    word_chunk = [word]
                    word_chars = word_len
                else:
                    word_chunk.append(word)
                    word_chars += word_len + (1 if len(word_chunk) > 1 else 0)
            if word_chunk:
                chunks.append(" ".join(word_chunk))
            continue

        current.append(part)
        current_chars += part_len + (1 if len(current) > 1 else 0)

    flush_current()
    return chunks


def truncate_text_for_prompt(text: str, max_chars: int) -> str:
    normalized = normalize_space(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 3)].rstrip() + "..."


@lru_cache(maxsize=8)
def load_sentence_transformer(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def is_local_embedding_model(model_name: str) -> bool:
    return os.path.isdir(model_name) or model_name.startswith("models/")


def embed_locally(model_name: str, texts: List[str]) -> np.ndarray:
    model = load_sentence_transformer(model_name)
    vectors = []

    for text in texts:
        chunks = split_text_for_embedding(text)
        if not chunks:
            chunks = [""]

        chunk_embs = model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        pooled = chunk_embs if chunk_embs.ndim == 1 else np.mean(chunk_embs, axis=0)
        vectors.append(np.asarray(pooled, dtype=np.float32))

    return np.array(vectors, dtype=np.float32)


# ============================================================
# OLLAMA CLIENT
# ============================================================
class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _embed_raw(self, model: str, texts: List[str]) -> np.ndarray:
        if is_local_embedding_model(model):
            return embed_locally(model, texts)

        # Try modern batch embed API first
        try:
            url = f"{self.base_url}/api/embed"
            payload = {"model": model, "input": texts}
            r = requests.post(url, json=payload, timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                embs = data.get("embeddings")
                if embs is not None:
                    return np.array(embs, dtype=np.float32)
        except Exception:
            pass

        # Fallback to older /api/embeddings one-by-one
        out = []
        url = f"{self.base_url}/api/embeddings"
        for t in texts:
            payload = {"model": model, "prompt": t}
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            if emb is None:
                raise RuntimeError(f"Embedding failed for model={model}")
            out.append(emb)
        return np.array(out, dtype=np.float32)

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        timeout: int | None = None,
        retries: int = DEFAULT_GENERATE_RETRIES,
        keep_alive: str = "10m",
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
            },
        }
        request_timeout = timeout or self.timeout

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=request_timeout)
                r.raise_for_status()
                data = r.json()
                return data.get("response", "").strip()
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt >= retries:
                    raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Ollama generate failed without an exception")

    def embed(self, model: str, texts: List[str]) -> np.ndarray:
        outputs = []

        for text in texts:
            chunks = split_text_for_embedding(text)
            if not chunks:
                chunks = [""]

            chunk_embs = self._embed_raw(model, chunks)
            pooled = chunk_embs if chunk_embs.ndim == 1 else np.mean(chunk_embs, axis=0)
            outputs.append(pooled)

        return np.array(outputs, dtype=np.float32)


# ============================================================
# LABEL PROTOTYPES
# ============================================================
def load_label_prototypes(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "labels" in obj:
        labels = obj["labels"]
    else:
        labels = obj

    # expected format: {label: {definition: ..., examples:[...]}}
    return labels


def make_label_prototype_text(label: str, payload: Dict[str, Any]) -> str:
    definition = payload.get("definition", "")
    examples = payload.get("examples", []) or []
    ex_text = " ; ".join(examples)
    return f"LABEL: {label}\nDEFINITION: {definition}\nEXAMPLES: {ex_text}"


# ============================================================
# TERM INPUT
# ============================================================
def read_terms_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    seen = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = normalize_space(row.get("term", ""))
            norm = normalize_space(row.get("normalized_term", ""))
            score = row.get("score", "")
            method = row.get("method", "")

            # Prefer normalized term if available and meaningful
            final_term = norm if norm else term
            final_term = normalize_space(final_term)
            if not final_term:
                continue

            # Filter obvious junk/header-like rows
            low = final_term.lower()
            if low in {
                "here are the extracted terms",
                "here are extracted terms",
                "extracted terms",
            }:
                continue
            if len(final_term) < 2:
                continue

            if final_term.lower() in seen:
                continue
            seen.add(final_term.lower())

            rows.append({
                "id": str(len(rows) + 1),
                "term": final_term,
                "raw_term": term,
                "normalized_term": norm,
                "score": score,
                "method": method,
            })
    return rows


# ============================================================
# MMD CONTEXT PROCESSING
# ============================================================
# def clean_mmd_text(text: str) -> str:
#     # Remove code fences if present
#     text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
#     # Remove markdown headings / bullets markers but keep text
#     text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
#     text = re.sub(r"^[\-\*\+]\s*", "", text, flags=re.MULTILINE)
#     # Remove table pipes lightly
#     text = text.replace("|", " ")
#     return normalize_space(text)


# def split_into_sentences(text: str) -> List[str]:
#     text = clean_mmd_text(text)
#     parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
#     sents = []
#     for p in parts:
#         p = normalize_space(p)
#         if not p:
#             continue
#         if len(p.split()) < 3:
#             continue
#         sents.append(p)
#     return sents


# # def read_mmd_sentences(path: str) -> List[str]:
# #     with open(path, "r", encoding="utf-8") as f:
# #         text = f.read()
# #     return split_into_sentences(text)


# def read_mmd_sentences(path: str) -> List[str]:
#     paths = []
#     for base, _, files in os.walk(root_dir):
#         for name in files:
#             if name.lower().endswith(".mmd"):
#                 paths.append(os.path.join(base, name))
#     paths.sort()
#     return paths


# def read_all_mmd_sentences(root_dir: str) -> List[str]:
#     paths = list_mmd_files(root_dir)
#     if not paths:
#         raise RuntimeError(f"No .mmd files found under {root_dir}")

#     all_sentences = []
#     seen = set()

#     for p in paths:
#         try:
#             sents = read_mmd_sentences(p)
#         except Exception as e:
#             print(f"Warning: failed to read {p}: {e}")
#             continue

#         for s in sents:
#             key = s.lower()
#             if key not in seen:
#                 seen.add(key)
#                 all_sentences.append(s)

#     return all_sentences

# def list_mmd_files(root_dir: str) -> List[str]:
#     paths = []
#     for base, _, files in os.walk(root_dir):
#         for name in files:
#             if name.lower().endswith('.mmd'):
#                 paths.append(os.path.join(base, name))
#     paths.sort()
#     return paths

import os
import re
from typing import List


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def clean_mmd_text(text: str) -> str:
    """
    Keep useful prose-like content from markdown/mmd, but do not over-delete.
    """
    # Remove fenced code blocks
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)

    # Remove HTML comments if any
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

    # Replace markdown headings markers but keep heading text
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Replace bullet markers but keep item text
    text = re.sub(r"^[\-\*\+]\s*", "", text, flags=re.MULTILINE)

    # Replace tabs
    text = text.replace("\t", " ")

    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Robust extraction for handbook-like .mmd files:
    1) try sentence splitting
    2) fallback to line-based chunks
    3) fallback to paragraph chunks
    """
    text = clean_mmd_text(text)

    candidates = []

    # ---------- A. punctuation-based sentences ----------
    text_one_line = normalize_space(text)
    sent_parts = re.split(r"(?<=[.!?])\s+", text_one_line)
    for s in sent_parts:
        s = normalize_space(s)
        if len(s.split()) >= 3:
            candidates.append(s)

    # ---------- B. line-based fallback ----------
    if len(candidates) == 0:
        for line in text.splitlines():
            line = normalize_space(line)
            if not line:
                continue
            # Skip extremely short fragments
            if len(line.split()) < 3:
                continue
            candidates.append(line)

    # ---------- C. paragraph fallback ----------
    if len(candidates) == 0:
        paras = re.split(r"\n\s*\n", text)
        for p in paras:
            p = normalize_space(p)
            if not p:
                continue
            if len(p.split()) < 3:
                continue
            candidates.append(p)

    # ---------- D. deduplicate while preserving order ----------
    final = []
    seen = set()
    for c in candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            final.append(c)

    return final


def read_mmd_sentences(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return split_into_sentences(text)


def list_mmd_files(root_dir: str) -> List[str]:
    paths = []
    for base, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(".mmd"):
                paths.append(os.path.join(base, name))
    paths.sort()
    return paths


def read_all_mmd_sentences(root_dir: str) -> List[str]:
    paths = list_mmd_files(root_dir)
    if not paths:
        raise RuntimeError(f"No .mmd files found under {root_dir}")

    print(f"Found {len(paths)} .mmd files under {root_dir}")

    all_sentences = []
    seen = set()

    for p in paths:
        try:
            sents = read_mmd_sentences(p)
            print(f"{p}: extracted {len(sents)} text chunks")
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
            continue

        for s in sents:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                all_sentences.append(s)

    return all_sentences

    # with open(path, "r", encoding="utf-8") as f:
    #     text = f.read()
    # return split_into_sentences(text)

# ============================================================
# CONTEXT RETRIEVER FROM MMD SENTENCES
# ============================================================
class SentenceContextRetriever:
    def __init__(self, sentence_index: List[dict]):
        self.sentence_index = sentence_index

    def retrieve(self, query_term: str, top_k: int = 10) -> List[Tuple[str, float]]:
        max_sentences = min(max(top_k, 1), DEFAULT_CONTEXT_SENTENCES)
        context_text, _ = retrieve_large_context_for_term(
            term=query_term,
            term_norm=normalize_term(query_term),
            sentence_index=self.sentence_index,
            max_sentences=max_sentences,
            max_chars=5000,
        )

        if not context_text:
            return []

        sentences = [part.strip() for part in context_text.split("\n---\n") if part.strip()]
        return [(sentence, 1.0) for sentence in sentences]


# ============================================================
# ZERO-SHOT EMBEDDING TYPER
# ============================================================
@dataclass
class ZeroShotResult:
    label: str
    scores: Dict[str, float]
    confidence: float
    per_model_scores: Dict[str, Dict[str, float]]
    contexts: List[str]


class ZeroShotEmbeddingTyper:
    def __init__(
        self,
        ollama: OllamaClient,
        embedding_models: List[str],
        label_prototypes: Dict[str, Dict[str, Any]],
        context_retriever: SentenceContextRetriever,
        top_k_contexts: int = 0,
        temperature: float = 0.08,
    ):
        self.ollama = ollama
        self.embedding_models = embedding_models
        self.label_prototypes = label_prototypes
        self.labels = list(label_prototypes.keys())
        self.context_retriever = context_retriever
        self.top_k_contexts = top_k_contexts
        self.temperature = temperature

        self.label_texts = [
            make_label_prototype_text(lbl, label_prototypes[lbl]) for lbl in self.labels
        ]

        self.label_embs_by_model: Dict[str, np.ndarray] = {}
        for m in self.embedding_models:
            embs = self.ollama.embed(m, self.label_texts)
            self.label_embs_by_model[m] = l2_normalize_rows(embs)

    @staticmethod
    def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        z = x / max(temperature, 1e-8)
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    @staticmethod
    def margin_confidence(scores: np.ndarray) -> float:
        ordered = np.sort(scores)[::-1]
        if len(ordered) < 2:
            return float(ordered[0])
        return float(max(ordered[0] - ordered[1], 1e-6))

    def build_query_text(self, term: str, contexts: List[str]) -> str:
        joined = " ".join(contexts)
        return (
            f"TERM: {term}\n"
            f"CONTEXT_SENTENCES: {joined}\n"
            f"TASK: classify the UAV-domain term into one ontology label using the term and context."
        )

    def predict_one(self, term: str) -> ZeroShotResult:
        retrieved = self.context_retriever.retrieve(term, top_k=self.top_k_contexts)
        contexts = [s for s, _ in retrieved]
        query_text = self.build_query_text(term, contexts)

        weighted_probs = []
        model_weights = []
        per_model_scores: Dict[str, Dict[str, float]] = {}

        for m in self.embedding_models:
            q_emb = self.ollama.embed(m, [query_text])
            q_emb = l2_normalize_rows(q_emb)[0]
            label_embs = self.label_embs_by_model[m]
            sims = cosine_similarity([q_emb], label_embs)[0]

            per_model_scores[m] = {
                label: float(score) for label, score in zip(self.labels, sims)
            }

            probs = self.softmax(sims, self.temperature)
            conf = self.margin_confidence(sims)
            weighted_probs.append(probs)
            model_weights.append(conf)

        model_weights = np.array(model_weights, dtype=np.float32)
        model_weights = model_weights / model_weights.sum()

        combined = np.zeros(len(self.labels), dtype=np.float32)
        for w, probs in zip(model_weights, weighted_probs):
            combined += w * probs

        best_idx = int(np.argmax(combined))
        best_label = self.labels[best_idx]
        score_map = {label: float(score) for label, score in zip(self.labels, combined)}
        ordered = sorted(score_map.values(), reverse=True)
        confidence = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])

        return ZeroShotResult(
            label=best_label,
            scores=score_map,
            confidence=confidence,
            per_model_scores=per_model_scores,
            contexts=contexts,
        )

    def predict_many(self, terms: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        out = []
        for row in terms:
            res = self.predict_one(row["term"])
            out.append({
                "id": row["id"],
                "term": row["term"],
                "label": res.label,
                "confidence": res.confidence,
                "scores": res.scores,
                "contexts": res.contexts,
                "per_model_scores": res.per_model_scores,
            })
        return out


# ============================================================
# FEW-SHOT RETRIEVER + PROMPTING
# ============================================================
def build_prompt_header() -> str:
    return """You are a UAV-domain ontology term typer.

Allowed labels:
system
part
mission
capability
parameter
environment
domain
unknown

label definitions:

system:
A general engineered object, vehicle, platform, device, subsystem, or operational system treated as a functional whole.
Use system for:
- system types
- categories
- variants
- subtypes
- named platforms
- named vehicles
- named system models

part:
A physical, structural, electrical, communication, software, sensor, launcher, interface, or logical component that belongs to or is contained in a system.

mission:
A goal, use case, operational purpose, or task for which a system is employed.
Ask: why is the system being used?

capability:
An ability, operational function, or service the system can provide.
Ask: what can the system do?

parameter:
A measurable, configurable, constrained, or value-like property.
Ask: how much / how fast / how large / what value / what range?

environment:
An external operating condition, medium, location, terrain, airspace, weather condition, or surrounding context in which a system operates.

domain:
A broad operational or institutional domain such as civil, military, public, maritime, etc.

unknown:
Use only if the context is insufficient or the term does not clearly fit any allowed label.

Methodology:
1. Use BOTH the term and the context.
2. Prefer the label best supported by context.
3. Terms may already be normalized or lemmatized.
4. If a term is instance-like but clearly refers to a named system/platform, label it as system.

Return format:
Return ONLY TSV lines in exactly this format:
id<TAB>term<TAB>label

Do not include markdown.
Do not include explanations.
Do not include extra text.
""".strip()


def build_few_shot_prompt(
    target_rows: List[Dict[str, str]],
    retrieved_examples: List[Dict[str, str]],
) -> str:
    header = build_prompt_header()
    lines = [header, "", "Here are some labeled examples:", ""]
    for ex in retrieved_examples:
        lines.append(f"ID: {ex['id']}")
        lines.append(f"TERM: {ex['term']}")
        lines.append(f"CONTEXT: {truncate_text_for_prompt(ex['context'], DEFAULT_FEWSHOT_MAX_EXAMPLE_CONTEXT_CHARS)}")
        lines.append(f"LABEL: {ex['label']}")
        lines.append("")

    lines.append("Items to classify:")
    for row in target_rows:
        lines.append(f"ID: {row['id']}")
        lines.append(f"TERM: {row['term']}")
        lines.append(f"CONTEXT: {truncate_text_for_prompt(row['context'], DEFAULT_FEWSHOT_MAX_CONTEXT_CHARS)}")
        lines.append("")
    return "\n".join(lines)


def parse_tsv(text: str) -> List[Tuple[str, str, str]]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        rid, term, label = [p.strip() for p in parts]
        rows.append((rid, term, label))
    return rows


def read_fewshot_examples_csv(path: str) -> List[Dict[str, str]]:
    examples = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ex = {
                "id": normalize_space(row.get("id", "")),
                "term": normalize_space(row.get("term", "")),
                "context": normalize_space(row.get("context", "")),
                "label": normalize_space(row.get("label", "")),
            }
            if ex["term"] and ex["label"]:
                examples.append(ex)
    return examples


class ExampleRetriever:
    def __init__(
        self,
        ollama: OllamaClient,
        embedding_model: str,
        labeled_examples: List[Dict[str, str]],
    ):
        self.ollama = ollama
        self.embedding_model = embedding_model
        self.examples = labeled_examples
        self.example_texts = [
            f"TERM: {ex['term']}\nCONTEXT: {ex.get('context', '')}\nLABEL: {ex['label']}"
            for ex in self.examples
        ]
        self.example_embs = self.ollama.embed(self.embedding_model, self.example_texts)
        self.example_embs = l2_normalize_rows(self.example_embs)

    def retrieve(self, term: str, context: str, top_k: int = 5) -> List[Dict[str, str]]:
        q_text = f"TERM: {term}\nCONTEXT: {context}"
        q_emb = self.ollama.embed(self.embedding_model, [q_text])
        q_emb = l2_normalize_rows(q_emb)[0]
        sims = cosine_similarity([q_emb], self.example_embs)[0]
        idxs = np.argsort(-sims)[:top_k]
        return [self.examples[i] for i in idxs]


def few_shot_predict_one_by_one(
    ollama: OllamaClient,
    llm_model: str,
    retriever: ExampleRetriever,
    context_retriever: SentenceContextRetriever,
    terms: List[Dict[str, str]],
    top_k_contexts: int = 10,
    top_k_examples: int = 5,
) -> List[Dict[str, Any]]:
    results = []

    for row in terms:
        ctx_pairs = context_retriever.retrieve(row["term"], top_k=top_k_contexts)
        context = " ".join(s for s, _ in ctx_pairs)
        retrieved_examples = retriever.retrieve(row["term"], context, top_k=min(top_k_examples, 3))
        target_rows = [{
            "id": row["id"],
            "term": row["term"],
            "context": context,
        }]
        prompt = build_few_shot_prompt(target_rows, retrieved_examples)
        raw = ollama.generate(
            llm_model,
            prompt,
            temperature=0.0,
            timeout=DEFAULT_GENERATE_TIMEOUT,
            retries=DEFAULT_GENERATE_RETRIES,
        )
        parsed = parse_tsv(raw)
        if parsed:
            _, _, label = parsed[0]
            if label not in LABELS_CANONICAL:
                label = "unknown"
        else:
            label = "unknown"

        results.append({
            "id": row["id"],
            "term": row["term"],
            "label": label,
            "contexts": [s for s, _ in ctx_pairs],
            "prompt": prompt,
            "raw_output": raw,
        })
    return results


# ============================================================
# OUTPUT WRITERS
# ============================================================
def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    safe_mkdir_for(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {}
            for fn in fieldnames:
                val = r.get(fn, "")
                if isinstance(val, (dict, list)):
                    val = json.dumps(val, ensure_ascii=False)
                out[fn] = val
            writer.writerow(out)


def write_tsv(path: str, rows: List[Dict[str, Any]]) -> None:
    safe_mkdir_for(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(f"{r['id']}\t{r['term']}\t{r['label']}\n")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="UAV term typing with Ollama")
    parser.add_argument("--terms-csv", default=DEFAULT_TERMS_CSV)
    # parser.add_argument("--context-mmd", default=DEFAULT_CONTEXT_MMD)
    parser.add_argument("--context-dir", default=DEFAULT_CONTEXT_DIR)
    parser.add_argument("--labels-json", default=DEFAULT_LABELS_JSON)
    parser.add_argument("--fewshot-csv", default=DEFAULT_FEWSHOT_CSV)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embed-models", nargs="+", default=DEFAULT_EMBED_MODELS)
    parser.add_argument("--context-embed-model", default=DEFAULT_CONTEXT_RETRIEVAL_MODEL)
    parser.add_argument("--top-k-contexts", type=int, default=DEFAULT_TOP_K_CONTEXTS)
    parser.add_argument("--top-k-examples", type=int, default=DEFAULT_TOP_K_FEWSHOT)
    parser.add_argument("--low-confidence-threshold", type=float, default=DEFAULT_LOW_CONFIDENCE_THRESHOLD)
    parser.add_argument("--run-zero-shot", action="store_true")
    parser.add_argument("--run-few-shot", action="store_true")
    parser.add_argument("--run-hybrid", action="store_true")
    parser.add_argument("--zero-csv-out", default=DEFAULT_ZEROSHOT_OUT_CSV)
    parser.add_argument("--zero-tsv-out", default=DEFAULT_ZEROSHOT_OUT_TSV)
    parser.add_argument("--few-csv-out", default=DEFAULT_FEWSHOT_OUT_CSV)
    parser.add_argument("--few-tsv-out", default=DEFAULT_FEWSHOT_OUT_TSV)
    parser.add_argument("--hybrid-csv-out", default=DEFAULT_HYBRID_OUT_CSV)
    parser.add_argument("--hybrid-tsv-out", default=DEFAULT_HYBRID_OUT_TSV)
    args = parser.parse_args()

    # Default behavior: run all if none explicitly chosen
    if not (args.run_zero_shot or args.run_few_shot or args.run_hybrid):
        args.run_zero_shot = True
        args.run_few_shot = True
        args.run_hybrid = True

    terms = read_terms_csv(args.terms_csv)
    if not terms:
        raise RuntimeError(f"No usable terms found in {args.terms_csv}")

    sentences = read_all_mmd_sentences(args.context_dir)
    if not sentences:
        raise RuntimeError(f"No usable context sentences found across .mmd files in {args.context_dir}")

    sentence_index = build_sentence_index("\n\n".join(sentences), normalize_term)
    ollama = OllamaClient()
    context_retriever = SentenceContextRetriever(
        sentence_index=sentence_index,
    )

    label_prototypes = load_label_prototypes(args.labels_json)

    if args.run_zero_shot or args.run_hybrid:
        zero_typer = ZeroShotEmbeddingTyper(
            ollama=ollama,
            embedding_models=args.embed_models,
            label_prototypes=label_prototypes,
            context_retriever=context_retriever,
            top_k_contexts=args.top_k_contexts,
            temperature=0.08,
        )
    else:
        zero_typer = None

    if args.run_few_shot or args.run_hybrid:
        fewshot_examples = read_fewshot_examples_csv(args.fewshot_csv)
        if not fewshot_examples:
            raise RuntimeError(f"No few-shot examples found in {args.fewshot_csv}")
        retriever = ExampleRetriever(
            ollama=ollama,
            embedding_model=args.context_embed_model,
            labeled_examples=fewshot_examples,
        )
    else:
        retriever = None

    if args.run_zero_shot:
        zero_results = zero_typer.predict_many(terms)
        write_csv(
            args.zero_csv_out,
            zero_results,
            fieldnames=["id", "term", "label", "confidence", "scores", "contexts", "per_model_scores"],
        )
        write_tsv(args.zero_tsv_out, zero_results)
        print(f"Wrote zero-shot CSV: {args.zero_csv_out}")
        print(f"Wrote zero-shot TSV: {args.zero_tsv_out}")

    if args.run_few_shot:
        few_results = few_shot_predict_one_by_one(
            ollama=ollama,
            llm_model=args.llm_model,
            retriever=retriever,
            context_retriever=context_retriever,
            terms=terms,
            top_k_contexts=args.top_k_contexts,
            top_k_examples=args.top_k_examples,
        )
        write_csv(
            args.few_csv_out,
            few_results,
            fieldnames=["id", "term", "label", "contexts", "prompt", "raw_output"],
        )
        write_tsv(args.few_tsv_out, few_results)
        print(f"Wrote few-shot CSV: {args.few_csv_out}")
        print(f"Wrote few-shot TSV: {args.few_tsv_out}")

    if args.run_hybrid:
        hybrid_rows = []
        for row in terms:
            z = zero_typer.predict_one(row["term"])
            final_label = z.label
            source = "zero_shot"
            prompt = None
            raw_output = None
            if z.confidence < args.low_confidence_threshold:
                ctx = " ".join(z.contexts)
                retrieved_examples = retriever.retrieve(row["term"], ctx, top_k=min(args.top_k_examples, 3))
                target_rows = [{"id": row["id"], "term": row["term"], "context": ctx}]
                prompt = build_few_shot_prompt(target_rows, retrieved_examples)
                raw_output = ollama.generate(
                    args.llm_model,
                    prompt,
                    temperature=0.0,
                    timeout=DEFAULT_GENERATE_TIMEOUT,
                    retries=DEFAULT_GENERATE_RETRIES,
                )
                parsed = parse_tsv(raw_output)
                if parsed:
                    _, _, llm_label = parsed[0]
                    if llm_label in LABELS_CANONICAL:
                        final_label = llm_label
                        source = "few_shot"
            hybrid_rows.append({
                "id": row["id"],
                "term": row["term"],
                "label": final_label,
                "source": source,
                "zero_shot_confidence": z.confidence,
                "zero_shot_scores": z.scores,
                "contexts": z.contexts,
                "prompt": prompt,
                "raw_output": raw_output,
            })

        write_csv(
            args.hybrid_csv_out,
            hybrid_rows,
            fieldnames=["id", "term", "label", "source", "zero_shot_confidence", "zero_shot_scores", "contexts", "prompt", "raw_output"],
        )
        write_tsv(args.hybrid_tsv_out, hybrid_rows)
        print(f"Wrote hybrid CSV: {args.hybrid_csv_out}")
        print(f"Wrote hybrid TSV: {args.hybrid_tsv_out}")


if __name__ == "__main__":
    main()