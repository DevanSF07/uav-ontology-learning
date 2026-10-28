#!/usr/bin/env python3
"""Generic multi-agent Qwen entity typing with top-5 context retrieval.

Pipeline
--------
1. Load retained terms, a source corpus, aliases, and entity-typing GT.
2. Retrieve five diverse corpus sentences for each term with MPNet.
3. Load the closed type vocabulary and official definitions from TYPE_DEFINITIONS.
   Add representative examples from eligible S1_TERM_TYPING rows; definitions are never rewritten.
4. Candidate-Type Agent reads ALL generated profiles and returns five candidates.
5. Primary Typing Agent chooses one candidate type or Uncertain.
6. A Critic Agent runs only for low-confidence / disagreement cases.
7. Evaluate candidate recall and final entity typing against GT.

The default evaluation mode is full for the 66-row S4 benchmark. Use heldout for a smaller
deterministic test split; type profiles then contain training examples only.

Expected local model service
----------------------------
Ollama with a Qwen instruct model, for example:
    ollama pull qwen2.5:7b-instruct
    ollama serve

Example
-------
python test_10_multiagent_qwen_entity_typing.py \
  --terms stage_outputs/07b_qwen_validation_yake_mpnet/best_qwen_retained.csv \
  --corpus data/combined_corpus_raw.txt \
  --ground-truth data/Ground_Truth.xlsx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


PROJECT_DIR = Path("/home/faujdar/onto_gen/ontology_experiment_framework")
DEFAULT_TERMS = PROJECT_DIR / "stage_outputs/07b_qwen_validation_yake_mpnet/best_qwen_retained.csv"
DEFAULT_CORPUS = PROJECT_DIR / "data/combined_corpus_raw.txt"
DEFAULT_GT = PROJECT_DIR / "data/Ground_Truth_S1_Term_Typing.xlsx"
DEFAULT_ALIAS_GT = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_OUTPUT = PROJECT_DIR / "stage_outputs/10_multiagent_qwen_entity_typing_final"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
UNCERTAIN = "Uncertain"
MANUAL_REVIEW = "needs_manual_review"


# ---------------------------------------------------------------------------
# Generic text utilities
# ---------------------------------------------------------------------------

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize(value: object) -> str:
    text = safe_text(value).casefold()
    for dash in ("‐", "‑", "‒", "–", "—"):
        text = text.replace(dash, "-")
    text = re.sub(r"\(s\)", "s", text)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"[^a-z0-9&/+ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def singularize_head(token: str) -> str:
    protected = {
        "uas", "uav", "rpas", "gps", "isr", "los", "blos", "mtow",
        "aircraft", "data", "analysis", "status",
    }
    if token in protected:
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 4 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def morph(value: object) -> str:
    tokens = normalize(value).split()
    if tokens:
        tokens[-1] = singularize_head(tokens[-1])
    return " ".join(tokens)


def stable_unit_interval(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        try:
            result = json.loads(fenced.group(1))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            ch = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(text[start : end + 1])
                            if isinstance(result, dict):
                                return result
                        except json.JSONDecodeError:
                            break
    raise ValueError("No valid JSON object found in model response")


def validate_confidence(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return min(1.0, max(0.0, x))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Ollama client with cache, retry, and audit log
# ---------------------------------------------------------------------------
class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        cache_dir: Path,
        request_log: Path,
        timeout: int,
        retries: int,
        temperature: float,
        seed: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache_dir = cache_dir
        self.request_log = request_log
        self.timeout = timeout
        self.retries = retries
        self.options = {"temperature": temperature, "seed": seed}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.live_calls = 0
        self.cache_hits = 0

    def check(self) -> None:
        req = urllib.request.Request(self.base_url + "/api/tags", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=min(10, self.timeout)) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama health check returned HTTP {response.status}")
        except Exception as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self.base_url}. Start 'ollama serve'. Error: {exc}"
            ) from exc

    def _cache_key(self, messages: list[dict[str, str]], stage: str) -> str:
        payload = {
            "model": self.model,
            "stage": stage,
            "messages": messages,
            "options": self.options,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def chat_json(
        self,
        messages: list[dict[str, str]],
        stage: str,
        validator,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        key = self._cache_key(messages, stage)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.is_file():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            parsed = validator(cached["parsed"])
            self.cache_hits += 1
            return parsed, {"cache_hit": True, "elapsed_seconds": 0.0, "key": key}

        last_error = ""
        working_messages = list(messages)
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            body = json.dumps({
                "model": self.model,
                "messages": working_messages,
                "stream": False,
                "format": "json",
                "options": self.options,
            }).encode("utf-8")
            request = urllib.request.Request(
                self.base_url + "/api/chat",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            raw = ""
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    envelope = json.loads(response.read().decode("utf-8"))
                raw = envelope.get("message", {}).get("content", "")
                parsed = validator(extract_json_object(raw))
                elapsed = time.perf_counter() - started
                audit = {
                    "timestamp_utc": now_utc(), "stage": stage, "model": self.model,
                    "cache_key": key, "attempt": attempt + 1, "elapsed_seconds": elapsed,
                    "messages": working_messages, "raw_response": raw, "parsed": parsed,
                }
                with self.request_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
                cache_file.write_text(
                    json.dumps({"raw": raw, "parsed": parsed}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self.live_calls += 1
                return parsed, {"cache_hit": False, "elapsed_seconds": elapsed, "key": key}
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.retries:
                    working_messages = messages + [{
                        "role": "user",
                        "content": (
                            "Your previous response was invalid. Return exactly one valid JSON object "
                            f"that follows the requested schema. Validation error: {last_error[:500]}"
                        ),
                    }]
                    time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"Qwen call failed after retries: {last_error}")


# ---------------------------------------------------------------------------
# Data discovery and loading
# ---------------------------------------------------------------------------
TERM_COLUMN_CANDIDATES = ["candidate", "term", "keyword", "phrase", "normalized_candidate"]
TYPE_COLUMN_CANDIDATES = [
    "entity_type", "expected_type", "type", "term_type", "ontological_type",
    "gold_type", "gt_type", "class",
]
GT_TERM_COLUMN_CANDIDATES = ["term_canonical", "term", "canonical", "entity", "mention", "label"]
ID_COLUMN_CANDIDATES = ["gt_id", "id", "term_id", "entity_id"]


def choose_column(columns: Iterable[str], candidates: list[str], purpose: str) -> str:
    cols = list(columns)
    lower = {c.casefold(): c for c in cols}
    for candidate in candidates:
        if candidate.casefold() in lower:
            return lower[candidate.casefold()]
    # Conservative fuzzy discovery by token containment.
    for col in cols:
        norm_col = normalize(col).replace(" ", "_")
        if purpose == "type" and "type" in norm_col and "concept" not in norm_col:
            return col
        if purpose == "term" and any(token in norm_col for token in ("term", "entity", "canonical")):
            return col
    raise ValueError(f"Could not detect {purpose} column. Available columns: {cols}")


def find_typing_sheet(path: Path, explicit_sheet: str | None) -> tuple[str, pd.DataFrame, str, str]:
    book = pd.ExcelFile(path, engine="openpyxl")
    candidates = [explicit_sheet] if explicit_sheet else book.sheet_names
    scored = []
    for sheet in candidates:
        if sheet not in book.sheet_names:
            raise ValueError(f"Sheet '{sheet}' not found. Available: {book.sheet_names}")
        frame = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        try:
            term_col = choose_column(frame.columns, GT_TERM_COLUMN_CANDIDATES, "term")
            type_col = choose_column(frame.columns, TYPE_COLUMN_CANDIDATES, "type")
        except ValueError:
            continue
        nonempty_types = frame[type_col].fillna("").astype(str).str.strip().ne("").sum()
        unique_types = frame[type_col].dropna().astype(str).str.strip().nunique()
        name_bonus = 1000 if re.search(r"s4|typing|entity.?type", sheet, flags=re.I) else 0
        scored.append((name_bonus + nonempty_types + 10 * min(unique_types, 50), sheet, frame, term_col, type_col))
    if not scored:
        raise ValueError(
            "Could not automatically find an entity-typing sheet. Use --typing-sheet, "
            "--gt-term-column, and --gt-type-column."
        )
    _, sheet, frame, term_col, type_col = max(scored, key=lambda x: x[0])
    return sheet, frame, term_col, type_col


def load_gt(
    path: Path,
    typing_sheet: str | None,
    gt_term_column: str | None,
    gt_type_column: str | None,
    include_review: bool,
) -> tuple[pd.DataFrame, str, str, str]:
    sheet, frame, detected_term, detected_type = find_typing_sheet(path, typing_sheet)
    term_col = gt_term_column or detected_term
    type_col = gt_type_column or detected_type
    if term_col not in frame.columns or type_col not in frame.columns:
        raise ValueError(f"Required columns not found in sheet {sheet}")
    frame = frame.copy()
    if not include_review and "review_status" in frame.columns:
        review_status = frame["review_status"].fillna("").astype(str).str.strip().str.casefold()
        frame = frame[review_status.eq("typed")].copy()
    if not include_review and "grounded_status_v2" in frame.columns:
        status = frame["grounded_status_v2"].fillna("").astype(str).str.casefold()
        frame = frame[status.ne(MANUAL_REVIEW)].copy()
    frame["gt_term"] = frame[term_col].map(safe_text)
    frame["gt_type"] = frame[type_col].map(safe_text)
    frame = frame[frame.gt_term.ne("") & frame.gt_type.ne("")].copy()
    frame["gt_norm"] = frame.gt_term.map(normalize)
    frame["gt_morph"] = frame.gt_term.map(morph)
    id_col = next((c for c in ID_COLUMN_CANDIDATES if c in frame.columns), None)
    frame["gt_id_eval"] = frame[id_col].map(safe_text) if id_col else [f"GT_{i:05d}" for i in range(len(frame))]
    frame = frame.drop_duplicates(["gt_norm", "gt_type"]).reset_index(drop=True)
    return frame, sheet, term_col, type_col


def load_aliases(path: Path, requested_sheet: str | None) -> pd.DataFrame:
    book = pd.ExcelFile(path, engine="openpyxl")
    sheet = requested_sheet
    if not sheet:
        sheet = next((s for s in book.sheet_names if re.search(r"s2|alias", s, re.I)), None)
    if not sheet:
        return pd.DataFrame(columns=["canonical", "alias", "canonical_norm", "alias_norm"])
    frame = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if not {"canonical", "alias"}.issubset(frame.columns):
        return pd.DataFrame(columns=["canonical", "alias", "canonical_norm", "alias_norm"])
    if "grounded_status_v2" in frame.columns:
        status = frame["grounded_status_v2"].fillna("").astype(str).str.casefold()
        frame = frame[status.ne(MANUAL_REVIEW)].copy()
    frame["canonical_norm"] = frame["canonical"].map(normalize)
    frame["alias_norm"] = frame["alias"].map(normalize)
    return frame[frame.canonical_norm.ne("") & frame.alias_norm.ne("")].copy()


def load_type_definitions(path: Path, sheet: str) -> list[dict[str, Any]]:
    book = pd.ExcelFile(path, engine="openpyxl")
    if sheet not in book.sheet_names:
        raise ValueError(f"Definitions sheet '{sheet}' not found. Available: {book.sheet_names}")
    frame = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    required = {"entity_type", "definition"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Definitions sheet must contain {sorted(required)}; found {list(frame.columns)}")
    frame = frame.copy()
    frame["entity_type"] = frame["entity_type"].map(safe_text)
    frame["definition"] = frame["definition"].map(safe_text)
    frame = frame[frame.entity_type.ne("") & frame.definition.ne("")].drop_duplicates("entity_type")
    return frame[["entity_type", "definition"]].to_dict("records")


def load_terms(path: Path, term_column: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    col = term_column or choose_column(frame.columns, TERM_COLUMN_CANDIDATES, "term")
    frame = frame.copy()
    frame["candidate"] = frame[col].map(safe_text)
    frame["candidate_norm"] = frame[col].map(normalize)
    frame["candidate_morph"] = frame[col].map(morph)
    return frame[frame.candidate_norm.ne("")].drop_duplicates("candidate_norm").reset_index(drop=True)


def build_alias_graph(aliases: pd.DataFrame) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for row in aliases.itertuples(index=False):
        graph[row.canonical_norm].add(row.alias_norm)
        graph[row.alias_norm].add(row.canonical_norm)
    return graph


def map_terms_to_gt(terms: pd.DataFrame, gt: pd.DataFrame, aliases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact: dict[str, list[pd.Series]] = defaultdict(list)
    morphological: dict[str, list[pd.Series]] = defaultdict(list)
    for _, row in gt.iterrows():
        exact[row.gt_norm].append(row)
        morphological[row.gt_morph].append(row)
    alias_graph = build_alias_graph(aliases)

    mapped_rows = []
    ambiguous_rows = []
    for _, term in terms.iterrows():
        candidates: list[tuple[pd.Series, str]] = []
        seen = set()
        for row in exact.get(term.candidate_norm, []):
            candidates.append((row, "exact")); seen.add(row.gt_id_eval)
        if not candidates:
            for row in morphological.get(term.candidate_morph, []):
                if row.gt_id_eval not in seen:
                    candidates.append((row, "morphological")); seen.add(row.gt_id_eval)
        if not candidates:
            labels = alias_graph.get(term.candidate_norm, set())
            for label in labels:
                for row in exact.get(label, []):
                    if row.gt_id_eval not in seen:
                        candidates.append((row, "alias")); seen.add(row.gt_id_eval)
        if not candidates:
            continue
        types = sorted({row.gt_type for row, _ in candidates})
        payload = term.to_dict()
        payload["matched_gt_ids"] = ";".join(row.gt_id_eval for row, _ in candidates)
        payload["matched_gt_terms"] = ";".join(row.gt_term for row, _ in candidates)
        payload["match_rules"] = ";".join(rule for _, rule in candidates)
        if len(types) != 1:
            payload["conflicting_gt_types"] = ";".join(types)
            ambiguous_rows.append(payload)
            continue
        payload["gt_type"] = types[0]
        payload["canonical_term"] = candidates[0][0].gt_term
        mapped_rows.append(payload)
    return pd.DataFrame(mapped_rows), pd.DataFrame(ambiguous_rows)


# ---------------------------------------------------------------------------
# Deterministic grouped splitting, profile examples, and type cards
# ---------------------------------------------------------------------------

def assign_split(key: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    x = stable_unit_interval(normalize(key), seed)
    if x < train_ratio:
        return "train"
    if x < train_ratio + val_ratio:
        return "validation"
    return "test"


def ensure_type_coverage(gt: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float) -> pd.DataFrame:
    out = gt.copy()
    out["split"] = out.gt_norm.map(lambda x: assign_split(x, seed, train_ratio, val_ratio))
    # For rare labels, deterministically ensure at least one training example.
    for entity_type, group in out.groupby("gt_type"):
        if not group.split.eq("train").any():
            chosen = sorted(group.index, key=lambda i: stable_unit_interval(out.loc[i, "gt_norm"], seed + 1))[0]
            out.loc[chosen, "split"] = "train"
    return out


def representative_examples(frame: pd.DataFrame, max_examples: int, seed: int) -> list[str]:
    values = frame.gt_term.dropna().astype(str).drop_duplicates().tolist()
    return sorted(values, key=lambda x: stable_unit_interval(x, seed))[:max_examples]


def profile_system_prompt() -> str:
    return (
        "You are an ontology annotation-policy analyst. Build a precise, domain-agnostic "
        "entity-type profile from labeled training examples. Infer the annotation boundary "
        "demonstrated by the examples. Never invent examples, never mention hidden test data, "
        "and do not map individual terms outside the supplied examples. Return JSON only."
    )


def build_profile_prompt(entity_type: str, examples: list[str], all_types: list[str]) -> str:
    return f"""Create a high-quality profile for one entity type.

TYPE NAME:
{entity_type}

LABELED TRAINING EXAMPLES:
{json.dumps(examples, ensure_ascii=False, indent=2)}

ALL TYPES IN THE SCHEMA:
{json.dumps(all_types, ensure_ascii=False)}

Write a precise profile that helps another agent distinguish this type from overlapping types.
The definition must describe what members of this type DENOTE, not merely list common words.
The inclusion and exclusion criteria must be operational and transferable to unseen domains.
Select representative_examples only from LABELED TRAINING EXAMPLES.
Choose confusable_types only from ALL TYPES IN THE SCHEMA and exclude the current type.
For each confusable type, explain one concise annotation-boundary distinction.
If the examples are sparse or heterogeneous, explicitly state that limitation in profile_notes.

Return exactly this JSON schema:
{{
  "type_name": {json.dumps(entity_type)},
  "definition": "one precise natural-language definition",
  "inclusion_criteria": ["criterion 1", "criterion 2", "criterion 3"],
  "exclusion_criteria": ["exclusion 1", "exclusion 2"],
  "representative_examples": ["example copied from supplied list"],
  "confusable_types": ["valid schema type"],
  "distinctions": {{"confusable type": "how annotation policy separates it"}},
  "profile_notes": "limitations or annotation-policy notes",
  "profile_confidence": 0.0
}}"""


def validate_profile(obj: dict[str, Any], entity_type: str, examples: set[str], all_types: set[str]) -> dict[str, Any]:
    if safe_text(obj.get("type_name")) != entity_type:
        raise ValueError("Profile type_name does not match requested type")
    definition = safe_text(obj.get("definition"))
    if len(definition) < 25:
        raise ValueError("Definition is too short")
    reps = [safe_text(x) for x in obj.get("representative_examples", []) if safe_text(x)]
    reps = [x for x in reps if x in examples]
    confusables = [safe_text(x) for x in obj.get("confusable_types", []) if safe_text(x)]
    confusables = [x for x in confusables if x in all_types and x != entity_type]
    distinctions = {
        safe_text(k): safe_text(v)
        for k, v in (obj.get("distinctions") or {}).items()
        if safe_text(k) in confusables and safe_text(v)
    }
    return {
        "type_name": entity_type,
        "definition": definition,
        "inclusion_criteria": [safe_text(x) for x in obj.get("inclusion_criteria", []) if safe_text(x)],
        "exclusion_criteria": [safe_text(x) for x in obj.get("exclusion_criteria", []) if safe_text(x)],
        "representative_examples": reps,
        "confusable_types": confusables,
        "distinctions": distinctions,
        "profile_notes": safe_text(obj.get("profile_notes")),
        "profile_confidence": validate_confidence(obj.get("profile_confidence"), 0.5),
    }


# ---------------------------------------------------------------------------
# Corpus sentence index and retrieval
# ---------------------------------------------------------------------------

def split_corpus(text: str) -> pd.DataFrame:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    section = "Document"
    rows = []
    buffer = []
    position = 0

    def flush(paragraph: str, section_name: str) -> None:
        nonlocal position
        paragraph = re.sub(r"\\begin\b.*?|\\end\b.*?|\\hline|\\cline\{.*?\}", " ", paragraph)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            return
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])", paragraph)
        for piece in pieces:
            sentence = piece.strip(" -\t")
            word_count = len(sentence.split())
            if 6 <= word_count <= 180 and not re.fullmatch(r"[\W\d_]+", sentence):
                rows.append({
                    "sentence_id": f"SENT_{position:06d}",
                    "section": section_name,
                    "sentence": sentence,
                    "sentence_norm": normalize(sentence),
                    "word_count": word_count,
                    "position": position,
                })
                position += 1

    for line in text.split("\n"):
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+", stripped):
            if buffer:
                flush(" ".join(buffer), section)
                buffer = []
            section = re.sub(r"^#{1,6}\s+", "", stripped).strip()
        elif not stripped:
            if buffer:
                flush(" ".join(buffer), section)
                buffer = []
        else:
            buffer.append(stripped)
    if buffer:
        flush(" ".join(buffer), section)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No usable corpus sentences were produced")
    return frame.drop_duplicates("sentence_norm").reset_index(drop=True)


def encode_cached(
    model: Any,
    texts: list[str],
    cache_path: Path,
    metadata_path: Path,
    signature: dict[str, Any],
    batch_size: int,
) -> np.ndarray:
    sig_hash = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature_hash") == sig_hash:
            embeddings = np.load(cache_path)
            if len(embeddings) == len(texts):
                return embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    np.save(cache_path, embeddings)
    metadata_path.write_text(
        json.dumps({"signature_hash": sig_hash, "signature": signature}, indent=2),
        encoding="utf-8",
    )
    return embeddings


def mention_bonus(term_norm: str, aliases: list[str], sentence_norm: str) -> tuple[float, str]:
    labels = [term_norm] + [normalize(x) for x in aliases]
    labels = [x for x in labels if x]
    for label in labels:
        pattern = r"(?<![a-z0-9])" + re.escape(label).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        if re.search(pattern, sentence_norm):
            return 1.0, label
    return 0.0, ""


def retrieve_contexts(
    terms: pd.DataFrame,
    aliases: pd.DataFrame,
    sentences: pd.DataFrame,
    sentence_embeddings: np.ndarray,
    model: Any,
    top_pool: int,
    top_k: int,
    mention_weight: float,
    diversity_threshold: float,
    batch_size: int,
) -> pd.DataFrame:
    alias_graph = build_alias_graph(aliases)
    query_texts = terms.canonical_term.tolist()
    query_embeddings = model.encode(
        query_texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype("float32")
    rows = []
    for term_idx, term in terms.reset_index(drop=True).iterrows():
        query = query_embeddings[term_idx]
        similarities = sentence_embeddings @ query
        pool_size = min(top_pool, len(similarities))
        indices = np.argpartition(-similarities, pool_size - 1)[:pool_size]
        records = []
        aliases_for_term = sorted(alias_graph.get(term.candidate_norm, set()))
        for idx in indices:
            sent = sentences.iloc[int(idx)]
            bonus, matched_label = mention_bonus(term.candidate_norm, aliases_for_term, sent.sentence_norm)
            quality = min(1.0, sent.word_count / 20.0) * min(1.0, 100.0 / sent.word_count)
            score = (1.0 - mention_weight - 0.05) * float(similarities[idx]) + mention_weight * bonus + 0.05 * quality
            records.append((score, float(similarities[idx]), int(idx), bonus, matched_label))
        records.sort(reverse=True)
        selected: list[tuple] = []
        for record in records:
            _, _, idx, _, _ = record
            if all(float(sentence_embeddings[idx] @ sentence_embeddings[old[2]]) < diversity_threshold for old in selected):
                selected.append(record)
            if len(selected) >= top_k:
                break
        # If diversity filtering is too strict, fill from remaining ranked items.
        selected_ids = {item[2] for item in selected}
        for record in records:
            if len(selected) >= top_k:
                break
            if record[2] not in selected_ids:
                selected.append(record); selected_ids.add(record[2])
        for rank, (score, cosine, idx, bonus, matched_label) in enumerate(selected, start=1):
            sent = sentences.iloc[idx]
            rows.append({
                "canonical_term": term.canonical_term,
                "candidate_norm": term.candidate_norm,
                "gt_type": term.gt_type,
                "sentence_rank": rank,
                "sentence_id": sent.sentence_id,
                "section": sent.section,
                "sentence": sent.sentence,
                "cosine_similarity": cosine,
                "mention_bonus": bonus,
                "matched_label": matched_label,
                "retrieval_score": score,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Agent prompts and validators
# ---------------------------------------------------------------------------

def compact_profile(profile: dict[str, Any], include_distinctions: bool = True) -> dict[str, Any]:
    result = {
        "type_name": profile["type_name"],
        "definition": profile["definition"],
        "inclusion_criteria": profile.get("inclusion_criteria", []),
        "exclusion_criteria": profile.get("exclusion_criteria", []),
        "representative_examples": profile.get("representative_examples", []),
    }
    if include_distinctions:
        result["distinctions"] = profile.get("distinctions", {})
    return result


def candidate_prompt(term: str, aliases: list[str], evidence: list[dict[str, str]], profiles: list[dict[str, Any]]) -> str:
    return f"""You are the Candidate-Type Retrieval Agent in a generic ontology-learning pipeline.
Your objective is HIGH RECALL: preserve the correct type in a shortlist of exactly five types.
Do not make the final classification.

TERM:
{term}

ALIASES:
{json.dumps(aliases, ensure_ascii=False)}

CORPUS EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

ALL TYPE PROFILES:
{json.dumps([compact_profile(p) for p in profiles], ensure_ascii=False, indent=2)}

First write a concise factual entity_description explaining what the TERM denotes in this evidence.
Then rank exactly five distinct type names from the profiles, from most to least plausible.
Use profile definitions, inclusion/exclusion boundaries, and labeled examples.
A type can be shortlisted even if not the most likely, when it remains reasonably plausible.
Do not invent type names. Do not classify the surrounding paragraph topic instead of the term.

Return exactly one JSON object:
{{
  "term": {json.dumps(term)},
  "entity_description": "concise evidence-grounded description",
  "evidence_quality": 0.0,
  "candidate_types": [
    {{"type": "valid type name", "plausibility": 0.0, "reason": "brief reason"}}
  ]
}}"""


def validate_candidate_result(obj: dict[str, Any], term: str, valid_types: set[str], shortlist_size: int) -> dict[str, Any]:
    description = safe_text(obj.get("entity_description"))
    if len(description) < 15:
        raise ValueError("entity_description is missing or too short")
    raw = obj.get("candidate_types")
    if not isinstance(raw, list):
        raise ValueError("candidate_types must be a list")
    candidates = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        entity_type = safe_text(item.get("type"))
        if entity_type in valid_types and entity_type not in seen:
            candidates.append({
                "type": entity_type,
                "plausibility": validate_confidence(item.get("plausibility"), 0.0),
                "reason": safe_text(item.get("reason")),
            })
            seen.add(entity_type)
    if len(candidates) != shortlist_size:
        raise ValueError(f"Expected exactly {shortlist_size} valid distinct candidate types, got {len(candidates)}")
    return {
        "term": term,
        "entity_description": description,
        "evidence_quality": validate_confidence(obj.get("evidence_quality"), 0.5),
        "candidate_types": candidates,
    }


def primary_prompt(
    term: str,
    aliases: list[str],
    evidence: list[dict[str, str]],
    candidate_result: dict[str, Any],
    profile_map: dict[str, dict[str, Any]],
) -> str:
    selected = [profile_map[x["type"]] for x in candidate_result["candidate_types"]]
    return f"""You are the Primary Entity-Typing Agent. Make the final fine-grained decision under the supplied annotation policy.

TERM:
{term}

ALIASES:
{json.dumps(aliases, ensure_ascii=False)}

ENTITY DESCRIPTION FROM SCREENING AGENT:
{candidate_result['entity_description']}

CORPUS EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

FIVE CANDIDATE TYPE PROFILES:
{json.dumps([compact_profile(p) for p in selected], ensure_ascii=False, indent=2)}

SCREENING RANKING:
{json.dumps(candidate_result['candidate_types'], ensure_ascii=False, indent=2)}

Instructions:
1. Classify what the TERM itself denotes in the evidence, not the paragraph's topic.
2. Follow the annotation boundaries demonstrated by definitions and training examples.
3. Compare all five candidates and explicitly distinguish the winner from the runner-up.
4. Choose exactly one of the five candidate types, or {UNCERTAIN} if evidence is insufficient.
5. Cite only sentence IDs that actually support the decision.
6. Confidence must reflect uncertainty and evidence quality, not rhetorical certainty.

Return exactly one JSON object:
{{
  "term": {json.dumps(term)},
  "predicted_type": "one candidate type or {UNCERTAIN}",
  "confidence": 0.0,
  "runner_up_type": "another candidate type or empty string",
  "supporting_sentence_ids": ["SENT_..."],
  "key_distinction": "why winner is better than runner-up",
  "evidence_supported": true
}}"""


def validate_primary_result(
    obj: dict[str, Any],
    term: str,
    candidates: set[str],
    sentence_ids: set[str],
) -> dict[str, Any]:
    predicted = safe_text(obj.get("predicted_type"))
    if predicted not in candidates | {UNCERTAIN}:
        raise ValueError("predicted_type is not in candidate shortlist")
    runner = safe_text(obj.get("runner_up_type"))
    if runner and runner not in candidates:
        raise ValueError("runner_up_type is not in candidate shortlist")
    supports = [safe_text(x) for x in obj.get("supporting_sentence_ids", []) if safe_text(x) in sentence_ids]
    return {
        "term": term,
        "predicted_type": predicted,
        "confidence": validate_confidence(obj.get("confidence"), 0.0),
        "runner_up_type": runner,
        "supporting_sentence_ids": supports,
        "key_distinction": safe_text(obj.get("key_distinction")),
        "evidence_supported": bool(obj.get("evidence_supported", False)),
    }


def critic_prompt(
    term: str,
    evidence: list[dict[str, str]],
    candidate_result: dict[str, Any],
    primary: dict[str, Any],
    profile_map: dict[str, dict[str, Any]],
) -> str:
    possibilities = []
    for x in [primary["predicted_type"], primary["runner_up_type"], candidate_result["candidate_types"][0]["type"]]:
        if x and x != UNCERTAIN and x not in possibilities:
            possibilities.append(x)
    possibilities = possibilities[:3]
    profiles = [profile_map[x] for x in possibilities]
    return f"""You are the Critic Agent. Resolve an ambiguous entity-typing decision.

TERM: {term}
ENTITY DESCRIPTION: {candidate_result['entity_description']}
EVIDENCE: {json.dumps(evidence, ensure_ascii=False, indent=2)}
PRIMARY RESULT: {json.dumps(primary, ensure_ascii=False, indent=2)}
SCREENING RESULT: {json.dumps(candidate_result['candidate_types'], ensure_ascii=False, indent=2)}
DISPUTED TYPE PROFILES: {json.dumps([compact_profile(p) for p in profiles], ensure_ascii=False, indent=2)}

Re-examine the denotation of the term and the annotation boundaries.
Choose one disputed type or {UNCERTAIN}. Do not introduce a new type.
Return JSON only:
{{
  "recommended_type": "one disputed type or {UNCERTAIN}",
  "confidence": 0.0,
  "primary_supported": true,
  "reason": "concise evidence-grounded resolution",
  "supporting_sentence_ids": ["SENT_..."]
}}"""


def validate_critic_result(obj: dict[str, Any], possibilities: set[str], sentence_ids: set[str]) -> dict[str, Any]:
    recommendation = safe_text(obj.get("recommended_type"))
    if recommendation not in possibilities | {UNCERTAIN}:
        raise ValueError("Critic recommendation is not an allowed disputed type")
    return {
        "recommended_type": recommendation,
        "confidence": validate_confidence(obj.get("confidence"), 0.0),
        "primary_supported": bool(obj.get("primary_supported", False)),
        "reason": safe_text(obj.get("reason")),
        "supporting_sentence_ids": [
            safe_text(x) for x in obj.get("supporting_sentence_ids", []) if safe_text(x) in sentence_ids
        ],
    }


# ---------------------------------------------------------------------------
# Evaluation and plotting
# ---------------------------------------------------------------------------

def evaluate_results(results: pd.DataFrame, all_types: list[str], output_dir: Path) -> dict[str, Any]:
    scored = results[results.final_type.ne(UNCERTAIN)].copy()
    y_true_all = results.gt_type.tolist()
    y_pred_all = results.final_type.tolist()
    coverage = len(scored) / len(results) if len(results) else 0.0
    selective_accuracy = accuracy_score(scored.gt_type, scored.final_type) if len(scored) else 0.0
    overall_accuracy = sum(a == b for a, b in zip(y_true_all, y_pred_all)) / len(results) if len(results) else 0.0
    macro_f1 = f1_score(y_true_all, y_pred_all, labels=all_types, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_all, y_pred_all, labels=all_types, average="weighted", zero_division=0)

    candidate_top = {}
    for k in (1, 3, 5):
        candidate_top[f"candidate_top_{k}_recall"] = float(
            results.apply(lambda r: r.gt_type in r.candidate_types_list[:k], axis=1).mean()
        ) if len(results) else 0.0
    reciprocal_ranks = []
    for row in results.itertuples(index=False):
        try:
            reciprocal_ranks.append(1.0 / (row.candidate_types_list.index(row.gt_type) + 1))
        except ValueError:
            reciprocal_ranks.append(0.0)

    overall = {
        "evaluable_terms": len(results),
        "non_abstained_terms": len(scored),
        "coverage": coverage,
        "abstention_rate": 1.0 - coverage,
        "overall_accuracy_uncertain_is_wrong": overall_accuracy,
        "selective_accuracy": selective_accuracy,
        "macro_f1_uncertain_is_wrong": macro_f1,
        "weighted_f1_uncertain_is_wrong": weighted_f1,
        "candidate_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        **candidate_top,
        "critic_invocation_rate": float(results.critic_invoked.mean()) if len(results) else 0.0,
        "average_llm_calls_per_term": float(results.llm_calls.mean()) if len(results) else 0.0,
        "average_runtime_seconds_per_term": float(results.runtime_seconds.mean()) if len(results) else 0.0,
    }
    pd.DataFrame([overall]).to_csv(output_dir / "10_overall_metrics.csv", index=False)

    p, r, f, support = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=all_types, zero_division=0
    )
    per_type = pd.DataFrame({
        "entity_type": all_types, "precision": p, "recall": r, "f1": f, "support": support
    })
    per_type.to_csv(output_dir / "11_per_type_metrics.csv", index=False)

    matrix = confusion_matrix(y_true_all, y_pred_all, labels=all_types + [UNCERTAIN])
    matrix_frame = pd.DataFrame(matrix, index=all_types + [UNCERTAIN], columns=all_types + [UNCERTAIN])
    matrix_frame.to_csv(output_dir / "12_confusion_matrix.csv")
    fig_size = max(12, len(all_types) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, interpolation="nearest")
    ax.set_title("Entity Typing Confusion Matrix")
    ax.set_xlabel("Predicted type")
    ax.set_ylabel("Ground-truth type")
    ax.set_xticks(range(len(all_types) + 1), all_types + [UNCERTAIN], rotation=90, fontsize=7)
    ax.set_yticks(range(len(all_types) + 1), all_types + [UNCERTAIN], fontsize=7)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "13_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    errors = results[results.final_type.ne(results.gt_type)].copy()
    errors.to_csv(output_dir / "14_error_analysis.csv", index=False)
    results[results.final_type.eq(UNCERTAIN)].to_csv(output_dir / "15_manual_review_queue.csv", index=False)
    return overall


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--terms", type=Path, default=DEFAULT_TERMS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--term-column", default=None)
    parser.add_argument("--typing-sheet", default="S1_TERM_TYPING")
    parser.add_argument("--gt-term-column", default="term_canonical")
    parser.add_argument("--gt-type-column", default="entity_type")
    parser.add_argument("--alias-workbook", type=Path, default=DEFAULT_ALIAS_GT)
    parser.add_argument("--alias-sheet", default="S2_GT_ALIASES")
    parser.add_argument("--definitions-sheet", default="TYPE_DEFINITIONS")
    parser.add_argument("--include-review-gt", action="store_true")
    parser.add_argument("--evaluation-mode", choices=("heldout", "full"), default="full")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--retrieval-pool", type=int, default=20)
    parser.add_argument("--top-contexts", type=int, default=5)
    parser.add_argument("--mention-weight", type=float, default=0.25)
    parser.add_argument("--diversity-threshold", type=float, default=0.90)
    parser.add_argument("--shortlist-size", type=int, default=5)
    parser.add_argument("--profile-examples", type=int, default=12)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--primary-confidence-threshold", type=float, default=0.82)
    parser.add_argument("--evidence-quality-threshold", type=float, default=0.55)
    parser.add_argument("--max-terms", type=int, default=0, help="Pilot limit; 0 means all evaluable terms")
    parser.add_argument("--skip-ollama-check", action="store_true")
    parser.add_argument("--rebuild-profiles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    for path in (args.terms, args.corpus, args.ground_truth, args.alias_workbook):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0 < args.train_ratio < 1 or not 0 <= args.validation_ratio < 1:
        raise ValueError("Invalid train/validation ratios")
    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("train-ratio + validation-ratio must be below 1")
    if args.shortlist_size < 2:
        raise ValueError("shortlist-size must be at least 2")

    client = OllamaClient(
        args.ollama_url, args.model, cache_dir / "qwen", args.output_dir / "qwen_requests.jsonl",
        args.timeout, args.retries, args.temperature, args.seed,
    )
    if not args.skip_ollama_check:
        client.check()

    print("Loading ground truth and retained terms...")
    gt, typing_sheet, gt_term_col, gt_type_col = load_gt(
        args.ground_truth, args.typing_sheet, args.gt_term_column, args.gt_type_column,
        args.include_review_gt,
    )
    aliases = load_aliases(args.alias_workbook, args.alias_sheet)
    terms = load_terms(args.terms, args.term_column)
    official_definitions = load_type_definitions(args.ground_truth, args.definitions_sheet)
    # Ground truth is attached only for evaluation. Every retained Qwen term
    # remains in the application population and is entity-typed.
    mapped, ambiguous = map_terms_to_gt(terms, gt, aliases)
    ambiguous.to_csv(args.output_dir / "ambiguous_gt_mappings.csv", index=False)

    gt = ensure_type_coverage(gt, args.seed, args.train_ratio, args.validation_ratio)
    split_map = gt.drop_duplicates("gt_norm").set_index("gt_norm")["split"].to_dict()
    if not mapped.empty:
        mapped["split"] = mapped.canonical_term.map(normalize).map(split_map).fillna("test")

    # Build one row per Qwen-retained term. Mapped terms receive GT metadata;
    # unmatched and ambiguous terms still proceed through all typing agents.
    mapped_by_candidate = (
        mapped.drop_duplicates("candidate_norm").set_index("candidate_norm").to_dict("index")
        if not mapped.empty else {}
    )
    processing_rows = []
    for _, term in terms.iterrows():
        payload = term.to_dict()
        gt_payload = mapped_by_candidate.get(term.candidate_norm)
        if gt_payload:
            payload.update({
                "canonical_term": gt_payload.get("canonical_term", term.candidate),
                "gt_type": gt_payload.get("gt_type", ""),
                "split": gt_payload.get("split", ""),
                "gt_matched": True,
                "matched_gt_ids": gt_payload.get("matched_gt_ids", ""),
                "matched_gt_terms": gt_payload.get("matched_gt_terms", ""),
                "match_rules": gt_payload.get("match_rules", ""),
            })
        else:
            payload.update({
                "canonical_term": term.candidate,
                "gt_type": "",
                "split": "",
                "gt_matched": False,
                "matched_gt_ids": "",
                "matched_gt_terms": "",
                "match_rules": "",
            })
        processing_rows.append(payload)
    terms_to_process = pd.DataFrame(processing_rows)

    if args.evaluation_mode == "heldout":
        profile_gt = gt[gt.split.eq("train")].copy()
    else:
        profile_gt = gt.copy()

    if args.max_terms > 0:
        terms_to_process = (
            terms_to_process.sort_values("canonical_term").head(args.max_terms).copy()
        )
    gt_types = set(gt.gt_type.unique().tolist())
    definition_map = {row["entity_type"]: row["definition"] for row in official_definitions}
    missing_definitions = sorted(gt_types - set(definition_map))
    if missing_definitions:
        raise ValueError(f"GT entity types missing from TYPE_DEFINITIONS: {missing_definitions}")
    all_types = [row["entity_type"] for row in official_definitions if row["entity_type"] in gt_types]
    if len(all_types) < args.shortlist_size:
        raise ValueError("shortlist-size exceeds number of GT types")

    terms_to_process.to_csv(args.output_dir / "01_all_qwen_terms_for_typing.csv", index=False)
    mapped.to_csv(args.output_dir / "01_gt_mapped_terms.csv", index=False)
    gt.to_csv(args.output_dir / "gt_splits.csv", index=False)

    print(f"Typing sheet: {typing_sheet} | term={gt_term_col} | type={gt_type_col}")
    print(
        f"GT types: {len(all_types)} | Qwen terms to type: {len(terms_to_process)} | "
        f"GT-mapped terms available for evaluation: {len(mapped)}"
    )

    # Sentence index and embeddings.
    print("Building corpus sentence index and embeddings...")
    corpus_text = args.corpus.read_text(encoding="utf-8", errors="replace")
    sentences = split_corpus(corpus_text)
    sentences.to_csv(args.output_dir / "02_corpus_sentences.csv", index=False)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency sentence-transformers. Install with: "
            "python -m pip install -U sentence-transformers"
        ) from exc
    embedding_model = SentenceTransformer(args.embedding_model)
    signature = {
        "corpus_sha256": hashlib.sha256(corpus_text.encode("utf-8")).hexdigest(),
        "embedding_model": args.embedding_model,
        "sentence_count": len(sentences),
    }
    sentence_embeddings = encode_cached(
        embedding_model, sentences.sentence.tolist(),
        cache_dir / "corpus_sentence_embeddings.npy",
        cache_dir / "corpus_sentence_embeddings.json",
        signature, args.embedding_batch_size,
    )
    contexts = retrieve_contexts(
        terms_to_process, aliases, sentences, sentence_embeddings, embedding_model,
        args.retrieval_pool, args.top_contexts, args.mention_weight,
        args.diversity_threshold, args.embedding_batch_size,
    )
    contexts.to_csv(args.output_dir / "03_retrieved_contexts.csv", index=False)

    # Build type profiles from authoritative TYPE_DEFINITIONS and GT examples.
    # Definitions are never generated or rewritten by Qwen.
    profile_path = args.output_dir / "04_type_profiles.json"
    profiles = []
    print("Building type profiles from authoritative TYPE_DEFINITIONS...")
    for idx, entity_type in enumerate(all_types, start=1):
        examples = representative_examples(
            profile_gt[profile_gt.gt_type.eq(entity_type)], args.profile_examples, args.seed + idx
        )
        profiles.append({
            "type_name": entity_type,
            "definition": definition_map[entity_type],
            "inclusion_criteria": [],
            "exclusion_criteria": [],
            "representative_examples": examples,
            "confusable_types": [],
            "distinctions": {},
            "profile_notes": "Official definition from TYPE_DEFINITIONS; examples from eligible GT rows.",
            "profile_confidence": 1.0,
        })
        print(f"  profile {idx}/{len(all_types)}: {entity_type}")
    profile_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    profile_map = {p["type_name"]: p for p in profiles}
    if set(profile_map) != set(all_types):
        raise ValueError("Type profile names do not match the authoritative definition vocabulary")

    alias_graph = build_alias_graph(aliases)
    partial_path = args.output_dir / "typing_results.partial.jsonl"
    completed = {}
    if partial_path.is_file():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                completed[record["candidate_norm"]] = record

    candidate_records = []
    primary_records = []
    critic_records = []
    final_records = []

    print("=" * 100)
    print("Running strict sequential term-by-term entity typing")
    print("=" * 100)
    print("Qwen terms per prompt: 1")
    print("Concurrent Qwen workers: 1")
    print("Processing order: Candidate Agent -> Primary Agent -> optional Critic -> durable save")
    print("Each completed term is flushed to disk before the next term begins.")
    print()
    for number, (_, term_row) in enumerate(terms_to_process.reset_index(drop=True).iterrows(), start=1):
        key = term_row.candidate_norm
        print()
        print("-" * 100)
        print(f"TERM {number}/{len(terms_to_process)}: {term_row.canonical_term}")
        print("-" * 100)
        if key in completed:
            final_records.append(completed[key])
            print("  Resume cache: matching completed result loaded; no Qwen call required.")
            continue
        term_started = time.perf_counter()
        term_context = contexts[contexts.candidate_norm.eq(key)].sort_values("sentence_rank")
        evidence = [
            {"sentence_id": row.sentence_id, "section": row.section, "text": row.sentence}
            for row in term_context.itertuples(index=False)
        ]
        sentence_ids = {x["sentence_id"] for x in evidence}
        term_aliases = sorted(alias_graph.get(key, set()))
        llm_calls = 0

        print("  Stage 1/3: Candidate-Type Agent")
        candidate_messages = [
            {"role": "system", "content": (
                "You are a high-recall candidate-type retrieval agent. Use only the supplied "
                "type profiles and evidence. Return valid JSON only."
            )},
            {"role": "user", "content": candidate_prompt(
                term_row.canonical_term, term_aliases, evidence, profiles
            )},
        ]
        candidate_result, candidate_meta = client.chat_json(
            candidate_messages,
            stage=f"candidate::{key}",
            validator=lambda obj, t=term_row.canonical_term: validate_candidate_result(
                obj, t, set(all_types), args.shortlist_size
            ),
        )
        llm_calls += 0 if candidate_meta["cache_hit"] else 1
        candidate_types = [x["type"] for x in candidate_result["candidate_types"]]

        print("  Stage 2/3: Primary Typing Agent")
        primary_messages = [
            {"role": "system", "content": (
                "You are a precise entity-typing agent. Follow the supplied annotation profiles, "
                "classify the term itself, and return valid JSON only."
            )},
            {"role": "user", "content": primary_prompt(
                term_row.canonical_term, term_aliases, evidence, candidate_result, profile_map
            )},
        ]
        primary, primary_meta = client.chat_json(
            primary_messages,
            stage=f"primary::{key}",
            validator=lambda obj, t=term_row.canonical_term, cs=set(candidate_types): validate_primary_result(
                obj, t, cs, sentence_ids
            ),
        )
        llm_calls += 0 if primary_meta["cache_hit"] else 1

        candidate_rank1 = candidate_types[0]
        critic_needed = (
            primary["predicted_type"] == UNCERTAIN
            or primary["confidence"] < args.primary_confidence_threshold
            or candidate_result["evidence_quality"] < args.evidence_quality_threshold
            or primary["predicted_type"] != candidate_rank1
            or not primary["evidence_supported"]
        )
        critic = None
        final_type = primary["predicted_type"]
        final_confidence = primary["confidence"]
        decision_path = "candidate_agent->primary_agent"
        if critic_needed:
            print("  Stage 3/3: Critic Agent required")
            # The critic may legitimately prefer any type from the Candidate Agent
            # shortlist, not only the primary label, runner-up, and rank-1 label.
            # Restricting validation to three labels caused valid critic outputs to
            # abort the entire 564-term run.
            possibilities = {
                x for x in candidate_types
                if x and x != UNCERTAIN
            }
            possibilities.update(
                x for x in [primary["predicted_type"], primary["runner_up_type"], candidate_rank1]
                if x and x != UNCERTAIN
            )
            critic_messages = [
                {"role": "system", "content": (
                    "You are an entity-typing critic. Resolve only the supplied disputed labels "
                    "using evidence and annotation profiles. Return JSON only."
                )},
                {"role": "user", "content": critic_prompt(
                    term_row.canonical_term, evidence, candidate_result, primary, profile_map
                )},
            ]
            try:
                critic, critic_meta = client.chat_json(
                    critic_messages,
                    stage=f"critic::{key}",
                    validator=lambda obj, poss=possibilities: validate_critic_result(obj, poss, sentence_ids),
                )
                llm_calls += 0 if critic_meta["cache_hit"] else 1
                final_type = critic["recommended_type"]
                final_confidence = critic["confidence"]
                decision_path += "->critic_agent"
                critic_records.append({
                    "canonical_term": term_row.canonical_term,
                    "candidate_norm": key,
                    **critic,
                })
            except RuntimeError as critic_error:
                # A malformed/out-of-scope critic response must not destroy a long
                # sequential run. Preserve the validated Primary Agent decision,
                # record the fallback, durably save the term, and continue.
                print(f"  Critic failed validation; using Primary Agent fallback: {critic_error}")
                critic = {
                    "recommended_type": primary["predicted_type"],
                    "confidence": primary["confidence"],
                    "reason": f"Critic validation failed; primary fallback used. {critic_error}",
                    "supporting_sentence_ids": primary["supporting_sentence_ids"],
                }
                final_type = primary["predicted_type"]
                final_confidence = primary["confidence"]
                decision_path += "->critic_failed->primary_fallback"
                critic_records.append({
                    "canonical_term": term_row.canonical_term,
                    "candidate_norm": key,
                    "fallback_used": True,
                    **critic,
                })
        else:
            print("  Stage 3/3: Critic Agent skipped")

        runtime = time.perf_counter() - term_started
        record = {
            "canonical_term": term_row.canonical_term,
            "candidate": term_row.candidate,
            "candidate_norm": key,
            "aliases": term_aliases,
            "gt_type": term_row.gt_type,
            "split": term_row.split,
            "entity_description": candidate_result["entity_description"],
            "evidence_quality": candidate_result["evidence_quality"],
            "candidate_types_list": candidate_types,
            "candidate_types_json": candidate_result["candidate_types"],
            "primary_type": primary["predicted_type"],
            "primary_confidence": primary["confidence"],
            "runner_up_type": primary["runner_up_type"],
            "primary_key_distinction": primary["key_distinction"],
            "supporting_sentence_ids": primary["supporting_sentence_ids"],
            "critic_invoked": critic_needed,
            "critic_type": critic["recommended_type"] if critic else "",
            "critic_confidence": critic["confidence"] if critic else np.nan,
            "final_type": final_type,
            "final_confidence": final_confidence,
            "decision_path": decision_path,
            "gt_matched": bool(term_row.gt_matched),
            "correct": (final_type == term_row.gt_type) if term_row.gt_matched else np.nan,
            "llm_calls": llm_calls,
            "runtime_seconds": runtime,
        }
        final_records.append(record)
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        candidate_records.append({
            "canonical_term": term_row.canonical_term, "candidate_norm": key,
            "gt_type": term_row.gt_type, **candidate_result,
        })
        primary_records.append({
            "canonical_term": term_row.canonical_term, "candidate_norm": key,
            "gt_type": term_row.gt_type, **primary,
        })
        print(
            f"  Completed and durably saved: {term_row.canonical_term} -> {final_type} "
            f"(GT={term_row.gt_type if term_row.gt_matched else 'N/A'}, "
            f"correct={(final_type == term_row.gt_type) if term_row.gt_matched else 'N/A'}, "
            f"critic={critic_needed}, runtime={runtime:.2f}s)"
        )

    # Normalize list/dict columns for CSV while retaining an internal list for metrics.
    final_frame = pd.DataFrame(final_records)
    if final_frame.empty:
        raise ValueError("No typing results were produced")
    def ensure_list(value):
        if isinstance(value, list): return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return []
    final_frame["candidate_types_list"] = final_frame["candidate_types_list"].map(ensure_list)

    # Application output: every Qwen-retained term, including terms without GT.
    all_terms_frame = final_frame.copy()

    # Evaluation output: only unambiguous GT matches, and only test rows in
    # held-out mode. Ground truth never controls which terms are classified.
    metric_frame = final_frame[final_frame.gt_matched.eq(True)].copy()
    if args.evaluation_mode == "heldout":
        metric_frame = metric_frame[metric_frame.split.eq("test")].copy()

    export_all = all_terms_frame.copy()
    export_eval = metric_frame.copy()
    for col in ["aliases", "candidate_types_list", "candidate_types_json", "supporting_sentence_ids"]:
        if col in export_all:
            export_all[col] = export_all[col].map(json_dump)
        if col in export_eval:
            export_eval[col] = export_eval[col].map(json_dump)

    export_all.to_csv(args.output_dir / "08_all_qwen_terms_with_entity_types.csv", index=False)
    # Backward-compatible all-terms alias.
    export_all.to_csv(args.output_dir / "08_final_entity_types.csv", index=False)
    export_eval.to_csv(args.output_dir / "09_typing_results_with_gt.csv", index=False)
    pd.DataFrame(candidate_records).to_json(
        args.output_dir / "05_candidate_type_results.jsonl", orient="records", lines=True, force_ascii=False
    )
    pd.DataFrame(primary_records).to_json(
        args.output_dir / "06_primary_typing_results.jsonl", orient="records", lines=True, force_ascii=False
    )
    pd.DataFrame(critic_records).to_json(
        args.output_dir / "07_critic_results.jsonl", orient="records", lines=True, force_ascii=False
    )

    if metric_frame.empty:
        overall = {
            "evaluable_terms": 0,
            "note": "No GT-matched terms were present in this processing subset.",
        }
        pd.DataFrame([overall]).to_csv(args.output_dir / "10_overall_metrics.csv", index=False)
    else:
        overall = evaluate_results(metric_frame, all_types, args.output_dir)
    runtime_total = time.perf_counter() - started
    runtime_summary = pd.DataFrame([{
        "runtime_seconds": runtime_total,
        "live_qwen_calls": client.live_calls,
        "qwen_cache_hits": client.cache_hits,
        "typed_qwen_terms": len(all_terms_frame),
        "evaluated_terms": len(metric_frame),
        "average_seconds_per_typed_term": runtime_total / len(all_terms_frame) if len(all_terms_frame) else 0,
    }])
    runtime_summary.to_csv(args.output_dir / "16_runtime_summary.csv", index=False)

    manifest = {
        "created_utc": now_utc(),
        "terms": str(args.terms), "corpus": str(args.corpus), "ground_truth": str(args.ground_truth),
        "alias_workbook": str(args.alias_workbook), "definitions_sheet": args.definitions_sheet,
        "typing_sheet": typing_sheet, "gt_term_column": gt_term_col, "gt_type_column": gt_type_col,
        "evaluation_mode": args.evaluation_mode,
        "warning": (
            "Held-out profiles use training examples only."
            if args.evaluation_mode == "heldout"
            else "Exploratory in-sample mode: profiles and evaluation may share GT examples."
        ),
        "model": args.model, "embedding_model": args.embedding_model,
        "all_types": all_types, "profile_count": len(profiles),
        "qwen_terms_typed": len(all_terms_frame),
        "mapped_terms": len(mapped), "evaluable_terms": len(metric_frame),
        "parameters": vars(args) | {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
        "overall_metrics": overall,
        "runtime_seconds": runtime_total,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print("=" * 100)
    print("Multi-agent entity typing completed for all Qwen-retained terms")
    print("=" * 100)
    for key, value in overall.items():
        if isinstance(value, float):
            print(f"{key:40s}: {value:.4f}")
        else:
            print(f"{key:40s}: {value}")
    print(f"All Qwen terms typed: {len(all_terms_frame)}")
    print(f"GT-matched terms evaluated: {len(metric_frame)}")
    print(f"Live Qwen calls: {client.live_calls}; cache hits: {client.cache_hits}")
    print(f"Runtime: {runtime_total:.2f}s")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
