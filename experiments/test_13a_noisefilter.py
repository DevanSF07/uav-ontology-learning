#!/usr/bin/env python3
"""Corpus Presence and Noise Filter for taxonomy-node candidates.

Purpose
-------
Filter terms predicted as ``System`` or ``Subsystem`` before taxonomy
canonicalization, embeddings, clustering, or parent-child induction.

The program uses a precision-first cascade:

1. normalized exact corpus occurrence
2. conservative singular/plural occurrence
3. explicit alias/acronym occurrence from S2_GT_ALIASES
4. controlled compound reconstruction
5. one-term-at-a-time Qwen review for unresolved candidates

No semantic similarity alone can prove corpus presence.  Generated structural
nodes are kept separate from corpus-grounded nodes.

Default inputs match the current UAV ontology-learning project layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR

DEFAULT_TYPING_RESULTS = (
    PROJECT_DIR
    / "stage_outputs/10_entity_typing_all_qwen_terms/08_all_qwen_terms_with_entity_types.csv"
)
DEFAULT_CORPUS = PROJECT_DIR / "data/combined_corpus_raw.txt"
DEFAULT_ALIAS_WORKBOOK = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "stage_outputs/13a_corpus_presence_noise_filter"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

PROMPT_VERSION = "corpus-presence-noise-filter-v1"
ALLOWED_LLM_DECISIONS = {
    "accept_variant",
    "accept_alias",
    "accept_reconstructed",
    "reject_not_present",
    "reject_fragment",
    "reject_artifact",
    "reject_context_only",
    "manual_review",
}
ACCEPTED_STATUSES = {
    "exact_present",
    "morphological_present",
    "alias_present",
    "corpus_reconstructed",
}
REJECTED_STATUSES = {
    "not_present",
    "fragment",
    "artifact",
    "context_only",
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def normalize(text: Any) -> str:
    """Normalize only superficial differences; do not stem technical terms."""
    value = unicodedata.normalize("NFKC", safe_text(text)).casefold()
    value = value.replace("–", "-").replace("—", "-").replace("−", "-")
    value = re.sub(r"[‐‑‒]", "-", value)
    value = re.sub(r"[-_/]+", " ", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenise(text: Any) -> list[str]:
    return normalize(text).split()


def singularize_token(token: str) -> str:
    """Conservative English plural reduction for phrase-head matching."""
    t = token.casefold()
    irregular = {
        "aircraft": "aircraft",
        "systems": "system",
        "vehicles": "vehicle",
        "stations": "station",
        "links": "link",
        "sensors": "sensor",
        "platforms": "platform",
        "subsystems": "subsystem",
        "applications": "application",
        "operations": "operation",
        "missions": "mission",
        "classes": "class",
        "categories": "category",
        "capabilities": "capability",
        "facilities": "facility",
        "technologies": "technology",
    }
    if t in irregular:
        return irregular[t]
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("sses"):
        return t[:-2]
    if len(t) > 4 and t.endswith("xes"):
        return t[:-2]
    if len(t) > 4 and t.endswith("zes"):
        return t[:-1]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def morph_key(text: Any) -> str:
    tokens = tokenise(text)
    if not tokens:
        return ""
    # Phrase-head normalization only; avoids control/controller collapse.
    tokens[-1] = singularize_token(tokens[-1])
    return " ".join(tokens)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def append_jsonl_durable(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def first_existing_column(columns: Iterable[str], names: list[str]) -> str | None:
    lookup = {str(c).casefold(): str(c) for c in columns}
    for name in names:
        if name.casefold() in lookup:
            return lookup[name.casefold()]
    return None


# ---------------------------------------------------------------------------
# Corpus indexing and deterministic matching
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Preserve acronym periods and decimals reasonably well while splitting.
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[]|[\"'])|\n{2,}", text)
    output: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        sentence = re.sub(r"\s+", " ", chunk).strip()
        if len(sentence.split()) < 4 or len(sentence) < 20:
            continue
        key = normalize(sentence)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(sentence)
    return output


@dataclass
class CorpusIndex:
    sentences: list[str]
    normalized_sentences: list[str]
    token_sentences: list[list[str]]
    inverted: dict[str, set[int]]

    @classmethod
    def build(cls, corpus_text: str) -> "CorpusIndex":
        sentences = split_sentences(corpus_text)
        normalized_sentences = [normalize(s) for s in sentences]
        token_sentences = [s.split() for s in normalized_sentences]
        inverted: dict[str, set[int]] = defaultdict(set)
        for index, tokens in enumerate(token_sentences):
            for token in set(tokens):
                inverted[token].add(index)
        return cls(sentences, normalized_sentences, token_sentences, dict(inverted))

    def candidate_sentence_ids(self, phrase: str) -> list[int]:
        tokens = tokenise(phrase)
        if not tokens:
            return []
        groups = [self.inverted.get(token, set()) for token in set(tokens)]
        if not groups or any(not group for group in groups):
            return []
        return sorted(set.intersection(*groups))

    def phrase_hits(self, phrase: str) -> list[int]:
        key = normalize(phrase)
        tokens = key.split()
        if not tokens:
            return []
        hits: list[int] = []
        for index in self.candidate_sentence_ids(key):
            sentence = self.normalized_sentences[index]
            if re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", sentence):
                hits.append(index)
        return hits

    def morph_hits(self, phrase: str) -> list[int]:
        key = morph_key(phrase)
        if not key:
            return []
        source_tokens = tokenise(phrase)
        broad: set[int] = set()
        for token in set(source_tokens):
            variants = {token, singularize_token(token)}
            for variant in variants:
                broad.update(self.inverted.get(variant, set()))
        hits: list[int] = []
        for index in sorted(broad):
            if morph_key(self.normalized_sentences[index]).find(key) >= 0:
                # Token sequence comparison rather than raw substring.
                sentence_tokens = self.token_sentences[index]
                width = len(key.split())
                for start in range(0, max(0, len(sentence_tokens) - width + 1)):
                    window = " ".join(sentence_tokens[start:start + width])
                    if morph_key(window) == key:
                        hits.append(index)
                        break
        return hits

    def evidence_window(self, hit_ids: list[int], limit: int = 5) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        used: set[int] = set()
        for hit in hit_ids:
            for idx in (hit, hit - 1, hit + 1):
                if idx < 0 or idx >= len(self.sentences) or idx in used:
                    continue
                used.add(idx)
                selected.append({
                    "sentence_id": f"SENT_{idx:06d}",
                    "sentence": self.sentences[idx],
                    "is_direct_hit": idx == hit,
                })
                if len(selected) >= limit:
                    return selected
        return selected


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_typing_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    term_col = first_existing_column(
        frame.columns,
        ["canonical_term", "candidate", "term", "original_term", "preferred_label"],
    )
    type_col = first_existing_column(
        frame.columns,
        ["final_type", "predicted_type", "entity_type"],
    )
    if not term_col or not type_col:
        raise ValueError(
            f"Typing results need a term column and final type column. Columns: {list(frame.columns)}"
        )
    out = frame.copy()
    out["taxonomy_term"] = out[term_col].map(safe_text)
    out["taxonomy_term_norm"] = out[term_col].map(normalize)
    out["predicted_entity_type"] = out[type_col].map(safe_text)
    out = out[
        out["predicted_entity_type"].isin(["System", "Subsystem"])
        & out["taxonomy_term_norm"].ne("")
    ].copy()
    return out.drop_duplicates("taxonomy_term_norm").reset_index(drop=True)


def load_aliases(path: Path, sheet: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["canonical", "alias", "canonical_norm", "alias_norm"])
    book = pd.ExcelFile(path, engine="openpyxl")
    if sheet not in book.sheet_names:
        return pd.DataFrame(columns=["canonical", "alias", "canonical_norm", "alias_norm"])
    frame = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    canonical_col = first_existing_column(frame.columns, ["canonical", "canonical_term", "term_canonical"])
    alias_col = first_existing_column(frame.columns, ["alias", "alternative_label", "synonym"])
    if not canonical_col or not alias_col:
        return pd.DataFrame(columns=["canonical", "alias", "canonical_norm", "alias_norm"])
    out = pd.DataFrame({
        "canonical": frame[canonical_col].map(safe_text),
        "alias": frame[alias_col].map(safe_text),
    })
    out["canonical_norm"] = out["canonical"].map(normalize)
    out["alias_norm"] = out["alias"].map(normalize)
    return out[out.canonical_norm.ne("") & out.alias_norm.ne("")].drop_duplicates()


def build_alias_lookup(aliases: pd.DataFrame) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for row in aliases.itertuples(index=False):
        lookup[row.canonical_norm].add(row.alias)
        lookup[row.alias_norm].add(row.canonical)
    return dict(lookup)


# ---------------------------------------------------------------------------
# Controlled compound reconstruction
# ---------------------------------------------------------------------------


def reconstruct_candidates(term: str) -> list[str]:
    """Generate only conservative corpus-search forms; never assert equivalence here."""
    value = safe_text(term)
    candidates: set[str] = set()
    # Slash compounds: MALE/HALE UAVs -> MALE UAVs, HALE UAVs.
    match = re.match(r"^\s*([A-Za-z0-9&+.-]+)\s*/\s*([A-Za-z0-9&+.-]+)\s+(.+?)\s*$", value)
    if match:
        left, right, tail = match.groups()
        candidates.add(f"{left} {tail}")
        candidates.add(f"{right} {tail}")
    # Parenthetical plural: station(s) -> station / stations.
    if "(s)" in value.casefold():
        candidates.add(re.sub(r"\(s\)", "", value, flags=re.I))
        candidates.add(re.sub(r"\(s\)", "s", value, flags=re.I))
    return sorted(candidates)


# ---------------------------------------------------------------------------
# Ollama client and Filter Agent
# ---------------------------------------------------------------------------


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        output_dir: Path,
        temperature: float,
        timeout: int,
        retries: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.cache_dir = output_dir / "cache" / "qwen"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_log = output_dir / "qwen_filter_requests.jsonl"
        self.live_calls = 0
        self.cache_hits = 0

    def check(self) -> None:
        request = urllib.request.Request(f"{self.base_url}/api/tags")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama returned HTTP {response.status}")
        except Exception as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self.base_url}. Start 'ollama serve'. Error: {exc}"
            ) from exc

    def _cache_key(self, stage: str, prompt: str) -> str:
        payload = json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "stage": stage,
                "model": self.model,
                "temperature": self.temperature,
                "prompt": prompt,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            value = json.loads(text[start:end + 1])
            if isinstance(value, dict):
                return value
        raise ValueError("No JSON object found in response")

    def chat_json(
        self,
        stage: str,
        prompt: str,
        validator,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        key = self._cache_key(stage, prompt)
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            try:
                result = validator(cached["result"])
                self.cache_hits += 1
                return result, {"cache_hit": True, "runtime_seconds": 0.0}
            except Exception:
                cache_path.unlink(missing_ok=True)

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            started = time.perf_counter()
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a conservative corpus-grounding auditor. "
                            "Return valid JSON only and never invent corpus text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": self.temperature},
            }
            request = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    envelope = json.loads(response.read().decode("utf-8"))
                raw = envelope["message"]["content"]
                parsed = self._extract_json(raw)
                result = validator(parsed)
                elapsed = time.perf_counter() - started
                record = {
                    "created_utc": now_utc(),
                    "stage": stage,
                    "attempt": attempt,
                    "prompt": prompt,
                    "raw_response": raw,
                    "result": result,
                    "runtime_seconds": elapsed,
                }
                cache_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                append_jsonl_durable(self.request_log, record)
                self.live_calls += 1
                return result, {"cache_hit": False, "runtime_seconds": elapsed}
            except Exception as exc:
                last_error = exc
                time.sleep(min(2 * attempt, 6))
        raise RuntimeError(f"Qwen filter call failed after retries: {last_error}")


def validate_filter_response(value: dict[str, Any]) -> dict[str, Any]:
    decision = safe_text(value.get("decision"))
    if decision not in ALLOWED_LLM_DECISIONS:
        raise ValueError(f"Invalid filter decision: {decision}")
    confidence = float(value.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0,1]")
    matched_text = safe_text(value.get("matched_text"))
    possible_complete_form = safe_text(value.get("possible_complete_form"))
    reason = safe_text(value.get("reason"))
    ids = value.get("evidence_sentence_ids", [])
    if not isinstance(ids, list):
        raise ValueError("evidence_sentence_ids must be a list")
    return {
        "decision": decision,
        "confidence": confidence,
        "matched_text": matched_text,
        "possible_complete_form": possible_complete_form,
        "reason": reason,
        "evidence_sentence_ids": [safe_text(item) for item in ids if safe_text(item)],
    }


def make_filter_prompt(
    term: str,
    entity_type: str,
    description: str,
    evidence: list[dict[str, Any]],
    original_context: str,
) -> str:
    evidence_text = "\n".join(
        f"[{row['sentence_id']}] {row['sentence']}" for row in evidence
    ) or "No reliable sentence match was retrieved."
    return f"""You are the Corpus Presence and Noise Filter Agent for a taxonomy-building pipeline.

CANDIDATE TERM:
{term}

PREDICTED ENTITY TYPE FROM THE PREVIOUS STAGE:
{entity_type}

ENTITY DESCRIPTION FROM THE PREVIOUS STAGE:
{description or 'Not available'}

ORIGINAL EXTRACTION CONTEXT, IF AVAILABLE:
{original_context or 'Not available'}

CORPUS EVIDENCE:
{evidence_text}

TASK:
Decide whether the candidate is a complete, reusable technical concept genuinely expressed in the corpus. The term may be accepted when the corpus explicitly contains a safe lexical variant, documented alias/acronym, or controlled reconstructed form. Semantic similarity alone is not corpus presence.

ALLOWED DECISIONS:
- accept_variant: a complete technical concept is explicitly represented by a minor lexical/morphological variant.
- accept_alias: a documented alias or acronym in the evidence expresses the same complete concept.
- accept_reconstructed: the complete concept can be safely reconstructed from a controlled compound or coordination in the evidence.
- reject_not_present: neither the candidate nor a safe equivalent expression is present.
- reject_fragment: the candidate is an incomplete extraction span rather than a standalone concept.
- reject_artifact: formatting, bibliographic, OCR, table, or extraction noise.
- reject_context_only: the phrase occurs but is generic, descriptive, or not a reusable standalone technical concept.
- manual_review: evidence is genuinely ambiguous.

IMPORTANT RULES:
1. Do not accept merely because the surrounding passage is about UAVs.
2. Do not invent a corrected term.
3. A possible complete form may be suggested, but it remains a review note and is not automatically accepted.
4. Prefer rejection or manual review when the evidence is insufficient.
5. Evidence sentence identifiers must come from the supplied evidence.

Return JSON only:
{{
  "decision": "one allowed decision",
  "confidence": 0.0,
  "matched_text": "exact supporting surface form, or empty",
  "possible_complete_form": "possible complete form, or empty",
  "evidence_sentence_ids": ["SENT_000001"],
  "reason": "brief evidence-grounded explanation"
}}
"""


# ---------------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------------


def get_row_description(row: pd.Series) -> str:
    for column in ("entity_description", "description", "candidate_description"):
        if column in row.index and safe_text(row[column]):
            return safe_text(row[column])
    return ""


def get_original_context(row: pd.Series) -> str:
    for column in ("source_sentence", "context_text", "retrieved_contexts", "supporting_sentences"):
        if column in row.index and safe_text(row[column]):
            return safe_text(row[column])[:2500]
    return ""


def deterministic_filter(
    term: str,
    index: CorpusIndex,
    alias_lookup: dict[str, set[str]],
) -> dict[str, Any] | None:
    exact_hits = index.phrase_hits(term)
    if exact_hits:
        evidence = index.evidence_window(exact_hits)
        return {
            "presence_status": "exact_present",
            "filter_method": "normalized_exact",
            "matched_surface_form": term,
            "occurrence_count": len(exact_hits),
            "evidence": evidence,
            "filter_confidence": 1.0,
            "filter_reason": "Normalized token-boundary occurrence found in corpus.",
        }

    morph_hits = index.morph_hits(term)
    if morph_hits:
        evidence = index.evidence_window(morph_hits)
        return {
            "presence_status": "morphological_present",
            "filter_method": "conservative_phrase_head_morphology",
            "matched_surface_form": "",
            "occurrence_count": len(morph_hits),
            "evidence": evidence,
            "filter_confidence": 0.97,
            "filter_reason": "Conservative singular/plural lexical variant found in corpus.",
        }

    key = normalize(term)
    alias_forms = sorted(alias_lookup.get(key, set()))
    alias_hits: list[tuple[str, list[int]]] = []
    for alias in alias_forms:
        hits = index.phrase_hits(alias)
        if not hits:
            hits = index.morph_hits(alias)
        if hits:
            alias_hits.append((alias, hits))
    if alias_hits:
        alias, hits = max(alias_hits, key=lambda item: len(item[1]))
        return {
            "presence_status": "alias_present",
            "filter_method": "explicit_alias_sheet",
            "matched_surface_form": alias,
            "occurrence_count": len(hits),
            "evidence": index.evidence_window(hits),
            "filter_confidence": 0.98,
            "filter_reason": "Explicit alias/canonical counterpart occurs in the corpus.",
        }

    reconstruction_hits: list[tuple[str, list[int]]] = []
    for reconstructed in reconstruct_candidates(term):
        hits = index.phrase_hits(reconstructed) or index.morph_hits(reconstructed)
        if hits:
            reconstruction_hits.append((reconstructed, hits))
    if reconstruction_hits:
        reconstructed, hits = max(reconstruction_hits, key=lambda item: len(item[1]))
        return {
            "presence_status": "corpus_reconstructed",
            "filter_method": "controlled_compound_reconstruction",
            "matched_surface_form": reconstructed,
            "occurrence_count": len(hits),
            "evidence": index.evidence_window(hits),
            "filter_confidence": 0.88,
            "filter_reason": "Controlled compound/coordination reconstruction found in corpus.",
        }
    return None


def map_llm_decision(result: dict[str, Any]) -> str:
    return {
        "accept_variant": "morphological_present",
        "accept_alias": "alias_present",
        "accept_reconstructed": "corpus_reconstructed",
        "reject_not_present": "not_present",
        "reject_fragment": "fragment",
        "reject_artifact": "artifact",
        "reject_context_only": "context_only",
        "manual_review": "manual_review",
    }[result["decision"]]


def process_terms(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    typing = load_typing_results(args.typing_results)
    aliases = load_aliases(args.alias_workbook, args.alias_sheet)
    alias_lookup = build_alias_lookup(aliases)
    corpus_text = args.corpus.read_text(encoding="utf-8", errors="replace")
    index = CorpusIndex.build(corpus_text)

    sentence_rows = [
        {
            "sentence_id": f"SENT_{i:06d}",
            "sentence": sentence,
            "normalized_sentence": index.normalized_sentences[i],
        }
        for i, sentence in enumerate(index.sentences)
    ]
    pd.DataFrame(sentence_rows).to_csv(output_dir / "00_corpus_sentences.csv", index=False)

    client = OllamaClient(
        args.ollama_url,
        args.model,
        output_dir,
        args.temperature,
        args.timeout,
        args.retries,
    )
    if not args.skip_ollama_check:
        client.check()

    partial_path = output_dir / "01_corpus_presence_results.partial.jsonl"
    completed = {
        normalize(row.get("candidate")): row
        for row in load_jsonl(partial_path)
        if normalize(row.get("candidate"))
    }

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for number, (_, row) in enumerate(typing.iterrows(), start=1):
        term = safe_text(row["taxonomy_term"])
        key = normalize(term)
        print("\n" + "-" * 100)
        print(f"TERM {number}/{len(typing)}: {term}")
        print("-" * 100)

        if key in completed:
            record = completed[key]
            records.append(record)
            print("  Resume: completed filter result loaded.")
            continue

        item_start = time.perf_counter()
        deterministic = deterministic_filter(term, index, alias_lookup)
        llm_invoked = False
        llm_runtime = 0.0
        possible_complete_form = ""

        if deterministic is not None:
            result = deterministic
            print(f"  Deterministic: {result['presence_status']}")
        else:
            llm_invoked = True
            # Gather loose lexical evidence for the agent; no acceptance is implied.
            loose_ids: set[int] = set()
            for token in tokenise(term):
                loose_ids.update(index.inverted.get(token, set()))
            ranked_loose = sorted(
                loose_ids,
                key=lambda idx: -len(set(tokenise(term)) & set(index.token_sentences[idx])),
            )[:5]
            evidence = index.evidence_window(ranked_loose, limit=5)
            prompt = make_filter_prompt(
                term,
                safe_text(row["predicted_entity_type"]),
                get_row_description(row),
                evidence,
                get_original_context(row),
            )
            try:
                llm_result, meta = client.chat_json(
                    f"filter::{hashlib.sha1(key.encode()).hexdigest()[:12]}",
                    prompt,
                    validate_filter_response,
                )
                llm_runtime = float(meta.get("runtime_seconds", 0.0))
                status = map_llm_decision(llm_result)
                possible_complete_form = llm_result["possible_complete_form"]
                result = {
                    "presence_status": status,
                    "filter_method": "qwen_filter_agent",
                    "matched_surface_form": llm_result["matched_text"],
                    "occurrence_count": 0,
                    "evidence": evidence,
                    "filter_confidence": llm_result["confidence"],
                    "filter_reason": llm_result["reason"],
                    "filter_agent_decision": llm_result["decision"],
                }
                print(f"  Filter Agent: {status}")
            except Exception as exc:
                result = {
                    "presence_status": "manual_review",
                    "filter_method": "qwen_filter_agent_failed",
                    "matched_surface_form": "",
                    "occurrence_count": 0,
                    "evidence": evidence,
                    "filter_confidence": 0.0,
                    "filter_reason": f"Filter Agent failed safely: {exc}",
                    "filter_agent_decision": "manual_review",
                }
                print("  Filter Agent failed; routed to manual review.")

        status = result["presence_status"]
        taxonomy_action = (
            "accept"
            if status in ACCEPTED_STATUSES
            else "reject"
            if status in REJECTED_STATUSES
            else "manual_review"
        )
        evidence_ids = [item["sentence_id"] for item in result.get("evidence", [])]
        evidence_texts = [item["sentence"] for item in result.get("evidence", [])]
        record = row.to_dict()
        record.update({
            "candidate": term,
            "candidate_norm": key,
            "presence_status": status,
            "taxonomy_action": taxonomy_action,
            "matched_surface_form": result.get("matched_surface_form", ""),
            "possible_complete_form": possible_complete_form,
            "occurrence_count": int(result.get("occurrence_count", 0)),
            "document_count": 1 if result.get("occurrence_count", 0) else 0,
            "evidence_sentence_ids": evidence_ids,
            "evidence_sentences": evidence_texts,
            "filter_method": result.get("filter_method", ""),
            "filter_confidence": float(result.get("filter_confidence", 0.0)),
            "filter_reason": result.get("filter_reason", ""),
            "filter_agent_invoked": llm_invoked,
            "filter_agent_decision": result.get("filter_agent_decision", ""),
            "llm_runtime_seconds": llm_runtime,
            "runtime_seconds": time.perf_counter() - item_start,
            "node_origin": "corpus" if taxonomy_action == "accept" else "",
        })
        append_jsonl_durable(partial_path, record)
        completed[key] = record
        records.append(record)
        print(
            f"  Saved: {taxonomy_action} | {status} | "
            f"confidence={record['filter_confidence']:.2f}"
        )

    results = pd.DataFrame(records)
    results.to_csv(output_dir / "01_corpus_presence_results.csv", index=False)
    accepted = results[results.taxonomy_action.eq("accept")].copy()
    rejected = results[results.taxonomy_action.eq("reject")].copy()
    review = results[results.taxonomy_action.eq("manual_review")].copy()
    proposed = pd.DataFrame(columns=[
        "proposed_label",
        "reason",
        "node_origin",
        "requires_human_approval",
    ])
    accepted.to_csv(output_dir / "02_taxonomy_terms_accepted.csv", index=False)
    rejected.to_csv(output_dir / "03_taxonomy_terms_rejected.csv", index=False)
    review.to_csv(output_dir / "04_taxonomy_terms_manual_review.csv", index=False)
    proposed.to_csv(output_dir / "05_proposed_structural_nodes.csv", index=False)

    status_counts = results["presence_status"].value_counts(dropna=False).to_dict()
    method_counts = results["filter_method"].value_counts(dropna=False).to_dict()
    summary = {
        "created_utc": now_utc(),
        "input_system_subsystem_terms": int(len(typing)),
        "accepted_terms": int(len(accepted)),
        "rejected_terms": int(len(rejected)),
        "manual_review_terms": int(len(review)),
        "acceptance_rate": float(len(accepted) / len(typing)) if len(typing) else 0.0,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "method_counts": {str(k): int(v) for k, v in method_counts.items()},
        "live_qwen_calls": client.live_calls,
        "qwen_cache_hits": client.cache_hits,
        "runtime_seconds": time.perf_counter() - started,
    }
    pd.DataFrame([
        {"metric": key, "value": json_dumps(value) if isinstance(value, dict) else value}
        for key, value in summary.items()
    ]).to_csv(output_dir / "06_filter_summary.csv", index=False)
    return results, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typing-results", type=Path, default=DEFAULT_TYPING_RESULTS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--alias-workbook", type=Path, default=DEFAULT_ALIAS_WORKBOOK)
    parser.add_argument("--alias-sheet", default="S2_GT_ALIASES")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--skip-ollama-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.typing_results, args.corpus, args.alias_workbook):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _, summary = process_terms(args)
    manifest = {
        "script": Path(__file__).name,
        "prompt_version": PROMPT_VERSION,
        "created_utc": now_utc(),
        "inputs": {
            "typing_results": str(args.typing_results),
            "typing_results_sha256": sha256_file(args.typing_results),
            "corpus": str(args.corpus),
            "corpus_sha256": sha256_file(args.corpus),
            "alias_workbook": str(args.alias_workbook),
            "alias_sheet": args.alias_sheet,
        },
        "model": args.model,
        "ollama_url": args.ollama_url,
        "summary": summary,
        "total_runtime_seconds": time.perf_counter() - started,
        "accepted_statuses": sorted(ACCEPTED_STATUSES),
        "rejected_statuses": sorted(REJECTED_STATUSES),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("=" * 100)
    print("CORPUS PRESENCE AND NOISE FILTER COMPLETED")
    print("=" * 100)
    print(f"Input System/Subsystem terms: {summary['input_system_subsystem_terms']}")
    print(f"Accepted corpus-grounded terms: {summary['accepted_terms']}")
    print(f"Rejected noise/unsupported terms: {summary['rejected_terms']}")
    print(f"Manual review terms: {summary['manual_review_terms']}")
    print(f"Live Qwen calls: {summary['live_qwen_calls']}")
    print(f"Cache hits: {summary['qwen_cache_hits']}")
    print(f"Runtime: {summary['runtime_seconds']:.2f}s")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
