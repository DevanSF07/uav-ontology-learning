#!/usr/bin/env python3
"""Qwen2.5 validation of the high-recall YAKE + MPNet term inventory.

This experiment starts from the earlier high-recall YAKE + MPNet inventory:

    stage_outputs/03c_weighted_voting/best_f1_candidates.csv

For every candidate, the script:

1. Uses the original source sentence from the candidate inventory.
2. Retrieves up to N additional exact-occurrence corpus sentences and ranks
   them with TF-IDF similarity.
3. Sends only candidate evidence, ontology scope, and validation instructions
   to a local Ollama model.
4. Requires a structured JSON judgment from qwen2.5:7b-instruct.
5. Validates and normalizes the response.
6. Computes a numeric LLM ontology-suitability score.
7. Applies acronym/strong-evidence protection to unsafe hard rejections.
8. Sweeps score thresholds from 0.10 through 0.90.
9. Evaluates two policies against strict GT:
      a. retain_only
      b. retain_or_review
10. Saves requests, raw responses, parsed results, metrics, inventories, and
    F1-versus-threshold plots.

Ground-truth information is NEVER included in an LLM request. GT is loaded
only after inference for offline evaluation.

Ollama setup:
    ollama pull qwen2.5:7b-instruct
    ollama serve

Example pilot:
    python test_07b_qwen_yake_mpnet_validation.py --limit 25

Complete run:
    python test_07b_qwen_yake_mpnet_validation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path("/home/faujdar/onto_gen/ontology_experiment_framework")
DEFAULT_INPUT = (
    PROJECT_DIR
    / "stage_outputs/03c_weighted_voting/best_f1_candidates.csv"
)
DEFAULT_CHUNKS = PROJECT_DIR / "stage_outputs/02_chunking/chunks.csv"
DEFAULT_GT = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_OUTPUT = PROJECT_DIR / "stage_outputs/07b_qwen_validation_yake_mpnet"

DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
PROMPT_VERSION = "uav-term-validator-yake-mpnet-v2.1"
GT_SHEET = "S1_GT_TERMS"
NEGATIVE_SHEET = "S7_NEGATIVE_TERMS"
MANUAL_REVIEW_STATUS = "needs_manual_review"

ALLOWED_DECISIONS = {"retain", "review", "reject"}
ALLOWED_CATEGORIES = {
    "core_uav_term",
    "platform_class",
    "platform_variant",
    "classification",
    "structural_design",
    "platform_design_performance",
    "communication_system",
    "operation_control",
    "mission_application",
    "civilian_application",
    "regulatory_airspace_risk",
    "other_uav_domain_concept",
    "none",
}
ALLOWED_CONCEPT_TYPES = {
    "technical_concept",
    "class",
    "system",
    "component",
    "property",
    "application",
    "operation",
    "regulatory_concept",
    "alias_candidate",
    "generic_expression",
    "artifact",
    "other",
}
ALLOWED_REJECTION_REASONS = {
    "generic_word",
    "generic_phrase",
    "action_or_verb",
    "descriptive_fragment",
    "document_artifact",
    "table_artifact",
    "bibliographic_metadata",
    "person_name",
    "organization_name",
    "geographic_entity",
    "publication_title",
    "context_only_relevance",
    "malformed_candidate",
    "overly_broad_concept",
    "unsupported_by_context",
    "outside_ontology_scope",
    "ambiguous_without_sufficient_evidence",
    "duplicate_or_alias",
}
PRIORITY_WEIGHTS = {"high": 3.0, "medium": 2.0, "low": 1.0}

ARTIFACT_TERMS = {
    "hline",
    "mathrm",
    "tabular",
    "end tabular",
}

CORE_SCOPE_TERMS = {
    "aircraft",
    "uav",
    "uas",
    "unmanned aircraft",
    "unmanned aerial vehicle",
    "unmanned aircraft system",
    "ground control station",
    "airspace",
    "payload",
}

SYSTEM_PROMPT = """You are a conservative validator of candidate terms for an ontology about unmanned aircraft, UAV systems, aviation systems, UAV operations, applications, regulations, risks, and supporting technologies.

Your primary task is to evaluate the CANDIDATE PHRASE ITSELF. A sentence can be highly relevant to UAVs while the candidate phrase is still invalid. Never transfer the relevance of the surrounding sentence to a generic candidate.

A candidate is valid when the candidate itself names a reusable technical concept, class, system, component, property, capability, operational concept, application, regulatory concept, risk concept, or aviation concept needed to describe UAV systems.

Positive examples include aircraft, unmanned aircraft, ground control station, flight control system, payload, airspace, satellite data link, endurance, maximum takeoff weight, and beyond line of sight.

Negative examples include public, important, different, several, significantly, based, use, operated, chapter, figure, and hline.

Names of military organizations, companies, universities, countries, cities, publishers, and research institutions are normally incidental entities rather than ontology concepts. Terms such as Army, NASA, Boeing, Denver, and Oxford University should normally be rejected unless the ontology explicitly models named organizations.

Aircraft is a valid aviation and UAV superclass. Do not reject aircraft merely because aircraft is broader than UAV.

Use review instead of reject when evidence is insufficient, an acronym meaning is uncertain, the phrase may be an alias, the candidate could be valid under a clear technical interpretation, or rejection confidence is low.

A reject decision requires strong evidence that the candidate itself is generic, malformed, incidental, artifact-like, bibliographic, or outside the stated scope.

Use only the supplied evidence. Do not invent definitions, categories, or relationships. Return one valid JSON object only. Do not return Markdown, analysis, comments, or text outside the JSON object."""

ONTOLOGY_CATEGORIES = [
    "core_uav_term",
    "platform_class",
    "platform_variant",
    "classification",
    "structural_design",
    "platform_design_performance",
    "communication_system",
    "operation_control",
    "mission_application",
    "civilian_application",
    "regulatory_airspace_risk",
    "other_uav_domain_concept",
    "none",
]


@dataclass
class LLMResult:
    parsed: dict[str, Any]
    raw_content: str
    elapsed_seconds: float
    attempts: int
    cache_hit: bool
    cache_key: str
    error: str = ""


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_term(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).casefold().strip()
    for dash in ("‐", "‑", "‒", "–", "—"):
        text = text.replace(dash, "-")
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"[^a-z0-9&/+ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_search_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).casefold()
    for dash in ("‐", "‑", "‒", "–", "—"):
        text = text.replace(dash, "-")
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"[^a-z0-9&/+.!? ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def split_sentences(text: str) -> list[str]:
    text = safe_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [part.strip() for part in parts if part.strip()]


def find_text_column(frame: pd.DataFrame) -> str:
    preferred = (
        "source_sentence",
        "sentence",
        "chunk_text",
        "context_text",
        "text",
        "chunk",
        "content",
    )
    lookup = {str(column).casefold(): column for column in frame.columns}
    for name in preferred:
        if name in lookup:
            return str(lookup[name])
    object_columns = [column for column in frame.columns if frame[column].dtype == "object"]
    if not object_columns:
        raise ValueError("No text-like column exists in the chunks file")
    return str(max(
        object_columns,
        key=lambda column: frame[column].fillna("").astype(str).str.len().mean(),
    ))


def load_candidates(path: Path, limit: int | None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8")
    if "candidate" not in frame.columns:
        raise ValueError("Input candidate CSV must contain a 'candidate' column")

    frame = frame.copy()
    if "normalized_candidate" not in frame.columns:
        frame["normalized_candidate"] = frame["candidate"].map(normalize_term)
    else:
        frame["normalized_candidate"] = frame["normalized_candidate"].map(normalize_term)

    frame["candidate"] = frame["candidate"].map(safe_text)
    for column in ("source_sentence", "context_text"):
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].map(safe_text)

    frame = frame[frame["normalized_candidate"].ne("")]
    frame = frame.drop_duplicates("normalized_candidate", keep="first").reset_index(drop=True)
    if limit is not None:
        frame = frame.head(limit).copy()
    frame.insert(0, "candidate_row_id", np.arange(len(frame), dtype=int))
    return frame


def load_corpus_sentences(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    chunks = pd.read_csv(path, encoding="utf-8")
    text_column = find_text_column(chunks)
    rows: list[dict[str, Any]] = []

    for chunk_position, value in enumerate(chunks[text_column].fillna("")):
        chunk_id = (
            chunks.iloc[chunk_position]["chunk_id"]
            if "chunk_id" in chunks.columns
            else chunk_position
        )
        for sentence_position, sentence in enumerate(split_sentences(str(value))):
            rows.append({
                "chunk_id": chunk_id,
                "sentence_position": sentence_position,
                "sentence": sentence,
                "normalized_sentence": normalize_search_text(sentence),
            })

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No corpus sentences could be extracted")
    return result.drop_duplicates("sentence").reset_index(drop=True)


def term_pattern(normalized_term: str) -> re.Pattern[str]:
    tokens = normalized_term.split()
    body = r"[\s\-]+".join(re.escape(token) for token in tokens)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)


def retrieve_contexts(
    candidates: pd.DataFrame,
    sentences: pd.DataFrame,
    top_k: int,
    maximum_occurrences: int,
) -> pd.DataFrame:
    normalized_sentences = sentences["normalized_sentence"].tolist()
    records: list[dict[str, Any]] = []

    for position, row in enumerate(candidates.itertuples(index=False), start=1):
        pattern = term_pattern(row.normalized_candidate)
        occurrence_indices = [
            index
            for index, sentence in enumerate(normalized_sentences)
            if pattern.search(sentence)
        ][:maximum_occurrences]
        occurrence_sentences = sentences.iloc[occurrence_indices]["sentence"].tolist()

        source_sentence = safe_text(row.source_sentence)
        query = f"{row.candidate} {source_sentence}".strip()
        selected: list[str] = []
        scores: list[float] = []

        if occurrence_sentences:
            if len(occurrence_sentences) == 1:
                selected = occurrence_sentences
                scores = [1.0]
            else:
                documents = [query] + occurrence_sentences
                vectorizer = TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                )
                matrix = vectorizer.fit_transform(documents)
                similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
                ranked = np.argsort(-similarities)[:top_k]
                selected = [occurrence_sentences[index] for index in ranked]
                scores = [float(similarities[index]) for index in ranked]
        elif source_sentence:
            selected = [source_sentence]
            scores = [1.0]

        # Do not duplicate the original source sentence in additional evidence.
        additional = []
        additional_scores = []
        source_normalized = normalize_search_text(source_sentence)
        for sentence, score in zip(selected, scores):
            if source_normalized and normalize_search_text(sentence) == source_normalized:
                continue
            additional.append(sentence)
            additional_scores.append(score)
            if len(additional) >= top_k:
                break

        records.append({
            "candidate_row_id": row.candidate_row_id,
            "corpus_occurrence_count": len(occurrence_sentences),
            "additional_contexts": additional,
            "additional_context_scores": additional_scores,
        })

        if position % 100 == 0 or position == len(candidates):
            print(f"Retrieved corpus evidence for {position}/{len(candidates)} candidates")

    return candidates.merge(
        pd.DataFrame(records),
        on="candidate_row_id",
        how="left",
        validate="one_to_one",
    )


def parse_method_support(row: pd.Series) -> int:
    possible_columns = (
        "method_count",
        "extractor_count",
        "support_count",
        "n_methods",
        "vote_count",
    )
    for column in possible_columns:
        if column in row.index:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return int(value)

    method_columns = [
        column
        for column in row.index
        if str(column).casefold().startswith(("found_by_", "selected_by_", "method_"))
    ]
    count = 0
    for column in method_columns:
        value = row[column]
        if isinstance(value, bool) and value:
            count += 1
        elif str(value).strip().casefold() in {"1", "true", "yes"}:
            count += 1
    return count


def is_protected_acronym(candidate: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", safe_text(candidate))
    return (
        2 <= len(cleaned) <= 10
        and any(character.isalpha() for character in cleaned)
        and cleaned.upper() == cleaned
    )


def is_obvious_artifact(candidate: str) -> bool:
    normalized = normalize_term(candidate)
    raw = safe_text(candidate)
    if normalized in ARTIFACT_TERMS:
        return True
    if re.search(r"\bdoi\b", normalized):
        return True
    if "@" in raw:
        return True
    if re.search(r"\\[A-Za-z]+", raw):
        return True
    return False


def candidate_payload(row: pd.Series) -> dict[str, Any]:
    contexts = row.get("additional_contexts", [])
    if not isinstance(contexts, list):
        contexts = []
    return {
        "candidate": safe_text(row["candidate"]),
        "normalized_candidate": normalize_term(row["normalized_candidate"]),
        "source_sentence": safe_text(row.get("source_sentence", "")),
        "additional_corpus_evidence": contexts,
        "corpus_occurrence_count": int(row.get("corpus_occurrence_count", 0) or 0),
        "independent_method_support": parse_method_support(row),
        "is_acronym_shape": is_protected_acronym(str(row["candidate"])),
        "is_obvious_artifact": is_obvious_artifact(str(row["candidate"])),
        "is_core_scope_term": normalize_term(row["normalized_candidate"]) in CORE_SCOPE_TERMS,
    }


def build_user_prompt(payload: dict[str, Any]) -> str:
    categories = "\n".join(f"- {category}" for category in ONTOLOGY_CATEGORIES)
    evidence = payload["additional_corpus_evidence"]
    evidence_text = (
        "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(evidence, 1))
        if evidence
        else "No additional exact-occurrence sentence was retrieved."
    )

    return f"""Evaluate this candidate for inclusion in a UAV-domain ontology.

Perform these tests in order:

TEST 1, candidate-alone test:
Ignore the context briefly. Does the candidate phrase itself name a recognizable concept or entity type?

TEST 2, ontology-scope test:
Could the candidate be used as a node, class, attribute, capability, application, operational concept, or regulatory concept in a UAV ontology?

TEST 3, context-grounding test:
Does the supplied evidence support the candidate's technical meaning?

TEST 4, genericity test:
Would the phrase remain meaningful as a reusable ontology label outside this exact sentence?

TEST 5, incidental-entity test:
Is the candidate merely a named organization, military organization, company, university, person, place, publication, or author affiliation?

Do not retain a candidate solely because its sentence discusses UAVs.
Specific guidance:
- Retain aircraft as a valid superclass of unmanned aircraft.
- Reject ordinary adjectives such as public when they do not name a technical class or concept.
- Treat named organizations such as Army as incidental unless the evidence clearly establishes an intentionally modeled ontology category.
- Reject formatting artifacts such as hline.

Allowed categories:
{categories}

Allowed decisions:
- retain: the candidate itself is a specific, reusable ontology concept and evidence supports the interpretation
- review: potentially valid, ambiguous, overly broad, alias-like, insufficiently evidenced, or low-confidence
- reject: the candidate itself is clearly generic, malformed, incidental, artifact-like, bibliographic, or outside the ontology scope

Candidate: {payload['candidate']}
Normalized candidate: {payload['normalized_candidate']}
Original source sentence: {payload['source_sentence'] or 'Not available'}
Corpus occurrence count: {payload['corpus_occurrence_count']}
Independent extraction-method support: {payload['independent_method_support']}
Acronym-shaped candidate: {payload['is_acronym_shape']}
Obvious deterministic artifact: {payload['is_obvious_artifact']}
Core ontology-scope anchor: {payload['is_core_scope_term']}

Additional exact-occurrence corpus evidence:
{evidence_text}

Return exactly this JSON structure:
{{
  "candidate": "string",
  "candidate_is_concept": true,
  "context_is_domain_relevant": true,
  "candidate_depends_on_context": false,
  "is_incidental_named_entity": false,
  "decision": "retain|review|reject",
  "ontology_category": "allowed category",
  "concept_type": "technical_concept|class|system|component|property|application|operation|regulatory_concept|alias_candidate|generic_expression|artifact|other",
  "domain_relevance": 0.0,
  "term_specificity": 0.0,
  "context_support": 0.0,
  "ontology_suitability": 0.0,
  "genericity": 0.0,
  "confidence": 0.0,
  "rejection_reasons": [],
  "brief_reason": "one short evidence-grounded sentence"
}}"""


def response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidate": {"type": "string"},
            "candidate_is_concept": {"type": "boolean"},
            "context_is_domain_relevant": {"type": "boolean"},
            "candidate_depends_on_context": {"type": "boolean"},
            "is_incidental_named_entity": {"type": "boolean"},
            "decision": {"type": "string", "enum": sorted(ALLOWED_DECISIONS)},
            "ontology_category": {"type": "string", "enum": sorted(ALLOWED_CATEGORIES)},
            "concept_type": {"type": "string", "enum": sorted(ALLOWED_CONCEPT_TYPES)},
            "domain_relevance": {"type": "number", "minimum": 0, "maximum": 1},
            "term_specificity": {"type": "number", "minimum": 0, "maximum": 1},
            "context_support": {"type": "number", "minimum": 0, "maximum": 1},
            "ontology_suitability": {"type": "number", "minimum": 0, "maximum": 1},
            "genericity": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rejection_reasons": {
                "type": "array",
                "items": {"type": "string"},
            },
            "brief_reason": {"type": "string"},
        },
        "required": [
            "candidate",
            "candidate_is_concept",
            "context_is_domain_relevant",
            "candidate_depends_on_context",
            "is_incidental_named_entity",
            "decision",
            "ontology_category",
            "concept_type",
            "domain_relevance",
            "term_specificity",
            "context_support",
            "ontology_suitability",
            "genericity",
            "confidence",
            "rejection_reasons",
            "brief_reason",
        ],
        "additionalProperties": False,
    }


def cache_key_for(payload: dict[str, Any], model: str, options: dict[str, Any]) -> str:
    material = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "options": options,
        "payload": payload,
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ollama_chat(
    url: str,
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, Any],
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    request_payload = {
        "model": model,
        "messages": messages,
        "format": response_schema(),
        "stream": False,
        "options": options,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body.get("message", {}).get("content", "")
    if not content:
        raise ValueError(f"Ollama response did not contain message.content: {body}")
    return str(content), body


def extract_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        value = json.loads(content)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(content[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("No valid JSON object could be extracted from model response")


def bounded_number(value: Any, field: str) -> float:
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise ValueError(f"{field} must be finite")
    return float(np.clip(number, 0.0, 1.0))


def validate_result(parsed: dict[str, Any], expected_candidate: str) -> dict[str, Any]:
    decision = str(parsed.get("decision", "")).strip().casefold()
    category = str(parsed.get("ontology_category", "")).strip().casefold()
    concept_type = str(parsed.get("concept_type", "")).strip().casefold()

    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"Invalid decision: {decision!r}")
    if category not in ALLOWED_CATEGORIES:
        raise ValueError(f"Invalid ontology_category: {category!r}")
    if concept_type not in ALLOWED_CONCEPT_TYPES:
        raise ValueError(f"Invalid concept_type: {concept_type!r}")

    reasons = parsed.get("rejection_reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    if not isinstance(reasons, list):
        raise ValueError("rejection_reasons must be a list")
    normalized_reasons = []
    for reason in reasons:
        normalized = str(reason).strip().casefold().replace(" ", "_")
        if normalized in ALLOWED_REJECTION_REASONS:
            normalized_reasons.append(normalized)
        elif normalized:
            normalized_reasons.append(normalized)

    result = {
        "candidate": expected_candidate,
        "candidate_is_concept": bool(parsed.get("candidate_is_concept", False)),
        "context_is_domain_relevant": bool(parsed.get("context_is_domain_relevant", False)),
        "candidate_depends_on_context": bool(parsed.get("candidate_depends_on_context", False)),
        "is_incidental_named_entity": bool(parsed.get("is_incidental_named_entity", False)),
        "decision": decision,
        "ontology_category": category,
        "concept_type": concept_type,
        "domain_relevance": bounded_number(parsed.get("domain_relevance"), "domain_relevance"),
        "term_specificity": bounded_number(parsed.get("term_specificity"), "term_specificity"),
        "context_support": bounded_number(parsed.get("context_support"), "context_support"),
        "ontology_suitability": bounded_number(parsed.get("ontology_suitability"), "ontology_suitability"),
        "genericity": bounded_number(parsed.get("genericity"), "genericity"),
        "confidence": bounded_number(parsed.get("confidence"), "confidence"),
        "rejection_reasons": sorted(set(normalized_reasons)),
        "brief_reason": safe_text(parsed.get("brief_reason", ""))[:500],
    }
    return result


def fallback_review(candidate: str, error: str) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "candidate_is_concept": False,
        "context_is_domain_relevant": False,
        "candidate_depends_on_context": False,
        "is_incidental_named_entity": False,
        "decision": "review",
        "ontology_category": "none",
        "concept_type": "other",
        "domain_relevance": 0.5,
        "term_specificity": 0.5,
        "context_support": 0.0,
        "ontology_suitability": 0.5,
        "genericity": 0.5,
        "confidence": 0.0,
        "rejection_reasons": ["ambiguous_without_sufficient_evidence"],
        "brief_reason": f"Automatic review fallback after invalid model output: {error[:300]}",
    }


def call_with_cache(
    payload: dict[str, Any],
    model: str,
    url: str,
    cache_dir: Path,
    options: dict[str, Any],
    timeout: int,
    retries: int,
    force: bool,
) -> LLMResult:
    key = cache_key_for(payload, model, options)
    cache_file = cache_dir / f"{key}.json"

    if cache_file.is_file() and not force:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return LLMResult(
            parsed=cached["parsed"],
            raw_content=cached.get("raw_content", ""),
            elapsed_seconds=float(cached.get("elapsed_seconds", 0.0)),
            attempts=int(cached.get("attempts", 1)),
            cache_hit=True,
            cache_key=key,
            error=str(cached.get("error", "")),
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(payload)},
    ]
    started = time.time()
    raw_content = ""
    final_error = ""

    for attempt in range(1, retries + 2):
        try:
            raw_content, _ = ollama_chat(url, model, messages, options, timeout)
            parsed = validate_result(
                extract_json_object(raw_content),
                payload["candidate"],
            )
            elapsed = time.time() - started
            cached = {
                "cache_key": key,
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "payload": payload,
                "parsed": parsed,
                "raw_content": raw_content,
                "elapsed_seconds": elapsed,
                "attempts": attempt,
                "error": "",
            }
            cache_file.write_text(
                json.dumps(cached, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return LLMResult(parsed, raw_content, elapsed, attempt, False, key)
        except Exception as exc:
            final_error = f"{type(exc).__name__}: {exc}"
            if attempt <= retries:
                time.sleep(min(2 ** attempt, 8))

    elapsed = time.time() - started
    parsed = fallback_review(payload["candidate"], final_error)
    cached = {
        "cache_key": key,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "payload": payload,
        "parsed": parsed,
        "raw_content": raw_content,
        "elapsed_seconds": elapsed,
        "attempts": retries + 1,
        "error": final_error,
    }
    cache_file.write_text(
        json.dumps(cached, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return LLMResult(parsed, raw_content, elapsed, retries + 1, False, key, final_error)


def qwen_score(row: pd.Series) -> float:
    relevance = float(row["domain_relevance"])
    specificity = float(row["term_specificity"])
    context = float(row["context_support"])
    suitability = float(row["ontology_suitability"])
    genericity = float(row["genericity"])

    score = (
        0.20 * relevance
        + 0.30 * specificity
        + 0.15 * context
        + 0.35 * suitability
        - 0.40 * genericity
    )

    # Context must not rescue a candidate that is not itself a concept.
    if not bool(row.get("candidate_is_concept", False)):
        score *= 0.20
    if bool(row.get("candidate_depends_on_context", False)):
        score -= 0.25
    if bool(row.get("is_incidental_named_entity", False)):
        score -= 0.35
    if bool(row.get("is_obvious_artifact", False)):
        score = 0.0

    return float(np.clip(score, 0.0, 1.0))


def apply_protection(row: pd.Series) -> tuple[str, bool, str]:
    decision = str(row["decision"])
    confidence = float(row["confidence"])
    suitability = float(row["ontology_suitability"])
    genericity = float(row["genericity"])
    normalized = normalize_term(row["normalized_candidate"])

    # Deterministic artifacts remain rejected, regardless of LLM inconsistency.
    if bool(row.get("is_obvious_artifact", False)):
        return "reject", False, "deterministic_artifact_rule"

    # Core ontology anchors cannot be hard-rejected by a single 7B judgment.
    if normalized in CORE_SCOPE_TERMS and decision == "reject":
        return "review", True, "core_scope_anchor"

    # No low-confidence hard decision is permitted.
    if confidence < 0.60:
        return "review", True, f"low_decision_confidence:{confidence:.2f}"

    if decision != "reject":
        return decision, False, ""

    # A categorical rejection must also contain meaningful negative evidence.
    if genericity < 0.55 and suitability >= 0.40:
        return "review", True, "weak_negative_evidence"

    protections = []
    if bool(row["is_acronym_shape"]):
        protections.append("acronym_shape")
    if int(row["independent_method_support"]) >= 3:
        protections.append("three_or_more_extraction_methods")
    if (
        int(row["corpus_occurrence_count"]) >= 3
        and float(row["context_support"]) >= 0.60
        and suitability >= 0.50
    ):
        protections.append("repeated_technical_context_support")

    if protections:
        return "review", True, ";".join(protections)
    return decision, False, ""


def hard_reject_mask(frame: pd.DataFrame) -> pd.Series:
    """Safe rejection gate used by the recommended filtering policy."""
    return (
        frame["protected_decision"].eq("reject")
        & frame["confidence"].ge(0.75)
        & frame["genericity"].ge(0.60)
        & frame["ontology_suitability"].le(0.40)
    )


def thresholds(start: float, stop: float, step: float) -> list[float]:
    if step <= 0 or start > stop:
        raise ValueError("Invalid threshold range")
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def load_gt(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt = pd.read_excel(path, sheet_name=GT_SHEET, engine="openpyxl")
    negatives = pd.read_excel(path, sheet_name=NEGATIVE_SHEET, engine="openpyxl")
    gt = gt[
        gt["keep_for_tasks"].astype(str).str.contains(
            "term_extraction", case=False, na=False
        )
    ].copy()
    gt["normalized_gt"] = gt["term_canonical"].map(normalize_term)
    gt = gt[gt["normalized_gt"].ne("")]
    status = gt["grounded_status_v2"].fillna("").astype(str).str.casefold()
    strict = gt[status.ne(MANUAL_REVIEW_STATUS)].drop_duplicates("normalized_gt").copy()
    review = gt[status.eq(MANUAL_REVIEW_STATUS)].drop_duplicates("normalized_gt").copy()
    negatives["normalized_negative"] = negatives["candidate"].map(normalize_term)
    return strict, review, negatives


def annotate_for_evaluation(
    results: pd.DataFrame,
    strict: pd.DataFrame,
    review: pd.DataFrame,
    negatives: pd.DataFrame,
) -> pd.DataFrame:
    strict_set = set(strict["normalized_gt"])
    review_set = set(review["normalized_gt"])
    negative_set = set(negatives["normalized_negative"])
    output = results.copy()
    output["match_type"] = output["normalized_candidate"].map(
        lambda term: "strict_exact" if term in strict_set else (
            "manual_review_exact" if term in review_set else "unmatched"
        )
    )
    output["negative_benchmark_hit"] = output["normalized_candidate"].isin(negative_set)
    return output


def evaluate_mask(
    results: pd.DataFrame,
    retain_mask: pd.Series,
    strict: pd.DataFrame,
    policy: str,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    retained = results[retain_mask].copy()
    rejected = results[~retain_mask].copy()

    scored = retained[retained["match_type"].ne("manual_review_exact")]
    predicted = set(scored["normalized_candidate"])
    gold = set(strict["normalized_gt"])
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    weights = (
        strict["priority"].fillna("low").astype(str).str.casefold()
        .map(PRIORITY_WEIGHTS).fillna(1.0)
    )
    weighted_recall = float(
        weights[strict["normalized_gt"].isin(predicted)].sum() / weights.sum()
    ) if weights.sum() else 0.0

    tp_removed = int(rejected["match_type"].eq("strict_exact").sum())
    fp_removed = int(rejected["match_type"].eq("unmatched").sum())
    metric = {
        "policy": policy,
        "threshold": threshold,
        "input_candidate_count": len(results),
        "retained_candidate_count": len(retained),
        "rejected_candidate_count": len(rejected),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "priority_weighted_recall": weighted_recall,
        "tp_removed": tp_removed,
        "fp_removed": fp_removed,
        "negative_hits_retained": int(retained["negative_benchmark_hit"].sum()),
        "negative_hits_removed": int(rejected["negative_benchmark_hit"].sum()),
        "smoothed_filter_efficiency": fp_removed / (tp_removed + 1),
    }
    return metric, retained, rejected


def plot_metrics(metrics: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, group in metrics.groupby("policy", sort=False):
        group = group.sort_values("threshold")
        label = {
            "retain_only_gated": "Retain only, concept-gated",
            "retain_or_review_gated": "Retain + review, safe-gated",
            "safe_hard_rejection": "Safe hard rejection",
        }.get(policy, policy)
        ax.plot(group["threshold"], group["f1"], marker="o", linewidth=2, label=label)
        best = group.loc[group["f1"].idxmax()]
        ax.annotate(
            f"{best['f1']:.3f} at {best['threshold']:.2f}",
            (best["threshold"], best["f1"]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Qwen validation-score threshold")
    ax.set_ylabel("F1 score")
    ax.set_title("Qwen2.5:7B-Instruct UAV term validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def check_ollama(url: str, timeout: int) -> None:
    base = url.split("/api/")[0]
    tags_url = f"{base}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=min(timeout, 10)) as response:
            json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {base}. Start it with 'ollama serve'. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evidence-grounded UAV ontology-term validation with local Qwen via Ollama."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--limit", type=int, default=0, help="0 processes every candidate")
    parser.add_argument("--top-contexts", type=int, default=1)
    parser.add_argument("--max-occurrences", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=500)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.10)
    parser.add_argument("--force", action="store_true", help="Ignore valid cache entries")
    parser.add_argument("--skip-ollama-check", action="store_true")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run LLM inference but skip GT loading and evaluation",
    )
    args = parser.parse_args()

    limit = None if args.limit <= 0 else args.limit
    if not args.inference_only and not args.ground_truth.is_file():
        raise FileNotFoundError(args.ground_truth)
    if args.top_contexts < 0:
        raise ValueError("--top-contexts cannot be negative")

    output = args.output_dir
    cache_dir = output / "cache"
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    run_started_perf = time.perf_counter()
    run_started_utc = utc_now_iso()
    stage_timings: dict[str, float] = {}

    ollama_check_started = time.perf_counter()
    if not args.skip_ollama_check:
        check_ollama(args.ollama_url, args.timeout)
    stage_timings["ollama_check_seconds"] = time.perf_counter() - ollama_check_started

    evidence_started = time.perf_counter()
    candidates = load_candidates(args.input, limit)
    sentences = load_corpus_sentences(args.chunks)
    candidates = retrieve_contexts(
        candidates,
        sentences,
        args.top_contexts,
        args.max_occurrences,
    )
    stage_timings["evidence_retrieval_seconds"] = time.perf_counter() - evidence_started

    context_export = candidates.copy()
    context_export["additional_contexts"] = context_export["additional_contexts"].map(
        lambda value: json.dumps(value if isinstance(value, list) else [], ensure_ascii=False)
    )
    context_export["additional_context_scores"] = context_export["additional_context_scores"].map(
        lambda value: json.dumps(value if isinstance(value, list) else [])
    )
    context_export.to_csv(output / "candidate_evidence.csv", index=False)

    options = {
        "temperature": args.temperature,
        "seed": args.seed,
        "num_predict": args.num_predict,
        "num_ctx": args.num_ctx,
    }

    request_log = output / "qwen_requests.jsonl"
    response_log = output / "qwen_raw_responses.jsonl"
    rows: list[dict[str, Any]] = []

    inference_started = time.perf_counter()
    inference_records: list[dict[str, Any]] = []
    print(f"Processing {len(candidates)} YAKE + MPNet candidates with {args.model}")
    print(f"Run started UTC: {run_started_utc}")
    print(f"Evidence retrieval time: {format_duration(stage_timings['evidence_retrieval_seconds'])}")

    for position, (_, row) in enumerate(candidates.iterrows(), start=1):
        payload = candidate_payload(row)
        request_record = {
            "candidate_row_id": int(row["candidate_row_id"]),
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "payload": payload,
        }
        with request_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(request_record, ensure_ascii=False) + "\n")

        result = call_with_cache(
            payload,
            args.model,
            args.ollama_url,
            cache_dir,
            options,
            args.timeout,
            args.retries,
            args.force,
        )

        parsed = result.parsed
        record = row.to_dict()
        record.update(parsed)
        record["rejection_reasons"] = json.dumps(parsed["rejection_reasons"])
        record["raw_response"] = result.raw_content
        record["elapsed_seconds"] = result.elapsed_seconds
        record["attempts"] = result.attempts
        record["cache_hit"] = result.cache_hit
        record["cache_key"] = result.cache_key
        record["inference_error"] = result.error
        record["is_acronym_shape"] = payload["is_acronym_shape"]
        record["independent_method_support"] = payload["independent_method_support"]
        record["is_obvious_artifact"] = payload["is_obvious_artifact"]
        record["is_core_scope_term"] = payload["is_core_scope_term"]
        rows.append(record)

        elapsed_total = time.perf_counter() - inference_started
        average_seconds = elapsed_total / position
        remaining_candidates = len(candidates) - position
        estimated_remaining = average_seconds * remaining_candidates
        estimated_total = average_seconds * len(candidates)
        inference_records.append({
            "candidate_row_id": int(row["candidate_row_id"]),
            "candidate": row["candidate"],
            "position": position,
            "candidate_elapsed_seconds": result.elapsed_seconds,
            "run_elapsed_seconds": elapsed_total,
            "average_seconds_per_candidate": average_seconds,
            "estimated_remaining_seconds": estimated_remaining,
            "estimated_total_seconds": estimated_total,
            "cache_hit": result.cache_hit,
            "attempts": result.attempts,
            "inference_error": result.error,
        })

        with response_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "candidate_row_id": int(row["candidate_row_id"]),
                "candidate": row["candidate"],
                "parsed": parsed,
                "raw_content": result.raw_content,
                "elapsed_seconds": result.elapsed_seconds,
                "attempts": result.attempts,
                "cache_hit": result.cache_hit,
                "error": result.error,
            }, ensure_ascii=False) + "\n")

        print(
            f"[{position:4d}/{len(candidates)}] "
            f"{row['candidate'][:38]:38s} "
            f"decision={parsed['decision']:6s} "
            f"conf={parsed['confidence']:.2f} "
            f"item={format_duration(result.elapsed_seconds)} "
            f"elapsed={format_duration(elapsed_total)} "
            f"ETA={format_duration(estimated_remaining)} "
            f"avg={average_seconds:.2f}s "
            f"cache={result.cache_hit}"
        )

        # Persist partial outputs after every candidate, including efficiency records.
        pd.DataFrame(rows).to_csv(output / "qwen_validation_results.partial.csv", index=False)
        pd.DataFrame(inference_records).to_csv(output / "inference_timing.partial.csv", index=False)

    stage_timings["llm_inference_seconds"] = time.perf_counter() - inference_started
    postprocessing_started = time.perf_counter()
    timing_frame = pd.DataFrame(inference_records)
    timing_frame.to_csv(output / "inference_timing.csv", index=False)

    results = pd.DataFrame(rows)
    results["llm_score"] = results.apply(qwen_score, axis=1)

    protection = results.apply(apply_protection, axis=1, result_type="expand")
    protection.columns = ["protected_decision", "protected_from_rejection", "protection_reason"]
    results = pd.concat([results, protection], axis=1)

    results["additional_contexts"] = results["additional_contexts"].map(
        lambda value: json.dumps(value if isinstance(value, list) else [], ensure_ascii=False)
    )
    results["additional_context_scores"] = results["additional_context_scores"].map(
        lambda value: json.dumps(value if isinstance(value, list) else [])
    )
    results.to_csv(output / "qwen_validation_results.csv", index=False)

    invalid = results[results["inference_error"].astype(str).str.len().gt(0)]
    invalid.to_csv(output / "invalid_qwen_responses.csv", index=False)
    results[results["protected_decision"].eq("review")].to_csv(
        output / "qwen_review_candidates.csv", index=False
    )

    category_summary = (
        results.groupby(["protected_decision", "ontology_category"], dropna=False)
        .size().rename("candidate_count").reset_index()
    )
    category_summary.to_csv(output / "category_summary.csv", index=False)

    reason_counter: Counter[str] = Counter()
    for value in results["rejection_reasons"]:
        try:
            reason_counter.update(json.loads(value))
        except Exception:
            pass
    pd.DataFrame([
        {"rejection_reason": reason, "candidate_count": count}
        for reason, count in reason_counter.most_common()
    ]).to_csv(output / "rejection_reason_summary.csv", index=False)

    manifest: dict[str, Any] = {
        "input": str(args.input),
        "chunks": str(args.chunks),
        "ground_truth": None if args.inference_only else str(args.ground_truth),
        "model": args.model,
        "ollama_url": args.ollama_url,
        "prompt_version": PROMPT_VERSION,
        "options": options,
        "candidate_count": len(results),
        "cache_hits": int(results["cache_hit"].sum()),
        "invalid_or_failed_responses": len(invalid),
        "mean_elapsed_seconds": float(results["elapsed_seconds"].mean()),
        "decision_counts": results["protected_decision"].value_counts().to_dict(),
    }

    if not args.inference_only:
        strict, review, negatives = load_gt(args.ground_truth)
        evaluated = annotate_for_evaluation(results, strict, review, negatives)
        evaluated.to_csv(output / "qwen_validation_results_with_gt.csv", index=False)

        metric_rows: list[dict[str, Any]] = []
        inventories: dict[tuple[str, float], tuple[pd.DataFrame, pd.DataFrame]] = {}

        # Explicit baseline for the YAKE + MPNet inventory before Qwen filtering.
        baseline_mask = pd.Series(True, index=evaluated.index)
        metric, retained, rejected = evaluate_mask(
            evaluated, baseline_mask, strict, "baseline_no_qwen_filter", 0.0
        )
        metric_rows.append(metric)
        inventories[("baseline_no_qwen_filter", 0.0)] = (retained, rejected)

        # Recommended safety-first policy: remove only high-confidence,
        # high-genericity, low-suitability rejections. This policy does not
        # depend on a score threshold, so it is evaluated once at threshold 0.
        safe_mask = ~hard_reject_mask(evaluated)
        metric, retained, rejected = evaluate_mask(
            evaluated, safe_mask, strict, "safe_hard_rejection", 0.0
        )
        metric_rows.append(metric)
        inventories[("safe_hard_rejection", 0.0)] = (retained, rejected)

        for threshold in thresholds(
            args.threshold_start,
            args.threshold_stop,
            args.threshold_step,
        ):
            retain_only_mask = (
                evaluated["protected_decision"].eq("retain")
                & evaluated["candidate_is_concept"].astype(bool)
                & ~evaluated["candidate_depends_on_context"].astype(bool)
                & ~evaluated["is_incidental_named_entity"].astype(bool)
                & evaluated["llm_score"].ge(threshold)
            )
            metric, retained, rejected = evaluate_mask(
                evaluated, retain_only_mask, strict, "retain_only_gated", threshold
            )
            metric_rows.append(metric)
            inventories[("retain_only_gated", threshold)] = (retained, rejected)

            retain_review_mask = (
                evaluated["protected_decision"].isin(["retain", "review"])
                & ~hard_reject_mask(evaluated)
                & evaluated["llm_score"].ge(threshold)
            )
            metric, retained, rejected = evaluate_mask(
                evaluated, retain_review_mask, strict, "retain_or_review_gated", threshold
            )
            metric_rows.append(metric)
            inventories[("retain_or_review_gated", threshold)] = (retained, rejected)

        metrics = pd.DataFrame(metric_rows).sort_values(
            ["f1", "precision", "recall"], ascending=[False, False, False]
        ).reset_index(drop=True)
        metrics.insert(0, "rank_by_f1", np.arange(1, len(metrics) + 1))
        metrics.to_csv(output / "qwen_threshold_metrics.csv", index=False)
        plot_metrics(metrics, output / "qwen_f1_vs_threshold.png")

        best = metrics.iloc[0]
        key = (str(best["policy"]), float(best["threshold"]))
        best_retained, best_rejected = inventories[key]
        best_retained.to_csv(output / "best_qwen_retained.csv", index=False)
        best_rejected.to_csv(output / "best_qwen_rejected.csv", index=False)

        manifest.update({
            "strict_gt_count": len(strict),
            "best_policy": key[0],
            "best_threshold": key[1],
            "best_f1": float(best["f1"]),
            "best_precision": float(best["precision"]),
            "best_recall": float(best["recall"]),
        })

        display = [
            "rank_by_f1", "policy", "threshold", "retained_candidate_count",
            "true_positives", "false_positives", "false_negatives",
            "precision", "recall", "f1", "tp_removed", "fp_removed",
        ]
        print("\n" + "=" * 105)
        print("Top Qwen filtering configurations")
        print("=" * 105)
        print(metrics[display].head(20).to_string(index=False))
        print(f"\nBest retained inventory: {output / 'best_qwen_retained.csv'}")
        print(f"Best rejected inventory: {output / 'best_qwen_rejected.csv'}")
        print(f"Metrics: {output / 'qwen_threshold_metrics.csv'}")
        print(f"F1 plot: {output / 'qwen_f1_vs_threshold.png'}")

    stage_timings["postprocessing_and_evaluation_seconds"] = time.perf_counter() - postprocessing_started
    total_runtime_seconds = time.perf_counter() - run_started_perf
    run_finished_utc = utc_now_iso()

    cache_hit_count = int(results["cache_hit"].sum())
    live_inference_count = len(results) - cache_hit_count
    live_rows = timing_frame[~timing_frame["cache_hit"]] if not timing_frame.empty else timing_frame
    mean_live_seconds = (
        float(live_rows["candidate_elapsed_seconds"].mean())
        if not live_rows.empty else 0.0
    )
    throughput_per_minute = (
        live_inference_count / (stage_timings["llm_inference_seconds"] / 60.0)
        if stage_timings["llm_inference_seconds"] > 0 else 0.0
    )
    efficiency_summary = pd.DataFrame([{
        "run_started_utc": run_started_utc,
        "run_finished_utc": run_finished_utc,
        "total_runtime_seconds": total_runtime_seconds,
        "total_runtime_hhmmss": format_duration(total_runtime_seconds),
        "ollama_check_seconds": stage_timings.get("ollama_check_seconds", 0.0),
        "evidence_retrieval_seconds": stage_timings.get("evidence_retrieval_seconds", 0.0),
        "llm_inference_seconds": stage_timings.get("llm_inference_seconds", 0.0),
        "postprocessing_and_evaluation_seconds": stage_timings.get("postprocessing_and_evaluation_seconds", 0.0),
        "candidate_count": len(results),
        "cache_hit_count": cache_hit_count,
        "live_inference_count": live_inference_count,
        "mean_live_seconds_per_candidate": mean_live_seconds,
        "candidates_per_minute": throughput_per_minute,
        "error_count": int(results["inference_error"].astype(str).str.len().gt(0).sum()),
    }])
    efficiency_summary.to_csv(output / "runtime_efficiency_summary.csv", index=False)

    manifest.update({
        "experiment": "qwen_validation_yake_mpnet_high_recall_inventory",
        "run_started_utc": run_started_utc,
        "run_finished_utc": run_finished_utc,
        "total_runtime_seconds": total_runtime_seconds,
        "total_runtime_hhmmss": format_duration(total_runtime_seconds),
        "stage_timings_seconds": stage_timings,
        "cache_hit_count": cache_hit_count,
        "live_inference_count": live_inference_count,
        "mean_live_seconds_per_candidate": mean_live_seconds,
        "candidates_per_minute": throughput_per_minute,
    })

    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("Runtime and efficiency summary")
    print("=" * 90)
    print(f"Total runtime: {format_duration(total_runtime_seconds)} ({total_runtime_seconds:.2f}s)")
    print(f"Evidence retrieval: {format_duration(stage_timings.get('evidence_retrieval_seconds', 0.0))}")
    print(f"LLM inference: {format_duration(stage_timings.get('llm_inference_seconds', 0.0))}")
    print(f"Postprocessing/evaluation: {format_duration(stage_timings.get('postprocessing_and_evaluation_seconds', 0.0))}")
    print(f"Live model calls: {live_inference_count}; cache hits: {cache_hit_count}")
    print(f"Mean live seconds/candidate: {mean_live_seconds:.2f}")
    print(f"Throughput: {throughput_per_minute:.2f} candidates/minute")
    print(f"Timing details: {output / 'inference_timing.csv'}")
    print(f"Efficiency summary: {output / 'runtime_efficiency_summary.csv'}")
    print(f"Structured Qwen results: {output / 'qwen_validation_results.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Cached and partial results were preserved.", file=sys.stderr)
        raise
