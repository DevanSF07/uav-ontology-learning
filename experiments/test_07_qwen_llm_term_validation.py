#!/usr/bin/env python3
"""Evidence-grounded UAV ontology-term validation with llama 3t.

This experiment starts from the current maximum-F1 inventory, expected by
past experiments to be:

    stage_outputs/05_seed_similarity_after_gliner/
        best_seed_filtered_candidates.csv

For every candidate, the script:

1. Uses the original source sentence from the candidate inventory.
2. Retrieves up to N additional exact-occurrence corpus sentences and ranks
   them with TF-IDF similarity.
3. Sends only candidate evidence, ontology scope, and validation instructions
   to a local Ollama model.
4. Requires a structured JSON judgment from llama3
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
    ollama pull llama
    ollama serve

Example pilot:
    python test_07_llama_llm_term_validation.py --limit 25

Complete run:
    python test_07_llama_llm_term_validation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
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
    / "stage_outputs/05_seed_similarity_after_gliner/"
      "best_seed_filtered_candidates.csv"
)
DEFAULT_CHUNKS = PROJECT_DIR / "stage_outputs/02_chunking/chunks.csv"
DEFAULT_GT = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_OUTPUT = PROJECT_DIR / "stage_outputs/llama_llm_term_validation"

DEFAULT_MODEL = "llama3"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
PROMPT_VERSION = "uav-term-validator-v1.0"
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

SYSTEM_PROMPT = """You validate candidate terms for a technical ontology about unmanned aircraft and UAV systems.

A valid candidate must represent a reusable domain concept, class, component, system, platform, operational concept, property, application, regulatory concept, risk concept, or relation-relevant entity.

Do not retain a candidate merely because it appears in UAV-related text.

Reject generic words and phrases, ordinary actions, isolated verbs, descriptive fragments, document or table artifacts, bibliographic metadata, incidental names, geographic references, phrases supported only by surrounding context, and concepts outside the ontology scope.

Preserve specialized technical terms and acronyms when the supplied corpus evidence supports a UAV-domain interpretation.

Use only the supplied evidence. Do not invent definitions, categories, or relationships. If evidence is insufficient or the candidate may be an alias, use review rather than making an unsupported decision.

Return one valid JSON object only. Do not return Markdown, analysis, comments, or text outside the JSON object."""

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

Allowed categories:
{categories}

Allowed decisions:
- retain: a specific, reusable UAV-domain ontology concept strongly supported by evidence
- review: potentially valid, ambiguous, overly broad, alias-like, or insufficiently evidenced
- reject: generic, malformed, incidental, artifact-like, bibliographic, or outside ontology scope

Candidate: {payload['candidate']}
Normalized candidate: {payload['normalized_candidate']}
Original source sentence: {payload['source_sentence'] or 'Not available'}
Corpus occurrence count: {payload['corpus_occurrence_count']}
Independent extraction-method support: {payload['independent_method_support']}
Acronym-shaped candidate: {payload['is_acronym_shape']}

Additional exact-occurrence corpus evidence:
{evidence_text}

Return exactly this JSON structure:
{{
  "candidate": "string",
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


def llama_score(row: pd.Series) -> float:
    score = (
        0.30 * float(row["domain_relevance"])
        + 0.25 * float(row["term_specificity"])
        + 0.20 * float(row["context_support"])
        + 0.25 * float(row["ontology_suitability"])
        - 0.30 * float(row["genericity"])
    )
    return float(np.clip(score, 0.0, 1.0))


def apply_protection(row: pd.Series) -> tuple[str, bool, str]:
    decision = str(row["decision"])
    if decision != "reject":
        return decision, False, ""

    protections = []
    if bool(row["is_acronym_shape"]):
        protections.append("acronym_shape")
    if int(row["independent_method_support"]) >= 3:
        protections.append("three_or_more_extraction_methods")
    if int(row["corpus_occurrence_count"]) >= 3 and float(row["context_support"]) >= 0.6:
        protections.append("repeated_corpus_support")

    if protections:
        return "review", True, ";".join(protections)
    return decision, False, ""


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
        label = "Retain only" if policy == "retain_only" else "Retain + review"
        ax.plot(group["threshold"], group["f1"], marker="o", linewidth=2, label=label)
        best = group.loc[group["f1"].idxmax()]
        ax.annotate(
            f"{best['f1']:.3f} at {best['threshold']:.2f}",
            (best["threshold"], best["f1"]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("llaman validation-score threshold")
    ax.set_ylabel("F1 score")
    ax.set_title("llama2.5:7B-Instruct UAV term validation")
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
        description="Evidence-grounded UAV ontology-term validation with local llama via Ollama."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--limit", type=int, default=0, help="0 processes every candidate")
    parser.add_argument("--top-contexts", type=int, default=3)
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

    if not args.skip_ollama_check:
        check_ollama(args.ollama_url, args.timeout)

    candidates = load_candidates(args.input, limit)
    sentences = load_corpus_sentences(args.chunks)
    candidates = retrieve_contexts(
        candidates,
        sentences,
        args.top_contexts,
        args.max_occurrences,
    )

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

    request_log = output / "lama.jsonl"
    response_log = output / "llama_raw_responses.jsonl"
    rows: list[dict[str, Any]] = []

    print(f"Processing {len(candidates)} candidates with {args.model}")
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
        rows.append(record)

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
            f"{row['candidate'][:45]:45s} "
            f"decision={parsed['decision']:6s} "
            f"confidence={parsed['confidence']:.2f} "
            f"cache={result.cache_hit}"
        )

        # Persist partial structured output after every candidate.
        pd.DataFrame(rows).to_csv(output / "llama_validation_results.partial.csv", index=False)

    results = pd.DataFrame(rows)
    results["llm_score"] = results.apply(llama_score, axis=1)

    protection = results.apply(apply_protection, axis=1, result_type="expand")
    protection.columns = ["protected_decision", "protected_from_rejection", "protection_reason"]
    results = pd.concat([results, protection], axis=1)

    results["additional_contexts"] = results["additional_contexts"].map(
        lambda value: json.dumps(value if isinstance(value, list) else [], ensure_ascii=False)
    )
    results["additional_context_scores"] = results["additional_context_scores"].map(
        lambda value: json.dumps(value if isinstance(value, list) else [])
    )
    results.to_csv(output / "llama_validation_results.csv", index=False)

    invalid = results[results["inference_error"].astype(str).str.len().gt(0)]
    invalid.to_csv(output / "invalid_llama_responses.csv", index=False)
    results[results["protected_decision"].eq("review")].to_csv(
        output / "llama_review_candidates.csv", index=False
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
        evaluated.to_csv(output / "llama_validation_results_with_gt.csv", index=False)

        metric_rows: list[dict[str, Any]] = []
        inventories: dict[tuple[str, float], tuple[pd.DataFrame, pd.DataFrame]] = {}
        for threshold in thresholds(
            args.threshold_start,
            args.threshold_stop,
            args.threshold_step,
        ):
            retain_only_mask = (
                evaluated["protected_decision"].eq("retain")
                & evaluated["llm_score"].ge(threshold)
            )
            metric, retained, rejected = evaluate_mask(
                evaluated, retain_only_mask, strict, "retain_only", threshold
            )
            metric_rows.append(metric)
            inventories[("retain_only", threshold)] = (retained, rejected)

            retain_review_mask = (
                evaluated["protected_decision"].isin(["retain", "review"])
                & evaluated["llm_score"].ge(threshold)
            )
            metric, retained, rejected = evaluate_mask(
                evaluated, retain_review_mask, strict, "retain_or_review", threshold
            )
            metric_rows.append(metric)
            inventories[("retain_or_review", threshold)] = (retained, rejected)

        metrics = pd.DataFrame(metric_rows).sort_values(
            ["f1", "precision", "recall"], ascending=[False, False, False]
        ).reset_index(drop=True)
        metrics.insert(0, "rank_by_f1", np.arange(1, len(metrics) + 1))
        metrics.to_csv(output / "llama_threshold_metrics.csv", index=False)
        plot_metrics(metrics, output / "llama_f1_vs_threshold.png")

        best = metrics.iloc[0]
        key = (str(best["policy"]), float(best["threshold"]))
        best_retained, best_rejected = inventories[key]
        best_retained.to_csv(output / "best_llama_retained.csv", index=False)
        best_rejected.to_csv(output / "best_llama_rejected.csv", index=False)

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
        print("Top llama filtering configurations")
        print("=" * 105)
        print(metrics[display].head(20).to_string(index=False))
        print(f"\nBest retained inventory: {output / 'best_llama_retained.csv'}")
        print(f"Best rejected inventory: {output / 'best_llama_rejected.csv'}")
        print(f"Metrics: {output / 'llama_threshold_metrics.csv'}")
        print(f"F1 plot: {output / 'llama_f1_vs_threshold.png'}")

    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nStructured llama results: {output / 'llama_validation_results.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Cached and partial results were preserved.", file=sys.stderr)
        raise
