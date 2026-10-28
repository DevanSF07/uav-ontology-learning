#!/usr/bin/env python3
"""
Precision-first ontology pipeline v11: recovery similarity and one batched LLM revalidation inside each chunk pass.
- Reuses completed baseline artifacts whenever final_ontology_candidates.csv contains chunk_id and context_text.
- Within each chunk, current accepted terms are the only seeds.
- Candidate/backlog terms only; original rejected terms are excluded.
- Term and exact occurrence-sentence embeddings are scored independently.
- Previous-one-chunk accepted term/sentence is used only for embedding retrieval; it is never sent to the LLM.
- The current chunk remains the sole LLM evidence and acceptance authority.
- Strict LLM revalidation returns ACCEPT, REVIEW, or REJECT.

Controlled baseline policy:
- Run the same broad chunk-first extraction policy first.
- Use the original Ground_Truth.xlsx / S1_GT_TERMS for broad evaluation.
- Filter saved predictions by structural labels and evaluate the subset against the final structural GT.
- Keep the established metric names: strict, safe_extended, context_extended.
- Do not perform semantic recall expansion or automatic promotion in this baseline.
- Keep the extractor set fixed to YAKE, acronyms, TF-IDF, and noun phrases.

Main fixes vs previous version:
1. Semantic seed similarity is diagnostic only. It never promotes a term.
2. LLM-grounded terms may retain multiple valid ontology labels; multi-label output is never rejected solely for having multiple labels.
3. Extractor-only terms become CANDIDATE/BACKLOG, never PROMOTED.
4. PROMOTED is reserved for a future LLM revalidation stage; this script does not auto-promote.
5. Obvious OCR/page/parser artifacts are hard rejected before scoring.
6. Gate prompt includes explicit out-of-scope rules.
7. Accepted output contains only final_state == accepted.

Dual evaluation: broad S1_GT_TERMS from Ground_Truth.xlsx plus final structural/classification GT. The same normalization, conservative lemmatization, delimiter splitting, and alias-family logic are used for predictions and both GTs.

Recommended first debug run:
Open this file in VS Code and select Run Python File. Defaults run YAKE, acronyms, TF-IDF, noun phrases, TextRank, and KeyBERT-MPNet.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from olbench.extractors import acronym_extract, keybert_extract, noun_phrases, textrank, tfidf, yake_extract  # noqa: E402
from olbench.preprocess import chunk_text, clean_mmd  # noqa: E402
from olbench.seed_similarity import DOMAIN_SEEDS, normalize_term, safe_text, score_seed_similarity  # noqa: E402
import gt_evaluator as controlled_eval  # noqa: E402

DEFAULT_INPUT_FILE = PROJECT_DIR / "data/combined_corpus_raw.txt"
DEFAULT_BROAD_GT_FILE = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_STRUCTURAL_GT_FILE = PROJECT_DIR / "data/UAV_FINAL_STRUCTURAL_CLASSIFICATION_GT_V1.xlsx"
BROAD_GT_SHEET = "S1_GT_TERMS"
BROAD_GT_SCOPE = "text_observed"
STRUCTURAL_GT_SHEET = "GT_TERMS"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "stage_outputs/precision_first_ontology_all_extractors"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b-instruct"

S1_SHEET = "S1_GT_TERMS"
S3_SHEET = "S3_GT_CONCEPTS"
MANUAL_REVIEW_STATUS = "needs_manual_review"

LLM_GATE_SYSTEM_PROMPT = "You are a precise ontology gate for technical document chunks. Return JSON only."
LLM_EVIDENCE_SYSTEM_PROMPT = "You extract grounded ontology terms from one chunk. Return JSON only."

HIGH_CONFIDENCE = {"high", "medium"}
ACCEPTED_STATE = "accepted"
PROMOTED_STATE = "promoted"        # reserved for future LLM revalidation script
CANDIDATE_STATE = "candidate"
BACKLOG_STATE = "backlog"
REJECTED_STATE = "rejected"

VALID_ONTOLOGY_TYPES = {
    "classification",
    "mission_usecase",
    "operating_condition",
    "environment",
    "scenario_operation",
    "system",
    "subsystem",
    "component",
    "part",
    "flow",
    "port",
    "interface",
}

STRUCTURAL_ONTOLOGY_TYPES = {
    "classification",
    "system",
    "subsystem",
    "component",
    "part",
    "flow",
    "interface",
}

# Conservative hard-reject patterns for obvious artifacts/fragments.
# Add more patterns here as you inspect outputs.
HARD_REJECT_PATTERNS = [
    r"missingpagefail",
    r"pagefail",
    r"missingpage",
    r"^\s*$",
    r"^nan$",
    r"^none$",
    r"^null$",
    r"^\d+$",
    r"^\W+$",
    r"divides\s+far\s+part",
    r"true\s+in\s+manned\s+aviation",
    r"\bfar\s+part\b",
    r"\bac\s*\d+(\.\d+)?\b",
    r"^table\s+\d+(\.\d+)?$",
    r"^figure\s+\d+(\.\d+)?$",
]

GENERIC_BAD_TERMS = {
    "characteristics",
    "example",
    "examples",
    "section",
    "table",
    "figure",
    "point of view",
    "points of view",
    "wide variety",
    "large number",
    "comprehensive classification",  # too broad as a term; classification alone is okay
}


def parse_json_object(content: str) -> dict[str, Any]:
    content = safe_text(content)
    if not content:
        raise ValueError("Empty model response")
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(content[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No JSON object found in model response")


def ollama_chat_json(
    url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
) -> tuple[dict[str, Any], str]:
    import urllib.request

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    message = body.get("message", {}) if isinstance(body, dict) else {}
    content = safe_text(message.get("content", ""))
    return parse_json_object(content), content


def load_chunks(input_file: Path, max_words: int, overlap_sentences: int) -> pd.DataFrame:
    if input_file.suffix.lower() == ".csv":
        chunks = pd.read_csv(input_file, encoding="utf-8")
        if "text" not in chunks.columns:
            if "chunk_text" in chunks.columns:
                chunks = chunks.rename(columns={"chunk_text": "text"})
            else:
                raise ValueError(f"{input_file} must contain a text or chunk_text column")
        if "chunk_id" not in chunks.columns:
            chunks = chunks.copy()
            chunks["chunk_id"] = [f"chunk_{index:06d}" for index in range(len(chunks))]
        if "chunk_index" not in chunks.columns:
            chunks = chunks.copy()
            chunks["chunk_index"] = range(len(chunks))
        if "section" not in chunks.columns:
            chunks = chunks.copy()
            chunks["section"] = "document"
        chunks["text"] = chunks["text"].fillna("").astype(str)
        return chunks[["chunk_id", "chunk_index", "section", "text"]].reset_index(drop=True)

    cleaned_text, _ = clean_mmd(input_file)
    chunks = chunk_text(cleaned_text, max_words=max_words, overlap_sentences=overlap_sentences)
    if chunks.empty:
        raise RuntimeError("No chunks were produced from the input corpus.")
    return chunks[["chunk_id", "chunk_index", "section", "text"]].reset_index(drop=True)


def build_gate_prompt(chunk: pd.Series) -> str:
    return f"""
Analyse one document chunk for ontology evidence.

Chunk ID: {safe_text(chunk.get('chunk_id', ''))}
Section: {safe_text(chunk.get('section', ''))}

Target ontology categories:
- classification
- mission_usecase
- operating_condition
- environment
- scenario_operation
- system
- subsystem
- component
- part
- flow
- port
- interface

Out of scope unless the sentence directly supports one of the target ontology categories:
- regulation
- certification
- legal compliance
- organisation names
- country names
- history
- market discussion
- generic background
- OCR artifacts
- page markers
- broken table/page extraction fragments

Question:
Does this chunk contain direct or indirect evidence useful for one or more target ontology categories?

Return JSON only:
{{
  "gate_decision": "YES" or "NO",
  "reason": "one short reason",
  "possible_categories": ["..."],
  "out_of_scope_theme": "... or null"
}}

Rules:
- Use only the given chunk.
- Do not use previous chunks.
- Do not invent concepts outside the text.
- If most of the chunk is regulatory/legal/historical/background and has no reusable system ontology evidence, return NO.
- If the chunk contains both out-of-scope and target ontology evidence, return YES but list only the target categories.

Chunk text:
{safe_text(chunk.get('text', ''))}
""".strip()


def build_evidence_prompt(chunk: pd.Series) -> str:
    return f"""
The chunk has passed the ontology evidence gate.

Chunk ID: {safe_text(chunk.get('chunk_id', ''))}

Extract ontology-relevant evidence from this chunk.

Return JSON only with this structure:
{{
  "chunk_id": "...",
  "concepts_present": [
    {{
      "category": "classification | mission_usecase | operating_condition | environment | scenario_operation | system | subsystem | component | part | flow | port | interface | other",
      "evidence_sentence": "exact sentence from chunk",
      "confidence": "high | medium | low"
    }}
  ],
  "candidate_terms": [
    {{
      "term": "...",
      "proposed_type": "classification | mission_usecase | operating_condition | environment | scenario_operation | system | subsystem | component | part | flow | port | interface | other",
      "evidence_sentence": "exact sentence from chunk",
      "confidence": "high | medium | low",
      "reason": "short reason"
    }}
  ]
}}

Rules:
- Extract only terms grounded in the chunk.
- Prefer explicit noun phrases.
- Allow multiple ontology categories in the same chunk and for the same term.
- If a term has multiple valid labels, join them with " | "; preserve all valid labels.
- A term must not be rejected merely because it has multiple valid labels.
- Platform variants such as Mini UAVs, MALE UAVs, HALE UAVs, VTOL UAVs, MUAVs, and MAVs may be typed as system, classification, or both when supported by the sentence.
- When a slash expresses alternatives, return the combined source term; the pipeline will split it into child terms while preserving evidence and labels. Examples: MUAV/MAV -> MUAV and MAV; EO/IR sensor -> EO sensor and IR sensor.
- Do not split composite operating labels where the slash expresses a combined condition, such as VLA/LOS or VLA/BLOS.
- Type each term using its local evidence sentence, not the whole document.
- Do not include regulation, certification, organisation, country, legal terms, OCR artifacts, page markers, or broken phrase fragments unless the exact sentence directly supports a target ontology category.
- Do not output generic fragments such as "large number", "wide variety", "true in manned aviation", "divides FAR Part", or page-failure markers.
- Do not type uncertain terms as accepted categories. Use proposed_type="other" and confidence="low" if uncertain.

Chunk text:
{safe_text(chunk.get('text', ''))}
""".strip()


def is_hard_noise(term: str, normalized: str) -> bool:
    raw = safe_text(term)
    norm = safe_text(normalized)
    text = f"{raw} {norm}".casefold().strip()
    if norm in GENERIC_BAD_TERMS:
        return True
    if len(norm) <= 1:
        return True
    return any(re.search(pattern, text) for pattern in HARD_REJECT_PATTERNS)


TYPE_ALIASES = {
    "mission": "mission_usecase",
    "use_case": "mission_usecase",
    "usecase": "mission_usecase",
    "scenario": "scenario_operation",
    "operation": "scenario_operation",
    "condition": "operating_condition",
    "communication": "interface",
    "link": "interface",
    "platform_variant": "system",
    "system_variant": "system",
}

PROTECTED_SLASH_TERMS = {
    "vla/los",
    "vla/blos",
}

# Splitting applies identically to predictions and GT. Add a phrase here only when
# punctuation is an intrinsic part of a single ontology label.
PROTECTED_DELIMITED_TERMS = {
    "vla/los",
    "vla/blos",
}


def parse_proposed_types(value: Any) -> list[str]:
    """Return every valid ontology label from scalar, list, or pipe-separated LLM output."""
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = re.split(r"\s*(?:\||,|;)\s*", safe_text(value))

    labels: list[str] = []
    for raw in raw_values:
        label = safe_text(raw).casefold().strip().replace(" ", "_")
        label = TYPE_ALIASES.get(label, label)
        if label in VALID_ONTOLOGY_TYPES and label not in labels:
            labels.append(label)
    return labels


def canonical_type_text(value: Any) -> str:
    labels = parse_proposed_types(value)
    return " | ".join(labels) if labels else "other"


def has_valid_type(proposed_type: Any) -> bool:
    return bool(parse_proposed_types(proposed_type))


def expand_term_forms(term: str) -> list[str]:
    """Expand slash, comma, and ampersand alternatives identically for GT and predictions.

    The original surface form is retained by callers. This function emits child forms for
    matching/canonicalization. Composite labels in PROTECTED_DELIMITED_TERMS are not split.
    """
    raw = safe_text(term).strip()
    if not raw or raw.casefold() in PROTECTED_DELIMITED_TERMS:
        return [raw] if raw else []

    # Split comma and ampersand alternatives first.
    coarse = [part.strip() for part in re.split(r"\s*(?:,|&)\s*", raw) if part.strip()]
    expanded: list[str] = []
    for item in coarse:
        if "/" not in item or item.casefold() in PROTECTED_SLASH_TERMS:
            expanded.append(item)
            continue
        left, right = [part.strip() for part in item.split("/", 1)]
        if not left or not right:
            expanded.append(item)
            continue
        lt, rt = left.split(), right.split()
        if len(rt) > 1 and len(lt) == 1:
            suffix = " ".join(rt[1:])
            expanded.extend([f"{left} {suffix}".strip(), right])
        elif len(lt) > 1 and len(rt) == 1:
            prefix = " ".join(lt[:-1])
            expanded.extend([left, f"{prefix} {right}".strip()])
        elif len(lt) == 1 and len(rt) == 1:
            expanded.extend([left, right])
        else:
            expanded.append(item)

    result: list[str] = []
    seen: set[str] = set()
    for item in expanded or [raw]:
        key = normalize_term(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


ACRONYM_PROTECTED = {
    "uav", "uas", "rpa", "rpas", "rpv", "gcs", "rvt", "muav", "mav",
    "tuav", "suas", "male", "hale", "ucav", "vtol", "los", "blos",
    "rf", "eo", "ir", "isr", "istar", "sead", "c-ied", "s&a",
}


def conservative_lemma(term: str) -> str:
    """Conservative noun-phrase lemmatization shared by prediction and GT evaluation.

    It singularizes only common English plural endings and preserves known engineering
    acronyms. It intentionally avoids aggressive stemming of verbs and technical forms.
    """
    normalized = normalize_term(term)
    tokens = normalized.split()
    out: list[str] = []
    for token in tokens:
        bare = token.casefold()
        if bare in ACRONYM_PROTECTED or len(bare) <= 3:
            out.append(token)
        elif re.fullmatch(r"[a-z]+ies", bare):
            out.append(token[:-3] + "y")
        elif re.fullmatch(r"[a-z]+(?:ches|shes|xes|zes|sses)", bare):
            out.append(token[:-2])
        elif re.fullmatch(r"[a-z]+s", bare) and not bare.endswith(("ss", "us", "is")):
            out.append(token[:-1])
        else:
            out.append(token)
    return " ".join(out)


def term_match_forms(term: str) -> dict[str, set[str]]:
    """Return strict and lemma forms using the exact same algorithm for GT/predictions."""
    children = expand_term_forms(term)
    strict = {normalize_term(term)} | {normalize_term(x) for x in children}
    strict.discard("")
    lemma = {conservative_lemma(x) for x in strict}
    lemma.discard("")
    return {"strict": strict, "lemma": lemma}


def has_llm_evidence(row: pd.Series) -> bool:
    source_methods = safe_text(row.get("source_methods", "")).casefold()
    return "llm_evidence" in source_methods


def has_evidence_sentence(row: pd.Series) -> bool:
    evidence = safe_text(row.get("evidence_sentence", ""))
    if not evidence.strip():
        return False
    if is_hard_noise(evidence, normalize_term(evidence)):
        return False
    return True


def confidence_rank(value: str) -> float:
    return {"high": 1.0, "medium": 0.65, "low": 0.35, "": 0.0}.get(safe_text(value).casefold(), 0.0)


def assign_state(row: pd.Series) -> tuple[str, str, float]:
    """Precision-first state assignment.

    Critical rule:
    - Seed similarity never promotes a term in this script.
    - PROMOTED is only for a future LLM revalidation stage.
    """
    term = safe_text(row.get("term", ""))
    normalized = safe_text(row.get("normalized_term", ""))
    llm_confidence = safe_text(row.get("llm_confidence", "")).casefold()
    extractor_count = int(row.get("extractor_count", 0) or 0)
    seed_similarity = float(row.get("max_seed_similarity", 0.0) or 0.0)
    proposed_type = canonical_type_text(row.get("proposed_type", ""))

    llm_present = has_llm_evidence(row)
    valid_type = has_valid_type(proposed_type)
    evidence_present = has_evidence_sentence(row)

    # 1. Hard reject artifacts/fragments immediately.
    if is_hard_noise(term, normalized):
        return REJECTED_STATE, REJECTED_STATE, 0.0

    # 2. Do not accept/promote anything typed as other/blank in this pass.
    if not valid_type:
        if extractor_count >= 2 and evidence_present:
            initial_state = CANDIDATE_STATE
        elif extractor_count >= 1 and evidence_present:
            initial_state = BACKLOG_STATE
        else:
            initial_state = REJECTED_STATE
        confidence_score = 0.25 * min(extractor_count, 3) / 3.0 + 0.15 * seed_similarity
        return initial_state, initial_state, confidence_score

    # 3. LLM-grounded terms with one or more valid labels are retained.
    if llm_present and evidence_present:
        if llm_confidence in HIGH_CONFIDENCE:
            initial_state = ACCEPTED_STATE if extractor_count >= 1 else CANDIDATE_STATE
        else:
            # Low-confidence but valid multi/single-label evidence is useful backlog, not rejection.
            initial_state = BACKLOG_STATE

    # 4. Extractor agreement without LLM typing is candidate only.
    elif extractor_count >= 2 and evidence_present:
        initial_state = CANDIDATE_STATE

    # 5. Single extractor term remains backlog.
    elif extractor_count == 1 and evidence_present:
        initial_state = BACKLOG_STATE

    else:
        initial_state = REJECTED_STATE

    # No auto-promotion from similarity in this script.
    final_state = initial_state

    confidence_score = (
        0.45 * confidence_rank(llm_confidence)
        + 0.30 * min(extractor_count, 3) / 3.0
        + 0.15 * seed_similarity
        + 0.10 * (1.0 if valid_type else 0.0)
    )
    return initial_state, final_state, confidence_score


def build_term_records(
    chunk: pd.Series,
    evidence_payload: dict[str, Any],
    extractor_terms: pd.DataFrame,
) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}

    def _candidate_item(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return item
        if isinstance(item, str):
            return {"term": item}
        return {"term": safe_text(item)}

    # LLM terms are the only terms that can carry LLM confidence/type.
    # Multi-label outputs are preserved. Genuine slash alternatives become child terms.
    for item in evidence_payload.get("candidate_terms", []) or []:
        item = _candidate_item(item)
        source_term = safe_text(item.get("term", ""))
        proposed_type_raw = item.get("proposed_types", item.get("proposed_type", "other"))
        proposed_type = canonical_type_text(proposed_type_raw)
        proposed_types = parse_proposed_types(proposed_type_raw)
        evidence_sentence = safe_text(item.get("evidence_sentence", "")) or safe_text(chunk.text)

        for term in expand_term_forms(source_term):
            normalized = normalize_term(term)
            if not normalized or is_hard_noise(term, normalized):
                continue

            existing = records.get(normalized)
            if existing is None:
                records[normalized] = {
                    "chunk_id": safe_text(chunk.chunk_id),
                    "term": term,
                    "source_compound_term": source_term if term != source_term else "",
                    "normalized_term": normalized,
                    "proposed_type": proposed_type,
                    "proposed_types_json": json.dumps(proposed_types, ensure_ascii=False),
                    "evidence_sentence": evidence_sentence,
                    "llm_confidence": safe_text(item.get("confidence", "low")).casefold(),
                    "reason": safe_text(item.get("reason", "")),
                    "extractor_name": "",
                    "extractor_count": 0,
                    "extractor_score": 0.0,
                    "source_sentence": evidence_sentence,
                    "context_text": safe_text(chunk.text),
                    "source_methods": "llm_evidence",
                }
            else:
                combined = parse_proposed_types(existing.get("proposed_type", ""))
                for label in proposed_types:
                    if label not in combined:
                        combined.append(label)
                existing["proposed_type"] = " | ".join(combined) if combined else "other"
                existing["proposed_types_json"] = json.dumps(combined, ensure_ascii=False)
                existing["source_methods"] = "llm_evidence"

    # Extractor terms are evidence reservoir only.
    for row in extractor_terms.itertuples(index=False):
        term = safe_text(getattr(row, "raw_term", ""))
        normalized = normalize_term(term)
        if not normalized or is_hard_noise(term, normalized):
            continue
        entry = records.get(normalized)
        extractor_name = safe_text(getattr(row, "extractor", ""))
        extractor_score = float(getattr(row, "score", 0.0) or 0.0)
        source_sentence = safe_text(getattr(row, "source_sentence", "")) or safe_text(chunk.text)

        if entry is None:
            records[normalized] = {
                "chunk_id": safe_text(chunk.chunk_id),
                "term": term,
                "source_compound_term": "",
                "normalized_term": normalized,
                "proposed_type": "other",
                "proposed_types_json": "[]",
                "evidence_sentence": source_sentence,
                "llm_confidence": "",
                "reason": "extractor_only",
                "extractor_name": extractor_name,
                "extractor_count": 1 if extractor_name else 0,
                "extractor_score": extractor_score,
                "source_sentence": source_sentence,
                "context_text": safe_text(chunk.text),
                "source_methods": extractor_name,
            }
            continue

        entry["term"] = entry["term"] or term
        old_extractors = set(filter(None, safe_text(entry.get("extractor_name", "")).split(" | ")))
        if extractor_name:
            old_extractors.add(extractor_name)
        entry["extractor_name"] = " | ".join(sorted(old_extractors))
        entry["extractor_count"] = len(old_extractors)
        entry["extractor_score"] = max(float(entry.get("extractor_score", 0.0) or 0.0), extractor_score)
        if not entry.get("source_sentence"):
            entry["source_sentence"] = source_sentence
        if not entry.get("evidence_sentence"):
            entry["evidence_sentence"] = source_sentence
        methods = set(filter(None, safe_text(entry.get("source_methods", "")).split(" | ")))
        if extractor_name:
            methods.add(extractor_name)
        entry["source_methods"] = " | ".join(sorted(methods))

    frame = pd.DataFrame(records.values())
    if frame.empty:
        return frame
    frame["extractor_count"] = pd.to_numeric(frame["extractor_count"], errors="coerce").fillna(0).astype(int)
    frame["extractor_score"] = pd.to_numeric(frame["extractor_score"], errors="coerce").fillna(0.0)
    return frame.reset_index(drop=True)


def extractor_plan(names: list[str], top_k: int, keybert_model: str) -> list[tuple[str, dict[str, object]]]:
    plan: list[tuple[str, dict[str, object]]] = []
    for name in names:
        if name == "yake":
            plan.append((name, {"top_k": top_k}))
        elif name == "acronyms":
            plan.append((name, {"top_k": None}))
        elif name == "tfidf":
            plan.append((name, {"top_k": top_k}))
        elif name == "noun_phrases":
            plan.append((name, {"top_k": top_k}))
        elif name == "textrank":
            plan.append((name, {"top_k": top_k}))
        elif name == "keybert_mpnet":
            plan.append((name, {"top_k": top_k, "model": keybert_model}))
        else:
            raise ValueError(f"Unknown extractor: {name}")
    return plan


def run_extractor(name: str, chunks: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
    if name == "yake":
        return yake_extract(chunks, **params)
    if name == "acronyms":
        return acronym_extract(chunks, **params)
    if name == "tfidf":
        return tfidf(chunks, **params)
    if name == "noun_phrases":
        return noun_phrases(chunks, **params)
    if name == "textrank":
        return textrank(chunks, **params)
    if name == "keybert_mpnet":
        return keybert_extract(chunks, **params)
    raise ValueError(f"Unknown extractor: {name}")


def safe_score_seed_similarity(terms: pd.DataFrame, embedding_model: Any, args: argparse.Namespace) -> pd.DataFrame:
    """Deprecated baseline compatibility helper.

    Domain seeds are intentionally NOT used. Accepted terms do not exist until the
    baseline inventory is complete, so accepted-seed scoring is performed only by
    score_local_chunk_similarity() during the controlled recovery stage.
    """
    scored = terms.copy()
    scored["model_key"] = ""
    scored["model_name"] = ""
    scored["representation"] = "baseline_not_seed_scored"
    scored["embedding_input"] = scored["term"].fillna("").astype(str)
    scored["max_seed_similarity"] = 0.0
    scored["mean_seed_similarity"] = 0.0
    scored["median_seed_similarity"] = 0.0
    scored["closest_seed"] = ""
    return scored



def _prediction_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["term", "normalized_prediction", "proposed_type", "final_state"])
    inventory = frame.copy()
    if "term" not in inventory.columns:
        inventory["term"] = ""
    inventory["term"] = inventory["term"].fillna("").astype(str)
    inventory["normalized_prediction"] = inventory["term"].map(normalize_term)
    inventory = inventory[inventory["normalized_prediction"].ne("")].copy()
    inventory = inventory.drop_duplicates("normalized_prediction", keep="first").reset_index(drop=True)
    return inventory


def _type_overlap(predicted: Any, expected: Any) -> bool:
    predicted_labels = set(parse_proposed_types(predicted))
    expected_labels = set(parse_proposed_types(expected))
    return bool(predicted_labels.intersection(expected_labels))


def evaluate_current_run_predictions(
    predictions: pd.DataFrame,
    current_gt: pd.DataFrame,
    evaluation_name: str,
    output_dir: Path,
    gt_file: Path,
    gt_sheet: str,
    gt_scope: str,
) -> dict[str, Any]:
    """Evaluate normalized exact term recovery against CURRENT_RUN_TERM_GT/include_now."""
    inventory = _prediction_inventory(predictions)
    gt_terms = set(current_gt["normalized_gt_term"])
    prediction_terms = set(inventory["normalized_prediction"])

    matched_terms = prediction_terms.intersection(gt_terms)
    unmatched_terms = prediction_terms.difference(gt_terms)
    missing_terms = gt_terms.difference(prediction_terms)

    retained = len(prediction_terms)
    matched = len(matched_terms)
    eligible = len(gt_terms)
    precision = matched / retained if retained else 0.0
    recall = matched / eligible if eligible else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    mappings = inventory.merge(
        current_gt,
        left_on="normalized_prediction",
        right_on="normalized_gt_term",
        how="inner",
        suffixes=("_pred", "_gt"),
    )
    if not mappings.empty:
        mappings["type_overlap"] = mappings.apply(
            lambda row: _type_overlap(
                row.get("proposed_type", ""),
                row.get("current_type_options", ""),
            ),
            axis=1,
        )
        typed_rows = mappings[
            mappings.get("proposed_type", pd.Series(index=mappings.index, dtype=object))
            .fillna("").astype(str).ne("")
        ]
        type_overlap_accuracy = (
            float(typed_rows["type_overlap"].mean()) if not typed_rows.empty else None
        )
    else:
        mappings["type_overlap"] = pd.Series(dtype=bool)
        type_overlap_accuracy = None

    unmatched = inventory[inventory["normalized_prediction"].isin(unmatched_terms)].copy()
    missing = current_gt[current_gt["normalized_gt_term"].isin(missing_terms)].copy()

    safe_name = re.sub(r"[^a-z0-9]+", "_", evaluation_name.casefold()).strip("_")
    mappings.to_csv(output_dir / f"{safe_name}_matched_terms.csv", index=False)
    unmatched.to_csv(output_dir / f"{safe_name}_unmatched_predictions.csv", index=False)
    missing.to_csv(output_dir / f"{safe_name}_missing_gt_terms.csv", index=False)

    metrics = {
        "evaluation_name": evaluation_name,
        "gt_file": str(gt_file),
        "gt_sheet": gt_sheet,
        "gt_scope": gt_scope,
        "retained_unique_terms": retained,
        "matched_unique_terms": matched,
        "unmatched_unique_terms": len(unmatched_terms),
        "eligible_current_run_gt_terms": eligible,
        "covered_current_run_gt_terms": matched,
        "missing_current_run_gt_terms": len(missing_terms),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "type_overlap_accuracy_on_matched": type_overlap_accuracy,
    }
    pd.DataFrame([metrics]).to_csv(output_dir / f"{safe_name}_metrics.csv", index=False)
    return metrics


def detect_corpus_aliases(chunks: pd.DataFrame) -> pd.DataFrame:
    """Detect explicit long-form (ACRONYM) pairs from corpus text, independent of GT."""
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"(?P<long>[A-Za-z][A-Za-z0-9\-/ ]{2,80}?)\s*\((?P<acro>[A-Z][A-Z0-9&\-]{1,12}s?)\)")
    for row in chunks.itertuples(index=False):
        text = safe_text(getattr(row, "text", ""))
        for match in pattern.finditer(text):
            long_form = safe_text(match.group("long")).strip(" ,;:")
            acronym = safe_text(match.group("acro"))
            # Keep the nearest noun phrase, not an entire leading sentence.
            long_form = re.split(r"[.;:]", long_form)[-1].strip()
            if 1 <= len(long_form.split()) <= 10:
                rows.append({
                    "chunk_id": safe_text(getattr(row, "chunk_id", "")),
                    "canonical_observed": long_form,
                    "alias_observed": acronym,
                    "normalized_canonical": normalize_term(long_form),
                    "normalized_alias": normalize_term(acronym),
                    "canonical_lemma": conservative_lemma(long_form),
                    "alias_lemma": conservative_lemma(acronym),
                    "evidence_text": text,
                    "alias_source": "corpus_parenthetical_acronym",
                })
    if not rows:
        return pd.DataFrame(columns=[
            "chunk_id", "canonical_observed", "alias_observed", "normalized_canonical",
            "normalized_alias", "canonical_lemma", "alias_lemma", "evidence_text", "alias_source",
        ])
    return pd.DataFrame(rows).drop_duplicates(
        ["normalized_canonical", "normalized_alias", "chunk_id"]
    ).reset_index(drop=True)


def load_gt_aliases(gt_file: Path, sheet_name: str, broad: bool) -> pd.DataFrame:
    if not gt_file.is_file():
        return pd.DataFrame(columns=["normalized_canonical", "normalized_alias"])
    frame = pd.read_excel(gt_file, sheet_name=sheet_name, engine="openpyxl")
    if broad:
        canonical_col, alias_col = "canonical", "alias"
    else:
        canonical_col, alias_col = "canonical_observed", "alias_observed"
    if canonical_col not in frame.columns or alias_col not in frame.columns:
        return pd.DataFrame(columns=["normalized_canonical", "normalized_alias"])
    frame = frame.copy()
    frame["normalized_canonical"] = frame[canonical_col].fillna("").astype(str).map(normalize_term)
    frame["normalized_alias"] = frame[alias_col].fillna("").astype(str).map(normalize_term)
    frame["canonical_lemma"] = frame[canonical_col].fillna("").astype(str).map(conservative_lemma)
    frame["alias_lemma"] = frame[alias_col].fillna("").astype(str).map(conservative_lemma)
    frame["alias_source"] = f"gt:{sheet_name}"
    return frame[(frame["normalized_canonical"] != "") & (frame["normalized_alias"] != "")]


def build_alias_family_map(alias_frames: list[pd.DataFrame]) -> dict[str, str]:
    """Union-find alias families; used only for canonicalization/evaluation, never promotion."""
    parent: dict[str, str] = {}
    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for frame in alias_frames:
        for row in frame.itertuples(index=False):
            a = safe_text(getattr(row, "normalized_canonical", ""))
            b = safe_text(getattr(row, "normalized_alias", ""))
            if a and b:
                union(a, b)
    return {key: find(key) for key in list(parent)}


def load_broad_gt(path: Path, sheet: str, scope: str) -> pd.DataFrame:
    """Load the broad S1 term GT.

    Supported schemas:
    1. Original workbook S1_GT_TERMS. Rows grounded as text_observed or
       text_observed_page are eligible; needs_manual_review rows are excluded.
    2. Earlier clarified CURRENT_RUN_TERM_GT workbook, retained for compatibility.
    """
    gt = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")

    if {"term_canonical", "grounded_status_v2"}.issubset(gt.columns):
        eligible_status = {"text_observed", "text_observed_page"}
        gt = gt[
            gt["grounded_status_v2"].fillna("").astype(str).str.strip().isin(eligible_status)
        ].copy()
        type_column = "expected_type" if "expected_type" in gt.columns else "gt_category"
        gt = gt.rename(columns={"term_canonical": "gt_term", type_column: "gt_type"})
        return gt

    required = {"current_run_scope", "term_canonical", "current_type_options"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(
            "Broad GT must be original S1_GT_TERMS or clarified CURRENT_RUN_TERM_GT; "
            f"missing columns: {sorted(missing)}"
        )
    gt = gt[gt["current_run_scope"].fillna("").astype(str).str.strip().eq(scope)].copy()
    return gt.rename(columns={"term_canonical": "gt_term", "current_type_options": "gt_type"})


def load_structural_gt(path: Path, sheet: str) -> pd.DataFrame:
    gt = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    required = {"term_observed", "ontology_type"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"Structural GT missing columns: {sorted(missing)}")
    gt = gt.rename(columns={"term_observed": "gt_term", "ontology_type": "gt_type"})
    return gt


def structural_filter(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = frame["proposed_type"].fillna("").map(
        lambda value: bool(set(parse_proposed_types(value)) & STRUCTURAL_ONTOLOGY_TYPES)
    )
    return frame[mask].copy().reset_index(drop=True)


def evaluate_levels(
    predictions: pd.DataFrame,
    gt: pd.DataFrame,
    alias_map: dict[str, str],
    evaluation_name: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Evaluate strict, safe_extended, and context_extended levels."""
    inventory = _prediction_inventory(predictions)
    pred_records: list[dict[str, Any]] = []
    for row in inventory.itertuples(index=False):
        forms = term_match_forms(getattr(row, "term", ""))
        pred_records.append({
            **row._asdict(),
            "strict_forms": forms["strict"],
            "lemma_forms": forms["lemma"],
        })
    gt_records: list[dict[str, Any]] = []
    for row in gt.itertuples(index=False):
        forms = term_match_forms(getattr(row, "gt_term", ""))
        gt_records.append({
            **row._asdict(),
            "strict_forms": forms["strict"],
            "lemma_forms": forms["lemma"],
        })

    levels: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for level in ("strict", "safe_extended", "context_extended"):
        matched_pred: set[int] = set()
        covered_gt: set[int] = set()
        for pi, pred in enumerate(pred_records):
            pforms = pred["strict_forms"] if level == "strict" else pred["lemma_forms"]
            if level == "context_extended":
                pforms = pforms | {alias_map.get(x, x) for x in pforms}
            for gi, gold in enumerate(gt_records):
                gforms = gold["strict_forms"] if level == "strict" else gold["lemma_forms"]
                if level == "context_extended":
                    gforms = gforms | {alias_map.get(x, x) for x in gforms}
                if pforms & gforms:
                    matched_pred.add(pi); covered_gt.add(gi)
                    mapping_rows.append({
                        "evaluation_name": evaluation_name,
                        "level": level,
                        "predicted_term": pred.get("term", ""),
                        "gt_term": gold.get("gt_term", ""),
                        "predicted_type": pred.get("proposed_type", ""),
                        "gt_type": gold.get("gt_type", ""),
                        "type_overlap": _type_overlap(pred.get("proposed_type", ""), gold.get("gt_type", "")),
                    })
        retained, matched, eligible, covered = len(pred_records), len(matched_pred), len(gt_records), len(covered_gt)
        precision = matched / retained if retained else 0.0
        recall = covered / eligible if eligible else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        levels.append({
            "evaluation_name": evaluation_name,
            "level": level,
            "retained_unique_terms": retained,
            "matched_unique_predictions": matched,
            "unmatched_unique_predictions": retained - matched,
            "eligible_gt_terms": eligible,
            "covered_gt_terms": covered,
            "missing_gt_terms": eligible - covered,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    safe = re.sub(r"[^a-z0-9]+", "_", evaluation_name.casefold()).strip("_")
    pd.DataFrame(levels).to_csv(output_dir / f"{safe}_metrics.csv", index=False)
    pd.DataFrame(mapping_rows).drop_duplicates().to_csv(output_dir / f"{safe}_matches.csv", index=False)
    return levels



def write_consolidated_metrics_workbook(
    evaluations: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write one comparison-ready Excel workbook with metrics, run counts, and charts."""
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, Reference
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    output_path = output_dir / "consolidated_metrics_dashboard.xlsx"
    workbook = Workbook()
    metrics_sheet = workbook.active
    metrics_sheet.title = "Metrics"

    metric_columns = [
        "evaluation_name", "level", "retained_unique_terms",
        "matched_unique_predictions", "unmatched_unique_predictions",
        "eligible_gt_terms", "covered_gt_terms", "missing_gt_terms",
        "precision", "recall", "f1",
    ]
    metrics_sheet.append(metric_columns)
    for record in evaluations:
        metrics_sheet.append([record.get(column, "") for column in metric_columns])

    # A compact comparison label is useful for charts and pivoting.
    metrics_sheet.insert_cols(1)
    metrics_sheet["A1"] = "comparison"
    for row_number in range(2, metrics_sheet.max_row + 1):
        metrics_sheet.cell(row_number, 1).value = (
            f"{metrics_sheet.cell(row_number, 2).value} | "
            f"{metrics_sheet.cell(row_number, 3).value}"
        )

    run_sheet = workbook.create_sheet("Run Summary")
    run_sheet.append(["Metric", "Value"])
    run_fields = [
        ("pipeline_version", summary.get("version", "")),
        ("input_file", summary.get("input_file", "")),
        ("broad_gt_file", summary.get("broad_gt_file", "")),
        ("structural_gt_file", summary.get("structural_gt_file", "")),
        ("chunks", summary.get("chunks", 0)),
        ("gate_yes_chunks", summary.get("gate_yes_chunks", 0)),
        ("accepted_terms_rows", summary.get("accepted_terms", 0)),
        ("promoted_terms_rows", summary.get("promoted_terms", 0)),
        ("candidate_terms_rows", summary.get("candidate_terms", 0)),
        ("backlog_terms_rows", summary.get("backlog_terms", 0)),
        ("rejected_terms_rows", summary.get("rejected_terms", 0)),
        ("detected_corpus_aliases", summary.get("detected_corpus_aliases", 0)),
        ("extractors", " | ".join(summary.get("extractors", []))),
        ("seed_similarity_role", summary.get("seed_similarity_note", "")),
        ("recovery_seed_source", summary.get("recovery_seed_source", "")),
        ("accepted_seed_terms", summary.get("accepted_seed_terms", 0)),
        ("local_scored_rows", summary.get("local_scored_rows", 0)),
        ("local_shortlisted_rows", summary.get("local_shortlisted_rows", 0)),
        ("local_recovered_accepted_unique_terms", summary.get("local_recovered_accepted_unique_terms", 0)),
        ("local_review_unique_terms", summary.get("local_review_unique_terms", 0)),
        ("local_revalidation_rejected_unique_terms", summary.get("local_revalidation_rejected_unique_terms", 0)),
        ("candidate_term_threshold", summary.get("recovery_thresholds", {}).get("candidate_term", "")),
        ("candidate_sentence_threshold", summary.get("recovery_thresholds", {}).get("candidate_sentence", "")),
        ("backlog_term_threshold", summary.get("recovery_thresholds", {}).get("backlog_term", "")),
        ("backlog_sentence_threshold", summary.get("recovery_thresholds", {}).get("backlog_sentence", "")),
    ]
    for row in run_fields:
        run_sheet.append(list(row))

    chart_sheet = workbook.create_sheet("Charts")
    chart_sheet.append(["Comparison", "Precision", "Recall", "F1"])
    for record in evaluations:
        chart_sheet.append([
            f"{record.get('evaluation_name', '')} | {record.get('level', '')}",
            record.get("precision", 0.0),
            record.get("recall", 0.0),
            record.get("f1", 0.0),
        ])

    chart = BarChart()
    chart.type = "col"
    chart.style = 10
    chart.title = "Controlled Dual-GT Metric Comparison"
    chart.y_axis.title = "Score"
    chart.x_axis.title = "Evaluation set and matching level"
    chart.height = 12
    chart.width = 27
    data = Reference(chart_sheet, min_col=2, max_col=4, min_row=1, max_row=chart_sheet.max_row)
    categories = Reference(chart_sheet, min_col=1, min_row=2, max_row=chart_sheet.max_row)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)
    chart.legend.position = "r"
    chart_sheet.add_chart(chart, "F2")

    # Consistent formatting.
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for sheet in (metrics_sheet, run_sheet, chart_sheet):
        sheet.sheet_view.showGridLines = False
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for column in range(1, sheet.max_column + 1):
            values = [len(str(sheet.cell(row, column).value or "")) for row in range(1, min(sheet.max_row, 50) + 1)]
            sheet.column_dimensions[get_column_letter(column)].width = min(max(max(values, default=10) + 2, 12), 45)

    # Percentage formats in Metrics and Charts.
    # Metrics columns moved by one after inserting comparison: precision=10, recall=11, f1=12.
    for row_number in range(2, metrics_sheet.max_row + 1):
        for column in (10, 11, 12):
            metrics_sheet.cell(row_number, column).number_format = "0.00%"
    for row_number in range(2, chart_sheet.max_row + 1):
        for column in (2, 3, 4):
            chart_sheet.cell(row_number, column).number_format = "0.00%"

    workbook.save(output_path)
    return output_path

def build_empty_final_terms() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "chunk_id", "term", "normalized_term", "proposed_type", "evidence_sentence",
        "llm_confidence", "source_methods", "extractor_count", "initial_state", "final_state",
        "confidence_score", "max_seed_similarity", "closest_seed",
    ])



RECOVERY_DECISIONS = {"ACCEPT", "REVIEW", "REJECT"}
RECOVERED_ACCEPTED_STATE = "recovered_accepted"
RECOVERY_CANDIDATE_STATE = "recovery_candidate"
RECOVERY_REJECTED_STATE = "recovery_rejected"


def split_local_sentences(text: str) -> list[str]:
    """Conservative sentence splitter used only inside the current chunk."""
    clean = safe_text(text).strip()
    if not clean:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", clean) if part.strip()]


def find_exact_term_sentence(term: str, chunk_text: str) -> str:
    """Return the first current-chunk sentence containing the normalized term; never fall back to the whole chunk."""
    normalized_term = normalize_term(term)
    if not normalized_term:
        return ""
    for sentence in split_local_sentences(chunk_text):
        if normalized_term in normalize_term(sentence):
            return sentence
    return ""


def build_local_revalidation_prompt(row: pd.Series) -> str:
    """Low-cost strict gate: the LLM receives only current-chunk evidence, never prior text/sentences."""
    return f"""
You are strictly revalidating one ontology term using evidence from ONE current chunk only.
Similarity retrieves a term for review; it is never proof of acceptance.
The selected accepted seed may have come from the current or immediately previous chunk,
but its source sentence and source chunk are NOT evidence and are not provided.
Do not use previous chunks, outside knowledge, global terms, or ground truth.

Chunk ID: {safe_text(row.get('chunk_id', ''))}
Candidate term: {safe_text(row.get('term', ''))}
Original state: {safe_text(row.get('recovery_source_state', ''))}
Candidate occurrence sentence: {safe_text(row.get('local_candidate_sentence', ''))}
Selected accepted seed term: {safe_text(row.get('selected_seed_term', ''))}
Seed scope: {safe_text(row.get('selected_seed_scope', ''))}
Term similarity: {float(row.get('selected_term_similarity', 0.0) or 0.0):.4f}
Sentence similarity: {float(row.get('selected_sentence_similarity', 0.0) or 0.0):.4f}
Extractor count: {int(row.get('extractor_count', 0) or 0)}
Source methods: {safe_text(row.get('source_methods', ''))}
Current chunk text:
{safe_text(row.get('context_text', ''))}

Allowed ontology categories:
classification, mission_usecase, operating_condition, environment, scenario_operation,
system, subsystem, component, part, flow, port, interface.

First classify term_nature as exactly one of:
ENTITY_OR_CLASS, ATTRIBUTE, ACTION, DESCRIPTIVE_PHRASE, DOCUMENT_FRAGMENT, OTHER.

Return ACCEPT only when all are true:
1. The exact candidate term occurs in the current chunk.
2. term_nature is ENTITY_OR_CLASS.
3. The term independently names a reusable ontology concept, not merely a related phrase.
4. The candidate occurrence sentence directly supports at least one allowed category.
5. Confidence is high.
6. Acceptance is not based only on similarity to the selected seed.
Return REVIEW, not REJECT, when the candidate appears meaningful and similarity is strong,
but current-chunk evidence is incomplete for confident acceptance or typing.
Return REJECT only for an attribute, action, descriptive phrase, document fragment,
generic phrase, unsupported term, or topic-only association.
Copy candidate_evidence_sentence exactly from the supplied candidate occurrence sentence.
Do not invent or paraphrase evidence.

Return JSON only:
{{
  "decision": "ACCEPT" or "REVIEW" or "REJECT",
  "term_nature": "ENTITY_OR_CLASS" or "ATTRIBUTE" or "ACTION" or "DESCRIPTIVE_PHRASE" or "DOCUMENT_FRAGMENT" or "OTHER",
  "proposed_types": ["..."],
  "confidence": "high" or "medium" or "low",
  "reason": "one short evidence-based reason",
  "candidate_evidence_sentence": "exact supplied candidate occurrence sentence, or empty string"
}}
""".strip()


def score_local_chunk_similarity(final_terms: pd.DataFrame, embedding_model: Any, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, int]]:
    """Score current candidate/backlog against current and previous-one-chunk original accepted seeds."""
    required = {"chunk_id", "chunk_index", "term", "final_state", "context_text"}
    missing = required - set(final_terms.columns)
    if missing:
        raise ValueError(
            "Existing final_ontology_candidates.csv cannot support local/previous recovery; missing columns: "
            f"{sorted(missing)}. Rerun once with --force-rerun to regenerate chunk-linked artifacts."
        )

    frame = final_terms.copy()
    frame["chunk_index"] = pd.to_numeric(frame["chunk_index"], errors="coerce")
    frame = frame.dropna(subset=["chunk_index"]).copy()
    frame["chunk_index"] = frame["chunk_index"].astype(int)
    rows: list[dict[str, Any]] = []
    stats = {
        "chunks_total_in_inventory": int(frame["chunk_id"].nunique()),
        "chunks_with_local_accepted_seed": 0,
        "chunks_without_local_accepted_seed": 0,
        "chunks_with_previous1_accepted_seed": 0,
        "chunks_without_previous1_accepted_seed": 0,
        "chunks_rescued_by_previous1_seed": 0,
        "chunks_with_recovery_pool": 0,
        "candidate_rows_scored": 0,
        "backlog_rows_scored": 0,
        "rows_missing_candidate_sentence": 0,
        "local_only_shortlisted": 0,
        "previous1_only_shortlisted": 0,
        "both_routes_shortlisted": 0,
    }

    accepted_all = frame[frame["final_state"].eq(ACCEPTED_STATE)].copy()
    for chunk_index in sorted(frame["chunk_index"].unique()):
        chunk_rows = frame[frame["chunk_index"].eq(chunk_index)].copy()
        local_pool = chunk_rows[chunk_rows["final_state"].isin([CANDIDATE_STATE, BACKLOG_STATE])].copy()
        if local_pool.empty:
            continue
        stats["chunks_with_recovery_pool"] += 1

        local_seeds = accepted_all[accepted_all["chunk_index"].eq(chunk_index)].copy()
        previous_seeds = accepted_all[accepted_all["chunk_index"].eq(chunk_index - 1)].copy()
        if local_seeds.empty:
            stats["chunks_without_local_accepted_seed"] += 1
        else:
            stats["chunks_with_local_accepted_seed"] += 1
        if previous_seeds.empty:
            stats["chunks_without_previous1_accepted_seed"] += 1
        else:
            stats["chunks_with_previous1_accepted_seed"] += 1
        if local_seeds.empty and not previous_seeds.empty:
            stats["chunks_rescued_by_previous1_seed"] += 1
        if local_seeds.empty and previous_seeds.empty:
            continue

        local_pool["normalized_term"] = local_pool["term"].map(normalize_term)
        local_pool = local_pool[local_pool["normalized_term"].ne("")].drop_duplicates("normalized_term")
        local_pool["local_candidate_sentence"] = local_pool.apply(
            lambda row: find_exact_term_sentence(row.get("term", ""), row.get("context_text", "")), axis=1
        )
        pool_term_vec = embedding_model.encode(
            local_pool["term"].fillna("").astype(str).tolist(), normalize_embeddings=True, show_progress_bar=False
        )
        pool_sentence_vec = embedding_model.encode(
            local_pool["local_candidate_sentence"].fillna("").astype(str).tolist(), normalize_embeddings=True, show_progress_bar=False
        )

        def prepare_seeds(seed_frame: pd.DataFrame) -> tuple[pd.DataFrame, Any, Any]:
            if seed_frame.empty:
                return seed_frame, None, None
            seed_frame = seed_frame.copy()
            seed_frame["normalized_term"] = seed_frame["term"].map(normalize_term)
            seed_frame = seed_frame[seed_frame["normalized_term"].ne("")].drop_duplicates("normalized_term")
            seed_frame["seed_occurrence_sentence"] = seed_frame.apply(
                lambda row: find_exact_term_sentence(row.get("term", ""), row.get("context_text", "")), axis=1
            )
            term_vec = embedding_model.encode(
                seed_frame["term"].fillna("").astype(str).tolist(), normalize_embeddings=True, show_progress_bar=False
            )
            sentence_vec = embedding_model.encode(
                seed_frame["seed_occurrence_sentence"].fillna("").astype(str).tolist(), normalize_embeddings=True, show_progress_bar=False
            )
            return seed_frame.reset_index(drop=True), term_vec, sentence_vec

        local_seeds, local_term_vec, local_sentence_vec = prepare_seeds(local_seeds)
        previous_seeds, previous_term_vec, previous_sentence_vec = prepare_seeds(previous_seeds)

        def best_pair(pool_i: int, seeds: pd.DataFrame, seed_term_vec: Any, seed_sentence_vec: Any) -> dict[str, Any]:
            if seeds.empty:
                return {"available": False}
            term_scores = pool_term_vec[pool_i] @ seed_term_vec.T
            sentence_scores = pool_sentence_vec[pool_i] @ seed_sentence_vec.T
            combined = args.term_similarity_weight * term_scores + args.sentence_similarity_weight * sentence_scores
            valid_sentence = seeds["seed_occurrence_sentence"].fillna("").astype(str).str.len().gt(0).to_numpy()
            candidate_sentence = safe_text(local_pool.iloc[pool_i].get("local_candidate_sentence", ""))
            ranked = combined.copy()
            if not candidate_sentence:
                ranked[:] = -1.0
            else:
                ranked[~valid_sentence] = -1.0
            best = int(ranked.argmax()) if len(ranked) else -1
            if best < 0 or ranked[best] < 0:
                return {"available": False}
            seed = seeds.iloc[best]
            return {
                "available": True,
                "seed_term": safe_text(seed.get("term", "")),
                "seed_chunk_id": safe_text(seed.get("chunk_id", "")),
                "seed_chunk_index": int(seed.get("chunk_index", -1)),
                "seed_sentence": safe_text(seed.get("seed_occurrence_sentence", "")),
                "term_similarity": float(term_scores[best]),
                "sentence_similarity": float(sentence_scores[best]),
                "combined_similarity": float(combined[best]),
            }

        for pool_i, (_, pool_row) in enumerate(local_pool.iterrows()):
            candidate_sentence = safe_text(pool_row.get("local_candidate_sentence", ""))
            if not candidate_sentence:
                stats["rows_missing_candidate_sentence"] += 1
            source_state = safe_text(pool_row.get("final_state", "")).casefold()
            local = best_pair(pool_i, local_seeds, local_term_vec, local_sentence_vec)
            previous = best_pair(pool_i, previous_seeds, previous_term_vec, previous_sentence_vec)

            if source_state == CANDIDATE_STATE:
                local_term_th = float(args.candidate_term_threshold)
                local_sentence_th = float(args.candidate_sentence_threshold)
                prev_term_th = float(args.previous_candidate_term_threshold)
                prev_sentence_th = float(args.previous_candidate_sentence_threshold)
                stats["candidate_rows_scored"] += 1
            else:
                local_term_th = float(args.backlog_term_threshold)
                local_sentence_th = float(args.backlog_sentence_threshold)
                prev_term_th = float(args.previous_backlog_term_threshold)
                prev_sentence_th = float(args.previous_backlog_sentence_threshold)
                stats["backlog_rows_scored"] += 1

            local_pass = bool(local.get("available") and local["term_similarity"] >= local_term_th and local["sentence_similarity"] >= local_sentence_th)
            previous_pass = bool(previous.get("available") and previous["term_similarity"] >= prev_term_th and previous["sentence_similarity"] >= prev_sentence_th)
            if local_pass and previous_pass:
                route = "both"
                selected = local
                stats["both_routes_shortlisted"] += 1
            elif local_pass:
                route = "local_only"
                selected = local
                stats["local_only_shortlisted"] += 1
            elif previous_pass:
                route = "previous1_only"
                selected = previous
                stats["previous1_only_shortlisted"] += 1
            else:
                route = "none"
                selected = local if local.get("available") else previous

            record = pool_row.to_dict()
            record.update({
                "recovery_scope": "local_plus_previous1_embedding_only",
                "recovery_source_state": source_state,
                "local_candidate_sentence": candidate_sentence,
                "local_seed_term": local.get("seed_term", ""),
                "local_seed_chunk_id": local.get("seed_chunk_id", ""),
                "local_term_similarity": local.get("term_similarity", None),
                "local_sentence_similarity": local.get("sentence_similarity", None),
                "local_combined_similarity": local.get("combined_similarity", None),
                "local_route_passed": local_pass,
                "previous1_seed_term": previous.get("seed_term", ""),
                "previous1_seed_chunk_id": previous.get("seed_chunk_id", ""),
                "previous1_term_similarity": previous.get("term_similarity", None),
                "previous1_sentence_similarity": previous.get("sentence_similarity", None),
                "previous1_combined_similarity": previous.get("combined_similarity", None),
                "previous1_route_passed": previous_pass,
                "selected_route": route,
                "selected_seed_scope": "local" if route in {"local_only", "both"} else ("previous1_embedding_only" if route == "previous1_only" else "none"),
                "selected_seed_term": selected.get("seed_term", "") if selected else "",
                "selected_seed_chunk_id": selected.get("seed_chunk_id", "") if selected else "",
                "selected_term_similarity": selected.get("term_similarity", None) if selected else None,
                "selected_sentence_similarity": selected.get("sentence_similarity", None) if selected else None,
                "selected_combined_similarity": selected.get("combined_similarity", None) if selected else None,
                "similarity_shortlisted": route != "none",
            })
            rows.append(record)

    return pd.DataFrame(rows), stats


def prepare_revalidation_pass(scored: pd.DataFrame, pass_name: str) -> pd.DataFrame:
    """Create an explicit LLM input frame for local or previous1 embedding evidence."""
    if scored.empty:
        return scored.copy()
    if pass_name == "local":
        frame = scored[scored["local_route_passed"].eq(True)].copy()
        frame["selected_route"] = "local_pass"
        frame["selected_seed_scope"] = "local"
        frame["selected_seed_term"] = frame["local_seed_term"]
        frame["selected_seed_chunk_id"] = frame["local_seed_chunk_id"]
        frame["selected_term_similarity"] = frame["local_term_similarity"]
        frame["selected_sentence_similarity"] = frame["local_sentence_similarity"]
        frame["selected_combined_similarity"] = frame["local_combined_similarity"]
    elif pass_name == "previous1":
        frame = scored[scored["previous1_route_passed"].eq(True)].copy()
        frame["selected_route"] = "previous1_second_pass"
        frame["selected_seed_scope"] = "previous1_embedding_only"
        frame["selected_seed_term"] = frame["previous1_seed_term"]
        frame["selected_seed_chunk_id"] = frame["previous1_seed_chunk_id"]
        frame["selected_term_similarity"] = frame["previous1_term_similarity"]
        frame["selected_sentence_similarity"] = frame["previous1_sentence_similarity"]
        frame["selected_combined_similarity"] = frame["previous1_combined_similarity"]
    else:
        raise ValueError(f"Unknown revalidation pass: {pass_name}")
    frame["revalidation_pass"] = pass_name
    return frame.reset_index(drop=True)


def revalidate_local_shortlist(shortlist: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Strict same-chunk ACCEPT/REVIEW/REJECT gate."""
    rows: list[dict[str, Any]] = []
    unique = shortlist.sort_values("selected_combined_similarity", ascending=False).drop_duplicates(
        ["chunk_id", "normalized_term"]
    )
    total_calls = int(len(unique))
    pass_name = safe_text(unique.iloc[0].get("revalidation_pass", "unspecified")) if total_calls else "unspecified"
    print(f"LLM revalidation pass '{pass_name}': {total_calls} call(s) scheduled")
    for call_number, source in enumerate(unique.itertuples(index=False), start=1):
        row = pd.Series(source._asdict())
        print(
            f"  [{pass_name} {call_number}/{total_calls}] "
            f"chunk={safe_text(row.get('chunk_id', ''))} "
            f"term={safe_text(row.get('term', ''))} "
            f"seed={safe_text(row.get('selected_seed_term', ''))}"
        )
        prompt = build_local_revalidation_prompt(row)
        started = time.perf_counter()
        try:
            payload, raw = ollama_chat_json(
                args.ollama_url, args.model, LLM_EVIDENCE_SYSTEM_PROMPT, prompt, args.timeout
            )
            error = ""
        except Exception as exc:
            payload, raw = {}, ""
            error = f"{type(exc).__name__}: {exc}"

        decision = safe_text(payload.get("decision", "")).upper()
        term_nature = safe_text(payload.get("term_nature", "")).upper()
        proposed_types = parse_proposed_types(payload.get("proposed_types", []))
        confidence = safe_text(payload.get("confidence", "low")).casefold()
        returned_evidence = safe_text(payload.get("candidate_evidence_sentence", ""))
        candidate_sentence = safe_text(row.get("local_candidate_sentence", ""))
        chunk_text = safe_text(row.get("context_text", ""))
        evidence_exact = bool(returned_evidence and returned_evidence.casefold() == candidate_sentence.casefold())
        evidence_in_chunk = bool(returned_evidence and returned_evidence.casefold() in chunk_text.casefold())
        term_in_chunk = bool(normalize_term(row.get("term", "")) in normalize_term(chunk_text))

        extractor_count = int(row.get("extractor_count", 0) or 0)
        source_methods = safe_text(row.get("source_methods", "")).casefold()
        has_prior_llm_evidence = "llm_evidence" in source_methods
        source_state = safe_text(row.get("recovery_source_state", "")).casefold()
        if source_state == CANDIDATE_STATE:
            extraction_support_ok = extractor_count >= 2 or has_prior_llm_evidence
        elif source_state == BACKLOG_STATE:
            extraction_support_ok = extractor_count >= 2 and has_prior_llm_evidence
        else:
            extraction_support_ok = False

        recovered_accepted = bool(
            decision == "ACCEPT"
            and term_nature == "ENTITY_OR_CLASS"
            and confidence == "high"
            and proposed_types
            and evidence_exact
            and evidence_in_chunk
            and term_in_chunk
            and extraction_support_ok
        )
        if recovered_accepted:
            recovery_state = RECOVERED_ACCEPTED_STATE
        elif decision == "REJECT":
            recovery_state = RECOVERY_REJECTED_STATE
        else:
            recovery_state = RECOVERY_CANDIDATE_STATE

        record = row.to_dict()
        record.update({
            "revalidation_decision": decision or "ERROR",
            "revalidation_term_nature": term_nature or "OTHER",
            "revalidated_types": " | ".join(proposed_types) if proposed_types else "other",
            "revalidation_confidence": confidence,
            "revalidation_reason": safe_text(payload.get("reason", "")),
            "revalidation_evidence_sentence": returned_evidence,
            "evidence_exact_to_local_sentence": evidence_exact,
            "evidence_grounded_in_current_chunk": evidence_in_chunk,
            "term_grounded_in_current_chunk": term_in_chunk,
            "revalidation_extraction_support": extraction_support_ok,
            "recovered_accepted": recovered_accepted,
            "recovery_state": recovery_state,
            "revalidation_prompt": prompt,
            "revalidation_raw_response": raw,
            "revalidation_error": error,
            "revalidation_duration_seconds": time.perf_counter() - started,
        })
        rows.append(record)
    return pd.DataFrame(rows)


def update_consolidated_stage_metrics(
    recovery_evaluations: list[dict[str, Any]],
    recovery_result: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    """Append/replace recovery-stage rows in the existing summary and rebuild one dashboard."""
    summary_csv = args.output_dir / "dual_gt_evaluation_summary.csv"
    recovery_names = {row.get("evaluation_name", "") for row in recovery_evaluations}

    existing_rows: list[dict[str, Any]] = []
    if summary_csv.is_file():
        existing = pd.read_csv(summary_csv).fillna("")
        if "evaluation_name" in existing.columns:
            names = existing["evaluation_name"].astype(str)
            existing = existing[~(names.str.startswith("recovery_") | names.str.startswith("local_"))]
        existing_rows = existing.to_dict(orient="records")

    all_evaluations = existing_rows + recovery_evaluations
    pd.DataFrame(all_evaluations).to_csv(summary_csv, index=False)

    prior_summary_path = args.output_dir / "summary.json"
    if prior_summary_path.is_file():
        try:
            dashboard_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            dashboard_summary = {}
    else:
        dashboard_summary = {}

    dashboard_summary.update({
        "version": "precision_first_ontology_pipeline_v11_in_chunk_recovery",
        "input_file": str(args.input_file),
        "broad_gt_file": str(args.broad_gt_file),
        "structural_gt_file": str(args.structural_gt_file),
        "output_dir": str(args.output_dir),
        "recovery_seed_source": "same-chunk rows where final_state == accepted",
        "accepted_seed_terms": recovery_result.get("accepted_seed_terms", 0),
        "local_scored_rows": recovery_result.get("local_scored_rows", 0),
        "local_shortlisted_rows": recovery_result.get("local_shortlisted_rows", 0),
        "local_recovered_accepted_unique_terms": recovery_result.get("local_recovered_accepted_unique_terms", 0),
        "local_review_unique_terms": recovery_result.get("local_review_unique_terms", 0),
        "local_revalidation_rejected_unique_terms": recovery_result.get("local_revalidation_rejected_unique_terms", 0),
        "recovery_thresholds": recovery_result.get("thresholds", {}),
        "all_stage_evaluations": all_evaluations,
    })
    dashboard_path = write_consolidated_metrics_workbook(
        evaluations=all_evaluations,
        summary=dashboard_summary,
        output_dir=args.output_dir,
    )
    dashboard_summary["consolidated_metrics_workbook"] = str(dashboard_path)
    dashboard_summary["consolidated_metrics_csv"] = str(summary_csv)
    prior_summary_path.write_text(
        json.dumps(dashboard_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dashboard_path

def write_single_experiment_observation(
    result: dict[str, Any],
    scored: pd.DataFrame,
    revalidated: pd.DataFrame,
    evaluations: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write one human-readable file containing configuration, funnel, outcomes, terms, and metrics."""
    path = output_dir / "experiment_observations.md"
    lines = [
        "# Ontology Recovery Experiment Observations",
        "",
        "## Experiment",
        f"- Name: `{result.get('experiment', '')}`",
        "- Previous-one-chunk accepted term and occurrence sentence are used only in embedding similarity; a distinct second LLM pass tests rows passing that route.",
        "- Neither LLM pass receives previous chunk text or previous seed sentence; the second pass receives only the selected previous seed term and similarity scores.",
        "- The current chunk is the only acceptance evidence.",
        "- Original rejected terms are excluded.",
        "",
        "## Configuration",
        "```json",
        json.dumps({"thresholds": result.get("thresholds", {}), "weights": result.get("weights", {})}, indent=2),
        "```",
        "",
        "## Funnel",
    ]
    funnel_keys = [
        "chunks_total_in_inventory", "chunks_with_recovery_pool",
        "chunks_with_local_accepted_seed", "chunks_without_local_accepted_seed",
        "chunks_with_previous1_accepted_seed", "chunks_without_previous1_accepted_seed",
        "chunks_rescued_by_previous1_seed", "candidate_rows_scored", "backlog_rows_scored",
        "rows_missing_candidate_sentence", "local_only_shortlisted",
        "previous1_only_shortlisted", "both_routes_shortlisted", "local_shortlisted_rows",
        "previous1_second_pass_shortlisted_rows", "local_llm_calls_scheduled",
        "previous1_llm_calls_scheduled",
        "local_recovered_accepted_unique_terms", "local_review_unique_terms",
        "local_revalidation_rejected_unique_terms",
    ]
    for key in funnel_keys:
        lines.append(f"- {key}: {result.get(key, 0)}")

    lines += ["", "## Route outcomes"]
    for key in ["shortlisted_by_route", "recovered_accepted_by_route", "review_by_route", "rejected_by_route"]:
        lines.append(f"- {key}: `{json.dumps(result.get(key, {}), sort_keys=True)}`")

    if evaluations:
        lines += ["", "## Evaluation metrics", "", "| Evaluation | Level | Terms | Matched | Covered GT | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in evaluations:
            lines.append(
                f"| {row.get('evaluation_name','')} | {row.get('level','')} | {row.get('retained_unique_terms',0)} | "
                f"{row.get('matched_unique_predictions',0)} | {row.get('covered_gt_terms',0)} | "
                f"{float(row.get('precision',0)):.4f} | {float(row.get('recall',0)):.4f} | {float(row.get('f1',0)):.4f} |"
            )

    def add_terms(title: str, state: str) -> None:
        lines.extend(["", f"## {title}"])
        if revalidated.empty or "recovery_state" not in revalidated.columns:
            lines.append("- None")
            return
        subset = revalidated[revalidated["recovery_state"].eq(state)].copy()
        if subset.empty:
            lines.append("- None")
            return
        columns = ["term", "recovery_source_state", "selected_route", "selected_seed_term", "selected_term_similarity", "selected_sentence_similarity", "revalidation_decision", "revalidation_reason"]
        columns = [col for col in columns if col in subset.columns]
        for _, row in subset[columns].drop_duplicates("term").iterrows():
            lines.append(
                f"- **{safe_text(row.get('term',''))}** | source={safe_text(row.get('recovery_source_state',''))} "
                f"| route={safe_text(row.get('selected_route',''))} | seed={safe_text(row.get('selected_seed_term',''))} "
                f"| term_sim={row.get('selected_term_similarity','')} | sentence_sim={row.get('selected_sentence_similarity','')} "
                f"| decision={safe_text(row.get('revalidation_decision',''))} | {safe_text(row.get('revalidation_reason',''))}"
            )

    add_terms("Recovered accepted terms", RECOVERED_ACCEPTED_STATE)
    add_terms("Review terms", RECOVERY_CANDIDATE_STATE)
    add_terms("Revalidation rejected terms", RECOVERY_REJECTED_STATE)

    lines += ["", "## Next-decision checklist", "- Compare local-only versus previous1-only accepted yield.", "- Check whether previous1-only terms add strict structural GT coverage.", "- Check whether previous1-only REVIEW dominates; if so, only those rows may justify a later cross-chunk LLM experiment.", "- Do not expand to previous 3–5 chunks unless previous1 improves recall without unacceptable precision loss."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_seed_recovery(
    final_terms: pd.DataFrame,
    embedding_model: Any,
    args: argparse.Namespace,
    detected_aliases: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Controlled experiment: same-chunk dual similarity followed by strict local LLM revalidation."""
    if embedding_model is None:
        raise RuntimeError("Local recovery requires sentence-transformers and an embedding model.")

    scored, local_stats = score_local_chunk_similarity(final_terms, embedding_model, args)
    scored.to_csv(args.output_dir / "local_plus_previous1_similarity_all.csv", index=False)

    # Pass 1: preserve the proven same-chunk experiment.
    local_shortlist = prepare_revalidation_pass(scored, "local")
    local_shortlist.to_csv(args.output_dir / "local_revalidation_shortlist.csv", index=False)

    # Pass 2: explicitly revalidate every candidate passing the previous-one-chunk embedding route.
    # This includes rows that also passed locally, so the effect of selecting the previous seed is auditable.
    previous1_shortlist = prepare_revalidation_pass(scored, "previous1")
    previous1_shortlist.to_csv(args.output_dir / "previous1_second_revalidation_shortlist.csv", index=False)

    if args.skip_revalidation:
        local_revalidated = pd.DataFrame()
        previous1_revalidated = pd.DataFrame()
        print("LLM revalidation skipped by --skip-revalidation")
    else:
        if local_shortlist.empty:
            local_revalidated = pd.DataFrame()
            print("LLM local revalidation pass: 0 calls (no local rows passed thresholds)")
        else:
            local_revalidated = revalidate_local_shortlist(local_shortlist, args)
        if previous1_shortlist.empty:
            previous1_revalidated = pd.DataFrame()
            print("LLM previous1 second revalidation pass: 0 calls (no previous1 rows passed thresholds)")
        else:
            previous1_revalidated = revalidate_local_shortlist(previous1_shortlist, args)

    local_revalidated.to_csv(args.output_dir / "local_first_revalidation.csv", index=False)
    previous1_revalidated.to_csv(args.output_dir / "previous1_second_revalidation.csv", index=False)
    revalidated = pd.concat(
        [local_revalidated, previous1_revalidated], ignore_index=True, sort=False
    ) if (not local_revalidated.empty or not previous1_revalidated.empty) else pd.DataFrame()
    revalidated.to_csv(args.output_dir / "two_pass_revalidation_all.csv", index=False)

    # A term accepted in either strict pass is accepted once. If both accept, local provenance wins.
    if not revalidated.empty:
        revalidated["pass_priority"] = revalidated["revalidation_pass"].map({"local": 0, "previous1": 1}).fillna(2)
        revalidated = revalidated.sort_values(
            ["recovered_accepted", "pass_priority"], ascending=[False, True]
        )
        decision_inventory = revalidated.drop_duplicates(
            ["chunk_id", "normalized_term"], keep="first"
        ).reset_index(drop=True)
    else:
        decision_inventory = pd.DataFrame()

    if decision_inventory.empty:
        accepted_recovery = pd.DataFrame(columns=final_terms.columns)
        review_recovery = pd.DataFrame(columns=final_terms.columns)
        rejected_recovery = pd.DataFrame(columns=final_terms.columns)
    else:
        accepted_recovery = decision_inventory[decision_inventory["recovered_accepted"].eq(True)].copy()
        review_recovery = decision_inventory[decision_inventory["recovery_state"].eq(RECOVERY_CANDIDATE_STATE)].copy()
        rejected_recovery = decision_inventory[decision_inventory["recovery_state"].eq(RECOVERY_REJECTED_STATE)].copy()
        if not accepted_recovery.empty:
            accepted_recovery["previous_final_state"] = accepted_recovery["final_state"]
            accepted_recovery["final_state"] = RECOVERED_ACCEPTED_STATE
            accepted_recovery["proposed_type"] = accepted_recovery["revalidated_types"]
            accepted_recovery["llm_confidence"] = accepted_recovery["revalidation_confidence"]
            accepted_recovery["reason"] = accepted_recovery["revalidation_reason"]
            accepted_recovery["evidence_sentence"] = accepted_recovery["revalidation_evidence_sentence"]
            accepted_recovery["source_methods"] = accepted_recovery["source_methods"].fillna("").astype(str).map(
                lambda value: " | ".join(filter(None, [value, "same_chunk_dual_similarity", "strict_local_llm_revalidation"]))
            )

    accepted_recovery.to_csv(args.output_dir / "local_recovered_accepted_terms.csv", index=False)
    review_recovery.to_csv(args.output_dir / "local_recovery_candidates_for_review.csv", index=False)
    rejected_recovery.to_csv(args.output_dir / "local_recovery_rejected.csv", index=False)

    accepted = final_terms[final_terms["final_state"].eq(ACCEPTED_STATE)].copy()
    trusted_expanded = pd.concat([accepted, accepted_recovery], ignore_index=True, sort=False).drop_duplicates("normalized_term")
    review_expanded = pd.concat([accepted, accepted_recovery, review_recovery], ignore_index=True, sort=False).drop_duplicates("normalized_term")
    trusted_expanded.to_csv(args.output_dir / "accepted_plus_local_recovered_strict.csv", index=False)
    review_expanded.to_csv(args.output_dir / "accepted_plus_local_review_pool.csv", index=False)

    evaluations: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        **local_stats,
        "experiment": "two_pass_local_then_previous1_embedding_only_llm_current_chunk_only",
        "accepted_seed_terms": int(accepted["normalized_term"].nunique()),
        "local_scored_rows": int(len(scored)),
        "local_shortlisted_rows": int(len(local_shortlist)),
        "previous1_second_pass_shortlisted_rows": int(len(previous1_shortlist)),
        "local_llm_calls_scheduled": 0 if args.skip_revalidation else int(len(local_shortlist)),
        "previous1_llm_calls_scheduled": 0 if args.skip_revalidation else int(len(previous1_shortlist)),
        "shortlisted_by_route": {
            "local_pass": int(len(local_shortlist)),
            "previous1_second_pass": int(len(previous1_shortlist)),
        },
        "local_recovered_accepted_unique_terms": int(accepted_recovery["normalized_term"].nunique()) if not accepted_recovery.empty else 0,
        "local_review_unique_terms": int(review_recovery["normalized_term"].nunique()) if not review_recovery.empty else 0,
        "local_revalidation_rejected_unique_terms": int(rejected_recovery["normalized_term"].nunique()) if not rejected_recovery.empty else 0,
        "recovered_accepted_by_route": {str(k): int(v) for k, v in accepted_recovery["selected_route"].value_counts().to_dict().items()} if not accepted_recovery.empty else {},
        "review_by_route": {str(k): int(v) for k, v in review_recovery["selected_route"].value_counts().to_dict().items()} if not review_recovery.empty else {},
        "rejected_by_route": {str(k): int(v) for k, v in rejected_recovery["selected_route"].value_counts().to_dict().items()} if not rejected_recovery.empty else {},
        "original_rejected_terms_scored": 0,
        "previous_chunk_context_used_by_llm": False,
        "previous1_context_used_by_embeddings": True,
        "thresholds": {
            "candidate_term": args.candidate_term_threshold,
            "candidate_sentence": args.candidate_sentence_threshold,
            "backlog_term": args.backlog_term_threshold,
            "backlog_sentence": args.backlog_sentence_threshold,
            "previous_candidate_term": args.previous_candidate_term_threshold,
            "previous_candidate_sentence": args.previous_candidate_sentence_threshold,
            "previous_backlog_term": args.previous_backlog_term_threshold,
            "previous_backlog_sentence": args.previous_backlog_sentence_threshold,
        },
        "weights": {
            "term": args.term_similarity_weight,
            "sentence": args.sentence_similarity_weight,
        },
    }

    if not args.skip_gt_evaluation:
        broad_gt = load_broad_gt(args.broad_gt_file, args.broad_gt_sheet, args.broad_gt_scope)
        structural_gt = load_structural_gt(args.structural_gt_file, args.structural_gt_sheet)
        broad_aliases = load_gt_aliases(args.broad_gt_file, "S2_GT_ALIASES", broad=True)
        structural_aliases = load_gt_aliases(args.structural_gt_file, "GT_ALIASES", broad=False)
        if detected_aliases is None:
            alias_path = args.output_dir / "detected_corpus_aliases.csv"
            detected_aliases = pd.read_csv(alias_path) if alias_path.is_file() else pd.DataFrame()
        alias_map = build_alias_family_map([detected_aliases, broad_aliases, structural_aliases])
        evaluations += evaluate_levels(accepted, broad_gt, alias_map, "local_before_broad_accepted", args.output_dir)
        evaluations += evaluate_levels(trusted_expanded, broad_gt, alias_map, "local_after_broad_strict_accepted", args.output_dir)
        evaluations += evaluate_levels(review_expanded, broad_gt, alias_map, "local_broad_review_pool", args.output_dir)
        evaluations += evaluate_levels(structural_filter(accepted), structural_gt, alias_map, "local_before_structural_accepted", args.output_dir)
        evaluations += evaluate_levels(structural_filter(trusted_expanded), structural_gt, alias_map, "local_after_structural_strict_accepted", args.output_dir)
        evaluations += evaluate_levels(structural_filter(review_expanded), structural_gt, alias_map, "local_structural_review_pool", args.output_dir)
        evaluation_frame = pd.DataFrame(evaluations)
        evaluation_frame.to_csv(args.output_dir / "local_seed_recovery_evaluation_summary.csv", index=False)
        evaluation_frame[evaluation_frame["level"].eq("strict")].to_csv(
            args.output_dir / "local_strict_precision_recall_summary.csv", index=False
        )
        result["evaluations"] = evaluations
        result["consolidated_metrics_workbook"] = str(update_consolidated_stage_metrics(evaluations, result, args))

    observation_path = write_single_experiment_observation(
        result=result,
        scored=scored,
        revalidated=revalidated,
        evaluations=evaluations,
        output_dir=args.output_dir,
    )
    result["single_observation_file"] = str(observation_path)

    (args.output_dir / "local_seed_recovery_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def build_chunk_recovery_batch_prompt(chunk_row: pd.Series, shortlist: pd.DataFrame) -> str:
    """One recovery prompt for all shortlisted candidate/backlog terms in the current chunk."""
    items = []
    for row in shortlist.itertuples(index=False):
        data = row._asdict()
        items.append({
            "term": safe_text(data.get("term", "")),
            "original_state": safe_text(data.get("recovery_source_state", "")),
            "candidate_occurrence_sentence": safe_text(data.get("local_candidate_sentence", "")),
            "selected_seed_term": safe_text(data.get("selected_seed_term", "")),
            "seed_scope": safe_text(data.get("selected_seed_scope", "")),
            "term_similarity": float(data.get("selected_term_similarity", 0.0) or 0.0),
            "sentence_similarity": float(data.get("selected_sentence_similarity", 0.0) or 0.0),
            "extractor_count": int(data.get("extractor_count", 0) or 0),
            "source_methods": safe_text(data.get("source_methods", "")),
        })
    return f"""
You are strictly revalidating ontology recovery candidates from ONE current chunk.
Return one decision for every supplied candidate. Similarity is retrieval evidence only.
A selected seed may come from the current or immediately previous chunk, but no previous
chunk text or previous seed sentence is provided. Decide only from the current chunk.

Chunk ID: {safe_text(chunk_row.get('chunk_id', ''))}
Current chunk text:
{safe_text(chunk_row.get('text', ''))}

Recovery candidates:
{json.dumps(items, ensure_ascii=False, indent=2)}

Allowed categories: classification, mission_usecase, operating_condition, environment,
scenario_operation, system, subsystem, component, part, flow, port, interface.

For each exact candidate term:
- ACCEPT only if it occurs in the current chunk, is ENTITY_OR_CLASS, independently names a
  reusable ontology concept, has a directly supported allowed category, and confidence is high.
- REVIEW when it appears meaningful but current-chunk evidence or typing is incomplete.
- REJECT only for attributes, actions, descriptive phrases, document fragments, generic phrases,
  unsupported terms, or topic-only associations.
- Copy candidate_evidence_sentence exactly from the supplied candidate occurrence sentence.
- Do not invent or paraphrase evidence.

Return JSON only:
{{
  "chunk_id": "...",
  "decisions": [
    {{
      "term": "exact candidate term",
      "decision": "ACCEPT" or "REVIEW" or "REJECT",
      "term_nature": "ENTITY_OR_CLASS" or "ATTRIBUTE" or "ACTION" or "DESCRIPTIVE_PHRASE" or "DOCUMENT_FRAGMENT" or "OTHER",
      "proposed_types": ["..."],
      "confidence": "high" or "medium" or "low",
      "reason": "one short evidence-based reason",
      "candidate_evidence_sentence": "exact supplied candidate occurrence sentence, or empty string"
    }}
  ]
}}
""".strip()


def apply_chunk_recovery_batch(
    chunk_terms: pd.DataFrame,
    scored: pd.DataFrame,
    chunk_row: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Select one route per term, make at most one batched recovery LLM call, and apply strict decisions."""
    if scored.empty:
        return chunk_terms, [], {"shortlisted": 0, "llm_called": False}
    shortlist = scored[scored["similarity_shortlisted"].eq(True)].copy()
    if shortlist.empty or args.skip_revalidation or args.skip_llm:
        return chunk_terms, [], {"shortlisted": int(len(shortlist)), "llm_called": False}

    prompt = build_chunk_recovery_batch_prompt(chunk_row, shortlist)
    print(
        f"chunk={safe_text(chunk_row.get('chunk_id',''))} recovery shortlist={len(shortlist)}; "
        "one batched LLM recovery call"
    )
    started = time.perf_counter()
    try:
        payload, raw = ollama_chat_json(
            args.ollama_url, args.model, LLM_EVIDENCE_SYSTEM_PROMPT, prompt, args.timeout
        )
        error = ""
    except Exception as exc:
        payload, raw = {}, ""
        error = f"{type(exc).__name__}: {exc}"

    decisions = payload.get("decisions", []) if isinstance(payload, dict) else []
    decision_map = {
        normalize_term(item.get("term", "")): item
        for item in decisions if isinstance(item, dict) and normalize_term(item.get("term", ""))
    }
    logs: list[dict[str, Any]] = []
    updated = chunk_terms.copy()

    for row in shortlist.itertuples(index=False):
        data = row._asdict()
        normalized = normalize_term(data.get("term", ""))
        item = decision_map.get(normalized, {})
        decision = safe_text(item.get("decision", "REVIEW")).upper()
        term_nature = safe_text(item.get("term_nature", "OTHER")).upper()
        proposed_types = parse_proposed_types(item.get("proposed_types", []))
        confidence = safe_text(item.get("confidence", "low")).casefold()
        returned_evidence = safe_text(item.get("candidate_evidence_sentence", ""))
        candidate_sentence = safe_text(data.get("local_candidate_sentence", ""))
        chunk_text = safe_text(chunk_row.get("text", ""))
        evidence_exact = bool(returned_evidence and returned_evidence.casefold() == candidate_sentence.casefold())
        evidence_in_chunk = bool(returned_evidence and returned_evidence.casefold() in chunk_text.casefold())
        term_in_chunk = bool(normalized and normalized in normalize_term(chunk_text))
        extractor_count = int(data.get("extractor_count", 0) or 0)
        has_llm = "llm_evidence" in safe_text(data.get("source_methods", "")).casefold()
        source_state = safe_text(data.get("recovery_source_state", "")).casefold()
        support_ok = (
            extractor_count >= 2 or has_llm
            if source_state == CANDIDATE_STATE
            else extractor_count >= 2 and has_llm
        )
        accepted = bool(
            decision == "ACCEPT" and term_nature == "ENTITY_OR_CLASS" and confidence == "high"
            and proposed_types and evidence_exact and evidence_in_chunk and term_in_chunk and support_ok
        )
        recovery_state = (
            RECOVERED_ACCEPTED_STATE if accepted
            else RECOVERY_REJECTED_STATE if decision == "REJECT"
            else RECOVERY_CANDIDATE_STATE
        )
        mask = updated["normalized_term"].map(normalize_term).eq(normalized)
        updated.loc[mask, "recovery_state"] = recovery_state
        updated.loc[mask, "revalidation_decision"] = decision
        updated.loc[mask, "revalidation_term_nature"] = term_nature
        updated.loc[mask, "revalidation_confidence"] = confidence
        updated.loc[mask, "revalidation_reason"] = safe_text(item.get("reason", ""))
        updated.loc[mask, "selected_route"] = safe_text(data.get("selected_route", ""))
        updated.loc[mask, "selected_seed_term"] = safe_text(data.get("selected_seed_term", ""))
        updated.loc[mask, "selected_seed_scope"] = safe_text(data.get("selected_seed_scope", ""))
        updated.loc[mask, "selected_term_similarity"] = data.get("selected_term_similarity", None)
        updated.loc[mask, "selected_sentence_similarity"] = data.get("selected_sentence_similarity", None)
        if accepted:
            updated.loc[mask, "previous_final_state"] = updated.loc[mask, "final_state"]
            updated.loc[mask, "final_state"] = RECOVERED_ACCEPTED_STATE
            updated.loc[mask, "proposed_type"] = " | ".join(proposed_types)
            updated.loc[mask, "llm_confidence"] = confidence
            updated.loc[mask, "evidence_sentence"] = returned_evidence
        logs.append({
            **data,
            "revalidation_decision": decision,
            "revalidation_term_nature": term_nature,
            "revalidated_types": " | ".join(proposed_types) if proposed_types else "other",
            "revalidation_confidence": confidence,
            "revalidation_reason": safe_text(item.get("reason", "")),
            "revalidation_evidence_sentence": returned_evidence,
            "evidence_exact_to_local_sentence": evidence_exact,
            "evidence_grounded_in_current_chunk": evidence_in_chunk,
            "term_grounded_in_current_chunk": term_in_chunk,
            "revalidation_extraction_support": support_ok,
            "recovered_accepted": accepted,
            "recovery_state": recovery_state,
            "revalidation_prompt": prompt,
            "revalidation_raw_response": raw,
            "revalidation_error": error,
            "revalidation_duration_seconds": time.perf_counter() - started,
        })
    return updated, logs, {"shortlisted": int(len(shortlist)), "llm_called": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Precision-first ontology pipeline v5 controlled baseline with dual GT evaluation and shared normalization.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--broad-gt-file", type=Path, default=DEFAULT_BROAD_GT_FILE)
    parser.add_argument("--structural-gt-file", type=Path, default=DEFAULT_STRUCTURAL_GT_FILE)
    parser.add_argument("--broad-gt-sheet", default=BROAD_GT_SHEET)
    parser.add_argument("--broad-gt-scope", default=BROAD_GT_SCOPE, help="Used only for clarified-schema compatibility; original S1 uses grounded_status_v2.")
    parser.add_argument("--structural-gt-sheet", default=STRUCTURAL_GT_SHEET)
    parser.add_argument("--skip-gt-evaluation", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--max-words", type=int, default=220)
    parser.add_argument("--overlap-sentences", type=int, default=1)
    parser.add_argument("--top-k-per-chunk", type=int, default=10)
    parser.add_argument("--extractors", nargs="+", default=["yake","keybert_mpnet","acronyms","tfidf","noun_phrases","textrank"])
    parser.add_argument("--keybert-model", default="all-mpnet-base-v2")
    parser.add_argument("--timeout", type=int, default=300, help="Seconds to wait for each Ollama request before continuing with an empty payload.")
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-seed-scoring", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-recovery", action="store_true", help="Skip accepted-seed recall recovery.")
    parser.add_argument("--skip-revalidation", action="store_true", help="Score and shortlist only; do not call the LLM revalidator.")
    parser.add_argument("--candidate-term-threshold", type=float, default=0.80)
    parser.add_argument("--candidate-sentence-threshold", type=float, default=0.70)
    parser.add_argument("--backlog-term-threshold", type=float, default=0.90)
    parser.add_argument("--backlog-sentence-threshold", type=float, default=0.75)
    parser.add_argument("--previous-candidate-term-threshold", type=float, default=0.85)
    parser.add_argument("--previous-candidate-sentence-threshold", type=float, default=0.75)
    parser.add_argument("--previous-backlog-term-threshold", type=float, default=0.92)
    parser.add_argument("--previous-backlog-sentence-threshold", type=float, default=0.80)
    parser.add_argument("--term-similarity-weight", type=float, default=0.70)
    parser.add_argument("--sentence-similarity-weight", type=float, default=0.30)
    args = parser.parse_args()

    embedding_model = None
    if not args.skip_recovery:
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(args.keybert_model)
        except ImportError as error:
            raise RuntimeError(
                "Accepted-seed recovery requires sentence-transformers. "
                "Install it or run with --skip-recovery."
            ) from error

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Integrated recovery is part of each chunk lifecycle; do not use a completed global baseline resume path.
    existing_inventory = args.output_dir / "final_ontology_candidates.csv"
    if existing_inventory.is_file() and not args.force_rerun:
        raise RuntimeError(
            "v11 performs recovery inside each chunk pass. Use a clean output directory or --force-rerun; "
            "a completed baseline inventory cannot reconstruct the in-chunk call order."
        )

    chunks = load_chunks(args.input_file, args.max_words, args.overlap_sentences)
    if args.max_chunks and args.max_chunks > 0:
        chunks = chunks.head(args.max_chunks).copy().reset_index(drop=True)

    detected_aliases = detect_corpus_aliases(chunks)
    detected_aliases.to_csv(args.output_dir / "detected_corpus_aliases.csv", index=False)

    extractor_plan_items = extractor_plan(args.extractors, args.top_k_per_chunk, args.keybert_model)

    chunk_analysis_rows: list[dict[str, Any]] = []
    gate_log_rows: list[dict[str, Any]] = []
    evidence_log_rows: list[dict[str, Any]] = []
    extractor_run_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    chunk_recovery_similarity_rows: list[dict[str, Any]] = []
    chunk_recovery_shortlist_rows: list[dict[str, Any]] = []
    chunk_revalidation_rows: list[dict[str, Any]] = []
    chunk_recovery_run_rows: list[dict[str, Any]] = []
    previous_chunk_original_accepted = pd.DataFrame()

    # Current script is single-process and always starts fresh unless you add resume logic.
    # force_rerun is retained for interface compatibility.
    for chunk in chunks.itertuples(index=False):
        chunk_id = safe_text(chunk.chunk_id)
        chunk_series = pd.Series(chunk._asdict())

        gate_prompt = build_gate_prompt(chunk_series)
        gate_started = time.perf_counter()
        if args.skip_llm:
            gate_payload = {
                "gate_decision": "YES",
                "reason": "LLM skipped by flag",
                "possible_categories": [],
                "out_of_scope_theme": None,
            }
            gate_raw = ""
        else:
            gate_payload, gate_raw = ollama_chat_json(
                args.ollama_url,
                args.model,
                LLM_GATE_SYSTEM_PROMPT,
                gate_prompt,
                args.timeout,
            )
        gate_duration = time.perf_counter() - gate_started
        gate_decision = safe_text(gate_payload.get("gate_decision", "")).upper()
        gate_decision = "YES" if gate_decision == "YES" else "NO"

        chunk_analysis_rows.append({
            "chunk_id": chunk_id,
            "chunk_index": int(chunk.chunk_index),
            "section": safe_text(chunk.section),
            "gate_decision": gate_decision,
            "gate_reason": safe_text(gate_payload.get("reason", "")),
            "possible_categories": " | ".join(gate_payload.get("possible_categories", []) or []),
            "out_of_scope_theme": safe_text(gate_payload.get("out_of_scope_theme", "")),
            "source_doc": safe_text(args.input_file.name),
            "gate_duration_seconds": gate_duration,
        })
        gate_log_rows.append({
            "chunk_id": chunk_id,
            "gate_prompt": gate_prompt,
            "gate_raw_response": gate_raw,
            "gate_payload_json": json.dumps(gate_payload, ensure_ascii=False),
            "gate_duration_seconds": gate_duration,
        })

        if gate_decision != "YES":
            previous_chunk_original_accepted = pd.DataFrame()
            continue

        evidence_prompt = build_evidence_prompt(chunk_series)
        evidence_started = time.perf_counter()
        if args.skip_llm:
            evidence_payload = {"chunk_id": chunk_id, "concepts_present": [], "candidate_terms": []}
            evidence_raw = ""
        else:
            evidence_payload, evidence_raw = ollama_chat_json(
                args.ollama_url,
                args.model,
                LLM_EVIDENCE_SYSTEM_PROMPT,
                evidence_prompt,
                args.timeout,
            )
        evidence_duration = time.perf_counter() - evidence_started
        evidence_log_rows.append({
            "chunk_id": chunk_id,
            "evidence_prompt": evidence_prompt,
            "evidence_raw_response": evidence_raw,
            "evidence_payload_json": json.dumps(evidence_payload, ensure_ascii=False),
            "evidence_duration_seconds": evidence_duration,
        })

        chunk_df = pd.DataFrame([chunk_series.to_dict()])
        extractor_frames: list[pd.DataFrame] = []
        for extractor_name, params in extractor_plan_items:
            try:
                extracted = run_extractor(extractor_name, chunk_df, params)
                status = "ok"
                count = int(len(extracted))
            except Exception as error:
                extracted = pd.DataFrame(columns=["raw_term", "score", "extractor", "chunk_id", "source_sentence", "context_text"])
                status = f"error: {type(error).__name__}: {error}"
                count = 0
            extractor_run_rows.append({
                "chunk_id": chunk_id,
                "extractor_name": extractor_name,
                "extractor_status": status,
                "candidate_count": count,
            })
            if not extracted.empty:
                extractor_frames.append(extracted)

        extractor_terms = (
            pd.concat(extractor_frames, ignore_index=True)
            if extractor_frames
            else pd.DataFrame(columns=["raw_term", "score", "extractor", "chunk_id", "source_sentence", "context_text"])
        )

        merged_terms = build_term_records(chunk_series, evidence_payload, extractor_terms)
        if merged_terms.empty:
            continue

        # Do not use static DOMAIN_SEEDS in the baseline. Accepted seeds are only
        # available after final state assignment and are used in the recovery stage.
        merged_terms = safe_score_seed_similarity(merged_terms, embedding_model, args)

        merged_terms["chunk_index"] = int(chunk.chunk_index)
        merged_terms["section"] = safe_text(chunk.section)
        merged_terms["source_doc"] = safe_text(args.input_file.name)
        merged_terms["gate_decision"] = gate_decision

        chunk_state_records: list[dict[str, Any]] = []
        for row in merged_terms.itertuples(index=False):
            record = row._asdict()
            initial_state, final_state, confidence_score = assign_state(pd.Series(record))
            record.update({
                "initial_state": initial_state,
                "final_state": final_state,
                "confidence_score": confidence_score,
                "evidence_sentence": safe_text(record.get("evidence_sentence", "")),
                "source_methods": safe_text(record.get("source_methods", "")),
                "max_seed_similarity": float(record.get("max_seed_similarity", 0.0) or 0.0),
                "closest_seed": safe_text(record.get("closest_seed", "")),
            })
            chunk_state_records.append(record)

        chunk_terms = pd.DataFrame(chunk_state_records)
        original_local_accepted = chunk_terms[
            chunk_terms["final_state"].eq(ACCEPTED_STATE)
        ].copy()

        if args.skip_recovery or embedding_model is None:
            scored_chunk = pd.DataFrame()
            recovery_logs: list[dict[str, Any]] = []
            recovery_run = {"shortlisted": 0, "llm_called": False}
        else:
            scoring_frame = pd.concat(
                [previous_chunk_original_accepted, chunk_terms], ignore_index=True, sort=False
            )
            scored_chunk, _ = score_local_chunk_similarity(scoring_frame, embedding_model, args)
            if not scored_chunk.empty:
                scored_chunk = scored_chunk[
                    pd.to_numeric(scored_chunk["chunk_index"], errors="coerce").eq(int(chunk.chunk_index))
                ].copy()
                chunk_recovery_similarity_rows.extend(scored_chunk.to_dict("records"))
                shortlist_chunk = scored_chunk[scored_chunk["similarity_shortlisted"].eq(True)].copy()
                chunk_recovery_shortlist_rows.extend(shortlist_chunk.to_dict("records"))
            chunk_terms, recovery_logs, recovery_run = apply_chunk_recovery_batch(
                chunk_terms, scored_chunk, chunk_series, args
            )
            chunk_revalidation_rows.extend(recovery_logs)

        chunk_recovery_run_rows.append({
            "chunk_id": chunk_id,
            "chunk_index": int(chunk.chunk_index),
            "original_accepted": int(len(original_local_accepted)),
            "candidate_backlog_scored": int(len(scored_chunk)),
            "shortlisted": int(recovery_run.get("shortlisted", 0)),
            "recovery_llm_called": bool(recovery_run.get("llm_called", False)),
        })
        final_rows.extend(chunk_terms.to_dict("records"))
        # Frozen rolling memory: only original accepted terms from this chunk feed the next chunk.
        previous_chunk_original_accepted = original_local_accepted.copy()

    chunk_analysis = pd.DataFrame(chunk_analysis_rows)
    gate_logs = pd.DataFrame(gate_log_rows)
    evidence_logs = pd.DataFrame(evidence_log_rows)
    extractor_runs = pd.DataFrame(extractor_run_rows)
    final_terms = pd.DataFrame(final_rows) if final_rows else build_empty_final_terms()

    if not final_terms.empty:
        final_terms = final_terms.drop_duplicates(["chunk_id", "normalized_term"], keep="first").reset_index(drop=True)
        final_terms["normalized_term"] = final_terms["normalized_term"].map(normalize_term)

    # v2: accepted means only final_state == accepted.
    # promoted is reserved for future revalidation script and should normally be empty in this script.
    accepted = final_terms[final_terms["final_state"].eq(ACCEPTED_STATE)].copy().reset_index(drop=True)
    promoted = final_terms[final_terms["final_state"].eq(PROMOTED_STATE)].copy().reset_index(drop=True)
    candidate = final_terms[final_terms["final_state"].eq(CANDIDATE_STATE)].copy().reset_index(drop=True)
    backlog = final_terms[final_terms["final_state"].eq(BACKLOG_STATE)].copy().reset_index(drop=True)
    rejected = final_terms[final_terms["final_state"].eq(REJECTED_STATE)].copy().reset_index(drop=True)

    recovered_accepted = final_terms[final_terms["final_state"].eq(RECOVERED_ACCEPTED_STATE)].copy().reset_index(drop=True)
    trusted_accepted = pd.concat([accepted, recovered_accepted], ignore_index=True, sort=False).drop_duplicates("normalized_term")
    recovered_accepted.to_csv(args.output_dir / "in_chunk_recovered_accepted_terms.csv", index=False)
    trusted_accepted.to_csv(args.output_dir / "accepted_plus_in_chunk_recovered.csv", index=False)
    chunk_analysis.to_csv(args.output_dir / "chunk_analysis.csv", index=False)
    gate_logs.to_csv(args.output_dir / "gate_responses.csv", index=False)
    evidence_logs.to_csv(args.output_dir / "llm_evidence_responses.csv", index=False)
    extractor_runs.to_csv(args.output_dir / "extractor_runs.csv", index=False)
    final_terms.to_csv(args.output_dir / "final_ontology_candidates.csv", index=False)
    accepted.to_csv(args.output_dir / "accepted_terms.csv", index=False)
    promoted.to_csv(args.output_dir / "promoted_terms.csv", index=False)
    candidate.to_csv(args.output_dir / "candidate_terms.csv", index=False)
    backlog.to_csv(args.output_dir / "backlog_terms.csv", index=False)
    rejected.to_csv(args.output_dir / "rejected_terms.csv", index=False)

    term_evidence_cols = [
        "term", "normalized_term", "chunk_id", "evidence_sentence", "source_methods",
        "proposed_type", "llm_confidence", "initial_state", "final_state", "confidence_score",
    ]
    for col in term_evidence_cols:
        if col not in final_terms.columns:
            final_terms[col] = ""
    final_terms[term_evidence_cols].to_csv(args.output_dir / "term_evidence.csv", index=False)
    final_terms.to_csv(args.output_dir / "seed_scored_terms_diagnostic.csv", index=False)

    recovery_summary: dict[str, Any] | None = {
        "architecture": "in_chunk_single_batched_revalidation",
        "chunks_with_recovery_llm_call": int(sum(bool(row["recovery_llm_called"]) for row in chunk_recovery_run_rows)),
        "total_shortlisted_terms": int(sum(int(row["shortlisted"]) for row in chunk_recovery_run_rows)),
        "previous_chunk_context_sent_to_llm": False,
    }
    pd.DataFrame(chunk_recovery_similarity_rows).to_csv(args.output_dir / "in_chunk_recovery_similarity.csv", index=False)
    pd.DataFrame(chunk_recovery_shortlist_rows).to_csv(args.output_dir / "in_chunk_recovery_shortlist.csv", index=False)
    pd.DataFrame(chunk_revalidation_rows).to_csv(args.output_dir / "in_chunk_revalidation.csv", index=False)
    pd.DataFrame(chunk_recovery_run_rows).to_csv(args.output_dir / "in_chunk_recovery_runs.csv", index=False)

    summary: dict[str, Any] = {
        "version": "precision_first_ontology_pipeline_v5_controlled_baseline",
        "input_file": str(args.input_file),
        "broad_gt_file": str(args.broad_gt_file),
        "structural_gt_file": str(args.structural_gt_file),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "extractors": args.extractors,
        "chunks": int(len(chunks)),
        "gate_yes_chunks": int((chunk_analysis["gate_decision"] == "YES").sum()) if not chunk_analysis.empty else 0,
        "accepted_terms": int(len(accepted)),
        "promoted_terms": int(len(promoted)),
        "candidate_terms": int(len(candidate)),
        "backlog_terms": int(len(backlog)),
        "rejected_terms": int(len(rejected)),
        "in_chunk_recovered_accepted_terms": int(len(recovered_accepted)),
        "trusted_accepted_terms": int(len(trusted_accepted)),
        "seed_similarity_note": "Recovery uses only current-run accepted terms as seeds; static DOMAIN_SEEDS are not used.",
        "recovery_seed_source": "final_ontology_candidates.csv rows where final_state == accepted",
        "seed_recovery": recovery_summary,
    }

    # Two-layer evaluation. GT aliases are evaluation-only and never change term state.
    if not args.skip_gt_evaluation:
        try:
            broad_gt = load_broad_gt(args.broad_gt_file, args.broad_gt_sheet, args.broad_gt_scope)
            structural_gt = load_structural_gt(args.structural_gt_file, args.structural_gt_sheet)
            broad_aliases = load_gt_aliases(args.broad_gt_file, "S2_GT_ALIASES", broad=True)
            structural_aliases = load_gt_aliases(args.structural_gt_file, "GT_ALIASES", broad=False)
            alias_map = build_alias_family_map([detected_aliases, broad_aliases, structural_aliases])

            all_discovery = pd.concat([accepted, candidate, backlog], ignore_index=True).drop_duplicates("normalized_term")
            structural_accepted = structural_filter(accepted)
            structural_discovery = structural_filter(all_discovery)

            structural_accepted.to_csv(args.output_dir / "structural_accepted_terms.csv", index=False)
            structural_discovery.to_csv(args.output_dir / "structural_discovery_pool.csv", index=False)

            evaluations: list[dict[str, Any]] = []
            evaluations += evaluate_levels(accepted, broad_gt, alias_map, "broad_accepted", args.output_dir)
            evaluations += evaluate_levels(all_discovery, broad_gt, alias_map, "broad_discovery_pool", args.output_dir)
            evaluations += evaluate_levels(structural_accepted, structural_gt, alias_map, "structural_accepted", args.output_dir)
            evaluations += evaluate_levels(structural_discovery, structural_gt, alias_map, "structural_discovery_pool", args.output_dir)
            pd.DataFrame(evaluations).to_csv(args.output_dir / "dual_gt_evaluation_summary.csv", index=False)
            summary["dual_gt_evaluation"] = evaluations
            summary["detected_corpus_aliases"] = int(len(detected_aliases))
            dashboard_path = write_consolidated_metrics_workbook(
                evaluations=evaluations,
                summary=summary,
                output_dir=args.output_dir,
            )
            summary["consolidated_metrics_workbook"] = str(dashboard_path)
        except Exception as error:
            summary["ground_truth_evaluation_error"] = f"{type(error).__name__}: {error}"
            print(f"WARN: dual GT evaluation failed: {type(error).__name__}: {error}")

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Precision-first ontology pipeline v11 in-chunk recovery completed")
    print(f"Output directory: {args.output_dir}")
    print(f"Chunks: {len(chunks)}")
    print(f"Gate YES chunks: {summary['gate_yes_chunks']}")
    print(f"Accepted terms: {len(accepted)}")
    print(f"Promoted terms: {len(promoted)}")
    print(f"Candidate terms: {len(candidate)}")
    print(f"Backlog terms: {len(backlog)}")
    print(f"Rejected terms: {len(rejected)}")
    if "dual_gt_evaluation" in summary:
        print("Dual GT metrics written to dual_gt_evaluation_summary.csv and consolidated_metrics_dashboard.xlsx")
        for row in summary["dual_gt_evaluation"]:
            if row["level"] in {"strict", "safe_extended", "context_extended"}:
                print(
                    f"{row['evaluation_name']} [{row['level']}]: "
                    f"P/R/F1={row['precision']:.4f}/{row['recall']:.4f}/{row['f1']:.4f}"
                )


if __name__ == "__main__":
    main()
