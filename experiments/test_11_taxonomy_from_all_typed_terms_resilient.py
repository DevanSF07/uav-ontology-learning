#!/usr/bin/env python3
"""Multi-agent taxonomy builder for entity-typed Qwen terms.

Accepted graph semantics
------------------------
1. System --is_a--> System
2. Subsystem --is_a--> Subsystem
3. System --has_subsystem--> Subsystem

The script consumes the completed entity-typing output, selects terms whose final
predicted type is System or Subsystem, canonicalizes nodes with aliases, retrieves
candidate relations with MPNet, validates each candidate with a Qwen Relation Agent
and Relation Critic, selects nearest direct is_a parents, performs deterministic graph
validation, and evaluates covered edges against Ground_Truth.xlsx / S5_GT_RELATIONS.

No term-specific relationship rules are hardcoded. Only the ontology schema and
allowed endpoint-type combinations are enforced deterministically.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


PROJECT_DIR = Path("/home/faujdar/onto_gen_product/ontology_experiment_framework")
DEFAULT_TYPING_DIR = PROJECT_DIR / "stage_outputs/10_entity_typing_all_qwen_terms"
DEFAULT_TYPING_RESULTS = DEFAULT_TYPING_DIR / "08_all_qwen_terms_with_entity_types.csv"
DEFAULT_CONTEXTS = DEFAULT_TYPING_DIR / "03_retrieved_contexts.csv"
DEFAULT_ALIAS_GT = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_CORPUS = PROJECT_DIR / "data/combined_corpus_raw.txt"
DEFAULT_OUTPUT = PROJECT_DIR / "stage_outputs/11_taxonomy_from_all_typed_terms"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

ALLOWED_NODE_TYPES = {"System", "Subsystem"}
POSITIVE_RELATIONS = {"is_a", "has_subsystem"}
SAME_TYPE_DECISIONS = {"is_a", "equivalent", "related_non_hierarchical", "no_relation", "uncertain"}
CROSS_TYPE_DECISIONS = {"has_subsystem", "related_non_hierarchical", "no_relation", "uncertain"}
CRITIC_DECISIONS = {"accept", "reject", "reverse_direction", "change_relation", "manual_review"}


def safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize(value: Any) -> str:
    text = safe_text(value).casefold().replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def singularize_head(value: Any) -> str:
    words = normalize(value).split()
    if not words:
        return ""
    word = words[-1]
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        word = word[:-1]
    words[-1] = word
    return " ".join(words)


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_json_like(value: Any, expected: type = list) -> Any:
    if isinstance(value, expected):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return expected()
    text = str(value).strip()
    if not text:
        return expected()
    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(text)
            return result if isinstance(result, expected) else expected()
        except Exception:
            continue
    return expected()


def validate_confidence(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def exact_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found")
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object")
    return obj


class OllamaClient:
    def __init__(self, model: str, base_url: str, output_dir: Path, timeout: int, retries: int, temperature: float):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.cache_dir = output_dir / "cache" / "qwen"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = output_dir / "qwen_requests.jsonl"
        self.live_calls = 0
        self.cache_hits = 0

    def check(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = {m.get("name", "") for m in payload.get("models", [])}
        if self.model not in names and not any(n.startswith(self.model + ":") for n in names):
            raise RuntimeError(f"Ollama model '{self.model}' not found. Available: {sorted(names)}")

    def chat_json(
        self,
        messages: list[dict[str, str]],
        stage: str,
        validator: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        signature = json.dumps(
            {"model": self.model, "stage": stage, "messages": messages, "temperature": self.temperature},
            sort_keys=True,
            ensure_ascii=False,
        )
        key = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            try:
                validated = validator(cached["parsed"])
                self.cache_hits += 1
                return validated, {**cached.get("meta", {}), "cache_hit": True}
            except Exception:
                cache_path.unlink(missing_ok=True)

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            started = time.perf_counter()
            request_payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": self.temperature},
            }
            request = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(request_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw_payload = json.loads(response.read().decode("utf-8"))
                raw_text = raw_payload.get("message", {}).get("content", "")
                parsed = exact_json_object(raw_text)
                validated = validator(parsed)
                runtime = time.perf_counter() - started
                meta = {"stage": stage, "attempt": attempt, "runtime_seconds": runtime, "cache_hit": False}
                cache_path.write_text(
                    json.dumps({"parsed": validated, "raw": raw_text, "meta": meta}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"stage": stage, "messages": messages, "response": validated, **meta}, ensure_ascii=False) + "\n")
                self.live_calls += 1
                return validated, meta
            except Exception as exc:
                last_error = exc
                if attempt <= self.retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Qwen stage '{stage}' failed: {last_error}")


def load_aliases(path: Path, sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    required = {"canonical", "alias"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Alias sheet missing columns {sorted(required)}")
    if "grounded_status_v2" in frame.columns:
        status = frame["grounded_status_v2"].fillna("").astype(str).str.casefold()
        frame = frame[status.ne("needs_manual_review")].copy()
    frame["canonical"] = frame["canonical"].map(safe_text)
    frame["alias"] = frame["alias"].map(safe_text)
    frame["canonical_norm"] = frame["canonical"].map(normalize)
    frame["alias_norm"] = frame["alias"].map(normalize)
    return frame[frame.canonical_norm.ne("") & frame.alias_norm.ne("")].copy()


def alias_components(aliases: pd.DataFrame) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, str] = {}
    for row in aliases.itertuples(index=False):
        graph[row.canonical_norm].add(row.alias_norm)
        graph[row.alias_norm].add(row.canonical_norm)
        labels[row.canonical_norm] = row.canonical
        labels[row.alias_norm] = row.alias
    components: dict[str, set[str]] = {}
    seen: set[str] = set()
    for node in graph:
        if node in seen:
            continue
        stack, comp = [node], set()
        while stack:
            current = stack.pop()
            if current in comp:
                continue
            comp.add(current)
            stack.extend(graph.get(current, set()) - comp)
        seen |= comp
        canonical_candidates = [r.canonical_norm for r in aliases.itertuples(index=False) if r.canonical_norm in comp]
        canonical = sorted(canonical_candidates, key=lambda x: (-len(x.split()), -len(x), x))[0] if canonical_candidates else sorted(comp)[0]
        for member in comp:
            components[member] = comp | {canonical}
    return components


def load_typing_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"canonical_term", "final_type", "final_confidence"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Typing results missing {sorted(required - set(frame.columns))}")
    frame = frame.copy()
    frame["canonical_term"] = frame["canonical_term"].map(safe_text)
    frame["term_norm"] = frame["canonical_term"].map(normalize)
    frame["final_type"] = frame["final_type"].map(safe_text)
    frame["final_confidence"] = pd.to_numeric(frame["final_confidence"], errors="coerce").fillna(0.0)
    if "evidence_quality" not in frame.columns:
        frame["evidence_quality"] = 0.5
    frame["evidence_quality"] = pd.to_numeric(frame["evidence_quality"], errors="coerce").fillna(0.5)
    return frame[frame.final_type.isin(ALLOWED_NODE_TYPES)].drop_duplicates("term_norm").reset_index(drop=True)


def load_contexts(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, low_memory=False)
    key = "candidate_norm" if "candidate_norm" in frame.columns else "canonical_term"
    if "sentence_rank" in frame.columns:
        frame = frame.sort_values([key, "sentence_rank"])
    result: dict[str, list[dict[str, Any]]] = {}
    for value, group in frame.groupby(key, dropna=False):
        result[normalize(value)] = group.head(5).to_dict("records")
    return result


def deterministic_node_key(term: str, alias_map: dict[str, set[str]]) -> str:
    norm = normalize(term)
    morph = singularize_head(term)
    candidates = set(alias_map.get(norm, {norm})) | {norm, morph}
    return sorted((c for c in candidates if c), key=lambda x: (-len(x.split()), -len(x), x))[0]


def context_text(contexts: list[dict[str, Any]]) -> str:
    return " ".join(safe_text(item.get("sentence")) for item in contexts if safe_text(item.get("sentence")))


def node_prompt(record: dict[str, Any], known_aliases: list[str], contexts: list[dict[str, Any]]) -> str:
    evidence = [
        {"sentence_id": safe_text(x.get("sentence_id")), "sentence": safe_text(x.get("sentence"))}
        for x in contexts
    ]
    return f"""Review one entity-typed term for use as a taxonomy node.

TERM: {record['canonical_term']}
PREDICTED TYPE: {record['final_type']}
TYPING CONFIDENCE: {record['final_confidence']}
ENTITY DESCRIPTION: {record.get('entity_description', '')}
KNOWN ALIASES: {json.dumps(known_aliases, ensure_ascii=False)}
EVIDENCE: {json.dumps(evidence, ensure_ascii=False, indent=2)}

A valid node must denote a standalone System or Subsystem class/concept. Reject activities,
properties, relations, sentence fragments, and incidental mentions. Do not invent a broader
or narrower label. Preserve the supplied type. Known aliases may be consolidated but semantic
similarity alone does not prove equivalence.

Return JSON:
{{
  "decision": "accept_as_node|reject_as_node|manual_review",
  "canonical_label": "best existing label from term or known aliases",
  "node_type": {json.dumps(record['final_type'])},
  "aliases": ["verified surface forms only"],
  "confidence": 0.0,
  "reason": "semantic reason",
  "supporting_sentence_ids": ["IDs"]
}}"""


def validate_node(obj: dict[str, Any], expected_type: str, available_labels: set[str]) -> dict[str, Any]:
    decision = safe_text(obj.get("decision"))
    if decision not in {"accept_as_node", "reject_as_node", "manual_review"}:
        raise ValueError("Invalid node decision")
    node_type = safe_text(obj.get("node_type"))
    if node_type != expected_type:
        raise ValueError("Node type changed")
    canonical = safe_text(obj.get("canonical_label"))
    if decision == "accept_as_node" and normalize(canonical) not in {normalize(x) for x in available_labels}:
        raise ValueError("Canonical label was invented")
    aliases = [safe_text(x) for x in obj.get("aliases", []) if safe_text(x)]
    aliases = [x for x in aliases if normalize(x) in {normalize(y) for y in available_labels}]
    return {
        "decision": decision,
        "canonical_label": canonical,
        "node_type": node_type,
        "aliases": aliases,
        "confidence": validate_confidence(obj.get("confidence")),
        "reason": safe_text(obj.get("reason")),
        "supporting_sentence_ids": [safe_text(x) for x in obj.get("supporting_sentence_ids", []) if safe_text(x)],
    }


def build_nodes(
    typed: pd.DataFrame,
    aliases: pd.DataFrame,
    contexts: dict[str, list[dict[str, Any]]],
    client: OllamaClient,
    typing_threshold: float,
    evidence_threshold: float,
    use_node_agent: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alias_map = alias_components(aliases)
    grouped: dict[tuple[str, str], list[pd.Series]] = defaultdict(list)
    for _, row in typed.iterrows():
        key = deterministic_node_key(row.canonical_term, alias_map)
        grouped[(key, row.final_type)].append(row)

    accepted, rejected, alias_rows = [], [], []
    node_number = 0
    for (key, node_type), rows in sorted(grouped.items()):
        best = max(rows, key=lambda r: (float(r.final_confidence), float(r.evidence_quality)))
        surface_forms = sorted({safe_text(r.canonical_term) for r in rows if safe_text(r.canonical_term)})
        for member in alias_map.get(normalize(best.canonical_term), set()):
            matching = aliases[(aliases.canonical_norm.eq(member)) | (aliases.alias_norm.eq(member))]
            surface_forms.extend(matching.canonical.tolist() + matching.alias.tolist())
        surface_forms = sorted(set(filter(None, map(safe_text, surface_forms))))
        evidence = contexts.get(normalize(best.get("candidate_norm", best.canonical_term)), contexts.get(normalize(best.canonical_term), []))
        record = best.to_dict()

        confidence_ok = float(best.final_confidence) >= typing_threshold
        evidence_ok = float(best.evidence_quality) >= evidence_threshold
        deterministic_profile = {
            "decision": "accept_as_node" if confidence_ok and evidence_ok else "manual_review",
            "canonical_label": best.canonical_term,
            "node_type": node_type,
            "aliases": [x for x in surface_forms if normalize(x) != normalize(best.canonical_term)],
            "confidence": min(float(best.final_confidence), float(best.evidence_quality)),
            "reason": "Deterministic eligibility using typing confidence and evidence quality.",
            "supporting_sentence_ids": [safe_text(x.get("sentence_id")) for x in evidence],
        }

        if use_node_agent:
            try:
                profile, _ = client.chat_json(
                    [
                        {"role": "system", "content": (
                            "You validate ontology nodes. Use evidence, preserve supplied types, "
                            "choose canonical_label verbatim from TERM or KNOWN ALIASES, and return JSON only."
                        )},
                        {"role": "user", "content": node_prompt(record, surface_forms, evidence)},
                    ],
                    stage=f"node::{key}::{node_type}",
                    validator=lambda obj, nt=node_type, labels=set(surface_forms): validate_node(obj, nt, labels),
                )
            except RuntimeError as node_error:
                # Never abort the complete taxonomy because one node-agent response
                # paraphrased or invented a canonical label. Fall back to the
                # original typed term, which is guaranteed to be an existing label.
                print(f"Node Agent failed validation for '{best.canonical_term}'; using deterministic fallback: {node_error}")
                profile = dict(deterministic_profile)
                profile["reason"] = (
                    "Node Agent validation failed; deterministic existing-label fallback used. "
                    + str(node_error)
                )
        else:
            profile = deterministic_profile

        target = accepted if profile["decision"] == "accept_as_node" else rejected
        node_id = ""
        if profile["decision"] == "accept_as_node":
            node_number += 1
            node_id = f"NODE_{node_number:04d}"
        payload = {
            "node_id": node_id,
            "canonical_label": profile["canonical_label"],
            "label_norm": normalize(profile["canonical_label"]),
            "aliases": json.dumps(profile["aliases"], ensure_ascii=False),
            "node_type": node_type,
            "typing_confidence": float(best.final_confidence),
            "evidence_quality": float(best.evidence_quality),
            "node_confidence": profile["confidence"],
            "entity_description": safe_text(best.get("entity_description")),
            "supporting_sentence_ids": json.dumps(profile["supporting_sentence_ids"], ensure_ascii=False),
            "evidence_text": context_text(evidence),
            "source_terms": json.dumps(surface_forms, ensure_ascii=False),
            "node_decision": profile["decision"],
            "node_reason": profile["reason"],
        }
        target.append(payload)
        if node_id:
            for label in surface_forms:
                alias_rows.append({"surface_form": label, "surface_norm": normalize(label), "node_id": node_id, "canonical_label": profile["canonical_label"]})

    return pd.DataFrame(accepted), pd.DataFrame(rejected), pd.DataFrame(alias_rows)


def load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def encode_cached(model: Any, texts: list[str], cache_path: Path, batch_size: int) -> np.ndarray:
    signature = hashlib.sha256((getattr(model, "model_card_data", None).__class__.__name__ + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()
    meta_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        matrix = np.load(cache_path)
        if meta.get("signature") == signature and matrix.shape[0] == len(texts):
            return matrix
    matrix = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    matrix = np.asarray(matrix, dtype=np.float32)
    np.save(cache_path, matrix)
    json_dump(meta_path, {"signature": signature, "rows": len(texts)})
    return matrix


def node_embedding_text(row: pd.Series) -> str:
    return " | ".join(filter(None, [
        safe_text(row.canonical_label),
        safe_text(row.aliases),
        safe_text(row.node_type),
        safe_text(row.entity_description),
        safe_text(row.evidence_text),
    ]))


def allowed_relation(parent_type: str, child_type: str) -> str | None:
    if parent_type == child_type and parent_type in ALLOWED_NODE_TYPES:
        return "is_a"
    if parent_type == "System" and child_type == "Subsystem":
        return "has_subsystem"
    return None


def generate_candidate_pairs(nodes: pd.DataFrame, embeddings: np.ndarray, top_k: int) -> pd.DataFrame:
    rows = []
    similarity = embeddings @ embeddings.T
    for child_idx, child in nodes.iterrows():
        pools = []
        if child.node_type == "System":
            pools.append(("System", "is_a"))
        else:
            pools.extend([("Subsystem", "is_a"), ("System", "has_subsystem")])
        for parent_type, relation in pools:
            candidates = []
            for parent_idx, parent in nodes[nodes.node_type.eq(parent_type)].iterrows():
                if parent_idx == child_idx:
                    continue
                candidates.append((float(similarity[child_idx, parent_idx]), parent_idx, parent))
            for rank, (score, _, parent) in enumerate(sorted(candidates, reverse=True, key=lambda x: x[0])[:top_k], start=1):
                rows.append({
                    "pair_id": f"PAIR_{len(rows)+1:06d}",
                    "parent_node_id": parent.node_id,
                    "parent_label": parent.canonical_label,
                    "parent_type": parent.node_type,
                    "child_node_id": child.node_id,
                    "child_label": child.canonical_label,
                    "child_type": child.node_type,
                    "allowed_relation": relation,
                    "semantic_similarity": score,
                    "candidate_rank": rank,
                })
    return pd.DataFrame(rows)


def relation_prompt(pair: pd.Series, parent: pd.Series, child: pd.Series) -> str:
    same_type = pair.allowed_relation == "is_a"
    allowed = sorted(SAME_TYPE_DECISIONS if same_type else CROSS_TYPE_DECISIONS)
    tests = (
        "Apply kind-of, definition entailment, generality, substitutability, equivalence exclusion, and association exclusion tests."
        if same_type
        else "Apply composition, major-functional-division, containment, type-direction, external-use exclusion, and association exclusion tests."
    )
    return f"""Classify one candidate ontology relationship.

PARENT CANDIDATE:
- ID: {parent.node_id}
- Label: {parent.canonical_label}
- Type: {parent.node_type}
- Aliases: {parent.aliases}
- Description: {parent.entity_description}
- Evidence: {parent.evidence_text}

CHILD CANDIDATE:
- ID: {child.node_id}
- Label: {child.canonical_label}
- Type: {child.node_type}
- Aliases: {child.aliases}
- Description: {child.entity_description}
- Evidence: {child.evidence_text}

ALLOWED DECISIONS: {json.dumps(allowed)}
EXPECTED POSITIVE RELATION: {pair.allowed_relation}

Definitions:
- is_a: the child is a more specific kind of the parent; every child instance satisfies the parent concept.
- has_subsystem: the parent System contains the child Subsystem as a major functional or structural division.
- equivalent: same concept, alias, acronym, or true synonym; not a hierarchy edge.
- related_non_hierarchical: meaningful association without subtype or composition entailment.
- no_relation: insufficient relationship.
- uncertain: evidence cannot decide reliably.

{tests}
Similarity is candidate retrieval only and never proves a relation. Do not invent evidence.

Return JSON:
{{
  "pair_id": {json.dumps(pair.pair_id)},
  "parent_node_id": {json.dumps(parent.node_id)},
  "child_node_id": {json.dumps(child.node_id)},
  "predicted_relation": "allowed decision",
  "confidence": 0.0,
  "direction_confidence": 0.0,
  "evidence_supported": true,
  "supporting_sentence_ids": ["IDs visible in evidence"],
  "semantic_test_result": "result of required tests",
  "reason": "concise evidence-based rationale",
  "alternative_relation": "allowed decision",
  "alternative_confidence": 0.0
}}"""


def validate_relation(obj: dict[str, Any], pair: pd.Series) -> dict[str, Any]:
    allowed = SAME_TYPE_DECISIONS if pair.allowed_relation == "is_a" else CROSS_TYPE_DECISIONS
    relation = safe_text(obj.get("predicted_relation"))
    if relation not in allowed:
        raise ValueError("Invalid relation")
    if safe_text(obj.get("parent_node_id")) != pair.parent_node_id or safe_text(obj.get("child_node_id")) != pair.child_node_id:
        raise ValueError("Pair IDs changed")
    alternative = safe_text(obj.get("alternative_relation"))
    if alternative not in allowed:
        alternative = "uncertain"
    return {
        "pair_id": pair.pair_id,
        "parent_node_id": pair.parent_node_id,
        "child_node_id": pair.child_node_id,
        "predicted_relation": relation,
        "confidence": validate_confidence(obj.get("confidence")),
        "direction_confidence": validate_confidence(obj.get("direction_confidence")),
        "evidence_supported": bool(obj.get("evidence_supported", False)),
        "supporting_sentence_ids": [safe_text(x) for x in obj.get("supporting_sentence_ids", []) if safe_text(x)],
        "semantic_test_result": safe_text(obj.get("semantic_test_result")),
        "reason": safe_text(obj.get("reason")),
        "alternative_relation": alternative,
        "alternative_confidence": validate_confidence(obj.get("alternative_confidence")),
    }


def critic_prompt(pair: pd.Series, primary: dict[str, Any], parent: pd.Series, child: pd.Series) -> str:
    return f"""Independently audit a proposed positive ontology edge.

PARENT: {parent.canonical_label} ({parent.node_type})
CHILD: {child.canonical_label} ({child.node_type})
ALLOWED POSITIVE RELATION: {pair.allowed_relation}
PRIMARY DECISION: {json.dumps(primary, ensure_ascii=False, indent=2)}
PARENT EVIDENCE: {parent.evidence_text}
CHILD EVIDENCE: {child.evidence_text}

Check semantic entailment, evidence sufficiency, type compatibility, and direction.
For is_a, reverse_direction is allowed if the child/parent direction is wrong.
For has_subsystem, direction must remain System -> Subsystem; otherwise reject or manual_review.

Return JSON:
{{
  "decision": "accept|reject|reverse_direction|change_relation|manual_review",
  "final_relation": "is_a|has_subsystem|no_relation|uncertain",
  "confidence": 0.0,
  "evidence_supported": true,
  "supporting_sentence_ids": ["IDs"],
  "reason": "audit rationale"
}}"""


def validate_critic(obj: dict[str, Any], pair: pd.Series) -> dict[str, Any]:
    decision = safe_text(obj.get("decision"))
    if decision not in CRITIC_DECISIONS:
        raise ValueError("Invalid critic decision")
    relation = safe_text(obj.get("final_relation"))
    if relation not in {pair.allowed_relation, "no_relation", "uncertain"}:
        raise ValueError("Invalid final relation")
    if pair.allowed_relation == "has_subsystem" and decision == "reverse_direction":
        decision, relation = "manual_review", "uncertain"
    return {
        "decision": decision,
        "final_relation": relation,
        "confidence": validate_confidence(obj.get("confidence")),
        "evidence_supported": bool(obj.get("evidence_supported", False)),
        "supporting_sentence_ids": [safe_text(x) for x in obj.get("supporting_sentence_ids", []) if safe_text(x)],
        "reason": safe_text(obj.get("reason")),
    }


def process_pairs(
    pairs: pd.DataFrame,
    nodes: pd.DataFrame,
    client: OllamaClient,
    relation_threshold: float,
    critic_all_positive: bool,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_map = {row.node_id: row for _, row in nodes.iterrows()}
    partial_path = output_dir / "relation_results.partial.jsonl"
    completed = {}
    if partial_path.is_file():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[item["pair_id"]] = item

    relation_rows, critic_rows = [], []
    for number, (_, pair) in enumerate(pairs.iterrows(), start=1):
        if pair.pair_id in completed:
            record = completed[pair.pair_id]
            relation_rows.append(record)
            continue
        parent, child = node_map[pair.parent_node_id], node_map[pair.child_node_id]
        print(f"[{number}/{len(pairs)}] {parent.canonical_label} -> {child.canonical_label} ({pair.allowed_relation})")
        try:
            primary, _ = client.chat_json(
                [
                    {"role": "system", "content": (
                        "You classify ontology relationships using semantic entailment and evidence. "
                        "predicted_relation must be exactly one of the allowed values in the user prompt. "
                        "Return JSON only."
                    )},
                    {"role": "user", "content": relation_prompt(pair, parent, child)},
                ],
                stage=f"relation::{pair.pair_id}",
                validator=lambda obj, p=pair: validate_relation(obj, p),
            )
        except RuntimeError as relation_error:
            # One malformed or out-of-schema Qwen response must not abort an
            # 860-pair taxonomy run. Preserve the pair as uncertain so it is
            # excluded from accepted edges and remains available for review.
            print(f"Relation Agent failed validation for {pair.pair_id}; using uncertain fallback: {relation_error}")
            primary = {
                "pair_id": pair.pair_id,
                "parent_node_id": pair.parent_node_id,
                "child_node_id": pair.child_node_id,
                "predicted_relation": "uncertain",
                "confidence": 0.0,
                "direction_confidence": 0.0,
                "evidence_supported": False,
                "supporting_sentence_ids": [],
                "semantic_test_result": "agent_validation_failed",
                "reason": "Relation Agent validation failed; conservative uncertain fallback used. " + str(relation_error),
                "alternative_relation": "uncertain",
                "alternative_confidence": 0.0,
            }
        positive = primary["predicted_relation"] == pair.allowed_relation and primary["confidence"] >= relation_threshold and primary["evidence_supported"]
        critic = None
        if primary["predicted_relation"] == pair.allowed_relation and (critic_all_positive or not positive):
            try:
                critic, _ = client.chat_json(
                    [
                        {"role": "system", "content": "You are a conservative ontology-relation critic. Return JSON only."},
                        {"role": "user", "content": critic_prompt(pair, primary, parent, child)},
                    ],
                    stage=f"critic::{pair.pair_id}",
                    validator=lambda obj, p=pair: validate_critic(obj, p),
                )
                critic_rows.append({"pair_id": pair.pair_id, **critic})
            except RuntimeError as critic_error:
                print(f"Critic Agent failed validation for {pair.pair_id}; keeping primary decision: {critic_error}")
                critic = None

        if critic:
            accepted = critic["decision"] in {"accept", "reverse_direction"} and critic["final_relation"] == pair.allowed_relation
            final_relation = pair.allowed_relation if accepted else critic["final_relation"]
            final_confidence = critic["confidence"]
            review_status = "accepted" if accepted else ("manual_review" if critic["decision"] == "manual_review" else "rejected")
            reverse = critic["decision"] == "reverse_direction"
        else:
            final_relation = primary["predicted_relation"]
            final_confidence = primary["confidence"]
            review_status = "accepted" if positive else ("manual_review" if final_relation == "uncertain" else "rejected")
            reverse = False

        final_parent = child.node_id if reverse else parent.node_id
        final_child = parent.node_id if reverse else child.node_id
        record = {
            **pair.to_dict(),
            **{f"primary_{k}": v for k, v in primary.items() if k not in {"pair_id", "parent_node_id", "child_node_id"}},
            "final_parent_node_id": final_parent,
            "final_child_node_id": final_child,
            "final_relation": final_relation,
            "final_confidence": final_confidence,
            "review_status": review_status,
            "critic_invoked": critic is not None,
        }
        relation_rows.append(record)
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return pd.DataFrame(relation_rows), pd.DataFrame(critic_rows)


def remove_is_a_cycles(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted, removed = [], []
    adjacency: dict[str, set[str]] = defaultdict(set)

    def reaches(start: str, target: str) -> bool:
        stack, visited = [start], set()
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency.get(current, set()) - visited)
        return False

    ordered = edges.sort_values("final_confidence", ascending=False)
    for _, edge in ordered.iterrows():
        parent, child = edge.final_parent_node_id, edge.final_child_node_id
        if parent == child or reaches(child, parent):
            removed.append({**edge.to_dict(), "validation_issue": "self_loop_or_cycle"})
        else:
            adjacency[parent].add(child)
            accepted.append(edge.to_dict())
    return pd.DataFrame(accepted), pd.DataFrame(removed)


def transitive_reduction(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if edges.empty:
        return edges.copy(), pd.DataFrame()
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in edges.itertuples(index=False):
        adjacency[row.final_parent_node_id].add(row.final_child_node_id)

    keep, redundant = [], []
    for _, edge in edges.iterrows():
        parent, child = edge.final_parent_node_id, edge.final_child_node_id
        adjacency[parent].discard(child)
        stack, visited, found = [parent], set(), False
        while stack:
            current = stack.pop()
            if current == child:
                found = True
                break
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency.get(current, set()) - visited)
        adjacency[parent].add(child)
        if found:
            redundant.append({**edge.to_dict(), "validation_issue": "transitively_redundant"})
        else:
            keep.append(edge.to_dict())
    return pd.DataFrame(keep), pd.DataFrame(redundant)


def select_nearest_parents(is_a: pd.DataFrame, max_parents: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if is_a.empty:
        return is_a.copy(), pd.DataFrame()
    kept, excess = [], []
    for _, group in is_a.groupby("final_child_node_id"):
        ordered = group.sort_values(["final_confidence", "semantic_similarity"], ascending=False)
        kept.extend(ordered.head(max_parents).to_dict("records"))
        for row in ordered.iloc[max_parents:].to_dict("records"):
            excess.append({**row, "validation_issue": "lower_ranked_additional_parent"})
    return pd.DataFrame(kept), pd.DataFrame(excess)


def validate_graph(relations: pd.DataFrame, nodes: pd.DataFrame, max_is_a_parents: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_types = dict(zip(nodes.node_id, nodes.node_type))
    accepted = relations[relations.review_status.eq("accepted") & relations.final_relation.isin(POSITIVE_RELATIONS)].copy()
    valid_rows, invalid_rows = [], []
    seen = set()
    for _, edge in accepted.iterrows():
        parent, child, relation = edge.final_parent_node_id, edge.final_child_node_id, edge.final_relation
        parent_type, child_type = node_types.get(parent), node_types.get(child)
        valid_combo = allowed_relation(parent_type, child_type) == relation
        signature = (parent, child, relation)
        if parent == child:
            invalid_rows.append({**edge.to_dict(), "validation_issue": "self_loop"})
        elif not valid_combo:
            invalid_rows.append({**edge.to_dict(), "validation_issue": "invalid_type_combination"})
        elif signature in seen:
            invalid_rows.append({**edge.to_dict(), "validation_issue": "duplicate_edge"})
        else:
            seen.add(signature)
            valid_rows.append(edge.to_dict())
    valid = pd.DataFrame(valid_rows)
    invalid = pd.DataFrame(invalid_rows)
    if valid.empty:
        return valid, invalid

    is_a = valid[valid.final_relation.eq("is_a")].copy()
    composition = valid[valid.final_relation.eq("has_subsystem")].copy()
    is_a, cycle_removed = remove_is_a_cycles(is_a)
    is_a, parent_removed = select_nearest_parents(is_a, max_is_a_parents)
    is_a, transitive_removed = transitive_reduction(is_a)
    removed = pd.concat([invalid, cycle_removed, parent_removed, transitive_removed], ignore_index=True)
    final = pd.concat([is_a, composition], ignore_index=True)
    return final, removed


def load_gt_relations(path: Path, sheet: str, aliases: pd.DataFrame) -> pd.DataFrame:
    gt = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    required = {"parent", "child", "relation"}
    if not required.issubset(gt.columns):
        raise ValueError(f"GT relation sheet missing {sorted(required - set(gt.columns))}")
    if "grounded_status_v2" in gt.columns:
        gt = gt[gt.grounded_status_v2.fillna("").astype(str).str.casefold().ne("needs_manual_review")].copy()
    gt["parent_norm"] = gt.parent.map(normalize)
    gt["child_norm"] = gt.child.map(normalize)
    gt["relation_norm"] = gt.relation.map(lambda x: normalize(x).replace(" ", "_"))
    relation_aliases = {
        "isa": "is_a", "is_a": "is_a", "subclass_of": "is_a", "kind_of": "is_a",
        "has_subsystem": "has_subsystem", "contains_subsystem": "has_subsystem",
        "part_of_system": "has_subsystem",
    }
    gt["relation_eval"] = gt.relation_norm.map(lambda x: relation_aliases.get(x, x))
    return gt


def evaluate_edges(edges: pd.DataFrame, nodes: pd.DataFrame, gt: pd.DataFrame, alias_map: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    node_labels = dict(zip(nodes.node_id, nodes.canonical_label))
    surface_to_canonical = {normalize(r.surface_form): normalize(r.canonical_label) for r in alias_map.itertuples(index=False)}

    def canonical(value: str) -> str:
        norm = normalize(value)
        return surface_to_canonical.get(norm, norm)

    predicted = set()
    for row in edges.itertuples(index=False):
        predicted.add((canonical(node_labels[row.final_parent_node_id]), canonical(node_labels[row.final_child_node_id]), row.final_relation))
    gold = {(canonical(r.parent), canonical(r.child), r.relation_eval) for r in gt.itertuples(index=False) if r.relation_eval in POSITIVE_RELATIONS}
    overlap = predicted & gold
    precision = len(overlap) / len(predicted) if predicted else 0.0
    recall = len(overlap) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "predicted_edges": len(predicted),
        "gt_edges": len(gold),
        "matched_edges": len(overlap),
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "25_taxonomy_metrics.csv", index=False)
    pd.DataFrame(sorted(predicted - gold), columns=["parent", "child", "relation"]).to_csv(output_dir / "27_incorrect_or_uncovered_edges.csv", index=False)
    pd.DataFrame(sorted(gold - predicted), columns=["parent", "child", "relation"]).to_csv(output_dir / "28_missing_gt_edges.csv", index=False)
    return metrics


def graph_quality(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, Any]:
    parents = set(edges.final_parent_node_id) if not edges.empty else set()
    children = set(edges.final_child_node_id) if not edges.empty else set()
    connected = parents | children
    roots = parents - children
    orphans = set(nodes.node_id) - connected
    is_a = edges[edges.final_relation.eq("is_a")] if not edges.empty else pd.DataFrame()
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "system_nodes": int(nodes.node_type.eq("System").sum()),
        "subsystem_nodes": int(nodes.node_type.eq("Subsystem").sum()),
        "system_is_a_edges": int(((edges.final_relation == "is_a") & edges.final_parent_node_id.map(dict(zip(nodes.node_id, nodes.node_type))).eq("System")).sum()) if not edges.empty else 0,
        "subsystem_is_a_edges": int(((edges.final_relation == "is_a") & edges.final_parent_node_id.map(dict(zip(nodes.node_id, nodes.node_type))).eq("Subsystem")).sum()) if not edges.empty else 0,
        "has_subsystem_edges": int(edges.final_relation.eq("has_subsystem").sum()) if not edges.empty else 0,
        "roots": len(roots),
        "orphans": len(orphans),
        "orphan_rate": len(orphans) / len(nodes) if len(nodes) else 0.0,
    }


def export_graphml(nodes: pd.DataFrame, edges: pd.DataFrame, path: Path) -> None:
    try:
        import networkx as nx
    except ImportError:
        return
    graph = nx.DiGraph()
    for row in nodes.itertuples(index=False):
        graph.add_node(row.node_id, label=row.canonical_label, entity_type=row.node_type, confidence=float(row.node_confidence))
    for row in edges.itertuples(index=False):
        graph.add_edge(row.final_parent_node_id, row.final_child_node_id, relation=row.final_relation, confidence=float(row.final_confidence))
    nx.write_graphml(graph, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typing-results", type=Path, default=DEFAULT_TYPING_RESULTS)
    parser.add_argument("--typing-contexts", type=Path, default=DEFAULT_CONTEXTS)
    parser.add_argument("--alias-workbook", type=Path, default=DEFAULT_ALIAS_GT)
    parser.add_argument("--alias-sheet", default="S2_GT_ALIASES")
    parser.add_argument("--relation-gt-sheet", default="S5_GT_RELATIONS")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--typing-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--evidence-quality-threshold", type=float, default=0.50)
    parser.add_argument("--relation-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--max-is-a-parents", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--skip-ollama-check", action="store_true")
    parser.add_argument("--disable-node-agent", action="store_true")
    parser.add_argument("--critic-positive-only-when-uncertain", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    for path in (args.typing_results, args.alias_workbook):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = OllamaClient(args.model, args.ollama_url, args.output_dir, args.timeout, args.retries, args.temperature)
    if not args.skip_ollama_check:
        client.check()

    print("Loading entity-typing results, aliases, and evidence...")
    typed = load_typing_results(args.typing_results)
    aliases = load_aliases(args.alias_workbook, args.alias_sheet)
    contexts = load_contexts(args.typing_contexts)
    typed.to_csv(args.output_dir / "11_system_subsystem_inventory.csv", index=False)

    print("Building canonical taxonomy nodes...")
    nodes, rejected, alias_map = build_nodes(
        typed, aliases, contexts, client,
        args.typing_confidence_threshold,
        args.evidence_quality_threshold,
        not args.disable_node_agent,
    )
    if args.max_nodes > 0:
        nodes = nodes.head(args.max_nodes).copy()
        alias_map = alias_map[alias_map.node_id.isin(nodes.node_id)].copy()
    nodes.to_csv(args.output_dir / "12_canonical_taxonomy_nodes.csv", index=False)
    alias_map.to_csv(args.output_dir / "13_alias_to_node_mapping.csv", index=False)
    rejected.to_csv(args.output_dir / "14_rejected_or_review_nodes.csv", index=False)
    if len(nodes) < 2:
        raise RuntimeError("Fewer than two accepted taxonomy nodes")

    print("Encoding nodes and retrieving candidate parent pairs...")
    model = load_embedding_model(args.embedding_model)
    texts = [node_embedding_text(row) for _, row in nodes.iterrows()]
    embeddings = encode_cached(model, texts, args.output_dir / "node_embeddings.npy", args.embedding_batch_size)
    pairs = generate_candidate_pairs(nodes.reset_index(drop=True), embeddings, args.top_k)
    if args.max_pairs > 0:
        pairs = pairs.head(args.max_pairs).copy()
    pairs.to_csv(args.output_dir / "15_candidate_relation_pairs.csv", index=False)

    print("Running Relation Agent and Relation Critic term-pair by term-pair...")
    relations, critics = process_pairs(
        pairs, nodes, client,
        args.relation_confidence_threshold,
        not args.critic_positive_only_when_uncertain,
        args.output_dir,
    )
    relations.to_csv(args.output_dir / "16_relation_agent_results.csv", index=False)
    critics.to_csv(args.output_dir / "17_relation_critic_results.csv", index=False)

    print("Applying deterministic schema and graph validation...")
    final_edges, removed = validate_graph(relations, nodes, args.max_is_a_parents)
    node_type_map = dict(zip(nodes.node_id, nodes.node_type))
    system_is_a = final_edges[(final_edges.final_relation.eq("is_a")) & final_edges.final_parent_node_id.map(node_type_map).eq("System")]
    subsystem_is_a = final_edges[(final_edges.final_relation.eq("is_a")) & final_edges.final_parent_node_id.map(node_type_map).eq("Subsystem")]
    has_subsystem = final_edges[final_edges.final_relation.eq("has_subsystem")]
    system_is_a.to_csv(args.output_dir / "18_system_is_a_edges.csv", index=False)
    subsystem_is_a.to_csv(args.output_dir / "19_subsystem_is_a_edges.csv", index=False)
    has_subsystem.to_csv(args.output_dir / "20_has_subsystem_edges.csv", index=False)
    nodes.to_csv(args.output_dir / "21_taxonomy_nodes.csv", index=False)
    final_edges.to_csv(args.output_dir / "22_taxonomy_edges.csv", index=False)
    removed.to_csv(args.output_dir / "23_removed_or_review_edges.csv", index=False)
    export_graphml(nodes, final_edges, args.output_dir / "24_taxonomy.graphml")

    quality = graph_quality(nodes, final_edges)
    pd.DataFrame([quality]).to_csv(args.output_dir / "26_graph_quality_metrics.csv", index=False)

    try:
        gt = load_gt_relations(args.alias_workbook, args.relation_gt_sheet, aliases)
        metrics = evaluate_edges(final_edges, nodes, gt, alias_map, args.output_dir)
    except Exception as exc:
        metrics = {"evaluation_error": str(exc)}
        json_dump(args.output_dir / "25_taxonomy_metrics.json", metrics)

    manifest = {
        "typing_results": str(args.typing_results),
        "typing_contexts": str(args.typing_contexts),
        "alias_workbook": str(args.alias_workbook),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "allowed_edges": [
            "System --is_a--> System",
            "Subsystem --is_a--> Subsystem",
            "System --has_subsystem--> Subsystem",
        ],
        "nodes": len(nodes),
        "candidate_pairs": len(pairs),
        "final_edges": len(final_edges),
        "live_qwen_calls": client.live_calls,
        "qwen_cache_hits": client.cache_hits,
        "runtime_seconds": time.perf_counter() - started,
        "quality": quality,
        "evaluation": metrics,
    }
    json_dump(args.output_dir / "run_manifest.json", manifest)

    print("=" * 100)
    print("Multi-agent System/Subsystem taxonomy completed")
    print("=" * 100)
    for key, value in quality.items():
        print(f"{key:35s}: {value}")
    for key, value in metrics.items():
        print(f"{key:35s}: {value}")
    print(f"Live Qwen calls: {client.live_calls}; cache hits: {client.cache_hits}")
    print(f"Runtime: {manifest['runtime_seconds']:.2f}s")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
