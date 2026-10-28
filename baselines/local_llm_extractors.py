# src/local_llm_extractors.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import requests

from llm_prompts import TERM_EXTRACTION_SYSTEM_PROMPT, build_term_extraction_user_prompt
from normalization import normalize_term
from performance import ollama_tokens_per_sec

OLLAMA_URL = "http://localhost:11434/api/chat"


def _clean_llm_terms_from_lines(content: str, method_name: str) -> List[dict]:
    rows = []
    seen = set()

    for line in content.splitlines():
        term = line.strip()
        term = term.lstrip("-•*").strip()

        if len(term) > 2 and term[0].isdigit():
            parts = term.split(".", 1)
            if len(parts) == 2 and parts[0].isdigit():
                term = parts[1].strip()

        if not term:
            continue

        norm = normalize_term(term)
        if not norm or norm in seen:
            continue

        seen.add(norm)
        rows.append({
            "term": term,
            "score": 1.0,
            "method": method_name,
            "normalized_term": norm,
        })

    return rows


def _chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """
    Split large text into chunks by paragraph boundaries where possible.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []

    current_len = 0
    for para in paras:
        para_len = len(para) + 2
        if current and current_len + para_len > max_chars:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _call_single_chunk(
    model: str,
    text: str,
    temperature: float = 0.0,
    keep_alive: str = "5m",
    timeout_sec: int = 900,
    num_ctx: int = 8192,
) -> Tuple[List[dict], Dict[str, Any], str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TERM_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": build_term_extraction_user_prompt(text)},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
        "keep_alive": keep_alive,
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    obj = resp.json()

    content = obj.get("message", {}).get("content", "")
    method_name = f"llm_{model.replace(':', '_').replace('/', '_')}"
    rows = _clean_llm_terms_from_lines(content, method_name)

    usage = {
        "model": obj.get("model", model),
        "total_duration_ns": obj.get("total_duration"),
        "load_duration_ns": obj.get("load_duration"),
        "prompt_eval_count": obj.get("prompt_eval_count"),
        "prompt_eval_duration_ns": obj.get("prompt_eval_duration"),
        "eval_count": obj.get("eval_count"),
        "eval_duration_ns": obj.get("eval_duration"),
        "generated_tokens_per_sec": ollama_tokens_per_sec(
            obj.get("eval_count"), obj.get("eval_duration")
        ),
    }

    return rows, usage, content


def call_ollama_extract_terms_plain(
    model: str,
    text: str,
    temperature: float = 0.0,
    keep_alive: str = "5m",
    timeout_sec: int = 900,
    num_ctx: int = 8192,
    chunk_chars: int = 4000,
):
    """
    Chunk the corpus, run extraction per chunk, then merge+dedupe all terms.
    """
    method_name = f"llm_{model.replace(':', '_').replace('/', '_')}"
    chunks = _chunk_text(text, max_chars=chunk_chars)

    all_rows = []
    total_usage = {
        "model": model,
        "total_duration_ns": 0,
        "load_duration_ns": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration_ns": 0,
        "eval_count": 0,
        "eval_duration_ns": 0,
        "generated_tokens_per_sec": None,
    }
    raw_outputs = []

    seen = set()

    for idx, chunk in enumerate(chunks, start=1):
        print(f"  -> chunk {idx}/{len(chunks)} | chars={len(chunk)}")

        rows, usage, raw_text = _call_single_chunk(
            model=model,
            text=chunk,
            temperature=temperature,
            keep_alive=keep_alive,
            timeout_sec=timeout_sec,
            num_ctx=num_ctx,
        )

        raw_outputs.append(raw_text)

        for row in rows:
            norm = row["normalized_term"]
            if norm not in seen:
                seen.add(norm)
                all_rows.append(row)

        for key in [
            "total_duration_ns",
            "load_duration_ns",
            "prompt_eval_count",
            "prompt_eval_duration_ns",
            "eval_count",
            "eval_duration_ns",
        ]:
            val = usage.get(key)
            if isinstance(val, int):
                total_usage[key] += val

    if total_usage["eval_count"] and total_usage["eval_duration_ns"]:
        total_usage["generated_tokens_per_sec"] = ollama_tokens_per_sec(
            total_usage["eval_count"],
            total_usage["eval_duration_ns"],
        )

    return all_rows, total_usage, "\n\n".join(raw_outputs)