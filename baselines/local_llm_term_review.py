from __future__ import annotations

import requests

from performance import ollama_tokens_per_sec
from term_review_prompts import REVIEW_SYSTEM_PROMPT, build_review_prompt

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

VALID_LABELS = {
    "system",
    "part",
    "mission",
    "capability",
    "parameter",
    "environment",
    "domain",
    "unknown",
}


def call_ollama_term_review(
    model: str,
    term: str,
    initial_label: str,
    context_text: str,
    timeout_sec: int = 600,
    temperature: float = 0.0,
    keep_alive: str = "5m",
    num_ctx: int = 8192,
) -> tuple[dict, dict, str]:
    """
    Send one term-review request to Ollama and return:
      - review_row: {
            "term": ...,
            "initial_label": ...,
            "reviewed_label": ...,
            "confidence_score": ...
        }
      - usage: Ollama timing/token stats
      - raw_text: raw model output
    """
    prompt = build_review_prompt(
        term=term,
        initial_label=initial_label,
        context_text=context_text,
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
        "keep_alive": keep_alive,
    }

    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout_sec)
    resp.raise_for_status()

    obj = resp.json()
    content = obj.get("message", {}).get("content", "")

    review_row = parse_review_output(
        raw_text=content,
        fallback_term=term,
        fallback_initial_label=initial_label,
    )

    usage = {
        "model": obj.get("model", model),
        "total_duration_ns": obj.get("total_duration"),
        "load_duration_ns": obj.get("load_duration"),
        "prompt_eval_count": obj.get("prompt_eval_count"),
        "prompt_eval_duration_ns": obj.get("prompt_eval_duration"),
        "eval_count": obj.get("eval_count"),
        "eval_duration_ns": obj.get("eval_duration"),
        "generated_tokens_per_sec": ollama_tokens_per_sec(
            obj.get("eval_count"),
            obj.get("eval_duration"),
        ),
    }

    return review_row, usage, content


def parse_review_output(
    raw_text: str,
    fallback_term: str,
    fallback_initial_label: str,
) -> dict:
    """
    Expected format:
    term<TAB>initial_label<TAB>reviewed_label<TAB>confidence_score

    If parsing fails, fall back to:
      term = fallback_term
      initial_label = fallback_initial_label
      reviewed_label = fallback_initial_label
      confidence_score = 0
    """
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) != 4:
            continue

        term = parts[0].strip()
        initial_label = parts[1].strip().lower()
        reviewed_label = parts[2].strip().lower()
        confidence_raw = parts[3].strip()

        if reviewed_label not in VALID_LABELS:
            continue

        try:
            confidence_score = int(float(confidence_raw))
        except ValueError:
            continue

        confidence_score = max(0, min(100, confidence_score))

        return {
            "term": term if term else fallback_term,
            "initial_label": initial_label if initial_label else fallback_initial_label,
            "reviewed_label": reviewed_label,
            "confidence_score": confidence_score,
        }

    # fallback if parsing fails
    return {
        "term": fallback_term,
        "initial_label": fallback_initial_label,
        "reviewed_label": fallback_initial_label,
        "confidence_score": 0,
    }
