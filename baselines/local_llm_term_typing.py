from __future__ import annotations

import requests

from performance import ollama_tokens_per_sec
from term_typing_prompts import SYSTEM_PROMPT, build_typing_prompt

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


def call_ollama_term_typing_batch(
    model: str,
    batch_rows: list[dict],
    timeout_sec: int = 600,
    temperature: float = 0.0,
    keep_alive: str = "5m",
    num_ctx: int = 8192,
) -> tuple[list[dict], dict, str]:
    """
    Send a batch of term+context rows to Ollama and return:
      - predictions: [{'id', 'term', 'predicted_label'}]
      - usage: Ollama timing/token stats
      - raw_text: raw model output

    batch_rows format:
    [
        {
            "id": ...,
            "term": ...,
            "normalized_term": ...,
            "context": ...
        },
        ...
    ]
    """
    prompt = build_typing_prompt(batch_rows)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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

    preds = parse_tsv_predictions(content, batch_rows)

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

    return preds, usage, content


def parse_tsv_predictions(raw_text: str, batch_rows: list[dict]) -> list[dict]:
    """
    Parse model output in the exact format:

    id<TAB>term<TAB>label

    Returns:
        [
            {"id": ..., "term": ..., "predicted_label": ...},
            ...
        ]

    Any missing batch items are backfilled with label='unknown'
    to keep evaluation deterministic.
    """
    by_id = {str(row["id"]): row for row in batch_rows}
    out = []
    seen_ids = set()

    for line in raw_text.splitlines():
        line = line.strip()

        if not line:
            continue

        # skip accidental header line
        if line.lower().startswith("id\t"):
            continue

        parts = line.split("\t")
        if len(parts) != 3:
            continue

        row_id = parts[0].strip()
        term = parts[1].strip()
        label = parts[2].strip().lower()

        if row_id not in by_id:
            continue

        if label not in VALID_LABELS:
            continue

        out.append(
            {
                "id": row_id,
                "term": term,
                "predicted_label": label,
            }
        )
        seen_ids.add(row_id)

    # backfill missing rows with unknown
    for row in batch_rows:
        row_id = str(row["id"])
        if row_id not in seen_ids:
            out.append(
                {
                    "id": row_id,
                    "term": row["term"],
                    "predicted_label": "unknown",
                }
            )

    # preserve original batch order
    order = {str(r["id"]): i for i, r in enumerate(batch_rows)}
    out.sort(key=lambda x: order[x["id"]])

    return out