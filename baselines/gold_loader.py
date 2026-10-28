import json
from pathlib import Path
from typing import List, Set, Dict

from normalization import normalize_term


def load_gold_json(gold_path: Path) -> List[dict]:
    """
    Load the JSON list of objects with keys:
    - term
    - aliases
    """
    with gold_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Gold JSON must be a list of objects with keys: term, aliases")

    return data


def build_gold_term_set(gold_items: List[dict]) -> Set[str]:
    """
    Flatten canonical terms + aliases into one normalized gold vocabulary set.
    """
    gold_terms = set()

    for item in gold_items:
        term = str(item.get("term", "")).strip()
        aliases = item.get("aliases", []) or []

        if term:
            gold_terms.add(normalize_term(term))

        for alias in aliases:
            alias = str(alias).strip()
            if alias:
                gold_terms.add(normalize_term(alias))

    return gold_terms


def build_gold_corpus_text(gold_items: List[dict]) -> str:
    """
    Create one flat gold corpus text from all terms and aliases.
    Useful for inspection/debugging.
    """
    all_terms = []

    for item in gold_items:
        term = str(item.get("term", "")).strip()
        aliases = item.get("aliases", []) or []

        if term:
            all_terms.append(term)

        for alias in aliases:
            alias = str(alias).strip()
            if alias:
                all_terms.append(alias)

    return "\n".join(all_terms)