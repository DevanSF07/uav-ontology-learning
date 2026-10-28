
from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Any, Callable, Dict, Optional


@dataclass
class RuntimeMetrics:
    method: str
    wall_clock_sec: float
    n_pred_unique: int
    # Optional Ollama-native timings (nanoseconds) / counts
    total_duration_ns: Optional[int] = None
    load_duration_ns: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration_ns: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration_ns: Optional[int] = None
    generated_tokens_per_sec: Optional[float] = None

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


def timed_call(func: Callable[..., Any], *args, **kwargs):
    start = perf_counter()
    result = func(*args, **kwargs)
    elapsed = perf_counter() - start
    return result, elapsed


def ollama_tokens_per_sec(eval_count: Optional[int], eval_duration_ns: Optional[int]) -> Optional[float]:
    if eval_count is None or eval_duration_ns is None or eval_duration_ns <= 0:
        return None
    return round(eval_count / (eval_duration_ns / 1_000_000_000), 4)
