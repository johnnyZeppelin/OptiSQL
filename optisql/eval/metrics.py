from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class MetricBundle:
    exacc: float
    ex_can: float
    tsr: float


def compute_tsr(text_token_counts: Iterable[int], optical_token_counts: Iterable[int]) -> float:
    text = list(text_token_counts)
    optical = list(optical_token_counts)
    if not text or not optical:
        return 0.0
    return sum(text) / sum(optical)
