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


def linearize_table(headers: list[str], rows: list[list[str]], types: list[str] | None = None) -> str:
    header_chunks = []
    for idx, header in enumerate(headers):
        col_type = types[idx] if types and idx < len(types) else ""
        header_chunks.append(f"{header}({col_type})" if col_type else header)
    header_line = " | ".join(header_chunks)
    row_lines = []
    for ridx, row in enumerate(rows, start=1):
        row_lines.append(f"r{ridx}: " + " | ".join(row))
    return "COLUMNS: " + header_line + " ROWS: " + " ".join(row_lines)
