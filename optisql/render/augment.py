from __future__ import annotations

from dataclasses import dataclass
import random

from optisql.render.html_templates import RenderStyle


@dataclass
class AugmentConfig:
    transpose_prob: float = 0.3
    style_pool_size: int = 4


def build_style_pool(base: RenderStyle, size: int) -> list[RenderStyle]:
    pool = []
    for i in range(size):
        pool.append(
            RenderStyle(
                font_family=random.choice(["Arial", "Times New Roman", "Calibri"]),
                font_size=random.choice([14, 16, 18]),
                header_bg=random.choice(["#f2f2f2", "#e6f7ff", "#fffbe6"]),
                stripe_bg=random.choice(["#fafafa", "#ffffff", "#f7f7f7"]),
                border_color=random.choice(["#333", "#555", "#111"]),
                padding_px=random.choice([4, 6, 8]),
                header_bold=random.choice([True, False]),
            )
        )
    return pool


def should_transpose(config: AugmentConfig) -> bool:
    return random.random() < config.transpose_prob


def pick_style(style_pool: list[RenderStyle]) -> RenderStyle:
    if not style_pool:
        raise ValueError("Style pool is empty.")
    return random.choice(style_pool)
