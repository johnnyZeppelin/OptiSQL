from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from PIL import Image, ImageDraw


@dataclass
class RobustnessResult:
    exacc_clean: float
    exacc_perturbed: float

    @property
    def drop(self) -> float:
        return self.exacc_clean - self.exacc_perturbed


def apply_header_mask(image: Image.Image, header_bbox: Tuple[float, float, float, float], ratio: float) -> Image.Image:
    ratio = max(0.0, min(ratio, 1.0 / 3.0))
    x, y, w, h = header_bbox
    mask_width = w * ratio
    if mask_width <= 1:
        return image
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    draw.rectangle([x, y, x + mask_width, y + h], fill=(240, 240, 240))
    return masked
