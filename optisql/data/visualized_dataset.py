from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from optisql.utils.io import read_jsonl


@dataclass
class VisualizedSample:
    image: Image.Image
    question: str
    sql: str
    db_path: Path
    render_meta: dict[str, Any]


class VisualizedDataset:
    def __init__(
        self,
        manifest_path: Path,
        images_root: Path,
        p_style: float = 1.0,
        p_transpose: float = 0.3,
    ) -> None:
        self.manifest_path = manifest_path
        self.images_root = images_root
        self.p_style = p_style
        self.p_transpose = p_transpose
        self.samples = read_jsonl(manifest_path)

    def __len__(self) -> int:
        return len(self.samples)

    def _select_image(self, item: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        images = item.get("images", {})
        base = images.get("base")
        styles = images.get("styles", [])
        transpose = images.get("transpose", {})
        use_transpose = self.p_transpose > 0 and transpose and random.random() < self.p_transpose
        if use_transpose:
            base = transpose.get("base", base)
            styles = transpose.get("styles", styles)
        if styles and self.p_style >= 1.0:
            chosen = random.choice(styles)
        elif styles and random.random() < self.p_style:
            chosen = random.choice(styles)
        else:
            chosen = base
        if not chosen:
            raise ValueError("No image path available in manifest entry.")
        return self.images_root / chosen, item.get("render_meta", {})

    def __getitem__(self, idx: int) -> VisualizedSample:
        item = self.samples[idx]
        image_path, render_meta = self._select_image(item)
        image = Image.open(image_path).convert("RGB")
        return VisualizedSample(
            image=image,
            question=item["question"],
            sql=item["sql"],
            db_path=Path(item["db_path"]),
            render_meta=render_meta,
        )
