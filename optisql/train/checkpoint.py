from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")
