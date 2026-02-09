from pathlib import Path
from typing import Any

from optisql.utils.io import read_jsonl


class Spider2SnowDataset:
    """Lightweight loader for Spider2-snow jsonl files."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root

    def load_split(self, split: str) -> list[dict[str, Any]]:
        split_path = self.data_root / f"{split}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        return read_jsonl(split_path)
