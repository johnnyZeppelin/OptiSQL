from __future__ import annotations

import argparse
from pathlib import Path
import sys

import requests

from optisql.utils.logging import setup_logging


SPIDER2_URL = "https://raw.githubusercontent.com/xlang-ai/Spider2/main/spider2-snow"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--auto_download", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("download")

    if args.auto_download:
        for split in ["train", "dev", "test"]:
            url = f"{SPIDER2_URL}/{split}.jsonl"
            dest = args.data_root / f"{split}.jsonl"
            logger.info("Downloading %s", url)
            try:
                download_file(url, dest)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to download %s: %s", url, exc)
                sys.exit(1)
    else:
        for split in ["train", "dev", "test"]:
            path = args.data_root / f"{split}.jsonl"
            if not path.exists():
                logger.error("Missing %s. Use --auto_download or place files manually.", path)
                sys.exit(1)
    logger.info("Spider2-snow data ready at %s", args.data_root)


if __name__ == "__main__":
    main()
