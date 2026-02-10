from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import subprocess
import sys

import requests

from optisql.utils.logging import setup_logging


CANDIDATE_BASE_URLS = [
    "https://raw.githubusercontent.com/xlang-ai/Spider2/main/spider2-snow",
    "https://raw.githubusercontent.com/xlang-ai/Spider2/main/spider2-snow/resource",
]


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)


def try_download_jsonl(data_root: Path, logger) -> bool:
    splits = ["train", "dev", "test"]
    for base in CANDIDATE_BASE_URLS:
        ok = True
        for split in splits:
            url = f"{base}/{split}.jsonl"
            dest = data_root / f"{split}.jsonl"
            logger.info("Trying download %s", url)
            try:
                download_file(url, dest)
            except Exception:
                ok = False
                break
        if ok:
            return True
    return False


def try_clone_repo(data_root: Path, logger) -> bool:
    tmp_root = data_root.parent / "Spider2_tmp"
    if tmp_root.exists():
        subprocess.run(["rm", "-rf", str(tmp_root)], check=False)
    cmd = ["git", "clone", "--depth", "1", "https://github.com/xlang-ai/Spider2", str(tmp_root)]
    logger.info("Fallback: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        return False
    candidates = [tmp_root / "spider2-snow", tmp_root / "spider2-snow" / "resource"]
    for candidate in candidates:
        if all((candidate / f"{split}.jsonl").exists() for split in ["train", "dev", "test"]):
            data_root.mkdir(parents=True, exist_ok=True)
            for split in ["train", "dev", "test"]:
                (data_root / f"{split}.jsonl").write_bytes((candidate / f"{split}.jsonl").read_bytes())
            return True
    return False


def validate_local(data_root: Path) -> tuple[bool, list[str]]:
    missing = []
    for split in ["train", "dev", "test"]:
        path = data_root / f"{split}.jsonl"
        if not path.exists():
            missing.append(str(path))
    return (len(missing) == 0, missing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--auto_download", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("download")

    if args.auto_download:
        if not try_download_jsonl(args.data_root, logger):
            logger.warning("Direct jsonl download failed, trying git clone fallback.")
            if not try_clone_repo(args.data_root, logger):
                logger.error("Failed to fetch spider2-snow automatically.")
                logger.error("Please manually place train/dev/test jsonl under %s", args.data_root)
                sys.exit(1)

    ok, missing = validate_local(args.data_root)
    if not ok:
        logger.error("Missing required files: %s", missing)
        sys.exit(1)

    logger.info("Spider2-snow data ready at %s", args.data_root)


if __name__ == "__main__":
    main()
