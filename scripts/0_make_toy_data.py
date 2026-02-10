from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root
    db_dir = data_root / "databases"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "toy.db"

    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS employees")
    conn.execute("CREATE TABLE employees(id INTEGER, name TEXT, dept TEXT, salary REAL)")
    conn.executemany(
        "INSERT INTO employees VALUES (?, ?, ?, ?)",
        [
            (1, "Alice", "HR", 100.0),
            (2, "Bob", "ENG", 200.0),
            (3, "Cara", "ENG", 300.0),
        ],
    )
    conn.commit()
    conn.close()

    rows = [
        {
            "id": "toy-1",
            "question": "What is the average salary?",
            "sql": "SELECT AVG(salary) FROM employees",
            "db_path": str(db_path.relative_to(data_root)),
        },
        {
            "id": "toy-2",
            "question": "How many employees are in ENG?",
            "sql": "SELECT COUNT(*) FROM employees WHERE dept = 'ENG'",
            "db_path": str(db_path.relative_to(data_root)),
        },
    ]

    for split in ["train", "dev", "test"]:
        with (data_root / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    print(f"Toy data generated at: {data_root}")


if __name__ == "__main__":
    main()
