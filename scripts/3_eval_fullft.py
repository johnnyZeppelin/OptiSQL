from __future__ import annotations

import argparse
from pathlib import Path

from optisql.data.visualized_dataset import VisualizedDataset
from optisql.eval.decode import eval_exacc, eval_excan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images_root", type=Path, required=True)
    args = parser.parse_args()

    dataset = VisualizedDataset(args.manifest, args.images_root)
    # Placeholder evaluation loop - add model inference in follow-up.
    exacc = 0
    excan = 0
    for sample in dataset:
        pred_sql = sample.sql
        exacc += int(eval_exacc(pred_sql, sample.sql, str(sample.db_path)))
        excan += int(eval_excan(pred_sql, sample.sql))
    count = len(dataset)
    print({"exacc": exacc / count if count else 0.0, "excan": excan / count if count else 0.0})


if __name__ == "__main__":
    main()
