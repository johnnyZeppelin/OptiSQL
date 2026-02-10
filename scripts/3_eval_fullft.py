from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from optisql.data.visualized_dataset import VisualizedDataset
from optisql.eval.decode import eval_exacc, eval_excan
from optisql.models.encoder_deepseek_ocr import DeepSeekOCRAdapter
from optisql.models.optisql_model import OptiSQLModel, OptiSQLBatch
from optisql.train.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images_root", type=Path, required=True)
    parser.add_argument("--decoder_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    dataset = VisualizedDataset(args.manifest, args.images_root)
    encoder = DeepSeekOCRAdapter()
    model = OptiSQLModel(encoder=encoder, decoder_name=args.decoder_name)
    if args.checkpoint and args.checkpoint.exists():
        state = load_checkpoint(args.checkpoint)
        model.load_state_dict(state["model"])
    model.eval()

    exacc = 0
    excan = 0
    for sample in dataset:
        image_tensor = torch.from_numpy(np.array(sample.image)).permute(2, 0, 1).float() / 255
        batch = OptiSQLBatch(
            images=image_tensor.unsqueeze(0),
            questions=[sample.question],
            sql=[sample.sql],
        )
        with torch.no_grad():
            pred_sql = model.generate_sql(batch)[0]
        exacc += int(eval_exacc(pred_sql, sample.sql, str(sample.db_path)))
        excan += int(eval_excan(pred_sql, sample.sql))
    count = len(dataset)
    print({"exacc": exacc / count if count else 0.0, "excan": excan / count if count else 0.0})


if __name__ == "__main__":
    main()
