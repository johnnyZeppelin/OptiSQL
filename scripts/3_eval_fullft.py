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
from optisql.eval.robustness import apply_header_mask
from optisql.models.encoder_deepseek_ocr import DeepSeekOCRAdapter
from optisql.models.optisql_model import OptiSQLModel, OptiSQLBatch
from optisql.train.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images_root", type=Path, required=True)
    parser.add_argument("--decoder_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--header_mask_ratio", type=float, default=0.0)
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
    exacc_header_mask = 0
    for sample in dataset:
        image_tensor = torch.from_numpy(np.array(sample.image)).permute(2, 0, 1).float() / 255
        batch = OptiSQLBatch(
            images=image_tensor.unsqueeze(0),
            questions=[sample.question],
            sql=[sample.sql],
        )
        with torch.no_grad():
            pred_sql = model.generate_sql(batch)[0]
        clean_ok = int(eval_exacc(pred_sql, sample.sql, str(sample.db_path)))
        exacc += clean_ok
        excan += int(eval_excan(pred_sql, sample.sql))

        if args.header_mask_ratio > 0 and sample.render_meta.get("header_bbox") is not None:
            masked = apply_header_mask(sample.image, tuple(sample.render_meta["header_bbox"]), args.header_mask_ratio)
            masked_tensor = torch.from_numpy(np.array(masked)).permute(2, 0, 1).float() / 255
            masked_batch = OptiSQLBatch(images=masked_tensor.unsqueeze(0), questions=[sample.question], sql=[sample.sql])
            with torch.no_grad():
                pred_masked = model.generate_sql(masked_batch)[0]
            exacc_header_mask += int(eval_exacc(pred_masked, sample.sql, str(sample.db_path)))
        else:
            exacc_header_mask += clean_ok
    count = len(dataset)
    result = {"exacc": exacc / count if count else 0.0, "excan": excan / count if count else 0.0}
    if count:
        result["exacc_header_mask"] = exacc_header_mask / count
        result["header_mask_drop"] = result["exacc"] - result["exacc_header_mask"]
    print(result)


if __name__ == "__main__":
    main()
