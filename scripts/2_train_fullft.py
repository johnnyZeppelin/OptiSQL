from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from optisql.data.visualized_dataset import VisualizedDataset
from optisql.models.encoder_deepseek_ocr import DeepSeekOCRAdapter
from optisql.models.optisql_model import OptiSQLModel
from optisql.train.optim import OptimConfig, build_optimizer
from optisql.train.trainer import TrainConfig, train_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images_root", type=Path, required=True)
    parser.add_argument("--decoder_name", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/fullft"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    dataset = VisualizedDataset(args.manifest, args.images_root)

    def collate(batch):
        return {
            "images": torch.stack(
                [
                    torch.from_numpy(np.array(sample.image)).permute(2, 0, 1).float() / 255
                    for sample in batch
                ]
            ),
            "questions": [sample.question for sample in batch],
            "sql": [sample.sql for sample in batch],
        }

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    encoder = DeepSeekOCRAdapter()
    model = OptiSQLModel(encoder=encoder, decoder_name=args.decoder_name)
    optimizer = build_optimizer(model.decoder.parameters(), model.encoder.parameters(), OptimConfig())
    train_loop(
        model,
        dataloader,
        optimizer,
        TrainConfig(
            max_steps=args.max_steps,
            gradient_accumulation=args.grad_accum,
            device=args.device,
            save_every=args.save_every,
            output_dir=args.output_dir,
        ),
    )


if __name__ == "__main__":
    main()
