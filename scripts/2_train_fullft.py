from __future__ import annotations

import argparse
from pathlib import Path

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
    args = parser.parse_args()

    dataset = VisualizedDataset(args.manifest, args.images_root)

    def collate(batch):
        return {
            "images": torch.stack([torch.tensor(sample.image).permute(2, 0, 1).float() / 255 for sample in batch]),
            "questions": [sample.question for sample in batch],
            "sql": [sample.sql for sample in batch],
        }

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)
    encoder = DeepSeekOCRAdapter()
    model = OptiSQLModel(encoder=encoder, decoder_name=args.decoder_name)
    optimizer = build_optimizer(model.decoder.parameters(), model.encoder.parameters(), OptimConfig())
    train_loop(model, dataloader, optimizer, TrainConfig(max_steps=args.max_steps, device="cpu"))


if __name__ == "__main__":
    main()
