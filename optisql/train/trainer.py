from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from optisql.models.optisql_model import OptiSQLModel, OptiSQLBatch


@dataclass
class TrainConfig:
    max_steps: int = 1000
    gradient_accumulation: int = 1
    device: str = "cpu"


def train_loop(
    model: OptiSQLModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> None:
    model.to(config.device)
    model.train()
    step = 0
    optimizer.zero_grad()
    progress = tqdm(total=config.max_steps, desc="train")
    while step < config.max_steps:
        for batch in dataloader:
            opti_batch = OptiSQLBatch(
                images=batch["images"].to(config.device),
                questions=batch["questions"],
                sql=batch["sql"],
            )
            loss = model.forward_train(opti_batch)
            loss.backward()
            if (step + 1) % config.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update(1)
            step += 1
            if step >= config.max_steps:
                break
    progress.close()
