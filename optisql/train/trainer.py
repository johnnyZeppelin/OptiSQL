from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from optisql.models.optisql_model import OptiSQLBatch, OptiSQLModel
from optisql.train.checkpoint import save_checkpoint


@dataclass
class TrainConfig:
    max_steps: int = 1000
    gradient_accumulation: int = 1
    device: str = "cpu"
    save_every: int = 500
    output_dir: Path = Path("checkpoints")


def train_loop(
    model: OptiSQLModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> None:
    model.to(config.device)
    model.train()
    micro_step = 0
    update_step = 0
    optimizer.zero_grad()
    progress = tqdm(total=config.max_steps, desc="train")

    while update_step < config.max_steps:
        for batch in dataloader:
            opti_batch = OptiSQLBatch(
                images=batch["images"].to(config.device),
                questions=batch["questions"],
                sql=batch["sql"],
            )
            loss = model.forward_train(opti_batch)
            loss.backward()
            micro_step += 1

            if micro_step % config.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                update_step += 1
                progress.update(1)
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

                if config.save_every > 0 and update_step % config.save_every == 0:
                    save_checkpoint(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": update_step,
                        },
                        config.output_dir / f"step_{update_step}.pt",
                    )

                if update_step >= config.max_steps:
                    break
    progress.close()

    save_checkpoint(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": update_step,
        },
        config.output_dir / "last.pt",
    )
