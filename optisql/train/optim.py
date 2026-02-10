from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class OptimConfig:
    lr_decoder: float = 1e-4
    lr_encoder: float = 5e-6
    weight_decay: float = 0.1


def build_optimizer(
    decoder_params: Iterable[torch.nn.Parameter],
    encoder_params: Iterable[torch.nn.Parameter],
    config: OptimConfig,
) -> torch.optim.Optimizer:
    param_groups = [
        {"params": list(decoder_params), "lr": config.lr_decoder, "weight_decay": config.weight_decay},
        {"params": list(encoder_params), "lr": config.lr_encoder, "weight_decay": config.weight_decay},
    ]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
