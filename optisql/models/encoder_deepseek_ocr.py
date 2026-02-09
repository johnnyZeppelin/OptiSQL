from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class EncoderOutput:
    token_embeddings: torch.Tensor


class DeepSeekOCRAdapter(nn.Module):
    """Placeholder adapter for an OCR-oriented visual encoder.

    This wrapper exposes a uniform interface for optical token extraction.
    """

    def __init__(self, embed_dim: int = 1024, token_budget: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.token_budget = token_budget
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(32, embed_dim)

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        features = self.backbone(images).squeeze(-1).squeeze(-1)
        projected = self.proj(features)
        tokens = projected.unsqueeze(1).repeat(1, self.token_budget, 1)
        return EncoderOutput(token_embeddings=tokens)


class OpticalTokenProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        return self.proj(token_embeddings)
