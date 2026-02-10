from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from optisql.models.encoder_deepseek_ocr import DeepSeekOCRAdapter, OpticalTokenProjector


@dataclass
class OptiSQLBatch:
    images: torch.Tensor
    questions: list[str]
    sql: list[str]


class OptiSQLModel(nn.Module):
    def __init__(
        self,
        encoder: DeepSeekOCRAdapter,
        decoder_name: str,
        token_budget: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.projector = OpticalTokenProjector(encoder.embed_dim, self.decoder.config.hidden_size)
        self.token_budget = token_budget

    def forward_train(self, batch: OptiSQLBatch) -> torch.Tensor:
        encoder_out = self.encoder(batch.images)
        optical_embeds = self.projector(encoder_out.token_embeddings)
        question_tokens = self.tokenizer(
            batch.questions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(optical_embeds.device)
        sql_tokens = self.tokenizer(
            batch.sql,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(optical_embeds.device)
        question_embeds = self.decoder.get_input_embeddings()(question_tokens.input_ids)
        sql_embeds = self.decoder.get_input_embeddings()(sql_tokens.input_ids)
        inputs_embeds = torch.cat([optical_embeds, question_embeds, sql_embeds], dim=1)
        labels = torch.full(
            (inputs_embeds.shape[0], inputs_embeds.shape[1]),
            fill_value=-100,
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        labels[:, -sql_tokens.input_ids.shape[1] :] = sql_tokens.input_ids
        outputs = self.decoder(inputs_embeds=inputs_embeds, labels=labels)
        return outputs.loss

    def generate_sql(self, batch: OptiSQLBatch, beam: int = 4, max_new_tokens: int = 256) -> list[str]:
        encoder_out = self.encoder(batch.images)
        optical_embeds = self.projector(encoder_out.token_embeddings)
        question_tokens = self.tokenizer(
            batch.questions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(optical_embeds.device)
        question_embeds = self.decoder.get_input_embeddings()(question_tokens.input_ids)
        inputs_embeds = torch.cat([optical_embeds, question_embeds], dim=1)
        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            num_beams=beam,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
