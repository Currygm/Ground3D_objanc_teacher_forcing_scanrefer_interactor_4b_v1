"""LLM-only training stage."""
from __future__ import annotations
from typing import Any
import torch
from torch import nn
from ..config import ARCHITECTURE_SIGNATURE, OBJECTIVE
from .llm import CompressedVisualObjAncLM

class ObjAncAnswerStage(nn.Module):
    objective = OBJECTIVE
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.model = CompressedVisualObjAncLM(config)

    def forward(self, batch: dict[str, list[Any]]):
        return self.model(
            batch["point_cloud"], batch["question"], batch["teacher_answer"]
        )

    def architecture_signature(self) -> dict[str, Any]:
        signature = dict(ARCHITECTURE_SIGNATURE)
        signature.update({
            "base_model": str(self.config["model"]["model_name_or_path"]),
            "hidden_size": 896,
            "structured_tokens": tuple(self.config["model"]["structured_tokens"]),
            "lora": dict(self.config["model"]["lora"]),
        })
        return signature

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return self.model.trainable_state_dict()

    def trainable_parameter_summary(self) -> dict[str, int]:
        return self.model.trainable_parameter_summary()

    def optimizer_parameter_groups(self, **kwargs):
        return self.model.optimizer_parameter_groups(**kwargs)
