from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TPVEPRConfig:
    numeric_features: list[str]
    categorical_features: list[str]
    relation_features: list[str]
    prior_features: list[str]
    embedding_dim: int = 8
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ffn_multiplier: int = 2
    dropout: float = 0.1
    learning_rate: float = 6e-4
    weight_decay: float = 2e-4
    epochs: int = 5
    patience: int = 2
    max_group_size: int = 256
    max_batch_groups: int = 256
    max_pair_tokens: int = 120_000
    list_temperature: float = 0.25
    listwise_weight: float = 0.35
    pairwise_weight: float = 0.40
    top1_weight: float = 0.20
    calibration_weight: float = 0.05
    ndcg_loss_k: int = 5
    device: str = "auto"
    seed: int = 20260626

    @classmethod
    def from_json(cls, path: Path) -> "TPVEPRConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        numeric = set(self.numeric_features)
        unknown_relation = sorted(set(self.relation_features) - numeric)
        unknown_prior = sorted(set(self.prior_features) - numeric)
        if unknown_relation or unknown_prior:
            raise ValueError(
                "relation_features and prior_features must occur in numeric_features; "
                f"unknown relation={unknown_relation}, prior={unknown_prior}"
            )
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must both be at least 1.")
        weights = (
            self.listwise_weight,
            self.pairwise_weight,
            self.top1_weight,
            self.calibration_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Loss weights must be non-negative and have a positive sum.")
