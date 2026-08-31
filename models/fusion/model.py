"""
SatQuery AI — Optical-SAR Multi-Task Fusion Neural Network Architecture.

Exact Architecture used during training:
- OpticalEncoder: Conv2D(4 -> 32 -> 64 -> 128) + AdaptiveAvgPool2d(1) -> 128-d
- SAREncoder: Conv2D(2 -> 32 -> 64 -> 128) + AdaptiveAvgPool2d(1) -> 128-d
- QuestionEncoder: Embedding(vocab_size, 128) + GRU(128, 128) -> 128-d
- Fusion Layer: Linear(384 -> 256) -> ReLU -> Dropout(0.2) -> Linear(256 -> 128) -> ReLU
- Heads:
    - binary_head: Linear(128, 2)
    - mcq_head: Linear(128, 4)
    - bbox_head: Linear(128, 64) -> ReLU -> Linear(64, 4) -> Sigmoid
"""

import torch
import torch.nn as nn


class OpticalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).flatten(1)


class SAREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).flatten(1)


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return hidden[-1]


class MultiTaskFusionModel(nn.Module):
    def __init__(self, vocab_size: int = 493):
        super().__init__()
        self.optical_encoder = OpticalEncoder()
        self.sar_encoder = SAREncoder()
        self.question_encoder = QuestionEncoder(vocab_size=vocab_size)

        self.fusion = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.binary_head = nn.Linear(128, 2)
        self.mcq_head = nn.Linear(128, 4)
        self.bbox_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        question: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        optical_features = self.optical_encoder(optical)
        sar_features = self.sar_encoder(sar)
        question_features = self.question_encoder(question)

        combined = torch.cat(
            [
                optical_features,
                sar_features,
                question_features,
            ],
            dim=1,
        )

        fused_features = self.fusion(combined)

        return {
            "binary": self.binary_head(fused_features),
            "mcq": self.mcq_head(fused_features),
            "bbox": self.bbox_head(fused_features),
            "features": fused_features,
        }
