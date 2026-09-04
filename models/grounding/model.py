"""
SatQuery AI — Region Grounding Neural Network Architecture.

Specialized architecture for spatial region grounding from satellite image + text query.
Uses Optical-SAR dual encoder + Question GRU + Multimodal Fusion + Box Regressor.
"""

from __future__ import annotations

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
    def __init__(self, vocab_size: int = 493, embedding_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return hidden[-1]


class SpatialGroundingNetwork(nn.Module):
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
        image: torch.Tensor,
        question: torch.Tensor,
        sar_image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if sar_image is None:
            # Handle optical (4-channel or 3-channel) vs SAR fallback
            if image.shape[1] == 3:
                # Pad RGB to 4 channels for optical_encoder
                zeros = torch.zeros((image.shape[0], 1, image.shape[2], image.shape[3]), device=image.device)
                optical_tensor = torch.cat([image, zeros], dim=1)
            elif image.shape[1] == 4:
                optical_tensor = image
            else:
                optical_tensor = image[:, :4, :, :]
            sar_tensor = torch.zeros((image.shape[0], 2, image.shape[2], image.shape[3]), device=image.device)
        else:
            optical_tensor = image
            sar_tensor = sar_image

        opt_feat = self.optical_encoder(optical_tensor)
        sar_feat = self.sar_encoder(sar_tensor)
        q_feat = self.question_encoder(question)

        combined = torch.cat([opt_feat, sar_feat, q_feat], dim=1)
        fused = self.fusion(combined)
        bbox = self.bbox_head(fused)
        binary_logits = self.binary_head(fused)

        return {
            "bbox": bbox,
            "confidence_logits": binary_logits,
            "features": fused,
        }
