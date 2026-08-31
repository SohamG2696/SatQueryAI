"""
SatQuery AI — Region Grounding Neural Network Architecture.

Specialized architecture for spatial region grounding from single satellite image + text query.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialGroundingNetwork(nn.Module):
    def __init__(self, vocab_size: int = 493, embed_dim: int = 128):
        super().__init__()
        # Image feature extractor (4-channel optical / 3-channel RGB)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        # Question encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

        # Cross-modal spatial attention and box regression
        self.spatial_proj = nn.Conv2d(128, embed_dim, kernel_size=1)
        self.box_regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid(),  # Output in [0, 1] normalized space
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, question: torch.Tensor) -> dict[str, torch.Tensor]:
        img_feats = self.image_encoder(image)  # [B, 128, 7, 7]
        img_pool = torch.mean(img_feats, dim=(2, 3))  # [B, 128]

        emb = self.embedding(question)
        _, hidden = self.gru(emb)
        q_feat = hidden[-1]  # [B, 128]

        fused = torch.cat([img_pool, q_feat], dim=1)
        bbox = self.box_regressor(fused)
        conf = self.confidence_head(fused)

        return {
            "bbox": bbox,
            "confidence": conf,
        }
