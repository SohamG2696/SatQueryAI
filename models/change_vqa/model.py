"""
SatQuery AI — Bi-Temporal Change-VQA Neural Network Architecture.

Accepts two temporal images (T1 Before, T2 After) + natural-language question
and classifies multi-temporal environmental and land-cover changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BiTemporalChangeNetwork(nn.Module):
    def __init__(self, vocab_size: int = 493, embed_dim: int = 128):
        super().__init__()
        # Shared siamese feature extractor for T1 and T2
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_img = nn.Linear(128, embed_dim)

        # Question encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

        # Difference and change fusion
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 256),  # [T1, T2, |T2 - T1|, Question]
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),  # Binary YES/NO or change indicator
        )

    def forward(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        question: torch.Tensor,
    ) -> torch.Tensor:
        feat_t1 = self.fc_img(self.encoder(t1).flatten(1))
        feat_t2 = self.fc_img(self.encoder(t2).flatten(1))
        diff = torch.abs(feat_t2 - feat_t1)

        emb = self.embedding(question)
        _, hidden = self.gru(emb)
        q_feat = hidden[-1]

        fused = torch.cat([feat_t1, feat_t2, diff, q_feat], dim=1)
        logits = self.classifier(fused)
        return logits
