import torch
import torch.nn as nn


class OpticalEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.network(x).flatten(1)


class SAREncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.network(x).flatten(1)


class QuestionEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return hidden[-1]


class MultiTaskFusionModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.optical_encoder = OpticalEncoder()
        self.sar_encoder = SAREncoder()

        self.question_encoder = QuestionEncoder(
            vocab_size=vocab_size
        )

        self.fusion = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.binary_head = nn.Linear(128, 2)

        self.mcq_head = nn.Linear(128, 4)

        self.bbox_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, optical, sar, question):

        optical_features = self.optical_encoder(optical)

        sar_features = self.sar_encoder(sar)

        question_features = self.question_encoder(question)

        combined = torch.cat(
            [
                optical_features,
                sar_features,
                question_features
            ],
            dim=1
        )

        fused_features = self.fusion(combined)

        return {
            "binary": self.binary_head(fused_features),
            "mcq": self.mcq_head(fused_features),
            "bbox": self.bbox_head(fused_features),
            "features": fused_features
        }