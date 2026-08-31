import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from fusion_model import OpticalSARFusion

BASE = Path("datasets/processed")
TRAIN_CSV = BASE / "splits" / "train.csv"
VAL_CSV = BASE / "splits" / "validation.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001


class FusionDataset(Dataset):

    def __init__(self, csv_file):
        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            self.files = [row["file"] for row in reader]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        data = np.load(self.files[index])

        s1 = data["s1"].astype(np.float32) / 255.0
        s2 = data["s2"].astype(np.float32) / 255.0

        return (
            torch.tensor(s2),
            torch.tensor(s1)
        )


class FusionTrainingModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fusion = OpticalSARFusion()

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, optical, sar):

        features = self.fusion(optical, sar)

        return self.classifier(features)


print("Device:", DEVICE)

train_dataset = FusionDataset(TRAIN_CSV)
val_dataset = FusionDataset(VAL_CSV)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = FusionTrainingModel().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss()

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for optical, sar in train_loader:

        optical = optical.to(DEVICE)
        sar = sar.to(DEVICE)

        optimizer.zero_grad()

        output = model(optical, sar)

        target = torch.zeros(
            output.size(0),
            dtype=torch.long,
            device=DEVICE
        )

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predictions = output.argmax(dim=1)

        train_correct += (predictions == target).sum().item()
        train_total += target.size(0)

    train_accuracy = train_correct / train_total

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for optical, sar in val_loader:

            optical = optical.to(DEVICE)
            sar = sar.to(DEVICE)

            output = model(optical, sar)

            target = torch.zeros(
                output.size(0),
                dtype=torch.long,
                device=DEVICE
            )

            loss = criterion(output, target)

            val_loss += loss.item()

            predictions = output.argmax(dim=1)

            val_correct += (predictions == target).sum().item()
            val_total += target.size(0)

    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Train Loss: {train_loss / len(train_loader):.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Val Loss: {val_loss / len(val_loader):.4f} "
        f"| Val Acc: {val_accuracy:.4f}"
    )

OUTPUT = BASE / "fusion_model.pth"

torch.save(model.state_dict(), OUTPUT)

print()
print("Training completed.")
print("Model saved to:", OUTPUT)