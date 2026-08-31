import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).parent))

from fusion_model import MultiTaskFusionModel
from vqa_dataset import create_loaders


BASE = Path("datasets/processed")

BATCH_SIZE = 2
EPOCHS = 1
MAX_BATCHES = 5000
LEARNING_RATE = 0.0001


if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


print("Device:", DEVICE)


with open(BASE / "vocabulary.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

VOCAB_SIZE = vocab["vocab_size"]

print("Vocabulary size:", VOCAB_SIZE)


train_loader, val_loader, test_loader = create_loaders(
    batch_size=BATCH_SIZE
)


model = MultiTaskFusionModel(
    vocab_size=VOCAB_SIZE
).to(DEVICE)


optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


binary_loss_fn = nn.CrossEntropyLoss()
mcq_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.SmoothL1Loss()


print("Model created.")
print("Starting training...")
print("Maximum batches:", MAX_BATCHES)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    batches = 0

    binary_count = 0
    mcq_count = 0
    bbox_count = 0

    for batch in train_loader:

        if batches >= MAX_BATCHES:
            break

        s1 = batch["s1"].to(DEVICE)
        s2 = batch["s2"].to(DEVICE)
        question = batch["question"].to(DEVICE)

        task_types = batch["type"]
        targets = batch["target"]

        optimizer.zero_grad()

        outputs = model(
            optical=s2,
            sar=s1,
            question=question
        )

        losses = []

        for i, task in enumerate(task_types):

            if task == "binary":

                target = targets[i].to(
                    DEVICE,
                    dtype=torch.long
                )

                prediction = outputs["binary"][i].unsqueeze(0)

                loss = binary_loss_fn(
                    prediction,
                    target.unsqueeze(0)
                )

                losses.append(loss)
                binary_count += 1

            elif task == "mcq":

                target = targets[i].to(
                    DEVICE,
                    dtype=torch.long
                )

                prediction = outputs["mcq"][i].unsqueeze(0)

                loss = mcq_loss_fn(
                    prediction,
                    target.unsqueeze(0)
                )

                losses.append(loss)
                mcq_count += 1

            elif task == "bounding box":

                target = targets[i].to(
                    DEVICE,
                    dtype=torch.float32
                )

                prediction = outputs["bbox"][i]

                loss = bbox_loss_fn(
                    prediction,
                    target
                )

                losses.append(loss)
                bbox_count += 1

        if not losses:
            continue

        loss = torch.stack(losses).mean()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        batches += 1

        if batches % 100 == 0:

            print(
                f"Batch {batches} | "
                f"Loss: {loss.item():.4f} | "
                f"Binary: {binary_count} | "
                f"MCQ: {mcq_count} | "
                f"BBox: {bbox_count}"
            )


    average_loss = total_loss / max(batches, 1)

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Average Loss: {average_loss:.4f}"
    )

    print(
        f"Samples used - "
        f"Binary: {binary_count}, "
        f"MCQ: {mcq_count}, "
        f"BBox: {bbox_count}"
    )


MODEL_FILE = BASE / "multitask_fusion_model.pth"

torch.save(
    model.state_dict(),
    MODEL_FILE
)


print()
print("Training completed.")
print("Batches completed:", batches)
print("Model saved to:", MODEL_FILE)