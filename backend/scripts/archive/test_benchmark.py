import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path("backend/scripts").resolve()))

from fusion_model import MultiTaskFusionModel
from vqa_dataset import create_loaders


BASE = Path("datasets/processed")
MODEL_PATH = BASE / "multitask_fusion_model_final.pth"


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_device(device):
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, test_loader, device):

    model.eval()

    binary_loss_fn = nn.CrossEntropyLoss()
    mcq_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()

    total_loss = 0.0
    loss_batches = 0

    binary_correct = 0
    binary_total = 0

    mcq_correct = 0
    mcq_total = 0

    bbox_loss_sum = 0.0
    bbox_total = 0

    caption_total = 0

    total_batches = len(test_loader)

    print("\n" + "=" * 70)
    print("             SATQUERY AI TEST SET EVALUATION")
    print("=" * 70)
    print(f"Test batches: {total_batches}")
    print("Running inference...")
    print()

    sync_device(device)
    start_time = time.time()

    for batch_idx, batch in enumerate(test_loader):

        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        question = batch["question"].to(device)

        task_types = batch["type"]
        targets = batch["target"]

        outputs = model(
            optical=s2,
            sar=s1,
            question=question
        )

        binary_indices = [
            i for i, t in enumerate(task_types)
            if t in ("binary", "captioning")
        ]

        mcq_indices = [
            i for i, t in enumerate(task_types)
            if t == "mcq"
        ]

        bbox_indices = [
            i for i, t in enumerate(task_types)
            if t == "bounding box"
        ]

        losses = []

        if binary_indices:

            bin_idx = torch.tensor(
                binary_indices,
                device=device
            )

            bin_targets = torch.stack(
                [targets[i] for i in binary_indices]
            ).to(device, dtype=torch.long)

            bin_preds = outputs["binary"][bin_idx]

            loss_bin = binary_loss_fn(
                bin_preds,
                bin_targets
            )

            losses.append(loss_bin)

            predictions = bin_preds.argmax(dim=1)

            for k, i in enumerate(binary_indices):

                if task_types[i] == "binary":

                    binary_total += 1

                    if predictions[k].item() == bin_targets[k].item():
                        binary_correct += 1

                else:
                    caption_total += 1

        if mcq_indices:

            mcq_idx = torch.tensor(
                mcq_indices,
                device=device
            )

            mcq_targets = torch.stack(
                [targets[i] for i in mcq_indices]
            ).to(device, dtype=torch.long)

            mcq_preds = outputs["mcq"][mcq_idx]

            loss_mcq = mcq_loss_fn(
                mcq_preds,
                mcq_targets
            )

            losses.append(loss_mcq)

            predictions = mcq_preds.argmax(dim=1)

            for k in range(len(mcq_indices)):

                mcq_total += 1

                if predictions[k].item() == mcq_targets[k].item():
                    mcq_correct += 1

        if bbox_indices:

            bbox_idx = torch.tensor(
                bbox_indices,
                device=device
            )

            bbox_targets = torch.stack(
                [targets[i] for i in bbox_indices]
            ).to(device, dtype=torch.float32)

            bbox_preds = outputs["bbox"][bbox_idx]

            loss_bbox = bbox_loss_fn(
                bbox_preds,
                bbox_targets
            )

            losses.append(loss_bbox)

            bbox_loss_sum += (
                loss_bbox.item() * len(bbox_indices)
            )

            bbox_total += len(bbox_indices)

        if losses:

            batch_loss = sum(losses) / len(losses)

            total_loss += batch_loss.item()
            loss_batches += 1

        if (batch_idx + 1) % 1000 == 0:

            binary_acc = (
                100.0 * binary_correct / binary_total
                if binary_total > 0 else 0.0
            )

            mcq_acc = (
                100.0 * mcq_correct / mcq_total
                if mcq_total > 0 else 0.0
            )

            print(
                f"Test Batch {batch_idx + 1}/{total_batches} "
                f"({100.0 * (batch_idx + 1) / total_batches:.1f}%) | "
                f"BinAcc: {binary_acc:.2f}% | "
                f"MCQAcc: {mcq_acc:.2f}%",
                flush=True
            )

    sync_device(device)

    elapsed = time.time() - start_time

    test_loss = (
        total_loss / loss_batches
        if loss_batches > 0 else 0.0
    )

    binary_accuracy = (
        100.0 * binary_correct / binary_total
        if binary_total > 0 else 0.0
    )

    mcq_accuracy = (
        100.0 * mcq_correct / mcq_total
        if mcq_total > 0 else 0.0
    )

    bbox_loss = (
        bbox_loss_sum / bbox_total
        if bbox_total > 0 else 0.0
    )

    print("\n" + "=" * 70)
    print("                 FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Test Loss:                 {test_loss:.4f}")
    print(f"Binary Accuracy:           {binary_accuracy:.2f}%")
    print(f"Binary Samples:            {binary_total}")
    print(f"MCQ Accuracy:              {mcq_accuracy:.2f}%")
    print(f"MCQ Samples:               {mcq_total}")
    print(f"BBox Loss:                 {bbox_loss:.4f}")
    print(f"BBox Samples:              {bbox_total}")
    print(f"Captioning Samples:        {caption_total}")
    print(f"Evaluation Time:           {elapsed / 60:.2f} minutes")

    print("=" * 70)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nDevice:")
    print(device)

    print("\nTEST EVALUATION COMPLETE.")


def main():

    print("=" * 70)
    print("             SatQuery AI Model Testing")
    print("=" * 70)

    device = get_device()

    print(f"Testing Device: {device}")

    if not MODEL_PATH.exists():

        print("\nERROR:")
        print(f"Model not found: {MODEL_PATH}")
        return

    with open(
        BASE / "vocabulary.json",
        "r",
        encoding="utf-8"
    ) as f:

        vocab = json.load(f)

    vocab_size = vocab["vocab_size"]

    print(f"Vocabulary Size: {vocab_size}")

    print("\nCreating model...")

    model = MultiTaskFusionModel(
        vocab_size=vocab_size
    ).to(device)

    print("Loading trained model...")

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    print("Model weights loaded successfully.")

    print("\nLoading test dataset...")

    _, _, test_loader = create_loaders(
        batch_size=4
    )

    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Test batches: {len(test_loader)}")

    evaluate(
        model,
        test_loader,
        device
    )


if __name__ == "__main__":
    main()