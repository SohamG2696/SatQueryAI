"""
SatQuery AI -- Multi-Task Remote-Sensing VQA Training (Epochs 3 to 5)
======================================================================

Resumes MultiTaskFusionModel training directly from Epoch 3 Batch 110,000
and trains through Epoch 5 (5 total epochs) on Intel Arc XPU.

Optimized Fast-Resume:
- Uses PyTorch Subset slicing so zero already-processed batches are loaded from disk.
- Resumes training at Batch 110,001 within 0.1 seconds.
- Evaluates full validation on 61,402 annotations after every completed epoch.
- Checkpoints saved every 2,500 batches; best checkpoint updated only on val improvement.
- Final model created from best_checkpoint.pth at completion.
"""

import sys
import os
import json
import time
import copy
from pathlib import Path

# Force unbuffered output so logs update immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

from fusion_model import MultiTaskFusionModel
from vqa_dataset import VQADataset, collate_fn, TRAIN_FILE, VAL_FILE, TEST_FILE


# ====================================================================
# Configuration
# ====================================================================

BASE = Path("datasets/processed")
CHECKPOINT_DIR = BASE / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
LR_PATIENCE = 2
EARLY_STOP_PATIENCE = 3

# Batch configuration
BATCH_SIZE = 4

# Checkpoint paths
LATEST_CKPT = CHECKPOINT_DIR / "latest_checkpoint.pth"
BEST_CKPT = CHECKPOINT_DIR / "best_checkpoint.pth"
HISTORY_FILE = CHECKPOINT_DIR / "training_history.json"
FINAL_MODEL = BASE / "multitask_fusion_model_final.pth"

# Progress logging frequency
LOG_EVERY = 100        # Log training progress every 100 batches
SAVE_EVERY = 2500      # Save latest checkpoint every 2500 batches
VAL_EVERY = 5000       # Intermediate validation every 5000 batches


# ====================================================================
# Device Setup & Synchronization
# ====================================================================

def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


def sync_device():
    """Synchronize XPU/CUDA before timing measurements."""
    if DEVICE.type == "xpu":
        torch.xpu.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ====================================================================
# Validation Function
# ====================================================================

@torch.no_grad()
def validate(model, loader, device, max_val_batches=None):
    model.eval()

    binary_loss_fn = nn.CrossEntropyLoss()
    mcq_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()

    total_loss = 0.0
    batches = 0

    binary_correct = 0
    binary_total = 0
    mcq_correct = 0
    mcq_total = 0
    bbox_loss_sum = 0.0
    bbox_total = 0
    caption_total = 0

    total_batches = len(loader) if max_val_batches is None else min(len(loader), max_val_batches)

    sync_device()
    val_start = time.time()

    for batch_idx, batch in enumerate(loader):
        if max_val_batches and batch_idx >= max_val_batches:
            break

        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        question = batch["question"].to(device)
        task_types = batch["type"]
        targets = batch["target"]

        outputs = model(optical=s2, sar=s1, question=question)

        binary_indices = [i for i, t in enumerate(task_types) if t in ("binary", "captioning")]
        mcq_indices = [i for i, t in enumerate(task_types) if t == "mcq"]
        bbox_indices = [i for i, t in enumerate(task_types) if t == "bounding box"]

        losses = []

        if binary_indices:
            bin_idx = torch.tensor(binary_indices, device=device)
            bin_targets = torch.stack([targets[i] for i in binary_indices]).to(device, dtype=torch.long)
            bin_preds = outputs["binary"][bin_idx]
            losses.append(binary_loss_fn(bin_preds, bin_targets))

            bin_preds_cls = bin_preds.argmax(dim=1)
            for k, i in enumerate(binary_indices):
                if task_types[i] == "binary":
                    binary_total += 1
                    if bin_preds_cls[k].item() == bin_targets[k].item():
                        binary_correct += 1
                else:
                    caption_total += 1

        if mcq_indices:
            mcq_idx = torch.tensor(mcq_indices, device=device)
            mcq_targets = torch.stack([targets[i] for i in mcq_indices]).to(device, dtype=torch.long)
            mcq_preds = outputs["mcq"][mcq_idx]
            losses.append(mcq_loss_fn(mcq_preds, mcq_targets))

            mcq_preds_cls = mcq_preds.argmax(dim=1)
            for k, i in enumerate(mcq_indices):
                mcq_total += 1
                if mcq_preds_cls[k].item() == mcq_targets[k].item():
                    mcq_correct += 1

        if bbox_indices:
            bbox_idx = torch.tensor(bbox_indices, device=device)
            bbox_targets = torch.stack([targets[i] for i in bbox_indices]).to(device, dtype=torch.float32)
            bbox_preds = outputs["bbox"][bbox_idx]
            loss_bbox = bbox_loss_fn(bbox_preds, bbox_targets)
            losses.append(loss_bbox)

            bbox_loss_sum += loss_bbox.item() * len(bbox_indices)
            bbox_total += len(bbox_indices)

        if losses:
            loss = sum(losses) / len(losses)
            total_loss += loss.item()
            batches += 1

        if batches % 1000 == 0:
            pct = 100.0 * batches / total_batches
            print(f"  [Validation] Batch {batches}/{total_batches} ({pct:.1f}%)", flush=True)

    sync_device()
    val_time = time.time() - val_start

    metrics = {
        "val_loss": total_loss / max(batches, 1),
        "binary_accuracy": (100.0 * binary_correct / binary_total) if binary_total > 0 else 0.0,
        "binary_total": binary_total,
        "mcq_accuracy": (100.0 * mcq_correct / mcq_total) if mcq_total > 0 else 0.0,
        "mcq_total": mcq_total,
        "bbox_avg_loss": (bbox_loss_sum / bbox_total) if bbox_total > 0 else 0.0,
        "bbox_total": bbox_total,
        "caption_total": caption_total,
        "batches": batches,
        "val_time": val_time,
    }

    return metrics


# ====================================================================
# Checkpoint Saving & Loading
# ====================================================================

def save_checkpoint(path, model, optimizer, epoch, batch_num, best_val_loss, history):
    torch.save({
        "epoch": epoch,
        "batch_num": batch_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "history": history,
    }, path)


# ====================================================================
# Main Training Routine
# ====================================================================

def main():
    # 1. Device and Vocabulary
    with open(BASE / "vocabulary.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"]

    # 2. Instantiate Model and Optimizer
    model = MultiTaskFusionModel(vocab_size=vocab_size).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=LR_PATIENCE)

    # 3. Verify and Load Checkpoint
    ckpt_path = LATEST_CKPT if LATEST_CKPT.exists() else BEST_CKPT

    if not ckpt_path.exists():
        print("ERROR: No valid checkpoint found! Aborting to prevent restart from scratch.", flush=True)
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Restore Model & Optimizer State
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"Warning: could not restore optimizer state ({e})", flush=True)

    start_epoch = ckpt.get("epoch", 0)
    start_batch = ckpt.get("batch_num", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    history = ckpt.get("history", [])

    # Sync with training_history.json if history file has existing records
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                file_history = json.load(f)
            if len(file_history) > len(history):
                history = file_history
        except Exception:
            pass

    # 4. Load Datasets
    train_dataset = VQADataset(TRAIN_FILE)
    val_dataset = VQADataset(VAL_FILE)

    total_train_samples = len(train_dataset)
    total_train_batches = total_train_samples // BATCH_SIZE

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    remaining_epoch_batches = total_train_batches - start_batch if start_epoch < TOTAL_EPOCHS else 0

    # 5. Startup Verification Report
    print("=" * 60, flush=True)
    print("CHECKPOINT RESUME VERIFICATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Checkpoint: {ckpt_path}", flush=True)
    print(f"Checkpoint Epoch: {start_epoch} (Epoch {start_epoch + 1})", flush=True)
    print(f"Checkpoint Batch: {start_batch} / {total_train_batches}", flush=True)
    print(f"Best Validation Loss: {best_val_loss:.4f}", flush=True)
    print("\nExisting History:", flush=True)
    for h in history:
        print(f"  Epoch {h['epoch']}: Train Loss {h['train_loss']:.4f} | Val Loss {h['val_loss']:.4f} | Val BinAcc {h.get('val_binary_acc', 0):.1f}% | Val MCQAcc {h.get('val_mcq_acc', 0):.1f}%", flush=True)
    print(f"\nTarget: Epoch {TOTAL_EPOCHS}", flush=True)
    print(f"Resume Position: Epoch {start_epoch + 1} / {TOTAL_EPOCHS}, Batch {start_batch} / {total_train_batches}", flush=True)
    print(f"Remaining Epoch {start_epoch + 1} batches: {remaining_epoch_batches}", flush=True)
    print("=" * 60, flush=True)

    if start_epoch >= TOTAL_EPOCHS and start_batch == 0:
        print(f"All {TOTAL_EPOCHS} epochs are already completed. Nothing to train.", flush=True)
        return

    # 6. Loss Functions
    binary_loss_fn = nn.CrossEntropyLoss()
    mcq_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()

    overall_start = time.time()
    epochs_no_improve = 0

    # 7. Main Training Loop (Epochs start_epoch to TOTAL_EPOCHS - 1)
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        model.train()
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n>>> Epoch {epoch + 1}/{TOTAL_EPOCHS} (Learning Rate: {current_lr:.6f})", flush=True)

        # Efficient DataLoader Creation:
        # If resuming mid-epoch (e.g. start_batch = 110000 on Epoch 3), slice with Subset so zero completed batches are loaded
        if epoch == start_epoch and start_batch > 0:
            start_sample_idx = start_batch * BATCH_SIZE
            print(f"Creating fast resume loader starting directly at sample {start_sample_idx:,} (Batch {start_batch:,}/{total_train_batches:,})...", flush=True)
            subset_dataset = Subset(train_dataset, range(start_sample_idx, total_train_samples))
            epoch_train_loader = DataLoader(
                subset_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn
            )
            batch_offset = start_batch
        else:
            epoch_train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn
            )
            batch_offset = 0

        total_loss = 0.0
        batches_processed = 0

        binary_correct = 0
        binary_total = 0
        mcq_correct = 0
        mcq_total = 0
        bbox_loss_sum = 0.0
        bbox_total = 0
        caption_total = 0

        for step_idx, batch in enumerate(epoch_train_loader):
            current_batch_in_epoch = batch_offset + step_idx + 1

            s1 = batch["s1"].to(DEVICE)
            s2 = batch["s2"].to(DEVICE)
            question = batch["question"].to(DEVICE)
            task_types = batch["type"]
            targets = batch["target"]

            optimizer.zero_grad()
            outputs = model(optical=s2, sar=s1, question=question)

            binary_indices = [i for i, t in enumerate(task_types) if t in ("binary", "captioning")]
            mcq_indices = [i for i, t in enumerate(task_types) if t == "mcq"]
            bbox_indices = [i for i, t in enumerate(task_types) if t == "bounding box"]

            losses = []

            if binary_indices:
                bin_idx = torch.tensor(binary_indices, device=DEVICE)
                bin_targets = torch.stack([targets[i] for i in binary_indices]).to(DEVICE, dtype=torch.long)
                bin_preds = outputs["binary"][bin_idx]
                loss_bin = binary_loss_fn(bin_preds, bin_targets)
                losses.append(loss_bin)

                bin_preds_cls = bin_preds.argmax(dim=1)
                for k, i in enumerate(binary_indices):
                    if task_types[i] == "binary":
                        binary_total += 1
                        if bin_preds_cls[k].item() == bin_targets[k].item():
                            binary_correct += 1
                    else:
                        caption_total += 1

            if mcq_indices:
                mcq_idx = torch.tensor(mcq_indices, device=DEVICE)
                mcq_targets = torch.stack([targets[i] for i in mcq_indices]).to(DEVICE, dtype=torch.long)
                mcq_preds = outputs["mcq"][mcq_idx]
                loss_mcq = mcq_loss_fn(mcq_preds, mcq_targets)
                losses.append(loss_mcq)

                mcq_preds_cls = mcq_preds.argmax(dim=1)
                for k, i in enumerate(mcq_indices):
                    mcq_total += 1
                    if mcq_preds_cls[k].item() == mcq_targets[k].item():
                        mcq_correct += 1

            if bbox_indices:
                bbox_idx = torch.tensor(bbox_indices, device=DEVICE)
                bbox_targets = torch.stack([targets[i] for i in bbox_indices]).to(DEVICE, dtype=torch.float32)
                bbox_preds = outputs["bbox"][bbox_idx]
                loss_bbox = bbox_loss_fn(bbox_preds, bbox_targets)
                losses.append(loss_bbox)

                bbox_loss_sum += loss_bbox.item() * len(bbox_indices)
                bbox_total += len(bbox_indices)

            if not losses:
                continue

            loss = sum(losses) / len(losses)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            batches_processed += 1

            # Progress Reporting
            if current_batch_in_epoch % LOG_EVERY == 0 or current_batch_in_epoch == total_train_batches:
                sync_device()
                now = time.time()
                elapsed = now - epoch_start
                pct = 100.0 * current_batch_in_epoch / total_train_batches
                avg_loss = total_loss / max(batches_processed, 1)
                batches_remaining = total_train_batches - current_batch_in_epoch
                sec_per_batch = elapsed / max(batches_processed, 1)
                eta_min = (batches_remaining * sec_per_batch) / 60.0

                bin_acc = (100.0 * binary_correct / binary_total) if binary_total > 0 else 0.0
                mcq_acc = (100.0 * mcq_correct / mcq_total) if mcq_total > 0 else 0.0
                avg_bbox = (bbox_loss_sum / bbox_total) if bbox_total > 0 else 0.0

                print(
                    f"Epoch {epoch + 1}/{TOTAL_EPOCHS} | Batch {current_batch_in_epoch}/{total_train_batches} ({pct:.1f}%) | "
                    f"Loss: {avg_loss:.4f} | "
                    f"BinAcc: {bin_acc:.1f}% | "
                    f"MCQAcc: {mcq_acc:.1f}% | "
                    f"BBoxL: {avg_bbox:.4f} | "
                    f"Elapsed: {elapsed/60:.1f}m | "
                    f"ETA: {eta_min:.1f}m",
                    flush=True
                )

            # Periodic Checkpoint Saving
            if current_batch_in_epoch % SAVE_EVERY == 0:
                save_checkpoint(
                    LATEST_CKPT, model, optimizer,
                    epoch, current_batch_in_epoch, best_val_loss, history
                )

            # Intermediate Validation Checkpoint
            if current_batch_in_epoch % VAL_EVERY == 0:
                print(f"\n--- Running Intermediate Validation at Batch {current_batch_in_epoch} ---", flush=True)
                val_metrics = validate(model, val_loader, DEVICE, max_val_batches=2500)
                print(
                    f"  [Inter-Val @ {current_batch_in_epoch}] Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val BinAcc: {val_metrics['binary_accuracy']:.1f}% | "
                    f"Val MCQAcc: {val_metrics['mcq_accuracy']:.1f}% | "
                    f"Val BBoxL: {val_metrics['bbox_avg_loss']:.4f}",
                    flush=True
                )
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    save_checkpoint(
                        BEST_CKPT, model, optimizer,
                        epoch, current_batch_in_epoch, best_val_loss, history
                    )
                    print(f"  ** New Best Model Saved! Val Loss: {best_val_loss:.4f} **", flush=True)
                model.train()

        # Reset start_batch for all subsequent full epochs
        start_batch = 0

        # Epoch Complete - Run Full Validation on all 61,402 annotations
        sync_device()
        epoch_time = time.time() - epoch_start
        train_loss = total_loss / max(batches_processed, 1)

        print(f"\n--- Epoch {epoch + 1}/{TOTAL_EPOCHS} Training Complete (Time: {epoch_time/60:.1f} mins) ---", flush=True)
        print("Running full validation on all 61,402 validation annotations...", flush=True)
        val_metrics = validate(model, val_loader, DEVICE)

        scheduler.step(val_metrics["val_loss"])

        # Record History
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "train_binary_acc": (100.0 * binary_correct / binary_total) if binary_total > 0 else 0.0,
            "val_binary_acc": val_metrics["binary_accuracy"],
            "train_mcq_acc": (100.0 * mcq_correct / mcq_total) if mcq_total > 0 else 0.0,
            "val_mcq_acc": val_metrics["mcq_accuracy"],
            "train_bbox_loss": (bbox_loss_sum / bbox_total) if bbox_total > 0 else 0.0,
            "val_bbox_loss": val_metrics["bbox_avg_loss"],
            "train_samples": {
                "binary": binary_total,
                "mcq": mcq_total,
                "bbox": bbox_total,
                "captioning": caption_total
            },
            "lr": current_lr,
            "epoch_time_sec": epoch_time,
        }

        # Update or append epoch history
        existing_idx = None
        for idx, h in enumerate(history):
            if h.get("epoch") == epoch + 1:
                existing_idx = idx
                break
        if existing_idx is not None:
            history[existing_idx] = epoch_record
        else:
            history.append(epoch_record)

        # Save History JSON
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Update Best Checkpoint if Val Loss improved
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            epochs_no_improve = 0
            save_checkpoint(
                BEST_CKPT, model, optimizer,
                epoch + 1, 0, best_val_loss, history
            )
            print(f"** New Best Model Checkpoint Saved! (Val Loss: {best_val_loss:.4f}) **", flush=True)
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve ({val_metrics['val_loss']:.4f} vs best {best_val_loss:.4f}). Consecutive epochs without improvement: {epochs_no_improve}", flush=True)

        # Save Latest Checkpoint at end of epoch
        save_checkpoint(
            LATEST_CKPT, model, optimizer,
            epoch + 1, 0, best_val_loss, history
        )

        print(f"\n==================================================", flush=True)
        print(f"Epoch {epoch + 1}/{TOTAL_EPOCHS} Summary:", flush=True)
        print(f"  Train Loss:     {train_loss:.4f}", flush=True)
        print(f"  Val Loss:       {val_metrics['val_loss']:.4f}", flush=True)
        print(f"  Binary Acc:     Train {epoch_record['train_binary_acc']:.1f}% | Val {val_metrics['binary_accuracy']:.1f}%", flush=True)
        print(f"  MCQ Acc:        Train {epoch_record['train_mcq_acc']:.1f}% | Val {val_metrics['mcq_accuracy']:.1f}%", flush=True)
        print(f"  BBox Avg Loss:  Train {epoch_record['train_bbox_loss']:.4f} | Val {val_metrics['bbox_avg_loss']:.4f}", flush=True)
        print(f"  Total Elapsed:  {(time.time() - overall_start)/60:.1f} mins", flush=True)
        print(f"==================================================\n", flush=True)

        # Early Stopping Guard
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n[EARLY STOPPING TRIGGERED] Validation loss has not improved for {epochs_no_improve} consecutive epochs.", flush=True)
            print(f"Stopping training early to prevent overfitting. Best checkpoint preserved at: {BEST_CKPT}", flush=True)
            break

    # 8. Save Final Model using the BEST checkpoint
    print(f"\nLoading best checkpoint weights ({BEST_CKPT}) into final model...", flush=True)
    best_state = torch.load(BEST_CKPT, map_location=DEVICE, weights_only=False)
    torch.save(best_state["model_state_dict"], FINAL_MODEL)
    print(f"Final verified model saved to: {FINAL_MODEL}", flush=True)

    # 9. Print Complete Report for All Completed Epochs
    total_training_time = time.time() - overall_start
    best_epoch_idx = min(range(len(history)), key=lambda i: history[i]["val_loss"])
    best_record = history[best_epoch_idx]

    print("\n" + "=" * 75, flush=True)
    print(f"  TRAINING COMPLETE -- FINAL {TOTAL_EPOCHS}-EPOCH REPORT", flush=True)
    print("=" * 75, flush=True)
    for i in range(TOTAL_EPOCHS):
        if i < len(history):
            h = history[i]
            print(f"Epoch {i+1} Metrics:", flush=True)
            print(f"  Train Loss: {h['train_loss']:.4f} | Val Loss: {h['val_loss']:.4f}", flush=True)
            print(f"  Binary Acc: Train {h['train_binary_acc']:.2f}% | Val {h['val_binary_acc']:.2f}%", flush=True)
            print(f"  MCQ Acc:    Train {h['train_mcq_acc']:.2f}% | Val {h['val_mcq_acc']:.2f}%", flush=True)
            print(f"  BBox Loss:  Train {h['train_bbox_loss']:.4f} | Val {h['val_bbox_loss']:.4f}", flush=True)
            print(f"  Time:       {h['epoch_time_sec']/60:.1f} mins", flush=True)
        else:
            print(f"Epoch {i+1} Metrics: Not reached / Early stopped", flush=True)
        print("-" * 50, flush=True)

    print(f"Best Epoch:                           Epoch {best_record['epoch']}", flush=True)
    print(f"Best Validation Loss:                 {best_record['val_loss']:.4f}", flush=True)
    print(f"Best Binary Validation Accuracy:      {best_record['val_binary_acc']:.2f}%", flush=True)
    print(f"Best MCQ Validation Accuracy:         {best_record['val_mcq_acc']:.2f}%", flush=True)
    print(f"Best BBox Validation Loss:            {best_record['val_bbox_loss']:.4f}", flush=True)
    print(f"Total Epochs Completed:               {len(history)}", flush=True)
    print(f"Total Training Time (this session):   {total_training_time/60:.1f} mins ({total_training_time/3600:.2f} hours)", flush=True)
    print(f"GPU/XPU Used:                         {DEVICE}", flush=True)
    print(f"Exact Location of Best Checkpoint:    {BEST_CKPT}", flush=True)
    print(f"Exact Location of Final Model:        {FINAL_MODEL}", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
