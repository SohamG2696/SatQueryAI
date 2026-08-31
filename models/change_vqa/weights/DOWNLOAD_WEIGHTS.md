# Checkpoint Weights

The trained ChangeFormerV6 checkpoint is excluded from Git via `.gitignore` (`*.pth`).

## Required file

```
models/change_vqa/weights/checkpoint_best.pth
```

## Details

| Field | Value |
|---|---|
| Model | ChangeFormerV6 |
| embed_dim | 256 |
| Parameters | ~41M |
| Trained on | SECOND dataset (2,968 image pairs) |
| Training epochs | 50 |
| Best Val IoU | 0.4678 |
| F1 Score | 0.6374 |
| Precision | 0.6610 |
| Recall | 0.6154 |
| Pixel Accuracy | 0.8543 |

## How to obtain

Copy from the SIH2026 training repository:

```powershell
Copy-Item "C:\Users\hp\OneDrive\Desktop\SIH2026\checkpoints\full\checkpoint_best.pth" `
          "models\change_vqa\weights\checkpoint_best.pth"
```

Or download from the shared team drive link (ask Person B).
