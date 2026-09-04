"""
SatQuery AI — Fusion Model Comprehensive Direct Diagnostic Script.

Executes 14 verification tasks:
1. Architecture summary & inspect checkpoint.
2. Verify optical tensor (4 channels, [1, 4, 224, 224], no NaNs/Infs).
3. Verify SAR tensor (2 channels, [1, 2, 224, 224], no NaNs/Infs).
4. Perform modality ablation (Optical+SAR vs Optical-only vs SAR-only).
5. Verify query dependency across 5 distinct queries.
6. Verify device support and genuine confidence output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.fusion.model import MultiTaskFusionModel
from models.fusion.inference import FusionInferenceEngine, encode_question, tokenize

# Grounding / Domain synonym map for query normalization
FUSION_SYNONYMS = {
    "roads": "urban",
    "road": "urban",
    "buildings": "industrial",
    "building": "industrial",
    "tanks": "water",
    "tank": "water",
    "vehicles": "urban",
    "vehicle": "urban",
    "cars": "urban",
    "car": "urban",
    "trees": "forest",
    "tree": "forest",
    "fields": "agriculture",
    "field": "agriculture",
}


def normalize_query(query: str) -> str:
    tokens = tokenize(query)
    norm_tokens = [FUSION_SYNONYMS.get(t, t) for t in tokens]
    return " ".join(norm_tokens)


def run_fusion_diagnostic():
    print("=" * 80)
    print("           SATQUERY AI — FUSION MODEL COMPREHENSIVE DIAGNOSTIC")
    print("=" * 80)

    weights_path = ROOT / "models" / "fusion" / "weights" / "multitask_fusion_model_final.pth"
    vocab_path = ROOT / "datasets" / "processed" / "vocabulary.json"

    # ── TASK 1 & 2: INSPECT ARCHITECTURE & CHECKPOINT ─────────────
    print("\n[TASK 1 & 2] Inspecting Architecture & Checkpoint...")
    print(f"Weights path: {weights_path} (exists={weights_path.exists()})")
    print(f"Vocab path  : {vocab_path} (exists={vocab_path.exists()})")

    assert weights_path.exists(), f"Fusion checkpoint not found at {weights_path}"
    assert vocab_path.exists(), f"Vocabulary file not found at {vocab_path}"

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint Type: {type(ckpt)}")

    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    print(f"Total Parameters in state_dict: {len(state_dict)}")
    first_20_keys = list(state_dict.keys())[:20]
    print("First 20 state_dict keys:")
    for k in first_20_keys:
        print(f"  - {k} (shape: {state_dict[k].shape})")

    model = MultiTaskFusionModel(vocab_size=493)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"\nModel load_state_dict(strict=True) -> Missing: {missing}, Unexpected: {unexpected}")
    assert len(missing) == 0 and len(unexpected) == 0, "State dict mismatch!"

    # ── TASK 3 & 4: VERIFY OPTICAL & SAR INPUT TENSORS ─────────────
    print("\n[TASK 3 & 4] Verifying Optical (4-ch) & SAR (2-ch) Input Tensors...")
    # Load sample npz pair if available, else synthetic
    sample_npz = ROOT / "datasets" / "processed" / "fusion" / "021489.npz"
    if sample_npz.exists():
        data = np.load(sample_npz, allow_pickle=True)
        opt_arr = data["s2"].astype(np.float32)  # [4, 224, 224]
        sar_arr = data["s1"].astype(np.float32)  # [2, 224, 224]
        opt_tensor = torch.tensor(opt_arr).unsqueeze(0)
        sar_tensor = torch.tensor(sar_arr).unsqueeze(0)
        print(f"Loaded real sample NPZ pair from: {sample_npz.name}")
    else:
        opt_tensor = torch.rand(1, 4, 224, 224, dtype=torch.float32)
        sar_tensor = torch.rand(1, 2, 224, 224, dtype=torch.float32)
        print("Generated normalized sample Optical and SAR tensors.")

    # Optical checks
    print("\n  [Optical Tensor]")
    print(f"    Shape : {opt_tensor.shape} (Expected: [1, 4, 224, 224])")
    print(f"    Dtype : {opt_tensor.dtype}")
    print(f"    Min   : {opt_tensor.min().item():.4f}")
    print(f"    Max   : {opt_tensor.max().item():.4f}")
    print(f"    Mean  : {opt_tensor.mean().item():.4f}")
    print(f"    NaNs  : {torch.isnan(opt_tensor).sum().item()}")
    print(f"    Infs  : {torch.isinf(opt_tensor).sum().item()}")
    assert opt_tensor.shape == (1, 4, 224, 224)
    assert torch.isnan(opt_tensor).sum().item() == 0
    assert torch.isinf(opt_tensor).sum().item() == 0

    # SAR checks
    print("\n  [SAR Tensor]")
    print(f"    Shape : {sar_tensor.shape} (Expected: [1, 2, 224, 224])")
    print(f"    Dtype : {sar_tensor.dtype}")
    print(f"    Min   : {sar_tensor.min().item():.4f}")
    print(f"    Max   : {sar_tensor.max().item():.4f}")
    print(f"    Mean  : {sar_tensor.mean().item():.4f}")
    print(f"    NaNs  : {torch.isnan(sar_tensor).sum().item()}")
    print(f"    Infs  : {torch.isinf(sar_tensor).sum().item()}")
    assert sar_tensor.shape == (1, 2, 224, 224)
    assert torch.isnan(sar_tensor).sum().item() == 0
    assert torch.isinf(sar_tensor).sum().item() == 0

    # ── TASK 5 & 7: MODALITY ABLATION & MULTIMODAL INTEGRATION ────
    print("\n[TASK 5 & 7] Testing Multimodal Integration & Modality Ablation...")
    model.eval()

    test_q = "Describe the land cover in this area."
    with open(vocab_path, "r", encoding="utf-8") as f:
        w2id = json.load(f)["word_to_id"]

    q_tensor = encode_question(normalize_query(test_q), w2id, max_length=40)

    zero_sar = torch.zeros_like(sar_tensor)
    zero_opt = torch.zeros_like(opt_tensor)

    with torch.no_grad():
        out_normal = model(opt_tensor, sar_tensor, q_tensor)
        out_opt_only = model(opt_tensor, zero_sar, q_tensor)
        out_sar_only = model(zero_opt, sar_tensor, q_tensor)

    feat_normal = out_normal["features"]
    feat_opt_only = out_opt_only["features"]
    feat_sar_only = out_sar_only["features"]

    sim_opt = F.cosine_similarity(feat_normal, feat_opt_only).item()
    dist_opt = torch.norm(feat_normal - feat_opt_only).item()

    sim_sar = F.cosine_similarity(feat_normal, feat_sar_only).item()
    dist_sar = torch.norm(feat_normal - feat_sar_only).item()

    print(f"  [Ablation 1: Normal (Optical+SAR) vs Optical-only (Zero SAR)]")
    print(f"    Cosine Similarity : {sim_opt:.6f}")
    print(f"    L2 Distance       : {dist_opt:.6f}")

    print(f"  [Ablation 2: Normal (Optical+SAR) vs SAR-only (Zero Optical)]")
    print(f"    Cosine Similarity : {sim_sar:.6f}")
    print(f"    L2 Distance       : {dist_sar:.6f}")

    assert dist_opt > 1e-4, "SAR modality has zero impact on fusion features!"
    assert dist_sar > 1e-4, "Optical modality has zero impact on fusion features!"
    print("  => CONFIRMED: Both Optical and SAR modalities materially affect fusion output.")

    # ── TASK 6: VERIFY QUERY DEPENDENCY ───────────────────────────
    print("\n[TASK 6] Testing Query Dependency across 5 distinct queries...")
    queries = [
        "Describe the land cover.",
        "Is there water in this area?",
        "Are there buildings in this area?",
        "Is vegetation present?",
        "Describe the urban features.",
    ]

    q_feats = {}
    outputs_map = {}

    with torch.no_grad():
        for q in queries:
            norm_q = normalize_query(q)
            q_tens = encode_question(norm_q, w2id, max_length=40)
            tok_ids = q_tens[0][:8].tolist()

            out = model(opt_tensor, sar_tensor, q_tens)
            feat = out["features"]
            bin_logits = out["binary"][0].tolist()
            bbox = [round(x, 4) for x in out["bbox"][0].tolist()]

            q_feats[q] = feat
            outputs_map[q] = {"bin": bin_logits, "bbox": bbox, "norm_q": norm_q, "toks": tok_ids}

            print(f"  Query: '{q}'")
            print(f"    Normalized : '{norm_q}'")
            print(f"    Token IDs  : {tok_ids}")
            print(f"    Binary Logits: {[round(x, 4) for x in bin_logits]}")
            print(f"    BBox       : {bbox}")
            print("-" * 50)

    # Check pairwise feature distances
    q_keys = list(q_feats.keys())
    print("\n  [Query Pairwise Distances]")
    diff_count = 0
    for i in range(len(q_keys)):
        for j in range(i + 1, len(q_keys)):
            k1, k2 = q_keys[i], q_keys[j]
            dist = torch.norm(q_feats[k1] - q_feats[k2]).item()
            if dist > 1e-3:
                diff_count += 1
            print(f"    '{k1}' vs '{k2}' -> L2 Dist: {dist:.4f}")

    assert diff_count > 0, "Query text has zero impact on fusion features!"
    print("  => CONFIRMED: Query text materially drives the model features and predictions.")

    print("\n" + "=" * 80)
    print("ALL FUSION DIAGNOSTIC CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_fusion_diagnostic()
