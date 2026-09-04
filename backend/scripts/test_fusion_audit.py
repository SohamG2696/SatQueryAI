"""
SatQuery AI — Comprehensive Optical-SAR Fusion Audit Script.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

# Insert backend directory into sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models.fusion import run_module as run_fusion
from app.services.image_service import prepare_optical_tensor, prepare_sar_tensor

def audit_fusion():
    print("==================================================")
    print("1. TESTING WITH REAL LOCAL SENTINEL TIFF FILES")
    print("==================================================")

    vv_path = Path("datasets/images/S1/S1A_IW_GRDH_1SDV_20170613T165043_33UUP_70_80_VV.tif")
    vh_path = Path("datasets/images/S1/S1A_IW_GRDH_1SDV_20170613T165043_33UUP_70_80_VH.tif")
    b02_path = Path("datasets/images/S2/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_38_90_B02.tif")
    b03_path = Path("datasets/images/S2/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_38_90_B03.tif")

    opt_tensor = prepare_optical_tensor(b02_path)
    sar_tensor = prepare_sar_tensor(vv_path)

    print("OPTICAL TENSOR STATS:")
    print("  shape:", opt_tensor.shape)
    print("  dtype:", opt_tensor.dtype)
    print("  min:  ", round(opt_tensor.min().item(), 6))
    print("  max:  ", round(opt_tensor.max().item(), 6))
    print("  mean: ", round(opt_tensor.mean().item(), 6))
    print("  std:  ", round(opt_tensor.std().item(), 6))
    print("  NaNs: ", torch.isnan(opt_tensor).sum().item())
    print("  Infs: ", torch.isinf(opt_tensor).sum().item())

    print("\nSAR TENSOR STATS:")
    print("  shape:", sar_tensor.shape)
    print("  dtype:", sar_tensor.dtype)
    print("  min:  ", round(sar_tensor.min().item(), 6))
    print("  max:  ", round(sar_tensor.max().item(), 6))
    print("  mean: ", round(sar_tensor.mean().item(), 6))
    print("  std:  ", round(sar_tensor.std().item(), 6))
    print("  NaNs: ", torch.isnan(sar_tensor).sum().item())
    print("  Infs: ", torch.isinf(sar_tensor).sum().item())

    print("\n==================================================")
    print("2. TESTING INFERENCE WITH MULTIPLE QUERIES")
    print("==================================================")

    queries = [
        "Does the target area contain buildings?",
        "Are there roads in the target area?",
        "Is the area mainly forest?",
        "Is the area mainly agriculture?",
        "Compare the optical and SAR images and determine whether the target area contains buildings.",
        "Which category best describes the image: agriculture, forest, water, or urban?"
    ]

    for q in queries:
        res = run_fusion(images=[b02_path, vv_path], query=q, metadata={"modalities": ["optical", "sar"]})
        print(f"\nQuery: '{q}'")
        print(f"  Answer:      {res['answer']}")
        print(f"  Confidence:  {res['confidence']}")
        print(f"  Task Subtype:{res['parameters']['task_sub_type']}")
        print(f"  Model Name:  {res['model_name']}")
        print(f"  Probs:       {res['parameters']['probabilities']}")

    print("\n==================================================")
    print("3. TESTING BOTH MODALITY ORDERS")
    print("==================================================")

    # Order 1: Optical -> SAR
    r1 = run_fusion(
        images=[b02_path, vv_path],
        query="Compare optical and SAR images.",
        metadata={"modalities": ["optical", "sar"]}
    )

    # Order 2: SAR -> Optical
    r2 = run_fusion(
        images=[vv_path, b02_path],
        query="Compare optical and SAR images.",
        metadata={"modalities": ["sar", "optical"]}
    )

    print("Order 1 (Optical -> SAR) answer:", r1["answer"], "conf:", r1["confidence"])
    print("Order 2 (SAR -> Optical) answer:", r2["answer"], "conf:", r2["confidence"])
    assert r1["answer"] == r2["answer"] and r1["confidence"] == r2["confidence"], "Modality swap mismatch!"
    print("Modality swap verification PASSED!")

if __name__ == "__main__":
    audit_fusion()
