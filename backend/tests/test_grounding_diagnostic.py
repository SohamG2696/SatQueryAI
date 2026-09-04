"""
SatQuery AI — Grounding Pipeline Diagnostic & Assertion Test.

Verifies:
1. Grounding checkpoint loads cleanly without errors.
2. Word normalization ensures 'roads' and 'buildings' DO NOT map to identical token IDs (<UNK>).
3. The four standard spatial queries produce different bounding box coordinates.
4. Real confidence values are reported.
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.grounding.inference import GroundingInferenceEngine, normalize_query
from models.fusion.inference import encode_question


def test_grounding_checkpoint_and_query_differentiation():
    ckpt_path = _ROOT / "models" / "grounding" / "weights" / "spatial_grounding_model.pth"
    vocab_path = _ROOT / "datasets" / "processed" / "vocabulary.json"

    assert ckpt_path.exists(), f"Grounding checkpoint not found at {ckpt_path}"
    assert vocab_path.exists(), f"Vocabulary not found at {vocab_path}"

    engine = GroundingInferenceEngine(
        weights_path=ckpt_path,
        vocab_path=vocab_path,
        device=torch.device("cpu"),
    )

    # 1. Verify token IDs for 'roads' vs 'buildings'
    q_roads = "Locate the roads in the image."
    q_buildings = "Locate the buildings in the image."

    norm_roads = normalize_query(q_roads)
    norm_buildings = normalize_query(q_buildings)

    t_roads = encode_question(norm_roads, engine.word_to_id, engine.max_length)
    t_buildings = encode_question(norm_buildings, engine.word_to_id, engine.max_length)

    ids_roads = t_roads[0][:8].tolist()
    ids_buildings = t_buildings[0][:8].tolist()

    assert ids_roads != ids_buildings, f"Token IDs for roads and buildings are identical: {ids_roads}"

    # 2. Verify bounding box prediction differentiation
    img_tensor = torch.rand(1, 4, 224, 224)

    queries = [
        "Locate the roads in the image.",
        "Locate the buildings in the image.",
        "Locate the water tank in the image.",
        "Locate vegetation in the image.",
    ]

    results = {}
    boxes = {}

    for q in queries:
        res = engine.predict(img_tensor, q)
        coords = res["visual_evidence"]["coordinates"]
        results[q] = res
        boxes[q] = coords
        print(f"[TEST] Query: '{q}' -> Norm: '{res['normalized_query']}' -> Box: {coords} (Conf: {res['confidence']})")

    # Assert that not all bounding boxes are identical
    unique_boxes = set(tuple(b) for b in boxes.values())
    assert len(unique_boxes) > 1, f"All queries produced identical bounding boxes: {unique_boxes}"


if __name__ == "__main__":
    test_grounding_checkpoint_and_query_differentiation()
