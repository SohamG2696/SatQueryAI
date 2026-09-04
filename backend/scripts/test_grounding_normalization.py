"""
SatQuery AI — Grounding Query Normalization Validation Script.
"""

import sys
import torch
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.grounding.inference import GroundingInferenceEngine, normalize_query, encode_question

def test_normalization():
    engine = GroundingInferenceEngine(
        weights_path="models/grounding/weights/spatial_grounding_model.pth",
        vocab_path="datasets/processed/vocabulary.json",
        device=torch.device("cpu")
    )

    test_queries = [
        "Where are the roads?",
        "Where are the buildings?",
        "Where are the trees?",
        "Where is the water?",
        "Where are the vehicles?",
        "Find urban areas.",
        "Find industrial areas."
    ]

    print("==================================================")
    print("GROUNDING QUERY NORMALIZATION TEST RESULTS")
    print("==================================================")

    for q in test_queries:
        norm_q = normalize_query(q)
        q_tensor = encode_question(norm_q, engine.word_to_id, engine.max_length)
        token_ids = q_tensor[0][:8].tolist()

        print(f"\nOriginal query:    '{q}'")
        print(f"-> normalized_query: '{norm_q}'")
        print(f"-> token IDs:        {token_ids}")

        # Assert object semantic preservation
        if "roads" in q.lower():
            assert "urban" not in norm_q, f"FAILED: roads mapped to urban in '{norm_q}'"
            assert "roads" in norm_q or "road" in norm_q
        if "buildings" in q.lower():
            assert "industrial" not in norm_q, f"FAILED: buildings mapped to industrial in '{norm_q}'"
            assert "buildings" in norm_q or "building" in norm_q

    print("\n==================================================")
    print("ALL NORMALIZATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    test_normalization()
