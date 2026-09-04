"""
SatQuery AI — Grounding Direct Diagnostic Script.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.grounding.inference import GroundingInferenceEngine
from models.fusion.inference import encode_question

def run_test():
    weights_path = ROOT / "models" / "grounding" / "weights" / "checkpoint_best.pth"
    vocab_path = ROOT / "datasets" / "processed" / "vocabulary.json"

    print("=" * 70)
    print("GROUNDING DIRECT DIAGNOSTIC TEST")
    print("=" * 70)
    print(f"Weights path : {weights_path} (exists={weights_path.exists()})")
    print(f"Vocab path   : {vocab_path} (exists={vocab_path.exists()})")

    engine = GroundingInferenceEngine(
        weights_path=weights_path if weights_path.exists() else None,
        vocab_path=vocab_path,
        device=torch.device("cpu"),
    )

    print(f"Engine class : {engine.__class__.__module__}.{engine.__class__.__name__}")
    print(f"Model class  : {engine.model.__class__.__module__}.{engine.model.__class__.__name__}")

    queries = [
        "Locate the roads in the image.",
        "Locate the buildings in the image.",
        "Locate the water tank in the image.",
        "Locate vegetation in the image.",
    ]

    img_tensor = torch.rand(1, 4, 224, 224)

    q_feats = {}

    engine.model.eval()
    with torch.no_grad():
        for q in queries:
            print("-" * 60)
            print(f"Query: '{q}'")
            q_tokens = encode_question(q, engine.word_to_id, engine.max_length, device=engine.device)
            token_ids = q_tokens[0][:10].tolist()
            print(f"  First 10 Token IDs: {token_ids}")
            
            # Words breakdown
            toks = q.lower().replace(".", "").split()
            word_id_breakdown = {w: engine.word_to_id.get(w, engine.word_to_id.get("<UNK>", 1)) for w in toks}
            print(f"  Word -> ID map    : {word_id_breakdown}")

            # Question features
            emb = engine.model.embedding(q_tokens)
            _, hidden = engine.model.gru(emb)
            q_feat = hidden[-1]  # [1, 128]
            q_feats[q] = q_feat

            res = engine.predict(img_tensor, q)
            print(f"  Raw Box           : {res['visual_evidence']['coordinates']}")
            print(f"  Confidence        : {res['confidence']}")

    print("\n" + "=" * 70)
    print("QUERY EMBEDDING / FEATURE DIFFERENCE (Cosine Similarity & L2 Dist)")
    print("=" * 70)
    q_keys = list(q_feats.keys())
    for i in range(len(q_keys)):
        for j in range(i + 1, len(q_keys)):
            k1, k2 = q_keys[i], q_keys[j]
            f1, f2 = q_feats[k1], q_feats[k2]
            cos_sim = F.cosine_similarity(f1, f2).item()
            l2_dist = torch.norm(f1 - f2).item()
            print(f"'{k1}' vs '{k2}':")
            print(f"   Cosine Sim: {cos_sim:.6f} | L2 Dist: {l2_dist:.6f}")

if __name__ == "__main__":
    run_test()
