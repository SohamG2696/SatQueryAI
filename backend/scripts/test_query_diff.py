"""
SatQuery AI — Query Differentiation & BBox Test Script.
"""

import sys
import json
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.fusion.model import MultiTaskFusionModel
from models.fusion.inference import encode_question, tokenize

vocab_path = ROOT / "datasets" / "processed" / "vocabulary.json"
word_to_id = json.load(open(vocab_path, "r", encoding="utf-8"))["word_to_id"]

SYNONYMS = {
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
}

def normalize_query(query: str) -> str:
    tokens = tokenize(query)
    norm_toks = [SYNONYMS.get(t, t) for t in tokens]
    return " ".join(norm_toks)

def run():
    queries = [
        "Locate the roads in the image.",
        "Locate the buildings in the image.",
        "Locate the water tank in the image.",
        "Locate vegetation in the image.",
    ]

    model = MultiTaskFusionModel()
    ckpt_path = ROOT / "models" / "fusion" / "weights" / "multitask_fusion_model_final.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()

    img_opt = torch.rand(1, 4, 224, 224)
    img_sar = torch.rand(1, 2, 224, 224)

    print("=" * 70)
    print("VERIFYING QUERY DIFFERENTIATION AND BBOX PREDICTIONS")
    print("=" * 70)

    q_feats = {}
    boxes = {}

    with torch.no_grad():
        for q in queries:
            norm_q = normalize_query(q)
            q_tensor = encode_question(norm_q, word_to_id, 40)
            tok_ids = q_tensor[0][:8].tolist()
            
            out = model(img_opt, img_sar, q_tensor)
            box = [round(x, 4) for x in out["bbox"][0].tolist()]
            q_feat = out["features"]
            
            print(f"Query      : \"{q}\"")
            print(f"  Normalized : \"{norm_q}\"")
            print(f"  Token IDs  : {tok_ids}")
            print(f"  BBox       : {box}")
            q_feats[q] = q_feat
            boxes[q] = box
            print("-" * 60)

    q_keys = list(q_feats.keys())
    print("\nQUERY FEATURE SIMILARITIES & DISTANCES:")
    for i in range(len(q_keys)):
        for j in range(i + 1, len(q_keys)):
            k1, k2 = q_keys[i], q_keys[j]
            cos_sim = F.cosine_similarity(q_feats[k1], q_feats[k2]).item()
            l2_dist = torch.norm(q_feats[k1] - q_feats[k2]).item()
            bbox_diff = boxes[k1] != boxes[k2]
            print(f"\"{k1}\" vs \"{k2}\":")
            print(f"   Cosine Sim: {cos_sim:.4f} | L2 Dist: {l2_dist:.4f} | BBox Different: {bbox_diff}")

if __name__ == "__main__":
    run()
