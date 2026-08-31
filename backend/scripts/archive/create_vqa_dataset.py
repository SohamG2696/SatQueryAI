import pandas as pd
from pathlib import Path

BASE = Path("datasets")

PAIR_FILE = BASE / "fusion_training_pairs_30k.csv"
META_FILE = BASE / "BigEarthNet.txt.parquet"
OUTPUT_FILE = BASE / "fusion_vqa_30k.csv"

pairs = pd.read_csv(PAIR_FILE)

print("Loading metadata...")

meta = pd.read_parquet(
    META_FILE,
    columns=[
        "patch_id",
        "input",
        "output",
        "type",
        "category",
        "split"
    ]
)

print("Metadata rows:", len(meta))
print("Selected image pairs:", len(pairs))

selected_ids = set(pairs["patch_id"].astype(str))

meta["patch_id"] = meta["patch_id"].astype(str)

vqa = meta[meta["patch_id"].isin(selected_ids)].copy()

vqa = vqa.merge(
    pairs[["patch_id", "s1_name"]],
    on="patch_id",
    how="inner"
)

vqa = vqa[
    [
        "patch_id",
        "s1_name",
        "input",
        "output",
        "type",
        "category",
        "split"
    ]
]

vqa = vqa.dropna(subset=["input", "output"])

vqa.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("VQA dataset created.")
print("Rows:", len(vqa))
print("Saved to:", OUTPUT_FILE)
print()
print(vqa.head())