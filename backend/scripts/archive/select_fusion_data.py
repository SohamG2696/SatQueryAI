import pandas as pd
from pathlib import Path

DATASET = Path("datasets/BigEarthNet.txt.parquet")
OUTPUT = Path("datasets/fusion_training_pairs.csv")

df = pd.read_parquet(DATASET)

train = df[df["split"] == "train"].copy()

patches = (
    train[["patch_id", "s1_name"]]
    .drop_duplicates(subset=["patch_id"])
    .sample(n=50000, random_state=42)
)

patches.to_csv(OUTPUT, index=False)

print(f"Selected {len(patches)} unique image pairs")
print(f"Saved to: {OUTPUT}")
print(patches.head(10).to_string(index=False))