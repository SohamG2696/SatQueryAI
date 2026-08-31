import pandas as pd
from pathlib import Path

BASE = Path("datasets")

VQA_FILE = BASE / "fusion_vqa_30k.csv"
SPLIT_DIR = BASE / "processed" / "splits"
OUTPUT_DIR = BASE / "processed" / "vqa_splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading VQA dataset...")

df = pd.read_csv(VQA_FILE)

print("Total VQA annotations:", len(df))

df["patch_id"] = df["patch_id"].astype(str)
df["s1_name"] = df["s1_name"].astype(str)

npz_files = list((BASE / "processed" / "fusion").glob("*.npz"))

print("Processed image files:", len(npz_files))

npz_map = {}

for path in npz_files:
    try:
        data = dict(__import__("numpy").load(path, allow_pickle=True))
        patch_id = str(data["patch_id"])
        npz_map[patch_id] = str(path)
    except Exception:
        pass

print("Indexed image patches:", len(npz_map))

df["image_file"] = df["patch_id"].map(npz_map)

before = len(df)

df = df.dropna(subset=["image_file"])

after = len(df)

print("Annotations with images:", after)
print("Annotations without images:", before - after)

train_files = set(
    pd.read_csv(SPLIT_DIR / "train.csv")["file"]
)

validation_files = set(
    pd.read_csv(SPLIT_DIR / "validation.csv")["file"]
)

test_files = set(
    pd.read_csv(SPLIT_DIR / "test.csv")["file"]
)

def get_split(path):
    path = str(Path(path))

    if path in train_files:
        return "train"

    if path in validation_files:
        return "validation"

    if path in test_files:
        return "test"

    return "unknown"

df["dataset_split"] = df["image_file"].apply(get_split)

df = df[df["dataset_split"] != "unknown"]

train = df[df["dataset_split"] == "train"]
validation = df[df["dataset_split"] == "validation"]
test = df[df["dataset_split"] == "test"]

train.to_csv(
    OUTPUT_DIR / "train_vqa.csv",
    index=False
)

validation.to_csv(
    OUTPUT_DIR / "validation_vqa.csv",
    index=False
)

test.to_csv(
    OUTPUT_DIR / "test_vqa.csv",
    index=False
)

print()
print("VQA splits created.")
print("Train annotations:", len(train))
print("Validation annotations:", len(validation))
print("Test annotations:", len(test))

print()
print("Task distribution:")
print(df["type"].value_counts())

print()
print("Saved to:")
print(OUTPUT_DIR)