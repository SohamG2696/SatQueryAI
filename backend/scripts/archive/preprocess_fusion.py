import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from PIL import Image

BASE = Path("datasets")

CSV_FILE = BASE / "fusion_training_pairs_30k.csv"
S1_DIR = BASE / "images" / "S1"
S2_DIR = BASE / "images" / "S2"

OUTPUT = BASE / "processed" / "fusion"
OUTPUT.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224

df = pd.read_csv(CSV_FILE)

s1_files = {}

for path in S1_DIR.glob("*.tif"):
    name = path.name

    if name.endswith("_VV.tif") or name.endswith("_VH.tif"):
        base = name.rsplit("_", 1)[0]
        s1_files.setdefault(base, {})[name[-6:-4]] = path

s2_files = {}

for path in S2_DIR.glob("*.tif"):
    name = path.name

    for band in ["B02", "B03", "B04", "B08"]:
        suffix = f"_{band}.tif"

        if name.endswith(suffix):
            base = name[:-len(suffix)]
            s2_files.setdefault(base, {})[band] = path
            break

print("File index created.")
print("S1 groups:", len(s1_files))
print("S2 groups:", len(s2_files))
print("Starting preprocessing...")

def normalize(data):
    data = data.astype(np.float32)

    low = np.percentile(data, 2)
    high = np.percentile(data, 98)

    if high <= low:
        return np.zeros_like(data, dtype=np.uint8)

    data = np.clip(data, low, high)
    data = (data - low) / (high - low)

    return (data * 255).astype(np.uint8)

def read_resize(path):
    with rasterio.open(path) as src:
        data = src.read(1)

    data = normalize(data)

    image = Image.fromarray(data)
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    return np.array(image)

processed = 0
skipped = 0

for i, row in df.iterrows():

    patch_id = str(row["patch_id"])
    s1_name = str(row["s1_name"])

    s1 = s1_files.get(s1_name, {})
    s2 = s2_files.get(patch_id, {})

    if not all(x in s1 for x in ["VV", "VH"]):
        skipped += 1
        continue

    if not all(x in s2 for x in ["B02", "B03", "B04", "B08"]):
        skipped += 1
        continue

    try:
        vv = read_resize(s1["VV"])
        vh = read_resize(s1["VH"])

        b02 = read_resize(s2["B02"])
        b03 = read_resize(s2["B03"])
        b04 = read_resize(s2["B04"])
        b08 = read_resize(s2["B08"])

        s1_data = np.stack([vv, vh])
        s2_data = np.stack([b02, b03, b04, b08])

        np.savez_compressed(
            OUTPUT / f"{i:06d}.npz",
            s1=s1_data,
            s2=s2_data,
            patch_id=patch_id,
            s1_name=s1_name
        )

        processed += 1

        if processed % 100 == 0:
            print(
                f"Processed: {processed}/{len(df)} | "
                f"Skipped: {skipped}"
            )

    except Exception as e:
        skipped += 1
        print(f"Error at {i}: {e}")

print()
print("Preprocessing completed.")
print("Processed:", processed)
print("Skipped:", skipped)
print("Output:", OUTPUT)