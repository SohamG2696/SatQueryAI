import numpy as np
import rasterio
from pathlib import Path
from PIL import Image

S2_DIR = Path("datasets/images/S2")
OUTPUT_DIR = S2_DIR / "previews"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def stretch_band(data):
    data = data.astype(np.float32)

    low = np.percentile(data, 2)
    high = np.percentile(data, 98)

    data = np.clip(data, low, high)

    if high > low:
        data = (data - low) / (high - low)
    else:
        data = np.zeros_like(data)

    # Slight gamma correction for better visual appearance
    data = np.power(data, 0.8)

    return (data * 255).astype(np.uint8)


def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1)


# Find all B04 files
b04_files = list(S2_DIR.glob("*_B04.tif"))

print(f"Found {len(b04_files)} S2 image groups.")
print("Creating RGB previews...\n")

created = 0
skipped = 0

for b04_path in b04_files:

    base = b04_path.name[:-len("_B04.tif")]

    b03_path = S2_DIR / f"{base}_B03.tif"
    b02_path = S2_DIR / f"{base}_B02.tif"

    if not b03_path.exists() or not b02_path.exists():
        skipped += 1
        print(f"Skipped: {base} - missing B02/B03")
        continue

    try:
        # Sentinel-2 bands
        red = read_band(b04_path)
        green = read_band(b03_path)
        blue = read_band(b02_path)

        # Contrast stretch
        red = stretch_band(red)
        green = stretch_band(green)
        blue = stretch_band(blue)

        # RGB order
        rgb = np.stack([red, green, blue], axis=-1)

        output_path = OUTPUT_DIR / f"{base}_RGB.jpg"

        Image.fromarray(rgb).save(
            output_path,
            quality=95
        )

        created += 1

        if created % 100 == 0:
            print(f"Created: {created}")

    except Exception as e:
        skipped += 1
        print(f"Error: {base} -> {e}")


print("\n--------------------------------")
print("S2 preview generation complete")
print("--------------------------------")
print("Created:", created)
print("Skipped:", skipped)
print("Output:", OUTPUT_DIR)