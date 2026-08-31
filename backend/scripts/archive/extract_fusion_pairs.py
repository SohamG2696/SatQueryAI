import pandas as pd
from pathlib import Path
import tarfile
import zstandard as zstd

BASE = Path("datasets")
CSV_FILE = BASE / "fusion_training_pairs.csv"

S1_ARCHIVE = BASE / "BigEarthNet-S1.tar.zst"
S2_ARCHIVE = BASE / "BigEarthNet-S2.tar.zst"

OUTPUT_S1 = BASE / "images" / "S1"
OUTPUT_S2 = BASE / "images" / "S2"

OUTPUT_S1.mkdir(parents=True, exist_ok=True)
OUTPUT_S2.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_FILE)

s1_ids = set(df["s1_name"].astype(str))
s2_ids = set(df["patch_id"].astype(str))


def extract_selected(archive_path, target_ids, output_dir, label):
    print(f"\nProcessing {label}...")
    found = set()

    with open(archive_path, "rb") as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:

                for member in tar:
                    if not member.isfile():
                        continue

                    parts = Path(member.name).parts

                    patch_id = None

                    for part in parts:
                        if part in target_ids:
                            patch_id = part
                            break

                    if patch_id is None:
                        continue

                    output_file = output_dir / Path(member.name).name

                    source = tar.extractfile(member)

                    if source:
                        with open(output_file, "wb") as destination:
                            destination.write(source.read())

                    found.add(patch_id)

                    if len(found) % 1000 == 0:
                        print(f"{label}: {len(found)} patches found")

    print(f"{label} completed: {len(found)} patches")


extract_selected(
    S1_ARCHIVE,
    s1_ids,
    OUTPUT_S1,
    "S1 SAR"
)

extract_selected(
    S2_ARCHIVE,
    s2_ids,
    OUTPUT_S2,
    "S2 Optical"
)

print("\nExtraction completed.")