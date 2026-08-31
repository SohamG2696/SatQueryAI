import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

INPUT = Path("datasets/processed/fusion")
OUTPUT = Path("datasets/processed/splits")
OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT.glob("*.npz"))

print("Total samples:", len(files))

train, temp = train_test_split(
    files,
    test_size=0.20,
    random_state=42
)

validation, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42
)

def save_split(name, items):
    df = pd.DataFrame({
        "file": [str(x) for x in items]
    })
    df.to_csv(OUTPUT / f"{name}.csv", index=False)
    print(f"{name}: {len(items)}")

save_split("train", train)
save_split("validation", validation)
save_split("test", test)

print("\nSplits created successfully.")