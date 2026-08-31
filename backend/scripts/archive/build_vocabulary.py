import pandas as pd
import re
import json
from collections import Counter
from pathlib import Path

INPUT = Path("datasets/fusion_vqa_30k.csv")
OUTPUT = Path("datasets/processed/vocabulary.json")

print("Loading VQA dataset...")

df = pd.read_csv(INPUT)

counter = Counter()

for question in df["input"].fillna("").astype(str):
    tokens = re.findall(r"\b\w+\b", question.lower())
    counter.update(tokens)

word_to_id = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word, count in counter.items():
    word_to_id[word] = len(word_to_id)

vocab = {
    "word_to_id": word_to_id,
    "vocab_size": len(word_to_id),
    "max_length": 40
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)

print()
print("Vocabulary created successfully.")
print("Vocabulary size:", len(word_to_id))
print("Maximum question length:", 40)
print("Saved to:", OUTPUT)