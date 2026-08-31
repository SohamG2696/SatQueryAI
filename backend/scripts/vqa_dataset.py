import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


BASE = Path("datasets/processed")

VOCAB_FILE = BASE / "vocabulary.json"
TRAIN_FILE = BASE / "vqa_splits" / "train_vqa.csv"
VAL_FILE = BASE / "vqa_splits" / "validation_vqa.csv"
TEST_FILE = BASE / "vqa_splits" / "test_vqa.csv"


with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

WORD_TO_ID = vocab_data["word_to_id"]
MAX_LENGTH = vocab_data["max_length"]

PAD_ID = WORD_TO_ID["<PAD>"]
UNK_ID = WORD_TO_ID["<UNK>"]


def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def encode_question(text):
    tokens = tokenize(text)

    ids = [
        WORD_TO_ID.get(token, UNK_ID)
        for token in tokens[:MAX_LENGTH]
    ]

    ids += [PAD_ID] * (MAX_LENGTH - len(ids))

    return ids


class VQADataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        self.df = self.df.dropna(
            subset=["image_file", "input", "output", "type"]
        ).reset_index(drop=True)

        print(
            f"Loaded {len(self.df)} annotations from "
            f"{csv_file.name}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        data = np.load(
            row["image_file"],
            allow_pickle=True
        )

        s1 = data["s1"].astype(np.float32)
        s2 = data["s2"].astype(np.float32)

        s1 = s1 / 255.0
        s2 = s2 / 255.0

        question = torch.tensor(
            encode_question(row["input"]),
            dtype=torch.long
        )

        task_type = row["type"]

        if task_type == "binary":

            answer = str(row["output"]).strip().lower()

            target = 1 if answer == "yes" else 0

            target = torch.tensor(
                target,
                dtype=torch.long
            )

        elif task_type == "mcq":

            answer = str(row["output"]).strip().lower()

            mapping = {
                "a": 0,
                "b": 1,
                "c": 2,
                "d": 3
            }

            target = torch.tensor(
                mapping.get(answer, 0),
                dtype=torch.long
            )

        elif task_type == "bounding box":

            values = re.findall(
                r"[-+]?\d*\.?\d+",
                str(row["output"])
            )

            values = [float(x) for x in values[:4]]

            while len(values) < 4:
                values.append(0.0)

            target = torch.tensor(
                values,
                dtype=torch.float32
            )

        else:

            target = torch.tensor(
                0,
                dtype=torch.long
            )

        return {
            "s1": torch.tensor(s1, dtype=torch.float32),
            "s2": torch.tensor(s2, dtype=torch.float32),
            "question": question,
            "target": target,
            "type": task_type,
            "question_text": str(row["input"]),
            "answer_text": str(row["output"])
        }


def collate_fn(batch):

    return {
        "s1": torch.stack([item["s1"] for item in batch]),
        "s2": torch.stack([item["s2"] for item in batch]),
        "question": torch.stack([item["question"] for item in batch]),
        "target": [item["target"] for item in batch],
        "type": [item["type"] for item in batch],
        "question_text": [item["question_text"] for item in batch],
        "answer_text": [item["answer_text"] for item in batch]
    }




def create_loaders(batch_size=8):

    train_dataset = VQADataset(TRAIN_FILE)
    val_dataset = VQADataset(VAL_FILE)
    test_dataset = VQADataset(TEST_FILE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    print("Creating VQA data loaders...")

    train_loader, val_loader, test_loader = create_loaders(
        batch_size=2
    )

    batch = next(iter(train_loader))

    print()
    print("DataLoader test successful.")
    print("S1 shape:", batch["s1"].shape)
    print("S2 shape:", batch["s2"].shape)
    print("Question shape:", batch["question"].shape)
    print("Task types:", batch["type"])

    print()
    print("Example question:")
    print(batch["question_text"][0])

    print()
    print("Example answer:")
    print(batch["answer_text"][0])