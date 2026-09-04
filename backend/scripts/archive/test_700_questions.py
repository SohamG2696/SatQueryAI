import sys
import json
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from backend.scripts.fusion_model import MultiTaskFusionModel


BASE = Path("datasets/processed")
TEST_FILE = BASE / "vqa_splits" / "test_vqa.csv"
MODEL_FILE = BASE / "multitask_fusion_model_final.pth"
VOCAB_FILE = BASE / "vocabulary.json"

RESULT_FILE = BASE / "700_question_evaluation.csv"

TOTAL_QUESTIONS = 700
BINARY_COUNT = 350
MCQ_COUNT = 350

BATCH_SIZE = 4


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


DEVICE = get_device()


def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def encode_question(text, word_to_id, max_length):
    tokens = tokenize(text)

    ids = [
        word_to_id.get(token, word_to_id["<UNK>"])
        for token in tokens[:max_length]
    ]

    ids += [word_to_id["<PAD>"]] * (max_length - len(ids))

    return torch.tensor(ids, dtype=torch.long)


class EvaluationDataset(Dataset):

    def __init__(self, dataframe, word_to_id, max_length):

        self.df = dataframe.reset_index(drop=True)
        self.word_to_id = word_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = Path(row["image_file"])

        data = np.load(
            image_path,
            allow_pickle=True
        )

        s1 = data["s1"].astype(np.float32) / 255.0
        s2 = data["s2"].astype(np.float32) / 255.0

        question = encode_question(
            row["input"],
            self.word_to_id,
            self.max_length
        )

        task_type = str(row["type"]).strip().lower()

        if task_type == "binary":

            answer = str(row["output"]).strip().lower()

            target = 1 if answer == "yes" else 0

        elif task_type == "mcq":

            answer = str(row["output"]).strip().lower()

            mapping = {
                "a": 0,
                "b": 1,
                "c": 2,
                "d": 3
            }

            target = mapping.get(answer, 0)

        else:
            target = 0

        return {
            "s1": torch.tensor(s1, dtype=torch.float32),
            "s2": torch.tensor(s2, dtype=torch.float32),
            "question": question,
            "target": torch.tensor(target, dtype=torch.long),
            "type": task_type,
            "question_text": str(row["input"]),
            "answer_text": str(row["output"]),
            "image_file": str(row["image_file"])
        }


def collate_fn(batch):

    return {
        "s1": torch.stack([x["s1"] for x in batch]),
        "s2": torch.stack([x["s2"] for x in batch]),
        "question": torch.stack([x["question"] for x in batch]),
        "target": torch.stack([x["target"] for x in batch]),
        "type": [x["type"] for x in batch],
        "question_text": [x["question_text"] for x in batch],
        "answer_text": [x["answer_text"] for x in batch],
        "image_file": [x["image_file"] for x in batch]
    }


def load_model(vocab_size):

    print("\nLoading model...", flush=True)

    model = MultiTaskFusionModel(
        vocab_size=vocab_size
    ).to(DEVICE)

    state = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=False
    )

    if "model_state_dict" in state:
        model.load_state_dict(
            state["model_state_dict"]
        )
    else:
        model.load_state_dict(state)

    model.eval()

    print("Model weights loaded successfully.", flush=True)

    return model


def binary_prediction(logits):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    prediction = probabilities.argmax(
        dim=1
    )

    confidence = probabilities.max(
        dim=1
    ).values

    return prediction, confidence


def mcq_prediction(logits):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    prediction = probabilities.argmax(
        dim=1
    )

    confidence = probabilities.max(
        dim=1
    ).values

    return prediction, confidence


def main():

    print("=" * 75)
    print("       SATQUERY AI - 700 QUESTION GENERALIZATION TEST")
    print("=" * 75)

    print(f"Device:       {DEVICE}")
    print(f"Model:        {MODEL_FILE}")
    print(f"Test dataset: {TEST_FILE}")

    if not TEST_FILE.exists():
        print("\nERROR: Test dataset not found.")
        return

    if not MODEL_FILE.exists():
        print("\nERROR: Trained model not found.")
        return

    with open(
        VOCAB_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        vocab = json.load(f)

    word_to_id = vocab["word_to_id"]
    max_length = vocab["max_length"]
    vocab_size = vocab["vocab_size"]

    print(f"Vocabulary Size: {vocab_size}")

    print("\nLoading test dataset...", flush=True)

    df = pd.read_csv(TEST_FILE)

    df = df.dropna(
        subset=[
            "image_file",
            "input",
            "output",
            "type"
        ]
    ).reset_index(drop=True)

    df["type_clean"] = (
        df["type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    binary_df = df[
        df["type_clean"] == "binary"
    ].copy()

    mcq_df = df[
        df["type_clean"] == "mcq"
    ].copy()

    print(f"Binary test annotations: {len(binary_df)}")
    print(f"MCQ test annotations:     {len(mcq_df)}")

    if len(binary_df) < BINARY_COUNT:
        print("\nERROR: Not enough binary questions.")
        return

    if len(mcq_df) < MCQ_COUNT:
        print("\nERROR: Not enough MCQ questions.")
        return

    print("\nSelecting 700 questions...")

    binary_sample = binary_df.sample(
        n=BINARY_COUNT,
        random_state=2026
    )

    mcq_sample = mcq_df.sample(
        n=MCQ_COUNT,
        random_state=2026
    )

    evaluation_df = pd.concat(
        [
            binary_sample,
            mcq_sample
        ],
        ignore_index=True
    )

    evaluation_df = evaluation_df.sample(
        frac=1,
        random_state=2026
    ).reset_index(drop=True)

    print(
        f"Selected {len(evaluation_df)} questions."
    )

    print("\nQuestion distribution:")
    print(
        evaluation_df["type_clean"]
        .value_counts()
        .to_string()
    )

    dataset = EvaluationDataset(
        evaluation_df,
        word_to_id,
        max_length
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = load_model(vocab_size)

    print("\n" + "=" * 75)
    print("                    RUNNING 700 TESTS")
    print("=" * 75)

    results = []

    total_correct = 0
    total_questions = 0

    binary_correct = 0
    binary_total = 0

    mcq_correct = 0
    mcq_total = 0

    total_confidence = 0.0

    start_time = time.time()

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader):

            s1 = batch["s1"].to(DEVICE)
            s2 = batch["s2"].to(DEVICE)
            question = batch["question"].to(DEVICE)

            outputs = model(
                optical=s2,
                sar=s1,
                question=question
            )

            binary_mask = [
                i
                for i, t in enumerate(batch["type"])
                if t == "binary"
            ]

            mcq_mask = [
                i
                for i, t in enumerate(batch["type"])
                if t == "mcq"
            ]

            if binary_mask:

                indices = torch.tensor(
                    binary_mask,
                    device=DEVICE
                )

                predictions, confidences = binary_prediction(
                    outputs["binary"][indices]
                )

                for local_idx, original_idx in enumerate(binary_mask):

                    predicted = predictions[
                        local_idx
                    ].item()

                    confidence = confidences[
                        local_idx
                    ].item() * 100

                    target = batch["target"][
                        original_idx
                    ].item()

                    correct = predicted == target

                    binary_total += 1

                    if correct:
                        binary_correct += 1

                    total_confidence += confidence

                    results.append({
                        "question_number": 0,
                        "image_file": batch["image_file"][original_idx],
                        "type": "binary",
                        "question": batch["question_text"][original_idx],
                        "expected_answer": "YES" if target == 1 else "NO",
                        "predicted_answer": "YES" if predicted == 1 else "NO",
                        "confidence": round(confidence, 2),
                        "correct": correct
                    })

            if mcq_mask:

                indices = torch.tensor(
                    mcq_mask,
                    device=DEVICE
                )

                predictions, confidences = mcq_prediction(
                    outputs["mcq"][indices]
                )

                for local_idx, original_idx in enumerate(mcq_mask):

                    predicted = predictions[
                        local_idx
                    ].item()

                    confidence = confidences[
                        local_idx
                    ].item() * 100

                    target = batch["target"][
                        original_idx
                    ].item()

                    correct = predicted == target

                    mcq_total += 1

                    if correct:
                        mcq_correct += 1

                    total_confidence += confidence

                    results.append({
                        "question_number": 0,
                        "image_file": batch["image_file"][original_idx],
                        "type": "mcq",
                        "question": batch["question_text"][original_idx],
                        "expected_answer": chr(ord("A") + target),
                        "predicted_answer": chr(ord("A") + predicted),
                        "confidence": round(confidence, 2),
                        "correct": correct
                    })

            total_questions = (
                binary_total + mcq_total
            )

            if total_questions % 100 < BATCH_SIZE:

                elapsed = time.time() - start_time

                accuracy = (
                    100.0 * (binary_correct + mcq_correct)
                    / max(total_questions, 1)
                )

                print(
                    f"Processed: {total_questions}/700 | "
                    f"Accuracy: {accuracy:.2f}% | "
                    f"Elapsed: {elapsed / 60:.1f} min",
                    flush=True
                )

    # Number results properly
    for i, result in enumerate(results, start=1):
        result["question_number"] = i

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8"
    )

    total_correct = (
        binary_correct +
        mcq_correct
    )

    overall_accuracy = (
        100.0 * total_correct /
        max(total_questions, 1)
    )

    binary_accuracy = (
        100.0 * binary_correct /
        max(binary_total, 1)
    )

    mcq_accuracy = (
        100.0 * mcq_correct /
        max(mcq_total, 1)
    )

    average_confidence = (
        total_confidence /
        max(total_questions, 1)
    )

    elapsed = time.time() - start_time

    print("\n")
    print("=" * 75)
    print("                 FINAL 700 QUESTION RESULTS")
    print("=" * 75)

    print(
        f"Total Questions:        {total_questions}"
    )

    print(
        f"Correct:                {total_correct}"
    )

    print(
        f"Incorrect:              {total_questions - total_correct}"
    )

    print(
        f"Overall Accuracy:       {overall_accuracy:.2f}%"
    )

    print("-" * 75)

    print(
        f"Binary Questions:       {binary_total}"
    )

    print(
        f"Binary Correct:         {binary_correct}"
    )

    print(
        f"Binary Incorrect:       "
        f"{binary_total - binary_correct}"
    )

    print(
        f"Binary Accuracy:        {binary_accuracy:.2f}%"
    )

    print("-" * 75)

    print(
        f"MCQ Questions:          {mcq_total}"
    )

    print(
        f"MCQ Correct:            {mcq_correct}"
    )

    print(
        f"MCQ Incorrect:          "
        f"{mcq_total - mcq_correct}"
    )

    print(
        f"MCQ Accuracy:           {mcq_accuracy:.2f}%"
    )

    print("-" * 75)

    print(
        f"Average Confidence:     "
        f"{average_confidence:.2f}%"
    )

    print(
        f"Evaluation Time:        "
        f"{elapsed / 60:.2f} minutes"
    )

    print("-" * 75)

    print(
        f"Detailed Results:        {RESULT_FILE}"
    )

    print("=" * 75)

    print("\n700-question evaluation complete.")


if __name__ == "__main__":
    main()