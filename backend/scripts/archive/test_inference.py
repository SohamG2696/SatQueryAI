import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from fusion_model import MultiTaskFusionModel


BASE = Path("datasets/processed")
MODEL_PATH = BASE / "multitask_fusion_model_final.pth"
VOCAB_PATH = BASE / "vocabulary.json"


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def encode_question(text, word_to_id, max_length):
    tokens = tokenize(text)

    ids = [
        word_to_id.get(token, word_to_id["<UNK>"])
        for token in tokens[:max_length]
    ]

    ids += [word_to_id["<PAD>"]] * (max_length - len(ids))

    return torch.tensor([ids], dtype=torch.long)


def load_model(device):
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    model = MultiTaskFusionModel(
        vocab_size=vocab["vocab_size"]
    ).to(device)

    state = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()

    return model, vocab


def load_image_pair(image_path, device):
    data = np.load(image_path, allow_pickle=True)

    s1 = data["s1"].astype(np.float32) / 255.0
    s2 = data["s2"].astype(np.float32) / 255.0

    s1 = torch.tensor(s1, dtype=torch.float32).unsqueeze(0).to(device)
    s2 = torch.tensor(s2, dtype=torch.float32).unsqueeze(0).to(device)

    return s1, s2


def binary_prediction(logits):
    probabilities = torch.softmax(logits, dim=1)
    prediction = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, prediction].item() * 100

    answer = "YES" if prediction == 1 else "NO"

    return answer, confidence


def mcq_prediction(logits):
    probabilities = torch.softmax(logits, dim=1)
    prediction = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, prediction].item() * 100

    option = chr(ord("A") + prediction)

    return option, confidence


def bbox_prediction(output):
    box = output[0].detach().cpu().numpy()

    return box


def main():
    device = get_device()

    print("=" * 70)
    print("             SatQuery AI - Model Inference Test")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model:  {MODEL_PATH}")

    model, vocab = load_model(device)

    print("Model loaded successfully.")

    default_image = BASE / "fusion" / "021489.npz"

    image_input = input(
        f"\nEnter image path [{default_image}]: "
    ).strip()

    if image_input:
        image_path = Path(image_input)
    else:
        image_path = default_image

    if not image_path.exists():
        print(f"\nERROR: Image file not found:")
        print(image_path)
        return

    s1, s2 = load_image_pair(image_path, device)

    print("\nImage loaded successfully.")
    print(f"S1 shape: {tuple(s1.shape)}")
    print(f"S2 shape: {tuple(s2.shape)}")

    print("\nChoose task:")
    print("1. Binary YES/NO")
    print("2. MCQ")
    print("3. Bounding Box")

    task = input("\nTask [1]: ").strip()

    if not task:
        task = "1"

    question = input("\nEnter your question: ").strip()

    if not question:
        print("ERROR: Question cannot be empty.")
        return

    question_tensor = encode_question(
        question,
        vocab["word_to_id"],
        vocab["max_length"]
    ).to(device)

    with torch.no_grad():
        outputs = model(
            optical=s2,
            sar=s1,
            question=question_tensor
        )

    print("\n" + "=" * 70)
    print("                         RESULT")
    print("=" * 70)

    print(f"Question: {question}")

    if task == "1":
        answer, confidence = binary_prediction(
            outputs["binary"]
        )

        print(f"\nAnswer: {answer}")
        print(f"Confidence: {confidence:.2f}%")

        probs = torch.softmax(outputs["binary"], dim=1)[0]

        print(f"NO probability:  {probs[0].item() * 100:.2f}%")
        print(f"YES probability: {probs[1].item() * 100:.2f}%")

    elif task == "2":
        answer, confidence = mcq_prediction(
            outputs["mcq"]
        )

        probabilities = torch.softmax(
            outputs["mcq"], dim=1
        )[0]

        print(f"\nPredicted Option: {answer}")
        print(f"Confidence: {confidence:.2f}%")

        print("\nOption probabilities:")

        for i, probability in enumerate(probabilities):
            option = chr(ord("A") + i)
            print(
                f"{option}: {probability.item() * 100:.2f}%"
            )

    elif task == "3":
        box = bbox_prediction(outputs["bbox"])

        print("\nPredicted Bounding Box:")
        print(
            f"x1: {box[0]:.4f}"
        )
        print(
            f"y1: {box[1]:.4f}"
        )
        print(
            f"x2: {box[2]:.4f}"
        )
        print(
            f"y2: {box[3]:.4f}"
        )

        print(
            f"\nRaw coordinates: {box}"
        )

    else:
        print("Invalid task.")

    print("=" * 70)


if __name__ == "__main__":
    main()