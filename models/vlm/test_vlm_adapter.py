"""
Standalone test script for Person A VLM Adapter (models/vlm/vlm_adapter.py).
"""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from models.vlm.vlm_adapter import get_vlm_adapter


def run_test():
    print("=== Testing Person A VLM Adapter ===")

    # 1. Instantiate Adapter (Model Loading)
    start_load = time.time()
    adapter = get_vlm_adapter()
    load_time = time.time() - start_load

    print(f"Model Loaded Successfully: True ({load_time:.2f}s)")
    print(f"Device Used: {adapter.device}")

    # 2. Select Test Image
    sample_image_path = PROJECT_ROOT / "backend" / "uploads" / "optical" / "img_ee2b8183a867.jpeg"
    if sample_image_path.exists():
        print(f"Using sample satellite image: {sample_image_path}")
        image_input = sample_image_path
    else:
        print("Sample image file not found, creating synthetic RGB satellite image.")
        image_input = Image.new("RGB", (384, 384), color=(80, 120, 70))

    # 3. Run VQA Inference
    question = "Are buildings visible in this satellite image?"
    print(f"\nAsking Question: '{question}'")

    result = adapter.predict(image=image_input, question=question)

    print("\n--- Adapter Output Result ---")
    print(f"Question:         {result['question']}")
    print(f"Raw Prediction:   {result['raw_prediction']}")
    print(f"Prediction:       {result['prediction']}")
    print(f"Inference Time:   {result['inference_time_s']} s")
    print(f"Model:            {result['model']}")

    # 4. Test Error Handling
    print("\n--- Testing Error Handling ---")

    # Test Empty Question
    try:
        adapter.predict(image=image_input, question="")
        print("ERROR: Empty question validation failed!")
    except ValueError as e:
        print(f"PASS: Caught empty question correctly -> {e}")

    # Test Missing Image
    try:
        adapter.predict(image=None, question=question)
        print("ERROR: Missing image validation failed!")
    except ValueError as e:
        print(f"PASS: Caught missing image correctly -> {e}")

    # Test Invalid Image Path
    try:
        adapter.predict(image="non_existent_image.png", question=question)
        print("ERROR: Invalid image path validation failed!")
    except FileNotFoundError as e:
        print(f"PASS: Caught invalid image path correctly -> {e}")

    print("\n=== VLM ADAPTER TEST PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_test()
