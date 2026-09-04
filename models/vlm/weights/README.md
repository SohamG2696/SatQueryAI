# SatQuery AI Model

> **SatQuery AI** — A fine-tuned Vision-Language Model for Remote Sensing Satellite Image Analysis

---

## Overview

This folder contains the **complete, self-contained, merged SatQuery AI model** ready for deployment.

- **Base model**: `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`
- **Architecture**: LLaVA-OneVision (Qwen2 0.5B language backbone + SigLIP vision encoder)
- **Training**: LoRA fine-tuning on satellite imagery (RSVQA / remote sensing VQA tasks)
- **Deployment type**: **Merged standalone** — LoRA weights are permanently folded into base model weights

The model does **not** require loading a separate LoRA adapter. It is a standard Hugging Face model loadable directly with `from_pretrained`.

---

## Folder Structure

```
SatQuery_AI_Model/
├── config.json                  — Model architecture configuration
├── generation_config.json       — Default generation parameters
├── model.safetensors            — Complete merged model weights (~3.4 GB)
├── tokenizer.json               — BPE tokenizer
├── tokenizer_config.json        — Tokenizer settings + chat template
├── special_tokens_map.json      — Special token definitions
├── processor_config.json        — LLaVA processor configuration
├── preprocessor_config.json     — Image preprocessor configuration
├── chat_template.jinja          — Chat template for conversation formatting
└── README.md                    — This file
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj` (language model attention) |
| Trainable params | 2,162,688 |
| Training framework | Transformers 4.49.0, PEFT 0.14.0, PyTorch 2.10.0+cu128 |
| Input resolution | 120×120 RGB |
| Task types | Binary VQA, MCQ VQA, Image Captioning |

---

## Requirements

```
pip install -r requirements-satquery.txt
```

Key versions:
```
transformers==4.49.0
peft==0.14.0
tokenizers==0.21.1
torch==2.1.0
pillow>=10.0.0
safetensors==0.4.3
accelerate>=0.26.0
```

---

## Loading the Model

```python
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import torch

MODEL_DIR = "./SatQuery_AI_Model"

processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,   # use float32 on CPU
    device_map="cuda",            # or "cpu"
    attn_implementation="eager",
)
model.eval()
```

---

## Running Inference

### Using the Inference Module (Recommended)

```python
from satquery_inference import SatQueryModel

# Load once at startup
model = SatQueryModel()  # loads from ./SatQuery_AI_Model/ by default

# Binary question
result = model.predict(
    "image.png",
    "Can broad-leaved forest be detected in the satellite image?"
)
print(result)
# {
#   "question": "Can broad-leaved forest be detected...",
#   "task_type": "binary",
#   "raw_prediction": "no",
#   "prediction": "no",
#   "inference_time_s": 12.3
# }

# MCQ question
result = model.predict(
    "image.png",
    "What type of land use is visible?\na) Urban\nb) Forest\nc) Farmland\nd) Water"
)
print(result["prediction"])  # "c"

# Captioning
result = model.predict(
    "image.png",
    "Describe the land cover in this satellite image.",
    task_type="captioning"
)
print(result["prediction"])
```

### Direct Transformers Usage

```python
from PIL import Image
import torch

img = Image.open("image.png").convert("RGB").resize((120, 120))

conversation = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": "Are buildings visible in this satellite image?"}
    ]
}]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        repetition_penalty=1.3,
    )

# Decode only new tokens
new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
print(answer)
```

---

## FastAPI Backend Integration

```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from satquery_inference import SatQueryModel
import tempfile, os, shutil

app = FastAPI(title="SatQuery AI API")

# Load model ONCE at startup
sat_model = SatQueryModel()

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(..., description="Satellite image (PNG/JPEG)"),
    question: str = Form(..., description="Question about the image"),
):
    """
    Analyze a satellite image and answer a question.

    Returns:
        - question: original question
        - prediction: model answer (normalized)
        - raw_prediction: unmodified model output
        - task_type: auto-detected (binary/mcq/captioning)
        - inference_time_s: seconds taken
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = sat_model.predict(tmp_path, question)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.unlink(tmp_path)

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Expected Input Format

- **Image**: Any RGB image, will be resized to 120×120 internally
- **Question formats**:
  - Binary: `"Are buildings visible in this image?"` → answers `yes` or `no`
  - MCQ: `"What land cover?\na) Urban\nb) Forest\nc) Farmland\nd) Water"` → answers `a`/`b`/`c`/`d`
  - Captioning: `"Describe the land cover in this image."` → free-form description

---

## Notes

- This model was trained for **remote sensing satellite imagery**. Performance on other image types is not guaranteed.
- Input images should ideally be **overhead satellite views** with natural RGB colors.
- The model uses **eager attention** (`attn_implementation="eager"`) for reliability.
- On CPU, inference takes approximately **10–30 seconds** per image depending on hardware.
- On GPU (CUDA), inference takes approximately **0.5–2 seconds** per image.

---

## Original Training Adapter

The original LoRA adapter is preserved at:
```
SatQuery_Best_Adapter.zip
```

This ZIP is your backup. The merged model in this folder was created by:
1. Loading `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`
2. Loading the LoRA adapter via PEFT
3. Calling `peft_model.merge_and_unload()`
4. Saving with `save_pretrained()`

---

*Generated by package_model.py — SatQuery AI packaging pipeline*
