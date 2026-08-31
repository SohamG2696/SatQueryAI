# SatQuery AI — Vision-Language Model (VLM) Module

This module is the placeholder interface for the general Remote Sensing Vision-Language Model (VLM) supporting Visual Question Answering (VQA) and Image Captioning.

## Architecture Specification
- **Vision Backbone**: Remote Sensing Vision Transformer / ConvNeXt encoder
- **Language Backbone**: Decoupled LLM / Multi-modal Decoder
- **Target Tasks**:
  - Open-ended Remote Sensing VQA
  - Natural Language Scene Description / Summarization

## Planned Integration
Once weights are ready, place weights under `models/vlm/weights/` and update `backend/app/models/vqa.py` and `backend/app/models/captioning.py`.
