# SatQuery AI — Optical-SAR Cross-Modal Fusion Module

## Status: READY 🏆

Integrated with the 5-epoch trained multi-task model weights.

## Metrics
- **Validation Binary Accuracy**: 78.00%
- **Validation MCQ Accuracy**: 63.85%
- **Validation BBox Loss**: 0.0236
- **Test Binary Accuracy**: 78.16%
- **Test MCQ Accuracy**: 63.82%

## Model Architecture
- **Optical Encoder**: 4 Channels (B02, B03, B04, B08) -> 128-d
- **SAR Encoder**: 2 Channels (VH, VV) -> 128-d
- **Question Encoder**: Vocabulary GRU (493 tokens, max length 40) -> 128-d
- **Fusion Layer**: Linear(384 -> 256 -> 128)
- **Heads**:
  - `binary_head`: Linear(128, 2)
  - `mcq_head`: Linear(128, 4)
  - `bbox_head`: Linear(128, 64) -> ReLU -> Linear(64, 4) -> Sigmoid
