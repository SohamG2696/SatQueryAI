# 🛰️ SatQuery AI — SIH26167

**An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

> **Organization:** Indian Space Research Organisation (ISRO)  
> **Department:** Department of Space  
> **Category:** Software | **Theme:** Space Technology

---

## What is SatQuery AI?

SatQuery AI is an **agentic multimodal remote-sensing assistant** that allows users to ask natural-language questions about satellite imagery. It automatically selects specialized AI models for VQA, captioning, grounding, change analysis, or optical-SAR fusion, and returns **evidence-grounded answers** with confidence scores and an auditable execution trace.

### How it works

```
User uploads satellite image(s)
        ↓
Asks a natural-language question
        ↓
🤖 Agentic Controller determines the task
        ↓
Selects the appropriate specialist model
        ↓
Analyzes the image(s)
        ↓
Returns answer + visual evidence + confidence + execution trace
```

---

## Supported Scenarios

| Scenario | Input | Example Query |
|----------|-------|---------------|
| **VQA** | 1 image + question | "What types of land cover are visible?" |
| **Captioning** | 1 image | "Describe this image." |
| **Region Grounding** | 1 image + spatial query | "Highlight the water body." |
| **Change Detection** | 2 images (different dates) | "Has the built-up area increased?" |
| **Optical-SAR Fusion** | Optical + SAR image pair | "Use both images to identify built-up regions." |

---

## Project Structure

```
SatQuery-AI/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── config.py            # Settings & environment configuration
│   │   ├── schemas/             # Pydantic request/response models
│   │   ├── api/                 # API route handlers
│   │   ├── agent/               # Agentic controller & routing
│   │   ├── models/              # ML model wrappers (future)
│   │   ├── services/            # Business logic layer (future)
│   │   └── utils/               # Utility functions (future)
│   ├── requirements.txt
│   └── .env.example
├── models/                      # ML model weights (future)
├── datasets/                    # Training/eval data (future)
├── outputs/                     # Inference outputs (future)
├── frontend/                    # React frontend (future)
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env as needed
```

### 3. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open API docs

Navigate to [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/upload` | Upload satellite image |
| POST | `/api/analyze` | General analysis (auto-routed) |
| POST | `/api/vqa` | Visual Question Answering |
| POST | `/api/caption` | Image captioning / scene description |
| POST | `/api/grounding` | Region grounding |
| POST | `/api/change` | Bi-temporal change detection |
| POST | `/api/fusion` | Optical-SAR cross-modal fusion |
| POST | `/api/agent` | Agentic routing (recommended entry point) |

### Standard Response Format

All analysis endpoints return a consistent structure:

```json
{
  "success": true,
  "task": "vqa",
  "answer": "Agricultural fields and water bodies are visible.",
  "confidence": 0.87,
  "models_used": ["RemoteSensingVQA"],
  "parameters": {},
  "evidence": [],
  "execution_trace": ["Input validated", "VQA model selected", "Answer generated"],
  "processing_time": 1.23
}
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, Python, Uvicorn |
| **Frontend** | React + Vite (future) |
| **AI/ML** | PyTorch, HuggingFace Transformers (future) |
| **Image Processing** | OpenCV, Pillow (future) |
| **Geospatial** | Rasterio, GDAL (future) |

---

## Team

Built for **Smart India Hackathon 2026** — Problem Statement SIH26167.

---

## License

This project is developed for the SIH 2026 hackathon.
