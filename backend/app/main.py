"""
SatQuery AI — FastAPI Application Entry Point.

Problem Statement: SIH26167 (ISRO)
"SatQuery AI - An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries"

Run with:
    uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.model_registry import registry
from app.api import all_routers
from app.config import settings
from app.services.inference_service import inference_service
from app.utils.device import get_device, get_device_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lifespan Hook — Startup & Shutdown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: runs model warmup and directory checks on startup."""
    # ── Startup ──────────────────────────────────────────────────
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    (settings.upload_path / "optical").mkdir(parents=True, exist_ok=True)
    (settings.upload_path / "sar").mkdir(parents=True, exist_ok=True)
    (settings.upload_path / "temporary").mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[{settings.app_name}] Starting up v{settings.app_version} ({settings.environment})")
    print(f"[{settings.app_name}] Compute Device: {device}")
    print(f"[{settings.app_name}] Upload directory: {settings.upload_path.resolve()}")

    # Warm up models in memory
    inference_service.warm_up()
    print(f"[{settings.app_name}] Ready models initialized: Fusion, Grounding, Change-VQA")

    yield

    # ── Shutdown ─────────────────────────────────────────────────
    print(f"[{settings.app_name}] Shutting down")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI Application Instance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(
    title=settings.app_name,
    description=(
        "An interactive vision-language assistant for multimodal remote-sensing "
        "image analysis through text queries. Built for SIH 2026 — Problem "
        "Statement SIH26167 (ISRO)."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ────────────────────────────────────────────
for router in all_routers:
    app.include_router(router)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Health & Status Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.get("/health", tags=["Health"])
@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check and model status inspection endpoint."""
    device = get_device()
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "device": device.type,
        "device_info": get_device_info(),
        "models": registry.get_status_dict(),
    }


@app.get("/api/models", tags=["Models"])
async def list_registered_models():
    """List all registered specialist models and their readiness status."""
    return {
        "models": registry.list_models(),
    }
