"""
SatQuery AI — Application Configuration.

Loads settings from environment variables and backend/.env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    app_name: str = "SatQuery AI"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True

    # ── Server ───────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8000

    # ── Compute Device ───────────────────────────────────────────
    device: str = "xpu"

    # ── File Uploads ─────────────────────────────────────────────
    upload_dir: str = "backend/uploads"
    max_upload_size_mb: int = 100
    allowed_image_extensions: str = ".tif,.tiff,.png,.jpg,.jpeg"

    # ── Model Paths ──────────────────────────────────────────────
    vqa_model_path: str = "models/vlm/weights"
    caption_model_path: str = "models/vlm/weights"

    # Trained Semantic Grounding checkpoint
    grounding_model_path: str = (
        "models/grounding/weights/spatial_grounding_model.pth"
    )

    # Trained ChangeFormerV6 checkpoint
    change_vqa_model_path: str = (
        "models/change_vqa/weights/checkpoint_best.pth"
    )

    # Fusion model
    fusion_model_path: str = (
        "models/fusion/weights/multitask_fusion_model_final.pth"
    )

    # ── Confidence & Evaluation ──────────────────────────────────
    confidence_threshold: float = 0.50

    # ── CORS ─────────────────────────────────────────────────────
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173"
    )

    # ── Logging ──────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── API Keys ─────────────────────────────────────────────────
    gemini_api_key: str = ""

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir).resolve()

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [
            ext.strip().lower()
            for ext in self.allowed_image_extensions.split(",")
            if ext.strip()
        ]

    @property
    def cors_origins_list(self) -> List[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


# Global singleton settings instance
settings = Settings()