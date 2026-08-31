"""
SatQuery AI — Logging Utility.

Configures structured logging for SatQuery AI without leaking
secrets, internal reasoning chains, or large binary payloads.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from app.config import settings


def setup_logger(name: str = "satquery") -> logging.Logger:
    """Initialize and configure the application logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        level_str = settings.log_level.upper()
        log_level = getattr(logging, level_str, logging.INFO)
        logger.setLevel(log_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


logger = setup_logger("satquery")


def log_request(task: str, model_name: str, query: str, image_count: int) -> None:
    """Log an incoming analysis request safely."""
    # Truncate query for logs
    clean_query = (query[:80] + "...") if len(query) > 80 else query
    logger.info(
        f"Request received | Task: {task} | Model: {model_name} | "
        f"Images: {image_count} | Query: '{clean_query}'"
    )


def log_inference(model_name: str, duration_ms: float, confidence: float | None) -> None:
    """Log completion of model inference."""
    conf_str = f"{confidence:.4f}" if confidence is not None else "N/A"
    logger.info(
        f"Inference complete | Model: {model_name} | "
        f"Duration: {duration_ms:.2f}ms | Confidence: {conf_str}"
    )


def log_error(context: str, error: Exception) -> None:
    """Log application error without exposing sensitive internals."""
    logger.error(f"Error in {context}: {type(error).__name__} - {str(error)}")
