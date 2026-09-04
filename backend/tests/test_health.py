"""
Tests for Health and Model Status Endpoints.
"""

from fastapi.testclient import TestClient
import pytest
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

client = TestClient(app)


def test_health_check():
    """Verify /health returns status ok and correct ready/not_ready model statuses."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert "device" in data
    assert "models" in data

    models = data["models"]
    assert models["fusion"] == "ready"
    assert models["grounding"] == "ready"
    assert models["change_vqa"] == "ready"
    assert models["vqa"] == "ready"
    assert models["captioning"] == "ready"


def test_api_models_endpoint():
    """Verify /api/models returns detailed list of all 5 specialist entries."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 5

    tasks = {m["task"] for m in data["models"]}
    assert {"fusion", "grounding", "change_vqa", "vqa", "captioning"}.issubset(tasks)
