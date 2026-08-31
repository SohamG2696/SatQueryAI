"""
Tests for Image Upload API Endpoint.
"""

import io
from PIL import Image
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

client = TestClient(app)


def test_upload_valid_image():
    """Verify upload of a valid PNG image."""
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/upload",
        files={"file": ("test_optical.png", buf, "image/png")},
        data={"modality": "optical"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "image_id" in data
    assert data["image_id"].startswith("img_")
    assert data["metadata_detected"]["modality"] == "optical"
    assert data["metadata_detected"]["width"] == 100
    assert data["metadata_detected"]["height"] == 100


def test_upload_invalid_extension():
    """Verify rejection of invalid file types."""
    buf = io.BytesIO(b"malicious executable payload")
    response = client.post(
        "/api/upload",
        files={"file": ("test.exe", buf, "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_empty_file():
    """Verify rejection of empty 0-byte file."""
    buf = io.BytesIO(b"")
    response = client.post(
        "/api/upload",
        files={"file": ("empty.png", buf, "image/png")},
    )
    assert response.status_code == 400
