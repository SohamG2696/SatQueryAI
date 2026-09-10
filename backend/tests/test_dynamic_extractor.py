"""
Unit tests for dynamic land-cover extractor and spatial validator.
"""

from pathlib import Path
import pytest
import sys

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.future_prediction.inference.dynamic_extractor import (
    extract_landcover_from_uploads,
    validate_same_location,
)

RAW_DIR = PROJECT_ROOT / "models" / "future_prediction" / "data" / "raw"
IMG_2019 = RAW_DIR / "region_01_2019.tif"
IMG_2021 = RAW_DIR / "region_01_2021.tif"


def test_validate_same_location_valid():
    """Verify validate_same_location returns valid=True for identical location images."""
    res = validate_same_location([IMG_2019, IMG_2021], tolerance_km=2.0)
    assert res["valid"] is True
    assert res["max_distance_km"] <= 0.01
    assert len(res["centroids"]) == 2


def test_extract_landcover_from_uploads_success():
    """Verify SCL landcover extraction matches expected values."""
    df = extract_landcover_from_uploads([IMG_2019, IMG_2021], [2019, 2021])
    assert len(df) == 2
    assert list(df["year"]) == [2019, 2021]

    # region_01 2019 should be 65.98% built, 34.02% veg
    row19 = df[df["year"] == 2019].iloc[0]
    assert abs(row19["built_up_pct"] - 65.98) < 0.1
    assert abs(row19["vegetation_pct"] - 34.02) < 0.1
    assert row19["low_confidence"] == False


def test_extract_landcover_non_scl_fails(tmp_path):
    """Verify that uploading a non-SCL image raises a clear ValueError."""
    fake_png = tmp_path / "fake_image.png"
    fake_png.write_bytes(b"PNG fake data")

    with pytest.raises(ValueError, match="not a valid Sentinel-2 GeoTIFF"):
        extract_landcover_from_uploads([fake_png, IMG_2019], [2019, 2021])
