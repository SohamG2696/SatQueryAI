"""
Tests for Validators and Geospatial Alignment Services.
"""

import pytest
import sys
from pathlib import Path
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.validators import (
    validate_file_extension,
    validate_file_size,
    validate_query,
    sanitize_filename,
)
from app.services.geospatial_service import check_coregistration
from app.services.metadata_service import detect_modality, extract_timestamp_from_filename


def test_file_extension_validation():
    """Verify supported and rejected extensions."""
    assert validate_file_extension("image.tif") == ".tif"
    assert validate_file_extension("image.tiff") == ".tiff"
    assert validate_file_extension("photo.png") == ".png"
    assert validate_file_extension("scene.jpg") == ".jpg"
    assert validate_file_extension("pair.npz") == ".npz"

    with pytest.raises(HTTPException) as exc:
        validate_file_extension("script.exe")
    assert exc.value.status_code == 400


def test_file_size_validation():
    """Verify empty and oversized files are caught."""
    validate_file_size(1024)  # 1 KB valid

    with pytest.raises(HTTPException) as exc:
        validate_file_size(0)
    assert exc.value.status_code == 400


def test_query_validation():
    """Verify query string sanitation."""
    assert validate_query("  Is there water?  ") == "Is there water?"

    with pytest.raises(HTTPException):
        validate_query("", required=True)


def test_sanitize_filename():
    """Verify path traversal prevention."""
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\secret.tif") == "secret.tif"
    assert sanitize_filename("valid_image_2024.tif") == "valid_image_2024.tif"


def test_geospatial_coregistration_check():
    """Verify co-registration evaluation."""
    meta1 = {"width": 224, "height": 224, "crs": "EPSG:32633", "bounds": [0, 0, 100, 100]}
    meta2 = {"width": 224, "height": 224, "crs": "EPSG:32633", "bounds": [0, 0, 100, 100]}
    res = check_coregistration(meta1, meta2)
    assert res["co_registered"] is True
    assert res["status"] == "aligned"

    # Mismatched CRS
    meta3 = {"width": 224, "height": 224, "crs": "EPSG:4326"}
    res_mismatch = check_coregistration(meta1, meta3)
    assert res_mismatch["crs_match"] is False


def test_metadata_service_detection():
    """Verify modality detection and timestamp parsing."""
    mod = detect_modality(filename="S1A_IW_GRDH_1SDV_20170613T165043_VH.tif")
    assert mod["modality"] == "sar"

    mod_opt = detect_modality(filename="S2A_MSIL2A_20240501_T33UUP.tif")
    assert mod_opt["modality"] == "optical"

    ts = extract_timestamp_from_filename("S1A_IW_GRDH_1SDV_20170613T165043_VH.tif")
    assert ts == "2017-06-13T16:50:43"
