"""
SatQuery AI — GEE Endpoint Tests.

Comprehensive mocked tests verifying:
- Parameter propagation for all datasets
- Date validation (invalid, reversed)
- Empty collection → 404
- SRTM buffer handling
- Sentinel-1 polarization/orbit/mode
- Spectral index date propagation
- Existing functionality regression
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def mock_ee():
    with patch("app.services.earth_engine.ee") as mock:
        yield mock


def _make_chainable_collection(mock_ee, size=1):
    """Create a mock collection with chainable filter methods."""
    coll = MagicMock()
    coll.filterBounds.return_value = coll
    coll.filterDate.return_value = coll
    coll.filter.return_value = coll
    coll.sort.return_value = coll
    coll.size.return_value.getInfo.return_value = size
    mock_ee.ImageCollection.return_value = coll
    mock_ee.Filter.lt.return_value = MagicMock()
    mock_ee.Filter.eq.return_value = MagicMock()
    mock_ee.Filter.listContains.return_value = MagicMock()
    return coll


def _make_image(coll, image_id="IMG_001", cloud=10.0, date="2024-06-15"):
    """Attach a mock image to a chainable collection."""
    img = MagicMock()
    img.get.side_effect = lambda key: MagicMock(getInfo=lambda: {
        "PRODUCT_ID": image_id,
        "CLOUDY_PIXEL_PERCENTAGE": cloud,
        "CLOUD_COVER": cloud,
        "LANDSAT_PRODUCT_ID": image_id,
        "system:index": image_id,
        "system:time_start": 1600000000000,
    }.get(key))
    img.getThumbURL.return_value = "http://example.com/thumb.png"
    img.select.return_value.multiply.return_value.add.return_value = img
    coll.first.return_value = img
    return img


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. Sentinel-2 Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_sentinel2_default_dates(mock_ee):
    """Sentinel-2 works with default dates (no explicit dates)."""
    coll = _make_chainable_collection(mock_ee)
    _make_image(coll)
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2025-05-05"

    response = client.get("/api/gee/sentinel2?lat=30.0&lon=75.0")
    assert response.status_code == 200
    data = response.json()
    assert data["image_id"] == "IMG_001"
    assert data["date"] == "2025-05-05"
    assert "cloud_percentage" in data


def test_sentinel2_date_range_respected(mock_ee):
    """Sentinel-2 filterDate is called with exact user-supplied dates."""
    coll = _make_chainable_collection(mock_ee)
    _make_image(coll, date="2024-06-15")
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-06-15"

    response = client.get(
        "/api/gee/sentinel2?lat=19.076&lon=72.8777"
        "&start_date=2024-06-01&end_date=2024-06-30&cloud_percentage=20"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["date"] == "2024-06-15"
    coll.filterDate.assert_called_with("2024-06-01", "2024-06-30")


def test_sentinel2_empty_collection_returns_404(mock_ee):
    """Empty filtered collection returns HTTP 404."""
    _make_chainable_collection(mock_ee, size=0)

    response = client.get(
        "/api/gee/sentinel2?lat=19.076&lon=72.8777"
        "&start_date=2024-06-01&end_date=2024-06-30"
    )
    assert response.status_code == 404
    assert "No Sentinel-2 imagery found" in response.json()["detail"]


def test_sentinel2_full_year(mock_ee):
    """Sentinel-2 query with full year date range."""
    coll = _make_chainable_collection(mock_ee)
    _make_image(coll)
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-03-15"

    response = client.get(
        "/api/gee/sentinel2?lat=30.0&lon=75.0"
        "&start_date=2024-01-01&end_date=2024-12-31"
    )
    assert response.status_code == 200
    assert response.json()["date"] == "2024-03-15"
    coll.filterDate.assert_called_with("2024-01-01", "2024-12-31")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. Sentinel-1 Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_sentinel1_date_range(mock_ee):
    """Sentinel-1 filterDate is called with user-supplied dates."""
    coll = _make_chainable_collection(mock_ee)
    _make_image(coll, image_id="S1_001")
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-02-10"

    response = client.get(
        "/api/gee/sentinel1?lat=30.0&lon=75.0"
        "&start_date=2024-01-01&end_date=2024-12-31"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["date"] == "2024-02-10"
    assert data["cloud_percentage"] is None
    coll.filterDate.assert_called_with("2024-01-01", "2024-12-31")


def test_sentinel1_polarization_vh(mock_ee):
    """Sentinel-1 VH polarization is correctly passed as filter."""
    coll = _make_chainable_collection(mock_ee)
    _make_image(coll, image_id="S1_VH")
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-05-01"

    response = client.get(
        "/api/gee/sentinel1?lat=30.0&lon=75.0&polarization=VH"
    )
    assert response.status_code == 200
    mock_ee.Filter.listContains.assert_called_with("transmitterReceiverPolarisation", "VH")


def test_sentinel1_orbit_ascending(mock_ee):
    """Sentinel-1 ASCENDING orbit is correctly passed as filter."""
    coll = _make_chainable_collection(mock_ee)
    _make_image(coll, image_id="S1_ASC")
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-05-01"

    response = client.get(
        "/api/gee/sentinel1?lat=30.0&lon=75.0&orbit_pass=ASCENDING"
    )
    assert response.status_code == 200
    mock_ee.Filter.eq.assert_any_call("orbitProperties_pass", "ASCENDING")


def test_sentinel1_empty_returns_404(mock_ee):
    """Sentinel-1 empty collection returns 404."""
    _make_chainable_collection(mock_ee, size=0)

    response = client.get(
        "/api/gee/sentinel1?lat=30.0&lon=75.0"
        "&start_date=2024-01-01&end_date=2024-01-02"
    )
    assert response.status_code == 404


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C. Landsat Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_landsat_date_cloud_propagation(mock_ee):
    """Landsat date and cloud parameters are propagated."""
    coll = _make_chainable_collection(mock_ee)
    img = _make_image(coll, image_id="LC09_001", cloud=5.0)
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-07-20"

    response = client.get(
        "/api/gee/landsat?lat=30.0&lon=75.0"
        "&start_date=2024-01-01&end_date=2024-12-31&cloud_percentage=15"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["date"] == "2024-07-20"
    coll.filterDate.assert_called_with("2024-01-01", "2024-12-31")
    mock_ee.Filter.lt.assert_any_call("CLOUD_COVER", 15.0)


def test_landsat_empty_returns_404(mock_ee):
    """Landsat empty collection returns 404."""
    _make_chainable_collection(mock_ee, size=0)

    response = client.get(
        "/api/gee/landsat?lat=30.0&lon=75.0"
        "&start_date=2024-06-01&end_date=2024-06-02"
    )
    assert response.status_code == 404


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# D. SRTM Elevation Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_elevation_point(mock_ee):
    """Elevation point query works."""
    mock_image = MagicMock()
    mock_image.sample.return_value.first.return_value.get.return_value.getInfo.return_value = 1500.5
    mock_ee.Image.return_value = mock_image

    response = client.get("/api/gee/elevation?lat=30.0&lon=75.0")
    assert response.status_code == 200
    data = response.json()
    assert data["elevation_m"] == 1500.5
    assert data["buffer_m"] == 0.0


def test_elevation_with_buffer(mock_ee):
    """Elevation with buffer returns stats."""
    mock_image = MagicMock()
    mock_image.reduceRegion.return_value.getInfo.return_value = {
        "elevation_mean": 1200.0,
        "elevation_min": 1100.0,
        "elevation_max": 1300.0,
    }
    mock_ee.Image.return_value = mock_image
    mock_ee.Reducer.mean.return_value.combine.return_value.combine.return_value = MagicMock()

    response = client.get("/api/gee/elevation?lat=30.0&lon=75.0&buffer=1000")
    assert response.status_code == 200
    data = response.json()
    assert data["elevation_mean"] == 1200.0
    assert data["elevation_min"] == 1100.0
    assert data["elevation_max"] == 1300.0
    assert data["buffer_m"] == 1000.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# E. Spectral Index Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_ndvi_date_propagation(mock_ee):
    """NDVI date range reaches underlying Sentinel-2 collection."""
    coll = _make_chainable_collection(mock_ee)
    img = MagicMock()
    coll.first.return_value = img
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-07-15"

    ndvi_img = MagicMock()
    img.normalizedDifference.return_value = ndvi_img
    ndvi_img.sample.return_value.first.return_value.get.return_value.getInfo.return_value = 0.75

    response = client.get(
        "/api/gee/ndvi?lat=30.0&lon=75.0"
        "&start_date=2024-06-01&end_date=2024-08-31"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["index_name"] == "NDVI"
    assert data["index_value"] == 0.75
    assert data["date"] == "2024-07-15"
    coll.filterDate.assert_called_with("2024-06-01", "2024-08-31")


def test_ndwi_date_propagation(mock_ee):
    """NDWI date range reaches underlying Sentinel-2 collection."""
    coll = _make_chainable_collection(mock_ee)
    img = MagicMock()
    coll.first.return_value = img
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-07-10"

    ndwi_img = MagicMock()
    img.normalizedDifference.return_value = ndwi_img
    ndwi_img.sample.return_value.first.return_value.get.return_value.getInfo.return_value = -0.10

    response = client.get(
        "/api/gee/ndwi?lat=30.0&lon=75.0"
        "&start_date=2024-06-01&end_date=2024-08-31"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["index_name"] == "NDWI"
    assert data["index_value"] == -0.10
    assert "low open-water" in data["explanation"]
    coll.filterDate.assert_called_with("2024-06-01", "2024-08-31")


def test_ndbi_date_propagation(mock_ee):
    """NDBI date range reaches underlying Sentinel-2 collection."""
    coll = _make_chainable_collection(mock_ee)
    img = MagicMock()
    coll.first.return_value = img
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-07-20"

    ndbi_img = MagicMock()
    img.normalizedDifference.return_value = ndbi_img
    ndbi_img.sample.return_value.first.return_value.get.return_value.getInfo.return_value = 0.11

    response = client.get(
        "/api/gee/ndbi?lat=30.0&lon=75.0"
        "&start_date=2024-06-01&end_date=2024-08-31"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["index_name"] == "NDBI"
    assert data["index_value"] == 0.11
    assert "built-up" in data["explanation"]
    coll.filterDate.assert_called_with("2024-06-01", "2024-08-31")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# F. Date Validation Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_invalid_date_format(mock_ee):
    """Invalid date format returns 400."""
    response = client.get(
        "/api/gee/sentinel2?lat=30.0&lon=75.0"
        "&start_date=2024/06/01&end_date=2024-06-30"
    )
    assert response.status_code == 400
    assert "format" in response.json()["detail"].lower()


def test_start_after_end_date(mock_ee):
    """start_date >= end_date returns 400."""
    response = client.get(
        "/api/gee/sentinel2?lat=30.0&lon=75.0"
        "&start_date=2024-06-30&end_date=2024-06-01"
    )
    assert response.status_code == 400
    assert "must be before" in response.json()["detail"]


def test_same_start_end_date(mock_ee):
    """start_date == end_date returns 400."""
    response = client.get(
        "/api/gee/sentinel2?lat=30.0&lon=75.0"
        "&start_date=2024-06-15&end_date=2024-06-15"
    )
    assert response.status_code == 400


def test_only_start_date_returns_400(mock_ee):
    """Providing only start_date without end_date returns 400."""
    response = client.get(
        "/api/gee/sentinel1?lat=30.0&lon=75.0"
        "&start_date=2024-01-01"
    )
    assert response.status_code == 400
    assert "must be provided together" in response.json()["detail"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# G. Query Planner Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_query_planner_elevation(mock_ee):
    """Query planner routes elevation queries correctly."""
    mock_image = MagicMock()
    mock_image.sample.return_value.first.return_value.get.return_value.getInfo.return_value = 1200.0
    mock_ee.Image.return_value = mock_image

    response = client.post("/api/gee/query", json={
        "query": "what is the elevation?",
        "latitude": 30.0,
        "longitude": 75.0,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "elevation"
    assert "elevation" in data["explanation"].lower()
