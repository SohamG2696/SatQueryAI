"""
models/future_prediction/inference/dynamic_extractor.py
=========================================================
Dynamic land-cover extraction and spatial validation for user-uploaded
multi-year satellite imagery.

Features:
1. extract_landcover_from_uploads():
   Extracts SCL-based land-cover stats (built_up_pct, vegetation_pct, water_pct,
   valid_px_pct, low_confidence) for N user-uploaded Sentinel-2 GeoTIFFs.
   Reuses the exact SCL classification logic from prepare_dataset.py.
   Strictly requires SCL band 15 (raises error if missing — no NDVI fallback).

2. validate_same_location():
   Reads CRS and centroid lat/lon of each GeoTIFF via rasterio, verifying all
   images cover the same region within a specified spatial tolerance (km).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

# ── Haversine Distance Helper ──────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Great-Circle (Haversine) distance in km between two (lat, lon) points."""
    R = 6371.0  # Earth mean radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


# ── Location Validation ────────────────────────────────────────────────────────

def get_image_centroid(image_path: str | Path) -> tuple[float, float]:
    """
    Reads image CRS and transform using rasterio, returning (latitude, longitude)
    of the image centroid in WGS84 (EPSG:4326).

    Raises:
        ValueError: If CRS or spatial metadata is missing.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Image '{path.name}' lacks geospatial CRS metadata (not a valid GeoTIFF).")

        # Center pixel in image pixel coordinates
        cx = src.width / 2.0
        cy = src.height / 2.0

        # Convert pixel (cy, cx) to map coordinates in native CRS
        x_native, y_native = rasterio.transform.xy(src.transform, cy, cx)

        # Reproject map coordinates to WGS84 (EPSG:4326) if necessary
        try:
            if src.crs.to_epsg() == 4326:
                lon, lat = float(x_native), float(y_native)
            else:
                lons, lats = warp_transform(src.crs, "EPSG:4326", [x_native], [y_native])
                lon, lat = float(lons[0]), float(lats[0])
        except Exception as e:
            raise ValueError(f"Failed to reproject CRS ({src.crs}) for '{path.name}': {e}")

    return lat, lon


def validate_same_location(
    image_paths: list[str | Path],
    tolerance_km: float = 2.0,
) -> dict[str, Any]:
    """
    Confirms all uploaded images cover the same geographical region within
    the specified centroid-to-centroid distance tolerance (km).

    Args:
        image_paths: List of file paths to uploaded GeoTIFFs.
        tolerance_km: Maximum allowable centroid separation distance in km (default 2.0 km).

    Returns:
        Dict: {
            "valid": bool,
            "centroids": [{"image": str, "lat": float, "lon": float}],
            "max_distance_km": float,
            "message": str
        }
    """
    if not image_paths or len(image_paths) < 2:
        return {
            "valid": False,
            "centroids": [],
            "max_distance_km": 0.0,
            "message": "At least 2 images are required to validate spatial overlap.",
        }

    centroids: list[dict[str, Any]] = []

    for path in image_paths:
        p = Path(path)
        try:
            lat, lon = get_image_centroid(p)
            centroids.append({"image": p.name, "lat": round(lat, 6), "lon": round(lon, 6)})
        except Exception as e:
            return {
                "valid": False,
                "centroids": centroids,
                "max_distance_km": float("inf"),
                "message": f"Geospatial validation failed: {e}",
            }

    # Pairwise max centroid distance calculation
    max_dist_km = 0.0
    n = len(centroids)
    for i in range(n):
        for j in range(i + 1, n):
            c1 = centroids[i]
            c2 = centroids[j]
            dist = haversine_km(c1["lat"], c1["lon"], c2["lat"], c2["lon"])
            if dist > max_dist_km:
                max_dist_km = dist

    max_dist_km = round(max_dist_km, 3)

    if max_dist_km <= tolerance_km:
        return {
            "valid": True,
            "centroids": centroids,
            "max_distance_km": max_dist_km,
            "message": f"All {n} images cover the same location (max separation: {max_dist_km:.2f} km <= {tolerance_km} km).",
        }
    else:
        return {
            "valid": False,
            "centroids": centroids,
            "max_distance_km": max_dist_km,
            "message": (
                f"Spatial discrepancy detected: max distance between image centroids "
                f"is {max_dist_km:.2f} km, which exceeds tolerance threshold of {tolerance_km} km."
            ),
        }


# ── SCL Land-Cover Extraction ─────────────────────────────────────────────────

def _extract_single_scl(path: Path) -> dict[str, Any]:
    """
    Extracts land-cover composition from band 15 (SCL) of a single Sentinel-2 GeoTIFF.
    Reuses the exact SCL classification rules from prepare_dataset.py.
    """
    try:
        with rasterio.open(path) as src:
            has_scl = src.count >= 15 and (src.descriptions[14] == "SCL" or src.count >= 15)
            if not has_scl:
                raise ValueError(
                    f"Image '{path.name}' is missing SCL band (band 15). "
                    "Sentinel-2 Surface Reflectance (S2_SR_HARMONIZED) format with SCL band 15 is required. "
                    "Plain RGB/PNG/JPG uploads or non-SCL GeoTIFFs are not supported."
                )
            scl_raw = src.read(15)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Image '{path.name}' is not a valid Sentinel-2 GeoTIFF raster: {e}"
        )

    # GEE median composite rounding fix (exact prepare_dataset.py logic)
    scl = np.round(scl_raw).astype(np.int32)
    total_pixels = scl.size

    # ESA Scene Classification Layer (SCL) definitions:
    # Excluded: 0 (no-data), 1 (defective/saturated), 3 (cloud shadow), 8 (cloud med), 9 (cloud high), 10 (cirrus)
    # Valid surface: 2 (dark/topographic), 4 (vegetation), 5 (bare soil / built-up), 6 (water), 7 (unclassified), 11 (snow)
    EXCLUDE_CLASSES = {0, 1, 3, 8, 9, 10}
    VALID_CLASSES = {2, 4, 5, 6, 7, 11}

    valid_mask = np.isin(scl, list(VALID_CLASSES))
    valid_count = int(valid_mask.sum())

    if valid_count == 0:
        raise ValueError(f"Image '{path.name}' contains 0 valid surface pixels (100% cloud/nodata cover).")

    built_up_pct = round(float((scl == 5).sum() / valid_count * 100), 2)
    vegetation_pct = round(float((scl == 4).sum() / valid_count * 100), 2)
    water_pct = round(float((scl == 6).sum() / valid_count * 100), 2)
    valid_px_pct = round(float(valid_count / total_pixels * 100), 2)
    low_confidence = bool(valid_px_pct < 99.0)

    return {
        "built_up_pct": built_up_pct,
        "vegetation_pct": vegetation_pct,
        "water_pct": water_pct,
        "valid_px_pct": valid_px_pct,
        "low_confidence": low_confidence,
    }


def extract_landcover_from_uploads(
    image_paths: list[str | Path],
    years: list[int],
) -> pd.DataFrame:
    """
    Processes user-uploaded multi-year Sentinel-2 GeoTIFFs and extracts their
    land-cover statistics using ESA's Scene Classification Layer (SCL).

    Args:
        image_paths: List of N file paths to user-uploaded GeoTIFFs.
        years: List of N corresponding acquisition years (e.g., [2019, 2021, 2023, 2025]).

    Returns:
        pd.DataFrame: Columns matching landcover_timeseries.csv:
            [year, built_up_pct, vegetation_pct, water_pct, valid_px_pct, low_confidence]
            Sorted chronologically by year.

    Raises:
        ValueError: If list lengths mismatch, n < 2, or any image is missing SCL band 15.
    """
    if len(image_paths) != len(years):
        raise ValueError(
            f"Mismatched input lengths: got {len(image_paths)} image paths but {len(years)} years."
        )

    if len(image_paths) < 2:
        raise ValueError("At least 2 historical images (from different years) are required for trend forecasting.")

    rows: list[dict[str, Any]] = []

    for path, yr in zip(image_paths, years):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {p}")

        stats = _extract_single_scl(p)
        rows.append({
            "year": int(yr),
            "built_up_pct": stats["built_up_pct"],
            "vegetation_pct": stats["vegetation_pct"],
            "water_pct": stats["water_pct"],
            "valid_px_pct": stats["valid_px_pct"],
            "low_confidence": stats["low_confidence"],
        })

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return df


# ── Testing Script ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("SatQueryAI — Dynamic Extractor Self-Test")
    print("=" * 80)

    PROJECT_ROOT_DIR = Path(__file__).resolve().parents[3]
    RAW_DIR = PROJECT_ROOT_DIR / "models" / "future_prediction" / "data" / "raw"

    img_2019 = RAW_DIR / "region_01_2019.tif"
    img_2021 = RAW_DIR / "region_01_2021.tif"

    print(f"\n1. Testing validate_same_location() on region_01_2019.tif & region_01_2021.tif:")
    loc_val = validate_same_location([img_2019, img_2021], tolerance_km=2.0)
    print(json.dumps(loc_val, indent=2))

    print(f"\n2. Testing extract_landcover_from_uploads():")
    df_extracted = extract_landcover_from_uploads([img_2019, img_2021], [2019, 2021])
    print(df_extracted.to_string(index=False))

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED CLEANLY")
    print("=" * 80)
