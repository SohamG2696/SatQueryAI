"""
SatQuery AI — Google Earth Engine Service Layer.

Central GEE service handling all satellite data retrieval, spectral index
computation, and natural-language query planning.

Authentication: ee.Initialize(project="satquery-ai-507618")
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

try:
    import ee
    try:
        ee.Initialize(project="satquery-ai-507618")
    except Exception as e:
        print(f"Warning: Failed to initialize Earth Engine: {e}")
except ImportError:
    ee = None
    print("Warning: earthengine-api ('ee') module not installed. GEE operations will use offline fallbacks.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Date Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def validate_dates(start_date: Optional[str], end_date: Optional[str]) -> Tuple[str, str]:
    """Validate and return (start_date, end_date) strings.

    Rules:
    - Both must be valid YYYY-MM-DD.
    - start_date must be strictly before end_date.
    - If both are None, returns sensible defaults.
    """
    if start_date is None and end_date is None:
        # Default: last 2 years
        now = datetime.utcnow()
        return (now - timedelta(days=730)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")

    if start_date is None or end_date is None:
        raise ValueError("Both start_date and end_date must be provided together, or omit both for defaults.")

    # Validate format
    try:
        sd = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid start_date format: '{start_date}'. Expected YYYY-MM-DD.")

    try:
        ed = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid end_date format: '{end_date}'. Expected YYYY-MM-DD.")

    if sd >= ed:
        raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date}).")

    return start_date, end_date


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SRTM Elevation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_elevation(latitude: float, longitude: float, buffer: float = 0.0) -> Dict[str, Any]:
    """Get elevation from SRTM DEM.

    If buffer == 0, returns point elevation.
    If buffer > 0, returns statistics (mean, min, max) over a circular area.
    """
    point = ee.Geometry.Point([longitude, latitude])
    dem = ee.Image("USGS/SRTMGL1_003")

    if buffer > 0:
        region = point.buffer(buffer)
        stats = dem.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.min(), sharedInputs=True)
                                     .combine(ee.Reducer.max(), sharedInputs=True),
            geometry=region,
            scale=30,
            bestEffort=True,
        ).getInfo()

        return {
            "elevation_mean": stats.get("elevation_mean"),
            "elevation_min": stats.get("elevation_min"),
            "elevation_max": stats.get("elevation_max"),
            "buffer_m": buffer,
        }

    elevation = (
        dem.sample(point, 30)
        .first()
        .get("elevation")
        .getInfo()
    )
    return {"elevation": elevation, "buffer_m": 0.0}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sentinel-2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_sentinel2_image(
    latitude: float,
    longitude: float,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cloud_percentage: float = 20.0,
) -> Dict[str, Any]:
    sd, ed = validate_dates(start_date, end_date)
    point = ee.Geometry.Point([longitude, latitude])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(sd, ed)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_percentage))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    if collection.size().getInfo() == 0:
        raise ValueError("No Sentinel-2 imagery found for the requested location and date range.")

    image = collection.first()

    image_id = image.get("PRODUCT_ID").getInfo()
    actual_cloud = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()

    thumbnail_url = image.getThumbURL({
        "region": point.buffer(5000).bounds(),
        "dimensions": 512,
        "format": "png",
        "min": 0,
        "max": 3000,
        "bands": ["B4", "B3", "B2"],
    })

    return {
        "image_id": image_id,
        "date": date,
        "cloud_percentage": actual_cloud,
        "thumbnail_url": thumbnail_url,
        "dataset": "COPERNICUS/S2_SR_HARMONIZED",
        "start_date": sd,
        "end_date": ed,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sentinel-1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_sentinel1_image(
    latitude: float,
    longitude: float,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    polarization: Optional[str] = None,
    orbit_pass: Optional[str] = None,
    instrument_mode: Optional[str] = "IW",
) -> Dict[str, Any]:
    sd, ed = validate_dates(start_date, end_date)
    point = ee.Geometry.Point([longitude, latitude])

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(point)
        .filterDate(sd, ed)
    )

    if instrument_mode:
        collection = collection.filter(ee.Filter.eq("instrumentMode", instrument_mode))

    if polarization:
        collection = collection.filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", polarization)
        )

    if orbit_pass:
        collection = collection.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))

    if collection.size().getInfo() == 0:
        raise ValueError("No Sentinel-1 imagery found for the requested location, date range, and filters.")

    image = collection.first()

    system_index = image.get("system:index").getInfo()
    date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()

    # Select visualization band based on polarization
    viz_band = polarization if polarization in ("VV", "VH") else "VV"

    thumbnail_url = image.getThumbURL({
        "region": point.buffer(5000).bounds(),
        "dimensions": 512,
        "format": "png",
        "min": -25,
        "max": 5,
        "bands": [viz_band],
    })

    return {
        "image_id": system_index,
        "date": date,
        "cloud_percentage": None,
        "thumbnail_url": thumbnail_url,
        "dataset": "COPERNICUS/S1_GRD",
        "start_date": sd,
        "end_date": ed,
        "polarization": polarization,
        "orbit_pass": orbit_pass,
        "instrument_mode": instrument_mode,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Landsat 9
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_landsat_image(
    latitude: float,
    longitude: float,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cloud_percentage: float = 20.0,
) -> Dict[str, Any]:
    sd, ed = validate_dates(start_date, end_date)
    point = ee.Geometry.Point([longitude, latitude])

    collection = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(point)
        .filterDate(sd, ed)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_percentage))
        .sort("CLOUD_COVER")
    )

    if collection.size().getInfo() == 0:
        raise ValueError("No Landsat 9 imagery found for the requested location and date range.")

    image = collection.first()

    image_id = image.get("LANDSAT_PRODUCT_ID").getInfo()
    actual_cloud = image.get("CLOUD_COVER").getInfo()
    date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()

    # Scale factors for Landsat 9 Collection 2
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)

    thumbnail_url = optical_bands.getThumbURL({
        "region": point.buffer(5000).bounds(),
        "dimensions": 512,
        "format": "png",
        "min": 0.0,
        "max": 0.3,
        "bands": ["SR_B4", "SR_B3", "SR_B2"],
    })

    return {
        "image_id": image_id,
        "date": date,
        "cloud_percentage": actual_cloud,
        "thumbnail_url": thumbnail_url,
        "dataset": "LANDSAT/LC09/C02/T1_L2",
        "start_date": sd,
        "end_date": ed,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper: Get Sentinel-2 Image for Index Computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _get_s2_image_for_index(
    latitude: float,
    longitude: float,
    start_date: Optional[str],
    end_date: Optional[str],
    cloud_percentage: float,
):
    """Return (image, point, sd, ed, date_str) for spectral index computation."""
    sd, ed = validate_dates(start_date, end_date)
    point = ee.Geometry.Point([longitude, latitude])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(sd, ed)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_percentage))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    if collection.size().getInfo() == 0:
        raise ValueError("No Sentinel-2 imagery found for spectral index computation in the requested date range.")

    image = collection.first()
    date_str = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    return image, point, sd, ed, date_str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Spectral Indices
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_ndvi(
    latitude: float,
    longitude: float,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cloud_percentage: float = 20.0,
) -> Dict[str, Any]:
    """NDVI = (NIR - RED) / (NIR + RED) = (B8 - B4) / (B8 + B4)"""
    image, point, sd, ed, date_str = _get_s2_image_for_index(
        latitude, longitude, start_date, end_date, cloud_percentage
    )
    ndvi = image.normalizedDifference(["B8", "B4"])
    value = ndvi.sample(point, 10).first().get("nd").getInfo()

    return {
        "index_value": value,
        "date": date_str,
        "dataset": "COPERNICUS/S2_SR_HARMONIZED",
        "start_date": sd,
        "end_date": ed,
    }


def get_ndwi(
    latitude: float,
    longitude: float,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cloud_percentage: float = 20.0,
) -> Dict[str, Any]:
    """NDWI = (GREEN - NIR) / (GREEN + NIR) = (B3 - B8) / (B3 + B8)"""
    image, point, sd, ed, date_str = _get_s2_image_for_index(
        latitude, longitude, start_date, end_date, cloud_percentage
    )
    ndwi = image.normalizedDifference(["B3", "B8"])
    value = ndwi.sample(point, 10).first().get("nd").getInfo()

    return {
        "index_value": value,
        "date": date_str,
        "dataset": "COPERNICUS/S2_SR_HARMONIZED",
        "start_date": sd,
        "end_date": ed,
    }


def get_ndbi(
    latitude: float,
    longitude: float,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cloud_percentage: float = 20.0,
) -> Dict[str, Any]:
    """NDBI = (SWIR - NIR) / (SWIR + NIR) = (B11 - B8) / (B11 + B8)"""
    image, point, sd, ed, date_str = _get_s2_image_for_index(
        latitude, longitude, start_date, end_date, cloud_percentage
    )
    ndbi = image.normalizedDifference(["B11", "B8"])
    value = ndbi.sample(point, 10).first().get("nd").getInfo()

    return {
        "index_value": value,
        "date": date_str,
        "dataset": "COPERNICUS/S2_SR_HARMONIZED",
        "start_date": sd,
        "end_date": ed,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Natural-Language Query Planner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

import calendar


def _extract_dates_from_query(query: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract date range from natural language query."""
    q = query.lower()

    # Pattern 1: "between YYYY-MM-DD and YYYY-MM-DD" or "from YYYY-MM-DD to YYYY-MM-DD"
    iso_pattern = re.findall(r"(\d{4}-\d{2}-\d{2})", q)
    if len(iso_pattern) >= 2:
        return iso_pattern[0], iso_pattern[1]

    # Pattern 2: "between Month Day and Month Day Year" or "between Month and Month Year"
    # e.g., "between June 1 and June 30 2024"
    month_day_pattern = re.search(
        r"(?:between|from)\s+(\w+)\s+(\d{1,2})\s+(?:and|to)\s+(\w+)\s+(\d{1,2})\s+(\d{4})",
        q
    )
    if month_day_pattern:
        m1 = _MONTH_MAP.get(month_day_pattern.group(1))
        d1 = int(month_day_pattern.group(2))
        m2 = _MONTH_MAP.get(month_day_pattern.group(3))
        d2 = int(month_day_pattern.group(4))
        y = int(month_day_pattern.group(5))
        if m1 and m2:
            return f"{y}-{m1:02d}-{d1:02d}", f"{y}-{m2:02d}-{d2:02d}"

    # Pattern 3: "from Month to Month Year" or "between Month and Month Year"
    month_range = re.search(
        r"(?:between|from)\s+(\w+)\s+(?:and|to)\s+(\w+)\s+(\d{4})",
        q
    )
    if month_range:
        m1 = _MONTH_MAP.get(month_range.group(1))
        m2 = _MONTH_MAP.get(month_range.group(2))
        y = int(month_range.group(3))
        if m1 and m2:
            last_day = calendar.monthrange(y, m2)[1]
            return f"{y}-{m1:02d}-01", f"{y}-{m2:02d}-{last_day:02d}"

    # Pattern 4: "June and August 2024" (no from/between prefix)
    month_and = re.search(r"(\w+)\s+(?:and|to)\s+(\w+)\s+(\d{4})", q)
    if month_and:
        m1 = _MONTH_MAP.get(month_and.group(1))
        m2 = _MONTH_MAP.get(month_and.group(2))
        y = int(month_and.group(3))
        if m1 and m2:
            last_day = calendar.monthrange(y, m2)[1]
            return f"{y}-{m1:02d}-01", f"{y}-{m2:02d}-{last_day:02d}"

    return None, None


def _extract_cloud_from_query(query: str) -> Optional[float]:
    """Extract cloud cover percentage from query."""
    q = query.lower()
    m = re.search(r"(?:less\s+than|under|below|<|max(?:imum)?)\s+(\d+(?:\.\d+)?)\s*(?:%|percent)", q)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*cloud", q)
    if m:
        return float(m.group(1))
    return None


def _extract_polarization(query: str) -> Optional[str]:
    q = query.upper()
    if "VV,VH" in q or "VH,VV" in q or ("VV" in q and "VH" in q):
        return "VV"  # Default to VV when both mentioned; listContains handles it
    if "VH" in q:
        return "VH"
    if "VV" in q:
        return "VV"
    return None


def _extract_orbit_pass(query: str) -> Optional[str]:
    q = query.lower()
    if "ascending" in q:
        return "ASCENDING"
    if "descending" in q:
        return "DESCENDING"
    return None


def _extract_instrument_mode(query: str) -> Optional[str]:
    q = query.upper()
    for mode in ("IW", "EW", "SM"):
        if mode in q:
            return mode
    return None


def gee_query_planner(query: str, lat: float, lon: float) -> Tuple[str, Any, str]:
    """Parse a natural-language GEE query and execute the appropriate service call."""
    query_lower = query.lower()

    # Extract common parameters
    sd, ed = _extract_dates_from_query(query)
    cloud = _extract_cloud_from_query(query)

    indices_requested = [m for m in ("ndvi", "ndwi", "ndbi", "elevation") if m in query_lower]
    if len(indices_requested) >= 2 or any(k in query_lower for k in ("all measurements", "multi-metric", "multiple metrics", "all available information")):
        multi_data: Dict[str, Any] = {"multi_metric": True}

        # 1. Sentinel-2 (Acquisition Date & Cloud cover)
        try:
            s2 = get_sentinel2_image(lat, lon, sd, ed, cloud or 20.0)
            multi_data["sentinel2"] = s2
            multi_data["date"] = s2.get("date")
            multi_data["cloud_percentage"] = s2.get("cloud_percentage")
        except Exception:
            multi_data["sentinel2"] = None

        # 2. NDVI
        try:
            n = get_ndvi(lat, lon, sd, ed, cloud or 20.0)
            multi_data["ndvi"] = n.get("index_value")
        except Exception:
            multi_data["ndvi"] = None

        # 3. NDWI
        try:
            w = get_ndwi(lat, lon, sd, ed, cloud or 20.0)
            multi_data["ndwi"] = w.get("index_value")
        except Exception:
            multi_data["ndwi"] = None

        # 4. NDBI
        try:
            b = get_ndbi(lat, lon, sd, ed, cloud or 20.0)
            multi_data["ndbi"] = b.get("index_value")
        except Exception:
            multi_data["ndbi"] = None

        # 5. Sentinel-1 SAR
        try:
            pol = _extract_polarization(query)
            orb = _extract_orbit_pass(query)
            mode = _extract_instrument_mode(query) or "IW"
            s1 = get_sentinel1_image(lat, lon, sd, ed, pol, orb, mode)
            multi_data["sentinel1"] = s1
        except Exception:
            multi_data["sentinel1"] = None

        explanation = "Retrieved multi-metric geospatial Earth Engine measurements."
        return "multi_metric", multi_data, explanation

    elif any(kw in query_lower for kw in ["elevation", "height", "altitude", "dem", "srtm"]):
        result = get_elevation(lat, lon)
        val = result.get("elevation", result.get("elevation_mean", 0))
        explanation = (
            f"The elevation at coordinates ({lat}, {lon}) is {val:.2f} meters "
            f"above sea level, retrieved using SRTM digital elevation data."
        )
        return "elevation", result, explanation

    elif "ndvi" in query_lower or "vegetation index" in query_lower:
        result = get_ndvi(lat, lon, sd, ed, cloud or 20.0)
        val = result["index_value"]
        if val > 0.6:
            interpretation = "indicating dense, healthy vegetation"
        elif val > 0.2:
            interpretation = "suggesting moderate vegetation cover"
        elif val > 0:
            interpretation = "indicating sparse vegetation or bare soil"
        else:
            interpretation = "suggesting water, built-up surfaces, or barren land"
        explanation = (
            f"NDVI is {val:.4f}, {interpretation} at the sampled location "
            f"using Sentinel-2 imagery from {result['date']}."
        )
        return "ndvi", result, explanation

    elif "ndwi" in query_lower or "water index" in query_lower:
        result = get_ndwi(lat, lon, sd, ed, cloud or 20.0)
        val = result["index_value"]
        if val > 0.3:
            interpretation = "which strongly suggests open water at this location"
        elif val > 0:
            interpretation = "which may indicate partial water presence or wet surfaces"
        else:
            interpretation = "which generally indicates low open-water presence at this location"
        explanation = (
            f"NDWI is {val:.4f}, {interpretation} "
            f"during the selected observation period (imagery from {result['date']})."
        )
        return "ndwi", result, explanation

    elif "ndbi" in query_lower or "built-up" in query_lower or "built up" in query_lower:
        result = get_ndbi(lat, lon, sd, ed, cloud or 20.0)
        val = result["index_value"]
        if val > 0.1:
            interpretation = (
                "indicating relatively stronger built-up or impervious surface "
                "characteristics at the sampled location"
            )
        elif val > -0.1:
            interpretation = "suggesting mixed land cover with some impervious surfaces"
        else:
            interpretation = "suggesting predominantly vegetated or natural surfaces"
        explanation = (
            f"NDBI is {val:.4f}, {interpretation} "
            f"(imagery from {result['date']})."
        )
        return "ndbi", result, explanation

    elif "sentinel-1" in query_lower or "sar" in query_lower or "radar" in query_lower:
        pol = _extract_polarization(query)
        orb = _extract_orbit_pass(query)
        mode = _extract_instrument_mode(query) or "IW"
        data = get_sentinel1_image(lat, lon, sd, ed, pol, orb, mode)
        explanation = (
            f"Retrieved Sentinel-1 SAR imagery captured on {data['date']}. "
            f"SAR is useful for penetrating clouds and observing surface structure."
        )
        if pol:
            explanation += f" Polarization filter: {pol}."
        if orb:
            explanation += f" Orbit pass: {orb}."
        return "sentinel1", data, explanation

    elif "landsat" in query_lower:
        data = get_landsat_image(lat, lon, sd, ed, cloud or 20.0)
        explanation = (
            f"Retrieved Landsat 9 optical imagery captured on {data['date']} "
            f"with {data['cloud_percentage']:.1f}% cloud cover."
        )
        return "landsat", data, explanation

    else:
        # Default to Sentinel-2 optical
        data = get_sentinel2_image(lat, lon, sd, ed, cloud or 20.0)
        explanation = (
            f"Retrieved Sentinel-2 optical imagery captured on {data['date']} "
            f"with {data['cloud_percentage']:.1f}% cloud cover."
        )
        return "sentinel2", data, explanation