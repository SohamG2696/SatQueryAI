"""SatQuery AI — Google Earth Engine API Router."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas.gee import (
    GEEElevationResponse,
    GEEImageResponse,
    GEEIndexResponse,
    GEENaturalQueryRequest,
    GEEQueryResponse,
)
from app.services.earth_engine import (
    get_elevation,
    get_landsat_image,
    get_ndbi,
    get_ndvi,
    get_ndwi,
    get_sentinel1_image,
    get_sentinel2_image,
    gee_query_planner,
)

router = APIRouter(prefix="/api/gee", tags=["Earth Engine"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SRTM Elevation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get("/elevation", response_model=GEEElevationResponse)
async def fetch_elevation(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    buffer: float = Query(0.0, description="Buffer radius in meters (0 = point query)"),
):
    """Get SRTM elevation at a point or area statistics with buffer."""
    try:
        result = get_elevation(lat, lon, buffer)

        if buffer > 0:
            mean_val = result.get("elevation_mean", 0)
            return GEEElevationResponse(
                latitude=lat,
                longitude=lon,
                elevation_mean=result.get("elevation_mean"),
                elevation_min=result.get("elevation_min"),
                elevation_max=result.get("elevation_max"),
                buffer_m=buffer,
                explanation=(
                    f"Elevation statistics within {buffer:.0f}m radius of ({lat}, {lon}): "
                    f"mean={result.get('elevation_mean'):.2f}m, "
                    f"min={result.get('elevation_min'):.2f}m, "
                    f"max={result.get('elevation_max'):.2f}m."
                ),
            )

        val = result["elevation"]
        return GEEElevationResponse(
            latitude=lat,
            longitude=lon,
            elevation_m=val,
            buffer_m=0.0,
            explanation=f"The elevation at coordinates ({lat}, {lon}) is {val:.2f} meters above sea level.",
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sentinel-2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get("/sentinel2", response_model=GEEImageResponse)
async def fetch_sentinel2(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD, exclusive in GEE)"),
    cloud_percentage: float = Query(20.0, description="Maximum cloud cover percentage"),
):
    """Retrieve Sentinel-2 Surface Reflectance imagery."""
    try:
        data = get_sentinel2_image(lat, lon, start_date, end_date, cloud_percentage)
        return GEEImageResponse(
            image_id=data["image_id"],
            date=data["date"],
            cloud_percentage=data.get("cloud_percentage"),
            thumbnail_url=data["thumbnail_url"],
            explanation=f"Sentinel-2 optical image captured on {data['date']} with {data.get('cloud_percentage', 0):.1f}% cloud cover.",
            dataset=data.get("dataset"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            metadata=data,
        )
    except ValueError as ve:
        msg = str(ve)
        if "format" in msg or "must be before" in msg or "must be provided" in msg:
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=404, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sentinel-1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get("/sentinel1", response_model=GEEImageResponse)
async def fetch_sentinel1(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD, exclusive in GEE)"),
    polarization: Optional[str] = Query(None, description="Polarization: VV, VH, or VV,VH"),
    orbit_pass: Optional[str] = Query(None, description="Orbit pass: ASCENDING or DESCENDING"),
    instrument_mode: Optional[str] = Query("IW", description="Instrument mode: IW, EW, or SM"),
):
    """Retrieve Sentinel-1 GRD SAR imagery."""
    try:
        data = get_sentinel1_image(
            lat, lon, start_date, end_date, polarization, orbit_pass, instrument_mode
        )
        explanation = f"Sentinel-1 SAR image captured on {data['date']}."
        if polarization:
            explanation += f" Polarization: {polarization}."
        if orbit_pass:
            explanation += f" Orbit: {orbit_pass}."

        return GEEImageResponse(
            image_id=data["image_id"],
            date=data["date"],
            cloud_percentage=None,
            thumbnail_url=data["thumbnail_url"],
            explanation=explanation,
            dataset=data.get("dataset"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            metadata=data,
        )
    except ValueError as ve:
        msg = str(ve)
        if "format" in msg or "must be before" in msg or "must be provided" in msg:
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=404, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Landsat 9
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get("/landsat", response_model=GEEImageResponse)
async def fetch_landsat(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD, exclusive in GEE)"),
    cloud_percentage: float = Query(20.0, description="Maximum cloud cover percentage"),
):
    """Retrieve Landsat 9 Collection 2 Level-2 imagery."""
    try:
        data = get_landsat_image(lat, lon, start_date, end_date, cloud_percentage)
        return GEEImageResponse(
            image_id=data["image_id"],
            date=data["date"],
            cloud_percentage=data.get("cloud_percentage"),
            thumbnail_url=data["thumbnail_url"],
            explanation=f"Landsat 9 image captured on {data['date']} with {data.get('cloud_percentage', 0):.1f}% cloud cover.",
            dataset=data.get("dataset"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            metadata=data,
        )
    except ValueError as ve:
        msg = str(ve)
        if "format" in msg or "must be before" in msg or "must be provided" in msg:
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=404, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Spectral Indices
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get("/ndvi", response_model=GEEIndexResponse)
async def fetch_ndvi(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    cloud_percentage: float = Query(20.0, description="Maximum cloud cover percentage"),
):
    """Calculate NDVI = (B8 - B4) / (B8 + B4) from Sentinel-2."""
    try:
        result = get_ndvi(lat, lon, start_date, end_date, cloud_percentage)
        val = result["index_value"]

        if val > 0.6:
            interp = "indicating dense, healthy vegetation"
        elif val > 0.2:
            interp = "suggesting moderate vegetation cover"
        elif val > 0:
            interp = "indicating sparse vegetation or bare soil"
        else:
            interp = "suggesting water, built-up surfaces, or barren land"

        return GEEIndexResponse(
            latitude=lat,
            longitude=lon,
            index_name="NDVI",
            index_value=val,
            date=result.get("date"),
            dataset=result.get("dataset"),
            start_date=result.get("start_date"),
            end_date=result.get("end_date"),
            explanation=f"NDVI is {val:.4f}, {interp} (imagery from {result['date']}).",
        )
    except ValueError as ve:
        msg = str(ve)
        if "format" in msg or "must be before" in msg or "must be provided" in msg:
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=404, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ndwi", response_model=GEEIndexResponse)
async def fetch_ndwi(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    cloud_percentage: float = Query(20.0, description="Maximum cloud cover percentage"),
):
    """Calculate NDWI = (B3 - B8) / (B3 + B8) from Sentinel-2."""
    try:
        result = get_ndwi(lat, lon, start_date, end_date, cloud_percentage)
        val = result["index_value"]

        if val > 0.3:
            interp = "which strongly suggests open water at this location"
        elif val > 0:
            interp = "which may indicate partial water presence or wet surfaces"
        else:
            interp = "which generally indicates low open-water presence at this location"

        return GEEIndexResponse(
            latitude=lat,
            longitude=lon,
            index_name="NDWI",
            index_value=val,
            date=result.get("date"),
            dataset=result.get("dataset"),
            start_date=result.get("start_date"),
            end_date=result.get("end_date"),
            explanation=f"NDWI is {val:.4f}, {interp} during the selected observation period (imagery from {result['date']}).",
        )
    except ValueError as ve:
        msg = str(ve)
        if "format" in msg or "must be before" in msg or "must be provided" in msg:
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=404, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ndbi", response_model=GEEIndexResponse)
async def fetch_ndbi(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    cloud_percentage: float = Query(20.0, description="Maximum cloud cover percentage"),
):
    """Calculate NDBI = (B11 - B8) / (B11 + B8) from Sentinel-2."""
    try:
        result = get_ndbi(lat, lon, start_date, end_date, cloud_percentage)
        val = result["index_value"]

        if val > 0.1:
            interp = "indicating relatively stronger built-up or impervious surface characteristics at the sampled location"
        elif val > -0.1:
            interp = "suggesting mixed land cover with some impervious surfaces"
        else:
            interp = "suggesting predominantly vegetated or natural surfaces"

        return GEEIndexResponse(
            latitude=lat,
            longitude=lon,
            index_name="NDBI",
            index_value=val,
            date=result.get("date"),
            dataset=result.get("dataset"),
            start_date=result.get("start_date"),
            end_date=result.get("end_date"),
            explanation=f"NDBI is {val:.4f}, {interp} (imagery from {result['date']}).",
        )
    except ValueError as ve:
        msg = str(ve)
        if "format" in msg or "must be before" in msg or "must be provided" in msg:
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=404, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Natural-Language Query Planner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post("/query", response_model=GEEQueryResponse)
async def execute_gee_query(request: GEENaturalQueryRequest):
    """Execute a natural-language GEE query with automatic parameter extraction."""
    try:
        intent, result, explanation = gee_query_planner(
            request.query, request.latitude, request.longitude
        )
        return GEEQueryResponse(
            intent=intent,
            result=result,
            explanation=explanation,
        )
    except ValueError as ve:
        msg = str(ve)
        if "format" in msg or "must be before" in msg or "must be provided" in msg:
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=404, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
