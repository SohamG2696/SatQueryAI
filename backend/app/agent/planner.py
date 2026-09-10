"""
SatQuery AI — Task-Aware Natural Language Query Planner & Input Validator.

Parses user natural-language queries, metadata, and image inputs into a
structured AnalysisPlan. Determines task-specific input requirements and
performs validation AFTER task resolution.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Structured Analysis Plan Schema
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class AnalysisPlan:
    """Structured execution plan produced by the NLP / AI Analysis Planner."""

    primary_task: str  # 'gee', 'vqa', 'captioning', 'grounding', 'change_vqa', 'fusion', 'multi_model'
    operation: str  # 'imagery_retrieval', 'index_calculation', 'elevation_query', 'visual_qa', 'scene_description', 'spatial_grounding', 'bi_temporal_change', 'cross_modal_fusion', 'multi_task'
    target: Optional[str] = None
    dataset: Optional[str] = None
    modality: Optional[str] = None
    requires_uploaded_images: bool = True
    min_images: int = 1
    required_modalities: List[str] = field(default_factory=list)
    requires_location: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    cloud_percentage: Optional[float] = None
    polarization: Optional[str] = None
    orbit_pass: Optional[str] = None
    instrument_mode: Optional[str] = None
    visualization_requested: bool = False
    comparison_requested: bool = False
    spatial_localization_requested: bool = False
    sub_tasks: List[str] = field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Regex & Extraction Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


def extract_coordinates(text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[Optional[float], Optional[float]]:
    """Extract latitude and longitude from metadata or natural language text."""
    meta = metadata or {}
    lat = meta.get("latitude", meta.get("lat"))
    lon = meta.get("longitude", meta.get("lon"))

    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon)
        except (ValueError, TypeError):
            pass

    q = text.lower()

    # Pattern A: latitude 19.076 and longitude 72.8777 or lat 19.076 lon 72.8777
    m = re.search(r"(?:lat(?:itude)?\s*[:=]?\s*)([-+]?\d+(?:\.\d+)?)\s*(?:,?\s*and\s*|\s*,?\s*)(?:lon(?:gitude)?\s*[:=]?\s*)([-+]?\d+(?:\.\d+)?)", q)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Pattern B: lat/lon in reverse or generic lat/long
    m = re.search(r"at\s+lat(?:itude)?\s+([-+]?\d+(?:\.\d+)?)\s+and\s+lon(?:gitude)?\s+([-+]?\d+(?:\.\d+)?)", q)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Pattern C: at (19.076, 72.8777) or coordinates 19.076, 72.8777
    m = re.search(r"(?:at|coordinates)\s*\(?\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)\s*\)?", q)
    if m:
        return float(m.group(1)), float(m.group(2))

    return None, None


def extract_dates(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract start_date and end_date from text."""
    q = text.lower()

    # Pattern 1: ISO YYYY-MM-DD pair
    iso_matches = re.findall(r"(\d{4}-\d{2}-\d{2})", q)
    if len(iso_matches) >= 2:
        return iso_matches[0], iso_matches[1]

    # Pattern 2: "between Month Day and Month Day Year"
    m_day = re.search(
        r"(?:between|from)\s+(\w+)\s+(\d{1,2})\s+(?:and|to)\s+(\w+)\s+(\d{1,2})\s+(\d{4})",
        q
    )
    if m_day:
        m1 = _MONTH_MAP.get(m_day.group(1))
        d1 = int(m_day.group(2))
        m2 = _MONTH_MAP.get(m_day.group(3))
        d2 = int(m_day.group(4))
        y = int(m_day.group(5))
        if m1 and m2:
            return f"{y}-{m1:02d}-{d1:02d}", f"{y}-{m2:02d}-{d2:02d}"

    # Pattern 3: "between Month and Month Year" or "from Month to Month Year"
    m_range = re.search(r"(?:between|from)\s+(\w+)\s+(?:and|to)\s+(\w+)\s+(\d{4})", q)
    if m_range:
        m1 = _MONTH_MAP.get(m_range.group(1))
        m2 = _MONTH_MAP.get(m_range.group(2))
        y = int(m_range.group(3))
        if m1 and m2:
            last_day = calendar.monthrange(y, m2)[1]
            return f"{y}-{m1:02d}-01", f"{y}-{m2:02d}-{last_day:02d}"

    # Pattern 4: "during 2024" or "in 2024"
    m_year = re.search(r"(?:during|in|for)\s+(\d{4})\b", q)
    if m_year:
        y = int(m_year.group(1))
        return f"{y}-01-01", f"{y}-12-31"

    return None, None


def extract_cloud(text: str) -> Optional[float]:
    """Extract cloud cover percentage from query."""
    q = text.lower()
    m = re.search(r"(?:less\s+than|under|below|<|max(?:imum)?)\s+(\d+(?:\.\d+)?)\s*(?:%|percent)", q)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*cloud", q)
    if m:
        return float(m.group(1))
    return None


def extract_sar_params(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract polarization, orbit_pass, and instrument_mode."""
    q = text.upper()
    q_lower = text.lower()

    pol = None
    if "VV,VH" in q or "VH,VV" in q or ("VV" in q and "VH" in q):
        pol = "VV"
    elif "VH" in q:
        pol = "VH"
    elif "VV" in q:
        pol = "VV"

    orb = None
    if "ascending" in q_lower:
        orb = "ASCENDING"
    elif "descending" in q_lower:
        orb = "DESCENDING"

    mode = "IW"
    for m in ("IW", "EW", "SM"):
        if m in q:
            mode = m
            break

    return pol, orb, mode


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP / AI Analysis Planner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_analysis_plan(
    query: str,
    image_count: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> AnalysisPlan:
    """Analyze query and metadata to produce a structured AnalysisPlan.

    This function resolves what task is requested BEFORE input validation.
    """
    meta = metadata or {}
    q_lower = query.lower().strip()

    modalities = [m.lower().strip() for m in (meta.get("modalities") or [])]
    has_optical = any("optical" in m or "rgb" in m or "s2" in m for m in modalities)
    has_sar = any("sar" in m or "radar" in m or "s1" in m for m in modalities)
    dates_meta = meta.get("dates") or []
    has_diff_dates = len(dates_meta) >= 2 and dates_meta[0] != dates_meta[1]

    # Extract coordinates, dates, cloud, SAR params
    lat, lon = extract_coordinates(query, meta)
    sd, ed = extract_dates(query)
    if not sd and dates_meta and len(dates_meta) >= 2:
        sd, ed = dates_meta[0], dates_meta[1]
    cloud = extract_cloud(query) or meta.get("cloud_percentage")
    pol, orb, mode = extract_sar_params(query)

    # 0. Detect Future Prediction Intent
    is_future_pred = False
    try:
        from models.future_prediction.inference.query_parser import parse_prediction_query
        pq = parse_prediction_query(query)
        is_future_pred = pq.get("is_prediction_query", False)
    except Exception:
        is_future_pred = False

    # 1. Detect GEE Intent
    # Keywords indicating GEE dataset retrieval, index computation, elevation
    gee_dataset_kw = (
        "sentinel-1", "sentinel-2", "landsat", "ndvi", "ndwi", "ndbi",
        "elevation", "srtm", "dem", "earth engine", "gee", "fetch",
        "spectral index", "vegetation index", "water index", "built-up index"
    )

    is_gee_explicit = any(k in q_lower for k in gee_dataset_kw)

    # Disambiguation: Is it GEE retrieval vs image analysis on uploaded images?
    # GEE requests ask to retrieve/fetch data or calculate indices/elevation (e.g. "Find a Sentinel-2 image", "Calculate NDVI").
    # If the user uploaded an image AND asked e.g. "Find the buildings in this image", that is Grounding!
    is_gee_query = False
    gee_operation = "imagery_retrieval"
    gee_dataset = None

    if is_gee_explicit:
        # Check if query requests multiple GEE metrics (e.g. NDVI + NDWI + NDBI)
        indices_requested = [m for m in ("ndvi", "ndwi", "ndbi", "elevation") if m in q_lower]
        is_multi_request = len(indices_requested) >= 2 or any(k in q_lower for k in ("all measurements", "multi-metric", "multiple metrics", "all available information"))
        
        if is_multi_request:
            gee_operation = "multi_metric"
            gee_dataset = "COPERNICUS/S2_SR_HARMONIZED"
        elif any(k in q_lower for k in ("elevation", "height", "altitude", "dem", "srtm")):
            gee_operation = "elevation_query"
            gee_dataset = "USGS/SRTMGL1_003"
        elif "ndvi" in q_lower or "vegetation index" in q_lower:
            gee_operation = "index_calculation"
            gee_dataset = "COPERNICUS/S2_SR_HARMONIZED"
        elif "ndwi" in q_lower or "water index" in q_lower:
            gee_operation = "index_calculation"
            gee_dataset = "COPERNICUS/S2_SR_HARMONIZED"
        elif "ndbi" in q_lower or "built-up index" in q_lower:
            gee_operation = "index_calculation"
            gee_dataset = "COPERNICUS/S2_SR_HARMONIZED"
        elif "sentinel-1" in q_lower or "sar image" in q_lower or "sar imagery" in q_lower:
            gee_operation = "imagery_retrieval"
            gee_dataset = "COPERNICUS/S1_GRD"
        elif "landsat" in q_lower:
            gee_operation = "imagery_retrieval"
            gee_dataset = "LANDSAT/LC09/C02/T1_L2"
        else:
            gee_operation = "imagery_retrieval"
            gee_dataset = "COPERNICUS/S2_SR_HARMONIZED"

        # Check if this GEE query is a standalone data/index request (no uploaded images or explicit request to fetch)
        if image_count == 0 or any(k in q_lower for k in ("fetch", "get", "find a sentinel", "find sentinel", "calculate", "what is the elevation", "for this location")):
            is_gee_query = True

    # 2. Detect Change Detection Intent (Bi-temporal comparison)
    change_semantic_kw = (
        "what changed", "changed between", "difference between", "has changed", "have changed",
        "detect change", "detect changes", "change detection", "increased between", "decreased between",
        "urban expansion", "deforestation", "growth between", "loss between", "before and after",
        "changes in", "change in", "area changed", "areas changed", "buildings changed",
        "building changed", "vegetation changed"
    )
    change_patterns = (
        r"\b(have|has|did)\b.+\b(changed?|increased|decreased|grown|shrunk)\b",
        r"\b(are|is)\s+there\s+.*\bchanges?\b",
        r"\bchanges?\s+in\b",
        r"\b(detect|show|identify|find|analyze)\s+.*\bchanges?\b",
    )
    has_change_pattern = any(re.search(pat, q_lower) for pat in change_patterns)

    is_change_semantic = (
        any(k in q_lower for k in change_semantic_kw)
        or has_change_pattern
        or (
            "change" in q_lower and any(k in q_lower for k in ("between", "images", "dates", "before", "after"))
        )
    )

    is_change_query = (
        is_change_semantic or has_diff_dates
    ) and not is_gee_query and not is_future_pred

    # 3. Detect Optical-SAR Fusion Intent
    fusion_explicit_kw = (
        "optical and sar", "sar and optical", "both modalities", "cross-modal",
        "fuse optical", "fusion model", "confirm using both", "support optical"
    )
    is_fusion_explicit = any(k in q_lower for k in fusion_explicit_kw)
    is_fusion_query = (
        (has_optical and has_sar) or is_fusion_explicit
    ) and not is_gee_query and not is_future_pred

    # 4. Detect Grounding Intent
    grounding_kw = (
        "find the", "locate the", "show where", "bounding box", "bbox",
        "where are the", "where is the", "segment", "highlight the", "locate"
    )
    is_grounding_query = any(k in q_lower for k in grounding_kw) and not is_gee_query and not is_future_pred

    # 5. Detect Captioning Intent
    caption_kw = ("describe", "caption", "summarize", "overview", "what is in this image")
    is_caption_query = (not q_lower or any(k in q_lower for k in caption_kw)) and not (is_grounding_query or is_change_query or is_fusion_query or is_gee_query or is_future_pred)

    # 6. Detect DL + GEE Combined Intent
    geospatial_evidence_kw = (
        "use available geospatial data", "geospatial data", "geospatial evidence",
        "supporting evidence", "verify with geospatial", "use gee"
    )
    is_combined_dl_gee = image_count >= 1 and any(k in q_lower for k in geospatial_evidence_kw)

    # 7. Assemble Sub-tasks and Primary Task
    sub_tasks: List[str] = []

    if is_future_pred:
        sub_tasks.append("future_prediction")
    if is_gee_query:
        sub_tasks.append("gee")
    if is_grounding_query:
        sub_tasks.append("grounding")
    if is_change_query:
        sub_tasks.append("change_vqa")
    if is_fusion_query:
        sub_tasks.append("fusion")
    if is_caption_query:
        sub_tasks.append("captioning")

    # If user requested DL + GEE combined analysis or GEE alongside image analysis
    if is_combined_dl_gee or (is_gee_explicit and image_count >= 1 and not is_gee_query):
        if "gee" not in sub_tasks:
            sub_tasks.append("gee")
        if not any(t in sub_tasks for t in ("grounding", "change_vqa", "fusion", "captioning")):
            sub_tasks.append("vqa")

    if not sub_tasks:
        sub_tasks.append("vqa")

    # Resolve Primary Task
    if len(sub_tasks) > 1:
        primary_task = "multi_model"
        operation = "multi_task"
    else:
        primary_task = sub_tasks[0]
        op_map = {
            "future_prediction": "multi_year_landcover_forecasting",
            "gee": gee_operation,
            "grounding": "spatial_grounding",
            "change_vqa": "bi_temporal_change",
            "fusion": "cross_modal_fusion",
            "captioning": "scene_description",
            "vqa": "visual_qa",
        }
        operation = op_map.get(primary_task, "visual_qa")

    # Determine Input Requirements
    requires_uploaded_images = True
    min_images = 1
    required_modalities: List[str] = []
    requires_location = False

    if primary_task == "future_prediction":
        requires_uploaded_images = True
        min_images = 2
    elif primary_task == "gee":
        requires_uploaded_images = False
        min_images = 0
        requires_location = True
    elif primary_task == "change_vqa":
        requires_uploaded_images = True
        min_images = 2
    elif primary_task == "fusion":
        requires_uploaded_images = True
        min_images = 2
        required_modalities = ["optical", "sar"]
    elif primary_task == "multi_model":
        # Dynamic requirements based on sub-tasks
        requires_uploaded_images = any(st != "gee" for st in sub_tasks)
        min_images = 0
        if "change_vqa" in sub_tasks or "fusion" in sub_tasks:
            min_images = 2
        elif requires_uploaded_images:
            min_images = 1
        requires_location = "gee" in sub_tasks
        if "fusion" in sub_tasks:
            required_modalities = ["optical", "sar"]
    else:
        # VQA, Grounding, Captioning
        requires_uploaded_images = True
        min_images = 1

    return AnalysisPlan(
        primary_task=primary_task,
        operation=operation,
        target=None,
        dataset=gee_dataset,
        modality="sar" if "sentinel-1" in q_lower else ("optical" if "sentinel-2" in q_lower or "landsat" in q_lower else None),
        requires_uploaded_images=requires_uploaded_images,
        min_images=min_images,
        required_modalities=required_modalities,
        requires_location=requires_location,
        latitude=lat,
        longitude=lon,
        start_date=sd,
        end_date=ed,
        cloud_percentage=cloud,
        polarization=pol,
        orbit_pass=orb,
        instrument_mode=mode,
        visualization_requested=is_grounding_query or "show" in q_lower or "visualize" in q_lower,
        comparison_requested=is_change_query or is_fusion_query,
        spatial_localization_requested=is_grounding_query,
        sub_tasks=sub_tasks,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input Requirement Validator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def validate_plan_inputs(
    plan: AnalysisPlan,
    image_count: int,
    modalities: Optional[List[str]] = None,
) -> None:
    """Validate provided inputs against the resolved AnalysisPlan requirements.

    Raises task-specific ValueError if input requirements are not met.
    Must be called AFTER the plan has been resolved.
    """
    # 1. Location Validation for GEE
    if plan.requires_location:
        if plan.latitude is None or plan.longitude is None:
            raise ValueError("Latitude and longitude are required for this GEE operation.")

    # 2. Uploaded Image Count Validation
    if plan.requires_uploaded_images and image_count < plan.min_images:
        task = plan.primary_task
        if task == "future_prediction":
            raise ValueError("Multi-year land-cover future forecasting requires at least 2 historical satellite images.")
        elif task in ("vqa", "captioning", "question"):
            raise ValueError("This query requires one satellite image.")
        elif task == "grounding":
            raise ValueError("Spatial grounding requires one satellite image.")
        elif task in ("change_vqa", "change"):
            raise ValueError("Bi-temporal change analysis requires two satellite images: before and after.")
        elif task == "fusion":
            raise ValueError("Optical-SAR fusion requires both an optical image and a SAR image.")
        elif task == "multi_model":
            if plan.min_images >= 2:
                raise ValueError("This multi-task query requires two satellite images.")
            else:
                raise ValueError("This multi-task query requires at least one satellite image.")
        else:
            raise ValueError(f"Task '{task}' requires at least {plan.min_images} satellite image(s).")

    # 3. Modality Validation for Fusion
    if plan.required_modalities and image_count >= 2:
        mods = [m.lower().strip() for m in (modalities or [])]
        has_opt = any("optical" in m or "rgb" in m or "s2" in m for m in mods)
        has_sar = any("sar" in m or "radar" in m or "s1" in m for m in mods)
        if mods and not (has_opt and has_sar):
            raise ValueError("Optical-SAR fusion requires both an optical image and a SAR image.")
