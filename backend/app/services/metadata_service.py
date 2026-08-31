"""
SatQuery AI — Metadata & Modality Detection Service.

Extracts acquisition timestamps, determines sensor modalities (Optical vs SAR),
and resolves spatial and spectral attributes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def detect_modality(
    filename: str = "",
    explicit_modality: str | None = None,
    bands: int = 3,
    raster_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Detect or confirm image modality (optical vs sar).

    Parameters
    ----------
    filename : str
        Original filename.
    explicit_modality : str | None
        Explicit modality provided by the client/frontend.
    bands : int
        Number of image bands.
    raster_metadata : dict[str, Any] | None
        Optional raster metadata dictionary.

    Returns
    -------
    dict[str, str]
        {
            "modality": "optical" | "sar" | "multispectral",
            "detection_method": "metadata" | "heuristic" | "filename"
        }
    """
    # 1. Explicit Client Metadata Preference
    if explicit_modality and explicit_modality.strip():
        mod = explicit_modality.strip().lower()
        if mod in ("optical", "rgb", "true_color", "multispectral"):
            return {"modality": "optical", "detection_method": "metadata"}
        elif mod in ("sar", "radar", "sentinel1", "s1"):
            return {"modality": "sar", "detection_method": "metadata"}
        return {"modality": mod, "detection_method": "metadata"}

    # 2. Filename Pattern Heuristic
    fname_upper = filename.upper()
    if any(k in fname_upper for k in ("S1", "SAR", "GRD", "_VH", "_VV", "SENTINEL-1")):
        return {"modality": "sar", "detection_method": "filename"}

    if any(k in fname_upper for k in ("S2", "OPTICAL", "RGB", "B02", "B03", "B04", "B08", "SENTINEL-2")):
        return {"modality": "optical", "detection_method": "filename"}

    # 3. Band Count & Channel Characteristics Heuristic
    if bands in (1, 2):
        return {"modality": "sar", "detection_method": "heuristic"}
    elif bands in (3, 4, 12, 13):
        return {"modality": "optical", "detection_method": "heuristic"}

    return {"modality": "optical", "detection_method": "heuristic"}


def extract_timestamp_from_filename(filename: str) -> str | None:
    """Extract standard ISO timestamp from satellite filenames.

    Example patterns:
    - 20170613T165043 -> 2017-06-13T16:50:43
    - 2024-05-18 -> 2024-05-18
    """
    # ISO-like date in filename (YYYY-MM-DD)
    match_iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if match_iso:
        return match_iso.group(0)

    # Sentinel compact timestamp (YYYYMMDDTHHMMSS)
    match_sentinel = re.search(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", filename)
    if match_sentinel:
        y, m, d, hh, mm, ss = match_sentinel.groups()
        return f"{y}-{m}-{d}T{hh}:{mm}:{ss}"

    # Compact 8-digit date (YYYYMMDD)
    match_compact = re.search(r"\b(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b", filename)
    if match_compact:
        y, m, d = match_compact.groups()
        return f"{y}-{m}-{d}"

    return None
