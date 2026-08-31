"""
SatQuery AI — Geospatial & Co-Registration Service.

Validates spatial alignment, CRS compatibility, bounding box overlaps,
and geometric consistency for multi-temporal and cross-modal image pairs.
"""

from __future__ import annotations

from typing import Any


def check_coregistration(
    meta1: dict[str, Any],
    meta2: dict[str, Any],
    tolerance_pct: float = 0.05,
) -> dict[str, Any]:
    """Determine whether two images are spatially aligned / co-registered.

    Parameters
    ----------
    meta1 : dict[str, Any]
        Metadata dict for first image (from image_service).
    meta2 : dict[str, Any]
        Metadata dict for second image (from image_service).
    tolerance_pct : float
        Allowed dimension/extent discrepancy percentage.

    Returns
    -------
    dict[str, Any]
        {
            "co_registered": bool,
            "status": "aligned" | "compatible" | "unaligned",
            "crs_match": bool,
            "dimension_match": bool,
            "details": str
        }
    """
    w1, h1 = meta1.get("width", 0), meta1.get("height", 0)
    w2, h2 = meta2.get("width", 0), meta2.get("height", 0)
    crs1, crs2 = meta1.get("crs"), meta2.get("crs")
    bounds1, bounds2 = meta1.get("bounds"), meta2.get("bounds")

    # 1. Dimension check
    dim_match = (w1 == w2 and h1 == h2)
    if not dim_match and w1 > 0 and w2 > 0 and h1 > 0 and h2 > 0:
        w_diff = abs(w1 - w2) / max(w1, w2)
        h_diff = abs(h1 - h2) / max(h1, h2)
        dim_match = (w_diff <= tolerance_pct and h_diff <= tolerance_pct)

    # 2. CRS check
    crs_match = True
    if crs1 and crs2:
        crs_match = (str(crs1).strip().upper() == str(crs2).strip().upper())

    # 3. Bounds / Extent check if available
    bounds_match = True
    if bounds1 and bounds2 and len(bounds1) == 4 and len(bounds2) == 4:
        # Check IoU or coordinate overlap
        min_x1, min_y1, max_x1, max_y1 = bounds1
        min_x2, min_y2, max_x2, max_y2 = bounds2

        overlap_min_x = max(min_x1, min_x2)
        overlap_min_y = max(min_y1, min_y2)
        overlap_max_x = min(max_x1, max_x2)
        overlap_max_y = min(max_y1, max_y2)

        if overlap_max_x <= overlap_min_x or overlap_max_y <= overlap_min_y:
            bounds_match = False

    is_aligned = dim_match and crs_match and bounds_match

    status = "aligned" if is_aligned else ("compatible" if dim_match else "unaligned")
    details = (
        "Images are co-registered with matching dimensions and spatial extent."
        if is_aligned
        else f"Spatial discrepancies detected: dimensions_match={dim_match}, crs_match={crs_match}, bounds_match={bounds_match}."
    )

    return {
        "co_registered": is_aligned,
        "status": status,
        "crs_match": crs_match,
        "dimension_match": dim_match,
        "details": details,
    }
