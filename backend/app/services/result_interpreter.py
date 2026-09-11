from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from app.schemas.execution import VisualEvidence

"""
SatQuery AI — Query-Aware Result Interpretation & Multi-Model Synthesis Layer.

Converts structured Deep Learning specialist model outputs and supporting GEE
geospatial evidence into factual, natural-language explanations without overclaiming,
speculating, or inventing unverified facts.

Architectural Guarantees:
- DL Specialist Models (ChangeFormer, Grounding, Fusion, VLM) remain the PRIMARY AI analysis.
- GEE provides additional/supporting geospatial evidence when invoked.
- Model confidence is explicitly distinguished from ground-truth accuracy.
- Mathematically derived values (pixel ratios, percentages) are cleanly calculated.
- Low-confidence spatial bounding boxes are flagged as unreliable.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Output Schema
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Interpretation:
    """Structured interpretation of a model or GEE result."""

    answer: str = ""
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    technical_interpretation: str = ""
    confidence_explanation: str = ""
    limitations: str = ""
    raw_result: Any = None

    def to_formatted_answer(self) -> str:
        """Format the interpretation into a clean, ChatGPT-style user-facing answer."""
        parts = []
        if self.summary:
            clean_sum = re.sub(r"\.{2,}", ".", self.summary).strip()
            parts.append(clean_sum)

        if self.key_findings:
            bullets = "\n".join("- " + re.sub(r"\.{2,}", ".", f).strip() for f in self.key_findings)
            parts.append(f"Key findings:\n{bullets}")

        if self.technical_interpretation:
            clean_tech = re.sub(r"\.{2,}", ".", self.technical_interpretation).strip()
            parts.append(clean_tech)

        if self.limitations:
            clean_lim = re.sub(r"\.{2,}", ".", self.limitations).strip()
            if not clean_lim.lower().startswith("limitations"):
                parts.append(f"Limitations and uncertainty:\n{clean_lim}")
            else:
                parts.append(clean_lim)

        return "\n\n".join(parts).strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def interpret_result(
    query: str,
    task: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]] = None,
    confidence: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Interpretation:
    """Interpret a structured model/GEE result into query-aware, human-readable language."""
    task_lower = task.lower().strip()
    conf = confidence if confidence is not None else result.get("confidence")

    dispatch = {
        "vqa": _interpret_vqa,
        "captioning": _interpret_captioning,
        "grounding": _interpret_grounding,
        "change_vqa": _interpret_change,
        "change": _interpret_change,
        "fusion": _interpret_fusion,
        "gee": _interpret_gee,
        "elevation": _interpret_gee_elevation,
        "ndvi": _interpret_gee_ndvi,
        "ndwi": _interpret_gee_ndwi,
        "ndbi": _interpret_gee_ndbi,
        "sentinel1": _interpret_gee_imagery,
        "sentinel2": _interpret_gee_imagery,
        "landsat": _interpret_gee_imagery,
    }

    handler = dispatch.get(task_lower, _interpret_generic)
    interp = handler(query, result, evidence, metadata)

    # Attach factual confidence explanation
    if conf is not None:
        interp.confidence_explanation = _explain_confidence(conf, task_lower)

    # Always preserve the raw result
    interp.raw_result = result

    return interp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Confidence Helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _explain_confidence(conf: float, task: str) -> str:
    """Generate a factual confidence explanation distinguishing confidence from accuracy."""
    c = conf if conf <= 1.0 else conf / 100.0
    pct = c * 100

    if c < 0.20:
        return f"The model assigned very low confidence ({pct:.2f}%) to this prediction; interpret with caution as model confidence is not equivalent to ground-truth accuracy."
    elif c < 0.50:
        return f"The model assigned low confidence ({pct:.2f}%) to this prediction; interpret with caution as model confidence is not equivalent to ground-truth accuracy."
    elif c <= 0.75:
        return f"The model assigned moderate confidence ({pct:.2f}%) to this prediction; note that model confidence is not equivalent to ground-truth accuracy."
    elif c <= 0.90:
        return f"The model assigned relatively high confidence ({pct:.2f}%) to this prediction; note that model confidence is not equivalent to ground-truth accuracy."
    else:
        return f"The model assigned high confidence ({pct:.2f}%) to this prediction; note that model confidence is not equivalent to ground-truth accuracy."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VLM / VQA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _interpret_vqa(
    query: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Interpretation:
    raw_answer = _safe_str(result.get("answer", ""))
    clean_answer = re.sub(r"\.{2,}", ".", raw_answer).strip()

    # Format bare digits/numbers into natural sentences for counting queries
    is_counting_query = bool(re.search(r"\b(how many|count|number of)\b", query.lower()))
    if is_counting_query and (clean_answer.isdigit() or re.match(r"^\d+\.?$", clean_answer)):
        num = clean_answer.rstrip(".")
        counting_match = re.search(r"\b(how many|count|number of)\s+([a-zA-Z\s_-]+?)(?:\s+(?:are|is|in|on|visible)|\?|$)", query.lower())
        entity = counting_match.group(2).strip() if counting_match else "features"
        clean_answer = f"There {'is' if num == '1' else 'are'} approximately {num} {entity} visible in the image."

    model_name = _safe_str(result.get("model_name", "satquery-vlm-person-a"))
    conf = result.get("confidence")

    findings = [f"Model answer: {clean_answer}"]
    if model_name:
        findings.append(f"Model used: {model_name}")
    if conf is not None:
        findings.append(f"Model confidence: {_fmt_pct(conf)}")

    summary = f'In response to your query "{query}", the vision-language model answered: {clean_answer}.'

    tech = (
        f"The visual question answering model ({model_name}) jointly processes "
        "the satellite imagery and text prompt to identify scene features and generate "
        "a natural-language answer."
    )

    limitations = ""
    if conf is not None and _norm_conf(conf) < 0.50:
        limitations = (
            f"The model assigned low confidence ({_fmt_pct(conf)}) to this prediction. "
            "Visual QA responses under low confidence should be interpreted with caution."
        )

    return Interpretation(
        answer=clean_answer,
        summary=summary,
        key_findings=findings,
        technical_interpretation=tech,
        limitations=limitations,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Captioning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _interpret_captioning(
    query: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Interpretation:
    raw_caption = _safe_str(result.get("answer", result.get("caption", "")))
    clean_caption = re.sub(r"\.{2,}", ".", raw_caption).strip()
    model_name = _safe_str(result.get("model_name", "satquery-vlm-person-a"))
    conf = result.get("confidence")

    summary = f"Scene description: {clean_caption}."

    findings = [f"Generated caption: {clean_caption}"]
    if model_name:
        findings.append(f"Model used: {model_name}")
    if conf is not None:
        findings.append(f"Model confidence: {_fmt_pct(conf)}")

    tech = (
        f"The caption was generated by {model_name}, analyzing visible land cover "
        "features across the satellite image and producing a natural-language description."
    )

    return Interpretation(
        answer=clean_caption,
        summary=summary,
        key_findings=findings,
        technical_interpretation=tech,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Grounding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _interpret_grounding(
    query: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Interpretation:
    raw_answer = _safe_str(result.get("answer", ""))
    clean_answer = re.sub(r"\.{2,}", ".", raw_answer).strip()
    conf = result.get("confidence")
    norm_query = _safe_str(result.get("normalized_query", query))
    model_name = _safe_str(result.get("model_name", "satquery-region-grounding-v1"))

    vis_ev = evidence or result.get("visual_evidence", {})
    if hasattr(vis_ev, "model_dump"):
        vis_ev = vis_ev.model_dump()
    coords = vis_ev.get("coordinates") if isinstance(vis_ev, dict) else None
    ev_type = vis_ev.get("type", "none") if isinstance(vis_ev, dict) else "none"
    method = vis_ev.get("classification_method") if isinstance(vis_ev, dict) else None
    target_class = vis_ev.get("target_class", norm_query) if isinstance(vis_ev, dict) else norm_query
    coverage = vis_ev.get("target_coverage_pct") if isinstance(vis_ev, dict) else None

    # ── MASK REGION GROUNDING (SCL OR RGB HEURISTIC) ──────────────────────────
    if ev_type == "mask" or method:
        findings = [f"Model used: {model_name}", f"Query target: {target_class}"]
        if method == "spectral_scl":
            findings.append("Classification method: Sentinel-2 SCL Spectral Classification (Validated)")
        else:
            findings.append("Classification method: Visible-spectrum RGB texture & spectral heuristic (Estimate)")

        if coverage is not None:
            findings.append(f"Target coverage: {coverage}% of scene")
        if conf is not None:
            findings.append(f"Confidence score: {_fmt_pct(conf)}")

        if method == "spectral_scl":
            summary = (
                f"Identified and highlighted {target_class} regions across {coverage if coverage is not None else ''}% "
                f"of the satellite scene using Sentinel-2 Scene Classification Layer (SCL)."
            )
            explanation = (
                "Spectral classification extracted ESA Scene Classification Layer (SCL) surface categories directly "
                "from multi-spectral Sentinel-2 imagery, generating a pixel-level region highlight mask for validated land-cover identification."
            )
            limitations = ""
        else:
            summary = (
                f"Identified and highlighted estimated {target_class} regions across ~{coverage if coverage is not None else ''}% "
                f"of the satellite scene using visible-spectrum spectral and texture heuristics (SCL unavailable)."
            )
            explanation = (
                "The uploaded imagery lacks multi-spectral SCL band data. Grounding applied visible-spectrum RGB spectral and texture gradient heuristics to segment likely target regions."
            )
            limitations = (
                "Multi-spectral SCL band is unavailable in this image format. The highlighted regions represent a visible-spectrum RGB heuristic (lower precision than Sentinel-2 SCL)."
            )

        return Interpretation(
            answer=clean_answer or summary,
            summary=summary,
            key_findings=findings,
            technical_interpretation=explanation,
            limitations=limitations,
        )

    # ── BBOX MODEL GROUNDING ──────────────────────────────────────────────────
    findings = [f"Model used: {model_name}", f"Query target: {norm_query}"]

    if conf is not None:
        findings.append(f"Localization confidence: {_fmt_pct(conf)}")

    if coords and ev_type == "bbox":
        findings.append(f"Bounding box (normalized [x1, y1, x2, y2]): {coords}")
        coord_str = f" The predicted target region is located at normalized bounding box coordinates {coords}."
    else:
        coord_str = " No spatial bounding box was localized."

    norm_c = _norm_conf(conf) if conf is not None else 1.0

    if norm_c < 0.50:
        summary = (
            f"The grounding model attempted to locate '{norm_query}', but assigned low confidence ({_fmt_pct(conf)}) to the prediction.{coord_str}"
        )
        explanation = (
            f"Spatial grounding evaluated the image for '{norm_query}' using {model_name}. "
            f"Because the localization confidence is only {_fmt_pct(conf)}, the spatial bounding box should be treated as unreliable."
        )
        limitations = (
            f"The grounding model assigned only {_fmt_pct(conf)} confidence to the predicted location, "
            "so the spatial localization should be treated as unreliable and cannot confidently verify target presence."
        )
    else:
        summary = (
            f"The grounding model localized the spatial region for '{norm_query}' with {_fmt_pct(conf)} confidence.{coord_str}"
        )
        explanation = (
            f"Spatial grounding uses {model_name} to predict the normalized bounding box coordinates "
            f"of '{norm_query}' within the satellite image."
        )
        limitations = ""

    return Interpretation(
        answer=clean_answer or f"Spatial grounding result for: {norm_query}",
        summary=summary,
        key_findings=findings,
        technical_interpretation=explanation,
        limitations=limitations,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ChangeFormer / Change VQA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _interpret_change(
    query: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Interpretation:
    raw_answer = _safe_str(result.get("answer", ""))
    clean_answer = re.sub(r"\.{2,}", ".", raw_answer).strip().rstrip(".")
    conf = result.get("confidence")
    model_name = _safe_str(result.get("model_name", "ChangeFormerV6"))

    vis_ev = evidence or result.get("visual_evidence", {})
    ev_type = vis_ev.get("type", "none") if isinstance(vis_ev, dict) else "none"
    changed_pixels = vis_ev.get("changed_pixels") if isinstance(vis_ev, dict) else None
    total_pixels = vis_ev.get("total_pixels") if isinstance(vis_ev, dict) else None

    # Derive change_ratio mathematically if changed_pixels & total_pixels are present
    change_ratio = result.get("change_ratio", result.get("global_change_ratio"))
    if change_ratio is None and changed_pixels is not None and total_pixels and total_pixels > 0:
        change_ratio = changed_pixels / total_pixels

    params = result.get("parameters", {})
    mean_change_prob = params.get("mean_change_probability") if isinstance(params, dict) else None

    findings = []
    if model_name:
        findings.append(f"Model used: {model_name}")
    if clean_answer:
        findings.append(f"Model classification: {clean_answer}")
    if conf is not None:
        findings.append(f"Model confidence: {_fmt_pct(conf)}")

    if change_ratio is not None:
        ratio_pct = change_ratio * 100 if change_ratio <= 1.0 else change_ratio
        if changed_pixels is not None and total_pixels is not None:
            findings.append(f"Changed area: {ratio_pct:.2f}% ({changed_pixels:,} of {total_pixels:,} pixels)")
        else:
            findings.append(f"Change ratio: {ratio_pct:.2f}% ({change_ratio:.4f})")

    if mean_change_prob is not None:
        findings.append(f"Mean change probability: {_fmt_pct(mean_change_prob)}")

    has_spatial = ev_type in ("change_map", "change_mask", "mask")
    if has_spatial:
        findings.append("Spatial evidence: Change mask generated")

    ratio_str = f"across approximately {change_ratio * 100 if change_ratio <= 1.0 else change_ratio:.2f}% of the image area" if change_ratio is not None else ""
    conf_str = f" The model assigned {_fmt_pct(conf)} confidence to this prediction." if conf is not None else ""
    summary = f"ChangeFormer analyzed bi-temporal imagery and identified land-cover change {ratio_str}.{conf_str}".strip()

    explanation_parts = [
        f"The bi-temporal change detection model ({model_name}) analyzed the image pair for pixel-level land-cover modifications."
    ]
    if changed_pixels is not None and total_pixels is not None and change_ratio is not None:
        ratio_pct = change_ratio * 100 if change_ratio <= 1.0 else change_ratio
        explanation_parts.append(
            f"Specifically, {changed_pixels:,} out of {total_pixels:,} pixels ({ratio_pct:.2f}%) were classified as changed."
        )
    if mean_change_prob is not None:
        explanation_parts.append(
            f"The model detected a mean change probability of about {_fmt_pct(mean_change_prob)}, indicating evidence of pixel-level change."
        )

    # Query context check for specific target questions
    q_lower = query.lower()
    if any(k in q_lower for k in ("building", "urban", "construction", "structure")):
        explanation_parts.append(
            "Note that ChangeFormer measures pixel-level surface changes between dates, but cannot single-handedly confirm specific building construction without spatial grounding or cross-modal verification."
        )
    elif any(k in q_lower for k in ("vegetation", "forest", "tree", "crop")):
        explanation_parts.append(
            "Note that ChangeFormer measures surface change between dates, but specific vegetation species or crop types require spectral index or VLM analysis."
        )

    explanation = " ".join(explanation_parts)

    limitations = ""
    if conf is not None and _norm_conf(conf) < 0.50:
        limitations = f"The change detection model assigned low confidence ({_fmt_pct(conf)}). Pixel-level changes should be interpreted with caution."

    return Interpretation(
        answer=clean_answer or summary,
        summary=summary,
        key_findings=findings,
        technical_interpretation=explanation,
        limitations=limitations,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Optical-SAR Fusion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _interpret_fusion(
    query: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Interpretation:
    raw_answer = _safe_str(result.get("answer", ""))
    clean_answer = re.sub(r"\.{2,}", ".", raw_answer).strip().rstrip(".")
    conf = result.get("confidence")
    sub_type = _safe_str(result.get("task_sub_type", ""))
    probs = result.get("probabilities", {})
    model_name = _safe_str(result.get("model_name", "satquery-optical-sar-fusion-v1"))

    vis_ev = evidence or result.get("visual_evidence", {})
    ev_type = vis_ev.get("type", "none") if isinstance(vis_ev, dict) else "none"
    coords = vis_ev.get("coordinates") if isinstance(vis_ev, dict) else None

    findings = [
        f"Model used: {model_name}",
        f"Analysis type: Optical-SAR cross-modal fusion",
    ]
    if sub_type:
        findings.append(f"Task subtype: {sub_type}")
    if clean_answer:
        findings.append(f"Model prediction: {clean_answer}")
    if conf is not None:
        findings.append(f"Model confidence: {_fmt_pct(conf)}")

    close_probs = False
    if probs and isinstance(probs, dict):
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_val = sorted_probs[0]
        prob_strs = [f"{cat}: {_fmt_pct(val)}" for cat, val in sorted_probs]
        findings.append(f"Probability distribution: {', '.join(prob_strs)}")

        if len(sorted_probs) > 1:
            diff = top_val - sorted_probs[1][1]
            if diff < 0.10 or top_val < 0.35:
                close_probs = True

    if ev_type == "bbox" and coords:
        findings.append(f"Bounding box coordinates: {coords}")

    norm_c = _norm_conf(conf) if conf is not None else 0.85
    q_lower = query.lower()
    is_modal_comparison = any(k in q_lower for k in ("compare optical and sar", "how the two modalities", "represent the scene differently", "modalities represent", "differently"))

    if is_modal_comparison:
        summary = (
            f"Optical imagery captures spectral reflectance (color, vegetation), while SAR captures microwave backscatter "
            f"(surface roughness, physical structure, moisture). Fusing both modalities, the model classified the scene feature as '{clean_answer}' "
            f"with {_fmt_pct(conf)} model confidence."
        )
        explanation = (
            f"Optical imagery represents reflected solar electromagnetic radiation (visible, NIR, SWIR spectral bands), "
            f"providing sensitivity to surface color, spectral reflectance, and vegetation/chlorophyll absorption. "
            f"In contrast, SAR imagery measures active microwave radar backscatter, sensitive to surface roughness, "
            f"physical geometry, dielectric moisture, and structural layout. The fusion model ({model_name}) jointly evaluated "
            f"both complementary modalities, predicting '{clean_answer}' with {_fmt_pct(conf)} model confidence."
        )
        limitations = ""
    elif close_probs:
        summary = (
            f"Optical-SAR fusion prediction ({clean_answer}) exhibits high uncertainty across categories "
            f"(highest probability: {_fmt_pct(conf)})."
        )
        explanation = (
            f"The fusion model ({model_name}) jointly analyzed optical and SAR imagery. "
            f"While category '{clean_answer}' received the highest probability ({_fmt_pct(conf)}), "
            "the probabilities are closely distributed across candidate categories, indicating high classification uncertainty."
        )
        limitations = (
            f"The classification probabilities are closely split across options (top confidence: {_fmt_pct(conf)}), "
            "so the prediction should be treated as uncertain."
        )
    elif norm_c < 0.50:
        summary = (
            f"The fusion model prediction for '{clean_answer}' has low confidence ({_fmt_pct(conf)}) and should be treated as uncertain."
        )
        explanation = (
            f"Cross-modal fusion evaluated optical and SAR features using {model_name}. "
            f"The model assigned {_fmt_pct(conf)} confidence to prediction '{clean_answer}'."
        )
        limitations = f"The model assigned low confidence ({_fmt_pct(conf)}) to this cross-modal prediction."
    else:
        summary = (
            f"Optical-SAR fusion model classified the scene feature as {clean_answer} with {_fmt_pct(conf)} confidence."
        )
        explanation = (
            f"The cross-modal fusion model ({model_name}) jointly processed Sentinel-2 optical and Sentinel-1 SAR imagery "
            f"to evaluate the cross-modal features, predicting '{clean_answer}' with {_fmt_pct(conf)} model confidence."
        )
        limitations = ""

    return Interpretation(
        answer=clean_answer,
        summary=summary,
        key_findings=findings,
        technical_interpretation=explanation,
        limitations=limitations,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEE — Generic Router
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _interpret_gee(
    query: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Interpretation:
    """Route GEE results to the appropriate sub-interpreter."""
    intent = _safe_str(result.get("intent", result.get("gee_intent", "")))
    inner = result.get("result", result)

    sub_dispatch = {
        "multi_metric": _interpret_gee_multi_metric,
        "elevation": _interpret_gee_elevation,
        "ndvi": _interpret_gee_ndvi,
        "ndwi": _interpret_gee_ndwi,
        "ndbi": _interpret_gee_ndbi,
        "sentinel1": _interpret_gee_imagery,
        "sentinel2": _interpret_gee_imagery,
        "landsat": _interpret_gee_imagery,
    }

    handler = sub_dispatch.get(intent, _interpret_gee_imagery)
    data = inner if isinstance(inner, dict) else result
    return handler(query, data, evidence, metadata)


def _interpret_gee_multi_metric(
    query: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Interpretation:
    findings = []

    # 1. Date & Cloud
    date = result.get("date")
    cloud = result.get("cloud_percentage")
    if date:
        findings.append(f"Satellite acquisition date: {date}")
    else:
        findings.append("Satellite acquisition date: Unavailable for specified parameters")

    if cloud is not None:
        findings.append(f"Cloud cover: {cloud:.1f}%")
    else:
        findings.append("Cloud cover: Unavailable")

    # 2. NDVI
    ndvi_val = result.get("ndvi")
    if ndvi_val is not None:
        if ndvi_val > 0.6:
            meaning = "indicates relatively strong vegetation characteristics and provides supporting geospatial context"
        elif ndvi_val > 0.2:
            meaning = "indicates moderate vegetation cover and provides supporting geospatial context"
        elif ndvi_val > 0:
            meaning = "indicates sparse vegetation or bare soil and provides supporting geospatial context"
        else:
            meaning = "indicates water, built-up surfaces, or barren land and provides supporting geospatial context"
        findings.append(f"NDVI: {ndvi_val:.4f} ({meaning})")
    else:
        findings.append("NDVI: Measurement unavailable")

    # 3. NDWI
    ndwi_val = result.get("ndwi")
    if ndwi_val is not None:
        if ndwi_val > 0.3:
            meaning = "consistent with open water features and providing supporting geospatial context"
        elif ndwi_val > 0:
            meaning = "suggesting potential moisture or wet surfaces as supporting geospatial context"
        else:
            meaning = "indicating low open-water presence and providing supporting geospatial context"
        findings.append(f"NDWI: {ndwi_val:.4f} ({meaning})")
    else:
        findings.append("NDWI: Measurement unavailable")

    # 4. NDBI
    ndbi_val = result.get("ndbi")
    if ndbi_val is not None:
        if ndbi_val > 0.1:
            meaning = "providing supporting evidence consistent with built-up or impervious surface characteristics"
        elif ndbi_val > -0.1:
            meaning = "providing supporting evidence consistent with mixed land cover"
        else:
            meaning = "providing supporting evidence consistent with predominantly vegetated or natural surfaces"
        findings.append(f"NDBI: {ndbi_val:.4f} ({meaning})")
    else:
        findings.append("NDBI: Measurement unavailable")

    # 5. Sentinel-1 SAR
    s1_info = result.get("sentinel1")
    if isinstance(s1_info, dict) and s1_info.get("date"):
        findings.append(
            f"Sentinel-1 SAR: Image ID {s1_info.get('image_id', 'unknown')}, acquired {s1_info.get('date')} "
            f"(polarization: {s1_info.get('polarization', 'VV')}, mode: {s1_info.get('instrument_mode', 'IW')})"
        )
    elif s1_info:
        findings.append("Sentinel-1 SAR: Information retrieved")
    else:
        findings.append("Sentinel-1 SAR information: Unavailable for specified parameters")

    summary = (
        "Multimodal Earth Engine geospatial measurements were retrieved for the target location. "
        "The retrieved indices and sensor parameters supply supporting geospatial context."
    )
    tech = (
        "Multi-metric GEE analysis samples Sentinel-2 optical bands (B4, B3, B8, B11) for NDVI, NDWI, and NDBI "
        "spectral index calculation alongside Sentinel-1 SAR backscatter metadata as supporting geospatial evidence."
    )

    return Interpretation(
        answer=summary,
        summary=summary,
        key_findings=findings,
        technical_interpretation=tech,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEE — Elevation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _interpret_gee_elevation(
    query: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Interpretation:
    elev = result.get("elevation") or result.get("elevation_m")
    elev_mean = result.get("elevation_mean")
    elev_min = result.get("elevation_min")
    elev_max = result.get("elevation_max")
    buffer = result.get("buffer_m", 0)

    findings = []

    if buffer and buffer > 0 and elev_mean is not None:
        findings.append(f"Mean elevation: {elev_mean:.2f} m")
        if elev_min is not None:
            findings.append(f"Minimum elevation: {elev_min:.2f} m")
        if elev_max is not None:
            findings.append(f"Maximum elevation: {elev_max:.2f} m")
        findings.append(f"Buffer radius: {buffer:.0f} m")
        summary = (
            f"Within a {buffer:.0f}-meter radius, terrain elevation "
            f"ranges from {elev_min:.2f} m to {elev_max:.2f} m "
            f"(mean: {elev_mean:.2f} m above sea level)."
        )
        answer_text = summary
    elif elev is not None:
        findings.append(f"Point elevation: {elev:.2f} m above sea level")
        summary = (
            f"The elevation at the requested location is {elev:.2f} meters "
            f"above sea level, based on SRTM digital elevation data."
        )
        answer_text = summary
    else:
        summary = "Elevation data was retrieved but could not be parsed."
        answer_text = summary

    findings.append("Data source: SRTM (USGS/SRTMGL1_003)")

    tech = (
        "Elevation is retrieved from the Shuttle Radar Topography Mission "
        "(SRTM) dataset, which provides global elevation data at approximately "
        "30-meter resolution as supporting geospatial evidence."
    )

    return Interpretation(
        answer=answer_text,
        summary=summary,
        key_findings=findings,
        technical_interpretation=tech,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEE — NDVI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _interpret_gee_ndvi(
    query: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Interpretation:
    val = result.get("index_value")
    date = result.get("date")
    dataset = result.get("dataset", "Sentinel-2")

    findings = []
    if val is not None:
        findings.append(f"NDVI value: {val:.4f}")
    if date:
        findings.append(f"Observation date: {date}")
    findings.append(f"Data source: {dataset}")

    if val is not None:
        if val > 0.6:
            meaning = "indicates relatively strong vegetation characteristics and provides supporting geospatial context"
        elif val > 0.2:
            meaning = "indicates moderate vegetation cover and provides supporting geospatial context"
        elif val > 0:
            meaning = "indicates sparse vegetation or bare soil and provides supporting geospatial context"
        else:
            meaning = "indicates water, built-up surfaces, or barren land and provides supporting geospatial context"

        summary = (
            f"The Normalized Difference Vegetation Index (NDVI) at this "
            f"location is {val:.4f}, which {meaning}."
        )
        if date:
            summary += f" Based on Sentinel-2 imagery from {date}."
    else:
        summary = "NDVI data was retrieved but the value could not be parsed."

    tech = (
        "NDVI is a spectral index calculated as (NIR − Red) / (NIR + Red) "
        "using Sentinel-2 bands B8 (NIR) and B4 (Red). Higher positive values "
        "are consistent with denser vegetation cover as supporting geospatial evidence."
    )

    return Interpretation(
        answer=summary,
        summary=summary,
        key_findings=findings,
        technical_interpretation=tech,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEE — NDWI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _interpret_gee_ndwi(
    query: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Interpretation:
    val = result.get("index_value")
    date = result.get("date")
    dataset = result.get("dataset", "Sentinel-2")

    findings = []
    if val is not None:
        findings.append(f"NDWI value: {val:.4f}")
    if date:
        findings.append(f"Observation date: {date}")
    findings.append(f"Data source: {dataset}")

    if val is not None:
        if val > 0.3:
            meaning = "consistent with open water features"
        elif val > 0:
            meaning = "suggesting potential moisture or wet surfaces"
        else:
            meaning = "indicating low open-water presence"

        summary = (
            f"The Normalized Difference Water Index (NDWI) at this location "
            f"is {val:.4f}, {meaning}."
        )
        if date:
            summary += f" Based on imagery from {date}."
    else:
        summary = "NDWI data was retrieved but the value could not be parsed."

    tech = (
        "NDWI is a spectral index calculated as (Green − NIR) / (Green + NIR) "
        "using Sentinel-2 bands B3 (Green) and B8 (NIR). Positive values are "
        "associated with water or moisture features as supporting geospatial evidence."
    )

    return Interpretation(
        answer=summary,
        summary=summary,
        key_findings=findings,
        technical_interpretation=tech,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEE — NDBI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _interpret_gee_ndbi(
    query: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Interpretation:
    val = result.get("index_value")
    date = result.get("date")
    dataset = result.get("dataset", "Sentinel-2")

    findings = []
    if val is not None:
        findings.append(f"NDBI value: {val:.4f}")
    if date:
        findings.append(f"Observation date: {date}")
    findings.append(f"Data source: {dataset}")

    if val is not None:
        if val > 0.1:
            meaning = (
                "providing supporting evidence consistent with built-up "
                "or impervious surface characteristics"
            )
        elif val > -0.1:
            meaning = "suggesting mixed land cover with some impervious surfaces"
        else:
            meaning = "suggesting predominantly vegetated or natural surfaces"

        summary = (
            f"The Normalized Difference Built-up Index (NDBI) at this "
            f"location is {val:.4f}, {meaning}."
        )
        if date:
            summary += f" Based on imagery from {date}."
    else:
        summary = "NDBI data was retrieved but the value could not be parsed."

    tech = (
        "NDBI is a spectral index calculated as (SWIR − NIR) / (SWIR + NIR) "
        "using Sentinel-2 bands B11 (SWIR) and B8 (NIR). Higher positive values "
        "provide supporting geospatial evidence consistent with built-up surfaces."
    )

    return Interpretation(
        answer=summary,
        summary=summary,
        key_findings=findings,
        technical_interpretation=tech,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEE — Satellite Imagery Retrieval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _interpret_gee_imagery(
    query: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Interpretation:
    image_id = result.get("image_id", "unknown")
    date = result.get("date")
    cloud = result.get("cloud_percentage")
    dataset = result.get("dataset", "")
    polarization = result.get("polarization")
    orbit = result.get("orbit_pass")
    instrument_mode = result.get("instrument_mode")

    if "S1" in dataset or "S1" in str(image_id):
        sensor = "Sentinel-1 SAR"
    elif "S2" in dataset or "S2" in str(image_id):
        sensor = "Sentinel-2 optical"
    elif "LANDSAT" in dataset.upper() or "LC09" in str(image_id):
        sensor = "Landsat 9"
    else:
        sensor = "satellite"

    findings = [f"Image ID: {image_id}"]
    if date:
        findings.append(f"Acquisition date: {date}")
    if cloud is not None:
        findings.append(f"Cloud cover: {cloud:.1f}%")
    if polarization:
        findings.append(f"Polarization: {polarization}")
    if orbit:
        findings.append(f"Orbit pass: {orbit}")
    if instrument_mode:
        findings.append(f"Instrument mode: {instrument_mode}")
    if dataset:
        findings.append(f"Dataset: {dataset}")

    parts = [f"Retrieved {sensor} imagery"]
    if date:
        parts.append(f"captured on {date}")
    if cloud is not None:
        parts.append(f"with {cloud:.1f}% cloud cover")
    summary = " ".join(parts) + "."

    tech_parts = [f"The image was selected from the {dataset or sensor} collection as supporting geospatial data"]
    if polarization:
        tech_parts.append(f"Polarization filter ({polarization}) was applied.")
    if orbit:
        tech_parts.append(f"Orbit pass filter ({orbit}) was applied.")
    tech = ". ".join(tech_parts) + "."

    return Interpretation(
        answer=summary,
        summary=summary,
        key_findings=findings,
        technical_interpretation=tech,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generic Fallback
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _interpret_generic(
    query: str,
    result: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Interpretation:
    raw_answer = _safe_str(result.get("answer", ""))
    clean_answer = re.sub(r"\.{2,}", ".", raw_answer).strip()
    model_name = _safe_str(result.get("model_name", ""))

    summary = f"Analysis result: {clean_answer}." if clean_answer else "Analysis completed."
    findings = []
    if clean_answer:
        findings.append(f"Result: {clean_answer}")
    if model_name:
        findings.append(f"Model: {model_name}")

    return Interpretation(
        answer=clean_answer or "Analysis completed.",
        summary=summary,
        key_findings=findings,
        technical_interpretation="The result was produced by the SatQuery AI analysis pipeline.",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _safe_str(val: Any) -> str:
    """Safely convert a value to string, returning '' for None."""
    if val is None:
        return ""
    return str(val).strip()


def _norm_conf(c: float) -> float:
    """Normalize a confidence value to 0–1."""
    return c if c <= 1.0 else c / 100.0


def _fmt_pct(val: float) -> str:
    """Format a value as percentage string."""
    v = val * 100 if val <= 1.0 else val
    return f"{v:.2f}%"


def decompose_query(original_query: str, tasks_list: List[str]) -> Dict[str, str]:
    """Decompose a complex original multi-intent query into task-focused sub-queries."""
    query_lower = original_query.lower()
    decomposed: Dict[str, str] = {}

    for task in tasks_list:
        norm_task = task.lower().strip()

        if norm_task in ("change_vqa", "change"):
            if "change" in query_lower or "difference" in query_lower or "between" in query_lower:
                decomposed[task] = "What major land-cover changes occurred between the before and after satellite images?"
            else:
                decomposed[task] = "Analyze bi-temporal land cover changes between the images."

        elif norm_task == "grounding":
            if any(k in query_lower for k in ("built-up", "building", "urban", "structure")):
                decomposed[task] = "Locate the newly developed built-up areas."
            elif any(k in query_lower for k in ("forest", "tree", "vegetation")):
                decomposed[task] = "Locate the target vegetation and forest regions."
            elif any(k in query_lower for k in ("water", "river", "lake")):
                decomposed[task] = "Locate the target water body regions."
            else:
                decomposed[task] = "Locate the primary changed or target spatial regions."

        elif norm_task == "fusion":
            if any(k in query_lower for k in ("built-up", "building", "urban")):
                decomposed[task] = "Verify whether the identified changed regions represent built-up development using both optical and SAR imagery."
            elif any(k in query_lower for k in ("water", "flood")):
                decomposed[task] = "Verify whether the identified regions represent water bodies using both optical and SAR imagery."
            else:
                decomposed[task] = "Confirm land cover features and targets using optical and SAR cross-modal imagery."

        elif norm_task == "captioning":
            decomposed[task] = "Describe the scene and visible land cover features in the satellite image."

        elif norm_task == "vqa":
            decomposed[task] = original_query

        else:
            decomposed[task] = original_query

    return decomposed


def synthesize_multi_model_results(
    original_query: str,
    sub_task_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Synthesize outputs from multiple specialist model inferences into ONE coherent ChatGPT-style conclusion."""
    confidence_by_task: Dict[str, float] = {}
    findings: List[str] = []
    uncertainties: List[str] = []
    visual_evidence_map: Dict[str, Any] = {}
    task_results_map: Dict[str, Dict[str, Any]] = {}

    for res in sub_task_results:
        task = res["task"]
        ans = res.get("answer", "")
        conf = res.get("confidence")

        task_results_map[task] = res

        if conf is not None:
            confidence_by_task[task] = round(float(conf), 4)

        vis = res.get("visual_evidence")
        if isinstance(vis, dict) and vis.get("type") != "none":
            visual_evidence_map[task] = vis
        elif hasattr(vis, "model_dump"):
            dumped = vis.model_dump()
            if dumped.get("type") != "none":
                visual_evidence_map[task] = dumped

    # 1. Analyze Change detection finding
    change_res = task_results_map.get("change_vqa") or task_results_map.get("change")
    change_conf = change_res.get("confidence") if change_res else None
    if change_res:
        change_ans = change_res.get("answer", "")
        c_vis = change_res.get("visual_evidence") or {}
        ch_pixels = c_vis.get("changed_pixels") if isinstance(c_vis, dict) else None
        tot_pixels = c_vis.get("total_pixels") if isinstance(c_vis, dict) else None
        c_ratio = change_res.get("change_ratio", change_res.get("global_change_ratio"))
        change_ratio_str = None

        if c_ratio is not None:
            r_pct = c_ratio * 100 if c_ratio <= 1.0 else c_ratio
            change_ratio_str = f"{r_pct:.2f}%"
        else:
            match_ratio = re.search(r"(\d+\.?\d*)\s*%", change_ans)
            if match_ratio:
                change_ratio_str = f"{match_ratio.group(1)}%"

        change_detected = "change detected" in change_ans.lower() or "yes" in change_ans.lower() or bool(change_ratio_str)
        px_str = f" ({ch_pixels:,} of {tot_pixels:,} pixels)" if ch_pixels is not None and tot_pixels is not None else ""

        if change_conf is not None and change_conf >= 0.5:
            ratio_phrase = f"across approximately {change_ratio_str}{px_str}" if change_ratio_str else ""
            findings.append(
                f"[ChangeFormer] Change detection: ChangeFormer detected land-cover modifications {ratio_phrase} with {_fmt_pct(change_conf)} model confidence."
            )
        elif change_conf is not None:
            findings.append(
                f"[ChangeFormer] Change detection: Identified potential changes with low confidence ({_fmt_pct(change_conf)})."
            )
            uncertainties.append(f"Bi-temporal change detection confidence is low ({_fmt_pct(change_conf)}).")

    # 2. Analyze Grounding finding
    grounding_res = task_results_map.get("grounding")
    grounding_conf = None
    if grounding_res:
        grounding_conf = grounding_res.get("confidence")
        vis = grounding_res.get("visual_evidence")
        coords = vis.get("coordinates") if isinstance(vis, dict) else None
        is_default_bbox = (coords == [0, 0, 1, 1] or coords == [0.0, 0.0, 1.0, 1.0])

        if grounding_conf is not None and grounding_conf >= 0.50 and not is_default_bbox:
            coord_str = f" at normalized coordinates {coords}" if coords else ""
            findings.append(
                f"[Grounding] Spatial localization: Region grounding localized target spatial features{coord_str} with {_fmt_pct(grounding_conf)} model confidence."
            )
        else:
            conf_label = _fmt_pct(grounding_conf) if grounding_conf is not None else "low"
            findings.append(
                f"[Grounding] Spatial localization: Region grounding localization assigned low confidence ({conf_label}) to spatial coordinates; spatial localization is treated as low-confidence evidence."
            )
            uncertainties.append(
                f"Spatial localization of target regions has low grounding confidence ({conf_label})."
            )

    # 3. Analyze Fusion finding
    fusion_res = task_results_map.get("fusion")
    fusion_conf = None
    fusion_ans = None
    fusion_pred_upper = ""
    if fusion_res:
        fusion_conf = fusion_res.get("confidence")
        fusion_ans = str(fusion_res.get("answer", ""))
        fusion_pred_upper = fusion_ans.upper()

        if fusion_conf is not None and fusion_conf >= 0.50:
            findings.append(
                f"[Fusion] Optical-SAR evidence: Cross-modal fusion strongly classified the scene feature as '{fusion_ans}' with {_fmt_pct(fusion_conf)} model confidence."
            )
        elif fusion_conf is not None:
            findings.append(
                f"[Fusion] Optical-SAR evidence: Optical-SAR cross-modal verification returned low confidence ({_fmt_pct(fusion_conf)})."
            )
            uncertainties.append(f"Optical-SAR cross-modal verification has low confidence ({_fmt_pct(fusion_conf)}).")

    # 4. Analyze VQA / Captioning finding
    vqa_res = task_results_map.get("vqa") or task_results_map.get("captioning")
    if vqa_res:
        v_ans = vqa_res.get("answer", "")
        v_conf = vqa_res.get("confidence")
        v_conf_str = f" with {_fmt_pct(v_conf)} model confidence" if v_conf is not None else ""
        findings.append(f"[VLM Image Analysis] Primary visual interpretation: {v_ans}{v_conf_str}.")

    # 5. Analyze GEE supporting evidence (if present)
    gee_res = task_results_map.get("gee") or task_results_map.get("ndvi") or task_results_map.get("ndbi") or task_results_map.get("elevation")
    if gee_res:
        g_ans = gee_res.get("answer", "")
        findings.append(f"[GEE Evidence] {g_ans}")

    # Check Conflict
    is_conflicting = False
    if change_res and fusion_res:
        c_conf = change_conf or 0.0
        f_conf = fusion_conf or 0.0
        if c_conf >= 0.50 and f_conf >= 0.50:
            if fusion_pred_upper in ("AGRICULTURE", "FOREST", "WATER", "NO", "FALSE", "NON_BUILT_UP"):
                query_lower = original_query.lower()
                if any(k in query_lower for k in ("built-up", "building", "urban", "construction", "development")) or change_detected:
                    is_conflicting = True
                    uncertainties.append(
                        f"Conflict detected between ChangeFormer change detection ({_fmt_pct(c_conf)} confidence) and Optical-SAR cross-modal fusion classification ('{fusion_ans}', {_fmt_pct(f_conf)} confidence)."
                    )

    # 6. Determine overall Evidence Quality
    conf_values = list(confidence_by_task.values())
    if is_conflicting:
        evidence_quality = "conflicting"
    elif not conf_values or all(c < 0.20 for c in conf_values):
        evidence_quality = "insufficient"
    elif all(c >= 0.70 for c in conf_values):
        evidence_quality = "strong"
    elif any(c >= 0.70 for c in conf_values) and any(c < 0.30 for c in conf_values):
        evidence_quality = "limited"
    elif all(c >= 0.40 for c in conf_values):
        evidence_quality = "moderate"
    else:
        evidence_quality = "limited"

    # 7. Formulate ONE Coherent ChatGPT-style Synthesized Answer String
    bullets_text = "\n".join(f"- {f}" for f in findings)

    if is_conflicting:
        synthesized_answer = (
            f"Multi-model synthesis evaluated the query across {len(sub_task_results)} specialist analyses and identified conflicting evidence.\n\n"
            f"Key findings:\n{bullets_text}\n\n"
            f"Synthesis:\nBi-temporal change analysis detected land-cover modifications (confidence: {_fmt_pct(change_conf or 0)}), "
            f"but Optical-SAR cross-modal fusion classified the region as '{fusion_ans}' with high confidence ({_fmt_pct(fusion_conf or 0)}). "
            f"Because cross-modal fusion identifies the surface as {fusion_ans.lower()} rather than built-up structures, the available evidence does NOT support newly developed buildings.\n\n"
            f"Overall conclusion:\n"
            f"- Supported: Pixel-level land-cover change between image dates.\n"
            f"- Uncertain: Spatial localization due to weak grounding confidence ({_fmt_pct(grounding_conf or 0)}).\n"
            f"- Not supported: Confirmed target built-up development (cross-modal fusion predicted {fusion_ans.lower()} with {_fmt_pct(fusion_conf or 0)} confidence)."
        )
    elif change_res and grounding_res and fusion_res:
        if (change_conf or 0) >= 0.5 and (grounding_conf or 1) < 0.30 and (fusion_conf or 1) < 0.30:
            ratio_desc = f"across approximately {change_ratio_str}" if change_ratio_str else ""
            synthesized_answer = (
                f"Multi-model synthesis evaluated the query across ChangeFormer, Region Grounding, and Optical-SAR Fusion.\n\n"
                f"Key findings:\n{bullets_text}\n\n"
                f"Synthesis:\nBi-temporal change analysis indicates that land-cover changes occurred {ratio_desc} with "
                f"{_fmt_pct(change_conf or 0)} model confidence. However, both region grounding ({_fmt_pct(grounding_conf or 0)}) "
                f"and Optical-SAR verification ({_fmt_pct(fusion_conf or 0)}) returned very low confidence scores. Therefore, "
                f"the available evidence is insufficient to reliably confirm that the detected changes correspond to newly developed built-up areas.\n\n"
                f"Overall conclusion:\n"
                f"- Supported: Pixel-level land-cover change between dates.\n"
                f"- Uncertain: Spatial localization of specific target regions.\n"
                f"- Unsupported: Definitive confirmation of new built-up construction."
            )
        elif (change_conf or 0) >= 0.70 and (grounding_conf or 0) >= 0.70 and (fusion_conf or 0) >= 0.70:
            ratio_desc = f"across approximately {change_ratio_str}" if change_ratio_str else ""
            synthesized_answer = (
                f"Multi-model synthesis evaluated the query across ChangeFormer, Region Grounding, and Optical-SAR Fusion.\n\n"
                f"Key findings:\n{bullets_text}\n\n"
                f"Synthesis:\nChangeFormer detected land-cover modifications {ratio_desc} with high confidence ({_fmt_pct(change_conf or 0)}). "
                f"Region grounding localized the target spatial region ({_fmt_pct(grounding_conf or 0)} confidence), "
                f"and Optical-SAR cross-modal fusion confirmed the land cover features ({_fmt_pct(fusion_conf or 0)} confidence). "
                f"All specialist models demonstrate strong agreement.\n\n"
                f"Overall conclusion:\n"
                f"- Supported: Pixel-level change, target spatial localization, and cross-modal feature confirmation."
            )
        else:
            synthesized_answer = (
                f"Multi-model evaluation combined outputs across {len(sub_task_results)} specialist tasks.\n\n"
                f"Key findings:\n{bullets_text}\n\n"
                f"Synthesis:\nSpecialist models produced outputs with an overall evidence quality evaluated as '{evidence_quality}'. "
                f"Individual model confidence scores reflect varying levels of certainty across tasks."
            )
    else:
        uncertainty_text = f"\n\nNote: {' '.join(uncertainties)}" if uncertainties else ""
        synthesized_answer = (
            f"Multi-model evaluation combined outputs across {len(sub_task_results)} sub-tasks ({', '.join(task_results_map.keys())}).\n\n"
            f"Key findings:\n{bullets_text}{uncertainty_text}"
        )

    # 8. Build High-Level Conclusion
    if evidence_quality == "conflicting":
        conclusion = "Specialist outputs show material disagreement between optical change detection and SAR cross-modal verification."
    elif evidence_quality == "limited":
        conclusion = (
            "Land-cover change is detected with high confidence, but specific target classification "
            "and spatial localization remain uncertain due to low grounding and fusion confidence scores."
        )
    elif evidence_quality in ("strong", "moderate"):
        conclusion = "Specialist models demonstrate consistent evidence supporting the identified features and spatial regions."
    else:
        conclusion = "Specialist analyses show low confidence or insufficient evidence across the evaluated tasks."

    summary = f"Multi-modal synthesis across {len(sub_task_results)} specialist tasks: {', '.join(task_results_map.keys())}."

    synthesis_block = {
        "summary": summary,
        "findings": findings,
        "conclusion": conclusion,
        "evidence_quality": evidence_quality,
        "uncertainties": uncertainties,
    }

    # 9. Construct Primary Visual Evidence
    primary_visual_evidence = VisualEvidence(type="none")
    g_vis = visual_evidence_map.get("grounding")
    g_coords = g_vis.get("coordinates") if isinstance(g_vis, dict) else None
    is_default_g_coords = g_coords in ([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0])
    g_conf = grounding_conf

    if g_vis and g_coords and not is_default_g_coords and (g_conf is not None and g_conf >= 0.20):
        primary_visual_evidence = VisualEvidence(
            type=g_vis.get("type", "bbox"),
            coordinates=g_coords,
            coordinate_system=g_vis.get("coordinate_system", "normalized"),
            data=g_vis.get("data"),
        )
    elif "change_vqa" in visual_evidence_map or "change" in visual_evidence_map:
        c_vis = visual_evidence_map.get("change_vqa") or visual_evidence_map.get("change")
        primary_visual_evidence = VisualEvidence(
            type=c_vis.get("type", "change_map"),
            coordinates=c_vis.get("coordinates"),
            coordinate_system=c_vis.get("coordinate_system", "normalized"),
            data=c_vis.get("data"),
        )

    return {
        "synthesized_answer": synthesized_answer,
        "confidence_by_task": confidence_by_task,
        "synthesis": synthesis_block,
        "primary_visual_evidence": primary_visual_evidence,
        "visual_evidence_map": visual_evidence_map,
    }
