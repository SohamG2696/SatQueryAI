"""
SatQuery AI — Query Validation & Normalization Layer.

Validates user natural-language queries against supported remote-sensing tasks,
rejects out-of-domain requests, and normalizes semantically equivalent queries
into canonical task representations and fixed specialist model prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Canonical Prompts & Templates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CANONICAL_SCENE_DESCRIPTION_PROMPT = (
    "Describe this satellite image in detail, including landscape features, "
    "land cover, terrain, infrastructure, and geographical elements."
)

CANONICAL_GROUNDING_TEMPLATE = "Locate and segment the {target} in this satellite image."

CANONICAL_CHANGE_PROMPT = (
    "Analyze and identify the physical and environmental changes between the "
    "pre-event and post-event satellite images."
)

CANONICAL_FUSION_PROMPT = (
    "Analyze the complementary optical and SAR satellite imagery jointly to "
    "characterize surface features and verify structures."
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Validation Result Dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ValidationResult:
    """Structured result produced by the Query Validation & Normalization Layer."""

    valid: bool
    intent: str  # 'scene_description', 'vqa', 'grounding', 'change_analysis', 'optical_sar_analysis', 'ndvi_analysis', 'area_analysis', 'temporal_analysis', 'invalid'
    canonical_task: str  # Standard task identifier ('scene_description', 'vqa', 'grounding', 'change_vqa', 'fusion', 'gee', 'invalid')
    canonical_prompt: str
    target: Optional[str] = None
    operation: Optional[str] = None
    confidence: float = 1.0
    reason: Optional[str] = None
    extracted_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary representation."""
        data = {
            "valid": self.valid,
            "intent": self.intent,
            "canonical_task": self.canonical_task,
            "canonical_prompt": self.canonical_prompt,
            "confidence": self.confidence,
        }
        if self.target is not None:
            data["target"] = self.target
        if self.operation is not None:
            data["operation"] = self.operation
        if self.reason is not None:
            data["reason"] = self.reason
        if self.extracted_params:
            data["extracted_params"] = self.extracted_params
        return data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Keyword Dictionaries & Patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Explicit Rejection / Out-of-Domain Keywords & Patterns
_INVALID_PATTERNS: List[re.Pattern] = [
    # Programming & Code
    re.compile(r"\b(write|generate|debug|fix|create)\s+(python|javascript|java|c\+\+|html|css|code|script|sql|regex)\b", re.I),
    re.compile(r"\b(write\s+code|print\(|def\s+\w+\(|function\s+\w+\(|class\s+\w+:)\b", re.I),
    # General Trivia & World Facts
    re.compile(r"\bcapital\s+of\s+(france|germany|italy|spain|india|japan|china|usa|canada|brazil|australia|russia|uk|egypt)\b", re.I),
    re.compile(r"\bwho\s+(won|scored|played)\s+(the\s+)?(football|cricket|fifa|world\s+cup|super\s+bowl|match|game|nba|tennis|olympics)\b", re.I),
    re.compile(r"\bwho\s+(is|was)\s+(albert\s+einstein|newton|shakespeare|napoleon|gandhi|lincoln|elon\s+musk|bill\s+gates|the\s+president|the\s+prime\s+minister|ceo\s+of)\b", re.I),
    re.compile(r"\bwhat\s+is\s+the\s+(population|currency|anthem|flag)\s+of\b", re.I),
    # General Math / Science unrelated to remote sensing
    re.compile(r"\b(solve|calculate|what\s+is)\s+\d+\s*[\+\-\*\/x\^]\s*\d+\b", re.I),
    re.compile(r"\b(solve\s+the\s+equation|quadratic\s+equation|quantum\s+physics|quantum\s+computing|speed\s+of\s+light)\b", re.I),
    # Creative writing, recipes, translation, conversational chit-chat, household how-tos
    re.compile(r"\b(write|compose)\s+(a\s+)?(poem|story|song|essay|joke|riddle|letter)\b", re.I),
    re.compile(r"\b(recipe\s+for|how\s+(do|can|to|should)\s+(i|we|you)?\s*(cook|bake|make\s+pasta|make\s+cake|fix|repair|change\s+a\s+tire|clean))\b", re.I),
    re.compile(r"\b(translate\s+.+\s+to\s+(french|spanish|german|hindi|chinese|japanese|russian|latin))\b", re.I),
    re.compile(r"\b(tell\s+me\s+a\s+joke|how\s+are\s+you\s+doing|what\s+is\s+your\s+favorite\s+(movie|food|color))\b", re.I),
]

# Scene Description Paraphrases & Patterns
_SCENE_DESCRIPTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"^(describe|caption|summarize|overview)\s+(the|this|an)?\s*(satellite\s+|aerial\s+)?(image|scene|photo|photograph|picture)?\.?$", re.I),
    re.compile(r"^what\s+(is\s+(in|there\s+in)|do\s+you\s+see\s+in|can\s+you\s+see(\s+here)?\s*(in)?)\s*(the|this|an)?\s*(satellite\s+|aerial\s+)?(image|scene|photo|picture)?\??$", re.I),
    re.compile(r"^tell\s+me\s+about\s+(the|this)\s*(satellite\s+|aerial\s+)?(image|scene|photo|picture)\.?$", re.I),
    re.compile(r"^(give\s+me\s+an\s+overview|provide\s+a\s+detailed\s+description|explain\s+what\s+is\s+visible|explain\s+what\s+you\s+see)\s+(of|in|about)\s+(the|this)\s*(satellite\s+|aerial\s+)?(image|scene|landscape|photo|picture)?\.?$", re.I),
    re.compile(r"^what\s+does\s+this\s*(aerial|satellite)?\s*(photograph|image|scene|photo)\s*show\??$", re.I),
    re.compile(r"^describe\s+(the|this)?\s*(scene|landscape|satellite\s+image|aerial\s+image)\.?$", re.I),
]

# Grounding Target Extraction Map
_GROUNDING_PREFIXES = (
    "where is the", "where are the", "where is", "where are",
    "locate the", "locate", "find the", "find",
    "highlight the", "highlight", "show where the", "show where is", "show where are",
    "show where", "pinpoint the", "pinpoint", "segment the", "segment",
    "spot the", "spot", "detect where", "bounding box for", "bbox for", "coordinates of",
)

_TARGET_NORMALIZATIONS = {
    "water body": "water_body",
    "water bodies": "water_body",
    "water-covered area": "water_body",
    "water covered area": "water_body",
    "water": "water_body",
    "river": "river",
    "rivers": "river",
    "lake": "lake",
    "lakes": "lake",
    "reservoir": "reservoir",
    "ocean": "ocean",
    "sea": "ocean",
    "building": "building",
    "buildings": "building",
    "residential building": "building",
    "residential buildings": "building",
    "house": "building",
    "houses": "building",
    "industrial building": "industrial_facility",
    "industrial facility": "industrial_facility",
    "industrial buildings": "industrial_facility",
    "storage tank": "tanks",
    "storage tanks": "tanks",
    "oil tank": "tanks",
    "oil tanks": "tanks",
    "oil storage tank": "tanks",
    "oil storage tanks": "tanks",
    "tanks": "tanks",
    "tank": "tanks",
    "road": "roads",
    "roads": "roads",
    "road network": "roads",
    "highway": "roads",
    "highways": "roads",
    "runway": "runway",
    "runways": "runway",
    "airport runway": "runway",
    "airport": "airport",
    "train": "train",
    "trains": "train",
    "train track": "railway",
    "train tracks": "railway",
    "railway": "railway",
    "railroad": "railway",
    "bridge": "bridge",
    "bridges": "bridge",
    "forest": "forest",
    "forests": "forest",
    "forest area": "forest",
    "trees": "forest",
    "tree": "forest",
    "vegetation": "vegetation",
    "agricultural field": "agriculture",
    "agricultural fields": "agriculture",
    "agriculture": "agriculture",
    "farm": "agriculture",
    "farms": "agriculture",
    "farmland": "agriculture",
    "fields": "agriculture",
    "field": "agriculture",
    "crop": "agriculture",
    "crops": "agriculture",
    "stadium": "stadium",
    "sports stadium": "stadium",
    "harbor": "harbor",
    "harbour": "harbor",
    "port": "harbor",
    "dock": "harbor",
    "ship": "ship",
    "ships": "ship",
    "vessel": "ship",
    "vessels": "ship",
    "airplane": "airplane",
    "airplanes": "airplane",
    "aircraft": "airplane",
    "plane": "airplane",
    "planes": "airplane",
    "car": "vehicles",
    "cars": "vehicles",
    "vehicle": "vehicles",
    "vehicles": "vehicles",
}

# Domain Remote-Sensing Keywords (Used to confirm remote sensing context for general VQA)
_REMOTE_SENSING_ENTITIES = {
    "image", "scene", "satellite", "aerial", "remote sensing", "earth", "sensor",
    "land", "terrain", "landscape", "urban", "rural", "city", "town", "village",
    "building", "buildings", "roof", "roofs", "structure", "structures", "house", "houses",
    "road", "roads", "highway", "railway", "train", "track", "airport", "runway", "tarmac",
    "bridge", "harbor", "port", "dock", "pier", "ship", "ships", "boat", "vessel", "airplane", "aircraft",
    "water", "river", "lake", "ocean", "sea", "pond", "reservoir", "coast", "coastline", "shore",
    "forest", "tree", "trees", "woodland", "vegetation", "grass", "field", "fields", "agriculture",
    "crop", "crops", "farm", "farmland", "pasture", "canopy", "greenery",
    "soil", "sand", "desert", "mountain", "hill", "peak", "valley", "glacier", "snow", "ice",
    "cloud", "clouds", "shadow", "haze", "modality", "optical", "sar", "radar", "multispectral",
    "sentinel", "landsat", "ndvi", "ndwi", "ndbi", "elevation", "srtm", "dem",
    "industrial", "factory", "plant", "tank", "tanks", "storage", "solar", "panel", "panels",
    "parking", "lot", "car", "cars", "vehicle", "vehicles", "quarry", "mine", "mining",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Extraction Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_grounding_target(query: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract and normalize spatial grounding target and operation.

    Returns:
        (canonical_target, operation) or (None, None)
    """
    q_clean = query.strip().rstrip("?.!").lower()

    matched_prefix = None
    for prefix in _GROUNDING_PREFIXES:
        if q_clean.startswith(prefix + " ") or q_clean == prefix:
            matched_prefix = prefix
            break

    if matched_prefix:
        raw_target = q_clean[len(matched_prefix):].strip()
        # Clean articles
        raw_target = re.sub(r"^(the|a|an|any|all)\s+", "", raw_target).strip()
        # Clean trailing descriptors
        raw_target = re.sub(r"\s+(in|on|across|inside|within)\s+(the|this)\s+(satellite\s+|aerial\s+)?(image|scene|photo).*$", "", raw_target).strip()

        if raw_target in _TARGET_NORMALIZATIONS:
            return _TARGET_NORMALIZATIONS[raw_target], "locate"

        # Partial substring match against normalization keys
        for key, canonical in sorted(_TARGET_NORMALIZATIONS.items(), key=lambda x: -len(x[0])):
            if key in raw_target:
                return canonical, "locate"

        # Fallback sanitized target if non-empty
        cleaned = re.sub(r"[^\w\s]", "", raw_target).strip().replace(" ", "_")
        if cleaned:
            return cleaned, "locate"

    return None, None


def is_explicitly_invalid(query: str) -> Tuple[bool, Optional[str]]:
    """Check if query matches known out-of-domain patterns.

    Returns:
        (is_invalid, rejection_reason)
    """
    for pattern in _INVALID_PATTERNS:
        if pattern.search(query):
            return True, "Query does not pertain to remote-sensing, satellite imagery, geospatial analysis, or supported visual tasks."

    return False, None


def is_scene_description(query: str) -> bool:
    """Check if query is a paraphrase of scene description / captioning."""
    q_clean = query.strip()
    if not q_clean:
        return True

    for pattern in _SCENE_DESCRIPTION_PATTERNS:
        if pattern.match(q_clean):
            return True

    q_lower = q_clean.lower().rstrip("?.!")
    generic_phrases = {
        "describe the image", "describe this image", "describe image",
        "describe the scene", "describe this scene", "describe scene",
        "what is in the image", "what is in this image", "what is there in the image",
        "what can you see", "what can you see here", "what do you see in this image",
        "what do you see", "tell me about this image", "tell me about this scene",
        "explain this image", "give an overview", "summarize the image",
        "summarize this image", "caption this image", "caption the image",
        "what is visible in this image", "what does this image show",
        "what does this aerial photograph show", "provide a detailed description of the landscape",
        "give me an overview of this satellite image", "describe the satellite image",
    }
    return q_lower in generic_phrases


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Validation & Normalization API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_query_intent(
    query: str | None,
    image_count: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """Validate query against supported remote sensing domains and normalize to canonical representation.

    Parameters
    ----------
    query : str | None
        User's natural-language query.
    image_count : int
        Number of uploaded or referenced satellite images.
    metadata : Dict[str, Any] | None
        Optional request metadata.

    Returns
    -------
    ValidationResult
        Structured validation status, canonical intent, prompt, and parameters.
    """
    meta = metadata or {}
    raw_query = (query or "").strip()

    # Empty query with image(s) defaults to scene description
    if not raw_query:
        if image_count >= 1:
            return ValidationResult(
                valid=True,
                intent="scene_description",
                canonical_task="scene_description",
                canonical_prompt=CANONICAL_SCENE_DESCRIPTION_PROMPT,
                confidence=1.0,
                operation="scene_description",
            )
        else:
            return ValidationResult(
                valid=False,
                intent="invalid",
                canonical_task="invalid",
                canonical_prompt="",
                confidence=0.0,
                reason="Query string is empty and no satellite image was provided.",
            )

    # 1. Check for explicit out-of-domain rejection patterns
    is_invalid, reason = is_explicitly_invalid(raw_query)
    if is_invalid:
        return ValidationResult(
            valid=False,
            intent="invalid",
            canonical_task="invalid",
            canonical_prompt="",
            confidence=0.0,
            reason=reason or "Query is outside SatQuery's supported remote-sensing tasks.",
        )

    q_lower = raw_query.lower()

    # 2. Check for Scene Description Paraphrases
    if is_scene_description(raw_query):
        return ValidationResult(
            valid=True,
            intent="scene_description",
            canonical_task="scene_description",
            canonical_prompt=CANONICAL_SCENE_DESCRIPTION_PROMPT,
            confidence=0.98,
            operation="scene_description",
        )

    # 3. Check for Optical-SAR Fusion Intent
    fusion_keywords = (
        "optical and sar", "sar and optical", "both modalities", "cross-modal",
        "fuse optical", "fusion model", "optical and radar", "radar and optical",
        "sentinel-1 and sentinel-2", "sentinel-2 and sentinel-1",
        "combine optical and sar", "use the optical and sar", "use optical and sar",
    )
    if any(k in q_lower for k in fusion_keywords):
        return ValidationResult(
            valid=True,
            intent="optical_sar_analysis",
            canonical_task="fusion",
            canonical_prompt=raw_query or CANONICAL_FUSION_PROMPT,
            confidence=0.95,
            operation="cross_modal_fusion",
        )

    # 4. Check for Bi-Temporal Change Detection Intent
    change_keywords = (
        "what changed", "changed between", "difference between", "has changed",
        "detect change", "increased between", "decreased between", "increase or decrease",
        "urban expansion", "expansion between", "deforestation", "growth between", "loss between",
        "before and after", "between before and after", "between these images",
        "between the two images", "between image 1 and image 2", "between the two dates",
        "difference across time", "temporal change",
    )
    is_change = any(k in q_lower for k in change_keywords) or (
        any(c in q_lower for c in ("change", "changed", "changes", "increase", "decrease", "growth", "loss"))
        and any(t in q_lower for t in ("between", "before", "after", "temporal", "dates", "images", "two dates", "over time"))
    )
    if is_change:
        return ValidationResult(
            valid=True,
            intent="change_analysis",
            canonical_task="change_vqa",
            canonical_prompt=raw_query or CANONICAL_CHANGE_PROMPT,
            confidence=0.95,
            operation="bi_temporal_change",
        )

    # 5. Check for Geospatial / GEE Intents (NDVI, NDWI, NDBI, Elevation, Temporal retrieval, Sentinel/Landsat search)
    if any(k in q_lower for k in ("ndvi", "vegetation index", "calculate ndvi", "compute ndvi")):
        return ValidationResult(
            valid=True,
            intent="ndvi_analysis",
            canonical_task="gee",
            canonical_prompt=raw_query,
            confidence=0.95,
            operation="index_calculation",
            extracted_params={"metric": "NDVI"},
        )
    if any(k in q_lower for k in ("ndwi", "water index", "calculate ndwi", "ndbi", "built-up index")):
        return ValidationResult(
            valid=True,
            intent="ndvi_analysis",
            canonical_task="gee",
            canonical_prompt=raw_query,
            confidence=0.95,
            operation="index_calculation",
        )
    if any(k in q_lower for k in ("elevation", "height at", "altitude at", "srtm", "dem")):
        return ValidationResult(
            valid=True,
            intent="area_analysis",
            canonical_task="gee",
            canonical_prompt=raw_query,
            confidence=0.95,
            operation="elevation_query",
            extracted_params={"metric": "ELEVATION"},
        )
    gee_retrieval_keywords = (
        "sentinel-1", "sentinel-2", "sentinel", "landsat", "earth engine", "gee",
        "fetch imagery", "search imagery", "retrieve imagery", "retrieve sentinel", "find sentinel",
    )
    if any(k in q_lower for k in gee_retrieval_keywords) and (
        image_count == 0 or any(k in q_lower for k in ("find", "fetch", "get", "retrieve", "search", "download", "between", "cloud", "polarization", "latitude", "lat", "longitude", "lon"))
    ):
        return ValidationResult(
            valid=True,
            intent="temporal_analysis",
            canonical_task="gee",
            canonical_prompt=raw_query,
            confidence=0.95,
            operation="imagery_retrieval",
        )

    # 6. Check for Spatial Grounding Intent
    target, operation = extract_grounding_target(raw_query)
    if target is not None:
        target_display = target.replace("_", " ")
        canonical_prompt = CANONICAL_GROUNDING_TEMPLATE.format(target=target_display)
        return ValidationResult(
            valid=True,
            intent="grounding",
            canonical_task="grounding",
            canonical_prompt=canonical_prompt,
            target=target,
            operation=operation or "locate",
            confidence=0.92,
            extracted_params={"target": target, "operation": operation or "locate"},
        )

    # 7. Check for Visual Question Answering (VQA)
    # Questions about visual presence, count, classification, attributes
    is_question_format = (
        raw_query.endswith("?")
        or re.match(r"^(is|are|does|do|can|how|what|which|where|why|has|have|could|would)\b", q_lower)
    )

    # Check for domain keywords or spatial context in VQA
    tokens = set(re.findall(r"\b\w+\b", q_lower))
    has_domain_keywords = bool(tokens.intersection(_REMOTE_SENSING_ENTITIES))

    if is_question_format or has_domain_keywords:
        # Standardize question formatting (ensure ends with question mark if question)
        normalized_q = raw_query.strip()
        if is_question_format and not normalized_q.endswith("?"):
            normalized_q += "?"

        return ValidationResult(
            valid=True,
            intent="vqa",
            canonical_task="vqa",
            canonical_prompt=normalized_q,
            confidence=0.90,
            operation="visual_qa",
        )

    # 8. Fallback: If no recognized remote sensing keywords and not a valid visual inquiry -> Reject
    return ValidationResult(
        valid=False,
        intent="invalid",
        canonical_task="invalid",
        canonical_prompt="",
        confidence=0.0,
        reason="Query does not contain recognized remote sensing or geospatial terms, nor does it ask a supported visual analysis question.",
    )
