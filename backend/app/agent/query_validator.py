"""
SatQuery AI — Query Validation, Domain Guardrail & Normalization Layer.

Enforces strict domain guardrails before specialist model execution:
1. Rejects out-of-domain / general world-knowledge queries (e.g. "How does a dog walk?").
2. Differentiates between observational visual queries about objects in the image vs. general knowledge.
3. Checks image-query compatibility (e.g. bi-temporal change requires 2+ images).
4. Normalizes semantically equivalent queries into canonical task representations and fixed specialist model prompts.
5. Returns a 3-state validation result: VALID / NEEDS_CLARIFICATION / INVALID.
   - VALID            : query is in-domain and all required inputs are present.
   - NEEDS_CLARIFICATION : intent is recognisable but inputs are insufficient
                          (e.g. change_analysis with only 1 image uploaded).
   - INVALID          : query is genuinely out-of-domain; no specialist models run.
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
# 3-State Guardrail Status Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GuardrailStatus:
    """Enumeration of the three validation states."""
    VALID = "valid"
    INVALID = "invalid"
    NEEDS_CLARIFICATION = "needs_clarification"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Domain Guardrail Result Dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class DomainGuardrailResult:
    """Structured result produced by the SatQuery Domain Guardrail.

    ``status`` is the canonical 3-state field:
        - ``GuardrailStatus.VALID``               → execute specialist models
        - ``GuardrailStatus.NEEDS_CLARIFICATION`` → do NOT run models; ask user
        - ``GuardrailStatus.INVALID``             → out-of-domain; reject

    ``domain_valid`` is kept for backward compatibility and is ``True`` only
    when ``status == GuardrailStatus.VALID``.
    """

    domain_valid: bool
    domain: str  # 'remote_sensing', 'unsupported', 'incompatible_input'
    domain_confidence: float = 1.0
    reason: Optional[str] = None
    # 3-state extension fields
    status: str = GuardrailStatus.INVALID  # set explicitly in every code-path
    task_detected: Optional[str] = None    # inferred task even when not yet valid

    def to_dict(self) -> Dict[str, Any]:
        """Convert guardrail result to dictionary representation."""
        return {
            "domain_valid": self.domain_valid,
            "domain": self.domain,
            "domain_confidence": self.domain_confidence,
            "reason": self.reason,
            "status": self.status,
            "task_detected": self.task_detected,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Validation Result Dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ValidationResult:
    """Structured result produced by the Query Validation & Normalization Layer.

    ``status`` mirrors the 3-state guardrail:
        - ``'valid'``                → proceed to specialist model execution
        - ``'needs_clarification'``  → do NOT run models; surface message to user
        - ``'invalid'``              → out-of-domain rejection
    """

    valid: bool
    intent: str  # 'scene_description', 'vqa', 'grounding', 'change_analysis', 'optical_sar_analysis', 'ndvi_analysis', 'area_analysis', 'temporal_analysis', 'invalid'
    canonical_task: str  # Standard task identifier ('scene_description', 'vqa', 'grounding', 'change_vqa', 'fusion', 'gee', 'invalid')
    canonical_prompt: str
    target: Optional[str] = None
    operation: Optional[str] = None
    confidence: float = 1.0
    reason: Optional[str] = None
    extracted_params: Dict[str, Any] = field(default_factory=dict)
    # 3-state status field
    status: str = GuardrailStatus.INVALID

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary representation.

        Returns the exact JSON shapes specified by the SatQuery guardrail contract:

        VALID::

            {"valid": true, "status": "valid",
             "task_detected": "change_analysis",
             "canonical_task": "change_analysis"}

        NEEDS_CLARIFICATION::

            {"valid": false, "status": "needs_clarification",
             "task_detected": "change_analysis",
             "canonical_task": "change_analysis",
             "reason": "Two compatible images are required for change analysis."}

        INVALID::

            {"valid": false, "status": "invalid",
             "task_detected": "invalid", "canonical_task": null,
             "reason": "Query is outside SatQuery's supported remote-sensing capabilities."}
        """
        data: Dict[str, Any] = {
            "valid": self.valid,
            "status": self.status,
            "task_detected": self.intent,
            "intent": self.intent,
            "canonical_task": self.canonical_task if self.canonical_task != "invalid" else None,
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

# ── Change-Analysis Semantic Patterns ──────────────────────────────
# These patterns detect change/comparison intent from the query text
# ALONE — without requiring temporal context words ("between",
# "before", "after", etc.).  When image_count >= 2 these are VALID;
# when image_count == 1 they are NEEDS_CLARIFICATION.
# The patterns are intentionally broad and typo-tolerant.
_CHANGE_ANALYSIS_SEMANTIC_PATTERNS: List[re.Pattern] = [
    # "what changed?", "what has changed?"
    re.compile(r"\bwhat\s+(has\s+)?changed\b", re.I),
    # "what are the changes?", "what are the changes and similar sort of?"
    re.compile(r"\bwhat\s+are\s+the\s+changes?\b", re.I),
    # "compare these images", "compare the two images", "compare both images"
    re.compile(r"\bcompare\s+(these|the|both|two)\s+(images?|photos?|pictures?|scenes?)\b", re.I),
    # "compare images", "compare the images"
    re.compile(r"\bcompare\s+(the\s+)?images?\b", re.I),
    # "how are these images different?", "how do these images differ?"
    re.compile(r"\bhow\s+(are|do|were|did)\s+(these|the|both|two)\s+(images?|photos?|pictures?|scenes?)\s+(different|differ)", re.I),
    # "differences between the images", "differences in the images"
    re.compile(r"\bdifferences?\s+(between|in|across)\s+(the|these|both|two)\s+(images?|photos?|scenes?)", re.I),
    # "show the differences", "identify differences"
    re.compile(r"\b(show|identify|list|highlight|detect|find|analyze|analyse)\s+the\s+differences?\b", re.I),
    # "what's different between these?", "what is different between these?"
    re.compile(r"\bwhat\s+('?s|is|are)\s+(different|the\s+difference)\b", re.I),
    # "changes and similar sort of", "changes and things like that" — trailing vague phrases
    re.compile(r"\bchanges?\s+(and|or)\s+(similar|like|such)", re.I),
    # "any changes?", "are there any changes?"
    re.compile(r"\b(any\s+changes?|are\s+there\s+(any\s+)?changes?)\b", re.I),
    # "identify the changes", "show the changes", "list all changes"
    re.compile(r"\b(identify|show|list|highlight|detect|find|analyze|analyse)\s+(all\s+)?(?:the\s+)?changes?\b", re.I),
    # "what has been modified?", "what was modified?"
    re.compile(r"\bwhat\s+(has\s+been|was|were|have\s+been)\s+(modified|altered|changed)\b", re.I),
]

# Strict Out-of-Domain / General World Knowledge Patterns
_INVALID_PATTERNS: List[re.Pattern] = [
    # Programming, Code, Scripts
    re.compile(r"\b(write|generate|debug|fix|create)\s+(python|javascript|java|c\+\+|html|css|code|script|sql|regex)\b", re.I),
    re.compile(r"\b(write\s+code|print\(|def\s+\w+\(|function\s+\w+\(|class\s+\w+:)\b", re.I),
    
    # General Trivia, World Facts, Governments, Celebrities
    re.compile(r"\bcapital\s+of\b", re.I),
    re.compile(r"\bwho\s+(won|scored|played)\s+(the\s+)?(football|cricket|fifa|world\s+cup|super\s+bowl|match|game|nba|tennis|olympics)\b", re.I),
    re.compile(r"\bwho\s+(is|was|are|were)\s+(albert\s+einstein|newton|shakespeare|napoleon|gandhi|lincoln|elon\s+musk|bill\s+gates|the\s+president|the\s+prime\s+minister|prime\s+minister|ceo\s+of)\b", re.I),
    re.compile(r"\bwhat\s+is\s+the\s+(population|currency|anthem|flag|gdp|motto)\s+of\b", re.I),
    
    # General Math / Science / Physics unrelated to remote sensing
    re.compile(r"\b(solve|calculate|what\s+is)\s+\d+\s*[\+\-\*\/x\^]\s*\d+\b", re.I),
    re.compile(r"\b(solve\s+the\s+equation|quadratic\s+equation|quantum\s+mechanics|quantum\s+physics|quantum\s+computing|speed\s+of\s+light|theory\s+of\s+relativity)\b", re.I),
    re.compile(r"\bexplain\s+(quantum\s+mechanics|quantum\s+physics|relativity|gravity|photosynthesis|evolution|machine\s+learning|blockchain|string\s+theory|thermodynamics)\b", re.I),
    
    # Creative writing, recipes, cooking, translation, chit-chat, email, weather
    re.compile(r"\b(write|compose)\s+(a\s+)?(poem|story|song|essay|joke|riddle|letter|email)\b", re.I),
    re.compile(r"\b(recipe\s+for|how\s+(do|can|to|should)\s+(i|we|you)?\s*(cook|bake|make\s+pasta|make\s+cake|fix|repair|change\s+a\s+tire|clean|learn|draw))\b", re.I),
    re.compile(r"\b(how\s+do\s+i\s+cook\b|how\s+to\s+cook\b|cook\s+pasta\b)", re.I),
    re.compile(r"\b(translate\s+.+\s+to\s+(french|spanish|german|hindi|chinese|japanese|russian|latin))\b", re.I),
    re.compile(r"\b(tell\s+me\s+a\s+joke|how\s+are\s+you\s+doing|what\s+is\s+your\s+favorite\s+(movie|food|color))\b", re.I),
    re.compile(r"\b(what\s+is\s+today'?s\s+weather|current\s+weather|weather\s+forecast|what('?s|\s+is)\s+the\s+weather)\b", re.I),

    # How-To & General Explanatory / Biological / Physical Mechanics of objects (OUT-OF-DOMAIN)
    # e.g., "How does a dog walk?", "How do birds fly?", "How does an airplane fly?", "How do engines work?"
    re.compile(r"\bhow\s+(does|do|can)\s+[\w\s]+\s+(walk|run|fly|work|function|operate|live|eat|grow|reproduce|breathe|behave|move|speak|think|digest)\b", re.I),
    # e.g., "How is water purified?", "How are buildings constructed?", "How is cement manufactured?"
    re.compile(r"\bhow\s+(is|are)\s+[\w\s]+\s+(purified|constructed|built|manufactured|produced|created|cooked|prepared|formed|generated|synthesized|processed|refined)\b", re.I),
    # e.g., "How fast does a train travel?", "How fast do planes fly?"
    re.compile(r"\bhow\s+fast\s+(does|do|can|is|are)\b", re.I),
    # e.g., "How much does a house cost?", "How much does a plane weigh?"
    re.compile(r"\bhow\s+much\s+(does|do|is|are)\s+[\w\s]+\s+(cost|weigh|worth)\b", re.I),
    # e.g., "How long does a dog live?"
    re.compile(r"\bhow\s+long\s+(does|do|will|did)\s+[\w\s]+\s+(take|live|last)\b", re.I),
    # e.g., "Why is the sky blue?", "Why is water wet?"
    re.compile(r"\bwhy\s+(is|are|do|does)\s+(the\s+sky\s+blue|water\s+wet|birds\s+sing|grass\s+green)\b", re.I),
]

# Scene Description Paraphrases & Patterns
_SCENE_DESCRIPTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"^(describe|caption|summarize|overview)\s+(the|this|an)?\s*(satellite\s+|aerial\s+)?(image|scene|photo|photograph|picture)?\.?$", re.I),
    re.compile(r"^what\s+(is\s+(in|there\s+in)|do\s+you\s+see\s+in|can\s+you\s+see(\s+here)?\s*(in)?)\s*(the|this|an)?\s*(satellite\s+|aerial\s+)?(image|scene|photo|picture)?\??$", re.I),
    re.compile(r"^tell\s+me\s+about\s+(the|this)\s*(satellite\s+|aerial\s+)?(image|scene|photo|picture)\.?$", re.I),
    re.compile(r"^(give\s+me\s+an\s+overview|provide\s+a\s+detailed\s+description|explain\s+what\s+is\s+visible|explain\s+what\s+you\s+see)\s+(of|in|about)\s+(the|this)\s*(satellite\s+|aerial\s+)?(image|scene|landscape|photo|picture)?\.?$", re.I),
    re.compile(r"^what\s+does\s+this\s*(aerial|satellite)?\s*(photograph|image|scene|photo)\s*show\??$", re.I),
    re.compile(r"^describe\s+(the|this)?\s*(scene|landscape|satellite\s+image|aerial\s+image|image)\.?$", re.I),
]

# Grounding Target Extraction Map
_GROUNDING_PREFIXES = (
    "show where are the", "show where is the", "show where the",
    "show where are", "show where is", "show where",
    "show me where are", "show me where is", "show me where",
    "show me the", "show me", "show the",
    "where are the", "where is the", "where are", "where is",
    "identify the", "identify",
    "locate the", "locate",
    "find the", "find",
    "highlight the", "highlight",
    "pinpoint the", "pinpoint",
    "segment the", "segment",
    "spot the", "spot",
    "detect where", "bounding box for", "bbox for", "coordinates of",
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
    "parking": "parking",
    "quarry": "quarry",
    "mine": "mine",
    "vehicles": "vehicles",
    "car": "vehicles",
    "cars": "vehicles",
    "vehicle": "vehicles",
}

# Explicitly out-of-domain non-remote-sensing entities (pets, domestic animals, humans, indoor items, consumer objects)
_OUT_OF_DOMAIN_ENTITIES = {
    "dog", "dogs", "puppy", "puppies", "cat", "cats", "kitten", "kittens",
    "pet", "pets", "cow", "cows", "horse", "horses", "pig", "pigs", "sheep",
    "goat", "goats", "bird", "birds", "lion", "tiger", "bear", "elephant",
    "monkey", "person", "people", "man", "woman", "child", "children", "baby",
    "human", "humans", "face", "faces", "hand", "hands", "finger",
    "food", "pizza", "burger", "coffee", "tea", "cake", "bread", "apple", "banana",
    "table", "chair", "sofa", "bed", "furniture", "desk", "room", "bedroom", "kitchen",
    "laptop", "computer", "phone", "mobile", "cellphone", "tv", "television", "keyboard",
    "mouse", "bottle", "cup", "glass", "fork", "spoon", "knife", "plate",
    "shoe", "shoes", "shirt", "pants", "dress", "hat", "clothes", "clothing",
    "bicycle", "bike", "motorcycle", "scooter", "car key", "wallet", "bag",
}

# Domain Remote-Sensing Entities (Used to confirm remote sensing context for visual tasks)
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
# Helper Extraction & Matching Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _is_change_analysis_intent(query: str) -> bool:
    """Return True if the query expresses a change / comparison intent.

    Evaluates three layers, from most-specific to most-semantic:
    1. Exact multi-word keyword phrases (e.g. ``"what changed between"``)
    2. Compound condition: a change word + a temporal context word
    3. Semantic regex patterns that catch informal / vague phrasings
       (e.g. ``"What are the changes and similar sort of?"``)

    The function does NOT consider image_count — that decision belongs
    in ``evaluate_domain_guardrail`` which applies the 3-state logic.
    """
    q_lower = query.lower()

    # Layer 1: exact multi-word keyword phrases
    _change_keywords = (
        "what changed", "changed between", "difference between", "has changed",
        "detect change", "increased between", "decreased between", "increase or decrease",
        "urban expansion", "expansion between", "deforestation", "growth between", "loss between",
        "before and after", "between before and after", "between these images",
        "between the two images", "between image 1 and image 2", "between the two dates",
        "difference across time", "temporal change", "has vegetation decreased",
        "has vegetation increased",
    )
    if any(k in q_lower for k in _change_keywords):
        return True

    # Layer 2: compound condition — change word + temporal word
    _change_words = ("change", "changed", "changes", "increase", "decrease", "growth", "loss")
    _temporal_words = (
        "between", "before", "after", "temporal", "dates", "image", "images",
        "two dates", "over time", "next", "previous", "prior", "later",
    )
    if (
        any(c in q_lower for c in _change_words)
        and any(t in q_lower for t in _temporal_words)
    ):
        return True

    # Layer 3: semantic regex patterns (informal / vague phrasings)
    for pattern in _CHANGE_ANALYSIS_SEMANTIC_PATTERNS:
        if pattern.search(query):
            return True

    return False

def extract_grounding_target(query: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract and normalize spatial grounding target and operation.

    Returns:
        (canonical_target, operation) or (None, None)
    """
    q_clean = query.strip().rstrip("?.!").lower()

    # Guard: Counting queries must never be classified as grounding
    if re.search(r"\b(how many|count|number of)\b", q_clean):
        return None, None

    matched_prefix = None
    for prefix in _GROUNDING_PREFIXES:
        if q_clean.startswith(prefix + " ") or q_clean == prefix:
            matched_prefix = prefix
            break

    if matched_prefix:
        raw_target = q_clean[len(matched_prefix):].strip()
        # Clean articles
        raw_target = re.sub(r"^(the|a|an|any|all)\s+", "", raw_target).strip()
        # Clean trailing descriptors and spatial verbs
        raw_target = re.sub(r"\s+(are\s+located|are\s+situated|located|situated|found|are)\b.*$", "", raw_target).strip()
        raw_target = re.sub(r"\s+(in|on|across|inside|within)\s+(the|this)\s+(satellite\s+|aerial\s+)?(image|scene|photo).*$", "", raw_target).strip()

        if raw_target in _TARGET_NORMALIZATIONS:
            return _TARGET_NORMALIZATIONS[raw_target], "locate"

        # Partial substring match against normalization keys
        for key, canonical in sorted(_TARGET_NORMALIZATIONS.items(), key=lambda x: -len(x[0])):
            if key in raw_target:
                return canonical, "locate"

        # For generic verbs (identify, show, show me), only treat as grounding if target refers to a remote-sensing entity
        generic_prefixes = ("identify", "identify the", "show", "show the", "show me", "show me the")
        non_spatial_tokens = {"image", "scene", "photo", "types", "what", "classes", "difference", "change", "changes", "land"}

        if matched_prefix in generic_prefixes:
            target_tokens = set(re.findall(r"\b\w+\b", raw_target))
            valid_spatial = bool(target_tokens.intersection(_REMOTE_SENSING_ENTITIES - non_spatial_tokens))
            if not valid_spatial:
                return None, None

        # Fallback sanitized target if non-empty
        cleaned = re.sub(r"[^\w\s]", "", raw_target).strip().replace(" ", "_")
        if cleaned and cleaned not in non_spatial_tokens:
            return cleaned, "locate"

    return None, None


def is_explicitly_invalid(query: str) -> Tuple[bool, Optional[str]]:
    """Check if query matches known out-of-domain patterns.

    Returns:
        (is_invalid, rejection_reason)
    """
    for pattern in _INVALID_PATTERNS:
        if pattern.search(query):
            return True, "The query requests general information unrelated to remote-sensing image analysis."

    return False, None


def is_scene_description(query: str) -> bool:
    """Check if query is a paraphrase of scene description / captioning with typo tolerance."""
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
        "what is ther in the image", "what is there in image", "what is in image",
        "what can you see", "what can you see here", "what do you see in this image",
        "what do you see", "tell me about this image", "tell me about this scene",
        "explain this image", "give an overview", "summarize the image",
        "summarize this image", "caption this image", "caption the image",
        "what is visible in this image", "what does this image show",
        "what does this aerial photograph show", "provide a detailed description of the landscape",
        "give me an overview of this satellite image", "describe the satellite image",
        "desribe the image", "desribe this image", "describe the buildings",
        "describe the landscape",
    }
    if q_lower in generic_phrases:
        return True

    # Fuzzy / typo tolerance for common inputs like "whar iaa desceibed i image"
    tokens = set(re.findall(r"\b\w+\b", q_lower))
    has_image_term = any(t in tokens for t in ("image", "scene", "photo", "picture", "aerial", "satellite", "landscape", "imge", "imaeg"))
    has_describe_term = any(t in tokens for t in ("describe", "desribe", "desceibe", "desceibed", "described", "caption", "summarize", "overview", "explanation"))
    has_view_term = any(t in tokens for t in ("see", "visible", "show", "visable"))
    
    if has_image_term and (has_describe_term or (("what" in tokens or "whar" in tokens) and (has_view_term or "in" in tokens or "i" in tokens))):
        return True

    return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Domain Guardrail API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_domain_guardrail(
    query: str | None,
    image_count: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
    images: Optional[List[Any]] = None,
) -> DomainGuardrailResult:
    if images is not None and image_count == 0:
        image_count = len(images)
    """Evaluate whether the query belongs to SatQuery's supported remote-sensing domain.
    
    Executes BEFORE specialist model execution to:
    1. Reject out-of-domain / general world knowledge queries.
    2. Distinguish queries about objects in the image vs. general knowledge.
    3. Check image-query compatibility (e.g. change detection requiring 2+ images).
    """
    meta = metadata or {}
    raw_query = (query or "").strip()

    # Empty query with image(s) is valid (defaults to scene description)
    if not raw_query:
        if image_count >= 1:
            return DomainGuardrailResult(
                domain_valid=True,
                domain="remote_sensing",
                domain_confidence=1.0,
                reason=None,
                status=GuardrailStatus.VALID,
                task_detected="scene_description",
            )
        return DomainGuardrailResult(
            domain_valid=False,
            domain="unsupported",
            domain_confidence=1.0,
            reason="Query string is empty and no satellite image was provided.",
            status=GuardrailStatus.INVALID,
            task_detected=None,
        )

    # Step 1: Explicit out-of-domain pattern rejection
    is_invalid, reason = is_explicitly_invalid(raw_query)
    if is_invalid:
        return DomainGuardrailResult(
            domain_valid=False,
            domain="unsupported",
            domain_confidence=0.98,
            reason=reason or "The query requests general information unrelated to remote-sensing image analysis.",
            status=GuardrailStatus.INVALID,
            task_detected="invalid",
        )

    q_lower = raw_query.lower()
    tokens = set(re.findall(r"\b[a-z]+\b", q_lower))

    # Check for non-remote-sensing entities (pets, domestic animals, humans, indoor items, consumer objects)
    out_of_domain_matches = tokens.intersection(_OUT_OF_DOMAIN_ENTITIES)
    if out_of_domain_matches:
        matched_terms = ", ".join(sorted(out_of_domain_matches))
        return DomainGuardrailResult(
            domain_valid=False,
            domain="unsupported",
            domain_confidence=0.99,
            reason=f"The query asks about '{matched_terms}', which is a domestic, biological, or ground-level object not observable in satellite remote-sensing imagery.",
            status=GuardrailStatus.INVALID,
            task_detected="invalid",
        )

    # Step 2: Recognized Remote-Sensing Canonical Tasks
    if is_scene_description(raw_query):
        return DomainGuardrailResult(
            domain_valid=True,
            domain="remote_sensing",
            domain_confidence=0.98,
            reason=None,
            status=GuardrailStatus.VALID,
            task_detected="scene_description",
        )

    # ── Change-Analysis: 3-state detection ──────────────────────────
    # Evaluate query semantics + image_count together.
    # We use the unified helper so informal phrasings are also caught.
    if _is_change_analysis_intent(raw_query):
        if image_count >= 2:
            # All required inputs are present → VALID
            return DomainGuardrailResult(
                domain_valid=True,
                domain="remote_sensing",
                domain_confidence=0.95,
                reason=None,
                status=GuardrailStatus.VALID,
                task_detected="change_analysis",
            )
        if image_count == 1:
            # Intent is clear but only one image uploaded → NEEDS_CLARIFICATION
            return DomainGuardrailResult(
                domain_valid=False,
                domain="incompatible_input",
                domain_confidence=0.95,
                reason="Two compatible images are required for change analysis.",
                status=GuardrailStatus.NEEDS_CLARIFICATION,
                task_detected="change_analysis",
            )
        # image_count == 0: still recognise the intent, request images
        return DomainGuardrailResult(
            domain_valid=False,
            domain="incompatible_input",
            domain_confidence=0.93,
            reason="Two compatible images are required for change analysis.",
            status=GuardrailStatus.NEEDS_CLARIFICATION,
            task_detected="change_analysis",
        )

    # Future Prediction / Land-Cover Forecasting
    if any(k in q_lower for k in ("predict future", "future land-cover", "future land cover", "predict land cover", "predict land-cover", "forecasting", "future changes", "future change", "predict 20", "forecast 20", "prediction for 20", "future prediction", "forecast", "predict")):
        return DomainGuardrailResult(
            domain_valid=True,
            domain="remote_sensing",
            domain_confidence=0.96,
            reason=None,
            status=GuardrailStatus.VALID,
            task_detected="future_prediction",
        )

    # Optical-SAR Fusion Compatibility
    fusion_keywords = (
        "optical and sar", "sar and optical", "both modalities", "cross-modal",
        "fuse optical", "fusion model", "optical and radar", "radar and optical",
        "sentinel-1 and sentinel-2", "sentinel-2 and sentinel-1",
        "combine optical and sar", "use the optical and sar", "use optical and sar",
    )
    if any(k in q_lower for k in fusion_keywords):
        modalities = meta.get("modalities", [])
        has_sar = "sar" in modalities or "radar" in modalities or any("sar" in str(m).lower() for m in modalities)
        if (image_count < 2 and not has_sar) or (modalities and not has_sar):
            return DomainGuardrailResult(
                domain_valid=False,
                domain="incompatible_input",
                domain_confidence=0.95,
                reason="Optical-SAR fusion requires both an optical image and a SAR image.",
                status=GuardrailStatus.NEEDS_CLARIFICATION,
                task_detected="optical_sar_analysis",
            )
        return DomainGuardrailResult(
            domain_valid=True,
            domain="remote_sensing",
            domain_confidence=0.95,
            reason=None,
            status=GuardrailStatus.VALID,
            task_detected="optical_sar_analysis",
        )

    # GEE / Geospatial / Spectral Indices
    if any(k in q_lower for k in ("ndvi", "ndwi", "ndbi", "vegetation index", "water index", "built-up index", "elevation", "srtm", "dem", "sentinel", "landsat", "earth engine")):
        return DomainGuardrailResult(
            domain_valid=True,
            domain="remote_sensing",
            domain_confidence=0.95,
            reason=None,
            status=GuardrailStatus.VALID,
            task_detected="gee",
        )

    # Grounding / Spatial Localization
    target, operation = extract_grounding_target(raw_query)
    if target is not None:
        return DomainGuardrailResult(
            domain_valid=True,
            domain="remote_sensing",
            domain_confidence=0.94,
            reason=None,
            status=GuardrailStatus.VALID,
            task_detected="grounding",
        )

    # Step 3: Visual Question Answering (VQA)
    is_presence_q = bool(re.match(r"^(is\s+there|are\s+there|do\s+you\s+see|can\s+you\s+see|does\s+(this|the)\s+(image|scene|area|region|landscape)\s+(have|show|contain))\b", q_lower))
    is_count_q = bool(re.match(r"^how\s+many\b", q_lower))
    is_visual_attribute_q = bool(re.match(r"^(what\s+color|what\s+type|is\s+this\s+area|is\s+the\s+cloud|are\s+the\s+roofs|what\s+kind|is\s+it\s+(urban|rural|forested|cloudy))\b", q_lower))
    has_explicit_rs_scope = any(scope in q_lower for scope in ("in the satellite image", "in this satellite image", "in the aerial image", "in the remote sensing image", "in this aerial photo", "satellite image", "aerial scene"))
    has_visual_describe = bool(re.match(r"^describe\s+(the\s+)?\w+", q_lower))

    has_domain_entity = bool(tokens.intersection(_REMOTE_SENSING_ENTITIES))

    # A presence or count query MUST target a recognized remote-sensing entity
    if (is_presence_q or is_count_q) and has_domain_entity:
        return DomainGuardrailResult(
            domain_valid=True,
            domain="remote_sensing",
            domain_confidence=0.92,
            reason=None,
            status=GuardrailStatus.VALID,
            task_detected="vqa",
        )

    if (is_visual_attribute_q or has_visual_describe or has_explicit_rs_scope) and (has_domain_entity or has_explicit_rs_scope):
        return DomainGuardrailResult(
            domain_valid=True,
            domain="remote_sensing",
            domain_confidence=0.90,
            reason=None,
            status=GuardrailStatus.VALID,
            task_detected="vqa",
        )

    if has_domain_entity and (raw_query.endswith("?") or re.match(r"^(is|are|does|do|can|what|which|where|has|have|how)\b", q_lower)):
        return DomainGuardrailResult(
            domain_valid=True,
            domain="remote_sensing",
            domain_confidence=0.88,
            reason=None,
            status=GuardrailStatus.VALID,
            task_detected="vqa",
        )

    # Step 4: Out-of-domain fallback
    return DomainGuardrailResult(
        domain_valid=False,
        domain="unsupported",
        domain_confidence=0.98,
        reason="The query is outside SatQuery's supported remote-sensing analysis domain.",
        status=GuardrailStatus.INVALID,
        task_detected="invalid",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Validation & Normalization API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_query_intent(
    query: str | None,
    image_count: int = 1,
    images: Optional[List[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    if images is not None:
        effective_count = len(images)
    else:
        effective_count = image_count
    """Validate query against supported remote sensing domains and normalize to canonical representation.

    Pipeline:
        USER QUERY + IMAGE(S)
                ↓
        DOMAIN GUARDRAIL (evaluate_domain_guardrail)
                ↓
        If unrelated / incompatible → REJECT immediately
                ↓
        If remote-sensing relevant
                ↓
        QUERY NORMALIZATION & CANONICAL TASK MAPPING
    """
    meta = metadata or {}
    raw_query = (query or "").strip()

    # 1. Execute Strict Domain Guardrail
    guardrail = evaluate_domain_guardrail(
        query=raw_query,
        image_count=effective_count,
        metadata=meta,
        images=images,
    )

    if not guardrail.domain_valid:
        # NEEDS_CLARIFICATION: intent is clear but inputs are insufficient
        # (e.g. change_analysis requested with only 1 image).
        # Surface a helpful message to the user but DO NOT run any specialist models.
        if guardrail.status == GuardrailStatus.NEEDS_CLARIFICATION:
            return ValidationResult(
                valid=False,
                status=GuardrailStatus.NEEDS_CLARIFICATION,
                intent=guardrail.task_detected or "change_analysis",
                canonical_task=guardrail.task_detected or "change_analysis",
                canonical_prompt="",
                confidence=0.0,
                reason=guardrail.reason or "Two compatible images are required for change analysis.",
            )
        # INVALID: genuine out-of-domain rejection
        return ValidationResult(
            valid=False,
            status=GuardrailStatus.INVALID,
            intent="invalid",
            canonical_task="invalid",
            canonical_prompt="",
            confidence=0.0,
            reason=guardrail.reason or "The query is outside SatQuery's supported remote-sensing analysis domain.",
        )

    # Empty query with image(s) defaults to scene description
    if not raw_query:
        return ValidationResult(
            valid=True,
            status=GuardrailStatus.VALID,
            intent="scene_description",
            canonical_task="scene_description",
            canonical_prompt=CANONICAL_SCENE_DESCRIPTION_PROMPT,
            confidence=1.0,
            operation="scene_description",
        )

    q_lower = raw_query.lower()

    # 2. Check for Scene Description Paraphrases
    if is_scene_description(raw_query):
        return ValidationResult(
            valid=True,
            status=GuardrailStatus.VALID,
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
            status=GuardrailStatus.VALID,
            intent="optical_sar_analysis",
            canonical_task="fusion",
            canonical_prompt=raw_query or CANONICAL_FUSION_PROMPT,
            confidence=0.95,
            operation="cross_modal_fusion",
        )

    # 4. Check for Bi-Temporal Change Detection Intent
    # Reuse the unified helper (covers keyword phrases, compound conditions,
    # and informal/vague phrasings like "what are the changes and similar sort of?").
    if _is_change_analysis_intent(raw_query):
        return ValidationResult(
            valid=True,
            status=GuardrailStatus.VALID,
            intent="change_analysis",
            canonical_task="change_vqa",
            canonical_prompt=raw_query or CANONICAL_CHANGE_PROMPT,
            confidence=0.95,
            operation="bi_temporal_change",
        )

    # 5. Check for Geospatial / GEE Intents
    if any(k in q_lower for k in ("ndvi", "vegetation index", "calculate ndvi", "compute ndvi")):
        return ValidationResult(
            valid=True,
            status=GuardrailStatus.VALID,
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
            status=GuardrailStatus.VALID,
            intent="ndvi_analysis",
            canonical_task="gee",
            canonical_prompt=raw_query,
            confidence=0.95,
            operation="index_calculation",
        )
    if any(k in q_lower for k in ("elevation", "height at", "altitude at", "srtm", "dem")):
        return ValidationResult(
            valid=True,
            status=GuardrailStatus.VALID,
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
        effective_count == 0 or any(k in q_lower for k in ("find", "fetch", "get", "retrieve", "search", "download", "between", "cloud", "polarization", "latitude", "lat", "longitude", "lon"))
    ):
        return ValidationResult(
            valid=True,
            status=GuardrailStatus.VALID,
            intent="temporal_analysis",
            canonical_task="gee",
            canonical_prompt=raw_query,
            confidence=0.95,
            operation="imagery_retrieval",
        )

    # 6. Check for Future Prediction / Land-Cover Forecasting Intent
    if guardrail.task_detected == "future_prediction" or any(k in q_lower for k in ("predict future", "future land-cover", "future land cover", "predict land cover", "predict land-cover", "forecasting", "future changes", "future change", "predict 20", "forecast 20", "prediction for 20", "future prediction", "forecast", "predict")):
        return ValidationResult(
            valid=True,
            status=GuardrailStatus.VALID,
            intent="future_prediction",
            canonical_task="future_prediction",
            canonical_prompt=raw_query,
            confidence=0.96,
            operation="future_prediction",
        )

    # 7. Check for Spatial Grounding Intent
    target, operation = extract_grounding_target(raw_query)
    if target is not None:
        target_display = target.replace("_", " ")
        if any(k in q_lower for k in ("bounding box", "bbox", "smallest", "largest", "single", "contiguous", "bounding", "box")):
            canonical_prompt = raw_query
        else:
            canonical_prompt = CANONICAL_GROUNDING_TEMPLATE.format(target=target_display)
        return ValidationResult(
            valid=True,
            status=GuardrailStatus.VALID,
            intent="grounding",
            canonical_task="grounding",
            canonical_prompt=canonical_prompt,
            target=target,
            operation=operation or "locate",
            confidence=0.92,
            extracted_params={"target": target, "operation": operation or "locate"},
        )

    # 7. Check for Visual Question Answering (VQA)
    is_question_format = (
        raw_query.endswith("?")
        or re.match(r"^(is|are|does|do|can|how|what|which|where|why|has|have|could|would|describe)\b", q_lower)
    )

    normalized_q = raw_query.strip()
    if is_question_format and not normalized_q.endswith("?") and not normalized_q.lower().startswith("describe"):
        normalized_q += "?"

    return ValidationResult(
        valid=True,
        status=GuardrailStatus.VALID,
        intent="vqa",
        canonical_task="vqa",
        canonical_prompt=normalized_q,
        confidence=0.90,
        operation="visual_qa",
    )
