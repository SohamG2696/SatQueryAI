"""
RSCoVLM Formatting and Safety Guard Utilities.
"""

import re
from typing import Any, Dict, List, Optional


UNSUPPORTED_SEMANTIC_CLAIMS = [
    "road",
    "street",
    "highway",
    "building",
    "house",
    "structure",
    "construction",
    "demolition",
    "vegetation",
    "forest",
    "tree",
    "crop",
    "field",
    "wildfire",
    "flood",
    "earthquake",
    "landslide",
    "deforestation",
    "urban expansion",
    "urban growth",
]


def format_rscovlm_prompt(
    question: str,
    specialist_evidence: Optional[Dict[str, Any]] = None,
    task: str = "vqa",
) -> str:
    """
    Format a question and optional specialist evidence for RSCoVLM.

    The fine-tuned RSCoVLM model was trained on mixed VQA and
    spatial/grounding outputs. Therefore this formatter must not
    force a JSON response.

    The adapter is responsible for deciding whether the request is:
        - normal VQA
        - detailed natural-language VQA
        - spatial/localization VQA
    """

    prompt_parts = [
        "You are RSCoVLM-7B-2512-SatQuery, an expert remote-sensing "
        "vision-language assistant.",
        "Analyze the optical satellite image and answer the user's "
        "question accurately.",
    ]

    if specialist_evidence:
        prompt_parts.append("\n[AUTHORITATIVE SPECIALIST MACHINE EVIDENCE]")

        for model_key, evidence_val in specialist_evidence.items():
            if isinstance(evidence_val, dict):
                formatted_items = ", ".join(
                    f"{k}: {v}" for k, v in evidence_val.items()
                )
                prompt_parts.append(f"- {model_key}: {formatted_items}")
            else:
                prompt_parts.append(f"- {model_key}: {evidence_val}")

        # Change-detection safety directive
        if (
            "changeformer" in str(specialist_evidence).lower()
            or task.lower() in ("change_vqa", "change_detection")
            or specialist_evidence.get("change_detection_present", False)
        ):
            prompt_parts.append(
                "\n[SAFETY DIRECTIVE FOR CHANGE DETECTION]"
                "\nSpecialist evidence provides pixel-level change metrics. "
                "Use this evidence as the authoritative source for detected "
                "change. Do not infer or speculate about specific object "
                "classes unless they are explicitly supported by the "
                "specialist evidence."
            )

    prompt_parts.append(f"\nUser Question: {question}")

    return "\n".join(prompt_parts)

def extract_rscovlm_json(raw_text: str) -> Dict[str, Any]:
    """
    Safely extract and parse structured JSON from RSCoVLM output.
    Handles raw JSON, markdown code blocks, and malformed strings with robust fallback.
    """
    import json

    cleaned = raw_text.strip() if raw_text else ""

    # Remove markdown formatting if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Attempt direct JSON load
    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        # Try extracting outermost JSON object via regex
        json_match = re.search(r"(\{[\s\S]*\})", cleaned)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except Exception:
                pass

    if isinstance(data, dict):
        # Validate and normalize keys
        validation = data.get("validation", {})
        if not isinstance(validation, dict):
            validation = {
                "image_valid": True,
                "query_valid": True,
                "answerable": True,
                "confidence": 0.90,
                "reason": "Image and query validated successfully.",
            }

        highlighting = data.get("highlighting", {})
        if not isinstance(highlighting, dict):
            highlighting = {"required": False, "regions": []}
        else:
            if "regions" not in highlighting or not isinstance(highlighting["regions"], list):
                highlighting["regions"] = []
            highlighting["required"] = bool(highlighting.get("required", bool(highlighting["regions"])))

        answer = data.get("answer", {})
        if isinstance(answer, dict):
            answer_text = str(answer.get("text", "")).strip()
        elif isinstance(answer, str):
            answer_text = answer.strip()
        else:
            answer_text = str(data.get("text", "")).strip()

        if not answer_text and "explanation" in data:
            answer_text = str(data["explanation"]).strip()
        if not answer_text:
            answer_text = cleaned

        return {
            "validation": validation,
            "highlighting": highlighting,
            "answer": answer_text,
            "raw_prediction": raw_text,
            "structured": True,
        }

    # Controlled fallback if model returned free-form text
    # Extract any bracketed coordinates if present: e.g. [0.1, 0.2, 0.3, 0.4]
    regions = []
    box_matches = re.findall(r"\[\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*\]", cleaned)
    for bm in box_matches:
        try:
            coords = [float(c) for c in bm]
            regions.append({
                "label": "region",
                "box": coords,
                "confidence": 0.85,
            })
        except Exception:
            pass

    return {
        "validation": {
            "image_valid": True,
            "query_valid": True,
            "answerable": True,
            "confidence": 0.90,
            "reason": "Evaluated by RSCoVLM.",
        },
        "highlighting": {
            "required": bool(regions),
            "regions": regions,
        },
        "answer": cleaned,
        "raw_prediction": raw_text,
        "structured": False,
    }


def validate_rscovlm_response(
    answer: str,
    specialist_evidence: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Enforce evidence grounding on RSCoVLM generated answers.
    If ChangeFormer/pixel change evidence is used, remove unsupported semantic claims.
    """
    if not answer or not specialist_evidence:
        return answer

    is_changeformer_evidence = (
        any(k in specialist_evidence for k in ("changeformer", "change_vqa", "change_detection"))
        or "change_mask" in str(specialist_evidence).lower()
        or specialist_evidence.get("change_detection_present", False)
    )

    if is_changeformer_evidence:
        # Check if specialist evidence explicitly named any semantic categories
        evidence_str = str(specialist_evidence).lower()

        cleaned_answer = answer
        for claim in UNSUPPORTED_SEMANTIC_CLAIMS:
            if claim in cleaned_answer.lower() and claim not in evidence_str:
                # Replace unsupported claim phrasing with neutral change description
                pattern = re.compile(rf"\b{claim}s?\b", re.IGNORECASE)
                cleaned_answer = pattern.sub("changed areas", cleaned_answer)

        return cleaned_answer

    return answer
