"""
SatQuery AI — RSCoVLM Python Highlighting Service.

Receives satellite images and structured spatial regions identified by RSCoVLM.
Validates, normalizes, and clamps coordinates to image dimensions.
Renders crisp bounding boxes and label badges using pure PIL/Pillow.
Saves highlighted images to the configured temporary upload directory.
"""

from __future__ import annotations

import base64
from io import BytesIO
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from PIL import Image, ImageDraw, ImageFont

from app.config import settings

logger = logging.getLogger("satquery.rscovlm.highlight")

# High-contrast color palette for clear remote sensing visual grounding
PALETTE_RGB = [
    (0, 255, 127),    # Neon Spring Green
    (0, 229, 255),    # Electric Cyan
    (255, 215, 0),    # Golden Yellow
    (255, 64, 129),   # Vivid Pink / Magenta
    (255, 112, 67),   # Coral Orange
    (171, 71, 188),   # Lavender Purple
]


def _parse_box_coordinates(
    box: Any,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[int, int, int, int, Tuple[float, float, float, float]]]:
    """
    Parse, convert, and clamp a bounding box into pixel coordinates (px1, py1, px2, py2)
    and normalized float coordinates (nx1, ny1, nx2, ny2).
    
    Supports:
    - [x1, y1, x2, y2]
    - [ymin, xmin, ymax, xmax] if explicitly tagged
    - {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
    - 0.0 -> 1.0 normalized float
    - 0 -> 1000 normalized integer (Qwen convention)
    - Direct pixel coordinates
    """
    if not box:
        return None

    raw_coords = None
    if isinstance(box, (list, tuple)):
        if len(box) >= 4:
            raw_coords = [float(c) for c in box[:4]]
    elif isinstance(box, dict):
        if all(k in box for k in ("x1", "y1", "x2", "y2")):
            raw_coords = [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]
        elif all(k in box for k in ("ymin", "xmin", "ymax", "xmax")):
            raw_coords = [float(box["xmin"]), float(box["ymin"]), float(box["xmax"]), float(box["ymax"])]
        elif "box_2d" in box and len(box["box_2d"]) >= 4:
            raw_coords = [float(c) for c in box["box_2d"][:4]]
        elif "box" in box and len(box["box"]) >= 4:
            raw_coords = [float(c) for c in box["box"][:4]]

    if not raw_coords:
        return None

    c1, c2, c3, c4 = raw_coords

    # Determine coordinate space
    max_val = max(c1, c2, c3, c4)
    min_val = min(c1, c2, c3, c4)

    if max_val <= 2.0 and min_val >= -1.0:
        # Case A: 0.0 to 1.0 normalized (with minor model float boundary overshoot)
        nx1, ny1, nx2, ny2 = c1, c2, c3, c4
        px1 = int(round(nx1 * img_w))
        py1 = int(round(ny1 * img_h))
        px2 = int(round(nx2 * img_w))
        py2 = int(round(ny2 * img_h))
    elif max_val <= 1000.0 and (img_w > 1000 or img_h > 1000 or (c1 <= 1000 and c3 <= 1000) or max_val > 2.0):
        # Case B: 0 to 1000 normalized space (common Qwen2.5-VL box tokens)
        nx1 = c1 / 1000.0
        ny1 = c2 / 1000.0
        nx2 = c3 / 1000.0
        ny2 = c4 / 1000.0
        px1 = int(round(nx1 * img_w))
        py1 = int(round(ny1 * img_h))
        px2 = int(round(nx2 * img_w))
        py2 = int(round(ny2 * img_h))
    else:
        # Case C: Pixel coordinates
        px1 = int(round(c1))
        py1 = int(round(c2))
        px2 = int(round(c3))
        py2 = int(round(c4))
        nx1 = px1 / float(img_w) if img_w > 0 else 0.0
        ny1 = py1 / float(img_h) if img_h > 0 else 0.0
        nx2 = px2 / float(img_w) if img_w > 0 else 1.0
        ny2 = py2 / float(img_h) if img_h > 0 else 1.0

    # Ensure correct ordering: x1 < x2, y1 < y2
    if px1 > px2:
        px1, px2 = px2, px1
        nx1, nx2 = nx2, nx1
    if py1 > py2:
        py1, py2 = py2, py1
        ny1, ny2 = ny2, ny1

    # Clamp pixel coordinates to image boundary
    px1 = max(0, min(px1, img_w - 1))
    py1 = max(0, min(py1, img_h - 1))
    px2 = max(px1 + 1, min(px2, img_w))
    py2 = max(py1 + 1, min(py2, img_h))

    # Clamp normalized coordinates
    nx1 = round(max(0.0, min(nx1, 1.0)), 4)
    ny1 = round(max(0.0, min(ny1, 1.0)), 4)
    nx2 = round(max(nx1 + 0.001, min(nx2, 1.0)), 4)
    ny2 = round(max(ny1 + 0.001, min(ny2, 1.0)), 4)

    # Filter degenerated boxes
    if px2 - px1 < 2 and py2 - py1 < 2:
        return None

    return (px1, py1, px2, py2, (nx1, ny1, nx2, ny2))


def highlight_image_regions(
    image: Union[str, Path, Image.Image],
    regions: List[Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    save_format: str = "JPEG",
) -> Dict[str, Any]:
    """
    Render visual highlights on the satellite image based on RSCoVLM regions.

    Parameters
    ----------
    image : Union[str, Path, Image.Image]
        Source satellite image (file path or PIL Image).
    regions : List[Dict[str, Any]]
        List of region dictionaries:
        [{'label': 'runway', 'box': [x1, y1, x2, y2], 'confidence': 0.88}, ...]
    output_dir : Optional[Union[str, Path]]
        Directory to save highlighted output image.
    save_format : str
        Format to save ('JPEG' or 'PNG').

    Returns
    -------
    Dict[str, Any]
        Structured result containing:
        - highlighted_image_path
        - highlighted_image_filename
        - highlighted_image_base64
        - regions (normalized and validated)
        - region_count
        - width, height
    """
    # 1. Load source image safely
    if isinstance(image, Image.Image):
        src_img = image.convert("RGB")
    elif isinstance(image, (str, Path)):
        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f"Source image file not found for highlighting: {p}")
        src_img = Image.open(p).convert("RGB")
    else:
        raise ValueError(f"Unsupported image type for highlighting: {type(image)}")

    img_w, img_h = src_img.size

    # Create a copy so original image remains untouched
    canvas = src_img.copy()
    draw = ImageDraw.Draw(canvas)

    # Dynamic line thickness based on image resolution
    thickness = max(2, min(8, int(min(img_w, img_h) / 180)))

    validated_regions = []
    rendered_count = 0

    # Try loading default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, reg in enumerate(regions or []):
        raw_box = reg.get("box", reg.get("bbox", reg.get("coordinates")))
        parsed = _parse_box_coordinates(raw_box, img_w, img_h)
        if not parsed:
            continue

        px1, py1, px2, py2, (nx1, ny1, nx2, ny2) = parsed
        color = PALETTE_RGB[idx % len(PALETTE_RGB)]
        label = str(reg.get("label", "region")).strip()
        confidence = float(reg.get("confidence", 0.90))

        # Draw multi-pixel thickness border
        for t in range(thickness):
            draw.rectangle(
                [px1 - t, py1 - t, px2 + t, py2 + t],
                outline=color,
            )

        # Draw label pill badge
        label_text = f"{label} ({int(confidence * 100)}%)" if confidence else label
        if font:
            try:
                bbox_text = draw.textbbox((px1, py1), label_text, font=font)
                text_w = bbox_text[2] - bbox_text[0]
                text_h = bbox_text[3] - bbox_text[1]
            except Exception:
                text_w = len(label_text) * 7
                text_h = 12

            pad = 3
            pill_y1 = max(0, py1 - text_h - pad * 2)
            pill_y2 = pill_y1 + text_h + pad * 2
            pill_x2 = min(img_w, px1 + text_w + pad * 2)

            # Draw label background
            draw.rectangle([px1, pill_y1, pill_x2, pill_y2], fill=(20, 20, 20))
            draw.rectangle([px1, pill_y1, pill_x2, pill_y2], outline=color, width=1)
            # Draw label text
            draw.text((px1 + pad, pill_y1 + pad), label_text, fill=color, font=font)

        validated_regions.append({
            "label": label,
            "box": [nx1, ny1, nx2, ny2],
            "pixel_box": [px1, py1, px2, py2],
            "confidence": round(confidence, 4),
        })
        rendered_count += 1

    # 2. Determine target save directory
    if output_dir:
        dest_dir = Path(output_dir)
    else:
        dest_dir = settings.upload_path / "temporary"
    dest_dir.mkdir(parents=True, exist_ok=True)

    out_filename = f"highlight_{int(time.time())}_{uuid.uuid4().hex[:8]}.{save_format.lower()}"
    out_path = dest_dir / out_filename

    # Save highlighted image
    if save_format.upper() in ("JPG", "JPEG"):
        canvas.save(out_path, format="JPEG", quality=92)
    else:
        canvas.save(out_path, format="PNG")

    # Generate Base64 preview string
    buffer = BytesIO()
    canvas.save(buffer, format="JPEG", quality=90)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "highlighted_image_path": str(out_path.resolve()),
        "highlighted_image_filename": out_filename,
        "highlighted_image_base64": b64_str,
        "regions": validated_regions,
        "region_count": rendered_count,
        "width": img_w,
        "height": img_h,
    }
