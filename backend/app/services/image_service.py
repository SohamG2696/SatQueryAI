"""
SatQuery AI — Image Loading & Preprocessing Service.

Supports:
- GeoTIFF / TIFF reading via Rasterio (preserving geospatial metadata, CRS, bounds, transform)
- Standard image formats (PNG, JPG, JPEG) via Pillow
- NumPy archives (.npz) used in remote sensing multimodal pairs
- Channel extraction, spatial resizing, and tensor normalization for ML inference
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def load_image_array(
    source: str | Path | bytes | io.BytesIO | np.ndarray,
) -> Tuple[np.ndarray, dict[str, Any]]:
    """Load image data into a NumPy array [C, H, W] along with geospatial metadata.

    Parameters
    ----------
    source : str | Path | bytes | io.BytesIO | np.ndarray
        Path to file, bytes in memory, or raw array.

    Returns
    -------
    Tuple[np.ndarray, dict[str, Any]]
        - Array of shape [channels, height, width], float32.
        - Metadata dictionary containing width, height, bands, crs, bounds, transform.
    """
    metadata: dict[str, Any] = {
        "width": 0,
        "height": 0,
        "bands": 0,
        "crs": None,
        "bounds": None,
        "transform": None,
        "format": "unknown",
    }

    # Case 1: Already a numpy array
    if isinstance(source, np.ndarray):
        arr = source.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[2] in (1, 2, 3, 4, 12) and arr.shape[0] not in (1, 2, 3, 4, 12):
            arr = np.transpose(arr, (2, 0, 1))
        metadata["bands"] = arr.shape[0]
        metadata["height"] = arr.shape[1]
        metadata["width"] = arr.shape[2]
        metadata["format"] = "numpy"
        return arr, metadata

    # Case 2: File Path
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        suffix = path.suffix.lower()

        # NumPy npz pair
        if suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            if "s2" in data and "s1" in data:
                # Store both in metadata dict
                s2 = data["s2"].astype(np.float32)
                s1 = data["s1"].astype(np.float32)
                metadata["format"] = "npz_fusion_pair"
                metadata["s1"] = s1
                metadata["s2"] = s2
                metadata["bands"] = s2.shape[0]
                metadata["height"] = s2.shape[1]
                metadata["width"] = s2.shape[2]
                return s2, metadata
            elif "image" in data:
                arr = data["image"].astype(np.float32)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                metadata["bands"] = arr.shape[0]
                metadata["height"] = arr.shape[1]
                metadata["width"] = arr.shape[2]
                metadata["format"] = "npz"
                return arr, metadata

        # GeoTIFF via Rasterio
        if suffix in (".tif", ".tiff") and RASTERIO_AVAILABLE:
            try:
                with rasterio.open(path) as src:
                    arr = src.read().astype(np.float32)
                    metadata.update({
                        "width": src.width,
                        "height": src.height,
                        "bands": src.count,
                        "crs": str(src.crs) if src.crs else None,
                        "bounds": list(src.bounds) if src.bounds else None,
                        "transform": [float(x) for x in src.transform] if src.transform else None,
                        "format": "geotiff",
                    })
                    return arr, metadata
            except Exception:
                pass  # fallback to PIL below

        # Standard image (or fallback) via PIL
        with Image.open(path) as img:
            img_rgba = img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB")
            arr = np.array(img_rgba, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            else:
                arr = np.transpose(arr, (2, 0, 1))  # [H, W, C] -> [C, H, W]

            metadata.update({
                "width": img.width,
                "height": img.height,
                "bands": arr.shape[0],
                "format": suffix.lstrip("."),
            })
            return arr, metadata

    # Case 3: Bytes buffer
    if isinstance(source, (bytes, io.BytesIO)):
        buf = io.BytesIO(source) if isinstance(source, bytes) else source
        buf.seek(0)

        # Try Rasterio MemoryFile first if geotiff
        if RASTERIO_AVAILABLE:
            try:
                from rasterio.io import MemoryFile
                with MemoryFile(buf.getvalue()) as memfile:
                    with memfile.open() as src:
                        arr = src.read().astype(np.float32)
                        metadata.update({
                            "width": src.width,
                            "height": src.height,
                            "bands": src.count,
                            "crs": str(src.crs) if src.crs else None,
                            "bounds": list(src.bounds) if src.bounds else None,
                            "transform": [float(x) for x in src.transform] if src.transform else None,
                            "format": "geotiff_memory",
                        })
                        return arr, metadata
            except Exception:
                buf.seek(0)

        # Fallback to PIL
        with Image.open(buf) as img:
            img_rgba = img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB")
            arr = np.array(img_rgba, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            else:
                arr = np.transpose(arr, (2, 0, 1))

            metadata.update({
                "width": img.width,
                "height": img.height,
                "bands": arr.shape[0],
                "format": "pillow_memory",
            })
            return arr, metadata

    raise ValueError(f"Unsupported image source type: {type(source)}")


def prepare_optical_tensor(
    source: str | Path | bytes | io.BytesIO | np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    device: torch.device | None = None,
) -> torch.Tensor:
    """Load and format an optical image into [1, 4, H, W] normalized float tensor [0, 1].

    If input has 3 channels (RGB), adds a simulated 4th NIR channel or duplicates.
    If input has >4 channels (e.g. 12-band Sentinel-2), selects standard B02, B03, B04, B08.
    """
    arr, _ = load_image_array(source)

    # Normalize max range
    if arr.max() > 1.0:
        arr = arr / 255.0

    channels, h, w = arr.shape

    # Channel adjustment to 4 channels
    if channels == 1:
        arr = np.repeat(arr, 4, axis=0)
    elif channels == 2:
        arr = np.concatenate([arr, arr], axis=0)
    elif channels == 3:
        # Create NIR approximation from Red/Green average
        nir = (arr[0:1] * 0.5 + arr[1:2] * 0.5)
        arr = np.concatenate([arr, nir], axis=0)
    elif channels >= 4:
        arr = arr[:4]

    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # [1, 4, H, W]

    # Spatial resize to target_size (e.g. 224x224)
    if (h, w) != target_size:
        tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)

    tensor = torch.clamp(tensor, 0.0, 1.0)
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def prepare_sar_tensor(
    source: str | Path | bytes | io.BytesIO | np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    device: torch.device | None = None,
) -> torch.Tensor:
    """Load and format a SAR image into [1, 2, H, W] normalized float tensor [0, 1].

    If input has 1 channel (e.g. VV), duplicates to 2 channels [VH, VV].
    If input has >=2 channels, takes first 2 channels.
    """
    arr, _ = load_image_array(source)

    if arr.max() > 1.0:
        arr = arr / 255.0

    channels, h, w = arr.shape

    if channels == 1:
        arr = np.concatenate([arr, arr], axis=0)
    elif channels >= 2:
        arr = arr[:2]

    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # [1, 2, H, W]

    if (h, w) != target_size:
        tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)

    tensor = torch.clamp(tensor, 0.0, 1.0)
    if device is not None:
        tensor = tensor.to(device)

    return tensor
