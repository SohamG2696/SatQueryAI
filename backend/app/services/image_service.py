"""
SatQuery AI — Image Ingestion and Tensor Preprocessing Service.

Supports:
- GeoTIFF / TIFF preservation of spatial CRS, bounds, and transform via Rasterio
- PNG, JPG, JPEG, NPZ array loading via PIL / NumPy
- Tensor conversion for Optical (4 channels) and SAR (2 channels)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import rasterio

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def load_image_array(
    source: str | Path | bytes | io.BytesIO | np.ndarray | Image.Image,
) -> Tuple[np.ndarray, dict[str, Any]]:
    """Load image data into a NumPy array [C, H, W] along with geospatial metadata.

    Parameters
    ----------
    source : str | Path | bytes | io.BytesIO | np.ndarray | Image.Image
        Path to file, bytes in memory, raw array, or PIL Image.

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

    # Case 2: Direct PIL Image instance
    if isinstance(source, Image.Image):
        img_rgba = source.convert("RGBA") if source.mode == "RGBA" else source.convert("RGB")
        arr = np.array(img_rgba, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        else:
            arr = np.transpose(arr, (2, 0, 1))
        metadata.update({
            "width": source.width,
            "height": source.height,
            "bands": arr.shape[0],
            "format": "pil_image",
        })
        return arr, metadata

    # Case 3: File Path
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        suffix = path.suffix.lower()

        # NumPy npz pair
        if suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            if "s2" in data and "s1" in data:
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

        # Standard image via PIL
        with Image.open(path) as img:
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
                "format": suffix.lstrip("."),
            })
            return arr, metadata

    # Case 4: Bytes buffer
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


def _normalize_channel_2d(arr: np.ndarray) -> np.ndarray:
    """Apply 2%-98% percentile min-max normalization matching Fusion training pipeline."""
    data = arr.astype(np.float32)

    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    low = float(np.percentile(data, 2))
    high = float(np.percentile(data, 98))

    if high <= low:
        high = float(data.max())
        low = float(data.min())
        if high <= low:
            return np.zeros_like(data, dtype=np.float32)

    data = np.clip(data, low, high)
    data = (data - low) / (high - low)
    return data.astype(np.float32)


def prepare_optical_tensor(
    source: str | Path | bytes | io.BytesIO | np.ndarray | Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    device: torch.device | None = None,
) -> torch.Tensor:
    """Load and format Optical satellite data into [1, 4, H, W] float32 tensor matching Fusion training.

    Channel Order: [B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)]
    Normalization: 2%-98% Percentile Min-Max Scaling [0.0, 1.0]
    """
    if source is None:
        raise ValueError("Optical image source cannot be None.")

    arr, meta = load_image_array(source)

    if arr is None or arr.size == 0:
        raise ValueError("Optical image array is empty or corrupt.")

    if np.isnan(arr).any() or np.isinf(arr).any():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Handle NPZ fusion pair directly if s2 array exists
    if meta.get("format") == "npz_fusion_pair" and "s2" in meta:
        s2_arr = meta["s2"]
        if s2_arr.dtype == np.uint8 or s2_arr.max() > 1.0:
            stacked = (s2_arr.astype(np.float32) / 255.0)
        else:
            stacked = s2_arr.astype(np.float32)
        tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
        if tensor.shape[2:] != target_size:
            tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
        tensor = torch.clamp(tensor, 0.0, 1.0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    channels, h, w = arr.shape

    # Apply 2%-98% percentile normalization to each band
    norm_channels = [_normalize_channel_2d(arr[i]) for i in range(channels)]

    if channels >= 4:
        if meta.get("format") in ("pil_image", "png", "jpg", "jpeg"):
            # Standard PIL Image RGBA: ch0=Red, ch1=Green, ch2=Blue, ch3=Alpha/NIR
            b02 = norm_channels[2]  # Blue
            b03 = norm_channels[1]  # Green
            b04 = norm_channels[0]  # Red
            if np.allclose(norm_channels[3], norm_channels[3][0, 0]):
                b08 = (b04 * 0.5 + b03 * 0.5)
            else:
                b08 = norm_channels[3]  # NIR
            stacked = np.stack([b02, b03, b04, b08])
        else:
            # Multi-band GeoTIFF or S2 stack [B02, B03, B04, B08]
            stacked = np.stack([norm_channels[0], norm_channels[1], norm_channels[2], norm_channels[3]])
    elif channels == 3:
        # Standard RGB image [Red, Green, Blue] -> Map to [B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)]
        b04 = norm_channels[0]  # Red
        b03 = norm_channels[1]  # Green
        b02 = norm_channels[2]  # Blue
        b08 = (b04 * 0.5 + b03 * 0.5)  # Synthetic NIR
        stacked = np.stack([b02, b03, b04, b08])
    elif channels == 2:
        stacked = np.stack([norm_channels[0], norm_channels[1], norm_channels[0], norm_channels[1]])
    elif channels == 1:
        ch = norm_channels[0]
        stacked = np.stack([ch, ch, ch, ch])
    else:
        raise ValueError(f"Invalid channel count for Optical image: {channels}")

    tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)

    if (h, w) != target_size:
        tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)

    tensor = torch.clamp(tensor, 0.0, 1.0)
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def prepare_sar_tensor(
    source: str | Path | bytes | io.BytesIO | np.ndarray | Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    device: torch.device | None = None,
) -> torch.Tensor:
    """Load and format SAR satellite data into [1, 2, H, W] float32 tensor matching Fusion training.

    Channel Order: [VV, VH]
    Normalization: 2%-98% Percentile Min-Max Scaling [0.0, 1.0]
    """
    if source is None:
        raise ValueError("SAR image source cannot be None.")

    arr, meta = load_image_array(source)

    if arr is None or arr.size == 0:
        raise ValueError("SAR image array is empty or corrupt.")

    if np.isnan(arr).any() or np.isinf(arr).any():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Handle NPZ fusion pair directly if s1 array exists
    if meta.get("format") == "npz_fusion_pair" and "s1" in meta:
        s1_arr = meta["s1"]
        if s1_arr.dtype == np.uint8 or s1_arr.max() > 1.0:
            stacked = (s1_arr.astype(np.float32) / 255.0)
        else:
            stacked = s1_arr.astype(np.float32)
        tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
        if tensor.shape[2:] != target_size:
            tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
        tensor = torch.clamp(tensor, 0.0, 1.0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    channels, h, w = arr.shape

    # Apply 2%-98% percentile normalization to each band
    norm_channels = [_normalize_channel_2d(arr[i]) for i in range(channels)]

    if channels >= 2:
        # Expected [VV, VH]
        stacked = np.stack([norm_channels[0], norm_channels[1]])
    elif channels == 1:
        # Single SAR polarization band (e.g. VV) -> duplicate across both bands
        ch = norm_channels[0]
        stacked = np.stack([ch, ch])
    else:
        raise ValueError(f"Invalid channel count for SAR image: {channels}")

    tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)

    if (h, w) != target_size:
        tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)

    tensor = torch.clamp(tensor, 0.0, 1.0)
    if device is not None:
        tensor = tensor.to(device)

    return tensor
