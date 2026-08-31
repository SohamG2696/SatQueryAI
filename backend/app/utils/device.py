"""
SatQuery AI — Compute Device Utility.

Handles hardware acceleration selection with strict priority:
    1. Intel Arc XPU (torch.xpu)
    2. NVIDIA CUDA (torch.cuda)
    3. CPU fallback
"""

from __future__ import annotations

import torch

from app.config import settings


def get_device(preference: str | None = None) -> torch.device:
    """Resolve and return the appropriate PyTorch device.

    Parameters
    ----------
    preference : str | None
        Optional explicit device string (e.g. 'xpu', 'cuda', 'cpu').
        If None, uses settings.device or hardware detection.

    Returns
    -------
    torch.device
        The initialized device object.
    """
    target = (preference or settings.device or "").lower()

    if target == "xpu" or (not target and hasattr(torch, "xpu")):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")

    if target == "cuda" or (not target and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")

    # Fallback to general detection
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def sync_device(device: torch.device | None = None) -> None:
    """Synchronize compute device streams before timing measurements."""
    dev = device or get_device()
    if dev.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    elif dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def get_device_info() -> dict[str, str | bool]:
    """Return factual device information for health checks and status APIs."""
    device = get_device()
    info: dict[str, str | bool] = {
        "active_device": str(device),
        "xpu_available": bool(hasattr(torch, "xpu") and torch.xpu.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_type": device.type,
    }

    if device.type == "xpu":
        info["device_name"] = "Intel Arc Graphics"
    elif device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(0)
    else:
        info["device_name"] = "CPU"

    return info
