"""
SatQuery AI — Lightweight In-Memory Inference Cache.

Caches specialist model inference results based on:
1. Multi-modal image content hashes (SHA-256)
2. Canonical intent / task representation
3. Canonical target / prompt / spatial parameters

Prevents redundant specialist model executions on semantically equivalent queries.
"""

from __future__ import annotations

import hashlib
import io
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from app.schemas.response import QueryResponse


def _hash_image_source(image_src: Any) -> str:
    """Compute deterministic SHA-256 fingerprint for diverse image source types."""
    hasher = hashlib.sha256()

    if isinstance(image_src, (bytes, bytearray)):
        hasher.update(image_src)
    elif isinstance(image_src, (str, Path)):
        p = Path(image_src)
        if p.exists() and p.is_file():
            # Hash file content
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
        else:
            hasher.update(str(image_src).encode("utf-8"))
    elif isinstance(image_src, Image.Image):
        # Hash image raw bytes
        hasher.update(image_src.tobytes())
        hasher.update(str(image_src.size).encode("utf-8"))
        hasher.update(str(image_src.mode).encode("utf-8"))
    elif isinstance(image_src, np.ndarray):
        hasher.update(image_src.tobytes())
        hasher.update(str(image_src.shape).encode("utf-8"))
        hasher.update(str(image_src.dtype).encode("utf-8"))
    elif isinstance(image_src, io.BytesIO):
        pos = image_src.tell()
        image_src.seek(0)
        hasher.update(image_src.read())
        image_src.seek(pos)
    else:
        hasher.update(str(image_src).encode("utf-8"))

    return hasher.hexdigest()[:32]


class InferenceCache:
    """Thread-safe LRU cache for query responses."""

    def __init__(self, maxsize: int = 500):
        self._maxsize = maxsize
        self._cache: OrderedDict[Tuple[Any, ...], QueryResponse] = OrderedDict()
        self._lock = threading.Lock()

    def _build_key(
        self,
        images: List[Any],
        canonical_intent: str,
        canonical_target_or_prompt: str,
        extra_key: Optional[str] = None,
    ) -> Tuple[Any, ...]:
        """Generate structured cache key tuple."""
        img_hashes = tuple(_hash_image_source(img) for img in images) if images else ("no_image",)
        norm_intent = str(canonical_intent).strip().lower()
        norm_target = str(canonical_target_or_prompt).strip().lower()
        return (img_hashes, norm_intent, norm_target, extra_key or "")

    def get(
        self,
        images: List[Any],
        canonical_intent: str,
        canonical_target_or_prompt: str,
        extra_key: Optional[str] = None,
    ) -> Optional[QueryResponse]:
        """Retrieve cached QueryResponse if available."""
        key = self._build_key(images, canonical_intent, canonical_target_or_prompt, extra_key)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                cached = self._cache[key]
                # Return deep copy or replica with cached flag
                response_copy = cached.model_copy(deep=True)
                if response_copy.execution_summary and response_copy.execution_summary.parameters:
                    response_copy.execution_summary.parameters["cached"] = True
                return response_copy
        return None

    def put(
        self,
        images: List[Any],
        canonical_intent: str,
        canonical_target_or_prompt: str,
        response: QueryResponse,
        extra_key: Optional[str] = None,
    ) -> None:
        """Store QueryResponse in LRU cache."""
        key = self._build_key(images, canonical_intent, canonical_target_or_prompt, extra_key)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = response.model_copy(deep=True)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all entries in the cache."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get number of cached entries."""
        with self._lock:
            return len(self._cache)


# Singleton cache instance
inference_cache = InferenceCache(maxsize=500)
