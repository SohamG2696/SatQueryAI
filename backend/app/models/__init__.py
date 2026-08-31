"""
SatQuery AI — Model Adapters Package.

Exposes unified specialist adapters conforming to the standard contract:
    run_module(images, query, metadata) -> dict
"""

from .fusion import run_module as run_fusion
from .grounding import run_module as run_grounding
from .change_vqa import run_module as run_change_vqa
from .vqa import run_module as run_vqa
from .captioning import run_module as run_captioning

__all__ = [
    "run_fusion",
    "run_grounding",
    "run_change_vqa",
    "run_vqa",
    "run_captioning",
]
