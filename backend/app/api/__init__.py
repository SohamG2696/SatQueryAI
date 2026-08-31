"""SatQuery AI — API Routers Package."""

from .history import router as history_router
from .query import router as query_router
from .upload import router as upload_router
from .analyze import router as analyze_router
from .vqa import router as vqa_router
from .caption import router as caption_router
from .grounding import router as grounding_router
from .change import router as change_router
from .fusion import router as fusion_router
from .agent import router as agent_router

all_routers = [
    query_router,
    upload_router,
    history_router,
    agent_router,
    analyze_router,
    vqa_router,
    caption_router,
    grounding_router,
    change_router,
    fusion_router,
]

__all__ = [
    "query_router",
    "upload_router",
    "history_router",
    "agent_router",
    "analyze_router",
    "vqa_router",
    "caption_router",
    "grounding_router",
    "change_router",
    "fusion_router",
    "all_routers",
]
