"""
SatQuery AI — History Endpoint.

Provides session-based query history storage and retrieval.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api", tags=["History"])

# In-memory history store keyed by session_id
_HISTORY_STORE: Dict[str, List[Dict[str, Any]]] = {}


def record_history(
    session_id: str,
    query: str,
    task_detected: str,
    answer: str,
    confidence: float | None = None,
    visual_evidence: dict | None = None,
) -> None:
    """Record a completed query transaction into the session history."""
    if not session_id:
        return

    if session_id not in _HISTORY_STORE:
        _HISTORY_STORE[session_id] = []

    _HISTORY_STORE[session_id].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "task_detected": task_detected,
        "answer": answer,
        "confidence": confidence,
        "visual_evidence": visual_evidence,
    })


@router.get("/history/{session_id}", response_model=List[Dict[str, Any]])
async def get_session_history(session_id: str) -> List[Dict[str, Any]]:
    """Retrieve historical queries and responses for a specific session."""
    return _HISTORY_STORE.get(session_id, [])


@router.delete("/history/{session_id}")
async def clear_session_history(session_id: str) -> dict[str, str]:
    """Clear query history for a session."""
    if session_id in _HISTORY_STORE:
        del _HISTORY_STORE[session_id]
    return {"message": f"History for session '{session_id}' cleared."}
