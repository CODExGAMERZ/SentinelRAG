from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    doc_id: str
    chunk_id: str
    source_path: str
    text: str
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class Evidence:
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    score: float
    temporal_status: Literal["current", "stale", "unknown"] = "unknown"
    facts: list[str] = field(default_factory=list)
