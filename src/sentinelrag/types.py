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


@dataclass(frozen=True, slots=True)
class MarkdownBlockIR:
    block_id: str
    content_hash: str
    source_path: str
    block_type: str
    content: str
    headers: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    created_at: float | None = None
    updated_at: float | None = None


@dataclass(frozen=True, slots=True)
class MergedEvidence:
    content: str
    source_type: Literal["vector", "graph", "hybrid"]
    source_id: str
    semantic_score: float | None
    centrality_score: float | None
    recency_score: float
    final_score: float
    provenance: dict[str, Any] = field(default_factory=dict)
