from __future__ import annotations

import atexit
import logging
from pathlib import Path
import threading
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, FilterSelector

from ..embedding import embed_text, VECTOR_DIM
from ..paths import ensure_within_base, validate_collection_name
from ..types import Evidence, ChunkRecord, MarkdownBlockIR

logger = logging.getLogger(__name__)

# Thread-safe client cache with reference counting
_client_cache: dict[str, tuple[QdrantClient, int]] = {}
_cache_lock = threading.Lock()


def acquire_client(path: str) -> QdrantClient:
    global _client_cache
    with _cache_lock:
        if path not in _client_cache:
            client = QdrantClient(path=path)
            _client_cache[path] = (client, 1)
        else:
            client, count = _client_cache[path]
            _client_cache[path] = (client, count + 1)
        return client


def release_client(path: str) -> None:
    global _client_cache
    with _cache_lock:
        if path in _client_cache:
            client, count = _client_cache[path]
            if count <= 1:
                try:
                    client.close()
                except Exception as exc:
                    logger.debug("Error closing client during release: %s", exc)
                _client_cache.pop(path, None)
            else:
                _client_cache[path] = (client, count - 1)


@atexit.register
def cleanup_all_clients() -> None:
    global _client_cache
    with _cache_lock:
        for path, (client, _) in list(_client_cache.items()):
            try:
                client.close()
            except Exception as exc:
                logger.debug("Error closing client during atexit cleanup: %s", exc)
        _client_cache.clear()


class VectorStore:
    def __init__(self, base_dir: Path, collection: str) -> None:
        self.collection = validate_collection_name(collection)
        root = ensure_within_base(base_dir, base_dir / "qdrant")
        self.base_dir = ensure_within_base(root, root / self.collection)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize/Acquire client from local disk cache
        self.client_path = str(self.base_dir)
        self.client = acquire_client(self.client_path)
        self.backend = "qdrant"
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(collection_name=self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

    def reset(self) -> None:
        if self.client.collection_exists(collection_name=self.collection):
            self.client.delete(
                collection_name=self.collection,
                points_selector=FilterSelector(filter=Filter()),
            )
        else:
            self._ensure_collection()

    def upsert_chunks(self, chunks: list[ChunkRecord | MarkdownBlockIR]) -> None:
        points = []
        for chunk in chunks:
            if isinstance(chunk, MarkdownBlockIR):
                chunk_id = chunk.block_id
                doc_id = chunk.block_id  # fallback
                source_path = chunk.source_path
                text = chunk.content
                payload = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "source_path": source_path,
                    "text": text,
                    "type": "block",
                    "block_type": chunk.block_type,
                    "headers": chunk.headers,
                    "tags": chunk.tags,
                    "links": chunk.links,
                }
            else:
                chunk_id = chunk.chunk_id
                doc_id = chunk.doc_id
                source_path = chunk.source_path
                text = chunk.text
                payload = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "source_path": source_path,
                    "text": text,
                    "type": "chunk",
                    **chunk.metadata,
                }
            
            vector = embed_text(text)
            
            # Create a deterministic UUID/int ID from the chunk_id string
            import uuid
            point_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)
            
            points.append(
                PointStruct(
                    id=str(point_uuid),
                    vector=vector,
                    payload=payload,
                )
            )
        
        if points:
            self.client.upsert(
                collection_name=self.collection,
                wait=True,
                points=points,
            )

    def delete_file(self, source_path: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_path",
                            match=MatchValue(value=source_path),
                        )
                    ]
                )
            ),
        )

    def search(self, query: str, top_k: int) -> list[Evidence]:
        query_vector = embed_text(query)
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
        )
        evidences = []
        for hit in response.points:
            payload = hit.payload or {}
            evidences.append(
                Evidence(
                    chunk_id=payload.get("chunk_id", ""),
                    doc_id=payload.get("doc_id", ""),
                    source_path=payload.get("source_path", ""),
                    text=payload.get("text", ""),
                    score=round(hit.score, 4),
                    temporal_status="unknown",
                    facts=[],
                )
            )
        return evidences

    def count(self) -> int:
        info = self.client.get_collection(collection_name=self.collection)
        return info.points_count or 0

    def close(self) -> None:
        if hasattr(self, "client_path"):
            release_client(self.client_path)

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

