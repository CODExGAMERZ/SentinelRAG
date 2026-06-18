from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
from pathlib import Path

from .embedding import cosine_similarity, embed_text
from .paths import ensure_within_base, validate_collection_name
from .types import ChunkRecord, Evidence


class VectorStore:
    def __init__(self, base_dir: Path, collection: str) -> None:
        self.collection = validate_collection_name(collection)
        root = ensure_within_base(base_dir, base_dir / "qdrant")
        self.base_dir = ensure_within_base(root, root / self.collection)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / "vectors.json"
        self.backend = "json"
        self._records: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        raw = self.path.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            backup = self.path.with_suffix(f".json.corrupt.{uuid.uuid4().hex}")
            self.path.replace(backup)
            return []
        return data

    def _save(self) -> None:
        _atomic_write_json(self.path, self._records)

    def reset(self) -> None:
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._records = []
        self._save()

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        existing = {record["chunk"]["chunk_id"]: record for record in self._records}
        for chunk in chunks:
            existing[chunk.chunk_id] = {
                "chunk": {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "source_path": chunk.source_path,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "created_at": chunk.created_at,
                },
                "vector": embed_text(chunk.text),
            }
        self._records = list(existing.values())
        self._save()

    def search(self, query: str, top_k: int) -> list[Evidence]:
        query_vector = embed_text(query)
        scored: list[tuple[float, dict]] = []
        for record in self._records:
            score = cosine_similarity(query_vector, record["vector"])
            scored.append((score, record["chunk"]))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            Evidence(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                source_path=chunk["source_path"],
                text=chunk["text"],
                score=round(score, 4),
                temporal_status="unknown",
            )
            for score, chunk in scored[:top_k]
            if score > 0
        ]

    def count(self) -> int:
        return len(self._records)


ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]{2,}(?:\s+[A-Z][A-Za-z0-9_]{2,}){0,3}\b")


class GraphMemory:
    def __init__(self, base_dir: Path, collection: str) -> None:
        self.collection = validate_collection_name(collection)
        root = ensure_within_base(base_dir, base_dir / "falkor")
        self.base_dir = ensure_within_base(root, root / self.collection)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / "graph.json"
        self.backend = "json"
        self._graph = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"entities": {}, "facts": []}
        raw = self.path.read_text(encoding="utf-8")
        if not raw.strip():
            return {"entities": {}, "facts": []}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            backup = self.path.with_suffix(f".json.corrupt.{uuid.uuid4().hex}")
            self.path.replace(backup)
            return {"entities": {}, "facts": []}
        if not isinstance(data, dict):
            return {"entities": {}, "facts": []}
        data.setdefault("entities", {})
        data.setdefault("facts", [])
        return data

    def _save(self) -> None:
        _atomic_write_json(self.path, self._graph)

    def reset(self) -> None:
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._graph = {"entities": {}, "facts": []}
        self._save()

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        existing_facts = {(fact["chunk_id"], fact["entity"]) for fact in self._graph["facts"]}
        for chunk in chunks:
            for entity in sorted(set(ENTITY_RE.findall(chunk.text))):
                self._graph["entities"].setdefault(entity, {"name": entity, "mentions": 0})
                self._graph["entities"][entity]["mentions"] += 1
                key = (chunk.chunk_id, entity)
                if key not in existing_facts:
                    self._graph["facts"].append(
                        {
                            "entity": entity,
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "source_path": chunk.source_path,
                            "valid_from": chunk.created_at,
                            "valid_to": None,
                            "temporal_status": "current",
                        }
                    )
                    existing_facts.add(key)
        self._save()

    def expand_evidence(self, evidence: list[Evidence]) -> list[Evidence]:
        facts_by_chunk: dict[str, list[str]] = {}
        status_by_chunk: dict[str, str] = {}
        for fact in self._graph["facts"]:
            facts_by_chunk.setdefault(fact["chunk_id"], []).append(fact["entity"])
            status_by_chunk[fact["chunk_id"]] = fact.get("temporal_status", "unknown")

        expanded: list[Evidence] = []
        for item in evidence:
            expanded.append(
                Evidence(
                    chunk_id=item.chunk_id,
                    doc_id=item.doc_id,
                    source_path=item.source_path,
                    text=item.text,
                    score=item.score,
                    temporal_status=status_by_chunk.get(item.chunk_id, item.temporal_status),  # type: ignore[arg-type]
                    facts=sorted(set(facts_by_chunk.get(item.chunk_id, [])))[:12],
                )
            )
        return expanded

    def count_facts(self) -> int:
        return len(self._graph["facts"])


def _atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(payload)
        handle.flush()
        temp_path = Path(handle.name)
    temp_path.replace(path)
