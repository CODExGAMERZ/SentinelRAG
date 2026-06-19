from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_store import GraphStore


class EntityGraph:
    def __init__(self, graph_store: GraphStore) -> None:
        self.graph_store = graph_store

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source_block_id: str,
        source_path: str,
        confidence: float = 1.0,
    ) -> None:
        with self.graph_store.get_conn() as conn:
            conn.execute("""
                INSERT INTO triples (subject, predicate, object, source_block_id, source_path, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (subject, predicate, obj, source_block_id, source_path, confidence))
            conn.commit()

    def delete_triples_for_note(self, source_path: str) -> None:
        with self.graph_store.get_conn() as conn:
            conn.execute("DELETE FROM triples WHERE source_path = ?", (source_path,))
            conn.commit()

    def search_triples(self, entity: str) -> list[dict[str, Any]]:
        """Finds triples where subject or object matches the entity (case-insensitive)."""
        with self.graph_store.get_conn() as conn:
            cursor = conn.execute("""
                SELECT * FROM triples 
                WHERE LOWER(subject) = ? OR LOWER(object) = ?
            """, (entity.lower(), entity.lower()))
            return [dict(row) for row in cursor]

    def get_all_triples(self) -> list[dict[str, Any]]:
        with self.graph_store.get_conn() as conn:
            cursor = conn.execute("SELECT * FROM triples")
            return [dict(row) for row in cursor]
