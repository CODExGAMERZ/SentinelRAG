from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..types import MarkdownBlockIR


class GraphStore:
    def __init__(self, base_dir: Path, db_name: str = "sentinelrag.db") -> None:
        self.db_path = base_dir / db_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = "sqlite"
        self._init_db()

    def get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    path TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    centrality REAL DEFAULT 1.0,
                    is_evergreen INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    source TEXT,
                    target TEXT,
                    ambiguous_link INTEGER DEFAULT 0,
                    PRIMARY KEY (source, target),
                    FOREIGN KEY (source) REFERENCES nodes(path) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    block_id TEXT,
                    source_path TEXT,
                    content_hash TEXT NOT NULL,
                    block_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    headers TEXT NOT NULL, -- JSON string list
                    tags TEXT NOT NULL,    -- JSON string list
                    links TEXT NOT NULL,   -- JSON string list
                    created_at REAL,
                    updated_at REAL,
                    PRIMARY KEY (source_path, block_id),
                    FOREIGN KEY (source_path) REFERENCES nodes(path) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    source_block_id TEXT,
                    source_path TEXT,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (source_path) REFERENCES nodes(path) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def reset(self) -> None:
        with self.get_conn() as conn:
            conn.execute("DROP TABLE IF EXISTS triples")
            conn.execute("DROP TABLE IF EXISTS blocks")
            conn.execute("DROP TABLE IF EXISTS edges")
            conn.execute("DROP TABLE IF EXISTS nodes")
            conn.commit()
        self._init_db()

    def upsert_note(self, path: str, title: str, mtime: float, is_evergreen: bool = False) -> None:
        with self.get_conn() as conn:
            conn.execute("""
                INSERT INTO nodes (path, title, mtime, is_evergreen)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    title = excluded.title,
                    mtime = excluded.mtime,
                    is_evergreen = excluded.is_evergreen
            """, (path, title, mtime, 1 if is_evergreen else 0))
            conn.commit()

    def delete_note(self, path: str) -> None:
        with self.get_conn() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("DELETE FROM nodes WHERE path = ?", (path,))
            conn.commit()

    def get_note_mtimes(self) -> dict[str, float]:
        with self.get_conn() as conn:
            cursor = conn.execute("SELECT path, mtime FROM nodes")
            return {row["path"]: row["mtime"] for row in cursor}

    def get_all_notes(self) -> list[dict[str, Any]]:
        with self.get_conn() as conn:
            cursor = conn.execute("SELECT path, title, mtime, centrality, is_evergreen FROM nodes")
            return [dict(row) for row in cursor]

    def get_note_by_title(self, title: str) -> list[dict[str, Any]]:
        with self.get_conn() as conn:
            cursor = conn.execute("SELECT path, title, mtime, centrality, is_evergreen FROM nodes WHERE title = ?", (title,))
            return [dict(row) for row in cursor]

    def get_all_edges(self) -> list[tuple[str, str]]:
        with self.get_conn() as conn:
            cursor = conn.execute("SELECT source, target FROM edges")
            return [(row["source"], row["target"]) for row in cursor]

    def upsert_block(self, block: MarkdownBlockIR) -> None:
        with self.get_conn() as conn:
            conn.execute("""
                INSERT INTO blocks (block_id, source_path, content_hash, block_type, content, headers, tags, links, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path, block_id) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    block_type = excluded.block_type,
                    content = excluded.content,
                    headers = excluded.headers,
                    tags = excluded.tags,
                    links = excluded.links,
                    updated_at = excluded.updated_at
            """, (
                block.block_id,
                block.source_path,
                block.content_hash,
                block.block_type,
                block.content,
                json.dumps(block.headers),
                json.dumps(block.tags),
                json.dumps(block.links),
                block.created_at,
                block.updated_at,
            ))
            conn.commit()

    def delete_blocks_for_note(self, source_path: str) -> None:
        with self.get_conn() as conn:
            conn.execute("DELETE FROM blocks WHERE source_path = ?", (source_path,))
            conn.commit()

    def delete_block_by_id(self, source_path: str, block_id: str) -> None:
        with self.get_conn() as conn:
            conn.execute("DELETE FROM blocks WHERE source_path = ? AND block_id = ?", (source_path, block_id))
            conn.commit()

    def get_blocks_for_note(self, source_path: str) -> list[MarkdownBlockIR]:
        with self.get_conn() as conn:
            cursor = conn.execute("SELECT * FROM blocks WHERE source_path = ?", (source_path,))
            blocks = []
            for row in cursor:
                blocks.append(
                    MarkdownBlockIR(
                        block_id=row["block_id"],
                        content_hash=row["content_hash"],
                        source_path=row["source_path"],
                        block_type=row["block_type"],
                        content=row["content"],
                        headers=json.loads(row["headers"]),
                        tags=json.loads(row["tags"]),
                        links=json.loads(row["links"]),
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                )
            return blocks

    def add_edge(self, source: str, target: str, ambiguous_link: bool = False) -> None:
        with self.get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO edges (source, target, ambiguous_link)
                VALUES (?, ?, ?)
            """, (source, target, 1 if ambiguous_link else 0))
            conn.commit()

    def delete_edges_for_source(self, source: str) -> None:
        with self.get_conn() as conn:
            conn.execute("DELETE FROM edges WHERE source = ?", (source,))
            conn.commit()

    def update_centrality_cache(self, centrality_scores: dict[str, float]) -> None:
        with self.get_conn() as conn:
            for path, score in centrality_scores.items():
                conn.execute("UPDATE nodes SET centrality = ? WHERE path = ?", (score, path))
            conn.commit()
