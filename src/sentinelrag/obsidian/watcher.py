from __future__ import annotations

import json
import logging
import os
import queue
import re
import time
from pathlib import Path
from threading import Thread, Event
from typing import TYPE_CHECKING, cast

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileMovedEvent, FileSystemEvent

from .parser import parse_markdown_file
from ..config import AppConfig
from ..graph.link_resolver import resolve_wikilink
from ..graph.entity_graph import EntityGraph
from ..llm import generate_with_ollama, ollama_status
from ..types import MarkdownBlockIR

if TYPE_CHECKING:
    from ..storage.vector_store import VectorStore
    from ..graph.graph_store import GraphStore

logger = logging.getLogger(__name__)


class VaultWatcher(FileSystemEventHandler):
    def __init__(
        self,
        vault_path: Path,
        config: AppConfig,
        vector_store: VectorStore,
        graph_store: GraphStore,
    ) -> None:
        super().__init__()
        self.vault_path = vault_path.resolve()
        self.config = config
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.entity_graph = EntityGraph(graph_store)

        # Thread-safe debounce mechanism
        self.event_queue: queue.Queue[tuple[float, FileSystemEvent]] = queue.Queue()
        self.stop_event = Event()
        self.worker_thread = Thread(target=self._process_queue, daemon=True)
        self.observer = Observer()

    def sync_all(self, force: bool = False) -> None:
        """Performs a startup delta sync comparing file mtimes against SQLite records."""
        logger.info("Starting startup delta sync for vault: %s", self.vault_path)
        
        # 1. Discover all current markdown files in the vault
        current_files = {}
        for root, _, files in os.walk(self.vault_path):
            for file in files:
                if file.endswith(".md"):
                    path = Path(root) / file
                    try:
                        current_files[str(path.resolve())] = path.stat().st_mtime
                    except OSError:
                        pass

        # 2. Retrieve existing mtimes from database
        db_files = self.graph_store.get_note_mtimes()

        # Check if vector store is empty to force a full re-sync
        force_full_sync = force or (self.vector_store.count() == 0 and len(current_files) > 0)
        if force_full_sync:
            logger.info("VectorStore is empty or force sync requested. Forcing full sync of all files.")
            db_files = {}

        # 3. Detect changes
        deleted_files = set(db_files.keys()) - set(current_files.keys())
        modified_files = []
        new_files = []

        for path, mtime in current_files.items():
            if path not in db_files:
                new_files.append(path)
            elif mtime > db_files[path] + 0.1:  # small margin
                modified_files.append(path)

        # 4. Process deletions
        for path in deleted_files:
            logger.info("Sync deletion detected for note: %s", path)
            self._handle_note_deleted(path)

        # 5. Process new and modified files
        for path in new_files:
            logger.info("Sync new note detected: %s", path)
            self._index_file(Path(path), current_files[path], force_reindex=force_full_sync)

        for path in modified_files:
            logger.info("Sync modification detected for note: %s", path)
            self._index_file(Path(path), current_files[path], force_reindex=force_full_sync)

        # 6. Re-resolve all Wikilink edges now that notes are indexed
        self._rebuild_all_edges()
        logger.info("Startup delta sync completed.")

    def _index_file(self, path: Path, mtime: float, force_reindex: bool = False) -> None:
        """Indices or re-indices a markdown file block-by-block."""
        str_path = str(path.resolve())
        title = path.stem
        
        # 1. Parse markdown file to MarkdownBlockIR
        try:
            new_blocks = parse_markdown_file(path, self.config.retrieval.parser_tier)
        except Exception as exc:
            logger.error("Failed to parse markdown file %s: %s", str_path, exc)
            return

        # 2. Check if evergreen (contains #evergreen tag in any block)
        is_evergreen = any("#evergreen" in block.tags for block in new_blocks)

        # 3. Upsert note metadata
        self.graph_store.upsert_note(str_path, title, mtime, is_evergreen)

        # 4. Load previous blocks from GraphStore (empty if forcing re-index)
        old_blocks = [] if force_reindex else self.graph_store.get_blocks_for_note(str_path)
        old_by_id = {b.block_id: b for b in old_blocks}
        new_by_id = {b.block_id: b for b in new_blocks}

        # 5. Determine block diffs
        inserts = [b for b in new_blocks if b.block_id not in old_by_id]
        deletes = [b for b in old_blocks if b.block_id not in new_by_id]
        updates = []
        for b in new_blocks:
            if b.block_id in old_by_id and b.content_hash != old_by_id[b.block_id].content_hash:
                updates.append(b)

        # 6. Apply deletions
        for block in deletes:
            self.vector_store.delete_file(block.source_path)  # Qdrant deletes by metadata filter (e.g. block_id)
            # Actually, delete specific point UUID:
            import uuid
            point_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, block.block_id)
            self.vector_store.client.delete(
                collection_name=self.vector_store.collection,
                points_selector=[str(point_uuid)]
            )
            self.graph_store.delete_block_by_id(str_path, block.block_id)
            # Delete triples for this block
            with self.graph_store.get_conn() as conn:
                conn.execute("DELETE FROM triples WHERE source_block_id = ?", (block.block_id,))
                conn.commit()

        # 7. Apply inserts and updates
        to_upsert_vector = []
        for block in inserts + updates:
            self.graph_store.upsert_block(block)
            to_upsert_vector.append(block)
            
            # Extract SPO triples
            self._extract_and_store_triples(block)

        if to_upsert_vector:
            self.vector_store.upsert_chunks(to_upsert_vector)

    def _extract_and_store_triples(self, block: MarkdownBlockIR) -> None:
        """Extracts and stores Subject-Predicate-Object triples for a block."""
        # Clean existing triples for this block first
        with self.graph_store.get_conn() as conn:
            conn.execute("DELETE FROM triples WHERE source_block_id = ?", (block.block_id,))
            conn.commit()

        # Ingestion-time LLM entity extraction
        triples = []
        status = ollama_status()
        
        # Only use LLM if Ollama is available, otherwise fall back to simple regex triple extractor
        if status.available:
            prompt = (
                "Extract subject-predicate-object triples from the following text. "
                "Each triple should connect two entities via a relation. "
                "Output ONLY a raw JSON list of lists of strings, e.g. [[\"Qwen3\", \"outperformed\", \"Gemma3\"]]. "
                "Do not write conversational preamble. "
                f"Text:\n{block.content}"
            )
            try:
                response = generate_with_ollama(
                    prompt,
                    model=self.config.model.name,
                    num_ctx=2048,
                    num_parallel=1,
                )
                # Attempt to parse JSON from response
                match = re.search(r"(\[.*\])", response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(1))
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, list) and len(item) >= 3:
                                triples.append((str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip()))
            except Exception:
                pass

        # Regex fallback extractor: search for simple patterns
        # e.g., "A is a B" or "A outperformed B"
        if not triples:
            triples = self._regex_triple_extractor(block.content)

        for s, p, o in triples:
            self.entity_graph.add_triple(s, p, o, block.block_id, block.source_path)

    def _regex_triple_extractor(self, text: str) -> list[tuple[str, str, str]]:
        # Match "EntityA relation EntityB" or similar simple structures
        triples = []
        # Match capitalized terms
        entity_matches = re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text)
        if len(entity_matches) >= 2:
            # Create a generic relation chain between consecutive capitalized words
            for i in range(len(entity_matches) - 1):
                triples.append((entity_matches[i], "mentions", entity_matches[i + 1]))
        return triples

    def _handle_note_deleted(self, path: str) -> None:
        self.vector_store.delete_file(path)
        self.graph_store.delete_note(path)

    def _handle_note_renamed(self, old_path: str, new_path: str) -> None:
        new_title = Path(new_path).stem
        with self.graph_store.get_conn() as conn:
            conn.execute("UPDATE nodes SET path = ?, title = ? WHERE path = ?", (new_path, new_title, old_path))
            conn.execute("UPDATE blocks SET source_path = ? WHERE source_path = ?", (new_path, old_path))
            conn.execute("UPDATE triples SET source_path = ? WHERE source_path = ?", (new_path, old_path))
            conn.execute("UPDATE edges SET source = ? WHERE source = ?", (new_path, old_path))
            conn.execute("UPDATE edges SET target = ? WHERE target = ?", (new_path, old_path))
            conn.commit()

        # Update Qdrant payloads with new path
        # Since local Qdrant is on-disk, we retrieve all points for the old path, delete them, and upsert with new path.
        # But wait! A simpler way is to just re-index the note. Or update the payload.
        # To avoid re-embedding, we can query Qdrant points and update their payload source_path.
        # However, for simplicity and perfect correctness, we delete old points and re-upsert them with the updated path.
        # Since embedding is fast (deterministic hash-based vector), reindexing is extremely cheap!
        try:
            mtime = Path(new_path).stat().st_mtime
        except OSError:
            mtime = time.time()
        self._index_file(Path(new_path), mtime)
        self.vector_store.delete_file(old_path)

    def _rebuild_all_edges(self) -> None:
        """Rebuilds Wikilink edges in the SQLite database."""
        notes = self.graph_store.get_all_notes()
        for note in notes:
            path = note["path"]
            self.graph_store.delete_edges_for_source(path)
            
            blocks = self.graph_store.get_blocks_for_note(path)
            for block in blocks:
                for link in block.links:
                    resolved, ambiguous = resolve_wikilink(self.graph_store, link, path)
                    for target_path in resolved:
                        self.graph_store.add_edge(path, target_path, ambiguous_link=ambiguous)

    # Watchdog Handlers
    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and event.src_path.endswith(".md"):
            self.event_queue.put((time.time(), event))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and event.src_path.endswith(".md"):
            self.event_queue.put((time.time(), event))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and event.src_path.endswith(".md"):
            self.event_queue.put((time.time(), event))

    def on_moved(self, event: FileMovedEvent) -> None:
        if event.src_path.endswith(".md") or event.dest_path.endswith(".md"):
            self.event_queue.put((time.time(), event))

    def start(self) -> None:
        """Starts the debounced worker thread and the Watchdog observer."""
        self.worker_thread.start()
        self.observer.schedule(self, str(self.vault_path), recursive=True)
        self.observer.start()

    def stop(self) -> None:
        """Stops the watcher."""
        self.stop_event.set()
        self.observer.stop()
        self.observer.join()
        self.worker_thread.join()

    def _process_queue(self) -> None:
        """Worker thread processing and debouncing file events by 500ms."""
        debounce_ms = self.config.watcher.debounce_ms
        pending_events: dict[str, tuple[float, FileSystemEvent]] = {}

        while not self.stop_event.is_set():
            # 1. Pull events from queue and populate pending events dict
            try:
                while True:
                    timestamp, event = self.event_queue.get_nowait()
                    # Keep only the latest event for each file path
                    path = event.src_path
                    if isinstance(event, FileMovedEvent):
                        path = event.dest_path
                    pending_events[path] = (timestamp, event)
            except queue.Empty:
                pass

            # 2. Process debounced events
            now = time.time()
            keys_to_remove = []
            for path, (timestamp, event) in list(pending_events.items()):
                if now - timestamp >= (debounce_ms / 1000.0):
                    keys_to_remove.append(path)
                    try:
                        self._dispatch_debounced_event(event)
                    except Exception as exc:
                        logger.error("Error processing event: %s", exc)

            for key in keys_to_remove:
                del pending_events[key]

            time.sleep(0.1)

    def _dispatch_debounced_event(self, event: FileSystemEvent) -> None:
        """Processes a single debounced file event."""
        from ..rag import get_arbiter
        from ..hardware_profiler import detect_hardware
        
        profile = detect_hardware()
        tier = self.config.hardware.tier if self.config.hardware.tier != "auto" else profile.recommended_tier
        arbiter = get_arbiter(tier)

        with arbiter.ingestion_slot(f"ingest-{Path(event.src_path).name}"):
            if event.event_type == "created":
                logger.info("Watcher: created %s", event.src_path)
                path = Path(event.src_path)
                if path.exists():
                    self._index_file(path, path.stat().st_mtime)
                    self._rebuild_all_edges()
            elif event.event_type == "modified":
                logger.info("Watcher: modified %s", event.src_path)
                path = Path(event.src_path)
                if path.exists():
                    self._index_file(path, path.stat().st_mtime)
                    self._rebuild_all_edges()
            elif event.event_type == "deleted":
                logger.info("Watcher: deleted %s", event.src_path)
                self._handle_note_deleted(event.src_path)
                self._rebuild_all_edges()
            elif event.event_type == "moved":
                moved_event = cast(FileMovedEvent, event)
                logger.info("Watcher: moved from %s to %s", moved_event.src_path, moved_event.dest_path)
                if moved_event.src_path.endswith(".md") and moved_event.dest_path.endswith(".md"):
                    self._handle_note_renamed(moved_event.src_path, moved_event.dest_path)
                elif moved_event.src_path.endswith(".md"):
                    self._handle_note_deleted(moved_event.src_path)
                elif moved_event.dest_path.endswith(".md"):
                    path = Path(moved_event.dest_path)
                    if path.exists():
                        self._index_file(path, path.stat().st_mtime)
                self._rebuild_all_edges()
