from pathlib import Path
from watchdog.events import FileMovedEvent
from sentinelrag.obsidian.watcher import VaultWatcher
from sentinelrag.config import AppConfig
from sentinelrag.storage.vector_store import VectorStore
from sentinelrag.graph.graph_store import GraphStore
from sentinelrag.types import MarkdownBlockIR


def test_watcher_rename_integrity(tmp_path: Path) -> None:
    config = AppConfig()
    vector_store = VectorStore(tmp_path, "test_collection")
    graph_store = GraphStore(tmp_path, "test.db")
    
    old_file = tmp_path / "OldNote.md"
    old_file.write_text("# OldNote\n\nContent here.", encoding="utf-8")
    
    watcher = VaultWatcher(tmp_path, config, vector_store, graph_store)
    
    watcher._index_file(old_file, old_file.stat().st_mtime)
    
    assert graph_store.get_note_mtimes() != {}
    assert vector_store.count() == 2  # heading and paragraph
    
    new_file = tmp_path / "NewNote.md"
    new_file.write_text("# NewNote\n\nContent here.", encoding="utf-8")
    
    event = FileMovedEvent(str(old_file), str(new_file))
    watcher._dispatch_debounced_event(event)
    
    assert str(old_file.resolve()) not in graph_store.get_note_mtimes()
    assert str(new_file.resolve()) in graph_store.get_note_mtimes()
    assert vector_store.count() == 2
