import time
from pathlib import Path
from watchdog.events import FileModifiedEvent
from sentinelrag.obsidian.watcher import VaultWatcher
from sentinelrag.config import AppConfig
from sentinelrag.storage.vector_store import VectorStore
from sentinelrag.graph.graph_store import GraphStore


def test_watcher_debounce_coalesces_events(tmp_path: Path) -> None:
    config = AppConfig()
    config.watcher.debounce_ms = 100
    
    # Initialize mock stores
    vector_store = VectorStore(tmp_path, "test_collection")
    graph_store = GraphStore(tmp_path, "test.db")
    
    watcher = VaultWatcher(tmp_path, config, vector_store, graph_store)
    
    # Mock index method to track calls
    indexed_files = []
    def mock_index_file(path: Path, mtime: float):
        indexed_files.append(path)
        
    watcher._index_file = mock_index_file
    watcher._rebuild_all_edges = lambda: None
    
    # Start queue processor (without watchdog observer)
    watcher.worker_thread.start()
    
    # Queue multiple rapid events for the same file
    test_file = tmp_path / "Note.md"
    test_file.write_text("# Note\nContent", encoding="utf-8")
    event = FileModifiedEvent(str(test_file))
    
    watcher.on_modified(event)
    watcher.on_modified(event)
    watcher.on_modified(event)
    
    # Wait for debouncer to process
    time.sleep(0.3)
    
    watcher.stop_event.set()
    watcher.worker_thread.join()
    
    # Assert it was only indexed once
    assert len(indexed_files) == 1
