from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from sentinelrag.cli import main
from sentinelrag.storage.vector_store import VectorStore
from sentinelrag.graph.graph_store import GraphStore
from sentinelrag.agents.workflow import select_workflow_topology


def test_workflow_topology_none_guard() -> None:
    topology = select_workflow_topology(None)
    assert topology.tier == "C"
    assert "retriever" in topology.nodes
    assert "synthesizer" in topology.nodes


def test_cli_empty_question_guard(capsys) -> None:
    rc = main(["ask", ""])
    assert rc == 1
    captured = capsys.readouterr()
    assert "Error: question is empty" in captured.err

    rc = main(["ask", "   "])
    assert rc == 1


def test_cli_nonexistent_path_ingest_guard(capsys) -> None:
    rc = main(["ingest", "nonexistent_path_xyz_123_456"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "Error: path" in captured.err
    assert "does not exist" in captured.err


def test_consecutive_ingest_and_ask_same_process(tmp_path: Path) -> None:
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    note_path = vault_dir / "note.md"
    note_path.write_text("# Test Title\nSome content for testing.", encoding="utf-8")

    rc_ingest = main(["ingest", str(vault_dir), "--collection", "test-lock-col"])
    assert rc_ingest == 0

    rc_ask = main(["ask", "What is testing?", "--collection", "test-lock-col"])
    assert rc_ask == 0


def test_empty_vector_store_force_sync(tmp_path: Path) -> None:
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    note_path = vault_dir / "note.md"
    note_path.write_text("# Test Note\nThis is a block of text.", encoding="utf-8")

    col = "test-sync-col"
    vector_store = VectorStore(tmp_path, col)
    graph_store = GraphStore(tmp_path, "test_sync_db.db")

    vector_store.reset()
    graph_store.reset()

    from sentinelrag.config import load_config
    config = load_config()
    from sentinelrag.obsidian.watcher import VaultWatcher
    watcher = VaultWatcher(vault_dir, config, vector_store, graph_store)

    watcher.sync_all()
    assert graph_store.get_note_mtimes() != {}
    blocks_count = len(graph_store.get_blocks_for_note(str(note_path.resolve())))
    assert blocks_count > 0
    assert vector_store.count() == blocks_count

    vector_store.reset()
    assert vector_store.count() == 0
    assert graph_store.get_note_mtimes() != {}

    watcher.sync_all()
    assert vector_store.count() == blocks_count  # Should be successfully re-upserted

    vector_store.close()
