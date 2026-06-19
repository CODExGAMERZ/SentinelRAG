from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import AppConfig, ensure_app_dirs
from .hardware_profiler import detect_hardware
from .resource_arbiter import ResourceArbiter
from .storage.vector_store import VectorStore
from .graph.graph_store import GraphStore
from .agents.workflow import build_agent_graph
from .types import Evidence

logger = logging.getLogger(__name__)

_arbiter: ResourceArbiter | None = None


def get_arbiter(tier: str) -> ResourceArbiter:
    global _arbiter
    if _arbiter is None:
        _arbiter = ResourceArbiter(tier)
    return _arbiter


def ask_question(
    question: str,
    config: AppConfig,
    collection: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    Core RAG entrypoint. Detects hardware tier, obtains a query slot from the ResourceArbiter,
    runs the LangGraph agent workflow, and formats the grounded response.
    """
    started = time.perf_counter()
    base = ensure_app_dirs(config)
    active_collection = collection or config.storage.collection

    # 1. Initialize stores
    vector_store = VectorStore(base, active_collection)
    graph_store = GraphStore(base, config.storage.sqlite_filename)

    try:
        # 2. Hardware and topology profiling
        profile = detect_hardware()
        tier = config.hardware.tier if config.hardware.tier != "auto" else profile.recommended_tier
        
        # Override top_k inside retrieval config if passed
        if top_k is not None:
            config.retrieval.top_k = top_k

        # 3. Resource Arbitration
        arbiter = get_arbiter(tier)
        
        with arbiter.query_slot():
            # 4. Build and run LangGraph
            graph = build_agent_graph(tier)
            
            inputs = {
                "query": question,
                "tier": tier,
                "plans": [],
                "vector_hits": [],
                "graph_hits": [],
                "retrieved_evidence": [],
                "validation_status": "unknown",
                "critique": "",
                "draft_answer": "",
                "final_answer": "",
            }
            
            graph_config = {
                "configurable": {
                    "config": config,
                    "vector_store": vector_store,
                    "graph_store": graph_store,
                }
            }
            
            try:
                outputs = graph.invoke(inputs, config=graph_config)
                final_answer = outputs.get("final_answer", "No response generated.")
                evidence = outputs.get("retrieved_evidence", [])
            except Exception as exc:
                logger.error("LangGraph execution failed: %s", exc)
                final_answer = f"Error during query execution: {exc}"
                evidence = []
    finally:
        vector_store.close()

    elapsed = round(time.perf_counter() - started, 3)

    return {
        "question": question,
        "answer": final_answer,
        "model": config.model.name if config.model.name != "auto" else profile.recommended_ollama_model,
        "elapsed_seconds": elapsed,
        "evidence": [asdict(item) for item in evidence],
    }


def result_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2)


def format_answer(result: dict[str, Any]) -> str:
    lines = [result["answer"], "", "Sources:"]
    for index, item in enumerate(result["evidence"], start=1):
        lines.append(f"[{index}] {Path(item['source_id']).name} ({item.get('final_score', item.get('score', 0.0))})")
    lines.append("")
    lines.append(f"Model: {result['model']} | elapsed: {result['elapsed_seconds']}s")
    return "\n".join(lines)
