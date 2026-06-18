from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .config import AppConfig, ensure_app_dirs
from .llm import choose_model, generate_with_ollama, ollama_status
from .storage import GraphMemory, VectorStore
from .types import Evidence


def extractive_answer(question: str, evidence: list[Evidence]) -> str:
    if not evidence:
        return "I do not have indexed evidence relevant to that question."
    query_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]+", question) if len(term) > 3}
    candidates: list[tuple[int, int, str]] = []
    for source_index, item in enumerate(evidence[:5], start=1):
        for line in item.text.splitlines():
            cleaned = _clean_evidence_line(line)
            if not cleaned:
                continue
            score = sum(1 for term in query_terms if term in cleaned.lower())
            if score > 0:
                candidates.append((score, source_index, cleaned))

    if not candidates:
        for source_index, item in enumerate(evidence[:3], start=1):
            cleaned = _clean_evidence_line(item.text)
            if cleaned:
                candidates.append((0, source_index, cleaned[:260]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    bullets: list[str] = []
    for _, source_index, text in candidates:
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        bullets.append(f"- {text} [{source_index}]")
        if len(bullets) == 4:
            break

    return "Based on the indexed evidence:\n" + "\n".join(bullets)


def _clean_evidence_line(line: str) -> str:
    cleaned = re.sub(r"\s+", " ", line).strip()
    cleaned = cleaned.strip("| ")
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    cleaned = re.sub(r"^- \[[ xX]\]\s*", "", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "").strip()
    cleaned = re.sub(r"^[-*]\s*", "", cleaned)
    cleaned = cleaned.strip("*_ ")
    if not cleaned or cleaned in {"---"}:
        return ""
    lowered = cleaned.lower()
    if lowered.startswith(("sentinelrag ", "python -m sentinelrag ", "export pythonpath=", "$env:pythonpath=")):
        return ""
    if cleaned.startswith("```"):
        return ""
    if set(cleaned) <= {"-", "|", " "}:
        return ""
    if len(cleaned) < 24:
        return ""
    return cleaned[:320]


def build_prompt(question: str, evidence: list[Evidence]) -> str:
    stable_context = "\n\n".join(
        f"[{index}] Source: {item.source_path}\nScore: {item.score}\nTemporal: {item.temporal_status}\nFacts: {', '.join(item.facts) if item.facts else 'none'}\n{item.text}"
        for index, item in enumerate(evidence, start=1)
    )
    boundary = f"BOUNDARY-{uuid.uuid4()}"
    return (
        "You are SentinelRAG, a local-first retrieval augmented assistant. "
        "Answer only from the provided evidence. If the evidence is insufficient, say so. "
        "Cite sources with bracket numbers like [1].\n\n"
        f"Evidence:\n{stable_context}\n\n"
        f"{boundary}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def ask_question(question: str, config: AppConfig, collection: str | None = None, top_k: int | None = None) -> dict:
    started = time.perf_counter()
    base = ensure_app_dirs(config)
    active_collection = collection or config.storage.collection
    vector_store = VectorStore(base, active_collection)
    graph = GraphMemory(base, active_collection)
    raw_evidence = vector_store.search(question, top_k or config.retrieval.top_k)
    evidence = graph.expand_evidence(raw_evidence)

    status = ollama_status()
    if not status.available:
        answer = extractive_answer(question, evidence)
        model = choose_model(config.model.name, [])
    elif not evidence:
        answer = "I do not have indexed evidence relevant to that question."
        model = choose_model(config.model.name, status.models)
    else:
        model = choose_model(config.model.name, status.models)
        answer = generate_with_ollama(
            build_prompt(question, evidence),
            model=model,
            num_ctx=config.model.num_ctx,
            num_parallel=config.model.num_parallel,
        )

    elapsed = round(time.perf_counter() - started, 3)
    return {
        "question": question,
        "answer": answer,
        "model": model,
        "elapsed_seconds": elapsed,
        "evidence": [asdict(item) for item in evidence],
    }


def result_json(result: dict) -> str:
    return json.dumps(result, indent=2)


def format_answer(result: dict) -> str:
    lines = [result["answer"], "", "Sources:"]
    for index, item in enumerate(result["evidence"], start=1):
        lines.append(f"[{index}] {Path(item['source_path']).name} ({item['score']})")
    lines.append("")
    lines.append(f"Model: {result['model']} | elapsed: {result['elapsed_seconds']}s")
    return "\n".join(lines)
