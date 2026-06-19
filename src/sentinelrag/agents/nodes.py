from __future__ import annotations

import re
from typing import Any, TypedDict, cast
from langchain_core.runnables import RunnableConfig

from ..types import Evidence, MergedEvidence
from ..graph.traversal import traverse_graph
from ..agents.evidence_merger import merge_evidence
from ..llm import generate_with_ollama
from ..retrieval.query_classifier import classify_query

class AgentState(TypedDict):
    query: str
    tier: str
    plans: list[str]
    vector_hits: list[Evidence]
    graph_hits: list[Evidence]
    retrieved_evidence: list[MergedEvidence]
    validation_status: str  # "valid" | "invalid"
    critique: str
    draft_answer: str
    final_answer: str


def planner_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Planner agent node: outlines reasoning and key retrieval seeds."""
    app_config = config["configurable"]["config"]
    model = app_config.model.name
    num_ctx = app_config.model.num_ctx
    num_parallel = app_config.model.num_parallel

    prompt = (
        "You are the Planner agent for SentinelRAG. "
        "Analyze the user query and output a concise plan of what concepts, "
        "entities, and facts are needed to answer it.\n"
        f"Query: {state['query']}\n"
        "Plan:"
    )
    try:
        plan = generate_with_ollama(prompt, model, num_ctx, num_parallel)
    except Exception as exc:
        plan = f"Default retrieval plan (fallback due to error: {exc})"

    return {"plans": state.get("plans", []) + [plan]}


def retriever_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Retriever agent node: fetches candidates from vector search and graph traversal."""
    app_config = config["configurable"]["config"]
    vector_store = config["configurable"]["vector_store"]
    graph_store = config["configurable"]["graph_store"]

    query = state["query"]
    top_k = app_config.retrieval.top_k
    depth = app_config.retrieval.graph_expansion_depth

    vector_hits = vector_store.search(query, top_k)

    entity_seeds = list(set(re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", query)))
    
    seed_paths = []
    for entity in entity_seeds:
        notes = graph_store.get_note_by_title(entity)
        for n in notes:
            seed_paths.append(n["path"])

    traversed_nodes = traverse_graph(graph_store, seed_paths, depth)

    graph_hits = []
    for path, d in traversed_nodes.items():
        blocks = graph_store.get_blocks_for_note(path)
        for b in blocks:
            score = round(1.0 - (d * 0.25), 4)
            graph_hits.append(
                Evidence(
                    chunk_id=b.block_id,
                    doc_id=b.block_id,
                    source_path=b.source_path,
                    text=b.content,
                    score=score,
                    facts=b.tags,
                )
            )

    return {
        "vector_hits": vector_hits,
        "graph_hits": graph_hits,
    }


def evidence_merger_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Evidence Merger node: consolidates vector and graph results using RRF and blending."""
    app_config = config["configurable"]["config"]
    graph_store = config["configurable"]["graph_store"]

    query = state["query"]
    classification = classify_query(query)
    query_is_temporal = classification.label == "temporal"

    vector_hits = state.get("vector_hits", [])
    graph_hits = state.get("graph_hits", [])

    merged = merge_evidence(vector_hits, graph_hits, graph_store, query_is_temporal)
    
    top_k = app_config.retrieval.top_k
    return {"retrieved_evidence": merged[:top_k]}


def validator_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Validator agent node: checks if retrieved evidence is sufficient to ground an answer."""
    app_config = config["configurable"]["config"]
    model = app_config.model.name
    num_ctx = app_config.model.num_ctx
    num_parallel = app_config.model.num_parallel

    evidence_text = "\n\n".join(
        f"[{idx}] (Source: {e.source_id}, Type: {e.source_type})\n{e.content}"
        for idx, e in enumerate(state.get("retrieved_evidence", []), start=1)
    )

    prompt = (
        "You are the Validator agent for SentinelRAG.\n"
        "Your task is to review the query and the retrieved evidence and determine if the "
        "evidence has sufficient information to answer the query.\n"
        "Reply with exactly 'valid' if there is sufficient evidence, "
        "or 'invalid' if the evidence is insufficient.\n\n"
        f"Query: {state['query']}\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Validation Status (valid/invalid):"
    )
    try:
        response = generate_with_ollama(prompt, model, num_ctx, num_parallel).lower().strip()
        status = "valid" if "valid" in response and "invalid" not in response else "invalid"
    except Exception:
        status = "valid"  # fallback

    return {"validation_status": status}


def critic_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Critic agent node: reviews draft and provides feedback/critique."""
    app_config = config["configurable"]["config"]
    model = app_config.model.name
    num_ctx = app_config.model.num_ctx
    num_parallel = app_config.model.num_parallel

    prompt = (
        "You are the Critic agent for SentinelRAG.\n"
        "Critique the plans and validation status. Suggest adjustments or flags if there are gaps "
        "in the retrieved evidence or logic.\n"
        f"Query: {state['query']}\n"
        f"Plans: {state.get('plans', [])}\n"
        f"Validation Status: {state.get('validation_status', 'unknown')}\n"
        "Critique:"
    )
    try:
        critique = generate_with_ollama(prompt, model, num_ctx, num_parallel)
    except Exception as exc:
        critique = f"Critique skipped (fallback due to error: {exc})"

    return {"critique": critique}


def validator_critic_combined_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Tier B combined node: validates evidence and critiques logic in a single LLM call."""
    app_config = config["configurable"]["config"]
    model = app_config.model.name
    num_ctx = app_config.model.num_ctx
    num_parallel = app_config.model.num_parallel

    evidence_text = "\n\n".join(
        f"[{idx}] {e.content}"
        for idx, e in enumerate(state.get("retrieved_evidence", []), start=1)
    )

    prompt = (
        "You are the Validator-Critic agent for SentinelRAG.\n"
        "Perform a combined check. First, check if the evidence is sufficient (output 'status: valid' or 'status: invalid'). "
        "Second, output a brief critique of the reasoning.\n\n"
        f"Query: {state['query']}\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Response:"
    )
    try:
        response = generate_with_ollama(prompt, model, num_ctx, num_parallel)
        status = "valid" if "status: valid" in response.lower() else "invalid"
        critique = response
    except Exception:
        status = "valid"
        critique = "Combined check fallback."

    return {
        "validation_status": status,
        "critique": critique,
    }


def synthesizer_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Synthesizer agent node: drafts the final grounded answer with citations."""
    app_config = config["configurable"]["config"]
    model = app_config.model.name
    num_ctx = app_config.model.num_ctx
    num_parallel = app_config.model.num_parallel

    evidence_list = state.get("retrieved_evidence", [])
    if not evidence_list:
        return {"final_answer": "I do not have indexed evidence relevant to that question."}

    context = "\n\n".join(
        f"[{idx}] Source: {e.source_id}\n{e.content}"
        for idx, e in enumerate(evidence_list, start=1)
    )

    prompt = (
        "You are the Synthesizer agent for SentinelRAG.\n"
        "Generate a grounded answer for the user query using ONLY the provided evidence. "
        "If the evidence is insufficient, say so. Cite sources with bracket numbers like [1].\n\n"
        f"Evidence:\n{context}\n\n"
        f"Critique: {state.get('critique', 'none')}\n\n"
        f"Query: {state['query']}\n"
        "Grounded Answer:"
    )
    try:
        final_answer = generate_with_ollama(prompt, model, num_ctx, num_parallel)
    except Exception as exc:
        final_answer = f"Fallback answer based on source [1]: {evidence_list[0].content[:200]}..."

    return {"final_answer": final_answer}
