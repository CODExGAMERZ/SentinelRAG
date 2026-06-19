from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..types import Evidence, MergedEvidence
from ..graph.ranking import compute_recency_score, min_max_normalize

if TYPE_CHECKING:
    from ..graph.graph_store import GraphStore

logger = logging.getLogger(__name__)


def merge_evidence(
    vector_hits: list[Evidence],
    graph_hits: list[Evidence],
    graph_store: GraphStore,
    query_is_temporal: bool,
) -> list[MergedEvidence]:
    """
    Combines vector and graph search results using RRF (Reciprocal Rank Fusion)
    and scores candidates using a multi-factor blend of Semantic Similarity,
    Normalized Centrality, and Temporal Relevance.
    """
    # 1. Reciprocal Rank Fusion (RRF) to merge candidate lists
    rrf_scores: dict[str, float] = {}
    
    # Map chunk_id to its respective raw hits
    vector_map = {item.chunk_id: item for item in vector_hits}
    graph_map = {item.chunk_id: item for item in graph_hits}
    
    all_chunk_ids = set(vector_map.keys()) | set(graph_map.keys())
    
    # Calculate rank dictionaries
    vector_ranks = {item.chunk_id: idx for idx, item in enumerate(vector_hits)}
    graph_ranks = {item.chunk_id: idx for idx, item in enumerate(graph_hits)}
    
    for chunk_id in all_chunk_ids:
        score = 0.0
        if chunk_id in vector_ranks:
            score += 1.0 / (60.0 + vector_ranks[chunk_id] + 1)
        if chunk_id in graph_ranks:
            score += 1.0 / (60.0 + graph_ranks[chunk_id] + 1)
        rrf_scores[chunk_id] = score

    # Sort candidates by RRF score to select top hits
    sorted_chunks = sorted(all_chunk_ids, key=lambda c: rrf_scores[c], reverse=True)
    
    # 2. Gather notes metadata from SQLite to compute centrality and recency
    # We query the SQLite nodes table for candidate files
    candidates: list[MergedEvidence] = []
    
    # Fetch note properties for all candidate source files
    note_metadata: dict[str, dict] = {}
    with graph_store.get_conn() as conn:
        cursor = conn.execute("SELECT path, centrality, mtime, is_evergreen FROM nodes")
        for row in cursor:
            note_metadata[row["path"]] = {
                "centrality": row["centrality"],
                "mtime": row["mtime"],
                "is_evergreen": bool(row["is_evergreen"]),
            }

    # Normalize centrality scores within the retrieved candidate set
    raw_centralities: dict[str, float] = {}
    for chunk_id in sorted_chunks:
        hit = vector_map.get(chunk_id) or graph_map[chunk_id]
        meta = note_metadata.get(hit.source_path, {"centrality": 1.0, "mtime": 0.0, "is_evergreen": False})
        raw_centralities[chunk_id] = meta["centrality"]
    
    normalized_centralities = min_max_normalize(raw_centralities)

    for chunk_id in sorted_chunks:
        # Determine source type
        in_vector = chunk_id in vector_map
        in_graph = chunk_id in graph_map
        
        if in_vector and in_graph:
            source_type = "hybrid"
            hit = vector_map[chunk_id]
        elif in_vector:
            source_type = "vector"
            hit = vector_map[chunk_id]
        else:
            source_type = "graph"
            hit = graph_map[chunk_id]

        meta = note_metadata.get(hit.source_path, {
            "centrality": 1.0,
            "mtime": hit.score if source_type == "graph" else 0.0,  # fallback if not in DB
            "is_evergreen": False
        })
        
        # Semantic Score (None if not in vector_hits)
        semantic_score = vector_map[chunk_id].score if chunk_id in vector_map else None
        
        # Centrality Score
        centrality_score = normalized_centralities[chunk_id]
        
        # Recency Score
        if query_is_temporal:
            recency_score = compute_recency_score(meta["mtime"], meta["is_evergreen"])
        else:
            recency_score = 1.0

        # Multi-factor Blending
        sem = semantic_score if semantic_score is not None else 0.0
        final_score = (0.60 * sem) + (0.30 * centrality_score) + (0.10 * recency_score)

        # Build provenance
        provenance = {
            "chunk_id": chunk_id,
            "facts": list(hit.facts),
        }

        candidates.append(
            MergedEvidence(
                content=hit.text,
                source_type=source_type,  # type: ignore
                source_id=hit.source_path,
                semantic_score=semantic_score,
                centrality_score=meta["centrality"],  # raw centrality
                recency_score=recency_score,
                final_score=round(final_score, 4),
                provenance=provenance,
            )
        )

    # Sort final merged candidates by final_score
    return sorted(candidates, key=lambda item: item.final_score, reverse=True)
