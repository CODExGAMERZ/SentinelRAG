from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from langgraph.graph import StateGraph, START, END

from .nodes import (
    AgentState,
    planner_node,
    retriever_node,
    evidence_merger_node,
    validator_node,
    critic_node,
    validator_critic_combined_node,
    synthesizer_node,
)


@dataclass(frozen=True, slots=True)
class WorkflowTopology:
    tier: str
    nodes: tuple[str, ...]


def select_workflow_topology(tier: str | None) -> WorkflowTopology:
    normalized = (tier or "C").upper()
    if normalized == "A":
        return WorkflowTopology(
            tier="A",
            nodes=("planner", "retriever", "evidence_merger", "validator", "critic", "synthesizer"),
        )
    if normalized == "B":
        return WorkflowTopology(
            tier="B",
            nodes=("planner", "retriever", "evidence_merger", "validator_critic", "synthesizer"),
        )
    return WorkflowTopology(
        tier="C",
        nodes=("retriever", "evidence_merger", "synthesizer"),
    )


def build_agent_graph(tier: str) -> Any:
    """
    Constructs and compiles the LangGraph StateGraph based on the hardware tier.
    """
    topology = select_workflow_topology(tier)
    builder = StateGraph(AgentState)

    # Add all required nodes dynamically based on topology
    if "planner" in topology.nodes:
        builder.add_node("planner", planner_node)
    if "retriever" in topology.nodes:
        builder.add_node("retriever", retriever_node)
    if "evidence_merger" in topology.nodes:
        builder.add_node("evidence_merger", evidence_merger_node)
    if "validator" in topology.nodes:
        builder.add_node("validator", validator_node)
    if "critic" in topology.nodes:
        builder.add_node("critic", critic_node)
    if "validator_critic" in topology.nodes:
        builder.add_node("validator_critic", validator_critic_combined_node)
    if "synthesizer" in topology.nodes:
        builder.add_node("synthesizer", synthesizer_node)

    # Add linear edges following the node sequence
    nodes_seq = list(topology.nodes)
    builder.add_edge(START, nodes_seq[0])
    for i in range(len(nodes_seq) - 1):
        builder.add_edge(nodes_seq[i], nodes_seq[i + 1])
    builder.add_edge(nodes_seq[-1], END)

    return builder.compile()
