from sentinelrag.agents.workflow import select_workflow_topology


def test_selects_full_topology_for_tier_a() -> None:
    topology = select_workflow_topology("A")
    assert topology.nodes == ("planner", "retriever", "evidence_merger", "validator", "critic", "synthesizer")


def test_selects_minimal_topology_for_tier_c() -> None:
    topology = select_workflow_topology("C")
    assert topology.nodes == ("retriever", "evidence_merger", "synthesizer")
