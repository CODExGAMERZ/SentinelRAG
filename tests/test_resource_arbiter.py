from sentinelrag.resource_arbiter import ResourceArbiter


def test_tier_a_allows_immediate_ingestion() -> None:
    arbiter = ResourceArbiter("A")
    with arbiter.query_slot():
        with arbiter.ingestion_slot("job1"):
            state = arbiter.state
            assert state.active_queries == 1
            assert state.active_ingestion_jobs == 1


def test_tier_c_records_query_precedence() -> None:
    arbiter = ResourceArbiter("C")
    with arbiter.query_slot():
        state = arbiter.state
        assert state.active_queries == 1
