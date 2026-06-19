from sentinelrag.retrieval.query_classifier import classify_query


def test_classifies_temporal_queries() -> None:
    result = classify_query("What changed in Project X this week?")
    assert result.label == "temporal"
    assert result.matched_patterns


def test_classifies_conceptual_queries() -> None:
    result = classify_query("What is transformer architecture?")
    assert result.label == "conceptual"
