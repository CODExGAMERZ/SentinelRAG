from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


TEMPORAL_PATTERNS = [
    r"\btoday\b",
    r"\byesterday\b",
    r"\bthis week\b",
    r"\bthis month\b",
    r"\brecent\b",
    r"\blatest\b",
    r"\bupdated\b",
    r"\bchanged\b",
    r"\bwhen\b",
    r"\bnew\b",
    r"\b\d{4}\b",
]


@dataclass(frozen=True, slots=True)
class QueryClassification:
    label: Literal["conceptual", "temporal"]
    matched_patterns: list[str]


def classify_query(query: str) -> QueryClassification:
    lowered = query.lower()
    matches = [pattern for pattern in TEMPORAL_PATTERNS if re.search(pattern, lowered)]
    if matches:
        return QueryClassification(label="temporal", matched_patterns=matches)
    return QueryClassification(label="conceptual", matched_patterns=[])
