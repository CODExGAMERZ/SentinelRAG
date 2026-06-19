from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Condition
from typing import Iterator


@dataclass(slots=True)
class ArbitrationState:
    tier: str
    active_queries: int = 0
    active_ingestion_jobs: int = 0
    queued_ingestion_jobs: int = 0
    history: list[str] = field(default_factory=list)


class ResourceArbiter:
    def __init__(self, tier: str) -> None:
        self._tier = tier.upper()
        self._condition = Condition()
        self._queue: deque[str] = deque()
        self._state = ArbitrationState(tier=self._tier)

    @property
    def state(self) -> ArbitrationState:
        with self._condition:
            return ArbitrationState(
                tier=self._state.tier,
                active_queries=self._state.active_queries,
                active_ingestion_jobs=self._state.active_ingestion_jobs,
                queued_ingestion_jobs=self._state.queued_ingestion_jobs,
                history=list(self._state.history),
            )

    @contextmanager
    def query_slot(self) -> Iterator[None]:
        with self._condition:
            self._state.active_queries += 1
            self._state.history.append("query:start")
        try:
            yield
        finally:
            with self._condition:
                self._state.active_queries -= 1
                self._state.history.append("query:end")
                self._condition.notify_all()

    @contextmanager
    def ingestion_slot(self, job_id: str = "ingest") -> Iterator[None]:
        if self._tier == "A":
            with self._condition:
                self._state.active_ingestion_jobs += 1
                self._state.history.append(f"{job_id}:start")
            try:
                yield
            finally:
                with self._condition:
                    self._state.active_ingestion_jobs -= 1
                    self._state.history.append(f"{job_id}:end")
                    self._condition.notify_all()
            return

        with self._condition:
            self._queue.append(job_id)
            self._state.queued_ingestion_jobs = len(self._queue)
            while self._state.active_queries > 0 or self._queue[0] != job_id:
                self._condition.wait()
            self._queue.popleft()
            self._state.queued_ingestion_jobs = len(self._queue)
            self._state.active_ingestion_jobs += 1
            self._state.history.append(f"{job_id}:start")
        try:
            yield
        finally:
            with self._condition:
                self._state.active_ingestion_jobs -= 1
                self._state.history.append(f"{job_id}:end")
                self._condition.notify_all()
