"""Structured JSON logging with request correlation IDs and stage timing.

Log records never include raw document text or secrets — only identifiers
(doc id, chunk id, file name) and numeric/status fields, per the assignment's
observability requirement (2.7, Engineering/Logging).
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-"
)


def new_correlation_id() -> str:
    cid = uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    return _correlation_id.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", get_correlation_id()),
        }
        for key, value in vars(record).items():
            if key in payload or key in (
                "args", "msg", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno",
                "funcName", "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "name",
            ):
                continue
            try:
                json.dumps(value)
            except TypeError:
                value = str(value)
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


class StageTimer:
    """Collects named stage durations (ms) for one request, for logs and API responses."""

    def __init__(self) -> None:
        self._stages: dict[str, float] = {}

    @contextmanager
    def measure(self, stage_name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stages[stage_name] = round((time.perf_counter() - start) * 1000, 2)

    @property
    def stages_ms(self) -> dict[str, float]:
        return dict(self._stages)

    @property
    def total_ms(self) -> float:
        return round(sum(self._stages.values()), 2)
