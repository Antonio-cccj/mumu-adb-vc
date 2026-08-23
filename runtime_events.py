from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from time_utils import beijing_now


class EventLevel(StrEnum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass(frozen=True)
class RunEvent:
    level: EventLevel
    category: str
    message: str
    timestamp: datetime = field(default_factory=beijing_now)
    node: str | None = None
    template: str | None = None
    score: float | None = None
    action: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["level"] = self.level.value
        record["timestamp"] = self.timestamp.isoformat(timespec="milliseconds")
        return record

    def to_display_text(self) -> str:
        parts = [
            self.timestamp.strftime("%H:%M:%S"),
            f"[{self.level.value}]",
            self.message,
        ]
        details = []
        if self.node:
            details.append(f"node={self.node}")
        if self.template:
            details.append(f"template={self.template}")
        if self.score is not None:
            details.append(f"score={self.score:.4f}")
        if self.action:
            details.append(f"action={self.action}")
        if details:
            parts.append(f"({', '.join(details)})")
        return " ".join(parts)


EventSubscriber = Callable[[RunEvent], None]


class EventHub:
    def __init__(self) -> None:
        self._subscribers: list[EventSubscriber] = []
        self._lock = Lock()

    def subscribe(self, subscriber: EventSubscriber) -> None:
        with self._lock:
            self._subscribers.append(subscriber)

    def emit(
        self,
        level: str | EventLevel,
        category: str,
        message: str,
        *,
        node: str | None = None,
        template: str | None = None,
        score: float | None = None,
        action: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> RunEvent:
        event = RunEvent(
            level=EventLevel(str(level).upper()),
            category=category,
            message=message,
            node=node,
            template=template,
            score=score,
            action=action,
            data=data or {},
        )
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber(event)
        return event


class RunEventRecorder(EventHub):
    def __init__(self, *, log_dir: str | Path = "logs", run_name: str = "run") -> None:
        super().__init__()
        timestamp = beijing_now().strftime("%Y%m%d-%H%M%S")
        safe_run_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / f"{timestamp}_{safe_run_name}.jsonl"
        self.text_path = self.log_dir / f"{timestamp}_{safe_run_name}.log"
        self._jsonl_file = self.jsonl_path.open("a", encoding="utf-8")
        self._text_file = self.text_path.open("a", encoding="utf-8")
        self._write_lock = Lock()
        self.subscribe(self._write_event)

    def close(self) -> None:
        with self._write_lock:
            self._jsonl_file.close()
            self._text_file.close()

    def _write_event(self, event: RunEvent) -> None:
        with self._write_lock:
            self._jsonl_file.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            self._jsonl_file.flush()
            self._text_file.write(event.to_display_text() + "\n")
            self._text_file.flush()
