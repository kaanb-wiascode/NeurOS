import dataclasses
import json
import pathlib
import threading
import time
import typing


@dataclasses.dataclass(frozen=True, slots=True)
class AuditEvent:
    event_type: str
    action_id: str
    timestamp: float
    intent: str
    confidence: float
    tool_name: str | None
    status: str
    reason: str
    metadata: dict[str, typing.Any]


class AuditLog:
    """Append-only local JSONL audit log.

    Raw EEG is intentionally outside this schema.
    """

    def __init__(self, path: str | pathlib.Path = ".neuros/audit/events.jsonl") -> None:
        self.path = pathlib.Path(path)
        self._lock = threading.Lock()

    def append(self, event: AuditEvent) -> None:
        if not event.event_type.strip():
            raise ValueError("event_type must be non-empty")
        if not event.action_id.strip():
            raise ValueError("action_id must be non-empty")

        encoded = json.dumps(dataclasses.asdict(event), separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")

    def record(
        self,
        *,
        event_type: str,
        action_id: str,
        intent: str,
        confidence: float,
        tool_name: str | None,
        status: str,
        reason: str,
        metadata: dict[str, typing.Any] | None = None,
    ) -> None:
        self.append(
            AuditEvent(
                event_type=event_type,
                action_id=action_id,
                timestamp=time.time(),
                intent=intent,
                confidence=confidence,
                tool_name=tool_name,
                status=status,
                reason=reason,
                metadata=dict(metadata or {}),
            )
        )
