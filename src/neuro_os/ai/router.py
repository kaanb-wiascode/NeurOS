import dataclasses
import enum
import threading
import typing
import uuid

from neuro_os.ai.audit import AuditLog
from neuro_os.domain import DecodedIntent, Intent
from neuro_os.intent.safety import Decision, SafetyPolicy


class ToolRisk(enum.StrEnum):
    LOW = "LOW"
    HIGH = "HIGH"


class ActionStatus(enum.StrEnum):
    EXECUTED = "EXECUTED"
    REJECTED = "REJECTED"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"


ToolHandler = typing.Callable[[dict[str, typing.Any]], typing.Any]


@dataclasses.dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    risk: ToolRisk
    handler: ToolHandler


@dataclasses.dataclass(frozen=True, slots=True)
class CortexAction:
    action_id: str
    intent: Intent
    confidence: float
    tool_name: str | None
    risk: ToolRisk | None
    status: ActionStatus
    reason: str
    result: typing.Any = None


@dataclasses.dataclass(slots=True)
class _PendingAction:
    action_id: str
    event: DecodedIntent
    tool: ToolDefinition
    context: dict[str, typing.Any]


class CortexRouter:
    """Safety-first structured intent router.

    Neural input can propose a high-impact action, but only an explicit confirmation
    supplied through a non-neural channel can execute it.
    """

    _ALLOWED_CONFIRMATION_CHANNELS = frozenset({"keyboard", "touch", "voice"})

    def __init__(
        self,
        *,
        safety_policy: SafetyPolicy | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self.safety_policy = safety_policy or SafetyPolicy()
        self.audit_log = audit_log or AuditLog()
        self._tools: dict[Intent, ToolDefinition] = {}
        self._pending: dict[str, _PendingAction] = {}
        self._lock = threading.Lock()

    def register(self, intent: Intent, tool: ToolDefinition) -> None:
        if intent is Intent.UNKNOWN:
            raise ValueError("UNKNOWN cannot be registered to a tool")
        if not tool.name.strip():
            raise ValueError("tool name must be non-empty")
        with self._lock:
            self._tools[intent] = tool

    def route(
        self,
        event: DecodedIntent,
        *,
        context: dict[str, typing.Any] | None = None,
    ) -> CortexAction:
        action_id = uuid.uuid4().hex
        tool = self._tools.get(event.intent)
        if tool is None:
            return self._reject(
                action_id=action_id,
                event=event,
                reason="no tool registered for decoded intent",
            )

        decision = self.safety_policy.evaluate(
            event,
            high_impact_action=tool.risk is ToolRisk.HIGH,
        )
        if decision.decision is Decision.REJECT:
            return self._reject(
                action_id=action_id,
                event=event,
                tool=tool,
                reason=decision.reason,
            )

        action_context = dict(context or {})
        if decision.decision is Decision.REQUIRE_CONFIRMATION:
            with self._lock:
                self._pending[action_id] = _PendingAction(
                    action_id=action_id,
                    event=event,
                    tool=tool,
                    context=action_context,
                )
            action = CortexAction(
                action_id=action_id,
                intent=event.intent,
                confidence=event.confidence,
                tool_name=tool.name,
                risk=tool.risk,
                status=ActionStatus.PENDING_CONFIRMATION,
                reason=decision.reason,
            )
            self._audit(action, event_type="route")
            return action

        return self._execute(
            action_id=action_id,
            event=event,
            tool=tool,
            context=action_context,
            reason=decision.reason,
            event_type="route",
        )

    def confirm(
        self,
        action_id: str,
        *,
        channel: str,
    ) -> CortexAction:
        normalized_channel = channel.strip().lower()
        if normalized_channel not in self._ALLOWED_CONFIRMATION_CHANNELS:
            raise ValueError(
                "confirmation channel must be one of: "
                + ", ".join(sorted(self._ALLOWED_CONFIRMATION_CHANNELS))
            )

        with self._lock:
            pending = self._pending.pop(action_id, None)
        if pending is None:
            raise KeyError(f"pending action does not exist: {action_id}")

        context = dict(pending.context)
        context["confirmation_channel"] = normalized_channel
        return self._execute(
            action_id=pending.action_id,
            event=pending.event,
            tool=pending.tool,
            context=context,
            reason=f"confirmed via non-neural channel: {normalized_channel}",
            event_type="confirmation",
        )

    def cancel(self, action_id: str, *, reason: str = "cancelled by user") -> CortexAction:
        with self._lock:
            pending = self._pending.pop(action_id, None)
        if pending is None:
            raise KeyError(f"pending action does not exist: {action_id}")

        action = CortexAction(
            action_id=action_id,
            intent=pending.event.intent,
            confidence=pending.event.confidence,
            tool_name=pending.tool.name,
            risk=pending.tool.risk,
            status=ActionStatus.REJECTED,
            reason=reason,
        )
        self._audit(action, event_type="cancellation")
        return action

    @property
    def pending_action_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._pending)

    def _execute(
        self,
        *,
        action_id: str,
        event: DecodedIntent,
        tool: ToolDefinition,
        context: dict[str, typing.Any],
        reason: str,
        event_type: str,
    ) -> CortexAction:
        result = tool.handler(context)
        action = CortexAction(
            action_id=action_id,
            intent=event.intent,
            confidence=event.confidence,
            tool_name=tool.name,
            risk=tool.risk,
            status=ActionStatus.EXECUTED,
            reason=reason,
            result=result,
        )
        self._audit(action, event_type=event_type)
        return action

    def _reject(
        self,
        *,
        action_id: str,
        event: DecodedIntent,
        reason: str,
        tool: ToolDefinition | None = None,
    ) -> CortexAction:
        action = CortexAction(
            action_id=action_id,
            intent=event.intent,
            confidence=event.confidence,
            tool_name=tool.name if tool is not None else None,
            risk=tool.risk if tool is not None else None,
            status=ActionStatus.REJECTED,
            reason=reason,
        )
        self._audit(action, event_type="route")
        return action

    def _audit(self, action: CortexAction, *, event_type: str) -> None:
        self.audit_log.record(
            event_type=event_type,
            action_id=action.action_id,
            intent=action.intent.value,
            confidence=action.confidence,
            tool_name=action.tool_name,
            status=action.status.value,
            reason=action.reason,
            metadata={
                "risk": action.risk.value if action.risk is not None else None,
            },
        )


def create_navigation_router(
    *,
    safety_policy: SafetyPolicy | None = None,
    audit_log: AuditLog | None = None,
) -> CortexRouter:
    """Create the first low-risk Cortex tool map for UI navigation."""

    def navigation_handler(action: str) -> ToolHandler:
        def handle(context: dict[str, typing.Any]) -> dict[str, typing.Any]:
            return {
                "action": action,
                "context": dict(context),
            }

        return handle

    router = CortexRouter(
        safety_policy=safety_policy,
        audit_log=audit_log,
    )
    router.register(
        Intent.LEFT,
        ToolDefinition(
            name="ui.navigate_left",
            risk=ToolRisk.LOW,
            handler=navigation_handler("navigate_left"),
        ),
    )
    router.register(
        Intent.RIGHT,
        ToolDefinition(
            name="ui.navigate_right",
            risk=ToolRisk.LOW,
            handler=navigation_handler("navigate_right"),
        ),
    )
    router.register(
        Intent.SELECT,
        ToolDefinition(
            name="ui.select",
            risk=ToolRisk.LOW,
            handler=navigation_handler("select"),
        ),
    )
    router.register(
        Intent.BACK,
        ToolDefinition(
            name="ui.back",
            risk=ToolRisk.LOW,
            handler=navigation_handler("back"),
        ),
    )
    return router
