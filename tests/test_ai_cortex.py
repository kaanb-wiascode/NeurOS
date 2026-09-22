import json
from pathlib import Path

import pytest

from neuro_os.ai.audit import AuditLog
from neuro_os.ai.router import (
    ActionStatus,
    CortexRouter,
    ToolDefinition,
    ToolRisk,
    create_navigation_router,
)
from neuro_os.domain import DecodedIntent, Intent
from neuro_os.intent.safety import SafetyPolicy


def test_navigation_router_executes_safe_high_confidence_intent(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    router = create_navigation_router(audit_log=audit)

    action = router.route(
        DecodedIntent.create(Intent.SELECT, 0.95, "test"),
        context={"screen": "home"},
    )

    assert action.status is ActionStatus.EXECUTED
    assert action.tool_name == "ui.select"
    assert action.result["action"] == "select"
    assert action.result["context"]["screen"] == "home"

    event = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert event["intent"] == "SELECT"
    assert event["status"] == "EXECUTED"


def test_low_confidence_intent_is_rejected_before_tool_execution(tmp_path: Path) -> None:
    executions: list[str] = []
    router = CortexRouter(
        safety_policy=SafetyPolicy(minimum_confidence=0.70),
        audit_log=AuditLog(tmp_path / "audit.jsonl"),
    )
    router.register(
        Intent.SELECT,
        ToolDefinition(
            name="ui.select",
            risk=ToolRisk.LOW,
            handler=lambda context: executions.append("executed"),
        ),
    )

    action = router.route(DecodedIntent.create(Intent.SELECT, 0.40, "test"))

    assert action.status is ActionStatus.REJECTED
    assert executions == []


def test_high_impact_tool_requires_separate_non_neural_confirmation(tmp_path: Path) -> None:
    executions: list[str] = []
    router = CortexRouter(audit_log=AuditLog(tmp_path / "audit.jsonl"))
    router.register(
        Intent.SELECT,
        ToolDefinition(
            name="external.send_message",
            risk=ToolRisk.HIGH,
            handler=lambda context: executions.append(
                str(context["confirmation_channel"])
            ),
        ),
    )

    proposed = router.route(DecodedIntent.create(Intent.SELECT, 0.95, "test"))

    assert proposed.status is ActionStatus.PENDING_CONFIRMATION
    assert executions == []
    assert proposed.action_id in router.pending_action_ids

    confirmed = router.confirm(proposed.action_id, channel="touch")

    assert confirmed.status is ActionStatus.EXECUTED
    assert executions == ["touch"]
    assert router.pending_action_ids == ()


def test_high_impact_action_can_be_cancelled(tmp_path: Path) -> None:
    router = CortexRouter(audit_log=AuditLog(tmp_path / "audit.jsonl"))
    router.register(
        Intent.BACK,
        ToolDefinition(
            name="dangerous.example",
            risk=ToolRisk.HIGH,
            handler=lambda context: context,
        ),
    )
    proposed = router.route(DecodedIntent.create(Intent.BACK, 0.90, "test"))

    cancelled = router.cancel(proposed.action_id)

    assert cancelled.status is ActionStatus.REJECTED
    assert proposed.action_id not in router.pending_action_ids


def test_neural_confirmation_channel_is_not_allowed(tmp_path: Path) -> None:
    router = CortexRouter(audit_log=AuditLog(tmp_path / "audit.jsonl"))
    router.register(
        Intent.SELECT,
        ToolDefinition(
            name="dangerous.example",
            risk=ToolRisk.HIGH,
            handler=lambda context: context,
        ),
    )
    proposed = router.route(DecodedIntent.create(Intent.SELECT, 0.90, "test"))

    with pytest.raises(ValueError, match="confirmation channel"):
        router.confirm(proposed.action_id, channel="eeg")
