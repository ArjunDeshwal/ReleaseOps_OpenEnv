"""ReleaseOps-Env: Typed models for production change review environment."""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State


# ──────────────────────────────────────────────
# ACTION MODEL
# ──────────────────────────────────────────────


class ReleaseAction(Action):
    """
    Single action type with discriminator routing.

    action_type determines which fields are used:
    - "inspect_change": section
    - "inspect_services": service
    - "inspect_dependencies": service
    - "search_incidents": keywords
    - "check_policy": (no extra fields)
    - "query_telemetry": metric, service, window
    - "request_artifact": artifact_type
    - "control_rollout": decision
    - "submit_decision": final_decision, reason_codes
    """

    action_type: Literal[
        "inspect_change",
        "inspect_services",
        "inspect_dependencies",
        "search_incidents",
        "check_policy",
        "query_telemetry",
        "request_artifact",
        "control_rollout",
        "submit_decision",
    ]

    # inspect_change
    section: Optional[Literal["diff", "tests", "approvals", "files_changed"]] = None

    # inspect_services, inspect_dependencies, query_telemetry
    service: Optional[str] = None

    # search_incidents
    keywords: Optional[list[str]] = None

    # query_telemetry
    metric: Optional[
        Literal["p50", "p95", "p99", "error_rate", "queue_depth", "cpu", "rps"]
    ] = None
    window: Optional[Literal["5m", "15m", "1h"]] = None

    # request_artifact
    artifact_type: Optional[
        Literal[
            "load_test",
            "rollback_plan",
            "approval",
            "runbook",
            "security_review",
            "compliance_check",
        ]
    ] = None

    # control_rollout
    decision: Optional[Literal["start_canary", "pause", "promote", "rollback"]] = None

    # submit_decision
    final_decision: Optional[
        Literal["approve", "request_changes", "block", "escalate"]
    ] = None
    reason_codes: Optional[list[str]] = None


# ──────────────────────────────────────────────
# RISK SIGNAL (sub-model for observations)
# ──────────────────────────────────────────────


class RiskSignal(BaseModel):
    """A discovered risk signal surfaced through investigation."""

    signal_id: str
    category: Literal[
        "latency",
        "error_rate",
        "dependency_health",
        "test_coverage",
        "blast_radius",
        "policy_violation",
        "missing_approval",
        "incident_similarity",
        "compliance",
        "resource_saturation",
    ]
    severity: Literal["low", "medium", "high", "critical"]
    summary: str
    numeric_value: Optional[float] = None
    threshold: Optional[float] = None


# ──────────────────────────────────────────────
# TOOL RESULT (sub-model for observations)
# ──────────────────────────────────────────────


class ToolResult(BaseModel):
    """Result from the last tool/action invocation."""

    tool_name: str
    success: bool
    content: str
    structured_payload: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────
# OBSERVATION MODEL
# ──────────────────────────────────────────────


class ReleaseObservation(Observation):
    """
    What the agent sees after each step.

    Starts sparse (just change_summary) and fills in as agent investigates.
    """

    # Context
    task_id: str = ""
    change_summary: str = ""

    # Discovery state
    known_risk_signals: list[RiskSignal] = Field(default_factory=list)
    last_tool_result: Optional[ToolResult] = None

    # Navigation
    allowed_actions: list[str] = Field(default_factory=list)
    rollout_phase: str = "precheck"

    # Budget
    time_remaining: int = 12
    cumulative_reward: float = 0.0

    # Terminal
    final_score: Optional[float] = None
    grader_breakdown: Optional[dict[str, float]] = None


# ──────────────────────────────────────────────
# STATE MODEL
# ──────────────────────────────────────────────


class ReleaseState(State):
    """
    Full environment state (server-side).

    Contains both public trajectory info and hidden simulator truth.
    The agent never sees hidden_* fields — only the Observation is sent.
    """

    # Public tracking
    task_id: str = ""
    rollout_phase: Literal[
        "precheck", "canary", "paused", "promoted", "rolled_back", "terminal"
    ] = "precheck"
    steps_in_phase: int = 0
    evidence_gathered: list[str] = Field(default_factory=list)
    risk_signals_found: list[str] = Field(default_factory=list)
    artifacts_requested: list[str] = Field(default_factory=list)
    actions_taken: list[str] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    terminal: bool = False
    final_score: Optional[float] = None
