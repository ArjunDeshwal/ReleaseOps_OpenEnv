"""ReleaseOps Environment: production change review simulator."""

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from releaseops_env.models import (
    ReleaseAction,
    ReleaseObservation,
    ReleaseState,
    RiskSignal,
    ToolResult,
)

TASKS_DIR = Path(__file__).parent.parent / "tasks"
INCIDENTS_DB = Path(__file__).parent.parent / "data" / "incidents.db"


class ReleaseOpsEnvironment(Environment):
    """
    Simulates production change review and rollout governance.

    The agent investigates a proposed change using tools (inspect, query,
    search) and makes a release decision. Scenarios are pre-authored JSON
    files loaded at reset(). No live computation — all reactions are array
    lookups on pre-authored telemetry curves.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state = ReleaseState(episode_id=str(uuid4()))
        self._scenario: dict = {}
        # Tracks discovered RiskSignal objects (signal_id → RiskSignal)
        self._discovered_risks: dict[str, RiskSignal] = {}

    # ── RESET ──────────────────────────────────────────────

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        task_id = kwargs.get("task_id", "easy_001")

        self._scenario = self._load_scenario(task_id)
        self._discovered_risks = {}

        self._state = ReleaseState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            rollout_phase="precheck",
        )

        return ReleaseObservation(
            done=False,
            reward=0.0,
            task_id=task_id,
            change_summary=self._scenario["change"]["description"],
            known_risk_signals=[],
            last_tool_result=None,
            allowed_actions=self._get_allowed_actions(),
            rollout_phase="precheck",
            time_remaining=self._max_steps,
            cumulative_reward=0.0,
            metadata={"status": "ready", "task_id": task_id},
        )

    # ── STEP ───────────────────────────────────────────────

    def step(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        if not isinstance(action, ReleaseAction):
            return ReleaseObservation(
                done=False,
                reward=-0.05,
                task_id=self._state.task_id,
                change_summary=self._scenario.get("change", {}).get("description", ""),
                allowed_actions=self._get_allowed_actions(),
                rollout_phase=self._state.rollout_phase,
                time_remaining=self._max_steps - self._state.step_count,
                cumulative_reward=self._state.cumulative_reward,
                metadata={
                    "error": f"Expected ReleaseAction, got {type(action).__name__}"
                },
            )

        self._state.step_count += 1
        self._state.steps_in_phase += 1
        self._state.actions_taken.append(action.action_type)

        reward = 0.0
        tool_result = None
        new_risks: list[RiskSignal] = []

        try:
            if action.action_type == "inspect_change":
                reward, tool_result, new_risks = self._handle_inspect_change(action)
            elif action.action_type == "inspect_services":
                reward, tool_result, new_risks = self._handle_inspect_services(action)
            elif action.action_type == "inspect_dependencies":
                reward, tool_result, new_risks = self._handle_inspect_dependencies(
                    action
                )
            elif action.action_type == "search_incidents":
                reward, tool_result, new_risks = self._handle_search_incidents(action)
            elif action.action_type == "check_policy":
                reward, tool_result, new_risks = self._handle_check_policy(action)
            elif action.action_type == "query_telemetry":
                reward, tool_result, new_risks = self._handle_query_telemetry(action)
            elif action.action_type == "request_artifact":
                reward, tool_result, new_risks = self._handle_request_artifact(action)
            elif action.action_type == "control_rollout":
                reward, tool_result, new_risks = self._handle_control_rollout(action)
            elif action.action_type == "submit_decision":
                return self._handle_submit_decision(action)
            else:
                reward = -0.05
                tool_result = ToolResult(
                    tool_name=action.action_type,
                    success=False,
                    content=f"Unknown action type: {action.action_type}",
                )
        except Exception as e:
            reward = -0.05
            tool_result = ToolResult(
                tool_name=action.action_type,
                success=False,
                content=f"Error handling action: {e}",
            )

        # Register new risk signals
        for risk in new_risks:
            if risk.signal_id not in self._discovered_risks:
                self._discovered_risks[risk.signal_id] = risk
                self._state.risk_signals_found.append(risk.signal_id)

        self._state.cumulative_reward += reward

        # Check max steps (timeout)
        done = self._state.step_count >= self._max_steps
        if done and not self._state.terminal:
            self._state.terminal = True
            self._state.rollout_phase = "terminal"
            timeout_penalty = -0.20
            reward += timeout_penalty
            self._state.cumulative_reward += timeout_penalty

        return ReleaseObservation(
            done=done,
            reward=reward,
            task_id=self._state.task_id,
            change_summary=self._scenario["change"]["description"],
            known_risk_signals=self._get_visible_risks(),
            last_tool_result=tool_result,
            allowed_actions=self._get_allowed_actions(),
            rollout_phase=self._state.rollout_phase,
            time_remaining=max(0, self._max_steps - self._state.step_count),
            cumulative_reward=self._state.cumulative_reward,
            metadata={"step": self._state.step_count},
        )

    # ── STATE ──────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    # ── CONSTANTS ──────────────────────────────────────────

    @property
    def _max_steps(self) -> int:
        return self._scenario.get("max_steps", 12)

    # ── SCENARIO LOADING ───────────────────────────────────

    def _load_scenario(self, task_id: str) -> dict:
        """Load all 6 JSON files for a scenario into one dict."""
        task_dir = TASKS_DIR / task_id
        scenario: dict[str, Any] = {}
        for name in [
            "services",
            "change",
            "telemetry_curves",
            "incidents",
            "policy",
            "ground_truth",
        ]:
            with open(task_dir / f"{name}.json") as f:
                scenario[name] = json.load(f)
        scenario["max_steps"] = scenario.get("ground_truth", {}).get("max_steps", 12)
        return scenario

    # ── ACTION HANDLERS ────────────────────────────────────
    # Each returns (reward, ToolResult, list[RiskSignal])

    def _handle_inspect_change(self, action: ReleaseAction):
        section = action.section or "diff"
        evidence_key = f"inspected_{section}"
        change = self._scenario["change"]

        if evidence_key in self._state.evidence_gathered:
            return (
                -0.05,
                ToolResult(
                    tool_name="inspect_change",
                    success=True,
                    content=f"Already inspected '{section}'. No new information.",
                ),
                [],
            )

        self._state.evidence_gathered.append(evidence_key)

        if section == "diff":
            content = change.get("diff_summary", "No diff available.")
            payload = {"diff": content, "files": change.get("files_changed", [])}
        elif section == "tests":
            content = json.dumps(change.get("test_results", {}), indent=2)
            payload = change.get("test_results", {})
        elif section == "approvals":
            content = json.dumps(change.get("approvals", {}), indent=2)
            payload = change.get("approvals", {})
        elif section == "files_changed":
            files = change.get("files_changed", [])
            content = "\n".join(files)
            payload = {"files": files}
        else:
            content = "Unknown section."
            payload = {}

        new_risks = self._discover_risks_from_section(section)
        reward = 0.10 if evidence_key in self._get_required_evidence() else 0.0

        return (
            reward,
            ToolResult(
                tool_name="inspect_change",
                success=True,
                content=content,
                structured_payload=payload,
            ),
            new_risks,
        )

    def _handle_inspect_services(self, action: ReleaseAction):
        service_name = action.service
        if not service_name:
            return (
                -0.02,
                ToolResult(
                    tool_name="inspect_services",
                    success=False,
                    content="Must specify service. Use inspect_services with a service name.",
                ),
                [],
            )

        services = self._scenario["services"].get("services", {})
        if service_name not in services:
            available = list(services.keys())
            return (
                0.0,
                ToolResult(
                    tool_name="inspect_services",
                    success=False,
                    content=f"Service '{service_name}' not found. Available: {available}",
                ),
                [],
            )

        evidence_key = f"inspected_service_{service_name}"
        is_new = evidence_key not in self._state.evidence_gathered
        if is_new:
            self._state.evidence_gathered.append(evidence_key)
            # Also mark the generic "inspected_services" evidence key (first service inspected)
            if "inspected_services" not in self._state.evidence_gathered:
                self._state.evidence_gathered.append("inspected_services")

        svc = services[service_name]
        new_risks: list[RiskSignal] = []
        if is_new:
            if svc.get("current_health") == "degraded":
                new_risks.append(
                    RiskSignal(
                        signal_id=f"degraded_{service_name}",
                        category="dependency_health",
                        severity="medium",
                        summary=f"{service_name} is currently degraded",
                        numeric_value=svc.get("metrics", {}).get("error_rate"),
                    )
                )
            # Detect connection capacity risk for database services
            max_conn = svc.get("max_connections")
            cur_conn = svc.get("current_connections")
            if max_conn and cur_conn:
                change_services = self._scenario["change"].get("services_touched", [])
                # Find pool size increase in the diff
                diff = self._scenario["change"].get("diff_summary", "").lower()
                if ("connection pool" in diff or "max_connections" in diff) and any(
                    s in service_name for s in ["postgres", "mysql", "db", "database"]
                ):
                    replicas = None
                    for svc_name, svc_data in self._scenario["services"].get("services", {}).items():
                        if svc_data.get("replicas"):
                            replicas = svc_data["replicas"]
                            break
                    # Try to estimate total connections (pool_size × replicas)
                    new_risks.append(
                        RiskSignal(
                            signal_id="db_connection_exhaustion",
                            category="resource_saturation",
                            severity="high",
                            summary=(
                                f"{service_name} at {cur_conn}/{max_conn} connections "
                                f"({int(cur_conn/max_conn*100)}% capacity). "
                                f"Pool increase may exhaust server connection limit."
                            ),
                            numeric_value=cur_conn,
                            threshold=max_conn,
                        )
                    )

        reward = 0.05 if is_new else -0.05
        return (
            reward,
            ToolResult(
                tool_name="inspect_services",
                success=True,
                content=json.dumps(svc, indent=2),
                structured_payload=svc,
            ),
            new_risks,
        )

    def _handle_inspect_dependencies(self, action: ReleaseAction):
        evidence_key = "inspected_dependencies"
        is_new = evidence_key not in self._state.evidence_gathered
        if is_new:
            self._state.evidence_gathered.append(evidence_key)

        deps = self._scenario["services"].get("dependencies", {})
        blast = self._scenario["change"].get("blast_radius", [])
        payload = {
            "dependencies": deps,
            "blast_radius": blast,
            "services_touched": self._scenario["change"].get("services_touched", []),
        }

        new_risks: list[RiskSignal] = []
        if is_new and len(blast) >= 3:
            new_risks.append(
                RiskSignal(
                    signal_id="large_blast_radius",
                    category="blast_radius",
                    severity="high",
                    summary=f"Change affects {len(blast)} services: {blast}",
                )
            )

        reward = (
            0.10
            if is_new and "inspected_dependencies" in self._get_required_evidence()
            else (0.02 if is_new else -0.05)
        )
        return (
            reward,
            ToolResult(
                tool_name="inspect_dependencies",
                success=True,
                content=json.dumps(payload, indent=2),
                structured_payload=payload,
            ),
            new_risks,
        )

    def _handle_search_incidents(self, action: ReleaseAction):
        keywords = action.keywords or []
        evidence_key = "checked_incidents"
        is_new = evidence_key not in self._state.evidence_gathered
        if is_new:
            self._state.evidence_gathered.append(evidence_key)

        # ── Scenario incidents (used for risk signal emission) ──────────────
        scenario_incidents = self._scenario["incidents"].get("incidents", [])
        if keywords:
            matched_scenario = [
                inc
                for inc in scenario_incidents
                if any(kw.lower() in json.dumps(inc).lower() for kw in keywords)
            ]
        else:
            matched_scenario = scenario_incidents

        # ── Real incident database (SQLite) ────────────────────────���────────
        db_results = self._query_incident_db(keywords)

        # Combine: scenario incidents first (with similarity scores), then
        # real-world incidents from the database for broader context.
        all_results = matched_scenario + db_results
        content = (
            json.dumps(all_results, indent=2) if all_results else "No matching incidents found."
        )

        new_risks: list[RiskSignal] = []
        if is_new:
            for inc in matched_scenario:
                if inc.get("similarity_to_current", 0) > 0.6:
                    new_risks.append(
                        RiskSignal(
                            signal_id=f"similar_incident_{inc['incident_id']}",
                            category="incident_similarity",
                            severity="high",
                            summary=f"Similar incident: {inc['title']} (similarity: {inc['similarity_to_current']})",
                        )
                    )

        reward = (
            0.10
            if is_new and "checked_incidents" in self._get_required_evidence()
            else (0.02 if is_new else -0.05)
        )
        if new_risks and any(r.severity == "high" for r in new_risks):
            reward += 0.05

        return (
            reward,
            ToolResult(
                tool_name="search_incidents",
                success=True,
                content=content,
                structured_payload={
                    "scenario_incidents": matched_scenario,
                    "historical_incidents": db_results,
                    "count": len(all_results),
                },
            ),
            new_risks,
        )

    def _query_incident_db(self, keywords: list[str]) -> list[dict]:
        """Query the real incident database for historically similar incidents."""
        if not INCIDENTS_DB.exists():
            return []
        try:
            conn = sqlite3.connect(f"file:{INCIDENTS_DB}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            if not keywords:
                rows = conn.execute(
                    "SELECT id, title, company, category, severity, description, root_cause, url "
                    "FROM incidents ORDER BY severity DESC LIMIT 5"
                ).fetchall()
            else:
                conditions = " OR ".join(
                    ["title LIKE ? OR description LIKE ? OR keywords LIKE ?"] * len(keywords)
                )
                params: list[str] = []
                for kw in keywords:
                    like = f"%{kw.lower()}%"
                    params.extend([like, like, like])
                rows = conn.execute(
                    f"SELECT id, title, company, category, severity, description, root_cause, url "
                    f"FROM incidents WHERE {conditions} "
                    f"ORDER BY severity DESC LIMIT 8",
                    params,
                ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def _handle_check_policy(self, action: ReleaseAction):
        evidence_key = "checked_policy"
        is_new = evidence_key not in self._state.evidence_gathered
        if is_new:
            self._state.evidence_gathered.append(evidence_key)

        policy = self._scenario["policy"]
        new_risks = self._evaluate_policy_violations(policy) if is_new else []

        reward = (
            0.10
            if is_new and "checked_policy" in self._get_required_evidence()
            else (0.02 if is_new else -0.05)
        )
        if new_risks and any(r.severity in ("high", "critical") for r in new_risks):
            reward += 0.05

        return (
            reward,
            ToolResult(
                tool_name="check_policy",
                success=True,
                content=json.dumps(policy, indent=2),
                structured_payload=policy,
            ),
            new_risks,
        )

    def _handle_query_telemetry(self, action: ReleaseAction):
        metric = action.metric or "p99"
        service = action.service
        window = action.window or "5m"

        if not service:
            return (
                -0.02,
                ToolResult(
                    tool_name="query_telemetry",
                    success=False,
                    content="Must specify service.",
                ),
                [],
            )

        curves = self._scenario["telemetry_curves"]
        phase = self._state.rollout_phase

        curve_key = {
            "precheck": "pre_canary",
            "canary": "during_canary",
            "promoted": "after_promotion",
            "rolled_back": "after_rollback",
            "paused": "during_canary",
        }.get(phase, "pre_canary")

        phase_data = curves.get(curve_key, {})
        service_data = phase_data.get(service, {})
        # Support both "p99" and "p99_ms" key variants in telemetry JSON
        metric_data = service_data.get(metric, [])
        if not metric_data and metric in ("p50", "p95", "p99"):
            metric_data = service_data.get(f"{metric}_ms", [])

        if not metric_data:
            return (
                0.0,
                ToolResult(
                    tool_name="query_telemetry",
                    success=True,
                    content=f"No {metric} data for {service} in {phase} phase.",
                    structured_payload={
                        "metric": metric,
                        "service": service,
                        "values": [],
                    },
                ),
                [],
            )

        idx = min(self._state.steps_in_phase, len(metric_data) - 1)
        current_value = metric_data[idx]

        evidence_key = f"telemetry_{service}_{metric}_{phase}"
        is_new = evidence_key not in self._state.evidence_gathered
        if is_new:
            self._state.evidence_gathered.append(evidence_key)

        new_risks: list[RiskSignal] = []
        if is_new and isinstance(current_value, (int, float)):
            baseline = metric_data[0]
            if isinstance(baseline, (int, float)) and baseline > 0:
                ratio = current_value / baseline
                if ratio > 2.0:
                    new_risks.append(
                        RiskSignal(
                            signal_id=f"{metric}_breach_{service}",
                            category="latency" if "p" in metric else "error_rate",
                            severity="critical" if ratio > 3.0 else "high",
                            summary=f"{service} {metric} is {ratio:.1f}x baseline ({current_value} vs {baseline})",
                            numeric_value=current_value,
                            threshold=baseline * 2,
                        )
                    )

        reward = 0.05 if is_new else -0.05
        if new_risks:
            reward += 0.15

        return (
            reward,
            ToolResult(
                tool_name="query_telemetry",
                success=True,
                content=f"{service} {metric} ({window}, {phase} phase): {current_value}",
                structured_payload={
                    "metric": metric,
                    "service": service,
                    "window": window,
                    "phase": phase,
                    "current_value": current_value,
                    "history": metric_data[: idx + 1],
                },
            ),
            new_risks,
        )

    def _handle_request_artifact(self, action: ReleaseAction):
        artifact_type = action.artifact_type
        if not artifact_type:
            return (
                -0.02,
                ToolResult(
                    tool_name="request_artifact",
                    success=False,
                    content="Must specify artifact_type.",
                ),
                [],
            )

        change = self._scenario["change"]
        if artifact_type == "load_test":
            data = change.get("test_results", {}).get("load_test")
        elif artifact_type == "approval":
            data = change.get("approvals", {})
        elif artifact_type == "rollback_plan":
            data = change.get("rollback_plan")
        else:
            data = change.get(artifact_type)

        is_new = artifact_type not in self._state.artifacts_requested
        if is_new:
            self._state.artifacts_requested.append(artifact_type)

        if data is None:
            missing_code = f"MISSING_{artifact_type.upper()}"
            new_risks: list[RiskSignal] = []
            if is_new:
                new_risks.append(
                    RiskSignal(
                        signal_id=f"missing_{artifact_type}",
                        category=(
                            "policy_violation"
                            if artifact_type in ("load_test", "approval")
                            else "compliance"
                        ),
                        severity="high",
                        summary=f"Required artifact '{artifact_type}' is not available",
                    )
                )
            reward = (
                0.20
                if is_new and missing_code in self._get_required_reason_codes()
                else (0.05 if is_new else -0.05)
            )
            return (
                reward,
                ToolResult(
                    tool_name="request_artifact",
                    success=False,
                    content=f"Artifact '{artifact_type}' is not available.",
                ),
                new_risks,
            )

        content = (
            json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)
        )
        reward = 0.05 if is_new else -0.05
        return (
            reward,
            ToolResult(
                tool_name="request_artifact",
                success=True,
                content=content,
                structured_payload={"artifact_type": artifact_type, "data": data},
            ),
            [],
        )

    def _handle_control_rollout(self, action: ReleaseAction):
        decision = action.decision
        if not decision:
            return (
                -0.02,
                ToolResult(
                    tool_name="control_rollout",
                    success=False,
                    content="Must specify decision.",
                ),
                [],
            )

        phase = self._state.rollout_phase
        valid_transitions = {
            "precheck": ["start_canary"],
            "canary": ["pause", "promote", "rollback"],
            "paused": ["promote", "rollback"],
        }

        if decision not in valid_transitions.get(phase, []):
            allowed = valid_transitions.get(phase, [])
            return (
                -0.10,
                ToolResult(
                    tool_name="control_rollout",
                    success=False,
                    content=f"Cannot '{decision}' from phase '{phase}'. Allowed: {allowed}",
                ),
                [],
            )

        prev_phase = phase
        reward = 0.0
        new_risks: list[RiskSignal] = []

        if decision == "start_canary":
            self._state.rollout_phase = "canary"
            self._state.steps_in_phase = 0
            gt = self._scenario["ground_truth"]
            reward = (
                0.25
                if "CANARY_REQUIRED" in gt.get("required_reason_codes", [])
                else 0.05
            )

        elif decision == "pause":
            self._state.rollout_phase = "paused"
            reward = 0.05

        elif decision == "promote":
            self._state.rollout_phase = "promoted"
            self._state.steps_in_phase = 0
            gt = self._scenario["ground_truth"]
            if gt.get("optimal_decision") in ("block", "request_changes"):
                reward = -0.40
                new_risks.append(
                    RiskSignal(
                        signal_id="catastrophic_promotion",
                        category="blast_radius",
                        severity="critical",
                        summary="Promoted a change that should have been blocked!",
                    )
                )
                self._state.terminal = True
                self._state.rollout_phase = "terminal"
            else:
                reward = 0.15

        elif decision == "rollback":
            self._state.rollout_phase = "rolled_back"
            self._state.steps_in_phase = 0
            if phase == "canary" and self._has_observed_bad_telemetry():
                reward = 0.30
            else:
                reward = 0.05

        return (
            reward,
            ToolResult(
                tool_name="control_rollout",
                success=True,
                content=f"Rollout phase: {prev_phase} → {self._state.rollout_phase}",
                structured_payload={
                    "previous_phase": prev_phase,
                    "new_phase": self._state.rollout_phase,
                },
            ),
            new_risks,
        )

    def _handle_submit_decision(self, action: ReleaseAction) -> ReleaseObservation:
        """Terminal action — grades the full trajectory."""
        final_decision = action.final_decision
        reason_codes = action.reason_codes or []

        if not final_decision:
            return ReleaseObservation(
                done=False,
                reward=-0.05,
                task_id=self._state.task_id,
                change_summary=self._scenario["change"]["description"],
                known_risk_signals=self._get_visible_risks(),
                allowed_actions=self._get_allowed_actions(),
                rollout_phase=self._state.rollout_phase,
                time_remaining=max(0, self._max_steps - self._state.step_count),
                cumulative_reward=self._state.cumulative_reward,
                metadata={"error": "Must specify final_decision."},
            )

        gt = self._scenario["ground_truth"]
        grader_result = self._grade(final_decision, reason_codes, gt)

        self._state.terminal = True
        self._state.rollout_phase = "terminal"
        self._state.final_score = grader_result["score"]

        decision_reward = (
            0.40
            if grader_result["decision_correct"]
            else (0.20 if grader_result["decision_partial"] else -0.15)
        )
        decision_reward += 0.10 * grader_result["risk_signal_discovery"]

        self._state.cumulative_reward += decision_reward

        return ReleaseObservation(
            done=True,
            reward=decision_reward,
            task_id=self._state.task_id,
            change_summary=self._scenario["change"]["description"],
            known_risk_signals=self._get_visible_risks(),
            last_tool_result=ToolResult(
                tool_name="submit_decision",
                success=True,
                content=f"Decision: {final_decision}. Score: {grader_result['score']:.3f}",
                structured_payload=grader_result,
            ),
            allowed_actions=[],
            rollout_phase="terminal",
            time_remaining=0,
            cumulative_reward=self._state.cumulative_reward,
            final_score=grader_result["score"],
            grader_breakdown=grader_result["breakdown"],
            metadata={"final_decision": final_decision, "reason_codes": reason_codes},
        )

    # ── GRADER ─────────────────────────────────────────────

    def _grade(
        self, final_decision: str, reason_codes: list[str], ground_truth: dict
    ) -> dict:
        """Deterministic grading formula. Returns score in [0.0, 1.0].

        Components:
          0.35  evidence_coverage     — did the agent inspect the right sources?
          0.25  risk_signal_discovery — did the agent trigger discovery of key signals?
          0.30  decision_correctness  — was the final decision correct?
          0.10  efficiency            — did the agent avoid wasted steps?

        risk_signal_discovery is objective: it checks state.risk_signals_found
        (signal_ids actually emitted by the environment during the episode)
        against required_risk_signals in ground_truth. This mirrors the
        Calendar env SQL verifier pattern — the environment measures what the
        agent observed, not what strings the agent typed.
        """

        # 1. Evidence coverage (weight 0.35)
        required_ev = set(ground_truth.get("required_evidence", []))
        gathered_ev = set(self._state.evidence_gathered)
        evidence_coverage = len(required_ev & gathered_ev) / max(len(required_ev), 1)

        # 2. Risk signal discovery (weight 0.25) — objective: checks env-emitted signal_ids
        required_signals = set(ground_truth.get("required_risk_signals", []))
        discovered_signals = set(self._state.risk_signals_found)
        if required_signals:
            signal_discovery = len(required_signals & discovered_signals) / len(required_signals)
        else:
            # No required signals (e.g. trivial approve scenario) — full credit
            signal_discovery = 1.0

        # 3. Decision correctness (weight 0.30)
        optimal = ground_truth.get("optimal_decision", "")
        acceptable = set(ground_truth.get("acceptable_decisions", [optimal]))
        if final_decision == optimal:
            decision_score = 1.0
        elif final_decision in acceptable:
            decision_score = 0.5
        else:
            decision_score = 0.0

        # 4. Efficiency (weight 0.10)
        usage = self._state.step_count / self._max_steps
        if 0.3 <= usage <= 0.7:
            efficiency = 1.0
        elif usage < 0.3:
            efficiency = usage / 0.3
        else:
            efficiency = max(0.0, 1.0 - (usage - 0.7) / 0.3)

        # 5. Forbidden action penalty
        forbidden = ground_truth.get("forbidden_actions", [])
        took_forbidden = any(fa in self._state.actions_taken for fa in forbidden)
        forbidden_penalty = 0.3 if took_forbidden else 0.0

        raw_score = (
            0.35 * evidence_coverage
            + 0.25 * signal_discovery
            + 0.30 * decision_score
            + 0.10 * efficiency
        )
        score = max(0.0, min(1.0, raw_score - forbidden_penalty))

        return {
            "score": round(score, 3),
            "decision_correct": final_decision == optimal,
            "decision_partial": final_decision in acceptable,
            "risk_signal_discovery": round(signal_discovery, 3),
            "breakdown": {
                "evidence_coverage": round(evidence_coverage, 3),
                "risk_signal_discovery": round(signal_discovery, 3),
                "decision_correctness": round(decision_score, 3),
                "efficiency": round(efficiency, 3),
                "forbidden_penalty": round(forbidden_penalty, 3),
            },
        }

    # ── HELPERS ─────────────────────────────────────────────

    def _get_allowed_actions(self) -> list[str]:
        phase = self._state.rollout_phase
        if phase == "terminal":
            return []

        base = [
            "inspect_change",
            "inspect_services",
            "inspect_dependencies",
            "search_incidents",
            "check_policy",
            "query_telemetry",
            "request_artifact",
            "submit_decision",
        ]
        if phase in ("precheck", "canary", "paused"):
            base.append("control_rollout")
        return base

    def _get_visible_risks(self) -> list[RiskSignal]:
        return list(self._discovered_risks.values())

    def _get_required_evidence(self) -> set[str]:
        return set(self._scenario.get("ground_truth", {}).get("required_evidence", []))

    def _get_required_reason_codes(self) -> set[str]:
        return set(
            self._scenario.get("ground_truth", {}).get("required_reason_codes", [])
        )

    def _discover_risks_from_section(self, section: str) -> list[RiskSignal]:
        """Surface-level risks visible from inspecting a change section."""
        risks: list[RiskSignal] = []
        change = self._scenario["change"]

        if section == "tests":
            test_results = change.get("test_results", {})
            integration = test_results.get("integration", {})
            if integration.get("failed", 0) > 0:
                risks.append(
                    RiskSignal(
                        signal_id="integration_test_failure",
                        category="test_coverage",
                        severity="high",
                        summary=f"Integration test failures: {integration.get('failures', [])}",
                    )
                )
            if test_results.get("load_test") is None:
                risks.append(
                    RiskSignal(
                        signal_id="missing_load_test",
                        category="test_coverage",
                        severity="medium",
                        summary="No load test results available",
                    )
                )
            elif integration.get("failed", 0) == 0:
                risks.append(
                    RiskSignal(
                        signal_id="all_tests_passed",
                        category="test_coverage",
                        severity="low",
                        summary="All unit, integration, and load tests passed",
                    )
                )

        elif section == "approvals":
            approvals = change.get("approvals", {})
            pending = [role for role, status in approvals.items() if status == "pending"]
            for role in pending:
                risks.append(
                    RiskSignal(
                        signal_id=f"missing_approval_{role}",
                        category="missing_approval",
                        severity="high",
                        summary=f"Approval pending: {role}",
                    )
                )
            if not pending and approvals:
                risks.append(
                    RiskSignal(
                        signal_id="all_approvals_granted",
                        category="compliance",
                        severity="low",
                        summary="All required approvals are in place",
                    )
                )

        elif section == "diff":
            diff = change.get("diff_summary", "").lower()
            if any(kw in diff for kw in ("backward-compatible", "backward compatible", "concurrently", "no code changes")):
                risks.append(
                    RiskSignal(
                        signal_id="backward_compatible_change",
                        category="compliance",
                        severity="low",
                        summary="Change is backward-compatible — existing interfaces and data are unaffected",
                    )
                )
            if any(
                kw in diff for kw in ("hot path", "sync", "synchronous", "blocking")
            ):
                risks.append(
                    RiskSignal(
                        signal_id="hot_path_sync_io",
                        category="latency",
                        severity="high",
                        summary="Change introduces synchronous I/O on a hot path",
                    )
                )
            if "retry" in diff and any(kw in diff for kw in ("concurrency", "queue", "consumer")):
                risks.append(
                    RiskSignal(
                        signal_id="retry_amplification_risk",
                        category="resource_saturation",
                        severity="high",
                        summary="Increased retry attempts combined with higher concurrency may cause thundering herd under partial failure",
                    )
                )
            if any(kw in diff for kw in ("connection pool", "max_connections", "pool size")):
                risks.append(
                    RiskSignal(
                        signal_id="missing_capacity_check",
                        category="resource_saturation",
                        severity="high",
                        summary="Connection pool increase without documented capacity validation against database server limits",
                    )
                )
            # Removing a protection mechanism (rate limiting, circuit breaker, etc.)
            # without a replacement is categorically critical — not fixable by adding approvals
            if "rate limit" in diff and any(kw in diff for kw in ("remov", "disabl", "delet")):
                risks.append(
                    RiskSignal(
                        signal_id="protection_mechanism_removal",
                        category="policy_violation",
                        severity="critical",
                        summary="Change removes rate limiting without introducing a replacement throttling mechanism — unthrottled traffic can cascade to downstream services",
                    )
                )

        return risks

    def _evaluate_policy_violations(self, policy: dict) -> list[RiskSignal]:
        """Check for policy violations given current scenario state."""
        risks: list[RiskSignal] = []
        change = self._scenario["change"]
        services = self._scenario["services"].get("services", {})

        for rule in policy.get("rules", []):
            rule_text = rule.get("rule", "").lower()
            rule_id = rule.get("id", "")

            # Rule: critical-tier services need owner approval
            if "critical" in rule_text and "approval" in rule_text:
                for svc_name in change.get("services_touched", []):
                    svc = services.get(svc_name, {})
                    if svc.get("tier") == "critical":
                        if (
                            change.get("approvals", {}).get("service_owner")
                            == "pending"
                        ):
                            risks.append(
                                RiskSignal(
                                    signal_id="policy_missing_owner_approval",
                                    category="policy_violation",
                                    severity="high",
                                    summary=f"{rule_id}: Critical service '{svc_name}' requires owner approval (pending)",
                                )
                            )

            # Rule: peak traffic window
            if "peak" in rule_text and "traffic" in rule_text:
                if policy.get("traffic_window") == "peak":
                    risks.append(
                        RiskSignal(
                            signal_id="policy_peak_traffic",
                            category="policy_violation",
                            severity="high",
                            summary=f"{rule_id}: Current time is peak traffic window ({policy.get('current_time', '?')})",
                        )
                    )

            # Rule: load test required
            if "load test" in rule_text:
                if change.get("test_results", {}).get("load_test") is None:
                    risks.append(
                        RiskSignal(
                            signal_id="policy_missing_load_test",
                            category="policy_violation",
                            severity="high",
                            summary=f"{rule_id}: Load test required but not provided",
                        )
                    )

            # Rule: canary required for schema/migration or retry/queue changes
            if "canary" in rule_text and any(kw in rule_text for kw in ("schema", "migration", "retry", "queue", "concurrency")):
                diff = change.get("diff_summary", "").lower()
                is_schema = any(kw in diff for kw in ("migration", "schema", "index", "create index", "alter table"))
                is_retry_queue = any(kw in diff for kw in ("retry", "concurrency", "queue consumer"))
                if is_schema or is_retry_queue:
                    risks.append(
                        RiskSignal(
                            signal_id="canary_required_policy",
                            category="policy_violation",
                            severity="high",
                            summary=f"{rule_id}: Change type requires canary deployment before full promotion",
                        )
                    )

        return risks

    def _has_observed_bad_telemetry(self) -> bool:
        """True if agent has seen a metric threshold breach during canary."""
        return any("breach" in sig_id for sig_id in self._state.risk_signals_found)
