"""
Heuristic baseline agent for ReleaseOps-Env.

A rule-based agent that follows a fixed investigation strategy.
Requires NO API key — fully deterministic and reproducible.
Used by the /baseline endpoint and as a reference implementation.
"""

import json
from pathlib import Path

from releaseops_env.models import ReleaseAction
from server.releaseops_environment import ReleaseOpsEnvironment

TASKS_DIR = Path(__file__).parent.parent / "tasks"


def _get_all_task_ids() -> list[str]:
    """Discover all task directories that contain ground_truth.json."""
    tasks = []
    for d in sorted(TASKS_DIR.iterdir()):
        if d.is_dir() and (d / "ground_truth.json").exists():
            tasks.append(d.name)
    return tasks


def play_heuristic_episode(env: ReleaseOpsEnvironment, task_id: str) -> dict:
    """
    Play one episode using a fixed heuristic strategy:

    1. Inspect diff
    2. Inspect tests
    3. Inspect approvals
    4. Inspect dependencies
    5. Search incidents (broad keywords)
    6. Check policy
    7. Analyze gathered risks and decide

    Returns dict with score, breakdown, trajectory.
    """
    obs = env.reset(task_id=task_id)
    trajectory = []

    # ── Phase 1: Gather evidence ────────────────────────────────────

    early_steps = [
        ReleaseAction(action_type="inspect_change", section="diff"),
        ReleaseAction(action_type="inspect_change", section="tests"),
        ReleaseAction(action_type="inspect_change", section="approvals"),
        ReleaseAction(action_type="inspect_dependencies"),
    ]

    blast_radius: list[str] = []
    for action in early_steps:
        obs = env.step(action)
        trajectory.append(action.action_type)
        # Capture blast radius right after inspect_dependencies
        if action.action_type == "inspect_dependencies" and obs.last_tool_result:
            blast_radius = obs.last_tool_result.structured_payload.get("blast_radius", [])
        if obs.done:
            break

    if obs.done:
        return _build_result(task_id, obs, trajectory)

    # ── Phase 1b: Inspect services in blast radius ──────────────────
    for svc_name in blast_radius[:2]:
        if obs.done:
            break
        obs = env.step(ReleaseAction(action_type="inspect_services", service=svc_name))
        trajectory.append(f"inspect_services:{svc_name}")

    if obs.done:
        return _build_result(task_id, obs, trajectory)

    late_steps = [
        ReleaseAction(
            action_type="search_incidents",
            keywords=["retry", "timeout", "latency", "outage", "connection", "pool"],
        ),
        ReleaseAction(action_type="check_policy"),
    ]

    for action in late_steps:
        obs = env.step(action)
        trajectory.append(action.action_type)
        if obs.done:
            break

    if obs.done:
        return _build_result(task_id, obs, trajectory)

    # ── Phase 2: Decide based on gathered risks ─────────────────────

    risks = obs.known_risk_signals
    critical_risks = [r for r in risks if r.severity == "critical"]
    high_risks = [r for r in risks if r.severity == "high"]
    # Positive-only signals: all low severity and recognizable as good news
    positive_signal_ids = {
        "all_tests_passed", "all_approvals_granted", "backward_compatible_change"
    }
    negative_risks = [
        r for r in risks
        if r.severity in ("medium", "high", "critical")
        or r.signal_id not in positive_signal_ids
    ]

    # Build reason codes from discovered risks
    reason_codes = []
    for risk in risks:
        code = _risk_to_reason_code(risk)
        if code and code not in reason_codes:
            reason_codes.append(code)

    # Decision logic
    if critical_risks:
        # Critical risk found → block
        final_decision = "block"
    elif high_risks:
        # High risks found → request changes
        final_decision = "request_changes"
    elif negative_risks:
        # Medium or unrecognized risks → request changes
        final_decision = "request_changes"
    else:
        # Only positive/low signals → approve
        final_decision = "approve"

    # If canary is mentioned in policy and decision is approve, try canary first
    policy_content = ""
    if obs.last_tool_result and obs.last_tool_result.tool_name == "check_policy":
        policy_content = obs.last_tool_result.content.lower()

    if final_decision == "approve" and "canary" in policy_content:
        # Start canary
        obs = env.step(
            ReleaseAction(action_type="control_rollout", decision="start_canary")
        )
        trajectory.append("control_rollout:start_canary")
        if not obs.done:
            # Query telemetry during canary
            services_touched = []
            if obs.last_tool_result and obs.last_tool_result.structured_payload:
                services_touched = obs.last_tool_result.structured_payload.get(
                    "services_touched", []
                )

            # Query a couple of metrics
            for metric in ["p99", "error_rate"]:
                if obs.done:
                    break
                obs = env.step(
                    ReleaseAction(
                        action_type="query_telemetry",
                        metric=metric,
                        service=(
                            services_touched[0] if services_touched else "api-gateway"
                        ),
                    )
                )
                trajectory.append(f"query_telemetry:{metric}")

            # Re-evaluate after canary telemetry
            new_risks = obs.known_risk_signals
            canary_critical = [
                r
                for r in new_risks
                if r.severity == "critical"
                and r.signal_id not in [r2.signal_id for r2 in risks]
            ]
            if canary_critical:
                # Canary showed problems → rollback
                if not obs.done:
                    obs = env.step(
                        ReleaseAction(
                            action_type="control_rollout", decision="rollback"
                        )
                    )
                    trajectory.append("control_rollout:rollback")
                final_decision = "request_changes"
                reason_codes.append("ROLLBACK_CONDITION_MET")
            else:
                # Canary looks good
                reason_codes.append("CANARY_PASSED")

    # ── Phase 3: Submit ─────────────────────────────────────────────

    if not obs.done:
        obs = env.step(
            ReleaseAction(
                action_type="submit_decision",
                final_decision=final_decision,
                reason_codes=reason_codes,
            )
        )
        trajectory.append(f"submit_decision:{final_decision}")

    return _build_result(task_id, obs, trajectory)


def _risk_to_reason_code(risk) -> str:
    """Map a RiskSignal to a standard reason code."""
    sid = risk.signal_id.lower()
    mapping = {
        "hot_path_sync_io": "HOT_PATH_SYNC_IO",
        "integration_test_failure": "INTEGRATION_TEST_FAILURE",
        "missing_load_test": "MISSING_LOAD_TEST",
        "missing_approval_service_owner": "MISSING_OWNER_APPROVAL",
        "missing_approval_dba": "MISSING_DBA_APPROVAL",
        "policy_missing_owner_approval": "MISSING_OWNER_APPROVAL",
        "policy_peak_traffic": "PEAK_TRAFFIC_WINDOW",
        "policy_missing_load_test": "MISSING_LOAD_TEST",
        "large_blast_radius": "LARGE_BLAST_RADIUS",
        "db_connection_exhaustion": "DB_CONNECTION_EXHAUSTION",
        "missing_capacity_check": "MISSING_CAPACITY_CHECK",
        "degraded_": "DEPENDENCY_DEGRADED",
        "similar_incident_": "INCIDENT_SIMILARITY",
        "catastrophic_promotion": "CATASTROPHIC_PROMOTION",
        "canary_required_policy": "CANARY_REQUIRED",
        "backward_compatible_change": "BACKWARD_COMPATIBLE_CHANGE",
        "all_tests_passed": "ALL_TESTS_PASSED",
        "all_approvals_granted": "ALL_APPROVALS_GRANTED",
    }
    for key, code in mapping.items():
        if key in sid:
            return code

    # Generic fallback based on category
    category_map = {
        "latency": "P99_REGRESSION",
        "error_rate": "ERROR_RATE_BREACH",
        "dependency_health": "DEPENDENCY_DEGRADED",
        "test_coverage": "MISSING_LOAD_TEST",
        "blast_radius": "LARGE_BLAST_RADIUS",
        "policy_violation": "POLICY_VIOLATION",
        "missing_approval": "MISSING_OWNER_APPROVAL",
        "incident_similarity": "INCIDENT_SIMILARITY",
        "compliance": "COMPLIANCE_VIOLATION",
        "resource_saturation": "RESOURCE_SATURATION",
    }
    return category_map.get(risk.category, "")


def _build_result(task_id: str, obs, trajectory: list[str]) -> dict:
    """Build a result dict from the final observation."""
    return {
        "task_id": task_id,
        "score": obs.final_score if obs.final_score is not None else 0.0,
        "grader_breakdown": obs.grader_breakdown,
        "trajectory": trajectory,
        "steps": len(trajectory),
        "cumulative_reward": obs.cumulative_reward,
    }


def run_heuristic_baseline() -> dict:
    """
    Run the heuristic agent on all available tasks.

    Returns a dict with per-task scores and an average.
    """
    task_ids = _get_all_task_ids()
    results = {}

    for task_id in task_ids:
        env = ReleaseOpsEnvironment()
        result = play_heuristic_episode(env, task_id)
        results[task_id] = result

    scores = [r["score"] for r in results.values()]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "agent": "heuristic_baseline",
        "seed": "deterministic",
        "tasks": results,
        "average_score": round(avg_score, 3),
        "num_tasks": len(task_ids),
    }


if __name__ == "__main__":
    import sys

    results = run_heuristic_baseline()

    print(f"\n{'='*60}")
    print("HEURISTIC BASELINE RESULTS")
    print(f"{'='*60}")
    for task_id, result in results["tasks"].items():
        print(f"\n  {task_id} (score: {result['score']:.3f})")
        print(f"    trajectory: {' → '.join(result['trajectory'])}")
        if result["grader_breakdown"]:
            for k, v in result["grader_breakdown"].items():
                print(f"    {k}: {v:.3f}")
    print(f"\n  Average: {results['average_score']:.3f}")
    print(f"{'='*60}")
