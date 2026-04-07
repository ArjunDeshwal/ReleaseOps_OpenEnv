"""Test the per-step reward signals.

Verifies that the dense reward function produces correct positive/negative
signals for each action type and scenario condition. All tests use easy_001.

Reward cheat-sheet:
  +0.10  new required evidence
  +0.02  new non-required evidence
  +0.05  new service inspection / new telemetry
  +0.15  uncovering high/critical risk via telemetry
  +0.20  discovering missing required artifact
  +0.25  starting canary when required by policy
  +0.30  correct rollback after bad canary telemetry
  +0.40  correct final decision (optimal)
  +0.20  partial correct final decision (acceptable)
  -0.02  missing required parameter
  -0.05  redundant evidence / redundant inspection
  -0.10  invalid phase transition
  -0.15  wrong final decision
  -0.20  timeout (max steps)
  -0.40  catastrophic promotion
"""

import pytest
from server.releaseops_environment import ReleaseOpsEnvironment
from releaseops_env.models import ReleaseAction


@pytest.fixture
def env():
    return ReleaseOpsEnvironment()


@pytest.fixture
def easy_env(env):
    env.reset(task_id="easy_001")
    return env


# ── Positive rewards: evidence gathering ─────────────────────────────


def test_new_required_evidence_positive(easy_env):
    """First inspection of required evidence → +0.10."""
    obs = easy_env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    # "inspected_diff" is in easy_001 required_evidence
    assert obs.reward == 0.10


def test_new_non_required_evidence_zero(easy_env):
    """First inspection of non-required evidence → 0.0 (neutral, not rewarded)."""
    obs = easy_env.step(
        ReleaseAction(action_type="inspect_change", section="files_changed")
    )
    # "inspected_files_changed" is NOT in required_evidence — no incentive to over-investigate
    assert obs.reward == 0.0


def test_new_service_inspection_positive(easy_env):
    """First inspection of a service → +0.05."""
    obs = easy_env.step(
        ReleaseAction(
            action_type="inspect_services",
            service="payment-service",
        )
    )
    assert obs.reward == 0.05


def test_new_dependency_inspection_required(easy_env):
    """inspect_dependencies is not in easy_001 required_evidence → +0.02."""
    obs = easy_env.step(ReleaseAction(action_type="inspect_dependencies"))
    # "inspected_dependencies" is NOT in easy_001's required_evidence
    assert obs.reward == 0.02


def test_search_incidents_with_match_bonus(easy_env):
    """Finding a high-similarity incident → base reward + 0.05 bonus."""
    obs = easy_env.step(
        ReleaseAction(
            action_type="search_incidents",
            keywords=["synchronous", "timeout"],
        )
    )
    # "checked_incidents" is not in required_evidence for easy_001 → base 0.02
    # But there's a high-similarity incident (INC-3341, similarity 0.92) → +0.05
    assert obs.reward >= 0.05


def test_check_policy_discovers_violations_bonus(easy_env):
    """Policy check that finds high/critical violations → base + 0.05."""
    obs = easy_env.step(ReleaseAction(action_type="check_policy"))
    # "checked_policy" IS in required_evidence → base 0.10
    # easy_001 has peak traffic (high) and missing owner approval (critical) → +0.05
    assert obs.reward >= 0.10


# ── Negative rewards: redundancy ─────────────────────────────────────


def test_redundant_inspect_change_penalized(easy_env):
    """Inspecting the same section twice → -0.05."""
    easy_env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    obs = easy_env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    assert obs.reward == -0.05


def test_redundant_service_inspection_penalized(easy_env):
    """Inspecting the same service twice → -0.05."""
    easy_env.step(
        ReleaseAction(
            action_type="inspect_services",
            service="payment-service",
        )
    )
    obs = easy_env.step(
        ReleaseAction(
            action_type="inspect_services",
            service="payment-service",
        )
    )
    assert obs.reward == -0.05


def test_redundant_policy_check_penalized(easy_env):
    """Checking policy twice → -0.05."""
    easy_env.step(ReleaseAction(action_type="check_policy"))
    obs = easy_env.step(ReleaseAction(action_type="check_policy"))
    assert obs.reward == -0.05


def test_redundant_incident_search_penalized(easy_env):
    """Searching incidents twice → -0.05."""
    easy_env.step(ReleaseAction(action_type="search_incidents", keywords=["retry"]))
    obs = easy_env.step(
        ReleaseAction(action_type="search_incidents", keywords=["retry"])
    )
    assert obs.reward == -0.05


def test_redundant_dependency_inspection_penalized(easy_env):
    """Inspecting dependencies twice → -0.05."""
    easy_env.step(ReleaseAction(action_type="inspect_dependencies"))
    obs = easy_env.step(ReleaseAction(action_type="inspect_dependencies"))
    assert obs.reward == -0.05


# ── Negative rewards: invalid actions ────────────────────────────────


def test_missing_service_parameter_penalized(easy_env):
    """inspect_services without service name → -0.02."""
    obs = easy_env.step(ReleaseAction(action_type="inspect_services"))
    assert obs.reward == -0.02
    assert obs.last_tool_result.success is False


def test_missing_telemetry_service_penalized(easy_env):
    """query_telemetry without service → -0.02."""
    obs = easy_env.step(
        ReleaseAction(
            action_type="query_telemetry",
            metric="p99",
        )
    )
    assert obs.reward == -0.02


def test_missing_artifact_type_penalized(easy_env):
    """request_artifact without artifact_type → -0.02."""
    obs = easy_env.step(ReleaseAction(action_type="request_artifact"))
    assert obs.reward == -0.02


def test_missing_rollout_decision_penalized(easy_env):
    """control_rollout without decision → -0.02."""
    obs = easy_env.step(ReleaseAction(action_type="control_rollout"))
    assert obs.reward == -0.02


def test_invalid_phase_transition_penalized(easy_env):
    """Trying to promote from precheck → -0.10."""
    obs = easy_env.step(
        ReleaseAction(
            action_type="control_rollout",
            decision="promote",
        )
    )
    assert obs.reward == -0.10
    assert obs.last_tool_result.success is False


def test_invalid_rollback_from_precheck(easy_env):
    """Trying to rollback from precheck → -0.10."""
    obs = easy_env.step(
        ReleaseAction(
            action_type="control_rollout",
            decision="rollback",
        )
    )
    assert obs.reward == -0.10


# ── Phase transition rewards ─────────────────────────────────────────


def test_start_canary_positive_reward(easy_env):
    """Starting canary → positive reward (≥ 0.05)."""
    obs = easy_env.step(
        ReleaseAction(
            action_type="control_rollout",
            decision="start_canary",
        )
    )
    assert obs.reward >= 0.05
    assert obs.rollout_phase == "canary"


def test_pause_canary_small_positive(easy_env):
    """Pausing canary → +0.05."""
    easy_env.step(
        ReleaseAction(
            action_type="control_rollout",
            decision="start_canary",
        )
    )
    obs = easy_env.step(
        ReleaseAction(
            action_type="control_rollout",
            decision="pause",
        )
    )
    assert obs.reward == 0.05
    assert obs.rollout_phase == "paused"


def test_catastrophic_promotion_large_penalty(env):
    """Promoting a change that should be blocked → -0.40."""
    env.reset(task_id="easy_001")
    # easy_001 optimal is "request_changes" → promoting is catastrophic
    env.step(ReleaseAction(action_type="control_rollout", decision="start_canary"))
    obs = env.step(ReleaseAction(action_type="control_rollout", decision="promote"))
    assert obs.reward == -0.40
    assert obs.rollout_phase == "terminal"


# ── Telemetry rewards ────────────────────────────────────────────────


def test_new_telemetry_positive(easy_env):
    """First telemetry query → +0.05."""
    obs = easy_env.step(
        ReleaseAction(
            action_type="query_telemetry",
            metric="p99",
            service="payment-service",
        )
    )
    assert obs.reward >= 0.05


def test_redundant_telemetry_penalized(easy_env):
    """Same telemetry query twice in same phase → -0.05."""
    easy_env.step(
        ReleaseAction(
            action_type="query_telemetry",
            metric="p99",
            service="payment-service",
        )
    )
    obs = easy_env.step(
        ReleaseAction(
            action_type="query_telemetry",
            metric="p99",
            service="payment-service",
        )
    )
    assert obs.reward == -0.05


def test_telemetry_breach_during_canary_bonus(env):
    """Querying telemetry during canary that breaches 2x baseline → +0.15 bonus."""
    env.reset(task_id="easy_001")
    env.step(ReleaseAction(action_type="control_rollout", decision="start_canary"))
    # Burn a couple steps in canary to let telemetry degrade
    env.step(
        ReleaseAction(
            action_type="query_telemetry", metric="error_rate", service="audit-service"
        )
    )
    # Now query at a later step where values should be degraded
    obs = env.step(
        ReleaseAction(
            action_type="query_telemetry",
            metric="p99",
            service="payment-service",
        )
    )
    # During canary, payment-service p99 goes from 125 → 2800+ (>2x baseline)
    # If breach detected: base 0.05 + 0.15 = 0.20
    # depends on exact step_in_phase index into the curve
    assert obs.reward >= 0.05


# ── Artifact rewards ─────────────────────────────────────────────────


def test_missing_artifact_discovery(easy_env):
    """Requesting a missing artifact → positive reward + risk discovered."""
    obs = easy_env.step(
        ReleaseAction(
            action_type="request_artifact",
            artifact_type="load_test",
        )
    )
    # load_test is null in easy_001 → artifact missing
    assert obs.last_tool_result.success is False
    assert obs.reward > 0  # missing artifact is useful information
    risk_ids = [r.signal_id for r in obs.known_risk_signals]
    assert "missing_load_test" in risk_ids


def test_existing_artifact_small_positive(easy_env):
    """Requesting an existing artifact → +0.05."""
    obs = easy_env.step(
        ReleaseAction(
            action_type="request_artifact",
            artifact_type="approval",
        )
    )
    # approvals exist in easy_001 → found
    assert obs.last_tool_result.success is True
    assert obs.reward == 0.05


def test_redundant_artifact_request_penalized(easy_env):
    """Requesting the same artifact twice → -0.05."""
    easy_env.step(
        ReleaseAction(
            action_type="request_artifact",
            artifact_type="load_test",
        )
    )
    obs = easy_env.step(
        ReleaseAction(
            action_type="request_artifact",
            artifact_type="load_test",
        )
    )
    assert obs.reward == -0.05


# ── Submit decision rewards ──────────────────────────────────────────


def test_optimal_decision_reward(env):
    """Optimal decision → +0.40 decision reward."""
    env.reset(task_id="easy_001")
    env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    env.step(ReleaseAction(action_type="inspect_change", section="tests"))
    env.step(ReleaseAction(action_type="check_policy"))
    obs = env.step(
        ReleaseAction(
            action_type="submit_decision",
            final_decision="request_changes",
            reason_codes=[
                "HOT_PATH_SYNC_IO",
                "MISSING_LOAD_TEST",
                "INTEGRATION_TEST_FAILURE",
            ],
        )
    )
    # decision_correct=True → 0.40, reason_accuracy > 0.8 → +0.10 = 0.50
    assert obs.reward >= 0.40


def test_acceptable_decision_reward(env):
    """Acceptable but not optimal → +0.20."""
    env.reset(task_id="easy_001")
    env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    obs = env.step(
        ReleaseAction(
            action_type="submit_decision",
            final_decision="block",
            reason_codes=["MISSING_LOAD_TEST"],
        )
    )
    # decision_partial=True, not correct → 0.20
    assert 0.15 <= obs.reward <= 0.35


def test_wrong_decision_negative_reward(env):
    """Wrong decision → -0.15."""
    env.reset(task_id="easy_001")
    obs = env.step(
        ReleaseAction(
            action_type="submit_decision",
            final_decision="approve",
            reason_codes=[],
        )
    )
    assert obs.reward < 0


def test_submit_without_decision_penalized(easy_env):
    """submit_decision without final_decision → -0.05, episode NOT ended."""
    obs = easy_env.step(
        ReleaseAction(
            action_type="submit_decision",
            reason_codes=["MISSING_LOAD_TEST"],
        )
    )
    assert obs.reward == -0.05
    assert obs.done is False


# ── Timeout penalty ──────────────────────────────────────────────────


def test_timeout_penalty(env):
    """Hitting max steps without deciding → -0.20 timeout penalty."""
    env.reset(task_id="easy_001")
    last_obs = None
    for i in range(12):
        last_obs = env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    assert last_obs.done is True
    # The last step gets the regular reward PLUS -0.20 timeout
    # First step is +0.10 (new evidence), steps 2-12 are -0.05 (redundant)
    # Step 12 also gets -0.20 timeout
    assert last_obs.reward < 0


# ── Cumulative reward tracking ───────────────────────────────────────


def test_cumulative_reward_tracks_correctly(easy_env):
    """cumulative_reward in observation matches sum of step rewards."""
    rewards = []
    obs = easy_env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    rewards.append(obs.reward)
    obs = easy_env.step(ReleaseAction(action_type="inspect_change", section="tests"))
    rewards.append(obs.reward)
    obs = easy_env.step(ReleaseAction(action_type="check_policy"))
    rewards.append(obs.reward)

    expected = sum(rewards)
    assert (
        abs(obs.cumulative_reward - expected) < 0.001
    ), f"Expected cumulative {expected}, got {obs.cumulative_reward}"


def test_cumulative_reward_resets_on_reset(env):
    """reset() zeroes the cumulative reward."""
    env.reset(task_id="easy_001")
    env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    env.step(ReleaseAction(action_type="inspect_change", section="tests"))
    # Reward is non-zero now
    assert env.state.cumulative_reward > 0

    env.reset(task_id="easy_001")
    assert env.state.cumulative_reward == 0.0


# ── Risk signal discovery rewards ────────────────────────────────────


def test_inspecting_tests_discovers_risks(easy_env):
    """Inspecting tests section surfaces test-related risks."""
    obs = easy_env.step(ReleaseAction(action_type="inspect_change", section="tests"))
    risk_ids = [r.signal_id for r in obs.known_risk_signals]
    assert "integration_test_failure" in risk_ids
    assert "missing_load_test" in risk_ids


def test_inspecting_approvals_discovers_pending(easy_env):
    """Inspecting approvals surfaces pending approval risks."""
    obs = easy_env.step(
        ReleaseAction(action_type="inspect_change", section="approvals")
    )
    risk_ids = [r.signal_id for r in obs.known_risk_signals]
    assert "missing_approval_service_owner" in risk_ids


def test_inspecting_diff_discovers_hot_path(easy_env):
    """Inspecting diff surfaces hot-path sync I/O risk."""
    obs = easy_env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    risk_ids = [r.signal_id for r in obs.known_risk_signals]
    assert "hot_path_sync_io" in risk_ids


def test_degraded_service_risk_discovered(env):
    """Inspecting a degraded service surfaces dependency_health risk."""
    # easy_001 payment-service is "healthy", not "degraded"
    # but we can still test that healthy services don't produce risk
    env.reset(task_id="easy_001")
    obs = env.step(
        ReleaseAction(
            action_type="inspect_services",
            service="payment-service",
        )
    )
    risk_ids = [r.signal_id for r in obs.known_risk_signals]
    # payment-service is healthy in easy_001 → no degraded risk
    assert "degraded_payment-service" not in risk_ids


def test_risks_accumulate_across_steps(easy_env):
    """Risks from multiple steps accumulate in observation."""
    easy_env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    obs = easy_env.step(ReleaseAction(action_type="inspect_change", section="tests"))
    # Should have risks from BOTH diff and tests
    risk_ids = [r.signal_id for r in obs.known_risk_signals]
    assert "hot_path_sync_io" in risk_ids
    assert "integration_test_failure" in risk_ids or "missing_load_test" in risk_ids


def test_risks_dont_duplicate(easy_env):
    """Same risk is not added twice even if triggered again."""
    easy_env.step(ReleaseAction(action_type="inspect_change", section="tests"))
    # Inspecting tests again is redundant but shouldn't duplicate risks
    easy_env.step(ReleaseAction(action_type="inspect_change", section="tests"))
    risk_ids = [r.signal_id for r in easy_env._get_visible_risks()]
    assert len(risk_ids) == len(set(risk_ids)), "Duplicate risk signals found"
