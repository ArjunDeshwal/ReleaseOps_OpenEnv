"""Core contract tests for ReleaseOps-Env."""

import pytest
from server.releaseops_environment import ReleaseOpsEnvironment
from releaseops_env.models import ReleaseAction


# ── RESET ──────────────────────────────────────────────────────────────────


def test_reset_returns_observation(env):
    obs = env.reset(task_id="easy_001")
    assert obs.done is False
    assert obs.change_summary != ""
    assert obs.rollout_phase == "precheck"
    assert obs.time_remaining == 12
    assert obs.cumulative_reward == 0.0
    assert obs.known_risk_signals == []


def test_reset_sets_state(env):
    env.reset(task_id="easy_001")
    state = env.state
    assert state.task_id == "easy_001"
    assert state.step_count == 0
    assert state.rollout_phase == "precheck"
    assert state.evidence_gathered == []


def test_reset_is_idempotent(env):
    env.reset(task_id="easy_001")
    env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    env.reset(task_id="easy_001")  # reset again
    assert env.state.step_count == 0
    assert env.state.evidence_gathered == []


# ── INSPECT CHANGE ────────────────────────────────────────────────────────


def test_inspect_change_diff_returns_content(easy_env):
    obs = easy_env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    assert obs.last_tool_result is not None
    assert obs.last_tool_result.success is True
    assert obs.last_tool_result.content != ""
    assert obs.reward > 0  # new required evidence


def test_inspect_change_tests_discovers_risks(easy_env):
    obs = easy_env.step(ReleaseAction(action_type="inspect_change", section="tests"))
    assert obs.last_tool_result.success is True
    risk_ids = [r.signal_id for r in obs.known_risk_signals]
    # easy_001 has a failed integration test and no load test
    assert "integration_test_failure" in risk_ids or "missing_load_test" in risk_ids


def test_redundant_inspection_penalized(easy_env):
    easy_env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    obs = easy_env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    assert obs.reward < 0  # penalty for redundancy


def test_inspect_approvals_discovers_pending(easy_env):
    obs = easy_env.step(ReleaseAction(action_type="inspect_change", section="approvals"))
    risk_ids = [r.signal_id for r in obs.known_risk_signals]
    assert any("approval" in rid for rid in risk_ids)


# ── CHECK POLICY ─────────────────────────────────────────────────────────


def test_check_policy_returns_rules(easy_env):
    obs = easy_env.step(ReleaseAction(action_type="check_policy"))
    assert obs.last_tool_result.success is True
    assert "POL-" in obs.last_tool_result.content
    assert obs.reward > 0


def test_check_policy_discovers_violations(easy_env):
    obs = easy_env.step(ReleaseAction(action_type="check_policy"))
    risk_ids = [r.signal_id for r in obs.known_risk_signals]
    # easy_001 has peak traffic window + missing owner approval
    assert len(obs.known_risk_signals) > 0


# ── SUBMIT DECISION ───────────────────────────────────────────────────────


def test_submit_decision_ends_episode(easy_env):
    obs = easy_env.step(
        ReleaseAction(
            action_type="submit_decision",
            final_decision="block",
            reason_codes=["MISSING_LOAD_TEST"],
        )
    )
    assert obs.done is True
    assert obs.final_score is not None
    assert 0.0 <= obs.final_score <= 1.0
    assert obs.grader_breakdown is not None
    assert obs.rollout_phase == "terminal"


def test_optimal_trajectory_high_score(env):
    """Agent that gathers required evidence + correct decision → score ≥ 0.85."""
    env.reset(task_id="easy_001")
    env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    env.step(ReleaseAction(action_type="inspect_change", section="tests"))
    env.step(ReleaseAction(action_type="check_policy"))
    obs = env.step(
        ReleaseAction(
            action_type="submit_decision",
            final_decision="request_changes",
            reason_codes=["HOT_PATH_SYNC_IO", "MISSING_LOAD_TEST", "INTEGRATION_TEST_FAILURE"],
        )
    )
    assert obs.final_score >= 0.80, f"Expected ≥0.80, got {obs.final_score}. Breakdown: {obs.grader_breakdown}"


def test_wrong_decision_low_score(env):
    """Agent that approves without evidence → score < 0.45."""
    env.reset(task_id="easy_001")
    obs = env.step(
        ReleaseAction(
            action_type="submit_decision",
            final_decision="approve",
            reason_codes=[],
        )
    )
    assert obs.final_score < 0.45, f"Expected <0.45, got {obs.final_score}"


def test_submit_without_final_decision_doesnt_end(easy_env):
    obs = easy_env.step(ReleaseAction(action_type="submit_decision", reason_codes=[]))
    assert obs.done is False
    assert obs.reward < 0


def test_grader_is_deterministic(env):
    """Same trajectory must produce the same score every run."""
    scores = []
    for _ in range(3):
        env.reset(task_id="easy_001")
        env.step(ReleaseAction(action_type="inspect_change", section="diff"))
        env.step(ReleaseAction(action_type="inspect_change", section="tests"))
        env.step(ReleaseAction(action_type="check_policy"))
        obs = env.step(
            ReleaseAction(
                action_type="submit_decision",
                final_decision="request_changes",
                reason_codes=["HOT_PATH_SYNC_IO", "MISSING_LOAD_TEST", "INTEGRATION_TEST_FAILURE"],
            )
        )
        scores.append(obs.final_score)
    assert len(set(scores)) == 1, f"Non-deterministic scores: {scores}"


# ── PHASE TRANSITIONS ─────────────────────────────────────────────────────


def test_invalid_phase_transition_penalized(easy_env):
    # Can't promote from precheck
    obs = easy_env.step(ReleaseAction(action_type="control_rollout", decision="promote"))
    assert obs.reward < 0
    assert obs.rollout_phase == "precheck"  # phase unchanged


def test_start_canary_transitions_phase(easy_env):
    obs = easy_env.step(ReleaseAction(action_type="control_rollout", decision="start_canary"))
    assert obs.rollout_phase == "canary"
    assert obs.reward >= 0


def test_max_steps_terminates(env):
    env.reset(task_id="easy_001")
    for _ in range(12):
        obs = env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    assert obs.done is True


# ── STATE ─────────────────────────────────────────────────────────────────


def test_state_tracks_evidence(easy_env):
    easy_env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    easy_env.step(ReleaseAction(action_type="inspect_change", section="tests"))
    state = easy_env.state
    assert "inspected_diff" in state.evidence_gathered
    assert "inspected_tests" in state.evidence_gathered


def test_state_tracks_step_count(easy_env):
    easy_env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    easy_env.step(ReleaseAction(action_type="check_policy"))
    assert easy_env.state.step_count == 2


def test_allowed_actions_in_observation(easy_env):
    obs = easy_env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    assert "inspect_change" in obs.allowed_actions
    assert "submit_decision" in obs.allowed_actions


def test_allowed_actions_empty_after_terminal(env):
    env.reset(task_id="easy_001")
    obs = env.step(
        ReleaseAction(
            action_type="submit_decision",
            final_decision="block",
            reason_codes=["MISSING_LOAD_TEST"],
        )
    )
    assert obs.allowed_actions == []
