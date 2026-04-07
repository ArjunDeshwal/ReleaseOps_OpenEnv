"""Test the deterministic grading system.

Verifies that the grader produces correct, reproducible scores across
different trajectories and decision qualities. All tests use easy_001.

Grading formula:
  score = 0.35 * evidence_coverage
        + 0.25 * risk_signal_discovery
        + 0.30 * decision_correctness
        + 0.10 * efficiency
        - forbidden_penalty (0.3 if any forbidden action taken)
  clamped to [0.0, 1.0]

risk_signal_discovery measures which required_risk_signals were actually
triggered (emitted by the environment) during the episode — objective,
not dependent on what strings the agent typed in reason_codes.
"""

import pytest
from server.releaseops_environment import ReleaseOpsEnvironment
from releaseops_env.models import ReleaseAction


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def env():
    return ReleaseOpsEnvironment()


def _gather_all_required_evidence(env):
    """Play the 3 required evidence steps for easy_001."""
    env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    env.step(ReleaseAction(action_type="inspect_change", section="tests"))
    env.step(ReleaseAction(action_type="check_policy"))


def _submit(env, decision, reason_codes):
    """Submit a final decision and return the observation."""
    return env.step(
        ReleaseAction(
            action_type="submit_decision",
            final_decision=decision,
            reason_codes=reason_codes,
        )
    )


# ── Score range tests ────────────────────────────────────────────────


def test_perfect_play_scores_above_085(env):
    """All evidence + optimal decision + all reason codes → ≥ 0.85."""
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)
    obs = _submit(
        env,
        "request_changes",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )
    assert (
        obs.final_score >= 0.85
    ), f"Got {obs.final_score}, breakdown: {obs.grader_breakdown}"


def test_wrong_decision_no_evidence_scores_below_020(env):
    """No evidence + wrong decision + no codes → very low."""
    env.reset(task_id="easy_001")
    obs = _submit(env, "approve", [])
    # decision_correctness=0, evidence=0, reason_accuracy=0.5 (no codes, no required → but there ARE required)
    # Actually: required_codes exist, submitted=none → precision=0/1=0, recall=0/3=0 → 0.0
    # efficiency: 1 step / 12 = 0.083 → 0.083/0.3 = 0.278
    # score = 0.35*0 + 0.25*0 + 0.30*0 + 0.10*0.278 = 0.028
    assert obs.final_score < 0.20, f"Got {obs.final_score}"


def test_acceptable_decision_gets_partial_credit(env):
    """'block' is acceptable but not optimal for easy_001 → decision_score = 0.5."""
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)
    obs = _submit(
        env,
        "block",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )
    bd = obs.grader_breakdown
    assert bd["decision_correctness"] == 0.5
    # Still gets full evidence + reason codes, so score should be decent
    assert 0.55 <= obs.final_score <= 0.85, f"Got {obs.final_score}"


def test_completely_wrong_decision_zero_decision_score(env):
    """'escalate' is not in acceptable_decisions → decision_score = 0."""
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)
    obs = _submit(
        env,
        "escalate",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )
    assert obs.grader_breakdown["decision_correctness"] == 0.0


# ── Evidence coverage component ──────────────────────────────────────


def test_full_evidence_coverage(env):
    """Gathering all 3 required evidence items → evidence_coverage = 1.0."""
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)
    obs = _submit(env, "request_changes", ["HOT_PATH_SYNC_IO"])
    assert obs.grader_breakdown["evidence_coverage"] == 1.0


def test_partial_evidence_coverage(env):
    """Gathering 1 of 3 required → evidence_coverage ≈ 0.333."""
    env.reset(task_id="easy_001")
    env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    obs = _submit(env, "request_changes", ["HOT_PATH_SYNC_IO"])
    ec = obs.grader_breakdown["evidence_coverage"]
    assert abs(ec - 1.0 / 3.0) < 0.01, f"Expected ~0.333, got {ec}"


def test_zero_evidence_coverage(env):
    """No evidence gathered → evidence_coverage = 0.0."""
    env.reset(task_id="easy_001")
    obs = _submit(env, "request_changes", ["HOT_PATH_SYNC_IO"])
    assert obs.grader_breakdown["evidence_coverage"] == 0.0


def test_extra_evidence_doesnt_hurt(env):
    """Gathering more than required evidence still gives 1.0 coverage."""
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)
    # Extra steps beyond the 3 required
    env.step(ReleaseAction(action_type="inspect_change", section="approvals"))
    env.step(ReleaseAction(action_type="inspect_dependencies"))
    obs = _submit(
        env,
        "request_changes",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )
    assert obs.grader_breakdown["evidence_coverage"] == 1.0


# ── Risk signal discovery component ─────────────────────────────────
# required_risk_signals for easy_001: hot_path_sync_io, integration_test_failure,
# missing_load_test, policy_peak_traffic
# hot_path_sync_io + missing_load_test → inspect_change(diff) + inspect_change(tests)
# integration_test_failure → inspect_change(tests)
# policy_peak_traffic → check_policy


def test_full_signal_discovery_after_complete_investigation(env):
    """Gathering all required evidence triggers all 4 required risk signals → 1.0."""
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)  # diff + tests + check_policy
    obs = _submit(env, "request_changes", [])
    assert obs.grader_breakdown["risk_signal_discovery"] == 1.0


def test_partial_signal_discovery_diff_only(env):
    """Only inspecting diff discovers hot_path_sync_io → 1/4 = 0.25."""
    env.reset(task_id="easy_001")
    env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    obs = _submit(env, "request_changes", [])
    rsd = obs.grader_breakdown["risk_signal_discovery"]
    # Only hot_path_sync_io triggered; missing_load_test, integration_test_failure,
    # policy_peak_traffic not yet discovered → 1/4 = 0.25
    assert abs(rsd - 0.25) < 0.01, f"Expected ~0.25, got {rsd}"


def test_reason_codes_do_not_affect_signal_discovery(env):
    """Submitting wrong/extra reason_codes has no effect on risk_signal_discovery."""
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)
    obs_correct = _submit(env, "request_changes", ["hot_path_sync_io"])

    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)
    obs_bogus = _submit(env, "request_changes", ["BOGUS_1", "BOGUS_2", "BOGUS_3"])

    # Both should have the same risk_signal_discovery — it's env-side, not string-matching
    assert obs_correct.grader_breakdown["risk_signal_discovery"] == \
           obs_bogus.grader_breakdown["risk_signal_discovery"]


def test_zero_signal_discovery_no_investigation(env):
    """Skipping all investigation → no signals discovered → 0.0."""
    env.reset(task_id="easy_001")
    obs = _submit(env, "request_changes", [])
    assert obs.grader_breakdown["risk_signal_discovery"] == 0.0


# ── Efficiency component ─────────────────────────────────────────────


def test_efficiency_sweet_spot(env):
    """Using 4-8 of 12 steps (33-67%) → efficiency = 1.0."""
    env.reset(task_id="easy_001")
    # 4 steps: 3 evidence + 1 submit = 4/12 = 0.333 (just in sweet spot)
    _gather_all_required_evidence(env)
    obs = _submit(
        env,
        "request_changes",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )
    assert obs.grader_breakdown["efficiency"] == 1.0


def test_efficiency_too_fast(env):
    """1 step out of 12 = 8.3% → efficiency < 1.0 (too hasty)."""
    env.reset(task_id="easy_001")
    obs = _submit(env, "request_changes", ["HOT_PATH_SYNC_IO"])
    eff = obs.grader_breakdown["efficiency"]
    # usage = 1/12 = 0.083, efficiency = 0.083/0.3 = 0.278
    assert eff < 0.5, f"Expected < 0.5, got {eff}"


def test_efficiency_too_slow(env):
    """Using 11 of 12 steps = 91.7% → efficiency tapers."""
    env.reset(task_id="easy_001")
    # Burn 10 steps with redundant inspections (each gets -0.05 after first)
    env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    env.step(ReleaseAction(action_type="inspect_change", section="tests"))
    env.step(ReleaseAction(action_type="check_policy"))
    for _ in range(7):
        env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    # Now at step 10, submit at step 11
    obs = _submit(
        env,
        "request_changes",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )
    eff = obs.grader_breakdown["efficiency"]
    # usage = 11/12 = 0.917, eff = max(0, 1-(0.917-0.7)/0.3) = 1-0.723 = 0.277
    assert eff < 0.5, f"Expected < 0.5, got {eff}"


# ── Forbidden actions ────────────────────────────────────────────────


def test_forbidden_action_penalty_applied(env):
    """Taking a forbidden action deducts 0.3 from score."""
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)

    # "approve_with_missing_approval" is in easy_001's forbidden_actions.
    # We fake it by directly injecting into actions_taken.
    env._state.actions_taken.append("approve_with_missing_approval")

    obs = _submit(
        env,
        "request_changes",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )
    assert obs.grader_breakdown["forbidden_penalty"] == 0.3
    # Without penalty this would be ~1.00, with penalty = 0.70
    assert obs.final_score <= 0.70


def test_no_forbidden_penalty_by_default(env):
    """Normal trajectory has no forbidden penalty."""
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)
    obs = _submit(
        env,
        "request_changes",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )
    assert obs.grader_breakdown["forbidden_penalty"] == 0.0


# ── Score clamping ───────────────────────────────────────────────────


def test_score_never_exceeds_one(env):
    """Score is clamped to [0.0, 1.0]."""
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)
    obs = _submit(
        env,
        "request_changes",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )
    assert 0.0 <= obs.final_score <= 1.0


def test_score_never_below_zero(env):
    """Even with forbidden penalty, score clamps to 0.0."""
    env.reset(task_id="easy_001")
    env._state.actions_taken.append("approve_with_missing_approval")
    obs = _submit(env, "escalate", [])  # wrong decision + forbidden
    assert obs.final_score >= 0.0


# ── Determinism ──────────────────────────────────────────────────────


def test_grader_deterministic_across_runs(env):
    """Identical trajectory produces identical score every time."""
    scores = []
    for _ in range(5):
        env.reset(task_id="easy_001")
        _gather_all_required_evidence(env)
        obs = _submit(
            env,
            "request_changes",
            [
                "HOT_PATH_SYNC_IO",
                "MISSING_LOAD_TEST",
                "INTEGRATION_TEST_FAILURE",
            ],
        )
        scores.append(obs.final_score)
    assert len(set(scores)) == 1, f"Scores vary: {scores}"


def test_grader_breakdown_deterministic(env):
    """Every breakdown component is identical across runs."""
    breakdowns = []
    for _ in range(3):
        env.reset(task_id="easy_001")
        _gather_all_required_evidence(env)
        obs = _submit(
            env,
            "request_changes",
            [
                "HOT_PATH_SYNC_IO",
                "MISSING_LOAD_TEST",
                "INTEGRATION_TEST_FAILURE",
            ],
        )
        breakdowns.append(obs.grader_breakdown)
    assert breakdowns[0] == breakdowns[1] == breakdowns[2]


# ── Score ordering (difficulty spread) ───────────────────────────────


def test_optimal_beats_partial_beats_wrong(env):
    """Optimal decision > acceptable decision > wrong decision."""
    # Optimal: request_changes
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)
    obs_opt = _submit(
        env,
        "request_changes",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )

    # Acceptable: block
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)
    obs_partial = _submit(
        env,
        "block",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )

    # Wrong: approve
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)
    obs_wrong = _submit(
        env,
        "approve",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )

    assert (
        obs_opt.final_score > obs_partial.final_score > obs_wrong.final_score
    ), f"Expected optimal({obs_opt.final_score}) > partial({obs_partial.final_score}) > wrong({obs_wrong.final_score})"


def test_more_evidence_higher_score(env):
    """Gathering more required evidence → higher score (same decision)."""
    # 0 evidence
    env.reset(task_id="easy_001")
    obs_0 = _submit(
        env,
        "request_changes",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )

    # 1 evidence
    env.reset(task_id="easy_001")
    env.step(ReleaseAction(action_type="inspect_change", section="diff"))
    obs_1 = _submit(
        env,
        "request_changes",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )

    # 3 evidence (all required)
    env.reset(task_id="easy_001")
    _gather_all_required_evidence(env)
    obs_3 = _submit(
        env,
        "request_changes",
        [
            "HOT_PATH_SYNC_IO",
            "MISSING_LOAD_TEST",
            "INTEGRATION_TEST_FAILURE",
        ],
    )

    assert (
        obs_3.final_score > obs_1.final_score > obs_0.final_score
    ), f"Expected 3ev({obs_3.final_score}) > 1ev({obs_1.final_score}) > 0ev({obs_0.final_score})"
