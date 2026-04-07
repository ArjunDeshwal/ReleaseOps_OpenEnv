"""
ReleaseOps rubrics — composable grading components.

Inspired by the REPL env's rubric pattern. Each rubric is an isolated,
testable unit that grades one dimension of agent behavior. The composite
ReleaseOpsRubric combines them into the final [0, 1] score.

Grading dimensions:
  EvidenceRubric       0.35 — did the agent inspect the right information sources?
  RiskDiscoveryRubric  0.25 — did the agent trigger discovery of key risk signals?
  DecisionRubric       0.30 — was the final release decision correct?
  EfficiencyRubric     0.10 — did the agent avoid wasted / redundant steps?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class EpisodeTrace:
    """Snapshot of relevant episode state passed to rubrics."""
    evidence_gathered: list[str]         # keys accumulated via inspect_* actions
    risk_signals_found: list[str]        # signal_ids emitted by the environment
    final_decision: str                  # "approve" | "request_changes" | "block" | "escalate"
    step_count: int
    max_steps: int
    actions_taken: list[str]             # action_type per step


@dataclass
class RubricResult:
    name: str
    score: float          # [0.0, 1.0]
    weight: float
    details: dict


# ── Rubric protocol ───────────────────────────────────────────────────────────

class Rubric(Protocol):
    name: str
    weight: float

    def score(self, trace: EpisodeTrace, ground_truth: dict) -> RubricResult:
        ...


# ── Individual rubrics ────────────────────────────────────────────────────────

class EvidenceRubric:
    """
    Measures information-gathering breadth.

    Reward = (required evidence keys gathered) / (total required).
    A thorough investigator inspects the diff, tests, approvals, policy,
    dependencies, and incidents before deciding.
    """

    name = "evidence_coverage"
    weight = 0.35

    def score(self, trace: EpisodeTrace, ground_truth: dict) -> RubricResult:
        required = set(ground_truth.get("required_evidence", []))
        gathered = set(trace.evidence_gathered)
        if not required:
            value = 1.0
            matched = set()
        else:
            matched = required & gathered
            value = len(matched) / len(required)

        missing = sorted(required - gathered)
        return RubricResult(
            name=self.name,
            score=round(value, 3),
            weight=self.weight,
            details={
                "required": sorted(required),
                "gathered": sorted(gathered & required),
                "missing": missing,
            },
        )


class RiskDiscoveryRubric:
    """
    Objective measure of signal discovery.

    Checks state.risk_signals_found (signal_ids the *environment* emitted
    during the episode) against required_risk_signals in ground_truth.

    This is analogous to the Calendar env's SQL verifiers: the environment
    measures what the agent actually observed, not what strings the agent typed.
    An agent that skips inspect_tests will never trigger 'missing_load_test',
    even if it guesses the string correctly in reason_codes.
    """

    name = "risk_signal_discovery"
    weight = 0.25

    def score(self, trace: EpisodeTrace, ground_truth: dict) -> RubricResult:
        required = set(ground_truth.get("required_risk_signals", []))
        discovered = set(trace.risk_signals_found)
        if not required:
            # No required signals (e.g. trivial approve) — full credit
            value = 1.0
            matched = set()
        else:
            matched = required & discovered
            value = len(matched) / len(required)

        missing = sorted(required - discovered)
        return RubricResult(
            name=self.name,
            score=round(value, 3),
            weight=self.weight,
            details={
                "required": sorted(required),
                "discovered": sorted(discovered & required),
                "missing": missing,
                "extra_discovered": sorted(discovered - required),
            },
        )


class DecisionRubric:
    """
    Scores the final release decision.

    optimal  → 1.0   (exactly right)
    acceptable but not optimal → 0.5   (e.g. block when request_changes was best)
    wrong    → 0.0
    """

    name = "decision_correctness"
    weight = 0.30

    def score(self, trace: EpisodeTrace, ground_truth: dict) -> RubricResult:
        optimal = ground_truth.get("optimal_decision", "")
        acceptable = set(ground_truth.get("acceptable_decisions", [optimal]))
        decision = trace.final_decision

        if decision == optimal:
            value = 1.0
            label = "optimal"
        elif decision in acceptable:
            value = 0.5
            label = "acceptable"
        else:
            value = 0.0
            label = "wrong"

        return RubricResult(
            name=self.name,
            score=round(value, 3),
            weight=self.weight,
            details={
                "submitted": decision,
                "optimal": optimal,
                "acceptable": sorted(acceptable),
                "verdict": label,
            },
        )


class EfficiencyRubric:
    """
    Rewards investigators who complete the task without wasted steps.

    The efficiency band [0.3, 0.7] of max_steps scores 1.0. Outside that
    window the score degrades linearly. Agents that decide after only 1-2 steps
    are penalized (they skipped evidence) as are agents that thrash up to
    the step budget.
    """

    name = "efficiency"
    weight = 0.10

    def score(self, trace: EpisodeTrace, ground_truth: dict) -> RubricResult:
        usage = trace.step_count / max(trace.max_steps, 1)
        if 0.3 <= usage <= 0.7:
            value = 1.0
        elif usage < 0.3:
            value = usage / 0.3
        else:
            value = max(0.0, 1.0 - (usage - 0.7) / 0.3)

        return RubricResult(
            name=self.name,
            score=round(value, 3),
            weight=self.weight,
            details={
                "steps_taken": trace.step_count,
                "max_steps": trace.max_steps,
                "usage_fraction": round(usage, 3),
            },
        )


# ── Composite rubric ──────────────────────────────────────────────────────────

class ReleaseOpsRubric:
    """
    Composite rubric combining all four dimensions.

    Usage:
        rubric = ReleaseOpsRubric()
        result = rubric.score(trace, ground_truth)
        print(result["score"], result["breakdown"])
    """

    def __init__(self):
        self._rubrics: list[Rubric] = [
            EvidenceRubric(),
            RiskDiscoveryRubric(),
            DecisionRubric(),
            EfficiencyRubric(),
        ]

    def score(self, trace: EpisodeTrace, ground_truth: dict) -> dict:
        results = [r.score(trace, ground_truth) for r in self._rubrics]

        # Forbidden action penalty
        forbidden = ground_truth.get("forbidden_actions", [])
        took_forbidden = any(fa in trace.actions_taken for fa in forbidden)
        forbidden_penalty = 0.3 if took_forbidden else 0.0

        raw = sum(r.score * r.weight for r in results)
        final_score = max(0.0, min(1.0, raw - forbidden_penalty))

        return {
            "score": round(final_score, 3),
            "breakdown": {r.name: round(r.score, 3) for r in results}
            | {"forbidden_penalty": round(forbidden_penalty, 3)},
            "details": {r.name: r.details for r in results},
        }
