"""
local.py — in-process test runner for ReleaseOpsEnvironment.

Runs a sequence of actions against a task without starting the HTTP server.
Useful for debugging scenarios and validating ground_truth.json.

Usage:
    python local.py easy_001
    python local.py hard_002 --trace
    python local.py all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure the server package is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent / "server"))
sys.path.insert(0, str(Path(__file__).parent))

from releaseops_environment import ReleaseOpsEnvironment
from releaseops_env.models import ReleaseAction

TASKS = ["easy_001", "easy_002", "medium_001", "medium_002", "hard_001", "hard_002"]

# Heuristic investigation sequence — covers all required evidence for every task
HEURISTIC_SEQUENCE = [
    {"action_type": "inspect_change", "section": "diff"},
    {"action_type": "inspect_change", "section": "tests"},
    {"action_type": "inspect_change", "section": "approvals"},
    {"action_type": "inspect_dependencies"},
    {"action_type": "search_incidents", "keywords": ["auth", "timeout", "rollback", "regression", "rate", "cascade"]},
    {"action_type": "check_policy"},
]


def _make_action(d: dict) -> ReleaseAction:
    return ReleaseAction(**d)


def run_heuristic(task_id: str, trace: bool = False) -> dict:
    env = ReleaseOpsEnvironment()
    obs = env.reset(task_id=task_id)

    if trace:
        print(f"\n[{task_id}] Change: {obs.change_summary[:80]}")

    # Execute investigation steps
    for step_dict in HEURISTIC_SEQUENCE:
        action = _make_action(step_dict)
        result = env.step(action)
        if trace:
            signals = [r.signal_id for r in result.known_risk_signals]
            print(f"  {step_dict['action_type']:25s}  reward={result.reward:+.2f}  signals={signals}")
        if result.done:
            break

    # Inspect top services from blast_radius
    blast = env._scenario["change"].get("blast_radius", [])
    for svc in blast[:2]:
        action = _make_action({"action_type": "inspect_services", "service": svc})
        result = env.step(action)
        if trace:
            print(f"  inspect_services({svc:20s})  reward={result.reward:+.2f}")
        if result.done:
            break

    # Build decision from discovered signals
    risk_signals = result.known_risk_signals
    codes = [r.signal_id for r in risk_signals]
    has_critical = any(r.severity == "critical" for r in risk_signals)
    has_high = any(r.severity == "high" for r in risk_signals)
    positive_only = all(r.severity in ("low", "info") for r in risk_signals)

    if positive_only or not risk_signals:
        decision = "approve"
    elif has_critical:
        decision = "block"
    else:
        decision = "request_changes"

    action = _make_action({
        "action_type": "submit_decision",
        "final_decision": decision,
        "reason_codes": codes,
    })
    final = env.step(action)

    score = final.final_score or 0.0
    breakdown = final.grader_breakdown or {}
    if trace:
        print(f"  submit({decision:15s})  score={score:.3f}  {breakdown}")

    return {"task_id": task_id, "decision": decision, "score": score, "breakdown": breakdown}


def run_all(trace: bool = False) -> None:
    results = [run_heuristic(t, trace=trace) for t in TASKS]
    total = sum(r["score"] for r in results)
    print(f"\n{'='*62}")
    print(f"{'Task':<15}  {'Decision':<15}  {'Score':>6}  Breakdown")
    print(f"{'='*62}")
    for r in results:
        bd = "  ".join(f"{k}={v:.2f}" for k, v in r["breakdown"].items())
        print(f"{r['task_id']:<15}  {r['decision']:<15}  {r['score']:>6.3f}  {bd}")
    print(f"{'='*62}")
    print(f"{'AVERAGE':<15}  {'':15}  {total/len(results):>6.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReleaseOps heuristic locally")
    parser.add_argument("task", nargs="?", default="all",
                        help="Task ID (e.g. easy_001) or 'all' (default)")
    parser.add_argument("--trace", action="store_true",
                        help="Print step-by-step trace")
    args = parser.parse_args()

    if args.task == "all":
        run_all(trace=args.trace)
    elif args.task in TASKS:
        result = run_heuristic(args.task, trace=True)
        print(f"\nFinal: {result}")
    else:
        print(f"Unknown task '{args.task}'. Choose from: {TASKS} or 'all'")
        sys.exit(1)
