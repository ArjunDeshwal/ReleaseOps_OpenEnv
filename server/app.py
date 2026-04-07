"""FastAPI application for ReleaseOps-Env.

Exposes the core OpenEnv endpoints (reset/step/state/health) via create_app,
plus the hackathon-required endpoints: /tasks, /grader, /baseline.
"""

import json
import os
from pathlib import Path

from fastapi import HTTPException
from openenv.core.env_server.http_server import create_app
from releaseops_env.models import ReleaseAction, ReleaseObservation
from server.releaseops_environment import ReleaseOpsEnvironment

TASKS_DIR = Path(__file__).parent.parent / "tasks"

# ── Core OpenEnv app ────────────────────────────────────────────────
app = create_app(
    ReleaseOpsEnvironment,
    ReleaseAction,
    ReleaseObservation,
    env_name="releaseops_env",
)


# ── /tasks — list available tasks and action schema ─────────────────
@app.get("/tasks")
def list_tasks():
    """Return all available tasks with metadata and the action schema."""
    tasks = []
    for task_dir in sorted(TASKS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        gt_path = task_dir / "ground_truth.json"
        if not gt_path.exists():
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        tasks.append(
            {
                "task_id": task_dir.name,
                "difficulty": gt.get("difficulty", "unknown"),
                "optimal_decision": gt.get("optimal_decision", ""),
                "max_steps": gt.get("max_steps", 12),
                "expected_score_range": gt.get("expected_score_range", {}),
            }
        )

    return {
        "tasks": tasks,
        "action_schema": ReleaseAction.model_json_schema(),
        "observation_schema": ReleaseObservation.model_json_schema(),
    }


# ── /grader — run grader on a specific task with a given trajectory ──
@app.post("/grader")
def run_grader(task_id: str = "easy_001"):
    """
    Run a full episode with an optimal-ish trajectory and return the grader score.

    This endpoint creates a fresh environment, plays a reference trajectory
    for the given task, and returns the grading result.
    """
    env = ReleaseOpsEnvironment()
    obs = env.reset(task_id=task_id)

    gt_path = TASKS_DIR / task_id / "ground_truth.json"
    if not gt_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    with open(gt_path) as f:
        gt = json.load(f)

    # Play a standard evidence-gathering trajectory
    evidence_actions = [
        ReleaseAction(action_type="inspect_change", section="diff"),
        ReleaseAction(action_type="inspect_change", section="tests"),
        ReleaseAction(action_type="inspect_change", section="approvals"),
        ReleaseAction(action_type="inspect_dependencies"),
        ReleaseAction(
            action_type="search_incidents", keywords=["retry", "timeout", "latency"]
        ),
        ReleaseAction(action_type="check_policy"),
    ]

    for action in evidence_actions:
        obs = env.step(action)
        if obs.done:
            break

    if not obs.done:
        obs = env.step(
            ReleaseAction(
                action_type="submit_decision",
                final_decision=gt.get("optimal_decision", "block"),
                reason_codes=gt.get("required_reason_codes", []),
            )
        )

    return {
        "task_id": task_id,
        "score": obs.final_score,
        "grader_breakdown": obs.grader_breakdown,
        "done": obs.done,
        "steps_taken": env.state.step_count,
        "cumulative_reward": obs.cumulative_reward,
    }


# ── /baseline — run baseline agent on all tasks ─────────────────────
@app.post("/baseline")
def run_baseline_endpoint():
    """
    Run the built-in heuristic baseline agent against all tasks.

    Returns scores for each task. Does NOT require an LLM API key —
    uses a rule-based heuristic agent for reproducibility.
    """
    from baseline.heuristic_agent import run_heuristic_baseline

    results = run_heuristic_baseline()
    return results


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
