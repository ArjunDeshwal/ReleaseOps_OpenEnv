"""
Baseline inference script for ReleaseOps-Env.

Two modes:
  1. LLM mode (default) — uses OpenAI API to run a model against all tasks
     Requires OPENAI_API_KEY environment variable.

  2. Heuristic mode (--heuristic) — uses a rule-based agent, no API key needed

Usage:
    # LLM baseline (requires API key)
    export OPENAI_API_KEY=sk-...
    python baseline/run_baseline.py

    # Heuristic baseline (no API key)
    python baseline/run_baseline.py --heuristic

    # Against a remote server
    export ENV_URL=http://localhost:8000
    python baseline/run_baseline.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from releaseops_env.models import ReleaseAction, ReleaseObservation
from server.releaseops_environment import ReleaseOpsEnvironment

TASKS_DIR = Path(__file__).parent.parent / "tasks"

SYSTEM_PROMPT = """You are an expert SRE agent reviewing a proposed production change.
Your job is to investigate the change thoroughly and make the right operational decision.

Available actions (set action_type to one of these):
- "inspect_change": Look at the change. Set section to "diff", "tests", "approvals", or "files_changed".
- "inspect_services": Check a service. Set service to the service name.
- "inspect_dependencies": View dependency graph and blast radius.
- "search_incidents": Search past incidents. Set keywords to a list of search terms.
- "check_policy": View rollout policy rules and current constraints.
- "query_telemetry": Query metrics. Set metric (p50/p95/p99/error_rate/queue_depth/cpu/rps), service, and window (5m/15m/1h).
- "request_artifact": Request artifacts. Set artifact_type to load_test, rollback_plan, approval, runbook, security_review, or compliance_check.
- "control_rollout": Control the rollout. Set decision to start_canary, pause, promote, or rollback.
- "submit_decision": Final decision. Set final_decision to approve, request_changes, block, or escalate. Set reason_codes to a list of specific codes.

Strategy:
1. First gather evidence: inspect diff, tests, approvals, dependencies
2. Search for similar historical incidents
3. Check the rollout policy for constraints
4. If needed, start a canary and monitor telemetry
5. Submit your decision with specific reason codes

Respond with ONLY a JSON object matching the action schema. No explanation text.
Example: {"action_type": "inspect_change", "section": "diff"}
Example: {"action_type": "submit_decision", "final_decision": "request_changes", "reason_codes": ["MISSING_LOAD_TEST", "HOT_PATH_SYNC_IO"]}"""


def build_user_prompt(obs: ReleaseObservation) -> str:
    """Build a context-rich prompt from the current observation."""
    parts = []
    parts.append(f"## Current State")
    parts.append(f"Change: {obs.change_summary}")
    parts.append(f"Rollout phase: {obs.rollout_phase}")
    parts.append(f"Steps remaining: {obs.time_remaining}")
    parts.append(f"Cumulative reward: {obs.cumulative_reward:.2f}")
    parts.append(f"Allowed actions: {obs.allowed_actions}")

    if obs.known_risk_signals:
        parts.append(f"\n## Discovered Risks ({len(obs.known_risk_signals)})")
        for r in obs.known_risk_signals:
            parts.append(f"  - [{r.severity.upper()}] {r.summary}")

    if obs.last_tool_result:
        parts.append(f"\n## Last Action Result ({obs.last_tool_result.tool_name})")
        parts.append(f"Success: {obs.last_tool_result.success}")
        content = obs.last_tool_result.content
        if len(content) > 1500:
            content = content[:1500] + "... (truncated)"
        parts.append(content)

    parts.append(f"\n## What should you do next? Respond with a JSON action.")
    return "\n".join(parts)


def run_llm_episode(
    client, model: str, env: ReleaseOpsEnvironment, task_id: str
) -> dict:
    """Run one episode using an LLM agent."""
    obs = env.reset(task_id=task_id)
    trajectory = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not obs.done:
        user_prompt = build_user_prompt(obs)
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        messages.append({"role": "assistant", "content": raw})

        try:
            action_data = json.loads(raw)
            action = ReleaseAction(**action_data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"    [!] Failed to parse action: {e}")
            print(f"    [!] Raw: {raw[:200]}")
            # Fallback: submit a generic block
            action = ReleaseAction(
                action_type="submit_decision",
                final_decision="block",
                reason_codes=["PARSE_ERROR"],
            )

        step_desc = action.action_type
        if action.action_type == "submit_decision":
            step_desc += f" → {action.final_decision} (codes: {action.reason_codes})"
        elif action.section:
            step_desc += f"({action.section})"
        elif action.service:
            step_desc += f"({action.service})"

        trajectory.append(step_desc)
        print(f"    Step {len(trajectory)}: {step_desc}")

        obs = env.step(action)

    return {
        "task_id": task_id,
        "score": obs.final_score if obs.final_score is not None else 0.0,
        "grader_breakdown": obs.grader_breakdown,
        "trajectory": trajectory,
        "steps": len(trajectory),
        "cumulative_reward": obs.cumulative_reward,
    }


def get_task_ids() -> list[str]:
    """Discover all available task IDs."""
    tasks = []
    for d in sorted(TASKS_DIR.iterdir()):
        if d.is_dir() and (d / "ground_truth.json").exists():
            tasks.append(d.name)
    return tasks


def run_llm_baseline(model: str = "gpt-4o"):
    """Run the LLM baseline agent on all tasks."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Use --heuristic for no-API baseline.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    task_ids = get_task_ids()
    results = {}

    for task_id in task_ids:
        print(f"\n{'='*60}")
        print(f"  Task: {task_id}")
        print(f"{'='*60}")

        env = ReleaseOpsEnvironment()
        result = run_llm_episode(client, model, env, task_id)
        results[task_id] = result
        print(f"\n  Score: {result['score']:.3f}")

    scores = [r["score"] for r in results.values()]
    avg = sum(scores) / len(scores) if scores else 0.0

    return {
        "agent": f"llm_baseline ({model})",
        "seed": "temperature=0",
        "tasks": results,
        "average_score": round(avg, 3),
        "num_tasks": len(task_ids),
    }


def print_results(results: dict):
    """Pretty-print baseline results."""
    print(f"\n{'='*60}")
    print(f"  BASELINE RESULTS — {results['agent']}")
    print(f"{'='*60}")

    for task_id, result in results["tasks"].items():
        print(f"\n  {task_id}: {result['score']:.3f}")
        print(f"    steps: {result['steps']}")
        print(f"    trajectory: {' → '.join(result['trajectory'][:8])}")
        if result.get("grader_breakdown"):
            for k, v in result["grader_breakdown"].items():
                print(f"    {k}: {v:.3f}")

    print(f"\n  Average: {results['average_score']:.3f}")
    print(f"  Tasks: {results['num_tasks']}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run ReleaseOps-Env baseline agent")
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Use heuristic agent (no API key needed)",
    )
    parser.add_argument(
        "--model", default="gpt-4o", help="OpenAI model name (default: gpt-4o)"
    )
    parser.add_argument("--task", default=None, help="Run only a specific task ID")
    args = parser.parse_args()

    if args.heuristic:
        from baseline.heuristic_agent import run_heuristic_baseline

        results = run_heuristic_baseline()
    else:
        results = run_llm_baseline(model=args.model)

    print_results(results)

    # Write results to file for reproducibility
    output_path = Path(__file__).parent / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
