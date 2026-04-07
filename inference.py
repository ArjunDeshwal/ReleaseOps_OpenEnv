"""
Inference Script — ReleaseOps-Env

Env vars:
    API_BASE_URL      LLM API endpoint   (default: https://router.huggingface.co/v1)
    MODEL_NAME        Model identifier   (set via environment)
    OPENAI_API_KEY    API key            (or HF_TOKEN)
    HF_TOKEN          API key fallback
    ENV_URL           Environment URL    (default: http://localhost:7860)

Install: pip install openenv-core openai

Structured stdout logs follow [START], [STEP], [END] format per hackathon requirements.
"""

import json
import os
import textwrap
import time
from typing import List, Optional

from openai import OpenAI
from openenv.core import GenericEnvClient

# ── Config ──────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME   = os.getenv("MODEL_NAME")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")


# ── Structured logging helpers ──────────────────────────────────────────────────
def log_start(task_id: str, model_name: str, env_url: str):
    """Emit [START] log with task metadata."""
    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "model_name": model_name,
        "env_url": env_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }))


def log_step(task_id: str, step: int, action: dict, reward: float, done: bool):
    """Emit [STEP] log with action and reward."""
    print(json.dumps({
        "type": "[STEP]",
        "task_id": task_id,
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }))


def log_end(task_id: str, final_score: float, steps_taken: int, grader_breakdown: dict):
    """Emit [END] log with final results."""
    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "final_score": final_score,
        "steps_taken": steps_taken,
        "grader_breakdown": grader_breakdown,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }))

TASKS       = ["easy_001", "easy_002", "medium_001", "medium_002", "hard_001", "hard_002"]
MAX_STEPS   = 14
TEMPERATURE = 0.0  # reproducible

# ── Prompt ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an SRE agent reviewing a proposed software change for production rollout.
    Investigate thoroughly, then submit a final decision.

    Investigation tools (use as needed — not all are required for every change):
      inspect_change       section: "diff"|"tests"|"approvals"|"files_changed"
      inspect_services     service: <service_name_exactly_as_shown>
      inspect_dependencies (no extra params) — reveals blast radius
      search_incidents     keywords: ["word1", "word2"] — queries real incident DB
      check_policy         (no extra params) — checks rollout policy rules
      query_telemetry      metric: "p99"|"error_rate"|"p95"|"cpu"|"rps"|"queue_depth"
                           service: <name>, window: "5m"|"15m"|"1h"
                           — check CURRENT live metrics before deciding; risky changes
                             (concurrency, rate limiting, connection pools) often show
                             pre-existing anomalies that static inspection misses
      request_artifact     artifact_type: "load_test"|"rollback_plan"|"approval"|"runbook"
      control_rollout      decision: "start_canary"|"promote"|"pause"|"rollback"
      submit_decision      final_decision: "approve"|"request_changes"|"block"|"escalate"
                           reason_codes: [<signal_id>, ...]  — use signal_ids from known_risk_signals

    Investigation strategy:
    - ALWAYS start: inspect_change(diff) → inspect_change(tests) → inspect_change(approvals) → inspect_dependencies
    - inspect_dependencies reveals the EXACT service names you MUST use for inspect_services and query_telemetry
    - NEVER guess service names — only use names returned by inspect_dependencies
    - For changes touching high-traffic services, query live telemetry — pre-existing
      degradation is a blocker even if tests pass
    - Search incidents with keywords from the change type (e.g. "retry", "rate_limit", "pool")
    - Check policy after gathering evidence
    - reason_codes MUST be string signal_ids from known_risk_signals (e.g. "missing_load_test"), never numbers

    Respond with ONLY a valid JSON object. No explanation. No markdown.
    Example: {"action_type": "query_telemetry", "metric": "error_rate", "service": "api-gateway", "window": "5m"}
""").strip()


# ── Prompt builder ───────────────────────────────────────────────────────────────
def build_prompt(step: int, obs: dict, history: List[str]) -> str:
    risk_lines = "\n".join(
        f"  [{r['severity'].upper()}] {r['signal_id']}: {r['summary']}"
        for r in obs.get("known_risk_signals", [])
    ) or "  (none discovered yet)"

    last = obs.get("last_tool_result")
    if last:
        status = "OK" if last["success"] else "FAIL"
        last_result = f"[{status}] {last['tool_name']}:\n{last['content'][:600]}"
    else:
        last_result = "(none)"

    return textwrap.dedent(f"""
        Step: {step}/{MAX_STEPS} | Phase: {obs['rollout_phase']} | Budget: {obs['time_remaining']}
        Task: {obs['task_id']}
        Change: {obs['change_summary']}

        Risk signals (use signal_id as reason_codes):
        {risk_lines}

        Last result:
        {last_result}

        Actions taken so far — DO NOT repeat:
        {chr(10).join(history) or '(none)'}

        Output next action as JSON.
    """).strip()


# ── Action parsing ───────────────────────────────────────────────────────────────
# Fields allowed per action_type — prevents hallucinated fields from failing validation
_ACTION_FIELDS: dict[str, set] = {
    "inspect_change":       {"section"},
    "inspect_services":     {"service"},
    "inspect_dependencies": set(),
    "search_incidents":     {"keywords"},
    "check_policy":         set(),
    "query_telemetry":      {"metric", "service", "window"},
    "request_artifact":     {"artifact_type"},
    "control_rollout":      {"decision"},
    "submit_decision":      {"final_decision", "reason_codes"},
}

_VALID_SECTIONS   = {"diff", "tests", "approvals", "files_changed"}
_VALID_METRICS    = {"p50", "p95", "p99", "error_rate", "queue_depth", "cpu", "rps"}
_VALID_WINDOWS    = {"5m", "15m", "1h"}
_VALID_ARTIFACTS  = {"load_test", "rollback_plan", "approval", "runbook", "security_review", "compliance_check"}
_VALID_ROLLOUT    = {"start_canary", "pause", "promote", "rollback"}
_VALID_DECISIONS  = {"approve", "request_changes", "block", "escalate"}


def parse_action(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.startswith("```")).strip()
    s, e = text.find("{"), text.rfind("}") + 1
    if s >= 0 and e > s:
        text = text[s:e]
    try:
        data = json.loads(text)
    except Exception:
        return None

    action_type = data.get("action_type", "")
    if action_type not in _ACTION_FIELDS:
        return None

    # Keep only fields valid for this action_type
    allowed = _ACTION_FIELDS[action_type]
    result: dict = {"action_type": action_type}
    for k in allowed:
        if k in data:
            result[k] = data[k]

    # Validate enum fields — strip if invalid to avoid Pydantic rejection
    if "section" in result and result["section"] not in _VALID_SECTIONS:
        del result["section"]
    if "metric" in result and result["metric"] not in _VALID_METRICS:
        del result["metric"]
    if "window" in result and result["window"] not in _VALID_WINDOWS:
        del result["window"]
    if "artifact_type" in result and result["artifact_type"] not in _VALID_ARTIFACTS:
        del result["artifact_type"]
    if "decision" in result and result["decision"] not in _VALID_ROLLOUT:
        del result["decision"]
    if "final_decision" in result and result["final_decision"] not in _VALID_DECISIONS:
        del result["final_decision"]
    if "keywords" in result and not isinstance(result["keywords"], list):
        result["keywords"] = [str(result["keywords"])]
    if "reason_codes" in result:
        if not isinstance(result["reason_codes"], list):
            result["reason_codes"] = [str(result["reason_codes"])]
        else:
            result["reason_codes"] = [str(rc) for rc in result["reason_codes"] if rc is not None]

    return result


# ── Task runner ──────────────────────────────────────────────────────────────────
def run_task(llm: OpenAI, task_id: str) -> dict:
    log_start(task_id, MODEL_NAME, ENV_URL)

    with GenericEnvClient(base_url=ENV_URL).sync() as env:
        result = env.reset(task_id=task_id)
        obs = result.observation if hasattr(result, "observation") else result
        if isinstance(obs, dict):
            obs_dict = obs
        else:
            obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)

        history: List[str] = []
        step = 0
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            done = result.done if hasattr(result, "done") else obs_dict.get("done", False)
            if done:
                break

            response_text = ""
            for attempt in range(4):
                try:
                    completion = llm.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": build_prompt(step, obs_dict, history)},
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=200,
                    )
                    response_text = completion.choices[0].message.content or ""
                    break
                except Exception as exc:
                    msg = str(exc)
                    if "429" in msg or "rate" in msg.lower():
                        wait = 15 * (attempt + 1)
                        time.sleep(wait)
                    else:
                        break

            action = parse_action(response_text)
            if action is None:
                action = {"action_type": "check_policy"}

            # Force submit on last step
            if step == MAX_STEPS and action.get("action_type") != "submit_decision":
                risks = obs_dict.get("known_risk_signals", [])
                codes = [r["signal_id"] for r in risks] or ["INSUFFICIENT_EVIDENCE"]
                has_high = any(r["severity"] in ("high", "critical") for r in risks)
                action = {
                    "action_type": "submit_decision",
                    "final_decision": "request_changes" if has_high else "approve",
                    "reason_codes": codes,
                }

            result = env.step(action)
            obs = result.observation if hasattr(result, "observation") else result
            obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else (obs if isinstance(obs, dict) else dict(obs))

            last_reward = getattr(result, 'reward', 0) or 0
            done = getattr(result, "done", False) or obs_dict.get("done", False)

            # Emit structured [STEP] log
            log_step(task_id, step, action, last_reward, done)

            history.append(
                f"Step {step}: {action.get('action_type')}"
                f"({action.get('section') or action.get('metric') or action.get('decision') or ''})"
                f" -> reward {last_reward:+.2f}"
            )

            if done:
                break

    score = obs_dict.get("final_score") or 0.0
    breakdown = obs_dict.get("grader_breakdown") or {}

    # Emit structured [END] log
    log_end(task_id, score, step, breakdown)

    return {
        "task_id": task_id,
        "final_score": score,
        "grader_breakdown": breakdown,
        "steps_taken": step,
    }


# ── Entry point ──────────────────────────────────────────────────────────────────
def main():

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results = [run_task(llm, t) for t in TASKS]

    print(f"\n{'='*60}\nResults\n{'='*60}")
    total = 0.0
    for r in results:
        total += r["final_score"]
        print(f"  {r['task_id']:15s}  score={r['final_score']:.3f}  steps={r['steps_taken']}")
    print(f"  {'AVERAGE':15s}  score={total / len(results):.3f}")
    return results


if __name__ == "__main__":
    main()
