---
title: ReleaseOps-Env
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - sre
  - release-management
  - benchmark
---

# ReleaseOps-Env

A production-grade OpenEnv benchmark for evaluating whether AI agents can safely approve, canary, pause, or roll back risky software changes under incomplete information.

Agents act as SRE reviewers: investigate a proposed change, gather evidence, and submit a final decision. The environment rewards thorough investigation and correct decisions, and penalizes wasted steps and missed risks.

## Setup

```bash
pip install -e ".[dev]"

# Seed the real incident database (requires GitHub PAT with public_repo scope)
GITHUB_TOKEN=<your_token> python3 scripts/seed_db.py

# Or run without a token — uses the 12 curated SRE incidents bundled in the repo
python3 scripts/seed_db.py
```

The incident database (`data/incidents.db`) is pre-seeded with 100+ real incidents from
GitHub Issues (prometheus/prometheus, kubernetes/kubernetes) and curated post-mortems
from companies including Cloudflare, Stripe, AWS, PagerDuty, and Discord. The
`search_incidents` tool queries this real SQLite database, not static JSON.

## Running Locally

```bash
# Start the server
uvicorn server.app:app --port 7860

# In another terminal — run inference (requires MODEL_NAME + API key)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_..."
export ENV_URL="http://localhost:7860"
python3 inference.py

# Or test locally without a server (no API key needed)
python3 local.py all --trace
```

## Quick Start (API)

```bash
# Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_001"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "inspect_change", "section": "diff"}}'

# List tasks and schemas
curl http://localhost:7860/tasks

# Run deterministic baseline (no API key needed)
curl -X POST http://localhost:7860/baseline
```

## Tasks

| Task | Difficulty | Optimal Decision | Description |
|------|-----------|-----------------|-------------|
| `easy_001` | Easy | `request_changes` | Synchronous audit logging on payment hot path — obvious latency risk |
| `easy_002` | Easy | `request_changes` | Connection pool increase risks DB exhaustion — missing DBA approval |
| `medium_001` | Medium | `approve` | Backward-compatible DB index migration — all approvals in place |
| `medium_002` | Medium | `approve` | JWT HS256→RS256 migration — backward-compatible, all checks pass |
| `hard_001` | Hard | `request_changes` | Multi-service retry/concurrency change — requires live telemetry to detect payments-service degradation |
| `hard_002` | Hard | `block` | Rate limit removal from API gateway — requires telemetry to confirm traffic surge risk |

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `inspect_change` | `section`: diff\|tests\|approvals\|files_changed | Read the proposed change |
| `inspect_services` | `service`: name | Check service health and SLA metrics |
| `inspect_dependencies` | — | View blast radius and dependency graph |
| `search_incidents` | `keywords`: list | Search historical incident database |
| `check_policy` | — | Evaluate current rollout policy rules |
| `query_telemetry` | `metric`, `service`, `window` | Query live metrics per rollout phase |
| `request_artifact` | `artifact_type` | Fetch load tests, rollback plans, approvals |
| `control_rollout` | `decision`: start_canary\|promote\|pause\|rollback | Advance the rollout state machine |
| `submit_decision` | `final_decision`, `reason_codes` | End the episode with a final verdict |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | Current task identifier |
| `change_summary` | str | One-line description of the proposed change |
| `known_risk_signals` | list[RiskSignal] | Risks discovered so far (signal_id, severity, summary) |
| `last_tool_result` | ToolResult | Result of the last action taken |
| `allowed_actions` | list[str] | Actions valid in the current rollout phase |
| `rollout_phase` | str | precheck → canary → promoted \| rolled_back |
| `time_remaining` | int | Steps remaining before timeout |
| `cumulative_reward` | float | Running reward total |
| `final_score` | float\|null | Grader score 0.0–1.0 (set on terminal step) |

## Grading Formula

```
score = 0.35 * evidence_coverage
      + 0.25 * risk_signal_discovery
      + 0.30 * decision_correctness
      + 0.10 * efficiency
      - 0.30 * forbidden_penalty
```

Scores clamped to [0.0, 1.0]. Fully deterministic — no LLM judge.

- **evidence_coverage**: fraction of required evidence sources the agent inspected
- **risk_signal_discovery**: fraction of required risk signals the environment emitted during the episode (objective — measures what the agent actually observed, not what strings it typed)
- **decision_correctness**: 1.0 for optimal decision, 0.5 for acceptable, 0.0 for wrong
- **efficiency**: peaks at 1.0 for 30–70% step usage, degrades toward 0 at extremes

Hard tasks require `query_telemetry` to discover critical pre-deployment anomalies. A rule-based
agent that skips telemetry inspection will score ~0.77 on hard tasks, while an agent that
queries live metrics across all affected services scores ~0.98. Easy/medium tasks are solvable
without telemetry.

## Baseline Scores (Heuristic Agent)

| Task | Score | Decision |
|------|-------|----------|
| easy_001 | 0.983 | request_changes |
| easy_002 | 0.983 | request_changes |
| medium_001 | 0.983 | approve |
| medium_002 | 0.983 | approve |
| hard_001 | 0.773 | request_changes |
| hard_002 | 0.760 | block |
| **Average** | **0.911** | |

The gap between easy (0.983) and hard (0.767) scores reflects genuine difficulty: hard tasks
require `query_telemetry` on multiple services to surface pre-deployment metric anomalies that
static diff/test inspection cannot reveal.

Heuristic baseline runs via `curl -X POST http://localhost:7860/baseline` — no LLM required.

## Rollout State Machine

```
precheck --start_canary--> canary --promote--> promoted  [terminal]
                               |
                            rollback --> rolled_back      [terminal]
submit_decision ends the episode from any phase.
```

## Running Inference Script

```bash
export API_BASE_URL="https://openrouter.ai/api/v1"   # or any OpenAI-compatible endpoint
export MODEL_NAME="meta-llama/llama-3.3-70b-instruct"
export OPENAI_API_KEY="sk-..."                         # or HF_TOKEN
export ENV_URL="https://your-space.hf.space"
python3 inference.py
```
# ReleaseOps_OpenEnv
