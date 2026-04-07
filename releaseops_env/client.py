"""ReleaseOps-Env Client.

Provides the HTTP/WebSocket client for connecting to a ReleaseOps-Env server.
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import (
    ReleaseAction,
    ReleaseObservation,
    ReleaseState,
    RiskSignal,
    ToolResult,
)


class ReleaseOpsEnv(EnvClient[ReleaseAction, ReleaseObservation, ReleaseState]):
    """
    Client for connecting to a ReleaseOps-Env server.

    Example:
        >>> with ReleaseOpsEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(task_id="easy_001")
        ...     print(result.observation.change_summary)
        ...
        ...     result = env.step(ReleaseAction(
        ...         action_type="inspect_change", section="diff"
        ...     ))
        ...     print(result.observation.last_tool_result.content)
        ...
        ...     result = env.step(ReleaseAction(
        ...         action_type="submit_decision",
        ...         final_decision="request_changes",
        ...         reason_codes=["MISSING_LOAD_TEST"],
        ...     ))
        ...     print(f"Score: {result.observation.final_score}")

    Example with Docker:
        >>> client = ReleaseOpsEnv.from_docker_image("releaseops-env:latest")
        >>> try:
        ...     result = client.reset(task_id="easy_001")
        ...     result = client.step(ReleaseAction(
        ...         action_type="inspect_change", section="diff"
        ...     ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ReleaseAction) -> Dict[str, Any]:
        """Serialize action to JSON payload, dropping None fields."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ReleaseObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", payload)

        # Parse nested models
        risk_signals = [
            RiskSignal(**r) if isinstance(r, dict) else r
            for r in obs_data.get("known_risk_signals", [])
        ]

        tool_result = None
        tr_data = obs_data.get("last_tool_result")
        if tr_data and isinstance(tr_data, dict):
            tool_result = ToolResult(**tr_data)

        observation = ReleaseObservation(
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            task_id=obs_data.get("task_id", ""),
            change_summary=obs_data.get("change_summary", ""),
            known_risk_signals=risk_signals,
            last_tool_result=tool_result,
            allowed_actions=obs_data.get("allowed_actions", []),
            rollout_phase=obs_data.get("rollout_phase", "precheck"),
            time_remaining=obs_data.get("time_remaining", 0),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),
            final_score=obs_data.get("final_score"),
            grader_breakdown=obs_data.get("grader_breakdown"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ReleaseState:
        """Parse state response."""
        return ReleaseState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            rollout_phase=payload.get("rollout_phase", "precheck"),
            steps_in_phase=payload.get("steps_in_phase", 0),
            evidence_gathered=payload.get("evidence_gathered", []),
            risk_signals_found=payload.get("risk_signals_found", []),
            artifacts_requested=payload.get("artifacts_requested", []),
            actions_taken=payload.get("actions_taken", []),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            terminal=payload.get("terminal", False),
            final_score=payload.get("final_score"),
        )
