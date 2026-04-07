"""ReleaseOps-Env: Production change review environment for OpenEnv."""

from releaseops_env.models import (
    ReleaseAction,
    ReleaseObservation,
    ReleaseState,
    RiskSignal,
    ToolResult,
)


# Client import is deferred to avoid circular imports and to allow
# usage without openenv.core installed (e.g., server-side only).
def __getattr__(name):
    if name == "ReleaseOpsEnv":
        from releaseops_env.client import ReleaseOpsEnv

        return ReleaseOpsEnv
    raise AttributeError(f"module 'releaseops_env' has no attribute {name!r}")


__all__ = [
    "ReleaseOpsEnv",
    "ReleaseAction",
    "ReleaseObservation",
    "ReleaseState",
    "RiskSignal",
    "ToolResult",
]
