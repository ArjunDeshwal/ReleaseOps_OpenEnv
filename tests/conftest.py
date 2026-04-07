"""Shared fixtures for ReleaseOps-Env tests."""

import pytest
from server.releaseops_environment import ReleaseOpsEnvironment
from releaseops_env.models import ReleaseAction


@pytest.fixture
def env():
    return ReleaseOpsEnvironment()


@pytest.fixture
def easy_env(env):
    env.reset(task_id="easy_001")
    return env


def make_action(**kwargs) -> ReleaseAction:
    return ReleaseAction(**kwargs)
