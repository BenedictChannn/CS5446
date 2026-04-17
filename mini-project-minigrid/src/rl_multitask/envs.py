"""Environment factories for multi-task MiniGrid navigation training."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import minigrid

from .tasks import ENV_NAME_TO_TASK_ID
from .wrappers import MiniGridActionSubsetWrapper, TaskIdWrapper


def make_env_factory(env_name: str, seed: int, action_ids: tuple[int, ...]) -> Callable[[], gym.Env]:
    """Create a thunk that constructs a wrapped training environment.

    Args:
        env_name: Registered MiniGrid environment id.
        seed: Deterministic seed assigned to the thunk instance.
        action_ids: Policy-index to MiniGrid-action mapping for this tranche.

    Returns:
        A zero-argument environment factory suitable for vectorized rollout
        creation.
    """

    task_id = int(ENV_NAME_TO_TASK_ID[env_name])

    def _factory() -> gym.Env:
        env = gym.make(env_name)
        env = MiniGridActionSubsetWrapper(env, action_ids=action_ids)
        env = TaskIdWrapper(env, task_id=task_id)
        env.reset(seed=seed)
        return env

    return _factory
