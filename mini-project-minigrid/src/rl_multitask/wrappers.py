"""Environment wrappers for multi-task MiniGrid training."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class MiniGridActionSubsetWrapper(gym.ActionWrapper):
    """Map compact policy actions to a subset of MiniGrid environment actions.

    Args:
        env: Wrapped MiniGrid environment.
        action_ids: Tuple mapping policy indices to raw MiniGrid action ids.
    """

    def __init__(self, env: gym.Env, action_ids: tuple[int, ...]):
        super().__init__(env)
        self.action_ids = tuple(int(action_id) for action_id in action_ids)
        self.action_space = gym.spaces.Discrete(len(self.action_ids))

    def action(self, action: int) -> int:
        """Translate a compact policy action into a MiniGrid action id.

        Args:
            action: Policy action index.

        Returns:
            Raw MiniGrid action id.
        """

        return int(self.action_ids[int(action)])


class ThreeActionWrapper(MiniGridActionSubsetWrapper):
    """Backward-compatible wrapper for the historical 3-action policy."""

    def __init__(self, env: gym.Env):
        super().__init__(env=env, action_ids=(0, 1, 2))


class TaskIdWrapper(gym.ObservationWrapper):
    """Append task metadata directly to the observation dictionary."""

    def __init__(self, env: gym.Env, task_id: int):
        super().__init__(env)
        self.task_id = int(task_id)
        self.observation_space = gym.spaces.Dict(
            {
                "image": env.observation_space["image"],
                "direction": env.observation_space["direction"],
                "task_id": gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.int64),
            }
        )

    def observation(self, observation: dict) -> dict:
        return {
            "image": observation["image"],
            "direction": observation["direction"],
            "task_id": np.array([self.task_id], dtype=np.int64),
        }
