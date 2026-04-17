"""Shared task identifiers and deployment-time task inference."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class TaskId(IntEnum):
    """Stable task identifiers shared by training and deployment."""

    EMPTY = 0
    LAVA_GAP = 1
    FOUR_ROOMS = 2
    MEMORY = 3
    DYNAMIC_OBSTACLES = 4
    DOOR_KEY = 5


ENV_NAME_TO_TASK_ID: dict[str, TaskId] = {
    "MiniGrid-Empty-8x8-v0": TaskId.EMPTY,
    "MiniGrid-LavaGapS7-v0": TaskId.LAVA_GAP,
    "MiniGrid-FourRooms-v0": TaskId.FOUR_ROOMS,
    "MiniGrid-MemoryS13Random-v0": TaskId.MEMORY,
    "MiniGrid-Dynamic-Obstacles-6x6-v0": TaskId.DYNAMIC_OBSTACLES,
    "MiniGrid-DoorKey-8x8-v0": TaskId.DOOR_KEY,
}

MISSION_TO_TASK_ID: dict[str, TaskId] = {
    "avoid the lava and get to the green goal square": TaskId.LAVA_GAP,
    "reach the goal": TaskId.FOUR_ROOMS,
    "go to the matching object at the end of the hallway": TaskId.MEMORY,
    "use the key to open the door and then get to the goal": TaskId.DOOR_KEY,
}


def infer_task_id_from_observation(obs: dict) -> TaskId:
    """Infer the deployment-time task id from public observation fields.

    Args:
        obs: Raw MiniGrid observation dictionary containing at least `mission`
            and `image`.

    Returns:
        The stable task identifier used by the shared policy.

    Raises:
        ValueError: If the observation mission is unsupported.
    """

    mission = obs["mission"]
    if mission == "get to the green goal square":
        object_ids = set(int(value) for value in np.asarray(obs["image"])[..., 0].reshape(-1).tolist())
        if 6 in object_ids:
            return TaskId.DYNAMIC_OBSTACLES
        return TaskId.EMPTY

    if mission not in MISSION_TO_TASK_ID:
        raise ValueError(f"Unsupported mission: {mission!r}")
    return MISSION_TO_TASK_ID[mission]
