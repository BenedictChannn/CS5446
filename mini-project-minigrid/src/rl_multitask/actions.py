"""Shared policy-action definitions and task-specific action masking."""

from __future__ import annotations

from enum import IntEnum

import torch

from .tasks import TaskId


class MiniGridAction(IntEnum):
    """Subset of MiniGrid actions used by the learned policies."""

    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    PICKUP = 3
    TOGGLE = 5


DEFAULT_POLICY_ACTION_IDS: tuple[int, ...] = (
    int(MiniGridAction.LEFT),
    int(MiniGridAction.RIGHT),
    int(MiniGridAction.FORWARD),
)
DOORKEY_POLICY_ACTION_IDS: tuple[int, ...] = (
    int(MiniGridAction.LEFT),
    int(MiniGridAction.RIGHT),
    int(MiniGridAction.FORWARD),
    int(MiniGridAction.PICKUP),
    int(MiniGridAction.TOGGLE),
)


def allowed_env_actions_for_task(task_id: int) -> tuple[int, ...]:
    """Return the allowed environment actions for a given task.

    Args:
        task_id: Stable task identifier.

    Returns:
        Tuple of MiniGrid environment action ids permitted for the task.
    """

    if task_id == int(TaskId.DOOR_KEY):
        return DOORKEY_POLICY_ACTION_IDS
    return DEFAULT_POLICY_ACTION_IDS


def mask_policy_logits(
    logits: torch.Tensor,
    task_ids: torch.Tensor,
    action_ids: tuple[int, ...],
) -> torch.Tensor:
    """Mask invalid actions from policy logits using task semantics.

    Args:
        logits: Tensor with shape `(..., action_dim)`.
        task_ids: Tensor with the same leading shape as `logits[..., 0]`.
        action_ids: Policy-index to MiniGrid-action mapping.

    Returns:
        Logits with invalid task-specific actions set to a large negative value.
    """

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_task_ids = task_ids.reshape(-1)
    mask = torch.zeros_like(flat_logits, dtype=torch.bool)

    for row_index in range(flat_logits.shape[0]):
        allowed_actions = allowed_env_actions_for_task(int(flat_task_ids[row_index].item()))
        for action_index, env_action in enumerate(action_ids):
            if env_action in allowed_actions:
                mask[row_index, action_index] = True

    masked = flat_logits.masked_fill(~mask, -1e9)
    return masked.reshape_as(logits)
