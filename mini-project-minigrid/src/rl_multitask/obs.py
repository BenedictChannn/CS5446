"""Observation preprocessing for MiniGrid navigation policies."""

from __future__ import annotations

from typing import TypedDict

import torch


class ObsBatch(TypedDict):
    """Typed batch of tensors used by the policy network."""

    obj: torch.Tensor
    color: torch.Tensor
    state: torch.Tensor
    direction: torch.Tensor
    task_id: torch.Tensor


def obs_to_tensors(obs: dict, device: torch.device) -> ObsBatch:
    """Convert vectorized Gym observations into torch tensors."""

    image = torch.as_tensor(obs["image"], device=device, dtype=torch.long)
    direction = torch.as_tensor(obs["direction"], device=device, dtype=torch.long)
    task_id = torch.as_tensor(obs["task_id"], device=device, dtype=torch.long).squeeze(-1)

    return {
        "obj": image[..., 0],
        "color": image[..., 1],
        "state": image[..., 2],
        "direction": direction,
        "task_id": task_id,
    }
