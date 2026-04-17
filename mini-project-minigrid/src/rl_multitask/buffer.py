"""Sequence-major rollout buffer for recurrent PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    """Sequence-major PPO batch."""

    obj: torch.Tensor
    color: torch.Tensor
    state: torch.Tensor
    direction: torch.Tensor
    task_id: torch.Tensor
    episode_starts: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    """Fixed-shape rollout buffer for recurrent PPO sequences."""

    def __init__(self, rollout_steps: int, num_envs: int, device: torch.device):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device
        self.obj = torch.zeros((rollout_steps, num_envs, 7, 7), dtype=torch.long, device=device)
        self.color = torch.zeros((rollout_steps, num_envs, 7, 7), dtype=torch.long, device=device)
        self.state = torch.zeros((rollout_steps, num_envs, 7, 7), dtype=torch.long, device=device)
        self.direction = torch.zeros((rollout_steps, num_envs), dtype=torch.long, device=device)
        self.task_id = torch.zeros((rollout_steps, num_envs), dtype=torch.long, device=device)
        self.episode_starts = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.actions = torch.zeros((rollout_steps, num_envs), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)

    def store(
        self,
        step: int,
        obs: dict[str, torch.Tensor],
        episode_starts: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Store one vectorized rollout step."""

        self.obj[step] = obs["obj"]
        self.color[step] = obs["color"]
        self.state[step] = obs["state"]
        self.direction[step] = obs["direction"]
        self.task_id[step] = obs["task_id"]
        self.episode_starts[step] = episode_starts
        self.actions[step] = actions
        self.logprobs[step] = logprobs
        self.rewards[step] = rewards
        self.dones[step] = dones
        self.values[step] = values

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Run backward GAE on the rollout."""

        next_advantage = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        next_value = last_value

        for step in reversed(range(self.rollout_steps)):
            mask = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_value * mask - self.values[step]
            next_advantage = delta + gamma * gae_lambda * mask * next_advantage
            self.advantages[step] = next_advantage
            self.returns[step] = next_advantage + self.values[step]
            next_value = self.values[step]

    def batch(self) -> RolloutBatch:
        """Return the rollout tensors without flattening sequence structure."""

        return RolloutBatch(
            obj=self.obj,
            color=self.color,
            state=self.state,
            direction=self.direction,
            task_id=self.task_id,
            episode_starts=self.episode_starts,
            actions=self.actions,
            logprobs=self.logprobs,
            advantages=self.advantages,
            returns=self.returns,
            values=self.values,
        )
