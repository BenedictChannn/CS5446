"""Evaluation helpers for trained navigation policies."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch
from torch.distributions.categorical import Categorical

from .actions import mask_policy_logits
from .config import TrainConfig
from .model import MultiTaskNavActorCritic
from .obs import obs_to_tensors
from .tasks import infer_task_id_from_observation


@dataclass
class PolicyEvalSummary:
    """Aggregate policy metrics for one environment."""

    env_id: str
    success_rate: float
    mean_reward: float
    mean_steps: float


def evaluate_policy(
    model: MultiTaskNavActorCritic,
    config: TrainConfig,
    device: torch.device,
) -> list[PolicyEvalSummary]:
    """Run deployment-faithful deterministic evaluations across configured tasks.

    Args:
        model: Policy network to evaluate.
        config: Evaluation configuration. `env_names` is deduplicated before use.
        device: Torch device used for forward passes.

    Returns:
        One aggregate summary per configured environment.
    """

    summaries: list[PolicyEvalSummary] = []
    model.eval()

    for env_name in dict.fromkeys(config.env_names):
        env = gym.make(env_name)
        successes = 0
        total_reward = 0.0
        total_steps = 0
        hidden_state = model.initial_hidden(batch_size=1, device=device)
        episode_starts = torch.ones(1, device=device, dtype=torch.float32)

        for episode in range(config.eval_episodes):
            obs, _ = env.reset(seed=config.seed + config.eval_seed_offset + episode)
            done = False
            steps = 0
            episode_reward = 0.0
            hidden_state.zero_()
            episode_starts.fill_(1.0)

            while not done:
                batched_obs = {
                    "image": obs["image"][None, ...],
                    "direction": [obs["direction"]],
                    "task_id": [[int(infer_task_id_from_observation(obs))]],
                }
                tensors = obs_to_tensors(batched_obs, device=device)
                with torch.no_grad():
                    hidden_state = hidden_state * (1.0 - episode_starts).view(1, 1, 1)
                    logits, _, hidden_state = model.forward_step(tensors, hidden_state=hidden_state)
                    masked_logits = mask_policy_logits(
                        logits=logits,
                        task_ids=tensors["task_id"],
                        action_ids=config.action_ids,
                    )
                    policy_action = Categorical(logits=masked_logits).probs.argmax(dim=-1).item()
                    action = int(config.action_ids[policy_action])

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)
                episode_starts.fill_(1.0 if done else 0.0)
                if terminated:
                    successes += int(float(reward) > 0.0)

            total_reward += episode_reward
            total_steps += steps

        env.close()
        summaries.append(
            PolicyEvalSummary(
                env_id=env_name,
                success_rate=100.0 * successes / config.eval_episodes,
                mean_reward=total_reward / config.eval_episodes,
                mean_steps=total_steps / config.eval_episodes,
            )
        )

    return summaries
