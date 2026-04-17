"""Behavior cloning warm-start and recurrent PPO updates."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader

from .actions import mask_policy_logits
from .config import TrainConfig
from .dataset import TeacherSequenceDataset, collate_teacher_sequences
from .model import MultiTaskNavActorCritic
from .obs import ObsBatch
from .teacher import load_teacher_dataset


def _build_task_mask(task_ids: torch.Tensor, allowed_task_ids: tuple[int, ...] | None) -> torch.Tensor:
    """Return a boolean mask selecting the requested task ids.

    Args:
        task_ids: Flattened task-id tensor.
        allowed_task_ids: Optional tuple of allowed task ids.

    Returns:
        Boolean mask with the same shape as `task_ids`.
    """

    if allowed_task_ids is None:
        return torch.ones_like(task_ids, dtype=torch.bool)

    mask = torch.zeros_like(task_ids, dtype=torch.bool)
    for task_id in allowed_task_ids:
        mask |= task_ids == int(task_id)
    return mask


def _kl_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute KL distillation loss between student and teacher logits."""

    teacher_probs = torch.softmax(teacher_logits.detach(), dim=-1)
    return F.kl_div(
        input=torch.log_softmax(student_logits, dim=-1),
        target=teacher_probs,
        reduction="batchmean",
    )


def bc_pretrain(
    model: MultiTaskNavActorCritic,
    config: TrainConfig,
    dataset_path: Path,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    distill_teacher: MultiTaskNavActorCritic | None = None,
) -> None:
    """Warm-start the shared encoder and policy with teacher data."""

    dataset = TeacherSequenceDataset(load_teacher_dataset(dataset_path))
    loader = DataLoader(
        dataset,
        batch_size=config.bc_sequence_batch_size,
        shuffle=True,
        collate_fn=collate_teacher_sequences,
    )
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for _ in range(config.bc_epochs):
        for batch in loader:
            obs: ObsBatch = {
                "obj": batch["image"][..., 0].to(device),
                "color": batch["image"][..., 1].to(device),
                "state": batch["image"][..., 2].to(device),
                "direction": batch["direction"].to(device),
                "task_id": batch["task_id"].to(device),
            }
            logits, _ = model.forward_sequence(
                obs=obs,
                episode_starts=batch["episode_starts"].to(device),
            )
            mask = batch["mask"].to(device)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_action = batch["action"].to(device).reshape(-1)
            flat_task_ids = batch["task_id"].to(device).reshape(-1)
            flat_mask = mask.reshape(-1) > 0.0
            ce_mask = flat_mask & _build_task_mask(flat_task_ids, config.trainable_task_ids)
            losses: list[torch.Tensor] = []
            if torch.any(ce_mask):
                losses.append(loss_fn(flat_logits[ce_mask], flat_action[ce_mask]))

            if distill_teacher is not None and config.distill_coef > 0.0:
                with torch.no_grad():
                    teacher_logits, _ = distill_teacher.forward_sequence(
                        obs=obs,
                        episode_starts=batch["episode_starts"].to(device),
                    )
                    teacher_logits = mask_policy_logits(
                        logits=teacher_logits,
                        task_ids=batch["task_id"].to(device),
                        action_ids=config.action_ids,
                    )
                distill_mask = flat_mask & _build_task_mask(flat_task_ids, config.distill_task_ids)
                if torch.any(distill_mask):
                    losses.append(
                        config.distill_coef
                        * _kl_distill_loss(
                            student_logits=flat_logits[distill_mask],
                            teacher_logits=teacher_logits.reshape(-1, teacher_logits.shape[-1])[distill_mask],
                        )
                    )

            if not losses:
                continue
            loss = sum(losses)
            if not loss.requires_grad:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()


def iterate_sequence_minibatches(
    num_envs: int,
    minibatch_envs: int,
    device: torch.device,
) -> Iterator[torch.Tensor]:
    """Yield shuffled environment indices for recurrent PPO minibatches."""

    permutation = torch.randperm(num_envs, device=device)
    for start in range(0, num_envs, minibatch_envs):
        yield permutation[start : start + minibatch_envs]


def ppo_update(
    model: MultiTaskNavActorCritic,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    batch,
    distill_teacher: MultiTaskNavActorCritic | None = None,
) -> dict[str, float]:
    """Run one PPO optimization phase."""

    model.train()
    advantages = batch.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
    }
    updates = 0

    for _ in range(config.ppo_epochs):
        for env_indices in iterate_sequence_minibatches(
            num_envs=batch.actions.shape[1],
            minibatch_envs=config.recurrent_minibatch_envs,
            device=batch.actions.device,
        ):
            obs: ObsBatch = {
                "obj": batch.obj[:, env_indices],
                "color": batch.color[:, env_indices],
                "state": batch.state[:, env_indices],
                "direction": batch.direction[:, env_indices],
                "task_id": batch.task_id[:, env_indices],
            }
            logits, values = model.forward_sequence(
                obs,
                episode_starts=batch.episode_starts[:, env_indices],
            )
            masked_logits = mask_policy_logits(
                logits=logits,
                task_ids=batch.task_id[:, env_indices],
                action_ids=config.action_ids,
            )
            flat_logits = masked_logits.reshape(-1, masked_logits.shape[-1])
            flat_values = values.reshape(-1)
            flat_actions = batch.actions[:, env_indices].reshape(-1)
            flat_old_logprobs = batch.logprobs[:, env_indices].reshape(-1)
            flat_advantages = advantages[:, env_indices].reshape(-1)
            flat_returns = batch.returns[:, env_indices].reshape(-1)
            flat_task_ids = batch.task_id[:, env_indices].reshape(-1)
            train_mask = _build_task_mask(flat_task_ids, config.trainable_task_ids)
            if not torch.any(train_mask):
                continue

            dist = Categorical(logits=flat_logits[train_mask])
            new_logprobs = dist.log_prob(flat_actions[train_mask])
            entropy = dist.entropy().mean()

            ratio = (new_logprobs - flat_old_logprobs[train_mask]).exp()
            clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
            surrogate_a = ratio * flat_advantages[train_mask]
            surrogate_b = clipped_ratio * flat_advantages[train_mask]
            policy_loss = -torch.min(surrogate_a, surrogate_b).mean()

            value_loss = nn.functional.mse_loss(flat_values[train_mask], flat_returns[train_mask])
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )
            if distill_teacher is not None and config.distill_coef > 0.0:
                with torch.no_grad():
                    teacher_logits, _ = distill_teacher.forward_sequence(
                        obs,
                        episode_starts=batch.episode_starts[:, env_indices],
                    )
                    teacher_logits = mask_policy_logits(
                        logits=teacher_logits,
                        task_ids=batch.task_id[:, env_indices],
                        action_ids=config.action_ids,
                    )
                distill_mask = _build_task_mask(flat_task_ids, config.distill_task_ids)
                if torch.any(distill_mask):
                    loss = loss + config.distill_coef * _kl_distill_loss(
                        student_logits=flat_logits[distill_mask],
                        teacher_logits=teacher_logits.reshape(-1, teacher_logits.shape[-1])[distill_mask],
                    )
            if not loss.requires_grad:
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy.item())
            updates += 1

    if updates == 0:
        return stats
    return {key: value / updates for key, value in stats.items()}
