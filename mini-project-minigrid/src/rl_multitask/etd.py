"""Episodic Temporal Distance utilities for recurrent PPO training.

This module implements a lightweight ETD-style intrinsic reward head. It is
training-only: the exported policy checkpoint does not depend on this module at
inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import TrainConfig
from .obs import ObsBatch
from .tasks import TaskId


@dataclass
class ETDUpdateStats:
    """Aggregate metrics from one ETD auxiliary update.

    Attributes:
        loss: Average temporal-distance regression loss.
        mean_bonus: Mean intrinsic bonus assigned during the rollout.
    """

    loss: float = 0.0
    mean_bonus: float = 0.0


class EpisodicTemporalDistance(nn.Module):
    """Approximate ETD module with episodic memory and temporal-distance loss.

    The implementation is intentionally compact. It uses the policy encoder's
    pre-recurrent features, learns a temporal-distance regressor on top of those
    features, and computes episodic novelty bonuses from the minimum predicted
    distance to prior states in the same episode.

    Args:
        config: Training configuration for the current tranche.
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.intrinsic_coef = config.etd_intrinsic_coef
        self.loss_coef = config.etd_loss_coef
        self.max_future_delta = config.etd_max_future_delta
        self.pairs_per_env = config.etd_pairs_per_env
        self.memory_limit = config.etd_memory_limit
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.etd_projection_dim),
            nn.ReLU(),
            nn.Linear(config.etd_projection_dim, config.etd_projection_dim),
        )
        self.distance_head = nn.Sequential(
            nn.Linear(config.etd_projection_dim * 3, config.etd_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.etd_hidden_dim, 1),
            nn.Softplus(),
        )
        task_weights = torch.zeros(config.task_vocab_size, dtype=torch.float32)
        task_weights[int(TaskId.FOUR_ROOMS)] = config.etd_fourrooms_weight
        self.register_buffer("task_weights", task_weights)
        self._episode_memory: list[list[torch.Tensor]] = []

    def reset(self, num_envs: int) -> None:
        """Reset episodic memory for a new vectorized rollout.

        Args:
            num_envs: Number of parallel environments in the rollout.
        """

        self._episode_memory = [[] for _ in range(num_envs)]

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project policy features into the temporal-distance embedding space.

        Args:
            features: Tensor of shape `(B, H)` or `(T, B, H)`.

        Returns:
            Projected embeddings with the same leading dimensions.
        """

        return self.projection(features)

    def predict_distance(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict asymmetric temporal distance between two embeddings.

        Args:
            source_embedding: Source tensor with shape `(..., D)`.
            target_embedding: Target tensor with shape `(..., D)`.

        Returns:
            Positive scalar distances with shape `(...)`.
        """

        features = torch.cat(
            [
                source_embedding,
                target_embedding,
                torch.abs(target_embedding - source_embedding),
            ],
            dim=-1,
        )
        return self.distance_head(features).squeeze(-1)

    def compute_bonus(
        self,
        encoded_features: torch.Tensor,
        task_ids: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ETD bonuses and update episodic memory.

        Args:
            encoded_features: Policy encoder features with shape `(B, H)`.
            task_ids: Task identifiers with shape `(B,)`.
            episode_starts: Float reset markers with shape `(B,)`.

        Returns:
            Intrinsic reward bonuses with shape `(B,)`.
        """

        if len(self._episode_memory) != encoded_features.shape[0]:
            self.reset(num_envs=encoded_features.shape[0])

        projected = self.project_features(encoded_features)
        bonuses = torch.zeros(encoded_features.shape[0], device=encoded_features.device)

        for env_index in range(encoded_features.shape[0]):
            if float(episode_starts[env_index].item()) > 0.5:
                self._episode_memory[env_index] = []

            task_weight = self.task_weights[int(task_ids[env_index].item())]
            current = projected[env_index]
            if task_weight > 0.0 and self._episode_memory[env_index]:
                memory = torch.stack(self._episode_memory[env_index], dim=0).to(encoded_features.device)
                repeated_current = current.unsqueeze(0).expand(memory.shape[0], -1)
                distances = self.predict_distance(repeated_current, memory)
                bonuses[env_index] = self.intrinsic_coef * task_weight * distances.min()

            self._episode_memory[env_index].append(current.detach().cpu())
            if len(self._episode_memory[env_index]) > self.memory_limit:
                self._episode_memory[env_index].pop(0)

        return bonuses

    def temporal_distance_loss(
        self,
        encoded_sequence: torch.Tensor,
        task_ids: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a temporal-distance regression loss over rollout sequences.

        Args:
            encoded_sequence: Policy encoder features with shape `(T, B, H)`.
            task_ids: Task identifiers with shape `(T, B)`.
            episode_starts: Episode reset markers with shape `(T, B)`.

        Returns:
            Scalar ETD auxiliary loss.
        """

        projected = self.project_features(encoded_sequence)
        episode_ids = torch.cumsum(episode_starts.to(torch.int64), dim=0)
        losses: list[torch.Tensor] = []
        time_steps = projected.shape[0]
        batch_size = projected.shape[1]

        for env_index in range(batch_size):
            task_weight = self.task_weights[int(task_ids[0, env_index].item())]
            if task_weight <= 0.0:
                continue

            valid_pairs = 0
            for _ in range(self.pairs_per_env):
                start_index = int(
                    torch.randint(low=0, high=max(time_steps - 1, 1), size=(1,), device=projected.device).item()
                )
                max_delta = min(self.max_future_delta, time_steps - start_index - 1)
                if max_delta <= 0:
                    continue
                delta = int(torch.randint(low=1, high=max_delta + 1, size=(1,), device=projected.device).item())
                end_index = start_index + delta
                if int(episode_ids[start_index, env_index].item()) != int(episode_ids[end_index, env_index].item()):
                    continue

                predicted = self.predict_distance(
                    projected[start_index, env_index],
                    projected[end_index, env_index],
                )
                target = torch.tensor(
                    delta / float(self.max_future_delta),
                    device=projected.device,
                    dtype=predicted.dtype,
                )
                losses.append(task_weight * nn.functional.smooth_l1_loss(predicted, target))
                valid_pairs += 1

            if valid_pairs == 0:
                continue

        if not losses:
            return torch.zeros((), device=encoded_sequence.device)
        return self.loss_coef * torch.stack(losses).mean()


def etd_update(
    etd_module: EpisodicTemporalDistance,
    model_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_obs: ObsBatch,
    task_ids: torch.Tensor,
    episode_starts: torch.Tensor,
    mean_bonus: float,
) -> ETDUpdateStats:
    """Run one ETD auxiliary update on the collected rollout.

    Args:
        etd_module: ETD module being optimized.
        model_encoder: Policy model exposing an `encode` method.
        optimizer: Optimizer for ETD parameters.
        batch_obs: Sequence-major rollout observations.
        task_ids: Sequence-major task identifiers.
        episode_starts: Sequence-major episode reset markers.
        mean_bonus: Mean intrinsic bonus assigned during the rollout.

    Returns:
        Summary statistics for the ETD update.
    """

    flat_obs: ObsBatch = {
        "obj": batch_obs["obj"].reshape(-1, 7, 7),
        "color": batch_obs["color"].reshape(-1, 7, 7),
        "state": batch_obs["state"].reshape(-1, 7, 7),
        "direction": batch_obs["direction"].reshape(-1),
        "task_id": batch_obs["task_id"].reshape(-1),
    }
    with torch.no_grad():
        encoded = model_encoder.encode(flat_obs).reshape(
            batch_obs["obj"].shape[0],
            batch_obs["obj"].shape[1],
            -1,
        )

    loss = etd_module.temporal_distance_loss(
        encoded_sequence=encoded,
        task_ids=task_ids,
        episode_starts=episode_starts,
    )

    if not loss.requires_grad:
        return ETDUpdateStats(loss=0.0, mean_bonus=mean_bonus)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return ETDUpdateStats(loss=float(loss.item()), mean_bonus=mean_bonus)
