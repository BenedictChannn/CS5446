"""Shared multi-task actor-critic for MiniGrid navigation tasks."""

from __future__ import annotations

import torch
from torch import nn

from .config import TrainConfig
from .obs import ObsBatch


class MultiTaskNavActorCritic(nn.Module):
    """Compact symbolic-spatial actor-critic network."""

    def __init__(self, config: TrainConfig):
        super().__init__()
        embed_dim = config.obs_embed_dim
        self.object_embedding = nn.Embedding(11, embed_dim)
        self.color_embedding = nn.Embedding(6, embed_dim)
        self.state_embedding = nn.Embedding(3, embed_dim)
        self.direction_embedding = nn.Embedding(4, config.direction_embed_dim)
        self.task_embedding = nn.Embedding(config.task_vocab_size, config.task_embed_dim)

        input_channels = embed_dim * 3
        conv_a, conv_b = config.cnn_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, conv_a, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_a, conv_b, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        spatial_dim = conv_b * 7 * 7
        fused_dim = spatial_dim + config.direction_embed_dim + config.task_embed_dim
        self.pre_recurrent = nn.Sequential(
            nn.Linear(fused_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.use_recurrence = config.use_recurrence
        self.task_recurrent_task_ids = {int(task_id) for task_id in config.task_recurrent_task_ids}
        if self.use_recurrence:
            self.recurrent = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                batch_first=False,
            )
            self.task_recurrents = nn.ModuleDict(
                {
                    str(task_id): nn.GRU(
                        input_size=config.hidden_dim,
                        hidden_size=config.hidden_dim,
                        batch_first=False,
                    )
                    for task_id in sorted(self.task_recurrent_task_ids)
                }
            )
        self.post_recurrent = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.use_task_adapters = config.use_task_adapters
        if self.use_task_adapters:
            self.task_adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(config.hidden_dim, config.hidden_dim),
                        nn.ReLU(),
                    )
                    for _ in range(config.task_vocab_size)
                ]
            )
        self.use_task_heads = config.use_task_heads
        if self.use_task_heads:
            self.policy_heads = nn.ModuleList(
                [nn.Linear(config.hidden_dim, config.action_dim) for _ in range(config.task_vocab_size)]
            )
        else:
            self.policy_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def _policy_logits(self, latent: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Project latent features into task-aware policy logits.

        Args:
            latent: Tensor with shape `(B, H)` or `(T, B, H)`.
            task_ids: Matching task-id tensor with shape `(B,)` or `(T, B)`.

        Returns:
            Policy logits with shape `(..., action_dim)`.
        """

        if not self.use_task_heads:
            return self.policy_head(latent)

        flat_latent = latent.reshape(-1, latent.shape[-1])
        flat_task_ids = task_ids.reshape(-1)
        flat_logits = torch.zeros(
            (flat_latent.shape[0], self.policy_heads[0].out_features),
            device=flat_latent.device,
            dtype=flat_latent.dtype,
        )
        for task_id, head in enumerate(self.policy_heads):
            task_mask = flat_task_ids == task_id
            if torch.any(task_mask):
                flat_logits[task_mask] = head(flat_latent[task_mask])
        return flat_logits.reshape(*latent.shape[:-1], -1)

    def _adapt_latent(self, latent: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Apply the shared or task-specific post-recurrent adapter.

        Args:
            latent: Tensor with shape `(B, H)` or `(T, B, H)`.
            task_ids: Matching task-id tensor.

        Returns:
            Adapted latent tensor with the same leading dimensions as `latent`.
        """

        if not self.use_task_adapters:
            return self.post_recurrent(latent)

        flat_latent = latent.reshape(-1, latent.shape[-1])
        flat_task_ids = task_ids.reshape(-1)
        adapted = torch.zeros_like(flat_latent)
        for task_id, adapter in enumerate(self.task_adapters):
            task_mask = flat_task_ids == task_id
            if torch.any(task_mask):
                adapted[task_mask] = adapter(flat_latent[task_mask])
        return adapted.reshape_as(latent)

    def encode(self, obs: ObsBatch) -> torch.Tensor:
        """Encode symbolic observations into compact latent features."""

        object_feat = self.object_embedding(obs["obj"])
        color_feat = self.color_embedding(obs["color"])
        state_feat = self.state_embedding(obs["state"])

        features = torch.cat([object_feat, color_feat, state_feat], dim=-1)
        features = features.permute(0, 3, 1, 2)
        spatial = self.encoder(features)

        direction = self.direction_embedding(obs["direction"])
        task = self.task_embedding(obs["task_id"])
        latent = torch.cat([spatial, direction, task], dim=-1)
        return self.pre_recurrent(latent)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create an initial recurrent state tensor."""

        return torch.zeros(1, batch_size, self.value_head.in_features, device=device)

    def _forward_recurrent_step(
        self,
        latent: torch.Tensor,
        task_ids: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one recurrent update with optional task-specific GRU routing.

        Args:
            latent: Encoded latent tensor with shape `(B, H)`.
            task_ids: Task-id tensor with shape `(B,)`.
            hidden_state: Prior hidden state with shape `(1, B, H)`.

        Returns:
            Tuple of `(recurrent_output, next_hidden_state)`.
        """

        if hidden_state is None:
            hidden_state = self.initial_hidden(batch_size=latent.shape[0], device=latent.device)

        if not self.task_recurrent_task_ids:
            recurrent_output, next_hidden = self.recurrent(latent.unsqueeze(0), hidden_state)
            return recurrent_output.squeeze(0), next_hidden

        next_hidden = hidden_state.clone()
        outputs = torch.zeros_like(latent)
        assigned_mask = torch.zeros(latent.shape[0], device=latent.device, dtype=torch.bool)

        for task_id in sorted(self.task_recurrent_task_ids):
            task_mask = task_ids == task_id
            if not torch.any(task_mask):
                continue
            recurrent_output, updated_hidden = self.task_recurrents[str(task_id)](
                latent[task_mask].unsqueeze(0),
                hidden_state[:, task_mask, :],
            )
            outputs[task_mask] = recurrent_output.squeeze(0)
            next_hidden[:, task_mask, :] = updated_hidden
            assigned_mask |= task_mask

        shared_mask = ~assigned_mask
        if torch.any(shared_mask):
            recurrent_output, updated_hidden = self.recurrent(
                latent[shared_mask].unsqueeze(0),
                hidden_state[:, shared_mask, :],
            )
            outputs[shared_mask] = recurrent_output.squeeze(0)
            next_hidden[:, shared_mask, :] = updated_hidden

        return outputs, next_hidden

    def forward_step(
        self,
        obs: ObsBatch,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run one batched environment step through the actor-critic."""

        latent = self.encode(obs)
        next_hidden = hidden_state

        if self.use_recurrence:
            latent, next_hidden = self._forward_recurrent_step(
                latent=latent,
                task_ids=obs["task_id"],
                hidden_state=hidden_state,
            )

        latent = self._adapt_latent(latent, obs["task_id"])
        logits = self._policy_logits(latent, obs["task_id"])
        value = self.value_head(latent).squeeze(-1)
        return logits, value, next_hidden

    def forward_sequence(
        self,
        obs: ObsBatch,
        episode_starts: torch.Tensor,
        initial_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a full rollout sequence through the actor-critic.

        Args:
            obs: Sequence-major observation batch with shape `(T, B, ...)`.
            episode_starts: Float tensor of shape `(T, B)` marking episode starts.
            initial_hidden: Optional recurrent state with shape `(1, B, H)`.

        Returns:
            Tuple of action logits and value predictions, both sequence-major.
        """

        time_steps = obs["obj"].shape[0]
        batch_size = obs["obj"].shape[1]
        flat_obs: ObsBatch = {
            "obj": obs["obj"].reshape(-1, 7, 7),
            "color": obs["color"].reshape(-1, 7, 7),
            "state": obs["state"].reshape(-1, 7, 7),
            "direction": obs["direction"].reshape(-1),
            "task_id": obs["task_id"].reshape(-1),
        }
        encoded = self.encode(flat_obs).reshape(time_steps, batch_size, -1)

        if not self.use_recurrence:
            latent = self.post_recurrent(encoded.reshape(-1, encoded.shape[-1])).reshape(
                time_steps,
                batch_size,
                -1,
            )
            logits = self.policy_head(latent)
            value = self.value_head(latent).squeeze(-1)
            return logits, value

        if initial_hidden is None:
            hidden = self.initial_hidden(batch_size=batch_size, device=encoded.device)
        else:
            hidden = initial_hidden

        outputs: list[torch.Tensor] = []
        for step in range(time_steps):
            reset_mask = (1.0 - episode_starts[step]).view(1, batch_size, 1)
            hidden = hidden * reset_mask
            recurrent_output, hidden = self._forward_recurrent_step(
                latent=encoded[step],
                task_ids=obs["task_id"][step],
                hidden_state=hidden,
            )
            outputs.append(recurrent_output)

        latent = torch.stack(outputs, dim=0)
        latent = self._adapt_latent(latent, obs["task_id"])
        logits = self._policy_logits(latent, obs["task_id"])
        value = self.value_head(latent).squeeze(-1)
        return logits, value

    def forward(self, obs: ObsBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility forward method for single-step callers."""

        logits, value, _ = self.forward_step(obs)
        return logits, value
