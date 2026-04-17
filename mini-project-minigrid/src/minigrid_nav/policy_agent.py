"""Deployment-time policy bundle for the shipped MiniGrid environments."""

from __future__ import annotations

from dataclasses import dataclass
import io
import lzma
from pathlib import Path

import numpy as np
import torch

from src.rl_multitask.actions import mask_policy_logits
from src.rl_multitask.config import TrainConfig
from src.rl_multitask.model import MultiTaskNavActorCritic
from src.rl_multitask.obs import obs_to_tensors
from src.rl_multitask.tasks import TaskId, infer_task_id_from_observation

CHECKPOINT_PATH = Path(__file__).with_name("policy_bundle.pt.xz")


@dataclass
class _LoadedPolicyExpert:
    """Loaded expert checkpoint with its recurrent deployment state."""

    config: TrainConfig
    model: MultiTaskNavActorCritic
    hidden_state: torch.Tensor
    episode_starts: torch.Tensor

    def reset(self) -> None:
        """Reset the expert recurrent state for a new episode."""

        self.hidden_state.zero_()
        self.episode_starts.fill_(1.0)


class NavigationPolicyController:
    """Deterministic policy-only controller backed by one bundled checkpoint.

    The bundle contains two learned experts:
    - a strong 5-task recurrent policy for `Empty`, `LavaGap`, `FourRooms`,
      `Memory`, and `Dynamic Obstacles`
    - a DoorKey-specialized recurrent policy for `DoorKey`

    Both experts are pure learned policies; no planner or symbolic fallback is
    used at inference time.
    """

    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH) -> None:
        """Load the bundled policy experts.

        Args:
            checkpoint_path: Path to the serialized policy bundle.

        Raises:
            FileNotFoundError: If the bundled checkpoint is missing.
        """

        self.device = torch.device("cpu")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Bundled policy checkpoint not found: {checkpoint_path}")
        with lzma.open(checkpoint_path, "rb") as handle:
            payload = torch.load(io.BytesIO(handle.read()), map_location=self.device)
        self.base_expert = self._load_expert(payload["base"])
        self.doorkey_expert = self._load_expert(payload["doorkey"])
        self.task_id: int | None = None

    def _load_expert(self, payload: dict[str, object]) -> _LoadedPolicyExpert:
        """Deserialize one expert from the policy bundle.

        Args:
            payload: Serialized expert payload containing config and state.

        Returns:
            Loaded expert ready for CPU inference.
        """

        config = TrainConfig(**payload["config"])
        model = MultiTaskNavActorCritic(config).to(self.device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        hidden_state = model.initial_hidden(batch_size=1, device=self.device)
        episode_starts = torch.ones(1, device=self.device, dtype=torch.float32)
        return _LoadedPolicyExpert(
            config=config,
            model=model,
            hidden_state=hidden_state,
            episode_starts=episode_starts,
        )

    def _select_expert(self, task_id: int) -> _LoadedPolicyExpert:
        """Return the deployment expert for the inferred task id."""

        if task_id == int(TaskId.DOOR_KEY):
            return self.doorkey_expert
        return self.base_expert

    def reset(self) -> None:
        """Reset episode-local recurrent state for all bundled experts."""

        self.base_expert.reset()
        self.doorkey_expert.reset()
        self.task_id = None

    def act(self, obs: dict) -> int:
        """Run the appropriate learned expert and return the greedy action.

        Args:
            obs: Raw MiniGrid observation dictionary containing `mission`,
                `image`, and `direction`.

        Returns:
            Greedy environment action from the selected learned expert.

        Raises:
            ValueError: If the observation mission is unsupported.
        """

        # Keep the inferred task identity fixed within an episode. Dynamic
        # Obstacles can temporarily hide all balls from the local 7x7 view,
        # which would otherwise cause a spurious downgrade to Empty.
        if self.task_id is None:
            self.task_id = int(infer_task_id_from_observation(obs))

        expert = self._select_expert(self.task_id)
        batched_obs = {
            "image": obs["image"][None, ...],
            "direction": np.asarray([obs["direction"]], dtype=np.int64),
            "task_id": np.asarray([[self.task_id]], dtype=np.int64),
        }
        tensors = obs_to_tensors(batched_obs, device=self.device)
        with torch.no_grad():
            expert.hidden_state = expert.hidden_state * (1.0 - expert.episode_starts).view(1, 1, 1)
            logits, _, expert.hidden_state = expert.model.forward_step(
                tensors,
                hidden_state=expert.hidden_state,
            )
            masked_logits = mask_policy_logits(
                logits=logits,
                task_ids=tensors["task_id"],
                action_ids=expert.config.action_ids,
            )
            policy_action = torch.argmax(masked_logits, dim=-1).item()

        expert.episode_starts.fill_(0.0)
        return int(expert.config.action_ids[policy_action])
