"""Training entry points for the multi-task navigation policy."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import torch
from torch.distributions.categorical import Categorical

from .actions import mask_policy_logits
from .buffer import RolloutBuffer
from .config import TrainConfig
from .etd import EpisodicTemporalDistance, etd_update
from .envs import make_env_factory
from .eval import evaluate_policy
from .model import MultiTaskNavActorCritic
from .obs import obs_to_tensors
from .ppo import bc_pretrain, ppo_update
from .teacher import collect_teacher_dataset


def _build_vector_env(config: TrainConfig) -> gym.vector.VectorEnv:
    """Create a vectorized environment with balanced task sampling."""

    factories = []
    for env_index in range(config.num_envs):
        env_name = config.env_names[env_index % len(config.env_names)]
        factories.append(
            make_env_factory(
                env_name=env_name,
                seed=config.seed + env_index,
                action_ids=config.action_ids,
            )
        )
    return gym.vector.SyncVectorEnv(factories)


def _build_policy_optimizer(
    model: MultiTaskNavActorCritic,
    config: TrainConfig,
) -> torch.optim.Optimizer:
    """Create the optimizer with a lower LR on context-encoder parameters.

    Args:
        model: Shared navigation actor-critic model.
        config: Training configuration for the current tranche.

    Returns:
        Adam optimizer with explicit parameter groups.
    """

    context_parameters = [
        *model.object_embedding.parameters(),
        *model.color_embedding.parameters(),
        *model.state_embedding.parameters(),
        *model.direction_embedding.parameters(),
        *model.task_embedding.parameters(),
        *model.encoder.parameters(),
        *model.pre_recurrent.parameters(),
    ]
    if model.use_recurrence:
        context_parameters.extend(model.recurrent.parameters())
    head_parameters = [*model.value_head.parameters()]
    if getattr(model, "task_recurrents", None) is not None:
        for task_recurrent in model.task_recurrents.values():
            head_parameters.extend(task_recurrent.parameters())
    if model.use_task_adapters:
        for task_adapter in model.task_adapters:
            head_parameters.extend(task_adapter.parameters())
    else:
        head_parameters.extend(model.post_recurrent.parameters())
    if model.use_task_heads:
        for policy_head in model.policy_heads:
            head_parameters.extend(policy_head.parameters())
    else:
        head_parameters.extend(model.policy_head.parameters())
    context_parameters = [parameter for parameter in context_parameters if parameter.requires_grad]
    head_parameters = [parameter for parameter in head_parameters if parameter.requires_grad]
    return torch.optim.Adam(
        [
            {
                "params": context_parameters,
                "lr": config.learning_rate * config.recurrent_lr_scale,
            },
            {
                "params": head_parameters,
                "lr": config.learning_rate,
            },
        ]
    )


def _apply_trainability(
    model: MultiTaskNavActorCritic,
    config: TrainConfig,
) -> None:
    """Freeze or unfreeze model parameters according to the current tranche.

    Args:
        model: Shared navigation actor-critic.
        config: Training configuration for the current tranche.
    """

    if config.freeze_shared_backbone:
        shared_modules = [
            model.object_embedding,
            model.color_embedding,
            model.state_embedding,
            model.direction_embedding,
            model.task_embedding,
            model.encoder,
            model.pre_recurrent,
        ]
        if model.use_recurrence:
            shared_modules.append(model.recurrent)
        for module in shared_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    if model.use_task_heads and config.trainable_task_ids is not None:
        allowed_task_ids = set(int(task_id) for task_id in config.trainable_task_ids)
        for task_id, policy_head in enumerate(model.policy_heads):
            trainable = task_id in allowed_task_ids
            for parameter in policy_head.parameters():
                parameter.requires_grad = trainable
        if model.use_task_adapters:
            for task_id, task_adapter in enumerate(model.task_adapters):
                trainable = task_id in allowed_task_ids
                for parameter in task_adapter.parameters():
                    parameter.requires_grad = trainable
        if getattr(model, "task_recurrents", None) is not None:
            for task_id_string, task_recurrent in model.task_recurrents.items():
                trainable = int(task_id_string) in allowed_task_ids
                for parameter in task_recurrent.parameters():
                    parameter.requires_grad = trainable


def _checkpoint_selection_score(
    config: TrainConfig,
    summaries,
) -> tuple[float, float]:
    """Compute the training-time checkpoint score from eval summaries.

    Args:
        config: Training configuration for the current tranche.
        summaries: Evaluation summaries for the current checkpoint.

    Returns:
        Tuple of `(selection_score, balanced_success)`.
    """

    if not summaries:
        return 0.0, 0.0

    balanced_success = sum(summary.success_rate for summary in summaries) / len(summaries)
    total_weight = 0.0
    weighted_success = 0.0
    for summary in summaries:
        weight = 1.0
        if summary.env_id == "MiniGrid-FourRooms-v0":
            weight = config.fourrooms_score_weight
        if summary.env_id == "MiniGrid-DoorKey-8x8-v0":
            weight = config.doorkey_score_weight
        weighted_success += weight * summary.success_rate
        total_weight += weight
    selection_score = weighted_success / max(total_weight, 1e-8)
    return selection_score, balanced_success


def _load_resume_checkpoint(
    model: MultiTaskNavActorCritic,
    etd_module: EpisodicTemporalDistance | None,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Load a prior checkpoint, expanding the task embedding when needed.

    Args:
        model: Target model receiving the restored weights.
        checkpoint_path: Path to a serialized checkpoint.
        etd_module: Optional ETD module receiving restored weights.
        checkpoint_path: Path to a serialized checkpoint.
        device: Device used to deserialize the checkpoint payload.

    Raises:
        ValueError: If a non-task-embedding tensor shape is incompatible.
    """

    payload = torch.load(checkpoint_path, map_location=device)
    saved_state = payload["model_state"]
    model_state = model.state_dict()

    for key, value in saved_state.items():
        if key in {"policy_head.weight", "policy_head.bias"} and model.use_task_heads:
            suffix = key.split(".", maxsplit=1)[1]
            for target_key in model_state:
                if not target_key.startswith("policy_heads.") or not target_key.endswith(suffix):
                    continue
                rows = min(model_state[target_key].shape[0], value.shape[0])
                if key.endswith("weight"):
                    model_state[target_key][:rows, :] = value[:rows, :]
                else:
                        model_state[target_key][:rows] = value[:rows]
            continue
        if key.startswith("post_recurrent.") and model.use_task_adapters:
            suffix = key.split(".", maxsplit=1)[1]
            for target_key in model_state:
                if not target_key.startswith("task_adapters.") or not target_key.endswith(suffix):
                    continue
                if model_state[target_key].shape == value.shape:
                    model_state[target_key] = value
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
            continue
        if key not in model_state:
            continue
        if model_state[key].shape == value.shape:
            model_state[key] = value
            continue
        if key == "task_embedding.weight" and model_state[key].shape[1] == value.shape[1]:
            # Copy the overlapping task rows when new environments expand the
            # vocabulary between tranches.
            rows = min(model_state[key].shape[0], value.shape[0])
            model_state[key][:rows] = value[:rows]
            continue
        if key in {"policy_head.weight", "policy_head.bias"}:
            if key in model_state:
                rows = min(model_state[key].shape[0], value.shape[0])
                if key.endswith("weight"):
                    model_state[key][:rows, :] = value[:rows, :]
                else:
                    model_state[key][:rows] = value[:rows]
                continue
            continue
        raise ValueError(
            f"Cannot resume checkpoint for parameter {key!r}: "
            f"expected {tuple(model_state[key].shape)}, got {tuple(value.shape)}"
        )

    model.load_state_dict(model_state)
    if etd_module is not None and "etd_state" in payload:
        etd_module.load_state_dict(payload["etd_state"])


def train_multitask_policy(config: TrainConfig) -> Path:
    """Train a shared navigation policy with BC warm-start and PPO.

    Args:
        config: Training configuration for the current tranche.

    Returns:
        Path to the best checkpoint produced during the run.
    """

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = save_dir / config.teacher_dataset_name
    checkpoint_path = config.save_path()
    best_checkpoint_path = config.best_save_path()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = MultiTaskNavActorCritic(config).to(device)
    etd_module = EpisodicTemporalDistance(config).to(device) if config.use_etd else None
    distill_teacher = None

    if config.resume_from is not None:
        _load_resume_checkpoint(
            model=model,
            etd_module=etd_module,
            checkpoint_path=config.resume_from,
            device=device,
        )
    if config.distill_from is not None and config.distill_coef > 0.0:
        distill_teacher = MultiTaskNavActorCritic(config).to(device)
        _load_resume_checkpoint(
            model=distill_teacher,
            etd_module=None,
            checkpoint_path=config.distill_from,
            device=device,
        )
        distill_teacher.eval()
        for parameter in distill_teacher.parameters():
            parameter.requires_grad = False
    _apply_trainability(model=model, config=config)
    optimizer = _build_policy_optimizer(model=model, config=config)
    etd_optimizer = (
        torch.optim.Adam(etd_module.parameters(), lr=config.etd_learning_rate)
        if etd_module is not None
        else None
    )

    if config.bc_epochs > 0:
        if not dataset_path.exists():
            collect_teacher_dataset(config=config, output_path=dataset_path)
        bc_pretrain(
            model=model,
            config=config,
            dataset_path=dataset_path,
            optimizer=optimizer,
            device=device,
            distill_teacher=distill_teacher,
        )

    envs = _build_vector_env(config)
    obs, _ = envs.reset(seed=config.seed)
    episode_starts = torch.ones(config.num_envs, device=device, dtype=torch.float32)
    best_score = float("-inf")

    def maybe_save_best(selection_score: float, balanced_success: float) -> None:
        """Persist the best checkpoint seen so far."""

        nonlocal best_score
        if selection_score < best_score:
            return
        best_score = selection_score
        payload: dict[str, object] = {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "balanced_success": balanced_success,
            "selection_score": selection_score,
        }
        if etd_module is not None:
            payload["etd_state"] = etd_module.state_dict()
        torch.save(payload, best_checkpoint_path)

    if config.eval_episodes > 0:
        warmstart_summaries = evaluate_policy(
            model=model,
            config=config,
            device=device,
        )
        warmstart_score, warmstart_success = _checkpoint_selection_score(
            config=config,
            summaries=warmstart_summaries,
        )
        print(
            f"warmstart balanced_success={warmstart_success:.2f} "
            f"selection_score={warmstart_score:.2f}"
        )
        for summary in warmstart_summaries:
            print(
                f"  {summary.env_id}: success={summary.success_rate:.1f} "
                f"reward={summary.mean_reward:.4f} steps={summary.mean_steps:.1f}"
            )
        maybe_save_best(warmstart_score, warmstart_success)

    for update in range(config.total_updates):
        buffer = RolloutBuffer(
            rollout_steps=config.rollout_steps,
            num_envs=config.num_envs,
            device=device,
        )
        hidden_state = model.initial_hidden(batch_size=config.num_envs, device=device)
        etd_bonus_sum = 0.0
        etd_bonus_count = 0
        if etd_module is not None:
            etd_module.train()
            etd_module.reset(num_envs=config.num_envs)

        for step in range(config.rollout_steps):
            tensors = obs_to_tensors(obs, device=device)
            reward_bonus = torch.zeros(config.num_envs, device=device, dtype=torch.float32)
            if etd_module is not None:
                with torch.no_grad():
                    encoded_features = model.encode(tensors)
                    reward_bonus = etd_module.compute_bonus(
                        encoded_features=encoded_features,
                        task_ids=tensors["task_id"],
                        episode_starts=episode_starts,
                    )
                etd_bonus_sum += float(reward_bonus.mean().item())
                etd_bonus_count += 1

            with torch.no_grad():
                hidden_state = hidden_state * (1.0 - episode_starts).view(1, config.num_envs, 1)
                logits, values, hidden_state = model.forward_step(tensors, hidden_state=hidden_state)
                masked_logits = mask_policy_logits(
                    logits=logits,
                    task_ids=tensors["task_id"],
                    action_ids=config.action_ids,
                )
                dist = Categorical(logits=masked_logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            next_obs, rewards, terminated, truncated, _ = envs.step(actions.cpu().numpy())
            dones = torch.as_tensor(
                terminated | truncated,
                device=device,
                dtype=torch.float32,
            )
            reward_tensor = torch.as_tensor(rewards, device=device, dtype=torch.float32) + reward_bonus
            buffer.store(
                step=step,
                obs=tensors,
                episode_starts=episode_starts,
                actions=actions,
                logprobs=logprobs,
                rewards=reward_tensor,
                dones=dones,
                values=values,
            )
            hidden_state = hidden_state * (1.0 - dones).view(1, config.num_envs, 1)
            episode_starts = dones
            obs = next_obs

        with torch.no_grad():
            last_tensors = obs_to_tensors(obs, device=device)
            hidden_state = hidden_state * (1.0 - episode_starts).view(1, config.num_envs, 1)
            _, last_value, _ = model.forward_step(last_tensors, hidden_state=hidden_state)
        buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        batch = buffer.batch()
        ppo_stats = ppo_update(
            model=model,
            optimizer=optimizer,
            config=config,
            batch=batch,
            distill_teacher=distill_teacher,
        )
        if etd_module is not None and etd_optimizer is not None:
            etd_stats = etd_update(
                etd_module=etd_module,
                model_encoder=model,
                optimizer=etd_optimizer,
                batch_obs={
                    "obj": batch.obj,
                    "color": batch.color,
                    "state": batch.state,
                    "direction": batch.direction,
                    "task_id": batch.task_id,
                },
                task_ids=batch.task_id,
                episode_starts=batch.episode_starts,
                mean_bonus=etd_bonus_sum / max(etd_bonus_count, 1),
            )
            ppo_stats["etd_loss"] = etd_stats.loss
            ppo_stats["etd_bonus"] = etd_stats.mean_bonus

        if (update + 1) % config.eval_interval == 0 or update + 1 == config.total_updates:
            summaries = evaluate_policy(
                model=model,
                config=config,
                device=device,
            )
            selection_score, balanced_success = _checkpoint_selection_score(
                config=config,
                summaries=summaries,
            )
            print(
                f"update={update + 1} "
                f"balanced_success={balanced_success:.2f} "
                f"selection_score={selection_score:.2f} "
                f"policy_loss={ppo_stats['policy_loss']:.4f} "
                f"value_loss={ppo_stats['value_loss']:.4f} "
                f"entropy={ppo_stats['entropy']:.4f}"
            )
            if "etd_loss" in ppo_stats and "etd_bonus" in ppo_stats:
                print(
                    f"  etd_loss={ppo_stats['etd_loss']:.4f} "
                    f"etd_bonus={ppo_stats['etd_bonus']:.4f}"
                )
            for summary in summaries:
                print(
                    f"  {summary.env_id}: success={summary.success_rate:.1f} "
                    f"reward={summary.mean_reward:.4f} steps={summary.mean_steps:.1f}"
                )
            maybe_save_best(selection_score, balanced_success)

    payload = {
        "model_state": model.state_dict(),
        "config": asdict(config),
    }
    if etd_module is not None:
        payload["etd_state"] = etd_module.state_dict()
    torch.save(payload, checkpoint_path)
    envs.close()
    return best_checkpoint_path if best_checkpoint_path.exists() else checkpoint_path
