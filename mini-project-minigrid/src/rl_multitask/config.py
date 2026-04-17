"""Typed configuration for multi-task MiniGrid navigation training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .actions import DEFAULT_POLICY_ACTION_IDS
from .tasks import ENV_NAME_TO_TASK_ID


@dataclass
class TrainConfig:
    """Configuration for teacher collection, BC warm-start, and PPO training.

    Attributes:
        env_names: Training tasks active in the current tranche.
        teacher_episodes_per_env: Demonstrations collected per environment.
        num_envs: Number of parallel rollout environments.
        rollout_steps: Number of environment steps per PPO rollout.
        total_updates: PPO updates to run in the current tranche.
        ppo_epochs: PPO epochs per rollout batch.
        minibatch_size: Legacy flat minibatch size kept for compatibility with
            older PPO experiments.
        recurrent_minibatch_envs: Number of environment sequences per PPO minibatch.
        learning_rate: Optimizer learning rate.
        recurrent_lr_scale: Multiplier applied to the encoder/recurrent context
            parameters relative to `learning_rate`.
        gamma: Reward discount factor.
        gae_lambda: Lambda for generalized advantage estimation.
        clip_coef: PPO clipping coefficient.
        entropy_coef: PPO entropy regularization coefficient.
        value_coef: PPO value loss multiplier.
        max_grad_norm: Global gradient clipping threshold.
        bc_epochs: Behavior cloning epochs before PPO.
        bc_batch_size: Legacy flat BC batch size kept for compatibility.
        bc_sequence_batch_size: Episode batch size for sequence behavior cloning.
        eval_episodes: Evaluation episodes per environment.
        eval_interval: PPO update interval between full evaluations.
        eval_seed_offset: Offset applied to evaluation seeds to avoid teacher-data
            leakage into reported metrics.
        seed: Global random seed.
        save_dir: Directory for datasets, logs, and checkpoints.
        checkpoint_name: Last checkpoint filename.
        best_checkpoint_name: Best-eval checkpoint filename.
        teacher_dataset_name: Demonstration dataset filename.
        resume_from: Optional checkpoint to load before training.
        distill_from: Optional checkpoint used as a frozen policy teacher during
            BC/PPO updates.
        distill_coef: Scalar multiplier for the policy-distillation loss.
        distill_task_ids: Optional task ids whose logits should be anchored to
            the frozen teacher policy.
        device: Preferred torch device.
        obs_embed_dim: Embedding width for symbolic observation channels.
        direction_embed_dim: Embedding width for the direction token.
        task_embed_dim: Embedding width for the task token.
        hidden_dim: Shared latent width.
        cnn_channels: Spatial encoder channel widths.
        use_recurrence: Whether to enable the GRU backbone.
        task_recurrent_task_ids: Optional task ids that receive dedicated GRU
            modules instead of the shared recurrent backbone.
        use_task_heads: Whether to use one policy head per task id.
        use_task_adapters: Whether to use one post-recurrent adapter per task id.
        freeze_shared_backbone: Whether to freeze the shared encoder/recurrent
            backbone during optimization.
        trainable_task_ids: Optional tuple of task ids whose task-specific heads
            remain trainable. When unset, all task heads are trainable.
        action_ids: Policy-index to MiniGrid-action mapping used by the current
            checkpoint.
        action_dim: Derived action vocabulary size.
        task_vocab_size: Optional task-id vocabulary size override. When unset,
            it is inferred from `env_names`.
        use_etd: Whether to enable ETD intrinsic rewards during PPO.
        etd_intrinsic_coef: Scalar multiplier for the ETD intrinsic bonus.
        etd_loss_coef: Scalar multiplier for the ETD auxiliary loss.
        etd_projection_dim: Hidden width for the ETD projection head.
        etd_hidden_dim: Hidden width for the ETD distance regressor.
        etd_max_future_delta: Maximum future-step distance used for ETD targets.
        etd_pairs_per_env: Number of ETD supervision pairs sampled per
            environment sequence.
        etd_memory_limit: Maximum episodic-memory entries retained per
            environment.
        etd_learning_rate: Optimizer learning rate for ETD parameters.
        etd_fourrooms_weight: Task-specific ETD multiplier for FourRooms.
        fourrooms_score_weight: Selection-time weight applied to FourRooms
            success when picking the best checkpoint.
        doorkey_score_weight: Selection-time weight applied to DoorKey success
            when picking the best checkpoint.
    """

    env_names: tuple[str, ...] = (
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-FourRooms-v0",
    )
    teacher_episodes_per_env: int = 256
    num_envs: int = 16
    rollout_steps: int = 128
    total_updates: int = 400
    ppo_epochs: int = 4
    minibatch_size: int = 512
    recurrent_minibatch_envs: int = 4
    learning_rate: float = 3e-4
    recurrent_lr_scale: float = 0.3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    bc_epochs: int = 10
    bc_batch_size: int = 1024
    bc_sequence_batch_size: int = 16
    eval_episodes: int = 20
    eval_interval: int = 20
    eval_seed_offset: int = 100_000
    seed: int = 7
    save_dir: str = "artifacts"
    checkpoint_name: str = "multitask_nav.pt"
    best_checkpoint_name: str = "multitask_nav_best.pt"
    teacher_dataset_name: str = "teacher_nav.npz"
    resume_from: str | None = None
    distill_from: str | None = None
    distill_coef: float = 0.0
    distill_task_ids: tuple[int, ...] | None = None
    device: str = "cuda"
    obs_embed_dim: int = 16
    direction_embed_dim: int = 4
    task_embed_dim: int = 8
    hidden_dim: int = 128
    cnn_channels: tuple[int, int] = field(default_factory=lambda: (64, 64))
    use_recurrence: bool = True
    task_recurrent_task_ids: tuple[int, ...] = ()
    use_task_heads: bool = False
    use_task_adapters: bool = False
    freeze_shared_backbone: bool = False
    trainable_task_ids: tuple[int, ...] | None = None
    action_ids: tuple[int, ...] = field(default_factory=lambda: DEFAULT_POLICY_ACTION_IDS)
    action_dim: int | None = None
    task_vocab_size: int | None = None
    use_etd: bool = False
    etd_intrinsic_coef: float = 0.05
    etd_loss_coef: float = 0.1
    etd_projection_dim: int = 64
    etd_hidden_dim: int = 64
    etd_max_future_delta: int = 16
    etd_pairs_per_env: int = 16
    etd_memory_limit: int = 256
    etd_learning_rate: float = 1e-3
    etd_fourrooms_weight: float = 1.0
    fourrooms_score_weight: float = 3.0
    doorkey_score_weight: float = 3.0

    def __post_init__(self) -> None:
        """Infer derived configuration fields after dataclass initialization."""

        if self.action_dim is None:
            self.action_dim = len(self.action_ids)
        if self.task_vocab_size is None:
            max_task_id = max(int(ENV_NAME_TO_TASK_ID[env_name]) for env_name in self.env_names)
            self.task_vocab_size = max_task_id + 1

    def save_path(self) -> Path:
        """Return the checkpoint path for the current tranche."""

        return Path(self.save_dir) / self.checkpoint_name

    def best_save_path(self) -> Path:
        """Return the best-checkpoint path for the current tranche."""

        return Path(self.save_dir) / self.best_checkpoint_name
