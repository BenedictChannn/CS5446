"""Train a 6-task checkpoint with DoorKey learning and old-policy distillation."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rl_multitask.actions import DOORKEY_POLICY_ACTION_IDS
from src.rl_multitask.config import TrainConfig
from src.rl_multitask.tasks import TaskId
from src.rl_multitask.train import train_multitask_policy


def build_config() -> TrainConfig:
    """Build the DoorKey + distillation consolidation tranche.

    Returns:
        Training configuration that trains DoorKey-specific branches while
        distilling the previous strong 5-task policy on the original tasks.
    """

    save_dir = Path("artifacts") / "single_checkpoint_doorkey_distill"
    resume_path = REPO_ROOT / "src" / "minigrid_nav" / "policy_nav.pt"
    old_task_ids = (
        int(TaskId.EMPTY),
        int(TaskId.LAVA_GAP),
        int(TaskId.FOUR_ROOMS),
        int(TaskId.MEMORY),
        int(TaskId.DYNAMIC_OBSTACLES),
    )
    return TrainConfig(
        env_names=(
            "MiniGrid-FourRooms-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-Dynamic-Obstacles-6x6-v0",
            "MiniGrid-Dynamic-Obstacles-6x6-v0",
            "MiniGrid-MemoryS13Random-v0",
            "MiniGrid-MemoryS13Random-v0",
            "MiniGrid-LavaGapS7-v0",
            "MiniGrid-Empty-8x8-v0",
        ),
        teacher_episodes_per_env=256,
        num_envs=20,
        rollout_steps=512,
        total_updates=30,
        ppo_epochs=4,
        recurrent_minibatch_envs=5,
        learning_rate=1e-4,
        recurrent_lr_scale=0.1,
        entropy_coef=0.02,
        bc_epochs=60,
        bc_sequence_batch_size=16,
        eval_episodes=100,
        eval_interval=5,
        save_dir=str(save_dir),
        resume_from=str(resume_path),
        distill_from=str(resume_path),
        distill_coef=0.5,
        distill_task_ids=old_task_ids,
        use_task_heads=True,
        use_task_adapters=True,
        task_recurrent_task_ids=(int(TaskId.DOOR_KEY),),
        trainable_task_ids=(int(TaskId.DOOR_KEY),),
        action_ids=DOORKEY_POLICY_ACTION_IDS,
        use_etd=False,
        fourrooms_score_weight=5.0,
        doorkey_score_weight=4.0,
    )


def main() -> None:
    """Run the DoorKey + distillation consolidation tranche."""

    config = build_config()
    best_path = train_multitask_policy(config=config)
    print(best_path)


if __name__ == "__main__":
    main()
