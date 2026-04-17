"""Train one shared navigation checkpoint with FourRooms-focused PPO."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rl_multitask.config import TrainConfig
from src.rl_multitask.train import train_multitask_policy


def build_config() -> TrainConfig:
    """Build the reproducible PPO tranche for the single-checkpoint target.

    Returns:
        Training configuration for a FourRooms-heavy recurrent PPO run.
    """

    save_dir = Path("artifacts") / "single_checkpoint_ppo"
    resume_path = REPO_ROOT / "src" / "minigrid_nav" / "policy_nav.pt"
    return TrainConfig(
        env_names=(
            "MiniGrid-FourRooms-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-Dynamic-Obstacles-6x6-v0",
            "MiniGrid-Dynamic-Obstacles-6x6-v0",
            "MiniGrid-Dynamic-Obstacles-6x6-v0",
            "MiniGrid-MemoryS13Random-v0",
            "MiniGrid-MemoryS13Random-v0",
            "MiniGrid-LavaGapS7-v0",
            "MiniGrid-Empty-8x8-v0",
        ),
        num_envs=15,
        rollout_steps=256,
        total_updates=80,
        ppo_epochs=4,
        recurrent_minibatch_envs=5,
        learning_rate=1e-4,
        recurrent_lr_scale=0.2,
        entropy_coef=0.03,
        bc_epochs=0,
        eval_episodes=50,
        eval_interval=5,
        save_dir=str(save_dir),
        resume_from=str(resume_path),
        use_etd=False,
        fourrooms_score_weight=4.0,
    )


def main() -> None:
    """Run the single-checkpoint PPO training tranche."""

    config = build_config()
    best_path = train_multitask_policy(config=config)
    print(best_path)


if __name__ == "__main__":
    main()
