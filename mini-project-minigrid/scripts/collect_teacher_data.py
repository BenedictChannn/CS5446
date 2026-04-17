"""Collect teacher demonstrations for the multi-task navigation family."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rl_multitask.config import TrainConfig
from src.rl_multitask.teacher import collect_teacher_dataset


def main() -> None:
    """Collect and persist teacher demonstrations."""

    config = TrainConfig()
    output_path = Path(config.save_dir) / config.teacher_dataset_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = collect_teacher_dataset(config=config, output_path=output_path)
    print(output_path)
    print(dataset.image.shape[0])


if __name__ == "__main__":
    main()
