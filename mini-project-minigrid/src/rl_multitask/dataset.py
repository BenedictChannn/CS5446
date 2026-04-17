"""PyTorch dataset wrapper for teacher demonstrations."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .teacher import TeacherDataset


class TeacherTorchDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset wrapper exposing teacher demonstrations to DataLoader."""

    def __init__(self, dataset: TeacherDataset):
        self.image = torch.as_tensor(dataset.image, dtype=torch.long)
        self.direction = torch.as_tensor(dataset.direction, dtype=torch.long)
        self.task_id = torch.as_tensor(dataset.task_id, dtype=torch.long)
        self.action = torch.as_tensor(dataset.action, dtype=torch.long)
        self.episode_starts = torch.as_tensor(dataset.episode_starts, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.action.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.image[index],
            "direction": self.direction[index],
            "task_id": self.task_id[index],
            "action": self.action[index],
            "episode_starts": self.episode_starts[index],
        }


class TeacherSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Episode-wise dataset wrapper for recurrent behavior cloning."""

    def __init__(self, dataset: TeacherDataset):
        self.image = torch.as_tensor(dataset.image, dtype=torch.long)
        self.direction = torch.as_tensor(dataset.direction, dtype=torch.long)
        self.task_id = torch.as_tensor(dataset.task_id, dtype=torch.long)
        self.action = torch.as_tensor(dataset.action, dtype=torch.long)
        self.episode_starts = torch.as_tensor(dataset.episode_starts, dtype=torch.long)

        self.episode_ranges: list[tuple[int, int]] = []
        start_indices = torch.nonzero(self.episode_starts, as_tuple=False).flatten().tolist()
        for range_index, start in enumerate(start_indices):
            end = start_indices[range_index + 1] if range_index + 1 < len(start_indices) else len(self.action)
            self.episode_ranges.append((start, end))

    def __len__(self) -> int:
        return len(self.episode_ranges)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start, end = self.episode_ranges[index]
        length = end - start
        return {
            "image": self.image[start:end],
            "direction": self.direction[start:end],
            "task_id": self.task_id[start:end],
            "action": self.action[start:end],
            "episode_starts": torch.cat(
                [
                    torch.ones(1, dtype=torch.float32),
                    torch.zeros(max(length - 1, 0), dtype=torch.float32),
                ]
            ),
            "mask": torch.ones(length, dtype=torch.float32),
        }


def collate_teacher_sequences(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length teacher episodes into sequence-major tensors."""

    max_length = max(item["action"].shape[0] for item in batch)
    batch_size = len(batch)
    image = torch.zeros((max_length, batch_size, 7, 7, 3), dtype=torch.long)
    direction = torch.zeros((max_length, batch_size), dtype=torch.long)
    task_id = torch.zeros((max_length, batch_size), dtype=torch.long)
    action = torch.zeros((max_length, batch_size), dtype=torch.long)
    episode_starts = torch.zeros((max_length, batch_size), dtype=torch.float32)
    mask = torch.zeros((max_length, batch_size), dtype=torch.float32)

    for batch_index, item in enumerate(batch):
        length = item["action"].shape[0]
        image[:length, batch_index] = item["image"]
        direction[:length, batch_index] = item["direction"]
        task_id[:length, batch_index] = item["task_id"]
        action[:length, batch_index] = item["action"]
        episode_starts[:length, batch_index] = item["episode_starts"]
        mask[:length, batch_index] = item["mask"]

    return {
        "image": image,
        "direction": direction,
        "task_id": task_id,
        "action": action,
        "episode_starts": episode_starts,
        "mask": mask,
    }
