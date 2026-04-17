"""Teacher data generation using oracle access to MiniGrid state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy

import numpy as np
from minigrid.core.constants import DIR_TO_VEC

from utils import make_env

from .config import TrainConfig
from .tasks import ENV_NAME_TO_TASK_ID


@dataclass
class TeacherDataset:
    """Teacher dataset stored as numpy arrays."""

    image: np.ndarray
    direction: np.ndarray
    task_id: np.ndarray
    action: np.ndarray
    episode_starts: np.ndarray


@dataclass
class MemoryTeacher:
    """Episode-stateful causal teacher for the Memory environment."""

    cue_observed: bool = False

    def reset(self) -> None:
        """Reset episode-local teacher memory."""

        self.cue_observed = False

    def act(self, inner_env) -> int:
        """Return the next causal teacher action."""

        inspect_position = (3, inner_env.height // 2)
        inspect_direction = 3
        current_position = tuple(int(value) for value in inner_env.agent_pos)

        if not self.cue_observed:
            if current_position != inspect_position:
                original_success = inner_env.success_pos
                inner_env.success_pos = inspect_position
                try:
                    return _oracle_action(inner_env)
                finally:
                    inner_env.success_pos = original_success

            if int(inner_env.agent_dir) != inspect_direction:
                return _turn_action(int(inner_env.agent_dir), inspect_direction)

            self.cue_observed = True

        return _oracle_action(inner_env)


def _find_goal_position(inner_env) -> tuple[int, int]:
    """Return the environment's success target in world coordinates."""

    if hasattr(inner_env, "success_pos"):
        success_x, success_y = inner_env.success_pos
        return int(success_x), int(success_y)

    for x in range(inner_env.width):
        for y in range(inner_env.height):
            cell = inner_env.grid.get(x, y)
            if cell is not None and type(cell).__name__ == "Goal":
                return x, y
    raise ValueError("Could not find a target goal position in the environment grid.")


def _find_positions_by_type(inner_env, cell_name: str) -> list[tuple[int, int]]:
    """Return all world positions matching a MiniGrid cell type name.

    Args:
        inner_env: Unwrapped MiniGrid environment.
        cell_name: Concrete world-object class name.

    Returns:
        World positions containing the requested cell type.
    """

    positions: list[tuple[int, int]] = []
    for x in range(inner_env.width):
        for y in range(inner_env.height):
            cell = inner_env.grid.get(x, y)
            if cell is not None and type(cell).__name__ == cell_name:
                positions.append((x, y))
    return positions


def _cell_is_traversable(inner_env, x: int, y: int) -> bool:
    """Return whether the oracle planner may enter a world cell."""

    cell = inner_env.grid.get(x, y)
    if cell is None:
        return True
    cell_name = type(cell).__name__
    return cell_name not in {"Wall", "Lava", "Ball", "Key", "Box"}


def _cell_is_traversable_for_doorkey(
    inner_env,
    x: int,
    y: int,
    *,
    allow_goal: bool = True,
) -> bool:
    """Return whether a DoorKey planner may occupy a world cell.

    Args:
        inner_env: Unwrapped MiniGrid environment.
        x: World x coordinate.
        y: World y coordinate.
        allow_goal: Whether the goal cell should be treated as traversable.

    Returns:
        True when the planner may stand on the cell.
    """

    cell = inner_env.grid.get(x, y)
    if cell is None:
        return True
    cell_name = type(cell).__name__
    if cell_name == "Wall":
        return False
    if cell_name == "Door":
        return bool(cell.is_open)
    if cell_name == "Goal":
        return allow_goal
    return cell_name not in {"Lava", "Ball", "Box", "Key"}


def _neighbors(inner_env, position: tuple[int, int]) -> list[tuple[int, int]]:
    """Return traversable neighbors inside the grid bounds."""

    x, y = position
    candidates = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
    valid: list[tuple[int, int]] = []
    for next_x, next_y in candidates:
        if next_x < 0 or next_x >= inner_env.width:
            continue
        if next_y < 0 or next_y >= inner_env.height:
            continue
        if not _cell_is_traversable(inner_env, next_x, next_y):
            continue
        valid.append((next_x, next_y))
    return valid


def _neighbors_for_doorkey(inner_env, position: tuple[int, int]) -> list[tuple[int, int]]:
    """Return traversable neighbors for DoorKey planning."""

    x, y = position
    candidates = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
    valid: list[tuple[int, int]] = []
    for next_x, next_y in candidates:
        if next_x < 0 or next_x >= inner_env.width:
            continue
        if next_y < 0 or next_y >= inner_env.height:
            continue
        if not _cell_is_traversable_for_doorkey(inner_env, next_x, next_y):
            continue
        valid.append((next_x, next_y))
    return valid


def _oracle_path(inner_env) -> list[tuple[int, int]]:
    """Plan an optimal path from the true state to the success target."""

    start = tuple(int(value) for value in inner_env.agent_pos)
    target = _find_goal_position(inner_env)
    if start == target:
        return [start]

    frontier: list[tuple[int, int]] = [start]
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    for current in frontier:
        for neighbor in _neighbors(inner_env, current):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            if neighbor == target:
                path = [target]
                node = target
                while parents[node] is not None:
                    node = parents[node]
                    path.append(node)
                path.reverse()
                return path
            frontier.append(neighbor)

    raise ValueError(f"No oracle path found from {start!r} to {target!r}.")


def _path_to_any_target(
    inner_env,
    targets: set[tuple[int, int]],
    *,
    use_doorkey_neighbors: bool = False,
) -> list[tuple[int, int]]:
    """Plan a shortest path from the agent to any target position.

    Args:
        inner_env: Unwrapped MiniGrid environment.
        targets: Accepting world positions.
        use_doorkey_neighbors: Whether to use DoorKey traversability semantics.

    Returns:
        Planned path including the current position.

    Raises:
        ValueError: If no target is reachable.
    """

    start = tuple(int(value) for value in inner_env.agent_pos)
    if start in targets:
        return [start]

    frontier: list[tuple[int, int]] = [start]
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    neighbor_fn = _neighbors_for_doorkey if use_doorkey_neighbors else _neighbors

    for current in frontier:
        for neighbor in neighbor_fn(inner_env, current):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            if neighbor in targets:
                path = [neighbor]
                node = neighbor
                while parents[node] is not None:
                    node = parents[node]
                    path.append(node)
                path.reverse()
                return path
            frontier.append(neighbor)

    raise ValueError(f"No oracle path found from {start!r} to any of {targets!r}.")


def _desired_direction(current: tuple[int, int], next_position: tuple[int, int]) -> int:
    """Convert a world-grid step into a MiniGrid direction id."""

    dx = next_position[0] - current[0]
    dy = next_position[1] - current[1]
    if dx == 1 and dy == 0:
        return 0
    if dx == 0 and dy == 1:
        return 1
    if dx == -1 and dy == 0:
        return 2
    if dx == 0 and dy == -1:
        return 3
    raise ValueError(f"Expected adjacent oracle path cells, got {current!r} -> {next_position!r}.")


def _oracle_action(inner_env) -> int:
    """Return the optimal one-step action from the true environment state."""

    path = _oracle_path(inner_env)
    if len(path) < 2:
        return 2

    current = path[0]
    next_position = path[1]
    desired = _desired_direction(current, next_position)
    current_direction = int(inner_env.agent_dir)
    delta = (desired - current_direction) % 4

    if delta == 0:
        return 2
    if delta == 1:
        return 1
    if delta == 3:
        return 0
    return 1


def _shortest_distance(inner_env, target: tuple[int, int]) -> int:
    """Return the current BFS distance to a target over traversable cells."""

    start = tuple(int(value) for value in inner_env.agent_pos)
    if start == target:
        return 0

    frontier: list[tuple[int, int]] = [start]
    seen = {start}
    distance = 0
    while frontier:
        next_frontier: list[tuple[int, int]] = []
        for current in frontier:
            if current == target:
                return distance
            for neighbor in _neighbors(inner_env, current):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                next_frontier.append(neighbor)
        frontier = next_frontier
        distance += 1
    return 10**6


def _dynamic_teacher_action(inner_env) -> int:
    """Return a one-step lookahead teacher action for Dynamic Obstacles."""

    goal_position = _find_goal_position(inner_env)
    scored_actions: list[tuple[tuple[int, float, int, int], int]] = []

    for action in (0, 1, 2):
        cloned_env = copy.deepcopy(inner_env)
        _, reward, terminated, truncated, _ = cloned_env.step(action)
        reward_value = float(reward)
        if terminated and reward_value > 0.0:
            score = (3, reward_value, 0, -action)
        elif terminated or truncated:
            score = (0, reward_value, -10**6, -action)
        else:
            distance = _shortest_distance(cloned_env, goal_position)
            # Prefer safe actions that keep decreasing the oracle distance while
            # using action id as a stable final tie-breaker.
            score = (2, reward_value, -distance, -action)
        scored_actions.append((score, action))

    scored_actions.sort(reverse=True)
    return scored_actions[0][1]


def _turn_action(current_direction: int, desired_direction: int) -> int:
    """Return the one-step turn action that reduces heading error."""

    delta = (desired_direction - current_direction) % 4
    if delta == 0:
        return 2
    if delta == 1:
        return 1
    if delta == 3:
        return 0
    return 1


def _front_position(inner_env) -> tuple[int, int]:
    """Return the world coordinate directly in front of the agent."""

    direction_vector = DIR_TO_VEC[int(inner_env.agent_dir)]
    agent_x, agent_y = (int(value) for value in inner_env.agent_pos)
    return agent_x + int(direction_vector[0]), agent_y + int(direction_vector[1])


def _interaction_targets(
    inner_env,
    target_position: tuple[int, int],
) -> list[tuple[tuple[int, int], int]]:
    """Return reachable stand positions and headings for interacting with a target.

    Args:
        inner_env: Unwrapped MiniGrid environment.
        target_position: World position of the key or door.

    Returns:
        Candidate `(stand_position, desired_direction)` pairs.
    """

    target_x, target_y = target_position
    candidates = [
        ((target_x - 1, target_y), 0),
        ((target_x + 1, target_y), 2),
        ((target_x, target_y - 1), 1),
        ((target_x, target_y + 1), 3),
    ]
    valid: list[tuple[tuple[int, int], int]] = []
    for stand_position, desired_direction in candidates:
        stand_x, stand_y = stand_position
        if stand_x < 0 or stand_x >= inner_env.width:
            continue
        if stand_y < 0 or stand_y >= inner_env.height:
            continue
        if not _cell_is_traversable_for_doorkey(inner_env, stand_x, stand_y, allow_goal=False):
            continue
        valid.append((stand_position, desired_direction))
    return valid


def _doorkey_teacher_action(inner_env) -> int:
    """Return the next oracle action for DoorKey.

    The policy follows three phases derived from the true simulator state:
    approach and pick up the key, approach and toggle the locked door, then
    navigate through the open doorway to the goal.
    """

    door_position = _find_positions_by_type(inner_env, "Door")[0]
    goal_position = _find_goal_position(inner_env)
    front_position = _front_position(inner_env)
    door = inner_env.grid.get(*door_position)

    if inner_env.carrying is None:
        key_positions = _find_positions_by_type(inner_env, "Key")
        if not key_positions:
            raise ValueError("DoorKey teacher expected an uncollected key, but none was found.")
        key_position = key_positions[0]
        if front_position == key_position:
            return 3
        key_targets = _interaction_targets(inner_env, key_position)
        key_target_positions = {position for position, _ in key_targets}
        path = _path_to_any_target(inner_env, key_target_positions, use_doorkey_neighbors=True)
        current_position = tuple(int(value) for value in inner_env.agent_pos)
        if len(path) > 1:
            return _turn_or_forward_action(
                current_position=current_position,
                current_direction=int(inner_env.agent_dir),
                next_position=path[1],
            )
        desired_direction = next(
            direction for position, direction in key_targets if position == current_position
        )
        return _turn_action(int(inner_env.agent_dir), desired_direction)

    if not door.is_open:
        if front_position == door_position:
            return 5
        door_targets = _interaction_targets(inner_env, door_position)
        door_target_positions = {position for position, _ in door_targets}
        path = _path_to_any_target(inner_env, door_target_positions, use_doorkey_neighbors=True)
        current_position = tuple(int(value) for value in inner_env.agent_pos)
        if len(path) > 1:
            return _turn_or_forward_action(
                current_position=current_position,
                current_direction=int(inner_env.agent_dir),
                next_position=path[1],
            )
        desired_direction = next(
            direction for position, direction in door_targets if position == current_position
        )
        return _turn_action(int(inner_env.agent_dir), desired_direction)

    path = _path_to_any_target(inner_env, {goal_position}, use_doorkey_neighbors=True)
    if len(path) < 2:
        return 2
    return _turn_or_forward_action(
        current_position=path[0],
        current_direction=int(inner_env.agent_dir),
        next_position=path[1],
    )


def _turn_or_forward_action(
    current_position: tuple[int, int],
    current_direction: int,
    next_position: tuple[int, int],
) -> int:
    """Convert a desired grid step into a turning/forward action.

    Args:
        current_position: Current world coordinate.
        current_direction: Current MiniGrid direction id.
        next_position: Next desired world coordinate.

    Returns:
        One-step action that reduces the heading error or moves forward.
    """

    desired = _desired_direction(current_position, next_position)
    delta = (desired - current_direction) % 4
    if delta == 0:
        return 2
    if delta == 1:
        return 1
    if delta == 3:
        return 0
    return 1


def collect_teacher_dataset(config: TrainConfig, output_path: Path) -> TeacherDataset:
    """Collect supervised demonstrations from oracle teachers.

    Args:
        config: Training configuration defining the active task set and dataset size.
        output_path: Output `.npz` path for serialized demonstrations.

    Returns:
        The in-memory teacher dataset that was also written to `output_path`.
    """

    images: list[np.ndarray] = []
    directions: list[int] = []
    task_ids: list[int] = []
    actions: list[int] = []
    episode_starts: list[int] = []
    env_action_to_policy_action = {
        int(env_action): action_index for action_index, env_action in enumerate(config.action_ids)
    }

    for env_name in config.env_names:
        env = make_env(env_name)
        task_id = int(ENV_NAME_TO_TASK_ID[env_name])
        memory_teacher = MemoryTeacher()
        rng = np.random.default_rng(config.seed + task_id)

        for episode in range(config.teacher_episodes_per_env):
            episode_seed = int(rng.integers(0, 2_147_483_647))
            obs, _ = env.reset(seed=episode_seed)
            done = False
            first_step = True
            memory_teacher.reset()

            while not done:
                if env_name == "MiniGrid-MemoryS13Random-v0":
                    env_action = memory_teacher.act(env.unwrapped)
                elif env_name == "MiniGrid-Dynamic-Obstacles-6x6-v0":
                    env_action = _dynamic_teacher_action(env.unwrapped)
                elif env_name == "MiniGrid-DoorKey-8x8-v0":
                    env_action = _doorkey_teacher_action(env.unwrapped)
                else:
                    env_action = _oracle_action(env.unwrapped)
                if env_action not in env_action_to_policy_action:
                    raise ValueError(
                        f"Teacher produced env action {env_action} not present in policy action ids "
                        f"{config.action_ids!r}."
                    )
                action = env_action_to_policy_action[env_action]
                images.append(np.asarray(obs["image"], dtype=np.int64))
                directions.append(int(obs["direction"]))
                task_ids.append(task_id)
                actions.append(int(action))
                episode_starts.append(1 if first_step else 0)
                first_step = False

                obs, reward, terminated, truncated, _ = env.step(env_action)
                done = bool(terminated or truncated)

        env.close()

    dataset = TeacherDataset(
        image=np.stack(images, axis=0),
        direction=np.asarray(directions, dtype=np.int64),
        task_id=np.asarray(task_ids, dtype=np.int64),
        action=np.asarray(actions, dtype=np.int64),
        episode_starts=np.asarray(episode_starts, dtype=np.int64),
    )
    np.savez_compressed(
        output_path,
        image=dataset.image,
        direction=dataset.direction,
        task_id=dataset.task_id,
        action=dataset.action,
        episode_starts=dataset.episode_starts,
    )
    return dataset


def load_teacher_dataset(path: Path) -> TeacherDataset:
    """Load a serialized teacher dataset."""

    payload = np.load(path)
    return TeacherDataset(
        image=payload["image"],
        direction=payload["direction"],
        task_id=payload["task_id"],
        action=payload["action"],
        episode_starts=payload["episode_starts"],
    )
