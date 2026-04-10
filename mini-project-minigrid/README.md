# CS5446 Mini Project: MiniGrid

This repository is the starter scaffold for the CS5446 mini project on AI decision making and reinforcement learning.

The goal is to build an `Agent` that can act in a family of MiniGrid environments and solve them reliably under strict runtime and submission constraints.

The scaffold currently gives:
- the assignment brief and submission contract in [mini-project.ipynb](/home/benedict/Documents/CS5446/mini-project-minigrid/mini-project.ipynb)
- helper utilities in [utils.py](/home/benedict/Documents/CS5446/mini-project-minigrid/utils.py)
- the allowed runtime dependencies in [environment.yml](/home/benedict/Documents/CS5446/mini-project-minigrid/environment.yml)

It does not yet contain a real solution agent, training code, experiment scripts, datasets, or checkpoints.

## 1. What this project is about

MiniGrid is a lightweight grid-world benchmark used in reinforcement learning research. The agent does not observe the full map. Instead, it sees a small egocentric `7 x 7` view in front of it and must act based on that partial observation.

That means this project is not just about shortest path planning. Depending on the environment, the agent may need:
- exploration
- memory
- object interaction
- obstacle avoidance
- safe navigation
- efficient action selection under time pressure

The assignment explicitly allows different solution styles:
- rule-based / heuristic agents
- search / planning approaches
- learned policies such as PPO, DQN, imitation learning, or hybrids

The main challenge is building something that fits the grader constraints and still scores well.

## 2. What the grader expects

Coursemology imports and runs an `Agent` class. The contract is:

```python
agent = Agent(obs_space=env.observation_space, action_space=env.action_space)
```

Then, for each episode:
- `agent.reset()` is called once at the start
- `agent.act(obs)` is called at every step

Your submission must preserve:
- class name: `Agent`
- method: `reset(self)`
- method: `act(self, obs) -> int`

Inside `Agent`, you can load a model, keep episode state, use heuristics, or dispatch to environment-specific logic.

## 3. Hard constraints that matter

These constraints should drive the design from day one:

- Time limit per task: `30 seconds`
- Time limit per episode: `3 seconds`
- Episodes per task: `10`
- Memory limit: `1.5 GB`
- CPU only
- Assume single core
- Upload size limit: `2 MB`
- Only libraries in [environment.yml](/home/benedict/Documents/CS5446/mini-project-minigrid/environment.yml) are allowed

Implications:
- very large models are risky
- expensive search at test time is risky
- multiprocessing is not a valid assumption
- the final pasted Coursemology submission must stay compact and self-contained

## 4. The six environments

There are 6 tasks total:
- 1 warm-up task
- 5 main tasks

The environment IDs are defined in the notebook.

### 4.1 Empty

Environment ID: `MiniGrid-Empty-8x8-v0`

This is the warm-up environment. It is an open room with a goal square and no meaningful obstacles.

What it tests:
- basic navigation
- action semantics
- whether the agent can move toward a visible goal efficiently

Why it matters:
- it is the easiest task
- it is a good debugging baseline
- if this is not solved, the rest of the project is not ready

### 4.2 DoorKey

Environment ID: `MiniGrid-DoorKey-8x8-v0`

The agent must:
1. find the key
2. pick it up
3. unlock/open the door
4. move through the door
5. reach the goal

What it tests:
- sequential decision making
- interaction with objects
- understanding of key and door mechanics
- exploration under partial observability

Why it is harder:
- success requires the correct order of subgoals
- naive reactive behavior is often not enough

### 4.3 FourRooms

Environment ID: `MiniGrid-FourRooms-v0`

The map consists of four rooms connected by narrow openings. The agent and goal are randomly placed.

What it tests:
- exploration
- room-to-room navigation
- planning through bottlenecks
- robustness to randomized starts and goals

Why it is harder:
- the goal may not be visible early
- local greedy behavior can get stuck wandering

### 4.4 Dynamic Obstacles

Environment ID: `MiniGrid-Dynamic-Obstacles-6x6-v0`

This is a compact environment with moving obstacles. A collision ends the episode.

What it tests:
- dynamic obstacle avoidance
- short-horizon reactivity
- safe forward movement

Important detail:
- upstream Minigrid narrows this environment's action space to 3 actions: left, right, forward

Why it is harder:
- a good static path is not enough
- the agent must react to obstacle motion online

### 4.5 Lava Gap

Environment ID: `MiniGrid-LavaGapS7-v0`

The agent must cross to the goal through a single gap in a lava barrier. Touching lava ends the episode.

What it tests:
- safe navigation
- finding a narrow passage
- movement precision under partial observability

Why it is harder:
- local mistakes are terminal
- the gap position changes across episodes

### 4.6 Memory

Environment ID: `MiniGrid-MemoryS13Random-v0`

The agent first sees a cue object, then must remember it and choose the matching branch later.

What it tests:
- memory over time
- handling partial observability
- decision making based on earlier information, not just current view

Why it is harder:
- the correct action later depends on information seen earlier
- a memoryless policy can struggle badly

Important grading note:
- this environment uses a `50%` minimum success-rate baseline because random guessing at the final split gives non-zero success

## 5. Observation and action format

Each observation is a dictionary:

- `obs["image"]`: NumPy array of shape `(7, 7, 3)`
- `obs["direction"]`: integer in `{0, 1, 2, 3}`
- `obs["mission"]`: task description string

The `image` encoding stores each visible cell as:

```python
(object_idx, color_idx, state)
```

Some important object indices:
- `1` = empty
- `2` = wall
- `4` = door
- `5` = key
- `8` = goal
- `9` = lava
- `10` = agent

Important door states:
- `0` = open
- `1` = closed
- `2` = locked

Default action meanings in MiniGrid:
- `0` = turn left
- `1` = turn right
- `2` = move forward
- `3` = pick up
- `4` = drop
- `5` = toggle
- `6` = done

For most tasks, the full 7-action interface exists even if some actions are unused.

## 6. How performance is graded

The project has two parts:
- performance: `12 marks`
- report: `8 marks`

Each of the 6 tasks is worth up to `3 marks`, but the total performance score is capped at `12`.

For grading:
- you do not need to master every single environment to get full performance marks
- warm-up plus 3 strong main tasks is enough for full performance marks if success rates are high enough

It may be better to:
- guarantee the warm-up
- pick 3 or 4 main environments to solve well
- avoid over-investing in the hardest remaining case if the mark cap is already reachable

## 7. What needs to be submitted

The assignment requires more than just the final `Agent`.

You must submit:
- the `Agent` code for Coursemology
- any helper code needed by the agent
- if using PyTorch, a self-contained `get_model()` or equivalent model-loading logic
- all supplementary files used to build the solution

Supplementary files include things like:
- training scripts
- data generation scripts
- datasets
- checkpoints
- experiment code

This is explicitly required for originality checking.

If supplementary files are missing, the assignment note says the project can receive zero marks.

## 8. What this scaffold already supports

### Local visualization

The notebook includes cells to:
- render each environment
- run a single episode with visualization
- run a 10-episode evaluation loop similar to the grader

### Agent stub

The notebook defines a minimal placeholder:
- `Agent.__init__`
- `Agent.reset()`
- `Agent.act(obs)`

Right now it samples random actions. It is only a template.

### Model embedding helper

[utils.py](/home/benedict/Documents/CS5446/mini-project-minigrid/utils.py) includes `generate_torch_loader_snippet(...)`, which can inline model weights into source code as a base64 blob. This is useful for making a Coursemology submission self-contained.

This is especially relevant because:
- the upload must be self-contained
- external files may not be available in the grader
- upload size is constrained

## 9. Suggested workflow

1. Understand the six environments before choosing an approach.
2. Set up local evaluation first.
3. Solve the warm-up environment as a sanity check.
4. Decide whether to use:
   - a single multi-task agent
   - environment-specific logic
   - a hybrid of rules and learned components
5. Design with Coursemology limits in mind from the start.
6. Keep training scripts, checkpoints, and experiment artifacts organized for the final submission and report.

## 10. Approach options

There is no required method. Common options include:

### Heuristic / rule-based

Useful when:
- environment structure is regular
- task-specific logic is easy to encode
- low inference cost matters

Often suitable for:
- Empty
- Lava Gap
- DoorKey
- parts of Dynamic Obstacles

### Learned policy

Useful when:
- exploration is hard to hand-code
- partial observability matters
- the behavior needs to generalize over randomized layouts

Often suitable for:
- FourRooms
- Dynamic Obstacles
- Memory

### Hybrid

A hybrid approach can combine:
- rules for easy and deterministic parts
- learned policy for harder perception or memory-heavy parts

This can be more practical than forcing one universal policy across all tasks.

## 11. Known implementation details worth remembering

- The grader creates the agent once per task, not once per episode.
- `reset()` is therefore the correct place to clear episode-specific state.
- `mission` alone is not always enough to identify the environment, because different tasks can share similar language.
- Dynamic Obstacles differs slightly from the notebook's simplified action-space summary.
- Success in the local evaluation cells is determined by whether termination happened with positive reward.

## 12. Suggested repository structure

As the project grows, the repository will likely need:

- `README.md`
- `src/` or `agents/` for the actual implementation
- `train_*.py` scripts
- evaluation scripts
- experiment logs or summaries
- saved checkpoints if needed
- report materials such as plots and tables

This makes the report and supplementary submission easier to prepare.

## 13. Files in this repository

- [mini-project.ipynb](/home/benedict/Documents/CS5446/mini-project-minigrid/mini-project.ipynb): assignment brief, environment list, agent interface, local evaluation cells, Coursemology submission notes
- [utils.py](/home/benedict/Documents/CS5446/mini-project-minigrid/utils.py): visualization helpers and PyTorch model embedding utilities
- [environment.yml](/home/benedict/Documents/CS5446/mini-project-minigrid/environment.yml): allowed dependencies for the project runtime

## 14. Summary

This mini project combines:
- partial observability
- multiple environment types
- strict runtime limits
- strict packaging constraints
- a report that must justify the design choices

The goal is to build a reliable submission pipeline:
- understand the tasks
- choose a scoring strategy
- implement and evaluate systematically
- keep all artifacts needed for final submission

This is the core of the project.
