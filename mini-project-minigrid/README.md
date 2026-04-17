# CS5446 Mini Project: MiniGrid

This repository contains the final retained code and artifacts for a MiniGrid
agent submission.

The shipped agent is a learned policy-only controller:
- a bundled 5-task base expert for `Empty`, `LavaGap`, `FourRooms`, `Memory`,
  and `Dynamic Obstacles`
- a DoorKey-specialized expert for `DoorKey`

The final Coursemology submission is self-contained in
`coursemology_submission.py`.

## Repository layout

### Runtime files

- `agent.py`
  Public assignment entry point.
- `coursemology_submission.py`
  Final self-contained Coursemology submission file.
- `src/minigrid_nav/agent.py`
  Local runtime wrapper used by the project code.
- `src/minigrid_nav/policy_agent.py`
  Deployment controller that loads the bundled experts and runs inference.
- `src/minigrid_nav/policy_bundle.pt.xz`
  Final bundled runtime checkpoint used by the shipped agent.
- `src/minigrid_nav/policy_nav.pt`
  Retained upstream checkpoint used in the final training path.

### Training code

- `src/rl_multitask/`
  Core model, PPO training, teacher rollout generation, task inference,
  observation processing, and evaluation utilities.
- `scripts/collect_teacher_data.py`
  Generates teacher demonstrations.
- `scripts/train_single_checkpoint_ppo.py`
  Trains the retained 5-task base checkpoint.
- `scripts/train_single_checkpoint_doorkey_distill.py`
  Trains the retained DoorKey consolidation checkpoint with distillation.

### Retained artifacts

- `artifacts/teacher_nav.npz`
  Teacher dataset retained at the project root.
- `artifacts/single_checkpoint_ppo/`
  Retained base PPO checkpoint family.
- `artifacts/single_checkpoint_doorkey_distill/`
  Retained final DoorKey distillation checkpoint family.

### Supporting files

- `mini-project.ipynb`
  Local notebook used during development and evaluation.
- `utils.py`
  Visualization helpers and model-embedding utilities.
- `environment.yml`
  Main environment specification.
- `environment.dev.gpu.yml`
  Optional development environment specification.

## Agent interface

The assignment expects:

```python
agent = Agent(obs_space=env.observation_space, action_space=env.action_space)
```

And then:

- `agent.reset()` once per episode
- `agent.act(obs)` each step

The public entry point in this repo is `agent.py`.

## Kept training lineage

This cleaned repository intentionally keeps only a minimal, defensible training
and deployment path:

1. collect teacher data
2. train the retained 5-task PPO checkpoint
3. train the retained DoorKey distillation checkpoint
4. bundle runtime experts for deployment
5. embed the final runtime logic into `coursemology_submission.py`

Older experiments, probe files, debug artifacts, screenshots, and superseded
checkpoint families were removed during cleanup.

## Notes

- The runtime agent is CPU-only.
- The current deployment controller keeps inferred task identity fixed within an
  episode to avoid mid-rollout task flips in Dynamic Obstacles.
- All remaining Python files in the cleaned repo compile successfully.
