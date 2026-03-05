# CS5446 Assignment 1: Deep Reinforcement Learning

> Work walkthrough — CS5446 AI Planning and Decision Making (NUS)

## Assignment Overview

Learn a policy for the **Elevator environment** (evening rush, passengers going down to ground floor) using **Actor–Critic** with **PPO**, without knowing the environment dynamics in advance.

- **Actor:** stochastic policy over discrete actions  
- **Critic:** value function for low-variance learning  
- **PPO:** clipped surrogate objective for stable updates  

Reference: Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

---

## Codebase Structure

```
CS5446_Assignment1/
├── docs/
│   └── elevator-domain-guide.md   # Domain & instance explanation (RDDL)
├── elevator/
│   ├── domain.rddl      # RDDL domain: dynamics, reward, actions
│   └── instance.rddl    # Instance: 1 elevator, 5 floors, horizon 200
├── utils.py             # DictToListWrapper, live_plot, model serialization
├── environment.yml      # Conda env spec
├── Elevator.ipynb       # Environment intro and exploration
├── Deep Reinforcement Learning.ipynb  # Main assignment (PPO)
└── README.md            # This walkthrough
```

---

## Environment (Elevator)

- **5 floors** (f0–f4), f0 = bottom/destination  
- **1 elevator:** move up/down, open/close door  
- **Poisson arrivals** on f1–f4  
- **Actions:** `move-current-dir`, `open-door`, `close-door` (max one per step)  
- **Reward:** penalties for waiting/in-elevator; reward for deliveries  

`Elevator.ipynb` explores the env in detail. See [docs/elevator-domain-guide.md](docs/elevator-domain-guide.md) for a concise explanation of the RDDL domain and instance.

---

## Architecture

**Data flow:**

```
RDDLEnv (domain + instance)
    → DictToListWrapper (Dict → Box/Discrete)
    → RecordEpisodeStatistics
    → SyncVectorEnv (4 envs)
    → ACAgent (actor–critic)
    → PPO training loop
    → model.pth → Coursemology
```

**Components:**

- **`DictToListWrapper`** (`utils.py`): Dict obs/actions → Box/Discrete for the agent.
- **`ACAgent`**: Actor (logits) + critic (value); shared observation input.
- **PPO loop:** Rollout → GAE advantages → mini-batch updates (policy + value + entropy).

**Hyperparameters:** `LEARNING_RATE=2.5e-4`, `ROLLOUT_STEPS=128`, `NUM_EPOCHS=4`, `TOTAL_STEPS=800000`, `GAMMA=0.99`, `GAE_LAMBDA=0.95`, `CLIP_COEF=0.2`, etc.

---

## Task Status

| Task | Description | Status |
|------|-------------|--------|
| **Task 1.1** | Define `actor_input_dim`, `actor_output_dim`, `critic_input_dim`, `critic_output_dim` in `ACAgent` | TODO |
| **Task 1.2** | Implement `get_value()` — value predictions for batch of states | TODO |
| **Task 1.3** | Implement `get_probs()` — categorical distribution over actions | TODO |
| **Task 1.4** | Implement `get_action()` — sample action from distribution | TODO |
| **Task 1.5** | Implement `get_action_logprob()` — log prob of given action | TODO |
| **Task 2** | Implement `get_deltas()` — TD error δ_t for advantage estimation | TODO |
| **Task 3.1.1** | Implement `get_ratio()` — probability ratio ρ_t (new/old policy) | TODO |
| **Task 3.1.2** | Implement `get_policy_objective()` — clipped surrogate objective | TODO |
| **Task 3.2** | Implement `get_value_loss()` — value loss with clipping | TODO |
| **Task 3.3** | Implement `get_total_loss()` — combined PPO loss | TODO |
| **Task 4** | Train agent, generate submission, submit to Coursemology | TODO |

**Grading:** Task 1 (5), Task 2 (1), Task 3 (4), Task 4 (5).

---

*Updated as tasks are completed.*
