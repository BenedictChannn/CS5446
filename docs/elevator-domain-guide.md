# Elevator RDDL Domain & Instance — Brief Explanation

## Overview

The elevator domain models a **stochastic MDP** where one or more elevators deliver passengers from upper floors to the bottom floor. Passengers arrive randomly (Poisson process), and the goal is to maximize reward by minimizing waiting time and maximizing deliveries.

---

## Domain (`domain.rddl`)

### What It Defines

The domain specifies the **rules** of the problem: types, state variables, actions, dynamics, and reward.

| Component | Purpose |
|-----------|---------|
| **Types** | `elevator`, `floor` — objects in the world |
| **Non-fluents** | Fixed parameters (penalties, arrival rates, capacities, floor layout) |
| **State fluents** | What changes: people waiting, people in elevator, elevator position, door state, direction |
| **Actions** | `move-current-dir`, `open-door`, `close-door` — one per elevator per step |
| **CPFs** | How state evolves: arrivals, boarding, movement, door/direction logic |
| **Reward** | Penalties for waiting + reward for deliveries |

### Key Behaviours

- **Boarding:** People board only when the elevator is on that floor, door is open, and elevator is going down.
- **Movement:** Elevator cannot move with door open; must close door before moving.
- **Direction:** Up at bottom; down at top or when door opens on an intermediate floor.
- **Pickups on the way down:** Allowed — stop at floor, open door, board, close door, continue.

---

## Instance (`instance.rddl`)

### What It Defines

The instance **instantiates** the domain with concrete objects and parameters.

| Setting | Value | Meaning |
|---------|-------|---------|
| **Objects** | 1 elevator (`e0`), 5 floors (`f0`–`f4`) | Single elevator, 5-floor building |
| **Initial state** | `elevator-at-floor(e0, f0)` | Elevator starts at bottom |
| **Floor layout** | f0 (bottom) ↔ f1 ↔ f2 ↔ f3 ↔ f4 (top) | Linear chain |
| **Arrival rates** | f1: 0.1; f2–f4: 0.15 | Poisson arrivals per floor |
| **Horizon** | 200 | 200 time steps per episode |
| **Discount** | 1.0 | No discounting |

---

## Diagram: Floor Layout & Action Flow

```mermaid
flowchart TB
    subgraph floors["Floor Layout"]
        f4["f4 (top)"]
        f3["f3"]
        f2["f2"]
        f1["f1"]
        f0["f0 (bottom)"]
        f4 --> f3 --> f2 --> f1 --> f0
    end

    subgraph actions["Actions (per elevator)"]
        move["move-current-dir"]
        open["open-door"]
        close["close-door"]
    end

    subgraph conditions["Boarding conditions"]
        cond["Door open ∧ Going down ∧ On floor"]
    end

    subgraph reward["Reward"]
        r1["− People waiting"]
        r2["− People in elevator"]
        r3["+ Deliveries to bottom"]
    end

    actions --> conditions
    conditions --> reward
```

---

## Diagram: Pickup Sequence (Example)

```mermaid
sequenceDiagram
    participant E as Elevator
    participant F4 as Floor 4
    participant F3 as Floor 3

    Note over E: At floor 4, going up
    E->>E: open-door
    E->>F4: Board passengers
    E->>E: close-door
    E->>E: move-current-dir (down)
    E->>F3: Arrive at floor 3
    E->>E: open-door
    E->>F3: Board passengers
    E->>E: close-door
    E->>E: move-current-dir (down)
```

---

*Last updated: March 2025*
