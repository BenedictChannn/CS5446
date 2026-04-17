# Adding Your Agent to the GUI

This guide shows you how to make your custom agent appear in the GUI dropdown so you can
play against it interactively.

---

## Quick overview

Two steps:

1. **Import** your agent class at the top of `gui/server.py`.
2. **Add a registry entry** to `_AGENT_REGISTRY` in the same file.

The server reads `_AGENT_REGISTRY` to build the dropdown. Each entry tells the GUI which
class to instantiate for each game variant (tiny_hana / medium_hana), or `None` if your
agent does not support that variant.

---

## Prerequisites

Your agent must:

- Be a subclass of `Agent` (from `engine.agents`)
- Implement `get_action_distribution(info_state: RoundInfoState)`

Your implementation lives in `student_attempts/` — for example, you might load a saved
policy-table agent or wrap your CFR solver output in an agent class there.

---

## Step 1 — Import your agent in `gui/server.py`

Open `gui/server.py` and find the import block (roughly lines 1–48). Add your import
**after** the existing imports:

```python
# --- add after the existing import block ---
from student_attempts.your_module import YourAgent
```

The server already adds the repo root to `sys.path`, so any module under `student_attempts/`
is importable using dot notation.

---

## Step 2 — Add a registry entry to `_AGENT_REGISTRY`

Find `_AGENT_REGISTRY` (around line 68) and add a new dict to the list:

```python
_AGENT_REGISTRY = [
    # ... existing entries ...

    # --- add your entry here ---
    {"id": "your_agent", "label": "Your Agent", "tiny": YourAgent, "medium": YourAgent},
]
```

### Field reference

| Field | Type | Purpose |
|-------|------|---------|
| `"id"` | unique lowercase string | Used internally to look up the agent; never shown to users |
| `"label"` | string | Display name shown in the GUI dropdown |
| `"tiny"` | class or `None` | Class to instantiate for tiny_hana; `None` hides it for that variant |
| `"medium"` | class or `None` | Class to instantiate for medium_hana; `None` hides it for that variant |

### Three common patterns

**Works with both variants (same class on both sides)**

```python
{"id": "your_agent", "label": "Your Agent", "tiny": YourAgent, "medium": YourAgent},
```

**Works with tiny_hana only**

```python
{"id": "your_agent", "label": "Your Agent", "tiny": YourAgent, "medium": None},
```

**Works with medium_hana only**

```python
{"id": "your_agent", "label": "Your Agent", "tiny": None, "medium": YourAgent},
```

> **Note:** If you want your agent to work across both variants, make sure its logic only
> relies on `RoundInfoState` helpers (like `get_available_action_types()`) rather than
> hardcoding variant-specific actions. The same class can often run on both variants unchanged.

---

## Step 3 — Verify

Run the server from the **repo root**:

```bash
python gui/server.py
```

Then open `http://localhost:8080` in your browser:

1. Select a game variant (tiny or medium).
2. Open the **opponent** dropdown.
3. Confirm your agent's label appears.
4. Start a game and play a round.

---

## Troubleshooting

### Import error on startup

```
ModuleNotFoundError: No module named 'student_attempts.your_module'
```

- Make sure you are running from the **repo root**: `python gui/server.py`, not
  `python server.py` from inside `gui/`.
- Double-check that the filename and class name match exactly (Python is case-sensitive).

### Agent missing from the dropdown

- Check that the `"id"` value is **unique** — duplicate ids cause the second entry to be
  silently shadowed.
- Check that the correct variant's class is **not `None`**. If you selected
  *medium_hana* in the GUI but your entry has `"medium": None`, the agent will not appear.
