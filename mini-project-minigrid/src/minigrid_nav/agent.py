"""Agent implementation for the shipped MiniGrid environments."""

from __future__ import annotations

from .policy_agent import NavigationPolicyController


class Agent:
    """MiniGrid agent backed only by learned policy experts.

    The shipped agent uses one bundled checkpoint containing multiple learned
    experts. No planner or symbolic controller is used at inference.
    """

    def __init__(self, obs_space, action_space):
        """Construct the public agent wrapper.

        Args:
            obs_space: Environment observation space, kept for compatibility with
                the assignment interface.
            action_space: Environment action space, kept for compatibility with
                the assignment interface.
        """

        self.obs_space = obs_space
        self.action_space = action_space
        self.controller: NavigationPolicyController | None = None

    def reset(self) -> None:
        """Reset episode-local state."""

        if self.controller is not None:
            self.controller.reset()

    def act(self, obs: dict) -> int:
        """Return an action for the current task observation.

        Args:
            obs: Raw MiniGrid observation dictionary for the current timestep.

        Returns:
            The action chosen by the bundled recurrent policy controller.
        """

        if self.controller is None:
            self.controller = NavigationPolicyController()
            self.controller.reset()

        return self.controller.act(obs)
