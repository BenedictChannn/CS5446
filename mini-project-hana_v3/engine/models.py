"""Core immutable/mutable value objects for the engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence, Set, Tuple

from .enums import ActionType


@dataclass(frozen=True)
class GameConfig:
    """Configuration that defines a Hanamikoji variant."""

    name: str
    suits: type
    available_actions: FrozenSet[ActionType]
    initial_hand_size: int
    win_favor_points: Optional[int]
    win_geisha_count: int

    @property
    def turns_per_round(self) -> int:
        """Total turns per round = 2 * number of available actions."""

        return 2 * len(self.available_actions)


@dataclass(frozen=True)
class Card:
    """Represents a single card in the game."""

    suit: object

    def __repr__(self):
        return f"Card({self.suit.name})"


@dataclass(frozen=True, init=False, repr=False)
class Action:
    """Represents a player's action."""

    action_type: ActionType
    cards: Tuple[Card, ...]
    choice: Optional[int]

    def __init__(self, action_type: ActionType, cards: Sequence[Card], choice: Optional[int] = None):
        object.__setattr__(self, "action_type", action_type)
        # Defensive copy prevents caller-side list mutation from affecting action payload.
        cards_tuple = tuple(cards)
        # Canonicalize card ordering for SPLIT and DISCARD so that info states
        # are identical regardless of the order the caller provided.
        # CHANCE_DEAL encodes the deal permutation and must not be reordered.
        if action_type in (ActionType.SPLIT, ActionType.DISCARD) and len(cards_tuple) == 2:
            c0, c1 = cards_tuple
            if c0.suit.value > c1.suit.value:
                cards_tuple = (c1, c0)
                # choice is a direct index into cards (revealed card for DISCARD),
                # so it must be flipped when the cards are swapped.
                if choice is not None:
                    choice = 1 - choice
        object.__setattr__(self, "cards", cards_tuple)
        object.__setattr__(self, "choice", choice)

    def validate(
        self,
        hand: List[Card],
        used_actions: Set[ActionType],
        pending_split: bool = False,
        available_actions: Optional[FrozenSet[ActionType]] = None,
    ) -> bool:
        """Validate if this action is legal given the current hand and used actions."""

        # Chance actions are validated separately inside execute_action
        if self.action_type in (ActionType.CHANCE_START, ActionType.CHANCE_DEAL, ActionType.CHANCE_DRAW):
            return True

        # SPLIT_RESPONSE is only valid when there's a pending split
        if self.action_type == ActionType.SPLIT_RESPONSE:
            if not pending_split:
                return False
            if self.choice not in [0, 1]:
                return False
            return True

        # Other actions are invalid when there's a pending split
        if pending_split:
            return False

        # Check action is available in this variant
        if available_actions is not None and self.action_type not in available_actions:
            return False

        if self.action_type in used_actions:
            return False

        hand_copy = hand.copy()
        for card in self.cards:
            if card not in hand_copy:
                return False
            hand_copy.remove(card)

        if self.action_type == ActionType.RESERVE and len(self.cards) != 1:
            return False
        elif self.action_type == ActionType.DISCARD:
            if len(self.cards) != 2:
                return False
            if self.choice not in [0, 1]:
                return False
        elif self.action_type == ActionType.SPLIT and len(self.cards) != 2:
            return False

        return True

    def __repr__(self):
        return f"Action({self.action_type.name}, {self.cards}, choice={self.choice})"
