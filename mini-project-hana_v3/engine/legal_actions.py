"""Shared legal-action enumeration helpers for player decision nodes.

Ordering guarantee:
- Actions are yielded in a canonical, deterministic order.
- For player actions, ordering is based on suit-value ordering, not the
  incidental in-hand card order.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Generator, List, Sequence, Set

from .enums import ActionType
from .models import Action, Card


def iter_split_response_actions() -> Generator[Action, None, None]:
    """Yield the two legal SPLIT_RESPONSE actions."""
    yield Action(ActionType.SPLIT_RESPONSE, [], choice=0)
    yield Action(ActionType.SPLIT_RESPONSE, [], choice=1)


def iter_player_legal_actions(
    hand: Sequence[Card],
    used_actions: Set[ActionType],
    available_actions: FrozenSet[ActionType],
) -> Generator[Action, None, None]:
    """Yield deduplicated legal non-chance actions for a player node."""
    # Group cards by suit; preserve per-suit index order for deterministic
    # representative selection when multiple identical-suit cards exist.
    suit_to_indices: Dict[object, List[int]] = {}
    for idx, card in enumerate(hand):
        suit_to_indices.setdefault(card.suit, []).append(idx)

    suits_sorted = sorted(suit_to_indices.keys(), key=lambda suit: suit.value)

    # RESERVE: one action per suit, ordered by suit value.
    if ActionType.RESERVE in available_actions and ActionType.RESERVE not in used_actions:
        for suit in suits_sorted:
            idx = suit_to_indices[suit][0]
            yield Action(ActionType.RESERVE, [hand[idx]])

    if len(hand) < 2:
        return

    # DISCARD: ordered by (low_suit, high_suit), then choice (0 before 1).
    # For distinct suits, yield two choices against one canonical card order.
    # For same-suit pairs, only one unique reveal exists (choice=0).
    if ActionType.DISCARD in available_actions and ActionType.DISCARD not in used_actions:
        for i, suit_a in enumerate(suits_sorted):
            for suit_b in suits_sorted[i:]:
                if suit_a == suit_b:
                    idxs = suit_to_indices[suit_a]
                    if len(idxs) < 2:
                        continue
                    cards = [hand[idxs[0]], hand[idxs[1]]]
                    yield Action(ActionType.DISCARD, cards, choice=0)
                    continue

                idx_a = suit_to_indices[suit_a][0]
                idx_b = suit_to_indices[suit_b][0]
                cards = [hand[idx_a], hand[idx_b]]
                yield Action(ActionType.DISCARD, cards, choice=0)
                yield Action(ActionType.DISCARD, cards, choice=1)

    # SPLIT: ordered by (low_suit, high_suit) in canonical card order.
    if ActionType.SPLIT in available_actions and ActionType.SPLIT not in used_actions:
        for i, suit_a in enumerate(suits_sorted):
            for suit_b in suits_sorted[i:]:
                if suit_a == suit_b:
                    idxs = suit_to_indices[suit_a]
                    if len(idxs) < 2:
                        continue
                    yield Action(ActionType.SPLIT, [hand[idxs[0]], hand[idxs[1]]])
                    continue

                idx_a = suit_to_indices[suit_a][0]
                idx_b = suit_to_indices[suit_b][0]
                yield Action(ActionType.SPLIT, [hand[idx_a], hand[idx_b]])
