"""Immutable round information state and query helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Generator, List, Optional, Set, Tuple

from .enums import ActionType
from .legal_actions import iter_player_legal_actions, iter_split_response_actions
from .models import Action, Card


@dataclass(frozen=True)
class RoundInfoState:
    """Immutable key representing a unique information state."""

    player_id: int
    round_number: int
    events: Tuple[ObservationEvent, ...]

    def __str__(self) -> str:
        parts = [f"p{self.player_id}r{self.round_number}"]
        for event in self.events:
            parts.append(str(event))
        return "|".join(parts)

    def __repr__(self) -> str:
        return f"RoundInfoState(player={self.player_id}, round={self.round_number}, events={len(self.events)})"

    def get_available_action_types(self) -> FrozenSet[ActionType]:
        """Return variant action types from the round's first observation event.

        Why this works:
        - The engine emits `RoundStartEvent` exactly once per player at round setup.
        - That event carries the variant's `available_actions`.
        - The event is recorded before any player decision is requested.

        Invariant:
        - `events[0]` must be `RoundStartEvent` for every decision-time info state.

        This method is intentionally fail-fast so invariant violations are caught
        immediately instead of silently falling back to guessed actions.
        """
        if not self.events:
            raise ValueError("RoundInfoState has no events; expected RoundStartEvent.")

        first_event = self.events[0]
        if not isinstance(first_event, RoundStartEvent):
            raise ValueError(
                "RoundInfoState invariant violated: first event must be RoundStartEvent, "
                f"got {type(first_event).__name__}."
            )

        return first_event.available_actions

    def get_current_hand_and_actions(self) -> Tuple[List[Card], Set[ActionType]]:
        hand_suits: List[Any] = []
        used_actions: Set[ActionType] = set()

        for event in self.events:
            if isinstance(event, RoundStartEvent):
                hand_suits = list(event.initial_hand)
            elif isinstance(event, DrawCardEvent):
                hand_suits.append(event.card)
            elif isinstance(event, OwnReserveEvent):
                hand_suits.remove(event.card)
                used_actions.add(ActionType.RESERVE)
            elif isinstance(event, OwnDiscardEvent):
                hand_suits.remove(event.revealed_card)
                hand_suits.remove(event.hidden_card)
                used_actions.add(ActionType.DISCARD)
            elif isinstance(event, OwnSplitEvent):
                hand_suits.remove(event.cards[0])
                hand_suits.remove(event.cards[1])
                used_actions.add(ActionType.SPLIT)

        return [Card(suit) for suit in hand_suits], used_actions

    def get_collected_cards(self) -> Tuple[List[Card], List[Card]]:
        my_collected: List[Card] = []
        opponent_collected: List[Card] = []

        for event in self.events:
            if isinstance(event, SplitChoiceEvent):
                if event.chooser == self.player_id:
                    my_collected.append(Card(event.chooser_gets))
                    opponent_collected.append(Card(event.offerer_gets))
                else:
                    my_collected.append(Card(event.offerer_gets))
                    opponent_collected.append(Card(event.chooser_gets))

        return my_collected, opponent_collected

    def get_favors(self) -> dict:
        favors: dict = {}
        for event in self.events:
            if isinstance(event, RoundStartEvent):
                favors = dict(event.favors)
            elif isinstance(event, RoundEndEvent):
                favors = dict(event.final_favors)
        return favors

    def get_my_reserved(self) -> List[Card]:
        reserved: List[Card] = []
        for event in self.events:
            if isinstance(event, OwnReserveEvent):
                reserved.append(Card(event.card))
        return reserved

    def get_opponent_reserved_count(self) -> int:
        count = 0
        for event in self.events:
            if isinstance(event, OpponentReserveEvent):
                count += 1
        return count

    def get_discarded_cards(self) -> Tuple[List[Card], List[Card], List[Card]]:
        my_revealed: List[Card] = []
        my_hidden: List[Card] = []
        opponent_revealed: List[Card] = []

        for event in self.events:
            if isinstance(event, OwnDiscardEvent):
                my_revealed.append(Card(event.revealed_card))
                my_hidden.append(Card(event.hidden_card))
            elif isinstance(event, OpponentDiscardEvent):
                opponent_revealed.append(Card(event.revealed_card))

        return my_revealed, my_hidden, opponent_revealed

    def get_opponent_used_actions(self) -> Set[ActionType]:
        used: Set[ActionType] = set()
        for event in self.events:
            if isinstance(event, OpponentReserveEvent):
                used.add(ActionType.RESERVE)
            elif isinstance(event, OpponentDiscardEvent):
                used.add(ActionType.DISCARD)
            elif isinstance(event, OpponentSplitEvent):
                used.add(ActionType.SPLIT)
        return used

    def get_current_turn(self) -> int:
        turn = 0
        for event in self.events:
            turn = event.turn
        return turn

    def get_pending_split(self) -> Optional[List[Card]]:
        pending: Optional[List[Card]] = None
        for event in self.events:
            if isinstance(event, OpponentSplitEvent):
                pending = [Card(event.cards[0]), Card(event.cards[1])]
            elif isinstance(event, SplitChoiceEvent):
                pending = None
        return pending

    def get_legal_actions(self) -> Generator[Action, None, None]:
        """Yield deduplicated legal actions at this information state.

        Note:
        - This is only for player decision nodes (no chance actions).
        - `get_available_action_types()` remains a variant-level capability set,
          not the current legal-action set.
        """
        if self.get_pending_split() is not None:
            yield from iter_split_response_actions()
            return

        hand, used_actions = self.get_current_hand_and_actions()
        yield from iter_player_legal_actions(
            hand=hand,
            used_actions=used_actions,
            available_actions=self.get_available_action_types(),
        )


from .events import (  # noqa: E402
    DrawCardEvent,
    ObservationEvent,
    OpponentDiscardEvent,
    OpponentReserveEvent,
    OpponentSplitEvent,
    OwnDiscardEvent,
    OwnReserveEvent,
    OwnSplitEvent,
    RoundEndEvent,
    RoundStartEvent,
    SplitChoiceEvent,
)
