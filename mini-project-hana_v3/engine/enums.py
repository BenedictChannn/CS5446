"""Core enums used by the Hanamikoji engine."""

from enum import Enum


class ActionType(Enum):
    """All action types across all variants."""

    RESERVE = "reserve"
    DISCARD = "discard"
    SPLIT = "split"
    SPLIT_RESPONSE = "split_response"
    CHANCE_START = "chance_start"
    CHANCE_DEAL = "chance_deal"
    CHANCE_DRAW = "chance_draw"


class ObservationType(Enum):
    """All observable event types across all variants."""

    ROUND_START = "round_start"
    DRAW_CARD = "draw_card"
    OWN_RESERVE = "own_reserve"
    OWN_DISCARD = "own_discard"
    OWN_SPLIT = "own_split"
    OPPONENT_RESERVE = "opponent_reserve"
    OPPONENT_DISCARD = "opponent_discard"
    OPPONENT_SPLIT = "opponent_split"
    SPLIT_CHOICE = "split_choice"
    ROUND_END = "round_end"
