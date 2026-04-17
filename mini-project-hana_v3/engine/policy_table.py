"""Policy-table artifact helpers and an artifact-backed agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .agents import Agent
from .enums import ActionType
from .info_state import RoundInfoState
from .models import Action, Card


POLICY_TABLE_FORMAT = "policy_table_v1"
POLICY_TABLE_KEY_ENCODING = "round_info_state_str_v1"

_EPS = 1e-6
_POLICY_ACTION_TYPES = frozenset(
    {
        ActionType.RESERVE,
        ActionType.DISCARD,
        ActionType.SPLIT,
        ActionType.SPLIT_RESPONSE,
    }
)
_ACTION_TYPE_BY_VALUE = {action_type.value: action_type for action_type in ActionType}

_InternalEncodedAction = Tuple[ActionType, Tuple[Any, ...], Optional[int]]
_InternalEncodedDistribution = Tuple[Tuple[_InternalEncodedAction, float], ...]
_InternalEncodedPolicyTable = Dict[str, _InternalEncodedDistribution]


def _coerce_policy_key(raw_key: Union[str, RoundInfoState]) -> str:
    if isinstance(raw_key, RoundInfoState):
        return str(raw_key)
    if isinstance(raw_key, str):
        return raw_key
    raise TypeError(f"Policy key must be str or RoundInfoState, got {type(raw_key).__name__}.")


def _validate_policy_action_shape(
    action_type: ActionType,
    cards: Sequence[Any],
    choice: Optional[int],
    *,
    context: str,
) -> None:
    if action_type not in _POLICY_ACTION_TYPES:
        raise ValueError(f"{context}: unsupported action type '{action_type.value}' for policy tables.")

    if action_type == ActionType.RESERVE:
        if len(cards) != 1:
            raise ValueError(f"{context}: reserve action must contain exactly one card.")
        if choice is not None:
            raise ValueError(f"{context}: reserve action must not set choice.")
        return

    if action_type == ActionType.DISCARD:
        if len(cards) != 2:
            raise ValueError(f"{context}: discard action must contain exactly two cards.")
        if choice not in (0, 1):
            raise ValueError(f"{context}: discard action choice must be 0 or 1.")
        return

    if action_type == ActionType.SPLIT:
        if len(cards) != 2:
            raise ValueError(f"{context}: split action must contain exactly two cards.")
        if choice is not None:
            raise ValueError(f"{context}: split action must not set choice.")
        return

    if action_type == ActionType.SPLIT_RESPONSE:
        if len(cards) != 0:
            raise ValueError(f"{context}: split_response action must not include cards.")
        if choice not in (0, 1):
            raise ValueError(f"{context}: split_response choice must be 0 or 1.")
        return

    raise ValueError(f"{context}: unsupported action type '{action_type.value}'.")


def _encode_action(action: Action, *, context: str) -> _InternalEncodedAction:
    if not isinstance(action, Action):
        raise TypeError(f"{context}: expected Action, got {type(action).__name__}.")

    cards: List[Any] = []
    for idx, card in enumerate(action.cards):
        if not isinstance(card, Card):
            raise TypeError(f"{context}: card {idx} is not a Card.")
        cards.append(card.suit)

    _validate_policy_action_shape(action.action_type, cards, action.choice, context=context)
    return (action.action_type, tuple(cards), action.choice)


def _encode_distribution(raw_distribution: Mapping[Action, float], *, context: str) -> _InternalEncodedDistribution:
    if not isinstance(raw_distribution, Mapping):
        raise TypeError(f"{context}: distribution must be a mapping.")
    if len(raw_distribution) == 0:
        raise ValueError(f"{context}: distribution cannot be empty.")

    encoded: List[Tuple[_InternalEncodedAction, float]] = []
    total_prob = 0.0

    for idx, (action, prob) in enumerate(raw_distribution.items()):
        entry_context = f"{context} entry {idx}"
        encoded_action = _encode_action(action, context=entry_context)
        try:
            weight = float(prob)
        except (TypeError, ValueError):
            raise ValueError(f"{entry_context}: probability must be numeric.")
        if weight < 0:
            raise ValueError(f"{entry_context}: probability must be non-negative.")
        encoded.append((encoded_action, weight))
        total_prob += weight

    if total_prob <= 0:
        raise ValueError(f"{context}: distribution total probability must be positive.")
    if abs(total_prob - 1.0) > _EPS:
        raise ValueError(f"{context}: probabilities must sum to 1.0, got {total_prob:.6f}.")

    return tuple(encoded)


def _encode_strategy_map(
    strategy_map: Mapping[Union[str, RoundInfoState], Mapping[Action, float]]
) -> _InternalEncodedPolicyTable:
    if not isinstance(strategy_map, Mapping):
        raise TypeError(f"strategy_map must be a mapping, got {type(strategy_map).__name__}.")

    encoded_table: _InternalEncodedPolicyTable = {}
    for raw_key, raw_distribution in strategy_map.items():
        key = _coerce_policy_key(raw_key)
        if key in encoded_table:
            raise ValueError(f"Duplicate policy key after string coercion: {key}")
        encoded_table[key] = _encode_distribution(raw_distribution, context=f"state '{key}'")

    return encoded_table


def _decode_distribution(encoded_distribution: _InternalEncodedDistribution) -> Dict[Action, float]:
    return {
        Action(action_type, [Card(suit) for suit in cards], choice=choice): prob
        for (action_type, cards, choice), prob in encoded_distribution
    }


def _serialize_suit(suit: Any, *, context: str) -> Any:
    if not hasattr(suit, "value"):
        raise ValueError(f"{context}: suit '{suit}' is missing '.value' for artifact serialization.")
    return suit.value


def _build_artifact_payload(
    encoded_table: _InternalEncodedPolicyTable,
    *,
    variant: Optional[str],
) -> Dict[str, Any]:
    entries: Dict[str, List[Dict[str, Any]]] = {}
    for key, distribution in encoded_table.items():
        encoded_distribution: List[Dict[str, Any]] = []
        for idx, ((action_type, cards, choice), prob) in enumerate(distribution):
            context = f"state '{key}' entry {idx}"
            encoded_distribution.append(
                {
                    "action_type": action_type.value,
                    "cards": [_serialize_suit(suit, context=context) for suit in cards],
                    "choice": choice,
                    "prob": prob,
                }
            )
        entries[key] = encoded_distribution

    return {
        "format": POLICY_TABLE_FORMAT,
        "key_encoding": POLICY_TABLE_KEY_ENCODING,
        "variant": variant,
        "entries": entries,
    }


def dump_policy_table_artifact(
    path: str,
    strategy_map: Mapping[Union[str, RoundInfoState], Mapping[Action, float]],
    *,
    variant: Optional[str] = None,
) -> None:
    """Serialize a strategy table to JSON artifact format."""
    encoded_table = _encode_strategy_map(strategy_map)
    payload = _build_artifact_payload(encoded_table, variant=variant)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def _decode_distribution_from_payload(
    key: str,
    raw_distribution: Any,
    suit_lookup: Mapping[Any, Any],
) -> _InternalEncodedDistribution:
    if not isinstance(raw_distribution, list):
        raise ValueError(f"state '{key}': distribution must be a list.")
    if len(raw_distribution) == 0:
        raise ValueError(f"state '{key}': distribution cannot be empty.")

    encoded_distribution: List[Tuple[_InternalEncodedAction, float]] = []
    total_prob = 0.0

    for idx, raw_entry in enumerate(raw_distribution):
        context = f"state '{key}' entry {idx}"
        if not isinstance(raw_entry, dict):
            raise ValueError(f"{context}: entry must be an object.")

        action_type_value = raw_entry.get("action_type")
        if action_type_value not in _ACTION_TYPE_BY_VALUE:
            raise ValueError(f"{context}: unknown action_type '{action_type_value}'.")
        action_type = _ACTION_TYPE_BY_VALUE[action_type_value]

        raw_cards = raw_entry.get("cards")
        if not isinstance(raw_cards, list):
            raise ValueError(f"{context}: cards must be a list.")

        cards: List[Any] = []
        for card_idx, raw_suit_value in enumerate(raw_cards):
            if raw_suit_value not in suit_lookup:
                raise ValueError(f"{context}: unknown suit value at cards[{card_idx}] -> {raw_suit_value}.")
            cards.append(suit_lookup[raw_suit_value])

        choice = raw_entry.get("choice")
        _validate_policy_action_shape(action_type, cards, choice, context=context)

        raw_prob = raw_entry.get("prob")
        try:
            prob = float(raw_prob)
        except (TypeError, ValueError):
            raise ValueError(f"{context}: prob must be numeric.")
        if prob < 0:
            raise ValueError(f"{context}: prob must be non-negative.")

        encoded_distribution.append(((action_type, tuple(cards), choice), prob))
        total_prob += prob

    if total_prob <= 0:
        raise ValueError(f"state '{key}': distribution total probability must be positive.")
    if abs(total_prob - 1.0) > _EPS:
        raise ValueError(f"state '{key}': probabilities must sum to 1.0, got {total_prob:.6f}.")

    return tuple(encoded_distribution)


def _load_policy_table_artifact_internal(
    path: str,
    suits_enum: type,
    *,
    expected_variant: Optional[str] = None,
) -> _InternalEncodedPolicyTable:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if not isinstance(payload, dict):
        raise ValueError("Artifact root must be an object.")

    artifact_format = payload.get("format")
    if artifact_format != POLICY_TABLE_FORMAT:
        raise ValueError(f"Unsupported artifact format '{artifact_format}', expected '{POLICY_TABLE_FORMAT}'.")

    key_encoding = payload.get("key_encoding")
    if key_encoding != POLICY_TABLE_KEY_ENCODING:
        raise ValueError(
            f"Unsupported key_encoding '{key_encoding}', expected '{POLICY_TABLE_KEY_ENCODING}'."
        )

    variant = payload.get("variant")
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(f"Artifact variant '{variant}' does not match expected '{expected_variant}'.")

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, dict):
        raise ValueError("Artifact 'entries' must be an object mapping key -> distribution.")

    suit_lookup = {suit.value: suit for suit in suits_enum}
    if not suit_lookup:
        raise ValueError("suits_enum must define at least one suit.")

    encoded_table: _InternalEncodedPolicyTable = {}
    for key, raw_distribution in raw_entries.items():
        if not isinstance(key, str):
            raise ValueError("Artifact entries keys must be strings.")
        encoded_table[key] = _decode_distribution_from_payload(key, raw_distribution, suit_lookup)

    return encoded_table


def load_policy_table_artifact(
    path: str,
    suits_enum: type,
    *,
    expected_variant: Optional[str] = None,
) -> Dict[str, Dict[Action, float]]:
    """Load a JSON policy-table artifact into an in-memory strategy mapping."""
    encoded_table = _load_policy_table_artifact_internal(path, suits_enum, expected_variant=expected_variant)
    return {key: _decode_distribution(distribution) for key, distribution in encoded_table.items()}


class PolicyTableAgent(Agent):
    """Agent that serves a precomputed policy table keyed by RoundInfoState string."""

    def __init__(
        self,
        strategy_map: Mapping[Union[str, RoundInfoState], Mapping[Action, float]],
    ):
        self._table = _encode_strategy_map(strategy_map)

    @classmethod
    def from_artifact(
        cls,
        path: str,
        suits_enum: type,
        *,
        expected_variant: Optional[str] = None,
    ) -> "PolicyTableAgent":
        encoded_table = _load_policy_table_artifact_internal(
            path,
            suits_enum,
            expected_variant=expected_variant,
        )
        agent = cls.__new__(cls)
        agent._table = encoded_table
        return agent

    def dump_artifact(self, path: str, *, variant: Optional[str] = None) -> None:
        """Write this agent's internal policy table to JSON artifact format."""
        payload = _build_artifact_payload(self._table, variant=variant)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)

    def get_action_distribution(self, info_state: RoundInfoState) -> Dict[Action, float]:
        key = str(info_state)
        if key not in self._table:
            raise ValueError(f"Policy table missing state key: {key}")
        return _decode_distribution(self._table[key])
