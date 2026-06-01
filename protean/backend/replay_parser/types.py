from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Winner(Enum):
    P1 = "p1"
    P2 = "p2"
    TIE = "tie"


@dataclass
class BattlePokemon:
    species: str
    nickname: str
    level: int
    hp: float  # current HP as fraction 0.0–1.0
    status: Optional[str] = None  # "par" | "brn" | "slp" | "frz" | "psn" | "tox"
    boosts: dict[str, int] = field(
        default_factory=lambda: {
            "atk": 0, "def": 0, "spa": 0, "spd": 0,
            "spe": 0, "accuracy": 0, "evasion": 0,
        }
    )
    revealed_moves: list[str] = field(default_factory=list)  # in order first seen
    item: Optional[str] = None
    ability: Optional[str] = None
    fainted: bool = False
    seen_in_battle: bool = False  # True once this pokemon has switched into the field


@dataclass
class SideState:
    player: str
    active: Optional[BattlePokemon] = None
    team: list[BattlePokemon] = field(default_factory=list)  # revealed, in switch-in order
    conditions: dict[str, int] = field(default_factory=dict)  # e.g. {"Reflect": 1}


@dataclass
class FieldState:
    weather: Optional[str] = None
    conditions: dict[str, int] = field(default_factory=dict)  # e.g. {"Trick Room": 1}


@dataclass
class Action:
    kind: str   # "move" | "switch"
    value: str  # move name or species switched to
    forced: bool = False  # True for drag or post-faint forced switch


@dataclass
class TurnSnapshot:
    turn_number: int
    p1: SideState      # state at the START of this turn (before actions resolve)
    p2: SideState
    field: FieldState
    p1_action: Optional[Action]  # what p1 did this turn (None if not reconstructable)
    p2_action: Optional[Action]


@dataclass
class ParsedBattle:
    battle_id: str
    format: str
    gen: int
    p1_name: str
    p2_name: str
    p1_rating: Optional[int]
    p2_rating: Optional[int]
    turns: list[TurnSnapshot]
    winner: Winner


@dataclass
class POVSnapshot:
    """One turn from a single player's perspective.

    Both my_side and opp_side have been enriched by team inference before
    the snapshot is built: missing moves, items, and abilities are filled
    from usage stats, and unrevealed team slots are filled with sampled
    pokemon based on teammates co-occurrence data.
    """
    turn_number: int
    my_side: SideState      # own team, inference-completed
    opp_side: SideState     # opponent's state, inference-completed
    field: FieldState
    my_action: Action       # what the POV player did (always present — filtered turns excluded)
    opp_action: Optional[Action] = None  # opponent's action (visible in replay; unknown in live play)


@dataclass
class POVReplay:
    """All usable turns from one player's viewpoint within a single battle."""
    battle_id: str
    format: str
    gen: int
    pov_player: str         # "p1" or "p2"
    player_name: str
    opponent_name: str
    winner: Winner
    snapshots: list[POVSnapshot] = field(default_factory=list)

    @property
    def won(self) -> bool:
        return (
            (self.pov_player == "p1" and self.winner == Winner.P1)
            or (self.pov_player == "p2" and self.winner == Winner.P2)
        )
