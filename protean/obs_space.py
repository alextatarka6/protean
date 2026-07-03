"""
Gen1OU observation and action spaces.

Produces observations in the same numbers + text format as metamon's
DefaultObservationSpace, adapted for gen1 (no items, no abilities, no tera).

Observation dict:
    "numbers"  np.ndarray[float32, shape=(48,)]   numerical features
    "text"     str                                 space-separated token string

Action space: 9 discrete slots
    0-3  use move 1-4 (alphabetical order within the active pokemon's moveset)
    4-8  switch to bench pokemon 1-5 (team order, skipping active + fainted)

Both state_to_obs (live POVSnapshot) and row_to_obs (HF dataset row + turn index)
produce identical output for the same game state, so the same model works for
BC training and live play.

Usage:
    obs_space  = Gen1OUObservationSpace()
    action_space = Gen1ActionSpace()

    # From dataset row
    obs = obs_space.row_to_obs(row, turn_idx=5)
    action_idx = action_space.row_to_action_idx(row, turn_idx=5)

    # From live snapshot (poke-env / replay replay)
    obs = obs_space.state_to_obs(snapshot, prev_my_action, prev_opp_action)
"""
from __future__ import annotations

import json
import re
from typing import Optional

import numpy as np

from protean.pokedex import get_base_stats, get_types, get_move_data
from protean.tokenizer import get_tokenizer, _clean

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_MOVE_SLOTS = 4
N_SWITCH_SLOTS = 5
NUMBERS_DIM = 48   # must stay 48 to match metamon's architecture

_PAD_FLOAT = -2.0  # padding value for unknown/missing numerical features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_status(status: str) -> str:
    """Map a raw status string from the dataset to a tokenizable status token."""
    if not status:
        return "nostatus"
    s = status.lower().strip()
    return s if s else "nostatus"


def _norm_weather(weather: str) -> str:
    if not weather:
        return "noweather"
    return _clean(weather) or "noweather"


def _norm_conditions(conditions_json: str) -> str:
    """
    Parse a JSON list of condition names and return a single space-joined token string.
    Returns "noconditions" if the list is empty.
    """
    try:
        conds = json.loads(conditions_json)
    except (json.JSONDecodeError, TypeError):
        conds = []
    if not conds:
        return "noconditions"
    return " ".join(_clean(c) for c in conds if c)


def _sorted_moves(moves: list[str]) -> list[str]:
    """Return moves in a consistent alphabetical order (same as metamon's consistent_move_order)."""
    return sorted(moves, key=lambda m: _clean(m))


# ---------------------------------------------------------------------------
# Numbers builder
# ---------------------------------------------------------------------------

def _pokemon_numerical(
    hp: float,
    species: str,
    boosts: dict[str, int],
    level: int = 100,
    is_active: bool = True,
) -> list[float]:
    """
    Build the numerical feature block for one Pokémon.

    Active pokemon: 1 (hp) + 1 (lvl) + 6 (base stats) + 7 (boosts) = 15 values
    Bench pokemon:  1 (hp) only = 1 value
    """
    if not is_active:
        return [float(hp)]

    stats = get_base_stats(species)
    stat_feats = [
        stats["atk"] / 255.0,
        stats["spc"] / 255.0,   # spa in gen1 = spc
        stats["def"] / 255.0,
        stats["spc"] / 255.0,   # spd in gen1 = spc
        stats["spe"] / 255.0,
        stats["hp"]  / 255.0,
    ]
    boost_feats = [
        boosts.get("atk",      0) / 6.0,
        boosts.get("spa",      0) / 6.0,
        boosts.get("def",      0) / 6.0,
        boosts.get("spd",      0) / 6.0,
        boosts.get("spe",      0) / 6.0,
        boosts.get("accuracy", 0) / 6.0,
        boosts.get("evasion",  0) / 6.0,
    ]
    return [float(hp), level / 100.0] + stat_feats + boost_feats


def _move_numerical(move_name: str) -> list[float]:
    """3-dim numerical features for one move: [power/200, accuracy, priority/5]."""
    d = get_move_data(move_name)
    return [
        d["base_power"] / 200.0,
        d["accuracy"],
        d["priority"] / 5.0,
    ]


def _pad_move_numerical() -> list[float]:
    return [_PAD_FLOAT, _PAD_FLOAT, _PAD_FLOAT]


# ---------------------------------------------------------------------------
# Text builder
# ---------------------------------------------------------------------------

def _pokemon_text_active(species: str, status: str) -> list[str]:
    """Text tokens for the active pokemon (species + types + status). No item/ability in gen1."""
    types = get_types(species)
    return [_clean(species)] + types + [_norm_status(status)]


def _pokemon_text_bench(species: str, moves: list[str], status: str = "") -> list[str]:
    """Text tokens for a bench pokemon: species + status + <moveset> + move names (padded to 4)."""
    sorted_m = _sorted_moves(moves)
    tokens = [_clean(species), _norm_status(status), "<moveset>"]
    for i in range(N_MOVE_SLOTS):
        tokens.append(_clean(sorted_m[i]) if i < len(sorted_m) else "<blank>")
    return tokens


def _move_text(move_name: str) -> list[str]:
    """Text tokens for a move on the active pokemon: name + type + category."""
    d = get_move_data(move_name)
    return [_clean(move_name), d["type"], d["category"]]


# ---------------------------------------------------------------------------
# Core observation builder (shared by row_to_obs and state_to_obs)
# ---------------------------------------------------------------------------

def _build_obs(
    *,
    my_active_species: str,
    my_active_hp: float,
    my_active_status: str,
    my_active_boosts: dict[str, int],
    my_active_moves: list[str],
    my_bench: list[dict],          # list of {species, hp, status, fainted, revealed_moves}
    opp_active_species: str,
    opp_active_hp: float,
    opp_active_status: str,
    opp_active_boosts: dict[str, int],
    opp_bench: list[dict],         # opponent's non-active non-fainted revealed pokemon
    opp_remaining: int,
    weather: str,
    my_conditions: str,
    opp_conditions: str,
    prev_my_move: str,
    prev_opp_move: str,
    forced_switch: bool,
) -> dict[str, np.ndarray]:

    # ---- numbers ----
    numerical: list[float] = [opp_remaining / 6.0]

    # player active
    numerical += _pokemon_numerical(
        my_active_hp, my_active_species, my_active_boosts, is_active=True
    )

    # player moves (sorted alphabetically, padded to 4)
    sorted_moves = _sorted_moves(my_active_moves)
    for i in range(N_MOVE_SLOTS):
        if i < len(sorted_moves):
            numerical += _move_numerical(sorted_moves[i])
        else:
            numerical += _pad_move_numerical()

    # player bench (hp only, padded to 5)
    alive_bench = [p for p in my_bench if not p.get("fainted", False)]
    for i in range(N_SWITCH_SLOTS):
        if i < len(alive_bench):
            numerical.append(float(alive_bench[i]["hp"]))
        else:
            numerical.append(_PAD_FLOAT)

    # opponent active
    numerical += _pokemon_numerical(
        opp_active_hp, opp_active_species, opp_active_boosts, is_active=True
    )

    assert len(numerical) == NUMBERS_DIM, (
        f"Numbers dim mismatch: got {len(numerical)}, expected {NUMBERS_DIM}"
    )

    # ---- text ----
    choice_token = "<forcedswitch>" if forced_switch else "<anychoice>"

    # player active
    player_tokens = (
        ["<player>"]
        + _pokemon_text_active(my_active_species, my_active_status)
    )

    # player moves
    move_tokens: list[str] = []
    for i in range(N_MOVE_SLOTS):
        if i < len(sorted_moves):
            move_tokens += ["<move>"] + _move_text(sorted_moves[i])
        else:
            move_tokens += ["<move>", "<blank>", "<blank>", "<blank>"]

    # player bench
    switch_tokens: list[str] = []
    for i in range(N_SWITCH_SLOTS):
        if i < len(alive_bench):
            p = alive_bench[i]
            switch_tokens += (
                ["<switch>"]
                + _pokemon_text_bench(p["species"], p.get("revealed_moves", []),
                                      p.get("status", ""))
            )
        else:
            switch_tokens += ["<switch>", "<blank>", "<blank>", "<moveset>",
                              "<blank>", "<blank>", "<blank>", "<blank>"]

    # opponent active
    opp_tokens = (
        ["<opponent>"]
        + _pokemon_text_active(opp_active_species, opp_active_status)
    )

    # conditions
    cond_tokens = [
        "<conditions>",
        _norm_weather(weather),
        my_conditions,
        opp_conditions,
    ]

    # previous moves
    prev_tokens = [
        "<player_prev>", _clean(prev_my_move) if prev_my_move else "<blank>",
        "<opp_prev>",    _clean(prev_opp_move) if prev_opp_move else "<blank>",
    ]

    full_tokens = (
        ["<gen1ou>", choice_token]
        + player_tokens
        + move_tokens
        + switch_tokens
        + opp_tokens
        + cond_tokens
        + prev_tokens
    )

    text = " ".join(full_tokens)
    return {
        "numbers": np.array(numerical, dtype=np.float32),
        "text":    np.array(text, dtype=np.str_),
    }


# ---------------------------------------------------------------------------
# Observation Space
# ---------------------------------------------------------------------------

class Gen1OUObservationSpace:
    """
    Converts game state at a single timestep to model inputs.

    Two entry points:
        row_to_obs(row, turn_idx)           — from HuggingFace dataset row
        state_to_obs(snap, prev_my, prev_opp) — from live POVSnapshot
    """

    @property
    def numbers_dim(self) -> int:
        return NUMBERS_DIM

    # ------------------------------------------------------------------
    # Dataset row entry point
    # ------------------------------------------------------------------

    def row_to_obs(self, row: dict, turn_idx: int) -> dict[str, np.ndarray]:
        """
        Build an observation from a single turn of a stored dataset row.

        Args:
            row:       One row from the atatark2/protean-gen1ou HF dataset.
            turn_idx:  Which turn to encode (0 = first turn of the trajectory).
        """
        t = turn_idx

        # Parse JSON blobs
        my_team: list[dict] = json.loads(row["my_team"][t])
        opp_team: list[dict] = json.loads(row["opp_seen_team"][t])
        my_active_boosts: dict = json.loads(row["my_active_boosts"][t])
        my_side_conds = _norm_conditions(row["my_side_conditions"][t])
        opp_side_conds = _norm_conditions(row["opp_side_conditions"][t])

        my_active_species = row["my_active_species"][t]
        opp_active_species = row["opp_active_species"][t]

        # Separate active from bench
        my_bench = [p for p in my_team
                    if p["species"] != my_active_species and not p["fainted"]]
        opp_active_dict = next(
            (p for p in opp_team if p["species"] == opp_active_species), None
        )
        opp_bench = [p for p in opp_team
                     if p["species"] != opp_active_species and not p["fainted"]]
        opp_boosts = opp_active_dict["boosts"] if opp_active_dict else {}

        # Opponent remaining (non-fainted in revealed team)
        opp_remaining = sum(1 for p in opp_team if not p["fainted"])

        # Active pokemon's moves (from team entry, not just active slot)
        my_active_dict = next(
            (p for p in my_team if p["species"] == my_active_species), None
        )
        my_active_moves = my_active_dict["revealed_moves"] if my_active_dict else []

        # Previous move (from prior turn's action, blank on turn 0)
        prev_my_move  = row["my_action_value"][t - 1]  if t > 0 else ""
        prev_opp_move = row["opp_action_value"][t - 1] if t > 0 else ""
        # Only keep moves, not switches
        if t > 0 and row["my_action_kind"][t - 1]  != "move": prev_my_move  = ""
        if t > 0 and row["opp_action_kind"][t - 1] != "move": prev_opp_move = ""

        forced = bool(row["my_action_forced"][t])

        return _build_obs(
            my_active_species=my_active_species,
            my_active_hp=float(row["my_active_hp"][t]),
            my_active_status=row["my_active_status"][t],
            my_active_boosts=my_active_boosts,
            my_active_moves=my_active_moves,
            my_bench=my_bench,
            opp_active_species=opp_active_species,
            opp_active_hp=float(row["opp_active_hp"][t]),
            opp_active_status=row["opp_active_status"][t],
            opp_active_boosts=opp_boosts,
            opp_bench=opp_bench,
            opp_remaining=opp_remaining,
            weather=row["weather"][t],
            my_conditions=my_side_conds,
            opp_conditions=opp_side_conds,
            prev_my_move=prev_my_move,
            prev_opp_move=prev_opp_move,
            forced_switch=forced,
        )

    # ------------------------------------------------------------------
    # Live POVSnapshot entry point
    # ------------------------------------------------------------------

    def state_to_obs(
        self,
        snap,                        # POVSnapshot
        prev_my_action=None,         # Optional[Action]
        prev_opp_action=None,        # Optional[Action]
    ) -> dict[str, np.ndarray]:
        """
        Build an observation from a live POVSnapshot (replay parser or poke-env).
        """
        from protean.backend.replay_parser.types import POVSnapshot, Action

        my_active = snap.my_side.active
        opp_active = snap.opp_side.active

        my_active_species = my_active.species if my_active else "missingno"
        my_active_hp      = float(my_active.hp) if my_active else 1.0
        my_active_status  = my_active.status or "" if my_active else ""
        my_active_boosts  = dict(my_active.boosts) if my_active else {}
        my_active_moves   = list(my_active.revealed_moves) if my_active else []

        opp_active_species = opp_active.species if opp_active else "missingno"
        opp_active_hp      = float(opp_active.hp) if opp_active else 1.0
        opp_active_status  = opp_active.status or "" if opp_active else ""
        opp_active_boosts  = dict(opp_active.boosts) if opp_active else {}

        def _pk_to_bench_dict(pk):
            return {
                "species":       pk.species,
                "hp":            float(pk.hp),
                "status":        pk.status or "",
                "fainted":       pk.fainted,
                "revealed_moves": list(pk.revealed_moves),
            }

        my_bench = [
            _pk_to_bench_dict(p)
            for p in snap.my_side.team
            if p.species != my_active_species and not p.fainted
        ]
        opp_bench = [
            _pk_to_bench_dict(p)
            for p in snap.opp_side.team
            if p.species != opp_active_species and not p.fainted
        ]
        opp_remaining = sum(1 for p in snap.opp_side.team if not p.fainted)

        prev_my_move  = (prev_my_action.value
                         if prev_my_action and prev_my_action.kind == "move" else "")
        prev_opp_move = (prev_opp_action.value
                         if prev_opp_action and prev_opp_action.kind == "move" else "")

        weather     = snap.field.weather or ""
        my_conds    = _norm_conditions(json.dumps(list(snap.my_side.conditions.keys())))
        opp_conds   = _norm_conditions(json.dumps(list(snap.opp_side.conditions.keys())))
        forced      = snap.my_action.forced

        return _build_obs(
            my_active_species=my_active_species,
            my_active_hp=my_active_hp,
            my_active_status=my_active_status,
            my_active_boosts=my_active_boosts,
            my_active_moves=my_active_moves,
            my_bench=my_bench,
            opp_active_species=opp_active_species,
            opp_active_hp=opp_active_hp,
            opp_active_status=opp_active_status,
            opp_active_boosts=opp_active_boosts,
            opp_bench=opp_bench,
            opp_remaining=opp_remaining,
            weather=weather,
            my_conditions=my_conds,
            opp_conditions=opp_conds,
            prev_my_move=prev_my_move,
            prev_opp_move=prev_opp_move,
            forced_switch=forced,
        )


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

class Gen1ActionSpace:
    """
    Fixed 9-slot discrete action space:
        slots 0-3  → use move 1-4 (alphabetical order of the active pokemon's moves)
        slots 4-8  → switch to bench pokemon 1-5 (alive bench order)

    Provides helpers to encode actions from dataset rows and to decode
    slot indices back to (kind, value) pairs for sending to the battle engine.
    """

    N_ACTIONS = 9
    N_MOVE_SLOTS   = N_MOVE_SLOTS
    N_SWITCH_SLOTS = N_SWITCH_SLOTS

    def row_to_action_idx(self, row: dict, turn_idx: int) -> int:
        """
        Map the action taken at turn_idx in a dataset row to a slot index 0-8.
        Returns -1 if the action can't be mapped (e.g. missing move data).
        """
        t = turn_idx
        kind  = row["my_action_kind"][t]
        value = row["my_action_value"][t]

        my_team: list[dict] = json.loads(row["my_team"][t])
        my_active_species = row["my_active_species"][t]

        if kind == "move":
            my_active_dict = next(
                (p for p in my_team if p["species"] == my_active_species), None
            )
            moves = _sorted_moves(my_active_dict["revealed_moves"]) if my_active_dict else []
            if len(moves) > N_MOVE_SLOTS:
                import warnings
                warnings.warn(
                    f"{my_active_species!r} has {len(moves)} revealed_moves "
                    f"{moves!r} — team_inference overfill; skipping turn"
                )
                return -1
            value_clean = _clean(value)
            for i, m in enumerate(moves):
                if _clean(m) == value_clean:
                    return i  # slot 0-3
            return -1

        elif kind == "switch":
            alive_bench = [
                p for p in my_team
                if p["species"] != my_active_species and not p["fainted"]
            ]
            if len(alive_bench) > N_SWITCH_SLOTS:
                import warnings
                warnings.warn(
                    f"alive_bench has {len(alive_bench)} entries "
                    f"(active={my_active_species!r}, team={[p['species'] for p in my_team]!r}) "
                    f"— likely a parser/inference duplicate; skipping turn"
                )
                return -1
            value_clean = _clean(value)
            for i, p in enumerate(alive_bench):
                if _clean(p["species"]) == value_clean:
                    return N_MOVE_SLOTS + i  # slot 4-8
            return -1

        return -1

    def idx_to_action(
        self,
        idx: int,
        my_active_moves: list[str],
        alive_bench_species: list[str],
    ) -> tuple[str, str]:
        """
        Decode a slot index back to (kind, value).

        Args:
            idx:                 Slot index 0-8.
            my_active_moves:     Current active pokemon's moves (unsorted; will be sorted internally).
            alive_bench_species: Species names of alive bench pokemon in team order.

        Returns:
            (kind, value) where kind is "move" or "switch" and value is the name.
        """
        if idx < N_MOVE_SLOTS:
            sorted_moves = _sorted_moves(my_active_moves)
            if idx < len(sorted_moves):
                return ("move", sorted_moves[idx])
            raise ValueError(f"Move slot {idx} out of range (only {len(sorted_moves)} moves known)")
        else:
            switch_idx = idx - N_MOVE_SLOTS
            if switch_idx < len(alive_bench_species):
                return ("switch", alive_bench_species[switch_idx])
            raise ValueError(f"Switch slot {idx} out of range (only {len(alive_bench_species)} bench pokemon)")

    def action_mask(self, row: dict, turn_idx: int) -> np.ndarray:
        """
        Return a boolean mask of shape (9,) — True where the action is valid.
        Forced switches mask out all move slots.
        """
        t = turn_idx
        mask = np.zeros(self.N_ACTIONS, dtype=bool)

        my_team: list[dict] = json.loads(row["my_team"][t])
        my_active_species = row["my_active_species"][t]
        my_active_dict = next(
            (p for p in my_team if p["species"] == my_active_species), None
        )
        alive_bench = [
            p for p in my_team
            if p["species"] != my_active_species and not p["fainted"]
        ]
        forced = bool(row["my_action_forced"][t])

        if not forced and my_active_dict:
            n_moves = min(len(my_active_dict["revealed_moves"]), N_MOVE_SLOTS)
            mask[:n_moves] = True

        for i in range(min(len(alive_bench), N_SWITCH_SLOTS)):
            mask[N_MOVE_SLOTS + i] = True

        return mask
