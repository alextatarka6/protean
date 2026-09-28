"""
Gen9 random-battle observation encoder.

Builds {"numbers": float32[NUMBERS_DIM], "text": str} from plain per-Pokémon
dicts, so the same encoder can serve live poke-env battles (battle.py) and,
later, parsed replays.

Pokémon dict ("mon") keys:
    species, hp (fraction), level, status ("" | brn | par | ...), boosts {stat: int},
    types [..], tera_type ("" if unknown), terastallized (bool),
    item (id | "unknownitem" | "noitem"), ability (id | "unknownability"),
    moves [ids], move_pp {id: fraction}   (move_pp only needed for my active)

Text (TEXT_LEN = 112 tokens, fixed):
    <gen9randombattle> <anychoice|forcedswitch>
    <player> species type1 type2 tera item ability status
      <move> name type category                                   ×4 (alphabetical)
      <switch> species item ability tera status <moveset> m1..m4  ×5 (alive bench, team order)
    <opponent> species type1 type2 tera item ability status
    <opp_moves> m1 m2 m3 m4          (revealed first, then inferred from randbats sets)
    <opp_bench> (species status)×5   (revealed, alive)
    <conditions> weather terrain
    <player_prev> move|<blank>  <opp_prev> move|<blank>

Numbers (NUMBERS_DIM = 88):
    [0]      my alive / 6
    [1]      opp alive / 6            (unrevealed opp mons count as alive)
    [2-18]   my active   (17, see _active_numbers)
    [19-34]  my moves ×4: base_power/200, accuracy, priority/5, pp fraction
    [35-44]  my bench ×5: hp, level/100
    [45-61]  opp active  (17)
    [62]     opp revealed moves / 4
    [63-67]  opp bench ×5: hp
    [68-69]  trick room, gravity
    [70-78]  my side conditions  (9, SIDE_CONDITIONS order)
    [79-87]  opp side conditions (9)
"""
from __future__ import annotations

import numpy as np

from protean.formats.gen9randombattle.dex import base_stats, infer_opponent, move_data

N_MOVE_SLOTS   = 4
N_SWITCH_SLOTS = 5
NUMBERS_DIM    = 88
TEXT_LEN       = 112

_PAD = -2.0   # numerical padding for missing moves / bench slots

BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
STAT_KEYS  = ["hp", "atk", "def", "spa", "spd", "spe"]

# (name, divisor) — divisor > 1 for stackable hazards (value = layers)
SIDE_CONDITIONS = [
    ("stealthrock", 1), ("spikes", 3), ("toxicspikes", 2), ("stickyweb", 1),
    ("reflect", 1), ("lightscreen", 1), ("auroraveil", 1), ("tailwind", 1), ("safeguard", 1),
]


def _status_token(status: str) -> str:
    return status or "nostatus"


def _type_tokens(mon: dict) -> list[str]:
    types = [t for t in mon["types"] if t][:2]
    return types + ["notype"] * (2 - len(types))


def _active_numbers(mon: dict, tera_available: bool) -> list[float]:
    """17 values: hp, level, 6 base stats, 7 boosts, terastallized, tera still available."""
    stats = base_stats(mon["species"])
    return (
        [float(mon["hp"]), mon["level"] / 100.0]
        + [stats[k] / 255.0 for k in STAT_KEYS]
        + [mon["boosts"].get(k, 0) / 6.0 for k in BOOST_KEYS]
        + [float(mon["terastallized"]), float(tera_available)]
    )


def _side_numbers(conditions: dict[str, int]) -> list[float]:
    return [min(conditions.get(name, 0), div) / div for name, div in SIDE_CONDITIONS]


def _mon_header(mon: dict) -> list[str]:
    return (
        [mon["species"]] + _type_tokens(mon)
        + [mon["tera_type"] or "unknowntera", mon["item"], mon["ability"], _status_token(mon["status"])]
    )


def build_obs(
    *,
    my_active: dict,
    my_bench: list[dict],          # alive, non-active, team order (= switch slot order)
    my_alive: int,
    my_tera_available: bool,
    opp_active: dict,
    opp_bench: list[dict],         # revealed, alive, non-active
    opp_alive: int,
    opp_tera_available: bool,
    weather: str,
    terrain: str,
    trick_room: bool,
    gravity: bool,
    my_side: dict[str, int],       # condition name → layers (stackable) or 1
    opp_side: dict[str, int],
    prev_my_move: str,
    prev_opp_move: str,
    forced_switch: bool,
) -> dict[str, np.ndarray]:
    my_moves = sorted(my_active["moves"])[:N_MOVE_SLOTS]
    bench    = my_bench[:N_SWITCH_SLOTS]

    # Opponent active: fill unrevealed moves / ability / tera from randbats sets
    opp_revealed = sorted(opp_active["moves"])[:N_MOVE_SLOTS]
    known_ability = opp_active["ability"] if opp_active["ability"] != "unknownability" else None
    opp_moves, opp_ability, opp_tera = infer_opponent(
        opp_active["species"], opp_revealed, known_ability, opp_active["tera_type"] or None,
    )
    opp_view = dict(opp_active, ability=opp_ability or "unknownability", tera_type=opp_tera or "")

    # ---- numbers ----
    nums: list[float] = [my_alive / 6.0, opp_alive / 6.0]
    nums += _active_numbers(my_active, my_tera_available)
    for i in range(N_MOVE_SLOTS):
        if i < len(my_moves):
            d = move_data(my_moves[i])
            nums += [d["base_power"] / 200.0, d["accuracy"], d["priority"] / 5.0,
                     float(my_active.get("move_pp", {}).get(my_moves[i], 1.0))]
        else:
            nums += [_PAD] * 4
    for i in range(N_SWITCH_SLOTS):
        nums += [float(bench[i]["hp"]), bench[i]["level"] / 100.0] if i < len(bench) else [_PAD, _PAD]
    nums += _active_numbers(opp_view, opp_tera_available)
    nums.append(len(opp_revealed) / 4.0)
    for i in range(N_SWITCH_SLOTS):
        nums.append(float(opp_bench[i]["hp"]) if i < len(opp_bench) else _PAD)
    nums += [float(trick_room), float(gravity)]
    nums += _side_numbers(my_side) + _side_numbers(opp_side)
    assert len(nums) == NUMBERS_DIM, f"numbers dim {len(nums)} != {NUMBERS_DIM}"

    # ---- text ----
    toks = ["<gen9randombattle>", "<forcedswitch>" if forced_switch else "<anychoice>"]
    toks += ["<player>"] + _mon_header(my_active)
    for i in range(N_MOVE_SLOTS):
        if i < len(my_moves):
            d = move_data(my_moves[i])
            toks += ["<move>", my_moves[i], d["type"], d["category"]]
        else:
            toks += ["<move>", "<blank>", "<blank>", "<blank>"]
    for i in range(N_SWITCH_SLOTS):
        if i < len(bench):
            p = bench[i]
            moves = sorted(p["moves"])[:N_MOVE_SLOTS]
            toks += ["<switch>", p["species"], p["item"], p["ability"],
                     p["tera_type"] or "unknowntera", _status_token(p["status"]), "<moveset>"]
            toks += moves + ["<blank>"] * (N_MOVE_SLOTS - len(moves))
        else:
            toks += ["<switch>"] + ["<blank>"] * 5 + ["<moveset>"] + ["<blank>"] * N_MOVE_SLOTS
    toks += ["<opponent>"] + _mon_header(opp_view)
    toks += ["<opp_moves>"] + opp_moves + ["<blank>"] * (N_MOVE_SLOTS - len(opp_moves))
    toks.append("<opp_bench>")
    for i in range(N_SWITCH_SLOTS):
        if i < len(opp_bench):
            toks += [opp_bench[i]["species"], _status_token(opp_bench[i]["status"])]
        else:
            toks += ["<blank>", "<blank>"]
    toks += ["<conditions>", weather or "noweather", terrain or "noterrain"]
    toks += ["<player_prev>", prev_my_move or "<blank>", "<opp_prev>", prev_opp_move or "<blank>"]
    assert len(toks) == TEXT_LEN, f"text len {len(toks)} != {TEXT_LEN}"
    assert all(t and " " not in t for t in toks), f"empty or multi-word token in {toks}"

    return {
        "numbers": np.array(nums, dtype=np.float32),
        "text":    np.array(" ".join(toks), dtype=np.str_),
    }
