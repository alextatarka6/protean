"""
Gen9 OU observation encoder: metamon UniversalState dict → {"numbers", "text"}.

Used for both dataset rows (metamon parsed replays) and live battles
(universal.state_from_battle), so BC training and live play see identical inputs.

Move and switch order match metamon's action indices (consistent_move_order /
consistent_pokemon_order): alphabetical by cleaned name.

Text (TEXT_LEN = 112 tokens, fixed):
    <gen9ou> <anychoice|forcedswitch>
    <player> name type1 type2 tera item ability status effect
      <move> name type category                                  ×4
      <switch> name tera item ability status <moveset> m1..m4    ×5
    <opponent> name type1 type2 tera item ability status effect
    <opp_moves> m1 m2 m3 m4           (revealed, alphabetical)
    <opp_team> s1 .. s6               (team preview species, alphabetical)
    <conditions> weather field my_condition opp_condition
    <player_prev> move <opp_prev> move

Numbers (NUMBERS_DIM = 55):
    [0]      opponents remaining / 6
    [1]      my remaining / 6
    [2]      can tera
    [3-17]   my active: hp, lvl/100, base atk/spa/def/spd/spe/hp /255, 7 boosts /6
    [18-33]  my moves ×4: base_power/200, accuracy, priority/5, pp fraction
    [34-38]  my bench ×5: hp
    [39-53]  opp active (same 15 as my active)
    [54]     opp revealed moves / 4
"""
from __future__ import annotations

import numpy as np

N_MOVE_SLOTS   = 4
N_SWITCH_SLOTS = 5
TEAM_SIZE      = 6
NUMBERS_DIM    = 55
TEXT_LEN       = 112

_PAD = -2.0

_BASE_STATS = ["atk", "spa", "def", "spd", "spe", "hp"]
_BOOSTS     = ["atk", "spa", "def", "spd", "spe", "accuracy", "evasion"]


def sorted_moves(pokemon: dict) -> list[dict]:
    return sorted(pokemon["moves"], key=lambda m: m["name"])[:N_MOVE_SLOTS]


def sorted_switches(state: dict) -> list[dict]:
    return sorted(state["available_switches"], key=lambda p: p["name"])[:N_SWITCH_SLOTS]


def _pokemon_numbers(p: dict) -> list[float]:
    return (
        [float(p["hp_pct"]), p["lvl"] / 100.0]
        + [p[f"base_{s}"] / 255.0 for s in _BASE_STATS]
        + [p[f"{b}_boost"] / 6.0 for b in _BOOSTS]
    )


def _pokemon_header(p: dict) -> list[str]:
    types = p["types"].split()
    types = (types + ["notype", "notype"])[:2]
    return [p["name"], *types, p["tera_type"], p["item"], p["ability"], p["status"], p["effect"]]


def state_to_obs(state: dict) -> dict[str, np.ndarray]:
    me, opp  = state["player_active_pokemon"], state["opponent_active_pokemon"]
    moves    = sorted_moves(me)
    switches = sorted_switches(state)
    opp_moves = sorted(m["name"] for m in opp["moves"])[:N_MOVE_SLOTS]

    # ---- numbers ----
    my_remaining = (me["hp_pct"] > 0) + len(state["available_switches"])
    nums: list[float] = [state["opponents_remaining"] / 6.0, my_remaining / 6.0,
                         float(state["can_tera"])]
    nums += _pokemon_numbers(me)
    for i in range(N_MOVE_SLOTS):
        if i < len(moves):
            m = moves[i]
            nums += [m["base_power"] / 200.0, float(m["accuracy"]), m["priority"] / 5.0,
                     m["current_pp"] / m["max_pp"] if m["max_pp"] else 1.0]
        else:
            nums += [_PAD] * 4
    for i in range(N_SWITCH_SLOTS):
        nums.append(float(switches[i]["hp_pct"]) if i < len(switches) else _PAD)
    nums += _pokemon_numbers(opp)
    nums.append(len(opp_moves) / 4.0)
    assert len(nums) == NUMBERS_DIM, f"numbers dim {len(nums)} != {NUMBERS_DIM}"

    # ---- text ----
    toks = ["<gen9ou>", "<forcedswitch>" if state["forced_switch"] else "<anychoice>"]
    toks += ["<player>"] + _pokemon_header(me)
    for i in range(N_MOVE_SLOTS):
        if i < len(moves):
            toks += ["<move>", moves[i]["name"], moves[i]["move_type"], moves[i]["category"]]
        else:
            toks += ["<move>", "<blank>", "<blank>", "<blank>"]
    for i in range(N_SWITCH_SLOTS):
        if i < len(switches):
            p = switches[i]
            ms = [m["name"] for m in sorted_moves(p)]
            toks += ["<switch>", p["name"], p["tera_type"], p["item"], p["ability"], p["status"],
                     "<moveset>"] + ms + ["<blank>"] * (N_MOVE_SLOTS - len(ms))
        else:
            toks += ["<switch>"] + ["<blank>"] * 5 + ["<moveset>"] + ["<blank>"] * N_MOVE_SLOTS
    toks += ["<opponent>"] + _pokemon_header(opp)
    toks += ["<opp_moves>"] + opp_moves + ["<blank>"] * (N_MOVE_SLOTS - len(opp_moves))
    preview = sorted(state.get("opponent_teampreview") or [])[:TEAM_SIZE]
    toks += ["<opp_team>"] + preview + ["<blank>"] * (TEAM_SIZE - len(preview))
    toks += ["<conditions>", state["weather"], state["battle_field"],
             state["player_conditions"], state["opponent_conditions"]]
    my_prev, opp_prev = state["player_prev_move"]["name"], state["opponent_prev_move"]["name"]
    toks += ["<player_prev>", my_prev if my_prev != "nomove" else "<blank>",
             "<opp_prev>", opp_prev if opp_prev != "nomove" else "<blank>"]
    # Every slot must stay exactly one whitespace-free token so positions are fixed.
    toks = [t.replace(" ", "") or "<blank>" for t in toks]
    assert len(toks) == TEXT_LEN, f"text len {len(toks)} != {TEXT_LEN}"

    return {
        "numbers": np.array(nums, dtype=np.float32),
        "text":    np.array(" ".join(toks), dtype=np.str_),
    }
