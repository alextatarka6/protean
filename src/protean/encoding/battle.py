"""Phase 1: battle state → flat float32 feature vector.

Feature vector layout  (FEATURE_DIM = 451):

  [0:49]    our active mon
  [49:98]   opponent active mon
    Each active-mon block (49 features):
      [0]     HP fraction                        (float, 0–1)
      [1:8]   status one-hot                     (7: none, BRN, FRZ, PAR, PSN, SLP, TOX)
      [8:26]  type_1 one-hot                     (18 real types, alphabetical)
      [26:44] type_2 one-hot                     (18; all-zero if single-type)
      [44:49] stat boosts: ATK DEF SPA SPD SPE   (each divided by 6 → [-1, 1])

  [98:146]  available moves — 4 slots × 12 features:
      [0]   move exists                          (1 or 0)
      [1]   base_power / 150
      [2]   type effectiveness vs opponent       (immune→0, ×0.25→0.25, ×0.5→0.5,
                                                  ×1→0.75, ×2→0.875, ×4→1.0)
      [3]   STAB                                 (1 if move type ∈ user's types)
      [4]   accuracy                             (0–1; 1.0 if always hits)
      [5]   priority / 7
      [6]   PP fraction                          (current_pp / max_pp)
      [7]   is_physical
      [8]   is_special
      [9]   is_status
      [10]  makes_contact                        ("contact" in move.flags)
      [11]  has_positive_priority                (priority > 0)

  [146:152] weather one-hot  (6: none, SUN, RAIN, SAND, SNOW, HAIL)
  [152:157] terrain one-hot  (5: none, ELECTRIC, GRASSY, MISTY, PSYCHIC)
  [157:161] our side hazards:     SR(/1)  Spikes(/3)  ToxicSpikes(/2)  StickyWeb(/1)
  [161:165] opponent side hazards: same
  [165:168] our screens:      Reflect  LightScreen  AuroraVeil        (binary)
  [168:171] opponent screens: same

  [171:311] our bench — 5 slots × 28 features
  [311:451] opponent bench — 5 slots × 28 features
    Each bench-mon slot (28 features):
      [0]     exists / known                     (1 if slot is filled / revealed)
      [1]     HP fraction
      [2]     fainted
      [3:10]  status one-hot                     (7, same order as active-mon block)
      [10:28] type_1 one-hot                     (18)
"""
import numpy as np
from poke_env.battle.battle import Battle
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition

# The 18 real competitive types in alphabetical order (matches PokemonType enum minus
# the pseudo-types THREE_QUESTION_MARKS and STELLAR).
_REAL_TYPES = [
    t for t in PokemonType
    if t not in (PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR)
]  # 18 types

# Status conditions used in the 7-dim one-hot (index 0 = no status).
_STATUS_ORDER = [Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX]

# Map raw type-effectiveness multiplier → normalised scalar in [0, 1].
_EFF_SCALE = {0.0: 0.0, 0.25: 0.25, 0.5: 0.5, 1.0: 0.75, 2.0: 0.875, 4.0: 1.0}

# Weathers encoded at indices 1-5 (index 0 = no weather).
_WEATHER_ORDER = [
    Weather.SUNNYDAY, Weather.RAINDANCE, Weather.SANDSTORM,
    Weather.SNOWSCAPE, Weather.HAIL,
]

# Terrains encoded at indices 1-4 (index 0 = no terrain).
_TERRAIN_ORDER = [
    Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN,
    Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN,
]

FEATURE_DIM = 451


# ---------------------------------------------------------------------------
# Helper encoders
# ---------------------------------------------------------------------------

def _status_vec(status) -> list:
    vec = [0.0] * 7
    if status is None or status == Status.FNT:
        vec[0] = 1.0
    elif status in _STATUS_ORDER:
        vec[_STATUS_ORDER.index(status) + 1] = 1.0
    else:
        vec[0] = 1.0
    return vec


def _type_vec(ptype) -> list:
    vec = [0.0] * 18
    if ptype in _REAL_TYPES:
        vec[_REAL_TYPES.index(ptype)] = 1.0
    return vec


def _encode_active_mon(mon) -> list:
    if mon is None:
        return [0.0] * 49
    feats = [mon.current_hp_fraction]
    feats += _status_vec(mon.status)
    feats += _type_vec(mon.type_1)
    feats += _type_vec(mon.type_2)
    boosts = mon.boosts
    feats += [boosts.get(s, 0) / 6.0 for s in ("atk", "def", "spa", "spd", "spe")]
    return feats


def _encode_move(move, battle: Battle) -> list:
    if move is None:
        return [0.0] * 12

    opp = battle.opponent_active_pokemon
    if opp is not None and move.base_power > 0:
        raw_eff = opp.damage_multiplier(move)
        eff = _EFF_SCALE.get(raw_eff, 0.75)
    else:
        eff = 0.75

    mon = battle.active_pokemon
    stab = 1.0 if (mon is not None and move.type in mon.types) else 0.0
    acc = 1.0 if move.accuracy is True else (move.accuracy / 100.0)
    pp_frac = (move.current_pp / move.max_pp) if move.max_pp else 0.0

    return [
        1.0,
        move.base_power / 150.0,
        eff,
        stab,
        acc,
        move.priority / 7.0,
        pp_frac,
        1.0 if move.category == MoveCategory.PHYSICAL else 0.0,
        1.0 if move.category == MoveCategory.SPECIAL else 0.0,
        1.0 if move.category == MoveCategory.STATUS else 0.0,
        1.0 if "contact" in move.flags else 0.0,
        1.0 if move.priority > 0 else 0.0,
    ]


def _encode_bench_mon(mon, known: bool = True) -> list:
    if mon is None or not known:
        return [0.0] * 28
    feats = [
        1.0,
        mon.current_hp_fraction,
        1.0 if mon.fainted else 0.0,
    ]
    feats += _status_vec(mon.status)
    feats += _type_vec(mon.type_1)
    return feats


def _hazard_vec(sc: dict) -> list:
    sr = 1.0 if SideCondition.STEALTH_ROCK in sc else 0.0
    spikes = sc.get(SideCondition.SPIKES, 0) / 3.0
    tspikes = sc.get(SideCondition.TOXIC_SPIKES, 0) / 2.0
    sweb = 1.0 if SideCondition.STICKY_WEB in sc else 0.0
    return [sr, spikes, tspikes, sweb]


def _screen_vec(sc: dict) -> list:
    return [
        1.0 if SideCondition.REFLECT in sc else 0.0,
        1.0 if SideCondition.LIGHT_SCREEN in sc else 0.0,
        1.0 if SideCondition.AURORA_VEIL in sc else 0.0,
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_battle(battle: Battle) -> np.ndarray:
    """Return a float32 feature vector of length FEATURE_DIM (451).

    See module docstring for the full schema.
    """
    feats: list = []

    # Active mons (49 + 49 = 98).
    feats += _encode_active_mon(battle.active_pokemon)
    feats += _encode_active_mon(battle.opponent_active_pokemon)

    # Available moves (4 × 12 = 48).
    moves = battle.available_moves
    for i in range(4):
        feats += _encode_move(moves[i] if i < len(moves) else None, battle)

    # Weather (6 one-hot).
    current_weather = next(iter(battle.weather), None) if battle.weather else None
    weather_vec = [0.0] * 6
    if current_weather in _WEATHER_ORDER:
        weather_vec[_WEATHER_ORDER.index(current_weather) + 1] = 1.0
    else:
        weather_vec[0] = 1.0
    feats += weather_vec

    # Terrain (5 one-hot).
    terrain_vec = [0.0] * 5
    active_terrain = next(
        (f for f in _TERRAIN_ORDER if f in battle.fields), None
    )
    if active_terrain is not None:
        terrain_vec[_TERRAIN_ORDER.index(active_terrain) + 1] = 1.0
    else:
        terrain_vec[0] = 1.0
    feats += terrain_vec

    # Side conditions (8 hazard + 6 screen = 14 total).
    feats += _hazard_vec(battle.side_conditions)
    feats += _hazard_vec(battle.opponent_side_conditions)
    feats += _screen_vec(battle.side_conditions)
    feats += _screen_vec(battle.opponent_side_conditions)

    # Our bench — 5 non-active mons (5 × 28 = 140).
    active = battle.active_pokemon
    our_bench = [m for m in battle.team.values() if m is not active][:5]
    for i in range(5):
        feats += _encode_bench_mon(our_bench[i] if i < len(our_bench) else None)

    # Opponent bench — revealed non-active mons + unknown slots (5 × 28 = 140).
    opp_active = battle.opponent_active_pokemon
    opp_bench = [m for m in battle.opponent_team.values() if m is not opp_active][:5]
    for i in range(5):
        known = i < len(opp_bench)
        feats += _encode_bench_mon(opp_bench[i] if known else None, known=known)

    vec = np.array(feats, dtype=np.float32)
    assert vec.shape == (FEATURE_DIM,), f"Expected {FEATURE_DIM} features, got {vec.shape[0]}"
    return vec
