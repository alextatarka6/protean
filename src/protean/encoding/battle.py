"""Phase 1 stub: battle state → flat feature vector."""
import numpy as np
from poke_env.battle.battle import Battle


# Current feature dimension — will grow as encoding gets richer.
FEATURE_DIM = 6


def encode_battle(battle: Battle) -> np.ndarray:
    """Return a float32 vector representing the current battle state.

    Features (all normalised to roughly [0, 1]):
      0  — our active Pokémon HP fraction
      1  — opponent active Pokémon HP fraction
      2–5 — base power of available moves 0–3, divided by 150
    """
    features: list[float] = []

    mon = battle.active_pokemon
    features.append(mon.current_hp_fraction if mon else 0.0)

    opp = battle.opponent_active_pokemon
    features.append(opp.current_hp_fraction if opp else 0.0)

    for i in range(4):
        if i < len(battle.available_moves):
            features.append(battle.available_moves[i].base_power / 150.0)
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)
