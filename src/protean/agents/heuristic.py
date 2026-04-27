"""Phase 1 stub: max-damage heuristic bot."""
from poke_env.player.player import Player
from poke_env.battle.battle import Battle


class MaxDamagePlayer(Player):
    """Always picks the available move with the highest effective base power.

    STAB (same-type attack bonus) is applied as a 1.5× multiplier.
    Falls back to a random move if no damaging moves are available.
    """

    def choose_move(self, battle: Battle):
        if battle.available_moves:
            best = max(
                battle.available_moves,
                key=lambda m: m.base_power * (
                    1.5 if m.type in battle.active_pokemon.types else 1.0
                ),
            )
            return self.create_order(best)
        return self.choose_random_move(battle)
