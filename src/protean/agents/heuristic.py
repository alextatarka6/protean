"""Phase 1: heuristic player with type-effectiveness scoring and switch logic."""
from poke_env.player.player import Player
from poke_env.battle.battle import Battle


def _move_score(move, battle: Battle) -> float:
    """Score a move by expected effective damage.

    Formula: base_power × type_effectiveness × STAB × accuracy.
    Returns 0 for status moves (base_power == 0).
    """
    if move.base_power == 0:
        return 0.0

    opp = battle.opponent_active_pokemon
    eff = opp.damage_multiplier(move) if opp is not None else 1.0
    stab = 1.5 if move.type in battle.active_pokemon.types else 1.0
    acc = 1.0 if move.accuracy is True else (move.accuracy / 100.0)
    return move.base_power * eff * stab * acc


def _best_switch(battle: Battle):
    """Return the available switch with the best defensive typing vs the opponent.

    Scores each candidate by HP fraction divided by the worst incoming type
    multiplier from the opponent's STAB types.
    """
    opp = battle.opponent_active_pokemon
    if not battle.available_switches or opp is None:
        return None

    def score(mon):
        if mon.fainted:
            return -1.0
        worst_incoming = max(
            (mon.damage_multiplier(t) for t in opp.types if t is not None),
            default=1.0,
        )
        return mon.current_hp_fraction / max(worst_incoming, 0.001)

    return max(battle.available_switches, key=score)


class HeuristicPlayer(Player):
    """Heuristic bot that scores moves by effective damage and switches on bad matchups.

    Switching triggers:
      - ×4 type disadvantage from any of the opponent's types, or
      - below 25 % HP with a ×2+ type disadvantage.
    """

    def choose_move(self, battle: Battle):
        mon = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        max_incoming = 1.0
        if opp is not None:
            max_incoming = max(
                (mon.damage_multiplier(t) for t in opp.types if t is not None),
                default=1.0,
            )

        should_switch = battle.available_switches and (
            max_incoming >= 4.0
            or (mon.current_hp_fraction < 0.25 and max_incoming >= 2.0)
        )

        if should_switch:
            target = _best_switch(battle)
            if target is not None:
                return self.create_order(target)

        if battle.available_moves:
            best = max(battle.available_moves, key=lambda m: _move_score(m, battle))
            if _move_score(best, battle) > 0:
                return self.create_order(best)

        return self.choose_random_move(battle)


class MaxDamagePlayer(Player):
    """Baseline: picks the move with the highest base power × STAB multiplier.

    No type-effectiveness or switching logic. Used as the Phase 1 benchmark.
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
