"""
RL environment wrapper for Gen1OU self-play via poke-env.

Gen1OUPlayer
  Subclasses poke-env's Player. Overrides choose_move() to:
    1. Convert the live Battle object to our obs format
    2. Run the policy model to select an action
    3. Translate the slot index back to a Showdown move/switch order
    4. Record the (obs, action, log_prob, value, reward) transition

The rollout buffer is filled across multiple concurrent battles and drained
by the PPO training loop.

Usage:
    player = Gen1OUPlayer(model=model, format_stats=stats, device=device)
    opponent = Gen1OUPlayer(model=opponent_model, format_stats=stats, device=device)
    await player.battle_against(opponent, n_battles=N)
    transitions = player.drain_buffer()
"""
from __future__ import annotations

import asyncio
import copy
import json
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from poke_env.environment import Battle, Move, Pokemon
from poke_env.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration

# ---------------------------------------------------------------------------
# poke-env 0.8.x compatibility patch
# to_id_str() crashes on None ability (gen1 has no abilities). Patch it to
# return "" for None before any poke-env code runs.
# ---------------------------------------------------------------------------
import poke_env.data.normalize as _poke_norm
import poke_env.environment.pokemon as _poke_pk

_orig_to_id_str = _poke_norm.to_id_str

def _safe_to_id_str(name):  # type: ignore[override]
    if name is None:
        return ""
    return _orig_to_id_str(name)

_poke_norm.to_id_str = _safe_to_id_str
_poke_pk.to_id_str   = _safe_to_id_str   # patch the already-imported ref

from protean.obs_space import (
    Gen1OUObservationSpace, Gen1ActionSpace,
    _sorted_moves, _clean, _build_obs, _norm_conditions,
    N_MOVE_SLOTS, N_SWITCH_SLOTS,
)
from protean.tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
# Server configuration helper
# ---------------------------------------------------------------------------

from poke_env import ServerConfiguration, ShowdownServerConfiguration

LOCAL_SERVER = ServerConfiguration(
    websocket_url="ws://localhost:8001/showdown/websocket",
    authentication_url="https://play.pokemonshowdown.com/action.php?",
)

SHOWDOWN_SERVER = ShowdownServerConfiguration

# ---------------------------------------------------------------------------
# Transition dataclass
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """One step of experience from a single agent in a single battle."""
    tokens:     np.ndarray   # int32 (T,)
    numbers:    np.ndarray   # float32 (48,)
    action_mask: np.ndarray  # bool (9,)
    action:     int          # chosen slot index
    log_prob:   float        # log π(a|s) at time of action
    value:      float        # V(s) at time of action
    reward:     float        # shaped reward for this step
    done:       bool         # True on the final step of the battle


# ---------------------------------------------------------------------------
# Observation bridge: poke-env Battle → our obs format
# ---------------------------------------------------------------------------

_obs_space    = Gen1OUObservationSpace()
_action_space = Gen1ActionSpace()

_STATUS_MAP = {
    "brn": "brn",
    "par": "par",
    "slp": "slp",
    "frz": "frz",
    "psn": "psn",
    "tox": "tox",
    None:  "",
}


def _poke_status(pokemon: Pokemon) -> str:
    if pokemon.status is None:
        return ""
    return _STATUS_MAP.get(pokemon.status.name.lower(), pokemon.status.name.lower())


def _poke_boosts(pokemon: Pokemon) -> dict[str, int]:
    # poke-env stores boosts as a dict keyed by stat name strings
    return dict(pokemon.boosts) if hasattr(pokemon, "boosts") else {}


def _pk_to_bench_dict(pokemon: Pokemon) -> dict:
    return {
        "species":        _clean(pokemon.species),
        "hp":             float(pokemon.current_hp_fraction),
        "status":         _poke_status(pokemon),
        "fainted":        pokemon.fainted,
        "revealed_moves": [_clean(m) for m in pokemon.moves],
    }


def _known_moves(battle: Battle) -> list[str]:
    """
    Return all currently known move IDs for the active pokemon.

    poke-env populates battle.active_pokemon.moves only as moves are *used*,
    so on turn 1 it may be empty. battle.available_moves always contains the
    moves available *this turn* from the server's |request| message.
    Union the two so we always have the full move set once all 4 are revealed.
    """
    active = battle.active_pokemon
    known: set[str] = set(active.moves.keys()) if active else set()
    if not battle.force_switch:
        known |= {_clean(m.id) for m in battle.available_moves}
    return _sorted_moves(list(known))


def battle_to_obs(
    battle: Battle,
    prev_my_move:  str = "",
    prev_opp_move: str = "",
) -> dict[str, np.ndarray]:
    """
    Convert a live poke-env Battle to our observation format.

    Own team: all moves are known (it's our team) — no inference needed.
    Opponent: only revealed information is used (no inference during play;
    the obs format handles unknown moves gracefully via <blank> tokens).
    """
    my_active  = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon

    # --- my active ---
    my_species  = _clean(my_active.species) if my_active else "missingno"
    my_hp       = float(my_active.current_hp_fraction) if my_active else 1.0
    my_status   = _poke_status(my_active) if my_active else ""
    my_boosts   = _poke_boosts(my_active) if my_active else {}
    # Union observed moves with available_moves so turn-1 moves are always visible
    my_moves    = _known_moves(battle)

    # --- my bench ---
    my_bench = [
        _pk_to_bench_dict(p)
        for p in battle.team.values()
        if not p.active and not p.fainted
    ]

    # --- opponent active ---
    opp_species = _clean(opp_active.species) if opp_active else "missingno"
    opp_hp      = float(opp_active.current_hp_fraction) if opp_active else 1.0
    opp_status  = _poke_status(opp_active) if opp_active else ""
    opp_boosts  = _poke_boosts(opp_active) if opp_active else {}

    # --- opponent bench (revealed only) ---
    opp_bench = [
        _pk_to_bench_dict(p)
        for p in battle.opponent_team.values()
        if not p.active and not p.fainted
    ]

    opp_remaining = sum(
        1 for p in battle.opponent_team.values() if not p.fainted
    )

    # --- field ---
    # battle.weather is a Weather enum or None
    weather   = battle.weather.name.lower() if battle.weather else ""
    # side_conditions is a dict {SideCondition: int}
    my_conds  = json.dumps([c.name.lower() for c in battle.side_conditions])
    opp_conds = json.dumps([c.name.lower() for c in battle.opponent_side_conditions])

    # --- forced switch ---
    forced = battle.force_switch

    return _build_obs(
        my_active_species=my_species,
        my_active_hp=my_hp,
        my_active_status=my_status,
        my_active_boosts=my_boosts,
        my_active_moves=my_moves,
        my_bench=my_bench,
        opp_active_species=opp_species,
        opp_active_hp=opp_hp,
        opp_active_status=opp_status,
        opp_active_boosts=opp_boosts,
        opp_bench=opp_bench,
        opp_remaining=opp_remaining,
        weather=weather,
        my_conditions=_norm_conditions(my_conds),
        opp_conditions=_norm_conditions(opp_conds),
        prev_my_move=prev_my_move,
        prev_opp_move=prev_opp_move,
        forced_switch=forced,
    )


def battle_to_action_mask(battle: Battle) -> np.ndarray:
    """
    Build the 9-slot boolean action mask from the live battle state.
    Slots 0-3: available moves (alphabetically ordered against full moveset).
    Slots 4-8: available switches.
    """
    mask = np.zeros(9, dtype=bool)

    if battle.force_switch:
        for i, _ in enumerate(battle.available_switches[:N_SWITCH_SLOTS]):
            mask[N_MOVE_SLOTS + i] = True
        return mask

    # Move slots — map available_moves back to their alphabetical slot index.
    # Use _known_moves() which unions active_pokemon.moves with available_moves
    # so slot assignment is correct even on turn 1 when no moves have been used.
    all_moves_sorted = _known_moves(battle)
    available_ids    = {_clean(m.id) for m in battle.available_moves}
    for i, m in enumerate(all_moves_sorted[:N_MOVE_SLOTS]):
        if _clean(m) in available_ids:
            mask[i] = True

    # Switch slots
    for i, _ in enumerate(battle.available_switches[:N_SWITCH_SLOTS]):
        mask[N_MOVE_SLOTS + i] = True

    # Emergency fallback — should never be needed
    if not mask.any():
        mask[0] = True

    return mask


def action_idx_to_order(idx: int, battle: Battle):
    """
    Convert a 9-slot action index to a poke-env BattleOrder.
    Move slots 0-3 map to the active pokemon's alphabetically-sorted moves.
    Switch slots 4-8 map to available_switches in the order poke-env provides.
    """

    if idx < N_MOVE_SLOTS:
        sorted_moves = _known_moves(battle)
        if idx < len(sorted_moves):
            move_id = sorted_moves[idx]
            for m in battle.available_moves:
                if _clean(m.id) == _clean(move_id):
                    return Player.create_order(m)
        # Fallback — shouldn't happen if mask is correct
        if battle.available_moves:
            return Player.create_order(battle.available_moves[0])
        if battle.available_switches:
            return Player.create_order(battle.available_switches[0])

    else:
        switch_idx = idx - N_MOVE_SLOTS
        switches = battle.available_switches
        if switch_idx < len(switches):
            return Player.create_order(switches[switch_idx])
        if switches:
            return Player.create_order(switches[0])

    return Player.choose_default_move(battle)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

@dataclass
class BattleState:
    """Minimal snapshot of relevant state for reward computation."""
    my_hp_total:   float   # sum of HP fractions across my team
    opp_hp_total:  float   # sum of HP fractions across opponent team
    my_fainted:    int     # number of my fainted mons
    opp_fainted:   int     # number of opponent fainted mons
    my_status:     int     # number of my mons with a status
    opp_status:    int     # number of opponent mons with a status


def _extract_state(battle: Battle) -> BattleState:
    my_team  = list(battle.team.values())
    opp_team = list(battle.opponent_team.values())
    return BattleState(
        my_hp_total  = sum(p.current_hp_fraction for p in my_team),
        opp_hp_total = sum(p.current_hp_fraction for p in opp_team),
        my_fainted   = sum(1 for p in my_team  if p.fainted),
        opp_fainted  = sum(1 for p in opp_team if p.fainted),
        my_status    = sum(1 for p in my_team  if p.status is not None),
        opp_status   = sum(1 for p in opp_team if p.status is not None),
    )


def compute_reward(
    prev: BattleState,
    curr: BattleState,
    won:  Optional[bool],   # None if battle not yet finished
) -> float:
    """
    Dense shaped reward faithful to the metamon paper (Appendix E.1).
    All shaping terms are scaled to O(0.01)/turn so GAE returns stay in [-1.5, +1.5].

    Spec:
        -0.005                          (per-step cost — discourages stalling)
        0.01 * net_hp                   (net HP differential: Δmy_hp − Δopp_hp)
      + 0.005 * (gave_status − took_status)
      + 0.01  * (KOs_dealt − KOs_taken)
      + 1.0   * victory   (terminal, undiscounted)

    net_hp = hp_dealt + hp_gained = (prev.opp − curr.opp) + (curr.my − prev.my)
           = Δmy_hp − Δopp_hp  — mirrors metamon's r_hp term.

    Key property: in a stall mirror where both sides spam recovery equally,
    net_hp ≈ 0 every turn, so the -0.002 step penalty dominates and the policy
    is pushed toward faster resolutions.  Raw healing alone (opponent idle) still
    gives a positive signal, which is desirable — it rewards preserving your team.

    The turn penalty sums to -0.5 over a 100-turn game (small vs ±1 terminal),
    but to -5.0 over a 1000-turn stall. A 200-turn stall costs -1.0 — equal to
    losing outright, creating a strong deterrent against passive play.
    """
    hp_dealt    = prev.opp_hp_total - curr.opp_hp_total  # positive = good
    hp_gained   = curr.my_hp_total  - prev.my_hp_total   # positive = good (healing)
    gave_status = curr.opp_status   - prev.opp_status
    took_status = curr.my_status    - prev.my_status
    kos_dealt   = curr.opp_fainted  - prev.opp_fainted
    kos_taken   = curr.my_fainted   - prev.my_fainted

    r = (-0.005                                           # per-step stall penalty
       + 0.01 * (hp_dealt + hp_gained)                   # net HP differential
       + 0.005 * (gave_status - took_status)
       + 0.01  * (kos_dealt  - kos_taken))

    if won is True:
        r += 1.0
    elif won is False:
        r -= 1.0

    return float(r)


# ---------------------------------------------------------------------------
# Gen1OUPlayer
# ---------------------------------------------------------------------------

class Gen1OUPlayer(Player):
    """
    poke-env Player that uses Gen1OUPolicy to choose moves.

    Collects (obs, action, log_prob, value, reward, done) transitions into a
    buffer that the PPO loop drains after each rollout.
    """

    def __init__(
        self,
        model:                torch.nn.Module,
        device:               torch.device,
        username:             str  = "ProteanBot",
        password:             str | None = None,
        sample:               bool = True,
        verbose:              bool = False,
        battle_format:        str  = "gen1ou",
        team:                 str | None = None,
        server_configuration: ServerConfiguration = None,
        **kwargs,
    ):
        # ALL instance attributes must be set before super().__init__() because
        # poke-env's PSClient starts the POKE_LOOP listener (a separate thread) at
        # line 123 of Player.__init__ — before Player.__init__ sets any of its own
        # attributes at line 149+. If a battle message arrives in that window,
        # choose_move or _battle_finished_callback will be called on a half-initialised
        # object and raise AttributeError.
        self._team = None

        self.model   = model
        self.device  = device
        self.sample  = sample
        self.verbose = verbose
        self._tokenizer = get_tokenizer()
        self._server_cfg = server_configuration or LOCAL_SERVER

        # Per-battle state tracking
        self._prev_state:     dict[str, BattleState] = {}
        self._prev_my_move:   dict[str, str]          = {}
        self._prev_opp_move:  dict[str, str]          = {}
        # Set of opp move IDs seen so far; used to detect newly revealed moves
        self._opp_moves_seen: dict[str, set[str]]     = {}
        self._pending:        dict[str, Transition]   = {}  # step waiting for next reward

        # Completed transitions ready for PPO
        # _lock guards _buffer and _pending across the main thread (drain_buffer)
        # and the POKE_LOOP background thread (choose_move / _battle_finished_callback).
        # Must be a threading.Lock, not asyncio.Lock — they run on different event loops.
        self._buffer: list[Transition] = []
        self._lock = threading.Lock()

        account_cfg = AccountConfiguration(username, password)
        super().__init__(
            account_configuration=account_cfg,
            server_configuration=self._server_cfg,
            battle_format=battle_format,
            max_concurrent_battles=1,
            team=team,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core poke-env interface
    # ------------------------------------------------------------------

    def choose_move(self, battle: Battle):
        """Called by poke-env each time this agent needs to pick an action."""
        battle_id = battle.battle_tag

        prev_my_move  = self._prev_my_move.get(battle_id, "")
        prev_opp_move = self._prev_opp_move.get(battle_id, "")

        # Build observation
        obs  = battle_to_obs(battle, prev_my_move, prev_opp_move)
        mask = battle_to_action_mask(battle)

        token_ids = self._tokenizer.tokenize(str(obs["text"]))
        tokens  = torch.from_numpy(token_ids).long().unsqueeze(0).to(self.device)
        numbers = torch.from_numpy(obs["numbers"]).float().unsqueeze(0).to(self.device)
        amask   = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.verbose:
                log_probs_t, value_t = self.model(tokens, numbers, amask)
                probs_np = log_probs_t[0].exp().cpu().numpy()
                if self.sample:
                    from torch.distributions import Categorical
                    dist   = Categorical(logits=log_probs_t)
                    act_t  = dist.sample()[0]
                    log_prob  = dist.log_prob(act_t).item()
                    action_idx = int(act_t.item())
                else:
                    action_idx = int(log_probs_t[0].argmax().item())
                    log_prob   = log_probs_t[0, action_idx].item()
                value = float(value_t[0, 0].item())
            else:
                action_idx, log_prob, value = self.model.act(
                    tokens, numbers, amask, sample=self.sample
                )
                probs_np = None

        # Decode action → poke-env BattleOrder (needed to record move name below)
        order = action_idx_to_order(action_idx, battle)

        # Update prev-move tracking for next turn's obs context.
        # My move: decode the chosen slot back to a move/switch name.
        if action_idx < N_MOVE_SLOTS:
            sorted_moves = _known_moves(battle)
            if action_idx < len(sorted_moves):
                self._prev_my_move[battle_id] = sorted_moves[action_idx]
        else:
            switch_idx = action_idx - N_MOVE_SLOTS
            switches = battle.available_switches
            if switch_idx < len(switches):
                self._prev_my_move[battle_id] = _clean(switches[switch_idx].species)

        # Opponent's last move: poke-env doesn't expose this directly.
        # Detect it by comparing the opp's current revealed moves against the set we
        # saw last turn — any newly appeared move was the one just used.
        opp = battle.opponent_active_pokemon
        if opp is not None:
            current_opp_moves = set(opp.moves.keys())
            prev_seen = self._opp_moves_seen.get(battle_id, set())
            new_moves = current_opp_moves - prev_seen
            if new_moves:
                # Pick the newly revealed move (usually exactly one)
                self._prev_opp_move[battle_id] = _clean(next(iter(new_moves)))
            self._opp_moves_seen[battle_id] = current_opp_moves

        if self.verbose and probs_np is not None:
            self._log_decision(battle, action_idx, probs_np, mask)

        # Close out the previous step's transition now that we have a new state
        curr_state = _extract_state(battle)
        self._close_step(battle_id, curr_state, won=None)

        # Store pending transition (reward filled on next step or episode end)
        with self._lock:
            self._pending[battle_id] = Transition(
                tokens=token_ids,
                numbers=obs["numbers"],
                action_mask=mask,
                action=action_idx,
                log_prob=log_prob,
                value=value,
                reward=0.0,    # filled in next call or _on_battle_finished
                done=False,
            )
        self._prev_state[battle_id] = curr_state

        return order

    def _battle_finished_callback(self, battle: Battle) -> None:
        """Called by poke-env when a battle ends."""
        battle_id  = battle.battle_tag
        won        = battle.won
        curr_state = _extract_state(battle)
        self._close_step(battle_id, curr_state, won=won, done=True)
        # Clean up per-battle state (not lock-protected: only ever touched on POKE_LOOP)
        self._prev_state.pop(battle_id, None)
        self._prev_my_move.pop(battle_id, None)
        self._prev_opp_move.pop(battle_id, None)
        self._opp_moves_seen.pop(battle_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_decision(
        self, battle: Battle, chosen: int, probs: np.ndarray, mask: np.ndarray
    ) -> None:
        sorted_moves = _known_moves(battle)
        switches     = battle.available_switches

        names: list[str] = []
        for i in range(N_MOVE_SLOTS):
            names.append(sorted_moves[i] if i < len(sorted_moves) else f"move{i}")
        for i in range(N_SWITCH_SLOTS):
            names.append(f">{_clean(switches[i].species)}" if i < len(switches) else f"sw{i}")

        valid = [(names[i], probs[i], i == chosen) for i in range(9) if mask[i]]
        valid.sort(key=lambda x: -x[1])

        parts = [f"{'▶ ' if c else ''}{n}({p:.0%})" for n, p, c in valid]
        print(f"  Bot: {'  '.join(parts)}")

    def _close_step(
        self,
        battle_id: str,
        curr_state: BattleState,
        won: Optional[bool],
        done: bool = False,
    ) -> None:
        """Fill in reward for the pending transition and move it to the buffer."""
        with self._lock:
            pending = self._pending.pop(battle_id, None)
        if pending is None:
            return
        prev_state = self._prev_state.get(battle_id)
        if prev_state is not None:
            reward = compute_reward(prev_state, curr_state, won if done else None)
        else:
            reward = 0.0
        pending.reward = reward
        pending.done   = done
        with self._lock:
            self._buffer.append(pending)

    def drain_buffer(self) -> list[Transition]:
        """Return and clear all completed transitions. Thread-safe."""
        with self._lock:
            buf, self._buffer = self._buffer, []
        return buf

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# HumanPlayer — interactive terminal player
# ---------------------------------------------------------------------------

def _dn(s: str) -> str:
    """Display name: bodyslam → Body Slam."""
    return s.replace("-", " ").replace("_", " ").title()


def _fmt_boosts(boosts: dict) -> str:
    parts = [f"{'+'if v>0 else ''}{v} {k}" for k, v in boosts.items() if v != 0]
    return f"  [{', '.join(parts)}]" if parts else ""


def _human_print_state(battle: Battle) -> None:
    print(f"\n{'─'*44}")
    print(f"Turn {battle.turn}")

    opp = battle.opponent_active_pokemon
    me  = battle.active_pokemon

    if opp:
        opp_hp  = f"{opp.current_hp_fraction:.0%}"
        opp_st  = f"  [{opp.status.name}]" if opp.status else ""
        opp_bst = _fmt_boosts(dict(opp.boosts)) if hasattr(opp, "boosts") else ""
        print(f"Opp:  {_dn(opp.species):<14} {opp_hp}{opp_st}{opp_bst}")
        opp_bench = [p for p in battle.opponent_team.values() if not p.active and not p.fainted]
        if opp_bench:
            bench_str = "  ".join(f"{_dn(p.species)} {p.current_hp_fraction:.0%}" for p in opp_bench)
            print(f"      bench: {bench_str}")

    print()

    if me:
        me_hp  = f"{me.current_hp_fraction:.0%}"
        me_st  = f"  [{me.status.name}]" if me.status else ""
        me_bst = _fmt_boosts(dict(me.boosts)) if hasattr(me, "boosts") else ""
        print(f"You:  {_dn(me.species):<14} {me_hp}{me_st}{me_bst}")

    print()

    n = 1
    if not battle.force_switch and battle.available_moves:
        for m in battle.available_moves:
            print(f"  {n}. {_dn(m.id)}")
            n += 1
        print()

    for sw in battle.available_switches:
        print(f"  {n}. → {_dn(sw.species):<16} {sw.current_hp_fraction:.0%}")
        n += 1

    print()


def _human_build_options(battle: Battle) -> list:
    orders = []
    if not battle.force_switch:
        for m in battle.available_moves:
            orders.append(Player.create_order(m))
    for sw in battle.available_switches:
        orders.append(Player.create_order(sw))
    return orders


def _human_read_choice(options: list):
    n = len(options)
    while True:
        try:
            raw = input(f"Choice (1-{n}): ").strip()
            idx = int(raw) - 1
            if 0 <= idx < n:
                return options[idx]
            print(f"  Enter a number from 1 to {n}.")
        except ValueError:
            print(f"  Enter a number from 1 to {n}.")
        except EOFError:
            return options[0]


class HumanPlayer(Player):
    """
    Interactive player: prints game state to stdout, reads choice from stdin.

    choose_move blocks on input() which freezes the asyncio event loop — fine
    for a single local battle, but don't use this in concurrent/training contexts.

    Future: replace with a WebSocket bridge so the human plays through the
    Showdown browser UI at http://localhost:8001.
    """

    def __init__(self, username: str = "Human", team: str | None = None):
        super().__init__(
            account_configuration=AccountConfiguration(username, None),
            server_configuration=LOCAL_SERVER,
            battle_format="gen1ou",
            max_concurrent_battles=1,
            team=team,
        )

    def choose_move(self, battle: Battle):
        _human_print_state(battle)
        options = _human_build_options(battle)
        return _human_read_choice(options)
