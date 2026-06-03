"""
Parse raw Pokémon Showdown spectator logs into turn-by-turn sequences.

Produces TurnSnapshot objects containing:
  - full revealed state for both sides at the START of each turn
  - the action each player took that turn (move name or switch target)

Scoped to Gen 1–4 singles (no Terastallization, no Dynamax).
Returns None for incomplete battles, ties with no winner, or unsupported formats.
"""
from __future__ import annotations

import copy
import re
from typing import Optional

from .types import (
    Action,
    BattlePokemon,
    FieldState,
    ParsedBattle,
    SideState,
    TurnSnapshot,
    Winner,
)

_SUPPORTED_FORMAT = re.compile(
    r"^gen[1-4](ou|ubers|uu|ru|nu|pu|lc|monotype|1v1|randombattle|doublesou|anythinggoes)$"
)

_BOOST_STATS = {"atk", "def", "spa", "spd", "spe", "accuracy", "evasion"}

_STATUSES = {"par", "brn", "slp", "frz", "psn", "tox"}

# Moves that call another move where the called move is NOT in the pokemon's real moveset.
# When we see |move| with [from] pointing to one of these, don't reveal the called move.
_MOVE_OVERRIDE = {
    "Metronome", "Copycat", "Assist", "Mirror Move",
    "Nature Power", "Snatch", "Magic Coat",
}

# Sleep Talk calls a move that IS in the pokemon's real moveset — still reveal it.
_MOVE_OVERRIDE_BUT_REVEAL_ANYWAY = {"Sleep Talk"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _supported(format_id: str) -> bool:
    return bool(_SUPPORTED_FORMAT.match(format_id.lower()))


def _parse_hp(hp_str: str) -> Optional[float]:
    """'75/100', '0 fnt', '100/100 par' → fraction 0.0–1.0. None if unparseable."""
    hp_str = hp_str.strip()
    if not hp_str or hp_str == "0 fnt":
        return 0.0
    token = hp_str.split()[0]
    if "/" in token:
        cur, mx = token.split("/", 1)
        try:
            return float(cur) / float(mx)
        except ValueError:
            return None
    return None


def _parse_status_from_hp(hp_str: str) -> Optional[str]:
    """Extract trailing status token from HP string, e.g. '75/100 par' → 'par'."""
    parts = hp_str.strip().split()
    if len(parts) > 1 and parts[-1] in _STATUSES:
        return parts[-1]
    return None


def _parse_ident(ident: str) -> tuple[str, str]:
    """'p1a: Nickname' → ('p1', 'Nickname').  Returns ('', raw) on failure."""
    m = re.match(r"(p[12])[a-z]:\s*(.+)", ident.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return "", ident.strip()


def _parse_detail(detail: str) -> tuple[str, int]:
    """'Gengar, L50, M' → ('Gengar', 50).  Level defaults to 100."""
    parts = [p.strip() for p in detail.split(",")]
    species = parts[0]
    level = 100
    for p in parts[1:]:
        if p.startswith("L"):
            try:
                level = int(p[1:])
            except ValueError:
                pass
    return species, level


def _blank_boosts() -> dict[str, int]:
    return {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}


def _parse_from_tag(args: list[str]) -> Optional[str]:
    """Extract the calling-move name from a [from] tag in move args."""
    for tag in args:
        tag = tag.strip()
        if tag.startswith("[from] move: "):
            return tag[len("[from] move: "):]
        if tag.startswith("[from] "):
            return tag[len("[from] "):]
    return None


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_battle(battle_id: str, log: str, format_id: str) -> Optional[ParsedBattle]:
    """
    Parse a raw Showdown spectator log into a ParsedBattle.
    Returns None if the battle is incomplete, unsupported, or has no clear winner.
    """
    if not _supported(format_id):
        return None

    gen = int(format_id[3])

    p1 = SideState(player="")
    p2 = SideState(player="")
    field = FieldState()

    p1_name: Optional[str] = None
    p2_name: Optional[str] = None
    p1_rating: Optional[int] = None
    p2_rating: Optional[int] = None
    winner: Optional[Winner] = None
    battle_started = False
    has_species_clause = False

    turns: list[TurnSnapshot] = []
    current_turn = 0

    # Per-turn action tracking
    p1_action: Optional[Action] = None
    p2_action: Optional[Action] = None

    # After a faint the next switch-in for that side is forced (not a chosen action)
    p1_must_switch = False
    p2_must_switch = False

    # Recharge state: queued means |-mustrecharge| was seen this turn (next turn is recharge).
    # is_recharge_turn is set at the |turn| boundary for the upcoming turn.
    p1_queued_recharge = False
    p2_queued_recharge = False
    p1_is_recharge_turn = False
    p2_is_recharge_turn = False

    # Snapshot of state at the start of the current turn
    snap_p1: Optional[SideState] = None
    snap_p2: Optional[SideState] = None
    snap_field: Optional[FieldState] = None

    def side(pid: str) -> SideState:
        return p1 if pid == "p1" else p2

    def find_or_add(s: SideState, nickname: str, species: str, level: int) -> BattlePokemon:
        # Exact nickname match
        for pk in s.team:
            if pk.nickname == nickname:
                return pk
        # Team-preview entry used species as placeholder nickname — update it
        for pk in s.team:
            if pk.species == species and pk.nickname == pk.species:
                pk.nickname = nickname
                return pk
        pk = BattlePokemon(species=species, nickname=nickname, level=level, hp=1.0)
        s.team.append(pk)
        return pk

    def snapshot() -> tuple[SideState, SideState, FieldState]:
        return copy.deepcopy(p1), copy.deepcopy(p2), copy.deepcopy(field)

    def flush_turn() -> None:
        """Save the completed turn and reset per-turn state."""
        nonlocal p1_action, p2_action, p1_must_switch, p2_must_switch
        nonlocal p1_queued_recharge, p2_queued_recharge
        nonlocal p1_is_recharge_turn, p2_is_recharge_turn
        if current_turn > 0 and snap_p1 is not None:
            # Fill in forced recharge when no other action was recorded
            if p1_is_recharge_turn and p1_action is None:
                p1_action = Action(kind="move", value="recharge", forced=True)
            if p2_is_recharge_turn and p2_action is None:
                p2_action = Action(kind="move", value="recharge", forced=True)
            turns.append(TurnSnapshot(
                turn_number=current_turn,
                p1=snap_p1,
                p2=snap_p2,
                field=snap_field,
                p1_action=p1_action,
                p2_action=p2_action,
            ))
        # Propagate queued recharge → active for the upcoming turn
        p1_is_recharge_turn = p1_queued_recharge
        p2_is_recharge_turn = p2_queued_recharge
        p1_queued_recharge = False
        p2_queued_recharge = False
        p1_action = None
        p2_action = None
        p1_must_switch = False
        p2_must_switch = False

    for raw_line in log.split("\n"):
        if not raw_line.startswith("|"):
            continue
        parts = raw_line.split("|")
        if len(parts) < 2:
            continue
        msg = parts[1]
        args = parts[2:]

        # ------------------------------------------------------------------
        # Battle setup
        # ------------------------------------------------------------------
        if msg == "player":
            # |player|p1|username|avatar|rating
            if len(args) >= 2:
                pid, uname = args[0], args[1]
                rating = int(args[3]) if len(args) >= 4 and args[3].isdigit() else None
                if pid == "p1":
                    p1_name, p1.player, p1_rating = uname, uname, rating
                elif pid == "p2":
                    p2_name, p2.player, p2_rating = uname, uname, rating

        elif msg == "teamsize":
            # |teamsize|p1|6
            if len(args) >= 2:
                try:
                    sz = int(args[1])
                except ValueError:
                    continue
                if sz != 6:
                    return None

        elif msg == "poke":
            # |poke|p1|Charizard, M|item  — team preview
            if len(args) >= 2:
                pid = args[0]
                species, level = _parse_detail(args[1])
                s = side(pid)
                if not any(pk.species == species for pk in s.team):
                    s.team.append(BattlePokemon(
                        species=species, nickname=species, level=level, hp=1.0
                    ))

        elif msg == "rule":
            if args and "Species Clause" in args[0]:
                has_species_clause = True

        elif msg == "start":
            battle_started = True

        # ------------------------------------------------------------------
        # Turn boundary
        # ------------------------------------------------------------------
        elif msg == "turn":
            flush_turn()
            current_turn = int(args[0])
            s1, s2, sf = snapshot()
            snap_p1, snap_p2, snap_field = s1, s2, sf

        # ------------------------------------------------------------------
        # Switches
        # ------------------------------------------------------------------
        elif msg in ("switch", "drag"):
            # |switch|p1a: Nick|Species, L100|75/100
            if len(args) < 3:
                continue
            pid, nickname = _parse_ident(args[0])
            species, level = _parse_detail(args[1])
            hp = _parse_hp(args[2]) or 1.0
            status = _parse_status_from_hp(args[2])

            s = side(pid)
            pk = find_or_add(s, nickname, species, level)
            pk.hp = hp
            pk.status = status
            pk.fainted = hp == 0.0
            pk.seen_in_battle = True

            # Reset boosts on the previously active pokemon when it leaves
            if s.active is not None and s.active is not pk:
                s.active.boosts = _blank_boosts()
            s.active = pk

            # Record as chosen action only for voluntary switches
            is_drag = msg == "drag"
            is_forced = is_drag or (pid == "p1" and p1_must_switch) or (pid == "p2" and p2_must_switch)
            if current_turn > 0:
                action = Action(kind="switch", value=species, forced=is_forced)
                if pid == "p1":
                    if p1_must_switch:
                        p1_must_switch = False
                    elif p1_action is None and not is_drag:
                        p1_action = action
                else:
                    if p2_must_switch:
                        p2_must_switch = False
                    elif p2_action is None and not is_drag:
                        p2_action = action

        # ------------------------------------------------------------------
        # Moves
        # ------------------------------------------------------------------
        elif msg == "move":
            # |move|p1a: Nick|Move Name|target|[from] move: Caller|...
            if len(args) < 2:
                continue
            pid, nickname = _parse_ident(args[0])
            move_name = args[1]
            s = side(pid)

            from_move = _parse_from_tag(args[2:])

            # Don't reveal the called move when it comes from a randomising caller
            if from_move not in _MOVE_OVERRIDE:
                if s.active is not None and move_name not in s.active.revealed_moves:
                    s.active.revealed_moves.append(move_name)

            if current_turn > 0:
                action = Action(kind="move", value=move_name)
                if pid == "p1" and p1_action is None:
                    p1_action = action
                elif pid == "p2" and p2_action is None:
                    p2_action = action

        # ------------------------------------------------------------------
        # Can't act — recover intended move from the move-name arg if present
        # ------------------------------------------------------------------
        elif msg == "cant":
            # |cant|p1a: Nick|reason[|move_name]
            if len(args) < 1:
                continue
            pid, _ = _parse_ident(args[0])
            intended_move = args[2].strip() if len(args) >= 3 else None
            if intended_move and current_turn > 0:
                action = Action(kind="move", value=intended_move)
                if pid == "p1" and p1_action is None:
                    p1_action = action
                elif pid == "p2" and p2_action is None:
                    p2_action = action

        # ------------------------------------------------------------------
        # Recharge (Hyper Beam, Blast Burn, etc.)
        # ------------------------------------------------------------------
        elif msg == "-mustrecharge":
            # Fires on the same turn the recharge move is used; next turn is the no-op.
            if not args:
                continue
            pid, _ = _parse_ident(args[0])
            if pid == "p1":
                p1_queued_recharge = True
            else:
                p2_queued_recharge = True

        # ------------------------------------------------------------------
        # Forme changes
        # ------------------------------------------------------------------
        elif msg in ("detailschange", "-formechange"):
            # |detailschange|p1a: Nick|Shaymin-Sky, M|100/100
            if len(args) < 2:
                continue
            pid, nickname = _parse_ident(args[0])
            new_species, _ = _parse_detail(args[1])
            s = side(pid)
            if s.active and s.active.nickname == nickname:
                s.active.species = new_species

        # ------------------------------------------------------------------
        # HP changes
        # ------------------------------------------------------------------
        elif msg in ("-damage", "-heal", "-sethp"):
            if len(args) < 2:
                continue
            pid, nickname = _parse_ident(args[0])
            hp = _parse_hp(args[1])
            status = _parse_status_from_hp(args[1])
            s = side(pid)
            if s.active is not None and s.active.nickname == nickname:
                if hp is not None:
                    s.active.hp = hp
                if status:
                    s.active.status = status

        # ------------------------------------------------------------------
        # Status
        # ------------------------------------------------------------------
        elif msg == "-status":
            if len(args) < 2:
                continue
            pid, _ = _parse_ident(args[0])
            side(pid).active and setattr(side(pid).active, "status", args[1])

        elif msg in ("-curestatus", "-cureall"):
            if not args:
                continue
            pid, _ = _parse_ident(args[0])
            if side(pid).active:
                side(pid).active.status = None

        # ------------------------------------------------------------------
        # Boosts
        # ------------------------------------------------------------------
        elif msg == "-boost":
            if len(args) < 3:
                continue
            pid, _ = _parse_ident(args[0])
            stat, amt = args[1], _safe_int(args[2])
            pk = side(pid).active
            if pk and stat in _BOOST_STATS:
                pk.boosts[stat] = max(-6, min(6, pk.boosts[stat] + amt))

        elif msg == "-unboost":
            if len(args) < 3:
                continue
            pid, _ = _parse_ident(args[0])
            stat, amt = args[1], _safe_int(args[2])
            pk = side(pid).active
            if pk and stat in _BOOST_STATS:
                pk.boosts[stat] = max(-6, min(6, pk.boosts[stat] - amt))

        elif msg == "-setboost":
            if len(args) < 3:
                continue
            pid, _ = _parse_ident(args[0])
            stat, amt = args[1], _safe_int(args[2])
            pk = side(pid).active
            if pk and stat in _BOOST_STATS:
                pk.boosts[stat] = max(-6, min(6, amt))

        elif msg == "-swapboost":
            # |-swapboost|p1a: Nick|p2a: Nick2|atk,spa,...
            if len(args) < 2:
                continue
            pid1, _ = _parse_ident(args[0])
            pid2, _ = _parse_ident(args[1])
            stats = [s.strip() for s in args[2].split(",")] if len(args) > 2 else list(_BOOST_STATS)
            pk1 = side(pid1).active
            pk2 = side(pid2).active
            if pk1 and pk2:
                for stat in stats:
                    if stat in _BOOST_STATS:
                        pk1.boosts[stat], pk2.boosts[stat] = pk2.boosts[stat], pk1.boosts[stat]

        elif msg == "-copyboost":
            # |-copyboost|p1a: Nick (dst)|p2a: Nick2 (src)|atk,spa,...
            if len(args) < 2:
                continue
            pid_dst, _ = _parse_ident(args[0])
            pid_src, _ = _parse_ident(args[1])
            stats = [s.strip() for s in args[2].split(",")] if len(args) > 2 else list(_BOOST_STATS)
            dst = side(pid_dst).active
            src = side(pid_src).active
            if dst and src:
                for stat in stats:
                    if stat in _BOOST_STATS:
                        dst.boosts[stat] = src.boosts[stat]

        elif msg == "-clearboost":
            if not args:
                continue
            pid, _ = _parse_ident(args[0])
            pk = side(pid).active
            if pk:
                pk.boosts = _blank_boosts()

        elif msg == "-clearallboost":
            for pk in (p1.active, p2.active):
                if pk:
                    pk.boosts = _blank_boosts()

        elif msg == "-clearpositiveboost":
            if not args:
                continue
            pid, _ = _parse_ident(args[0])
            pk = side(pid).active
            if pk:
                pk.boosts = {k: min(0, v) for k, v in pk.boosts.items()}

        elif msg == "-clearnegativeboost":
            if not args:
                continue
            pid, _ = _parse_ident(args[0])
            pk = side(pid).active
            if pk:
                pk.boosts = {k: max(0, v) for k, v in pk.boosts.items()}

        elif msg == "-invertboost":
            if not args:
                continue
            pid, _ = _parse_ident(args[0])
            pk = side(pid).active
            if pk:
                pk.boosts = {k: -v for k, v in pk.boosts.items()}

        elif msg == "-restoreboost":
            # Restores boosts from a prior snapshot (Baton Pass etc.); no-op here
            # since we don't track boost history.
            pass

        # ------------------------------------------------------------------
        # Faint
        # ------------------------------------------------------------------
        elif msg == "faint":
            if not args:
                continue
            pid, nickname = _parse_ident(args[0])
            s = side(pid)
            if s.active:
                s.active.hp = 0.0
                s.active.fainted = True
            # Next switch-in for this side is forced
            if pid == "p1":
                p1_must_switch = True
            else:
                p2_must_switch = True

        # ------------------------------------------------------------------
        # Weather
        # ------------------------------------------------------------------
        elif msg == "-weather":
            if args:
                w = args[0]
                field.weather = None if w in ("none", "RainUpkeep", "") else w
                if len(args) > 1 and args[1] == "[upkeep]":
                    pass  # keep current weather

        # ------------------------------------------------------------------
        # Field conditions (Trick Room, Gravity, etc.)
        # ------------------------------------------------------------------
        elif msg == "-fieldstart":
            if args:
                cond = args[0].removeprefix("move: ").removeprefix("ability: ")
                field.conditions[cond] = 1

        elif msg == "-fieldend":
            if args:
                cond = args[0].removeprefix("move: ").removeprefix("ability: ")
                field.conditions.pop(cond, None)

        # ------------------------------------------------------------------
        # Side conditions (Reflect, Spikes, Stealth Rock, etc.)
        # ------------------------------------------------------------------
        elif msg == "-sidestart":
            if len(args) >= 2:
                pid = args[0].split(":")[0].strip()
                cond = args[1].removeprefix("move: ").removeprefix("ability: ")
                side(pid).conditions[cond] = side(pid).conditions.get(cond, 0) + 1

        elif msg == "-sideend":
            if len(args) >= 2:
                pid = args[0].split(":")[0].strip()
                cond = args[1].removeprefix("move: ").removeprefix("ability: ")
                side(pid).conditions.pop(cond, None)

        # ------------------------------------------------------------------
        # Items / Abilities (revealed during battle)
        # ------------------------------------------------------------------
        elif msg == "-item":
            if len(args) >= 2:
                pid, _ = _parse_ident(args[0])
                pk = side(pid).active
                if pk:
                    pk.item = args[1]

        elif msg == "-enditem":
            if args:
                pid, _ = _parse_ident(args[0])
                pk = side(pid).active
                if pk:
                    pk.item = None

        elif msg == "-ability":
            if len(args) >= 2:
                pid, _ = _parse_ident(args[0])
                pk = side(pid).active
                if pk:
                    pk.ability = args[1]

        # ------------------------------------------------------------------
        # Win / Tie
        # ------------------------------------------------------------------
        elif msg == "win":
            if args:
                name = args[0]
                winner = Winner.P1 if name == p1_name else Winner.P2

        elif msg == "tie":
            winner = Winner.TIE

    # Save final turn
    flush_turn()

    # ------------------------------------------------------------------
    # Completeness checks
    # ------------------------------------------------------------------
    if not battle_started or winner is None or not turns or p1_name is None or p2_name is None:
        return None

    # Minimum meaningful battle length
    if len(turns) < 5:
        return None

    # Species Clause required for our nickname-based lookup assumptions
    if not has_species_clause:
        return None

    # Team preview (players see each other's full roster before picking a lead) was
    # introduced in Gen 5.  Gen 1–4 only reveal pokemon when they are sent into battle,
    # so battles that end early will have fewer than 6 seen per side.  Team inference
    # fills the unrevealed slots later; require only that at least 1 was seen per side.
    has_team_preview = gen >= 5
    if has_team_preview:
        if len(p1.team) != 6 or len(p2.team) != 6:
            return None
    else:
        if len(p1.team) == 0 or len(p2.team) == 0:
            return None

    # No duplicate species within either team (species clause applies to own team)
    p1_species = [pk.species for pk in p1.team]
    p2_species = [pk.species for pk in p2.team]
    if len(p1_species) != len(set(p1_species)) or len(p2_species) != len(set(p2_species)):
        return None

    return ParsedBattle(
        battle_id=battle_id,
        format=format_id,
        gen=gen,
        p1_name=p1_name,
        p2_name=p2_name,
        p1_rating=p1_rating,
        p2_rating=p2_rating,
        turns=turns,
        winner=winner,
    )


def _safe_int(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return 0
