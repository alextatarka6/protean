import re

from protean.backend.replay_parser import iter_parsed_battles, ParsedBattle
from protean.backend.replay_parser.loader import iter_raw_replays
from protean.backend.replay_parser.parser import parse_battle, _supported
from protean.backend.replay_parser.types import TurnSnapshot, BattlePokemon, Winner, POVReplay
from protean.backend.usage_stats import load_format_stats, MovesetStats
from protean.pov import reconstruct_pov, reconstruct_both_povs


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def fmt_pokemon(pk: BattlePokemon | None) -> str:
    if pk is None:
        return "???"
    hp = f"{pk.hp*100:.0f}%"
    status = f" [{pk.status.upper()}]" if pk.status else ""
    boosts = " ".join(
        f"{s[0].upper()}{v:+d}" for s, v in pk.boosts.items() if v != 0
    )
    boosts_str = f" ({boosts})" if boosts else ""
    moves = "/".join(pk.revealed_moves) if pk.revealed_moves else "—"
    return f"{pk.species} Lv{pk.level} {hp}{status}{boosts_str}  moves:[{moves}]"


def fmt_action(action) -> str:
    if action is None:
        return "unknown"
    tag = "USE" if action.kind == "move" else "SWITCH →"
    forced = " [forced]" if getattr(action, "forced", False) else ""
    return f"{tag} {action.value}{forced}"


def print_battle(battle: ParsedBattle) -> None:
    winner_name = battle.p1_name if battle.winner == Winner.P1 else (
        battle.p2_name if battle.winner == Winner.P2 else "TIE"
    )
    print(f"\n{'='*60}")
    print(f"  Replay: [{battle.format}] (#{battle.battle_id})")
    print(f"  Players: {battle.p1_name} vs {battle.p2_name}")
    print(f"  Winner:  {winner_name}  ({len(battle.turns)} turns)")
    print(f"{'='*60}")

    for snap in battle.turns:
        print(f"\n  |turn|{snap.turn_number}|")
        print(f"  P1 active : {fmt_pokemon(snap.p1.active)}")
        print(f"  P2 active : {fmt_pokemon(snap.p2.active)}")

        p1_bench = [p for p in snap.p1.team if p is not snap.p1.active]
        p2_bench = [p for p in snap.p2.team if p is not snap.p2.active]
        if p1_bench:
            print(f"  P1 bench  : {', '.join(p.species + ('✗' if p.fainted else '') for p in p1_bench)}")
        if p2_bench:
            print(f"  P2 bench  : {', '.join(p.species + ('✗' if p.fainted else '') for p in p2_bench)}")

        if snap.field.weather:
            print(f"  Weather   : {snap.field.weather}")
        if snap.field.conditions:
            print(f"  Field     : {', '.join(snap.field.conditions)}")
        if snap.p1.conditions:
            print(f"  P1 hazards: {', '.join(snap.p1.conditions)}")
        if snap.p2.conditions:
            print(f"  P2 hazards: {', '.join(snap.p2.conditions)}")

        print(f"  → {battle.p1_name}: {fmt_action(snap.p1_action)}")
        print(f"  → {battle.p2_name}: {fmt_action(snap.p2_action)}")

    print(f"\n  [win] {winner_name}\n")


# ---------------------------------------------------------------------------
# Step 3 diagnostic
# ---------------------------------------------------------------------------

def run_step3_diagnostic(n: int = 300, formats: list[str] | None = None) -> None:
    fmt_label = str(formats) if formats else "gen1–4 all"
    print(f"\nStep 3 feasibility filter diagnostic  (n={n}, formats={fmt_label})\n")

    counts = {
        "total": 0,
        "unsupported_format": 0,
        "bad_teamsize": 0,
        "no_species_clause": 0,
        "too_few_turns": 0,
        "parse_other": 0,
        "passed": 0,
    }
    examples: dict[str, list[str]] = {k: [] for k in counts}
    first_passed: ParsedBattle | None = None

    for entry in iter_raw_replays(formats=formats):
        if counts["total"] >= n:
            break
        counts["total"] += 1

        log = entry.get("log", "")
        fmt = entry.get("formatid", "")
        battle_id = entry.get("id", "?")
        label = f"{fmt} #{battle_id}"

        if not _supported(fmt):
            counts["unsupported_format"] += 1
            if len(examples["unsupported_format"]) < 3:
                examples["unsupported_format"].append(label)
            continue

        sizes = re.findall(r"\|teamsize\|p[12]\|(\d+)", log)
        if sizes and any(int(s) != 6 for s in sizes):
            counts["bad_teamsize"] += 1
            if len(examples["bad_teamsize"]) < 3:
                sizes_str = ",".join(sizes)
                examples["bad_teamsize"].append(f"{label}  sizes=[{sizes_str}]")
            continue

        if "|rule|Species Clause" not in log:
            counts["no_species_clause"] += 1
            if len(examples["no_species_clause"]) < 3:
                examples["no_species_clause"].append(label)
            continue

        turn_count = log.count("\n|turn|")
        if turn_count < 5:
            counts["too_few_turns"] += 1
            if len(examples["too_few_turns"]) < 3:
                examples["too_few_turns"].append(f"{label}  turns={turn_count}")
            continue

        battle = parse_battle(battle_id, log, fmt)
        if battle is None:
            counts["parse_other"] += 1
            if len(examples["parse_other"]) < 3:
                examples["parse_other"].append(label)
        else:
            counts["passed"] += 1
            if first_passed is None:
                first_passed = battle

    # --- summary table ---
    total = counts["total"]
    filter_keys = [
        "unsupported_format",
        "bad_teamsize",
        "no_species_clause",
        "too_few_turns",
        "parse_other",
    ]
    labels = {
        "unsupported_format": "unsupported format",
        "bad_teamsize":        "teamsize != 6",
        "no_species_clause":   "no Species Clause rule",
        "too_few_turns":       "< 5 turns",
        "parse_other":         "other (no winner / teams incomplete / dupes)",
    }

    print(f"  {'examined':<42}: {total:>5}")
    for key in filter_keys:
        c = counts[key]
        pct = 100 * c / total if total else 0
        print(f"  filtered – {labels[key]:<35}: {c:>5}  ({pct:5.1f}%)")
    passed = counts["passed"]
    pct = 100 * passed / total if total else 0
    print(f"  {'PASSED':<42}: {passed:>5}  ({pct:5.1f}%)")

    # --- examples per filter ---
    print()
    for key in filter_keys:
        exs = examples[key]
        if exs:
            print(f"  [{labels[key]}]")
            for ex in exs:
                print(f"    {ex}")
    print()

    # --- show one full passed battle ---
    if first_passed:
        print("First battle that passed all filters:")
        print_battle(first_passed)
    else:
        print("No battles passed all filters in this sample.")


# ---------------------------------------------------------------------------
# Step 4 diagnostic
# ---------------------------------------------------------------------------

def run_step4_diagnostic(
    n: int = 100,
    formats: list[str] | None = None,
    rank: str = "1630",
) -> None:
    fmt_label = str(formats) if formats else "gen1–4 all"
    print(f"\nStep 4 POV reconstruction diagnostic  (n={n}, formats={fmt_label})\n")

    # Load usage stats once per format before reconstructing POVs.
    # This downloads all monthly JSON files for each format and merges them.
    fmt_stats_cache: dict[str, dict[str, MovesetStats]] = {}
    if formats:
        for fmt in formats:
            print(f"  loading usage stats for {fmt}...")
            fmt_stats_cache[fmt] = load_format_stats(fmt, rank=rank)
            print(f"    {len(fmt_stats_cache[fmt])} species loaded")
    print()

    total_battles = 0
    total_pov_replays = 0
    total_snapshots = 0
    skipped_no_action = 0
    first_pov: POVReplay | None = None

    for battle in iter_parsed_battles(formats=formats):
        if total_battles >= n:
            break
        total_battles += 1

        fmt_stats = fmt_stats_cache.get(battle.format)
        p1_pov, p2_pov = reconstruct_both_povs(battle, format_stats=fmt_stats)

        for pov in (p1_pov, p2_pov):
            if pov is None:
                continue
            total_pov_replays += 1
            total_snapshots += len(pov.snapshots)
            if first_pov is None:
                first_pov = pov

        for turn in battle.turns:
            if turn.p1_action is None:
                skipped_no_action += 1
            if turn.p2_action is None:
                skipped_no_action += 1

    print(f"  battles parsed           : {total_battles}")
    print(f"  POV replays produced     : {total_pov_replays}  ({total_pov_replays / max(total_battles,1):.1f}/battle)")
    print(f"  total POV snapshots      : {total_snapshots}")
    avg = total_snapshots / max(total_pov_replays, 1)
    print(f"  avg snapshots / POV      : {avg:.1f}")
    print(f"  turns dropped (no action): {skipped_no_action}")
    print()

    if first_pov is None:
        print("No POV replays produced in this sample.")
        return

    # Print first 5 snapshots of the first POV replay
    pov = first_pov
    outcome = "WIN" if pov.won else "LOSS"
    print(f"First POV replay: [{pov.format}] #{pov.battle_id}")
    print(f"  Player   : {pov.player_name}  ({pov.pov_player})")
    print(f"  Opponent : {pov.opponent_name}")
    print(f"  Outcome  : {outcome}  ({len(pov.snapshots)} usable turns)")
    print()

    # Show inferred team for the POV player at the first snapshot
    first_snap = pov.snapshots[0]
    print(f"  My inferred team (turn {first_snap.turn_number}):")
    for pk in first_snap.my_side.team:
        moves_str = "/".join(pk.revealed_moves) if pk.revealed_moves else "—"
        item_str = f" @ {pk.item}" if pk.item else ""
        ability_str = f" [{pk.ability}]" if pk.ability else ""
        hp_str = f" {pk.hp*100:.0f}%"
        print(f"    {pk.species}{item_str}{ability_str}{hp_str}  [{moves_str}]")
    print()

    for snap in pov.snapshots[:5]:
        print(f"  turn {snap.turn_number}")
        active = snap.my_side.active
        print(f"    my active  : {fmt_pokemon(active)}")
        opp_team = snap.opp_side.team
        print(f"    opp team ({len(opp_team)}):")
        for pk in opp_team:
            inferred = " [inferred]" if pk.nickname == pk.species and pk.hp == 1.0 and not pk.revealed_moves[:1] == [] else ""
            moves_str = "/".join(pk.revealed_moves) if pk.revealed_moves else "—"
            item_str = f" @ {pk.item}" if pk.item else ""
            print(f"      {pk.species}{item_str}  moves:[{moves_str}]{inferred}")
        print(f"    my action  : {fmt_action(snap.my_action)}")
        print(f"    opp action : {fmt_action(snap.opp_action)}")
    if len(pov.snapshots) > 5:
        print(f"  ... ({len(pov.snapshots) - 5} more turns)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("loading dataset...")
# run_step3_diagnostic(n=300, formats=["gen1ou"])
run_step4_diagnostic(n=100, formats=["gen1ou"])
