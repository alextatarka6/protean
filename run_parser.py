import re

from protean.backend.replay_parser import iter_parsed_battles, ParsedBattle
from protean.backend.replay_parser.loader import iter_raw_replays
from protean.backend.replay_parser.parser import parse_battle, _supported
from protean.backend.replay_parser.types import TurnSnapshot, BattlePokemon, Winner


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
# Main
# ---------------------------------------------------------------------------

print("loading dataset...")
run_step3_diagnostic(n=300, formats=["gen4nu"])
