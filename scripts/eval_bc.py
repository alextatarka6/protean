"""
Evaluate a BC checkpoint against the protean-gen1ou dataset.

Reports:
  - Overall top-1 accuracy
  - Move accuracy  (action slots 0-3)
  - Switch accuracy (action slots 4-8)
  - Forced-switch accuracy (subset of switch)
  - Per-action-slot confusion (optional --confusion flag)

Usage:
    python scripts/eval_bc.py --checkpoint checkpoints/bc_final.pt                        # holdout (default)
    python scripts/eval_bc.py --checkpoint checkpoints/bc_final.pt --rows 2000
    python scripts/eval_bc.py --checkpoint checkpoints/bc_final.pt --confusion
    python scripts/eval_bc.py --checkpoint checkpoints/bc_final.pt --split train          # train rows only
    python scripts/eval_bc.py --checkpoint checkpoints/bc_final.pt --split all            # full dataset
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protean.model import Gen1OUPolicy
from protean.obs_space import Gen1OUObservationSpace, Gen1ActionSpace
from protean.tokenizer import get_tokenizer

DATASET_REPO = "atatark2/protean-gen1ou"
N_ACTIONS    = 9


def _is_holdout(battle_id: str, holdout_pct: int = 10) -> bool:
    """Deterministic train/holdout split via CRC32 of the battle_id."""
    return zlib.crc32(battle_id.encode()) % 100 < holdout_pct


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    tokenizer    = get_tokenizer()
    obs_space    = Gen1OUObservationSpace()
    action_space = Gen1ActionSpace()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = Gen1OUPolicy(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    trained_step = ckpt.get("step", "?")
    print(f"Loaded checkpoint: {args.checkpoint} (step {trained_step})")

    # Counters
    total = correct = 0
    move_total   = move_correct   = 0
    switch_total = switch_correct = 0
    forced_total = forced_correct = 0

    confusion = np.zeros((N_ACTIONS, N_ACTIONS), dtype=np.int64)  # [true, pred]

    ds = load_dataset(DATASET_REPO, split="train", streaming=True)

    rows_seen = 0
    with torch.no_grad():
        for row in ds:
            if args.rows and rows_seen >= args.rows:
                break
            if args.split == "holdout" and not _is_holdout(row["battle_id"]):
                continue
            if args.split == "train" and _is_holdout(row["battle_id"]):
                continue

            for t in range(row["num_turns"]):
                action_idx = action_space.row_to_action_idx(row, t)
                if not (0 <= action_idx < N_ACTIONS):
                    continue

                obs  = obs_space.row_to_obs(row, t)
                mask = action_space.action_mask(row, t)

                token_ids = tokenizer.tokenize(str(obs["text"]))
                tokens  = torch.from_numpy(token_ids).long().unsqueeze(0).to(device)
                numbers = torch.from_numpy(obs["numbers"]).float().unsqueeze(0).to(device)
                amask   = torch.from_numpy(mask).bool().unsqueeze(0).to(device)

                log_probs = model(tokens, numbers, amask)
                pred = int(log_probs.argmax(dim=-1).item())
                is_correct = (pred == action_idx)

                total   += 1
                correct += int(is_correct)

                is_forced = bool(row["my_action_forced"][t])
                if action_idx < 4:
                    move_total   += 1
                    move_correct += int(is_correct)
                else:
                    switch_total   += 1
                    switch_correct += int(is_correct)
                    if is_forced:
                        forced_total   += 1
                        forced_correct += int(is_correct)

                if args.confusion:
                    confusion[action_idx, pred] += 1

            rows_seen += 1
            if rows_seen % 500 == 0:
                acc = correct / total if total else 0
                print(f"  rows {rows_seen:>6,} | samples {total:>8,} | acc {acc:.3f}")

    print()
    print("=" * 50)
    print(f"Rows evaluated:   {rows_seen:,}")
    print(f"Total samples:    {total:,}")
    print(f"Overall accuracy: {correct/total:.4f}  ({correct}/{total})")
    print(f"Move   accuracy:  {move_correct/move_total:.4f}  ({move_correct}/{move_total})")
    if switch_total:
        print(f"Switch accuracy:  {switch_correct/switch_total:.4f}  ({switch_correct}/{switch_total})")
    if forced_total:
        print(f"Forced-sw accur:  {forced_correct/forced_total:.4f}  ({forced_correct}/{forced_total})")

    if args.confusion:
        print()
        print("Confusion matrix (rows=true, cols=pred):")
        header = "     " + "".join(f"{i:>6}" for i in range(N_ACTIONS))
        print(header)
        labels = [f"M{i}" for i in range(4)] + [f"S{i}" for i in range(5)]
        for i in range(N_ACTIONS):
            row_str = f"{labels[i]:>4} " + "".join(f"{confusion[i,j]:>6}" for j in range(N_ACTIONS))
            print(row_str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a BC checkpoint")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--split", type=str, default="holdout", choices=["holdout", "train", "all"],
                   help="Which rows to evaluate: holdout (default), train, or all")
    p.add_argument("--rows", type=int, default=0,
                   help="Number of dataset rows to evaluate (0 = full split)")
    p.add_argument("--confusion", action="store_true",
                   help="Print 9×9 confusion matrix")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
