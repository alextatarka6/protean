"""
Fine-tune a PPO/BC checkpoint on ladder games (data/ladder_dataset.jsonl, built by
scripts/build_ladder_dataset.py).

Supervised: NLL on the chosen action, with the same K-turn observation windows the
policy sees at inference (left-padded with zeros, exactly like ProteanPlayer).
A KL term keeps the fine-tuned policy close to the starting checkpoint so a small
dataset can't wreck what PPO learned.

    loss = NLL(action) + kl_beta * KL(pi_ft || pi_start)

By default it trains on the *human* POV of games the human won — that's the useful
signal; the bot's own moves are just its current policy. Use --which to change.

Usage:
    python scripts/finetune_ladder.py --checkpoint checkpoints/ppo_ep0010000.pt
    python scripts/finetune_ladder.py --checkpoint checkpoints/ppo_ep0010000.pt \\
        --which both --min-rating 1100 --epochs 5 --lr 2e-5
Writes checkpoints/<base>_ladderft.pt, playable with ladder.py --checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from protean.formats import get_format  # noqa: E402
from protean.formats.gen1ou.obs_space import Gen1ActionSpace, Gen1OUObservationSpace  # noqa: E402
from protean.model import ProteanPolicy  # noqa: E402



def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_holdout(battle_id: str, pct: int) -> bool:
    return zlib.crc32(battle_id.encode()) % 100 < pct


def select_rows(path: Path, which: str, wins_only: bool, min_rating: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if which == "humans" and r["is_bot"]:
                continue
            if which == "bot" and not r["is_bot"]:
                continue
            if wins_only and not r["won"]:
                continue
            if (r.get("own_rating") or 0) < min_rating:
                continue
            rows.append(r)
    return rows


def build_samples(rows: list[dict], fmt, K: int):
    """One sample per decision: (K,T) tokens, (K,D) numbers, action, legal mask."""
    obs_space, action_space, tok = Gen1OUObservationSpace(), Gen1ActionSpace(), fmt.tokenizer
    toks, nums, acts, masks, ids = [], [], [], [], []
    for row in rows:
        per_turn = []
        for t in range(row["num_turns"]):
            obs = obs_space.row_to_obs(row, t)
            per_turn.append((tok.tokenize(str(obs["text"])), obs["numbers"]))
        for t in range(row["num_turns"]):
            a = action_space.row_to_action_idx(row, t)
            if a == -1:
                continue
            window = per_turn[max(0, t - K + 1): t + 1]
            off = K - len(window)
            T = window[0][0].shape[0]
            tk = np.zeros((K, T), dtype=np.int32)
            nm = np.zeros((K, fmt.numbers_dim), dtype=np.float32)
            for i, (tt, nn) in enumerate(window):
                tk[off + i], nm[off + i] = tt, nn
            toks.append(tk); nums.append(nm); acts.append(a)
            masks.append(action_space.action_mask(row, t)); ids.append(row["battle_id"])
    return (np.stack(toks), np.stack(nums), np.array(acts), np.stack(masks), ids)


@torch.no_grad()
def accuracy(model, data, device) -> float:
    tk, nm, ac, mk = (torch.from_numpy(x).to(device) for x in
                      (data[0].astype(np.int64), data[1], data[2], data[3]))
    model.eval()
    lp, _ = model(tk, nm, mk)
    return (lp.argmax(-1) == ac).float().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "ladder_dataset.jsonl"))
    ap.add_argument("--out", default=None, help="default: checkpoints/<base>_ladderft.pt")
    ap.add_argument("--which", choices=["humans", "bot", "both"], default="humans")
    ap.add_argument("--wins-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-rating", type=int, default=0, help="min rating of the POV player")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-beta", type=float, default=1.0)
    ap.add_argument("--holdout-pct", type=int, default=10)
    ap.add_argument("--history-len", type=int, default=None,
                    help="obs history window; must match how the policy is played. "
                         "Default: 1 for bc_* checkpoints, else 10")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    stem = Path(args.checkpoint).stem
    K = args.history_len or (1 if stem.startswith("bc_") else 10)
    print(f"History window K={K}")
    device = get_device()
    fmt = get_format("gen1ou")

    rows = select_rows(Path(args.data), args.which, args.wins_only, args.min_rating)
    if not rows:
        sys.exit("No rows match the filters — play more games or loosen --which/--min-rating.")
    toks, nums, acts, masks, ids = build_samples(rows, fmt, K)
    hold = np.array([is_holdout(b, args.holdout_pct) for b in ids])
    tr, ho = ~hold, hold
    print(f"{len(rows)} POV rows -> {len(acts)} decisions  (train {tr.sum()}, holdout {ho.sum()})")
    if tr.sum() < args.batch_size:
        print(f"Warning: only {tr.sum()} training decisions — expect little effect; play more games first.")

    model = ProteanPolicy(**fmt.model_kwargs()).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    def split(m):
        return (toks[m], nums[m], acts[m], masks[m])
    train_d, hold_d = split(tr), split(ho)
    if ho.sum():
        print(f"holdout accuracy before: {accuracy(model, hold_d, device):.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    n = int(tr.sum())
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.default_rng(args.seed + epoch).permutation(n)
        tot_nll = tot_kl = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            tk = torch.from_numpy(train_d[0][idx].astype(np.int64)).to(device)
            nm = torch.from_numpy(train_d[1][idx]).to(device)
            ac = torch.from_numpy(train_d[2][idx]).to(device)
            lp, _ = model(tk, nm, None)  # unmasked, as in BC training
            with torch.no_grad():
                ref_lp, _ = ref(tk, nm, None)
            nll = F.nll_loss(lp, ac)
            kl = (lp.exp() * (lp - ref_lp)).sum(-1).mean()
            loss = nll + args.kl_beta * kl
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gn):
                opt.step()
            tot_nll += nll.item() * len(idx); tot_kl += kl.item() * len(idx)
        msg = f"epoch {epoch + 1}/{args.epochs}  nll {tot_nll / n:.4f}  kl {tot_kl / n:.4f}"
        if ho.sum():
            msg += f"  holdout acc {accuracy(model, hold_d, device):.3f}"
        print(msg)

    out = Path(args.out) if args.out else ROOT / "checkpoints" / f"{Path(args.checkpoint).stem}_ladderft.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": ckpt.get("episode", ckpt.get("step", 0)),
                "base": args.checkpoint, "history_len": K, "args": vars(args)}, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
