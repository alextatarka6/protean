"""
Behavioural cloning from pre-tokenised .npz shards (any format).

Shards come from a format's converter, e.g. scripts/build_gen9ou_bc_data.py, and
hold per-turn arrays: tokens, numbers, action (-1 = unrevealed), legal (bitmask),
rating, holdout. Training is single-turn (K=1), like gen1ou BC — the trajectory
transformer is left at its zero init and learned during PPO.

Data loading: a few shards are held in memory at a time, shuffled together, and
swapped for a new random group once exhausted, so the full dataset never has to
fit in RAM.

Usage:
    python scripts/train_bc_shards.py --format gen9ou --data-dir data/gen9ou_bc
    python scripts/train_bc_shards.py --format gen9ou --data-dir data/gen9ou_bc \\
        --min-rating 1500 --max-steps 300000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protean.formats import FORMAT_NAMES, get_format
from protean.model import ProteanPolicy


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _select(d, holdout: bool, min_rating: int, include_unrated: bool) -> np.ndarray:
    """Indices of labelled turns in the requested split that pass the rating filter."""
    keep = (d["action"] >= 0) & (d["holdout"] == holdout)
    if min_rating > 0:
        rating = d["rating"]
        keep &= (rating >= min_rating) | ((rating == 0) & include_unrated)
    return np.flatnonzero(keep)


def _load(path: Path, holdout: bool, args) -> dict[str, np.ndarray]:
    d = np.load(path)
    idx = _select(d, holdout, args.min_rating, args.include_unrated)
    return {k: d[k][idx] for k in ("tokens", "numbers", "action", "legal")}


def _legal_to_mask(legal: np.ndarray, n_actions: int) -> np.ndarray:
    return ((legal[:, None].astype(np.int32) >> np.arange(n_actions)) & 1).astype(bool)


def batch_stream(shards: list[Path], args, rng: np.random.Generator) -> Iterator[dict]:
    """Infinite stream of training batches (numpy)."""
    while True:
        order = rng.permutation(len(shards))
        for g in range(0, len(order), args.shards_in_memory):
            group = [_load(shards[i], False, args) for i in order[g:g + args.shards_in_memory]]
            pool = {k: np.concatenate([s[k] for s in group]) for k in group[0]}
            perm = rng.permutation(len(pool["action"]))
            for b in range(0, len(perm) - args.batch_size + 1, args.batch_size):
                sel = perm[b:b + args.batch_size]
                yield {k: v[sel] for k, v in pool.items()}


def holdout_set(shards: list[Path], args, max_turns: int) -> dict[str, np.ndarray]:
    parts, n = [], 0
    for path in shards:
        parts.append(_load(path, True, args))
        n += len(parts[-1]["action"])
        if n >= max_turns:
            break
    return {k: np.concatenate([p[k] for p in parts])[:max_turns] for k in parts[0]}


def to_device(batch: dict, n_actions: int, device: torch.device):
    return (
        torch.from_numpy(batch["tokens"].astype(np.int64)).to(device),
        torch.from_numpy(batch["numbers"].astype(np.float32)).to(device),
        torch.from_numpy(batch["action"].astype(np.int64)).to(device),
        torch.from_numpy(_legal_to_mask(batch["legal"], n_actions)).to(device),
    )


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def autocast(device: torch.device, amp: str):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    return torch.autocast(device.type, dtype=dtype, enabled=dtype is not None and device.type != "cpu")


@torch.no_grad()
def evaluate(model, data: dict, fmt, device, amp: str = "none", batch_size: int = 1024) -> dict[str, float]:
    """Holdout loss / accuracy with the legal mask applied (inference behaviour)."""
    model.eval()
    loss_sum, correct, n = 0.0, np.zeros(3), np.zeros(3)   # [move, switch, tera]
    for b in range(0, len(data["action"]), batch_size):
        tokens, nums, targets, masks = to_device({k: v[b:b + batch_size] for k, v in data.items()},
                                                 fmt.n_actions, device)
        with autocast(device, amp):
            log_probs, _ = model(tokens, nums, masks)
        log_probs = log_probs.float()
        loss_sum += F.nll_loss(log_probs, targets, reduction="sum").item()
        hit = (log_probs.argmax(-1) == targets).cpu().numpy()
        t = targets.cpu().numpy()
        kind = np.where(t < 4, 0, np.where(t < 9, 1, 2))
        for k in range(3):
            correct[k] += hit[kind == k].sum()
            n[k] += (kind == k).sum()
    model.train()
    total = n.sum()
    return {
        "loss": loss_sum / total, "acc": correct.sum() / total,
        "move_acc": correct[0] / max(n[0], 1), "switch_acc": correct[1] / max(n[1], 1),
        "tera_acc": correct[2] / max(n[2], 1),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device()
    fmt = get_format(args.format)
    shards = sorted(Path(args.data_dir).glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"No shard_*.npz in {args.data_dir}")
    ckpt_dir = Path(args.checkpoint_dir or f"checkpoints/{fmt.name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | amp: {args.amp} | format: {fmt.name} | {len(shards)} shards in {args.data_dir}")

    model = ProteanPolicy(**fmt.model_kwargs()).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    action_weights = torch.ones(fmt.n_actions, device=device)
    action_weights[4:9] = args.switch_weight   # metamon-order switch slots

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1)

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]
        print(f"Resumed from step {step}")

    holdout = holdout_set(shards, args, args.holdout_turns)
    print(f"Holdout: {len(holdout['action']):,} turns")

    def save(path: Path) -> None:
        torch.save({"step": step, "format": fmt.name, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "args": vars(args)}, path)
        print(f"  Saved → {path}", flush=True)

    stream = batch_stream(shards, args, rng)
    model.train()
    run_loss = run_correct = run_n = 0.0
    t0 = time.time()
    while step < args.max_steps:
        tokens, nums, targets, masks = to_device(next(stream), fmt.n_actions, device)

        if step < args.warmup_steps:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * (step + 1) / args.warmup_steps

        with autocast(device, args.amp):
            log_probs, _ = model(tokens, nums, masks if args.train_mask else None)
        log_probs = log_probs.float()
        loss = F.nll_loss(log_probs, targets, weight=action_weights)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(grad_norm):
            optimizer.step()
        if step >= args.warmup_steps:
            scheduler.step()

        run_loss    += loss.item()
        run_correct += (log_probs.argmax(-1) == targets).sum().item()
        run_n       += len(targets)
        step += 1

        if step % args.log_interval == 0:
            dt = time.time() - t0
            print(f"step {step:>7,} | loss {run_loss / args.log_interval:.4f} | acc {run_correct / run_n:.3f} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e} | {args.log_interval / dt:.1f} steps/s", flush=True)
            run_loss = run_correct = run_n = 0.0
            t0 = time.time()
        if step % args.eval_interval == 0 or step == args.max_steps:
            m = evaluate(model, holdout, fmt, device, args.amp)
            print(f"  holdout | loss {m['loss']:.4f} | acc {m['acc']:.3f} | move {m['move_acc']:.3f} "
                  f"| switch {m['switch_acc']:.3f} | tera {m['tera_acc']:.3f}", flush=True)
        if step % args.checkpoint_interval == 0:
            save(ckpt_dir / f"bc_step{step:07d}.pt")

    save(ckpt_dir / "bc_final.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BC training from pre-tokenised shards")
    p.add_argument("--format",              default="gen9ou", choices=FORMAT_NAMES)
    p.add_argument("--data-dir",            default="data/gen9ou_bc")
    p.add_argument("--checkpoint-dir",      default=None, help="default: checkpoints/<format>")
    p.add_argument("--batch-size",          type=int,   default=256)
    p.add_argument("--lr",                  type=float, default=3e-4)
    p.add_argument("--weight-decay",        type=float, default=1e-2)
    p.add_argument("--max-steps",           type=int,   default=200_000)
    p.add_argument("--warmup-steps",        type=int,   default=2_000)
    p.add_argument("--switch-weight",       type=float, default=1.0,
                   help="loss weight on switch slots 4-8 (gen1ou BC used 2.0)")
    p.add_argument("--train-mask",          action=argparse.BooleanOptionalAction, default=True,
                   help="mask illegal actions in the training loss (labels are always legal in gen9ou shards)")
    p.add_argument("--min-rating",          type=int,   default=0,
                   help="drop rated battles below this rating (0 = keep all)")
    p.add_argument("--include-unrated",     action=argparse.BooleanOptionalAction, default=True,
                   help="keep unrated / tournament battles when --min-rating is set")
    p.add_argument("--shards-in-memory",    type=int,   default=4)
    p.add_argument("--holdout-turns",       type=int,   default=50_000)
    p.add_argument("--log-interval",        type=int,   default=100)
    p.add_argument("--eval-interval",       type=int,   default=2_000)
    p.add_argument("--checkpoint-interval", type=int,   default=10_000)
    p.add_argument("--amp",                 choices=["bf16", "fp16", "none"], default="bf16",
                   help="autocast dtype on mps/cuda (~40%% faster on Apple Silicon); ignored on cpu")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--resume",              default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
