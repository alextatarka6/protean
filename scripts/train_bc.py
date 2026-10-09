"""
Behavioural-cloning training for ProteanPolicy.

Dataset: atatark2/protean-gen1ou (streamed from HuggingFace)
Loss:    NLL with invalid actions masked to -inf before log-softmax
Optim:   AdamW + cosine LR schedule
Logs:    loss / accuracy to stdout each LOG_INTERVAL steps
Saves:   checkpoints/bc_stepN.pt every CHECKPOINT_INTERVAL steps
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protean.formats import get_format
from protean.formats.gen1ou.obs_space import Gen1OUObservationSpace, Gen1ActionSpace
from protean.model import ProteanPolicy

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_REPO      = "atatark2/protean-gen1ou"
CHECKPOINT_DIR    = Path("checkpoints")
LOG_INTERVAL      = 100      # steps
CHECKPOINT_INTERVAL = 5_000  # steps

# Training hyperparameters (overridable via CLI)
DEFAULTS = dict(
    batch_size   = 256,
    lr           = 3e-4,
    weight_decay = 1e-2,
    max_steps    = 200_000,
    warmup_steps = 2_000,
    seed         = 42,
)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

obs_space    = Gen1OUObservationSpace()
action_space = Gen1ActionSpace()


def _row_to_samples(row: dict, tokenizer, history_len: int = 1) -> list[tuple]:
    """
    Expand one dataset row into one sample per valid turn.
    Returns list of (token_ids, numbers, action_idx, action_mask).

    history_len == 1: token_ids (T,), numbers (D,)  — single-turn (original BC).
    history_len  > 1: token_ids (K,T), numbers (K,D) — the last K turns, left-padded
    with zeros at the start of a battle, exactly like ProteanPlayer.choose_move.
    """
    n_turns = row["num_turns"]
    per_turn = []
    for t in range(n_turns):
        obs = obs_space.row_to_obs(row, t)
        per_turn.append((tokenizer.tokenize(str(obs["text"])), obs["numbers"]))

    samples = []
    for t in range(n_turns):
        action_idx = action_space.row_to_action_idx(row, t)
        if action_idx == -1:
            continue  # unmappable action, skip
        mask = action_space.action_mask(row, t)

        if history_len == 1:
            token_ids, numbers = per_turn[t]
        else:
            window = per_turn[max(0, t - history_len + 1): t + 1]
            off = history_len - len(window)
            token_ids = np.zeros((history_len, window[0][0].shape[0]), dtype=np.int32)
            numbers   = np.zeros((history_len, window[0][1].shape[0]), dtype=np.float32)
            for i, (tt, nn) in enumerate(window):
                token_ids[off + i], numbers[off + i] = tt, nn
        samples.append((token_ids, numbers, action_idx, mask))
    return samples


def _is_holdout(battle_id: str, holdout_pct: int = 10) -> bool:
    """Deterministic train/holdout split via CRC32 of the battle_id."""
    return zlib.crc32(battle_id.encode()) % 100 < holdout_pct


def sample_stream(tokenizer, shuffle_buffer: int = 10_000, history_len: int = 1) -> Iterator[tuple]:
    """
    Infinite iterator over (token_ids, numbers, action_idx, action_mask) tuples,
    cycling through the HF dataset with an in-memory shuffle buffer.
    Holdout rows (10% by battle_id hash) are excluded.
    """
    ds = load_dataset(DATASET_REPO, split="train", streaming=True)

    buffer: list[tuple] = []
    rng = np.random.default_rng()

    while True:
        for row in ds:
            if _is_holdout(row["battle_id"]):
                continue
            for sample in _row_to_samples(row, tokenizer, history_len):
                buffer.append(sample)
                if len(buffer) >= shuffle_buffer:
                    idx = rng.integers(len(buffer))
                    yield buffer[idx]
                    buffer[idx] = buffer[-1]
                    buffer.pop()

        # Flush remaining buffer at end of dataset epoch
        rng.shuffle(buffer)
        yield from buffer
        buffer = []


def make_batch(samples: list[tuple], device: torch.device) -> tuple:
    token_ids, numbers, action_idxs, masks = zip(*samples)

    # Pad / stack token sequences (last axis is T; history windows are (K, T))
    max_len = max(t.shape[-1] for t in token_ids)
    token_arr = np.zeros((len(samples), *token_ids[0].shape[:-1], max_len), dtype=np.int32)
    for i, t in enumerate(token_ids):
        token_arr[i, ..., :t.shape[-1]] = t

    tokens  = torch.from_numpy(token_arr).long().to(device)
    nums    = torch.from_numpy(np.stack(numbers)).float().to(device)
    targets = torch.tensor(action_idxs, dtype=torch.long, device=device)
    amasks  = torch.from_numpy(np.stack(masks)).bool().to(device)

    return tokens, nums, targets, amasks


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    tokenizer = get_format("gen1ou").tokenizer
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    model = ProteanPolicy(**get_format("gen1ou").model_kwargs()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Class weights: upweight switch slots to counteract ~3:1 move/switch imbalance
    action_weights = torch.ones(9, device=device)
    action_weights[4:] = args.switch_weight
    print(f"Action weights: moves=1.0  switches={args.switch_weight}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Cosine LR over max_steps (with linear warmup handled manually)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # Don't overwrite the single-turn BC checkpoints when training with history.
    prefix = "bc" if args.history_len == 1 else f"bc_k{args.history_len}"
    print(f"History window: {args.history_len} turn(s)  →  checkpoints/{prefix}_*.pt")

    # Resume from checkpoint if provided
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        if args.finetune:
            # Fine-tune: only restore model weights; use fresh optimizer/scheduler
            # so mismatched parameter groups (e.g. BC → PPO checkpoint) don't error.
            start_step = 0
            print(f"Fine-tuning from {args.resume} (fresh optimizer, lr={args.lr})")
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_step = ckpt["step"]
            print(f"Resumed from step {start_step}")

    stream = sample_stream(tokenizer, history_len=args.history_len)

    step        = start_step
    batch_buf   = []
    total_loss  = 0.0
    total_correct = 0
    total_valid   = 0
    t0 = time.time()

    model.train()

    while step < args.max_steps:
        # Collect a batch
        batch_buf.clear()
        while len(batch_buf) < args.batch_size:
            batch_buf.append(next(stream))

        tokens, nums, targets, amasks = make_batch(batch_buf, device)

        # Warmup: scale LR linearly for first warmup_steps
        if step < args.warmup_steps:
            scale = (step + 1) / args.warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * scale

        optimizer.zero_grad()
        # No mask during training — masking the target slot to -inf causes inf loss
        # when the parser's revealed-move state lags the action taken.
        # The mask is used only at inference time (model.act).
        log_probs, _ = model(tokens, nums, action_mask=None)   # (B, 9)
        loss = F.nll_loss(log_probs, targets, weight=action_weights, reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step >= args.warmup_steps:
            scheduler.step()

        # Metrics: accuracy uses masked logits (inference behaviour)
        with torch.no_grad():
            masked_log_probs, _ = model(tokens, nums, amasks)
            preds   = masked_log_probs.argmax(dim=-1)
            correct = (preds == targets).sum().item()

        total_loss    += loss.item()
        total_correct += correct
        total_valid   += len(targets)
        step += 1

        if step % LOG_INTERVAL == 0:
            elapsed   = time.time() - t0
            avg_loss  = total_loss / LOG_INTERVAL
            accuracy  = total_correct / total_valid
            cur_lr    = optimizer.param_groups[0]["lr"]
            steps_per_sec = LOG_INTERVAL / elapsed
            print(
                f"step {step:>7,} | loss {avg_loss:.4f} | acc {accuracy:.3f} "
                f"| lr {cur_lr:.2e} | {steps_per_sec:.1f} steps/s"
            )
            total_loss = total_correct = total_valid = 0
            t0 = time.time()

        if step % CHECKPOINT_INTERVAL == 0:
            ckpt_path = CHECKPOINT_DIR / f"{prefix}_step{step:07d}.pt"
            torch.save({
                "step":      step,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args":      vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    # Final checkpoint
    final_path = CHECKPOINT_DIR / f"{prefix}_final.pt"
    torch.save({
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args":      vars(args),
    }, final_path)
    print(f"Training complete. Final checkpoint → {final_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BC training for ProteanPolicy")
    p.add_argument("--batch-size",    type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",            type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay",  type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--max-steps",     type=int,   default=DEFAULTS["max_steps"])
    p.add_argument("--warmup-steps",  type=int,   default=DEFAULTS["warmup_steps"])
    p.add_argument("--seed",          type=int,   default=DEFAULTS["seed"])
    p.add_argument("--switch-weight",  type=float, default=2.0,
                   help="Loss weight for switch slots 4-8 (default: 2.0 — partial class-imbalance correction, biases toward moves in uncertain situations)")
    p.add_argument("--history-len",   type=int,   default=1,
                   help="Turns of observation history per sample (default 1 = original BC; use 10 "
                        "to match ProteanPlayer / PPO). Saves checkpoints as bc_k<N>_*.pt")
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to a checkpoint to resume training from")
    p.add_argument("--finetune",      action="store_true",
                   help="Only restore model weights from --resume; use fresh optimizer/scheduler")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
