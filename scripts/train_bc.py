"""
Behavioural-cloning training for Gen1OUPolicy.

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

from protean.model import Gen1OUPolicy
from protean.obs_space import Gen1OUObservationSpace, Gen1ActionSpace
from protean.tokenizer import get_tokenizer

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


def _row_to_samples(row: dict, tokenizer) -> list[tuple]:
    """
    Expand one dataset row into one sample per valid turn.
    Returns list of (token_ids, numbers, action_idx, action_mask).
    """
    samples = []
    n_turns = row["num_turns"]
    for t in range(n_turns):
        action_idx = action_space.row_to_action_idx(row, t)
        if action_idx == -1:
            continue  # unmappable action, skip

        obs  = obs_space.row_to_obs(row, t)
        mask = action_space.action_mask(row, t)

        text = str(obs["text"])
        token_ids = tokenizer.tokenize(text)   # np.int32 array, length 71

        samples.append((
            token_ids,
            obs["numbers"],   # float32 (48,)
            action_idx,
            mask,             # bool (9,)
        ))
    return samples


def sample_stream(tokenizer, shuffle_buffer: int = 10_000) -> Iterator[tuple]:
    """
    Infinite iterator over (token_ids, numbers, action_idx, action_mask) tuples,
    cycling through the HF dataset with an in-memory shuffle buffer.
    """
    ds = load_dataset(DATASET_REPO, split="train", streaming=True)

    buffer: list[tuple] = []
    rng = np.random.default_rng()

    while True:
        for row in ds:
            for sample in _row_to_samples(row, tokenizer):
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

    # Pad / stack token sequences (all should be length 71, but guard anyway)
    max_len = max(t.shape[0] for t in token_ids)
    token_arr = np.zeros((len(samples), max_len), dtype=np.int32)
    for i, t in enumerate(token_ids):
        token_arr[i, :t.shape[0]] = t

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

    tokenizer = get_tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    model = Gen1OUPolicy(vocab_size=tokenizer.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Cosine LR over max_steps (with linear warmup handled manually)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if provided
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    stream = sample_stream(tokenizer)

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
        log_probs = model(tokens, nums, action_mask=None)   # (B, 9)
        loss = F.nll_loss(log_probs, targets, reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step >= args.warmup_steps:
            scheduler.step()

        # Metrics: accuracy uses masked logits (inference behaviour)
        with torch.no_grad():
            masked_log_probs = model(tokens, nums, amasks)
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
            ckpt_path = CHECKPOINT_DIR / f"bc_step{step:07d}.pt"
            torch.save({
                "step":      step,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args":      vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    # Final checkpoint
    final_path = CHECKPOINT_DIR / "bc_final.pt"
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
    p = argparse.ArgumentParser(description="BC training for Gen1OUPolicy")
    p.add_argument("--batch-size",    type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",            type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay",  type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--max-steps",     type=int,   default=DEFAULTS["max_steps"])
    p.add_argument("--warmup-steps",  type=int,   default=DEFAULTS["warmup_steps"])
    p.add_argument("--seed",          type=int,   default=DEFAULTS["seed"])
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to a checkpoint to resume training from")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
