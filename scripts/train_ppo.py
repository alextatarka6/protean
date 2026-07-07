"""
PPO self-play fine-tuning for Gen1OUPolicy.

Initialises from a BC checkpoint, then runs self-play battles on the local
Showdown server (port 8001) to fine-tune with Proximal Policy Optimisation.

Design:
  - 4 learner players + 4 opponent players, all sharing the live model weights
  - Opponent weights synced to learner every --opponent-sync-interval episodes (default: 20)
  - Rollout buffer: --rollout-steps transitions across all agents
  - GAE (γ=0.99, λ=0.95) for advantage estimation
  - PPO clip ε=0.2, 4 epochs per rollout, minibatch 256
  - KL penalty β * KL(π_RL ‖ π_BC) anchors RL policy to BC knowledge
  - AdamW lr=1e-4, grad clip 1.0

Usage:
    python scripts/train_ppo.py --bc-checkpoint checkpoints/bc_final.pt
    python scripts/train_ppo.py --bc-checkpoint checkpoints/bc_final.pt \\
        --max-episodes 10000 --rollout-steps 1024 --kl-beta 0.01
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protean.model import Gen1OUPolicy
from protean.rl_env import Gen1OUPlayer, Transition, LOCAL_SERVER
from protean.tokenizer import get_tokenizer
from protean.teams import ALL_TEAMS

CHECKPOINT_DIR = Path("checkpoints")

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
# Rollout buffer helpers
# ---------------------------------------------------------------------------

def make_batch(transitions: list[Transition], device: torch.device):
    """Stack a list of Transition objects into tensors."""
    max_len = max(t.tokens.shape[0] for t in transitions)
    token_arr = np.zeros((len(transitions), max_len), dtype=np.int32)
    for i, t in enumerate(transitions):
        token_arr[i, :t.tokens.shape[0]] = t.tokens

    tokens   = torch.from_numpy(token_arr).long().to(device)
    numbers  = torch.from_numpy(np.stack([t.numbers     for t in transitions])).float().to(device)
    masks    = torch.from_numpy(np.stack([t.action_mask for t in transitions])).bool().to(device)
    actions  = torch.tensor([t.action   for t in transitions], dtype=torch.long,  device=device)
    old_lps  = torch.tensor([t.log_prob for t in transitions], dtype=torch.float, device=device)
    values   = torch.tensor([t.value    for t in transitions], dtype=torch.float, device=device)
    rewards  = torch.tensor([t.reward   for t in transitions], dtype=torch.float, device=device)
    dones    = torch.tensor([t.done     for t in transitions], dtype=torch.float, device=device)
    return tokens, numbers, masks, actions, old_lps, values, rewards, dones


def compute_gae(
    rewards: torch.Tensor,  # (N,)
    values:  torch.Tensor,  # (N,)
    dones:   torch.Tensor,  # (N,)
    gamma:   float = 0.99,
    lam:     float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation.
    Returns (advantages, returns) both shape (N,).
    """
    N = len(rewards)
    advantages = torch.zeros(N, device=rewards.device)
    gae = 0.0
    # Bootstrap: next value is 0 if done, else the stored value of the next step
    next_value = 0.0
    for t in reversed(range(N)):
        if dones[t]:
            next_value = 0.0
            gae = 0.0
        delta  = rewards[t] + gamma * next_value - values[t]
        gae    = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t].item()
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    model:       Gen1OUPolicy,
    bc_model:    Gen1OUPolicy,        # frozen BC reference for KL penalty
    optimizer:   AdamW,
    transitions: list[Transition],
    device:      torch.device,
    args:        argparse.Namespace,
) -> dict[str, float]:
    """Run PPO update on a full rollout buffer. Returns dict of metrics."""
    tokens, numbers, masks, actions, old_lps, values, rewards, dones = make_batch(
        transitions, device
    )

    # GAE advantages + returns
    with torch.no_grad():
        advantages, returns = compute_gae(rewards, values, dones, args.gamma, args.gae_lambda)

    # Normalise advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    N = len(transitions)
    indices = np.arange(N)

    total_pg_loss  = 0.0
    total_vf_loss  = 0.0
    total_kl_loss  = 0.0
    total_loss_sum = 0.0
    n_updates      = 0

    model.train()
    for _ in range(args.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, N, args.minibatch_size):
            idx = indices[start: start + args.minibatch_size]
            if len(idx) < 2:
                continue

            b_tokens  = tokens[idx]
            b_numbers = numbers[idx]
            b_masks   = masks[idx]
            b_actions = actions[idx]
            b_old_lps = old_lps[idx]
            b_adv     = advantages[idx]
            b_returns = returns[idx]

            # Current policy
            log_probs, new_values = model(b_tokens, b_numbers, b_masks)
            new_lps = log_probs.gather(1, b_actions.unsqueeze(1)).squeeze(1)

            # PPO clipped objective.
            # Clamp ratio to [0, 10] before multiplying — otherwise inf*0 = nan when
            # a valid action gains near-zero probability in the updated policy.
            ratio    = (new_lps - b_old_lps).exp().clamp(max=10.0)
            pg_loss1 = ratio * b_adv
            pg_loss2 = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * b_adv
            pg_loss  = -torch.min(pg_loss1, pg_loss2).mean()

            # Value function loss (MSE against GAE returns)
            vf_loss = F.mse_loss(new_values.squeeze(-1), b_returns)

            # KL penalty vs frozen BC policy
            with torch.no_grad():
                bc_log_probs, _ = bc_model(b_tokens, b_numbers, b_masks)
            # KL(π_RL ‖ π_BC) = Σ π_RL * (log π_RL - log π_BC)
            # Masked slots have log_prob = -inf → prob = 0; use where() to avoid 0*nan
            probs = log_probs.exp()
            kl_elementwise = probs * (log_probs - bc_log_probs)
            kl_loss = torch.where(
                probs > 1e-10,
                kl_elementwise,
                torch.zeros_like(kl_elementwise),
            ).sum(dim=-1).mean()

            loss = pg_loss + args.vf_coef * vf_loss + args.kl_beta * kl_loss

            if not torch.isfinite(loss):
                print(
                    f"  WARNING: non-finite loss  "
                    f"pg={pg_loss.item():.4f}  "
                    f"vf={vf_loss.item():.4f}  "
                    f"kl={kl_loss.item():.4f}  — skipping batch"
                )
                continue

            optimizer.zero_grad()
            loss.backward()

            # Guard: skip the optimizer step if any gradient went NaN/Inf.
            # This prevents a single bad backward pass from corrupting all weights
            # (which would then NaN every subsequent forward pass in this rollout).
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                print(f"  WARNING: non-finite gradient norm ({grad_norm:.4f}), skipping optimizer step")
                optimizer.zero_grad()
                continue

            optimizer.step()

            total_pg_loss  += pg_loss.item()
            total_vf_loss  += vf_loss.item()
            total_kl_loss  += kl_loss.item()
            total_loss_sum += loss.item()
            n_updates      += 1

    model.eval()
    denom = max(n_updates, 1)
    return {
        "pg_loss":    total_pg_loss  / denom,
        "vf_loss":    total_vf_loss  / denom,
        "kl_loss":    total_kl_loss  / denom,
        "total_loss": total_loss_sum / denom,
        "mean_reward": rewards.mean().item(),
        "mean_value":  values.mean().item(),
    }


# ---------------------------------------------------------------------------
# Self-play loop
# ---------------------------------------------------------------------------

async def run_battles(
    learners:  list[Gen1OUPlayer],
    opponents: list[Gen1OUPlayer],
    n_battles_each: int,
) -> None:
    """Run n_battles_each battles for each learner/opponent pair concurrently."""
    tasks = [
        learner.battle_against(opponent, n_battles=n_battles_each)
        for learner, opponent in zip(learners, opponents)
    ]
    await asyncio.gather(*tasks)


def sync_opponent(learner: Gen1OUPolicy, opponent: Gen1OUPolicy) -> None:
    """Copy learner weights into the opponent model."""
    opponent.load_state_dict(copy.deepcopy(learner.state_dict()))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    tokenizer = get_tokenizer()

    # Load BC checkpoint → learner model
    print(f"Loading BC checkpoint: {args.bc_checkpoint}")
    ckpt = torch.load(args.bc_checkpoint, map_location=device)
    learner_model = Gen1OUPolicy(vocab_size=tokenizer.vocab_size).to(device)
    missing, unexpected = learner_model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  New keys (randomly initialised): {missing}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")
    learner_model.eval()
    print(f"  BC step: {ckpt.get('step', '?')}")

    # Frozen BC reference for KL penalty
    bc_model = Gen1OUPolicy(vocab_size=tokenizer.vocab_size).to(device)
    bc_model.load_state_dict(ckpt["model"], strict=False)
    for p in bc_model.parameters():
        p.requires_grad_(False)
    bc_model.eval()

    optimizer = AdamW(learner_model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Resume PPO checkpoint if provided — must happen before opponent sync so
    # the opponent starts from the resumed weights, not the BC weights.
    start_episode = 0
    if args.resume:
        ppo_ckpt = torch.load(args.resume, map_location=device)
        learner_model.load_state_dict(ppo_ckpt["model"])
        optimizer.load_state_dict(ppo_ckpt["optimizer"])
        start_episode = ppo_ckpt.get("episode", 0)
        print(f"Resumed from PPO episode {start_episode}")

    # Opponent model (lagged copy of learner) — sync after resume so it starts
    # from the correct weights (PPO if resuming, BC if starting fresh).
    opponent_model = Gen1OUPolicy(vocab_size=tokenizer.vocab_size).to(device)
    sync_opponent(learner_model, opponent_model)
    opponent_model.eval()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Create player instances — each pair gets a rotating team assignment.
    # A fraction (--bc-opponent-frac) of opponent slots use the frozen BC model
    # to preserve diverse, generalizable training signal and anchor the policy
    # to human-quality play.  The rest use the lagged self-play opponent.
    n_envs    = args.n_envs
    n_teams   = len(ALL_TEAMS)
    n_bc_opps = max(0, round(n_envs * args.bc_opponent_frac))

    learners = [
        Gen1OUPlayer(
            model=learner_model,
            device=device,
            sample=True,
            username=f"Protean_L{i}",
            team=ALL_TEAMS[i % n_teams],
        )
        for i in range(n_envs)
    ]
    opponents = [
        Gen1OUPlayer(
            # First n_bc_opps slots use frozen BC; remainder use lagged self-play.
            model=bc_model if i < n_bc_opps else opponent_model,
            device=device,
            sample=True,
            username=f"Protean_O{i}",
            team=ALL_TEAMS[(i + 1) % n_teams],
        )
        for i in range(n_envs)
    ]

    total_episodes = start_episode
    total_steps    = 0
    rollout_buf: list[Transition] = []

    print(f"\nStarting PPO training | {n_envs} envs | rollout {args.rollout_steps} steps")
    print(f"  Opponents: {n_bc_opps} frozen BC + {n_envs - n_bc_opps} self-play")
    print(f"  KL β={args.kl_beta}  clip ε={args.clip_eps}  lr={args.lr}")
    print(f"  Self-play opponent sync every {args.opponent_sync_interval} episodes\n")

    t0 = time.time()

    while total_episodes < args.max_episodes:
        # Run one batch of battles (1 per env pair) to collect transitions.
        asyncio.run(run_battles(learners, opponents, n_battles_each=1))
        # Brief pause so POKE_LOOP's background thread can finish server-side
        # battle teardown before the next challenge is issued.  Without this,
        # the server occasionally sends "|popup|You are already challenging
        # someone" and drops the challenge, causing battle_against to hang.
        time.sleep(0.5)
        total_episodes += n_envs

        # Drain all learner buffers
        for learner in learners:
            rollout_buf.extend(learner.drain_buffer())
        total_steps += len(rollout_buf)

        # Win stats
        ep_wins   = sum(p.n_won_battles   for p in learners)
        ep_played = sum(p.n_finished_battles for p in learners)

        # Sync opponent periodically
        if total_episodes % args.opponent_sync_interval < n_envs:
            sync_opponent(learner_model, opponent_model)

        # PPO update when rollout buffer is full
        if len(rollout_buf) >= args.rollout_steps:
            batch = rollout_buf[:args.rollout_steps]
            rollout_buf = rollout_buf[args.rollout_steps:]

            # Drop transitions with non-finite log_probs (collected during a NaN-weight
            # window).  Keeps ratio = exp(new_lp - old_lp) from becoming inf/nan.
            n_before = len(batch)
            batch = [t for t in batch if math.isfinite(t.log_prob)]
            if len(batch) < n_before:
                print(f"  Dropped {n_before - len(batch)} transitions with non-finite log_prob")
            if len(batch) < 2:
                print("  Not enough valid transitions for PPO update, skipping")
                continue

            metrics = ppo_update(
                model=learner_model,
                bc_model=bc_model,
                optimizer=optimizer,
                transitions=batch,
                device=device,
                args=args,
            )

            # Detect NaN weights — can happen if value-head loss spikes on the first
            # update (randomly-initialised value head, large initial returns).
            # Reset to BC weights and reinitialise the optimiser rather than crash.
            nan_in_weights = any(
                not torch.isfinite(p).all() for p in learner_model.parameters()
            )
            if nan_in_weights:
                print("  WARNING: NaN detected in model weights — resetting to BC checkpoint")
                missing, _ = learner_model.load_state_dict(ckpt["model"], strict=False)
                optimizer = AdamW(learner_model.parameters(), lr=args.lr, weight_decay=1e-2)
                sync_opponent(learner_model, opponent_model)

            elapsed    = time.time() - t0
            win_rate   = ep_wins / max(ep_played, 1)
            steps_per_s = total_steps / max(elapsed, 1)
            print(
                f"ep {total_episodes:>6,} | "
                f"win {win_rate:.3f} | "
                f"rew {metrics['mean_reward']:+.4f} | "
                f"pg {metrics['pg_loss']:+.4f} | "
                f"vf {metrics['vf_loss']:.4f} | "
                f"kl {metrics['kl_loss']:.4f} | "
                f"{'[RESET] ' if nan_in_weights else ''}"
                f"{steps_per_s:.0f} steps/s"
            )

        # Checkpoint
        if total_episodes % args.checkpoint_interval < n_envs:
            path = CHECKPOINT_DIR / f"ppo_ep{total_episodes:07d}.pt"
            torch.save({
                "episode":   total_episodes,
                "model":     learner_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args":      vars(args),
            }, path)
            print(f"  Saved → {path}")

    # Final checkpoint
    final_path = CHECKPOINT_DIR / "ppo_final.pt"
    torch.save({
        "episode":   total_episodes,
        "model":     learner_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args":      vars(args),
    }, final_path)
    print(f"\nTraining complete. Final checkpoint → {final_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO self-play for Gen1OUPolicy")
    p.add_argument("--bc-checkpoint",  type=str, required=True,
                   help="Path to BC .pt checkpoint to initialise from")
    p.add_argument("--resume",         type=str, default=None,
                   help="Path to PPO checkpoint to resume from")
    p.add_argument("--max-episodes",   type=int, default=20_000)
    p.add_argument("--n-envs",         type=int, default=4,
                   help="Number of parallel battle pairs (default: 4)")
    p.add_argument("--rollout-steps",  type=int, default=1024,
                   help="Transitions to collect before each PPO update")
    p.add_argument("--ppo-epochs",     type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--clip-eps",       type=float, default=0.2)
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--gae-lambda",     type=float, default=0.95)
    p.add_argument("--vf-coef",        type=float, default=0.1,
                   help="Value function loss coefficient (lower = less value-head influence early)")
    p.add_argument("--kl-beta",          type=float, default=0.1,
                   help="KL penalty weight vs frozen BC policy (default: 0.1; was 0.01)")
    p.add_argument("--bc-opponent-frac", type=float, default=0.75,
                   help="Fraction of opponent slots using frozen BC (default: 0.75)")
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--opponent-sync-interval", type=int, default=20,
                   help="Sync self-play opponent weights to learner every N episodes (default: 20)")
    p.add_argument("--checkpoint-interval",    type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
