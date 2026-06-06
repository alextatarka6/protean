"""
Gen1OU policy network (BC pretraining + PPO fine-tuning).

Architecture:
    text (71 tokens) → TokenEmbedding(vocab, 256) + PosEmbedding(128, 256)
                     → [CLS] prepended → Transformer(4L, 256, 8H, 1024FFN)
                     → CLS output [256]
                            ↓
    numbers (48)    → Linear(48→256) → ReLU [256]
                            ↓
                     Concat [512] → MLP(512→256→256) → state repr [256]
                            ↓
              ┌─────────────────────────────┐
              │ policy_head: Linear(256→9) │  → masked log-softmax → log-probs
              │ value_head:  Linear(256→1) │  → scalar V(s) (for PPO)
              └─────────────────────────────┘

forward() returns (log_probs, value).
Value head operates on state.detach() — trunk trains only from policy gradient.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE   = 459
D_MODEL      = 256
N_HEADS      = 8
N_LAYERS     = 4
FFN_DIM      = 1024
MAX_SEQ_LEN  = 128   # covers 71 text tokens + 1 CLS
NUMBERS_DIM  = 48
N_ACTIONS    = 9


class Gen1OUPolicy(nn.Module):
    def __init__(
        self,
        vocab_size:   int = VOCAB_SIZE,
        d_model:      int = D_MODEL,
        nhead:        int = N_HEADS,
        num_layers:   int = N_LAYERS,
        ffn_dim:      int = FFN_DIM,
        max_seq_len:  int = MAX_SEQ_LEN,
        numbers_dim:  int = NUMBERS_DIM,
        n_actions:    int = N_ACTIONS,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Text branch
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_seq_len, d_model)
        self.cls_token   = nn.Parameter(torch.empty(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Numbers branch
        self.numbers_proj = nn.Sequential(
            nn.Linear(numbers_dim, d_model),
            nn.ReLU(),
        )

        # Fusion MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # Policy head
        self.action_head = nn.Linear(d_model, n_actions)

        # Value head (for PPO) — operates on detached state repr
        self.value_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Value head starts near zero so early updates don't explode.
        # Returns can be ±100 (terminal reward); a random init creates huge
        # MSE gradients on the first PPO update that can corrupt the policy trunk.
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T) int64 — clamp negatives to 0 before embedding.
        Returns: (B, d_model) — CLS output after transformer.
        """
        B, T = tokens.shape
        tokens = tokens.clamp(min=0)  # -1 (unknown) → 0

        x = self.token_embed(tokens)  # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)          # (B, T+1, d_model)

        # Positional embedding
        pos  = torch.arange(T + 1, device=tokens.device)
        x    = x + self.pos_embed(pos)

        x = self.transformer(x)   # (B, T+1, d_model)
        return x[:, 0]            # CLS position

    def forward(
        self,
        tokens:      torch.Tensor,                 # (B, T)  int64
        numbers:     torch.Tensor,                 # (B, 48) float32
        action_mask: torch.Tensor | None = None,   # (B, 9)  bool — True = valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (log_probs, value).
          log_probs: (B, 9)  — log-probabilities over action slots
          value:     (B, 1)  — scalar state-value estimate V(s)

        Invalid actions (mask=False) receive -1e9 before softmax (not -inf — see comment).
        Value head uses state.detach() so value loss doesn't flow through the trunk.
        """
        text_repr    = self.encode_text(tokens)                             # (B, 256)
        numbers_repr = self.numbers_proj(numbers)                           # (B, 256)
        state        = self.mlp(torch.cat([text_repr, numbers_repr], dim=-1))  # (B, 256)

        logits = self.action_head(state)                                    # (B, 9)
        if action_mask is not None:
            # Use a large finite negative instead of -inf.
            # In float32, exp(-1e9) underflows to 0 — identical softmax values to
            # -inf for valid slots.  But -inf causes NaN in log_softmax backward on
            # MPS (Apple Silicon GPU): the grad formula involves softmax_j * sum(grad)
            # where softmax_j is denormalized near-zero, yielding 0*inf = nan.
            logits = logits.masked_fill(~action_mask, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)

        value = self.value_head(state.detach())                             # (B, 1)

        return log_probs, value

    @torch.no_grad()
    def act(
        self,
        tokens:      torch.Tensor,
        numbers:     torch.Tensor,
        action_mask: torch.Tensor | None = None,
        sample:      bool = False,
    ) -> tuple[int, float, float]:
        """
        Select an action for a single observation (no batch dim required).

        Returns (action_idx, log_prob, value).
        If sample=True, samples from the distribution; otherwise argmax (greedy).
        """
        if tokens.dim() == 1:
            tokens  = tokens.unsqueeze(0)
            numbers = numbers.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)

        log_probs, value = self.forward(tokens, numbers, action_mask)

        if sample:
            probs = log_probs.exp().clamp(min=0.0)
            # If weights have gone NaN (e.g. after a bad PPO update), fall back to
            # a uniform distribution over valid actions so the battle can continue.
            if not torch.isfinite(probs).all():
                if action_mask is not None:
                    probs = action_mask.float()
                else:
                    probs = torch.ones_like(probs)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            if not torch.isfinite(log_probs).all():
                # Corrupt weights: fall back to first valid action
                if action_mask is not None:
                    action = action_mask[0].nonzero(as_tuple=False)[0, 0].unsqueeze(0)
                else:
                    action = torch.zeros(1, dtype=torch.long, device=log_probs.device)
            else:
                action = log_probs.argmax(dim=-1)

        action_idx = int(action.item())
        log_prob   = float(log_probs[0, action_idx].item())
        val        = float(value[0, 0].item())
        return action_idx, log_prob, val
