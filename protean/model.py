"""
Gen1OU policy network (BC pretraining + PPO fine-tuning).

Architecture (two-stage, matching metamon):

  Stage 1 — Turn Encoder (per-turn, shared weights):
    text (77 tokens) → TokenEmbedding(vocab, 256) + PosEmbedding(128, 256)
                     → [CLS] prepended → Transformer(4L, 256, 8H, 1024FFN)
                     → CLS output [256]
                            ↓
    numbers (48)    → Linear(48→256) → ReLU [256]
                            ↓
                     Concat [512] → MLP(512→256) → turn_emb [256]

  Stage 2 — Trajectory Encoder (causal, over K turns):
    [turn_emb_{t-K+1}, ..., turn_emb_t]  →  CausalTransformer(2L, 256, 8H)
                                          →  state [256]  (last position)
                                                 ↓
              ┌─────────────────────────────┐
              │ policy_head: Linear(256→9) │  → masked log-softmax → log-probs
              │ value_head:  Linear(256→1) │  → scalar V(s) (for PPO)
              └─────────────────────────────┘

forward() accepts (B, K, T) tokens and (B, K, 48) numbers.
For backward compat, (B, T) / (B, 48) are treated as K=1.

encode_turn_only() bypasses the trajectory encoder — used for KL computation
vs BC (which was trained on single-turn obs).

Value head uses state.detach() — trunk trains only from policy gradient.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE    = 459
D_MODEL       = 256
N_HEADS       = 8
N_LAYERS      = 4
FFN_DIM       = 1024
MAX_SEQ_LEN   = 128   # covers 77 text tokens + 1 CLS
NUMBERS_DIM   = 48
N_ACTIONS     = 9
HISTORY_LEN   = 10    # turns of battle history
TRAJ_LAYERS   = 2     # causal trajectory transformer depth


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
        history_len:  int = HISTORY_LEN,
        traj_layers:  int = TRAJ_LAYERS,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model     = d_model
        self.history_len = history_len

        # ── Stage 1: Turn Encoder ──────────────────────────────────────────
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_seq_len, d_model)
        self.cls_token   = nn.Parameter(torch.empty(1, 1, d_model))

        turn_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            turn_enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.numbers_proj = nn.Sequential(
            nn.Linear(numbers_dim, d_model),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # ── Stage 2: Causal Trajectory Encoder ────────────────────────────
        self.traj_pos_embed = nn.Embedding(history_len, d_model)

        traj_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.traj_transformer = nn.TransformerEncoder(
            traj_layer, num_layers=traj_layers, enable_nested_tensor=False
        )

        # ── Heads ──────────────────────────────────────────────────────────
        self.action_head = nn.Linear(d_model, n_actions)
        self.value_head  = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.traj_pos_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Value head near-zero init: PPO returns can be ±100; random init causes
        # huge MSE gradients that corrupt the policy trunk on the first update.
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        # Trajectory transformer near-zero: starts as an identity-ish pass-through
        # so early training isn't destabilised by random trajectory mixing.
        for module in self.traj_transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) → (B, d_model) via CLS-prepended transformer."""
        B, T = tokens.shape
        tokens = tokens.clamp(min=0)
        x   = self.token_embed(tokens)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        pos = torch.arange(T + 1, device=tokens.device)
        x   = x + self.pos_embed(pos)
        x   = self.transformer(x)
        return x[:, 0]  # CLS position

    def _encode_turn(self, tokens: torch.Tensor, numbers: torch.Tensor) -> torch.Tensor:
        """(B, T), (B, 48) → (B, d_model) turn embedding."""
        text = self._encode_text(tokens)
        nums = self.numbers_proj(numbers)
        return self.mlp(torch.cat([text, nums], dim=-1))

    def _apply_heads(
        self,
        state:       torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.action_head(state)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)
        value     = self.value_head(state.detach())
        return log_probs, value

    # ── Public API ─────────────────────────────────────────────────────────

    def forward(
        self,
        tokens:      torch.Tensor,                 # (B, K, T) or (B, T)
        numbers:     torch.Tensor,                 # (B, K, 48) or (B, 48)
        action_mask: torch.Tensor | None = None,   # (B, 9) bool — True = valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (log_probs, value).
          log_probs: (B, 9)
          value:     (B, 1)

        Accepts K-turn history (B, K, T) or single-turn (B, T) for BC compat.
        Padded (zero) history turns are processed but contribute near-zero
        embeddings — the trajectory transformer learns to ignore them.
        """
        # Normalise to (B, K, T)
        if tokens.dim() == 2:
            tokens  = tokens.unsqueeze(1)
            numbers = numbers.unsqueeze(1)

        B, K, T = tokens.shape

        # Encode all K turns in one batch pass
        turn_embs = self._encode_turn(
            tokens.reshape(B * K, T),
            numbers.reshape(B * K, -1),
        ).reshape(B, K, self.d_model)               # (B, K, d_model)

        # Add trajectory positional embeddings
        pos       = torch.arange(K, device=tokens.device)
        turn_embs = turn_embs + self.traj_pos_embed(pos)

        # Causal trajectory transformer — position i attends to 0..i only
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            K, device=tokens.device
        )
        state = self.traj_transformer(
            turn_embs, mask=causal_mask, is_causal=True
        )                                            # (B, K, d_model)
        state = state[:, -1]                         # current turn → (B, d_model)

        return self._apply_heads(state, action_mask)

    def encode_turn_only(
        self,
        tokens:      torch.Tensor,                 # (B, K, T) or (B, T)
        numbers:     torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Policy output using only the current turn's encoding — no trajectory context.
        Used for KL computation vs BC (which was trained on single-turn obs).
        If K-turn history is passed, only the last turn is used.
        """
        if tokens.dim() == 3:
            tokens  = tokens[:, -1, :]
            numbers = numbers[:, -1, :]
        state = self._encode_turn(tokens, numbers)
        return self._apply_heads(state, action_mask)

    @torch.no_grad()
    def act(
        self,
        tokens:      torch.Tensor,                 # (K, T), (T,), (1, K, T), or (1, T)
        numbers:     torch.Tensor,
        action_mask: torch.Tensor | None = None,
        sample:      bool = False,
    ) -> tuple[int, float, float]:
        """
        Select an action for a single observation (batch dim optional).
        Returns (action_idx, log_prob, value).
        """
        # Normalise to (1, K, T)
        if tokens.dim() == 1:
            tokens  = tokens.unsqueeze(0).unsqueeze(0)
            numbers = numbers.unsqueeze(0).unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)
        elif tokens.dim() == 2:
            # (K, T) single sample with history, or (1, T) single turn
            tokens  = tokens.unsqueeze(0)
            numbers = numbers.unsqueeze(0)
            if action_mask is not None and action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

        log_probs, value = self.forward(tokens, numbers, action_mask)

        if sample:
            probs = log_probs.exp().clamp(min=0.0)
            if not torch.isfinite(probs).all():
                probs = action_mask.float() if action_mask is not None else torch.ones_like(probs)
            probs  = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            if not torch.isfinite(log_probs).all():
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
