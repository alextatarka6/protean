"""
Gen1OU behavioural cloning policy.

Architecture:
    text (71 tokens) → TokenEmbedding(vocab, 256) + PosEmbedding(128, 256)
                     → [CLS] prepended → Transformer(4L, 256, 8H, 1024FFN)
                     → CLS output [256]
                            ↓
    numbers (48)    → Linear(48→256) → ReLU [256]
                            ↓
                     Concat [512] → MLP(512→256→256) → [256]
                            ↓
                     Linear(256→9) → action logits

Call forward(tokens, numbers, action_mask) to get log-probabilities over 9 slots.
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

        # Action head
        self.action_head = nn.Linear(d_model, n_actions)

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
        tokens:      torch.Tensor,          # (B, T)  int64
        numbers:     torch.Tensor,          # (B, 48) float32
        action_mask: torch.Tensor | None = None,  # (B, 9)  bool — True = valid
    ) -> torch.Tensor:
        """
        Returns log-probabilities over 9 action slots.
        Invalid actions (mask=False) receive -inf before softmax.
        """
        text_repr    = self.encode_text(tokens)            # (B, 256)
        numbers_repr = self.numbers_proj(numbers)          # (B, 256)

        state = self.mlp(torch.cat([text_repr, numbers_repr], dim=-1))  # (B, 256)
        logits = self.action_head(state)                   # (B, 9)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def act(
        self,
        tokens:      torch.Tensor,
        numbers:     torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> int:
        """Greedy action for a single observation (no batch dim required)."""
        if tokens.dim() == 1:
            tokens  = tokens.unsqueeze(0)
            numbers = numbers.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)
        log_probs = self.forward(tokens, numbers, action_mask)
        return int(log_probs.argmax(dim=-1).item())
