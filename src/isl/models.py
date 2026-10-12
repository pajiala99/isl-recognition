"""Sequence classifiers over keypoint features."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import F_TOTAL


class AttentionPool(nn.Module):
    """Learned temporal attention - lets the model weight the frames where the
    sign actually happens and ignore the idle lead-in/lead-out of each clip."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x):                      # x: (B, T, D)
        w = torch.softmax(self.score(x), dim=1)
        return (x * w).sum(1), w.squeeze(-1)


class BiLSTMAttention(nn.Module):
    def __init__(self, n_classes: int, in_dim: int = F_TOTAL, hidden: int = 192,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.pool = AttentionPool(hidden * 2)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

    def forward(self, x, return_attn: bool = False):
        h, _ = self.lstm(self.proj(self.norm(x)))
        pooled, attn = self.pool(h)
        logits = self.head(pooled)
        return (logits, attn) if return_attn else logits


class TemporalTransformer(nn.Module):
    """Alternative encoder, reported in the ablation table."""

    def __init__(self, n_classes: int, in_dim: int = F_TOTAL, d_model: int = 192,
                 heads: int = 4, layers: int = 3, dropout: float = 0.3, seq_len: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.pool = AttentionPool(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x, return_attn: bool = False):
        h = self.enc(self.proj(self.norm(x)) + self.pos[:, : x.size(1)])
        pooled, attn = self.pool(h)
        logits = self.head(pooled)
        return (logits, attn) if return_attn else logits


def build_model(name: str, n_classes: int, cfg) -> nn.Module:
    if name == "bilstm":
        return BiLSTMAttention(n_classes, hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout)
    if name == "transformer":
        return TemporalTransformer(n_classes, d_model=cfg.hidden, dropout=cfg.dropout,
                                   seq_len=cfg.seq_len)
    raise ValueError(f"unknown model {name}")
