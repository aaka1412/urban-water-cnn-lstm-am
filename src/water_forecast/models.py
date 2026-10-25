from __future__ import annotations

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_size: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_size, attention_size), nn.Tanh(), nn.Linear(attention_size, 1))

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context, weights


class Forecaster(nn.Module):
    """Configurable ablation family: LSTM, CNN-LSTM, LSTM-AM, CNN-LSTM-AM."""
    def __init__(self, variant: str, hidden_size: int, cnn_filters: int, kernel_size: int, dropout: float, attention_size: int):
        super().__init__()
        if variant not in {"lstm", "cnn_lstm", "lstm_attention", "cnn_lstm_attention"}:
            raise ValueError(f"Unsupported model variant: {variant}")
        self.variant = variant
        self.use_cnn = "cnn" in variant
        self.use_attention = "attention" in variant
        channels = cnn_filters if self.use_cnn else 1
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_filters, kernel_size, padding=kernel_size // 2), nn.ReLU(), nn.Dropout(dropout)
        ) if self.use_cnn else nn.Identity()
        # A single-layer LSTM ignores its built-in dropout argument; apply it explicitly instead.
        self.lstm = nn.LSTM(channels, hidden_size, batch_first=True)
        self.sequence_dropout = nn.Dropout(dropout)
        self.attention = AdditiveAttention(hidden_size, attention_size) if self.use_attention else None
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.lstm(x)
        sequence = self.sequence_dropout(sequence)
        context = self.attention(sequence)[0] if self.attention else sequence[:, -1, :]
        return self.output(context).squeeze(-1)
