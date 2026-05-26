"""
models.py

LSTM and GRU models for stock return prediction.
- StockLSTM: Stacked LSTM + Dropout + FC
- StockGRU: Stacked GRU + Dropout + FC
- StockBiLSTM: Bidirectional LSTM for turning point detection (binary classification)
"""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Stacked LSTM for d-day return prediction.
    Input: (batch, T, F) -> Output: (batch, D)
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_size: int = 5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out


class StockGRU(nn.Module):
    """
    Stacked GRU for d-day return prediction.
    Input: (batch, T, F) -> Output: (batch, D)
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_size: int = 5,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F)
        gru_out, h_n = self.gru(x)
        out = self.dropout(gru_out[:, -1, :])
        out = self.fc(out)
        return out


class StockBiLSTM(nn.Module):
    """
    Bidirectional LSTM for turning point detection (buy/pass signal).
    Input: (batch, T, F) -> Output: (batch, 1) sigmoid probability
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F)
        lstm_out, _ = self.lstm(x)
        # Use last time step output (concatenation of forward and backward)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out
