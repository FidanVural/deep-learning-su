"""
dataset.py

Financial time series dataset using yfinance.
Downloads OHLC data for S&P 500 tickers and creates sliding-window
input-output pairs for return ratio prediction.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from typing import List, Tuple, Optional


TICKERS = ["AAPL", "MSFT", "GOOGL"]

TRAIN_START = "2020-01-01"
TRAIN_END = "2024-07-31"
VAL_START = "2024-08-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

FEATURES = ["Open", "High", "Low", "Close"]
LOOKBACK = 20  # T
HORIZON = 5    # D (predict d=1,...,5 day returns)


def download_data(tickers: List[str], save_dir: str = "hw4/data") -> dict:
    """Download OHLC data from Yahoo Finance for given tickers."""
    os.makedirs(save_dir, exist_ok=True)
    data = {}
    for ticker in tickers:
        cache_path = os.path.join(save_dir, f"{ticker}.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            df = yf.download(ticker, start="2020-01-01", end="2025-12-31",
                             auto_adjust=True, progress=False)
            # yfinance returns MultiIndex columns; flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[FEATURES].copy()
            df.index.name = "Date"
            df.to_csv(cache_path)
        data[ticker] = df
    return data


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each feature column to zero mean and unit variance (per-stock)."""
    return (df - df.mean()) / (df.std() + 1e-8)


def create_sliding_windows(
    df: pd.DataFrame,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    mode: str = "return",
    rolling_window: int = 3,
    use_max_price: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window input-output pairs.

    Args:
        df: DataFrame with OHLC columns (raw, un-normalized).
        lookback: Number of past days to use as input (T=20).
        horizon: Number of future days to predict (D=5).
        mode: 'return' for exact d-day returns, 'rolling_avg' for weighted rolling avg.
        rolling_window: Window size l for rolling average (used when mode='rolling_avg').
        use_max_price: If True, use High price for return calculation (turning point detection).

    Returns:
        X: array of shape (num_samples, lookback, num_features)
        y: array of shape (num_samples, horizon)
    """
    close_prices = df["Close"].values
    high_prices = df["High"].values
    features = df[FEATURES].values

    # Normalize features for input
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0) + 1e-8
    features_norm = (features - feat_mean) / feat_std

    X_list, y_list = [], []
    n = len(df)

    for t in range(lookback - 1, n - horizon):
        x_window = features_norm[t - lookback + 1: t + 1]  # shape (T, F)
        pt = close_prices[t]

        if pt == 0:
            continue

        if mode == "return":
            # d-day return ratios: r_{t+d} = (p_{t+d} - p_t) / p_t
            returns = []
            for d in range(1, horizon + 1):
                if t + d >= n:
                    break
                if use_max_price:
                    p_future = high_prices[t + d]
                else:
                    p_future = close_prices[t + d]
                returns.append((p_future - pt) / pt)

            if len(returns) == horizon:
                X_list.append(x_window)
                y_list.append(returns)

        elif mode == "rolling_avg":
            # Weighted rolling average return
            returns = []
            l = rolling_window
            weights = np.ones(l + 1) / (l + 1)  # uniform weights

            for d in range(1, horizon + 1):
                if t + d >= n:
                    break
                # rolling avg: sum_{j=0}^{l} w_j * p_{t+d-j}
                prices_window = []
                for j in range(l + 1):
                    idx = t + d - j
                    if idx < 0 or idx >= n:
                        break
                    prices_window.append(close_prices[idx])

                if len(prices_window) == l + 1:
                    weighted_avg = np.dot(weights, prices_window)
                    returns.append((weighted_avg - pt) / pt)
                else:
                    break

            if len(returns) == horizon:
                X_list.append(x_window)
                y_list.append(returns)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


class StockDataset(Dataset):
    """PyTorch Dataset for stock return prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_datasets(
    tickers: List[str] = TICKERS,
    mode: str = "return",
    rolling_window: int = 3,
    use_max_price: bool = False,
    save_dir: str = "hw4/data",
) -> Tuple[StockDataset, StockDataset, StockDataset]:
    """
    Download data, split chronologically, create sliding windows, and return datasets.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    raw_data = download_data(tickers, save_dir=save_dir)

    train_X, train_y = [], []
    val_X, val_y = [], []
    test_X, test_y = [], []

    for ticker, df in raw_data.items():
        # Chronological split
        train_df = df[TRAIN_START:TRAIN_END]
        val_df = df[VAL_START:VAL_END]
        test_df = df[TEST_START:TEST_END]

        for split_df, X_list, y_list in [
            (train_df, train_X, train_y),
            (val_df, val_X, val_y),
            (test_df, test_X, test_y),
        ]:
            if len(split_df) < LOOKBACK + HORIZON:
                continue
            X, y = create_sliding_windows(
                split_df, lookback=LOOKBACK, horizon=HORIZON,
                mode=mode, rolling_window=rolling_window,
                use_max_price=use_max_price,
            )
            if len(X) > 0:
                X_list.append(X)
                y_list.append(y)

    train_X = np.concatenate(train_X, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    val_X = np.concatenate(val_X, axis=0)
    val_y = np.concatenate(val_y, axis=0)
    test_X = np.concatenate(test_X, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    print(f"Dataset sizes - Train: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}")
    print(f"Input shape: {train_X.shape}, Target shape: {train_y.shape}")

    return (
        StockDataset(train_X, train_y),
        StockDataset(val_X, val_y),
        StockDataset(test_X, test_y),
    )


def get_dataloaders(
    tickers: List[str] = TICKERS,
    batch_size: int = 64,
    mode: str = "return",
    rolling_window: int = 3,
    use_max_price: bool = False,
    save_dir: str = "hw4/data",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/val/test splits."""
    train_ds, val_ds, test_ds = prepare_datasets(
        tickers=tickers, mode=mode, rolling_window=rolling_window,
        use_max_price=use_max_price, save_dir=save_dir,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
