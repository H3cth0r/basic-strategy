#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supervised trade execution model with walk-forward LSTM and Plotly/Dash visualization.

References:
- Dixon, M. (2017). A High Frequency Trade Execution Model for Supervised Learning.
- Tang, F. (2023). Application of supervised learning models in the Chinese futures market.
- Kangin & Pugeault (2018). Continuous Control with a Combination of Supervised and Reinforcement Learning.

What this file does:
1) Load BTC-USD 1-min data (~130k rows).
2) Compute TA indicators (ta library).
3) Volatility-scaled barrier labeling (triple-barrier style time-out).
4) Train a PyTorch LSTM classifier with walk-forward splits (progress bars).
5) Convert predicted probabilities into trade points and sizes (target position sizing).
6) Backtest: execute trades, compute PnL, mark sell-win and sell-lose.
7) Print evaluation metrics to terminal; launch a single-tab Dash app with multiple figures.

Notes:
- To keep runtime reasonable on a laptop, default windows are modest and a subset of tail data is used. Adjust config in MAIN CONFIG.
- Uses mps (Mac), cuda (GPU), else cpu.
"""

import os
import sys
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

# TA features
import ta

# ML / Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Plotly/Dash
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html

# ---------------------------------------------------------
# 0) Data loader provided by the user
# ---------------------------------------------------------
def load_data() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"
    column_names = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
    df = pd.read_csv(
        url, skiprows=[1, 2], header=0, names=column_names,
        parse_dates=["Datetime"], index_col="Datetime",
        dtype={"Volume": "int64"}, na_values=["NA", "N/A", ""],
        keep_default_na=True,
    )
    df = df.sort_index()
    return df

# ---------------------------------------------------------
# 1) Feature Engineering using ta
# ---------------------------------------------------------
def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Basic sanity
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}")
    df["Ret1"] = df["Close"].pct_change().fillna(0.0)
    df["LogRet"] = np.log(df["Close"]).diff().fillna(0.0)

    # Volatility
    df["Vol_10"] = df["Ret1"].rolling(10).std().fillna(method="bfill")
    df["Vol_30"] = df["Ret1"].rolling(30).std().fillna(method="bfill")
    df["ATR_14"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()

    # Trend
    df["EMA_10"] = ta.trend.EMAIndicator(close=df["Close"], window=10).ema_indicator()
    df["EMA_20"] = ta.trend.EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df["SMA_50"] = ta.trend.SMAIndicator(close=df["Close"], window=50).sma_indicator()
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # Momentum
    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # Volume-based
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    try:
        df["MFI_14"] = ta.volume.MFIIndicator(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=14).money_flow_index()
    except:
        df["MFI_14"] = np.nan

    # Volatility bands
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_H"] = bb.bollinger_hband()
    df["BB_L"] = bb.bollinger_lband()
    df["BB_PctB"] = bb.bollinger_pband()

    # Price distances
    df["Dist_EMA20"] = (df["Close"] - df["EMA_20"]) / df["Close"]
    df["Dist_SMA50"] = (df["Close"] - df["SMA_50"]) / df["Close"]

    # Fill missing
    df = df.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    return df

# ---------------------------------------------------------
# 2) Labeling: volatility-scaled barrier (triple-barrier style)
#    If future return exceeds +k_up*vol, label = 1
#    If future return falls below -k_dn*vol, label = -1
#    Else after horizon timeout -> 0
# ---------------------------------------------------------
def label_triple_barrier(df: pd.DataFrame, horizon: int = 30, k_up: float = 2.0, k_dn: float = 2.0, vol_window: int = 60) -> pd.Series:
    close = df["Close"].values
    vol = df["Ret1"].rolling(vol_window).std().fillna(method="bfill").values
    labels = np.zeros(len(df), dtype=int)
    n = len(df)

    # This simplified triple-barrier checks horizon return against vol-scaled thresholds
    # (path-independent simplification for speed)
    fut_ret = (pd.Series(close).shift(-horizon) / pd.Series(close) - 1.0).fillna(0.0).values
    up_thr = k_up * vol
    dn_thr = k_dn * vol

    labels[fut_ret > up_thr] = 1
    labels[fut_ret < -dn_thr] = -1
    # else 0

    return pd.Series(labels, index=df.index, name="Label")

# ---------------------------------------------------------
# 3) Sequence dataset for LSTM
# ---------------------------------------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx+self.seq_len, :]
        y_t = self.y[idx+self.seq_len-1]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_t, dtype=torch.long)

class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 2, dropout: float = 0.2, n_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)    # (B, T, H)
        out = out[:, -1, :]      # (B, H)
        logits = self.head(out)  # (B, C)
        return logits

# ---------------------------------------------------------
# 4) Walk-forward generator
# ---------------------------------------------------------
def generate_walk_forward_splits(df: pd.DataFrame, train_len: int, val_len: int, test_len: int, step: int):
    """
    Yields (train_idx, val_idx, test_idx) index arrays for walk-forward.
    """
    n = len(df)
    start = 0
    while start + train_len + val_len + test_len <= n:
        train_idx = np.arange(start, start+train_len)
        val_idx = np.arange(start+train_len, start+train_len+val_len)
        test_idx = np.arange(start+train_len+val_len, start+train_len+val_len+test_len)
        yield train_idx, val_idx, test_idx
        start += step

# ---------------------------------------------------------
# 5) Backtest engine
# ---------------------------------------------------------
def backtest_minute(
    times: np.ndarray, prices: np.ndarray, probs: np.ndarray,
    fee_bps: float = 5.0,  # 5 bps = 0.05%
    risk_per_trade: float = 0.01,
    vol_est: np.ndarray = None,
    max_trade_units: float = 0.25,   # Max BTC per minute
    min_lot: float = 0.001,          # Min BTC
    initial_cash: float = 10_000.0
):
    """
    Convert predicted probabilities into target positions and execute trades.
    - probs: (N,3) for classes [-1,0,1] in that order.
    - Position sizing: target_pos_btc ~ (p_up - p_down) * (risk_budget/vol).
    - Buy/sell with market orders at next minute price, with fees.
    Returns dict with equity curve, cash, holdings, trades log, sell markers separated into win/lose.
    """
    n = len(prices)
    cash = initial_cash
    btc = 0.0
    equity = np.zeros(n)
    cash_series = np.zeros(n)
    hold_series = np.zeros(n)

    if vol_est is None:
        vol_est = pd.Series(prices).pct_change().rolling(60).std().fillna(method="bfill").values
    vol_safe = np.where(vol_est <= 1e-6, 1e-6, vol_est)

    fee_rate = fee_bps / 1e4

    # Trade logs
    trades = []  # dicts with time, side, qty, price, fee, pnl_on_close
    open_trades = []  # track open buys (avg cost), we support position flipping with partial closes

    # probability order: [-1,0,1]
    for i in range(n):
        price = prices[i]
        p_dn, p_neu, p_up = probs[i]
        signal_strength = p_up - p_dn
        # equity update
        equity[i] = cash + btc * price
        cash_series[i] = cash
        hold_series[i] = btc

        # Compute target position by volatility scaling
        target_notional = equity[i] * risk_per_trade * np.clip(signal_strength, -1.0, 1.0) / vol_safe[i]
        target_btc = target_notional / price
        # Clamp target BTC to a reasonable range (e.g., -2 BTC .. +2 BTC)
        target_btc = float(np.clip(target_btc, -2.0, 2.0))

        # Determine trade to move toward target position
        delta_btc = target_btc - btc

        # Limit per-minute change
        trade_btc = float(np.clip(delta_btc, -max_trade_units, max_trade_units))

        # Respect min lot
        if abs(trade_btc) < min_lot:
            continue

        # Execute at price with fee
        trade_value = trade_btc * price
        trade_fee = abs(trade_value) * fee_rate

        if trade_btc > 0:
            # BUY
            if cash >= (trade_value + trade_fee):
                cash -= (trade_value + trade_fee)
                btc += trade_btc
                open_trades.append({"qty": trade_btc, "price": price, "value": trade_value, "time": times[i]})
                trades.append({"time": times[i], "side": "BUY", "qty": trade_btc, "price": price, "fee": trade_fee, "pnl_close": None})
        else:
            # SELL (closing or shorting -> we only allow long-only plus reducing position; if negative target, we flatten and optionally go short=off)
            sell_qty = min(abs(trade_btc), btc)  # do not short; restrict selling up to current btc
            if sell_qty > 0:
                sell_val = sell_qty * price
                sell_fee = sell_val * fee_rate
                cash += (sell_val - sell_fee)
                btc -= sell_qty
                # Match FIFO to compute realized PnL
                qty_to_match = sell_qty
                realized = 0.0
                while qty_to_match > 1e-12 and len(open_trades) > 0:
                    lot = open_trades[0]
                    take_qty = min(qty_to_match, lot["qty"])
                    realized += (price - lot["price"]) * take_qty
                    lot["qty"] -= take_qty
                    qty_to_match -= take_qty
                    if lot["qty"] <= 1e-12:
                        open_trades.pop(0)
                trades.append({"time": times[i], "side": "SELL", "qty": sell_qty, "price": price, "fee": sell_fee, "pnl_close": realized})

    # Final equity recompute
    for i in range(n):
        equity[i] = cash_series[i] + hold_series[i] * prices[i]

    # Identify sell-win and sell-lose markers
    sell_times_win = []
    sell_prices_win = []
    sell_times_lose = []
    sell_prices_lose = []
    buy_times = []
    buy_prices = []

    for tr in trades:
        if tr["side"] == "BUY":
            buy_times.append(tr["time"])
            buy_prices.append(tr["price"])
        elif tr["side"] == "SELL":
            pnl = tr.get("pnl_close", 0.0)
            if pnl is not None and pnl > 0:
                sell_times_win.append(tr["time"])
                sell_prices_win.append(tr["price"])
            else:
                sell_times_lose.append(tr["time"])
                sell_prices_lose.append(tr["price"])

    return {
        "equity": equity,
        "cash": cash_series,
        "holdings": hold_series,
        "trades": trades,
        "buy_markers": (np.array(buy_times), np.array(buy_prices)),
        "sell_win_markers": (np.array(sell_times_win), np.array(sell_prices_win)),
        "sell_lose_markers": (np.array(sell_times_lose), np.array(sell_prices_lose)),
    }

# ---------------------------------------------------------
# 6) Trading metrics
# ---------------------------------------------------------
def kpis_from_equity(equity: np.ndarray, times: np.ndarray, rf: float = 0.0):
    """
    Compute trading KPIs for minute-level equity curve.
    Annualization factor assumes ~365*24*60 minutes/year.
    """
    eps = 1e-12
    rets = np.diff(equity) / (equity[:-1] + eps)
    ann_factor = 365 * 24 * 60

    if len(rets) == 0:
        return {"CAGR": 0, "Sharpe": 0, "Sortino": 0, "MaxDD": 0, "MaxDD_Days": 0}

    mean_ret = np.mean(rets)
    std_ret = np.std(rets) + eps
    downside = rets[rets < 0]
    dd_std = np.std(downside) + eps

    sharpe = (mean_ret - rf/ann_factor) / std_ret * math.sqrt(ann_factor)
    sortino = (mean_ret - rf/ann_factor) / dd_std * math.sqrt(ann_factor)

    # CAGR: from first to last
    total_ret = equity[-1] / equity[0] - 1.0
    years = max((times[-1] - times[0]).astype('timedelta64[m]').astype(int) / (60*24*365), 1e-9)
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    # Max Drawdown
    cum = equity
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / (peak + 1e-12)
    maxdd = drawdown.min()
    # approximate duration
    maxdd_days = 0
    if len(drawdown) > 0:
        end_idx = np.argmin(drawdown)
        start_idx = np.argmax(cum[:end_idx+1])
        maxdd_days = (times[end_idx] - times[start_idx]).astype('timedelta64[D]').astype(int)

    return {"CAGR": cagr, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": maxdd, "MaxDD_Days": maxdd_days}

# ---------------------------------------------------------
# 7) Main training & evaluation
# ---------------------------------------------------------
def main():
    print("Loading data...")
    df = load_data()
    print(f"Data loaded: {df.index.min()} to {df.index.max()}, rows={len(df)}")

    # Add features
    print("Computing TA features...")
    df = add_ta_features(df)

    # Label
    print("Labeling (volatility-scaled barriers)...")
    df["Label"] = label_triple_barrier(df, horizon=30, k_up=2.0, k_dn=2.0, vol_window=60)

    # Clean rows for which all features exist
    feature_cols = [
        "Ret1","LogRet","Vol_10","Vol_30","ATR_14","EMA_10","EMA_20","SMA_50",
        "MACD","MACD_Signal","MACD_Hist","RSI_14","Stoch_K","Stoch_D","OBV","MFI_14","BB_H","BB_L","BB_PctB","Dist_EMA20","Dist_SMA50"
    ]
    cols_needed = feature_cols + ["Label", "Close"]
    df = df[cols_needed].dropna().copy()

    # To keep runtime moderate, focus on a recent tail
    # You can expand this later to use full data.
    TAIL = 80_000  # adjust as needed (smaller=quicker)
    if len(df) > TAIL:
        df = df.iloc[-TAIL:].copy()

    # Standardize features (fit on training windows only later; here we prepare arrays)
    # We will fit scaler per fold, respecting walk-forward protocol.
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["Label"].values.astype(int)
    time_all = df.index.values
    close_all = df["Close"].values.astype(np.float32)

    # LSTM config
    seq_len = 32
    batch_size = 512
    hidden = 64
    layers = 2
    dropout = 0.2
    lr = 1e-3
    max_epochs = 6  # keep small for speed; can increase
    patience = 2

    # Walk-forward config (minutes)
    # Train 20 days, Val 3 days, Test 5 days, step 5 days
    train_len = 20*24*60
    val_len = 3*24*60
    test_len = 5*24*60
    step = test_len

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Using device: {device}")

    # Storage for concatenated test predictions for backtest
    all_test_idx = []
    all_test_probs = []
    all_test_true = []

    fold_num = 0

    # Walk-forward
    splits = list(generate_walk_forward_splits(df, train_len, val_len, test_len, step))
    if len(splits) == 0:
        print("Not enough rows for walk-forward with current window sizes.")
        sys.exit(0)

    print(f"Starting walk-forward with {len(splits)} folds...")
    for (tr_idx, va_idx, te_idx) in tqdm(splits, desc="Walk-forward folds"):
        fold_num += 1

        # Fit scaler on train only
        scaler = StandardScaler()
        scaler.fit(X_all[tr_idx])
        X_tr = scaler.transform(X_all[tr_idx])
        X_va = scaler.transform(X_all[va_idx])
        X_te = scaler.transform(X_all[te_idx])
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]
        y_te = y_all[te_idx]

        # Class weights to address imbalance
        classes_present = np.unique(y_tr)
        cw = compute_class_weight(class_weight="balanced", classes=np.array([-1,0,1]), y=y_tr)
        # cw mapping is aligned with classes [-1,0,1] by ordering:
        class_to_index = {-1:0, 0:1, 1:2}
        weights = torch.tensor([cw[class_to_index[c]] for c in [-1,0,1]], dtype=torch.float32).to(device)

        # Build datasets
        ds_tr = SeqDataset(X_tr, y_tr, seq_len)
        ds_va = SeqDataset(X_va, y_va, seq_len)
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

        # Model
        model = LSTMClassifier(n_features=X_tr.shape[1], hidden=hidden, num_layers=layers, dropout=dropout, n_classes=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss(weight=weights)

        best_val = np.inf
        no_improve = 0

        # Train
        for epoch in tqdm(range(1, max_epochs+1), desc=f"Fold {fold_num} train", leave=False):
            model.train()
            total_loss = 0.0
            for xb, yb in dl_tr:
                xb = xb.to(device)
                yb_idx = torch.tensor([class_to_index[int(y.item())] for y in yb], dtype=torch.long, device=device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb_idx)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_tr = total_loss / max(1, len(ds_tr))

            # Validate
            model.eval()
            with torch.no_grad():
                total_v = 0.0
                count_v = 0
                for xb, yb in dl_va:
                    xb = xb.to(device)
                    yb_idx = torch.tensor([class_to_index[int(y.item())] for y in yb], dtype=torch.long, device=device)
                    logits = model(xb)
                    loss = criterion(logits, yb_idx)
                    total_v += loss.item() * xb.size(0)
                    count_v += xb.size(0)
                avg_va = total_v / max(1, count_v)

            tqdm.write(f"Fold {fold_num} | Epoch {epoch} | TrainLoss={avg_tr:.4f} | ValLoss={avg_va:.4f}")

            if avg_va + 1e-6 < best_val:
                best_val = avg_va
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    tqdm.write(f"Early stop on fold {fold_num} at epoch {epoch}")
                    break

        # Load best
        model.load_state_dict(best_state)

        # Predict test
        ds_te = SeqDataset(X_te, y_te, seq_len)
        dl_te = DataLoader(ds_te, batch_size=1024, shuffle=False)
        model.eval()
        fold_probs = []
        fold_true = []
        with torch.no_grad():
            for xb, yb in tqdm(dl_te, desc=f"Fold {fold_num} predict", leave=False):
                xb = xb.to(device)
                logits = model(xb)
                prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                # map class order [0,1,2] => [-1,0,1] same order output:
                # our mapping used y index [-1,0,1] -> [0,1,2]
                # prob[:,0]=P(-1), prob[:,1]=P(0), prob[:,2]=P(1)
                fold_probs.append(prob)
                fold_true.append(yb.numpy())

        fold_probs = np.vstack(fold_probs)
        fold_true = np.hstack(fold_true)
        # Align test indices (sequence reduces length by seq_len-1)
        te_eff_idx = te_idx[seq_len-1:]

        # Metrics (classification)
        y_pred = np.argmax(fold_probs, axis=1)
        # convert y_pred idx back to class labels [-1,0,1]
        idx_to_class = {0:-1, 1:0, 2:1}
        y_pred_lab = np.array([idx_to_class[i] for i in y_pred])
        acc = accuracy_score(fold_true, y_pred_lab)
        f1m = f1_score(fold_true, y_pred_lab, average="macro")
        prec = precision_score(fold_true, y_pred_lab, average="macro", zero_division=0)
        rec = recall_score(fold_true, y_pred_lab, average="macro", zero_division=0)
        print(f"[Fold {fold_num}] Classification: Acc={acc:.4f} | F1_macro={f1m:.4f} | Prec_macro={prec:.4f} | Rec_macro={rec:.4f}")
        print("Confusion matrix (rows=true, cols=pred in order [-1,0,1]):")
        cm = confusion_matrix(fold_true, y_pred_lab, labels=[-1,0,1])
        print(cm)
        print(classification_report(fold_true, y_pred_lab, labels=[-1,0,1], digits=4))

        all_test_idx.append(te_eff_idx)
        all_test_probs.append(fold_probs)
        all_test_true.append(fold_true)

    # Concatenate all test predictions
    all_test_idx = np.concatenate(all_test_idx)
    all_test_probs = np.vstack(all_test_probs)
    all_test_true = np.concatenate(all_test_true)

    # Prepare backtest arrays (aligned)
    bt_times = df.index.values[all_test_idx]
    bt_prices = close_all[all_test_idx]
    bt_vol = pd.Series(close_all).pct_change().rolling(60).std().fillna(method="bfill").values[all_test_idx]

    # Backtest
    print("Running backtest on concatenated test windows...")
    bt = backtest_minute(
        times=bt_times,
        prices=bt_prices,
        probs=all_test_probs,
        fee_bps=5.0,
        risk_per_trade=0.01,
        vol_est=bt_vol,
        max_trade_units=0.25,
        min_lot=0.001,
        initial_cash=10_000.0
    )

    # Trading KPIs
    kpis = kpis_from_equity(bt["equity"], bt_times)
    print("=== Trading KPIs (Test) ===")
    for k, v in kpis.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    # Simple realized trade stats
    sells = [t for t in bt["trades"] if t["side"] == "SELL" and t["pnl_close"] is not None]
    if len(sells) > 0:
        pnl_list = [t["pnl_close"] for t in sells]
        win_rate = np.mean([1.0 if p > 0 else 0.0 for p in pnl_list])
        avg_win = np.mean([p for p in pnl_list if p > 0]) if any(p > 0 for p in pnl_list) else 0.0
        avg_loss = np.mean([abs(p) for p in pnl_list if p < 0]) if any(p < 0 for p in pnl_list) else 0.0
        print(f"Trades closed: {len(sells)} | WinRate={win_rate:.3f} | AvgWin={avg_win:.2f} | AvgLoss={avg_loss:.2f}")
    else:
        print("No closed sells with PnL computed.")

    # Create Plotly/Dash app
    print("Launching Plotly/Dash single-tab dashboard...")
    app = init_dash_app(df, all_test_idx, all_test_probs, bt, kpis)
    # Run server; comment out if running in notebook
    app.run_server(host="0.0.0.0", port=8050, debug=False)

def init_dash_app(df: pd.DataFrame, test_idx: np.ndarray, test_probs: np.ndarray, bt: dict, kpis: dict):
    # Extract series for plotting
    times = df.index.values[test_idx]
    price = df["Close"].values[test_idx]

    # Probabilities
    p_dn = test_probs[:, 0]
    p_neu = test_probs[:, 1]
    p_up = test_probs[:, 2]

    # Equity, cash, holdings
    equity = bt["equity"]
    cash = bt["cash"]
    holds = bt["holdings"]

    # Markers
    buy_t, buy_p = bt["buy_markers"]
    sw_t, sw_p = bt["sell_win_markers"]
    sl_t, sl_p = bt["sell_lose_markers"]

    # Figure 1: Price with trades
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=times, y=price, mode="lines", name="BTC Close", line=dict(color="royalblue")))
    if len(buy_t) > 0:
        fig_price.add_trace(go.Scatter(x=buy_t, y=buy_p, mode="markers", name="BUY", marker=dict(color="green", symbol="triangle-up", size=8)))
    if len(sw_t) > 0:
        fig_price.add_trace(go.Scatter(x=sw_t, y=sw_p, mode="markers", name="SELL-win", marker=dict(color="limegreen", symbol="x", size=8)))
    if len(sl_t) > 0:
        fig_price.add_trace(go.Scatter(x=sl_t, y=sl_p, mode="markers", name="SELL-lose", marker=dict(color="red", symbol="x", size=8)))
    fig_price.update_layout(title="BTC Price with Trades (BUY, SELL-win, SELL-lose)", legend=dict(orientation="h"))

    # Figure 2: Portfolio value
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=times, y=equity, mode="lines", name="Equity", line=dict(color="black")))
    fig_equity.update_layout(title="Portfolio Value (Equity)")

    # Figure 3: Cash/credit
    fig_cash = go.Figure()
    fig_cash.add_trace(go.Scatter(x=times, y=cash, mode="lines", name="Cash", line=dict(color="orange")))
    fig_cash.update_layout(title="Cash / Credit Over Time")

    # Figure 4: Holdings
    fig_hold = go.Figure()
    fig_hold.add_trace(go.Scatter(x=times, y=holds, mode="lines", name="BTC Holdings", line=dict(color="purple")))
    fig_hold.update_layout(title="Holdings (BTC) Over Time")

    # Figure 5: Predicted Probabilities
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(x=times, y=p_up, mode="lines", name="P(Up)", line=dict(color="green")))
    fig_prob.add_trace(go.Scatter(x=times, y=p_neu, mode="lines", name="P(Neutral)", line=dict(color="gray")))
    fig_prob.add_trace(go.Scatter(x=times, y=p_dn, mode="lines", name="P(Down)", line=dict(color="red")))
    fig_prob.update_layout(title="Predicted Class Probabilities")

    # Figure 6: KPI Table
    kpi_table = go.Figure(data=[go.Table(
        header=dict(values=list(kpis.keys()), fill_color='paleturquoise', align='left'),
        cells=dict(values=[[f"{kpis[k]:.6f}" if isinstance(kpis[k], float) else str(kpis[k])] for k in kpis.keys()],
                   fill_color='lavender',
                   align='left'))
    ])
    kpi_table.update_layout(title="Trading KPIs")

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H2("Supervised Trading Model: LSTM + Walk-Forward (Single Tab, Separate Figures)"),
        html.Div(children=[
            dcc.Graph(figure=fig_price),
            dcc.Graph(figure=fig_equity),
            dcc.Graph(figure=fig_cash),
            dcc.Graph(figure=fig_hold),
            dcc.Graph(figure=fig_prob),
            dcc.Graph(figure=kpi_table),
        ])
    ])
    return app

if __name__ == "__main__":
    main()
