# supervised_guided_rl_btcusd.py
# pip install pandas numpy scipy scikit-learn torch tqdm plotly ta

import os
import math
import copy
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.special import softmax
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ta

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# 1) Data loading (provided)
# ---------------------------

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

# ---------------------------
# 2) Features: TA, FFD, Triple Barrier
# ---------------------------

def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ret_1'] = df['Close'].pct_change()
    df['logret_1'] = np.log(df['Close']).diff()
    df['vol_ewm'] = df['ret_1'].ewm(span=60, min_periods=60).std()
    df['vol_ewm'] = df['vol_ewm'].bfill()

    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14, fillna=True).rsi()
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3, fillna=True)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2, fillna=True)
    df['bb_h'] = bb.bollinger_hband()
    df['bb_l'] = bb.bollinger_lband()
    df['bb_w'] = (df['bb_h'] - df['bb_l']) / df['Close'].replace(0, np.nan)
    df['ema_20'] = ta.trend.EMAIndicator(df['Close'], window=20, fillna=True).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['Close'], window=50, fillna=True).ema_indicator()
    df['ema_diff'] = (df['ema_20'] - df['ema_50']) / df['Close'].replace(0, np.nan)

    df['mfi'] = ta.volume.MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14, fillna=True).money_flow_index()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def fracdiff_fixed_width(x: pd.Series, d: float = 0.3, tol: float = 1e-3, max_lag: int = 500) -> pd.Series:
    x = x.dropna().astype(float)
    w = [1.0]
    k = 1
    while k < max_lag:
        w_k = w[-1] * (-(d - (k - 1))) / k
        if abs(w_k) < tol:
            break
        w.append(w_k)
        k += 1
    w = np.array(w)[::-1]
    out = np.full_like(x.values, np.nan, dtype=float)
    vals = x.values
    for i in range(len(vals)):
        if i >= len(w) - 1:
            window = vals[i - len(w) + 1: i + 1]
            out[i] = np.dot(w, window)
    return pd.Series(out, index=x.index, name=f'ffd_d{d}')

def add_ffd_feature(df: pd.DataFrame, d: float = 0.3) -> pd.DataFrame:
    df = df.copy()
    df['ffd_close'] = fracdiff_fixed_width(df['Close'], d=d, tol=1e-3, max_lag=500)
    df['ffd_close'] = df['ffd_close'].bfill()
    return df

def triple_barrier_labels(close: pd.Series, vol: pd.Series, up_mult: float = 5.0, dn_mult: float = 5.0, horizon: int = 120) -> pd.Series:
    """
    Triple Barrier labels with longer horizon and larger multipliers to produce more neutral labels,
    improving class balance for training stability.
    """
    n = len(close)
    labels = np.zeros(n, dtype=int)
    idx = close.index
    cvals = close.values
    vol = vol.reindex(idx)
    vol = vol.bfill()
    vvals = vol.values
    for i in tqdm(range(n - horizon), desc="Labeling (Triple Barrier)", leave=False):
        price0 = cvals[i]
        up_bar = price0 * (1 + up_mult * vvals[i])
        dn_bar = price0 * (1 - dn_mult * vvals[i])
        label = 0
        for j in range(1, horizon + 1):
            p = cvals[i + j]
            if p >= up_bar:
                label = 1
                break
            if p <= dn_bar:
                label = -1
                break
        labels[i] = label
    labels[-horizon:] = 0
    return pd.Series(labels, index=idx, name='tb_label')

# ---------------------------
# 3) LSTM classifier with focal loss (better for imbalance)
# ---------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps')
    except Exception:
        pass
    return torch.device('cpu')

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96, layers: int = 2, dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.head(out[:, -1, :])
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)  # [N]
        pt = torch.softmax(logits, dim=1)
        pt = pt.gather(1, targets.view(-1,1)).squeeze(1)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len:i, :])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)

# ---------------------------
# 4) TIM and position mapping
# ---------------------------

def confusion_and_tim(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    C = confusion_matrix(y_true, y_pred, labels=[-1,0,1])
    T = np.array([
        [ 1.0, -0.05, -1.0],   # true -1
        [-0.05, 0.10, -0.05],  # true 0
        [-1.0, -0.05,  1.0],   # true +1
    ], dtype=float)
    exp_value = float(np.trace(C @ T.T)) / max(1, C.sum())
    return C, T, exp_value

def probs_to_position(prob_up: float, prob_dn: float, prob_flat: float, T: np.ndarray, max_pos: float = 0.5, edge_thresh: float = 0.05) -> float:
    p_vec = np.array([prob_dn, prob_flat, prob_up], dtype=float)
    E_long  = float(np.dot(p_vec, T[:, 2]))
    E_flat  = float(np.dot(p_vec, T[:, 1]))
    E_short = float(np.dot(p_vec, T[:, 0]))
    best_action = np.argmax([E_short, E_flat, E_long])  # 0=short,1=flat,2=long
    edges = [E_short - E_flat, 0.0, E_long - E_flat]
    edge = edges[best_action]
    if edge <= edge_thresh:
        return 0.0
    if best_action == 2:
        return min(max_pos, max(0.0, edge))
    elif best_action == 0:
        return 0.0  # disable shorting for spot safety
    else:
        return 0.0

def smooth_positions(pos: pd.Series, ema_span: int = 30) -> pd.Series:
    return pos.ewm(span=ema_span, adjust=False).mean().clip(lower=0.0, upper=1.0)

# ---------------------------
# 5) Backtester (no margin, no short)
# ---------------------------

@dataclass
class BacktestResult:
    df_trades: pd.DataFrame
    equity_curve: pd.Series
    cash_curve: pd.Series
    holdings_curve: pd.Series
    metrics: Dict[str, float]

def compute_metrics(equity: pd.Series, returns: pd.Series) -> Dict[str, float]:
    ret = returns.dropna()
    ann_scale = np.sqrt(525600.0)  # 365*24*60
    sharpe = (ret.mean() / (ret.std() + 1e-12)) * ann_scale
    downside = ret[ret < 0]
    sortino = (ret.mean() / (downside.std() + 1e-12)) * ann_scale
    roll_max = equity.cummax()
    dd = (equity / roll_max - 1.0)
    max_dd = dd.min()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    return {
        "Total Return": float(total_return),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Max Drawdown": float(max_dd),
    }

def backtest_signals(
    df: pd.DataFrame,
    positions: pd.Series,
    fee_bps: float = 7.5,
    slippage_bps: float = 1.0,
    initial_cash: float = 10000.0,
) -> BacktestResult:
    fee = fee_bps / 10000.0
    slippage = slippage_bps / 10000.0
    price = df['Close']
    idx = df.index

    equity = []
    cash = []
    holdings = []
    trades = []

    current_cash = initial_cash
    current_units = 0.0

    for t in tqdm(range(len(idx)), desc="Backtest", leave=False):
        p = float(price.iloc[t])
        target_pos = float(np.clip(positions.iloc[t], 0.0, 1.0))  # long-only fraction
        current_equity = current_cash + current_units * p
        target_units = target_pos * current_equity / p if p > 0 else 0.0
        delta_units = target_units - current_units

        if abs(delta_units) > 1e-9:
            # Enforce no shorting and no margin
            if delta_units > 0:
                trade_price = p * (1 + slippage)
                max_buy_units = max(0.0, current_cash / (trade_price * (1 + fee)))
                delta_units = min(delta_units, max_buy_units)
                trade_cost = delta_units * trade_price
                fee_cost = trade_cost * fee
                current_cash -= (trade_cost + fee_cost)
                current_units += delta_units
                if delta_units > 0:
                    trades.append({"time": idx[t], "type": "BUY", "units": float(delta_units), "price": trade_price, "fee": fee_cost})
            else:
                # Sell only up to current holdings
                delta_units = max(delta_units, -current_units)
                trade_price = p * (1 - slippage)
                trade_proceeds = (-delta_units) * trade_price
                fee_cost = trade_proceeds * fee
                current_cash += (trade_proceeds - fee_cost)
                current_units += delta_units
                if delta_units < 0:
                    trades.append({"time": idx[t], "type": "SELL", "units": float(delta_units), "price": trade_price, "fee": fee_cost})

        current_equity = current_cash + current_units * p
        equity.append(current_equity)
        cash.append(current_cash)
        holdings.append(current_units)

    equity_s = pd.Series(equity, index=idx, name='equity')
    cash_s = pd.Series(cash, index=idx, name='cash')
    hold_s = pd.Series(holdings, index=idx, name='units')
    ret = equity_s.pct_change().fillna(0.0)
    metrics = compute_metrics(equity_s, ret)
    df_trades = pd.DataFrame(trades)

    # Mark Sell wins
    if not df_trades.empty:
        df_trades['is_win'] = False
        avg_entry_price = 0.0
        cum_units = 0.0
        for i, row in df_trades.iterrows():
            if row['type'] == 'BUY':
                total_cost = avg_entry_price * cum_units + row['price'] * row['units']
                cum_units += row['units']
                avg_entry_price = (total_cost / cum_units) if cum_units > 0 else 0.0
            else:
                pnl = (row['price'] - avg_entry_price) * (-row['units'])  # units negative
                df_trades.at[i, 'is_win'] = (pnl > 0)
                cum_units += row['units']
                if cum_units <= 1e-9:
                    avg_entry_price = 0.0

    return BacktestResult(df_trades=df_trades, equity_curve=equity_s, cash_curve=cash_s, holdings_curve=hold_s, metrics=metrics)

def plot_results(price: pd.Series, result: BacktestResult, title: str, filename: str = "plots.html"):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.35, 0.25, 0.20, 0.20],
                        subplot_titles=("BTC-USD Price & Trades", "Portfolio Value", "Cash (Credit) Value", "Holdings (Units)"))
    fig.add_trace(go.Scatter(x=price.index, y=price.values, name="Price", line=dict(color="black")), row=1, col=1)
    if not result.df_trades.empty:
        buys = result.df_trades[result.df_trades['type'] == 'BUY']
        sells = result.df_trades[result.df_trades['type'] == 'SELL']
        fig.add_trace(go.Scatter(x=buys['time'], y=[price.loc[t] if t in price.index else None for t in buys['time']],
                                 mode='markers', name='Buy',
                                 marker=dict(symbol='triangle-up', color='blue', size=8)), row=1, col=1)
        if not sells.empty:
            sells_win = sells[sells['is_win']]
            sells_lose = sells[~sells['is_win']]
            fig.add_trace(go.Scatter(x=sells_win['time'], y=[price.loc[t] if t in price.index else None for t in sells_win['time']],
                                     mode='markers', name='Sell (Win)',
                                     marker=dict(symbol='x', color='green', size=8)), row=1, col=1)
            fig.add_trace(go.Scatter(x=sells_lose['time'], y=[price.loc[t] if t in price.index else None for t in sells_lose['time']],
                                     mode='markers', name='Sell (Lose)',
                                     marker=dict(symbol='x', color='red', size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.equity_curve.index, y=result.equity_curve.values, name="Equity", line=dict(color="purple")), row=2, col=1)
    fig.add_trace(go.Scatter(x=result.cash_curve.index, y=result.cash_curve.values, name="Cash", line=dict(color="teal")), row=3, col=1)
    fig.add_trace(go.Scatter(x=result.holdings_curve.index, y=result.holdings_curve.values, name="Units", line=dict(color="orange")), row=4, col=1)
    fig.update_layout(title=title, height=1000, showlegend=True)
    fig.write_html(filename)
    print(f"Plot saved to {filename}")

# ---------------------------
# 6) DDPG with guided actions
# ---------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.buf = []
        self.pos = 0
    def push(self, s, a, r, s2, d):
        if len(self.buf) < self.capacity:
            self.buf.append(None)
        self.buf[self.pos] = (s, a, r, s2, d)
        self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r.reshape(-1,1), s2, d.reshape(-1,1)
    def __len__(self):
        return len(self.buf)

class Actor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),  # [-1,1]
        )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=-1))

def soft_update(target, source, tau=0.01):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

@dataclass
class RLEnvConfig:
    fee_bps: float = 7.5
    slippage_bps: float = 1.0
    risk_penalty: float = 0.0
    max_position: float = 1.0

class TradingEnv:
    """
    Minute environment, long-only, no margin.
    State = features + probs + prev_target.
    Action in [-1,1] mapped to [0,1] target position.
    Reward = log equity change minus small penalty.
    """
    def __init__(self, df: pd.DataFrame, X: np.ndarray, probs: np.ndarray, initial_cash: float = 10000.0, cfg: RLEnvConfig = RLEnvConfig()):
        self.df = df
        self.p = df['Close'].values.astype(float)
        self.X = X
        self.probs = probs
        self.initial_cash = initial_cash
        self.cfg = cfg
        self.reset()

    def reset(self, start_idx: int = 0) -> np.ndarray:
        self.t = start_idx
        self.cash = self.initial_cash
        self.units = 0.0
        self.prev_target = 0.0
        self.equity = self.initial_cash
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        feat = self.X[self.t]
        prob = self.probs[self.t]
        pos = np.array([self.prev_target], dtype=np.float32)
        return np.concatenate([feat, prob, pos], axis=0)

    def step(self, a_raw: float) -> Tuple[np.ndarray, float, bool]:
        # map [-1,1] -> [0,1]
        a_target = float(np.clip((a_raw + 1.0) / 2.0, 0.0, self.cfg.max_position))
        price_t = self.p[self.t]
        equity = self.cash + self.units * price_t
        target_units = a_target * equity / price_t if price_t > 0 else 0.0
        delta_units = target_units - self.units

        fee = self.cfg.fee_bps / 10000.0
        slip = self.cfg.slippage_bps / 10000.0

        if abs(delta_units) > 1e-9:
            if delta_units > 0:
                trade_price = price_t * (1 + slip)
                max_buy_units = max(0.0, self.cash / (trade_price * (1 + fee)))
                delta_units = min(delta_units, max_buy_units)
                trade_cost = delta_units * trade_price
                fee_cost = trade_cost * fee
                self.cash -= (trade_cost + fee_cost)
                self.units += delta_units
            else:
                delta_units = max(delta_units, -self.units)
                trade_price = price_t * (1 - slip)
                proceeds = (-delta_units) * trade_price
                fee_cost = proceeds * fee
                self.cash += (proceeds - fee_cost)
                self.units += delta_units

        t_next = self.t + 1
        done = t_next >= (len(self.p) - 1)
        price_next = self.p[t_next] if not done else self.p[self.t]
        equity_prev = self.cash + self.units * price_t
        equity_next = self.cash + self.units * price_next
        reward = math.log(max(equity_next, 1e-12)) - math.log(max(equity_prev, 1e-12))
        reward -= self.cfg.risk_penalty * abs(delta_units)
        self.t = t_next
        self.prev_target = a_target
        s_next = self._get_state()
        return s_next, float(reward), done

# ---------------------------
# 7) Main pipeline
# ---------------------------

def main():
    device = get_device()
    print("Device:", device)

    # Data and features
    df0 = load_data()
    df = add_ta_features(df0)
    df = add_ffd_feature(df, d=0.3)
    labels = triple_barrier_labels(df['Close'], df['vol_ewm'], up_mult=5.0, dn_mult=5.0, horizon=120)
    df = df.join(labels).dropna()

    # Split
    n = len(df)
    i_train = int(n * 0.6)
    i_val = int(n * 0.75)
    train_df = df.iloc[:i_train].copy()
    val_df = df.iloc[i_train:i_val].copy()
    test_df = df.iloc[i_val:].copy()

    use_cols = ['Close','High','Low','Open','Volume',
                'ret_1','logret_1','vol_ewm','rsi','macd','macd_signal','macd_hist',
                'stoch_k','stoch_d','bb_w','ema_diff','mfi','ffd_close']
    use_cols = [c for c in use_cols if c in df.columns]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[use_cols].values)
    X_val = scaler.transform(val_df[use_cols].values)
    X_test = scaler.transform(test_df[use_cols].values)
    y_train = train_df['tb_label'].values
    y_val = val_df['tb_label'].values
    y_test = test_df['tb_label'].values

    # Sequence data
    seq_len = 120
    Xtr_seq, ytr_seq = make_sequences(X_train, y_train, seq_len=seq_len)
    Xva_seq, yva_seq = make_sequences(X_val, y_val, seq_len=seq_len)
    Xte_seq, yte_seq = make_sequences(X_test, y_test, seq_len=seq_len)

    # LSTM model
    model = LSTMClassifier(input_dim=Xtr_seq.shape[-1], hidden=96, layers=2, dropout=0.2, num_classes=3).to(device)
    # Focal alpha: inverse of class frequencies (based on train labels)
    cls_counts = np.array([np.sum(y_train == -1), np.sum(y_train == 0), np.sum(y_train == 1)], dtype=float)
    freq = cls_counts / cls_counts.sum()
    alpha = torch.tensor(1.0 - freq, dtype=torch.float32, device=device)
    loss_fn = FocalLoss(alpha=alpha, gamma=2.0, reduction='mean')
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)  # verbose removed

    # Balanced minibatches
    bs = 256
    epochs = 12
    best_val = 1e9
    best_state = None
    patience = 3
    no_improve = 0

    # Build per-class index arrays
    idx_neg = np.where(ytr_seq == -1)[0]
    idx_zer = np.where(ytr_seq == 0)[0]
    idx_pos = np.where(ytr_seq == 1)[0]
    min_class = max(1, min(len(idx_neg), len(idx_zer), len(idx_pos)))

    for epoch in range(epochs):
        model.train()
        # Sample equal from each class
        sel_neg = np.random.choice(idx_neg, min_class, replace=True)
        sel_zer = np.random.choice(idx_zer, min_class, replace=True)
        sel_pos = np.random.choice(idx_pos, min_class, replace=True)
        idxs = np.concatenate([sel_neg, sel_zer, sel_pos])
        np.random.shuffle(idxs)

        total_loss = 0.0
        pbar = tqdm(range(0, len(idxs), bs), desc=f"LSTM Train Epoch {epoch+1}/{epochs}")
        for k in pbar:
            batch_idx = idxs[k:k+bs]
            xb = torch.tensor(Xtr_seq[batch_idx], dtype=torch.float32, device=device)
            yb = torch.tensor(ytr_seq[batch_idx] + 1, dtype=torch.long, device=device)  # {-1,0,1} -> {0,1,2}
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * len(batch_idx)
            pbar.set_postfix(loss=float(loss.item()))

        # Validation
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(Xva_seq, dtype=torch.float32, device=device)
            yv = torch.tensor(yva_seq + 1, dtype=torch.long, device=device)
            lv = loss_fn(model(xv), yv).item()
        print(f"Epoch {epoch+1}: val_loss={lv:.6f}")
        scheduler.step(lv)
        if lv < best_val - 1e-4:
            best_val = lv
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test inference
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(Xte_seq, dtype=torch.float32, device=device)
        logits = model(xt).cpu().numpy()
        probs = softmax(logits, axis=1)
        y_pred_idx = np.argmax(probs, axis=1)
        y_pred = y_pred_idx - 1
        y_true = yte_seq

    print("LSTM Classification Report (Test):")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix (rows true [-1,0,1], cols pred [-1,0,1]):")
    print(confusion_matrix(y_true, y_pred, labels=[-1,0,1]))

    # TIM + positions
    C, T, EV = confusion_and_tim(y_true, y_pred)
    print("Estimated TIM T:\n", T)
    print(f"Trace(C * T^T)/N ~ Expected unit payoff: {EV:.4f}")

    # Map probs to positions (aligned to test_df indices)
    prob_full = np.zeros((len(test_df), 3), dtype=float)
    prob_full[seq_len:, :] = probs
    raw_positions = []
    for i in range(len(test_df)):
        if i < seq_len:
            raw_positions.append(0.0)
            continue
        p_dn, p_flat, p_up = prob_full[i]
        pos = probs_to_position(prob_up=p_up, prob_dn=p_dn, prob_flat=p_flat, T=T, max_pos=0.5, edge_thresh=0.05)
        raw_positions.append(pos)
    positions = pd.Series(raw_positions, index=test_df.index, name='position')
    positions = smooth_positions(positions, ema_span=30)

    # Backtest supervised
    bt_sup = backtest_signals(test_df, positions, fee_bps=7.5, slippage_bps=1.0, initial_cash=10000.0)
    print("Supervised Backtest Metrics:")
    for k, v in bt_sup.metrics.items():
        print(f"  {k}: {v:.4f}")
    plot_results(test_df['Close'], bt_sup, title="Supervised LSTM+TIM (Safe, Long-Only)", filename="plots_supervised.html")

    # ---------------- RL Stage ----------------
    # Build probs for env
    probs_test = np.zeros((len(test_df), 3), dtype=np.float32)
    probs_test[seq_len:, :] = prob_full[seq_len:, :].astype(np.float32)

    # RL state dim
    state_dim = X_test.shape[1] + 3 + 1
    actor = Actor(in_dim=state_dim, hidden=128).to(device)
    critic = Critic(in_dim=state_dim, hidden=128).to(device)
    actor_tgt = copy.deepcopy(actor).to(device)
    critic_tgt = copy.deepcopy(critic).to(device)
    opt_a = optim.Adam(actor.parameters(), lr=1e-4)
    opt_c = optim.Adam(critic.parameters(), lr=1e-3)
    rb = ReplayBuffer(capacity=200_000)
    gamma = 0.99
    tau = 0.01
    batch_size = 256

    # Supervised guidance action from classifier probabilities
    a_sl = []
    for i in range(len(test_df)):
        if i < seq_len:
            a_sl.append(0.0)
        else:
            p_dn, p_flat, p_up = prob_full[i]
            signed = float(np.clip(p_up - p_dn, -1.0, 1.0))
            guided = (signed + 1.0) / 2.0  # [0,1]
            a_sl.append(guided)
    a_sl = np.array(a_sl, dtype=np.float32)

    # Env
    env_cfg = RLEnvConfig(fee_bps=7.5, slippage_bps=1.0, risk_penalty=0.0, max_position=1.0)
    env = TradingEnv(test_df, X_test.astype(np.float32), probs_test.astype(np.float32), initial_cash=10000.0, cfg=env_cfg)

    # Walk-forward by day
    dates = test_df.index.normalize().unique()
    total_steps_cap = 20_000
    steps_done = 0
    episodes = min(len(dates), 30)
    lambda_start = 0.6
    lambda_end = 0.15

    # Exploration noise
    def add_noise(a, scale=0.05):
        return float(np.clip(a + np.random.normal(0, scale), -1.0, 1.0))

    for ep in range(episodes):
        day = dates[ep]
        idx_mask = (test_df.index.normalize() == day)  # boolean array
        idxs = np.where(idx_mask)[0]
        if len(idxs) < (seq_len + 10):
            continue
        start = int(idxs[0])
        end = int(idxs[-1])
        _ = env.reset(start_idx=start)
        lam = float(lambda_start + (lambda_end - lambda_start) * (ep / max(1, episodes - 1)))

        for t in range(start, end):
            s = env._get_state()
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a_rl = actor(s_t).cpu().numpy()[0,0]
            a_rl = add_noise(a_rl, scale=0.05)
            a_super = 2.0 * a_sl[t] - 1.0  # back to [-1,1] for mixing
            a = (1.0 - lam) * a_rl + lam * a_super
            s2, r, done = env.step(a)
            rb.push(s, np.array([a], dtype=np.float32), np.array([r], dtype=np.float32), s2, np.array([done], dtype=np.float32))

            if len(rb) >= batch_size:
                s_b, a_b, r_b, s2_b, d_b = rb.sample(batch_size)
                s_b = torch.tensor(s_b, dtype=torch.float32, device=device)
                a_b = torch.tensor(a_b, dtype=torch.float32, device=device)
                r_b = torch.tensor(r_b, dtype=torch.float32, device=device)
                s2_b = torch.tensor(s2_b, dtype=torch.float32, device=device)
                d_b = torch.tensor(d_b, dtype=torch.float32, device=device)

                with torch.no_grad():
                    a2 = actor_tgt(s2_b)
                    q2 = critic_tgt(s2_b, a2)
                    y = r_b + gamma * (1.0 - d_b) * q2

                q1 = critic(s_b, a_b)
                loss_c = nn.MSELoss()(q1, y)
                opt_c.zero_grad()
                loss_c.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_c.step()

                a_pred = actor(s_b)
                loss_a = -critic(s_b, a_pred).mean()
                opt_a.zero_grad()
                loss_a.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt_a.step()

                soft_update(actor_tgt, actor, tau=tau)
                soft_update(critic_tgt, critic, tau=tau)

            steps_done += 1
            if steps_done >= total_steps_cap:
                break
        print(f"Episode {ep+1}/{episodes} finished. lambda={lam:.3f}, steps={steps_done}")
        if steps_done >= total_steps_cap:
            break

    # Evaluate RL-guided policy
    env.reset(start_idx=0)
    positions_rl = []
    for t in tqdm(range(len(test_df)), desc="RL Evaluation", leave=False):
        s = env._get_state()
        s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a_rl = actor(s_t).cpu().numpy()[0,0]
        a_super = 2.0 * a_sl[t] - 1.0
        a = (1.0 - lambda_end) * a_rl + lambda_end * a_super
        s2, r, done = env.step(a)
        positions_rl.append(float(np.clip((a + 1.0) / 2.0, 0.0, 1.0)))
        if done:
            break
    positions_rl = pd.Series(positions_rl, index=test_df.index[:len(positions_rl)], name='position')
    positions_rl = smooth_positions(positions_rl, ema_span=30)

    bt_rl = backtest_signals(test_df.iloc[:len(positions_rl)], positions_rl, fee_bps=7.5, slippage_bps=1.0, initial_cash=10000.0)
    print("RL-Guided Backtest Metrics:")
    for k, v in bt_rl.metrics.items():
        print(f"  {k}: {v:.4f}")
    plot_results(test_df['Close'].iloc[:len(positions_rl)], bt_rl, title="Supervised-Guided DDPG (Safe, Long-Only)", filename="plots_rl.html")

if __name__ == "__main__":
    main()
