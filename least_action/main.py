# least_action_trading_strategy_trades.py
# GPT-5 (reasoning) - Least-Action MPC with horizon alpha, friction, risk targeting, and trade visualization

import os
import sys
import math
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")

# TA indicators
try:
    import ta
except ImportError:
    print("Installing 'ta' package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ta"])
    import ta

# Torch for alpha model
try:
    import torch
    from torch import nn
except ImportError:
    print("Installing 'torch' package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
    import torch
    from torch import nn

data_url = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"

def _add_time_features(df):
    df['hour'] = df['Datetime'].dt.hour
    df['minute'] = df['Datetime'].dt.minute
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['month'] = df['Datetime'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60.0)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60.0)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    return df

def load_raw_data(url):
    print("Loading and preparing data...")
    df = pd.read_csv(url, skiprows=3, header=None)
    df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)

    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['Original_Close'] = df['Close'].astype(float)

    df['log_close'] = np.log(df['Original_Close'])
    df['ret_1m'] = df['log_close'].diff(1)

    for w in [5, 15, 60, 240]:
        df[f'ret_{w}m'] = df['log_close'].diff(w)
        df[f'vol_{w}m_raw'] = df['ret_1m'].rolling(w).std()

    df['RSI'] = ta.momentum.RSIIndicator(close=df['Original_Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Original_Close'])
    df['MACD'] = macd.macd_diff()

    df['ema_12'] = df['Original_Close'].ewm(span=12, adjust=False).mean()
    df['ema_48'] = df['Original_Close'].ewm(span=48, adjust=False).mean()
    df['ema_ratio'] = df['ema_12'] / (df['ema_48'] + 1e-8) - 1.0

    atr = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Original_Close'], window=14
    ).average_true_range()
    df['atr_raw'] = atr
    df['atr_pct_raw'] = df['atr_raw'] / (df['Original_Close'] + 1e-8)

    vol_mean = df['Volume'].rolling(240).mean()
    vol_std = df['Volume'].rolling(240).std()
    df['vol_z_240'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)

    _add_time_features(df)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    print("Data preparation complete.")
    return df.reset_index(drop=True)

# ----------------------
# Alpha model (multi-horizon)
# ----------------------

class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def build_features(df):
    return [
        'ret_1m','ret_5m','ret_15m','ret_60m',
        'ema_ratio','MACD','RSI',
        'vol_z_240','atr_pct_raw',
        'hour_sin','hour_cos','minute_sin','minute_cos','dow_sin','dow_cos'
    ]

def train_alpha_horizon(df, H=60, epochs=8, lr=1e-3, val_split=0.2):
    # Predict cumulative next-H-minute return (sum of future log returns)
    df = df.copy()
    ret = df['ret_1m'].values
    # cumulative next H: use convolution and shift
    cum_h = np.convolve(ret, np.ones(H, dtype=float), mode='full')[:len(ret)]
    cum_h = np.roll(cum_h, -H)
    df['target_next_H'] = cum_h
    df = df.iloc[:-H]  # drop last H rows (no target)
    df = df.dropna().reset_index(drop=True)

    feat_cols = build_features(df)
    X = df[feat_cols].values.astype(np.float32)
    y = df['target_next_H'].values.astype(np.float32).reshape(-1, 1)

    n = len(df)
    n_val = int(n * val_split)
    n_train = n - n_val
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std

    device = get_device()
    model = TinyMLP(in_dim=X.shape[1], hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train_n).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val_n).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        opt.step()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()
        print(f"Alpha(H={H}) epoch {ep+1}/{epochs} - train MSE: {loss.item():.6f} - val MSE: {val_loss:.6f}")

    # Predict for whole series (excluding last H we dropped)
    X_all = df[feat_cols].values.astype(np.float32)
    X_all_n = (X_all - x_mean) / x_std
    with torch.no_grad():
        preds = model(torch.from_numpy(X_all_n).to(device)).cpu().numpy().squeeze()

    alpha_df = pd.DataFrame({'Datetime': df['Datetime'], f'alpha_next_{H}m': preds})
    return alpha_df

def regime_multiplier(df):
    # Risk-on if trend up and RSI positive; reduce in illiquid periods
    trend = (df['ema_12'] > df['ema_48']).astype(float)
    rsi_up = (df['RSI'] > 50).astype(float)
    liquid = (df['vol_z_240'] > -0.5).astype(float)
    score = (0.5*trend + 0.3*rsi_up + 0.2*liquid)
    mult = 0.3 + 0.7 * score  # 0.3 in risk-off to 1.0 in risk-on
    return mult.values.astype(np.float64)

# ----------------------
# MPC core with friction
# ----------------------

def soft_threshold(x, thr):
    ax = abs(x)
    if ax <= thr: return 0.0
    return math.copysign(ax - thr, x)

def compute_mpc_step(w0, alpha_vec, sigma_vec, H, eta, c2, lam, beta, u_max, gamma_l1):
    L = np.tril(np.ones((H, H), dtype=np.float64), k=-1)
    L_aug = np.vstack([L, np.ones((1, H), dtype=np.float64)])
    D_diag = lam * (sigma_vec[:H] ** 2)
    D_aug = np.concatenate([D_diag, np.array([beta], dtype=np.float64)])
    D = np.diag(D_aug)
    I = np.eye(H, dtype=np.float64)
    M = (eta + c2) * I + L_aug.T @ D @ L_aug
    ones_aug = np.ones(L_aug.shape[0], dtype=np.float64)
    rhs = 0.5 * alpha_vec[:H] - (w0) * (L_aug.T @ (D @ ones_aug))

    try:
        u_star = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        reg = 1e-8
        u_star = np.linalg.solve(M + reg * I, rhs)

    u_raw = float(u_star[0])
    thr = gamma_l1 / (2.0 * (eta + c2) + 1e-12)
    u0 = soft_threshold(u_raw, thr)
    u0 = float(np.clip(u0, -u_max, u_max))
    return u0, u_star

# ----------------------
# Backtester with trades
# ----------------------

def backtest_least_action(df, config):
    H = config.get('H', 60)
    eta = config.get('eta', 1e-6)
    c2 = config.get('c2', 2e-6)
    lam = config.get('lam', 3e-3)
    beta = config.get('beta', 1e-3)
    u_max_base = config.get('u_max', 0.10)
    w_max_base = config.get('w_max', 2.0)
    use_model_alpha = config.get('use_model_alpha', True)
    alpha_scale = config.get('alpha_scale', 0.5)  # tamer default
    alpha_clip = config.get('alpha_clip', 0.001)  # per-minute clip
    spread_bps = config.get('spread_bps', 2.0)
    risk_budget_daily = config.get('risk_budget_daily', 0.10)
    dd_limit = config.get('dd_limit', 0.05)
    dd_cooldown = config.get('dd_cooldown_min', 120)
    trade_min = config.get('trade_min', 0.01)  # minimum |u| to count as trade

    subdf = df.copy()

    price = subdf['Original_Close'].values.astype(np.float64)
    ret_1m = subdf['ret_1m'].values.astype(np.float64)
    sigma = subdf['vol_60m_raw'].values.astype(np.float64)
    vol_z = subdf['vol_z_240'].values.astype(np.float64)

    # Alpha: horizon-aligned prediction
    if use_model_alpha:
        print("Training horizon-aligned alpha model (MPS/GPU if available)...")
        alpha_df = train_alpha_horizon(subdf, H=H, epochs=8, lr=1e-3, val_split=0.2)
        subdf = subdf.merge(alpha_df, on='Datetime', how='left')
        a_H = subdf[f'alpha_next_{H}m'].fillna(0.0).values.astype(np.float64)
    else:
        # simple indicator alpha
        a_H = (
            0.4 * subdf['ema_ratio'].fillna(0.0) +
            0.3 * subdf['MACD'].fillna(0.0) +
            0.2 * ((subdf['RSI'].fillna(50.0) - 50.0) / 100.0) +
            0.1 * subdf['ret_5m'].fillna(0.0)
        ).ewm(span=30, adjust=False).mean().clip(-0.003, 0.003).values.astype(np.float64)

    # Normalize the horizon alpha to per-minute scale and clip
    # Convert horizon expectation to per-minute using kernel, but scale by sqrt(H) to avoid excessive amplitude
    kernel = np.exp(-np.arange(H)/30.0)
    kernel = kernel / (kernel.sum() + 1e-12)
    regime_mult = regime_multiplier(subdf)

    # Arrays
    n = len(subdf)
    w = 0.0
    pnl = 0.0
    cum_pnl = np.zeros(n, dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    u_exec = np.zeros(n, dtype=np.float64)
    alpha_used = np.zeros(n, dtype=np.float64)

    spread_cost_per_unit = spread_bps / 10000.0
    gamma_l1 = spread_cost_per_unit
    flat_until = -1

    # Vol targeting
    vol_target_min = risk_budget_daily / math.sqrt(1440.0)

    trades = []  # list of dicts {Datetime, action, size, price, new_weight}

    for t in range(n - 1):
        sigma_t = sigma[t] if not np.isnan(sigma[t]) else np.nanmean(sigma[max(0, t-240):t+1])
        if np.isnan(sigma_t): sigma_t = 1e-4

        # Dynamic caps
        w_max_t = min(w_max_base, vol_target_min / (sigma_t + 1e-8))
        u_max_t = u_max_base * float(np.clip((vol_z[t] + 2.0) / 4.0, 0.2, 1.2))

        # Build per-minute alpha for horizon from the local horizon prediction
        a_local = alpha_scale * a_H[t] * regime_mult[t]
        # Scale down overly large horizon predictions by sqrt(H) to keep SNR stable
        a_local = a_local / math.sqrt(H)
        alpha_vec = a_local * kernel
        alpha_vec = np.clip(alpha_vec, -alpha_clip, alpha_clip)
        sigma_vec_t = np.full(H, sigma_t, dtype=np.float64)

        # No trading if in cooldown or tiny alpha relative to costs
        if flat_until > t or abs(alpha_vec[0]) < 1.5 * spread_cost_per_unit:
            u0 = 0.0
            u_star = np.zeros(H)
        else:
            u0, u_star = compute_mpc_step(
                w0=w, alpha_vec=alpha_vec, sigma_vec=sigma_vec_t, H=H,
                eta=eta, c2=c2, lam=lam, beta=beta, u_max=u_max_t, gamma_l1=gamma_l1
            )

        # Respect bounds
        if w + u0 > w_max_t:
            u0 = max(-u_max_t, w_max_t - w)
        if w + u0 < -w_max_t:
            u0 = min(u_max_t, -w_max_t - w)

        # Execution and cost
        trading_cost = eta * (u0 ** 2) + c2 * (u0 ** 2) + spread_cost_per_unit * abs(u0)

        # Record trade if meaningful
        if abs(u0) >= trade_min:
            action = "BUY" if u0 > 0 else "SELL"
            trades.append({
                'Datetime': subdf['Datetime'].iloc[t],
                'action': action,
                'size_u': float(u0),
                'price': float(price[t]),
                'new_weight': float(w + u0)
            })

        # Update state
        u_exec[t] = u0
        w = w + u0
        weights[t+1] = w
        alpha_used[t] = alpha_vec[0]

        realized_ret = ret_1m[t+1] if not np.isnan(ret_1m[t+1]) else 0.0
        pnl += w * realized_ret - trading_cost
        cum_pnl[t+1] = pnl

        # Drawdown monitor
        if t > 1440:
            window = cum_pnl[max(0, t-720):t+1]  # last 12 hours
            peak = np.max(window)
            dd = peak - cum_pnl[t+1]
            if dd > dd_limit:
                flat_until = t + dd_cooldown
                w = 0.0  # flatten immediately
                weights[t+1] = w

    # Outputs
    ts = pd.DataFrame({
        'Datetime': subdf['Datetime'],
        'price': price,
        'cum_pnl': cum_pnl,
        'weight': weights,
        'u': u_exec,
        'alpha_ret': alpha_used
    }).set_index('Datetime')

    trades_df = pd.DataFrame(trades)
    buys = int((trades_df['action'] == 'BUY').sum()) if not trades_df.empty else 0
    sells = int((trades_df['action'] == 'SELL').sum()) if not trades_df.empty else 0

    perf = {
        'final_pnl': float(cum_pnl[-1]),
        'final_weight': float(w),
        'turnover': float(np.sum(np.abs(u_exec))),
        'avg_weight': float(np.mean(np.abs(weights))),
        'num_buys': buys,
        'num_sells': sells
    }

    # Alpha stats
    alpha_series = pd.Series(alpha_used)
    perf['alpha_stats'] = {
        'mean': float(alpha_series.mean()),
        'std': float(alpha_series.std()),
        'p5': float(alpha_series.quantile(0.05)),
        'p95': float(alpha_series.quantile(0.95))
    }

    # Summaries by horizon
    def summarize_window(days):
        end_time = ts.index[-1]
        start_time = end_time - pd.Timedelta(days=days)
        window = ts.loc[ts.index >= start_time]
        if len(window) < 2:
            return None
        return {
            'days': days,
            'pnl_change': float(window['cum_pnl'].iloc[-1] - window['cum_pnl'].iloc[0]),
            'avg_weight': float(window['weight'].abs().mean()),
            'turnover': float(window['u'].abs().sum()),
            'minutes': int(len(window))
        }

    horizons = [1, 7, 30, 90, 365]
    perf['summaries'] = [s for s in [summarize_window(d) for d in horizons] if s is not None]

    return perf, ts, trades_df

# ----------------------
# Plotly visualization
# ----------------------

def make_plots(ts, trades_df, title="Least-Action MPC Strategy on BTC-USD (Trades & Weight)"):
    # 4-row figure: price + trades, weight, alpha/u, cumulative PnL
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("BTC Price with Trades", "Portfolio Weight", "Alpha and Control (u)", "Cumulative PnL"))

    # Row 1: Price
    fig.add_trace(go.Scatter(x=ts.index, y=ts['price'], name="Price", line=dict(color="#1f77b4")), row=1, col=1)

    # Trades as markers
    if trades_df is not None and len(trades_df) > 0:
        buys = trades_df[trades_df['action'] == 'BUY']
        sells = trades_df[trades_df['action'] == 'SELL']
        fig.add_trace(go.Scatter(
            x=buys['Datetime'], y=buys['price'],
            mode='markers', name='BUY',
            marker=dict(symbol='triangle-up', color='green', size=8, line=dict(width=1, color='darkgreen'))
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sells['Datetime'], y=sells['price'],
            mode='markers', name='SELL',
            marker=dict(symbol='triangle-down', color='red', size=8, line=dict(width=1, color='darkred'))
        ), row=1, col=1)

    # Row 2: Weight in its own axis
    fig.add_trace(go.Scatter(x=ts.index, y=ts['weight'], name="Weight", line=dict(color="#ff7f0e")), row=2, col=1)

    # Row 3: Alpha and control
    fig.add_trace(go.Scatter(x=ts.index, y=ts['alpha_ret'], name="Alpha (per min)", line=dict(color="#2ca02c")), row=3, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts['u'], name="Control u", line=dict(color="#d62728")), row=3, col=1)

    # Row 4: Cumulative PnL
    fig.add_trace(go.Scatter(x=ts.index, y=ts['cum_pnl'], name="Cumulative PnL", line=dict(color="#9467bd")), row=4, col=1)

    fig.update_layout(title=title, height=1100, legend_orientation="h", legend_yanchor="top")
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig

# ----------------------
# Main
# ----------------------

def main():
    df = load_raw_data(data_url)
    print(f"Loaded {len(df)} rows from {data_url}")

    config = {
        'H': 60,
        'eta': 1e-6,
        'c2': 2e-6,
        'lam': 3e-3,
        'beta': 1e-3,
        'u_max': 0.10,
        'w_max': 2.0,
        'use_model_alpha': True,
        'alpha_scale': 0.5,
        'alpha_clip': 0.001,
        'spread_bps': 2.0,
        'risk_budget_daily': 0.10,
        'dd_limit': 0.05,
        'dd_cooldown_min': 120,
        'trade_min': 0.01
    }

    perf, ts, trades_df = backtest_least_action(df, config)
    print("Backtest summary:")
    print(json.dumps(perf, indent=2))

    # Print trade counts and show a small trade log preview
    print(f"Total trades: {len(trades_df)} | Buys: {(trades_df['action']=='BUY').sum()} | Sells: {(trades_df['action']=='SELL').sum()}")
    if len(trades_df) > 0:
        print("Trade log preview:")
        print(trades_df.head(10).to_string(index=False))

    fig = make_plots(ts, trades_df, title="Least-Action MPC Strategy on BTC-USD (Trades & Weight)")
    try:
        fig.show()
    except Exception:
        print("Plotly rendering failed. Saving to HTML...")
        fig.write_html("least_action_trades_weight.html")
        print("Saved plot to least_action_trades_weight.html")

if __name__ == "__main__":
    main()
