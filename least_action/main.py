# least_action_trading_strategy_risk_tuned.py
# GPT-5 (reasoning)
# Least-Action MPC with "a bit more risky" tuning:
# - Walk-forward horizon alpha (early stopping, dropout, MPS/CUDA if available)
# - Volatility-relative alpha cap + tanh squashing
# - Online alpha calibration (EWMA; nonnegative, responsive; moderate floor/cap)
# - Signal gates (percentile, persistence) + short trade cooldown
# - Turnover governor and regime-based slowdown
# - Strict volatility targeting and moderate risk knobs
# - Aggregated and forward expected returns (asset and portfolio)
# - Trade-interval plotting

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
# Alpha model
# ----------------------

class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
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

def train_alpha_horizon_walkforward(df, H=60, epochs=12, lr=1e-3, window_min=70000, retrain_every=1440, dropout=0.1, patience=3):
    feat_cols = build_features(df)
    n = len(df)
    aH = np.zeros(n, dtype=np.float64)
    device = get_device()

    def make_targets(ret_series, H):
        cum_h = np.convolve(ret_series, np.ones(H, dtype=float), mode='full')[:len(ret_series)]
        return np.roll(cum_h, -H)

    start = max(window_min, H + 1)
    i = start
    while i < n - H:
        s = max(0, i - window_min); e = i
        win = df.iloc[s:e].copy()
        ret = win['ret_1m'].values
        tgt = make_targets(ret, H)
        win = win.iloc[:-H].reset_index(drop=True)
        tgt = tgt[:-H]
        if len(win) < 2000:
            i = min(n, i + retrain_every)
            continue

        X = win[feat_cols].values.astype(np.float32)
        y = tgt.astype(np.float32).reshape(-1, 1)

        n_win = len(win); n_val = int(n_win * 0.2); n_tr = n_win - n_val
        X_tr, y_tr = X[:n_tr], y[:n_tr]
        X_val, y_val = X[n_tr:], y[n_tr:]

        x_mean = X_tr.mean(axis=0, keepdims=True)
        x_std = X_tr.std(axis=0, keepdims=True) + 1e-8
        X_tr_n = (X_tr - x_mean) / x_std
        X_val_n = (X_val - x_mean) / x_std

        model = TinyMLP(in_dim=X.shape[1], hidden=64, dropout=dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        Xt = torch.from_numpy(X_tr_n).to(device)
        yt = torch.from_numpy(y_tr).to(device)
        Xv = torch.from_numpy(X_val_n).to(device)
        yv = torch.from_numpy(y_val).to(device)

        best_val = float('inf'); bad = 0
        model.train()
        for ep in range(epochs):
            opt.zero_grad()
            pred = model(Xt); loss = loss_fn(pred, yt)
            loss.backward(); opt.step()
            with torch.no_grad():
                val_pred = model(Xv)
                val_loss = loss_fn(val_pred, yv).item()
            if val_loss + 1e-6 < best_val:
                bad = 0; best_val = val_loss
            else:
                bad += 1
            if bad >= patience:
                break

        f_end = min(n, i + retrain_every)
        Xf = df.iloc[i:f_end][feat_cols].values.astype(np.float32)
        Xf_n = (Xf - x_mean) / x_std
        with torch.no_grad():
            preds = model(torch.from_numpy(Xf_n).to(device)).cpu().numpy().squeeze()
        aH[i:f_end] = preds
        i = f_end

    first_nonzero = np.nonzero(aH)[0]
    if len(first_nonzero) > 0:
        aH[:first_nonzero[0]] = aH[first_nonzero[0]]
    return aH

def regime_multiplier(df):
    # risk-on multiplier baseline
    trend = (df['ema_12'] > df['ema_48']).astype(float)
    rsi_up = (df['RSI'] > 50).astype(float)
    liquid = (df['vol_z_240'] > -0.5).astype(float)
    score = (0.5*trend + 0.3*rsi_up + 0.2*liquid)
    mult = 0.5 + 0.5 * score  # range ~[0.5, 1.0]
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
# Backtester with trades and account tracking
# ----------------------

def halflife_to_rho(halflife_min):
    return 1.0 - math.exp(-math.log(2) / max(1.0, float(halflife_min)))

def backtest_least_action(df, config):
    H = config.get('H', 60)
    eta = config.get('eta', 1.0e-6)             # moderate impact
    c2 = config.get('c2', 3.0e-6)               # L2 friction
    lam = config.get('lam', 1.5e-3)             # moderate inventory penalty
    beta = config.get('beta', 1e-3)
    u_max_base = config.get('u_max', 0.08)      # moderate trading speed
    w_max_base = config.get('w_max', 1.0)       # cap leverage to 1x
    epochs_alpha = config.get('epochs_alpha', 12)
    alpha_scale = config.get('alpha_scale', 0.8)
    alpha_clip_global = config.get('alpha_clip', 0.0012)
    k_sigma = config.get('k_sigma', 0.10)       # alpha_1m <= 0.10 * sigma_t
    spread_bps = config.get('spread_bps', 5.0)
    risk_budget_daily = config.get('risk_budget_daily', 0.08)  # stricter
    dd_limit = config.get('dd_limit', 0.06)
    dd_cooldown = config.get('dd_cooldown_min', 120)
    trade_min = config.get('trade_min', 0.004)
    initial_capital = config.get('initial_capital', 1.0)
    deadband_mult = config.get('deadband_mult', 0.60)          # need alpha to beat 0.6*spread
    kappa_decay = config.get('kappa_decay', 0.0005)
    walk_window_min = config.get('walk_window_min', 70000)
    retrain_every_min = config.get('retrain_every_min', 1440)
    dropout = config.get('dropout', 0.1)

    # online alpha calibration
    dyn_calib = config.get('dyn_calib', True)
    calib_halflife_min = config.get('calib_halflife_min', 1440) # 1 day
    calib_clip = config.get('calib_clip', (0.3, 1.5))           # floor/cap
    calib_allow_negative = False

    # gates
    gate_quantile = config.get('gate_quantile', 0.85)
    gate_window = config.get('gate_window', 3000)
    persist_min = config.get('persist_min', 4)
    trade_cooldown_min = config.get('trade_cooldown_min', 5)

    # optional target sizing (very tame); can disable by setting enable_target=False
    enable_target = config.get('enable_target', True)
    target_gain = config.get('target_gain', 0.3)
    gate_snr = config.get('gate_snr', 0.18)
    w_min_expo = config.get('w_min_expo', 0.0)
    snr_high = config.get('snr_high', 0.35)
    blend_mpc = config.get('blend_mpc', 0.85)

    # turnover governor
    turn_window_min = config.get('turn_window_min', 60)
    turn_cap = config.get('turn_cap', 0.80)     # cap total |u| over last 60m ~0.80
    turn_decay = config.get('turn_decay', 0.5)  # scale u_max when above cap

    subdf = df.copy()
    price = subdf['Original_Close'].values.astype(np.float64)
    ret_1m = subdf['ret_1m'].values.astype(np.float64)
    sigma = subdf['vol_60m_raw'].values.astype(np.float64)
    vol_z = subdf['vol_z_240'].values.astype(np.float64)

    print("Training horizon-aligned alpha model (walk-forward=True)...")
    a_H = train_alpha_horizon_walkforward(
        subdf, H=H, epochs=epochs_alpha, lr=1e-3,
        window_min=walk_window_min, retrain_every=retrain_every_min, dropout=dropout, patience=3
    )

    kernel = np.exp(-np.arange(H)/30.0); kernel /= (kernel.sum() + 1e-12)
    regime_mult = regime_multiplier(subdf)

    n = len(subdf)
    w = 0.0
    wealth = float(initial_capital)
    cum_pnl_log = np.zeros(n, dtype=np.float64)
    wealth_series = np.zeros(n, dtype=np.float64); wealth_series[0] = wealth
    weights = np.zeros(n, dtype=np.float64)
    u_exec = np.zeros(n, dtype=np.float64)
    alpha_used_1m = np.zeros(n, dtype=np.float64)
    margin_used = np.zeros(n, dtype=np.float64)
    avail_credit = np.zeros(n, dtype=np.float64)
    scale_dyn_series = np.zeros(n, dtype=np.float64)
    a_H_series = a_H.copy()

    spread_cost_per_unit = spread_bps / 10000.0
    gamma_l1 = spread_cost_per_unit
    flat_until = -1
    vol_target_min = risk_budget_daily / math.sqrt(1440.0)
    deadband_alpha = deadband_mult * spread_cost_per_unit

    trades = []
    last_trade_t = -10**9
    u_abs_hist = []

    # online calibration state
    rho = halflife_to_rho(calib_halflife_min)
    S_xx = 1e-8; S_xy = 0.0; scale_dyn = 1.0

    # gates state
    abs_alpha_hist = []
    persist_count = 0; last_sign = 0

    for t in range(n - 1):
        sigma_t = sigma[t] if not np.isnan(sigma[t]) else np.nanmean(sigma[max(0, t-240):t+1])
        if np.isnan(sigma_t): sigma_t = 1e-4

        # Volatility targeting -> weight cap
        w_max_t = min(w_max_base, vol_target_min / (sigma_t + 1e-8))
        w_max_t = max(w_max_t, 0.3)  # allow at least 0.3x in quiet markets

        # Base u_max with liquidity
        u_max_t = u_max_base * float(np.clip((vol_z[t] + 2.0) / 4.0, 0.4, 1.2))

        # Turnover governor (last 60m)
        start_turn = max(0, t - turn_window_min)
        turn = float(np.sum(np.abs(u_exec[start_turn:t])))
        if turn > turn_cap:
            u_max_t *= turn_decay

        # Regime slowdown in risk-off
        risk_off = (regime_mult[t] < 0.65)
        if risk_off:
            u_max_t *= 0.8
            w_max_t *= 0.9

        # Per-minute alpha from horizon prediction
        a_local = alpha_scale * a_H[t] * regime_mult[t]
        a_local = a_local / math.sqrt(H)
        if dyn_calib:
            if not calib_allow_negative and scale_dyn < 0.0:
                scale_dyn = 0.0
            a_local *= float(np.clip(scale_dyn, calib_clip[0], calib_clip[1]))

        # Volatility-relative clip and tanh squash for realism
        alpha_clip_abs = min(alpha_clip_global, k_sigma * sigma_t)
        # squash to (-alpha_clip_abs, alpha_clip_abs) dynamically
        squash = alpha_clip_abs
        a_local = squash * math.tanh(a_local / (squash + 1e-12))

        alpha_vec = np.clip(a_local * kernel, -alpha_clip_abs, alpha_clip_abs)
        sigma_vec_t = np.full(H, sigma_t, dtype=np.float64)

        # Gates
        a1 = float(alpha_vec[0])
        sgn = 1 if a1 > 0 else (-1 if a1 < 0 else 0)
        if sgn == 0 or sgn != last_sign:
            persist_count = 1 if sgn != 0 else 0
            last_sign = sgn
        else:
            persist_count += 1

        abs_alpha_hist.append(abs(a1))
        if len(abs_alpha_hist) > gate_window:
            abs_alpha_hist.pop(0)
        thresh = float(np.quantile(np.array(abs_alpha_hist), gate_quantile)) if len(abs_alpha_hist) >= 500 else 0.0

        cool_ok = (t - last_trade_t) >= trade_cooldown_min
        allowed = (abs(a1) >= max(deadband_alpha, thresh)) and (persist_count >= persist_min) and cool_ok

        # Tame target sizing
        if enable_target:
            denom = (sigma_t * math.sqrt(H) + 1e-9)
            snr = (a_H[t] / denom) if denom > 0 else 0.0
            w_target = 0.0
            if abs(snr) >= gate_snr:
                w_target = np.clip(target_gain * snr, -w_max_t, w_max_t)
                if abs(snr) >= snr_high and w_min_expo > 0:
                    w_target = math.copysign(max(abs(w_target), min(w_min_expo, w_max_t)), w_target)
            u_set = np.clip(w_target - w, -u_max_t, u_max_t)
        else:
            u_set = 0.0

        # Decide control
        if flat_until > t or not allowed:
            u_mpc = np.clip(-kappa_decay * w, -u_max_t, u_max_t)
            u0 = np.clip(blend_mpc * u_mpc + (1.0 - blend_mpc) * u_set, -u_max_t, u_max_t)
            u_star = np.zeros(H)
        else:
            u_mpc, u_star = compute_mpc_step(
                w0=w, alpha_vec=alpha_vec, sigma_vec=sigma_vec_t, H=H,
                eta=eta, c2=c2, lam=lam, beta=beta, u_max=u_max_t, gamma_l1=gamma_l1
            )
            u0 = np.clip(blend_mpc * u_mpc + (1.0 - blend_mpc) * u_set, -u_max_t, u_max_t)

        # Respect bounds
        if w + u0 > w_max_t:
            u0 = max(-u_max_t, w_max_t - w)
        if w + u0 < -w_max_t:
            u0 = min(u_max_t, -w_max_t - w)

        # Costs
        trading_cost = eta * (u0 ** 2) + c2 * (u0 ** 2) + spread_cost_per_unit * abs(u0)

        # Record trade
        if abs(u0) >= trade_min:
            action = "BUY" if u0 > 0 else "SELL"
            trades.append({
                'Datetime': subdf['Datetime'].iloc[t],
                'action': action,
                'size_u': float(u0),
                'price': float(price[t]),
                'new_weight': float(w + u0)
            })
            last_trade_t = t

        # Update
        u_exec[t] = u0
        u_abs_hist.append(abs(u0))
        if len(u_abs_hist) > turn_window_min: u_abs_hist.pop(0)

        w = w + u0
        weights[t+1] = w
        alpha_used_1m[t] = a1
        scale_dyn_series[t] = scale_dyn

        minute_log_ret = w * (ret_1m[t+1] if not np.isnan(ret_1m[t+1]) else 0.0) - trading_cost
        wealth *= math.exp(minute_log_ret)
        wealth_series[t+1] = wealth
        cum_pnl_log[t+1] = math.log(wealth / initial_capital)
        margin_used[t+1] = abs(w) * wealth
        avail_credit[t+1] = max(w_max_base - abs(w), 0.0) * wealth

        # Update calibration
        if dyn_calib:
            x = a1
            y = ret_1m[t+1] if not np.isnan(ret_1m[t+1]) else 0.0
            S_xx = (1 - rho) * S_xx + rho * (x * x)
            S_xy = (1 - rho) * S_xy + rho * (x * y)
            if S_xx > 1e-10:
                scale_dyn = S_xy / S_xx
                if not calib_allow_negative and scale_dyn < 0.0:
                    scale_dyn = 0.0

        # Drawdown monitor
        if t > 1440:
            window = cum_pnl_log[max(0, t-720):t+1]
            peak = np.max(window)
            dd = peak - cum_pnl_log[t+1]
            if dd > dd_limit:
                flat_until = t + dd_cooldown
                w = 0.0
                weights[t+1] = w

    # Outputs
    ts = pd.DataFrame({
        'Datetime': subdf['Datetime'],
        'price': price,
        'cum_pnl_log': cum_pnl_log,
        'wealth': wealth_series,
        'weight': weights,
        'u': u_exec,
        'alpha_1m_exp': alpha_used_1m,
        'margin_used': margin_used,
        'avail_credit': avail_credit,
        'alpha_H': a_H_series,
        'scale_dyn': scale_dyn_series
    }).set_index('Datetime')

    trades_df = pd.DataFrame(trades)
    buys = int((trades_df['action'] == 'BUY').sum()) if not trades_df.empty else 0
    sells = int((trades_df['action'] == 'SELL').sum()) if not trades_df.empty else 0

    engage_ratio = float((np.abs(weights) > 0.10).mean())

    perf = {
        'final_pnl_log': float(cum_pnl_log[-1]),
        'final_wealth': float(wealth),
        'final_weight': float(w),
        'turnover': float(np.sum(np.abs(u_exec))),
        'avg_weight': float(np.mean(np.abs(weights))),
        'num_buys': buys,
        'num_sells': sells,
        'engaged_frac_|w|>0.1': engage_ratio
    }

    def summarize_window(days):
        end_time = ts.index[-1]
        start_time = end_time - pd.Timedelta(days=days)
        window = ts.loc[ts.index >= start_time]
        if len(window) < 2:
            return None
        pnl_change_log = float(window['cum_pnl_log'].iloc[-1] - window['cum_pnl_log'].iloc[0])
        return {
            'days': days,
            'pnl_change_log': pnl_change_log,
            'pnl_change_simple': float(math.exp(pnl_change_log) - 1.0),
            'avg_weight': float(window['weight'].abs().mean()),
            'turnover': float(window['u'].abs().sum()),
            'minutes': int(len(window))
        }
    horizons = [1, 7, 30, 90, 365]
    perf['summaries'] = [s for s in [summarize_window(d) for d in horizons] if s is not None]

    sd = pd.Series(scale_dyn_series)
    print(f"Alpha calibration scale_dyn: mean={sd.replace(0,np.nan).mean():.3f}, "
          f"p10={sd.quantile(0.10):.3f}, p90={sd.quantile(0.90):.3f}")
    print(f"Engagement ratio (|w|>0.1): {engage_ratio:.2%}")

    return perf, ts, trades_df

# ----------------------
# Expected returns aggregation and forward curves (asset and portfolio)
# ----------------------

def aggregate_expected_returns_from_alpha_1m(ts):
    a1 = ts['alpha_1m_exp'].astype(float)
    daily_log = a1.resample('D').sum()
    weekly_log = a1.resample('W').sum()
    monthly_log = a1.resample('M').sum()
    return {
        'daily': pd.DataFrame({'exp_log': daily_log, 'exp_simple': np.exp(daily_log) - 1.0}),
        'weekly': pd.DataFrame({'exp_log': weekly_log, 'exp_simple': np.exp(weekly_log) - 1.0}),
        'monthly': pd.DataFrame({'exp_log': monthly_log, 'exp_simple': np.exp(monthly_log) - 1.0})
    }

def aggregate_expected_returns_portfolio(ts):
    a1 = (ts['alpha_1m_exp'] * ts['weight']).astype(float)
    daily_log = a1.resample('D').sum()
    weekly_log = a1.resample('W').sum()
    monthly_log = a1.resample('M').sum()
    return {
        'daily': pd.DataFrame({'exp_log': daily_log, 'exp_simple': np.exp(daily_log) - 1.0}),
        'weekly': pd.DataFrame({'exp_log': weekly_log, 'exp_simple': np.exp(weekly_log) - 1.0}),
        'monthly': pd.DataFrame({'exp_log': monthly_log, 'exp_simple': np.exp(monthly_log) - 1.0})
    }

def forward_expected_curves(ts, minutes_per={'1d':1440, '7d':10080, '30d':43200}):
    a_asset = ts['alpha_1m_exp'].astype(float).values
    a_portf = (ts['alpha_1m_exp'] * ts['weight']).astype(float).values
    n = len(a_asset)
    c_asset = np.cumsum(np.insert(a_asset, 0, 0.0))
    c_portf = np.cumsum(np.insert(a_portf, 0, 0.0))
    out_asset, out_portf = {}, {}
    for label, W in minutes_per.items():
        fwd_a = np.full(n, np.nan, dtype=np.float64)
        fwd_p = np.full(n, np.nan, dtype=np.float64)
        if W < n:
            sums_a = c_asset[W:] - c_asset[:-W]
            sums_p = c_portf[W:] - c_portf[:-W]
            upto = n - W + 1
            fwd_a[:upto] = np.exp(sums_a[:upto]) - 1.0
            fwd_p[:upto] = np.exp(sums_p[:upto]) - 1.0
        out_asset[label] = pd.Series(fwd_a, index=ts.index, name=f'exp_next_{label}_asset')
        out_portf[label] = pd.Series(fwd_p, index=ts.index, name=f'exp_next_{label}_portfolio')
    return out_asset, out_portf

# ----------------------
# Trade-interval slicing and plotting
# ----------------------

def slice_to_trade_interval(ts, trades_df, pad_min=720):
    if trades_df is None or len(trades_df) == 0:
        return ts
    t0 = trades_df['Datetime'].min()
    t1 = trades_df['Datetime'].max()
    start = t0 - pd.Timedelta(minutes=pad_min)
    end = t1 + pd.Timedelta(minutes=pad_min)
    return ts.loc[(ts.index >= start) & (ts.index <= end)]

# ----------------------
# $1,000 examples (realized and expected)
# ----------------------

def wealth_examples(ts, initial=1000.0):
    out = {}
    wealth = ts['wealth'].astype(float)
    daily_wealth = wealth.resample('D').last().dropna()
    daily_ret = daily_wealth.pct_change().dropna()

    exp_asset = aggregate_expected_returns_from_alpha_1m(ts)
    exp_portf = aggregate_expected_returns_portfolio(ts)

    def stats_and_path(simple_rets):
        if len(simple_rets) == 0:
            return {'count': 0}
        mean_r = float(simple_rets.mean())
        std_r = float(simple_rets.std())
        median_r = float(simple_rets.median())
        cum_r = float((1.0 + simple_rets).prod() - 1.0)
        end_wealth = float(initial * (1.0 + cum_r))
        return {
            'count': int(len(simple_rets)),
            'mean': mean_r,
            'std': std_r,
            'median': median_r,
            'cum_return': cum_r,
            'start_wealth': initial,
            'end_wealth': end_wealth,
            'pnl': end_wealth - initial
        }

    for label, N in [('7d', 7), ('30d', 30), ('90d', 90)]:
        out[f'realized_{label}'] = stats_and_path(daily_ret.tail(N))
        out[f'expected_asset_{label}'] = stats_and_path(exp_asset['daily']['exp_simple'].dropna().tail(N))
        out[f'expected_portfolio_{label}'] = stats_and_path(exp_portf['daily']['exp_simple'].dropna().tail(N))

    monthly_wealth = wealth.resample('M').last().dropna()
    monthly_ret = monthly_wealth.pct_change().dropna()
    out['realized_monthly'] = stats_and_path(monthly_ret)
    out['expected_asset_monthly'] = stats_and_path(exp_asset['monthly']['exp_simple'].dropna())
    out['expected_portfolio_monthly'] = stats_and_path(exp_portf['monthly']['exp_simple'].dropna())
    return out

# ----------------------
# Plotly figures
# ----------------------

def make_main_plots(ts, trades_df, title="Least-Action MPC (Risk-Tuned)"):
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        subplot_titles=("BTC Price with Trades", "Portfolio Weight", "Alpha and Control (u)",
                                        "Cumulative PnL (log)", "Alpha calibration scale (scale_dyn)"))

    fig.add_trace(go.Scatter(x=ts.index, y=ts['price'], name="Price", line=dict(color="#1f77b4")), row=1, col=1)
    if trades_df is not None and len(trades_df) > 0:
        buys = trades_df[trades_df['action'] == 'BUY']
        sells = trades_df[trades_df['action'] == 'SELL']
        fig.add_trace(go.Scatter(x=buys['Datetime'], y=buys['price'], mode='markers', name='BUY',
                                 marker=dict(symbol='triangle-up', color='green', size=7)), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells['Datetime'], y=sells['price'], mode='markers', name='SELL',
                                 marker=dict(symbol='triangle-down', color='red', size=7)), row=1, col=1)

    fig.add_trace(go.Scatter(x=ts.index, y=ts['weight'], name="Weight", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts['alpha_1m_exp'], name="Expected 1m log-return (alpha)", line=dict(color="#2ca02c")), row=3, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts['u'], name="Control u", line=dict(color="#d62728")), row=3, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts['cum_pnl_log'], name="Cum PnL (log)", line=dict(color="#9467bd")), row=4, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts['scale_dyn'], name="scale_dyn", line=dict(color="#8c564b")), row=5, col=1)

    fig.update_layout(title=title, height=1350, legend_orientation="h")
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig

def make_account_plots(ts, title="Account: Portfolio Value, Margin Used, Available Credit"):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Portfolio Value (wealth)", "Margin Used (notional)", "Available Credit (notional)"))
    fig.add_trace(go.Scatter(x=ts.index, y=ts['wealth'], name="Wealth", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts['margin_used'], name="Margin Used", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts['avail_credit'], name="Available Credit", line=dict(color="#2ca02c")), row=3, col=1)
    fig.update_layout(title=title, height=900, legend_orientation="h")
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig

def make_expected_plots(agg_asset, agg_portf, title="Expected Returns (Asset vs Portfolio)"):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=False,
                        subplot_titles=("Daily Expected Return", "Weekly Expected Return", "Monthly Expected Return"))
    fig.add_trace(go.Bar(x=agg_asset['daily'].index, y=agg_asset['daily']['exp_simple']*100, name="Asset Daily %", marker_color="#1f77b4"), row=1, col=1)
    fig.add_trace(go.Bar(x=agg_portf['daily'].index, y=agg_portf['daily']['exp_simple']*100, name="Portfolio Daily %", marker_color="#aec7e8"), row=1, col=1)
    fig.add_trace(go.Bar(x=agg_asset['weekly'].index, y=agg_asset['weekly']['exp_simple']*100, name="Asset Weekly %", marker_color="#ff7f0e"), row=2, col=1)
    fig.add_trace(go.Bar(x=agg_portf['weekly'].index, y=agg_portf['weekly']['exp_simple']*100, name="Portfolio Weekly %", marker_color="#ffbb78"), row=2, col=1)
    fig.add_trace(go.Bar(x=agg_asset['monthly'].index, y=agg_asset['monthly']['exp_simple']*100, name="Asset Monthly %", marker_color="#2ca02c"), row=3, col=1)
    fig.add_trace(go.Bar(x=agg_portf['monthly'].index, y=agg_portf['monthly']['exp_simple']*100, name="Portfolio Monthly %", marker_color="#98df8a"), row=3, col=1)
    fig.update_layout(title=title, height=950, legend_orientation="h", yaxis_title="Percent")
    return fig

def make_forward_expected_plots(fwd_asset, fwd_portf, title="Forward Cumulative Expected Returns (Asset vs Portfolio)"):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Expected next 1d simple return", "Expected next 7d simple return", "Expected next 30d simple return"))
    fig.add_trace(go.Scatter(x=fwd_asset['1d'].index, y=fwd_asset['1d'].values*100, name="Asset next 1d %", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=fwd_portf['1d'].index, y=fwd_portf['1d'].values*100, name="Portfolio next 1d %", line=dict(color="#aec7e8")), row=1, col=1)
    fig.add_trace(go.Scatter(x=fwd_asset['7d'].index, y=fwd_asset['7d'].values*100, name="Asset next 7d %", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Scatter(x=fwd_portf['7d'].index, y=fwd_portf['7d'].values*100, name="Portfolio next 7d %", line=dict(color="#ffbb78")), row=2, col=1)
    fig.add_trace(go.Scatter(x=fwd_asset['30d'].index, y=fwd_asset['30d'].values*100, name="Asset next 30d %", line=dict(color="#2ca02c")), row=3, col=1)
    fig.add_trace(go.Scatter(x=fwd_portf['30d'].index, y=fwd_portf['30d'].values*100, name="Portfolio next 30d %", line=dict(color="#98df8a")), row=3, col=1)
    fig.update_layout(title=title, height=950, legend_orientation="h", yaxis_title="Percent")
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
        'eta': 1.0e-6,
        'c2': 3.0e-6,
        'lam': 1.5e-3,
        'beta': 1e-3,
        'u_max': 0.08,
        'w_max': 1.0,
        'epochs_alpha': 12,
        'alpha_scale': 0.8,
        'alpha_clip': 0.0012,
        'k_sigma': 0.10,
        'spread_bps': 5.0,
        'risk_budget_daily': 0.08,
        'dd_limit': 0.06,
        'dd_cooldown_min': 120,
        'trade_min': 0.004,
        'initial_capital': 1.0,
        'deadband_mult': 0.60,
        'kappa_decay': 0.0005,
        'walk_window_min': 70000,
        'retrain_every_min': 1440,
        'dropout': 0.1,
        # online calibration
        'dyn_calib': True,
        'calib_halflife_min': 1440,
        'calib_clip': (0.3, 1.5),
        # gates
        'gate_quantile': 0.85,
        'gate_window': 3000,
        'persist_min': 4,
        'trade_cooldown_min': 5,
        # target sizing (tame)
        'enable_target': True,
        'target_gain': 0.3,
        'gate_snr': 0.18,
        'w_min_expo': 0.0,
        'snr_high': 0.35,
        'blend_mpc': 0.85,
        # turnover governor
        'turn_window_min': 60,
        'turn_cap': 0.80,
        'turn_decay': 0.5
    }

    perf, ts, trades_df = backtest_least_action(df, config)
    print("Backtest summary:")
    print(json.dumps(perf, indent=2))

    # Aggregated expected returns (asset and portfolio)
    agg_asset = aggregate_expected_returns_from_alpha_1m(ts)
    agg_portf = aggregate_expected_returns_portfolio(ts)
    print("Expected returns (Asset):")
    for k in ['daily','weekly','monthly']:
        dfk = agg_asset[k]
        print(f" Asset {k}: mean={dfk['exp_simple'].mean():.4%}, std={dfk['exp_simple'].std():.4%}, count={len(dfk)}")
    print("Expected returns (Portfolio = weight × alpha):")
    for k in ['daily','weekly','monthly']:
        dfk = agg_portf[k]
        print(f" Portfolio {k}: mean={dfk['exp_simple'].mean():.4%}, std={dfk['exp_simple'].std():.4%}, count={len(dfk)}")

    # Forward cumulative expected returns over next 1/7/30 days (asset and portfolio)
    fwd_asset, fwd_portf = forward_expected_curves(ts, {'1d':1440, '7d':10080, '30d':43200})

    # $1,000 examples
    examples = wealth_examples(ts, initial=1000.0)
    print("\n$1,000 examples (realized vs expected):")
    for key, val in examples.items():
        if 'count' in val and val['count'] > 0:
            print(f" {key}: days={val['count']}, mean={val['mean']:.2%}, std={val['std']:.2%}, "
                  f"median={val['median']:.2%}, cum={val['cum_return']:.2%}, "
                  f"start=${val['start_wealth']:.2f}, end=${val['end_wealth']:.2f}, pnl=${val['pnl']:.2f}")

    # Trades
    total_trades = len(trades_df)
    buys = int((trades_df['action']=='BUY').sum()) if total_trades > 0 else 0
    sells = int((trades_df['action']=='SELL').sum()) if total_trades > 0 else 0
    print(f"\nTotal trades: {total_trades} | Buys: {buys} | Sells: {sells}")
    if total_trades > 0:
        print("Trade log preview:")
        print(trades_df.head(10).to_string(index=False))

    # Plot only the trade interval (and you can switch to full-period plots if needed)
    ts_interval = slice_to_trade_interval(ts, trades_df, pad_min=720)
    fig_main = make_main_plots(ts_interval, trades_df, title="Least-Action MPC Strategy on BTC-USD (Risk-Tuned, Trade Interval)")
    fig_acct = make_account_plots(ts_interval, title="Account (Trade Interval)")
    fig_agg = make_expected_plots(agg_asset, agg_portf, title="Expected Returns (Daily/Weekly/Monthly): Asset vs Portfolio")
    fwd_asset_ts, fwd_portf_ts = forward_expected_curves(ts_interval, {'1d':1440, '7d':10080, '30d':43200})
    fig_fwd = make_forward_expected_plots(fwd_asset_ts, fwd_portf_ts, title="Forward Cumulative Expected Returns (1d/7d/30d): Asset vs Portfolio")

    try:
        fig_main.show(); fig_acct.show(); fig_agg.show(); fig_fwd.show()
    except Exception:
        print("Plotly rendering failed. Saving to HTML...")
        fig_main.write_html("least_action_main_risk_tuned_trades.html")
        fig_acct.write_html("least_action_account_risk_tuned_trades.html")
        fig_agg.write_html("least_action_expected_agg_risk_tuned.html")
        fig_fwd.write_html("least_action_expected_forward_risk_tuned.html")
        print("Saved plots: least_action_main_risk_tuned_trades.html, least_action_account_risk_tuned_trades.html, least_action_expected_agg_risk_tuned.html, least_action_expected_forward_risk_tuned.html")

if __name__ == "__main__":
    main()
