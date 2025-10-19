#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduces a simple Deep Reinforcement Learning trading agent (DQN) inspired by:
"Recommending Cryptocurrency Trading Points with Deep Reinforcement Learning Approach"
- Architecture: 64 -> 32 -> 8 -> 3 (ReLU; final linear), MSE/TD error
- Actions: 0=Hold, 1=Buy, 2=Sell
- Reward: realized on sell = proceeds - cumulative cost basis incl. fees, with penalties for too many holds/buys
- Environment: single asset, long-only, fractional position allowed up to 1.0 coin (fixes "not enough cash" at high BTC price),
features include returns/volatility/TA indicators.

It:
- Downloads BTC-USD data (provided URL)
- Prepares features (TA + time features)
- Trains a small DQN on the first split of data
- Tests on the last split of data
- Produces Plotly interactive plots: price with buy/sell markers, and equity curve
- Uses CUDA if available; otherwise Apple MPS if available; else CPU
- Uses tqdm progress bars during training and testing

Run:
  python main.py

Notes:
- This is a light, self-contained educational implementation that closely follows the paper's described ideas.
- For speed, training settings are modest. Increase episodes/steps for better performance if you have a GPU.

"""

import os
import sys
import math
import random
import subprocess
import importlib
from collections import deque

# --- lightweight auto-install for missing libs (optional) ---
def _ensure(pkg):
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"Installing missing package: {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for pkg in ["pandas", "numpy", "plotly", "torch", "ta", "tqdm"]:
    _ensure(pkg)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

# ---------- Configuration ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Detect device: CUDA -> MPS -> CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# Data URL (provided)
data_url = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"

# ---------- Data Loading & Feature Engineering ----------

def _add_time_features(df: pd.DataFrame) -> None:
    if "Datetime" not in df.columns:
        return
    df["minute"] = df["Datetime"].dt.minute
    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek
    df["dayofmonth"] = df["Datetime"].dt.day
    df["month"] = df["Datetime"].dt.month

    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60.0)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60.0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["dom_sin"] = np.sin(2 * np.pi * (df["dayofmonth"] - 1) / 31.0)
    df["dom_cos"] = np.cos(2 * np.pi * (df["dayofmonth"] - 1) / 31.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

def load_raw_data(url: str) -> pd.DataFrame:
    import ta

    print("Loading and preparing data...")
    df = pd.read_csv(url, skiprows=3, header=None)
    df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors="coerce")
    df = df.sort_values('Datetime').reset_index(drop=True)

    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['Original_Close'] = df['Close'].astype(float)

    df['log_close'] = np.log(df['Original_Close'] + 1e-12)
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

# ---------- RL Environment (fractional position support) ----------

class TradingEnv:
    """
    Long-only single-asset environment with fractional position support:
    - position_qty in [0, max_position_qty], default max_position_qty=1.0 coin
    - Buy action increases position up to cap based on available cash
    - Sell action closes entire position (market flat)
    - Reward (on sell): realized PnL = proceeds - cumulative cost basis including buy fees
    - Penalties for too many sequential holds/buys (as in paper)
    """

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        datetimes: np.ndarray = None,
        lookback: int = 60,
        fee_rate: float = 0.0015,
        hold_penalty_limit: int = 20,
        buy_penalty_limit: int = 20,
        invalid_action_penalty: float = -0.001,
        max_steps: int = None,
        initial_cash: float = 10000.0,
        max_position_qty: float = 1.0,  # cap at 1 coin (fractional allowed)
        min_trade_qty: float = 1e-6,    # avoid dust
    ):
        assert features.ndim == 2
        assert len(prices) == len(features)
        self.features = features
        self.prices = prices.astype(float)
        self.datetimes = datetimes
        self.lookback = lookback
        self.fee_rate = fee_rate
        self.hold_penalty_limit = hold_penalty_limit
        self.buy_penalty_limit = buy_penalty_limit
        self.invalid_action_penalty = invalid_action_penalty
        self.initial_cash = float(initial_cash)
        self.max_steps = max_steps if max_steps is not None else (len(prices) - 1)
        self.max_position_qty = float(max_position_qty)
        self.min_trade_qty = float(min_trade_qty)

        self.reset()

    def reset(self):
        self.t = self.lookback
        self.position_qty = 0.0
        self.cash = float(self.initial_cash)
        self.cost_basis_cash = 0.0  # cumulative buy cost including buy fees for current open position
        self.hold_count = 0
        self.buy_count = 0
        self.equity_curve = []
        self.actions_taken = []
        self.trade_marks = []  # tuples (t, action, price, qty)
        return self._get_state()

    def _get_state(self):
        start = self.t - self.lookback
        end = self.t
        window = self.features[start:end]  # (lookback, n_features)
        # Append position qty as continuous feature
        state = np.concatenate([window.flatten(), np.array([self.position_qty], dtype=np.float32)], axis=0)
        return state

    def _equity(self, price):
        return self.cash + self.position_qty * price

    def step(self, action: int):
        done = False
        reward = 0.0
        price = float(self.prices[self.t])

        # Track equity pre-action (will update after action)
        equity = self._equity(price)
        self.equity_curve.append(equity)
        self.actions_taken.append(action)

        if action == 0:
            # Hold
            self.hold_count += 1
            if self.hold_count > self.hold_penalty_limit:
                reward -= 0.001 * (self.hold_count - self.hold_penalty_limit)

        elif action == 1:
            # Buy (increase position towards cap as much as cash allows)
            desired_add = self.max_position_qty - self.position_qty
            affordable_qty = self.cash / (price * (1.0 + self.fee_rate))
            qty_to_buy = max(0.0, min(desired_add, affordable_qty))

            if qty_to_buy >= self.min_trade_qty:
                cost = price * qty_to_buy * (1.0 + self.fee_rate)
                self.cash -= cost
                self.position_qty += qty_to_buy
                self.cost_basis_cash += price * qty_to_buy * (1.0 + self.fee_rate)

                self.trade_marks.append((self.t, "buy", price, qty_to_buy))
                self.buy_count += 1
                self.hold_count = 0

                if self.buy_count > self.buy_penalty_limit:
                    reward -= 0.001 * (self.buy_count - self.buy_penalty_limit)
            else:
                # Can't buy meaningful quantity (no cash or already at cap)
                reward += self.invalid_action_penalty
                self.buy_count += 1
                if self.buy_count > self.buy_penalty_limit:
                    reward -= 0.001 * (self.buy_count - self.buy_penalty_limit)

        elif action == 2:
            # Sell (close entire position if any)
            if self.position_qty > 0.0:
                qty = self.position_qty
                proceeds = price * qty * (1.0 - self.fee_rate)
                realized = proceeds - self.cost_basis_cash  # cost basis included buy fees
                reward += realized

                self.cash += proceeds
                self.position_qty = 0.0
                self.cost_basis_cash = 0.0
                self.hold_count = 0
                self.buy_count = 0

                self.trade_marks.append((self.t, "sell", price, qty))
            else:
                reward += self.invalid_action_penalty
                self.hold_count += 1
                if self.hold_count > self.hold_penalty_limit:
                    reward -= 0.001 * (self.hold_count - self.hold_penalty_limit)
        else:
            raise ValueError("Invalid action")

        # Advance time
        self.t += 1
        if self.t >= min(len(self.prices) - 1, self.max_steps):
            done = True

        next_state = self._get_state() if not done else None

        # Update last equity after action using price at (t-1) (the price we just acted on)
        last_px = float(self.prices[self.t - 1])
        self.equity_curve[-1] = self._equity(last_px)
        info = {"equity": self.equity_curve[-1], "price": last_px}
        return next_state, reward, done, info

# ---------- DQN Agent ----------

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.out = nn.Linear(8, output_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q = self.out(x)
        return q

class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            ns,
            np.array(d, dtype=bool),
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 256,
        tau: float = 0.005,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.q = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_q = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

        self.replay = ReplayBuffer(capacity=200_000)
        self.train_steps = 0

    def act(self, state, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            qvals = self.q(s)
            return int(torch.argmax(qvals, dim=1).item())

    def push(self, s, a, r, ns, d):
        self.replay.push(s, a, r, ns if ns is not None else None, d)

    def soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.target_q.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None

        s, a, r, ns_raw, d = self.replay.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        a = torch.tensor(a, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        d = torch.tensor(d, dtype=torch.bool, device=DEVICE).unsqueeze(1)

        # Build next-state tensor with masking None
        ns_vals = np.zeros_like(s.cpu().numpy())
        mask_not_done = np.array([n is not None for n in ns_raw], dtype=bool)
        if mask_not_done.any():
            ns_vals[mask_not_done] = np.stack([ns_raw[i] for i in range(len(ns_raw)) if mask_not_done[i]])
        ns = torch.tensor(ns_vals, dtype=torch.float32, device=DEVICE)

        # Q(s,a)
        q_pred = self.q(s).gather(1, a)

        with torch.no_grad():
            next_actions = torch.argmax(self.q(ns), dim=1, keepdim=True)
            q_next = self.target_q(ns).gather(1, next_actions)
            q_target = r + (~d).float() * self.gamma * q_next

        loss = F.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optimizer.step()
        self.soft_update()
        self.train_steps += 1
        return float(loss.item())

# ---------- Utilities ----------

def standardize_train_test(train_arr: np.ndarray, test_arr: np.ndarray):
    mean = train_arr.mean(axis=0, keepdims=True)
    std = train_arr.std(axis=0, keepdims=True) + 1e-8
    return (train_arr - mean) / std, (test_arr - mean) / std, mean, std

def build_feature_matrix(df: pd.DataFrame):
    # Exclude raw OHLCV columns not used directly and some intermediates
    exclude = {"Datetime", "High", "Low", "Open", "Volume", "ema_12", "ema_48", "atr_raw"}
    feature_cols = [c for c in df.columns if c not in exclude]
    base = []
    for key in ["Original_Close", "log_close", "ret_1m"]:
        if key in feature_cols:
            base.append(key)
    rest = [c for c in feature_cols if c not in base]
    feature_cols = base + sorted(rest)
    feats = df[feature_cols].values.astype(np.float32)
    prices = df["Original_Close"].values.astype(np.float32)
    dts = df["Datetime"].values if "Datetime" in df.columns else np.arange(len(df))
    return feats, prices, dts, feature_cols

def split_train_test(features, prices, dts, train_ratio=0.7):
    n = len(features)
    split = int(n * train_ratio)
    return (features[:split], prices[:split], dts[:split]), (features[split:], prices[split:], dts[split:])

def to_plotly_price_with_trades(dts, prices, marks):
    fig = go.Figure()
    x = dts
    fig.add_trace(go.Scatter(x=x, y=prices, mode="lines", name="Price", line=dict(color="royalblue")))
    if marks:
        buy_x, buy_y = [], []
        sell_x, sell_y = [], []
        for (t, a, px, qty) in marks:
            if t >= len(x):
                continue
            if a == "buy":
                buy_x.append(x[t]); buy_y.append(prices[t])
            elif a == "sell":
                sell_x.append(x[t]); sell_y.append(prices[t])
        if buy_x:
            fig.add_trace(go.Scatter(
                x=buy_x, y=buy_y, mode="markers",
                marker=dict(color="yellow", size=8, line=dict(color="black", width=0.5)),
                name="Buy"
            ))
        if sell_x:
            fig.add_trace(go.Scatter(
                x=sell_x, y=sell_y, mode="markers",
                marker=dict(color="green", size=8, line=dict(color="black", width=0.5)),
                name="Sell"
            ))
    fig.update_layout(
        title="BTC-USD Price with Trades (DRL Agent)",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def to_plotly_equity(dts, equity):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dts, y=equity, mode="lines", name="Equity", line=dict(color="firebrick")))
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Time",
        yaxis_title="Equity (USD)",
        template="plotly_white"
    )
    return fig

# ---------- Main pipeline ----------

def main():
    df = load_raw_data(data_url)

    # Optional speed-up: keep last N rows (e.g., 120k ~ few months at 1-min)
    # df = df.tail(120_000).reset_index(drop=True)

    features, prices, dts, feat_names = build_feature_matrix(df)

    # Train/test split
    (f_tr, p_tr, dt_tr), (f_te, p_te, dt_te) = split_train_test(features, prices, dts, train_ratio=0.7)

    # Standardize features (fit on train, apply to test)
    f_tr_std, f_te_std, mu, sigma = standardize_train_test(f_tr, f_te)

    # Environment params
    lookback = 60
    fee_rate = 0.0015
    initial_cash = 10_000.0

    # Limit steps per episode to speed up training while still learning
    # Use most of training set but cap to avoid very long runs on CPU/MPS.
    cap_train_steps = min(len(p_tr) - 1, 60_000)
    cap_test_steps = len(p_te) - 1  # evaluate on full test

    env_train = TradingEnv(
        features=f_tr_std, prices=p_tr, datetimes=dt_tr, lookback=lookback,
        fee_rate=fee_rate, initial_cash=initial_cash, max_steps=cap_train_steps,
        max_position_qty=1.0,  # up to 1 coin (fractional allowed)
        min_trade_qty=1e-6,
    )
    env_test = TradingEnv(
        features=f_te_std, prices=p_te, datetimes=dt_te, lookback=lookback,
        fee_rate=fee_rate, initial_cash=initial_cash, max_steps=cap_test_steps,
        max_position_qty=1.0, min_trade_qty=1e-6,
    )

    state_dim = lookback * f_tr_std.shape[1] + 1  # +1 for position qty
    action_dim = 3

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        batch_size=256,
        tau=0.005,
    )

    # Training loop
    episodes = 12
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 10_000
    global_step = 0
    start_train_after = 2_000
    train_every = 4

    steps_per_episode = (env_train.max_steps - env_train.lookback)
    print("Starting training...")
    for ep in range(1, episodes + 1):
        state = env_train.reset()
        ep_reward = 0.0
        ep_steps = 0
        last_loss = None
        done = False

        pbar = tqdm(total=steps_per_episode, desc=f"Train Ep {ep}/{episodes}", leave=True)
        while not done:
            epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (global_step / epsilon_decay_steps))
            action = agent.act(state, epsilon=epsilon)

            next_state, reward, done, info = env_train.step(action)
            agent.push(state, action, reward, next_state, done)
            state = next_state if next_state is not None else state
            ep_reward += reward
            ep_steps += 1
            global_step += 1

            if global_step > start_train_after and global_step % train_every == 0:
                last_loss = agent.train_step()

            postfix = {
                "eps": f"{epsilon:.3f}",
                "r": f"{reward:+.2f}",
                "eq": f"{info['equity']:.2f}",
            }
            if last_loss is not None:
                postfix["loss"] = f"{last_loss:.4f}"
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        final_equity = env_train.equity_curve[-1] if env_train.equity_curve else initial_cash
        print(f"Episode {ep:02d} | Steps: {ep_steps:5d} | EpReward: {ep_reward:10.2f} | FinalEq: {final_equity:10.2f} | Epsilon: {epsilon:0.3f}")

    # Testing (no exploration)
    print("Testing policy...")
    state = env_test.reset()
    done = False
    test_reward = 0.0
    steps_test = (env_test.max_steps - env_test.lookback)
    pbar_t = tqdm(total=steps_test, desc="Testing", leave=True)
    while not done:
        action = agent.act(state, epsilon=0.0)
        next_state, reward, done, info = env_test.step(action)
        state = next_state if next_state is not None else state
        test_reward += reward

        action_name = {0: "H", 1: "B", 2: "S"}[action]
        pbar_t.set_postfix({
            "a": action_name,
            "r": f"{reward:+.2f}",
            "eq": f"{info['equity']:.2f}",
        })
        pbar_t.update(1)
    pbar_t.close()

    test_equity_curve = env_test.equity_curve
    test_marks = env_test.trade_marks
    test_prices = env_test.prices
    test_dts = env_test.datetimes
    final_equity = test_equity_curve[-1] if len(test_equity_curve) else initial_cash
    buy_count = sum(1 for (_, a, _, _) in test_marks if a == "buy")
    sell_count = sum(1 for (_, a, _, _) in test_marks if a == "sell")

    print(f"Test complete. Final equity: {final_equity:.2f} | Initial cash: {initial_cash:.2f} | Gain: {(final_equity - initial_cash):.2f} ({(final_equity/initial_cash - 1)*100:.2f}%)")
    print(f"Trades: buys={buy_count}, sells={sell_count}")

    # ---------- Plotly Visualizations ----------
    fig_price = to_plotly_price_with_trades(test_dts, test_prices, test_marks)
    fig_price.show()

    eq_x = test_dts[env_test.lookback:env_test.lookback+len(test_equity_curve)]
    fig_equity = to_plotly_equity(eq_x, test_equity_curve)
    fig_equity.show()

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    fig_price.write_html(os.path.join(out_dir, "price_with_trades.html"))
    fig_equity.write_html(os.path.join(out_dir, "equity_curve.html"))
    print(f"Saved Plotly HTML files to: outputs/price_with_trades.html, outputs/equity_curve.html")

if __name__ == "__main__":
    main()
