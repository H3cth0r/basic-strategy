import os
import math
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ta

warnings.filterwarnings("ignore")

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

# -----------------------------
# Feature engineering with `ta`
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Basic returns
    df["ret1"] = df["Close"].pct_change().fillna(0.0)
    df["logret1"] = np.log1p(df["ret1"]).fillna(0.0)

    # RSI
    rsi = ta.momentum.RSIIndicator(close=df["Close"], window=14)
    df["rsi"] = rsi.rsi()

    # MACD
    macd_ind = ta.trend.MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()

    # EMAs
    df["ema_12"] = ta.trend.EMAIndicator(close=df["Close"], window=12).ema_indicator()
    df["ema_26"] = ta.trend.EMAIndicator(close=df["Close"], window=26).ema_indicator()

    # ATR
    atr = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["atr"] = atr.average_true_range()

    # MFI
    try:
        mfi_ind = ta.volume.MFIIndicator(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=14)
        df["mfi"] = mfi_ind.money_flow_index()
    except Exception:
        df["mfi"] = np.nan

    # Fill and clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df

# --------------------------------------------
# Utility functions for metrics and processing
# --------------------------------------------
def sharpe_ratio(returns: np.ndarray, eps: float = 1e-9) -> float:
    m = np.mean(returns)
    s = np.std(returns)
    if s < eps:
        return 0.0
    return m / (s + eps)

def max_drawdown(series: np.ndarray) -> float:
    if len(series) == 0:
        return 0.0
    peak = np.maximum.accumulate(series)
    drawdown = (series - peak) / peak
    return -np.min(drawdown)

# --------------------------------
# Replay Buffer with shared labels
# --------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: Tuple[int, int], device):
        self.capacity = capacity
        self.device = device
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.reward_labels = []  # vector of rewards per action (for reward net)

    def push(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool, r_label_vec: torch.Tensor):
        if len(self.states) >= self.capacity:
            # FIFO
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.reward_labels.pop(0)
        self.states.append(state.detach())
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state.detach())
        self.dones.append(done)
        self.reward_labels.append(r_label_vec.detach())

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, len(self.states), size=batch_size)
        s = torch.stack([self.states[i] for i in idxs]).to(self.device)
        a = torch.tensor([self.actions[i] for i in idxs], dtype=torch.long, device=self.device)
        r = torch.tensor([self.rewards[i] for i in idxs], dtype=torch.float32, device=self.device)
        ns = torch.stack([self.next_states[i] for i in idxs]).to(self.device)
        d = torch.tensor([self.dones[i] for i in idxs], dtype=torch.bool, device=self.device)
        rl = torch.stack([self.reward_labels[i] for i in idxs]).to(self.device)
        return s, a, r, ns, d, rl

    def __len__(self):
        return len(self.states)

# ------------------------
# Reward Network (SRDRL)
# ------------------------
class RewardNet(nn.Module):
    """
    A compact 1D CNN + MLP reward network that approximates action-wise rewards from a state sequence.
    Outputs a 3-dim vector [r_hold, r_buy, r_sell].
    """
    def __init__(self, n_features: int, window: int, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3)
        )

    def forward(self, x):
        # x: [B, C, T]
        h = self.conv(x)  # [B, 32, 1]
        out = self.mlp(h)  # [B, 3]
        return out

# -------------------------
# Q-Network (DDQN, simple)
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, n_features: int, window: int, hidden: int = 128, n_actions: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        # x: [B, C, T]
        h = self.conv(x)
        q = self.fc(h)
        return q

# -----------------
# Trading Env
# -----------------
@dataclass
class TradeEvent:
    time: pd.Timestamp
    action: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    sell_win: Optional[bool] = None

class TradingEnv:
    """
    Long-only environment with actions: 0 Hold, 1 Buy all, 2 Sell all.
    Expert reward: short-term Sharpe ratio of forward returns (best per Tran et al. 2023).
    Final reward used for agent: rt = max(expert_reward[action], reward_net_prediction[action]) per Huang et al. 2024.
    """
    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        times: np.ndarray,
        window_size: int = 60,
        forward_horizon: int = 60,
        init_capital: float = 100000.0,
        fee_bps: float = 10.0  # 0.1% (10 basis points)
    ):
        self.prices = prices
        self.features = features
        self.times = times
        self.window_size = window_size
        self.forward_horizon = forward_horizon
        self.init_capital = init_capital
        self.fee_rate = fee_bps / 10000.0

        self.reset_indices(0, len(prices) - 1)  # default
        self.reset()

    def reset_indices(self, start_idx: int, end_idx: int):
        self.start_idx = max(start_idx, 0)
        self.end_idx = min(end_idx, len(self.prices) - 1)

    def reset(self):
        self.ptr = self.start_idx + self.window_size - 1
        self.cash = self.init_capital
        self.holdings = 0.0
        self.avg_cost = 0.0
        self.portfolio_values = []
        self.cash_series = []
        self.holdings_series = []
        self.time_series = []
        self.events: List[TradeEvent] = []
        return self._get_state()

    def _get_state(self) -> torch.Tensor:
        # Build state as [features] window: shape (C, T)
        s_feat = self.features[self.ptr - self.window_size + 1 : self.ptr + 1]  # [T, C]
        # Add position & cash normalized channels if desired
        # We'll keep it simple and rely on technical indicators
        s = torch.tensor(s_feat.T, dtype=torch.float32)
        return s  # [C, T]

    def _compute_expert_rewards(self) -> np.ndarray:
        """
        Compute expert reward per action using short-term Sharpe ratio of forward returns.
        Actions: [Hold, Buy, Sell]
        For long-only, we define:
          r_hold = pos_sign * sharpe
          r_buy  = +sharpe
          r_sell = -sharpe (discourages selling into uptrend)
        """
        end = min(self.ptr + self.forward_horizon, self.end_idx)
        base_price = self.prices[self.ptr]
        if end <= self.ptr + 1:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        forward_prices = self.prices[self.ptr+1 : end+1]
        # Percent returns vs base
        forward_returns = (forward_prices - base_price) / (base_price + 1e-12)
        sr = sharpe_ratio(forward_returns)
        pos_sign = 1.0 if self.holdings > 0 else 0.0
        r_hold = pos_sign * sr
        r_buy = sr
        r_sell = -sr
        return np.array([r_hold, r_buy, r_sell], dtype=np.float32)

    def _update_portfolio_series(self):
        pv = self.cash + self.holdings * self.prices[self.ptr]
        self.portfolio_values.append(pv)
        self.cash_series.append(self.cash)
        self.holdings_series.append(self.holdings)
        self.time_series.append(self.times[self.ptr])

    def step(self, action: int, reward_net: Optional[nn.Module] = None, device: Optional[torch.device] = None):
        price = self.prices[self.ptr]
        # Predicted rewards from RewardNet
        state = self._get_state().unsqueeze(0)  # [1, C, T]
        rst = None
        if reward_net is not None:
            reward_net.eval()
            with torch.no_grad():
                rst_t = reward_net(state.to(device))  # [1, 3]
                rst = rst_t.squeeze(0).cpu().numpy()
        else:
            rst = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Expert rewards per action
        ret = self._compute_expert_rewards()

        # Final reward vector: element-wise choose better for each action
        r_vec = np.where(rst >= ret, rst, ret)  # [3]

        # Execute action
        trade_event = None
        if action == 1:  # Buy all if flat
            if self.holdings == 0.0 and self.cash > 0.0:
                qty = (self.cash * (1.0 - self.fee_rate)) / price
                if qty > 0:
                    cost = qty * price
                    fee = cost * self.fee_rate
                    self.cash -= (cost + fee)
                    self.holdings += qty
                    self.avg_cost = price
                    trade_event = TradeEvent(time=self.times[self.ptr], action="BUY", price=price, quantity=qty)
        elif action == 2:  # Sell all if long
            if self.holdings > 0.0:
                proceeds = self.holdings * price
                fee = proceeds * self.fee_rate
                self.cash += (proceeds - fee)
                sell_win = price >= self.avg_cost
                trade_event = TradeEvent(time=self.times[self.ptr], action="SELL", price=price, quantity=self.holdings, sell_win=sell_win)
                self.holdings = 0.0
                self.avg_cost = 0.0
        # Hold: do nothing

        # Update series
        self._update_portfolio_series()

        # Move pointer
        done = False
        if self.ptr >= self.end_idx - 1:
            done = True
        else:
            self.ptr += 1

        next_state = self._get_state()
        # Reward scalar is the chosen action's element
        reward_scalar = float(r_vec[action])

        # Return reward label vector (for training reward net)
        r_label_vec = torch.tensor(r_vec, dtype=torch.float32)

        info = {
            "rst": rst,
            "ret": ret,
            "r_vec": r_vec,
            "trade_event": trade_event
        }
        return next_state, reward_scalar, done, info

# ----------------------------
# Walk-forward preparation
# ----------------------------
def split_walk_forward(df: pd.DataFrame, period_days: int = 36, train_ratio: float = 0.8) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of tuples: (train_start_idx, train_end_idx, test_start_idx, test_end_idx)
    """
    minutes_per_day = 1440
    period_len = period_days * minutes_per_day
    n = len(df)
    windows = []
    start = 0
    while start + period_len <= n:
        end = start + period_len
        train_end = start + int(period_len * train_ratio)
        windows.append((start, train_end, train_end, end))
        start = end  # non-overlapping walk-forward
    return windows

def standardize_features(train_feat: np.ndarray, test_feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(train_feat)
    train_scaled = scaler.transform(train_feat)
    test_scaled = scaler.transform(test_feat)
    return train_scaled, test_scaled

# ----------------------------
# Training & Evaluation (DDQN)
# ----------------------------
def train_srddqn_on_window(
    df: pd.DataFrame,
    window: Tuple[int, int, int, int],
    feature_cols: List[str],
    device: torch.device,
    ddqn_params: Dict,
    reward_net_params: Dict,
    verbose: bool = False
):
    (train_start, train_end, test_start, test_end) = window
    sub_train = df.iloc[train_start:train_end].copy()
    sub_test = df.iloc[test_start:test_end].copy()

    # Build feature matrices and standardize by train statistics
    train_feat_mat = sub_train[feature_cols].values
    test_feat_mat = sub_test[feature_cols].values
    train_feat_mat, test_feat_mat = standardize_features(train_feat_mat, test_feat_mat)

    # Prepare env & networks
    window_size = ddqn_params["window_size"]
    forward_horizon = ddqn_params["forward_horizon"]
    init_capital = ddqn_params["init_capital"]
    fee_bps = ddqn_params["fee_bps"]

    env_train = TradingEnv(
        prices=sub_train["Close"].values,
        features=train_feat_mat,
        times=sub_train.index.values,
        window_size=window_size,
        forward_horizon=forward_horizon,
        init_capital=init_capital,
        fee_bps=fee_bps
    )
    env_train.reset_indices(0, len(sub_train["Close"].values) - 1)
    env_train.reset()

    env_test = TradingEnv(
        prices=sub_test["Close"].values,
        features=test_feat_mat,
        times=sub_test.index.values,
        window_size=window_size,
        forward_horizon=forward_horizon,
        init_capital=init_capital,
        fee_bps=fee_bps
    )
    env_test.reset_indices(0, len(sub_test["Close"].values) - 1)
    env_test.reset()

    n_features = train_feat_mat.shape[1]
    q_online = QNetwork(n_features=n_features, window=window_size, hidden=ddqn_params["hidden"], n_actions=3).to(device)
    q_target = QNetwork(n_features=n_features, window=window_size, hidden=ddqn_params["hidden"], n_actions=3).to(device)
    q_target.load_state_dict(q_online.state_dict())

    reward_net = RewardNet(n_features=n_features, window=window_size, hidden=reward_net_params["hidden"]).to(device)

    q_opt = optim.Adam(q_online.parameters(), lr=ddqn_params["lr"])
    r_opt = optim.Adam(reward_net.parameters(), lr=reward_net_params["lr"])
    mse = nn.MSELoss()

    buffer = ReplayBuffer(capacity=ddqn_params["buffer_capacity"], state_shape=(n_features, window_size), device=device)

    episodes = ddqn_params["episodes"]
    batch_size = ddqn_params["batch_size"]
    gamma = ddqn_params["gamma"]
    target_update_step = ddqn_params["target_update_step"]

    epsilon_start = ddqn_params["epsilon_start"]
    epsilon_end = ddqn_params["epsilon_end"]
    epsilon_decay_steps = ddqn_params["epsilon_decay_steps"]

    global_step = 0
    ep_bar = tqdm(range(episodes), desc="Train episodes", disable=not verbose)
    for ep in ep_bar:
        state = env_train.reset()
        done = False
        ep_steps = 0
        price_series_len = len(env_train.prices)
        step_bar = tqdm(total=(price_series_len - env_train.window_size), desc=f"Episode {ep+1}", leave=False, disable=not verbose)
        while not done:
            # epsilon-greedy
            eps = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * global_step / max(1, epsilon_decay_steps))
            q_online.eval()
            with torch.no_grad():
                q_vals = q_online(state.unsqueeze(0).to(device))  # [1, 3]
            if np.random.rand() < eps:
                action = np.random.randint(0, 3)
            else:
                action = int(torch.argmax(q_vals, dim=1).item())

            next_state, reward_scalar, done, info = env_train.step(action, reward_net=reward_net, device=device)

            # Push to buffer
            buffer.push(state, action, reward_scalar, next_state, done, torch.tensor(info["r_vec"], dtype=torch.float32))

            state = next_state
            ep_steps += 1
            global_step += 1
            step_bar.update(1)

            # Train both networks if enough samples
            if len(buffer) >= batch_size:
                s, a, r, ns, d, rl = buffer.sample(batch_size)

                # Q update (Double DQN)
                q_online.train()
                q_target.eval()
                with torch.no_grad():
                    # online selects action
                    next_q_online = q_online(ns)
                    next_actions = torch.argmax(next_q_online, dim=1)  # [B]
                    # target evaluates
                    next_q_target = q_target(ns)
                    target_q = next_q_target.gather(1, next_actions.view(-1, 1)).squeeze(1)
                    y = r + (~d).float() * gamma * target_q
                q_pred = q_online(s).gather(1, a.view(-1, 1)).squeeze(1)
                q_loss = mse(q_pred, y)
                q_opt.zero_grad()
                q_loss.backward()
                q_opt.step()

                # RewardNet update (synchronous, shared buffer)
                reward_net.train()
                r_pred = reward_net(s)  # [B, 3]
                r_loss = mse(r_pred, rl)
                r_opt.zero_grad()
                r_loss.backward()
                r_opt.step()

                if global_step % target_update_step == 0:
                    q_target.load_state_dict(q_online.state_dict())

        step_bar.close()

    # ----------------------
    # Evaluation on test set
    # ----------------------
    q_online.eval()
    reward_net.eval()

    state = env_test.reset()
    done = False
    test_returns = []
    # For plotting trade markers
    buy_markers = []
    sell_win_markers = []
    sell_lose_markers = []

    with tqdm(total=(len(env_test.prices) - env_test.window_size), desc="Evaluate", disable=not verbose) as eval_bar:
        prev_pv = env_test.cash + env_test.holdings * env_test.prices[env_test.ptr]
        while not done:
            with torch.no_grad():
                q_vals = q_online(state.unsqueeze(0).to(device))
            action = int(torch.argmax(q_vals, dim=1).item())
            next_state, reward_scalar, done, info = env_test.step(action, reward_net=reward_net, device=device)
            # Compute minute return of portfolio
            pv = env_test.cash + env_test.holdings * env_test.prices[env_test.ptr]
            ret = (pv - prev_pv) / (prev_pv + 1e-12)
            test_returns.append(ret)
            prev_pv = pv

            # Collect trade markers
            evt = info.get("trade_event", None)
            if evt is not None:
                ts = evt.time
                if evt.action == "BUY":
                    buy_markers.append((ts, evt.price))
                elif evt.action == "SELL":
                    if evt.sell_win:
                        sell_win_markers.append((ts, evt.price))
                    else:
                        sell_lose_markers.append((ts, evt.price))

            state = next_state
            eval_bar.update(1)

    pv_series = np.array(env_test.portfolio_values)
    cash_series = np.array(env_test.cash_series)
    holdings_series = np.array(env_test.holdings_series)
    time_series = np.array(env_test.time_series)
    price_series = env_test.prices[env_test.ptr - len(pv_series) + 1 : env_test.ptr + 1]
    # Align price_series length with time_series
    if len(price_series) != len(time_series):
        price_series = env_test.prices[(env_test.ptr - len(time_series) + 1) : env_test.ptr + 1]

    # Metrics
    cum_ret = (pv_series[-1] / pv_series[0]) - 1.0 if len(pv_series) > 1 else 0.0
    sr = sharpe_ratio(np.array(test_returns))
    mdd = max_drawdown(pv_series)
    num_trades = len(buy_markers) + len(sell_win_markers) + len(sell_lose_markers)
    wins = len(sell_win_markers)
    loses = len(sell_lose_markers)
    win_rate = wins / max(1, (wins + loses))

    metrics = {
        "cumulative_return": cum_ret,
        "sharpe_ratio": sr,
        "max_drawdown": mdd,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "wins": wins,
        "loses": loses
    }

    plotting_payload = {
        "time": time_series,
        "price": price_series,
        "portfolio": pv_series,
        "cash": cash_series,
        "holdings": holdings_series,
        "buy_markers": buy_markers,
        "sell_win_markers": sell_win_markers,
        "sell_lose_markers": sell_lose_markers,
        "window": window
    }
    return metrics, plotting_payload

def plot_results(payload, title_suffix=""):
    time = payload["time"]
    price = payload["price"]
    portfolio = payload["portfolio"]
    cash = payload["cash"]
    holdings = payload["holdings"]
    buy_markers = payload["buy_markers"]
    sell_win_markers = payload["sell_win_markers"]
    sell_lose_markers = payload["sell_lose_markers"]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=("BTC Price with Trades", "Portfolio Value", "Credit (Cash)", "Holdings (BTC)"),
        vertical_spacing=0.06
    )

    # Price subplot with markers
    fig.add_trace(go.Scatter(x=time, y=price, name="BTC-USD Price", mode="lines", line=dict(color="royalblue")), row=1, col=1)

    if len(buy_markers) > 0:
        bm_t = [t for t, p in buy_markers]
        bm_p = [p for t, p in buy_markers]
        fig.add_trace(go.Scatter(
            x=bm_t, y=bm_p, name="BUY", mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=8)
        ), row=1, col=1)

    if len(sell_win_markers) > 0:
        sw_t = [t for t, p in sell_win_markers]
        sw_p = [p for t, p in sell_win_markers]
        fig.add_trace(go.Scatter(
            x=sw_t, y=sw_p, name="SELL-WIN", mode="markers",
            marker=dict(symbol="circle", color="limegreen", size=8)
        ), row=1, col=1)

    if len(sell_lose_markers) > 0:
        sl_t = [t for t, p in sell_lose_markers]
        sl_p = [p for t, p in sell_lose_markers]
        fig.add_trace(go.Scatter(
            x=sl_t, y=sl_p, name="SELL-LOSE", mode="markers",
            marker=dict(symbol="circle", color="red", size=8)
        ), row=1, col=1)

    # Portfolio subplot
    fig.add_trace(go.Scatter(x=time, y=portfolio, name="Portfolio Value", mode="lines", line=dict(color="darkorange")), row=2, col=1)

    # Cash subplot
    fig.add_trace(go.Scatter(x=time, y=cash, name="Cash", mode="lines", line=dict(color="purple")), row=3, col=1)

    # Holdings subplot
    fig.add_trace(go.Scatter(x=time, y=holdings, name="Holdings (BTC)", mode="lines", line=dict(color="teal")), row=4, col=1)

    fig.update_layout(
        title=f"Walk-Forward Test Results {title_suffix}",
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()

def main():
    # Device selection (mps if possible, else cuda, else cpu)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    df = load_data()
    # Compute indicators
    df = add_indicators(df)

    # Feature columns used for state (ta indicators and returns)
    feature_cols = [
        "ret1", "logret1", "rsi", "macd", "macd_signal", "macd_hist",
        "ema_12", "ema_26", "atr", "mfi"
    ]

    # Ensure data clean
    df = df.dropna()
    print(f"Data samples: {len(df)}; date range: {df.index.min()} -> {df.index.max()}")

    # Walk-forward windows: 36 days per paper (Tran et al.), 80/20 split
    windows = split_walk_forward(df, period_days=36, train_ratio=0.8)
    if len(windows) == 0:
        # Fallback to a single split (80/20)
        n = len(df)
        train_end = int(n * 0.8)
        windows = [(0, train_end, train_end, n)]
    print(f"Walk-forward windows: {len(windows)}")

    # Hyperparameters (inspired by both papers)
    ddqn_params = {
        "window_size": 60,            # 60 minutes history in state
        "forward_horizon": 60,        # short-term forward horizon for Sharpe reward (60 minutes)
        "init_capital": 100000.0,
        "fee_bps": 10.0,              # 0.1% per transaction
        "hidden": 128,
        "lr": 1e-3,
        "buffer_capacity": 10000,
        "episodes": 10,               # adjust higher for longer training (paper used ~100)
        "batch_size": 64,
        "gamma": 0.98,                # as per paper (future rewards matter)
        "target_update_step": 100,
        "epsilon_start": 1.0,
        "epsilon_end": 0.10,
        "epsilon_decay_steps": 3000
    }
    reward_net_params = {
        "hidden": 64,
        "lr": 1e-3
    }

    all_metrics = []
    last_payload = None

    wf_bar = tqdm(windows, desc="Walk-forward windows")
    for i, w in enumerate(wf_bar):
        metrics, payload = train_srddqn_on_window(
            df=df,
            window=w,
            feature_cols=feature_cols,
            device=device,
            ddqn_params=ddqn_params,
            reward_net_params=reward_net_params,
            verbose=True
        )
        all_metrics.append(metrics)
        last_payload = payload

        # Print per-window metrics
        train_start, train_end, test_start, test_end = w
        print("\n========== Walk-forward Window Results ==========")
        print(f"Train: {df.index[train_start]} -> {df.index[train_end-1]} | Test: {df.index[test_start]} -> {df.index[test_end-1]}")
        print(f"Cumulative Return: {metrics['cumulative_return']*100:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"Trades: {metrics['num_trades']} | Wins: {metrics['wins']} | Loses: {metrics['loses']} | Win Rate: {metrics['win_rate']*100:.2f}%")

    # Aggregate results across windows
    if len(all_metrics) > 0:
        avg_cr = np.mean([m["cumulative_return"] for m in all_metrics])
        avg_sr = np.mean([m["sharpe_ratio"] for m in all_metrics])
        avg_mdd = np.mean([m["max_drawdown"] for m in all_metrics])
        total_trades = np.sum([m["num_trades"] for m in all_metrics])
        total_wins = np.sum([m["wins"] for m in all_metrics])
        total_loses = np.sum([m["loses"] for m in all_metrics])
        win_rate = total_wins / max(1, (total_wins + total_loses))

        print("\n========== Overall Walk-forward Results ==========")
        print(f"Windows: {len(all_metrics)}")
        print(f"Average Cumulative Return: {avg_cr*100:.2f}%")
        print(f"Average Sharpe Ratio: {avg_sr:.3f}")
        print(f"Average Max Drawdown: {avg_mdd*100:.2f}%")
        print(f"Total Trades: {total_trades} | Wins: {total_wins} | Loses: {total_loses} | Win Rate: {win_rate*100:.2f}%")

    # Plot from the last test window
    if last_payload is not None:
        plot_results(last_payload, title_suffix="(Last Walk-Forward Test Window)")

if __name__ == "__main__":
    main()
