#!/usr/bin/env python3
"""
Deep-Q Attention Trader – single-file demo
Author : ChatGPT   (© OpenAI, 2025)
License: MIT

DISCLAIMER
==========
• This code is for research / educational purposes only.
• It is NOT financial advice and it does NOT guarantee profitability.
• Use real capital at your own risk.

Requires:
----------
pip install pandas numpy torch ta plotly tqdm
"""

import os
import math
import random
import time
import warnings
from collections import deque, namedtuple
from typing import Tuple, List

import numpy as np
import pandas as pd
from ta import add_all_ta_features
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

# ==============  CONFIG  ===================== #
CSV_URL = (
    "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/"
    "data/CRYPTO/BTC-USD/data_0.csv"
)

COLUMN_NAMES = ["Datetime", "Close", "High", "Low", "Open", "Volume"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trading & env parms
TRADING_FEE = 0.001      # 0.1 %
MAX_POSITION = 1.0       # we only hold 0 or 1 BTC in this simple example
HOLD_PENALTY = 1e-4      # discourages passive buy-and-hold
NO_TRADE_PENALTY = 2e-4  # discourages inactivity

SEQ_LEN      = 180       # minutes back for attention (120-180 works, choose 180)
FEATURE_DIM  = 64        # after the feature MLP
ATTN_DIM     = 64
ATTN_HEADS   = 4

ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}

# DQN parms
BATCH_SIZE       = 128
BUFFER_SIZE      = 100_000
GAMMA            = 0.99
LR               = 3e-4
EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY_STEPS  = 50_000
TARGET_UPDATE_EVERY = 1_000  # gradient steps
GRAD_CLIP        = 1.0

# Training schedule
EPISODE_DAYS   = 4                       # ≈5760 minutes / episode
VAL_DAYS       = 4
TEST_DAYS      = 4
MAX_EPISODES   = 500

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# ============================================= #

Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "done"))

def load_data() -> pd.DataFrame:
    """Load BTC-USD minute data and add technical indicators."""
    print("Loading raw CSV ...")
    df = pd.read_csv(
        CSV_URL,
        skiprows=[1, 2],
        header=0,
        names=COLUMN_NAMES,
        parse_dates=["Datetime"],
        index_col="Datetime",
        dtype={"Volume": "int64"},
        na_values=["NA", "N/A", ""],
    )
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    print(f"Data shape before TA: {df.shape}")
    #
    # Fill NaNs in OHLC with previous values – volume NaNs become 0
    #
    # FIX: Use .ffill() instead of the deprecated fillna(method="ffill")
    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].ffill()
    df["Volume"] = df["Volume"].fillna(0)
    #
    # Add TA features – many columns … we will later compress.
    #
    df_ta = add_all_ta_features(
        df.copy(),
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True,
    )
    print(f"Data shape after TA : {df_ta.shape}")
    return df_ta

# ----------------------------- ENV ----------------------------- #
class TradingEnv:
    """
    Simple 1-asset environment with minute bars.
    State  : last `SEQ_LEN` rows of pre-computed feature matrix.
    Action : 0 HOLD, 1 BUY (go 1 BTC), 2 SELL (go flat).  (No shorting.)
    Reward : Δ-portfolio minus fees, minus inactivity penalty.
    """

    def __init__(self, df: pd.DataFrame, start_idx: int, end_idx: int):
        self.df = df
        self.start_idx = start_idx
        self.end_idx   = end_idx
        self.reset()

    # -------- Helper getters -------- #
    def _price(self, idx: int) -> float:
        return float(self.df.iloc[idx]["Close"])

    # -------------------------------- #
    def reset(self) -> np.ndarray:
        self.current = self.start_idx
        self.credit  = 100_000.0      # start with 100k USD
        self.holdings = 0.0           # BTC
        self.last_trade_step = 0
        self.done = False
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return (state, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")

        info = {"action": ACTIONS[action], "index": self.current}
        price = self._price(self.current)

        reward = 0.0
        # -------- Trading logic ---------- #
        if action == 1 and self.holdings == 0:  # BUY
            cost = price * (1 + TRADING_FEE)
            tradable_units = min(MAX_POSITION, self.credit / cost)
            if tradable_units > 0:
                self.credit -= tradable_units * cost
                self.holdings += tradable_units
                info["executed"] = True
                self.last_trade_step = 0
        elif action == 2 and self.holdings > 0:  # SELL
            proceeds = price * self.holdings * (1 - TRADING_FEE)
            self.credit += proceeds
            self.holdings = 0.0
            info["executed"] = True
            self.last_trade_step = 0
        else:
            info["executed"] = False

        # portfolio value after action
        portfolio_now = self.credit + self.holdings * price

        # Move to next step
        self.current += 1
        self.last_trade_step += 1
        self.done = self.current >= self.end_idx

        # portfolio value after price moves one minute
        price_next = self._price(self.current) if not self.done else price
        portfolio_next = self.credit + self.holdings * price_next

        # --- reward --- #
        reward = portfolio_next - portfolio_now
        # encourage activity – penalise no trade streak
        reward -= NO_TRADE_PENALTY * (self.last_trade_step**1.1)
        # discourage buy&hold stagnation
        reward -= HOLD_PENALTY * abs(self.holdings)

        state_next = self._get_state()
        info["credit"] = self.credit
        info["holdings"] = self.holdings
        info["portfolio"] = portfolio_next

        return state_next, reward, self.done, info

    # ------------------- #
    def _get_state(self) -> np.ndarray:
        """
        Returns shape (SEQ_LEN, n_features)
        Missing past values are padding zeros.
        """
        start = self.current - SEQ_LEN + 1
        if start < 0:
            start = 0
        seq = self.df.iloc[start : self.current + 1].values
        if len(seq) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(seq), seq.shape[1]))
            seq = np.vstack([pad, seq])
        return seq.astype(np.float32)

    # additional helpers for monitoring
    def portfolio_value(self) -> float:
        return self.credit + self.holdings * self._price(self.current)

# ------------------ Replay buffer ------------------ #
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ------------------- Model ------------------- #
class AttentionEncoder(nn.Module):
    def __init__(self, num_inputs: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(num_inputs, FEATURE_DIM),
            nn.ReLU(),
            nn.Linear(FEATURE_DIM, FEATURE_DIM),
            nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=FEATURE_DIM, num_heads=ATTN_HEADS, batch_first=True
        )
        self.norm = nn.LayerNorm(FEATURE_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, SEQ_LEN, raw_features)
        """
        b, t, f = x.shape
        x = self.pre(x.view(-1, f)).view(b, t, FEATURE_DIM)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        h = self.norm(attn_out + x)
        # Aggregate over sequence – mean pooling
        return h.mean(dim=1)

class QNetwork(nn.Module):
    def __init__(self, raw_feature_dim: int, n_actions: int):
        super().__init__()
        self.encoder = AttentionEncoder(raw_feature_dim)
        self.head = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, state: torch.Tensor):
        z = self.encoder(state)
        return self.head(z)

# ------------------- Agent ------------------- #
class DQNAgent:
    def __init__(self, feature_dim: int):
        self.n_actions = len(ACTIONS)
        self.policy_net = QNetwork(feature_dim, self.n_actions).to(DEVICE)
        self.target_net = QNetwork(feature_dim, self.n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.steps_done = 0
        self.eps = EPS_START

    # --------------- #
    def select_action(self, state: np.ndarray) -> int:
        self.steps_done += 1
        self.eps = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY_STEPS
        )
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

    # --------------- #
    def optimize(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        transitions = self.buffer.sample(BATCH_SIZE)
        batch = Transition(*transitions)

        state_batch = torch.tensor(
            np.array(batch.state), dtype=torch.float32, device=DEVICE
        )
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(-1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32, device=DEVICE
        ).unsqueeze(-1)
        next_state_batch = torch.tensor(
            np.array(batch.next_state), dtype=torch.float32, device=DEVICE
        )
        done_mask = torch.tensor(
            batch.done, dtype=torch.bool, device=DEVICE
        ).unsqueeze(-1)

        # Q(s,a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Q target
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            target = reward_batch + (1 - done_mask.float()) * GAMMA * next_q

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        # update target network
        if self.steps_done % TARGET_UPDATE_EVERY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------- #
    @staticmethod
    def projected_return(total_return_per_episode: float, days: float, target_days: int):
        return (1 + total_return_per_episode) ** (target_days / days) - 1

# ------------------ Utility (plots, metrics) ------------------ #
def plot_episode(df_price: pd.Series, trades: List[dict], credit: List[float],
                 holdings_val: List[float], portfolio: List[float], title: str,
                 filename: str):
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=("Price + Trades",
                        "Credit (USD)",
                        "Holdings value (USD)",
                        "Portfolio value (USD)"))
    # 1) price
    fig.add_trace(go.Scatter(x=df_price.index, y=df_price.values, name="Price"), row=1, col=1)
    # Trades
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for tr in trades:
        if tr.get("executed"): # Use .get() for safer access
            if tr["action"] == "BUY":
                buy_x.append(df_price.index[tr["index"]])
                buy_y.append(df_price.iloc[tr["index"]])
            elif tr["action"] == "SELL":
                sell_x.append(df_price.index[tr["index"]])
                sell_y.append(df_price.iloc[tr["index"]])
    if buy_x:
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode="markers",
                                 marker_symbol="triangle-up",
                                 marker_color="green", name="Buys"), row=1, col=1)
    if sell_x:
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode="markers",
                                 marker_symbol="triangle-down",
                                 marker_color="red", name="Sells"), row=1, col=1)
    # 2) credit
    fig.add_trace(go.Scatter(x=df_price.index, y=credit, name="Credit"), row=2, col=1)
    # 3) holdings value
    fig.add_trace(go.Scatter(x=df_price.index, y=holdings_val, name="Holdings"),
                  row=3, col=1)
    # 4) portfolio
    fig.add_trace(go.Scatter(x=df_price.index, y=portfolio, name="Portfolio"),
                  row=4, col=1)

    fig.update_layout(height=900, width=1000, title_text=title, showlegend=True)
    fig.write_html(filename)  # interactive HTML
    print(f"Saved plot => {filename}")

# -------------------- Training loop -------------------- #
def train(agent: DQNAgent, df: pd.DataFrame,
          train_start: int, train_end: int,
          episodes: int = MAX_EPISODES):
    episode_length = EPISODE_DAYS * 24 * 60  # minutes
    indices = list(range(train_start, train_end - episode_length - 1))

    idx0 = indices[0] # Initialize idx0

    for ep in range(episodes):
        # sample start index – keep 10 % overlap with previous ep
        if ep > 0:
            shift = int(episode_length * 0.1)
            # Ensure idx0 does not exceed the last possible start index
            next_idx0 = idx0 + episode_length - shift
            if next_idx0 >= indices[-1]:
                # If it exceeds, restart from a random earlier position or break
                print("Reached end of training data for episodes. Stopping.")
                break
            idx0 = next_idx0

        idx1 = idx0 + episode_length
        env = TradingEnv(df, idx0, idx1)

        state = env.reset()
        done = False
        info_list = []
        credit_hist, holdings_hist, port_hist = [], [], []
        returns_hist = []

        with tqdm(total=episode_length, desc=f"Episode {ep+1}/{episodes}") as pbar:
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.buffer.push(state, action, reward, next_state, done)
                state = next_state
                agent.optimize()

                # logs
                info_list.append(info)
                returns_hist.append(reward)
                credit_hist.append(info["credit"])
                holdings_hist.append(info["holdings"] * df.iloc[info["index"]]["Close"])
                port_hist.append(info["portfolio"])

                pbar.update(1)

        # ---- episode metrics ---- #
        total_return = (port_hist[-1] - port_hist[0]) / port_hist[0]
        week_proj = agent.projected_return(total_return, EPISODE_DAYS, 7)
        month_proj = agent.projected_return(total_return, EPISODE_DAYS, 30)
        year_proj = agent.projected_return(total_return, EPISODE_DAYS, 365)

        n_buys = sum(1 for i in info_list if i.get("executed") and i["action"] == "BUY")
        n_sells = sum(1 for i in info_list if i.get("executed") and i["action"] == "SELL")

        print(f"\nEp {ep+1} done. "
              f"Trades – buys: {n_buys} sells: {n_sells}. "
              f"Eps-return: {total_return*100:6.2f}% "
              f"week: {week_proj*100:6.2f}%  "
              f"month: {month_proj*100:6.2f}%  "
              f"year: {year_proj*100:6.2f}%  "
              f"epsilon: {agent.eps:5.3f}")

        # ---------- plots ---------- #
        # Ensure the slice is within bounds
        price_slice = df.iloc[idx0:idx1+1]["Close"]

        plot_episode(
            price_slice,
            info_list,
            credit_hist,
            holdings_hist,
            port_hist,
            title=f"Training Episode {ep+1}",
            filename=f"train_ep_{ep+1}.html",
        )

def evaluate(agent: DQNAgent, df: pd.DataFrame,
             start_idx: int, end_idx: int,
             tag: str = "validation"):
    env = TradingEnv(df, start_idx, end_idx)
    state = env.reset()
    done = False

    info_list = []
    credit_hist, holdings_hist, port_hist = [], [], []

    pbar_eval = tqdm(total=(end_idx-start_idx), desc=f"Evaluating [{tag}]")

    while not done:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = int(agent.policy_net(s).argmax(dim=1).item())
        next_state, _, done, info = env.step(action)
        state = next_state

        info_list.append(info)
        credit_hist.append(info["credit"])
        holdings_hist.append(info["holdings"] * df.iloc[info["index"]]["Close"])
        port_hist.append(info["portfolio"])
        pbar_eval.update(1)

    pbar_eval.close()
    
    days = (end_idx - start_idx) / (60 * 24)
    total_return = (port_hist[-1] - port_hist[0]) / port_hist[0]
    week_proj = DQNAgent.projected_return(total_return, days, 7)
    month_proj = DQNAgent.projected_return(total_return, days, 30)
    year_proj = DQNAgent.projected_return(total_return, days, 365)

    n_buys = sum(1 for i in info_list if i.get("executed") and i["action"] == "BUY")
    n_sells = sum(1 for i in info_list if i.get("executed") and i["action"] == "SELL")

    print(f"\n[{tag}] return: {total_return*100:6.2f}%   "
          f"week: {week_proj*100:6.2f}%  month: {month_proj*100:6.2f}%  "
          f"year: {year_proj*100:6.2f}%   "
          f"buys: {n_buys}   sells: {n_sells}")

    # Ensure the slice is within bounds for plotting
    price_slice = df.iloc[start_idx:end_idx+1]["Close"]
    plot_episode(
        price_slice,
        info_list,
        credit_hist,
        holdings_hist,
        port_hist,
        title=f"{tag.capitalize()} run",
        filename=f"{tag}_run.html",
    )

# ===================== MAIN ===================== #
def main():
    df = load_data()

    # chronological split
    total_len = len(df)
    train_end = int(total_len * 0.7)
    val_end   = int(total_len * 0.85)

    train_start = SEQ_LEN  # skip the first 'SEQ_LEN' minutes for padding simplicity

    print(f"Dataset length  : {total_len}")
    print(f"Train   indices : {train_start} … {train_end}")
    print(f"Valid   indices : {train_end} … {val_end}")
    print(f"Test    indices : {val_end} … {total_len}")

    # FIX: Use the correct keyword argument 'feature_dim'
    agent = DQNAgent(feature_dim=df.shape[1])

    # --------------- training --------------- #
    train(
        agent,
        df,
        train_start=train_start,
        train_end=train_end,
        episodes=20  # fewer for demo; raise to MAX_EPISODES for long training
    )

    # --------------- validation ------------- #
    evaluate(agent, df, train_end, val_end, tag="validation")

    # --------------- testing ---------------- #
    evaluate(agent, df, val_end, total_len-1, tag="test")

if __name__ == "__main__":
    # Silence pandas SettingWithCopy warnings for readability
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    main()
