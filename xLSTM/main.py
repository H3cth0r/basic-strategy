# ------------------------------------------------------------
# xLSTM + Recurrent PPO for BTC-USD intraday trading (1-min)
# Single-file implementation with data loading, indicators (ta),
# custom xLSTM, environment, PPO training (epoch-based), and backtesting.
# Uses tqdm for progress and matplotlib for plots.
# Device: MPS/CUDA/CPU auto-detected.
# Numerically-stabilized to avoid NaNs on MPS/CUDA.
# Improved environment and training to reduce churn and seek profitability:
#  - Actions map to target position (0.0, 0.5, 1.0) instead of fixed buy/sell.
#  - Rebalance only when change exceeds threshold to avoid overtrading.
#  - Log-return reward with turnover penalty.
#  - Turbulence forces flat position.
#  - Longer epochs covering full training data.
#  - More detailed logging per epoch and tqdm in evaluation.
# ------------------------------------------------------------

import os
import math
import time
import random
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

import matplotlib.pyplot as plt
from tqdm import tqdm

# Technical indicators
from ta import add_all_ta_features

# Torch setup (MPS/CUDA/CPU)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (12, 6)

# ---------------------------
# Reproducibility
# ---------------------------
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# ---------------------------
# Device selection
# ---------------------------
def get_device():
    if torch.backends.mps.is_available():
        print("Using device: MPS")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using device: CUDA")
        return torch.device("cuda")
    else:
        print("Using device: CPU")
        return torch.device("cpu")

DEVICE = get_device()
TORCH_DTYPE = torch.float32

# Small helper to sanitize tensors
def safe_tensor(t: torch.Tensor, clamp_val: float = 50.0) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
    t = torch.clamp(t, -clamp_val, clamp_val)
    return t

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
def get_data():
    print("Downloading and preprocessing data…")
    url = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"
    column_names = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
    try:
        df = pd.read_csv(
            url, skiprows=[1, 2], header=0, names=column_names,
            parse_dates=["Datetime"], index_col="Datetime",
            dtype={"Close": "float64", "High": "float64", "Low": "float64", "Open": "float64", "Volume": "float64"},
            na_values=["NA", "N/A", "", "NaN", "nan", "INF", "-INF"],
            keep_default_na=True,
        )
        df.index = pd.to_datetime(df.index, utc=True)
    except Exception as e:
        print(f"Error reading data: {e}")
        return pd.DataFrame()

    # Forward/backward fill and drop NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(inplace=True)

    # Add technical indicators
    print("Calculating technical indicators…")
    add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )

    # Preserve original close
    original_close = df["Close"].copy()

    # Z-score normalize features (clip after to stabilize)
    for col in df.columns:
        if col != 'Original_Close':  # will be added after
            mu = df[col].mean()
            sigma = df[col].std()
            sigma = sigma if sigma and not np.isnan(sigma) else 1e-7
            df[col] = (df[col] - mu) / (sigma + 1e-7)

    df["Original_Close"] = original_close
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Clip features to avoid extreme values that may cause instabilities
    feature_cols = [c for c in df.columns if c not in ["Original_Close"]]
    df[feature_cols] = df[feature_cols].clip(-10.0, 10.0)

    # Minute returns for turbulence and Sharpe calculations
    df["Return"] = df["Original_Close"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    print("Data loaded and preprocessed successfully.")
    print(f"Data shape: {df.shape}")
    return df

# ---------------------------
# Train/Validation/Test Split
# ---------------------------
def split_data(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    return train_df, val_df, test_df

# ---------------------------
# Trading Environment (Gym-like)
# ---------------------------
class BTCRLEnv:
    """
    BTC Trading RL Environment with discrete actions mapped to target position ratios:
    Actions: 0 -> Keep current target; 1 -> target 0.5 long; 2 -> target 1.0 long
    - Turbulence forces target to 0.0 (flat).
    - Rebalance only if |target - current_pos| > rebalance_threshold to avoid churn.
    Reward: log-return of portfolio minus transaction cost and turnover penalty.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        time_window: int = 30,
        initial_balance: float = 1_000_000.0,
        cost_rate: float = 0.0005,     # 0.05% per trade
        rebalance_threshold: float = 0.10,  # rebalance only if change >10% of portfolio
        turnover_penalty: float = 0.10,     # penalize turnover in reward
        turbulence_window: int = 60,  # minutes
        turbulence_threshold: float = 3.0,  # abs z-score threshold
        feature_exclude: List[str] = None
    ):
        self.data = data.copy()
        self.time_window = time_window
        self.initial_balance = initial_balance
        self.cost_rate = cost_rate
        self.rebalance_threshold = rebalance_threshold
        self.turnover_penalty = turnover_penalty
        self.turbulence_window = turbulence_window
        self.turbulence_threshold = turbulence_threshold

        if feature_exclude is None:
            feature_exclude = ["Original_Close", "Return"]
        self.feature_cols = [c for c in self.data.columns if c not in feature_exclude]

        # Precompute turbulence as abs z-score of returns over window
        self.data["RetMean"] = self.data["Return"].rolling(self.turbulence_window, min_periods=1).mean()
        self.data["RetStd"] = self.data["Return"].rolling(self.turbulence_window, min_periods=1).std().replace(0, 1e-8)
        self.data["TurbulenceIndex"] = np.abs((self.data["Return"] - self.data["RetMean"]) / (self.data["RetStd"] + 1e-8)).fillna(0.0)

        # Sanitize
        self.data.replace([np.inf, -np.inf], 0.0, inplace=True)

        # Mapping of actions to target positions
        self.action_targets = {0: None, 1: 0.5, 2: 1.0}

        self.reset()

    def _obs_from_index(self, idx: int) -> np.ndarray:
        # Safe observation getter using bounded idx
        idx = max(0, min(idx, len(self.data) - 1))
        row = self.data.iloc[idx]
        obs = row[self.feature_cols].values.astype(np.float32)
        price = float(row["Original_Close"])
        position_value = self.shares * price
        cash_ratio = self.balance / (self.total_value + 1e-8)
        pos_ratio = position_value / (self.total_value + 1e-8)
        obs = np.concatenate([obs, np.array([cash_ratio, pos_ratio], dtype=np.float32)])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        return obs

    def reset(self) -> np.ndarray:
        self.idx = 0
        self.balance = self.initial_balance
        self.shares = 0.0
        self.total_value = self.initial_balance
        self.prev_total_value = self.initial_balance
        self.done = False

        self.target_pos = 0.0  # target position ratio
        self.n_trades = 0
        self.equity_curve = [self.initial_balance]
        self.step_returns = []
        self.peak_value = self.initial_balance
        self.drawdowns = []
        self.actions_taken = []

        # Skip initial time window as warmup (hold)
        self.idx = max(self.idx, self.time_window - 1)
        return self._obs_from_index(self.idx)

    def _current_pos_ratio(self, price: float) -> float:
        position_value = self.shares * price
        return position_value / (self.total_value + 1e-8)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Action: 0 keep target, 1 -> target 0.5, 2 -> target 1.0
        Turbulence forces target to 0.0 (flat)
        """
        if self.done:
            return self._obs_from_index(self.idx), 0.0, True, {}

        row = self.data.iloc[self.idx]
        price = float(row["Original_Close"])
        turbulence_index = float(row["TurbulenceIndex"])
        reward = 0.0

        # Update target position
        if turbulence_index > self.turbulence_threshold:
            new_target = 0.0  # force flat when turbulent
            turbulence_forced_flat = True
        else:
            target = self.action_targets.get(action, None)
            new_target = self.target_pos if target is None else target
            turbulence_forced_flat = False

        # Compute current position ratio
        current_pos = self._current_pos_ratio(price)
        change = new_target - current_pos

        turnover = 0.0
        if abs(change) > self.rebalance_threshold:
            # Rebalance using available cash and shares to reach target
            desired_position_value = new_target * self.total_value
            current_position_value = self.shares * price
            diff_value = desired_position_value - current_position_value

            if diff_value > 0:  # buy
                buy_cash = min(self.balance, diff_value)
                if buy_cash > 0:
                    shares_to_buy = buy_cash / (price + 1e-12)
                    fee = buy_cash * self.cost_rate
                    self.balance -= (buy_cash + fee)
                    self.shares += shares_to_buy
                    self.n_trades += 1
                    turnover = buy_cash / (self.prev_total_value + 1e-8)
            elif diff_value < 0:  # sell
                sell_shares = min(self.shares, abs(diff_value) / (price + 1e-12))
                if sell_shares > 0:
                    sell_cash = sell_shares * price
                    fee = sell_cash * self.cost_rate
                    self.balance += (sell_cash - fee)
                    self.shares -= sell_shares
                    self.n_trades += 1
                    turnover = sell_cash / (self.prev_total_value + 1e-8)

        # Update target_pos after rebalancing
        self.target_pos = new_target

        # Update portfolio value
        self.prev_total_value = self.total_value
        self.total_value = self.balance + self.shares * price

        # Reward: log-return minus turnover penalty; small penalty if turbulence forced flat
        log_ret = math.log(max(self.total_value / (self.prev_total_value + 1e-12), 1e-12))
        reward = log_ret - self.turnover_penalty * turnover
        if turbulence_forced_flat and action != 0:
            reward -= 0.0005  # small penalty for trying to trade in turbulence

        # Track returns and drawdown
        step_ret = (self.total_value / (self.prev_total_value + 1e-8)) - 1.0
        if not np.isfinite(step_ret):
            step_ret = 0.0
        self.step_returns.append(step_ret)
        self.equity_curve.append(self.total_value)
        self.peak_value = max(self.peak_value, self.total_value)
        dd = (self.total_value - self.peak_value) / (self.peak_value + 1e-8)
        self.drawdowns.append(dd)

        # Advance time
        self.idx += 1
        if self.idx >= len(self.data):
            self.done = True
            obs = self._obs_from_index(self.idx - 1)
        else:
            obs = self._obs_from_index(self.idx)

        info = {
            "price": price,
            "balance": self.balance,
            "shares": self.shares,
            "value": self.total_value,
            "turbulence": turbulence_index,
            "target_pos": self.target_pos,
            "turnover": turnover,
            "log_ret": log_ret,
        }
        self.actions_taken.append(action)
        return obs, float(reward), self.done, info

# ---------------------------
# xLSTM Implementation (simplified + stabilized)
# ---------------------------
class XLSTMCell(nn.Module):
    """
    Simplified xLSTM-style cell with exponential gating and memory mixing.
    Numerically-stabilized to avoid NaNs:
      - clamp pre-exp to [-4, 4]
      - nan_to_num on intermediates
      - LayerNorm on output
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Gates and updates
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_e = nn.Linear(input_size, hidden_size)
        self.U_e = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_m = nn.Linear(input_size, hidden_size)
        self.U_m = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_o = nn.Linear(hidden_size, hidden_size, bias=False)

        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_i, self.U_i, self.W_e, self.U_e, self.W_m, self.U_m, self.W_o, self.U_o, self.V_o]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, h_prev, m_prev):
        x = safe_tensor(x)
        h_prev = safe_tensor(h_prev)
        m_prev = safe_tensor(m_prev)

        # input gate (sigmoid-like)
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        i = safe_tensor(i)

        # exponential gate (positive scaling) - clipped for stability
        e_pre = torch.clamp(self.W_e(x) + self.U_e(h_prev), min=-4.0, max=4.0)
        e = torch.exp(e_pre)
        e = safe_tensor(e)

        # memory proposal
        m_tilde = self.act(self.W_m(x) + self.U_m(h_prev))
        m_tilde = safe_tensor(m_tilde)

        # memory mixing
        m = e * m_prev + i * m_tilde
        m = safe_tensor(m)

        # output
        o_pre = self.W_o(x) + self.U_o(h_prev) + self.V_o(m)
        o = self.act(o_pre)
        o = self.norm(o)
        o = safe_tensor(o)

        h = o
        return h, m

class XLSTM(nn.Module):
    """
    Stacked XLSTM layers (configurable).
    Maintains (h, m) state.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.layers.append(XLSTMCell(in_size, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, m):
        """
        x: [batch, input_size]
        h/m: list of states per layer, each [batch, hidden_size]
        returns new h/m and last layer output
        """
        outputs = safe_tensor(x)
        new_h = []
        new_m = []
        for i, layer in enumerate(self.layers):
            h_i, m_i = layer(outputs, h[i], m[i])
            outputs = self.dropout(h_i)
            outputs = safe_tensor(outputs)
            new_h.append(h_i)
            new_m.append(m_i)
        return outputs, new_h, new_m

    def init_state(self, batch_size: int, device: torch.device):
        h = [torch.zeros(batch_size, self.hidden_size, device=device, dtype=TORCH_DTYPE) for _ in range(self.num_layers)]
        m = [torch.zeros(batch_size, self.hidden_size, device=device, dtype=TORCH_DTYPE) for _ in range(self.num_layers)]
        return h, m

# ---------------------------
# Actor-Critic using XLSTM
# ---------------------------
class ActorXLSTM(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, num_actions: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.xlstm = XLSTM(128, hidden_size, num_layers=num_layers, dropout=dropout)
        self.policy_head = nn.Linear(hidden_size, num_actions)
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, obs, h, m):
        # obs: [batch, obs_size]
        obs = safe_tensor(obs)
        z = self.feature_extractor(obs)
        z = safe_tensor(z)
        out, h_new, m_new = self.xlstm(z, h, m)
        logits = self.policy_head(out)
        logits = safe_tensor(logits, clamp_val=20.0)
        return logits, h_new, m_new

    def init_state(self, batch_size: int, device: torch.device):
        return self.xlstm.init_state(batch_size, device)

class CriticXLSTM(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.xlstm = XLSTM(128, hidden_size, num_layers=num_layers, dropout=dropout)
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, h, m):
        obs = safe_tensor(obs)
        z = self.feature_extractor(obs)
        z = safe_tensor(z)
        out, h_new, m_new = self.xlstm(z, h, m)
        value = self.value_head(out)
        value = safe_tensor(value).squeeze(-1)
        return value, h_new, m_new

    def init_state(self, batch_size: int, device: torch.device):
        return self.xlstm.init_state(batch_size, device)

# ---------------------------
# PPO Agent (Recurrent, epoch-based training)
# ---------------------------
class PPOAgent:
    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        hidden_size: int = 128,
        xlstm_layers: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.num_actions = num_actions

        self.actor = ActorXLSTM(obs_size, hidden_size, num_actions, num_layers=xlstm_layers).to(self.device)
        self.critic = CriticXLSTM(obs_size, hidden_size, num_layers=xlstm_layers).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Learning rate schedulers (help stronger training)
        self.actor_sched = optim.lr_scheduler.CosineAnnealingLR(self.actor_opt, T_max=50, eta_min=1e-5)
        self.critic_sched = optim.lr_scheduler.CosineAnnealingLR(self.critic_opt, T_max=50, eta_min=1e-5)

        # Temperature for action sampling (decays over epochs to reduce exploration)
        self.temperature = 1.0

    def set_temperature(self, temp: float):
        self.temperature = max(0.5, float(temp))

    def act(self, obs_t: np.ndarray, actor_state: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> Tuple[int, float, Tuple, torch.Tensor]:
        obs_t = np.nan_to_num(obs_t, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        obs_t = torch.tensor(obs_t, device=self.device, dtype=TORCH_DTYPE).unsqueeze(0)
        h_a, m_a = actor_state
        logits, h_new, m_new = self.actor(obs_t, h_a, m_a)
        logits = safe_tensor(logits, clamp_val=20.0)
        # Temperature scaling (lower temp -> more exploitative)
        logits = logits / self.temperature
        if not torch.isfinite(logits).all():
            logits = torch.zeros_like(logits)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), (h_new, m_new), logits.detach()

    def value(self, obs_t: np.ndarray, critic_state: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> Tuple[float, Tuple]:
        obs_t = np.nan_to_num(obs_t, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        obs_t = torch.tensor(obs_t, device=self.device, dtype=TORCH_DTYPE).unsqueeze(0)
        h_c, m_c = critic_state
        value, h_new, m_new = self.critic(obs_t, h_c, m_c)
        value = safe_tensor(value)
        return float(value.item()), (h_new, m_new)

    def compute_gae(self, rewards, values, dones, last_value):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(values, dtype=np.float32)
        # Normalize advantages for stability
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        return advantages, returns

    def update(self, rollout: Dict[str, Any], epochs: int = 4, batch_size: int = 1024):
        obs = torch.tensor(rollout["obs"], device=self.device, dtype=TORCH_DTYPE)
        actions = torch.tensor(rollout["actions"], device=self.device, dtype=torch.long)
        old_logprobs = torch.tensor(rollout["logprobs"], device=self.device, dtype=TORCH_DTYPE)
        advantages = torch.tensor(rollout["advantages"], device=self.device, dtype=TORCH_DTYPE)
        returns = torch.tensor(rollout["returns"], device=self.device, dtype=TORCH_DTYPE)

        # Sanitize
        obs = safe_tensor(obs)
        advantages = safe_tensor(advantages, clamp_val=10.0)
        returns = safe_tensor(returns, clamp_val=10.0)

        dataset_size = obs.shape[0]
        idxs = np.arange(dataset_size)

        actor_losses = []
        critic_losses = []
        entropies = []
        approx_kls = []
        explained_vars = []

        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # Reset recurrent states for mini-batch (approximate; proper recurrent PPO needs sequence batching)
                h_a, m_a = self.actor.init_state(batch_size=mb_obs.shape[0], device=self.device)
                logits, _, _ = self.actor(mb_obs, h_a, m_a)
                logits = safe_tensor(logits, clamp_val=20.0)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Approx KL
                approx_kl = (mb_old_logprobs - new_logprobs).mean()

                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                # Critic update
                h_c, m_c = self.critic.init_state(batch_size=mb_obs.shape[0], device=self.device)
                values_pred, _, _ = self.critic(mb_obs, h_c, m_c)
                values_pred = safe_tensor(values_pred, clamp_val=10.0)
                value_loss = self.value_coef * (mb_returns - values_pred).pow(2).mean()

                # Explained variance
                var_y = torch.var(mb_returns)
                ev = 1.0 - torch.var(mb_returns - values_pred) / (var_y + 1e-8)

                self.critic_opt.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                approx_kls.append(float(approx_kl.item()))
                explained_vars.append(float(ev.item()))

        stats = {
            "actor_loss": np.mean(actor_losses) if actor_losses else 0.0,
            "critic_loss": np.mean(critic_losses) if critic_losses else 0.0,
            "entropy": np.mean(entropies) if entropies else 0.0,
            "approx_kl": np.mean(approx_kls) if approx_kls else 0.0,
            "explained_var": np.mean(explained_vars) if explained_vars else 0.0,
        }
        # Step schedulers once per update cycle (epoch)
        self.actor_sched.step()
        self.critic_sched.step()
        return stats

# ---------------------------
# Training Loop (epoch-based)
# ---------------------------
def train_agent(
    env: BTCRLEnv,
    agent: PPOAgent,
    epochs: int = 6,
    steps_per_epoch: int = None,  # default: full dataset
    rollout_len: int = 2_048,
    update_epochs: int = 6,
    batch_size: int = 1_024,
) -> Dict[str, Any]:
    # Use full training data per epoch by default
    if steps_per_epoch is None:
        steps_per_epoch = max(1, len(env.data) - env.time_window - 1)

    obs = env.reset()
    h_a, m_a = agent.actor.init_state(batch_size=1, device=agent.device)
    h_c, m_c = agent.critic.init_state(batch_size=1, device=agent.device)

    all_stats = []
    total_steps_done = 0

    for ep in range(1, epochs + 1):
        # Slightly reduce exploration temperature each epoch
        agent.set_temperature(temp=max(0.7, 1.0 - 0.05 * (ep - 1)))

        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {ep}/{epochs}", leave=True)
        ep_stats = []
        ep_rewards = []
        ep_actions = []
        ep_turnover = 0.0
        start_value = env.total_value

        steps_this_epoch = 0
        while steps_this_epoch < steps_per_epoch:
            # Rollout buffer
            rollout = {
                "obs": [],
                "actions": [],
                "logprobs": [],
                "rewards": [],
                "values": [],
                "dones": [],
            }

            # Collect rollout
            inner_len = min(rollout_len, steps_per_epoch - steps_this_epoch)
            for _ in range(inner_len):
                action, logprob, (h_a, m_a), _ = agent.act(obs, (h_a, m_a))
                value, (h_c, m_c) = agent.value(obs, (h_c, m_c))

                next_obs, reward, done, info = env.step(action)

                rollout["obs"].append(obs)
                rollout["actions"].append(action)
                rollout["logprobs"].append(logprob)
                rollout["rewards"].append(reward)
                rollout["values"].append(value)
                rollout["dones"].append(1.0 if done else 0.0)

                obs = next_obs
                steps_this_epoch += 1
                total_steps_done += 1
                pbar.update(1)

                # Accumulate monitoring
                ep_rewards.append(reward)
                ep_actions.append(action)
                ep_turnover += info.get("turnover", 0.0)

                if done:
                    # Reset environment and states
                    obs = env.reset()
                    h_a, m_a = agent.actor.init_state(batch_size=1, device=agent.device)
                    h_c, m_c = agent.critic.init_state(batch_size=1, device=agent.device)

                if steps_this_epoch >= steps_per_epoch:
                    break

            # Bootstrap value for GAE
            last_value, _ = agent.value(obs, (h_c, m_c))
            adv, returns = agent.compute_gae(
                rewards=np.array(rollout["rewards"], dtype=np.float32),
                values=np.array(rollout["values"], dtype=np.float32),
                dones=np.array(rollout["dones"], dtype=np.float32),
                last_value=last_value
            )

            # Prepare arrays
            rollout_np = {
                "obs": np.stack(rollout["obs"]).astype(np.float32),
                "actions": np.array(rollout["actions"], dtype=np.int64),
                "logprobs": np.array(rollout["logprobs"], dtype=np.float32),
                "advantages": adv.astype(np.float32),
                "returns": returns.astype(np.float32),
            }

            stats = agent.update(rollout_np, epochs=update_epochs, batch_size=batch_size)
            ep_stats.append(stats)

            # Show partial stats on pbar
            pbar.set_postfix({
                "AL": f"{stats['actor_loss']:.3f}",
                "CL": f"{stats['critic_loss']:.3f}",
                "Ent": f"{stats['entropy']:.3f}",
                "KL": f"{stats['approx_kl']:.3f}",
                "EV": f"{stats['explained_var']:.3f}",
            })

        pbar.close()
        # Epoch summary metrics
        end_value = env.total_value
        epoch_return = (end_value - start_value) / start_value
        avg_reward = np.mean(ep_rewards) if ep_rewards else 0.0
        action_counts = {0: 0, 1: 0, 2: 0}
        for a in ep_actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        total_actions = max(1, len(ep_actions))
        action_dist = {k: v / total_actions for k, v in action_counts.items()}
        mean_actor = np.mean([s["actor_loss"] for s in ep_stats]) if ep_stats else 0.0
        mean_critic = np.mean([s["critic_loss"] for s in ep_stats]) if ep_stats else 0.0
        mean_entropy = np.mean([s["entropy"] for s in ep_stats]) if ep_stats else 0.0
        mean_kl = np.mean([s["approx_kl"] for s in ep_stats]) if ep_stats else 0.0
        mean_ev = np.mean([s["explained_var"] for s in ep_stats]) if ep_stats else 0.0

        print(f"Epoch {ep} Summary | "
              f"ActorLoss: {mean_actor:.4f} | CriticLoss: {mean_critic:.4f} | Entropy: {mean_entropy:.4f} | "
              f"KL: {mean_kl:.4f} | EV: {mean_ev:.4f} | "
              f"AvgReward: {avg_reward:.6f} | Turnover: {ep_turnover:.4f} | "
              f"ActionDist: {action_dist} | "
              f"EpochReturn: {epoch_return:.4f} | Temp: {agent.temperature:.2f} | Trades: {env.n_trades}")

        all_stats.append({
            "epoch": ep,
            "actor_loss": mean_actor,
            "critic_loss": mean_critic,
            "entropy": mean_entropy,
            "approx_kl": mean_kl,
            "explained_var": mean_ev,
            "avg_reward": avg_reward,
            "turnover": ep_turnover,
            "action_dist": action_dist,
            "epoch_return": epoch_return,
            "trades": env.n_trades
        })

    return {"train_stats": all_stats, "total_steps": total_steps_done}

# ---------------------------
# Evaluation and Backtesting
# ---------------------------
def evaluate_env(env: BTCRLEnv, agent: PPOAgent, name: str = "Eval", verbose: bool = False) -> Dict[str, Any]:
    obs = env.reset()
    h_a, m_a = agent.actor.init_state(batch_size=1, device=agent.device)
    h_c, m_c = agent.critic.init_state(batch_size=1, device=agent.device)

    # tqdm over evaluation length
    total_eval_steps = max(1, len(env.data) - env.time_window - 1)
    pbar = tqdm(total=total_eval_steps, desc=f"{name} Evaluation", leave=False)

    steps = 0
    while True:
        action, logprob, (h_a, m_a), logits = agent.act(obs, (h_a, m_a))
        obs, reward, done, info = env.step(action)
        steps += 1
        if verbose and steps % 2000 == 0:
            pbar.set_postfix({
                "Price": f"{info['price']:.2f}",
                "Value": f"{info['value']:.2f}",
                "Target": f"{info['target_pos']:.2f}",
                "Ret": f"{info['log_ret']:.6f}"
            })
        pbar.update(1)
        if done or steps >= total_eval_steps:
            break

    pbar.close()

    # Metrics
    equity = np.array(env.equity_curve, dtype=np.float64)
    returns = np.array(env.step_returns, dtype=np.float64)
    init = env.initial_balance
    final = equity[-1]
    cr = (final - init) / init
    mer = np.max((equity - init) / init) if equity.size > 0 else 0.0
    # Maximum drawdown (positive number)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / (running_max + 1e-8)
    mpb = -np.min(drawdown) if drawdown.size > 0 else 0.0
    appt = (final - init) / (env.n_trades if env.n_trades > 0 else 1)
    # Sharpe ratio (minute data, risk-free ~0). Annualize by sqrt(minutes/year).
    mean_ret = returns.mean() if returns.size > 0 else 0.0
    std_ret = returns.std() + 1e-12
    ann_factor = math.sqrt(525600.0)
    sr = (mean_ret / std_ret) * ann_factor if std_ret > 0 else 0.0

    metrics = {"CR": float(cr), "MER": float(mer), "MPB": float(mpb), "APPT": float(appt), "SR": float(sr)}
    return {
        "metrics": metrics,
        "equity_curve": equity,
        "returns": returns,
        "trades": env.n_trades,
        "actions": env.actions_taken
    }

def plot_equity(equity_curve: np.ndarray, title: str):
    plt.figure()
    plt.plot(equity_curve, label="Equity")
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------------------
# Main
# ---------------------------
def main():
    print("You are interacting with GPT-5 (reasoning).")
    df = get_data()

    # Split data
    train_df, val_df, test_df = split_data(df, train_ratio=0.8, val_ratio=0.1)

    # Feature dimension (we append two ratios in env)
    feature_exclude = ["Original_Close", "Return"]
    feature_cols = [c for c in df.columns if c not in feature_exclude]
    obs_size = len(feature_cols) + 2  # + cash_ratio + pos_ratio

    # Hyperparameters
    time_window = 30
    initial_balance = 1_000_000.0
    turbulence_threshold = 3.0

    # Epoch config: Use full training dataset per epoch
    epochs = 6
    steps_per_epoch = len(train_df) - time_window - 1
    rollout_len = 2048
    update_epochs = 6
    batch_size = 1024

    # Environments (same settings across splits)
    env_kwargs = dict(
        time_window=time_window,
        initial_balance=initial_balance,
        cost_rate=0.0005,
        rebalance_threshold=0.10,
        turnover_penalty=0.10,
        turbulence_window=60,
        turbulence_threshold=turbulence_threshold,
        feature_exclude=feature_exclude
    )
    train_env = BTCRLEnv(train_df, **env_kwargs)
    val_env = BTCRLEnv(val_df, **env_kwargs)
    test_env = BTCRLEnv(test_df, **env_kwargs)

    # Agent (deeper xLSTM for stronger modeling)
    agent = PPOAgent(
        obs_size=obs_size,
        num_actions=3,
        hidden_size=128,
        xlstm_layers=2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=1.0,
        device=DEVICE,
    )

    print("Starting training (epoch-based) with full training data per epoch…")
    train_info = train_agent(
        env=train_env,
        agent=agent,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        rollout_len=rollout_len,
        update_epochs=update_epochs,
        batch_size=batch_size,
    )
    print("Training completed.")

    # Evaluate on Train (backtest on seen data)
    print("Evaluating on train data (backtest)…")
    train_eval = evaluate_env(train_env, agent, name="Train", verbose=True)
    print(f"Train Metrics: {train_eval['metrics']}")
    plot_equity(train_eval["equity_curve"], title="Train Equity Curve")

    # Evaluate on Validation
    print("Evaluating on validation data…")
    val_eval = evaluate_env(val_env, agent, name="Validation", verbose=True)
    print(f"Validation Metrics: {val_eval['metrics']}")
    plot_equity(val_eval["equity_curve"], title="Validation Equity Curve")

    # Evaluate on Test (final backtest)
    print("Evaluating on test data (backtest)…")
    test_eval = evaluate_env(test_env, agent, name="Test", verbose=True)
    print(f"Test Metrics: {test_eval['metrics']}")
    plot_equity(test_eval["equity_curve"], title="Test Equity Curve")

    # Print summary
    def fmt_metrics(m: Dict[str, float]) -> str:
        return " | ".join([f"{k}: {v:.4f}" for k, v in m.items()])

    print("Summary:")
    print(f"- Train: {fmt_metrics(train_eval['metrics'])}, Trades: {train_eval['trades']}")
    print(f"- Val:   {fmt_metrics(val_eval['metrics'])}, Trades: {val_eval['trades']}")
    print(f"- Test:  {fmt_metrics(test_eval['metrics'])}, Trades: {test_eval['trades']}")

    # Optional: plot actions distribution on test
    actions = test_eval["actions"]
    plt.figure()
    plt.hist(actions, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8)
    plt.xticks([0, 1, 2], ["Keep", "0.5 Long", "1.0 Long"])
    plt.title("Action Distribution (Test)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
