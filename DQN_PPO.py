# single_file_drl_trader.py
# Deep RL ensemble for BTC-USD trading using ta indicators, PPO/A2C/DDPG, tqdm progress, Plotly plots.
# Backtests: Walk-Forward, Time-series K-Fold, Hyperparameter Tuning with combinatorial CV and p(overfitting).
# MPS-safe: no Beta/Dirichlet ops; only Normal/Gaussian sampling.
# Plots: Portfolio Value, Cash, Holdings Value, Cumulative Returns, Price with Buy/Sell markers and RSI/MACD.
# Emphasis on reducing churn and drawdowns via position-target actions, trade smoothing, and volatility risk-off.

import os
import sys
import math
import time
import random
import itertools
import warnings
from typing import List, Tuple, Dict, Any

warnings.filterwarnings("ignore")

# Ensure required packages
def _ensure_pkg(pkg):
    try:
        __import__(pkg)
    except ImportError:
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except Exception as e:
            print(f"Warning: failed to install {pkg}: {e}")

for _pkg in ["torch", "numpy", "pandas", "ta", "plotly", "tqdm"]:
    _ensure_pkg(_pkg)

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Device selection: prefer MPS on macOS, else CUDA, else CPU
DEVICE = torch.device(
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
torch.set_float32_matmul_precision("medium")

SEED = 123
def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_all(SEED)

################################################################################
# Data Loading & Indicators (using ta)
################################################################################

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

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    import ta

    # Momentum indicators
    df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    df["ultosc"] = ta.momentum.UltimateOscillator(
        high=df["High"], low=df["Low"], close=df["Close"],
        window1=7, window2=14, window3=28
    ).ultimate_oscillator()
    df["williams_r"] = ta.momentum.WilliamsRIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], lbp=14
    ).williams_r()
    df["roc"] = ta.momentum.ROCIndicator(close=df["Close"], window=10).roc()

    # Trend indicators
    macd = ta.trend.MACD(close=df["Close"])
    df["macd"] = macd.macd()
    df["macd_diff"] = macd.macd_diff()
    df["macd_signal"] = macd.macd_signal()
    df["cci"] = ta.trend.CCIIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=20, constant=0.015
    ).cci()
    df["adx"] = ta.trend.ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).adx()

    # Volume indicators
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"]
    ).on_balance_volume()

    # Volatility (ATR)
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range()

    # Returns & realized volatility
    df["log_ret"] = np.log(df["Close"]).diff()
    df["realized_vol"] = df["log_ret"].rolling(72, min_periods=20).std()

    df = df.dropna()
    return df

def drop_correlated_features(df: pd.DataFrame, features: List[str], threshold: float=0.6) -> List[str]:
    corr = df[features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = []
    for c in upper.columns:
        if any(upper[c] > threshold):
            to_drop.append(c)
    keep = [f for f in features if f not in to_drop]
    return keep

def standardize_features(df: pd.DataFrame, feature_cols: List[str], ref_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[float,float]]]:
    stats = {}
    df = df.copy()
    for c in feature_cols:
        mu = float(ref_df[c].mean())
        sd = float(ref_df[c].std())
        sd = sd if sd > 1e-8 else 1.0
        stats[c] = (mu, sd)
        df[c] = (df[c] - mu) / sd
    return df, stats

################################################################################
# Trading Environment (single asset BTC-USD)
# Action: target position fraction a in [0, 1] (long-only), with smoothing and fee
################################################################################

class CryptoTradingEnv:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        initial_cash: float=100000.0,
        fee_rate: float=0.001,                # 0.1% fee
        slippage_bp: float=0.0,               # optional slippage (basis points)
        risk_vol_threshold: float=None,       # if realized_vol > threshold: liquidate + halt buys
        position_smooth: float=0.2,           # fraction of gap to target executed per step (0<alpha<=1)
        max_trade_frac: float=0.4,            # cap trade value as fraction of portfolio per step
        obs_stack: int=8,                     # stack last N observations to give context
        normalize_features: bool=True,
        feature_stats: Dict[str, Tuple[float,float]]=None
    ):
        self.df = df
        self.feature_cols = feature_cols
        self.initial_cash = float(initial_cash)
        self.fee_rate = float(fee_rate)
        self.slippage_bp = float(slippage_bp)
        self.risk_vol_threshold = risk_vol_threshold
        self.position_smooth = float(position_smooth)
        self.max_trade_frac = float(max_trade_frac)
        self.obs_stack = int(obs_stack)
        self.normalize_features = normalize_features
        self.feature_stats = feature_stats or {}
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.n = len(self.df)
        self.cash = float(self.initial_cash)
        self.holdings = 0.0  # BTC units
        self.done = False
        self.target_pos_frac = 0.0
        self.history = []  # [time, price, cash, holdings, portfolio, action]
        self._obs_buffer = []
        # Prime buffer with initial obs
        obs = self._get_base_obs()
        for _ in range(self.obs_stack):
            self._obs_buffer.append(obs)
        return self._stack_obs()

    def _norm_feat(self, row: pd.Series, col: str) -> float:
        x = float(row[col])
        if self.normalize_features and col in self.feature_stats:
            mu, sd = self.feature_stats[col]
            sd = sd if sd > 1e-8 else 1.0
            x = (x - mu) / sd
        return x

    def _get_base_obs(self) -> np.ndarray:
        row = self.df.iloc[self.t]
        price = float(row["Close"])
        port = self.cash + self.holdings * price
        pos_frac = (self.holdings * price) / (port + 1e-8)
        cash_frac = self.cash / (port + 1e-8)
        feats = [self._norm_feat(row, c) for c in self.feature_cols]
        # Include recent log_ret, realized_vol raw (normalized via feature_stats if present)
        extra = [
            float(row["log_ret"]),
            float(row["realized_vol"]) if not np.isnan(row["realized_vol"]) else 0.0,
            pos_frac,
            cash_frac
        ]
        return np.array(feats + extra, dtype=np.float32)

    def _stack_obs(self) -> np.ndarray:
        # Stack last obs_stack base observations
        return np.concatenate(self._obs_buffer, axis=0)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        row = self.df.iloc[self.t]
        price = float(row["Close"])
        realized_vol = float(row["realized_vol"]) if not np.isnan(row["realized_vol"]) else 0.0

        # Risk control: if vol above threshold => liquidate, freeze buys
        a = float(np.clip(action, 0.0, 1.0))
        if self.risk_vol_threshold is not None and realized_vol > self.risk_vol_threshold:
            if self.holdings > 1e-12:
                sell_qty = self.holdings
                fee = self.fee_rate * price * sell_qty
                self.cash += price * sell_qty - fee
                self.holdings = 0.0
            a = 0.0  # target zero position

        # Action is target position fraction in [0,1]
        self.target_pos_frac = a
        port = self.cash + self.holdings * price
        current_pos_frac = (self.holdings * price) / (port + 1e-8)

        # Move towards target with smoothing and max trade limit
        gap = self.target_pos_frac - current_pos_frac
        desired_change = np.clip(self.position_smooth * gap, -1.0, 1.0)
        trade_value = float(np.clip(desired_change, -self.max_trade_frac, self.max_trade_frac)) * port

        # Execute trade: sell if trade_value < 0, buy if >0
        if trade_value < -1e-12 and self.holdings > 1e-12:
            sell_qty = min(self.holdings, abs(trade_value) / price)
            exec_price = price * (1 - self.slippage_bp / 1e4)
            fee = self.fee_rate * exec_price * sell_qty
            self.cash += exec_price * sell_qty - fee
            self.holdings -= sell_qty

        elif trade_value > 1e-12 and self.cash > 1e-8:
            buy_val = min(self.cash, trade_value)
            exec_price = price * (1 + self.slippage_bp / 1e4)
            buy_qty = buy_val / exec_price
            fee = self.fee_rate * exec_price * buy_qty
            total_cost = buy_val + fee
            if total_cost <= self.cash:
                self.cash -= total_cost
                self.holdings += buy_qty

        # Compute reward: portfolio return minus small penalty on turnover
        prev_port = self.history[-1][4] if self.history else self.initial_cash
        curr_port = self.cash + self.holdings * price
        step_ret = (curr_port - prev_port) / (prev_port + 1e-8)  # step portfolio return
        # Penalty proportional to traded fraction
        turnover_pen = abs(trade_value) / (prev_port + 1e-8)
        reward = float(step_ret - 0.001 * turnover_pen)

        self.history.append([self.df.index[self.t], price, self.cash, self.holdings, curr_port, a])
        self.t += 1
        if self.t >= self.n - 1:
            self.done = True

        # Prepare next observation
        if not self.done:
            base_obs = self._get_base_obs()
            self._obs_buffer.pop(0)
            self._obs_buffer.append(base_obs)
            next_obs = self._stack_obs()
        else:
            next_obs = None
        return next_obs, reward, self.done, {}

################################################################################
# DRL Agents: PPO, A2C, DDPG (MPS-safe; only Normal sampling)
################################################################################

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, act: nn.Module=nn.ReLU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), act,
            nn.Linear(hidden, hidden), act,
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# PPO Agent
class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int=256,
        lr: float=3e-4,
        gamma: float=0.995,
        gae_lambda: float=0.95,
        clip_eps: float=0.2,
        vf_coef: float=0.5,
        ent_coef: float=0.01,
        max_grad_norm: float=0.5,
        name: str="PPO"
    ):
        self.name = name
        self.actor_body = MLP(obs_dim, hidden_dim, hidden_dim).to(DEVICE)
        self.mu_head = nn.Linear(hidden_dim, 1).to(DEVICE)   # action in [0,1]
        self.log_std = nn.Parameter(torch.zeros(1, device=DEVICE))
        self.critic = MLP(obs_dim, hidden_dim, 1).to(DEVICE)
        self.opt_actor = optim.Adam(list(self.actor_body.parameters()) + [self.mu_head.weight, self.mu_head.bias, self.log_std], lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def _dist(self, obs_t: torch.Tensor):
        h = self.actor_body(obs_t)
        mu = torch.sigmoid(self.mu_head(h))           # map to [0,1]
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)
        return Normal(mu, std)

    def select_action(self, obs: np.ndarray, deterministic: bool=False) -> Tuple[float, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            dist = self._dist(obs_t)
            a = dist.mean if deterministic else dist.sample()
            a = torch.clamp(a, 0.0, 1.0)
            logp = dist.log_prob(a).sum(dim=-1)
            value = self.critic(obs_t).squeeze(-1)
        return float(a.squeeze(0).cpu().numpy()), float(logp.cpu().numpy()), float(value.cpu().numpy())

    def evaluate_actions(self, obs_batch: torch.Tensor, act_batch: torch.Tensor):
        dist = self._dist(obs_batch)
        logp = dist.log_prob(act_batch).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(obs_batch).squeeze(-1)
        return logp, entropy, values

    def compute_gae(self, rewards, values, dones, next_values) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_values[t] * nonterminal - values[t]
            last_adv = delta + self.gamma * self.gae_lambda * nonterminal * last_adv
            advantages[t] = last_adv
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def update(self, traj: Dict[str, Any], batch_size: int=256, epochs: int=5):
        obs = torch.tensor(np.vstack(traj["obs"]), dtype=torch.float32, device=DEVICE)
        acts = torch.tensor(np.array(traj["acts"]).reshape(-1,1), dtype=torch.float32, device=DEVICE)
        old_logp = torch.tensor(np.array(traj["logp"]), dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(np.array(traj["returns"]), dtype=torch.float32, device=DEVICE)
        adv = torch.tensor(np.array(traj["advantages"]), dtype=torch.float32, device=DEVICE)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idxs = np.arange(len(adv))
        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), batch_size):
                bi = idxs[start:start+batch_size]
                ob_b = obs[bi]; act_b = acts[bi]; old_logp_b = old_logp[bi]
                ret_b = returns[bi]; adv_b = adv[bi]

                new_logp, entropy, values = self.evaluate_actions(ob_b, act_b)
                ratio = torch.exp(new_logp - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy.mean()
                critic_loss = self.vf_coef * (ret_b - values).pow(2).mean()
                loss = actor_loss + critic_loss

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor_body.parameters()) + list(self.critic.parameters()) + [self.mu_head.weight, self.mu_head.bias], self.max_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()

    def train_on_env(self, env: CryptoTradingEnv, total_steps: int=4000, rollout_horizon: int=512, update_epochs: int=5, batch_size: int=256, desc: str="PPO Training"):
        obs = env.reset()
        pbar = tqdm(total=total_steps, desc=desc)
        steps = 0
        while steps < total_steps:
            traj = {"obs": [], "acts": [], "logp": [], "values": [], "rewards": [], "dones": []}
            for _ in range(rollout_horizon):
                a, logp, val = self.select_action(obs, deterministic=False)
                next_obs, reward, done, _ = env.step(a)
                traj["obs"].append(obs)
                traj["acts"].append(a)
                traj["logp"].append(logp)
                traj["values"].append(val)
                traj["rewards"].append(reward)
                traj["dones"].append(done)
                obs = next_obs if not done else env.reset()
                steps += 1
                pbar.update(1)
                if steps >= total_steps:
                    break
            next_values = traj["values"][1:] + [traj["values"][-1]]
            adv, ret = self.compute_gae(traj["rewards"], traj["values"], traj["dones"], next_values)
            traj["advantages"] = adv
            traj["returns"] = ret
            self.update(traj, batch_size=batch_size, epochs=update_epochs)
        pbar.close()

# A2C Agent
class A2CAgent:
    def __init__(self, obs_dim: int, hidden_dim: int=256, lr: float=3e-4, gamma: float=0.995, ent_coef: float=0.01, name: str="A2C"):
        self.name = name
        self.actor_body = MLP(obs_dim, hidden_dim, hidden_dim).to(DEVICE)
        self.mu_head = nn.Linear(hidden_dim, 1).to(DEVICE)
        self.log_std = nn.Parameter(torch.zeros(1, device=DEVICE))
        self.critic = MLP(obs_dim, hidden_dim, 1).to(DEVICE)
        self.opt = optim.Adam(list(self.actor_body.parameters()) + [self.mu_head.weight, self.mu_head.bias, self.log_std] + list(self.critic.parameters()), lr=lr)
        self.gamma = gamma
        self.ent_coef = ent_coef

    def _dist(self, obs_t: torch.Tensor):
        h = self.actor_body(obs_t)
        mu = torch.sigmoid(self.mu_head(h))
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)
        return Normal(mu, std)

    def select_action(self, obs: np.ndarray, deterministic: bool=False) -> Tuple[float, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            dist = self._dist(obs_t)
            a = dist.mean if deterministic else dist.sample()
            a = torch.clamp(a, 0.0, 1.0)
            logp = dist.log_prob(a).sum(dim=-1)
            value = self.critic(obs_t).squeeze(-1)
        return float(a.squeeze(0).cpu().numpy()), float(logp.cpu().numpy()), float(value.cpu().numpy())

    def train_on_env(self, env: CryptoTradingEnv, total_steps: int=4000, rollout_horizon: int=512, desc: str="A2C Training"):
        obs = env.reset()
        pbar = tqdm(total=total_steps, desc=desc)
        steps = 0
        while steps < total_steps:
            obs_list, act_list, logp_list, val_list, rew_list = [], [], [], [], []
            for _ in range(rollout_horizon):
                a, logp, v = self.select_action(obs, deterministic=False)
                next_obs, r, done, _ = env.step(a)
                obs_list.append(obs); act_list.append(a); logp_list.append(logp); val_list.append(v); rew_list.append(r)
                obs = next_obs if not done else env.reset()
                steps += 1; pbar.update(1)
                if steps >= total_steps:
                    break
            # Compute returns with bootstrap last value
            returns = []
            G = val_list[-1]
            for r in reversed(rew_list):
                G = r + self.gamma * G
                returns.append(G)
            returns = list(reversed(returns))
            # Policy and value update
            ob_t = torch.tensor(np.vstack(obs_list), dtype=torch.float32, device=DEVICE)
            act_t = torch.tensor(np.array(act_list).reshape(-1,1), dtype=torch.float32, device=DEVICE)
            ret_t = torch.tensor(np.array(returns), dtype=torch.float32, device=DEVICE)
            val_t = self.critic(ob_t).squeeze(-1)
            adv_t = ret_t - val_t
            dist = self._dist(ob_t)
            logp_new = dist.log_prob(act_t).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            actor_loss = -(logp_new * adv_t.detach()).mean() - self.ent_coef * entropy.mean()
            critic_loss = 0.5 * (adv_t.pow(2)).mean()
            loss = actor_loss + critic_loss
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor_body.parameters()) + list(self.critic.parameters()) + [self.mu_head.weight, self.mu_head.bias], 0.5)
            self.opt.step()
        pbar.close()

# DDPG Agent
class ReplayBuffer:
    def __init__(self, capacity: int=200000):
        self.capacity = capacity
        self.storage = []
        self.pos = 0
    def push(self, s, a, r, s2, d):
        data = (s, a, r, s2, d)
        if len(self.storage) < self.capacity:
            self.storage.append(data)
        else:
            self.storage[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.storage), batch_size, replace=False)
        s, a, r, s2, d = zip(*[self.storage[i] for i in idx])
        return np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d)
    def __len__(self):
        return len(self.storage)

class DDPGAgent:
    def __init__(self, obs_dim: int, hidden_dim: int=256, lr_actor: float=1e-3, lr_critic: float=1e-3, gamma: float=0.995, tau: float=0.01, noise_std: float=0.1, name: str="DDPG"):
        self.name = name
        self.actor = MLP(obs_dim, hidden_dim, 1).to(DEVICE)
        self.critic = MLP(obs_dim + 1, hidden_dim, 1).to(DEVICE)
        self.actor_target = MLP(obs_dim, hidden_dim, 1).to(DEVICE)
        self.critic_target = MLP(obs_dim + 1, hidden_dim, 1).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.buffer = ReplayBuffer(capacity=200000)

    def select_action(self, obs: np.ndarray, deterministic: bool=False) -> float:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            a = torch.sigmoid(self.actor(obs_t))  # [0,1]
        a = a.squeeze(0).cpu().numpy()
        if not deterministic:
            a += np.random.normal(0.0, self.noise_std, size=a.shape)
        return float(np.clip(a, 0.0, 1.0))

    def train_on_env(self, env: CryptoTradingEnv, total_steps: int=4000, warmup: int=1000, batch_size: int=256, desc: str="DDPG Training"):
        obs = env.reset()
        pbar = tqdm(total=total_steps, desc=desc)
        steps = 0
        while steps < total_steps:
            a = self.select_action(obs, deterministic=False)
            next_obs, r, d, _ = env.step(a)
            self.buffer.push(obs, a, r, next_obs if not d else np.zeros_like(obs), d)
            obs = next_obs if not d else env.reset()
            steps += 1; pbar.update(1)
            # Update after warmup
            if len(self.buffer) >= max(batch_size, warmup):
                s, a, r, s2, d_ = self.buffer.sample(batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE)
                a_t = torch.tensor(a.reshape(-1,1), dtype=torch.float32, device=DEVICE)
                r_t = torch.tensor(r, dtype=torch.float32, device=DEVICE)
                s2_t = torch.tensor(s2, dtype=torch.float32, device=DEVICE)
                d_t = torch.tensor(d_.astype(np.float32), dtype=torch.float32, device=DEVICE)

                # Critic target
                with torch.no_grad():
                    a2 = torch.sigmoid(self.actor_target(s2_t))
                    q2 = self.critic_target(torch.cat([s2_t, a2], dim=-1)).squeeze(-1)
                    y = r_t + self.gamma * (1.0 - d_t) * q2
                q = self.critic(torch.cat([s_t, a_t], dim=-1)).squeeze(-1)
                critic_loss = nn.MSELoss()(q, y)
                self.opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_critic.step()

                # Actor loss (maximize Q)
                a_pred = torch.sigmoid(self.actor(s_t))
                actor_loss = -self.critic(torch.cat([s_t, a_pred], dim=-1)).mean()
                self.opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_((self.actor.parameters()), 1.0)
                self.opt_actor.step()

                # Soft update targets
                with torch.no_grad():
                    for tgt, src in zip(self.actor_target.parameters(), self.actor.parameters()):
                        tgt.data.mul_(1.0 - self.tau).add_(self.tau * src.data)
                    for tgt, src in zip(self.critic_target.parameters(), self.critic.parameters()):
                        tgt.data.mul_(1.0 - self.tau).add_(self.tau * src.data)
        pbar.close()

################################################################################
# Metrics, Trades, and Plotting
################################################################################

def history_to_df(env_history: List[List[Any]]) -> pd.DataFrame:
    dfh = pd.DataFrame(env_history, columns=["time", "price", "cash", "holdings", "portfolio", "action"])
    dfh = dfh.set_index("time").sort_index()
    return dfh

def portfolio_timeseries(env_history: List[List[Any]]):
    times = [h[0] for h in env_history]
    prices = [h[1] for h in env_history]
    cash = [h[2] for h in env_history]
    holdings = [h[3] for h in env_history]
    port = [h[4] for h in env_history]
    actions = [h[5] for h in env_history]
    hold_val = [p*q for p, q in zip(prices, holdings)]
    return times, port, cash, hold_val, prices, actions, holdings

def cumulative_return_from_port(port: List[float]) -> float:
    if len(port) < 2: return 0.0
    return (port[-1] - port[0]) / (port[0] + 1e-8)

def volatility_from_history(env_history: List[List[Any]]) -> float:
    ports = np.array([h[4] for h in env_history])
    ret = np.diff(ports) / (ports[:-1] + 1e-8)
    if len(ret) == 0: return 0.0
    return float(np.std(ret))

def max_drawdown_from_port(port: List[float]) -> float:
    if len(port) == 0: return 0.0
    arr = np.array(port, dtype=np.float64)
    roll_max = np.maximum.accumulate(arr)
    drawdowns = (arr - roll_max) / (roll_max + 1e-12)
    return float(drawdowns.min())

def sharpe_ratio(env_history: List[List[Any]], risk_free: float=0.0) -> float:
    ports = np.array([h[4] for h in env_history], dtype=np.float64)
    ret = np.diff(ports) / (ports[:-1] + 1e-8)
    if len(ret) == 0 or ret.std() < 1e-12:
        return 0.0
    excess = ret - risk_free / 365.0
    return float(np.mean(excess) / (np.std(ret) + 1e-8) * np.sqrt(365.0))

def extract_trades(env_history: List[List[Any]]) -> Tuple[List[Tuple[Any,float,float,float]], List[Tuple[Any,float,float,float]]]:
    buys, sells = []
    if not env_history:
        return [], []
    prev_hold = env_history[0][3]
    for i in range(1, len(env_history)):
        t_i, p_i, c_i, h_i, v_i, a_i = env_history[i]
        delta = h_i - prev_hold
        if delta > 1e-10:
            buys.append((t_i, p_i, delta, a_i))
        elif delta < -1e-10:
            sells.append((t_i, p_i, -delta, a_i))
        prev_hold = h_i
    return buys, sells

def summarize_and_print(env_history: List[List[Any]], label: str="Backtest"):
    ts, port, cash, hold_val, _, _, _ = portfolio_timeseries(env_history)
    if len(ts) == 0:
        print(f"{label}: No data.")
        return
    init_val = port[0]; final_val = port[-1]
    total_ret = (final_val - init_val) / (init_val + 1e-8)
    vol = volatility_from_history(env_history)
    mdd = max_drawdown_from_port(port)
    sp = sharpe_ratio(env_history, risk_free=0.0)
    buys, sells = extract_trades(env_history)
    print(f"=== {label} ===")
    print(f"Initial: ${init_val:,.2f} | Final: ${final_val:,.2f} | Return: {total_ret*100:.2f}% | Sharpe: {sp:.3f}")
    print(f"Vol (std): {vol:.6f} | Max Drawdown: {mdd*100:.2f}% | Trades: {len(buys)+len(sells)}")
    print("")

def plot_portfolio_cash_holdings(env_history: List[List[Any]], title_prefix: str="Backtest"):
    times, port, cash, hold_val, _, _, _ = portfolio_timeseries(env_history)
    # Portfolio
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=times, y=port, mode='lines', name='Portfolio Value'))
    fig1.update_layout(title=f"{title_prefix}: Portfolio Value", xaxis_title="Time", yaxis_title="USD", legend=dict(orientation="h"))
    fig1.show()
    # Cash
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=cash, mode='lines', name='Cash'))
    fig2.update_layout(title=f"{title_prefix}: Cash", xaxis_title="Time", yaxis_title="USD", legend=dict(orientation="h"))
    fig2.show()
    # Holdings value
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=times, y=hold_val, mode='lines', name='Holdings Value'))
    fig3.update_layout(title=f"{title_prefix}: Holdings Value", xaxis_title="Time", yaxis_title="USD", legend=dict(orientation="h"))
    fig3.show()

def plot_cumulative_returns(env_history: List[List[Any]], title: str="Cumulative Returns"):
    times, port, _, _, _, _, _ = portfolio_timeseries(env_history)
    if len(port) == 0: return
    base = port[0] if port[0] != 0 else 1.0
    cum_ret_series = [(p / base) - 1.0 for p in port]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=cum_ret_series, mode='lines', name='Cumulative Return'))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Return", legend=dict(orientation="h"))
    fig.show()

def plot_price_with_signals(env_history: List[List[Any]], base_df: pd.DataFrame, title: str="Price + Trades + RSI/MACD"):
    dfh = history_to_df(env_history)
    df_ind = base_df.loc[dfh.index.intersection(base_df.index)].copy()
    df_plot = dfh.join(df_ind[["rsi", "macd", "macd_signal"]], how="left")
    df_plot = df_plot.dropna(subset=["price"])
    buys, sells = extract_trades(env_history)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03,
                        subplot_titles=("Price with Trades", "RSI(14)", "MACD"))
    # price
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["price"], mode='lines', name='Price'), row=1, col=1)
    if len(buys) > 0:
        b_t = [b[0] for b in buys]; b_p = [b[1] for b in buys]
        fig.add_trace(go.Scatter(x=b_t, y=b_p, mode='markers', name='Buy',
                                 marker=dict(symbol='triangle-up', color='green', size=9)), row=1, col=1)
    if len(sells) > 0:
        s_t = [s[0] for s in sells]; s_p = [s[1] for s in sells]
        fig.add_trace(go.Scatter(x=s_t, y=s_p, mode='markers', name='Sell',
                                 marker=dict(symbol='triangle-down', color='red', size=9)), row=1, col=1)
    # RSI
    if "rsi" in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["rsi"], mode='lines', name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="orange", width=1, dash="dash"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="orange", width=1, dash="dash"), row=2, col=1)
    # MACD
    if "macd" in df_plot.columns and "macd_signal" in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["macd"], mode='lines', name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["macd_signal"], mode='lines', name='Signal'), row=3, col=1)
    fig.update_layout(title=title, legend=dict(orientation="h"))
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.show()

################################################################################
# Backtests: Walk-Forward, K-Fold, Hyperparameter Tuning + Combinatorial CV
################################################################################

def time_series_split(df: pd.DataFrame, train_ratio: float=0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(n * train_ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def evaluate_agent_on_df(agent, df_std: pd.DataFrame, feature_cols: List[str], stats: Dict, env_params: Dict) -> List[List[Any]]:
    env = CryptoTradingEnv(df_std, feature_cols, normalize_features=True, feature_stats=stats, **env_params)
    obs = env.reset()
    while True:
        a = agent.select_action(obs, deterministic=True) if isinstance(agent, DDPGAgent) else agent.select_action(obs, deterministic=True)[0]
        obs, r, done, _ = env.step(a)
        if done:
            break
    return env.history

def ensemble_select_and_test(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], env_params: Dict, agent_params: Dict) -> Tuple[List[List[Any]], str]:
    # Standardize by train stats
    train_std, stats = standardize_features(train_df.copy(), feature_cols, ref_df=train_df)
    val_std, _ = standardize_features(val_df.copy(), feature_cols, ref_df=train_df)
    test_std, _ = standardize_features(test_df.copy(), feature_cols, ref_df=train_df)

    obs_dim = len(feature_cols) + 4  # features + [log_ret, realized_vol, pos_frac, cash_frac], stacked later
    obs_dim *= env_params.get("obs_stack", 8)

    # Initialize agents
    ppo = PPOAgent(obs_dim=obs_dim, **agent_params.get("PPO", {}))
    a2c = A2CAgent(obs_dim=obs_dim, **agent_params.get("A2C", {}))
    ddpg = DDPGAgent(obs_dim=obs_dim, **agent_params.get("DDPG", {}))
    agents = [ppo, a2c, ddpg]

    # Train all on train set
    env_train = CryptoTradingEnv(train_std, feature_cols, normalize_features=True, feature_stats=stats, **env_params)
    for ag in agents:
        if isinstance(ag, PPOAgent):
            ag.train_on_env(env_train, **agent_params.get("PPO_train", {}), desc=f"{ag.name} Train")
        elif isinstance(ag, A2CAgent):
            ag.train_on_env(env_train, **agent_params.get("A2C_train", {}), desc=f"{ag.name} Train")
        else:
            ag.train_on_env(env_train, **agent_params.get("DDPG_train", {}), desc=f"{ag.name} Train")

    # Validate
    sharpe_scores = []
    for ag in agents:
        hist = evaluate_agent_on_df(ag, val_std, feature_cols, stats, env_params)
        sharpe_scores.append((ag.name, sharpe_ratio(hist)))

    # Pick best by Sharpe
    best_name, _ = max(sharpe_scores, key=lambda x: x[1])
    best_agent = {ag.name: ag for ag in agents}[best_name]

    # Test
    hist_test = evaluate_agent_on_df(best_agent, test_std, feature_cols, stats, env_params)
    return hist_test, best_name

def walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: List[str],
    wf_train_len: int=3000,
    wf_val_len: int=500,
    wf_test_len: int=500,
    env_params: Dict=None,
    agent_params: Dict=None
) -> Tuple[List[List[Any]], List[str]]:
    env_params = env_params or {}
    agent_params = agent_params or {}
    starts = list(range(0, len(df) - (wf_train_len + wf_val_len + wf_test_len), wf_test_len))
    combined_history = []
    chosen_agents = []
    for s in tqdm(starts, desc="Walk-Forward"):
        train_df = df.iloc[s : s + wf_train_len]
        val_df = df.iloc[s + wf_train_len : s + wf_train_len + wf_val_len]
        test_df = df.iloc[s + wf_train_len + wf_val_len : s + wf_train_len + wf_val_len + wf_test_len]
        hist, name = ensemble_select_and_test(train_df, val_df, test_df, feature_cols, env_params, agent_params)
        combined_history.extend(hist)
        chosen_agents.append(name)
    return combined_history, chosen_agents

def kfold_backtest(
    df: pd.DataFrame,
    feature_cols: List[str],
    k: int=5,
    env_params: Dict=None,
    agent_params: Dict=None
) -> Tuple[List[List[Any]], List[str]]:
    env_params = env_params or {}
    agent_params = agent_params or {}
    n = len(df)
    fold_size = n // k
    hist_all = []
    chosen_agents = []
    for i in tqdm(range(k), desc="K-Fold"):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k-1 else n
        test_df = df.iloc[test_start:test_end]
        # Train on prior data, with a validation tail
        if test_start < fold_size:
            train_df = df.iloc[:fold_size]
        else:
            train_df = df.iloc[:test_start - fold_size]  # leave last fold_size for validation
        val_df = df.iloc[test_start - fold_size:test_start] if test_start >= fold_size else df.iloc[:fold_size]
        hist, name = ensemble_select_and_test(train_df, val_df, test_df, feature_cols, env_params, agent_params)
        hist_all.extend(hist)
        chosen_agents.append(name)
    return hist_all, chosen_agents

# Combinatorial CV splits (Bailey et al.)
def combinatorial_splits(df: pd.DataFrame, N: int=5, k: int=2) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    n = len(df)
    group_size = n // N
    groups = []
    for i in range(N):
        start = i * group_size
        end = (i + 1) * group_size if i < N - 1 else n
        groups.append(df.iloc[start:end])
    indices = list(range(N))
    splits = []
    for val_idx in itertools.combinations(indices, k):
        train_idx = [j for j in indices if j not in val_idx]
        train_df = pd.concat([groups[j] for j in train_idx], axis=0)
        val_df = pd.concat([groups[j] for j in val_idx], axis=0)
        splits.append((train_df, val_df))
    return splits

def estimate_overfitting_probability(M: np.ndarray, num_slices: int=4) -> Tuple[float, np.ndarray]:
    # M shape [J_splits, H_trials], compute lambdas via IS/OOS relative rank
    J, H = M.shape
    slice_len = J // num_slices if num_slices > 0 else J
    subsets = []
    st = 0
    for s in range(num_slices - 1):
        subsets.append(M[st:st + slice_len, :]); st += slice_len
    subsets.append(M[st:, :])

    idx_list = list(range(len(subsets)))
    lambdas = []
    # Use half subsets as IS, rest OOS combinations
    r_size = max(1, len(subsets)//2)
    for IS_idx in itertools.combinations(idx_list, r=r_size):
        OOS_idx = [i for i in idx_list if i not in IS_idx]
        if len(OOS_idx) == 0: continue
        IS_mat = np.vstack([subsets[i] for i in IS_idx])
        OOS_mat = np.vstack([subsets[i] for i in OOS_idx])
        IS_perf = IS_mat.mean(axis=0)
        OOS_perf = OOS_mat.mean(axis=0)
        best_trial = int(np.argmax(IS_perf))
        oos_ranks = np.argsort(OOS_perf)  # low->high
        pos = int(np.where(oos_ranks == best_trial)[0][0]) + 1
        omega = pos / (H + 1.0)
        omega = max(min(omega, 1 - 1e-6), 1e-6)
        lam = math.log(omega / (1.0 - omega))
        lambdas.append(lam)
    lambdas = np.array(lambdas) if len(lambdas) > 0 else np.array([0.0])
    p_hat = float(np.mean(lambdas < 0.0))
    return p_hat, lambdas

def hyperparameter_tuning_backtest(
    df: pd.DataFrame,
    feature_cols: List[str],
    env_params: Dict=None,
    H: int=12,            # number of trials
    N: int=5, k: int=2    # combinatorial CV
) -> Tuple[List[List[Any]], float, float, Dict[str, Any], float, np.ndarray, pd.DataFrame]:
    env_params = env_params or {}
    # Grid for PPO
    lr_grid = [3e-4, 5e-4, 1e-4]
    gamma_grid = [0.99, 0.995]
    hidden_grid = [256, 128]
    clip_grid = [0.2, 0.1]
    ent_grid = [0.01, 0.00]
    combos = list(itertools.product(lr_grid, gamma_grid, hidden_grid, clip_grid, ent_grid))
    random.shuffle(combos)
    combos = combos[:H]

    splits = combinatorial_splits(df, N=N, k=k)
    J = len(splits)
    M = np.zeros((J, len(combos)), dtype=np.float32)
    trial_results = []

    for t_idx, (lr, gamma, hidden_dim, clip_eps, ent_coef) in enumerate(tqdm(combos, desc="Hyperparam Trials")):
        val_returns = []
        for j_idx, (train_df, val_df) in enumerate(tqdm(splits, desc="CombCV", leave=False)):
            train_std, stats = standardize_features(train_df.copy(), feature_cols, ref_df=train_df)
            val_std, _ = standardize_features(val_df.copy(), feature_cols, ref_df=train_df)
            obs_dim = (len(feature_cols) + 4) * env_params.get("obs_stack", 8)
            agent = PPOAgent(obs_dim=obs_dim, hidden_dim=hidden_dim, lr=lr, gamma=gamma, clip_eps=clip_eps, ent_coef=ent_coef)
            env_train = CryptoTradingEnv(train_std, feature_cols, normalize_features=True, feature_stats=stats, **env_params)
            agent.train_on_env(env_train, total_steps=2500, rollout_horizon=256, update_epochs=3, batch_size=256, desc=f"Tune Split {j_idx+1}/{J}")
            hist_val = evaluate_agent_on_df(agent, val_std, feature_cols, stats, env_params)
            ret = cumulative_return_from_port([h[4] for h in hist_val])
            M[j_idx, t_idx] = ret
            val_returns.append(ret)
        trial_results.append({
            "params": {"lr": lr, "gamma": gamma, "hidden_dim": hidden_dim, "clip_eps": clip_eps, "ent_coef": ent_coef},
            "avg_val_return": float(np.mean(val_returns)),
            "std_val_return": float(np.std(val_returns))
        })

    p_hat, lambdas = estimate_overfitting_probability(M, num_slices=4)
    best_idx, best_trial = max(enumerate(trial_results), key=lambda x: x[1]["avg_val_return"])
    best_params = best_trial["params"]

    # Retrain best on 70% train and test on remainder
    train_df, test_df = time_series_split(df, train_ratio=0.7)
    train_std, stats = standardize_features(train_df.copy(), feature_cols, ref_df=train_df)
    test_std, _ = standardize_features(test_df.copy(), feature_cols, ref_df=train_df)
    obs_dim = (len(feature_cols) + 4) * env_params.get("obs_stack", 8)
    best_agent = PPOAgent(obs_dim=obs_dim, **best_params)
    env_train = CryptoTradingEnv(train_std, feature_cols, normalize_features=True, feature_stats=stats, **env_params)
    best_agent.train_on_env(env_train, total_steps=5000, rollout_horizon=256, update_epochs=5, batch_size=256, desc="Retrain Best PPO")
    hist_test = evaluate_agent_on_df(best_agent, test_std, feature_cols, stats, env_params)
    ret = cumulative_return_from_port([h[4] for h in hist_test])
    vol = volatility_from_history(hist_test)
    return hist_test, ret, vol, best_params, p_hat, lambdas, test_df

################################################################################
# Main
################################################################################

def main():
    # Load and prepare data
    df_raw = load_data()
    df = compute_indicators(df_raw)

    # Feature selection
    base_features = [
        "Close", "High", "Low", "Open", "Volume",
        "rsi", "ultosc", "williams_r", "roc", "macd", "macd_diff", "macd_signal",
        "cci", "adx", "obv", "atr"
    ]
    feature_cols = [c for c in base_features if c in df.columns]
    feature_cols = drop_correlated_features(df, feature_cols, threshold=0.6)

    # Risk control threshold from training slice
    df_train_tmp, df_test_tmp = time_series_split(df, train_ratio=0.7)
    vol_thresh = float(df_train_tmp["realized_vol"].quantile(0.95))

    # Environment and agent parameters
    env_params = {
        "initial_cash": 100000.0,
        "fee_rate": 0.001,           # 0.1% fee to curb churn impact
        "slippage_bp": 0.0,
        "risk_vol_threshold": vol_thresh,
        "position_smooth": 0.25,     # smoother moves to reduce whipsaw
        "max_trade_frac": 0.35,      # cap trade per step
        "obs_stack": 8,
    }
    agent_params = {
        "PPO": {"hidden_dim": 256, "lr": 3e-4, "gamma": 0.995, "clip_eps": 0.2, "ent_coef": 0.01},
        "A2C": {"hidden_dim": 256, "lr": 3e-4, "gamma": 0.995, "ent_coef": 0.01},
        "DDPG": {"hidden_dim": 256, "lr_actor": 1e-3, "lr_critic": 1e-3, "gamma": 0.995, "tau": 0.01, "noise_std": 0.05},
        "PPO_train": {"total_steps": 4000, "rollout_horizon": 512, "update_epochs": 5, "batch_size": 256},
        "A2C_train": {"total_steps": 4000, "rollout_horizon": 512},
        "DDPG_train": {"total_steps": 4000, "warmup": 800, "batch_size": 256},
    }

    # Walk-Forward Backtest (train/val/test sliding windows)
    wf_history, wf_agents = walk_forward_backtest(
        df, feature_cols=feature_cols,
        wf_train_len=3000, wf_val_len=500, wf_test_len=500,
        env_params=env_params, agent_params=agent_params
    )
    summarize_and_print(wf_history, label=f"Walk-Forward (agents: {dict((x, wf_agents.count(x)) for x in set(wf_agents))})")
    plot_portfolio_cash_holdings(wf_history, title_prefix="Walk-Forward")
    plot_cumulative_returns(wf_history, title="Walk-Forward: Cumulative Returns")
    plot_price_with_signals(wf_history, df, title="Walk-Forward: Price + Trades + RSI/MACD")

    # K-Fold Backtest (time-series folds; ensemble per fold)
    kf_history, kf_agents = kfold_backtest(
        df, feature_cols=feature_cols, k=5,
        env_params=env_params, agent_params=agent_params
    )
    summarize_and_print(kf_history, label=f"K-Fold (agents: {dict((x, kf_agents.count(x)) for x in set(kf_agents))})")
    plot_portfolio_cash_holdings(kf_history, title_prefix="K-Fold")
    plot_cumulative_returns(kf_history, title="K-Fold: Cumulative Returns")
    plot_price_with_signals(kf_history, df, title="K-Fold: Price + Trades + RSI/MACD")

    # Hyperparameter Tuning Backtest (PPO with combinatorial CV + p(overfitting))
    tune_history, tune_ret, tune_vol, best_params, p_hat, lambdas, test_df_used = hyperparameter_tuning_backtest(
        df, feature_cols=feature_cols, env_params=env_params, H=10, N=5, k=2
    )
    summarize_and_print(tune_history, label="Hyperparameter Tuning (Test Split)")
    print(f"Best PPO params: {best_params}")
    print(f"Estimated probability of overfitting p_hat: {p_hat*100:.2f}% | Lambdas count: {len(lambdas)}\n")
    plot_portfolio_cash_holdings(tune_history, title_prefix="Tuning")
    plot_cumulative_returns(tune_history, title="Tuning: Cumulative Returns")
    plot_price_with_signals(tune_history, df, title="Tuning: Price + Trades + RSI/MACD")

if __name__ == "__main__":
    main()
