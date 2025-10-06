import os
import math
import time
import random
import statistics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import ta
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# Device Configuration
# =============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

# =============================================================================
# Reproducibility
# =============================================================================
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

# =============================================================================
# Custom Trading Environment with profit-friendly risk and churn controls
# =============================================================================

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df,
        initial_balance=10000,
        lookback_window=50,
        is_training=True,
        episode_duration_minutes=4320,
        transaction_fee=0.001,
        action_threshold=0.15,          # larger hysteresis: act only if allocation error > 15%
        action_interpretation="target", # 'delta' or 'target'
        max_trade_frac_per_step=0.10,   # rebalance cap per step to 10% PV
        cooldown_steps=15,              # force fewer trades
        allow_short=False,
        min_usd_trade=100.0,            # minimum trade size (USD)
        # Risk cap from realized volatility
        use_risk_cap=True,
        risk_cap_base=0.10,             # numerator for risk cap = risk_cap_base / daily_vol
        risk_cap_min=0.05,
        risk_cap_max=0.60,
        # Action smoothing to damp oscillations
        action_smoothing_alpha=0.30,
        # Reward shaping
        gain_scale=1.0,                 # positive log-return scale
        loss_scale=2.0,                 # stronger penalty on negative returns
        turnover_penalty_coeff=0.10,    # penalize turnover meaningfully
        trade_exec_penalty=0.0005,      # per-trade nuisance penalty
        trade_pnl_coeff=0.0,            # REMOVE realized-PnL bonus (it encourages churning)
        terminal_profit_scale=2.0,      # terminal bonus proportional to final PnL
        early_stop_drawdown=0.20,       # stop if DD > 20%
        early_stop_penalty=3.0          # penalty for hitting early-stop
    ):
        super(TradingEnv, self).__init__()
        self.full_df = df.dropna().reset_index(drop=True)
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.is_training = is_training
        self.episode_duration = episode_duration_minutes
        self.df = pd.DataFrame()

        self.transaction_fee = float(transaction_fee)
        self.action_threshold = float(action_threshold)
        self.action_interpretation = action_interpretation
        self.max_trade_frac_per_step = float(max_trade_frac_per_step)
        self.cooldown_steps = int(cooldown_steps)
        self.allow_short = bool(allow_short)
        self.min_usd_trade = float(min_usd_trade)

        # Risk cap
        self.use_risk_cap = bool(use_risk_cap)
        self.risk_cap_base = float(risk_cap_base)
        self.risk_cap_min = float(risk_cap_min)
        self.risk_cap_max = float(risk_cap_max)

        # Action smoothing
        self.action_smoothing_alpha = float(action_smoothing_alpha)

        # Reward shaping params
        self.gain_scale = float(gain_scale)
        self.loss_scale = float(loss_scale)
        self.turnover_penalty_coeff = float(turnover_penalty_coeff)
        self.trade_exec_penalty = float(trade_exec_penalty)
        self.trade_pnl_coeff = float(trade_pnl_coeff)
        self.terminal_profit_scale = float(terminal_profit_scale)
        self.early_stop_drawdown = float(early_stop_drawdown)
        self.early_stop_penalty = float(early_stop_penalty)

        # Action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation: all features except 'Original_Close' + 3 portfolio features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.full_df.columns) - 1 + 3,),
            dtype=np.float32
        )

        self.reset_trackers()

    def reset_trackers(self):
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.portfolio_value = self.initial_balance
        self.current_step = self.lookback_window

        self.buy_trades = 0
        self.sell_trades = 0
        self.num_trades = 0
        self.turnover = 0.0
        self.last_trade_step = -10**9

        self.max_pv = self.portfolio_value
        self.min_pv = self.portfolio_value
        self.peak_pv = self.portfolio_value
        self.max_drawdown = 0.0

        self.step_returns = []
        self.time_in_market_steps = 0
        self.time_above_initial_steps = 0

        # Trade win/loss via realized PnL
        self.cost_basis = 0.0
        self.realized_pnl = 0.0
        self.winning_trades = 0
        self.losing_trades = 0

        # Per-step win/loss
        self.winning_steps = 0
        self.losing_steps = 0

        # Action smoothing memory
        self.prev_action = 0.0

    def _get_observation(self):
        obs_df = self.df.drop(columns=['Original_Close'])
        market_obs = obs_df.loc[self.current_step].values.astype(np.float32)

        current_price = float(self.df.loc[self.current_step, 'Original_Close'])
        crypto_value = self.crypto_held * current_price
        pv_safe = max(1e-8, self.balance + crypto_value)

        portfolio_state = np.array([
            self.balance / pv_safe,         # cash fraction
            crypto_value / pv_safe,         # position fraction
            pv_safe / self.initial_balance  # normalized PV
        ], dtype=np.float32)

        return np.concatenate((market_obs, portfolio_state))

    def reset(self, seed=None):
        super().reset(seed=seed)
        if self.is_training:
            max_start_index = len(self.full_df) - self.episode_duration - self.lookback_window - 1
            start_index = np.random.randint(0, max_start_index)
            end_index = start_index + self.episode_duration
            self.df = self.full_df.iloc[start_index:end_index].reset_index(drop=True)
        else:
            self.df = self.full_df.copy()

        self.reset_trackers()
        return self._get_observation(), {}

    def _update_cost_basis_on_buy(self, buy_qty, price, fee_value):
        if buy_qty <= 0:
            return
        total_cost_value = self.cost_basis * self.crypto_held + price * buy_qty + fee_value
        new_qty = self.crypto_held + buy_qty
        self.cost_basis = total_cost_value / max(1e-8, new_qty)

    def _realize_pnl_on_sell(self, sell_qty, price, fee_value):
        if sell_qty <= 0:
            return 0.0
        gross_pnl = (price - self.cost_basis) * sell_qty
        pnl_after_fee = gross_pnl - fee_value
        self.realized_pnl += pnl_after_fee
        if pnl_after_fee > 0:
            self.winning_trades += 1
        elif pnl_after_fee < 0:
            self.losing_trades += 1
        return pnl_after_fee

    def _trade_allowed(self):
        if self.cooldown_steps <= 0:
            return True
        return (self.current_step - self.last_trade_step) >= self.cooldown_steps

    def _compute_risk_cap(self):
        if not self.use_risk_cap:
            return 1.0
        vol_raw = float(self.df.loc[self.current_step, 'vol_60m_raw']) if 'vol_60m_raw' in self.df.columns else None
        if vol_raw is None or vol_raw <= 0:
            return self.risk_cap_max
        daily_vol = vol_raw * math.sqrt(1440.0)
        cap = self.risk_cap_base / max(1e-8, daily_vol)
        return float(np.clip(cap, self.risk_cap_min, self.risk_cap_max))

    def step(self, action):
        # Smooth action to reduce jitter
        raw_action = float(action[0] if isinstance(action, (np.ndarray, list, tuple)) else action)
        action_val = self.prev_action * (1.0 - self.action_smoothing_alpha) + raw_action * self.action_smoothing_alpha
        self.prev_action = action_val

        current_price = float(self.df.loc[self.current_step, 'Original_Close'])
        prev_pv = max(1e-8, self.portfolio_value)
        trade_executed = False
        traded_value_abs = 0.0
        realized_pnl_step = 0.0

        # Interpret action
        if self.action_interpretation == "delta":
            # Discouraged; kept for compatibility if user flips mode
            if action_val > self.action_threshold and self._trade_allowed():
                trade_amount_usd = self.balance * action_val
                if trade_amount_usd >= self.min_usd_trade:
                    fee = trade_amount_usd * self.transaction_fee
                    buy_qty = trade_amount_usd / current_price
                    if (trade_amount_usd + fee) <= self.balance + 1e-8:
                        self.balance -= (trade_amount_usd + fee)
                        self.crypto_held += buy_qty
                        self._update_cost_basis_on_buy(buy_qty, current_price, fee)
                        self.buy_trades += 1
                        self.num_trades += 1
                        traded_value_abs = trade_amount_usd
                        self.last_trade_step = self.current_step
                        trade_executed = True
            elif action_val < -self.action_threshold and self._trade_allowed():
                trade_amount_crypto = self.crypto_held * min(1.0, abs(action_val))
                trade_amount_usd = trade_amount_crypto * current_price
                if trade_amount_crypto > 1e-6 and trade_amount_usd >= self.min_usd_trade:
                    fee = trade_amount_usd * self.transaction_fee
                    self.balance += (trade_amount_usd - fee)
                    self.crypto_held -= trade_amount_crypto
                    realized_pnl_step = self._realize_pnl_on_sell(trade_amount_crypto, current_price, fee)
                    self.sell_trades += 1
                    self.num_trades += 1
                    traded_value_abs = trade_amount_usd
                    self.last_trade_step = self.current_step
                    trade_executed = True

        elif self.action_interpretation == "target":
            # action is desired allocation; trade if allocation error > threshold, respecting risk cap
            target_alloc = float(np.clip(action_val, -1.0, 1.0))
            if not self.allow_short:
                target_alloc = max(0.0, target_alloc)
            # Dynamic risk cap based on realized volatility
            max_alloc = self._compute_risk_cap()
            target_alloc = float(np.clip(target_alloc, -max_alloc if self.allow_short else 0.0, max_alloc))

            crypto_value_now = self.crypto_held * current_price
            pv_now = max(1e-8, self.balance + crypto_value_now)
            current_alloc = crypto_value_now / pv_now
            alloc_error = target_alloc - current_alloc

            if abs(alloc_error) > self.action_threshold and self._trade_allowed():
                desired_crypto_value = target_alloc * pv_now
                delta_value = desired_crypto_value - crypto_value_now
                max_step_trade_value = self.max_trade_frac_per_step * pv_now
                trade_value = float(np.clip(delta_value, -max_step_trade_value, max_step_trade_value))

                if abs(trade_value) >= self.min_usd_trade:
                    if trade_value > 0:
                        trade_amount_usd = min(self.balance, trade_value)
                        if trade_amount_usd >= self.min_usd_trade:
                            fee = trade_amount_usd * self.transaction_fee
                            if (trade_amount_usd + fee) <= self.balance + 1e-8:
                                buy_qty = trade_amount_usd / current_price
                                self.balance -= (trade_amount_usd + fee)
                                self.crypto_held += buy_qty
                                self._update_cost_basis_on_buy(buy_qty, current_price, fee)
                                self.buy_trades += 1
                                self.num_trades += 1
                                traded_value_abs = trade_amount_usd
                                self.last_trade_step = self.current_step
                                trade_executed = True
                    else:
                        trade_amount_usd = min(crypto_value_now, abs(trade_value))
                        sell_qty = trade_amount_usd / current_price
                        if sell_qty > 1e-6 and trade_amount_usd >= self.min_usd_trade:
                            fee = trade_amount_usd * self.transaction_fee
                            self.balance += (trade_amount_usd - fee)
                            self.crypto_held -= sell_qty
                            realized_pnl_step = self._realize_pnl_on_sell(sell_qty, current_price, fee)
                            self.sell_trades += 1
                            self.num_trades += 1
                            traded_value_abs = trade_amount_usd
                            self.last_trade_step = self.current_step
                            trade_executed = True

        # Advance time
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1

        next_price = float(self.df.loc[self.current_step, 'Original_Close'])
        self.portfolio_value = float(self.balance + (self.crypto_held * next_price))

        # Metrics updates
        self.max_pv = max(self.max_pv, self.portfolio_value)
        self.min_pv = min(self.min_pv, self.portfolio_value)
        if self.portfolio_value > self.peak_pv:
            self.peak_pv = self.portfolio_value
        drawdown = 0.0 if self.peak_pv <= 0 else (self.peak_pv - self.portfolio_value) / self.peak_pv
        self.max_drawdown = max(self.max_drawdown, drawdown)

        step_return = math.log(max(1e-8, self.portfolio_value) / max(1e-8, prev_pv))
        self.step_returns.append(step_return)
        if step_return > 0:
            self.winning_steps += 1
        elif step_return < 0:
            self.losing_steps += 1

        if self.portfolio_value > self.initial_balance:
            self.time_above_initial_steps += 1

        if self.portfolio_value > 0 and traded_value_abs > 0:
            self.turnover += traded_value_abs / self.portfolio_value

        if (self.crypto_held * next_price) / max(1e-8, self.portfolio_value) > 0.01:
            self.time_in_market_steps += 1

        # Reward shaping: emphasize net returns, discourage churn
        reward = self.gain_scale * step_return if step_return >= 0 else -self.loss_scale * abs(step_return)

        if trade_executed:
            reward -= self.trade_exec_penalty
            if self.trade_pnl_coeff != 0.0 and realized_pnl_step != 0.0:
                reward += self.trade_pnl_coeff * (realized_pnl_step / prev_pv)

        if traded_value_abs > 0:
            reward -= self.turnover_penalty_coeff * (traded_value_abs / prev_pv)

        # Early stop on large drawdown
        if drawdown >= self.early_stop_drawdown and not terminated:
            terminated = True
            reward -= self.early_stop_penalty

        # Terminal profit bonus
        if terminated:
            reward += self.terminal_profit_scale * ((self.portfolio_value / self.initial_balance) - 1.0)

        obs = self._get_observation()
        return obs, reward, terminated, False, {}

    def get_metrics(self):
        sharpe = 0.0
        if len(self.step_returns) > 1:
            mean_r = statistics.fmean(self.step_returns)
            std_r = statistics.pstdev(self.step_returns)
            if std_r > 0:
                sharpe = (mean_r / std_r) * math.sqrt(1440.0)  # minute->day scaling

        trade_win_rate = 0.0
        decided = self.winning_trades + self.losing_trades
        if decided > 0:
            trade_win_rate = 100.0 * self.winning_trades / decided

        steps_this_episode = max(1, len(self.step_returns))
        time_in_market = self.time_in_market_steps / steps_this_episode
        step_win_rate = 100.0 * self.winning_steps / steps_this_episode

        return {
            'final_pv': self.portfolio_value,
            'max_pv': self.max_pv,
            'min_pv': self.min_pv,
            'max_drawdown': self.max_drawdown,
            'sharpe_per_day': sharpe,
            'turnover_pct': 100.0 * self.turnover,
            'trades_total': self.num_trades,
            'trades_buy': self.buy_trades,
            'trades_sell': self.sell_trades,
            'realized_pnl': self.realized_pnl,
            'trade_win_rate_pct': trade_win_rate,
            'time_in_market_pct': 100.0 * time_in_market,
            'steps': steps_this_episode,
            'step_win_rate_pct': step_win_rate,
            'time_above_initial_pct': 100.0 * (self.time_above_initial_steps / steps_this_episode)
        }

# =============================================================================
# Data loading and feature engineering (train-only normalization)
# =============================================================================

def load_raw_data(url):
    print("Loading and preparing data...")
    df = pd.read_csv(url, skiprows=3, header=None)
    df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)

    # Coerce numerics
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Preserve original close for pricing
    df['Original_Close'] = df['Close'].astype(float)

    # Core features
    df['log_close'] = np.log(df['Original_Close'])
    df['ret_1m'] = df['log_close'].diff(1)

    for w in [5, 15, 60, 240]:
        df[f'ret_{w}m'] = df['log_close'].diff(w)
        df[f'vol_{w}m_raw'] = df['ret_1m'].rolling(w).std()

    # RSI and MACD on original price
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Original_Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Original_Close'])
    df['MACD'] = macd.macd_diff()

    # EMA momentum
    df['ema_12'] = df['Original_Close'].ewm(span=12, adjust=False).mean()
    df['ema_48'] = df['Original_Close'].ewm(span=48, adjust=False).mean()
    df['ema_ratio'] = df['ema_12'] / (df['ema_48'] + 1e-8) - 1.0

    # ATR percentage
    # THIS IS THE CORRECTED LINE:
    atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Original_Close'], window=14).average_true_range()
    df['atr_raw'] = atr
    df['atr_pct_raw'] = df['atr_raw'] / (df['Original_Close'] + 1e-8)

    # Volume zscore (rolling)
    vol_mean = df['Volume'].rolling(240).mean()
    vol_std = df['Volume'].rolling(240).std()
    df['vol_z_240'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)

    # Drop early NA from indicators
    df.dropna(inplace=True)
    print("Data preparation complete.")
    return df.reset_index(drop=True)

def fit_train_normalization(train_df, exclude_cols):
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1e-8)
    return feature_cols, mean, std

def apply_normalization(df, feature_cols, mean, std):
    df_norm = df.copy()
    df_norm[feature_cols] = (df_norm[feature_cols] - mean) / std
    return df_norm

# =============================================================================
# Actor/Critic Networks (TD3)
# =============================================================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 384)
        self.l2 = nn.Linear(384, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 384)
        self.l2 = nn.Linear(384, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 384)
        self.l5 = nn.Linear(384, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)
        q1 = F.relu(self.l1(xu))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(xu))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, x, u):
        xu = torch.cat([x, u], dim=1)
        q1 = F.relu(self.l1(xu))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = zip(*[self.storage[i] for i in ind])
        return (
            np.array(x),
            np.array(y),
            np.array(u),
            np.array(r).reshape(-1, 1),
            np.array(d).reshape(-1, 1)
        )

class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.08,
        noise_clip=0.20,
        policy_delay=2,
        grad_clip_norm=1.0
    ):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.grad_clip_norm = grad_clip_norm

        self.total_it = 0
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, batch_size):
        if len(self.replay_buffer.storage) < batch_size:
            return None

        self.total_it += 1
        x, y, u, r, d = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(x).to(self.device)
        action = torch.FloatTensor(u).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)
        not_done = torch.FloatTensor(1.0 - d).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        actor_loss_val = None
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()
            actor_loss_val = float(actor_loss.detach().cpu().item())

            # Soft updates
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': float(critic_loss.detach().cpu().item()),
            'actor_loss': actor_loss_val
        }

# =============================================================================
# Plotting and Evaluation
# =============================================================================

def plot_validation_results(history, episode_number):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=('Price and Executed Trades', 'Agent Actions', 'Portfolio Value ($)', 'Portfolio Composition ($)')
    )
    fig.add_trace(go.Scatter(x=history['steps'], y=history['price'], mode='lines', name='BTC Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=history['buy_steps'], y=history['buy_prices'], mode='markers', name='Buy', marker=dict(color='green', symbol='triangle-up', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=history['sell_steps'], y=history['sell_prices'], mode='markers', name='Sell', marker=dict(color='red', symbol='triangle-down', size=10)), row=1, col=1)

    action_colors = ['green' if a > 0 else 'red' for a in history['actions']]
    fig.add_trace(go.Bar(x=history['steps'], y=history['actions'], name='Action Strength', marker_color=action_colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=history['steps'], y=history['portfolio_value'], mode='lines', name='Total Value', line=dict(color='purple')), row=3, col=1)
    fig.add_trace(go.Scatter(x=history['steps'], y=history['cash'], mode='lines', name='Cash (Credit)', stackgroup='one', line=dict(color='grey')), row=4, col=1)
    fig.add_trace(go.Scatter(x=history['steps'], y=history['crypto_value'], mode='lines', name='Crypto Value', stackgroup='one', line=dict(color='orange')), row=4, col=1)

    fig.update_layout(title_text=f"Validation Performance - After Episode {episode_number}", height=1200, showlegend=False)
    fig.show()

def evaluate_agent(agent, env, initial_balance):
    state, _ = env.reset()
    done = False
    history = {
        'steps': [], 'actions': [], 'portfolio_value': [], 'cash': [], 'crypto_value': [], 'price': [],
        'buy_steps': [], 'buy_prices': [], 'sell_steps': [], 'sell_prices': []
    }
    prev_balance, prev_crypto_held = env.balance, env.crypto_held

    # Action smoothing in eval as well (matches env internal smoothing)
    while not done:
        action = agent.select_action(state)  # deterministic eval
        next_state, _, terminated, _, _ = env.step(action)
        done = terminated

        current_price = env.df.loc[env.current_step - 1, 'Original_Close']
        history['steps'].append(env.current_step - 1)
        history['actions'].append(float(action[0]))
        history['price'].append(float(current_price))
        history['portfolio_value'].append(float(env.portfolio_value))
        history['cash'].append(float(env.balance))
        history['crypto_value'].append(float(env.crypto_held * current_price))

        if env.balance < prev_balance:
            history['buy_steps'].append(env.current_step - 1)
            history['buy_prices'].append(float(current_price))
        elif env.crypto_held < prev_crypto_held:
            history['sell_steps'].append(env.current_step - 1)
            history['sell_prices'].append(float(current_price))

        prev_balance, prev_crypto_held = env.balance, env.crypto_held
        state = next_state

    metrics = env.get_metrics()
    final_pv = metrics['final_pv']
    perf_pct = (final_pv / initial_balance - 1.0) * 100.0
    return final_pv, perf_pct, history, (env.buy_trades, env.sell_trades), metrics

# =============================================================================
# Training
# =============================================================================

if __name__ == "__main__":
    # --- Config ---
    data_url = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"
    total_training_episodes = 200
    batch_size = 256
    initial_balance = 10_000
    validation_freq = 10
    EPISODE_DURATION_DAYS = 3
    episode_duration_minutes = EPISODE_DURATION_DAYS * 24 * 60

    # Exploration schedule: smaller noise to avoid harmful churn
    EXPLORATION_STEPS = 5_000
    TRAIN_NOISE_STD = 0.02
    MAX_GRAD_NORM = 1.0

    # Checkpointing
    os.makedirs("checkpoints", exist_ok=True)
    best_val_perf = -1e9
    best_train_pv = -1e9

    # --- Data ---
    data_df_raw = load_raw_data(data_url)

    # Split before normalization to avoid leakage
    split_idx = int(len(data_df_raw) * 0.8)
    train_df_raw = data_df_raw.iloc[:split_idx].reset_index(drop=True)
    val_df_raw = data_df_raw.iloc[split_idx:].reset_index(drop=True)

    # Columns to exclude from normalization (keep as raw for risk cap and pricing)
    exclude_cols = [
        'Original_Close',
        'log_close',          # keep raw log close drift
        'atr_raw', 'atr_pct_raw',
        'vol_5m_raw', 'vol_15m_raw', 'vol_60m_raw', 'vol_240m_raw'
    ]
    # Create missing exclude keys if not present
    exclude_cols = [c for c in exclude_cols if c in train_df_raw.columns]

    feature_cols, mean, std = fit_train_normalization(train_df_raw, exclude_cols=exclude_cols)
    train_df = apply_normalization(train_df_raw, feature_cols, mean, std)
    val_df = apply_normalization(val_df_raw, feature_cols, mean, std)

    print(f"Training data: {len(train_df)} points\nValidation data: {len(val_df)} points")

    # --- Environments (aligned settings) ---
    env_kwargs = dict(
        initial_balance=initial_balance,
        episode_duration_minutes=episode_duration_minutes,
        transaction_fee=0.001,
        action_threshold=0.15,
        action_interpretation="target",
        max_trade_frac_per_step=0.10,
        cooldown_steps=15,
        allow_short=False,
        min_usd_trade=100.0,
        use_risk_cap=True,
        risk_cap_base=0.10,
        risk_cap_min=0.05,
        risk_cap_max=0.60,
        action_smoothing_alpha=0.30,
        gain_scale=1.0,
        loss_scale=2.0,
        turnover_penalty_coeff=0.10,
        trade_exec_penalty=0.0005,
        trade_pnl_coeff=0.0,           # no realized PnL reward
        terminal_profit_scale=2.0,
        early_stop_drawdown=0.20,
        early_stop_penalty=3.0
    )

    train_env = TradingEnv(df=train_df, is_training=True, **env_kwargs)
    val_env = TradingEnv(df=val_df, is_training=False, **env_kwargs)

    # --- Agent (TD3) ---
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    max_action = float(train_env.action_space.high[0])

    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.08,
        noise_clip=0.20,
        policy_delay=2,
        grad_clip_norm=MAX_GRAD_NORM
    )

    # --- Run-level trackers ---
    run_high = -1e18
    run_low = 1e18
    global_steps = 0

    for ep in range(total_training_episodes):
        state, _ = train_env.reset()
        episode_reward = 0.0
        terminated = False
        num_steps = len(train_env.df) - train_env.lookback_window - 1

        with tqdm(total=num_steps, desc=f"Training Ep {ep+1}/{total_training_episodes}") as pbar:
            for t in range(num_steps):
                if global_steps < EXPLORATION_STEPS:
                    action = train_env.action_space.sample()
                else:
                    action = agent.select_action(state)
                    noise = np.random.normal(0, TRAIN_NOISE_STD, size=action_dim).astype(np.float32)
                    action = np.clip(action + noise, -1.0, 1.0)

                next_state, reward, terminated, _, _ = train_env.step(action)

                agent.replay_buffer.add((state, next_state, action, reward, float(terminated)))
                global_steps += 1

                agent.update(batch_size)

                state = next_state
                episode_reward += reward

                pbar.set_postfix({
                    "PV": f"${train_env.portfolio_value:,.2f}",
                    "DD": f"{train_env.max_drawdown*100:5.2f}%",
                    "Trd": f"{train_env.num_trades}",
                })
                pbar.update(1)

                if terminated:
                    break

        metrics = train_env.get_metrics()
        portfolio_perf = (train_env.portfolio_value / initial_balance - 1.0) * 100.0
        run_high = max(run_high, metrics['max_pv'])
        run_low = min(run_low, metrics['min_pv'])

        print(
            f"Ep {ep+1} Finished | "
            f"Reward: {episode_reward:.4f} | "
            f"PV: ${train_env.portfolio_value:,.2f} ({portfolio_perf:+.2f}%) | "
            f"MaxPV: ${metrics['max_pv']:,.2f} | MinPV: ${metrics['min_pv']:,.2f} | "
            f"MaxDD: {metrics['max_drawdown']*100:5.2f}% | Sharpe(d): {metrics['sharpe_per_day']:.2f} | "
            f"Trades B/S/T: {metrics['trades_buy']}/{metrics['trades_sell']}/{metrics['trades_total']} | "
            f"StepWinRate: {metrics['step_win_rate_pct']:.1f}% | TradeWinRate: {metrics['trade_win_rate_pct']:.1f}% | "
            f"Turnover: {metrics['turnover_pct']:.1f}% | TiM: {metrics['time_in_market_pct']:.1f}% | "
            f"T>Init: {metrics['time_above_initial_pct']:.1f}% | RealPnL: ${metrics['realized_pnl']:,.2f}"
        )
        print(f"Run High so far: ${run_high:,.2f} | Run Low so far: ${run_low:,.2f}")

        # Save best training PV
        if train_env.portfolio_value > best_train_pv:
            best_train_pv = train_env.portfolio_value
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_target': agent.actor_target.state_dict(),
                'critic_target': agent.critic_target.state_dict(),
            }, "checkpoints/best_train.pt")

        # Validation
        if (ep + 1) % validation_freq == 0:
            val_portfolio, val_perf, history, trades, val_metrics = evaluate_agent(agent, val_env, initial_balance)
            print("------------------------------------------------------")
            print(
                f"VALIDATION after Ep {ep+1} | "
                f"PV: ${val_portfolio:,.2f} ({val_perf:+.2f}%) | "
                f"MaxPV: ${val_metrics['max_pv']:,.2f} | MinPV: ${val_metrics['min_pv']:,.2f} | "
                f"MaxDD: {val_metrics['max_drawdown']*100:5.2f}% | Sharpe(d): {val_metrics['sharpe_per_day']:.2f} | "
                f"Trades B/S/T: {trades[0]}/{trades[1]}/{val_metrics['trades_total']} | "
                f"StepWinRate: {val_metrics['step_win_rate_pct']:.1f}% | TradeWinRate: {val_metrics['trade_win_rate_pct']:.1f}% | "
                f"Turnover: {val_metrics['turnover_pct']:.1f}% | TiM: {val_metrics['time_in_market_pct']:.1f}% | "
                f"T>Init: {val_metrics['time_above_initial_pct']:.1f}% | RealPnL: ${val_metrics['realized_pnl']:,.2f}"
            )
            print("------------------------------------------------------")

            if val_perf > best_val_perf:
                best_val_perf = val_perf
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'actor_target': agent.actor_target.state_dict(),
                    'critic_target': agent.critic_target.state_dict(),
                }, "checkpoints/best_val.pt")
                print(f"New best validation performance: {best_val_perf:+.2f}% -> model saved to checkpoints/best_val.pt")

            plot_validation_results(history, ep + 1)

    train_env.close()
    val_env.close()
    print("\n--- Training Finished ---")
