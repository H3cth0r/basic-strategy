import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ta import add_all_ta_features
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from collections import deque
import math
import os
import json
from collections import defaultdict

# -------------------------------------------------
# 0. HYPERPARAMETERS AND CONFIGURATION
# -------------------------------------------------
class Config:
    # --- Training ---
    EPISODES = 50
    EPISODE_LENGTH_DAYS = 2.5
    BATCH_SIZE = 64
    GAMMA = 0.99
    LEARNING_RATE = 0.0001
    TARGET_UPDATE_FREQ = 10  # episodes

    # Replay and learning cadence
    MEMORY_CAPACITY = 100000
    LEARN_START_MEMORY = 2000  # start learning once memory >= this
    LEARN_EVERY = 1            # learn every step once above threshold

    # Exploration in validation
    EVAL_EPSILON = 0.1  # small epsilon-greedy during validation to avoid early all-HOLD

    # Decision frequency scheduling (minutes)
    DECISION_FREQUENCY_BASE = 1
    DECISION_FREQUENCY_LATER = 5
    DECISION_FREQ_SWITCH_EPISODE = 5  # after this episode, use later frequency

    # Cooldown measured in decision ticks (not raw minutes)
    ACTION_COOLDOWN = 3  # decision ticks

    # --- Model Architecture ---
    ATTENTION_DIM = 64
    ATTENTION_HEADS = 4
    FC_UNITS_1 = 256
    FC_UNITS_2 = 128

    # --- Environment ---
    INITIAL_CREDIT = 100.0
    WINDOW_SIZE = 180
    FEE = 0.001
    MIN_TRADE_CREDIT_BUY = 0.1     # lowered so small accounts can still trade
    MIN_TRADE_HOLDINGS_SELL = 0.5  # in currency terms

    # Reward shaping coefficients
    REWARD_REALIZED = 8.0           # realized pnl scaling
    REWARD_CREDIT_DELTA = 2.0       # reward for increases in cash
    REWARD_PV_DELTA = 0.3           # reward for portfolio value changes
    TRADE_PENALTY = 0.15            # fixed penalty per executed trade
    TIME_PENALTY = 0.002            # per step while in position
    DRAWDOWN_PENALTY = 0.05         # penalty proportional to drawdown

    # --- Prioritized Experience Replay (PER) ---
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_END = 1.0
    PER_BETA_ANNEAL_STEPS = 100000

    # --- Validation ---
    VAL_SEGMENT_MINUTES = 2 * 24 * 60

cfg = Config()
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# -------------------------------------------------
# Seeding for reproducibility
# -------------------------------------------------
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# -------------------------------------------------
# 1. DATA LOADING AND PRE-PROCESSING (NO LEAKAGE)
# -------------------------------------------------
def load_and_build_features():
    print("Downloading and building features…")
    url = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"
    column_names = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
    try:
        df = pd.read_csv(
            url, skiprows=[1, 2], header=0, names=column_names,
            parse_dates=["Datetime"], index_col="Datetime",
            dtype={"Volume": "int64"}, na_values=["NA", "N/A", ""],
            keep_default_na=True,
        )
        df.index = pd.to_datetime(df.index, utc=True)
    except Exception as e:
        print(f"Error reading data: {e}")
        return pd.DataFrame()

    df.ffill(inplace=True)
    df.dropna(inplace=True)

    print("Calculating technical indicators…")
    add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )

    # Keep unnormalized close for PnL calculations
    df["Original_Close"] = df["Close"].copy()

    # Drop any residual NaNs
    df.dropna(inplace=True)
    print("Features built.")
    return df

def split_and_normalize(df):
    # Split
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()

    # Compute normalization stats from train only (excluding Original_Close)
    feature_cols = [c for c in df.columns if c != "Original_Close"]
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std()
    std_replaced = std.replace(0, 1e-7)

    def apply_norm(x):
        out = x.copy()
        out[feature_cols] = (x[feature_cols] - mean) / (std_replaced + 1e-7)
        out["Original_Close"] = x["Original_Close"]  # ensure unchanged
        return out

    print("Applying train-based normalization to train/val/test…")
    train_norm = apply_norm(train_df)
    val_norm = apply_norm(val_df)
    test_norm = apply_norm(test_df)
    return train_norm, val_norm, test_norm

# -------------------------------------------------
# 2. RL COMPONENTS (MEMORY & NETWORK)
# -------------------------------------------------
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left, right = 2 * idx + 1, 2 * idx + 2
        if left >= len(self.tree):
            return idx
        if self.tree[left] == 0 and self.tree[right] == 0:
            return idx  # safeguard if priorities collapse
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(right, s - self.tree[left])

    def total(self):
        return max(self.tree[0], 1e-8)

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=cfg.PER_ALPHA):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.e = 0.01
        self.max_priority = 1.0

    def push(self, *args):
        self.tree.add(self.max_priority, args)

    def sample(self, batch_size, beta=cfg.PER_BETA_START):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max() + 1e-8
        return batch, idxs, torch.FloatTensor(is_weight).to(device)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            priority = (priority + self.e) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features, self.out_features, self.std_init = in_features, out_features, std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class SharedSelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim, attention_heads=1, dropout_rate=0.1):
        super().__init__()
        self.attention_dim, self.attention_heads = attention_dim, attention_heads
        self.head_dim = attention_dim // attention_heads
        self.query_proj = nn.Linear(input_dim, attention_dim)
        self.key_proj = nn.Linear(input_dim, attention_dim)
        self.value_proj = nn.Linear(input_dim, attention_dim)
        self.output_proj = nn.Linear(attention_dim, attention_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sequence_features):
        if sequence_features.ndim == 2:
            sequence_features = sequence_features.unsqueeze(0)
        batch_size, seq_len, _ = sequence_features.shape
        Q = self.query_proj(sequence_features).view(batch_size, seq_len, self.attention_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key_proj(sequence_features).view(batch_size, seq_len, self.attention_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value_proj(sequence_features).view(batch_size, seq_len, self.attention_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(energy, dim=-1)
        attention_weights = self.dropout(attention_weights)
        weighted_values = torch.matmul(attention_weights, V).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.attention_dim)
        output = self.output_proj(weighted_values)
        return output.mean(dim=1)

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        market_feature_dim = state_dim - 4
        self.attention = SharedSelfAttention(market_feature_dim, cfg.ATTENTION_DIM, cfg.ATTENTION_HEADS)
        self.fc1 = nn.Linear(cfg.ATTENTION_DIM + 4, cfg.FC_UNITS_1)
        self.value_stream = nn.Sequential(NoisyLinear(cfg.FC_UNITS_1, cfg.FC_UNITS_2), nn.ReLU(), NoisyLinear(cfg.FC_UNITS_2, 1))
        self.advantage_stream = nn.Sequential(NoisyLinear(cfg.FC_UNITS_1, cfg.FC_UNITS_2), nn.ReLU(), NoisyLinear(cfg.FC_UNITS_2, action_dim))

    def forward(self, state):
        market_data, portfolio_state = state[:, :, :-4], state[:, -1, -4:]
        attention_output = self.attention(market_data)
        combined_input = torch.cat([attention_output, portfolio_state], dim=1)
        features = F.relu(self.fc1(combined_input))
        value, advantage = self.value_stream(features), self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# -------------------------------------------------
# 3. DQN AGENT
# -------------------------------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.beta, self.learn_step_counter = cfg.PER_BETA_START, 0
        self.total_steps = 0  # training-only steps
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.LEARNING_RATE)
        self.memory = PrioritizedReplayMemory(cfg.MEMORY_CAPACITY)

    def _action_mask_from_state(self, state_np):
        # state_np: [window, features]; last 4 are [credit_ratio, holdings_ratio, unrealized_pnl_ratio, time]
        last = state_np[-1]
        credit_ratio = last[-4]
        holdings_ratio = last[-3]
        mask = np.ones(self.action_dim, dtype=bool)
        # Actions: 0:HOLD, 1:BUY_25, 2:BUY_50, 3:SELL_25, 4:SELL_50, 5:SELL_100, 6:BUY_100
        if credit_ratio < 0.01:
            mask[1] = False
            mask[2] = False
            if self.action_dim > 6:
                pass
            if self.action_dim >= 7:
                mask[6] = False
        if holdings_ratio < 0.01:
            mask[3] = False
            mask[4] = False
            if self.action_dim >= 6:
                mask[5] = False
        return mask

    def act(self, state, is_eval=False):
        # Count only training steps for warmup/learning schedules
        if not is_eval:
            self.total_steps += 1

        state_np = np.array(state, dtype=np.float32)
        mask = self._action_mask_from_state(state_np)
        valid_actions = np.where(mask)[0].tolist()
        if len(valid_actions) == 0:
            return 0  # fallback

        # Epsilon in eval to avoid all-HOLD early
        if is_eval and random.random() < cfg.EVAL_EPSILON:
            return random.choice(valid_actions)

        # Mode and NoisyNet reset
        if is_eval:
            self.policy_net.eval()
        else:
            self.policy_net.train()
            self.policy_net.reset_noise()

        with torch.no_grad():
            s = torch.FloatTensor(state_np).unsqueeze(0).to(device)
            q_values = self.policy_net(s).squeeze(0).cpu().numpy()

        # Apply mask: set invalid actions to very low
        q_values_masked = q_values.copy()
        q_values_masked[~mask] = -1e9
        return int(np.argmax(q_values_masked))

    def learn(self):
        if len(self.memory) < cfg.BATCH_SIZE:
            return
        self.policy_net.train()
        self.target_net.eval()

        # Anneal beta
        self.beta = min(cfg.PER_BETA_END, cfg.PER_BETA_START + self.learn_step_counter * (cfg.PER_BETA_END - cfg.PER_BETA_START) / cfg.PER_BETA_ANNEAL_STEPS)
        self.learn_step_counter += 1

        transitions, indices, is_weights = self.memory.sample(cfg.BATCH_SIZE, self.beta)
        states, actions, rewards, next_states, dones = zip(*transitions)

        state_batch = torch.FloatTensor(np.array(states)).to(device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(rewards).to(device)
        non_final_mask = torch.tensor([s is not None for s in next_states], device=device, dtype=torch.bool)
        if any(non_final_mask.cpu().numpy()):
            non_final_next_states = torch.FloatTensor(np.array([s for s in next_states if s is not None])).to(device)
        else:
            non_final_next_states = torch.empty((0,) + state_batch.shape[1:], device=device)

        next_q_values = torch.zeros(cfg.BATCH_SIZE, device=device)
        if non_final_next_states.size(0) > 0:
            with torch.no_grad():
                next_actions = self.policy_net(non_final_next_states).argmax(1).unsqueeze(1)
                next_q_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze(1)

        expected_q_values = reward_batch + (cfg.GAMMA * next_q_values)
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        td_errors = (expected_q_values - q_values).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        loss = (is_weights * F.mse_loss(q_values, expected_q_values, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Refresh NoisyNet noise post-update for continued exploration
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from {path}")

# -------------------------------------------------
# 4. TRADING ENVIRONMENT
# -------------------------------------------------
class TradingEnvironment:
    # Actions: 0:HOLD, 1:BUY_25, 2:BUY_50, 3:SELL_25, 4:SELL_50, 5:SELL_100, 6:BUY_100
    def __init__(self, data, initial_credit=cfg.INITIAL_CREDIT, window_size=cfg.WINDOW_SIZE):
        self.data = data
        self.normalized_data = data.drop(columns=["Original_Close"])
        self.initial_credit = initial_credit
        self.window_size = window_size
        self.n_features = self.normalized_data.shape[1] + 4
        self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.n_actions = len(self.action_space)

    def reset(self, episode_start_index=0, initial_credit=None, holdings=0.0):
        self.credit = initial_credit if initial_credit is not None else self.initial_credit
        self.start_credit = self.credit
        self.holdings = holdings
        self.average_buy_price = 0.0
        self.current_step = episode_start_index + self.window_size
        self.trades = []
        self.steps_in_position = 0
        self.cooldown = 0  # in decision ticks
        initial_price = self.data["Original_Close"].iloc[self.current_step - 1]
        self.max_portfolio_value = self.credit + self.holdings * initial_price
        return self._get_state()

    def _get_state(self):
        start, end = self.current_step - self.window_size, self.current_step
        market_data = self.normalized_data.iloc[start:end].values
        current_price = self.data["Original_Close"].iloc[self.current_step]
        portfolio_value = self.credit + self.holdings * current_price
        credit_ratio = self.credit / portfolio_value if portfolio_value > 0 else 1.0
        holdings_ratio = (self.holdings * current_price) / portfolio_value if portfolio_value > 0 else 0.0
        unrealized_pnl_ratio = (current_price - self.average_buy_price) / self.average_buy_price if self.average_buy_price > 0 else 0.0
        time_in_pos_norm = math.log(self.steps_in_position + 1) / 5.0
        portfolio_state = np.array([[credit_ratio, holdings_ratio, unrealized_pnl_ratio, time_in_pos_norm]] * self.window_size)
        return np.concatenate([market_data, portfolio_state], axis=1)

    def step(self, action_idx, decision_tick=False):
        # Cooldown is decremented only on decision ticks
        if decision_tick and self.cooldown > 0:
            action_idx = 0  # force HOLD
            self.cooldown -= 1

        action = self.action_space[action_idx]
        current_price = self.data["Original_Close"].iloc[self.current_step]
        next_price = self.data["Original_Close"].iloc[self.current_step + 1] if self.current_step + 1 < len(self.data) else current_price

        realized_pnl, trade_executed = 0.0, False
        buy_fraction, sell_fraction = 0.0, 0.0

        if action == 1:
            buy_fraction = 0.25
        elif action == 2:
            buy_fraction = 0.50
        elif action == 6:
            buy_fraction = 1.00
        elif action == 3:
            sell_fraction = 0.25
        elif action == 4:
            sell_fraction = 0.50
        elif action == 5:
            sell_fraction = 1.00

        credit_before = self.credit
        pv_before = self.credit + self.holdings * current_price

        # Execute Buy
        if buy_fraction > 0:
            # Prevent extreme over-investment (cap exposure ~99%)
            portfolio_value = self.credit + self.holdings * current_price
            current_exposure = (self.holdings * current_price) / portfolio_value if portfolio_value > 0 else 0.0
            if current_exposure < 0.99:  # soft cap
                investment = self.credit * buy_fraction
                if investment > cfg.MIN_TRADE_CREDIT_BUY:
                    buy_amount_asset = (investment * (1 - cfg.FEE)) / current_price
                    total_cost = (self.average_buy_price * self.holdings) + investment
                    self.holdings += buy_amount_asset
                    self.credit -= investment
                    self.average_buy_price = total_cost / self.holdings if self.holdings > 0 else 0
                    self.trades.append({"step": self.current_step, "type": "buy", "price": current_price, "amount": buy_amount_asset})
                    trade_executed = True

        # Execute Sell
        elif sell_fraction > 0 and self.holdings > 0:
            sell_amount_asset = min(self.holdings, self.holdings * sell_fraction)
            if sell_amount_asset * current_price > cfg.MIN_TRADE_HOLDINGS_SELL:
                sell_value = sell_amount_asset * current_price * (1 - cfg.FEE)
                self.credit += sell_value
                self.holdings -= sell_amount_asset
                realized_pnl = (current_price - self.average_buy_price) * sell_amount_asset
                if self.holdings < 1e-9:
                    self.holdings = 0.0
                    self.average_buy_price = 0.0
                self.trades.append({"step": self.current_step, "type": "sell", "price": current_price, "amount": sell_amount_asset})
                trade_executed = True

        # Post-trade management
        if trade_executed and decision_tick:
            self.cooldown = cfg.ACTION_COOLDOWN
            self.steps_in_position = 0
        elif self.holdings > 0:
            self.steps_in_position += 1

        # Advance time
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        pv_after = self.credit + self.holdings * next_price
        if pv_after > self.max_portfolio_value:
            self.max_portfolio_value = pv_after

        # If done, force liquidation for metrics consistency
        if done and self.holdings > 0:
            liquidate_value = self.holdings * next_price * (1 - cfg.FEE)
            realized_pnl += (next_price - self.average_buy_price) * self.holdings
            self.credit += liquidate_value
            self.trades.append({"step": self.current_step, "type": "sell", "price": next_price, "amount": self.holdings})
            self.holdings = 0.0
            self.average_buy_price = 0.0
            pv_after = self.credit

        # Reward calculation
        reward = self._calculate_reward(realized_pnl, pv_before, pv_after, credit_before, self.credit, trade_executed)

        next_state = self._get_state() if not done else None

        info = {
            "portfolio_value": pv_after,
            "credit": self.credit,
            "holdings": self.holdings,
            "trades": self.trades
        }
        return next_state, reward, done, info

    def _calculate_reward(self, realized_pnl, pv_before, pv_after, credit_before, credit_after, trade_executed):
        reward = 0.0

        # Realized PnL
        if realized_pnl != 0:
            reward += cfg.REWARD_REALIZED * (realized_pnl / self.start_credit)

        # Trading cost penalty
        if trade_executed:
            reward -= cfg.TRADE_PENALTY

        # Credit growth (only reward increases)
        credit_delta = max(0.0, credit_after - credit_before)
        reward += cfg.REWARD_CREDIT_DELTA * (credit_delta / self.start_credit)

        # Portfolio value momentum
        pv_delta = pv_after - pv_before
        reward += cfg.REWARD_PV_DELTA * (pv_delta / self.start_credit)

        # Time in position penalty
        if self.holdings > 0:
            reward -= cfg.TIME_PENALTY * self.steps_in_position

        # Drawdown penalty
        if self.max_portfolio_value > 0:
            drawdown = max(0.0, (self.max_portfolio_value - pv_after) / self.max_portfolio_value)
            reward -= cfg.DRAWDOWN_PENALTY * drawdown

        return reward

# -------------------------------------------------
# 5. PLOTTING
# -------------------------------------------------
def plot_results(df, episode, portfolio_history, credit_history, holdings_history, trades, plot_title_prefix="", segment_boundaries=None):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=(f"{plot_title_prefix} Price and Trades", "Portfolio Value", "Credit", "Holdings Value"))
    start_index = len(df) - len(portfolio_history)
    plot_df = df.iloc[start_index:].copy()
    plot_df["portfolio_value"], plot_df["credit"], plot_df["holdings_value"] = portfolio_history, credit_history, holdings_history
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Original_Close"], mode="lines", name="Price", line=dict(color="lightgrey")), row=1, col=1)
    buy_trades = [t for t in trades if t["type"] == "buy"]
    sell_trades = [t for t in trades if t["type"] == "sell"]
    if buy_trades:
        fig.add_trace(go.Scatter(x=[df.index[min(t["step"], len(df)-1)] for t in buy_trades], y=[t["price"] for t in buy_trades], mode="markers", marker=dict(color="green", symbol="triangle-up", size=8), name="Buy"), row=1, col=1)
    if sell_trades:
        fig.add_trace(go.Scatter(x=[df.index[min(t["step"], len(df)-1)] for t in sell_trades], y=[t["price"] for t in sell_trades], mode="markers", marker=dict(color="red", symbol="triangle-down", size=8), name="Sell"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["portfolio_value"], mode="lines", name="PV"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["credit"], mode="lines", name="Credit"), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["holdings_value"], mode="lines", name="Holdings"), row=4, col=1)
    if segment_boundaries:
        for boundary in segment_boundaries:
            if 0 <= boundary < len(df.index):
                fig.add_vline(x=df.index[boundary], line_width=1, line_dash="dash", line_color="black", opacity=0.4)
    fig.update_layout(height=1000, title_text=f"{plot_title_prefix} Results (Episode {episode})", showlegend=False)
    fig.show()

# -------------------------------------------------
# 6. TRAINING AND EVALUATION LOGIC
# -------------------------------------------------
def get_decision_frequency_for_episode(e_idx):
    # e_idx is 0-based
    return cfg.DECISION_FREQUENCY_LATER if (e_idx + 1) > cfg.DECISION_FREQ_SWITCH_EPISODE else cfg.DECISION_FREQUENCY_BASE

def run_episode(env, agent, data, is_eval=False, initial_credit=None, initial_holdings=0.0, decision_frequency=1):
    state = env.reset(initial_credit=initial_credit, holdings=initial_holdings)
    done = False
    portfolio_values, credits, holdings_values = [], [], []
    pbar_desc = "VALIDATING" if is_eval else "TRAINING"
    total_steps = len(data) - env.window_size - 1
    pbar = tqdm(total=total_steps, desc=pbar_desc, leave=False)

    action_counts = defaultdict(int)

    for step in range(total_steps):
        decision_tick = (step % decision_frequency == 0)

        if decision_tick:
            action = agent.act(state, is_eval)
            action_counts[action] += 1
        else:
            action = 0  # HOLD if not a decision tick
            action_counts[action] += 1

        next_state, reward, done, info = env.step(action, decision_tick=decision_tick)

        if not is_eval:
            agent.memory.push(state, action, reward, next_state, done)
            if len(agent.memory) >= cfg.LEARN_START_MEMORY and (step % cfg.LEARN_EVERY == 0):
                agent.learn()

        state = next_state

        # Append histories including the final step (post-liquidation)
        portfolio_values.append(info["portfolio_value"])
        credits.append(info["credit"])
        # For holdings value, use the current price at env.current_step if available, else last price
        current_index = min(env.current_step, len(data) - 1)
        current_price = data["Original_Close"].iloc[current_index]
        holdings_values.append(info["holdings"] * current_price)

        pbar.update(1)
        if done:
            break
    pbar.close()

    final_pv = portfolio_values[-1] if portfolio_values else initial_credit
    final_credit = credits[-1] if credits else initial_credit
    final_holdings = info["holdings"] if info else 0.0

    return portfolio_values, credits, holdings_values, env.trades, final_pv, final_credit, final_holdings, dict(action_counts)

def validate_in_segments(full_val_data, agent, window_size, initial_credit, decision_frequency):
    n_total = len(full_val_data)
    segment_starts = list(range(0, n_total - window_size - 1, cfg.VAL_SEGMENT_MINUTES))
    all_portfolio, all_credit, all_holdings, all_trades = [], [], [], []
    segment_metrics = []
    current_credit, current_holdings = initial_credit, 0.0

    for seg_idx, start in enumerate(segment_starts, 1):
        end = min(start + cfg.VAL_SEGMENT_MINUTES, n_total)
        if end - start <= window_size:
            continue
        segment_data = full_val_data.iloc[start:end].copy().reset_index(drop=True)
        val_env = TradingEnvironment(segment_data, initial_credit=current_credit, window_size=window_size)
        pv_hist, credit_hist, hold_hist, trades, final_pv, final_credit, final_holdings, action_counts = run_episode(
            val_env, agent, segment_data, is_eval=True, initial_credit=current_credit, initial_holdings=current_holdings,
            decision_frequency=decision_frequency
        )
        # Shift trade steps to full validation index
        for tr in trades:
            tr["step"] += start
        all_trades.extend(trades)
        all_portfolio.extend(pv_hist)
        all_credit.extend(credit_hist)
        all_holdings.extend(hold_hist)

        # Compute segment metrics with correct bases
        initial_pv = pv_hist[0] if pv_hist else (current_credit)
        seg_return = ((final_pv - initial_pv) / initial_pv * 100) if initial_pv > 0 else 0.0
        initial_credit_seg = credit_hist[0] if credit_hist else current_credit
        credit_growth = ((final_credit - initial_credit_seg) / initial_credit_seg * 100) if initial_credit_seg > 0 else 0.0

        buys = len([t for t in trades if t["type"] == "buy"])
        sells = len([t for t in trades if t["type"] == "sell"])
        segment_metrics.append({
            "seg": seg_idx,
            "start": full_val_data.index[start],
            "end": full_val_data.index[end - 1],
            "final_pv": final_pv,
            "ret": seg_return,
            "credit_growth": credit_growth,
            "buys": buys,
            "sells": sells,
            "final_credit": final_credit
        })

        # Carry forward credit and holdings (holdings should be 0 after auto-liquidation)
        current_credit, current_holdings = final_credit, final_holdings

    return all_portfolio, all_credit, all_holdings, all_trades, segment_metrics, segment_starts

# -------------------------------------------------
# 7. MAIN EXECUTION
# -------------------------------------------------
def main():
    print(f"Using device: {device}")
    full_data = load_and_build_features()
    if full_data.empty or len(full_data) < cfg.WINDOW_SIZE * 2:
        print("Not enough data to run.")
        return

    train_data, val_data, test_data = split_and_normalize(full_data)

    print(f"Training data size:   {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Test data size:       {len(test_data)}")

    dummy_env = TradingEnvironment(train_data)
    agent = DQNAgent(state_dim=dummy_env.n_features, action_dim=dummy_env.n_actions)

    os.makedirs("saved_models", exist_ok=True)
    best_val_credit_growth = -1e9

    for e in range(cfg.EPISODES):
        print(f"\n=== Episode {e + 1}/{cfg.EPISODES} ===")
        episode_minutes = int(cfg.EPISODE_LENGTH_DAYS * 24 * 60)
        max_start = len(train_data) - episode_minutes - 1
        if max_start <= 0:
            print("Training data is smaller than an episode length. Skipping.")
            continue

        start_idx = random.randint(0, max_start)
        episode_data = train_data.iloc[start_idx:start_idx + episode_minutes].copy().reset_index(drop=True)
        print(f"Training on slice {train_data.index[start_idx]} -> {train_data.index[start_idx + episode_minutes - 1]}")

        train_env = TradingEnvironment(episode_data, window_size=cfg.WINDOW_SIZE)
        decision_frequency = get_decision_frequency_for_episode(e)

        _pv, _cred, _hold, _trades, _fpv, _fcred, _fhold, action_counts = run_episode(
            train_env, agent, episode_data, is_eval=False, initial_credit=cfg.INITIAL_CREDIT, decision_frequency=decision_frequency
        )

        if (e + 1) % cfg.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
            print("Target network synchronized.")

        print("\n-> Validation phase…")
        val_freq = cfg.DECISION_FREQUENCY_LATER  # use stable freq for validation
        pv_hist, credit_hist, hold_hist, trades, seg_metrics, seg_starts = validate_in_segments(
            val_data, agent, window_size=cfg.WINDOW_SIZE, initial_credit=cfg.INITIAL_CREDIT, decision_frequency=val_freq
        )

        print("\nSegment-by-segment results:")
        for m in seg_metrics:
            print(f"  Seg {m['seg']:02d} [{m['start']} → {m['end']}]  "
                  f"PV: ${m['final_pv']:.2f} (Ret: {m['ret']:.2f}%) | "
                  f"Credit: ${m['final_credit']:.2f} (Growth: {m['credit_growth']:.2f}%) | "
                  f"Buys: {m['buys']}, Sells: {m['sells']}")

        if seg_metrics:
            credit_growths = [m["credit_growth"] for m in seg_metrics]
            mean_growth = float(np.mean(credit_growths))
            if mean_growth > best_val_credit_growth:
                best_val_credit_growth = mean_growth
                agent.save_model(f"saved_models/best_model.pth")
                print(f"New best validation credit growth: {best_val_credit_growth:.2f}%. Model saved.")
            total_buys = sum(m["buys"] for m in seg_metrics)
            total_sells = sum(m["sells"] for m in seg_metrics)
            print("\nAggregated validation:")
            print(f"  Mean segment credit growth: {mean_growth:.2f}%")
            print(f"  Total buys: {total_buys}, Total sells: {total_sells}")

        if e < 3 and pv_hist:
            plot_results(
                val_data, e + 1, pv_hist, credit_hist, hold_hist, trades,
                "Validation", [s + cfg.WINDOW_SIZE for s in seg_starts[1:]]
            )

    print("\n=== FINAL TEST ON UNSEEN DATA ===")
    agent.load_model("saved_models/best_model.pth")
    test_env = TradingEnvironment(test_data.reset_index(drop=True))
    pv, cred, hold, trades, final_pv, final_credit, final_holdings, act_counts = run_episode(
        test_env, agent, test_data.reset_index(drop=True), is_eval=True, initial_credit=cfg.INITIAL_CREDIT,
        decision_frequency=cfg.DECISION_FREQUENCY_LATER
    )
    if pv:
        pv_return = (final_pv - cfg.INITIAL_CREDIT) / cfg.INITIAL_CREDIT * 100
        credit_return = (final_credit - cfg.INITIAL_CREDIT) / cfg.INITIAL_CREDIT * 100
        buys, sells = len([t for t in trades if t["type"] == "buy"]), len([t for t in trades if t["type"] == "sell"])
        print(f"Final PV: ${final_pv:,.2f} (Return: {pv_return:.2f}%)")
        print(f"Final Credit: ${final_credit:,.2f} (Return: {credit_return:.2f}%)")
        print(f"Total Buys: {buys}, Total Sells: {sells}")
        plot_results(test_data, "Final", pv, cred, hold, trades, "Final Test")

if __name__ == "__main__":
    main()
