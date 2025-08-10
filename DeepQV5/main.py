import os
import math
import json
import random
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tqdm import tqdm

# Optional: curated features via 'ta' for a smaller, less noisy set
try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import AverageTrueRange
except Exception:
    RSIIndicator = None
    MACD = None
    AverageTrueRange = None

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

    # Soft target network (Polyak) update parameter (applied every learn() step)
    SOFT_TARGET_TAU = 0.01

    # Experience replay
    MEMORY_CAPACITY = 100000
    LEARN_START_MEMORY = 1500  # count of aggregated decision transitions (not raw minutes)

    # Learning cadence: run K gradient steps after each pushed decision transition
    LEARN_UPDATES_PER_PUSH = 2

    # EMA weights for evaluation
    EMA_MOMENTUM = 0.995  # higher => smoother; evaluate with ema_net

    # Decision frequency (stabilize training and eval to same cadence)
    TRAIN_DECISION_FREQUENCIES = [15]  # minutes per decision (use single cadence to reduce drift)
    VAL_DECISION_FREQUENCY = 15        # minutes per decision

    # Cooldown (minutes)
    ACTION_COOLDOWN_MINUTES = 30

    # --- Model Architecture ---
    ATTENTION_DIM = 64
    ATTENTION_HEADS = 4
    FC_UNITS_1 = 256
    FC_UNITS_2 = 128

    # --- Environment ---
    INITIAL_CREDIT = 100.0
    WINDOW_SIZE = 180
    FEE = 0.001
    MIN_TRADE_CREDIT_BUY = 2.0
    MIN_TRADE_HOLDINGS_SELL = 2.0  # in currency terms
    MAX_EXPOSURE = 0.90            # strict cap on exposure when buying

    # Reward shaping and scaling
    PV_LOGRET_SCALE = 100.0        # scale log-PV return to improve TD magnitude
    REWARD_REALIZED = 0.05         # small bonus on realized PnL (normalized by start PV)
    TRADE_PENALTY = 0.001          # small fixed penalty per executed trade
    TIME_PENALTY = 0.0
    DRAWDOWN_PENALTY = 0.005
    FEES_PENALTY_COEF = 0.25
    REALIZED_CLIP = 0.05           # cap realized component per step
    FEES_CLIP = 0.02               # cap fees penalty per step
    REWARD_CLIP = 0.1              # clip total reward per step

    # --- Prioritized Experience Replay (PER) ---
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_END = 1.0
    PER_BETA_ANNEAL_STEPS = 100000
    PER_PRIORITY_EPS = 1e-5
    PER_PRIORITY_MAX = 10.0

    # --- N-step returns ---
    N_STEP = 3  # across decision transitions

    # --- Validation ---
    VAL_SEGMENT_MINUTES = 2 * 24 * 60  # 2 days per segment

    # --- Features ---
    USE_CURATED_FEATURES = False  # set True to use a compact, curated feature set
    CLIP_UNREALIZED_PNL = 2.0     # clip for unrealized pnl ratio feature

    # --- Early stopping ---
    PATIENCE = 10  # stop if no new best for this many episodes

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

    print("Calculating features…")
    df["Original_Close"] = df["Close"].copy()

    if cfg.USE_CURATED_FEATURES and RSIIndicator is not None and MACD is not None:
        # Basic returns
        df["ret_1"] = np.log(df["Close"]).diff().fillna(0.0)
        df["ret_5"] = np.log(df["Close"]).diff(5).fillna(0.0)
        df["ret_15"] = np.log(df["Close"]).diff(15).fillna(0.0)
        df["ret_60"] = np.log(df["Close"]).diff(60).fillna(0.0)
        # Rolling volatility
        df["vol_30"] = df["ret_1"].rolling(30).std().fillna(method="bfill").fillna(0.0)
        # RSI
        rsi = RSIIndicator(close=df["Close"], window=14, fillna=True)
        df["rsi_14"] = rsi.rsi()
        # MACD
        macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
        # ATR
        if AverageTrueRange is not None:
            atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14, fillna=True)
            df["atr_14"] = atr.average_true_range()
        # Volume z-score
        df["vol_z"] = ((df["Volume"] - df["Volume"].rolling(60).mean()) / (df["Volume"].rolling(60).std() + 1e-8)).fillna(0.0)
        # Keep only curated features + Original_Close
        keep_cols = [c for c in df.columns if c in {
            "ret_1","ret_5","ret_15","ret_60","vol_30","rsi_14",
            "macd","macd_signal","macd_hist","atr_14","vol_z","Original_Close","Close","High","Low","Open","Volume"
        }]
        df = df[keep_cols]
    else:
        # Keep original features (Open/High/Low/Close/Volume) as-is + Original_Close
        pass

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
    std = train_df[feature_cols].std().replace(0, 1e-7)

    def apply_norm(x):
        out = x.copy()
        out[feature_cols] = (x[feature_cols] - mean) / (std + 1e-7)
        out["Original_Close"] = x["Original_Close"]  # do not normalize price used for PnL
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
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
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
        if self.tree[left] == 0.0 and self.tree[right] == 0.0:
            while left < len(self.tree):
                idx = left
                left = 2 * idx + 1
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return max(self.tree[0], 1e-8)

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        if data_idx < 0 or data_idx >= self.capacity:
            data_idx = (self.write - 1) % max(self.n_entries, 1)
            idx = data_idx + self.capacity - 1
        return (idx, self.tree[idx], self.data[data_idx])

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=cfg.PER_ALPHA):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.e = cfg.PER_PRIORITY_EPS
        self.max_priority = 1.0

    def push(self, *args):
        self.tree.add(self.max_priority, args)

    def sample(self, batch_size, beta=cfg.PER_BETA_START):
        batch, idxs, priorities = [], [], []
        total_p = self.tree.total()
        segment = total_p / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            p = max(p, 1e-8)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = np.array(priorities, dtype=np.float64) / max(self.tree.total(), 1e-8)
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= (is_weight.max() + 1e-8)
        return batch, idxs, torch.FloatTensor(is_weight).to(device)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            priority = float(np.clip((priority + self.e) ** self.alpha, 1e-8, cfg.PER_PRIORITY_MAX))
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
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
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
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
        # We now have 6 portfolio features appended at the end of each window row
        market_feature_dim = state_dim - 6
        self.attention = SharedSelfAttention(market_feature_dim, cfg.ATTENTION_DIM, cfg.ATTENTION_HEADS)
        self.fc1 = nn.Linear(cfg.ATTENTION_DIM + 6, cfg.FC_UNITS_1)
        self.value_stream = nn.Sequential(
            NoisyLinear(cfg.FC_UNITS_1, cfg.FC_UNITS_2), nn.ReLU(),
            NoisyLinear(cfg.FC_UNITS_2, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(cfg.FC_UNITS_1, cfg.FC_UNITS_2), nn.ReLU(),
            NoisyLinear(cfg.FC_UNITS_2, action_dim)
        )

    def forward(self, state):
        # state: [B, W, F]; last 6 columns are portfolio features copied across window; take the last row
        market_data, portfolio_state = state[:, :, :-6], state[:, -1, -6:]
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
# 3. DQN AGENT WITH N-STEP, SOFT TARGET, EMA
# -------------------------------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.beta, self.learn_step_counter = cfg.PER_BETA_START, 0
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.ema_net = DuelingDQN(state_dim, action_dim).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.ema_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.ema_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.LEARNING_RATE)
        self.memory = PrioritizedReplayMemory(cfg.MEMORY_CAPACITY)

        # N-step buffer for decision-level transitions
        self.nstep_buffer = deque(maxlen=cfg.N_STEP)

    def _soft_update(self, tau=None):
        if tau is None:
            tau = cfg.SOFT_TARGET_TAU
        with torch.no_grad():
            for tp, sp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * sp.data)

    def _update_ema(self):
        with torch.no_grad():
            for ep, sp in zip(self.ema_net.parameters(), self.policy_net.parameters()):
                ep.data.mul_(cfg.EMA_MOMENTUM).add_((1.0 - cfg.EMA_MOMENTUM) * sp.data)

    def _action_mask_from_state(self, state_np):
        # state_np: [window, features]; last 6 are [credit_ratio, holdings_ratio, unrealized_pnl_ratio_c, time, credit_norm, holdings_value_norm]
        last = state_np[-1]
        credit_ratio = float(last[-6])
        holdings_ratio = float(last[-5])
        # Conservative mask on ratios so it's stable independent of absolute PV
        mask = np.ones(self.action_dim, dtype=bool)
        # Actions: 0:HOLD, 1:BUY_25, 2:BUY_50, 3:SELL_25, 4:SELL_50, 5:SELL_100
        if credit_ratio < 0.01:
            mask[1] = False
            mask[2] = False
        if holdings_ratio < 0.01:
            mask[3] = False
            mask[4] = False
            mask[5] = False
        return mask

    def act(self, state, is_eval=False):
        state_np = np.array(state, dtype=np.float32)
        mask = self._action_mask_from_state(state_np)
        valid_actions = np.where(mask)[0].tolist()
        if len(valid_actions) == 0:
            return 0  # HOLD

        net = self.ema_net if is_eval else self.policy_net
        if is_eval:
            net.eval()
        else:
            net.train()
            self.policy_net.reset_noise()

        with torch.no_grad():
            s = torch.FloatTensor(state_np).unsqueeze(0).to(device)
            q_values = net(s).squeeze(0).cpu().numpy()

        q_values_masked = q_values.copy()
        q_values_masked[~mask] = -1e9
        return int(np.argmax(q_values_masked))

    def store_transition(self, state, action, reward, next_state, done):
        # Push decision-level transition into n-step buffer
        self.nstep_buffer.append((state, action, reward, next_state, done))
        if len(self.nstep_buffer) < cfg.N_STEP:
            return None
        # Build n-step transition from buffer head
        R, s0, a0, ns, d = 0.0, None, None, None, False
        for i, (s, a, r, n_s, dn) in enumerate(self.nstep_buffer):
            if i == 0:
                s0, a0 = s, a
            R += (cfg.GAMMA ** i) * r
            ns = n_s
            d = dn
            if dn:
                break
        self.memory.push(s0, a0, R, ns, d)
        return (s0, a0, R, ns, d)

    def finish_episode_flush(self):
        # Flush remaining in n-step buffer at end of episode
        while len(self.nstep_buffer) > 0:
            R, s0, a0, ns, d = 0.0, None, None, None, False
            for i, (s, a, r, n_s, dn) in enumerate(self.nstep_buffer):
                if i == 0:
                    s0, a0 = s, a
                R += (cfg.GAMMA ** i) * r
                ns = n_s
                d = dn
                if dn:
                    break
            self.memory.push(s0, a0, R, ns, d)
            self.nstep_buffer.popleft()

    def learn(self):
        if len(self.memory) < cfg.LEARN_START_MEMORY:
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
        dones_tensor = torch.tensor(dones, device=device, dtype=torch.float32)

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

        targets = reward_batch + (cfg.GAMMA ** cfg.N_STEP) * (1.0 - dones_tensor) * next_q_values
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        td_errors = (targets - q_values).detach().abs().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        loss = (is_weights * F.smooth_l1_loss(q_values, targets, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self._soft_update()
        self._update_ema()
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def save_model(self, path):
        to_save = {
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "ema": self.ema_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(to_save, path)
        print(f"Model saved to {path}")

    def load_model(self, path, strict=True):
        ckpt = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(ckpt["policy"], strict=strict)
        self.target_net.load_state_dict(ckpt.get("target", ckpt["policy"]), strict=False)
        self.ema_net.load_state_dict(ckpt.get("ema", ckpt["policy"]), strict=False)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Model loaded from {path}")

# -------------------------------------------------
# 4. TRADING ENVIRONMENT
# -------------------------------------------------
class TradingEnvironment:
    # Actions: 0:HOLD, 1:BUY_25, 2:BUY_50, 3:SELL_25, 4:SELL_50, 5:SELL_100
    def __init__(self, data, initial_credit=cfg.INITIAL_CREDIT, window_size=cfg.WINDOW_SIZE,
                 decision_frequency_minutes=1, force_liquidation_on_done=True):
        self.data = data
        self.normalized_data = data.drop(columns=["Original_Close"])
        self.initial_credit = initial_credit
        self.window_size = window_size
        # We append 6 portfolio features at the end
        self.n_features = self.normalized_data.shape[1] + 6
        self.action_space = [0, 1, 2, 3, 4, 5]
        self.n_actions = len(self.action_space)
        self.decision_frequency_minutes = max(int(decision_frequency_minutes), 1)
        self.cooldown_ticks = max(1, math.ceil(cfg.ACTION_COOLDOWN_MINUTES / self.decision_frequency_minutes))
        self.force_liquidation_on_done = force_liquidation_on_done

    def set_force_liquidation(self, force_flag: bool):
        self.force_liquidation_on_done = bool(force_flag)

    def reset(self, episode_start_index=0, initial_credit=None, holdings=0.0, average_buy_price=0.0):
        self.credit = initial_credit if initial_credit is not None else self.initial_credit
        self.start_credit = self.credit
        self.holdings = holdings
        self.average_buy_price = average_buy_price if holdings > 0 else 0.0
        self.current_step = episode_start_index + self.window_size
        self.trades = []
        self.steps_in_position = 0
        self.cooldown = 0  # in decision ticks
        initial_price = self.data["Original_Close"].iloc[self.current_step - 1]
        self.max_portfolio_value = self.credit + self.holdings * initial_price
        self.prev_drawdown = 0.0
        # fee/turnover trackers
        self.total_fees = 0.0
        self.total_buy_notional = 0.0
        self.total_sell_notional = 0.0
        # Start PV baseline for normalization
        self.start_pv = self.credit + self.holdings * initial_price
        if self.start_pv <= 0:
            self.start_pv = max(self.start_credit, 1e-8)
        return self._get_state()

    def _get_state(self):
        start, end = self.current_step - self.window_size, self.current_step
        market_data = self.normalized_data.iloc[start:end].values
        current_price = self.data["Original_Close"].iloc[self.current_step]
        portfolio_value = self.credit + self.holdings * current_price
        credit_ratio = self.credit / portfolio_value if portfolio_value > 0 else 1.0
        holdings_ratio = (self.holdings * current_price) / portfolio_value if portfolio_value > 0 else 0.0
        unrealized_pnl_ratio = (current_price - self.average_buy_price) / self.average_buy_price if self.average_buy_price > 0 else 0.0
        # Clip unrealized pnl for stability
        unrealized_pnl_ratio = float(np.clip(unrealized_pnl_ratio, -cfg.CLIP_UNREALIZED_PNL, cfg.CLIP_UNREALIZED_PNL))
        time_in_pos_norm = math.log(self.steps_in_position + 1) / 5.0
        # Absolute-normalized (by start PV) portfolio scalars for more stable masks (not used directly in mask here)
        credit_norm = self.credit / max(self.start_pv, 1e-8)
        holdings_value_norm = (self.holdings * current_price) / max(self.start_pv, 1e-8)
        portfolio_state = np.array([[credit_ratio, holdings_ratio, unrealized_pnl_ratio, time_in_pos_norm, credit_norm, holdings_value_norm]] * self.window_size)
        return np.concatenate([market_data, portfolio_state], axis=1)

    def step(self, action_idx, decision_tick=False):
        # Cooldown only on decision ticks
        if decision_tick and self.cooldown > 0:
            action_idx = 0  # HOLD
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
        elif action == 3:
            sell_fraction = 0.25
        elif action == 4:
            sell_fraction = 0.50
        elif action == 5:
            sell_fraction = 1.00

        pv_before = self.credit + self.holdings * current_price

        buy_notional = 0.0
        sell_notional = 0.0
        fees_step = 0.0

        # Execute Buy with strict exposure cap
        if buy_fraction > 0:
            portfolio_value = self.credit + self.holdings * current_price
            h_val = self.holdings * current_price
            c_val = self.credit
            f = cfg.FEE
            Emax = cfg.MAX_EXPOSURE

            # desired investment
            desired_I = self.credit * buy_fraction

            # maximum investment to keep final exposure <= Emax:
            # I <= (Emax*c + (Emax - 1)*h) / (1 - f + Emax*f)
            denom = (1.0 - f + Emax * f)
            numer = Emax * c_val + (Emax - 1.0) * h_val
            I_max = max(0.0, numer / max(denom, 1e-8))

            invest_I = min(desired_I, I_max)
            if invest_I > cfg.MIN_TRADE_CREDIT_BUY and invest_I > 0.0:
                buy_fee = invest_I * f
                buy_amount_asset = (invest_I * (1.0 - f)) / current_price
                # Update weighted average buy price with proper cost basis
                total_cost_excl_fees = (self.average_buy_price * self.holdings) + invest_I
                self.holdings += buy_amount_asset
                self.credit -= invest_I
                if self.holdings > 0:
                    self.average_buy_price = total_cost_excl_fees / self.holdings
                else:
                    self.average_buy_price = 0.0
                self.trades.append({"step": self.current_step, "type": "buy", "price": current_price, "amount": buy_amount_asset})
                trade_executed = True
                buy_notional = invest_I
                fees_step += buy_fee

        # Execute Sell
        elif sell_fraction > 0 and self.holdings > 0:
            sell_amount_asset = min(self.holdings, self.holdings * sell_fraction)
            if sell_amount_asset * current_price > cfg.MIN_TRADE_HOLDINGS_SELL:
                gross_value = sell_amount_asset * current_price
                sell_fee = gross_value * cfg.FEE
                sell_value = gross_value - sell_fee
                self.credit += sell_value
                self.holdings -= sell_amount_asset
                realized_pnl = (current_price - self.average_buy_price) * sell_amount_asset
                if self.holdings < 1e-9:
                    self.holdings = 0.0
                    self.average_buy_price = 0.0
                self.trades.append({"step": self.current_step, "type": "sell", "price": current_price, "amount": sell_amount_asset})
                trade_executed = True
                sell_notional = gross_value
                fees_step += sell_fee

        # Post-trade management
        if trade_executed and decision_tick:
            self.cooldown = self.cooldown_ticks
            self.steps_in_position = 0
        elif self.holdings > 0:
            self.steps_in_position += 1

        # Advance time
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        pv_after = self.credit + self.holdings * next_price
        if pv_after > self.max_portfolio_value:
            self.max_portfolio_value = pv_after

        # Optional forced liquidation on done
        if done and self.holdings > 0 and self.force_liquidation_on_done:
            gross_liq = self.holdings * next_price
            liq_fee = gross_liq * cfg.FEE
            liquidate_value = gross_liq - liq_fee
            realized_pnl += (next_price - self.average_buy_price) * self.holdings
            self.credit += liquidate_value
            self.trades.append({"step": self.current_step, "type": "sell", "price": next_price, "amount": self.holdings})
            pv_after = self.credit
            fees_step += liq_fee
            sell_notional += gross_liq
            self.holdings = 0.0
            self.average_buy_price = 0.0

        # Track fees and turnover
        self.total_fees += fees_step
        self.total_buy_notional += buy_notional
        self.total_sell_notional += sell_notional

        # Reward calculation (log PV return centric; scaled and clipped)
        reward, comps = self._calculate_reward(realized_pnl, pv_before, pv_after, trade_executed, fees_step)

        next_state = self._get_state() if not done else None

        info = {
            "portfolio_value": pv_after,
            "credit": self.credit,
            "holdings": self.holdings,
            "average_buy_price": self.average_buy_price,
            "trades": self.trades,
            "reward_components": comps,
            "fees_paid_step": fees_step,
            "buy_notional_step": buy_notional,
            "sell_notional_step": sell_notional
        }
        return next_state, reward, done, info

    def _calculate_reward(self, realized_pnl, pv_before, pv_after, trade_executed, fees_step):
        # Core: scaled log return on PV (encourages multiplicative growth)
        log_ret = math.log(max(pv_after, 1e-8) / max(pv_before, 1e-8))
        pv_component = cfg.PV_LOGRET_SCALE * log_ret

        # Realized PnL shaping (normalize by start PV)
        realized_component = cfg.REWARD_REALIZED * (realized_pnl / max(self.start_pv, 1e-8)) if realized_pnl != 0 else 0.0
        realized_component = float(np.clip(realized_component, -cfg.REALIZED_CLIP, cfg.REALIZED_CLIP))

        # Trade penalty
        trade_pen = -cfg.TRADE_PENALTY if trade_executed else 0.0

        # Time in position penalty (off by default)
        time_pen = -cfg.TIME_PENALTY * self.steps_in_position if self.holdings > 0 else 0.0

        # Drawdown increase penalty
        drawdown = 0.0
        drawdown_increase = 0.0
        if self.max_portfolio_value > 0:
            drawdown = max(0.0, (self.max_portfolio_value - pv_after) / self.max_portfolio_value)
            drawdown_increase = max(0.0, drawdown - self.prev_drawdown)
        dd_pen = -cfg.DRAWDOWN_PENALTY * drawdown_increase
        self.prev_drawdown = drawdown

        # Explicit fees penalty (normalized by start PV) and clipped
        fees_pen = -cfg.FEES_PENALTY_COEF * (fees_step / max(self.start_pv, 1e-8)) if fees_step > 0 else 0.0
        fees_pen = float(np.clip(fees_pen, -cfg.FEES_CLIP, cfg.FEES_CLIP))

        comps = {
            "pv_logret": pv_component,
            "realized": realized_component,
            "trade_pen": trade_pen,
            "time_pen": time_pen,
            "dd_pen": dd_pen,
            "fees_pen": fees_pen
        }
        reward = float(np.clip(sum(comps.values()), -cfg.REWARD_CLIP, cfg.REWARD_CLIP))
        return reward, comps

# -------------------------------------------------
# 5. PLOTTING
# -------------------------------------------------
def plot_results(df, episode, portfolio_history, credit_history, holdings_history, trades, plot_title_prefix="", segment_boundaries=None, save_name=None):
    os.makedirs("plots", exist_ok=True)
    aligned = len(portfolio_history) == len(df)
    if aligned:
        plot_df = df.copy()
        plot_df["portfolio_value"] = portfolio_history
        plot_df["credit"] = credit_history
        plot_df["holdings_value"] = holdings_history
    else:
        start_index = len(df) - len(portfolio_history)
        plot_df = df.iloc[start_index:].copy()
        plot_df["portfolio_value"] = portfolio_history
        plot_df["credit"] = credit_history
        plot_df["holdings_value"] = holdings_history

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=(f"{plot_title_prefix} Price and Trades", "Portfolio Value", "Credit", "Holdings Value")
    )

    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["Original_Close"], mode="lines", name="Price", line=dict(color="lightgrey")),
        row=1, col=1
    )

    if trades:
        buy_trades = [t for t in trades if t["type"] == "buy"]
        sell_trades = [t for t in trades if t["type"] == "sell"]
        if buy_trades:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[min(t["step"], len(df)-1)] for t in buy_trades],
                    y=[t["price"] for t in buy_trades],
                    mode="markers",
                    marker=dict(color="green", symbol="triangle-up", size=8),
                    name="Buy"
                ),
                row=1, col=1
            )
        if sell_trades:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[min(t["step"], len(df)-1)] for t in sell_trades],
                    y=[t["price"] for t in sell_trades],
                    mode="markers",
                    marker=dict(color="red", symbol="triangle-down", size=8),
                    name="Sell"
                ),
                row=1, col=1
            )

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["portfolio_value"], mode="lines", name="PV"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["credit"], mode="lines", name="Credit"), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["holdings_value"], mode="lines", name="Holdings"), row=4, col=1)

    if segment_boundaries:
        for boundary in segment_boundaries:
            if 0 <= boundary < len(df.index):
                fig.add_vline(x=df.index[boundary], line_width=1, line_dash="dash", line_color="black", opacity=0.4)

    fig.update_layout(height=1000, title_text=f"{plot_title_prefix} Results (Episode {episode})", showlegend=False)

    fig.show()
    if save_name is None:
        save_name = f"{plot_title_prefix.lower()}_episode_{episode}.html"
    fig.write_html(os.path.join("plots", save_name), auto_open=False)

# -------------------------------------------------
# 6. TRAINING AND EVALUATION LOGIC
# -------------------------------------------------
def run_episode(env, agent, data, is_eval=False, initial_credit=None, initial_holdings=0.0, initial_avg_buy_price=0.0, decision_frequency_minutes=15, force_liq_on_done=True):
    env.decision_frequency_minutes = max(1, int(decision_frequency_minutes))
    env.cooldown_ticks = max(1, math.ceil(cfg.ACTION_COOLDOWN_MINUTES / env.decision_frequency_minutes))
    env.set_force_liquidation(force_liq_on_done)

    state = env.reset(initial_credit=initial_credit, holdings=initial_holdings, average_buy_price=initial_avg_buy_price)
    done = False
    portfolio_values, credits, holdings_values = [], [], []
    indices_history = []
    pbar_desc = "VALIDATING" if is_eval else "TRAINING"
    total_steps = len(data) - env.window_size - 1
    pbar = tqdm(total=total_steps, desc=pbar_desc, leave=False)

    action_counts = defaultdict(int)
    comp_sums = {"pv_logret": 0.0, "realized": 0.0, "trade_pen": 0.0, "time_pen": 0.0, "dd_pen": 0.0, "fees_pen": 0.0}
    fees_total = 0.0
    buy_notional_total = 0.0
    sell_notional_total = 0.0

    # Transition aggregation across base steps (frame-skip between decisions)
    last_decision_state = None
    last_decision_action = None
    aggregated_reward = 0.0

    decision_period = max(1, int(decision_frequency_minutes))

    for step in range(total_steps):
        decision_tick = (step % decision_period == 0)

        if decision_tick:
            # Push previous aggregated decision transition (n-step builder inside agent)
            if not is_eval and last_decision_state is not None:
                # Store the just-finished (s,a,R,next_s,done)
                pushed = agent.store_transition(last_decision_state, last_decision_action, aggregated_reward, state, False)
                if pushed is not None:
                    # After pushing, run K gradient updates
                    for _ in range(cfg.LEARN_UPDATES_PER_PUSH):
                        agent.learn()
            # Choose new action at decision tick
            action = agent.act(state, is_eval)
            action_counts[action] += 1
            last_decision_state = state
            last_decision_action = action
            aggregated_reward = 0.0
        else:
            action = 0  # HOLD on non-decision ticks
            action_counts[action] += 1

        next_state, reward, done, info = env.step(action, decision_tick=decision_tick)
        aggregated_reward += reward

        state = next_state

        portfolio_values.append(info["portfolio_value"])
        credits.append(info["credit"])
        current_index = min(env.current_step, len(data) - 1)
        current_price = data["Original_Close"].iloc[current_index]
        holdings_values.append(info["holdings"] * current_price)
        indices_history.append(current_index)

        comps = info.get("reward_components", None)
        if comps is not None:
            for k in comp_sums.keys():
                comp_sums[k] += comps.get(k, 0.0)
        fees_total += info.get("fees_paid_step", 0.0)
        buy_notional_total += info.get("buy_notional_step", 0.0)
        sell_notional_total += info.get("sell_notional_step", 0.0)

        pbar.update(1)
        if done:
            break
    pbar.close()

    # Push the final aggregated transition at terminal
    if not is_eval and last_decision_state is not None:
        agent.store_transition(last_decision_state, last_decision_action, aggregated_reward, None, True)
        agent.finish_episode_flush()
        for _ in range(cfg.LEARN_UPDATES_PER_PUSH):
            agent.learn()

    final_pv = portfolio_values[-1] if portfolio_values else initial_credit
    final_credit = credits[-1] if credits else initial_credit
    final_holdings = info["holdings"] if info else 0.0
    final_avg_buy_price = info.get("average_buy_price", 0.0) if info else 0.0

    return (
        portfolio_values, credits, holdings_values, indices_history, env.trades,
        final_pv, final_credit, final_holdings, final_avg_buy_price, dict(action_counts),
        comp_sums, fees_total, buy_notional_total, sell_notional_total
    )

def validate_in_segments(full_val_data, agent, window_size, initial_credit, decision_frequency_minutes):
    n_total = len(full_val_data)
    segment_starts = list(range(0, n_total - window_size - 1, cfg.VAL_SEGMENT_MINUTES))

    all_portfolio = [np.nan] * n_total
    all_credit = [np.nan] * n_total
    all_holdings = [np.nan] * n_total
    all_trades = []
    segment_metrics = []

    current_credit, current_holdings, current_avg_buy_price = initial_credit, 0.0, 0.0

    for seg_idx, start in enumerate(segment_starts, 1):
        end = min(start + cfg.VAL_SEGMENT_MINUTES, n_total)
        if end - start <= window_size:
            continue

        is_last_segment = (seg_idx == len(segment_starts))
        force_liq = is_last_segment

        segment_data = full_val_data.iloc[start:end].copy().reset_index(drop=True)
        val_env = TradingEnvironment(
            segment_data, initial_credit=current_credit, window_size=window_size,
            decision_frequency_minutes=decision_frequency_minutes,
            force_liquidation_on_done=force_liq
        )

        (
            pv_hist, credit_hist, hold_hist, idx_hist, trades,
            final_pv, final_credit, final_holdings, final_avg_buy_price, action_counts,
            comp_sums, fees_total, buy_notional_total, sell_notional_total
        ) = run_episode(
            val_env, agent, segment_data, is_eval=True,
            initial_credit=current_credit, initial_holdings=current_holdings, initial_avg_buy_price=current_avg_buy_price,
            decision_frequency_minutes=decision_frequency_minutes, force_liq_on_done=force_liq
        )

        # Fill aligned histories
        for k in range(len(idx_hist)):
            global_idx = start + idx_hist[k]
            if 0 <= global_idx < n_total:
                all_portfolio[global_idx] = pv_hist[k]
                all_credit[global_idx] = credit_hist[k]
                all_holdings[global_idx] = hold_hist[k]

        # Shift trade steps to full validation index
        for tr in trades:
            tr["step"] += start
        all_trades.extend(trades)

        # Baselines for metrics at segment start (continuous evaluation)
        start_price = segment_data["Original_Close"].iloc[val_env.window_size]
        initial_pv_seg = current_credit + current_holdings * start_price
        initial_credit_seg = current_credit

        seg_return = ((final_pv - initial_pv_seg) / initial_pv_seg * 100) if initial_pv_seg > 0 else 0.0
        credit_growth = ((final_credit - initial_credit_seg) / initial_credit_seg * 100) if initial_credit_seg > 0 else 0.0

        seg_logret_pct = (math.exp(comp_sums["pv_logret"] / max(cfg.PV_LOGRET_SCALE, 1e-8)) - 1.0) * 100.0

        buys = len([t for t in trades if t["type"] == "buy"])
        sells = len([t for t in trades if t["type"] == "sell"])

        segment_metrics.append({
            "seg": seg_idx,
            "start": full_val_data.index[start],
            "end": full_val_data.index[end - 1],
            "final_pv": final_pv,
            "ret": seg_return,
            "logret_pct": seg_logret_pct,
            "credit_growth": credit_growth,
            "buys": buys,
            "sells": sells,
            "final_credit": final_credit,
            "final_holdings": final_holdings,
            "final_avg_buy_price": final_avg_buy_price,
            "reward_components": comp_sums,
            "fees_total": fees_total,
            "turnover_buy": buy_notional_total,
            "turnover_sell": sell_notional_total
        })

        # Carry forward credit, holdings, and average buy price
        current_credit, current_holdings, current_avg_buy_price = final_credit, final_holdings, final_avg_buy_price

    segment_boundaries = []
    for s in segment_starts[1:]:
        boundary = s + window_size
        if boundary < n_total:
            segment_boundaries.append(boundary)

    return all_portfolio, all_credit, all_holdings, all_trades, segment_metrics, segment_boundaries

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
    best_val_pv_return = -1e9
    patience_counter = 0

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

        # Single fixed cadence for stability
        train_decision_freq = cfg.TRAIN_DECISION_FREQUENCIES[0]

        train_env = TradingEnvironment(episode_data, window_size=cfg.WINDOW_SIZE, decision_frequency_minutes=train_decision_freq)
        (
            _pv, _cred, _hold, _idx, _trades,
            _fpv, _fcred, _fhold, _favgbp, action_counts, _c_sums, _fees, _buy_not, _sell_not
        ) = run_episode(
            train_env, agent, episode_data, is_eval=False, initial_credit=cfg.INITIAL_CREDIT,
            decision_frequency_minutes=train_decision_freq, force_liq_on_done=True
        )

        print("\n-> Validation phase…")
        val_freq = cfg.VAL_DECISION_FREQUENCY
        pv_hist, credit_hist, hold_hist, trades, seg_metrics, seg_boundaries = validate_in_segments(
            val_data, agent, window_size=cfg.WINDOW_SIZE, initial_credit=cfg.INITIAL_CREDIT, decision_frequency_minutes=val_freq
        )

        print("\nSegment-by-segment results:")
        for m in seg_metrics:
            rc = m["reward_components"]
            print(
                f"  Seg {m['seg']:02d} [{m['start']} → {m['end']}]  "
                f"PV: ${m['final_pv']:.2f} (Ret: {m['ret']:.2f}%, LogRet: {m['logret_pct']:.2f}%) | "
                f"Credit: ${m['final_credit']:.2f} (Growth: {m['credit_growth']:.2f}%) | "
                f"Buys: {m['buys']}, Sells: {m['sells']} | "
                f"Fees: ${m['fees_total']:.2f}, Turnover(B/S): ${m['turnover_buy']:.2f}/${m['turnover_sell']:.2f} | "
                f"Rewards Σ: pv_logret={rc['pv_logret']:.4f}, realized={rc['realized']:.4f}, trade_pen={rc['trade_pen']:.4f}, "
                f"dd_pen={rc['dd_pen']:.4f}, fees_pen={rc['fees_pen']:.4f}, time_pen={rc['time_pen']:.4f}"
            )

        if seg_metrics:
            total_final_credit = seg_metrics[-1]["final_credit"]
            total_initial_credit = cfg.INITIAL_CREDIT
            cumulative_val_return = (total_final_credit - total_initial_credit) / total_initial_credit * 100.0
            mean_seg_logret_pct = float(np.mean([m["logret_pct"] for m in seg_metrics]))

            improved = cumulative_val_return > best_val_pv_return
            if improved:
                best_val_pv_return = cumulative_val_return
                patience_counter = 0
                agent.save_model(f"saved_models/best_model.pth")
                print(f"New best validation cumulative PV return: {best_val_pv_return:.2f}%. Model saved.")
            else:
                patience_counter += 1
                print(f"No improvement. Patience {patience_counter}/{cfg.PATIENCE}.")

            total_buys = sum(m["buys"] for m in seg_metrics)
            total_sells = sum(m["sells"] for m in seg_metrics)

            print("\nAggregated validation:")
            print(f"  Cumulative PV return: {cumulative_val_return:.2f}%")
            print(f"  Mean segment PV log-return: {mean_seg_logret_pct:.2f}%")
            print(f"  Total buys: {total_buys}, Total sells: {total_sells}")

            if any(np.isfinite(v) for v in pv_hist):
                plot_results(
                    val_data, e + 1, pv_hist, credit_hist, hold_hist, trades,
                    "Validation", segment_boundaries=seg_boundaries, save_name=f"val_episode_{e+1}.html"
                )

            if patience_counter >= cfg.PATIENCE:
                print("Early stopping due to no improvement.")
                break

    print("\n=== FINAL TEST ON UNSEEN DATA ===")
    try:
        agent.load_model("saved_models/best_model.pth", strict=False)
    except Exception as e:
        print("No saved model found or failed to load; using current policy.")
    test_env = TradingEnvironment(test_data.reset_index(drop=True), decision_frequency_minutes=cfg.VAL_DECISION_FREQUENCY)
    (
        pv, cred, hold, idx_hist, trades, final_pv, final_credit, final_holdings, final_avg_buy_price,
        act_counts, comp_sums, fees_total, buy_not, sell_not
    ) = run_episode(
        test_env, agent, test_data.reset_index(drop=True), is_eval=True, initial_credit=cfg.INITIAL_CREDIT,
        decision_frequency_minutes=cfg.VAL_DECISION_FREQUENCY, force_liq_on_done=True
    )

    if pv:
        n_total_test = len(test_data)
        full_pv = [np.nan] * n_total_test
        full_cred = [np.nan] * n_total_test
        full_hold = [np.nan] * n_total_test
        for k in range(len(idx_hist)):
            gi = idx_hist[k]
            if 0 <= gi < n_total_test:
                full_pv[gi] = pv[k]
                full_cred[gi] = cred[k]
                full_hold[gi] = hold[k]

        pv_return = (final_pv - cfg.INITIAL_CREDIT) / cfg.INITIAL_CREDIT * 100
        credit_return = (final_credit - cfg.INITIAL_CREDIT) / cfg.INITIAL_CREDIT * 100
        buys, sells = len([t for t in trades if t["type"] == "buy"]), len([t for t in trades if t["type"] == "sell"])
        print(f"Final PV: ${final_pv:,.2f} (Return: {pv_return:.2f}%)")
        print(f"Final Credit: ${final_credit:,.2f} (Return: {credit_return:.2f}%)")
        print(f"Total Buys: {buys}, Total Sells: {sells}")
        print(f"Fees paid: ${fees_total:.2f}, Turnover(B/S): ${buy_not:.2f}/{sell_not:.2f}")
        print(f"Reward components Σ (test): {json.dumps({k: round(v, 6) for k, v in comp_sums.items()})}")
        plot_results(test_data, "Final", full_pv, full_cred, full_hold, trades, "Final Test", segment_boundaries=None, save_name="final_test.html")

if __name__ == "__main__":
    main()
