
Look, I have this code, which Im trying to optimize to make it make good trades. Please help me fix it. I dont understand why it doesnt try to make any trades. What I want is to make the capital/credit grow in a cumulative manner. 

```
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
from collections import deque, namedtuple
import math
import os
import json

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
    TARGET_UPDATE_FREQ = 10
    TRAINING_WARMUP_STEPS = 5000

    ## --- FIX 1: ADD PARAMETERS FOR TRADING FRICTION ---
    DECISION_FREQUENCY = 5  # Agent makes a decision only every 5 minutes.
    ACTION_COOLDOWN = 5     # Agent must wait 5 minutes after a trade before making another.

    # --- Model Architecture ---
    ATTENTION_DIM = 64
    ATTENTION_HEADS = 4
    FC_UNITS_1 = 256
    FC_UNITS_2 = 128

    # --- Environment ---
    INITIAL_CREDIT = 100.0
    WINDOW_SIZE = 180
    FEE = 0.001
    MIN_TRADE_CREDIT_BUY = 1.0
    MIN_TRADE_HOLDINGS_SELL = 0.01

    # --- Prioritized Experience Replay (PER) ---
    MEMORY_CAPACITY = 100000
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_END = 1.0
    PER_BETA_ANNEAL_STEPS = 100000

    # --- Validation ---
    VAL_SEGMENT_MINUTES = 2 * 24 * 60

cfg = Config()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -------------------------------------------------
# 1. DATA LOADING AND PRE-PROCESSING
# -------------------------------------------------
def get_data():
    print("Downloading and preprocessing data…")
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
    original_close = df["Close"].copy()
    for col in df.columns:
        if col != 'Original_Close':
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-7)
    df["Original_Close"] = original_close
    df.dropna(inplace=True)
    print("Data loaded and preprocessed successfully.")
    print(f"Data shape: {df.shape}")
    return df

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
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

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
        is_weight /= is_weight.max()
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
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
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
        self.beta, self.learn_step_counter, self.total_steps = cfg.PER_BETA_START, 0, 0
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.LEARNING_RATE)
        self.memory = PrioritizedReplayMemory(cfg.MEMORY_CAPACITY)

    def act(self, state, is_eval=False):
        self.total_steps += 1
        if not is_eval and self.total_steps < cfg.TRAINING_WARMUP_STEPS:
            return random.randrange(self.action_dim)
        self.policy_net.eval() if is_eval else self.policy_net.train()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def learn(self):
        if len(self.memory) < cfg.BATCH_SIZE:
            return
        self.policy_net.train()
        self.target_net.eval()
        self.beta = min(cfg.PER_BETA_END, cfg.PER_BETA_START + self.learn_step_counter * (cfg.PER_BETA_END - cfg.PER_BETA_START) / cfg.PER_BETA_ANNEAL_STEPS)
        self.learn_step_counter += 1
        transitions, indices, is_weights = self.memory.sample(cfg.BATCH_SIZE, self.beta)
        states, actions, rewards, next_states, dones = zip(*transitions)
        state_batch = torch.FloatTensor(np.array(states)).to(device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(rewards).to(device)
        non_final_mask = torch.tensor([s is not None for s in next_states], device=device, dtype=torch.bool)
        non_final_next_states = torch.FloatTensor(np.array([s for s in next_states if s is not None])).to(device)
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
    def __init__(self, data, initial_credit=cfg.INITIAL_CREDIT, window_size=cfg.WINDOW_SIZE):
        self.data = data
        self.normalized_data = data.drop(columns=["Original_Close"])
        self.initial_credit = initial_credit
        self.window_size = window_size
        self.n_features = self.normalized_data.shape[1] + 4
        self.action_space = [0, 1, 2, 3, 4] # 0:HOLD, 1:BUY_25, 2:BUY_50, 3:SELL_25, 4:SELL_50
        self.n_actions = len(self.action_space)

    def reset(self, episode_start_index=0, initial_credit=None, holdings=0.0):
        self.credit = initial_credit if initial_credit is not None else self.initial_credit
        self.start_credit = self.credit
        self.holdings = holdings
        self.average_buy_price = 0.0
        self.current_step = episode_start_index + self.window_size
        self.trades = []
        self.steps_in_position = 0
        self.cooldown = 0 # Cooldown counter
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

    def step(self, action_idx):
        ## --- FIX 2: APPLY ACTION COOLDOWN ---
        if self.cooldown > 0:
            action_idx = 0 # Force HOLD action
            self.cooldown -= 1

        action = self.action_space[action_idx]
        current_price = self.data["Original_Close"].iloc[self.current_step]
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

        if buy_fraction > 0:
            investment = self.credit * buy_fraction
            if investment > cfg.MIN_TRADE_CREDIT_BUY:
                buy_amount_asset = (investment * (1 - cfg.FEE)) / current_price
                total_cost = (self.average_buy_price * self.holdings) + investment
                self.holdings += buy_amount_asset
                self.credit -= investment
                self.average_buy_price = total_cost / self.holdings if self.holdings > 0 else 0
                self.trades.append({"step": self.current_step, "type": "buy", "price": current_price, "amount": buy_amount_asset})
                trade_executed = True
        elif sell_fraction > 0:
            sell_amount_asset = self.holdings * sell_fraction
            if sell_amount_asset * current_price > cfg.MIN_TRADE_HOLDINGS_SELL:
                sell_value = sell_amount_asset * current_price * (1 - cfg.FEE)
                self.credit += sell_value
                self.holdings -= sell_amount_asset
                realized_pnl = (current_price - self.average_buy_price) * sell_amount_asset
                if self.holdings < 1e-6:
                    self.average_buy_price = 0
                self.trades.append({"step": self.current_step, "type": "sell", "price": current_price, "amount": sell_amount_asset})
                trade_executed = True
        
        if trade_executed:
            self.cooldown = cfg.ACTION_COOLDOWN
            self.steps_in_position = 0
        elif self.holdings > 0:
            self.steps_in_position += 1

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        next_state = self._get_state() if not done else None
        next_price = self.data["Original_Close"].iloc[self.current_step] if not done else current_price
        
        reward = self._calculate_reward(realized_pnl, current_price, next_price, trade_executed)
        
        portfolio_value_after = self.credit + self.holdings * next_price
        if portfolio_value_after > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value_after
        if done:
            self.credit += self.holdings * current_price * (1 - cfg.FEE)
            self.holdings = 0
            portfolio_value_after = self.credit
        return next_state, reward, done, {"portfolio_value": portfolio_value_after, "credit": self.credit, "holdings": self.holdings, "trades": self.trades}

    def _calculate_reward(self, realized_pnl, price_now, price_next, trade_executed):
        reward = 0.0
        
        # 1. Strong reward for REALIZED profits, scaled by initial capital.
        if realized_pnl != 0:
            reward += 20.0 * (realized_pnl / self.start_credit)

        ## --- FIX 3: ADD EXPLICIT TRANSACTION COST PENALTY ---
        if trade_executed:
            reward -= 0.5 # A fixed penalty for the cost of trading.

        # 2. Penalty for UNREALIZED losses (Opportunity Cost).
        if self.holdings > 0:
            price_change_ratio = (price_next - price_now) / price_now
            unrealized_pnl_change = self.holdings * price_now * price_change_ratio
            if unrealized_pnl_change < 0:
                reward += 10.0 * (unrealized_pnl_change / self.start_credit)

        # 3. Time-in-position penalty to encourage closing trades.
        if self.holdings > 0:
            reward -= 0.005 * self.steps_in_position

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
        fig.add_trace(go.Scatter(x=[df.index[t["step"]] for t in buy_trades], y=[t["price"] for t in buy_trades], mode="markers", marker=dict(color="green", symbol="triangle-up", size=8), name="Buy"), row=1, col=1)
    if sell_trades:
        fig.add_trace(go.Scatter(x=[df.index[t["step"]] for t in sell_trades], y=[t["price"] for t in sell_trades], mode="markers", marker=dict(color="red", symbol="triangle-down", size=8), name="Sell"), row=1, col=1)
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
def run_episode(env, agent, data, is_eval=False, initial_credit=None, initial_holdings=0.0):
    state = env.reset(initial_credit=initial_credit, holdings=initial_holdings)
    done = False
    portfolio_values, credits, holdings_values = [], [], []
    pbar_desc = "VALIDATING" if is_eval else "TRAINING"
    total_steps = len(data) - env.window_size - 1
    pbar = tqdm(total=total_steps, desc=pbar_desc, leave=False)

    for step in range(total_steps):
        ## --- FIX 4: INTRODUCE DECISION FREQUENCY ---
        # The agent only makes a new decision every N steps.
        if step % cfg.DECISION_FREQUENCY == 0:
            action = agent.act(state, is_eval)
        else:
            action = 0 # Force HOLD action

        next_state, reward, done, info = env.step(action)
        if not is_eval:
            agent.memory.push(state, action, reward, next_state, done)
            if agent.total_steps > cfg.TRAINING_WARMUP_STEPS:
                agent.learn()
        state = next_state
        if done:
            break
        current_price = data["Original_Close"].iloc[env.current_step]
        portfolio_values.append(info["portfolio_value"])
        credits.append(info["credit"])
        holdings_values.append(info["holdings"] * current_price)
        pbar.update(1)
    pbar.close()
    return portfolio_values, credits, holdings_values, env.trades, info["credit"], info["holdings"]

def validate_in_segments(full_val_data, agent, window_size, initial_credit):
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
        pv_hist, credit_hist, hold_hist, trades, final_credit, final_holdings = run_episode(
            val_env, agent, segment_data, is_eval=True, initial_credit=current_credit, initial_holdings=current_holdings
        )
        for tr in trades:
            tr["step"] += start
        all_trades.extend(trades)
        all_portfolio.extend(pv_hist)
        all_credit.extend(credit_hist)
        all_holdings.extend(hold_hist)
        current_credit, current_holdings = final_credit, final_holdings
        final_pv = pv_hist[-1] if pv_hist else current_credit
        initial_segment_value = credit_hist[0] if credit_hist else current_credit
        seg_return = (final_pv - initial_segment_value) / initial_segment_value * 100 if initial_segment_value > 0 else 0
        final_credit_val = credit_hist[-1] if credit_hist else current_credit
        initial_credit_val = credit_hist[0] if credit_hist else current_credit
        credit_growth = (final_credit_val - initial_credit_val) / initial_credit_val * 100 if initial_credit_val > 0 else 0
        buys = len([t for t in trades if t["type"] == "buy"])
        sells = len([t for t in trades if t["type"] == "sell"])
        segment_metrics.append({
            "seg": seg_idx, "start": full_val_data.index[start], "end": full_val_data.index[end - 1],
            "final_pv": final_pv, "ret": seg_return, "credit_growth": credit_growth,
            "buys": buys, "sells": sells, "final_credit": final_credit_val
        })
    return all_portfolio, all_credit, all_holdings, all_trades, segment_metrics, segment_starts

# -------------------------------------------------
# 7. MAIN EXECUTION
# -------------------------------------------------
def main():
    print(f"Using device: {device}")
    full_data = get_data()
    if full_data.empty or len(full_data) < cfg.WINDOW_SIZE * 2:
        print("Not enough data to run.")
        return
    train_size = int(len(full_data) * 0.7)
    val_size = int(len(full_data) * 0.15)
    train_data = full_data[:train_size]
    val_data = full_data[train_size : train_size + val_size]
    test_data = full_data[train_size + val_size :]
    val_data = val_data.reset_index()
    print(f"Training data size:   {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Test data size:       {len(test_data)}")
    dummy_env = TradingEnvironment(full_data)
    agent = DQNAgent(state_dim=dummy_env.n_features, action_dim=dummy_env.n_actions)
    os.makedirs("saved_models", exist_ok=True)
    best_val_credit_growth = -100.0
    for e in range(cfg.EPISODES):
        print(f"\n=== Episode {e + 1}/{cfg.EPISODES} ===")
        episode_minutes = int(cfg.EPISODE_LENGTH_DAYS * 24 * 60)
        max_start = len(train_data) - episode_minutes - 1
        if max_start <= 0:
            print("Training data is smaller than an episode length. Skipping.")
            continue
        start_idx = random.randint(0, max_start)
        episode_data = train_data.iloc[start_idx : start_idx + episode_minutes].copy().reset_index(drop=True)
        print(f"Training on slice {train_data.index[start_idx]} -> {train_data.index[start_idx + episode_minutes -1]}")
        train_env = TradingEnvironment(episode_data, window_size=cfg.WINDOW_SIZE)
        run_episode(train_env, agent, episode_data, is_eval=False, initial_credit=cfg.INITIAL_CREDIT)
        if (e + 1) % cfg.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
            print("Target network synchronized.")
        print("\n-> Validation phase…")
        pv_hist, credit_hist, hold_hist, trades, seg_metrics, seg_starts = validate_in_segments(
            val_data.set_index('Datetime'), agent, window_size=cfg.WINDOW_SIZE, initial_credit=cfg.INITIAL_CREDIT
        )
        print("\nSegment-by-segment results:")
        for m in seg_metrics:
            print(f"  Seg {m['seg']:02d} [{m['start']} → {m['end']}]  "
                  f"PV: ${m['final_pv']:.2f} (Ret: {m['ret']:.2f}%) | "
                  f"Credit: ${m['final_credit']:.2f} (Growth: {m['credit_growth']:.2f}%) | "
                  f"Buys: {m['buys']}, Sells: {m['sells']}")
        if seg_metrics:
            credit_growths = [m["credit_growth"] for m in seg_metrics]
            mean_growth = np.mean(credit_growths)
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
                val_data.set_index('Datetime'), e + 1, pv_hist, credit_hist, hold_hist, trades,
                "Validation", [s + cfg.WINDOW_SIZE for s in seg_starts[1:]]
            )
    print("\n=== FINAL TEST ON UNSEEN DATA ===")
    agent.load_model("saved_models/best_model.pth")
    test_env = TradingEnvironment(test_data.reset_index(drop=True))
    pv, cred, hold, trades, _, _ = run_episode(test_env, agent, test_data.reset_index(drop=True), is_eval=True, initial_credit=cfg.INITIAL_CREDIT)
    if pv:
        final_pv, final_credit = pv[-1], cred[-1]
        pv_return = (final_pv - cfg.INITIAL_CREDIT) / cfg.INITIAL_CREDIT * 100
        credit_return = (final_credit - cfg.INITIAL_CREDIT) / cfg.INITIAL_CREDIT * 100
        buys, sells = len([t for t in trades if t["type"] == "buy"]), len([t for t in trades if t["type"] == "sell"])
        print(f"Final PV: ${final_pv:,.2f} (Return: {pv_return:.2f}%)")
        print(f"Final Credit: ${final_credit:,.2f} (Return: {credit_return:.2f}%)")
        print(f"Total Buys: {buys}, Total Sells: {sells}")
        plot_results(test_data, "Final", pv, cred, hold, trades, "Final Test")

if __name__ == "__main__":
    main()
```


```
=== Episode 1/50 ===
Training on slice 2025-05-24 12:42:00+00:00 -> 2025-05-27 08:14:00+00:00
                                                                                                                                                          
-> Validation phase…
                                                                                                                                                          
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $105.88 (Ret: 111.76%) | Credit: $1.56 (Growth: -96.88%) | Buys: 6, Sells: 0
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $106.53 (Ret: 101.42%) | Credit: $1.65 (Growth: -96.88%) | Buys: 6, Sells: 0
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $105.52 (Ret: 98.32%) | Credit: $1.66 (Growth: -96.87%) | Buys: 6, Sells: 0
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $104.47 (Ret: 98.19%) | Credit: $1.65 (Growth: -96.88%) | Buys: 6, Sells: 0
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $105.15 (Ret: 101.51%) | Credit: $1.63 (Growth: -96.88%) | Buys: 6, Sells: 0
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $106.49 (Ret: 102.75%) | Credit: $1.64 (Growth: -96.88%) | Buys: 6, Sells: 0
Model saved to saved_models/best_model.pth
New best validation credit growth: -96.88%. Model saved.

Aggregated validation:
  Mean segment credit growth: -96.88%
  Total buys: 36, Total sells: 0

=== Episode 2/50 ===
Training on slice 2025-05-24 03:54:00+00:00 -> 2025-05-26 23:34:00+00:00
                                                                                                                                                          
-> Validation phase…
                                                                                                                                                          
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $105.88 (Ret: 111.76%) | Credit: $1.56 (Growth: -96.88%) | Buys: 6, Sells: 0
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $106.53 (Ret: 101.42%) | Credit: $1.65 (Growth: -96.88%) | Buys: 6, Sells: 0
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $105.52 (Ret: 98.32%) | Credit: $1.66 (Growth: -96.87%) | Buys: 6, Sells: 0
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $104.47 (Ret: 98.19%) | Credit: $1.65 (Growth: -96.88%) | Buys: 6, Sells: 0
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $105.15 (Ret: 101.51%) | Credit: $1.63 (Growth: -96.88%) | Buys: 6, Sells: 0
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $106.49 (Ret: 102.75%) | Credit: $1.64 (Growth: -96.88%) | Buys: 6, Sells: 0

Aggregated validation:
  Mean segment credit growth: -96.88%
  Total buys: 36, Total sells: 0

=== Episode 3/50 ===
Training on slice 2025-06-27 03:52:00+00:00 -> 2025-06-30 03:58:00+00:00
                                                                                                                                                          
-> Validation phase…
                                                                                                                                                          
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
Model saved to saved_models/best_model.pth
New best validation credit growth: 0.00%. Model saved.

Aggregated validation:
  Mean segment credit growth: 0.00%
  Total buys: 0, Total sells: 0

=== Episode 4/50 ===
Training on slice 2025-05-28 16:17:00+00:00 -> 2025-05-31 16:47:00+00:00
                                                                                                                                                          
-> Validation phase…
                                                                                                                                                          
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0

Aggregated validation:
  Mean segment credit growth: 0.00%
  Total buys: 0, Total sells: 0

=== Episode 5/50 ===
Training on slice 2025-06-23 14:16:00+00:00 -> 2025-06-26 09:54:00+00:00
                                                                                                                                                          
-> Validation phase…
                                                                                                                                                          
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0

Aggregated validation:
  Mean segment credit growth: 0.00%
  Total buys: 0, Total sells: 0
=== Episode 6/50 ===
Training on slice 2025-05-23 14:33:00+00:00 -> 2025-05-26 09:48:00+00:00
TRAINING:   2%|██▏                                                           TRAINING:   2%|██▎                                                           TRAINING:   2%|██▍                                                           TRAINING:   2%|██▌                                                           TRAINING:   2%|██▌                                                           TRAINING:   3%|██▋                                                           TRAINING:   3%|██▊                                                           TRAINING:   3%|██▉                                                           TRAINING:   3%|██▉                                                           TRAINING:   3%|███                                                           TRAINING:   3%|███▏                                                          TRAINING:   3%|███▏                                                          TRAINING:   3%|███▎                                                          TRAINING:   3%|███▍                                                          TRAINING:   3%|███▍                                                          TRAINING:   3%|███▌                                                          TRAINING:   4%|███▋                                                          TRAINING:   4%|███▊                                                          TRAINING:   4%|███▊                                                          TRAINING:   4%|███▉                                                          TRAINING:   4%|████                                                          TRAINING:   4%|████▏                                                         TRAINING:   4%|████▏                                                         TRAINING:   4%|████▎                                                         TRAINING:   4%|████▍                                                         TRAINING:   4%|████▌                                                                                                                                                                                                                   
-> Validation phase…
                                                                                                                                                          
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $100.00 (Ret: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0

Aggregated validation:
  Mean segment credit growth: 0.00%
  Total buys: 0, Total sells: 0

```

List Of Recommendations to apply to the code
Below is a prioritized checklist of what to fix, add, or modify, with the why/where/what-for. It addresses (1) why you see no trades in later episodes, (2) how to make credit grow cumulatively, and (3) bugs in validation metrics that make results misleading.

A. Critical issues causing “no trades” and misleading validation

Greedy eval with an untrained network → always HOLD
Why: During validation (is_eval=True), you switch the network to eval() and disable NoisyNet noise. Early in training (or before warmup ends) the network is effectively random, and its tiny initialization biases often make argmax pick action 0 (HOLD) consistently. That’s why you see zero buys/sells in Episode 3+.
Where: DQNAgent.act
What to change:
Do not use fully greedy during early epochs; allow some exploration in validation, or keep NoisyNet noise for action selection in eval until you have a trained model. Easiest: add an eval_epsilon (e.g., 0.05–0.1) in act when is_eval=True to occasionally try non-HOLD actions. Alternatively, keep NoisyNet active by not setting eval() in validation (or explicitly reset noise before inference).
Also reset NoisyLinear noise before each decision step (training side) to avoid stale noise. Example patch:
Add cfg.EVAL_EPSILON = 0.05
In DQNAgent.act:
Do not increment self.total_steps in eval (see point 2).
If is_eval and random.random() < cfg.EVAL_EPSILON: return random.randrange(self.action_dim)
If training (is_eval=False), call self.policy_net.reset_noise() before forward to get fresh exploration.
Validation increments total_steps and burns the warmup budget
Why: self.total_steps += 1 is executed even in eval, which advances the warmup counter despite not adding experiences. This creates inconsistent training phases and confuses your warmup/learning schedule.
Where: DQNAgent.act
What to change: Only increment total_steps in training calls. Example:
In act: if not is_eval: self.total_steps += 1
Training warmup is too long relative to decision frequency
Why: You only act every 5 minutes (DECISION_FREQUENCY=5), so you get ~1/5 as many “decisions” as steps. With TRAINING_WARMUP_STEPS=5000, you need roughly 5000 decisions to start learning, which is 5000*5=25,000 minutes (> 17.3 days) of data just to START learning. Until then, the eval policy is greedy random → HOLD.
Where: Config + DQNAgent.learn gating
What to change:
Reduce cfg.TRAINING_WARMUP_STEPS substantially (e.g., 1000) or base warmup on memory length instead of total_steps, i.e., start learning when len(memory) >= some threshold (e.g., 5k).
Alternatively set DECISION_FREQUENCY=1 for the first few episodes to accelerate exploration and learning, then increase it later.
Validation metrics are wrong (show “Credit $1.56” even though you liquidate on done)
Why:
You only append credit/portfolio histories while not done. On the final step (done=True), histories miss the liquidation to credit. Segment metrics then use credit_hist[-1], which is pre-liquidation cash (very low after buys), not final credit.
Portfolio return is computed using credit_hist[0] as the base instead of portfolio value base.
Where: run_episode (history appending) and validate_in_segments (metrics calculation)
What to change:
Append the last (post-liquidation) values to histories before returning, or compute metrics from the final values returned by run_episode instead of the histories.
For PV return, base on portfolio value (not credit) and on the first PV in the segment. Example:
In run_episode: after loop ends (or inside when done), push the final info["portfolio_value"] and credit to the histories so plotting and metrics are consistent.
In validate_in_segments: compute
final_pv from the return value of run_episode (or append last into pv_hist).
PV return = (final_pv - initial_pv) / initial_pv.
Credit growth = (final_credit - initial_credit) / initial_credit.
B. Reward shaping to grow credit cumulatively (sell/harvest profits) Your current reward strongly rewards realized pnl on sells, penalizes trading and unrealized losses, and penalizes time-in-position. There is no positive reward for unrealized gains or for increasing cash. This can push the agent to a “do nothing” policy, or to buy-and-hold without ever selling.

Recommended changes: 5) Add explicit reward for positive credit delta and for harvesting gains

Why: You said you want credit (cash) to grow cumulatively. Reward credit increases directly to make harvesting profitable.
Where: TradingEnvironment._calculate_reward
What to change:
Track credit_now and credit_next and give a small positive reward for (credit_next - credit_now)/start_credit.
Keep realized_pnl reward on sells, but scale down to not dominate.
Keep a smaller time-in-position penalty, but add a small positive reward for unrealized gains too (so holding winners is not punished).
Optionally add a drawdown penalty relative to max_portfolio_value to avoid deep underwater positions.
Concrete components you can add:

r_credit = alpha * (credit_next - credit_now) / self.start_credit (alpha ~ 2.0)
r_pv = beta * (portfolio_value_next - portfolio_value_now) / self.start_credit (beta ~ 0.2..0.5)
r_realized = gamma * realized_pnl / self.start_credit (gamma ~ 5..10; you have 20 now, that’s high)
r_trade_cost = -fixed_penalty_per_trade (0.1..0.2) OR proportional to fee paid
r_drawdown = -lambda * max(0, (max_pv - pv_next)/max_pv) (lambda small, e.g., 0.1)
Reduce time-in-position penalty or make it kick in only after some holding time.
Important: compute credit_now/next and portfolio_value_now/next inside step() right where you already have price_now and price_next.

Encourage exits (so credit grows)
Why: Your action set has no “SELL ALL,” and the time penalty alone may not be enough. You want the agent to realize profits.
Where: TradingEnvironment.action_space and step
What to change:
Add SELL_ALL and BUY_ALL actions. E.g., action 5: SELL_100, action 6: BUY_100 (or at least SELL_ALL).
This makes it easier to realize profits cleanly, which directly increases credit.
Penalize being over-invested when trend is adverse
Why: Encourage the agent to reduce exposure into drawdowns and come back to cash (credit growth safety).
Where: reward
What to change:
Add a penalty proportional to negative pv delta while invested (already partially done via unrealized negative change) and to high exposure when volatility spikes (optional).
C. Action timing and friction 8) Cooldown vs decision frequency aliasing

Why: With DECISION_FREQUENCY=5 and ACTION_COOLDOWN=5, the cooldown often consumes the next decision tick (on the 5th step it’s still >0 at the beginning of the step). You end up acting every 10 minutes after a trade. It’s okay but non-intuitive.
Where: Config + TradingEnvironment.step
What to change:
Either reduce ACTION_COOLDOWN to 4, or interpret cooldown in “decision ticks” rather than minutes so it doesn’t collide with DECISION_FREQUENCY. Alternatively, only decrement cooldown on decision steps.
Minimum trade thresholds and capital starvation
Why: When credit shrinks after some buys, BUY_25 may be below MIN_TRADE_CREDIT_BUY (1.0), leaving only BUY_50 usable. If network favors BUY_25, you’ll see no trade. Also, a small fixed threshold in currency units might be too high for end-game phases.
Where: Config + trade checks in step
What to change:
Lower MIN_TRADE_CREDIT_BUY (e.g., 0.1) or make it relative (e.g., min notional = 0.1% of start_credit).
Ensure all buy/sell actions are only selectable when executable (optional: mask invalid actions by setting their Q-values to -inf during forward, or reject them in env and add a small negative reward).
D. Exploration and training stability 10) Reset NoisyNet noise per decision tick (training)

Why: With NoisyNet, exploration requires resetting noise each action selection, not only during learn(), otherwise you might get stale exploration between optimizer steps.
Where: DQNAgent.act
What to change:
If not is_eval: call self.policy_net.reset_noise() immediately before computing Qs.
Start learning based on memory size, not total_steps
Why: More robust and independent of DECISION_FREQUENCY.
Where: DQNAgent.learn gating and act
What to change:
Gate learning with len(self.memory) >= some_threshold (e.g., 5000) and remove or drastically reduce TRAINING_WARMUP_STEPS.
Keep the existing len(memory) >= BATCH_SIZE guard.
Keep some exploration in early validation (until you trust the model)
Why: Early greedy eval is not meaningful and produces “no trades”.
Where: DQNAgent.act
What to change:
Use cfg.EVAL_EPSILON > 0 for early episodes (e.g., linearly decay eval_epsilon to 0 across episodes), or keep NoisyNet in eval by not switching to eval() mode during validation.
E. Data and preprocessing 13) Data leakage in normalization

Why: You standardize using the entire dataset (train+val+test). This leaks future statistics into the past and biases results.
Where: get_data()
What to change:
Fit normalization on train only, then apply to val/test using train’s mean/std. Keep Original_Close un-normalized as you do.
Technical indicators quirks
Why: Some indicators can be noisy on 1-min data; very high dimensionality can slow learning.
Where: get_data()
What to change:
Consider selecting a subset of robust indicators (e.g., RSI, MACD, ATR, EMA(S), OBV) to reduce noise and state dimension.
Alternatively increase WINDOW_SIZE if you keep many features so attention can learn temporal patterns.
F. Logging, metrics, and plotting fixes 15) Fix PV/credit metrics in validation

Why: You’re reporting “credit growth” from credit_hist which misses liquidation. And PV return uses the wrong base (credit instead of PV).
Where: validate_in_segments / plot_results / run_episode
What to change:
In run_episode: ensure last (post-liquidation) point is appended to histories. If you don’t want to append, compute segment metrics from the returned final_credit and final PV.
In validate_in_segments:
initial_pv = pv_hist[0] (or credit + holdings*price at segment start)
final_pv = pv_hist[-1] (after liquidation) or use the return value
PV return = (final_pv - initial_pv)/initial_pv
Credit growth = (final_credit - initial_credit)/initial_credit
Save checkpoints based on correct metric (e.g., mean final_credit growth or PV growth), not the pre-liquidation series.
Track and print action distribution
Why: To see if the policy is stuck in HOLD or is exploring.
Where: run_episode
What to change:
Count how many times each action was chosen (on decision ticks) and add to logs.
Seed everything for reproducibility
Why: To make debugging consistent.
Where: main()
What to change:
Set np.random.seed, random.seed, torch.manual_seed and torch.use_deterministic_algorithms(True) if possible.
G. Environment/action design improvements to help credit growth 18) Add SELL_ALL (and optional BUY_ALL)

Why: Makes harvesting easier and cash grows in discrete chunks.
Where: TradingEnvironment.action_space and step
What to change:
action_space = [0, 1, 2, 3, 4, 5] with 5=SELL_100 (and optional 6=BUY_100)
Implement branch in step to liquidate full position.
Optional: Cap position size and/or add a max leverage penalty
Why: Prevents the agent from becoming fully invested forever and helps it learn to go back to cash.
Where: step (before executing buys)
What to change: If current exposure (holdingsprice)/(credit+holdingsprice) > cap (e.g., 90%), reduce or block further buys.
Optional: Mask invalid actions
Why: Prevent learning to choose obviously invalid actions.
Where: In DuelingDQN.forward or agent.act
What to change:
Build an action mask from current state (credit and holdings ratios). Set Q-values for invalid actions to a very low number before argmax.
H. Hyperparameter tweaks (quick wins)

DECISION_FREQUENCY: set to 1 for the first 5–10 episodes, then raise to 5.
ACTION_COOLDOWN: 3–4 (or measured in decision ticks).
TRAINING_WARMUP_STEPS: 500 → 1000 max; or use memory-sized warmup.
Reward scales: realized pnl gamma ~ 5–10; credit delta alpha ~ 2; pv delta beta ~ 0.2–0.5; trade penalty ~ 0.1–0.2.
Learning rate: 1e-4 is fine; if unstable, try 5e-5.
I. Potential warnings/errors to address

Misleading returns printed: As explained, fix the metric base and include terminal liquidation in histories. Otherwise PV “Return: 111.76%” when PV ~105 is a calculation bug.
SumTree sampling if total priority == 0: you set max_priority=1.0 on push, so safe. Keep e=0.01 in update_priorities to avoid zero priorities.
NoisyLinear.reset_noise usage: Currently only in learn(), not per action; exploration will be stale. Reset per decision tick (training).
Memory bloat: With WINDOW_SIZE=180 and many features, memory 100k will be large. Consider fewer indicators or lower memory cap while debugging.
Data normalization leakage: Fit on train only.
J. Minimal concrete patches (summarized)

DQNAgent.act:
Only increment total_steps in training
Add eval_epsilon for validation
Reset NoisyNet noise before action in training
DQNAgent.learn:
Optionally gate by len(memory) instead of total_steps
TradingEnvironment._calculate_reward:
Add credit delta and pv delta positive rewards
Scale realized pnl reward down (or tune)
Optionally add drawdown penalty
TradingEnvironment.action_space and step:
Add SELL_ALL (and optional BUY_ALL)
Adjust MIN_TRADE_CREDIT_BUY (lower) or make relative
Optionally mask invalid actions
run_episode and validate_in_segments:
Append/post-process final post-liquidation histories or compute metrics from returned finals
Fix PV return base and credit growth computation
Track action counts
Config:
Reduce TRAINING_WARMUP_STEPS to 500–1000 (or rely on memory length)
Consider DECISION_FREQUENCY=1 for initial episodes; ACTION_COOLDOWN=3–4
Add EVAL_EPSILON
K. Sanity checks after changes

During training:
Action distribution should include buys and sells; not 99% HOLD.
Replay memory length should grow quickly; learning starts early.
During validation:
PV and credit histories should end with liquidation values.
Credit should step up over segments when profitable trades are realized.
Metrics:
“Credit growth” should be based on final_credit of each segment, not last pre-liquidation point.
PV return should be realistic (~a few %), not 100%+ from $100 → $105.
If you make the above changes, you’ll (1) get trades during both training and early validation, (2) align the reward with your goal of cumulative credit growth, and (3) fix the reporting so improvements are measured correctly and the “best model” checkpoint really reflects credit growth instead of a pre-liquidation artifact.

I need you to take all this recomedations and things to apply to the code, in order to fix it. Please apply all of this things, to make sure that now the code will work. Please write the whole code with the fixes integrated in it.


I dont need you to write the whole fixed code, but to generate a list of things to fix, add or modify. Please be very descriptive of why , where and what for is things. Please make somethign to make the credit grow in a cumulative manner and start making grow the money. Also, please decribe the errors or warnings you find and how should the be addressed.























































Look, I have this code, which Im trying to optimize to make it make good trades. Please help me fix it. I dont understand why it makes bad trades, it is not learning to grow the credit. Maybe it is paying a lot of fees?. What I want is to make the capital/credit grow in a cumulative manner. Also, I think there is an error in the calculation of the `Mean segmend growth`, since trhe credit is not actually growing, it is decreasing. If possible, for the falidation print the score for each of the policy reward rules, to see what is the model profiting and get a better understanding of what is happening:

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
```
=== Episode 1/50 ===
Training on slice 2025-05-19 10:09:00+00:00 -> 2025-05-22 06:03:00+00:00
                                                                                                                                                          
-> Validation phase…
                                                                                                                                                          
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $98.10 (Ret: -1.90%) | Credit: $98.10 (Growth: -1.90%) | Buys: 34, Sells: 34
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $95.40 (Ret: -2.75%) | Credit: $95.40 (Growth: -2.75%) | Buys: 28, Sells: 27
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $90.63 (Ret: -4.95%) | Credit: $90.63 (Growth: 90.01%) | Buys: 35, Sells: 35
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $86.30 (Ret: -4.78%) | Credit: $86.30 (Growth: -4.78%) | Buys: 44, Sells: 44
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $83.27 (Ret: -3.40%) | Credit: $83.27 (Growth: 0.00%) | Buys: 68, Sells: 67
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $81.27 (Ret: -2.41%) | Credit: $81.27 (Growth: 30.12%) | Buys: 49, Sells: 51
Model saved to saved_models/best_model.pth
New best validation credit growth: 18.45%. Model saved.

Aggregated validation:
  Mean segment credit growth: 18.45%
  Total buys: 258, Total sells: 258

=== Episode 2/50 ===
Training on slice 2025-06-12 13:50:00+00:00 -> 2025-06-15 03:47:00+00:00
                                                                                                                                                          
-> Validation phase…
                                                                                                                                                          
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $97.57 (Ret: -2.43%) | Credit: $97.57 (Growth: -2.43%) | Buys: 27, Sells: 26
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $95.73 (Ret: -1.89%) | Credit: $95.73 (Growth: -1.89%) | Buys: 18, Sells: 18
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $91.57 (Ret: -4.35%) | Credit: $91.57 (Growth: -4.35%) | Buys: 36, Sells: 33
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $88.70 (Ret: -3.13%) | Credit: $88.70 (Growth: -3.13%) | Buys: 29, Sells: 30
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $86.38 (Ret: -2.61%) | Credit: $86.38 (Growth: -2.61%) | Buys: 28, Sells: 27
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $85.08 (Ret: -1.51%) | Credit: $85.08 (Growth: -1.51%) | Buys: 22, Sells: 22

Aggregated validation:
  Mean segment credit growth: -2.65%
  Total buys: 160, Total sells: 156
=== Episode 3/50 ===
Training on slice 2025-05-30 14:49:00+00:00 -> 2025-06-02 16:44:00+00:00
                                                                                                                                                          
-> Validation phase…
                                                                                                                                                          
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $97.94 (Ret: -2.06%) | Credit: $97.94 (Growth: -2.06%) | Buys: 45, Sells: 46
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $93.99 (Ret: -4.00%) | Credit: $93.99 (Growth: 27.96%) | Buys: 59, Sells: 57
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $90.42 (Ret: -3.77%) | Credit: $90.42 (Growth: 28.27%) | Buys: 53, Sells: 53
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $87.53 (Ret: -3.12%) | Credit: $87.53 (Growth: 0.00%) | Buys: 33, Sells: 34
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $84.60 (Ret: -3.29%) | Credit: $84.60 (Growth: 93.30%) | Buys: 68, Sells: 67
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $82.41 (Ret: -2.59%) | Credit: $82.41 (Growth: 29.89%) | Buys: 51, Sells: 49
Model saved to saved_models/best_model.pth
New best validation credit growth: 29.56%. Model saved.

Aggregated validation:
  Mean segment credit growth: 29.56%
  Total buys: 309, Total sells: 306

```

Great question. You have two separate problems here:

The agent is incentivized to make lots of trades that do not grow total wealth (credit + holdings), which bleeds fees. This is largely due to reward shaping that accidentally rewards “credit pumping” even when portfolio value falls.
Your validation metrics (credit growth and “mean segment growth”) are computed on the wrong baseline, which makes the printouts misleading (e.g., the 90% credit growth while PV is falling). So the model selection and your understanding of what’s happening are skewed.
Below is a prioritized checklist of what to change, why, and where. I also include small snippets so you know exactly what to edit, but I’m not rewriting your whole script.

A) Fix the validation metrics (this is why “Mean segment growth” is wrong) Why:

You compute segment “initial” credit and PV from the first element of the segment histories after the first trade has already happened. If the first action is a buy, credit_hist[0] is already lower than the true starting credit, making later “credit growth” look positive even as wealth declines. That’s why you saw things like Growth: 90.01% with PV falling.
Where:

validate_in_segments(), right after each run_episode() call and before segment_metrics.append(...).
What to change:

Use the true starting baselines instead of pv_hist[0] and credit_hist[0]:
initial_credit_seg must be the current_credit you carried into the segment (not credit_hist[0]).
initial_pv must be current_credit + current_holdings * first_bar_price_at_segment_start (i.e., the price at index window_size).
How: Replace this:

initial_pv = pv_hist[0] if pv_hist else (current_credit)
initial_credit_seg = credit_hist[0] if credit_hist else current_credit
With this:

start_price = segment_data["Original_Close"].iloc[val_env.window_size]
initial_pv = current_credit + current_holdings * start_price
initial_credit_seg = current_credit
And make model selection based on a portfolio metric, not credit growth:

Instead of mean segment credit growth, use cumulative PV return or mean log-returns across segments.
B) Remove the perverse incentive in the reward function (the main reason for bad trades) Why:

You reward positive changes in credit (cash) but do not penalize credit reductions. This encourages the agent to cycle “buy then sell” because sells increase credit (reward!), while buys don’t cause a symmetric negative reward. Combined with small trade penalties and time penalty, the optimal policy under this reward is to churn and earn reward from “credit increases,” even if total wealth declines due to fees.
You want cumulative growth of total portfolio value, not just cash.
Where:

TradingEnvironment._calculate_reward()
What to change (at minimum):

Remove or set to zero the credit_delta reward term, or make it symmetric (reward increases and penalize decreases equally).
Make portfolio value change the main term, ideally as a log return to align with compounding (additivity of log returns).
Recommended simple reward (strongly suggested):

reward = log(pv_after / pv_before), plus:
add realized PnL term only when positions are closed or reduced (optional),
penalize only increases in drawdown (not drawdown level every step),
a small per-trade penalty to reflect frictions.
Concrete changes:

Set cfg.REWARD_CREDIT_DELTA = 0.0
Replace pv_delta term with log return:
log_ret = math.log(max(pv_after, 1e-8) / max(pv_before, 1e-8))
reward += log_ret
Change drawdown penalty to penalize only when drawdown worsens:
Keep track of previous drawdown and penalize max(0, drawdown - prior_drawdown)
Reduce TIME_PENALTY or remove it at first; it’s currently pushing the agent to avoid holding.
C) Add reward diagnostics so you can see what the model is “profiting” from Why:

You asked to print per-rule scores. This is crucial to verify that after the changes the agent is no longer exploiting credit_delta or a constant drawdown penalty.
Where:

TradingEnvironment._calculate_reward()
TradingEnvironment.step()
run_episode()
validate_in_segments()
What to do:

Make _calculate_reward return both total reward and a dict of its components.
In step(), include that components dict in info.
In run_episode(), accumulate component sums during validation.
In validate_in_segments(), print the per-segment sum of each component.
Sketch:

In _calculate_reward, compute comps = {"pv_logret": log_ret, "realized": ..., "trade_pen": ..., "drawdown_pen": ..., "time_pen": ...}, reward = sum(comps.values())
return reward, comps
In step(), stash comps into info["reward_components"]
In run_episode(), accumulate during is_eval and return the sums; in validate_in_segments() print them per segment.
D) Make trading less “chatty” to reduce fee bleed Why:

You are evaluating every minute (or every 5). With small moves, frequent partial flips cause more fees than captured edge. Your fees plus trade penalty outweigh expected edge at this frequency.
Where:

Config and the environment action gating.
What to change:

Increase decision intervals and cooldown. For example:
DECISION_FREQUENCY_LATER = 15
ACTION_COOLDOWN = 10
Increase minimum trade size thresholds to avoid micro trades:
MIN_TRADE_CREDIT_BUY = 2.0 (or 5.0)
MIN_TRADE_HOLDINGS_SELL = 5.0
Optionally cap daily number of trades in validation (for analysis) to understand cost impact.
E) Align reward and transaction costs, and reduce constant penalties that bias behavior Why:

Your TRADE_PENALTY is small compared to the old credit_delta reward. After removing credit_delta, the PV log-return already includes fees; a small extra trade_penalty can still discourage churn.
Where:

Config and reward.
What to change:

After moving to log PV reward:
Set TRADE_PENALTY around 0.05–0.15 (start at lower end and adjust).
Reduce or remove TIME_PENALTY initially; re-introduce gently if it helps hold times (e.g., TIME_PENALTY = 0.0005).
DRAWDOWN_PENALTY: penalize drawdown increases only, and reduce to ~0.01–0.02 to avoid overwhelming the signal.
F) Correct the Q-learning target to explicitly mask terminals Why:

You rely on next_state=None to zero next_q, which is okay, but it’s standard and clearer to multiply by (1 - done). This also makes bugs easier to spot if next_state handling changes.
Where:

DQNAgent.learn()
What to change:

Build dones tensor and use it in the target:
dones_tensor = torch.tensor(dones, device=device, dtype=torch.float32)
expected_q_values = reward_batch + cfg.GAMMA * (1 - dones_tensor) * next_q_values
G) Reduce randomness during validation so metrics are stable Why:

EVAL_EPSILON = 0.1 introduces randomness in validation. Keep it at 0 for selection, or very small (0.01) only if you see degeneracy like persistent HOLDs.
Where:

Config and DQNAgent.act()
What to change:

Set cfg.EVAL_EPSILON = 0.0 during validation used for model selection.
You can keep a small epsilon for early episodes if needed, but don’t use it for selecting and saving best model.
H) Select and save the best model using a portfolio metric Why:

“Mean segment credit growth” is not robust and was incorrect anyway. Use wealth-based measures.
Where:

main(), validation block after validate_in_segments()
What to change:

Compute cumulative PV return across the whole validation pass:
total_initial = cfg.INITIAL_CREDIT
total_final = last segment’s final_pv (since you carry forward credit and liquidate at each segment end, this is additive across segments via compounding log-returns or multiplicative via PV).
Or compute average log-return per segment and use its mean.
Save the model by best cumulative PV return or mean log-return, not mean credit growth.
I) Small environment and metric bugs and cleanups

“Mean segmend growth” typo: fix the label to avoid confusion in logs.
Plot misalignment: portfolio_value uses next_price while holdings_value in plotting uses current_index post-step. It’s not the cause of bad performance, but you may wish to align both to next_price or both to current price for visual clarity.
Drawdown computation uses pv_after (based on next_price) while max_portfolio_value was set from previous step’s PV. It’s acceptable but be consistent across all PV calculations within a step to avoid off-by-one artifacts in analytics.
Consider adding a turnover metric per episode/segment to quantify fee drag: sum of dollar notional traded and sum of fee dollars paid. You can compute fee dollars in step():
buy_fee = investment * cfg.FEE
sell_fee = sell_amount_asset * current_price * cfg.FEE Aggregate and print per segment.
J) Optional but helpful improvements

Exposure cap: Lower soft cap from 99% to something like 80–90% to avoid being max-long all the time.
“Close position” action: Consider adding an explicit “flatten” action to close all holdings; it can simplify policy learning.
Observation cleanup: Instead of repeating portfolio_state for every row in the window, append it only to the last row (your net concatenation currently repeats it across the window – it works but is redundant). This is an architectural micro-optimization.
Concrete parameter starting points after changes

Rewards:
Use reward = log(pv_after / pv_before)
Optionally add realized PnL scaled by start_credit on sells only (keep small, e.g., 0.5x) if you want to emphasize profitable exits.
Penalize only increases in drawdown: reward -= 0.02 * max(0, dd_after - dd_before)
Trade penalty = 0.05–0.1
Remove TIME_PENALTY (set to 0.0) initially
Set REWARD_CREDIT_DELTA = 0.0 and REWARD_PV_DELTA term unused
Action cadence and frictions:
DECISION_FREQUENCY_LATER = 15
ACTION_COOLDOWN = 10
MIN_TRADE_CREDIT_BUY = 2.0 (or 5.0) and MIN_TRADE_HOLDINGS_SELL = 5.0
Validation:
EVAL_EPSILON = 0.0
Save best by cumulative validation PV return
Print per-segment reward components, turnover, fees, trades
Minimal code edits (snippets only)

validate_in_segments: correct baselines and print component totals
After running the segment:
start_price = segment_data["Original_Close"].iloc[val_env.window_size] initial_pv = current_credit + current_holdings * start_price initial_credit_seg = current_credit

seg_return = (final_pv - initial_pv) / initial_pv * 100 if initial_pv > 0 else 0.0 credit_growth = (final_credit - initial_credit_seg) / initial_credit_seg * 100 if initial_credit_seg > 0 else 0.0

Change model selection metric to cumulative PV return or mean log-return.
DQNAgent.learn: explicit terminal masking
Add:
dones_tensor = torch.tensor(dones, device=device, dtype=torch.float32) expected_q_values = reward_batch + cfg.GAMMA * (1 - dones_tensor) * next_q_values

TradingEnvironment._calculate_reward: switch to log PV and component reporting
Compute:
log_ret = math.log(max(pv_after, 1e-8) / max(pv_before, 1e-8)) drawdown = ... drawdown_increase = max(0.0, drawdown - self.prev_drawdown) trade_pen = cfg.TRADE_PENALTY if trade_executed else 0.0

comps = { "pv_logret": log_ret, "realized": (realized_pnl / self.start_credit) if realized_pnl != 0 else 0.0, "trade_pen": -trade_pen, "dd_pen": -0.02 * drawdown_increase } reward = sum(comps.values())

Save prev_drawdown in the environment between steps.
Return both reward and components, and pass components out via info in step().
Config: deprecate credit and PV delta weights
Set:
REWARD_CREDIT_DELTA = 0.0
REWARD_PV_DELTA unused
TIME_PENALTY = 0.0 (at least initially)
TRADE_PENALTY = 0.05–0.1
DRAWDOWN_PENALTY removed in favor of dd increase penalty as above
DECISION_FREQUENCY_LATER = 15
ACTION_COOLDOWN = 10
MIN_TRADE_CREDIT_BUY = 2.0–5.0
MIN_TRADE_HOLDINGS_SELL = 5.0
EVAL_EPSILON = 0.0
Validation/output
Fix the “Mean segmend growth” label typo.
Replace “Mean segment credit growth” with “Mean segment PV log-return (%)” or “Cumulative PV return (%)”.
Print aggregated reward components per segment:
e.g., Sum pv_logret, Sum realized, Sum trade_pen, Sum dd_pen for the segment.
Why these changes will make credit grow cumulatively

With reward = log(PV_after/PV_before), the agent is directly optimized to increase total wealth multiplicatively. There is no loophole for getting high reward by moving value from holdings to credit and back.
Higher decision interval, cooldown, and minimum trade sizes reduce churn and fee drag.
Proper validation baselines ensure you actually select models that grow wealth, not artifacts from the initial-credit mismeasurement.
Printing per-rule totals lets you confirm that most of the reward is coming from PV log-returns, not from constant penalties or artifacts.
Warnings you should expect and how to address them

After switching to log-return reward, the reward scale changes (around small numbers). That’s okay; you may see slower learning initially. You can slightly increase learning rate if convergence stalls, but first let buffer fill and PER anneal.
If you remove TIME_PENALTY and significantly reduce drawdown penalty, the agent may hold longer; monitor risk and reintroduce a mild drawdown-increase penalty if needed.
If you keep dropout in attention while training (which you do), remember to set model to eval in validation (you already do in act during is_eval), so you should be fine.
Final sanity checklist

Validate on EVAL_EPSILON=0.0 to select the model; avoid random actions affecting selection.
Save best by cumulative PV return (or mean log-return), not credit growth.
Verify per-segment component printouts show positive pv_logret sums on good segments and see trade_pen/fees small relative to pv_logret.
Expect far fewer trades after increasing cadence/cooldown and thresholds.
Do the A+B+C+D edits first. They fix the measurement bug and remove the reward loophole that’s causing bad trades. Then iterate on the cadence and penalties to balance trading frequency versus edge.

I need you to take all this recomedations and things to apply to the code, in order to fix it. Please apply all of this things, to make sure that now the code will work. Please write the whole code with the fixes integrated in it.


I dont need you to write the whole fixed code, but to generate a list of things to fix, add or modify. Please be very descriptive of why , where and what for is things. Please make somethign to make the credit grow in a cumulative manner and start making grow the money. Also, please decribe the errors or warnings you find and how should the be addressed.



















































































Look, I have this code, which Im trying to optimize to make it make good trades and the main objetive is to make the credit or capital grow. Please help me fix it. What I want is to make the capital/credit grow in a cumulative manner. 
Why after the 3 or 4 episode, it stopped plotting the validation data? I know it could be making a lot more money, but something is making it stop trying to push harder on each valiudation frame/episode; its like after the first frame(till the first dotted lines), it wont try to make more money on the next phase. Is it correctly learning to identify the moments it should have selled and it didnt?:
```
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

    # Exploration in validation (set to 0 for deterministic model selection)
    EVAL_EPSILON = 0.0

    # Decision frequency scheduling (minutes)
    DECISION_FREQUENCY_BASE = 1
    DECISION_FREQUENCY_LATER = 15
    DECISION_FREQ_SWITCH_EPISODE = 5  # after this episode, use later frequency

    # Cooldown measured in decision ticks (not raw minutes)
    ACTION_COOLDOWN = 10  # decision ticks

    # --- Model Architecture ---
    ATTENTION_DIM = 64
    ATTENTION_HEADS = 4
    FC_UNITS_1 = 256
    FC_UNITS_2 = 128

    # --- Environment ---
    INITIAL_CREDIT = 100.0
    WINDOW_SIZE = 180
    FEE = 0.001
    MIN_TRADE_CREDIT_BUY = 5.0      # avoid micro buys
    MIN_TRADE_HOLDINGS_SELL = 5.0   # in currency terms

    # Reward shaping coefficients (new scheme)
    REWARD_REALIZED = 0.5           # small bonus for realized pnl
    TRADE_PENALTY = 0.08            # fixed penalty per executed trade
    TIME_PENALTY = 0.0              # set 0 initially; re-introduce later if needed
    DRAWDOWN_PENALTY = 0.02         # penalize increases in drawdown only

    # Legacy reward weights (unused now; kept for clarity)
    REWARD_CREDIT_DELTA = 0.0
    REWARD_PV_DELTA = 0.0

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
            if self.action_dim >= 7:
                mask[6] = False
        if holdings_ratio < 0.01:
            mask[3] = False
            mask[4] = False
            if self.action_dim >= 6:
                mask[5] = False
        return mask

    def act(self, state, is_eval=False):
        if not is_eval:
            self.total_steps += 1

        state_np = np.array(state, dtype=np.float32)
        mask = self._action_mask_from_state(state_np)
        valid_actions = np.where(mask)[0].tolist()
        if len(valid_actions) == 0:
            return 0  # fallback

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

        # Apply mask
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
        dones_tensor = torch.tensor(dones, device=device, dtype=torch.float32)

        if any(non_final_mask.cpu().numpy()):
            non_final_next_states = torch.FloatTensor(np.array([s for s in next_states if s is not None])).to(device)
        else:
            non_final_next_states = torch.empty((0,) + state_batch.shape[1:], device=device)

        next_q_values = torch.zeros(cfg.BATCH_SIZE, device=device)
        if non_final_next_states.size(0) > 0:
            with torch.no_grad():
                next_actions = self.policy_net(non_final_next_states).argmax(1).unsqueeze(1)
                next_q_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze(1)

        expected_q_values = reward_batch + cfg.GAMMA * (1.0 - dones_tensor) * next_q_values
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
        self.prev_drawdown = 0.0
        # fee/turnover trackers
        self.total_fees = 0.0
        self.total_buy_notional = 0.0
        self.total_sell_notional = 0.0
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
        # Cooldown only on decision ticks
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
        buy_notional = 0.0
        sell_notional = 0.0
        fees_step = 0.0

        if buy_fraction > 0:
            # Prevent extreme over-investment (cap exposure ~90%)
            portfolio_value = self.credit + self.holdings * current_price
            current_exposure = (self.holdings * current_price) / portfolio_value if portfolio_value > 0 else 0.0
            if current_exposure < 0.90:  # soft cap
                investment = self.credit * buy_fraction
                if investment > cfg.MIN_TRADE_CREDIT_BUY:
                    buy_fee = investment * cfg.FEE
                    buy_amount_asset = (investment * (1 - cfg.FEE)) / current_price
                    total_cost = (self.average_buy_price * self.holdings) + investment
                    self.holdings += buy_amount_asset
                    self.credit -= investment
                    self.average_buy_price = total_cost / self.holdings if self.holdings > 0 else 0.0
                    self.trades.append({"step": self.current_step, "type": "buy", "price": current_price, "amount": buy_amount_asset})
                    trade_executed = True
                    buy_notional = investment
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

        # Reward calculation (log PV return-centric; penalize drawdown increases only)
        reward, comps = self._calculate_reward(realized_pnl, pv_before, pv_after, trade_executed)

        next_state = self._get_state() if not done else None

        info = {
            "portfolio_value": pv_after,
            "credit": self.credit,
            "holdings": self.holdings,
            "trades": self.trades,
            "reward_components": comps,
            "fees_paid_step": fees_step,
            "buy_notional_step": buy_notional,
            "sell_notional_step": sell_notional
        }
        return next_state, reward, done, info

    def _calculate_reward(self, realized_pnl, pv_before, pv_after, trade_executed):
        # Core: log return on portfolio value (encourages multiplicative growth)
        log_ret = math.log(max(pv_after, 1e-8) / max(pv_before, 1e-8))

        # Realized PnL shaping (small)
        realized_component = cfg.REWARD_REALIZED * (realized_pnl / max(self.start_credit, 1e-8)) if realized_pnl != 0 else 0.0

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

        comps = {
            "pv_logret": log_ret,
            "realized": realized_component,
            "trade_pen": trade_pen,
            "time_pen": time_pen,
            "dd_pen": dd_pen
        }
        reward = sum(comps.values())
        return reward, comps

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
    # Reward component and cost aggregators (per episode/segment)
    comp_sums = {"pv_logret": 0.0, "realized": 0.0, "trade_pen": 0.0, "time_pen": 0.0, "dd_pen": 0.0}
    fees_total = 0.0
    buy_notional_total = 0.0
    sell_notional_total = 0.0

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
        current_index = min(env.current_step, len(data) - 1)
        current_price = data["Original_Close"].iloc[current_index]
        holdings_values.append(info["holdings"] * current_price)

        # Aggregate reward components and costs (for validation diagnostics)
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

    final_pv = portfolio_values[-1] if portfolio_values else initial_credit
    final_credit = credits[-1] if credits else initial_credit
    final_holdings = info["holdings"] if info else 0.0

    return (
        portfolio_values, credits, holdings_values, env.trades,
        final_pv, final_credit, final_holdings, dict(action_counts),
        comp_sums, fees_total, buy_notional_total, sell_notional_total
    )

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
        (
            pv_hist, credit_hist, hold_hist, trades,
            final_pv, final_credit, final_holdings, action_counts,
            comp_sums, fees_total, buy_notional_total, sell_notional_total
        ) = run_episode(
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

        # Correct baselines: use true starting credit and PV at segment start (price at window_size)
        start_price = segment_data["Original_Close"].iloc[val_env.window_size]
        initial_pv = current_credit + current_holdings * start_price
        initial_credit_seg = current_credit

        seg_return = ((final_pv - initial_pv) / initial_pv * 100) if initial_pv > 0 else 0.0
        credit_growth = ((final_credit - initial_credit_seg) / initial_credit_seg * 100) if initial_credit_seg > 0 else 0.0

        # Convert summed pv_logret to percentage for segment
        seg_logret_pct = (math.exp(comp_sums["pv_logret"]) - 1.0) * 100.0

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
            "reward_components": comp_sums,
            "fees_total": fees_total,
            "turnover_buy": buy_notional_total,
            "turnover_sell": sell_notional_total
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
    best_val_pv_return = -1e9

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

        _pv, _cred, _hold, _trades, _fpv, _fcred, _fhold, action_counts, _c_sums, _fees, _buy_not, _sell_not = run_episode(
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
            rc = m["reward_components"]
            print(
                f"  Seg {m['seg']:02d} [{m['start']} → {m['end']}]  "
                f"PV: ${m['final_pv']:.2f} (Ret: {m['ret']:.2f}%, LogRet: {m['logret_pct']:.2f}%) | "
                f"Credit: ${m['final_credit']:.2f} (Growth: {m['credit_growth']:.2f}%) | "
                f"Buys: {m['buys']}, Sells: {m['sells']} | "
                f"Fees: ${m['fees_total']:.2f}, Turnover(B/S): ${m['turnover_buy']:.2f}/${m['turnover_sell']:.2f} | "
                f"Rewards Σ: pv_logret={rc['pv_logret']:.4f}, realized={rc['realized']:.4f}, trade_pen={rc['trade_pen']:.4f}, "
                f"dd_pen={rc['dd_pen']:.4f}, time_pen={rc['time_pen']:.4f}"
            )

        if seg_metrics:
            # Selection metric: cumulative PV return across validation
            total_final_credit = seg_metrics[-1]["final_credit"]
            total_initial_credit = cfg.INITIAL_CREDIT
            cumulative_val_return = (total_final_credit - total_initial_credit) / total_initial_credit * 100.0

            # Mean segment PV log-return (%), computed from per-seg logret_pct
            mean_seg_logret_pct = float(np.mean([m["logret_pct"] for m in seg_metrics]))

            if cumulative_val_return > best_val_pv_return:
                best_val_pv_return = cumulative_val_return
                agent.save_model(f"saved_models/best_model.pth")
                print(f"New best validation cumulative PV return: {best_val_pv_return:.2f}%. Model saved.")

            total_buys = sum(m["buys"] for m in seg_metrics)
            total_sells = sum(m["sells"] for m in seg_metrics)

            print("\nAggregated validation:")
            print(f"  Cumulative PV return: {cumulative_val_return:.2f}%")
            print(f"  Mean segment PV log-return: {mean_seg_logret_pct:.2f}%")
            print(f"  Total buys: {total_buys}, Total sells: {total_sells}")

        if e < 3 and pv_hist:
            plot_results(
                val_data, e + 1, pv_hist, credit_hist, hold_hist, trades,
                "Validation", [s + cfg.WINDOW_SIZE for s in seg_starts[1:]]
            )

    print("\n=== FINAL TEST ON UNSEEN DATA ===")
    try:
        agent.load_model("saved_models/best_model.pth")
    except Exception as e:
        print("No saved model found or failed to load; using current policy.")
    test_env = TradingEnvironment(test_data.reset_index(drop=True))
    pv, cred, hold, trades, final_pv, final_credit, final_holdings, act_counts, comp_sums, fees_total, buy_not, sell_not = run_episode(
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
        print(f"Fees paid: ${fees_total:.2f}, Turnover(B/S): ${buy_not:.2f}/${sell_not:.2f}")
        print(f"Reward components Σ (test): {json.dumps({k: round(v, 6) for k, v in comp_sums.items()})}")
        plot_results(test_data, "Final", pv, cred, hold, trades, "Final Test")

if __name__ == "__main__":
    main()
```

```
=== Episode 1/50 ===
Training on slice 2025-05-19 10:09:00+00:00 -> 2025-05-22 06:03:00+00:00
                                                                                                                                                          
-> Validation phase…
                                                                                                                                                          
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $101.09 (Ret: 1.09%, LogRet: 1.09%) | Credit: $101.09 (Growth: 1.09%) | Buys: 3, Sells: 7 | Fees: $0.24, Turnover(B/S): $120.29/$121.51 | Rewards Σ: pv_logret=0.0109, realized=0.0061, trade_pen=-0.7200, dd_pen=-0.0016, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $101.09 (Ret: -0.00%, LogRet: -0.00%) | Credit: $101.09 (Growth: -0.00%) | Buys: 1, Sells: 4 | Fees: $0.10, Turnover(B/S): $50.55/$50.59 | Rewards Σ: pv_logret=-0.0000, realized=0.0002, trade_pen=-0.3200, dd_pen=-0.0004, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $99.71 (Ret: -1.36%, LogRet: -1.36%) | Credit: $99.71 (Growth: -1.36%) | Buys: 3, Sells: 5 | Fees: $0.21, Turnover(B/S): $107.27/$106.00 | Rewards Σ: pv_logret=-0.0137, realized=-0.0063, trade_pen=-0.5600, dd_pen=-0.0022, time_pen=0.0000  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $98.85 (Ret: -0.86%, LogRet: -0.86%) | Credit: $98.85 (Growth: -0.86%) | Buys: 8, Sells: 7 | Fees: $0.43, Turnover(B/S): $217.17/$216.53 | Rewards Σ: pv_logret=-0.0087, realized=-0.0032, trade_pen=-1.1200, dd_pen=-0.0041, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $98.85 (Ret: -0.00%, LogRet: -0.00%) | Credit: $98.85 (Growth: -0.00%) | Buys: 1, Sells: 4 | Fees: $0.10, Turnover(B/S): $49.43/$49.47 | Rewards Σ: pv_logret=-0.0000, realized=0.0002, trade_pen=-0.3200, dd_pen=-0.0005, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $98.90 (Ret: 0.05%, LogRet: 0.05%) | Credit: $98.90 (Growth: 0.05%) | Buys: 1, Sells: 4 | Fees: $0.10, Turnover(B/S): $49.43/$49.53 | Rewards Σ: pv_logret=0.0005, realized=0.0005, trade_pen=-0.3200, dd_pen=-0.0008, time_pen=0.0000
Model saved to saved_models/best_model.pth
New best validation cumulative PV return: -1.10%. Model saved.

Aggregated validation:
  Cumulative PV return: -1.10%
  Mean segment PV log-return: -0.18%
  Total buys: 17, Total sells: 31

=== Episode 2/50 ===
Training on slice 2025-05-17 16:23:00+00:00 -> 2025-05-20 09:26:00+00:00
                                                                                                                                                          
-> Validation phase…
                                                                                                                                                                                                                                                                                                                   
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $100.43 (Ret: 0.43%, LogRet: 0.43%) | Credit: $100.43 (Growth: 0.43%) | Buys: 2, Sells: 7 | Fees: $0.19, Turnover(B/S): $96.86/$97.39 | Rewards Σ: pv_logret=0.0043, realized=0.0026, trade_pen=-0.6400, dd_pen=-0.0010, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $100.46 (Ret: 0.03%, LogRet: 0.03%) | Credit: $100.46 (Growth: 0.03%) | Buys: 1, Sells: 4 | Fees: $0.10, Turnover(B/S): $50.22/$50.30 | Rewards Σ: pv_logret=0.0003, realized=0.0004, trade_pen=-0.3200, dd_pen=-0.0005, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $101.19 (Ret: 0.72%, LogRet: 0.72%) | Credit: $101.19 (Growth: 0.72%) | Buys: 1, Sells: 4 | Fees: $0.10, Turnover(B/S): $50.23/$51.01 | Rewards Σ: pv_logret=0.0072, realized=0.0039, trade_pen=-0.3200, dd_pen=-0.0010, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $100.11 (Ret: -1.06%, LogRet: -1.06%) | Credit: $100.11 (Growth: -1.06%) | Buys: 7, Sells: 4 | Fees: $0.30, Turnover(B/S): $152.40/$151.48 | Rewards Σ: pv_logret=-0.0107, realized=-0.0046, trade_pen=-0.8000, dd_pen=-0.0033, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $100.11 (Ret: -0.00%, LogRet: -0.00%) | Credit: $100.11 (Growth: -0.00%) | Buys: 1, Sells: 4 | Fees: $0.10, Turnover(B/S): $50.06/$50.10 | Rewards Σ: pv_logret=-0.0000, realized=0.0002, trade_pen=-0.3200, dd_pen=-0.0005, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $100.16 (Ret: 0.05%, LogRet: 0.05%) | Credit: $100.16 (Growth: 0.05%) | Buys: 1, Sells: 4 | Fees: $0.10, Turnover(B/S): $50.06/$50.16 | Rewards Σ: pv_logret=0.0005, realized=0.0005, trade_pen=-0.3200, dd_pen=-0.0008, time_pen=0.0000
Model saved to saved_models/best_model.pth
New best validation cumulative PV return: 0.16%. Model saved.

Aggregated validation:
  Cumulative PV return: 0.16%
  Mean segment PV log-return: 0.03%
  Total buys: 13, Total sells: 27

=== Episode 3/50 ===
Training on slice 2025-05-26 23:21:00+00:00 -> 2025-05-29 14:51:00+00:00
                                                                                                                                                                                                                                                                                                                   
-> Validation phase…
                                                                                                                                                                                                                                                                                                                   
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $102.97 (Ret: 2.97%, LogRet: 2.97%) | Credit: $102.97 (Growth: 2.97%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.00/$53.03 | Rewards Σ: pv_logret=0.0293, realized=0.0151, trade_pen=-0.0800, dd_pen=-0.0037, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $103.10 (Ret: 0.12%, LogRet: 0.12%) | Credit: $103.10 (Growth: 0.12%) | Buys: 1, Sells: 1 | Fees: $0.05, Turnover(B/S): $25.74/$25.90 | Rewards Σ: pv_logret=0.0012, realized=0.0007, trade_pen=-0.0800, dd_pen=-0.0011, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $102.86 (Ret: -0.23%, LogRet: -0.23%) | Credit: $102.86 (Growth: -0.23%) | Buys: 1, Sells: 1 | Fees: $0.05, Turnover(B/S): $25.78/$25.56 | Rewards Σ: pv_logret=-0.0023, realized=-0.0010, trade_pen=-0.0800, dd_pen=-0.0025, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $102.82 (Ret: -0.04%, LogRet: -0.04%) | Credit: $102.82 (Growth: -0.04%) | Buys: 1, Sells: 3 | Fees: $0.05, Turnover(B/S): $25.72/$25.70 | Rewards Σ: pv_logret=-0.0004, realized=-0.0001, trade_pen=-0.2400, dd_pen=-0.0021, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $102.84 (Ret: 0.02%, LogRet: 0.02%) | Credit: $102.84 (Growth: 0.02%) | Buys: 1, Sells: 3 | Fees: $0.05, Turnover(B/S): $25.71/$25.75 | Rewards Σ: pv_logret=0.0002, realized=0.0002, trade_pen=-0.2400, dd_pen=-0.0004, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $102.95 (Ret: 0.11%, LogRet: 0.11%) | Credit: $102.95 (Growth: 0.11%) | Buys: 1, Sells: 3 | Fees: $0.05, Turnover(B/S): $25.71/$25.84 | Rewards Σ: pv_logret=0.0011, realized=0.0007, trade_pen=-0.2400, dd_pen=-0.0005, time_pen=0.0000
Model saved to saved_models/best_model.pth
New best validation cumulative PV return: 2.95%. Model saved.

Aggregated validation:
  Cumulative PV return: 2.95%
  Mean segment PV log-return: 0.49%
  Total buys: 6, Total sells: 12

=== Episode 4/50 ===
Training on slice 2025-07-01 10:38:00+00:00 -> 2025-07-04 13:46:00+00:00
                                                                                                                                                                                                                                                                                                                   
-> Validation phase…
                                                                                                                                                                                                                                                                                                                   
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $100.00 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $100.00 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $99.27 (Ret: -0.73%, LogRet: -0.73%) | Credit: $99.27 (Growth: -0.73%) | Buys: 1, Sells: 1 | Fees: $0.05, Turnover(B/S): $25.00/$24.29 | Rewards Σ: pv_logret=-0.0073, realized=-0.0035, trade_pen=-0.0800, dd_pen=-0.0023, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $98.85 (Ret: -0.42%, LogRet: -0.42%) | Credit: $98.85 (Growth: -0.42%) | Buys: 1, Sells: 2 | Fees: $0.05, Turnover(B/S): $24.82/$24.42 | Rewards Σ: pv_logret=-0.0042, realized=-0.0020, trade_pen=-0.1600, dd_pen=-0.0006, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $99.00 (Ret: 0.15%, LogRet: 0.15%) | Credit: $99.00 (Growth: 0.15%) | Buys: 1, Sells: 1 | Fees: $0.05, Turnover(B/S): $24.71/$24.89 | Rewards Σ: pv_logret=0.0015, realized=0.0009, trade_pen=-0.0800, dd_pen=-0.0014, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $99.34 (Ret: 0.34%, LogRet: 0.34%) | Credit: $99.34 (Growth: 0.34%) | Buys: 1, Sells: 1 | Fees: $0.05, Turnover(B/S): $24.75/$25.12 | Rewards Σ: pv_logret=0.0034, realized=0.0018, trade_pen=-0.0800, dd_pen=-0.0016, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: -0.66%
  Mean segment PV log-return: -0.11%
  Total buys: 4, Total sells: 5
=== Episode 5/50 ===
Training on slice 2025-06-22 19:57:00+00:00 -> 2025-06-25 16:42:00+00:00
                                                                                                                                                                                                                                                                                                                   
-> Validation phase…
                                                                                                                                                                                                                                                                                                                   
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $100.00 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $100.00 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $98.65 (Ret: -1.35%, LogRet: -1.35%) | Credit: $98.65 (Growth: -1.35%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.00/$48.70 | Rewards Σ: pv_logret=-0.0136, realized=-0.0065, trade_pen=-0.0800, dd_pen=-0.0048, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $98.07 (Ret: -0.59%, LogRet: -0.59%) | Credit: $98.07 (Growth: -0.59%) | Buys: 2, Sells: 3 | Fees: $0.20, Turnover(B/S): $98.66/$98.18 | Rewards Σ: pv_logret=-0.0059, realized=-0.0025, trade_pen=-0.3200, dd_pen=-0.0016, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $98.73 (Ret: 0.68%, LogRet: 0.68%) | Credit: $98.73 (Growth: 0.68%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $49.03/$49.75 | Rewards Σ: pv_logret=0.0067, realized=0.0036, trade_pen=-0.0800, dd_pen=-0.0008, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $99.34 (Ret: 0.62%, LogRet: 0.62%) | Credit: $99.34 (Growth: 0.62%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $49.36/$50.03 | Rewards Σ: pv_logret=0.0062, realized=0.0034, trade_pen=-0.0800, dd_pen=-0.0008, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: -0.66%
  Mean segment PV log-return: -0.11%
  Total buys: 5, Total sells: 6
=== Episode 6/50 ===
Training on slice 2025-06-01 22:06:00+00:00 -> 2025-06-05 02:21:00+00:00
                                                                                                                                                                                                                                                                                                                   
-> Validation phase…
                                                                                                                                                                                                                                                                                                                   
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $100.00 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $100.00 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $100.00 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $100.00 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $100.00 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $100.00 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: 0.00%
  Mean segment PV log-return: 0.00%
  Total buys: 0, Total sells: 0
=== Episode 7/50 ===
Training on slice 2025-06-13 15:22:00+00:00 -> 2025-06-16 05:05:00+00:00
                                                                                                                                                                                                                                                                                                                   
-> Validation phase…
                                                                                                                                                                                                                                                                                                                   
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $101.49 (Ret: 1.49%, LogRet: 1.49%) | Credit: $101.49 (Growth: 1.49%) | Buys: 1, Sells: 1 | Fees: $0.05, Turnover(B/S): $25.00/$26.51 | Rewards Σ: pv_logret=0.0148, realized=0.0076, trade_pen=-0.0800, dd_pen=-0.0019, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $101.74 (Ret: 0.25%, LogRet: 0.25%) | Credit: $101.74 (Growth: 0.25%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.74/$51.05 | Rewards Σ: pv_logret=0.0025, realized=0.0015, trade_pen=-0.0800, dd_pen=-0.0022, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $101.27 (Ret: -0.46%, LogRet: -0.46%) | Credit: $101.27 (Growth: -0.46%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.87/$50.45 | Rewards Σ: pv_logret=-0.0046, realized=-0.0021, trade_pen=-0.0800, dd_pen=-0.0050, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $100.77 (Ret: -0.49%, LogRet: -0.49%) | Credit: $100.77 (Growth: -0.49%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.63/$50.18 | Rewards Σ: pv_logret=-0.0050, realized=-0.0022, trade_pen=-0.0800, dd_pen=-0.0051, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $101.08 (Ret: 0.31%, LogRet: 0.31%) | Credit: $101.08 (Growth: 0.31%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.38/$50.75 | Rewards Σ: pv_logret=0.0031, realized=0.0018, trade_pen=-0.0800, dd_pen=-0.0028, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $101.77 (Ret: 0.69%, LogRet: 0.69%) | Credit: $101.77 (Growth: 0.69%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.54/$51.28 | Rewards Σ: pv_logret=0.0068, realized=0.0037, trade_pen=-0.0800, dd_pen=-0.0032, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: 1.77%
  Mean segment PV log-return: 0.30%
  Total buys: 6, Total sells: 6

=== Episode 8/50 ===
Training on slice 2025-06-14 10:11:00+00:00 -> 2025-06-17 00:48:00+00:00
                                                                                                                                                                                                                                                                                                                   
-> Validation phase…
                                                                                                                                                                                                                                                                                                                   
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $102.97 (Ret: 2.97%, LogRet: 2.97%) | Credit: $102.97 (Growth: 2.97%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.00/$53.03 | Rewards Σ: pv_logret=0.0293, realized=0.0151, trade_pen=-0.0800, dd_pen=-0.0037, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $103.23 (Ret: 0.25%, LogRet: 0.25%) | Credit: $103.23 (Growth: 0.25%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $51.49/$51.79 | Rewards Σ: pv_logret=0.0025, realized=0.0015, trade_pen=-0.0800, dd_pen=-0.0022, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $102.75 (Ret: -0.46%, LogRet: -0.46%) | Credit: $102.75 (Growth: -0.46%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $51.62/$51.19 | Rewards Σ: pv_logret=-0.0046, realized=-0.0021, trade_pen=-0.0800, dd_pen=-0.0050, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $102.24 (Ret: -0.49%, LogRet: -0.49%) | Credit: $102.24 (Growth: -0.49%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $51.38/$50.92 | Rewards Σ: pv_logret=-0.0050, realized=-0.0022, trade_pen=-0.0800, dd_pen=-0.0051, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $102.73 (Ret: 0.48%, LogRet: 0.48%) | Credit: $102.73 (Growth: 0.48%) | Buys: 2, Sells: 1 | Fees: $0.13, Turnover(B/S): $63.90/$64.46 | Rewards Σ: pv_logret=0.0048, realized=0.0027, trade_pen=-0.1600, dd_pen=-0.0030, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $103.44 (Ret: 0.69%, LogRet: 0.69%) | Credit: $103.44 (Growth: 0.69%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $51.37/$52.12 | Rewards Σ: pv_logret=0.0068, realized=0.0037, trade_pen=-0.0800, dd_pen=-0.0032, time_pen=0.0000
Model saved to saved_models/best_model.pth
New best validation cumulative PV return: 3.44%. Model saved.

Aggregated validation:
  Cumulative PV return: 3.44%
  Mean segment PV log-return: 0.57%
  Total buys: 7, Total sells: 6
=== Episode 9/50 ===
Training on slice 2025-06-17 00:16:00+00:00 -> 2025-06-20 02:05:00+00:00
                                                                                                                                                                                                                                                                                                                   
-> Validation phase…
                                                                                                                                                                                                                                                                                                                   
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $102.88 (Ret: 2.88%, LogRet: 2.88%) | Credit: $102.88 (Growth: 2.88%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.00/$52.93 | Rewards Σ: pv_logret=0.0283, realized=0.0146, trade_pen=-0.0800, dd_pen=-0.0034, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $103.13 (Ret: 0.25%, LogRet: 0.25%) | Credit: $103.13 (Growth: 0.25%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $51.44/$51.74 | Rewards Σ: pv_logret=0.0025, realized=0.0015, trade_pen=-0.0800, dd_pen=-0.0022, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $102.65 (Ret: -0.46%, LogRet: -0.46%) | Credit: $102.65 (Growth: -0.46%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $51.57/$51.14 | Rewards Σ: pv_logret=-0.0046, realized=-0.0021, trade_pen=-0.0800, dd_pen=-0.0050, time_pen=0.0000  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $102.15 (Ret: -0.49%, LogRet: -0.49%) | Credit: $102.15 (Growth: -0.49%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $51.33/$50.87 | Rewards Σ: pv_logret=-0.0050, realized=-0.0022, trade_pen=-0.0800, dd_pen=-0.0051, time_pen=0.0000  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $102.46 (Ret: 0.31%, LogRet: 0.31%) | Credit: $102.46 (Growth: 0.31%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $51.07/$51.44 | Rewards Σ: pv_logret=0.0031, realized=0.0018, trade_pen=-0.0800, dd_pen=-0.0028, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $103.16 (Ret: 0.69%, LogRet: 0.69%) | Credit: $103.16 (Growth: 0.69%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $51.23/$51.99 | Rewards Σ: pv_logret=0.0068, realized=0.0037, trade_pen=-0.0800, dd_pen=-0.0032, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: 3.16%
  Mean segment PV log-return: 0.53%
  Total buys: 6, Total sells: 6

=== Episode 10/50 ===
Training on slice 2025-06-24 08:09:00+00:00 -> 2025-06-27 01:13:00+00:00
Target network synchronized.                                                                                                                                                                                                                                                                                       

-> Validation phase…
                                                                                                                                                                                                                                                                                                                   
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $100.84 (Ret: 0.84%, LogRet: 0.84%) | Credit: $100.84 (Growth: 0.84%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.00/$50.89 | Rewards Σ: pv_logret=0.0084, realized=0.0045, trade_pen=-0.0800, dd_pen=-0.0019, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $100.85 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.85 (Growth: 0.00%) | Buys: 1, Sells: 1 | Fees: $0.05, Turnover(B/S): $25.21/$25.24 | Rewards Σ: pv_logret=0.0000, realized=0.0001, trade_pen=-0.0800, dd_pen=-0.0009, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $99.48 (Ret: -1.35%, LogRet: -1.35%) | Credit: $99.48 (Growth: -1.35%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.42/$49.11 | Rewards Σ: pv_logret=-0.0136, realized=-0.0065, trade_pen=-0.0800, dd_pen=-0.0048, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $99.12 (Ret: -0.36%, LogRet: -0.36%) | Credit: $99.12 (Growth: -0.36%) | Buys: 1, Sells: 1 | Fees: $0.05, Turnover(B/S): $24.87/$24.54 | Rewards Σ: pv_logret=-0.0036, realized=-0.0017, trade_pen=-0.0800, dd_pen=-0.0022, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $99.60 (Ret: 0.48%, LogRet: 0.48%) | Credit: $99.60 (Growth: 0.48%) | Buys: 2, Sells: 1 | Fees: $0.12, Turnover(B/S): $61.95/$62.49 | Rewards Σ: pv_logret=0.0048, realized=0.0027, trade_pen=-0.1600, dd_pen=-0.0030, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $100.28 (Ret: 0.69%, LogRet: 0.69%) | Credit: $100.28 (Growth: 0.69%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $49.80/$50.53 | Rewards Σ: pv_logret=0.0068, realized=0.0037, trade_pen=-0.0800, dd_pen=-0.0032, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: 0.28%
  Mean segment PV log-return: 0.05%
  Total buys: 7, Total sells: 6
=== Episode 11/50 ===
Training on slice 2025-06-16 03:31:00+00:00 -> 2025-06-19 01:36:00+00:00
                                                                                                                                                                                                                                                                                                                   
-> Validation phase…
                                                                                                                                                                                                                                                                                                                   
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $100.73 (Ret: 0.73%, LogRet: 0.73%) | Credit: $100.73 (Growth: 0.73%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.00/$50.79 | Rewards Σ: pv_logret=0.0073, realized=0.0039, trade_pen=-0.0800, dd_pen=-0.0023, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $100.73 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.73 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $99.62 (Ret: -1.10%, LogRet: -1.10%) | Credit: $99.62 (Growth: -1.10%) | Buys: 3, Sells: 1 | Fees: $0.14, Turnover(B/S): $72.40/$71.36 | Rewards Σ: pv_logret=-0.0111, realized=-0.0052, trade_pen=-0.2400, dd_pen=-0.0052, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $98.50 (Ret: -1.13%, LogRet: -1.13%) | Credit: $98.50 (Growth: -1.13%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $49.81/$48.73 | Rewards Σ: pv_logret=-0.0114, realized=-0.0054, trade_pen=-0.0800, dd_pen=-0.0021, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $99.11 (Ret: 0.62%, LogRet: 0.62%) | Credit: $99.11 (Growth: 0.62%) | Buys: 1, Sells: 1 | Fees: $0.20, Turnover(B/S): $98.50/$99.20 | Rewards Σ: pv_logret=0.0062, realized=0.0036, trade_pen=-0.0800, dd_pen=-0.0057, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $99.99 (Ret: 0.90%, LogRet: 0.90%) | Credit: $99.99 (Growth: 0.90%) | Buys: 2, Sells: 1 | Fees: $0.12, Turnover(B/S): $61.94/$62.89 | Rewards Σ: pv_logret=0.0089, realized=0.0048, trade_pen=-0.1600, dd_pen=-0.0034, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: -0.01%
  Mean segment PV log-return: 0.00%
  Total buys: 8, Total sells: 5
=== Episode 12/50 ===
Training on slice 2025-06-01 12:48:00+00:00 -> 2025-06-04 16:36:00+00:00
                                                                                                                                                                                                                                                                                                                   
-> Validation phase…
                                                                                                                                                                                                                                                                                                                   
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $100.47 (Ret: 0.47%, LogRet: 0.47%) | Credit: $100.47 (Growth: 0.47%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.00/$50.52 | Rewards Σ: pv_logret=0.0047, realized=0.0026, trade_pen=-0.0800, dd_pen=-0.0022, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $100.47 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.47 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $99.11 (Ret: -1.35%, LogRet: -1.35%) | Credit: $99.11 (Growth: -1.35%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $50.23/$48.92 | Rewards Σ: pv_logret=-0.0136, realized=-0.0065, trade_pen=-0.0800, dd_pen=-0.0048, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $98.03 (Ret: -1.09%, LogRet: -1.09%) | Credit: $98.03 (Growth: -1.09%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $49.55/$48.52 | Rewards Σ: pv_logret=-0.0109, realized=-0.0052, trade_pen=-0.0800, dd_pen=-0.0021, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $98.50 (Ret: 0.48%, LogRet: 0.48%) | Credit: $98.50 (Growth: 0.48%) | Buys: 2, Sells: 1 | Fees: $0.12, Turnover(B/S): $61.27/$61.80 | Rewards Σ: pv_logret=0.0048, realized=0.0027, trade_pen=-0.1600, dd_pen=-0.0030, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $99.17 (Ret: 0.69%, LogRet: 0.69%) | Credit: $99.17 (Growth: 0.69%) | Buys: 1, Sells: 1 | Fees: $0.10, Turnover(B/S): $49.25/$49.98 | Rewards Σ: pv_logret=0.0068, realized=0.0037, trade_pen=-0.0800, dd_pen=-0.0032, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: -0.83%
  Mean segment PV log-return: -0.13%
  Total buys: 6, Total sells: 5
```
Great question. You already have most of the plumbing for compounding growth and clean evaluation, but a few design choices are holding the agent back (and also explain the plotting behavior). Below is a prioritized list of what to fix, add, or modify — with the why, where, and what-for.

A. Things that look “broken” or misleading

Validation plots “stop” after episode 3
Symptom: After episode 3–4 you don’t see new validation charts.
Cause: You explicitly disable plotting after the first 3 episodes.
Where: main(), near the end of each training loop: if e < 3 and pv_hist: plot_results(...)
Fix:
Option 1: always plot latest validation if pv_hist: plot_results(...)
Option 2: only plot when a new best model is found: if cumulative_val_return > best_val_pv_return and pv_hist: plot_results(...)
Why: The code is doing exactly what you told it to; it is not a learning failure.
Validation plot alignment can be confusing
Symptom: Dotted vertical lines and curves don’t always feel aligned after concatenating segments.
Cause: plot_results chooses start_index = len(df) - len(portfolio_history), so you only see the last portion of val_data, while vlines are computed in full-data index space.
Where: plot_results()
Fix:
Option A: Always plot against the full validation index. Pre-pad your histories with NaNs so their length matches len(df) and keep vlines untouched.
Option B: Shift vline positions into the cropped window before plotting by subtracting start_index and dropping those outside the visible range.
Why: This prevents “drifting” visual artifacts when displaying concatenated segment results.
B. Why it sometimes “does nothing” after the first validation segment

Trade penalty is too large relative to achievable edge
Symptom: Several validations with 0 buys/sells for entire segments (e.g., Episode 6), or HOLD dominates.
Cause: cfg.TRADE_PENALTY = 0.08 is huge compared to typical per-step PV log returns and your 0.1% fee. The agent learns that not trading is safer.
Where: Config.TRADE_PENALTY
Fix:
Reduce drastically (e.g., 0.005–0.02), or set to 0.0 and let:
actual fees (already reflected in PV) and
drawdown penalty do the regularization.
Better: replace “per trade fixed penalty” with a “penalty proportional to actual fees or turnover” that you already track: reward -= fees_paid_step / self.start_credit
Why: You want the reward to reflect actual economics. A large fixed penalty dwarfs realistic signal and drives the agent to HOLD.
Cooldown measured in “decision ticks” explodes when decision_frequency is high
Symptom: After one trade it can sit for hours (10 ticks × 15 minutes = 150 minutes) unable to act, which looks like it “stops trying.”
Cause: cfg.ACTION_COOLDOWN is in decision ticks, but you validate with 15-minute ticks and train with 1-minute ticks (after ep 5). The same number of ticks means very different wall-clock time.
Where: TradingEnvironment.step()
Fix:
Make cooldown time-based:
Add ACTION_COOLDOWN_MINUTES (e.g., 30).
In run_episode, compute cooldown_ticks = ceil(ACTION_COOLDOWN_MINUTES / decision_frequency) and pass it into the environment.
Or store decision_frequency in env and compute cooldown per episode.
Why: A consistent wall-clock cooldown keeps the behavior stable across train/val frequencies.
Mismatch between training and validation decision cadence
Symptom: Behavior generalizes poorly across segments/frequency changes.
Cause: You train first at 1-minute decisions, then later at 15-minute. Validation is always at 15 minutes.
Where: get_decision_frequency_for_episode(); main() validation uses DECISION_FREQUENCY_LATER only.
Fix (pick one):
Train at the same frequency as you validate (15 minutes).
Or randomize the frequency per episode from a small set (e.g., {1, 5, 15}) and scale cooldown accordingly so the agent generalizes.
Why: Avoids learning a policy specialized to a cadence different from validation.
Memory is dominated by non-decision steps (forced HOLD)
Symptom: Agent learns to HOLD, Q-values don’t prioritize trade decisions.
Cause: You push transitions for every step, even when it’s not a decision tick (action forced to HOLD).
Where: run_episode(), the push happens unconditionally.
Fix:
Only push transitions on decision ticks: if not is_eval and decision_tick: memory.push(...)
If you want to retain all market evolution in the reward, accumulate rewards across non-decision steps and push the aggregated reward at the next decision tick (frame-skip style).
Why: Training on irrelevant “forced hold” frames dilutes the signal and biases the policy towards HOLD.
C. Make cumulative credit growth work end-to-end across validation segments

Don’t force liquidation between validation segments (unless it’s the final one)
Symptom: You want compounding, but positions are forcibly closed at the end of every segment. This removes the ability to carry profitable positions and “push harder” into the next one.
Cause: Environment forces liquidation on done.
Where: TradingEnvironment.step(), at “If done, force liquidation…”
Fix:
Add a flag force_liquidation_on_done to the environment.
In validate_in_segments:
For segments 1..N-1: set force_liquidation_on_done=False.
For the last segment: set True so you report clean credit/PV.
Carry both credit, holdings, and average_buy_price to the next segment via reset parameters.
Why: True compounding across segments requires carrying the actual position state, not just credit.
Carry average_buy_price into the next segment (if you carry holdings)
Symptom: If you decide to carry holdings across segments, unrealized_pnl_ratio is wrong at the start of next segment (average_buy_price resets to 0.0).
Cause: reset() sets average_buy_price=0.0 unconditionally.
Where: TradingEnvironment.reset()
Fix:
Extend reset() with average_buy_price parameter (default 0.0).
In run_episode/validate_in_segments, return avg_buy_price at the end and pass it into the next segment’s reset if holdings > 0.
Why: Features must reflect the true position. Otherwise the policy gets inconsistent state.
D. Reward shaping tweaks to encourage growth (without gaming)

Keep the log-PV return core; reduce extra penalties
You already use log_ret = log(PV_after/PV_before), which is exactly what you want for multiplicative growth.
Recommended changes:
Reduce/remodel TRADE_PENALTY (see item 3).
Optionally include fees directly (subtract fees_paid_step normalized by PV or start_credit; they’re already implicitly in PV but making them explicit sharpens the signal).
Keep drawdown penalty on increases only (good) but tune cfg.DRAWDOWN_PENALTY smaller initially (e.g., 0.005–0.01) if you see the agent getting too conservative.
Why: Let the log-return lead and avoid over-regularizing.
Use Huber loss instead of MSE
Where: DQNAgent.learn(), loss = (is_weights * F.mse_loss(...))
Fix:
Use F.smooth_l1_loss(q_values, expected_q_values, reduction='none') to make training less sensitive to outliers in TD errors.
Why: More stable gradients for noisy financial series.
E. Experience Replay (PER) correctness and stability

SumTree fallback bug can return non-leaf indices
Symptom: Potentially sampling an internal node if both children have zero priorities.
Cause: In SumTree._retrieve(), you return idx if both children are zero. idx may be an internal node; data_idx = idx - capacity + 1 becomes negative or wrong.
Where: class SumTree._retrieve()
Fix:
Ensure you always descend to leaves. If you encounter zeroed children, descend to the leftmost leaf in that subtree (or resample s). Alternatively, guard against total()==0 by early return and don’t call _retrieve in that case.
Why: Sampling internal nodes corrupts replay samples; can silently degrade learning.
PER beta annealing based on learn steps is fine, but ensure enough diversity early
Tip: When you switch to only pushing decision-tick transitions (item 6), you will have fewer, higher-quality samples. Consider reducing LEARN_START_MEMORY (e.g., to 500–1000) so learning starts earlier.
F. Action and trading microstructure

Minimum trade thresholds and exposure cap
Symptom: Small leftover positions can’t be sold, or the agent can’t resize quickly.
Cause:
MIN_TRADE_CREDIT_BUY=5.0 on $100 initial credit means buys smaller than $5 are disallowed.
MIN_TRADE_HOLDINGS_SELL=5.0 means you can’t sell positions worth <$5 except at forced liquidation.
Where: Config MIN_TRADE_*
Fix:
For $100 initial credit, consider MIN_TRADE_CREDIT_BUY=2.0 and MIN_TRADE_HOLDINGS_SELL=2.0 to give finer control without micromanaging.
Exposure cap at 90% is OK, but if you increase confidence later, consider 95%.
Why: Too-large minimums reduce the agent’s ability to rebalance, especially with cooldown.
Action set and mask
Your mask uses credit_ratio and holdings_ratio thresholds of 1%. That’s fine, but consider lowering to 0.005 to avoid “near-zero” positions that cannot be trimmed due to the sell threshold.
G. Training cadence and target updates

Target network updates tied to episodes
Symptom: Instability or stale targets when episodes vary in length.
Cause: TARGET_UPDATE_FREQ per episode, not per training step.
Where: main() after each episode
Fix:
Update target every fixed number of learn() calls (e.g., every 1000–5000 gradient steps), or switch to soft updates: target = tau*policy + (1-tau)*target with tau ~ 0.005
Why: More consistent stabilization across episodes.
H. Diagnostics to understand “should have sold but didn’t”

Add decision diagnostics during validation
What to log per decision tick:
Q-values per action (top-2 actions and margins).
Whether cooldown forced HOLD.
Immediate k-step forward returns (e.g., 30, 60, 120 minutes) after a SELL to see if sells were timely.
Share of decision ticks disabled by cooldown.
Why: You’ll immediately see if HOLD was chosen because of cooldown, action mask, or policy preference.
Baselines
Compare against:
Buy-and-hold (credit not changing, holdings initial invested),
SMA cross strategy,
Random policy (with same trade constraints).
Why: To quantify if the RL policy is improving on simple baselines.
I. Minor issues and warnings

torch.ger is deprecated
Where: NoisyLinear.reset_noise(): self.weight_epsilon.copy_(torch.ger(epsilon_out, epsilon_in))
Fix: Replace with torch.outer(epsilon_out, epsilon_in) for newer PyTorch.
Why: Avoid deprecation warnings and ensure forward-compatibility.
Plotly in non-notebook environments
fig.show() may not display in some environments. Consider:
fig.write_html("val_episode_X.html", auto_open=True) for reliable viewing.
Why: Removes environment-dependent behavior.
Numerical stability in normalization
You already guard std with 1e-7. Good. Keep it.
J. Concrete parameter changes to try immediately

Reward/penalties:
TRADE_PENALTY: 0.0 to 0.02 (start with 0.01)
DRAWDOWN_PENALTY: 0.005–0.01
Add explicit: reward -= fees_paid_step / max(self.start_credit, 1e-8)
Frequency/cooldown:
Train and validate at 15-minute cadence, or randomize among {5, 15}.
Set ACTION_COOLDOWN_MINUTES = 30 and compute ticks per episode.
Replay:
Push transitions only on decision ticks.
LEARN_START_MEMORY: 500–1000
Loss:
Switch to Huber (smooth_l1).
Trading thresholds:
MIN_TRADE_CREDIT_BUY = 2.0
MIN_TRADE_HOLDINGS_SELL = 2.0
K. Optional but impactful improvements

N-step returns (e.g., n=3–5) to better propagate outcomes through sparse trading decisions.
Distributional Q-learning (QR-DQN) for heavy-tailed outcomes common in markets.
Learning-rate scheduler (reduce on plateau of validation cumulative PV return).
Summary: The biggest practical blockers to “pushing harder” and compounding are (1) the hard per-trade penalty being too large, (2) cooldown scaled in ticks not minutes, (3) flooding replay with non-decision HOLD transitions, and (4) forced liquidation between validation segments. Fix those first. Then align train/val decision frequency, reduce minimum trade thresholds, switch to Huber loss, and add diagnostics. Also, the plotting “stop” is just the if e < 3 gate — not a learning issue. Implementing these changes should make the agent trade more, carry gains across segments, and focus on growing credit multiplicatively.

I need you to take all this recomedations and things to apply to the code, in order to fix it. Please apply all of this things, to make sure that now the code will work. Please write the whole code with the fixes integrated in it. Make sure to plot the plots correctly each episode, with pyplot in the browser still.


I dont need you to write the whole fixed code, but to generate a list of things to fix, add or modify. Please be very descriptive of why , where and what for is things. Please make somethign to make the credit grow in a cumulative manner and start making grow the money. Also, please decribe the errors or warnings you find and how should the be addressed.





















































































Look, I have this code, which Im trying to optimize to make it make good trades and make credit or capital grow. Please help me fix it. As you can see, it did pretty well on Episode 9 - 12, at 13 it started to move back down. Why is it no keeping the results or getting even better? What is happening. I think the code it is on a good track, Just I dont understand why it goes back to bad results and not making almost any money:
```

import os
import math
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ta import add_all_ta_features
from tqdm import tqdm

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

    # Target network update every N learn() steps
    TARGET_UPDATE_EVERY_STEPS = 2000

    # Replay and learning cadence
    MEMORY_CAPACITY = 100000
    LEARN_START_MEMORY = 1000  # start learning once memory >= this
    LEARN_EVERY = 1            # learn every step once above threshold

    # Exploration in validation (set to 0 for deterministic model selection)
    EVAL_EPSILON = 0.0

    # Decision frequency in minutes (we train with a random choice from this set)
    TRAIN_DECISION_FREQUENCIES = [5, 15]  # minutes per decision
    VAL_DECISION_FREQUENCY = 15           # minutes per decision

    # Cooldown measured in minutes (consistent across decision cadences)
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
    MIN_TRADE_CREDIT_BUY = 2.0      # allow finer buys
    MIN_TRADE_HOLDINGS_SELL = 2.0   # in currency terms

    # Reward shaping coefficients
    REWARD_REALIZED = 0.5           # bonus for realized pnl (scaled by start credit)
    TRADE_PENALTY = 0.01            # small fixed penalty per executed trade
    TIME_PENALTY = 0.0              # off
    DRAWDOWN_PENALTY = 0.01         # penalize increases in drawdown
    FEES_PENALTY_COEF = 1.0         # subtract fees explicitly (normalized by start credit)

    # --- Prioritized Experience Replay (PER) ---
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_END = 1.0
    PER_BETA_ANNEAL_STEPS = 100000

    # --- Validation ---
    VAL_SEGMENT_MINUTES = 2 * 24 * 60  # 2 days per segment

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
        # Leaf node
        if left >= len(self.tree):
            return idx
        # If both children priorities are zero, descend to the leftmost leaf deterministically
        if self.tree[left] == 0.0 and self.tree[right] == 0.0:
            # Walk down to the leftmost leaf
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
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        if data_idx < 0 or data_idx >= self.capacity:
            # Safety fallback: sample from a valid filled index
            data_idx = (self.write - 1) % max(self.n_entries, 1)
            idx = data_idx + self.capacity - 1
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
        total_p = self.tree.total()
        segment = total_p / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(max(p, 1e-8))
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = np.array(priorities, dtype=np.float64) / max(self.tree.total(), 1e-8)
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= (is_weight.max() + 1e-8)
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
        market_feature_dim = state_dim - 4
        self.attention = SharedSelfAttention(market_feature_dim, cfg.ATTENTION_DIM, cfg.ATTENTION_HEADS)
        self.fc1 = nn.Linear(cfg.ATTENTION_DIM + 4, cfg.FC_UNITS_1)
        self.value_stream = nn.Sequential(
            NoisyLinear(cfg.FC_UNITS_1, cfg.FC_UNITS_2), nn.ReLU(),
            NoisyLinear(cfg.FC_UNITS_2, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(cfg.FC_UNITS_1, cfg.FC_UNITS_2), nn.ReLU(),
            NoisyLinear(cfg.FC_UNITS_2, action_dim)
        )

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
        if credit_ratio < 0.005:
            mask[1] = False
            mask[2] = False
            if self.action_dim >= 7:
                mask[6] = False
        if holdings_ratio < 0.005:
            mask[3] = False
            mask[4] = False
            if self.action_dim >= 6:
                mask[5] = False
        return mask

    def act(self, state, is_eval=False):
        if not is_eval:
            self.total_steps += 1

        state_np = np.array(state, dtype=np.float32)
        mask = self._action_mask_from_state(state_np)
        valid_actions = np.where(mask)[0].tolist()
        if len(valid_actions) == 0:
            return 0  # fallback HOLD

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

        # Apply mask
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
        dones_tensor = torch.tensor(dones, device=device, dtype=torch.float32)

        if any(non_final_mask.cpu().numpy()):
            non_final_next_states = torch.FloatTensor(np.array([s for s in next_states if s is not None])).to(device)
        else:
            non_final_next_states = torch.empty((0,) + state_batch.shape[1:], device=device)

        next_q_values = torch.zeros(cfg.BATCH_SIZE, device=device)
        if non_final_next_states.size(0) > 0:
            with torch.no_grad():
                next_actions = self.policy_net(non_final_next_states).argmax(1).unsqueeze(1)
                next_q_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze(1)

        expected_q_values = reward_batch + cfg.GAMMA * (1.0 - dones_tensor) * next_q_values
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        td_errors = (expected_q_values - q_values).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # Huber loss (smooth L1)
        loss = (is_weights * F.smooth_l1_loss(q_values, expected_q_values, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Step-based target network synchronization
        if self.learn_step_counter % cfg.TARGET_UPDATE_EVERY_STEPS == 0:
            self.update_target_network()

        # Refresh NoisyNet noise post-update for continued exploration
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("Target network synchronized.")

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
    def __init__(self, data, initial_credit=cfg.INITIAL_CREDIT, window_size=cfg.WINDOW_SIZE,
                 decision_frequency_minutes=1, force_liquidation_on_done=True):
        self.data = data
        self.normalized_data = data.drop(columns=["Original_Close"])
        self.initial_credit = initial_credit
        self.window_size = window_size
        self.n_features = self.normalized_data.shape[1] + 4
        self.action_space = [0, 1, 2, 3, 4, 5, 6]
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
        # Cooldown only on decision ticks
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
        buy_notional = 0.0
        sell_notional = 0.0
        fees_step = 0.0

        if buy_fraction > 0:
            # Prevent extreme over-investment (cap exposure ~90%)
            portfolio_value = self.credit + self.holdings * current_price
            current_exposure = (self.holdings * current_price) / portfolio_value if portfolio_value > 0 else 0.0
            if current_exposure < 0.90:  # soft cap
                investment = self.credit * buy_fraction
                if investment > cfg.MIN_TRADE_CREDIT_BUY:
                    buy_fee = investment * cfg.FEE
                    buy_amount_asset = (investment * (1 - cfg.FEE)) / current_price
                    total_cost = (self.average_buy_price * self.holdings) + investment
                    self.holdings += buy_amount_asset
                    self.credit -= investment
                    self.average_buy_price = total_cost / self.holdings if self.holdings > 0 else 0.0
                    self.trades.append({"step": self.current_step, "type": "buy", "price": current_price, "amount": buy_amount_asset})
                    trade_executed = True
                    buy_notional = investment
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

        # Optional forced liquidation on done (e.g., end of final segment)
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

        # Reward calculation (log PV return-centric; penalize drawdown increases only)
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
        # Core: log return on portfolio value (encourages multiplicative growth)
        log_ret = math.log(max(pv_after, 1e-8) / max(pv_before, 1e-8))

        # Realized PnL shaping (small)
        realized_component = cfg.REWARD_REALIZED * (realized_pnl / max(self.start_credit, 1e-8)) if realized_pnl != 0 else 0.0

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

        # Explicit fees penalty (normalized by initial credit)
        fees_pen = -cfg.FEES_PENALTY_COEF * (fees_step / max(self.start_credit, 1e-8)) if fees_step > 0 else 0.0

        comps = {
            "pv_logret": log_ret,
            "realized": realized_component,
            "trade_pen": trade_pen,
            "time_pen": time_pen,
            "dd_pen": dd_pen,
            "fees_pen": fees_pen
        }
        reward = sum(comps.values())
        return reward, comps

# -------------------------------------------------
# 5. PLOTTING
# -------------------------------------------------
def plot_results(df, episode, portfolio_history, credit_history, holdings_history, trades, plot_title_prefix="", segment_boundaries=None, save_name=None):
    """
    If histories are length == len(df), they are assumed aligned with df.index.
    Otherwise, we crop to the last len(history) points.
    """
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

    # Price
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["Original_Close"], mode="lines", name="Price", line=dict(color="lightgrey")),
        row=1, col=1
    )

    # Trades (assumed to use df's global index)
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

    # PV, Credit, Holdings
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["portfolio_value"], mode="lines", name="PV"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["credit"], mode="lines", name="Credit"), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["holdings_value"], mode="lines", name="Holdings"), row=4, col=1)

    # Segment boundaries are global index positions
    if segment_boundaries:
        for boundary in segment_boundaries:
            if 0 <= boundary < len(df.index):
                fig.add_vline(x=df.index[boundary], line_width=1, line_dash="dash", line_color="black", opacity=0.4)

    fig.update_layout(height=1000, title_text=f"{plot_title_prefix} Results (Episode {episode})", showlegend=False)

    # Show and save
    fig.show()
    if save_name is None:
        save_name = f"{plot_title_prefix.lower()}_episode_{episode}.html"
    fig.write_html(os.path.join("plots", save_name), auto_open=False)

# -------------------------------------------------
# 6. TRAINING AND EVALUATION LOGIC
# -------------------------------------------------
def run_episode(env, agent, data, is_eval=False, initial_credit=None, initial_holdings=0.0, initial_avg_buy_price=0.0, decision_frequency_minutes=1, force_liq_on_done=True):
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
    # Reward component and cost aggregators (per episode/segment)
    comp_sums = {"pv_logret": 0.0, "realized": 0.0, "trade_pen": 0.0, "time_pen": 0.0, "dd_pen": 0.0, "fees_pen": 0.0}
    fees_total = 0.0
    buy_notional_total = 0.0
    sell_notional_total = 0.0

    # Transition aggregation: only push on decision ticks, aggregate rewards in between (frame-skip style)
    last_decision_state = None
    last_decision_action = None
    aggregated_reward = 0.0

    decision_period = max(1, int(decision_frequency_minutes))  # minutes per decision

    for step in range(total_steps):
        decision_tick = (step % decision_period == 0)

        # On a new decision tick, before choosing an action, push the previous aggregated transition
        if decision_tick:
            if not is_eval and last_decision_state is not None:
                agent.memory.push(last_decision_state, last_decision_action, aggregated_reward, state, False)
            # Choose action at decision tick
            action = agent.act(state, is_eval)
            action_counts[action] += 1
            # Store new decision context
            last_decision_state = state
            last_decision_action = action
            aggregated_reward = 0.0
        else:
            action = 0  # HOLD if not a decision tick
            action_counts[action] += 1

        next_state, reward, done, info = env.step(action, decision_tick=decision_tick)
        aggregated_reward += reward  # accumulate rewards across steps until next decision

        # Learning: only at training time; we learn every step (using aggregated transitions pushed at decision ticks)
        if not is_eval and len(agent.memory) >= cfg.LEARN_START_MEMORY and (step % cfg.LEARN_EVERY == 0):
            agent.learn()

        state = next_state

        # Append histories including the final step (post-liquidation if any)
        portfolio_values.append(info["portfolio_value"])
        credits.append(info["credit"])
        current_index = min(env.current_step, len(data) - 1)
        current_price = data["Original_Close"].iloc[current_index]
        holdings_values.append(info["holdings"] * current_price)
        indices_history.append(current_index)

        # Aggregate reward components and costs (for validation diagnostics)
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
        agent.memory.push(last_decision_state, last_decision_action, aggregated_reward, None, True)

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

    # Prepare full-length histories aligned to full_val_data index
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

        # For all but the last segment, do not force liquidation; for the last, force it
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

        # Baselines for metrics at segment start
        start_price = segment_data["Original_Close"].iloc[val_env.window_size]
        initial_pv_seg = current_credit + current_holdings * start_price
        initial_credit_seg = current_credit

        seg_return = ((final_pv - initial_pv_seg) / initial_pv_seg * 100) if initial_pv_seg > 0 else 0.0
        credit_growth = ((final_credit - initial_credit_seg) / initial_credit_seg * 100) if initial_credit_seg > 0 else 0.0

        # Convert summed pv_logret to percentage for segment
        seg_logret_pct = (math.exp(comp_sums["pv_logret"]) - 1.0) * 100.0

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

    # Segment boundary vlines: at each segment's first tradable index (start + window_size)
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

        # Randomize decision frequency during training for generalization
        train_decision_freq = random.choice(cfg.TRAIN_DECISION_FREQUENCIES)

        train_env = TradingEnvironment(episode_data, window_size=cfg.WINDOW_SIZE, decision_frequency_minutes=train_decision_freq)
        (
            _pv, _cred, _hold, _idx, _trades,
            _fpv, _fcred, _fhold, _favgbp, action_counts, _c_sums, _fees, _buy_not, _sell_not
        ) = run_episode(
            train_env, agent, episode_data, is_eval=False, initial_credit=cfg.INITIAL_CREDIT,
            decision_frequency_minutes=train_decision_freq, force_liq_on_done=True
        )

        print("\n-> Validation phase…")
        val_freq = cfg.VAL_DECISION_FREQUENCY  # use stable freq for validation
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
            # Selection metric: cumulative PV return across validation (final credit vs initial)
            total_final_credit = seg_metrics[-1]["final_credit"]
            total_initial_credit = cfg.INITIAL_CREDIT
            cumulative_val_return = (total_final_credit - total_initial_credit) / total_initial_credit * 100.0

            # Mean segment PV log-return (%), computed from per-seg logret_pct
            mean_seg_logret_pct = float(np.mean([m["logret_pct"] for m in seg_metrics]))

            improved = cumulative_val_return > best_val_pv_return
            if improved:
                best_val_pv_return = cumulative_val_return
                agent.save_model(f"saved_models/best_model.pth")
                print(f"New best validation cumulative PV return: {best_val_pv_return:.2f}%. Model saved.")

            total_buys = sum(m["buys"] for m in seg_metrics)
            total_sells = sum(m["sells"] for m in seg_metrics)

            print("\nAggregated validation:")
            print(f"  Cumulative PV return: {cumulative_val_return:.2f}%")
            print(f"  Mean segment PV log-return: {mean_seg_logret_pct:.2f}%")
            print(f"  Total buys: {total_buys}, Total sells: {total_sells}")

            # Plot full aligned histories each episode
            if any(np.isfinite(v) for v in pv_hist):
                # Ensure plotting arrays are aligned length == len(val_data)
                plot_results(
                    val_data, e + 1, pv_hist, credit_hist, hold_hist, trades,
                    "Validation", segment_boundaries=seg_boundaries, save_name=f"val_episode_{e+1}.html"
                )

    print("\n=== FINAL TEST ON UNSEEN DATA ===")
    try:
        agent.load_model("saved_models/best_model.pth")
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

    # Build aligned arrays for test plotting
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
```

```
=== Episode 1/50 ===
Training on slice 2025-05-19 10:09:00+00:00 -> 2025-05-22 06:03:00+00:00
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $100.00 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, fees_pen=0.0000, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $100.00 (Ret: 0.00%, LogRet: 0.00%) | Credit: $100.00 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, fees_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $99.21 (Ret: -0.79%, LogRet: -0.79%) | Credit: $78.64 (Growth: -21.36%) | Buys: 1, Sells: 3 | Fees: $0.08, Turnover(B/S): $50.00/$28.67 | Rewards Σ: pv_logret=-0.0079, realized=-0.0012, trade_pen=-0.0400, dd_pen=-0.0009, fees_pen=-0.0008, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $97.41 (Ret: -1.94%, LogRet: -1.94%) | Credit: $5.85 (Growth: -92.55%) | Buys: 9, Sells: 6 | Fees: $0.31, Turnover(B/S): $192.25/$119.59 | Rewards Σ: pv_logret=-0.0196, realized=-0.0076, trade_pen=-0.1500, dd_pen=-0.0023, fees_pen=-0.0040, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $97.65 (Ret: 0.14%, LogRet: 0.14%) | Credit: $0.00 (Growth: -100.00%) | Buys: 7, Sells: 7 | Fees: $0.34, Turnover(B/S): $173.92/$168.23 | Rewards Σ: pv_logret=0.0014, realized=-0.1198, trade_pen=-0.1400, dd_pen=-0.0026, fees_pen=-0.0584, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $97.33 (Ret: 0.37%, LogRet: 0.37%) | Credit: $97.33 (Growth: 0.00%) | Buys: 10, Sells: 12 | Fees: $0.58, Turnover(B/S): $240.27/$337.94 | Rewards Σ: pv_logret=0.0036, realized=41413802.9273, trade_pen=-0.2100, dd_pen=-0.0027, fees_pen=-57821293.2776, time_pen=0.0000
Model saved to saved_models/best_model.pth
New best validation cumulative PV return: -2.67%. Model saved.

Aggregated validation:
  Cumulative PV return: -2.67%
  Mean segment PV log-return: -0.37%
  Total buys: 27, Total sells: 28

=== Episode 2/50 ===
Training on slice 2025-06-14 10:08:00+00:00 -> 2025-06-17 00:45:00+00:00
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $102.10 (Ret: 2.10%, LogRet: 2.10%) | Credit: $98.81 (Growth: -1.19%) | Buys: 16, Sells: 27 | Fees: $1.11, Turnover(B/S): $554.98/$554.34 | Rewards Σ: pv_logret=0.0208, realized=0.0133, trade_pen=-0.4300, dd_pen=-0.0011, fees_pen=-0.0111, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $101.05 (Ret: -1.03%, LogRet: -1.03%) | Credit: $38.01 (Growth: -61.53%) | Buys: 26, Sells: 22 | Fees: $1.24, Turnover(B/S): $651.17/$590.97 | Rewards Σ: pv_logret=-0.0103, realized=-0.0007, trade_pen=-0.4800, dd_pen=-0.0011, fees_pen=-0.0126, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $99.44 (Ret: -1.91%, LogRet: -1.91%) | Credit: $96.31 (Growth: 153.35%) | Buys: 16, Sells: 21 | Fees: $1.41, Turnover(B/S): $673.48/$732.50 | Rewards Σ: pv_logret=-0.0193, realized=-0.0161, trade_pen=-0.3700, dd_pen=-0.0012, fees_pen=-0.0370, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $96.82 (Ret: -2.65%, LogRet: -2.65%) | Credit: $5.96 (Growth: -93.81%) | Buys: 9, Sells: 13 | Fees: $0.58, Turnover(B/S): $337.41/$247.31 | Rewards Σ: pv_logret=-0.0268, realized=-0.0036, trade_pen=-0.2200, dd_pen=-0.0022, fees_pen=-0.0061, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $97.67 (Ret: 0.77%, LogRet: 0.77%) | Credit: $5.96 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0077, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0026, fees_pen=0.0000, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $98.38 (Ret: 1.38%, LogRet: 1.38%) | Credit: $98.38 (Growth: 1550.04%) | Buys: 0, Sells: 1 | Fees: $0.09, Turnover(B/S): $0.00/$92.51 | Rewards Σ: pv_logret=0.0137, realized=-0.0012, trade_pen=0.0000, dd_pen=-0.0030, fees_pen=-0.0155, time_pen=0.0000
Model saved to saved_models/best_model.pth
New best validation cumulative PV return: -1.62%. Model saved.

Aggregated validation:
  Cumulative PV return: -1.62%
  Mean segment PV log-return: -0.22%
  Total buys: 67, Total sells: 84
=== Episode 3/50 ===
Training on slice 2025-05-29 23:45:00+00:00 -> 2025-06-02 02:46:00+00:00
TRAINING:   5%|████████████▎                                                                                                                                                                                                                                                          | 160/3419 [00:06<02:24, 22.58it/s]Target network synchronized.
TRAINING:  63%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                | 2158/3419 [01:32<00:54, 22.94it/s]Target network synchronized.
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $101.79 (Ret: 1.79%, LogRet: 1.79%) | Credit: $101.79 (Growth: 1.79%) | Buys: 24, Sells: 18 | Fees: $2.01, Turnover(B/S): $1003.51/$1006.30 | Rewards Σ: pv_logret=0.0177, realized=0.0140, trade_pen=-0.4200, dd_pen=-0.0021, fees_pen=-0.0201, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $99.72 (Ret: -2.03%, LogRet: -2.03%) | Credit: $49.98 (Growth: -50.90%) | Buys: 35, Sells: 25 | Fees: $2.83, Turnover(B/S): $1440.84/$1390.42 | Rewards Σ: pv_logret=-0.0206, realized=-0.0022, trade_pen=-0.6000, dd_pen=-0.0010, fees_pen=-0.0278, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $97.84 (Ret: -2.13%, LogRet: -2.13%) | Credit: $97.84 (Growth: 95.77%) | Buys: 33, Sells: 27 | Fees: $2.87, Turnover(B/S): $1410.59/$1459.92 | Rewards Σ: pv_logret=-0.0215, realized=-0.0065, trade_pen=-0.6000, dd_pen=-0.0019, fees_pen=-0.0574, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $94.65 (Ret: -3.26%, LogRet: -3.26%) | Credit: $13.89 (Growth: -85.81%) | Buys: 27, Sells: 28 | Fees: $2.03, Turnover(B/S): $1058.17/$975.19 | Rewards Σ: pv_logret=-0.0332, realized=-0.0082, trade_pen=-0.5500, dd_pen=-0.0022, fees_pen=-0.0208, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $95.46 (Ret: 0.75%, LogRet: 0.75%) | Credit: $6.94 (Growth: -50.00%) | Buys: 1, Sells: 0 | Fees: $0.01, Turnover(B/S): $6.94/$0.00 | Rewards Σ: pv_logret=0.0075, realized=0.0000, trade_pen=-0.0100, dd_pen=-0.0026, fees_pen=-0.0005, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $96.14 (Ret: 1.37%, LogRet: 1.37%) | Credit: $96.14 (Growth: 1284.50%) | Buys: 0, Sells: 1 | Fees: $0.09, Turnover(B/S): $0.00/$89.28 | Rewards Σ: pv_logret=0.0136, realized=0.0693, trade_pen=0.0000, dd_pen=-0.0029, fees_pen=-0.0129, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: -3.86%
  Mean segment PV log-return: -0.59%
  Total buys: 120, Total sells: 99
=== Episode 5/50 ===
Training on slice 2025-06-04 06:15:00+00:00 -> 2025-06-07 06:19:00+00:00
TRAINING:  39%|█████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                | 1322/3419 [01:00<01:34, 22.07it/s]
Target network synchronized.
TRAINING:  97%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍       | 3320/3419 [02:31<00:04, 22.29it/s]
Target network synchronized.
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $99.24 (Ret: -0.76%, LogRet: -0.76%) | Credit: $6.22 (Growth: -93.78%) | Buys: 22, Sells: 28 | Fees: $1.68, Turnover(B/S): $884.36/$791.36 | Rewards Σ: pv_logret=-0.0076, realized=0.0012, trade_pen=-0.5000, dd_pen=-0.0019, fees_pen=-0.0168, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $99.41 (Ret: 0.38%, LogRet: 0.38%) | Credit: $6.18 (Growth: -0.51%) | Buys: 4, Sells: 1 | Fees: $0.19, Turnover(B/S): $92.75/$92.82 | Rewards Σ: pv_logret=0.0038, realized=-0.0330, trade_pen=-0.0500, dd_pen=-0.0020, fees_pen=-0.0299, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $97.85 (Ret: -2.05%, LogRet: -2.05%) | Credit: $6.10 (Growth: -1.43%) | Buys: 24, Sells: 13 | Fees: $1.55, Turnover(B/S): $772.80/$773.48 | Rewards Σ: pv_logret=-0.0207, realized=-0.0517, trade_pen=-0.3700, dd_pen=-0.0032, fees_pen=-0.2501, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $97.94 (Ret: -0.47%, LogRet: -0.47%) | Credit: $48.86 (Growth: 701.57%) | Buys: 20, Sells: 19 | Fees: $1.47, Turnover(B/S): $712.53/$756.05 | Rewards Σ: pv_logret=-0.0048, realized=0.0777, trade_pen=-0.3900, dd_pen=-0.0031, fees_pen=-0.2409, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $95.35 (Ret: -2.70%, LogRet: -2.70%) | Credit: $47.77 (Growth: -2.23%) | Buys: 30, Sells: 30 | Fees: $2.90, Turnover(B/S): $1448.59/$1448.95 | Rewards Σ: pv_logret=-0.0274, realized=-0.0075, trade_pen=-0.6000, dd_pen=-0.0010, fees_pen=-0.0593, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $93.47 (Ret: -1.64%, LogRet: -1.64%) | Credit: $93.47 (Growth: 95.67%) | Buys: 22, Sells: 23 | Fees: $2.12, Turnover(B/S): $1038.85/$1085.63 | Rewards Σ: pv_logret=-0.0165, realized=-0.0103, trade_pen=-0.4500, dd_pen=-0.0009, fees_pen=-0.0445, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: -6.53%
  Mean segment PV log-return: -1.21%
  Total buys: 122, Total sells: 114
=== Episode 6/50 ===
Training on slice 2025-07-01 00:55:00+00:00 -> 2025-07-04 04:08:00+00:00
TRAINING:  56%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                    | 1901/3419 [01:25<01:10, 21.52it/s]
Target network synchronized.
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $99.70 (Ret: -0.30%, LogRet: -0.30%) | Credit: $99.70 (Growth: -0.30%) | Buys: 6, Sells: 16 | Fees: $0.49, Turnover(B/S): $243.61/$243.55 | Rewards Σ: pv_logret=-0.0030, realized=-0.0003, trade_pen=-0.2200, dd_pen=-0.0004, fees_pen=-0.0049, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $99.72 (Ret: 0.02%, LogRet: 0.02%) | Credit: $99.72 (Growth: 0.02%) | Buys: 1, Sells: 4 | Fees: $0.05, Turnover(B/S): $24.92/$24.97 | Rewards Σ: pv_logret=0.0002, realized=0.0002, trade_pen=-0.0500, dd_pen=-0.0000, fees_pen=-0.0005, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $98.45 (Ret: -1.27%, LogRet: -1.27%) | Credit: $95.36 (Growth: -4.37%) | Buys: 10, Sells: 31 | Fees: $0.94, Turnover(B/S): $470.88/$466.99 | Rewards Σ: pv_logret=-0.0128, realized=-0.0041, trade_pen=-0.4100, dd_pen=-0.0007, fees_pen=-0.0094, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $97.04 (Ret: -1.45%, LogRet: -1.45%) | Credit: $97.04 (Growth: 1.76%) | Buys: 14, Sells: 25 | Fees: $0.98, Turnover(B/S): $489.16/$491.33 | Rewards Σ: pv_logret=-0.0146, realized=-0.0047, trade_pen=-0.3900, dd_pen=-0.0006, fees_pen=-0.0103, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $95.67 (Ret: -1.42%, LogRet: -1.42%) | Credit: $95.67 (Growth: -1.42%) | Buys: 30, Sells: 30 | Fees: $1.44, Turnover(B/S): $722.53/$721.88 | Rewards Σ: pv_logret=-0.0143, realized=-0.0034, trade_pen=-0.6000, dd_pen=-0.0005, fees_pen=-0.0149, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $94.39 (Ret: -1.34%, LogRet: -1.34%) | Credit: $94.39 (Growth: -1.34%) | Buys: 22, Sells: 23 | Fees: $1.04, Turnover(B/S): $522.12/$521.36 | Rewards Σ: pv_logret=-0.0135, realized=-0.0040, trade_pen=-0.4500, dd_pen=-0.0005, fees_pen=-0.0109, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: -5.61%
  Mean segment PV log-return: -0.96%
  Total buys: 83, Total sells: 129
=== Episode 8/50 ===
Training on slice 2025-06-17 04:25:00+00:00 -> 2025-06-20 07:01:00+00:00
TRAINING:  31%|█████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                                                                    | 1064/3419 [00:45<01:38, 23.98it/s]
Target network synchronized.
TRAINING:  90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                           | 3065/3419 [02:09<00:14, 24.46it/s]
Target network synchronized.
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $102.29 (Ret: 2.29%, LogRet: 2.29%) | Credit: $17.80 (Growth: -82.20%) | Buys: 6, Sells: 0 | Fees: $0.08, Turnover(B/S): $82.20/$0.00 | Rewards Σ: pv_logret=0.0227, realized=0.0000, trade_pen=-0.0600, dd_pen=-0.0021, fees_pen=-0.0008, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $102.02 (Ret: -0.08%, LogRet: -0.08%) | Credit: $102.02 (Growth: 473.21%) | Buys: 0, Sells: 1 | Fees: $0.08, Turnover(B/S): $0.00/$84.31 | Rewards Σ: pv_logret=-0.0008, realized=0.0591, trade_pen=-0.0100, dd_pen=-0.0000, fees_pen=-0.0047, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $99.81 (Ret: -2.17%, LogRet: -2.17%) | Credit: $17.71 (Growth: -82.64%) | Buys: 20, Sells: 10 | Fees: $1.00, Turnover(B/S): $543.35/$459.50 | Rewards Σ: pv_logret=-0.0219, realized=-0.0101, trade_pen=-0.3000, dd_pen=-0.0017, fees_pen=-0.0098, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $101.02 (Ret: 0.71%, LogRet: 0.71%) | Credit: $75.62 (Growth: 327.01%) | Buys: 24, Sells: 14 | Fees: $1.07, Turnover(B/S): $504.84/$563.32 | Rewards Σ: pv_logret=0.0070, realized=0.0533, trade_pen=-0.3800, dd_pen=-0.0020, fees_pen=-0.0603, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $102.16 (Ret: 1.10%, LogRet: 1.10%) | Credit: $0.00 (Growth: -100.00%) | Buys: 1, Sells: 1 | Fees: $0.13, Turnover(B/S): $101.02/$25.42 | Rewards Σ: pv_logret=0.0109, realized=0.0014, trade_pen=-0.0200, dd_pen=-0.0007, fees_pen=-0.0017, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $102.94 (Ret: 1.47%, LogRet: 1.47%) | Credit: $102.94 (Growth: 0.00%) | Buys: 0, Sells: 1 | Fees: $0.10, Turnover(B/S): $0.00/$103.04 | Rewards Σ: pv_logret=0.0146, realized=101183636.1997, trade_pen=0.0000, dd_pen=-0.0031, fees_pen=-10304427.3454, time_pen=0.0000
Model saved to saved_models/best_model.pth
New best validation cumulative PV return: 2.94%. Model saved.

Aggregated validation:
  Cumulative PV return: 2.94%
  Mean segment PV log-return: 0.55%
  Total buys: 51, Total sells: 27

=== Episode 9/50 ===
Training on slice 2025-05-27 02:27:00+00:00 -> 2025-05-29 18:21:00+00:00
TRAINING:  48%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                       | 1646/3419 [01:07<01:11, 24.88it/s]
Target network synchronized.
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $105.78 (Ret: 5.78%, LogRet: 5.78%) | Credit: $0.00 (Growth: -100.00%) | Buys: 1, Sells: 0 | Fees: $0.10, Turnover(B/S): $100.00/$0.00 | Rewards Σ: pv_logret=0.0562, realized=0.0000, trade_pen=-0.0100, dd_pen=-0.0036, fees_pen=-0.0010, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $105.86 (Ret: 0.30%, LogRet: 0.30%) | Credit: $52.72 (Growth: 0.00%) | Buys: 0, Sells: 1 | Fees: $0.05, Turnover(B/S): $0.00/$52.77 | Rewards Σ: pv_logret=0.0030, realized=138497411.0152, trade_pen=-0.0100, dd_pen=-0.0011, fees_pen=-5276994.8220, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $104.10 (Ret: -1.92%, LogRet: -1.92%) | Credit: $77.97 (Growth: 47.90%) | Buys: 7, Sells: 6 | Fees: $0.55, Turnover(B/S): $262.60/$288.13 | Rewards Σ: pv_logret=-0.0194, realized=0.0145, trade_pen=-0.1300, dd_pen=-0.0020, fees_pen=-0.0104, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $106.11 (Ret: 1.78%, LogRet: 1.78%) | Credit: $79.44 (Growth: 1.89%) | Buys: 11, Sells: 10 | Fees: $0.84, Turnover(B/S): $421.15/$423.05 | Rewards Σ: pv_logret=0.0177, realized=0.0153, trade_pen=-0.2100, dd_pen=-0.0025, fees_pen=-0.0108, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $106.73 (Ret: 0.56%, LogRet: 0.56%) | Credit: $0.00 (Growth: -100.00%) | Buys: 1, Sells: 1 | Fees: $0.13, Turnover(B/S): $106.12/$26.70 | Rewards Σ: pv_logret=0.0055, realized=0.0014, trade_pen=-0.0200, dd_pen=-0.0028, fees_pen=-0.0017, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $107.55 (Ret: 1.47%, LogRet: 1.47%) | Credit: $107.55 (Growth: 0.00%) | Buys: 0, Sells: 1 | Fees: $0.11, Turnover(B/S): $0.00/$107.66 | Rewards Σ: pv_logret=0.0146, realized=77164319.2798, trade_pen=0.0000, dd_pen=-0.0031, fees_pen=-10766029.6935, time_pen=0.0000
Model saved to saved_models/best_model.pth
New best validation cumulative PV return: 7.55%. Model saved.

Aggregated validation:
  Cumulative PV return: 7.55%
  Mean segment PV log-return: 1.33%
  Total buys: 20, Total sells: 19
=== Episode 10/50 ===
Training on slice 2025-05-19 07:04:00+00:00 -> 2025-05-22 02:49:00+00:00
TRAINING:   7%|█████████████████▍                                                                                                                                                                                                                                                     | 227/3419 [00:09<02:31, 21.10it/s]Target network synchronized.
TRAINING:  65%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                           | 2225/3419 [01:35<00:51, 23.36it/s]
Target network synchronized.
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $105.36 (Ret: 5.36%, LogRet: 5.36%) | Credit: $0.00 (Growth: -100.00%) | Buys: 3, Sells: 1 | Fees: $0.31, Turnover(B/S): $203.77/$103.88 | Rewards Σ: pv_logret=0.0522, realized=0.0194, trade_pen=-0.0400, dd_pen=-0.0036, fees_pen=-0.0031, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $105.59 (Ret: 0.45%, LogRet: 0.45%) | Credit: $0.00 (Growth: 0.00%) | Buys: 1, Sells: 1 | Fees: $0.21, Turnover(B/S): $105.01/$105.12 | Rewards Σ: pv_logret=0.0044, realized=67312271.0462, trade_pen=-0.0200, dd_pen=-0.0021, fees_pen=-21013304.7467, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $106.67 (Ret: 0.50%, LogRet: 0.50%) | Credit: $79.90 (Growth: 0.00%) | Buys: 5, Sells: 4 | Fees: $0.50, Turnover(B/S): $209.25/$289.44 | Rewards Σ: pv_logret=0.0050, realized=90313155.6159, trade_pen=-0.0900, dd_pen=-0.0021, fees_pen=-49869578.8890, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $107.07 (Ret: 0.21%, LogRet: 0.21%) | Credit: $80.21 (Growth: 0.39%) | Buys: 9, Sells: 10 | Fees: $0.64, Turnover(B/S): $321.38/$322.01 | Rewards Σ: pv_logret=0.0021, realized=0.0046, trade_pen=-0.1900, dd_pen=-0.0023, fees_pen=-0.0081, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $107.69 (Ret: 0.56%, LogRet: 0.56%) | Credit: $0.00 (Growth: -100.00%) | Buys: 1, Sells: 1 | Fees: $0.13, Turnover(B/S): $107.07/$26.89 | Rewards Σ: pv_logret=0.0055, realized=0.0010, trade_pen=-0.0200, dd_pen=-0.0028, fees_pen=-0.0017, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $108.52 (Ret: 1.47%, LogRet: 1.47%) | Credit: $108.52 (Growth: 0.00%) | Buys: 0, Sells: 1 | Fees: $0.11, Turnover(B/S): $0.00/$108.63 | Rewards Σ: pv_logret=0.0146, realized=77857884.4119, trade_pen=0.0000, dd_pen=-0.0031, fees_pen=-10862796.4748, time_pen=0.0000
Model saved to saved_models/best_model.pth
New best validation cumulative PV return: 8.52%. Model saved.

Aggregated validation:
  Cumulative PV return: 8.52%
  Mean segment PV log-return: 1.43%
  Total buys: 19, Total sells: 18
=== Episode 11/50 ===
Training on slice 2025-06-06 19:02:00+00:00 -> 2025-06-09 16:51:00+00:00
TRAINING:  24%|██████████████████████████████████████████████████████████████                                                                                                                                                                                                         | 806/3419 [00:34<01:49, 23.97it/s]
Target network synchronized.
TRAINING:  82%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                               | 2807/3419 [02:00<00:26, 23.03it/s]
Target network synchronized.
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $105.51 (Ret: 5.51%, LogRet: 5.51%) | Credit: $0.00 (Growth: -100.00%) | Buys: 2, Sells: 1 | Fees: $0.31, Turnover(B/S): $203.92/$104.02 | Rewards Σ: pv_logret=0.0536, realized=0.0201, trade_pen=-0.0300, dd_pen=-0.0036, fees_pen=-0.0031, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $105.23 (Ret: -0.04%, LogRet: -0.04%) | Credit: $0.00 (Growth: 0.00%) | Buys: 2, Sells: 2 | Fees: $0.42, Turnover(B/S): $209.73/$209.94 | Rewards Σ: pv_logret=-0.0004, realized=52290937.1783, trade_pen=-0.0400, dd_pen=-0.0018, fees_pen=-41966403.7980, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $105.89 (Ret: 0.11%, LogRet: 0.11%) | Credit: $79.31 (Growth: 0.00%) | Buys: 5, Sells: 5 | Fees: $0.67, Turnover(B/S): $295.63/$375.32 | Rewards Σ: pv_logret=0.0011, realized=68519718.3414, trade_pen=-0.1000, dd_pen=-0.0019, fees_pen=-67095049.6419, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $106.53 (Ret: 0.45%, LogRet: 0.45%) | Credit: $106.53 (Growth: 34.32%) | Buys: 2, Sells: 3 | Fees: $0.35, Turnover(B/S): $159.49/$186.89 | Rewards Σ: pv_logret=0.0045, realized=0.0061, trade_pen=-0.0500, dd_pen=-0.0022, fees_pen=-0.0044, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $107.27 (Ret: 0.69%, LogRet: 0.69%) | Credit: $0.00 (Growth: -100.00%) | Buys: 1, Sells: 0 | Fees: $0.11, Turnover(B/S): $106.53/$0.00 | Rewards Σ: pv_logret=0.0069, realized=0.0000, trade_pen=-0.0100, dd_pen=-0.0028, fees_pen=-0.0010, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $108.09 (Ret: 1.47%, LogRet: 1.47%) | Credit: $108.09 (Growth: 0.00%) | Buys: 0, Sells: 1 | Fees: $0.11, Turnover(B/S): $0.00/$108.20 | Rewards Σ: pv_logret=0.0146, realized=83371864.9747, trade_pen=0.0000, dd_pen=-0.0031, fees_pen=-10819909.4517, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: 8.09%
  Mean segment PV log-return: 1.37%
  Total buys: 12, Total sells: 12
=== Episode 12/50 ===
Training on slice 2025-05-29 20:14:00+00:00 -> 2025-06-01 23:45:00+00:00
TRAINING:  41%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                           | 1388/3419 [00:58<01:33, 21.75it/s]
Target network synchronized.
TRAINING:  99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋  | 3389/3419 [02:25<00:01, 22.32it/s]
Target network synchronized.
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $104.93 (Ret: 4.93%, LogRet: 4.93%) | Credit: $0.00 (Growth: -100.00%) | Buys: 3, Sells: 2 | Fees: $0.33, Turnover(B/S): $215.72/$115.83 | Rewards Σ: pv_logret=0.0481, realized=0.0166, trade_pen=-0.0500, dd_pen=-0.0031, fees_pen=-0.0033, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $104.49 (Ret: -0.19%, LogRet: -0.19%) | Credit: $52.11 (Growth: 0.00%) | Buys: 2, Sells: 2 | Fees: $0.37, Turnover(B/S): $156.50/$208.82 | Rewards Σ: pv_logret=-0.0019, realized=60890714.3504, trade_pen=-0.0400, dd_pen=-0.0011, fees_pen=-36532546.6223, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $105.97 (Ret: 1.15%, LogRet: 1.15%) | Credit: $59.83 (Growth: 14.82%) | Buys: 5, Sells: 5 | Fees: $0.53, Turnover(B/S): $259.41/$267.40 | Rewards Σ: pv_logret=0.0114, realized=0.0232, trade_pen=-0.1000, dd_pen=-0.0022, fees_pen=-0.0101, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $106.56 (Ret: 0.29%, LogRet: 0.29%) | Credit: $106.56 (Growth: 78.10%) | Buys: 3, Sells: 3 | Fees: $0.59, Turnover(B/S): $273.30/$320.35 | Rewards Σ: pv_logret=0.0029, realized=0.0043, trade_pen=-0.0600, dd_pen=-0.0029, fees_pen=-0.0099, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $107.29 (Ret: 0.69%, LogRet: 0.69%) | Credit: $0.00 (Growth: -100.00%) | Buys: 1, Sells: 0 | Fees: $0.11, Turnover(B/S): $106.56/$0.00 | Rewards Σ: pv_logret=0.0069, realized=0.0000, trade_pen=-0.0100, dd_pen=-0.0028, fees_pen=-0.0010, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $108.12 (Ret: 1.48%, LogRet: 1.48%) | Credit: $108.12 (Growth: 0.00%) | Buys: 1, Sells: 2 | Fees: $0.21, Turnover(B/S): $53.22/$161.50 | Rewards Σ: pv_logret=0.0147, realized=86372969.1902, trade_pen=-0.0200, dd_pen=-0.0029, fees_pen=-21472472.6529, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: 8.12%
  Mean segment PV log-return: 1.39%
  Total buys: 15, Total sells: 14
=== Episode 13/50 ===
Training on slice 2025-05-15 09:27:00+00:00 -> 2025-05-18 00:12:00+00:00
TRAINING:  58%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                               | 1970/3419 [01:24<01:02, 23.35it/s]
Target network synchronized.
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $102.03 (Ret: 2.03%, LogRet: 2.03%) | Credit: $89.10 (Growth: -10.90%) | Buys: 2, Sells: 4 | Fees: $0.29, Turnover(B/S): $151.85/$141.09 | Rewards Σ: pv_logret=0.0201, realized=0.0099, trade_pen=-0.0600, dd_pen=-0.0012, fees_pen=-0.0029, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $102.70 (Ret: 0.68%, LogRet: 0.68%) | Credit: $0.00 (Growth: -100.00%) | Buys: 1, Sells: 2 | Fees: $0.11, Turnover(B/S): $101.99/$12.90 | Rewards Σ: pv_logret=0.0068, realized=0.0010, trade_pen=-0.0300, dd_pen=-0.0016, fees_pen=-0.0013, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $104.10 (Ret: 0.84%, LogRet: 0.84%) | Credit: $78.05 (Growth: 0.00%) | Buys: 6, Sells: 8 | Fees: $1.07, Turnover(B/S): $496.31/$574.94 | Rewards Σ: pv_logret=0.0083, realized=131924902.8033, trade_pen=-0.1400, dd_pen=-0.0026, fees_pen=-107125410.4026, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $104.64 (Ret: 0.37%, LogRet: 0.37%) | Credit: $104.64 (Growth: 34.06%) | Buys: 2, Sells: 2 | Fees: $0.39, Turnover(B/S): $182.67/$209.46 | Rewards Σ: pv_logret=0.0036, realized=0.0051, trade_pen=-0.0400, dd_pen=-0.0007, fees_pen=-0.0050, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $105.36 (Ret: 0.69%, LogRet: 0.69%) | Credit: $0.00 (Growth: -100.00%) | Buys: 1, Sells: 0 | Fees: $0.10, Turnover(B/S): $104.64/$0.00 | Rewards Σ: pv_logret=0.0069, realized=0.0000, trade_pen=-0.0100, dd_pen=-0.0028, fees_pen=-0.0010, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $106.37 (Ret: 1.67%, LogRet: 1.67%) | Credit: $106.37 (Growth: 0.00%) | Buys: 1, Sells: 2 | Fees: $0.32, Turnover(B/S): $104.71/$211.29 | Rewards Σ: pv_logret=0.0166, realized=97337726.0077, trade_pen=-0.0200, dd_pen=-0.0027, fees_pen=-31600171.7203, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: 6.37%
  Mean segment PV log-return: 1.05%
  Total buys: 13, Total sells: 18

=== Episode 14/50 ===
Training on slice 2025-05-26 03:29:00+00:00 -> 2025-05-28 19:27:00+00:00
TRAINING:  16%|██████████████████████████████████████████▍                                                                                                                                                                                                                            | 551/3419 [00:22<01:56, 24.69it/s]Target network synchronized.
TRAINING:  75%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                  | 2549/3419 [01:45<00:35, 24.46it/s]
Target network synchronized.
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $100.52 (Ret: 0.52%, LogRet: 0.52%) | Credit: $100.52 (Growth: 0.52%) | Buys: 9, Sells: 6 | Fees: $0.67, Turnover(B/S): $332.62/$333.47 | Rewards Σ: pv_logret=0.0052, realized=0.0043, trade_pen=-0.1500, dd_pen=-0.0008, fees_pen=-0.0067, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $101.02 (Ret: 0.49%, LogRet: 0.49%) | Credit: $0.00 (Growth: -100.00%) | Buys: 2, Sells: 1 | Fees: $0.30, Turnover(B/S): $200.85/$100.43 | Rewards Σ: pv_logret=0.0049, realized=-0.0005, trade_pen=-0.0300, dd_pen=-0.0015, fees_pen=-0.0030, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $100.64 (Ret: -0.89%, LogRet: -0.89%) | Credit: $75.46 (Growth: 0.00%) | Buys: 10, Sells: 11 | Fees: $1.50, Turnover(B/S): $711.66/$787.91 | Rewards Σ: pv_logret=-0.0090, realized=52663725.4669, trade_pen=-0.2100, dd_pen=-0.0010, fees_pen=-149956661.6645, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $99.52 (Ret: -1.26%, LogRet: -1.26%) | Credit: $99.52 (Growth: 31.89%) | Buys: 4, Sells: 5 | Fees: $0.60, Turnover(B/S): $288.70/$313.07 | Rewards Σ: pv_logret=-0.0127, realized=-0.0050, trade_pen=-0.0900, dd_pen=-0.0008, fees_pen=-0.0080, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $98.68 (Ret: -0.84%, LogRet: -0.84%) | Credit: $98.68 (Growth: -0.84%) | Buys: 11, Sells: 20 | Fees: $1.48, Turnover(B/S): $742.49/$742.40 | Rewards Σ: pv_logret=-0.0084, realized=-0.0005, trade_pen=-0.3100, dd_pen=-0.0013, fees_pen=-0.0149, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $100.38 (Ret: 1.71%, LogRet: 1.71%) | Credit: $100.38 (Growth: 1.71%) | Buys: 2, Sells: 2 | Fees: $0.30, Turnover(B/S): $148.15/$149.99 | Rewards Σ: pv_logret=0.0170, realized=0.0093, trade_pen=-0.0300, dd_pen=-0.0027, fees_pen=-0.0030, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: 0.38%
  Mean segment PV log-return: -0.04%
  Total buys: 38, Total sells: 45

=== Episode 15/50 ===
Training on slice 2025-05-18 04:24:00+00:00 -> 2025-05-20 21:58:00+00:00
TRAINING:  33%|██████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                                                               | 1130/3419 [00:46<01:31, 25.00it/s]
Target network synchronized.
TRAINING:  92%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                      | 3131/3419 [02:12<00:12, 23.80it/s]
Target network synchronized.
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $102.95 (Ret: 2.95%, LogRet: 2.95%) | Credit: $0.00 (Growth: -100.00%) | Buys: 7, Sells: 4 | Fees: $0.54, Turnover(B/S): $318.86/$219.08 | Rewards Σ: pv_logret=0.0291, realized=0.0170, trade_pen=-0.1100, dd_pen=-0.0014, fees_pen=-0.0054, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $102.68 (Ret: -0.05%, LogRet: -0.05%) | Credit: $0.00 (Growth: 0.00%) | Buys: 1, Sells: 1 | Fees: $0.20, Turnover(B/S): $102.43/$102.54 | Rewards Σ: pv_logret=-0.0005, realized=-31802772.6528, trade_pen=-0.0200, dd_pen=-0.0018, fees_pen=-20496775.4701, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $102.81 (Ret: -0.39%, LogRet: -0.39%) | Credit: $102.81 (Growth: 0.00%) | Buys: 6, Sells: 7 | Fees: $1.08, Turnover(B/S): $488.40/$591.80 | Rewards Σ: pv_logret=-0.0039, realized=48257824.8168, trade_pen=-0.1300, dd_pen=-0.0009, fees_pen=-108020011.9455, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $103.61 (Ret: 0.79%, LogRet: 0.79%) | Credit: $103.61 (Growth: 0.79%) | Buys: 3, Sells: 3 | Fees: $0.62, Turnover(B/S): $309.32/$310.44 | Rewards Σ: pv_logret=0.0078, realized=0.0054, trade_pen=-0.0600, dd_pen=-0.0010, fees_pen=-0.0060, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $102.04 (Ret: -1.52%, LogRet: -1.52%) | Credit: $102.04 (Growth: -1.52%) | Buys: 21, Sells: 32 | Fees: $2.77, Turnover(B/S): $1384.98/$1384.79 | Rewards Σ: pv_logret=-0.0153, realized=-0.0009, trade_pen=-0.5300, dd_pen=-0.0014, fees_pen=-0.0267, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $102.89 (Ret: 0.83%, LogRet: 0.83%) | Credit: $102.89 (Growth: 0.83%) | Buys: 12, Sells: 12 | Fees: $2.03, Turnover(B/S): $1015.84/$1017.70 | Rewards Σ: pv_logret=0.0083, realized=0.0092, trade_pen=-0.2300, dd_pen=-0.0017, fees_pen=-0.0199, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: 2.89%
  Mean segment PV log-return: 0.44%
  Total buys: 50, Total sells: 59

=== Episode 16/50 ===
Training on slice 2025-05-22 13:26:00+00:00 -> 2025-05-25 08:07:00+00:00
TRAINING:  50%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                  | 1712/3419 [01:11<01:07, 25.29it/s]
Target network synchronized.
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $102.77 (Ret: 2.77%, LogRet: 2.77%) | Credit: $102.77 (Growth: 2.77%) | Buys: 7, Sells: 4 | Fees: $0.42, Turnover(B/S): $208.06/$211.04 | Rewards Σ: pv_logret=0.0273, realized=0.0149, trade_pen=-0.1100, dd_pen=-0.0013, fees_pen=-0.0042, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $103.48 (Ret: 0.69%, LogRet: 0.69%) | Credit: $0.00 (Growth: -100.00%) | Buys: 1, Sells: 0 | Fees: $0.10, Turnover(B/S): $102.77/$0.00 | Rewards Σ: pv_logret=0.0068, realized=0.0000, trade_pen=-0.0100, dd_pen=-0.0016, fees_pen=-0.0010, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $103.51 (Ret: -0.49%, LogRet: -0.49%) | Credit: $103.51 (Growth: 0.00%) | Buys: 4, Sells: 5 | Fees: $0.68, Turnover(B/S): $286.14/$390.03 | Rewards Σ: pv_logret=-0.0049, realized=56294420.3989, trade_pen=-0.0900, dd_pen=-0.0007, fees_pen=-67616710.4408, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $104.02 (Ret: 0.50%, LogRet: 0.50%) | Credit: $104.02 (Growth: 0.50%) | Buys: 1, Sells: 1 | Fees: $0.21, Turnover(B/S): $103.51/$104.12 | Rewards Σ: pv_logret=0.0050, realized=0.0030, trade_pen=-0.0200, dd_pen=-0.0006, fees_pen=-0.0020, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $102.98 (Ret: -1.00%, LogRet: -1.00%) | Credit: $102.98 (Growth: -1.00%) | Buys: 6, Sells: 8 | Fees: $0.93, Turnover(B/S): $464.94/$464.36 | Rewards Σ: pv_logret=-0.0101, realized=-0.0028, trade_pen=-0.1400, dd_pen=-0.0009, fees_pen=-0.0089, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $103.53 (Ret: 0.54%, LogRet: 0.54%) | Credit: $103.53 (Growth: 0.54%) | Buys: 2, Sells: 2 | Fees: $0.31, Turnover(B/S): $154.12/$154.83 | Rewards Σ: pv_logret=0.0054, realized=0.0034, trade_pen=-0.0300, dd_pen=-0.0007, fees_pen=-0.0030, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: 3.53%
  Mean segment PV log-return: 0.50%
  Total buys: 21, Total sells: 20
=== Episode 17/50 ===
Training on slice 2025-05-08 21:03:00+00:00 -> 2025-05-11 17:09:00+00:00
TRAINING:   9%|██████████████████████▌                                                                                                                                                                                                                                                | 293/3419 [00:11<02:03, 25.41it/s]Target network synchronized.
TRAINING:  18%|███████████████████████████████████████████████▏                                                                                                                                                                                                                       | 614/3419 [00:25<01:53, 24.69it/s]TRAINING:  67%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                      | 2294/3419 [01:38<00:48, 22.97it/s]Target network synchronized.
                                                                                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $101.15 (Ret: 1.15%, LogRet: 1.15%) | Credit: $101.15 (Growth: 1.15%) | Buys: 8, Sells: 8 | Fees: $0.94, Turnover(B/S): $467.43/$469.05 | Rewards Σ: pv_logret=0.0114, realized=0.0081, trade_pen=-0.1600, dd_pen=-0.0009, fees_pen=-0.0094, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $101.90 (Ret: 0.74%, LogRet: 0.74%) | Credit: $0.00 (Growth: -100.00%) | Buys: 1, Sells: 0 | Fees: $0.10, Turnover(B/S): $101.15/$0.00 | Rewards Σ: pv_logret=0.0074, realized=0.0000, trade_pen=-0.0100, dd_pen=-0.0016, fees_pen=-0.0010, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $101.65 (Ret: -0.76%, LogRet: -0.76%) | Credit: $101.65 (Growth: 0.00%) | Buys: 5, Sells: 6 | Fees: $0.87, Turnover(B/S): $383.52/$485.66 | Rewards Σ: pv_logret=-0.0077, realized=49396752.1216, trade_pen=-0.1100, dd_pen=-0.0007, fees_pen=-86917479.7939, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $101.57 (Ret: -0.08%, LogRet: -0.08%) | Credit: $101.57 (Growth: -0.08%) | Buys: 1, Sells: 2 | Fees: $0.20, Turnover(B/S): $101.65/$101.67 | Rewards Σ: pv_logret=-0.0008, realized=0.0001, trade_pen=-0.0300, dd_pen=-0.0002, fees_pen=-0.0020, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $101.25 (Ret: -0.31%, LogRet: -0.31%) | Credit: $101.25 (Growth: -0.31%) | Buys: 6, Sells: 8 | Fees: $0.91, Turnover(B/S): $455.95/$456.08 | Rewards Σ: pv_logret=-0.0032, realized=0.0007, trade_pen=-0.1400, dd_pen=-0.0005, fees_pen=-0.0090, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $100.85 (Ret: -0.39%, LogRet: -0.39%) | Credit: $100.85 (Growth: -0.39%) | Buys: 3, Sells: 3 | Fees: $0.40, Turnover(B/S): $202.31/$202.11 | Rewards Σ: pv_logret=-0.0039, realized=-0.0010, trade_pen=-0.0600, dd_pen=-0.0002, fees_pen=-0.0040, time_pen=0.0000

Aggregated validation:
  Cumulative PV return: 0.85%
  Mean segment PV log-return: 0.06%
  Total buys: 24, Total sells: 27
```
The “did well then slid back” pattern is very common with off‑policy RL on financial series. There are a couple of concrete bugs, several subtle mismatches between training/eval, and a few stability and risk‑management gaps. Below is a prioritized list of changes with the why, where, how, and what-for, so you can keep the gains and make performance more consistent.

Fix the exploding reward components in validation (root cause of insane realized/fees numbers)
Why: In your logs you see realized and fees_pen exploding to millions. That happens when a validation segment starts with zero credit (because you carry positions across segments): your reward shaping normalizes by start_credit, which can be 0, so you divide by 1e-8, exploding the term.
Where: TradingEnvironment._calculate_reward and validate_in_segments/run_episode when you carry credit/positions across segments.
How:
Store and use the starting portfolio value, not “starting credit”, as the normalization baseline.
In reset(), compute self.start_pv = self.credit + self.holdings * initial_price.
In _calculate_reward(), replace max(self.start_credit, 1e-8) with max(self.start_pv, 1e-8) for realized_component and fees_pen.
Optionally cap these components to a sensible range to prevent outliers from dominating: e.g., realized_component = clip(realized_component, [-0.05, 0.05]); fees_pen = clip(fees_pen, [-0.02, 0.02]).
What for: Prevents spurious gigantic reward terms and keeps shaping consistent across segments even when credit is 0 but you’re fully invested. It also stabilizes diagnostics so you can trust the logs.
Align the reward to a single primary objective and tame shaping
Why: Your core reward (log PV return) is well-chosen. The shaping terms (realized bonus, fees_penalty, drawdown penalty, trade penalty) can sway learning unpredictably if their scale isn’t stable. After you fix normalization, still keep shaping small and consistent.
Where: TradingEnvironment._calculate_reward, Config.
How:
Make log PV return the dominant signal: keep as-is.
Reduce REWARD_REALIZED and FEES_PENALTY_COEF so they aid tie-breaking but don’t dominate. Start with REWARD_REALIZED ∈ [0.05, 0.1], FEES_PENALTY_COEF ∈ [0.25, 0.5] (after normalization fix).
Consider clipping total reward to a bounded range (e.g., [-0.1, 0.1]) to stabilize TD targets.
What for: More stable learning signals and less reward hacking.
Stabilize evaluation protocol to match what you optimize
Why: You’re training to maximize cumulative PV growth, but validation mixes per-segment carry and only liquidates at the very end. That’s fine, but make the normalization and metrics reflect that, and avoid artifacts from zero credit.
Where: validate_in_segments.
How (choose one of the two):
Option A (continuous evaluation): Keep carrying positions across segments but normalize diagnostics by segment PV at start (as per item 1). Report cumulative PV at the very end. This matches a live “always on” strategy.
Option B (segment-isolated): Force liquidation at end of every segment and reset credit=INITIAL_CREDIT, holdings=0. This evaluates repeatability of the policy across many independent slices.
What for: Comparable, stable validation metrics; easier model selection; no exploding logs.
Make target network updates smoother
Why: Hard updates every N steps can cause oscillations; a soft/Polyak update often stabilizes Q-learning in non-stationary series.
Where: DQNAgent.update_target_network and learn().
How:
Replace hard copy with soft updates: θ_target = τ θ_online + (1-τ) θ_target each learn step, with τ ~ 0.005–0.01. Keep the current TARGET_UPDATE_EVERY_STEPS or set it to 1 and apply the soft update each learn().
What for: Reduces policy/value “yank” and the episode-to-episode backslide.
Use n-step returns with PER
Why: Single-step TD can under-credit decisions when rewards realize after several minutes; your frame-skip already aggregates between decisions, but n-step targets help credit assignment further.
Where: Replay buffer and learn().
How:
Maintain an n-step buffer (n=3–5). Store (s_t, a_t, R_t^(n), s_{t+n}, done_{t+n}). Use R_t^(n) = sum_{i=0..n-1} γ^i r_{t+i}.
Keep PER priority on the n-step TD error.
What for: Better learning signal for delayed outcomes and improved stability.
Calibrate learning cadence vs samples
Why: You learn every step but only push a transition on decision ticks. With decision frequency 5–15 minutes and cooldown, the ratio of updates to fresh experiences can be high; this can overfit recent minibatches or stale parts of memory.
Where: run_episode() and learn().
How:
Learn only after you push a new aggregated transition (i.e., at decision ticks), and run K gradient steps per push (e.g., K=1–4). Or set LEARN_EVERY = decision_period for training runs.
Ensure LEARN_START_MEMORY considers aggregated transitions (not raw steps). For example, 1–2k aggregated transitions before learning.
What for: Better sample efficiency, less overfitting to stale experiences, more stable convergence.
Normalize and clip the portfolio-related inputs
Why: unrealized_pnl_ratio can explode when average_buy_price is close to zero; this can destabilize the network.
Where: TradingEnvironment._get_state().
How:
Clip unrealized_pnl_ratio to a bounded range, e.g., [-2, 2] or use tanh scaling, e.g., tanh(0.5 * ratio).
Consider adding portfolio_value_change over the window as an additional normalized feature (e.g., rolling log return), and remove redundant TA features.
What for: Prevents rare extreme feature values from dominating the forward pass.
Constrain exposure strictly (don’t just gate)
Why: The exposure cap logic checks current_exposure < 0.90 before a buy, but a buy can push exposure beyond the cap. This can result in “all-in” behavior that looks good in trending episodes and then hurts consistency.
Where: TradingEnvironment.step().
How:
Compute the maximum buy amount that keeps final exposure <= cap and clamp buy_amount_asset accordingly. That is, solve for additional_exposure_needed and reduce buy_notional if needed.
Consider removing the BUY_100 action or replacing it with BUY_75 to reduce all-in flips.
What for: Smoother risk profile, better generalization across market regimes.
Rebalance the action space and cooldown
Why: Too-aggressive actions (BUY_100, SELL_100) plus relatively short cooldown (2–6 decision ticks) can create whipsaw behavior that the agent learns and then forgets under changing data; this hurts consistency.
Where: TradingEnvironment.action_space, ACTION_COOLDOWN_MINUTES.
How:
Remove BUY_100, keep SELL_100 (exit is important), or add BUY_75/SELL_75 for granularity.
Increase cooldown or make it state-dependent (e.g., longer when exposure changes a lot).
What for: Reduces flip-flopping, cuts fees/turnover, improves stability segment-to-segment.
Tighten and simplify the feature set
Why: add_all_ta_features dumps dozens of indicators; many are collinear/noisy and invite overfitting. Your model is powerful (attention + MLP). Too many TA features make training unstable and inconsistent.
Where: load_and_build_features().
How:
Use a small curated set: returns (1, 5, 15, 60-min log returns), rolling volatility (ATR/stdev), RSI, MACD (signal and hist), OBV/volume z-score. Keep dimension tight.
Re-normalize only on train statistics.
What for: Better bias-variance tradeoff, improved robustness.
Make training/eval decision cadence consistent
Why: You train with [5, 15] minutes but evaluate at 15. That can help generalization, but also adds variance and instability.
Where: TRAIN_DECISION_FREQUENCIES and VAL_DECISION_FREQUENCY.
How:
Either keep both at 15 for now to stabilize, or if you want multi-cadence robustness, expand training to [5, 15, 30] and evaluate at all three cadences and average the validation score for model selection.
What for: Reduces surprise distribution shift from training to validation.
Use soft model averaging for evaluation (EMA of weights)
Why: EMA weights are smoother than the latest online policy and often validate better and more consistently.
Where: DQNAgent.
How:
Maintain an EMA copy of policy parameters (θ_ema). Update each learn step: θ_ema = α θ_ema + (1-α) θ_online, α ∈ [0.99, 0.999]. Evaluate with θ_ema; keep training with θ_online.
What for: More consistent validation curves and less backsliding.
Early stopping and patience
Why: You already save the best checkpoint. Continuing to train can easily drift away in off-policy RL. Use patience and stop.
Where: main().
How:
Track best cumulative PV return; if no improvement for N episodes (e.g., 10), stop training and keep best_model.pth.
Optionally reload best weights after each episode to continue improving from the current best rather than drifting.
What for: Prevents late-episode degradation from overwriting a good policy.
Reward scaling to improve TD signal magnitude
Why: Log returns per step are tiny; trade penalties are relatively large compared to per-step log returns; the scale mismatch can skew learning.
Where: _calculate_reward and Config.
How:
Multiply pv_logret by a constant scale (e.g., 100) and reduce TRADE_PENALTY proportionally (e.g., 0.001–0.002 after scaling), or clip rewards to a narrow band after summation.
What for: Keeps TD errors in a numerically healthy range and improves PER behavior.
PER details and safety checks
Why: Early in training, many priorities are equal/zero; SumTree get() fallback and zero-priority branches can bias sampling.
Where: SumTree and PrioritizedReplayMemory.
How:
Ensure initial transitions are inserted with a reasonable priority (you already use self.max_priority).
After updates, bound priorities (e.g., min/max clamp) to avoid extreme sampling bias.
Consider a small ε priority floor per-sample (you already add self.e).
What for: Prevents PER pathologies and improves training stability.
Make aggregated transitions clearer to the learner
Why: You push aggregated reward from several base steps, but the network’s Q target is still 1-step bootstrapped from the next decision state. That’s fine, but be explicit and consistent.
Where: run_episode().
How:
Only compute pv_logret and shaping on steps that will be aggregated to the next decision tick; you already sum reward across base steps, which is good. Make sure shaping terms are aggregated in the same units (no accidental double-counting).
What for: Clearer credit assignment.
Improve average cost and realized PnL accounting consistency
Why: Your approach is approximately correct, but to be fully coherent:
Where: TradingEnvironment.step().
How:
For buy: total_cost += current_price * buy_amount_asset + buy_fee; average_buy_price = total_cost / holdings.
For sell: realized_pnl += (current_price - average_buy_price) * sell_amount_asset - sell_fee_per_unit if you want realized PnL net fees; or keep fees separately but then make sure your reward shaping and PV line are consistent.
What for: Cleaner accounting reduces the chance of subtle reward/feature inconsistencies.
Exposure-aware action mask based on absolutes
Why: Using credit_ratio can behave oddly when PV changes a lot; also, when fully invested credit_ratio≈0 disables buys (good) but enabling condition may fluctuate due to price moves.
Where: DQNAgent._action_mask_from_state().
How:
Base on absolute credit (dollars) and absolute holdings value instead of ratios. E.g., disable buys if credit < MIN_TRADE_CREDIT_BUY; disable sells if holdings_value < MIN_TRADE_HOLDINGS_SELL.
What for: More stable masking independent of portfolio composition.
Longer training episodes and more diverse slices
Why: 2.5-day episodes might overfit to microstructure and not learn regime-level cues.
Where: Config.EPISODE_LENGTH_DAYS.
How:
Try 5–10 days per training episode, or sample episode lengths from a distribution (e.g., Uniform[2, 7] days).
What for: Better generalization and less episode-to-episode volatility.
Sanity checks and baselines
Why: You need to know if the learned policy beats trivial baselines consistently, not just in a few episodes.
Where: validation reporting.
How:
Add Buy&Hold and Cash-only baselines to the same validation splits and plot/print them.
Track turnover, fees, max drawdown for each segment model.
What for: Detects overfitting and confirms the policy really adds value.
Logging and diagnostics to catch regressions early
Why: To understand why performance slides when it does.
Where: run_episode and validate_in_segments outputs.
How:
Log the distribution of Q-values at decision ticks (mean/std), action entropy, and fraction of “big” actions (e.g., SELL_100/BUY_50+).
Track state feature ranges, especially unrealized_pnl_ratio, to ensure clipping works.
What for: Root-cause analysis when stability issues reappear.
Deterministic evaluation path
Why: Ensure eval is deterministic and not affected by NoisyNet stochasticity or training mode.
Where: DQNAgent.act().
How:
You already set eval() and NoisyLinear runs deterministically at eval. Keep it. Also ensure torch.backends.cudnn.deterministic set if you go deeper, but current ops are deterministic enough.
What for: Reproducible validation metrics.
Use EMA/Best-of ensemble for final test
Why: Smooth out last-mile variance.
Where: main() testing stage.
How:
Evaluate both the best checkpoint and an EMA-smoothed version; optionally average their actions or pick the better performer on validation before final test.
What for: Small, free consistency boost.
Optional: Distributional RL head (C51/QR-DQN)
Why: Distributional methods are known to improve stability of value estimation in noisy reward settings like markets.
Where: DuelingDQN output head.
How:
Replace scalar Q with categorical distribution (51 atoms) or quantile regression (n=51), keep dueling advantage/value decomposition.
What for: Often yields more stable and higher-quality policies.
Optional: Transaction-cost-aware action shaping
Why: To strongly discourage churn in chop without needing big trade penalties.
Where: Policy/Env.
How:
Include a “no-trade band” feature (e.g., price vs short-term MA in bps). Or add a learned inaction penalty only when predicted advantage over HOLD is small.
What for: Reduces overtrading while preserving responsiveness.
What’s specifically causing “did well then slid back”

Backsliding is expected with off-policy value-based RL on non-stationary data. You’re not doing anything “wrong” fundamentally; you’re just missing stabilizers. The biggest correctness bug is the reward normalization in validation (and diagnostics) that explodes when credit=0. The rest are stabilizers (soft targets, n-step returns, EMA, calibrated reward scales) and risk controls (exposure clamp, action set).
Implement items 1–4 first; they typically give the biggest stability boost. Then 6–9 to reduce churn and regime sensitivity. Finally 10–15 to cut variance further.
Minimal “Phase 1” checklist to try next

Normalize realized/fees by start_pv; clip both.
Switch target updates to soft with τ=0.005–0.01.
Learn K=2 times per decision tick; don’t learn every single base step.
Clip unrealized_pnl_ratio to [-2,2].
Strictly enforce 90% exposure cap when buying; remove BUY_100.
Set training and validation decision frequency to 15m for now.
Reduce REWARD_REALIZED to 0.05; FEES_PENALTY_COEF to 0.25; scale pv_logret by 100 and set TRADE_PENALTY to 0.001.
Add early stopping with patience=10 and evaluate EMA weights.
These changes usually turn the “good then bad” pattern into “good then flat then slightly better,” and your best checkpoint will arrive earlier and be more robust.

I need you to take all this recomedations and things to apply to the code, in order to fix it. Please apply all of this things, to make sure that now the code will work. Please write the whole code with the fixes integrated in it. Write the output inside a block of code.

I dont need you to write the whole fixed code, but to generate a list of things to fix, add or modify in order for this to keep work. I think in general it was doing for a while, but  I need to make it be consistent. Please be very descriptive of why , where, how and what for is things, on what to do.























































































































Look, I have this code, which Im trying to optimize to make it make good trades and make credit or capital grow. Please help me fix it. As you can see, it did pretty well till episode 8, and then at 9 it started to move back down and practically stop doing any gains. Why is it no keeping the results ?  I think the code is on a good track, Just I dont understand why it goes back to bad results and not making almost any money at episode 9. Are you storing a check point of the best model? The ending test plot, plotted the results with a very bad model version, but please, I would like to plot the best performing one, just for the end plot:

```
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
```

```
=== Episode 1/50 ===
Training on slice 2025-05-19 10:09:00+00:00 -> 2025-05-22 06:03:00+00:00
                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $104.79 (Ret: 4.79%, LogRet: 4.79%) | Credit: $11.18 (Growth: -88.82%) | Buys: 24, Sells: 12 | Fees: $0.51, Turnover(B/S): $298.58/$209.97 | Rewards Σ: pv_logret=4.6791, realized=0.0014, trade_pen=-0.0360, dd_pen=-0.0013, fees_pen=-0.0013, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $105.23 (Ret: 0.62%, LogRet: 0.62%) | Credit: $11.18 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.6215, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0010, fees_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $105.60 (Ret: -0.11%, LogRet: -0.11%) | Credit: $11.56 (Growth: 3.42%) | Buys: 12, Sells: 10 | Fees: $0.28, Turnover(B/S): $140.05/$140.58 | Rewards Σ: pv_logret=-0.1125, realized=0.0022, trade_pen=-0.0220, dd_pen=-0.0019, fees_pen=-0.0007, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $106.92 (Ret: 0.70%, LogRet: 0.70%) | Credit: $99.80 (Growth: 763.22%) | Buys: 7, Sells: 13 | Fees: $0.22, Turnover(B/S): $65.95/$154.34 | Rewards Σ: pv_logret=0.7004, realized=-0.0000, trade_pen=-0.0200, dd_pen=-0.0014, fees_pen=-0.0005, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $107.49 (Ret: 0.53%, LogRet: 0.53%) | Credit: $10.68 (Growth: -89.30%) | Buys: 8, Sells: 0 | Fees: $0.09, Turnover(B/S): $89.11/$0.00 | Rewards Σ: pv_logret=0.5247, realized=0.0000, trade_pen=-0.0080, dd_pen=-0.0012, fees_pen=-0.0002, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $108.24 (Ret: 1.33%, LogRet: 1.33%) | Credit: $108.24 (Growth: 913.21%) | Buys: 0, Sells: 1 | Fees: $0.10, Turnover(B/S): $0.00/$97.65 | Rewards Σ: pv_logret=1.3177, realized=0.0006, trade_pen=0.0000, dd_pen=-0.0014, fees_pen=-0.0002, time_pen=0.0000
Model saved to saved_models/best_model.pth
New best validation cumulative PV return: 8.24%. Model saved.

Aggregated validation:
  Cumulative PV return: 8.24%
  Mean segment PV log-return: 1.31%
  Total buys: 51, Total sells: 36

=== Episode 2/50 ===
Training on slice 2025-05-10 15:38:00+00:00 -> 2025-05-13 13:16:00+00:00
                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $104.79 (Ret: 4.79%, LogRet: 4.79%) | Credit: $11.18 (Growth: -88.82%) | Buys: 24, Sells: 12 | Fees: $0.51, Turnover(B/S): $298.58/$209.97 | Rewards Σ: pv_logret=4.6791, realized=0.0014, trade_pen=-0.0360, dd_pen=-0.0013, fees_pen=-0.0013, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $105.23 (Ret: 0.62%, LogRet: 0.62%) | Credit: $11.18 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.6215, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0010, fees_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $105.60 (Ret: -0.11%, LogRet: -0.11%) | Credit: $11.56 (Growth: 3.42%) | Buys: 12, Sells: 10 | Fees: $0.28, Turnover(B/S): $140.05/$140.58 | Rewards Σ: pv_logret=-0.1125, realized=0.0022, trade_pen=-0.0220, dd_pen=-0.0019, fees_pen=-0.0007, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $106.92 (Ret: 0.70%, LogRet: 0.70%) | Credit: $99.80 (Growth: 763.22%) | Buys: 7, Sells: 13 | Fees: $0.22, Turnover(B/S): $65.95/$154.34 | Rewards Σ: pv_logret=0.7004, realized=-0.0000, trade_pen=-0.0200, dd_pen=-0.0014, fees_pen=-0.0005, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $107.49 (Ret: 0.53%, LogRet: 0.53%) | Credit: $10.68 (Growth: -89.30%) | Buys: 8, Sells: 0 | Fees: $0.09, Turnover(B/S): $89.11/$0.00 | Rewards Σ: pv_logret=0.5247, realized=0.0000, trade_pen=-0.0080, dd_pen=-0.0012, fees_pen=-0.0002, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $108.24 (Ret: 1.33%, LogRet: 1.33%) | Credit: $108.24 (Growth: 913.21%) | Buys: 0, Sells: 1 | Fees: $0.10, Turnover(B/S): $0.00/$97.65 | Rewards Σ: pv_logret=1.3177, realized=0.0006, trade_pen=0.0000, dd_pen=-0.0014, fees_pen=-0.0002, time_pen=0.0000
No improvement. Patience 1/10.

Aggregated validation:
  Cumulative PV return: 8.24%
  Mean segment PV log-return: 1.31%
  Total buys: 51, Total sells: 36

=== Episode 3/50 ===
Training on slice 2025-06-05 17:28:00+00:00 -> 2025-06-08 15:14:00+00:00
                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $104.79 (Ret: 4.79%, LogRet: 4.79%) | Credit: $11.18 (Growth: -88.82%) | Buys: 24, Sells: 12 | Fees: $0.51, Turnover(B/S): $298.58/$209.97 | Rewards Σ: pv_logret=4.6791, realized=0.0014, trade_pen=-0.0360, dd_pen=-0.0013, fees_pen=-0.0013, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $105.23 (Ret: 0.62%, LogRet: 0.62%) | Credit: $11.18 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.6215, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0010, fees_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $105.60 (Ret: -0.11%, LogRet: -0.11%) | Credit: $11.56 (Growth: 3.42%) | Buys: 12, Sells: 10 | Fees: $0.28, Turnover(B/S): $140.05/$140.58 | Rewards Σ: pv_logret=-0.1125, realized=0.0022, trade_pen=-0.0220, dd_pen=-0.0019, fees_pen=-0.0007, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $106.92 (Ret: 0.70%, LogRet: 0.70%) | Credit: $99.80 (Growth: 763.22%) | Buys: 7, Sells: 13 | Fees: $0.22, Turnover(B/S): $65.95/$154.34 | Rewards Σ: pv_logret=0.7004, realized=-0.0000, trade_pen=-0.0200, dd_pen=-0.0014, fees_pen=-0.0005, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $107.49 (Ret: 0.53%, LogRet: 0.53%) | Credit: $10.68 (Growth: -89.30%) | Buys: 8, Sells: 0 | Fees: $0.09, Turnover(B/S): $89.11/$0.00 | Rewards Σ: pv_logret=0.5247, realized=0.0000, trade_pen=-0.0080, dd_pen=-0.0012, fees_pen=-0.0002, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $108.24 (Ret: 1.33%, LogRet: 1.33%) | Credit: $108.24 (Growth: 913.21%) | Buys: 0, Sells: 1 | Fees: $0.10, Turnover(B/S): $0.00/$97.65 | Rewards Σ: pv_logret=1.3177, realized=0.0006, trade_pen=0.0000, dd_pen=-0.0014, fees_pen=-0.0002, time_pen=0.0000
No improvement. Patience 2/10.

Aggregated validation:
  Cumulative PV return: 8.24%
  Mean segment PV log-return: 1.31%
  Total buys: 51, Total sells: 36

=== Episode 4/50 ===
Training on slice 2025-06-02 05:36:00+00:00 -> 2025-06-05 10:23:00+00:00
                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $104.79 (Ret: 4.79%, LogRet: 4.79%) | Credit: $11.18 (Growth: -88.82%) | Buys: 24, Sells: 12 | Fees: $0.51, Turnover(B/S): $298.58/$209.97 | Rewards Σ: pv_logret=4.6791, realized=0.0014, trade_pen=-0.0360, dd_pen=-0.0013, fees_pen=-0.0013, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $105.23 (Ret: 0.62%, LogRet: 0.62%) | Credit: $11.18 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.6215, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0010, fees_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $105.60 (Ret: -0.11%, LogRet: -0.11%) | Credit: $11.56 (Growth: 3.42%) | Buys: 12, Sells: 10 | Fees: $0.28, Turnover(B/S): $140.05/$140.58 | Rewards Σ: pv_logret=-0.1125, realized=0.0022, trade_pen=-0.0220, dd_pen=-0.0019, fees_pen=-0.0007, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $106.92 (Ret: 0.70%, LogRet: 0.70%) | Credit: $99.80 (Growth: 763.22%) | Buys: 7, Sells: 13 | Fees: $0.22, Turnover(B/S): $65.95/$154.34 | Rewards Σ: pv_logret=0.7004, realized=-0.0000, trade_pen=-0.0200, dd_pen=-0.0014, fees_pen=-0.0005, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $107.49 (Ret: 0.53%, LogRet: 0.53%) | Credit: $10.68 (Growth: -89.30%) | Buys: 8, Sells: 0 | Fees: $0.09, Turnover(B/S): $89.11/$0.00 | Rewards Σ: pv_logret=0.5247, realized=0.0000, trade_pen=-0.0080, dd_pen=-0.0012, fees_pen=-0.0002, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $108.24 (Ret: 1.33%, LogRet: 1.33%) | Credit: $108.24 (Growth: 913.21%) | Buys: 0, Sells: 1 | Fees: $0.10, Turnover(B/S): $0.00/$97.65 | Rewards Σ: pv_logret=1.3177, realized=0.0006, trade_pen=0.0000, dd_pen=-0.0014, fees_pen=-0.0002, time_pen=0.0000
No improvement. Patience 3/10.

Aggregated validation:
  Cumulative PV return: 8.24%
  Mean segment PV log-return: 1.31%
  Total buys: 51, Total sells: 36

=== Episode 5/50 ===
Training on slice 2025-05-30 19:39:00+00:00 -> 2025-06-02 21:36:00+00:00
                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $104.79 (Ret: 4.79%, LogRet: 4.79%) | Credit: $11.18 (Growth: -88.82%) | Buys: 24, Sells: 12 | Fees: $0.51, Turnover(B/S): $298.58/$209.97 | Rewards Σ: pv_logret=4.6791, realized=0.0014, trade_pen=-0.0360, dd_pen=-0.0013, fees_pen=-0.0013, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $105.23 (Ret: 0.62%, LogRet: 0.62%) | Credit: $11.18 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.6215, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0010, fees_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $105.60 (Ret: -0.11%, LogRet: -0.11%) | Credit: $11.56 (Growth: 3.42%) | Buys: 12, Sells: 10 | Fees: $0.28, Turnover(B/S): $140.05/$140.58 | Rewards Σ: pv_logret=-0.1125, realized=0.0022, trade_pen=-0.0220, dd_pen=-0.0019, fees_pen=-0.0007, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $106.92 (Ret: 0.70%, LogRet: 0.70%) | Credit: $99.80 (Growth: 763.22%) | Buys: 7, Sells: 13 | Fees: $0.22, Turnover(B/S): $65.95/$154.34 | Rewards Σ: pv_logret=0.7004, realized=-0.0000, trade_pen=-0.0200, dd_pen=-0.0014, fees_pen=-0.0005, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $107.49 (Ret: 0.53%, LogRet: 0.53%) | Credit: $10.68 (Growth: -89.30%) | Buys: 8, Sells: 0 | Fees: $0.09, Turnover(B/S): $89.11/$0.00 | Rewards Σ: pv_logret=0.5247, realized=0.0000, trade_pen=-0.0080, dd_pen=-0.0012, fees_pen=-0.0002, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $108.24 (Ret: 1.33%, LogRet: 1.33%) | Credit: $108.24 (Growth: 913.21%) | Buys: 0, Sells: 1 | Fees: $0.10, Turnover(B/S): $0.00/$97.65 | Rewards Σ: pv_logret=1.3177, realized=0.0006, trade_pen=0.0000, dd_pen=-0.0014, fees_pen=-0.0002, time_pen=0.0000
No improvement. Patience 4/10.

Aggregated validation:
  Cumulative PV return: 8.24%
  Mean segment PV log-return: 1.31%
  Total buys: 51, Total sells: 36

=== Episode 6/50 ===
Training on slice 2025-05-22 07:52:00+00:00 -> 2025-05-25 02:14:00+00:00
                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $104.79 (Ret: 4.79%, LogRet: 4.79%) | Credit: $11.18 (Growth: -88.82%) | Buys: 24, Sells: 12 | Fees: $0.51, Turnover(B/S): $298.58/$209.97 | Rewards Σ: pv_logret=4.6791, realized=0.0014, trade_pen=-0.0360, dd_pen=-0.0013, fees_pen=-0.0013, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $105.23 (Ret: 0.62%, LogRet: 0.62%) | Credit: $11.18 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.6215, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0010, fees_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $105.60 (Ret: -0.11%, LogRet: -0.11%) | Credit: $11.56 (Growth: 3.42%) | Buys: 12, Sells: 10 | Fees: $0.28, Turnover(B/S): $140.05/$140.58 | Rewards Σ: pv_logret=-0.1125, realized=0.0022, trade_pen=-0.0220, dd_pen=-0.0019, fees_pen=-0.0007, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $106.92 (Ret: 0.70%, LogRet: 0.70%) | Credit: $99.80 (Growth: 763.22%) | Buys: 7, Sells: 13 | Fees: $0.22, Turnover(B/S): $65.95/$154.34 | Rewards Σ: pv_logret=0.7004, realized=-0.0000, trade_pen=-0.0200, dd_pen=-0.0014, fees_pen=-0.0005, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $107.49 (Ret: 0.53%, LogRet: 0.53%) | Credit: $10.68 (Growth: -89.30%) | Buys: 8, Sells: 0 | Fees: $0.09, Turnover(B/S): $89.11/$0.00 | Rewards Σ: pv_logret=0.5247, realized=0.0000, trade_pen=-0.0080, dd_pen=-0.0012, fees_pen=-0.0002, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $108.24 (Ret: 1.33%, LogRet: 1.33%) | Credit: $108.24 (Growth: 913.21%) | Buys: 0, Sells: 1 | Fees: $0.10, Turnover(B/S): $0.00/$97.65 | Rewards Σ: pv_logret=1.3177, realized=0.0006, trade_pen=0.0000, dd_pen=-0.0014, fees_pen=-0.0002, time_pen=0.0000
No improvement. Patience 5/10.

Aggregated validation:
  Cumulative PV return: 8.24%
  Mean segment PV log-return: 1.31%
  Total buys: 51, Total sells: 36


=== Episode 7/50 ===
Training on slice 2025-05-18 14:29:00+00:00 -> 2025-05-21 08:27:00+00:00
                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $103.67 (Ret: 3.67%, LogRet: 3.67%) | Credit: $96.53 (Growth: -3.47%) | Buys: 16, Sells: 17 | Fees: $0.38, Turnover(B/S): $192.09/$188.81 | Rewards Σ: pv_logret=3.6042, realized=0.0017, trade_pen=-0.0330, dd_pen=-0.0007, fees_pen=-0.0010, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $103.70 (Ret: 0.05%, LogRet: 0.05%) | Credit: $96.53 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0480, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0001, fees_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $103.69 (Ret: -0.05%, LogRet: -0.05%) | Credit: $96.53 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=-0.0506, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0002, fees_pen=0.0000, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $103.68 (Ret: -0.05%, LogRet: -0.05%) | Credit: $96.53 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=-0.0549, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0002, fees_pen=0.0000, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $103.74 (Ret: 0.06%, LogRet: 0.06%) | Credit: $96.53 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0566, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0001, fees_pen=0.0000, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $103.80 (Ret: 0.10%, LogRet: 0.10%) | Credit: $103.80 (Growth: 7.53%) | Buys: 0, Sells: 1 | Fees: $0.01, Turnover(B/S): $0.00/$7.28 | Rewards Σ: pv_logret=0.1018, realized=0.0003, trade_pen=0.0000, dd_pen=-0.0001, fees_pen=-0.0000, time_pen=0.0000
No improvement. Patience 6/10.

Aggregated validation:
  Cumulative PV return: 3.80%
  Mean segment PV log-return: 0.63%
  Total buys: 16, Total sells: 18

=== Episode 8/50 ===
Training on slice 2025-06-24 17:25:00+00:00 -> 2025-06-27 10:48:00+00:00
                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $105.25 (Ret: 5.25%, LogRet: 5.25%) | Credit: $10.01 (Growth: -89.99%) | Buys: 8, Sells: 0 | Fees: $0.09, Turnover(B/S): $89.99/$0.00 | Rewards Σ: pv_logret=5.1178, realized=0.0000, trade_pen=-0.0080, dd_pen=-0.0016, fees_pen=-0.0002, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $105.70 (Ret: 0.63%, LogRet: 0.63%) | Credit: $10.01 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.6295, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0010, fees_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $106.11 (Ret: -0.08%, LogRet: -0.08%) | Credit: $11.61 (Growth: 16.01%) | Buys: 12, Sells: 10 | Fees: $0.28, Turnover(B/S): $140.49/$142.24 | Rewards Σ: pv_logret=-0.0836, realized=0.0036, trade_pen=-0.0220, dd_pen=-0.0019, fees_pen=-0.0007, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $107.44 (Ret: 0.70%, LogRet: 0.70%) | Credit: $100.28 (Growth: 763.39%) | Buys: 7, Sells: 13 | Fees: $0.22, Turnover(B/S): $66.26/$155.08 | Rewards Σ: pv_logret=0.7004, realized=0.0001, trade_pen=-0.0200, dd_pen=-0.0014, fees_pen=-0.0005, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $108.01 (Ret: 0.53%, LogRet: 0.53%) | Credit: $10.73 (Growth: -89.30%) | Buys: 8, Sells: 0 | Fees: $0.09, Turnover(B/S): $89.54/$0.00 | Rewards Σ: pv_logret=0.5247, realized=0.0000, trade_pen=-0.0080, dd_pen=-0.0012, fees_pen=-0.0002, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $108.76 (Ret: 1.33%, LogRet: 1.33%) | Credit: $108.76 (Growth: 913.21%) | Buys: 0, Sells: 1 | Fees: $0.10, Turnover(B/S): $0.00/$98.12 | Rewards Σ: pv_logret=1.3177, realized=0.0006, trade_pen=0.0000, dd_pen=-0.0014, fees_pen=-0.0002, time_pen=0.0000
Model saved to saved_models/best_model.pth
New best validation cumulative PV return: 8.76%. Model saved.

Aggregated validation:
  Cumulative PV return: 8.76%
  Mean segment PV log-return: 1.39%
  Total buys: 35, Total sells: 24

=== Episode 9/50 ===
Training on slice 2025-07-02 20:32:00+00:00 -> 2025-07-05 22:20:00+00:00
                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $99.31 (Ret: -0.69%, LogRet: -0.69%) | Credit: $97.16 (Growth: -2.84%) | Buys: 13, Sells: 18 | Fees: $0.45, Turnover(B/S): $224.68/$222.07 | Rewards Σ: pv_logret=-0.6972, realized=-0.0002, trade_pen=-0.0310, dd_pen=-0.0002, fees_pen=-0.0011, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $99.32 (Ret: 0.02%, LogRet: 0.02%) | Credit: $97.16 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0150, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0000, fees_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $99.31 (Ret: -0.02%, LogRet: -0.02%) | Credit: $97.16 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=-0.0158, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0001, fees_pen=0.0000, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $99.31 (Ret: -0.02%, LogRet: -0.02%) | Credit: $97.16 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=-0.0172, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0001, fees_pen=0.0000, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $99.33 (Ret: 0.02%, LogRet: 0.02%) | Credit: $97.16 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0177, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0000, fees_pen=0.0000, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $99.34 (Ret: 0.03%, LogRet: 0.03%) | Credit: $99.34 (Growth: 2.24%) | Buys: 0, Sells: 1 | Fees: $0.00, Turnover(B/S): $0.00/$2.18 | Rewards Σ: pv_logret=0.0319, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0000, fees_pen=-0.0000, time_pen=0.0000
No improvement. Patience 1/10.

Aggregated validation:
  Cumulative PV return: -0.66%
  Mean segment PV log-return: -0.11%
  Total buys: 13, Total sells: 19

=== Episode 10/50 ===
Training on slice 2025-06-22 12:42:00+00:00 -> 2025-06-25 10:29:00+00:00
                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $99.70 (Ret: -0.30%, LogRet: -0.30%) | Credit: $97.42 (Growth: -2.58%) | Buys: 9, Sells: 12 | Fees: $0.31, Turnover(B/S): $155.82/$153.40 | Rewards Σ: pv_logret=-0.2986, realized=-0.0001, trade_pen=-0.0210, dd_pen=-0.0001, fees_pen=-0.0008, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $99.71 (Ret: 0.02%, LogRet: 0.02%) | Credit: $97.42 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0159, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0000, fees_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $99.71 (Ret: -0.02%, LogRet: -0.02%) | Credit: $97.42 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=-0.0168, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0001, fees_pen=0.0000, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $99.70 (Ret: -0.02%, LogRet: -0.02%) | Credit: $97.42 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=-0.0182, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0001, fees_pen=0.0000, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $99.72 (Ret: 0.02%, LogRet: 0.02%) | Credit: $97.42 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0188, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0000, fees_pen=0.0000, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $99.74 (Ret: 0.03%, LogRet: 0.03%) | Credit: $99.74 (Growth: 2.38%) | Buys: 0, Sells: 1 | Fees: $0.00, Turnover(B/S): $0.00/$2.32 | Rewards Σ: pv_logret=0.0338, realized=0.0001, trade_pen=0.0000, dd_pen=-0.0000, fees_pen=-0.0000, time_pen=0.0000
No improvement. Patience 2/10.

Aggregated validation:
  Cumulative PV return: -0.26%
  Mean segment PV log-return: -0.04%
  Total buys: 9, Total sells: 13

=== Episode 17/50 ===
Training on slice 2025-06-28 12:34:00+00:00 -> 2025-07-01 12:18:00+00:00
                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $99.72 (Ret: -0.28%, LogRet: -0.28%) | Credit: $97.44 (Growth: -2.56%) | Buys: 8, Sells: 10 | Fees: $0.27, Turnover(B/S): $135.16/$132.73 | Rewards Σ: pv_logret=-0.2779, realized=-0.0001, trade_pen=-0.0180, dd_pen=-0.0001, fees_pen=-0.0007, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $99.73 (Ret: 0.02%, LogRet: 0.02%) | Credit: $97.44 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0159, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0000, fees_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $99.73 (Ret: -0.02%, LogRet: -0.02%) | Credit: $97.44 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=-0.0168, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0001, fees_pen=0.0000, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $99.72 (Ret: -0.02%, LogRet: -0.02%) | Credit: $97.44 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=-0.0182, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0001, fees_pen=0.0000, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $99.75 (Ret: 0.02%, LogRet: 0.02%) | Credit: $97.44 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0188, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0000, fees_pen=0.0000, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $99.76 (Ret: 0.03%, LogRet: 0.03%) | Credit: $99.76 (Growth: 2.38%) | Buys: 0, Sells: 1 | Fees: $0.00, Turnover(B/S): $0.00/$2.32 | Rewards Σ: pv_logret=0.0338, realized=0.0001, trade_pen=0.0000, dd_pen=-0.0000, fees_pen=-0.0000, time_pen=0.0000
No improvement. Patience 9/10.

Aggregated validation:
  Cumulative PV return: -0.24%
  Mean segment PV log-return: -0.04%
  Total buys: 8, Total sells: 11

=== Episode 18/50 ===
Training on slice 2025-05-29 23:09:00+00:00 -> 2025-06-02 02:17:00+00:00
                                                                                                                                                                                                                                                         
-> Validation phase…
                                                                                                                                                                                                                                                         
Segment-by-segment results:
  Seg 01 [2025-07-09 18:41:00+00:00 → 2025-07-11 21:24:00+00:00]  PV: $99.52 (Ret: -0.48%, LogRet: -0.48%) | Credit: $97.33 (Growth: -2.67%) | Buys: 12, Sells: 15 | Fees: $0.40, Turnover(B/S): $201.77/$199.30 | Rewards Σ: pv_logret=-0.4847, realized=-0.0002, trade_pen=-0.0270, dd_pen=-0.0002, fees_pen=-0.0010, time_pen=0.0000
  Seg 02 [2025-07-11 21:25:00+00:00 → 2025-07-13 22:21:00+00:00]  PV: $99.53 (Ret: 0.02%, LogRet: 0.02%) | Credit: $97.33 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0153, realized=0.0000, trade_pen=0.0000, dd_pen=-0.0000, fees_pen=0.0000, time_pen=0.0000
  Seg 03 [2025-07-13 22:22:00+00:00 → 2025-07-16 07:20:00+00:00]  PV: $98.81 (Ret: -0.74%, LogRet: -0.74%) | Credit: $95.77 (Growth: -1.60%) | Buys: 5, Sells: 8 | Fees: $0.25, Turnover(B/S): $124.21/$122.78 | Rewards Σ: pv_logret=-0.7378, realized=-0.0002, trade_pen=-0.0130, dd_pen=-0.0001, fees_pen=-0.0006, time_pen=0.0000
  Seg 04 [2025-07-16 07:22:00+00:00 → 2025-07-18 21:18:00+00:00]  PV: $98.27 (Ret: -0.56%, LogRet: -0.56%) | Credit: $73.57 (Growth: -23.18%) | Buys: 14, Sells: 17 | Fees: $0.67, Turnover(B/S): $344.71/$322.82 | Rewards Σ: pv_logret=-0.5593, realized=-0.0002, trade_pen=-0.0310, dd_pen=-0.0002, fees_pen=-0.0017, time_pen=0.0000
  Seg 05 [2025-07-18 21:20:00+00:00 → 2025-07-21 08:53:00+00:00]  PV: $98.28 (Ret: -0.03%, LogRet: -0.03%) | Credit: $98.28 (Growth: 33.58%) | Buys: 0, Sells: 1 | Fees: $0.02, Turnover(B/S): $0.00/$24.73 | Rewards Σ: pv_logret=-0.0252, realized=0.0001, trade_pen=-0.0010, dd_pen=-0.0000, fees_pen=-0.0001, time_pen=0.0000
  Seg 06 [2025-07-21 08:54:00+00:00 → 2025-07-22 23:33:00+00:00]  PV: $98.28 (Ret: 0.00%, LogRet: 0.00%) | Credit: $98.28 (Growth: 0.00%) | Buys: 0, Sells: 0 | Fees: $0.00, Turnover(B/S): $0.00/$0.00 | Rewards Σ: pv_logret=0.0000, realized=0.0000, trade_pen=0.0000, dd_pen=0.0000, fees_pen=0.0000, time_pen=0.0000
No improvement. Patience 10/10.

Aggregated validation:
  Cumulative PV return: -1.72%
  Mean segment PV log-return: -0.30%
  Total buys: 31, Total sells: 41
Early stopping due to no improvement.

=== FINAL TEST ON UNSEEN DATA ===
Model loaded from saved_models/best_model.pth
Final PV: $96.40 (Return: -3.60%)                                                                                                                                                                                                                        
Final Credit: $96.40 (Return: -3.60%)
Total Buys: 8, Total Sells: 1
Fees paid: $0.18, Turnover(B/S): $89.99/86.47
Reward components Σ (test): {"pv_logret": -3.671196, "realized": -0.001759, "trade_pen": -0.008, "time_pen": 0.0, "dd_pen": -0.009371, "fees_pen": -0.000441}
```

The big drop you saw starting around episode 9 is expected given your current training schedule: you don’t start learning until the replay buffer reaches 1,500 decision transitions, which happens only after ~7–8 episodes. Episodes 1–6 validation results come from an untrained network that happened to do fine on that particular validation period. Once learning starts (ep 7–9), the policy moves a lot and can easily collapse (noisy Q-values, PER bias, no masking in targets, relatively high LR), hence the sudden drop and then plateau.
Below is a prioritized list of changes (what, where, why, and how) to make training stable, keep the good model, and ensure the final plot uses the best-performing checkpoint.

Start learning at the right time and slow policy drift Why
Your LEARN_START_MEMORY is in decision-transitions (not raw minutes). With 2.5 days per episode and a 15-minute decision cadence, you only collect ~220–240 decision transitions per episode. You need ~7 episodes to cross 1,500, which is exactly why learning suddenly starts around episodes 7–8 and the policy jumps.
Once learning starts, the combination of PER, relatively high LR, soft-update every step, and 2 updates per push can move Q-values hard and cause policy collapse.
What to change

Warm-up settings and update rate:
Reduce LEARN_START_MEMORY so learning starts earlier and ramps up gradually on more diverse slices, or increase episode length so you collect more decisions per episode.
Reduce LEARN_UPDATES_PER_PUSH to 1 to slow the drift per decision.
Lower the learning rate slightly.
Use a “slower” eval policy via higher EMA_MOMENTUM.
Slow the target updates (smaller tau) or switch to periodic hard updates.
Concrete values to try

LEARN_START_MEMORY: 400–600 (instead of 1500), or increase EPISODE_LENGTH_DAYS to 4–5.
LEARN_UPDATES_PER_PUSH: 1 (instead of 2).
LEARNING_RATE: 5e-5 (instead of 1e-4).
EMA_MOMENTUM: 0.999 (instead of 0.995).
SOFT_TARGET_TAU: 0.005 (instead of 0.01), or use a hard update every N learn steps (e.g., copy target from policy every 1,000 learning steps).
What for

You want learning to begin earlier but move more cautiously, reducing the chance of the sudden policy collapse that you observed at episode 9.
Mask invalid actions when computing TD targets (training bug) Why
You correctly mask invalid actions at action selection time (act) but do not mask them during TD target computation. In Double DQN you currently do:
next_actions = argmax(policy(next_state))
target = Q_target(next_state, next_actions)
If argmax picks an invalid action (e.g., SELL when holdings_ratio ~0), you bootstrap off a Q-value that can never be taken, creating biased/unstable targets.
What to change (where)

In DQNAgent.learn(), when you compute next_actions and target Q-values, mask invalid actions for each next_state before argmax and before gather:
How (sketch)

Build a mask for each next_state using your existing _action_mask_from_state, then set Q-values of invalid actions to -1e9 prior to argmax and gather.
Example snippet (inside learn, where non_final_next_states exists)

Build masks and apply them both to policy_net and target_net Qs for next states before argmax and gather.
What for

This avoids bootstrapping on impossible actions which can inject large, misleading TD-errors and destabilize training.
Fix cost basis (average_buy_price) update Why
You update the average_buy_price with invest_I (credit deducted pre-fee) rather than the actual cost in units (buy_amount_asset * current_price). This slightly inflates cost basis and depresses realized PnL, biasing the agent against selling and increasing the chance it learns to avoid trades.
What to change (where)

In TradingEnvironment.step(), in the Buy branch, change average_buy_price update to be unit-weighted by price (you can choose to include or exclude fees in the cost basis, but be consistent with realized pnl).
Minimal fix (exclude fees from cost basis, keep fees handled separately by fees_penalty):
numerator = avg_buy_priceold_holdings + buy_amount_assetcurrent_price
denominator = old_holdings + buy_amount_asset
average_buy_price = numerator / denominator
What for

Correct cost basis ensures realized PnL and the reward components are not biased against trading.
Stabilize model selection and “keep the good one” Why
You are already saving the best validation checkpoint, but it’s easy to overfit to one validation period or save on noisy upticks.
What to change

Save top-K best checkpoints with clear filenames and also keep a “best_ema_only” file. Don’t overwrite blindly; give files a metric-based name and keep a stable symlink.
Example filenames:
saved_models/best_ep08_val+8.76_ema.pth (ema weights only)
saved_models/best_ep08_val+8.76_full.pth (policy, target, ema, optimizer)
Also update saved_models/best_model.pth as a symlink or copy of the best_ema file.
Add an improvement margin to reduce noise-triggered saves, e.g., save only if cumulative_val_return > best_val + 0.2%.
Optionally use a more robust validation metric for “best”:
Sum of pv_logret across segments (geometric return), not just last-segment final credit.
Or median segment log-return to reduce the impact of one lucky segment.
Reduce PATIENCE once you have more consistent selection (e.g., 5) so you stop earlier rather than drifting far from the best.
What for

This preserves the genuinely good policy and avoids overwriting the best model due to noise. It also makes the final test deterministic: you always evaluate the checkpoint that was actually best on a robust metric.
Guarantee your end plots use the best model Why
You already load saved_models/best_model.pth before testing, but you also want a “best model plot on validation” at the end, and to eliminate any ambiguity about which weights are being used.
What to change (how)

After training:
Load the best checkpoint
Re-run validate_in_segments on the validation set and plot it as “Best Validation”
Then run the test and plot it as “Best Test”
Also ensure best_model.pth points to the ema weights you validated with.
Consider naming the final plots with the metric embedded, e.g., best_val_ep08_+8.76.html.
Make validation more robust Why
Your validation window is a single contiguous period broken into 2-day segments. If that period trends up, a trivial buy-and-holdish policy will look good; if it chops, it will punish the same policy. That creates high variance in the “best” checkpoint.
What to change

Use multiple disjoint validation windows (e.g., 3–5 windows across the 15% val split) and aggregate metrics across them for model selection.
Use your geometric return measure (sum of pv_logret) or median across segments/windows as the selection metric.
Slightly lengthen segments (e.g., 3–4 days) so the measure is less noisy.
What for

Reduces the chance that you select a model that only fits one particular regime.
Reduce target/Q oscillation Why
DQN can oscillate with soft target updates each step, NoisyNets, PER, and a relatively high LR. You saw a big swing between episode 8 (good) and 9 (bad).
What to change

One of:
Use a smaller tau (0.005) as above, or
Switch to hard target updates every N learning steps (e.g., 1,000) and remove the soft updates.
Consider a cosine or plateau LR scheduler that halves the LR if validation does not improve for a few episodes.
What for

Slows down how quickly the target changes, leading to steadier Q-updates.
Reward shaping tweaks (optional) Why
The current reward is sensible, but fees_penalty and dd_penalty can push the agent to “do nothing” in choppy test data. It might learn a “cash is king” policy that does okay in some val segments but doesn’t capture gains when trends appear.
What to change (if you see inactivity)

Reduce DRAWDOWN_PENALTY slightly (e.g., 0.003–0.004).
Ensure REWARD_CLIP is not too tight; 0.1 is ok, but if your per-minute log returns during strong trends get clipped too often, consider 0.15.
Consider scaling the realized component up slightly (REWARD_REALIZED 0.07–0.10) to reward crystallizing PnL when it exists.
What for

Encourages trading when there is signal, without letting costs dominate.
Use action cadence and cooldown consistently Why
You already do, but double-check:
Same decision frequency for train/val/test (you do).
Cooldown measured in decision ticks (you do).
If you want more actions and tighter learning signals, shorten ACTION_COOLDOWN_MINUTES to 15–20; if you want fewer but higher conviction actions, keep 30 or raise to 45.
Add regularization to the optimizer (optional) Why
Q-nets can overfit quickly; a little weight decay can calm the net.
What to change

Optimizer: optim.Adam(..., weight_decay=1e-5).
Diagnostics to understand drift (do this once) What
Log and plot per-episode:
Action counts per action type
Mean/median Q-values and their std
TD-error histogram (or at least mean/max)
Average per-step fee and drawdown penalties
Exposure ratio over time
Inspect the first 2–3 episodes after learning starts (episode 7 onward in your current setup) to confirm that invalid-action masking fixed spurious targets and that the Q-value scale stabilizes.
Optional: feature compactness Why
If you set USE_CURATED_FEATURES = True, the network sees a cleaner set (returns, RSI, MACD, ATR, volume z-score). Often improves generalization on short windows.
What to change

Set cfg.USE_CURATED_FEATURES = True and keep WINDOW_SIZE = 180. Re-tune a bit if needed.
Housekeeping: prevent checkpoint confusion between runs Why
If you run multiple experiments in the same folder, you might overwrite “best_model.pth” from previous runs.
What to change

Create a unique run folder per training run (timestamped) and write checkpoints/plots inside it. Keep a symlink saved_models/best_model.pth pointing to that run’s best checkpoint.
Summary of the minimal set I would apply first

Learn sooner but more slowly:
LEARN_START_MEMORY: 600
LEARN_UPDATES_PER_PUSH: 1
LEARNING_RATE: 5e-5
EMA_MOMENTUM: 0.999
SOFT_TARGET_TAU: 0.005
Mask invalid actions in TD target computation.
Fix average_buy_price computation.
Save top-K best checkpoints; keep a “best_ema” checkpoint and always load that for final validation and test plots; add a margin (e.g., 0.2%) for “improvement.”
After training, reload the best checkpoint and:
Re-run validate_in_segments and plot “Best Validation”
Run test and plot “Best Test”
Why episode 8 looked good and episode 9 collapsed

Episodes 1–6: no learning (replay warm-up), policy constant; validation period happened to be friendly to that untrained policy.
Episode 7–8: learning begins; the agent updates rapidly and briefly finds a policy that works well on the validation window.
Episode 9+: continued updates (without masking invalid actions in TD targets, with PER and higher LR) cause Q oscillation/collapse and a conservative “don’t trade” policy that flatlines. Early stopping kicks in later, but the “best” from ep 8 remained the best on validation. It just didn’t generalize to test.
Ensuring the end plot is “the best model”

You already load saved_models/best_model.pth for the test. Add one more step right after loading:
Re-run validation and produce a “Best Validation” plot
Then run test and produce the “Best Test” plot
Save the ema-only checkpoint at the moment you save the best (e.g., best_ep08_val+8.76_ema.pth) and always load that for inference.
If you implement just the masking fix, the cost-basis fix, and the slower/earlier learning schedule, you should see:

No sudden collapse at episode 9
More consistent validation performance
The checkpoint you save as “best” remains competitive on test, and the final plots will reflect that best checkpoint unambiguously.


I need you to take all this recomedations and things to apply to the code, in order to fix it. Please apply all of this things, to make sure that now the code will work. Please write the whole code with the fixes integrated in it. Write the output inside a block of code.

I dont need you to write the whole fixed code, but to generate a list of things to fix, add or modify in order for this to keep work. I think in general it was doing for a while, till episode 8, but  I need to make it be consistent or at least, get results of the best performing model. I think the model does good, but dont understand why it doesnt keep the good results. Please be very descriptive of why , where, how and what for is things, on what to do.


























































































The model is doing great till Episode 5. It looks good till episode 5. At episode 6 it starts decreasing the growth of value. Instead of learning to keep that value or make it growth, training stops and the Best Test Results show that  is it loosing money. Why does it not keeps learning to make the capital growth? Why does training stops to get better? Please help me fix this to get even more capital or to make the capital grow in general. 

```
```

```
```

I dont need you to write the whole fixed code, but to generate a list of things to fix, add or modify in order for this to keep work. I think in general it was doing for a while, but  I need to make it be consistent or at least, get results of the best performing model. I think the model does good, but dont understand why it doesnt keep the good results or keeps trying to learn even further. Please be very descriptive of why , where, how and what for is things, on what to do.
