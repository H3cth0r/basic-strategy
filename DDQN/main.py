# Requirements (install before running):
# pip install pandas numpy ta plotly tqdm scikit-learn torch

import os
import math
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------------------
# Data loader (exactly as requested)
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# Configurations derived from both papers and user requirements
# --------------------------------------------------------------------------------------
class Config:
    # Device selection with mps if available (Apple), else cuda, else cpu
    DEVICE = (
        torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # DRL hyperparameters (Tran et al. 2023)
    EPISODES = 50              # You can raise to 100 for full training, kept moderate to be runnable
    BATCH_SIZE = 40
    REPLAY_SIZE = 10_000
    TARGET_UPDATE_EVERY = 10
    GAMMA = 0.98
    LR_Q = 1e-3
    LR_REWARD = 1e-3
    EPS_START = 1.0
    EPS_END = 0.12
    EPS_DECAY_STEPS = 300

    # Self-Rewarding (Huang et al. 2024) parameters
    REWARD_WINDOW_K = 30  # short-term horizon for Sharpe/MinMax (minutes)
    USE_SELF_REWARD = True
    # Shared buffer and synchronous updates are the default in this implementation.

    # Trading parameters consistent with paper settings
    INITIAL_CASH = 1_000_000.0
    FEE_RATE = 0.001  # 0.1% per trade
    TRADE_FRACTION = 0.98  # invest nearly all available cash when buying (simplified)

    # State representation
    STATE_WINDOW = 60  # minutes; input to CNN
    FEATURES = ["Close", "Open", "High", "Low", "Volume", "RSI", "Ret"]
    NUM_ACTIONS = 3  # 0: hold, 1: buy, 2: sell

    # Walk-forward evaluation
    TEST_SPLIT = 0.2  # last 20% of data for testing
    RANDOM_SEED = 42

    # Evaluation annualization (minute data)
    MINUTES_PER_YEAR = 365 * 24 * 60

# --------------------------------------------------------------------------------------
# Features: RSI using ta; returns; scaling
# --------------------------------------------------------------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Use ta for RSI
    try:
        import ta
    except ImportError:
        raise RuntimeError("Please install 'ta' with: pip install ta")

    # RSI(14)
    rsi = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    df["RSI"] = rsi

    # Minute returns
    df["Ret"] = df["Close"].pct_change().fillna(0.0)

    # Clean up potential NaNs
    df = df.dropna()
    return df

def split_train_test(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    test_size = int(n * test_ratio)
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    return train_df, test_df

def prepare_scalers(train_df: pd.DataFrame, feature_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler

def transform_features(df: pd.DataFrame, scaler: StandardScaler, feature_cols):
    X = scaler.transform(df[feature_cols].values)
    return X

# --------------------------------------------------------------------------------------
# Replay Buffer
# --------------------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done, rlt_vec):
        item = (state, action, reward, next_state, done, rlt_vec)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones, rlt_vecs = zip(*batch)
        return (
            torch.tensor(np.stack(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.stack(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(np.stack(rlt_vecs), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

# --------------------------------------------------------------------------------------
# Networks: QNetwork (Double DQN) and RewardNet (self-rewarding)
# 1D CNN across time with LeakyReLU; architecture aligned to paper guidance
# --------------------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, num_actions: int):
        super().__init__()
        # Two Conv1D layers (120 channels each), LeakyReLU; then FCs
        self.conv1 = nn.Conv1d(in_channels, 120, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(120, 120, kernel_size=5, stride=1, padding=2)
        self.act = nn.LeakyReLU(negative_slope=0.01)

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            out = self.conv2(self.act(self.conv1(dummy)))
            flatten_dim = out.numel()

        self.fc1 = nn.Linear(flatten_dim, 120)
        self.fc2 = nn.Linear(120, num_actions)  # Q-values for each action

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        q = self.fc2(x)
        return q

class RewardNet(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, num_actions: int):
        super().__init__()
        # RewardNet similar shape to QNetwork; GELU can be used as in SRDRL discussion
        self.conv1 = nn.Conv1d(in_channels, 120, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(120, 120, kernel_size=5, stride=1, padding=2)
        self.act = nn.GELU()

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            out = self.conv2(self.act(self.conv1(dummy)))
            flatten_dim = out.numel()

        self.fc1 = nn.Linear(flatten_dim, 120)
        self.fc2 = nn.Linear(120, num_actions)  # per-action reward prediction

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        y = self.fc2(x)
        return y

# --------------------------------------------------------------------------------------
# Environment: trading simulation for training reward shaping and evaluation
# - State: window of normalized features
# - Actions: 0 hold, 1 buy, 2 sell
# - Expert reward: Sharpe (short-term), Return (instant), MinMax (short-term). We use Sharpe as the
#   primary expert reward per Tran et al. (best) but also compute Return and MinMax to allow SRDRL selection.
# --------------------------------------------------------------------------------------
class TradingEnv:
    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        config: Config,
        episode_length: int,
        start_index: int = 0,
        eval_mode: bool = False
    ):
        self.prices = prices
        self.features = features
        self.cfg = config
        self.state_window = config.STATE_WINDOW
        self.k = config.REWARD_WINDOW_K
        self.eval_mode = eval_mode

        # Trading state
        self.cash = config.INITIAL_CASH
        self.holdings = 0.0
        self.position = 0  # 0 flat, +1 long
        self.avg_buy_price = None

        # Episode indexing
        self.t = 0
        self.start_index = max(start_index, self.state_window)
        self.end_index = min(len(prices) - 1, self.start_index + episode_length)
        self.done = False

        # Tracking
        self.portfolio_values = []
        self.cash_values = []
        self.holdings_values = []
        self.trade_log = []  # list of dict: time_idx, type, price, units, pnl, fee

    def reset(self):
        self.cash = self.cfg.INITIAL_CASH
        self.holdings = 0.0
        self.position = 0
        self.avg_buy_price = None

        self.t = 0
        self.done = False

        self.portfolio_values = []
        self.cash_values = []
        self.holdings_values = []
        self.trade_log = []

        return self._get_state()

    def step(self, action: int):
        """
        Execute action: 0 hold, 1 buy, 2 sell
        """
        idx = self.current_index()
        price = float(self.prices[idx])

        # Trade execution (simplified; only long allowed)
        fee = 0.0
        pnl = 0.0
        trade_type = None

        if action == 1:  # buy
            if self.position == 0 and self.cash > 0:
                # invest fraction of cash
                buy_cash = self.cash * self.cfg.TRADE_FRACTION
                fee = buy_cash * self.cfg.FEE_RATE
                buy_cash_after_fee = buy_cash - fee
                units = buy_cash_after_fee / price
                self.holdings += units
                self.cash -= buy_cash
                self.position = 1
                self.avg_buy_price = price
                trade_type = 'buy'
                self.trade_log.append({
                    "time_idx": idx, "type": trade_type, "price": price,
                    "units": units, "pnl": 0.0, "fee": fee
                })

        elif action == 2:  # sell (close long)
            if self.position == 1 and self.holdings > 0:
                sell_value = self.holdings * price
                fee = sell_value * self.cfg.FEE_RATE
                receive = sell_value - fee
                pnl = (price - self.avg_buy_price) * self.holdings - fee
                self.cash += receive
                self.position = 0
                # classify win vs loss
                trade_type = 'sell_win' if (price - self.avg_buy_price) > 0 else 'sell_loss'
                self.trade_log.append({
                    "time_idx": idx, "type": trade_type, "price": price,
                    "units": self.holdings, "pnl": pnl, "fee": fee
                })
                self.holdings = 0.0
                self.avg_buy_price = None

        # Update portfolio tracking
        port_val = self.cash + self.holdings * price
        self.portfolio_values.append(port_val)
        self.cash_values.append(self.cash)
        self.holdings_values.append(self.holdings)

        # Advance time
        self.t += 1
        if self.current_index() >= self.end_index - 1:
            self.done = True

        next_state = self._get_state()

        # Reward shaping (expert + self-reward)
        expert_rewards = self._compute_expert_rewards()
        return next_state, expert_rewards, self.done

    def _get_state(self):
        idx = self.current_index()
        seq = self.features[idx - self.state_window + 1: idx + 1]  # shape (state_window, n_features)
        # Convert to (channels, seq_len)
        seq = np.transpose(seq, (1, 0))  # (n_features, state_window)
        return seq.astype(np.float32)

    def current_index(self):
        return self.start_index + self.t

    def _future_returns(self, idx, k):
        p0 = float(self.prices[idx])
        # future idx sequence
        right = min(idx + k, len(self.prices) - 1)
        future_slice = self.prices[idx+1: right+1]
        if len(future_slice) == 0:
            return np.array([0.0], dtype=np.float32)
        rets = (future_slice - p0) / p0
        return rets.astype(np.float32)

    def _compute_expert_rewards(self):
        """
        Compute expert rewards for actions: hold(POSt=0), buy(POSt=+1), sell(POSt=-1).
        Expert includes: Sharpe (short-term window k), Return (instant), MinMax (window k).
        We select Sharpe as the main expert signal per Tran et al. (best), but we also carry
        Return and MinMax to allow per-action max selection with RewardNet outputs.
        Returns vector [r_hold, r_buy, r_sell] from expert design.
        """
        idx = self.current_index()
        k = self.k

        # POSt for each action [hold, buy, sell]
        pos_actions = np.array([0, 1, -1], dtype=np.float32)

        # Short-term vector returns starting at idx
        Rk = self._future_returns(idx, k)
        mean_r = float(np.mean(Rk)) if len(Rk) > 0 else 0.0
        std_r = float(np.std(Rk)) if len(Rk) > 1 else 0.0
        sharpe_term = (mean_r / std_r) if std_r > 1e-12 else 0.0

        # Return-based reward (instant)
        if idx > 0:
            r_inst = (float(self.prices[idx]) - float(self.prices[idx-1])) / float(self.prices[idx-1])
        else:
            r_inst = 0.0

        # MinMax-based reward (window k)
        if len(Rk) > 0:
            maxR = float(np.max(np.where(Rk > 0, Rk, 0.0)))
            minR = float(np.min(np.where(Rk < 0, Rk, 0.0)))
            # Following the described logic:
            if (maxR > 0) or (maxR + minR > 0):
                minmax_term = maxR
            elif (minR < 0) or (maxR + minR < 0):
                minmax_term = minR
            else:
                minmax_term = (maxR - minR)  # fallback
        else:
            minmax_term = 0.0

        # Expert-labeled rewards per action:
        r_sharpe = pos_actions * sharpe_term
        r_return = pos_actions * r_inst
        r_minmax = pos_actions * minmax_term

        # Combine to a single expert vector; per Tran et al. the best is Sharpe.
        # For SRDRL we will compare RewardNet predictions against each expert metric and take the max per action.
        # Here we choose Sharpe as "expert baseline" but return all three as a stacked for convenience.
        # We'll compute per-action expert "ret" as Sharpe, and also keep Return/MinMax available for the max.
        expert = {
            "sharpe": r_sharpe.astype(np.float32),
            "return": r_return.astype(np.float32),
            "minmax": r_minmax.astype(np.float32),
        }
        return expert

# --------------------------------------------------------------------------------------
# Epsilon scheduler (linear decay)
# --------------------------------------------------------------------------------------
class EpsilonScheduler:
    def __init__(self, start=1.0, end=0.12, decay_steps=300):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.step_count = 0

    def step(self):
        self.step_count += 1

    def value(self):
        frac = min(1.0, self.step_count / float(self.decay_steps))
        return self.start + (self.end - self.start) * frac

# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def compute_performance_metrics(times, prices, portfolio_values):
    # Compute returns series from portfolio values
    pv = np.array(portfolio_values, dtype=np.float64)
    rets = np.diff(pv) / pv[:-1]
    rets = np.nan_to_num(rets, nan=0.0)

    # Cumulative return (%)
    cr = (pv[-1] / pv[0] - 1.0) * 100.0

    # Annualized return based on minute returns
    mean_r = np.mean(rets)
    std_r = np.std(rets)
    # Annualize
    ar = (1.0 + mean_r) ** Config.MINUTES_PER_YEAR - 1.0
    ar *= 100.0

    # Sharpe ratio
    sr = (mean_r / std_r) * math.sqrt(Config.MINUTES_PER_YEAR) if std_r > 1e-12 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / peak
    mdd = -np.min(drawdown) * 100.0

    # Volatility
    vol = std_r * math.sqrt(Config.MINUTES_PER_YEAR) * 100.0

    return {
        "CR_%": cr,
        "AR_%": ar,
        "SR": sr,
        "MDD_%": mdd,
        "Vol_%": vol,
    }

def trade_statistics(trade_log):
    sells = [t for t in trade_log if t["type"].startswith("sell")]
    wins = [t for t in sells if t["type"] == "sell_win"]
    losses = [t for t in sells if t["type"] == "sell_loss"]
    n_trades = len(sells)
    win_rate = (len(wins) / n_trades * 100.0) if n_trades > 0 else 0.0
    avg_pnl = np.mean([t["pnl"] for t in sells]) if n_trades > 0 else 0.0
    return {
        "n_trades": n_trades,
        "win_rate_%": win_rate,
        "avg_trade_pnl": avg_pnl,
    }

# --------------------------------------------------------------------------------------
# Training (SRDDQN): Double DQN + RewardNet with self-rewarding mechanism
# --------------------------------------------------------------------------------------
def train_agent(train_prices, train_features):
    cfg = Config()

    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    # Initialize networks and optimizers
    in_channels = train_features.shape[1]
    seq_len = cfg.STATE_WINDOW
    num_actions = cfg.NUM_ACTIONS

    q_net = QNetwork(in_channels, seq_len, num_actions).to(cfg.DEVICE)
    target_net = QNetwork(in_channels, seq_len, num_actions).to(cfg.DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    reward_net = RewardNet(in_channels, seq_len, num_actions).to(cfg.DEVICE)

    opt_q = torch.optim.Adam(q_net.parameters(), lr=cfg.LR_Q)
    opt_r = torch.optim.Adam(reward_net.parameters(), lr=cfg.LR_REWARD)

    buffer = ReplayBuffer(cfg.REPLAY_SIZE)
    eps_sched = EpsilonScheduler(start=cfg.EPS_START, end=cfg.EPS_END, decay_steps=cfg.EPS_DECAY_STEPS)

    # Training episodes
    steps_since_target_update = 0
    losses_q = []
    losses_r = []

    episode_length = min(3000, len(train_prices) - cfg.STATE_WINDOW - 2)  # reasonable length for learning
    start_indices = np.linspace(cfg.STATE_WINDOW, len(train_prices) - episode_length - 2, num=cfg.EPISODES).astype(int)

    pbar = trange(cfg.EPISODES, desc="Training Episodes", leave=True)
    for ep_idx in pbar:
        env = TradingEnv(
            prices=train_prices,
            features=train_features,
            config=cfg,
            episode_length=episode_length,
            start_index=int(start_indices[ep_idx])
        )
        state = env.reset()

        ep_step_bar = trange(episode_length, desc=f"Episode {ep_idx+1}/{cfg.EPISODES}", leave=False)
        for _ in ep_step_bar:
            # epsilon-greedy
            eps = eps_sched.value()
            eps_sched.step()

            # Prepare state tensor
            s_t = torch.tensor(state, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)  # (1, C, T)

            if np.random.rand() < eps:
                action = np.random.randint(0, num_actions)
            else:
                with torch.no_grad():
                    q_vals = q_net(s_t)
                    action = int(torch.argmax(q_vals, dim=1).item())

            next_state, expert_rewards_dict, done = env.step(action)

            # Expert reward per action: use Sharpe (best) but enable SRDRL (compare predicted vs expert)
            r_exp_sharpe = expert_rewards_dict["sharpe"]  # vector of length 3
            r_exp_return = expert_rewards_dict["return"]
            r_exp_minmax = expert_rewards_dict["minmax"]

            # Build a per-action expert vector by combining Sharpe + (optionally Return/MinMax)
            # We'll take the maximum across expert metrics (Sharpe, Return, MinMax) per action to be robust
            ret_vec_expert = np.stack([r_exp_sharpe, r_exp_return, r_exp_minmax], axis=0)  # shape (3, 3)
            # max across metrics dimension
            ret_vec_expert = np.max(ret_vec_expert, axis=0).astype(np.float32)  # shape (3,)

            # Self-reward predicted by reward_net
            s_t_reward = torch.tensor(state, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)
            with torch.no_grad():
                r_pred = reward_net(s_t_reward).squeeze(0)  # (3,)
            r_pred_np = r_pred.detach().cpu().numpy().astype(np.float32)

            # SRDRL: Select higher elements between predicted and expert per action (vector)
            rlt_vec = np.where(r_pred_np >= ret_vec_expert, r_pred_np, ret_vec_expert).astype(np.float32)
            # Scalar reward to agent is that selected for the action taken
            reward = float(rlt_vec[action])

            buffer.push(state, action, reward, next_state, done, rlt_vec)

            # Update state
            state = next_state

            # Optimize networks if buffer large enough
            if len(buffer) >= cfg.BATCH_SIZE:
                batch = buffer.sample(cfg.BATCH_SIZE)
                b_states, b_actions, b_rewards, b_next_states, b_dones, b_rlt_vecs = batch

                b_states = b_states.to(cfg.DEVICE)  # (B, C, T)
                b_next_states = b_next_states.to(cfg.DEVICE)
                b_actions = b_actions.to(cfg.DEVICE)
                b_rewards = b_rewards.to(cfg.DEVICE)
                b_dones = b_dones.to(cfg.DEVICE)
                b_rlt_vecs = b_rlt_vecs.to(cfg.DEVICE)

                # Q-network update (Double DQN target)
                q_pred = q_net(b_states)  # (B, A)
                q_pred_a = q_pred.gather(1, b_actions.unsqueeze(1)).squeeze(1)  # (B,)

                with torch.no_grad():
                    next_q_main = q_net(b_next_states)  # (B, A)
                    next_actions = torch.argmax(next_q_main, dim=1)  # (B,)
                    next_q_target = target_net(b_next_states)  # (B, A)
                    next_q_target_a = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (B,)
                    td_target = b_rewards + cfg.GAMMA * (1.0 - b_dones) * next_q_target_a  # (B,)

                loss_q = F.mse_loss(q_pred_a, td_target)
                opt_q.zero_grad()
                loss_q.backward()
                opt_q.step()

                losses_q.append(float(loss_q.item()))
                steps_since_target_update += 1

                # RewardNet supervised update to predict rlt_vecs
                r_pred_batch = reward_net(b_states)  # (B, A)
                loss_r = F.mse_loss(r_pred_batch, b_rlt_vecs)
                opt_r.zero_grad()
                loss_r.backward()
                opt_r.step()
                losses_r.append(float(loss_r.item()))

                # Target network periodic update
                if steps_since_target_update >= cfg.TARGET_UPDATE_EVERY:
                    target_net.load_state_dict(q_net.state_dict())
                    steps_since_target_update = 0

            if done:
                break

        # Update episode progress bar postfix
        if len(losses_q) > 0 and len(losses_r) > 0:
            pbar.set_postfix({
                "loss_q": f"{np.mean(losses_q[-50:]):.4f}",
                "loss_r": f"{np.mean(losses_r[-50:]):.4f}",
                "eps": f"{eps:.3f}",
            })

    return q_net, reward_net

# --------------------------------------------------------------------------------------
# Evaluation: forward-moving across the test set, greedy policy (epsilon=0)
# --------------------------------------------------------------------------------------
def evaluate_agent(test_df, scaler, feature_cols, q_net):
    cfg = Config()

    test_prices = test_df["Close"].values.astype(np.float32)
    test_features = transform_features(test_df, scaler, feature_cols).astype(np.float32)

    env = TradingEnv(
        prices=test_prices,
        features=test_features,
        config=cfg,
        episode_length=len(test_prices) - cfg.STATE_WINDOW - 2,
        start_index=cfg.STATE_WINDOW,
        eval_mode=True
    )
    state = env.reset()

    # Greedy action selection
    eval_steps = env.end_index - env.start_index
    eval_bar = trange(eval_steps, desc="Evaluating (forward-moving)", leave=True)
    for _ in eval_bar:
        s_t = torch.tensor(state, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_vals = q_net(s_t)
            action = int(torch.argmax(q_vals, dim=1).item())

        next_state, _, done = env.step(action)
        state = next_state
        if done:
            break

    # Gather evaluation series
    times = test_df.index[env.start_index:env.start_index + len(env.portfolio_values)]
    prices_segment = test_df["Close"].iloc[env.start_index:env.start_index + len(env.portfolio_values)].values

    metrics = compute_performance_metrics(times, prices_segment, env.portfolio_values)
    stats = trade_statistics(env.trade_log)
    return env, times, metrics, stats

# --------------------------------------------------------------------------------------
# Plotting: single window with 4 separated figures (subplots)
# --------------------------------------------------------------------------------------
def plot_results(times, prices, env):
    # times: datetime index
    # env: TradingEnv after evaluation
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=("BTC-USD Close (with trades)",
                                        "Portfolio Value",
                                        "Credit (Cash) Value",
                                        "Holdings (BTC units)"))

    # 1) Price
    fig.add_trace(go.Scatter(x=times, y=prices, name="Close", line=dict(color="blue")), row=1, col=1)

    # Buy markers
    buys = [t for t in env.trade_log if t["type"] == "buy"]
    if len(buys) > 0:
        fig.add_trace(go.Scatter(
            x=[times.iloc[np.where(times == times[t["time_idx"]])[0][0]] if times[0] != times[t["time_idx"]] else times[0] for t in buys] if False else
            [times[t["time_idx"] - env.start_index] for t in buys],
            y=[t["price"] for t in buys],
            mode="markers",
            name="Buy",
            marker=dict(color="green", symbol="triangle-up", size=10)
        ), row=1, col=1)

    # Sell-win markers
    sell_wins = [t for t in env.trade_log if t["type"] == "sell_win"]
    if len(sell_wins) > 0:
        fig.add_trace(go.Scatter(
            x=[times[t["time_idx"] - env.start_index] for t in sell_wins],
            y=[t["price"] for t in sell_wins],
            mode="markers",
            name="Sell (Win)",
            marker=dict(color="orange", symbol="circle", size=9)
        ), row=1, col=1)

    # Sell-loss markers
    sell_losses = [t for t in env.trade_log if t["type"] == "sell_loss"]
    if len(sell_losses) > 0:
        fig.add_trace(go.Scatter(
            x=[times[t["time_idx"] - env.start_index] for t in sell_losses],
            y=[t["price"] for t in sell_losses],
            mode="markers",
            name="Sell (Loss)",
            marker=dict(color="red", symbol="x", size=9)
        ), row=1, col=1)

    # 2) Portfolio value
    fig.add_trace(go.Scatter(
        x=times, y=env.portfolio_values, name="Portfolio", line=dict(color="purple")
    ), row=2, col=1)

    # 3) Credit (Cash)
    fig.add_trace(go.Scatter(
        x=times, y=env.cash_values, name="Cash", line=dict(color="brown")
    ), row=3, col=1)

    # 4) Holdings
    fig.add_trace(go.Scatter(
        x=times, y=env.holdings_values, name="Holdings (BTC)", line=dict(color="black")
    ), row=4, col=1)

    fig.update_layout(height=1000, title="DRL BTC-USD Trading (DDQN + Self-Rewarding) — Walk-Forward Evaluation")
    fig.show()

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    print(f"Running on device: {Config.DEVICE.type}")
    print("Dependencies OK. Loading data...")

    df = load_data()
    print(f"Loaded {len(df):,} rows from {df.index.min()} to {df.index.max()}")

    df = compute_features(df)
    train_df, test_df = split_train_test(df, test_ratio=Config.TEST_SPLIT)

    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    feature_cols = Config.FEATURES
    scaler = prepare_scalers(train_df, feature_cols)
    train_features = transform_features(train_df, scaler, feature_cols).astype(np.float32)
    test_features = transform_features(test_df, scaler, feature_cols).astype(np.float32)

    train_prices = train_df["Close"].values.astype(np.float32)
    test_prices = test_df["Close"].values.astype(np.float32)

    # Train agent
    print("Training agent (DDQN + Self-Rewarding RewardNet)...")
    q_net, reward_net = train_agent(train_prices, train_features)

    # Evaluate agent forward-moving on test set
    print("Evaluating agent (walk-forward on test set)...")
    env, times, metrics, stats = evaluate_agent(test_df, scaler, feature_cols, q_net)

    # Print metrics
    print("\nEvaluation Metrics (Forward-Moving on Test):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nTrade Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Plot results
    plot_results(times, test_prices[env.start_index:env.start_index + len(env.portfolio_values)], env)

    # Print pip command for completeness
    print("\nTo install required dependencies:")
    print("pip install pandas numpy ta plotly tqdm scikit-learn torch")

if __name__ == "__main__":
    main()
