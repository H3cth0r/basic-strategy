import os
import math
import random
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit

import torch
import torch.nn as nn
import torch.optim as optim

import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ===========================
# Data loading (as provided)
# ===========================
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

# ===========================
# Indicators via ta
# ===========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # RSI
    try:
        from ta.momentum import RSIIndicator
    except ImportError:
        raise RuntimeError("Please install 'ta' (pip install ta).")

    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    # Minute returns
    df['ret1'] = df['Close'].pct_change().fillna(0.0)

    # Rolling volatility (std of returns) to normalize Sharpe computation
    df['ret_std_30'] = df['ret1'].rolling(30).std().fillna(0.0)

    # Normalize features using rolling mean/std (robust enough and supported by pandas)
    roll_win = 500
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        roll_mean = df[col].rolling(roll_win).mean()
        roll_std = df[col].rolling(roll_win).std()
        df[f'{col}_z'] = (df[col] - roll_mean) / (roll_std + 1e-8)
    roll_mean_rsi = df['RSI'].rolling(roll_win).mean()
    roll_std_rsi = df['RSI'].rolling(roll_win).std()
    df['RSI_z'] = (df['RSI'] - roll_mean_rsi) / (roll_std_rsi + 1e-8)

    df = df.dropna()
    return df

# ===========================
# Device selection (MPS -> CUDA -> CPU)
# ===========================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# ===========================
# Replay Buffer
# ===========================
@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    reward_vector: torch.Tensor  # rli: vector of rewards for actions

class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.pos = 0
        self.device = device

    def push(self, transition: Transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        # Stack
        states = torch.stack([t.state for t in batch]).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([t.next_state for t in batch]).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        rvecs = torch.stack([t.reward_vector for t in batch]).to(self.device)  # shape [B, 3]
        return states, actions, rewards, next_states, dones, rvecs

    def __len__(self):
        return len(self.buffer)

# ===========================
# Networks: Q-Net (DDQN) and Reward-Net
# ===========================
class QNetwork(nn.Module):
    # Two Conv1D layers with 120 filters each, LeakyReLU, followed by FC, output=3 actions
    def __init__(self, in_channels: int, window: int, n_actions: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=120, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=120, out_channels=120, kernel_size=5, padding=2)
        conv_output_len = window
        self.fc1 = nn.Linear(120 * conv_output_len, 120)
        self.fc2 = nn.Linear(120, 60)
        self.out = nn.Linear(60, n_actions)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = x.flatten(1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.out(x)

class RewardNetwork(nn.Module):
    # TimesNet-inspired Conv1D feature extractor producing a 3-dim reward vector (for actions)
    def __init__(self, in_channels: int, window: int, n_actions: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * window, 64)
        self.fc2 = nn.Linear(64, n_actions)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = x.flatten(1)
        x = self.act(self.fc1(x))
        return self.fc2(x)

# ===========================
# Trading Environment
# ===========================
@dataclass
class Trade:
    time: pd.Timestamp
    action: str  # 'buy' or 'sell'
    price: float
    qty: float
    pnl: float = 0.0
    win: bool = False

class TradingEnv:
    def __init__(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        window: int,
        transaction_cost_bps: float = 10.0,  # 0.1% = 10 bps
        initial_cash: float = 1_000_000.0,
        feature_cols: List[str] = None,
        horizon_k: int = 30,  # future horizon for expert reward
    ):
        self.df = df
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.window = window
        self.tc_bps = transaction_cost_bps
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0.0  # units of BTC
        self.trades: List[Trade] = []
        self.current_idx = start_idx
        self.feature_cols = feature_cols
        self.horizon_k = horizon_k

        self.net_worths = []  # portfolio values over time
        self.cash_series = []
        self.holdings_series = []
        self.price_series = []

        self.avg_cost_basis = 0.0

    def reset(self):
        self.cash = self.initial_cash
        self.position = 0.0
        self.trades = []
        self.net_worths = []
        self.cash_series = []
        self.holdings_series = []
        self.price_series = []
        self.avg_cost_basis = 0.0
        self.current_idx = self.start_idx
        return self._get_state()

    def _get_state(self) -> torch.Tensor:
        # Build state tensor [C, T] from window ending at current_idx
        i0 = self.current_idx - self.window + 1
        if i0 < 0:
            i0 = 0
        idx_slice = self.df.iloc[i0:self.current_idx + 1]
        # If not enough data at start, pad by repeating first row
        if len(idx_slice) < self.window:
            pad_needed = self.window - len(idx_slice)
            first_row = idx_slice.iloc[0]
            first_rows = pd.DataFrame([first_row.values] * pad_needed, columns=idx_slice.columns, index=[first_row.name]*pad_needed)
            idx_slice = pd.concat([first_rows, idx_slice])

        X = idx_slice[self.feature_cols].values.T  # [C, T]
        # Normalize per-channel for stability (z-score within window)
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        return torch.tensor(X, dtype=torch.float32)

    def _portfolio_value(self, price: float) -> float:
        return self.cash + self.position * price

    def _apply_transaction_cost(self, amount: float) -> float:
        cost = amount * (self.tc_bps / 1e4)
        return cost

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Actions:
        0: hold
        1: buy (use all cash)
        2: sell (liquidate all)
        """
        done = False
        info = {}
        price = float(self.df.iloc[self.current_idx]['Close'])
        next_price = float(self.df.iloc[min(self.current_idx + 1, self.end_idx)]['Close'])

        # Execute action
        if action == 1:  # buy with all cash
            if self.cash > 0:
                qty = self.cash / price
                gross = qty * price
                cost = self._apply_transaction_cost(gross)
                qty_net = (gross - cost) / price
                prev_pos = self.position
                self.position += qty_net
                self.cash -= gross
                # Update average cost basis
                if self.position > 1e-12:
                    self.avg_cost_basis = ((self.avg_cost_basis * prev_pos) + price * qty_net) / self.position
                else:
                    self.avg_cost_basis = 0.0
                self.trades.append(Trade(time=self.df.index[self.current_idx], action='buy', price=price, qty=qty_net))
        elif action == 2:  # sell all
            if self.position > 0:
                gross = self.position * price
                cost = self._apply_transaction_cost(gross)
                proceeds = gross - cost
                pnl = (price - self.avg_cost_basis) * self.position - cost
                win = pnl > 0.0
                self.cash += proceeds
                self.trades.append(Trade(time=self.df.index[self.current_idx], action='sell', price=price, qty=self.position, pnl=pnl, win=win))
                self.position = 0.0
                self.avg_cost_basis = 0.0

        # Immediate portfolio change
        pv = self._portfolio_value(price)
        pv_next = self._portfolio_value(next_price)
        step_ret = (pv_next - pv) / (pv + 1e-12)

        # Record series
        self.net_worths.append(pv)
        self.cash_series.append(self.cash)
        self.holdings_series.append(self.position)
        self.price_series.append(price)

        # Move forward
        self.current_idx += 1
        if self.current_idx >= self.end_idx:
            done = True

        next_state = self._get_state()

        info['step_ret'] = step_ret
        info['price'] = price
        info['pv'] = pv

        return next_state, step_ret, done, info

    def compute_expert_reward_vector(self) -> np.ndarray:
        """
        Short-term Sharpe-based expert labeled rewards for actions:
        SSR_k = mean(R_k) / std(R_k) with POSt factor.
        R_k computed as percent changes from current price to future prices up to horizon_k.
        POSt per action: buy -> +1, sell -> -1, hold -> 0.
        """
        i = self.current_idx
        if i >= self.end_idx:
            i = self.end_idx - 1
        price_t = float(self.df.iloc[i]['Close'])
        end_future = min(i + self.horizon_k, self.end_idx)
        future_prices = self.df.iloc[i+1:end_future]['Close'].values
        if len(future_prices) == 0:
            Rk = np.array([0.0], dtype=np.float32)
        else:
            Rk = (future_prices - price_t) / (price_t + 1e-12)

        mu = float(np.mean(Rk)) if len(Rk) > 1 else 0.0
        sigma = float(np.std(Rk)) if len(Rk) > 1 else 1e-8
        ssr = mu / (sigma + 1e-8)

        reward_hold = 0.0
        reward_buy = +1.0 * ssr
        reward_sell = -1.0 * ssr

        return np.array([reward_hold, reward_buy, reward_sell], dtype=np.float32)

# ===========================
# Metrics
# ===========================
def compute_metrics(timestamps: List[pd.Timestamp], price_series: List[float], pv_series: List[float], cash_series: List[float], trades: List[Trade], freq_per_year: int = 525600) -> Dict:
    if len(pv_series) < 2:
        return {}
    pv0 = pv_series[0]
    pv_end = pv_series[-1]
    cr = (pv_end / pv0) - 1.0

    rets = np.diff(pv_series) / (np.array(pv_series[:-1]) + 1e-12)
    mean_ret = np.mean(rets)
    std_ret = np.std(rets) + 1e-12
    sharpe = (mean_ret / std_ret) * math.sqrt(freq_per_year) if std_ret > 0 else 0.0

    n_steps = len(pv_series)
    ar = (pv_end / pv0) ** (freq_per_year / max(n_steps, 1)) - 1.0

    pv_arr = np.array(pv_series)
    peak = np.maximum.accumulate(pv_arr)
    drawdown = (pv_arr - peak) / (peak + 1e-12)
    mdd = float(np.min(drawdown))

    n_trades = len(trades)
    sell_trades = [t for t in trades if t.action == 'sell']
    wins = sum(1 for t in sell_trades if t.win)
    avg_pnl = np.mean([t.pnl for t in sell_trades]) if sell_trades else 0.0
    win_rate = wins / max(len(sell_trades), 1) if sell_trades else 0.0

    return {
        'CR': cr,
        'AR': ar,
        'Sharpe': sharpe,
        'MDD': mdd,
        'Trades': n_trades,
        'WinRate': win_rate,
        'AvgTradePnL': avg_pnl,
    }

# ===========================
# Training Loop (DDQN + SR)
# ===========================
@dataclass
class DRLConfig:
    episodes: int = 10
    gamma: float = 0.98
    lr: float = 1e-3
    eps_start: float = 1.0
    eps_end: float = 0.12
    eps_decay_steps: int = 300
    buffer_size: int = 10000
    batch_size: int = 40
    target_update_every: int = 10
    window: int = 60
    horizon_k: int = 30
    transaction_cost_bps: float = 10.0
    initial_cash: float = 1_000_000.0
    use_self_rewarding: bool = True

class SRDDQNAgent:
    def __init__(self, in_channels: int, window: int, device: torch.device, cfg: DRLConfig):
        self.device = device
        self.cfg = cfg
        self.q = QNetwork(in_channels=in_channels, window=window).to(device)
        self.q_target = QNetwork(in_channels=in_channels, window=window).to(device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.q_opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.loss_q = nn.MSELoss()

        self.reward_net = RewardNetwork(in_channels=in_channels, window=window).to(device)
        self.reward_opt = optim.Adam(self.reward_net.parameters(), lr=cfg.lr)
        self.loss_r = nn.MSELoss()

        self.buffer = ReplayBuffer(cfg.buffer_size, device)

        self.eps = cfg.eps_start
        self.total_steps = 0

    def select_action(self, state: torch.Tensor, greedy: bool = False) -> int:
        if (not greedy) and (random.random() < self.eps):
            return random.randint(0, 2)  # 3 actions
        with torch.no_grad():
            qvals = self.q(state.unsqueeze(0).to(self.device))  # [1, 3]
            return int(torch.argmax(qvals, dim=1).item())

    def update_eps(self):
        # Linear decay
        self.total_steps += 1
        if self.total_steps < self.cfg.eps_decay_steps:
            frac = self.total_steps / self.cfg.eps_decay_steps
            self.eps = self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)
        else:
            self.eps = self.cfg.eps_end

    def optimize(self):
        if len(self.buffer) < self.cfg.batch_size:
            return
        states, actions, rewards, next_states, dones, reward_vecs = self.buffer.sample(self.cfg.batch_size)

        # Q-network TD target (Double DQN)
        with torch.no_grad():
            q_next = self.q(next_states)                      # main net for argmax
            a_next = torch.argmax(q_next, dim=1)             # [B]
            q_target_next = self.q_target(next_states)       # target net for value
            q_next_selected = q_target_next.gather(1, a_next.view(-1, 1)).squeeze(1)  # [B]
            y = rewards + (1.0 - dones) * self.cfg.gamma * q_next_selected

        q_vals = self.q(states).gather(1, actions.view(-1, 1)).squeeze(1)
        loss_q = self.loss_q(q_vals, y)
        self.q_opt.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.q_opt.step()

        # Reward network supervised training on reward vectors (targets)
        pred_rvec = self.reward_net(states)
        loss_r = self.loss_r(pred_rvec, reward_vecs)
        self.reward_opt.zero_grad()
        loss_r.backward()
        torch.nn.utils.clip_grad_norm_((self.reward_net.parameters()), 5.0)
        self.reward_opt.step()

        # Periodically update target net
        if self.total_steps % self.cfg.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss_q.item()), float(loss_r.item())

# ===========================
# Plotting
# ===========================
def plot_results(
    df: pd.DataFrame,
    timestamps: List[pd.Timestamp],
    price_series: List[float],
    pv_series: List[float],
    cash_series: List[float],
    holdings_series: List[float],
    trades: List[Trade],
    title: str = "SRDDQN BTC-USD Evaluation"
):
    # Markers for buy/sell-win/sell-loss
    buy_times = [t.time for t in trades if t.action == 'buy']
    buy_prices = [t.price for t in trades if t.action == 'buy']
    sell_times_win = [t.time for t in trades if t.action == 'sell' and t.win]
    sell_prices_win = [t.price for t in trades if t.action == 'sell' and t.win]
    sell_times_loss = [t.time for t in trades if t.action == 'sell' and not t.win]
    sell_prices_loss = [t.price for t in trades if t.action == 'sell' and not t.win]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.35, 0.25, 0.20, 0.20],
        subplot_titles=("BTC-USD Price", "Portfolio Value", "Cash (Credit) Value", "Holdings (BTC)")
    )

    # Price
    fig.add_trace(go.Scatter(x=timestamps, y=price_series, name='Close', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=buy_times, y=buy_prices, mode='markers', name='Buy',
                             marker=dict(symbol='triangle-up', color='blue', size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_times_win, y=sell_prices_win, mode='markers', name='Sell-Win',
                             marker=dict(symbol='triangle-down', color='green', size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_times_loss, y=sell_prices_loss, mode='markers', name='Sell-Loss',
                             marker=dict(symbol='triangle-down', color='red', size=8)), row=1, col=1)

    # Portfolio
    fig.add_trace(go.Scatter(x=timestamps, y=pv_series, name='Portfolio Value', line=dict(color='royalblue')), row=2, col=1)
    # Cash
    fig.add_trace(go.Scatter(x=timestamps, y=cash_series, name='Cash', line=dict(color='orange')), row=3, col=1)
    # Holdings
    fig.add_trace(go.Scatter(x=timestamps, y=holdings_series, name='BTC Holdings', line=dict(color='purple')), row=4, col=1)

    fig.update_layout(height=900, width=1200, title=title, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()

# ===========================
# Main training & evaluation
# ===========================
def main():
    # Reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    device = get_device()
    print(f"Using device: {device}")

    # Load and prepare data
    df = load_data()
    df = add_indicators(df)

    # Feature columns for state
    feature_cols = ['Open_z', 'High_z', 'Low_z', 'Close_z', 'Volume_z', 'RSI_z', 'ret1', 'ret_std_30']

    # Forward-moving k-folds (TimeSeriesSplit)
    n_splits = 5
    tss = TimeSeriesSplit(n_splits=n_splits)

    cfg = DRLConfig(
        episodes=6,
        gamma=0.98,
        lr=1e-3,
        eps_start=1.0,
        eps_end=0.12,
        eps_decay_steps=300,
        buffer_size=10000,
        batch_size=40,
        target_update_every=10,
        window=60,
        horizon_k=30,
        transaction_cost_bps=10.0,
        initial_cash=1_000_000.0,
        use_self_rewarding=True
    )

    idx_all = np.arange(len(df))
    fold_results = []

    fold_pbar = tqdm(enumerate(tss.split(idx_all)), total=n_splits, desc="Folds")
    final_plot_data = None
    for fold_i, (train_idx, test_idx) in fold_pbar:
        tr_start, tr_end = train_idx[0], train_idx[-1]
        te_start, te_end = test_idx[0], test_idx[-1]

        tr_start = max(tr_start, cfg.window)
        te_start = max(te_start, cfg.window)

        in_channels = len(feature_cols)
        agent = SRDDQNAgent(in_channels=in_channels, window=cfg.window, device=device, cfg=cfg)

        env_train = TradingEnv(
            df=df, start_idx=tr_start, end_idx=tr_end, window=cfg.window,
            transaction_cost_bps=cfg.transaction_cost_bps,
            initial_cash=cfg.initial_cash,
            feature_cols=feature_cols,
            horizon_k=cfg.horizon_k
        )

        ep_pbar = tqdm(range(cfg.episodes), desc=f"Train fold {fold_i+1}/{n_splits}")
        for _ in ep_pbar:
            state = env_train.reset().to(device)
            done = False
            step_losses_q = []
            step_losses_r = []
            while not done:
                action = agent.select_action(state, greedy=False)
                next_state, base_reward, done, info = env_train.step(action)
                next_state = next_state.to(device)

                expert_rvec = env_train.compute_expert_reward_vector()  # [3]
                expert_rvec_t = torch.tensor(expert_rvec, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 3]

                with torch.no_grad():
                    pred_rvec = agent.reward_net(state.unsqueeze(0))  # [1, 3]

                if cfg.use_self_rewarding:
                    sel_rvec = torch.maximum(pred_rvec, expert_rvec_t)  # [1,3]
                else:
                    sel_rvec = expert_rvec_t

                agent_reward = float(sel_rvec[0, action].item())

                # Blend SR-based shaped reward with realized portfolio change
                reward_final = 0.7 * agent_reward + 0.3 * float(info['step_ret'])

                agent.buffer.push(Transition(
                    state=state.cpu(),
                    action=action,
                    reward=reward_final,
                    next_state=next_state.cpu(),
                    done=done,
                    reward_vector=sel_rvec.squeeze(0).detach().cpu()
                ))

                agent.update_eps()
                out = agent.optimize()
                if out is not None:
                    lq, lr = out
                    step_losses_q.append(lq)
                    step_losses_r.append(lr)

                state = next_state

            if step_losses_q:
                ep_pbar.set_postfix({'Lq': np.mean(step_losses_q), 'Lr': np.mean(step_losses_r) if step_losses_r else 0.0})

        # Evaluation on test set (greedy)
        env_test = TradingEnv(
            df=df, start_idx=te_start, end_idx=te_end, window=cfg.window,
            transaction_cost_bps=cfg.transaction_cost_bps,
            initial_cash=cfg.initial_cash,
            feature_cols=feature_cols,
            horizon_k=cfg.horizon_k
        )

        state = env_test.reset().to(device)
        done = False
        while not done:
            action = agent.select_action(state, greedy=True)
            next_state, _, done, _ = env_test.step(action)
            state = next_state.to(device)

        timestamps = list(df.index[te_start:te_end+1])[:len(env_test.price_series)]
        metrics = compute_metrics(
            timestamps=timestamps,
            price_series=env_test.price_series,
            pv_series=env_test.net_worths,
            cash_series=env_test.cash_series,
            trades=env_test.trades
        )
        fold_results.append(metrics)

        print(f"\nFold {fold_i+1}/{n_splits} metrics:")
        print(f"  Cumulative Return: {metrics['CR']*100:.2f}%")
        print(f"  Annualized Return: {metrics['AR']*100:.2f}%")
        print(f"  Sharpe Ratio: {metrics['Sharpe']:.2f}")
        print(f"  Max Drawdown: {metrics['MDD']*100:.2f}%")
        print(f"  Trades: {metrics['Trades']} | WinRate: {metrics['WinRate']*100:.2f}% | AvgTradePnL: {metrics['AvgTradePnL']:.2f}")

        if fold_i == n_splits - 1:
            final_plot_data = (timestamps, env_test.price_series, env_test.net_worths, env_test.cash_series, env_test.holdings_series, env_test.trades)

    if fold_results:
        agg = {}
        for k in fold_results[0].keys():
            vals = [fr[k] for fr in fold_results]
            agg[k] = float(np.mean(vals))
        print("\nAggregate metrics across folds:")
        print(f"  Avg Cumulative Return: {agg['CR']*100:.2f}%")
        print(f"  Avg Annualized Return: {agg['AR']*100:.2f}%")
        print(f"  Avg Sharpe Ratio: {agg['Sharpe']:.2f}")
        print(f"  Avg Max Drawdown: {agg['MDD']*100:.2f}%")
        print(f"  Avg Trades: {agg['Trades']:.2f} | Avg WinRate: {agg['WinRate']*100:.2f}% | Avg AvgTradePnL: {agg['AvgTradePnL']:.2f}")

    if final_plot_data is not None:
        plot_results(
            df=df,
            timestamps=final_plot_data[0],
            price_series=final_plot_data[1],
            pv_series=final_plot_data[2],
            cash_series=final_plot_data[3],
            holdings_series=final_plot_data[4],
            trades=final_plot_data[5],
            title="SRDDQN (Sharpe self-rewarding) on BTC-USD (Test Fold)"
        )

if __name__ == "__main__":
    main()
