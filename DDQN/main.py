import os
import sys
import math
import time
import random
import warnings
from collections import deque
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ta

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Device (MPS for Mac, CUDA for Nvidia, CPU otherwise)
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Data and Features
    SEQ_LEN = 60                      # Lookback window for LSTM/state
    PRED_HORIZON = 10                 # LSTM predicts next-k aggregated return
    SHARPE_K = 30                     # Lookahead window for Sharpe-shaped reward
    FEATURES = ["log_ret", "rsi", "macd_norm", "atr_norm", "cci_norm", "vol_norm"]

    # Trading
    INITIAL_BALANCE = 10000.0
    COMMISSION = 0.001                # 0.1% commission per trade
    STOP_LOSS = -0.03                 # 3%
    TAKE_PROFIT = 0.06                # 6%

    # DRL
    ACTION_DIM = 3                    # 0=Hold, 1=Buy, 2=Sell
    HIDDEN_DIM = 128                  # LSTM hidden size
    GAMMA = 0.99
    LR = 1e-4
    BATCH_SIZE = 64
    MEMORY_SIZE = 20000
    MIN_MEMORY_SIZE = 1000
    TARGET_UPDATE_EVERY = 250
    MAX_TRAIN_STEPS_PER_FOLD = 20000

    # Exploration
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.9995

    # Kangin-style regularization
    ALPHA_INIT = 0.15                 # weight for imitation loss initially
    ALPHA_DECAY = 0.995               # decay per 1k steps
    ALPHA_MIN = 0.02

    # LSTM forecasting
    LSTM_LR = 3e-4
    LSTM_EPOCHS = 10
    LSTM_PATIENCE = 3
    LSTM_HIDDEN = 64
    LSTM_LAYERS = 1
    LSTM_DROPOUT = 0.1

    # Walk-forward
    TRAIN_SIZE = 15000
    TEST_SIZE = 5000
    STEP_SIZE = 5000
    EPOCHS_PER_FOLD = 3               # DRL epochs per fold

    # Synthetic pretraining
    SYNTH_EPOCHS = 2
    SYNTH_SEQ = 5000                  # synthetic sequence length per epoch
    SYNTH_TREND_STRENGTH = 0.0015     # upward/downward drift

# ==========================================
# DATA LOADING
# ==========================================
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

def add_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Log returns
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0)

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi() / 100.0

    # MACD normalized by rolling z-score of macd_diff
    macd = ta.trend.MACD(df["Close"])
    df["macd_diff"] = macd.macd_diff()
    df["macd_norm"] = (df["macd_diff"] - df["macd_diff"].rolling(200).mean()) / (df["macd_diff"].rolling(200).std() + 1e-6)

    # ATR normalized
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    df["atr"] = atr
    df["atr_norm"] = df["atr"] / (df["Close"] + 1e-6)

    # CCI normalized
    cci = ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"]).cci()
    df["cci"] = cci
    df["cci_norm"] = df["cci"] / 200.0

    # Volume change clipped
    vol_chg = df["Volume"].pct_change().fillna(0.0)
    df["vol_norm"] = vol_chg.clip(-1.0, 1.0)

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df, Config.FEATURES

# ==========================================
# LSTM FORECASTER (SUPERVISED)
# Predict next-k aggregated return to be used as a feature in DRL
# ==========================================
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, num_layers=layers, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        h = out[:, -1, :]  # last step
        return self.head(h)

def prepare_lstm_data(df: pd.DataFrame, feature_cols: List[str], pred_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    # target: future aggregated return over pred_horizon
    close = df["Close"].values
    fut_ret = (close[pred_horizon:] - close[:-pred_horizon]) / (close[:-pred_horizon] + 1e-8)
    fut_ret = np.concatenate([fut_ret, np.zeros(pred_horizon)])  # pad last horizon
    X = df[feature_cols].values
    return X, fut_ret

def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    sequences = []
    targets = []
    for i in range(seq_len, n):
        sequences.append(X[i - seq_len:i])
        targets.append(y[i])
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

def train_lstm_forecaster(df_train, df_valid, feature_cols):
    Xtr, ytr = prepare_lstm_data(df_train, feature_cols, Config.PRED_HORIZON)
    Xva, yva = prepare_lstm_data(df_valid, feature_cols, Config.PRED_HORIZON)

    Xtr_seq, ytr_seq = make_sequences(Xtr, ytr, Config.SEQ_LEN)
    Xva_seq, yva_seq = make_sequences(Xva, yva, Config.SEQ_LEN)

    model = LSTMForecaster(input_dim=len(feature_cols), hidden=Config.LSTM_HIDDEN, layers=Config.LSTM_LAYERS, dropout=Config.LSTM_DROPOUT).to(Config.DEVICE)
    opt = optim.Adam(model.parameters(), lr=Config.LSTM_LR)
    crit = nn.SmoothL1Loss()

    best_val = float("inf")
    no_improve = 0

    train_ds = torch.utils.data.TensorDataset(torch.tensor(Xtr_seq), torch.tensor(ytr_seq).unsqueeze(1))
    val_ds = torch.utils.data.TensorDataset(torch.tensor(Xva_seq), torch.tensor(yva_seq).unsqueeze(1))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)

    for ep in range(1, Config.LSTM_EPOCHS + 1):
        model.train()
        ep_loss = []
        for xb, yb in tqdm(train_loader, desc=f"LSTM Ep {ep}/{Config.LSTM_EPOCHS}", leave=False):
            xb = xb.to(Config.DEVICE)
            yb = yb.to(Config.DEVICE)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss.append(loss.item())
        train_l = float(np.mean(ep_loss))

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(Config.DEVICE); yb = yb.to(Config.DEVICE)
                pred = model(xb)
                val_losses.append(crit(pred, yb).item())
        val_l = float(np.mean(val_losses))
        print(f"[LSTM] Ep {ep}: train {train_l:.5f} | val {val_l:.5f}")

        if val_l < best_val - 1e-4:
            best_val = val_l
            no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= Config.LSTM_PATIENCE:
            print("[LSTM] Early stopping.")
            break

    if 'best_state' in locals():
        model.load_state_dict(best_state)

    return model

def add_lstm_predictions(model: LSTMForecaster, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X, y = prepare_lstm_data(df, feature_cols, Config.PRED_HORIZON)
    Xseq, _ = make_sequences(X, y, Config.SEQ_LEN)

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xseq), 1024):
            xb = torch.tensor(Xseq[i:i+1024], dtype=torch.float32, device=Config.DEVICE)
            p = model(xb).squeeze(1).cpu().numpy()
            preds.extend(p.tolist())
    # Align back to original index
    pred_series = pd.Series([np.nan]*Config.SEQ_LEN + preds, index=df.index)
    df["lstm_pred_ret"] = pred_series.fillna(method="bfill").fillna(0.0)
    return df

# ==========================================
# REFERENCE ACTOR (RSI crossings)
# Kangin-style supervised guidance
# ==========================================
def reference_action_rsi(rsi_series: pd.Series) -> np.ndarray:
    # 0 hold, 1 buy, 2 sell; triggers on RSI crosses
    actions = np.zeros(len(rsi_series), dtype=np.int64)
    rsi = rsi_series.values
    for i in range(1, len(rsi)):
        # Oversold crossing upwards => buy
        if rsi[i-1] < 0.30 and rsi[i] >= 0.30:
            actions[i] = 1
        # Overbought crossing downwards => sell
        elif rsi[i-1] > 0.70 and rsi[i] <= 0.70:
            actions[i] = 2
        else:
            actions[i] = 0
    return actions

# ==========================================
# TRADING ENVIRONMENT
# ==========================================
class TradingEnv:
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], include_pred: bool = True):
        self.df = df.copy()
        self.feature_cols = feature_cols.copy()
        self.include_pred = include_pred
        if include_pred and "lstm_pred_ret" in self.df.columns:
            self.feature_cols = self.feature_cols + ["lstm_pred_ret"]
        self.n_steps = len(self.df)
        self.prices = self.df["Close"].values
        self.dates = self.df.index
        self.data_matrix = self.df[self.feature_cols].values
        self.ref_actions = reference_action_rsi(self.df["rsi"])
        self.reset()

    def reset(self):
        self.current = Config.SEQ_LEN
        self.balance = Config.INITIAL_BALANCE
        self.shares = 0.0
        self.entry_price = 0.0
        self.net_worth = Config.INITIAL_BALANCE
        self.prev_nw = Config.INITIAL_BALANCE
        self.trades = []
        self.history = []
        return self._state()

    def _state(self):
        window = self.data_matrix[self.current-Config.SEQ_LEN:self.current]
        pos_flag = 1.0 if self.shares > 0 else 0.0
        pos_col = np.full((Config.SEQ_LEN, 1), pos_flag, dtype=np.float32)
        state = np.hstack((window, pos_col))
        return state

    def _sharpe_reward(self, k: int) -> float:
        # Lookahead returns for Sharpe-like reward shaping
        if self.current + k >= self.n_steps:
            return 0.0
        p0 = self.prices[self.current]
        future = self.prices[self.current+1:self.current+k+1]
        ret = (future - p0) / (p0 + 1e-8)
        if ret.std() < 1e-6:
            return 0.0
        return ret.mean() / (ret.std() + 1e-8)

    def step(self, action: int):
        price = self.prices[self.current]
        date = self.dates[self.current]
        done = False
        info_trade = None

        # Apply action
        if action == 1:  # BUY
            if self.shares == 0 and self.balance > 0:
                cost_fee = self.balance * Config.COMMISSION
                self.shares = (self.balance - cost_fee) / price
                self.balance = 0.0
                self.entry_price = price
                self.trades.append({"date": date, "type": "buy", "price": price})
        elif action == 2:  # SELL
            if self.shares > 0:
                revenue = self.shares * price
                fee = revenue * Config.COMMISSION
                profit_pct = (price - self.entry_price) / (self.entry_price + 1e-8)
                self.balance = revenue - fee
                self.shares = 0.0
                self.entry_price = 0.0
                self.trades.append({"date": date, "type": "sell", "price": price, "win": profit_pct > 0, "profit": profit_pct})
                info_trade = self.trades[-1]

        # Risk management
        if self.shares > 0:
            unreal = (price - self.entry_price) / (self.entry_price + 1e-8)
            if unreal < Config.STOP_LOSS or unreal > Config.TAKE_PROFIT:
                revenue = self.shares * price
                fee = revenue * Config.COMMISSION
                self.balance = revenue - fee
                win_flag = unreal > 0
                self.trades.append({"date": date, "type": "sell", "price": price, "win": win_flag, "profit": unreal})
                self.shares = 0.0
                self.entry_price = 0.0

        # Update net worth
        self.net_worth = self.balance + self.shares * price

        # Immediate reward: net worth change (percent)
        r_imm = (self.net_worth - self.prev_nw) / (self.prev_nw + 1e-8)

        # Sharpe lookahead reward
        sharpe_r = self._sharpe_reward(Config.SHARPE_K)
        # Position-sensitive shaping
        if self.shares > 0:
            r_shape = sharpe_r
        else:
            r_shape = -0.1 * sharpe_r

        # Combine rewards (scale to %)
        reward = 100.0 * r_imm + 10.0 * r_shape

        self.prev_nw = self.net_worth

        # Record history
        self.history.append({
            "date": date,
            "price": price,
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares": self.shares,
            "reward": reward,
            "ref_action": int(self.ref_actions[self.current]),
            "action": int(action),
        })

        self.current += 1
        if self.current >= self.n_steps - Config.SHARPE_K - 1:
            done = True

        return self._state(), reward, done, info_trade

# ==========================================
# DDQN AGENT (DUELING + LSTM FEATURE)
# With Kangin-style imitation regularizer
# ==========================================
class DuelingSRDDQN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.adv = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_dim)
        )
        self.val = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        adv = self.adv(h)
        val = self.val(h)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q

class Agent:
    def __init__(self, input_dim, action_dim):
        self.action_dim = action_dim
        self.policy = DuelingSRDDQN(input_dim, Config.HIDDEN_DIM, action_dim).to(Config.DEVICE)
        self.target = DuelingSRDDQN(input_dim, Config.HIDDEN_DIM, action_dim).to(Config.DEVICE)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt = optim.AdamW(self.policy.parameters(), lr=Config.LR, amsgrad=True)
        self.huber = nn.SmoothL1Loss()

        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.epsilon = Config.EPSILON_START
        self.learn_steps = 0
        self.alpha = Config.ALPHA_INIT  # Kangin-style regularization weight

    def select_action(self, state: np.ndarray, is_eval=False) -> int:
        if not is_eval and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
            q = self.policy(s)
            return int(q.argmax().item())

    def store(self, state, action, reward, next_state, done, ref_action):
        self.memory.append((state, action, reward, next_state, done, ref_action))

    def train_step(self):
        if len(self.memory) < max(Config.MIN_MEMORY_SIZE, Config.BATCH_SIZE):
            return None

        batch = random.sample(self.memory, Config.BATCH_SIZE)
        st, act, rew, nst, dn, ref_act = zip(*batch)

        st = torch.tensor(np.array(st), dtype=torch.float32, device=Config.DEVICE)
        nst = torch.tensor(np.array(nst), dtype=torch.float32, device=Config.DEVICE)
        act = torch.tensor(act, dtype=torch.int64, device=Config.DEVICE).unsqueeze(1)
        rew = torch.tensor(rew, dtype=torch.float32, device=Config.DEVICE).unsqueeze(1)
        dn = torch.tensor(dn, dtype=torch.float32, device=Config.DEVICE).unsqueeze(1)
        ref_act = torch.tensor(ref_act, dtype=torch.int64, device=Config.DEVICE)

        # Current Q
        q_all = self.policy(st)
        q_curr = q_all.gather(1, act)

        # Double DQN target
        with torch.no_grad():
            q_next_online = self.policy(nst)
            next_best = q_next_online.argmax(dim=1, keepdim=True)
            q_next_target = self.target(nst)
            q_next = q_next_target.gather(1, next_best)
            target_q = rew + Config.GAMMA * q_next * (1.0 - dn)

        # Q-loss
        q_loss = self.huber(q_curr, target_q)

        # Kangin-style regularization:
        # Compute policy probabilities via softmax of Q-values, and cross-entropy toward ref action
        logits = q_all
        probs = torch.softmax(logits, dim=1)
        # CE loss toward one-hot ref action
        ref_onehot = torch.zeros_like(probs)
        ref_onehot.scatter_(1, ref_act.unsqueeze(1), 1.0)
        ce_loss = -(ref_onehot * torch.log(probs + 1e-8)).sum(dim=1).mean()

        # Weight alpha decays over time; optionally boost when ref action is not hold
        w = 1.0 + 0.5 * (ref_act != 0).float().mean().item()
        total_loss = q_loss + (self.alpha * w) * ce_loss

        self.opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

        # Update target periodically
        self.learn_steps += 1
        if self.learn_steps % Config.TARGET_UPDATE_EVERY == 0:
            self.target.load_state_dict(self.policy.state_dict())

        # Epsilon decay
        if self.epsilon > Config.EPSILON_END:
            self.epsilon *= Config.EPSILON_DECAY

        # Alpha decay (Kangin)
        if self.alpha > Config.ALPHA_MIN:
            self.alpha = max(Config.ALPHA_MIN, self.alpha * Config.ALPHA_DECAY)

        return float(q_loss.item()), float(ce_loss.item())

# ==========================================
# SYNTHETIC PRETRAINING (OPTIONAL)
# Teach basic money-making behavior
# ==========================================
def synthetic_series(n: int, drift: float, vol: float = 0.002) -> np.ndarray:
    ret = np.random.normal(loc=drift, scale=vol, size=n)
    price = 100.0 * np.exp(np.cumsum(ret))
    return price

def pretrain_on_synthetic(agent: Agent):
    print(">>> Synthetic pretraining...")
    for ep in range(1, Config.SYNTH_EPOCHS + 1):
        # Alternate upward/downward drift
        drift = Config.SYNTH_TREND_STRENGTH if ep % 2 == 1 else -Config.SYNTH_TREND_STRENGTH
        price = synthetic_series(Config.SYNTH_SEQ + 2 * Config.SEQ_LEN, drift)
        df = pd.DataFrame({"Close": price})
        # Minimal features
        df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0)
        df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi() / 100.0
        df["macd_norm"] = 0.0
        df["atr_norm"] = 0.0
        df["cci_norm"] = 0.0
        df["vol_norm"] = 0.0
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        env = TradingEnv(df, ["log_ret", "rsi", "macd_norm", "atr_norm", "cci_norm", "vol_norm"], include_pred=False)
        state = env.reset()
        done = False
        steps = 0

        pbar = tqdm(desc=f"Synth Ep {ep}/{Config.SYNTH_EPOCHS}", total=Config.SYNTH_SEQ, leave=False)
        while not done and steps < Config.SYNTH_SEQ:
            # Reference actor guidance
            ref_act = env.ref_actions[env.current]
            action = agent.select_action(state)
            next_state, reward, done, info_trade = env.step(action)
            agent.store(state, action, reward, next_state, done, ref_act)
            agent.train_step()
            state = next_state
            steps += 1
            pbar.update(1)
        pbar.close()
    print(">>> Synthetic pretraining finished.")

# ==========================================
# PLOTTING
# ==========================================
def plot_results(history: List[Dict], trades: List[Dict], fold_id: int, title_suffix: str = ""):
    dfh = pd.DataFrame(history).set_index("date")
    # Build buy & hold baseline
    initial_price = dfh["price"].iloc[0]
    initial_bal = dfh["net_worth"].iloc[0]
    dfh["buyhold"] = (dfh["price"] / initial_price) * initial_bal

    # Trades split
    buys = [t for t in trades if t.get("type") == "buy"]
    sells = [t for t in trades if t.get("type") == "sell"]
    win_sells = [t for t in sells if t.get("win", False)]
    loss_sells = [t for t in sells if not t.get("win", False)]

    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=(
            f"BTC Price & Trades (Fold {fold_id})",
            "Portfolio Value",
            "Cash (Credit) Balance",
            "Holdings (BTC)",
            "LSTM Predicted Next-k Return",
            "Per-step Reward"
        ),
        row_heights=[0.28, 0.18, 0.14, 0.14, 0.14, 0.12]
    )

    # Row 1: Price & trades
    fig.add_trace(go.Scatter(x=dfh.index, y=dfh["price"], name="BTC", line=dict(color="cornflowerblue")), row=1, col=1)
    if buys:
        fig.add_trace(go.Scatter(x=[t["date"] for t in buys], y=[t["price"] for t in buys],
                                 mode="markers", name="Buy", marker=dict(symbol="triangle-up", size=10, color="green")), row=1, col=1)
    if win_sells:
        fig.add_trace(go.Scatter(x=[t["date"] for t in win_sells], y=[t["price"] for t in win_sells],
                                 mode="markers", name="Sell (Win)", marker=dict(symbol="triangle-down", size=10, color="lime")), row=1, col=1)
    if loss_sells:
        fig.add_trace(go.Scatter(x=[t["date"] for t in loss_sells], y=[t["price"] for t in loss_sells],
                                 mode="markers", name="Sell (Loss)", marker=dict(symbol="triangle-down", size=10, color="red")), row=1, col=1)

    # Row 2: Portfolio vs Buy&Hold
    fig.add_trace(go.Scatter(x=dfh.index, y=dfh["net_worth"], name="Net Worth", line=dict(color="purple")), row=2, col=1)
    fig.add_trace(go.Scatter(x=dfh.index, y=dfh["buyhold"], name="Buy&Hold (baseline)", line=dict(color="gray", dash="dot")), row=2, col=1)

    # Row 3: Cash balance
    fig.add_trace(go.Scatter(x=dfh.index, y=dfh["balance"], name="Cash (Credit)", line=dict(color="goldenrod")), row=3, col=1)

    # Row 4: Holdings
    fig.add_trace(go.Scatter(x=dfh.index, y=dfh["shares"], name="BTC position", line=dict(color="orange"), fill="tozeroy"), row=4, col=1)

    # Row 5: LSTM predicted return
    if "lstm_pred_ret" in dfh.columns:
        fig.add_trace(go.Scatter(x=dfh.index, y=dfh["lstm_pred_ret"], name=f"LSTM pred next-{Config.PRED_HORIZON} ret", line=dict(color="teal")), row=5, col=1)

    # Row 6: reward (smoothed)
    fig.add_trace(go.Scatter(x=dfh.index, y=pd.Series(dfh["reward"]).rolling(20).mean(), name="Reward (rolling)", line=dict(color="firebrick")), row=6, col=1)

    fig.update_layout(height=1400, template="plotly_dark",
                      title=f"SR-DDQN Walk-Forward Results - Fold {fold_id} {title_suffix}")
    fig.show()

# ==========================================
# METRICS
# ==========================================
def compute_metrics(env: TradingEnv, df_test: pd.DataFrame):
    final_nw = env.net_worth
    bot_profit_pct = (final_nw - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE * 100.0
    bnh_profit_pct = (df_test["Close"].iloc[-1] - df_test["Close"].iloc[0]) / df_test["Close"].iloc[0] * 100.0

    # Win rate
    sells = [t for t in env.trades if t.get("type") == "sell"]
    win_sells = [t for t in sells if t.get("win", False)]
    win_rate = (len(win_sells) / len(sells) * 100.0) if sells else 0.0

    # Max drawdown over net_worth path
    h = pd.DataFrame(env.history)
    nw = h["net_worth"].values
    if len(nw) > 1:
        cummax = np.maximum.accumulate(nw)
        drawdowns = (nw - cummax) / (cummax + 1e-8)
        mdd = drawdowns.min() * 100.0
    else:
        mdd = 0.0

    # Sharpe over per-step net worth returns
    nw_ret = pd.Series(nw).pct_change().dropna()
    if len(nw_ret) > 10 and nw_ret.std() > 1e-8:
        sharpe = (nw_ret.mean() / (nw_ret.std() + 1e-8)) * np.sqrt(252*24*60)  # minute data -> annualize roughly
    else:
        sharpe = 0.0

    return {
        "bot_profit_pct": bot_profit_pct,
        "bnh_profit_pct": bnh_profit_pct,
        "win_rate": win_rate,
        "trades": len(env.trades),
        "mdd_pct": mdd,
        "sharpe": sharpe
    }

# ==========================================
# MAIN WALK-FORWARD
# ==========================================
def main():
    print(">>> Device:", Config.DEVICE)
    print(">>> Loading data...")
    raw = load_data()
    df, features = add_features(raw)
    df = df[~df.index.duplicated(keep="first")]
    print(f">>> Data loaded: {len(df)} rows")

    # Global LSTM Forecaster: trained per fold (train-valid split inside train segment)
    # Walk-forward
    total_len = len(df)
    current_idx = 0
    fold = 1

    # Agent init
    input_dim = len(features) + 1  # + position flag; LSTM pred will be appended dynamically if present
    agent = Agent(input_dim=input_dim + 1, action_dim=Config.ACTION_DIM)  # +1 for lstm_pred_ret

    # Synthetic pretraining (brief)
    pretrain_on_synthetic(agent)

    while current_idx + Config.TRAIN_SIZE + Config.TEST_SIZE < total_len:
        print("\n" + "="*70)
        print(f"FOLD {fold} | Walk-Forward")
        print("="*70)

        train_start = current_idx
        train_end = train_start + Config.TRAIN_SIZE
        test_start = train_end
        test_end = test_start + Config.TEST_SIZE

        df_train = df.iloc[train_start:train_end].copy()
        df_test = df.iloc[test_start:test_end].copy()

        # Split train into LSTM train/valid for forecaster
        split = int(len(df_train) * 0.8)
        df_lstm_tr = df_train.iloc[:split]
        df_lstm_va = df_train.iloc[split:]

        # Train LSTM forecaster on this fold
        lstm_model = train_lstm_forecaster(df_lstm_tr, df_lstm_va, features)

        # Append LSTM predictions to both train/test
        df_train = add_lstm_predictions(lstm_model, df_train, features)
        df_test = add_lstm_predictions(lstm_model, df_test, features)

        # Train DRL
        print(">>> Training DRL on train window")
        env = TradingEnv(df_train, features, include_pred=True)
        agent.epsilon = max(0.3, agent.epsilon)  # allow exploration per regime
        train_steps = 0
        for ep in range(1, Config.EPOCHS_PER_FOLD + 1):
            state = env.reset()
            done = False
            pbar = tqdm(total=min(Config.MAX_TRAIN_STEPS_PER_FOLD // Config.EPOCHS_PER_FOLD, len(df_train)), desc=f"Train Ep {ep}/{Config.EPOCHS_PER_FOLD}", leave=False)
            while not done and train_steps < Config.MAX_TRAIN_STEPS_PER_FOLD:
                ref_a = env.ref_actions[env.current]
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store(state, action, reward, next_state, done, ref_a)
                agent.train_step()
                state = next_state
                train_steps += 1
                pbar.update(1)
            pbar.close()
            print(f"   Ep {ep}: NW={env.net_worth:.2f} | epsilon={agent.epsilon:.3f} | alpha={agent.alpha:.3f}")

        # Test DRL (greedy)
        print(">>> Testing on test window")
        test_env = TradingEnv(df_test, features, include_pred=True)
        state = test_env.reset()
        done = False
        pbar_t = tqdm(total=len(df_test), desc=f"Test Fold {fold}", leave=False)
        while not done:
            action = agent.select_action(state, is_eval=True)
            state, _, done, info = test_env.step(action)
            pbar_t.update(1)
        pbar_t.close()

        metrics = compute_metrics(test_env, df_test)
        print(f"\nRESULTS FOLD {fold}:")
        print(f"Bot Profit:  {metrics['bot_profit_pct']:.2f}%")
        print(f"Buy & Hold:  {metrics['bnh_profit_pct']:.2f}%")
        print(f"Win Rate:    {metrics['win_rate']:.2f}% ({len([t for t in test_env.trades if t.get('type')=='sell' and t.get('win', False)])}/{len([t for t in test_env.trades if t.get('type')=='sell'])})")
        print(f"Trades:      {metrics['trades']}")
        print(f"MaxDD:       {metrics['mdd_pct']:.2f}%")
        print(f"Sharpe (~):  {metrics['sharpe']:.2f}")

        title_suffix = f"| Profit {metrics['bot_profit_pct']:.2f}% | MDD {metrics['mdd_pct']:.2f}% | Sharpe {metrics['sharpe']:.2f}"
        plot_results(test_env.history, test_env.trades, fold_id=fold, title_suffix=title_suffix)

        current_idx += Config.STEP_SIZE
        fold += 1

    print(">>> Walk-forward completed.")

if __name__ == "__main__":
    main()
