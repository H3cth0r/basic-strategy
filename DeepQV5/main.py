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


# -------------------------------------------------
# 1. DATA  LOADING AND PRE-PROCESSING
# -------------------------------------------------
def get_data():
    """
    Downloads and preprocesses the Bitcoin 1-minute interval data.
    """
    print("Downloading and preprocessing data …")
    url = (
        "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/"
        "data/CRYPTO/BTC-USD/data_0.csv"
    )
    column_names = ["Datetime", "Close", "High", "Low", "Open", "Volume"]

    try:
        df = pd.read_csv(
            url,
            skiprows=[1, 2],
            header=0,
            names=column_names,
            parse_dates=["Datetime"],
            index_col="Datetime",
            dtype={"Volume": "int64"},
            na_values=["NA", "N/A", ""],
            keep_default_na=True,
        )
        df.index = pd.to_datetime(df.index, utc=True)
    except Exception as e:
        print(f"Error reading data: {e}")
        return pd.DataFrame()

    df.ffill(inplace=True)
    df.dropna(inplace=True)

    print("Calculating technical indicators …")
    add_all_ta_features(
        df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True,
    )
    original_close = df["Close"].copy()
    # z-score normalisation
    for col in df.columns:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-7)

    df["Original_Close"] = original_close
    df.dropna(inplace=True)
    print("Data loaded and preprocessed successfully.")
    print(df.head())
    print(f"Data shape: {df.shape}")
    return df


# -------------------------------------------------
# 2. ATTENTION
# -------------------------------------------------
class SharedSelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim, attention_heads=1, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.attention_heads = attention_heads

        if self.attention_dim % self.attention_heads != 0:
            raise ValueError(
                f"Attention dim ({self.attention_dim}) must be divisible by "
                f"the number of heads ({self.attention_heads})."
            )

        self.head_dim = self.attention_dim // self.attention_heads
        self.query_proj = nn.Linear(input_dim, self.attention_dim)
        self.key_proj = nn.Linear(input_dim, self.attention_dim)
        self.value_proj = nn.Linear(input_dim, self.attention_dim)
        self.output_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sequence_features):
        if sequence_features.ndim == 2:
            sequence_features = sequence_features.unsqueeze(0)

        batch_size, seq_len, _ = sequence_features.shape
        Q = self.query_proj(sequence_features)
        K = self.key_proj(sequence_features)
        V = self.value_proj(sequence_features)

        Q = (
            Q.view(batch_size, seq_len, self.attention_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        K = (
            K.view(batch_size, seq_len, self.attention_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        V = (
            V.view(batch_size, seq_len, self.attention_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(energy, dim=-1)
        attention_weights = self.dropout(attention_weights)

        weighted_values = torch.matmul(attention_weights, V)
        weighted_values = weighted_values.permute(0, 2, 1, 3).contiguous()
        weighted_values = weighted_values.view(batch_size, seq_len, self.attention_dim)

        output = self.output_proj(weighted_values)
        return output.mean(dim=1)


# -------------------------------------------------
# 3. D Q N  AGENT
# -------------------------------------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, attention_dim, attention_heads):
        super().__init__()
        self.attention = SharedSelfAttention(
            state_dim - 2, attention_dim, attention_heads
        )
        self.fc1 = nn.Linear(attention_dim + 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        market_data = state[:, :, :-2]
        portfolio_state = state[:, -1, -2:]
        attention_output = self.attention(market_data)
        combined_input = torch.cat([attention_output, portfolio_state], dim=1)
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        attention_dim,
        attention_heads,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = QNetwork(
            state_dim, action_dim, attention_dim, attention_heads
        ).to(self.device)
        self.target_net = QNetwork(
            state_dim, action_dim, attention_dim, attention_heads
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(10000)

    def act(self, state, is_eval=False):
        if is_eval or random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()
        else:
            return random.randrange(self.action_dim)

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")


# -------------------------------------------------
# 4. TRADING  ENVIRONMENT
# -------------------------------------------------
class TradingEnvironment:
    def __init__(
        self,
        data,
        initial_credit=10000,
        fee=0.001,
        window_size=180,
        min_trade_amount_buy=100,
        min_trade_amount_sell=100,
    ):
        self.data = data
        self.normalized_data = data.drop(columns=["Original_Close"])
        self.initial_credit = initial_credit
        self.fee = fee
        self.window_size = window_size
        self.min_trade_amount_buy = min_trade_amount_buy
        self.min_trade_amount_sell = min_trade_amount_sell
        self.n_features = self.normalized_data.shape[1] + 2
        self.action_space = [-0.5, -0.25, 0, 0.25, 0.5]
        self.n_actions = len(self.action_space)

    def reset(self, episode_start_index=0, initial_credit=None, holdings=0.0):
        self.credit = initial_credit if initial_credit is not None else self.initial_credit
        self.holdings = holdings
        self.average_buy_price = 0.0
        self.current_step = episode_start_index + self.window_size
        self.trades = []
        
        # Initialize high-water marks
        self.max_credit = self.credit
        initial_price = self.data["Original_Close"].iloc[self.current_step - 1]
        self.max_portfolio_value = self.credit + self.holdings * initial_price
        return self._get_state()

    def _get_state(self):
        start = self.current_step - self.window_size
        end = self.current_step
        market_data = self.normalized_data.iloc[start:end].values
        current_price = self.data["Original_Close"].iloc[self.current_step]
        portfolio_value = self.credit + self.holdings * current_price
        holdings_ratio = (
            (self.holdings * current_price) / portfolio_value if portfolio_value > 0 else 0
        )
        credit_ratio = self.credit / portfolio_value if portfolio_value > 0 else 0
        portfolio_state = np.array([[holdings_ratio, credit_ratio]] * self.window_size)
        return np.concatenate([market_data, portfolio_state], axis=1)

    def step(self, action_idx):
        action = self.action_space[action_idx]
        current_price = self.data["Original_Close"].iloc[self.current_step]
        
        reward = 0.0
        done = False

        # ----- SELL
        if action < 0:
            sell_fraction = -action
            sell_amount_in_usd = self.holdings * sell_fraction * current_price
            if sell_amount_in_usd > self.min_trade_amount_sell:
                sell_amount = self.holdings * sell_fraction
                
                self.credit += sell_amount * current_price * (1 - self.fee)
                self.holdings -= sell_amount
                
                realized_pnl = (current_price - self.average_buy_price) * sell_amount
                reward += realized_pnl 
                
                if self.credit > self.max_credit:
                    credit_increase = self.credit - self.max_credit
                    reward += credit_increase * 5.0
                    self.max_credit = self.credit
                
                self.trades.append(
                    {
                        "step": self.current_step,
                        "type": "sell",
                        "price": current_price,
                        "amount": sell_amount,
                    }
                )
                if self.holdings < 1e-6:
                    self.average_buy_price = 0
        # ----- BUY
        elif action > 0:
            buy_fraction = action
            investment = self.credit * buy_fraction
            if investment > self.min_trade_amount_buy:
                buy_amount = (investment / current_price) * (1 - self.fee)
                total_cost = (self.average_buy_price * self.holdings) + investment
                self.holdings += buy_amount
                self.credit -= investment
                self.average_buy_price = (
                    total_cost / self.holdings if self.holdings > 0 else 0
                )
                self.trades.append(
                    {
                        "step": self.current_step,
                        "type": "buy",
                        "price": current_price,
                        "amount": buy_amount,
                    }
                )
        # ----- HOLD
        else:
            # *** FIX IS HERE ***
            # Apply a small penalty for inaction to encourage trading
            reward = -0.01 
        
        # ----- ADVANCE ONE STEP
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        next_state = self._get_state() if not done else None
        next_price = (
            self.data["Original_Close"].iloc[self.current_step] if not done else current_price
        )
        portfolio_value = self.credit + self.holdings * next_price

        # Portfolio value shaping reward
        if portfolio_value > self.max_portfolio_value:
            reward += (portfolio_value - self.max_portfolio_value) * 0.05
            self.max_portfolio_value = portfolio_value

        if done:
            # liquidate
            self.credit += self.holdings * current_price * (1 - self.fee)
            self.holdings = 0
            portfolio_value = self.credit

        return next_state, reward, done, {
            "portfolio_value": portfolio_value,
            "credit": self.credit,
            "holdings": self.holdings,
            "trades": self.trades,
        }


# -------------------------------------------------
# 5. PLOTTING
# -------------------------------------------------
def plot_results(
    df,
    episode,
    portfolio_history,
    credit_history,
    holdings_history,
    trades,
    plot_title_prefix="",
    segment_boundaries=None,
):
    """
    Draw price + metrics; if segment_boundaries (list of indices into df)
    is supplied, it draws a dashed vertical line at each boundary.
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f"{plot_title_prefix} Price and Trades",
            "Portfolio Value",
            "Credit",
            "Holdings Value",
        ),
    )
    
    # Ensure plotting dataframe aligns with the history data
    start_index = len(df) - len(portfolio_history)
    plot_df = df.iloc[start_index:].copy()


    plot_df["portfolio_value"] = portfolio_history
    plot_df["credit"] = credit_history
    plot_df["holdings_value"] = holdings_history

    # --- Price line
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["Original_Close"],
            mode="lines",
            name="Price",
            line=dict(color="lightgrey"),
        ),
        row=1,
        col=1,
    )

    # --- Trades
    buy_trades = [t for t in trades if t["type"] == "buy"]
    sell_trades = [t for t in trades if t["type"] == "sell"]
    if buy_trades:
        fig.add_trace(
            go.Scatter(
                x=[df.index[t["step"]] for t in buy_trades],
                y=[t["price"] for t in buy_trades],
                mode="markers",
                marker=dict(color="green", symbol="triangle-up", size=8),
                name="Buy",
            ),
            row=1,
            col=1,
        )
    if sell_trades:
        fig.add_trace(
            go.Scatter(
                x=[df.index[t["step"]] for t in sell_trades],
                y=[t["price"] for t in sell_trades],
                mode="markers",
                marker=dict(color="red", symbol="triangle-down", size=8),
                name="Sell",
            ),
            row=1,
            col=1,
        )

    # --- Metrics
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["portfolio_value"], mode="lines", name="PV"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["credit"], mode="lines", name="Credit"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index, y=plot_df["holdings_value"], mode="lines", name="Holdings"
        ),
        row=4,
        col=1,
    )

    # --- Vertical dashed lines for segment boundaries
    if segment_boundaries:
        for boundary in segment_boundaries:
            # Check if boundary is within the index range
            if 0 <= boundary < len(df.index):
                fig.add_vline(
                    x=df.index[boundary],
                    line_width=1,
                    line_dash="dash",
                    line_color="black",
                    opacity=0.4,
                )

    fig.update_layout(
        height=1000,
        title_text=f"{plot_title_prefix} Results (Episode {episode})",
        showlegend=False,
    )
    fig.show()


# -------------------------------------------------
# 6. RUN A SINGLE EPISODE
# -------------------------------------------------
def run_episode(env, agent, data, batch_size, is_eval=False, initial_credit=None, initial_holdings=0.0):
    state = env.reset(initial_credit=initial_credit, holdings=initial_holdings)
    done = False

    portfolio_values, credits, holdings_values = [], [], []

    pbar_desc = "TESTING" if is_eval else "TRAINING"
    # Adjust total for progress bar to be the number of steps in the episode
    total_steps = len(data) - env.window_size -1
    pbar = tqdm(total=total_steps, desc=pbar_desc)

    while not done:
        action = agent.act(state, is_eval)
        next_state, reward, done, info = env.step(action)

        if not is_eval:
            # Ensure next_state is not None before pushing to memory
            if next_state is not None:
                agent.memory.push(state, action, reward, next_state, done)
                agent.learn(batch_size)

        state = next_state
        
        # Break the loop if state is None (end of data)
        if state is None:
            break

        current_price = data["Original_Close"].iloc[
            env.current_step if not done else env.current_step - 1
        ]
        portfolio_values.append(info["portfolio_value"])
        credits.append(info["credit"])
        holdings_values.append(info["holdings"] * current_price)

        pbar.update(1)

    pbar.close()
    return portfolio_values, credits, holdings_values, env.trades, info["credit"], info["holdings"]


# -------------------------------------------------
# 7. VALIDATION SPLIT INTO SMALLER WINDOWS
# -------------------------------------------------
def validate_in_segments(
    full_val_data,
    agent,
    window_size,
    eval_window_minutes,
    initial_credit,
    batch_size,
    min_trade_amount_buy,
    min_trade_amount_sell,
):
    """
    Consecutively evaluates the agent on `full_val_data` split into chunks of
    length `eval_window_minutes`.
    Returns aggregated information and the concatenated history vectors that
    can be fed into the plotting routine.
    """
    n_total = len(full_val_data)
    segment_starts = list(
        range(0, n_total - window_size - 1, eval_window_minutes)
    )  # last window may be shorter
    all_portfolio, all_credit, all_holdings, all_trades = [], [], [], []
    segment_metrics = []  # list of dicts with final PV, ret, buy, sell
    
    current_credit = initial_credit
    current_holdings = 0.0

    for seg_idx, start in enumerate(segment_starts, 1):
        end = min(start + eval_window_minutes, n_total)
        # Ensure the segment has enough data
        if end - start <= window_size:
            continue
        
        segment_data = full_val_data.iloc[start:end].copy().reset_index(drop=True)

        val_env = TradingEnvironment(
            segment_data,
            initial_credit=current_credit,
            window_size=window_size,
            min_trade_amount_buy=min_trade_amount_buy,
            min_trade_amount_sell=min_trade_amount_sell,
        )
        (
            pv_hist,
            credit_hist,
            hold_hist,
            trades,
            final_credit,
            final_holdings,
        ) = run_episode(val_env, agent, segment_data, batch_size, is_eval=True, initial_credit=current_credit, initial_holdings=current_holdings)

        # Shift step indices so they refer to the full validation dataframe
        for tr in trades:
            tr["step"] += start
        all_trades.extend(trades)

        # Append histories (these already have correct chronological order)
        all_portfolio.extend(pv_hist)
        all_credit.extend(credit_hist)
        all_holdings.extend(hold_hist)

        # Update portfolio for the next segment
        current_credit = final_credit
        current_holdings = final_holdings
        
        final_pv = pv_hist[-1] if pv_hist else current_credit
        # Calculate return based on the initial value of the current segment
        initial_segment_value = credit_hist[0] if credit_hist else current_credit
        seg_return = (final_pv - initial_segment_value) / initial_segment_value * 100
        buys = len([t for t in trades if t["type"] == "buy"])
        sells = len([t for t in trades if t["type"] == "sell"])
        segment_metrics.append(
            dict(
                seg=seg_idx,
                start=full_val_data.index[start],
                end=full_val_data.index[end - 1],
                final_pv=final_pv,
                ret=seg_return,
                buys=buys,
                sells=sells,
            )
        )

    return (
        all_portfolio,
        all_credit,
        all_holdings,
        all_trades,
        segment_metrics,
        segment_starts,
    )


# -------------------------------------------------
# 8. MAIN
# -------------------------------------------------
def main():
    # ---------- HYPER-PARAMETERS ----------
    EPISODES = 50
    EPISODE_LENGTH_DAYS = 2
    BATCH_SIZE = 64
    ATTENTION_DIM = 32
    ATTENTION_HEADS = 4
    LEARNING_RATE = 0.0001
    TARGET_UPDATE = 5
    WINDOW_SIZE = 180
    MIN_TRADE_AMOUNT_BUY = 1
    MIN_TRADE_AMOUNT_SELL = 1
    INITIAL_CREDIT = 100
    # NEW: length (in minutes) of every validation evaluation segment
    VAL_EVAL_WINDOW_MINUTES = 48 * 60  # 2 days

    # ---------- DATA ----------
    full_data = get_data()
    if full_data.empty or len(full_data) < WINDOW_SIZE * 2:
        return

    # Train / Val / Test split
    train_size = int(len(full_data) * 0.7)
    val_size = int(len(full_data) * 0.15)
    train_data = full_data[:train_size]
    val_data = full_data[train_size : train_size + val_size]
    test_data = full_data[train_size + val_size :]
    
    # Reset index for validation data to ensure proper slicing
    val_data = val_data.reset_index()


    print(f"Training data size:   {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Test data size:       {len(test_data)}")

    # ---------- AGENT ----------
    dummy_env = TradingEnvironment(
        full_data,
        initial_credit=INITIAL_CREDIT,
        window_size=WINDOW_SIZE,
        min_trade_amount_buy=MIN_TRADE_AMOUNT_BUY,
        min_trade_amount_sell=MIN_TRADE_AMOUNT_SELL,
    )
    agent = DQNAgent(
        state_dim=dummy_env.n_features,
        action_dim=dummy_env.n_actions,
        attention_dim=ATTENTION_DIM,
        attention_heads=ATTENTION_HEADS,
        learning_rate=LEARNING_RATE,
    )

    os.makedirs("saved_models", exist_ok=True)

    # ---------- TRAINING LOOP ----------
    for e in range(EPISODES):
        print(f"\n=== Episode {e + 1}/{EPISODES} ===")

        # --- TRAINING PHASE
        episode_minutes = EPISODE_LENGTH_DAYS * 24 * 60
        max_start = len(train_data) - episode_minutes - 1
        if max_start <0:
            print("Training data is smaller than an episode length. Skipping training for this episode.")
            continue
        start_idx = random.randint(0, max_start)
        episode_data = train_data.iloc[start_idx : start_idx + episode_minutes].copy()
        print(
            f"Training on slice {episode_data.index[0]}  ->  {episode_data.index[-1]}"
        )
        train_env = TradingEnvironment(
            episode_data,
            initial_credit=INITIAL_CREDIT,
            window_size=WINDOW_SIZE,
            min_trade_amount_buy=MIN_TRADE_AMOUNT_BUY,
            min_trade_amount_sell=MIN_TRADE_AMOUNT_SELL,
        )
        run_episode(train_env, agent, episode_data, BATCH_SIZE, is_eval=False, initial_credit=INITIAL_CREDIT)
        agent.decay_epsilon()

        if (e + 1) % TARGET_UPDATE == 0:
            agent.update_target_network()
            print("Target network synchronised.")

        agent.save_model(f"saved_models/model_episode_{e + 1}.pth")

        # --- VALIDATION PHASE (multi-segment)
        print("\n-> Validation phase …")
        (
            pv_hist,
            credit_hist,
            hold_hist,
            trades,
            seg_metrics,
            seg_starts,
        ) = validate_in_segments(
            val_data.set_index('Datetime'), # Set index back for plotting
            agent,
            window_size=WINDOW_SIZE,
            eval_window_minutes=VAL_EVAL_WINDOW_MINUTES,
            initial_credit=INITIAL_CREDIT,
            batch_size=BATCH_SIZE,
            min_trade_amount_buy=MIN_TRADE_AMOUNT_BUY,
            min_trade_amount_sell=MIN_TRADE_AMOUNT_SELL,
        )

        # ---------- RESULTS ----------
        # csv output
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_csv_name = f"validation_trades_episode_{e + 1}.csv"
            trades_df.to_csv(trades_csv_name, index=False)
            print(f"Detailed trades saved to {trades_csv_name}")

        # console printout per segment
        print("\nSegment-by-segment results:")
        for m in seg_metrics:
            print(
                f"  Seg {m['seg']:02d}  [{m['start']} → {m['end']}]  "
                f"PV: ${m['final_pv']:.2f}  Ret: {m['ret']:.2f}%  "
                f"Buys: {m['buys']}  Sells: {m['sells']}"
            )

        # aggregate
        if seg_metrics:
            returns = [m["ret"] for m in seg_metrics]
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            total_buys = sum(m["buys"] for m in seg_metrics)
            total_sells = sum(m["sells"] for m in seg_metrics)
            print("\nAggregated validation:")
            print(f"  Mean segment return: {mean_ret:.2f}%  (σ={std_ret:.2f})")
            print(f"  Total buys:  {total_buys}")
            print(f"  Total sells: {total_sells}")

        # ---------- PLOT ----------
        if pv_hist:
            plot_results(
                val_data.set_index('Datetime'),
                episode=e + 1,
                portfolio_history=pv_hist,
                credit_history=credit_hist,
                holdings_history=hold_hist,
                trades=trades,
                plot_title_prefix="Validation",
                segment_boundaries=[s + WINDOW_SIZE for s in seg_starts[1:]],
            )

    # ------------- FINAL   TEST -------------
    print("\n=== FINAL TEST ===")
    test_env = TradingEnvironment(
        test_data,
        initial_credit=INITIAL_CREDIT,
        window_size=WINDOW_SIZE,
        min_trade_amount_buy=MIN_TRADE_AMOUNT_BUY,
        min_trade_amount_sell=MIN_TRADE_AMOUNT_SELL,
    )
    pv, cred, hold, trades, _, _ = run_episode(
        test_env, agent, test_data, BATCH_SIZE, is_eval=True, initial_credit=INITIAL_CREDIT
    )
    plot_results(
        test_data,
        episode=EPISODES,
        portfolio_history=pv,
        credit_history=cred,
        holdings_history=hold,
        trades=trades,
        plot_title_prefix="Final Test",
    )

    if pv:
        final_pv = pv[-1]
        test_return = (final_pv - INITIAL_CREDIT) / INITIAL_CREDIT * 100
        buys = len([t for t in trades if t["type"] == "buy"])
        sells = len([t for t in trades if t["type"] == "sell"])
        print(
            f"Final portfolio value: ${final_pv:,.2f}  |  Return: {test_return:.2f}%  "
            f"|  Buys: {buys}  Sells: {sells}"
        )


if __name__ == "__main__":
    main()
