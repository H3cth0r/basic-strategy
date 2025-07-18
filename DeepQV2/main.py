import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import math
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm # Import tqdm
import warnings

# Ignore all warnings from the 'ta' library
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration Constants ---
# Data and Environment
URL = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"
TRAIN_PCT = 0.7
VALIDATION_PCT = 0.15
# Note: TEST_PCT is implicitly 1 - TRAIN_PCT - VALIDATION_PCT

# Trading Simulation
INITIAL_CAPITAL = 10000.0
MAKER_FEE = 0.001  # Fee for making a trade (e.g., 0.1%)
TAKER_FEE = 0.001

# RL Agent
STATE_LOOKBACK = {
    "attention_window_minutes": 180, # Look back 3 hours
    "attention_sample_interval_minutes": 5, # Sample every 5 mins
    "long_term_lags_hours": [6, 12, 24] # Lags for macro context
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Hyperparameters
ATTENTION_DIM = 64
ATTENTION_HEADS = 4
HIDDEN_DIM = 256
GAMMA = 0.99  # Discount factor
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500 # Higher value means slower decay
LEARNING_RATE = 0.0001
MEMORY_SIZE = 50000
BATCH_SIZE = 128
TARGET_UPDATE_FREQ = 10 # Episodes

# Training
NUM_EPISODES = 50 # Increase for serious training
EPISODE_MAX_STEPS = 5000 # Max steps per episode to prevent infinite loops

# --- 1. Data Loading and Preprocessing ---

def load_and_prepare_data(url):
    """Loads, cleans, and adds technical indicators to the data."""
    print("Loading data...")
    column_names = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
    df = pd.read_csv(
        url,
        skiprows=[1, 2],
        header=0,
        names=column_names,
        parse_dates=['Datetime'],
        index_col='Datetime',
        dtype={'Volume': 'int64'},
        na_values=['NA', 'N/A', ''],
    )
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    df.dropna(inplace=True) # Drop rows with any missing values
    
    # Fill any remaining NaNs from indicators with the previous value
    df.ffill(inplace=True)
    
    print("Calculating technical indicators...")
    # Add a few key indicators from 'ta' library
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    
    # Remove initial rows with NaNs created by indicators
    df.dropna(inplace=True)

    print(f"Data loaded and preprocessed. Shape: {df.shape}")
    print(df.head())
    return df

# --- 2. RL Environment ---

class TradingEnv:
    def __init__(self, df, initial_capital=INITIAL_CAPITAL, episode_max_steps=EPISODE_MAX_STEPS):
        self.df = df
        self.initial_capital = initial_capital
        self.episode_max_steps = episode_max_steps
        
        self.feature_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'rsi', 'macd_diff']
        self.scaler = MinMaxScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[self.feature_cols].values)

        self.action_space_size = 3
        self.min_required_data_points = (
            STATE_LOOKBACK['attention_window_minutes'] + 
            max(STATE_LOOKBACK['long_term_lags_hours']) * 60
        )

    def reset(self, start_index=None):
        """Resets the environment for a new episode."""
        if start_index is None:
            max_start_index = len(self.df) - self.episode_max_steps - 1
            start_index = random.randint(self.min_required_data_points, max_start_index)
        
        self.start_step = start_index
        self.current_step = self.start_step
        
        max_possible_end_step = len(self.df) - 2 
        self.end_step = min(self.current_step + self.episode_max_steps, max_possible_end_step)

        self.cash = self.initial_capital
        self.holdings_value = 0.0
        self.btc_held = 0.0
        self.total_trades = 0
        self.buy_trades = 0
        self.sell_trades = 0
        
        self.portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        self.hold_streak = 0
        
        self.history = []

        return self._get_state()

    def _get_state(self):
        """Constructs the state dictionary for the agent."""
        norm_cash = self.cash / self.portfolio_value if self.portfolio_value > 0 else 0
        norm_holdings = self.holdings_value / self.portfolio_value if self.portfolio_value > 0 else 0
        portfolio_state = torch.tensor([norm_cash, norm_holdings], dtype=torch.float32, device=DEVICE)

        lags = []
        current_price = self.df['Close'].iloc[self.current_step]
        for hour in STATE_LOOKBACK['long_term_lags_hours']:
            lag_step = self.current_step - hour * 60
            lag_price = self.df['Close'].iloc[lag_step]
            lags.append((current_price - lag_price) / current_price)
        market_context = torch.tensor(lags, dtype=torch.float32, device=DEVICE)

        attention_window = STATE_LOOKBACK['attention_window_minutes']
        sample_interval = STATE_LOOKBACK['attention_sample_interval_minutes']
        start_idx = self.current_step - attention_window
        end_idx = self.current_step
        
        attention_seq_data = self.scaled_features[start_idx:end_idx:sample_interval]
        attention_input = torch.tensor(attention_seq_data, dtype=torch.float32, device=DEVICE)
        
        return {
            'portfolio': portfolio_state.unsqueeze(0),
            'market_context': market_context.unsqueeze(0),
            'attention_input': attention_input.unsqueeze(0)
        }

    def step(self, action):
        """Executes one time step in the environment."""
        current_price = self.df['Close'].iloc[self.current_step]
        
        # FIX: Default to Hold action (0) and only change if a trade is executed.
        executed_action = 0 
        
        if action == 1: # Buy Attempt
            if self.cash > 1: # Condition to execute
                btc_to_buy = (self.cash * (1 - TAKER_FEE)) / current_price
                self.btc_held += btc_to_buy
                self.cash = 0
                self.total_trades += 1
                self.buy_trades += 1
                self.hold_streak = 0
                executed_action = 1 # Mark that a buy was executed
        elif action == 2: # Sell Attempt
            if self.btc_held > 1e-6: # Condition to execute
                self.cash += self.btc_held * current_price * (1 - MAKER_FEE)
                self.btc_held = 0
                self.total_trades += 1
                self.sell_trades += 1
                self.hold_streak = 0
                executed_action = 2 # Mark that a sell was executed
        else: # Hold
            self.hold_streak += 1

        self.current_step += 1
        next_price = self.df['Close'].iloc[self.current_step]
        self.holdings_value = self.btc_held * next_price
        
        previous_portfolio_value = self.portfolio_value
        self.portfolio_value = self.cash + self.holdings_value
        
        reward = self._calculate_reward(previous_portfolio_value)
        done = (self.portfolio_value <= 0.2 * self.initial_capital) or (self.current_step >= self.end_step)
        
        # FIX: Log the executed_action instead of the original attempted action.
        self.history.append({
            'step': self.current_step,
            'price': current_price,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'holdings': self.holdings_value,
            'action': executed_action 
        })
        
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done, {}

    def _calculate_reward(self, previous_portfolio_value):
        reward = (self.portfolio_value - previous_portfolio_value) / self.initial_capital
        if self.portfolio_value > self.max_portfolio_value:
            reward += 0.05
            self.max_portfolio_value = self.portfolio_value
        if self.hold_streak > 10:
             reward -= self.hold_streak * 0.00001
        return reward

# --- 3. The DQN Model with Self-Attention ---

class SharedSelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim, attention_heads=4):
        super().__init__()
        self.head_dim = attention_dim // attention_heads
        self.attention_heads = attention_heads
        
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.fc_out = nn.Linear(attention_dim, attention_dim)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        q = self.query(x).view(batch_size, seq_length, self.attention_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).view(batch_size, seq_length, self.attention_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).view(batch_size, seq_length, self.attention_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        out = self.fc_out(out)
        context_vector = out.mean(dim=1)
        return context_vector

class AttentionDQN(nn.Module):
    def __init__(self, attention_input_dim, portfolio_dim, context_dim, num_actions):
        super().__init__()
        self.attention = SharedSelfAttention(attention_input_dim, ATTENTION_DIM, ATTENTION_HEADS)
        combined_dim = ATTENTION_DIM + portfolio_dim + context_dim
        self.fc1 = nn.Linear(combined_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, num_actions)
        
    def forward(self, state):
        attention_context = self.attention(state['attention_input'])
        combined = torch.cat([state['portfolio'], state['market_context'], attention_context], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- 4. The RL Agent ---

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Experience(*args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

class Agent:
    def __init__(self, attention_input_dim, portfolio_dim, context_dim, num_actions):
        self.num_actions = num_actions
        self.steps_done = 0
        self.policy_net = AttentionDQN(attention_input_dim, portfolio_dim, context_dim, num_actions).to(DEVICE)
        self.target_net = AttentionDQN(attention_input_dim, portfolio_dim, context_dim, num_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)

    def select_action(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=DEVICE, dtype=torch.long)

    def learn(self):
        if len(self.memory) < BATCH_SIZE: return None
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
        def batch_states(states):
            valid_states = [s for s in states if s is not None]
            if not valid_states: return None
            return {
                'portfolio': torch.cat([s['portfolio'] for s in valid_states]),
                'market_context': torch.cat([s['market_context'] for s in valid_states]),
                'attention_input': torch.cat([s['attention_input'] for s in valid_states]),
            }
        state_batch = batch_states(list(batch.state))
        next_state_batch = batch_states(list(batch.next_state))
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
        if next_state_batch is not None:
             with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(next_state_batch).max(1)[0]
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# --- 5. Plotting and Reporting ---

def plot_validation_results(history, episode):
    """Generates a multi-panel Plotly chart of the validation run."""
    if not history:
        print("No history to plot for validation.")
        return
    df_history = pd.DataFrame(history)
    df_history.set_index('step', inplace=True)
    
    # This filtering now works correctly because the 'action' in history is only 1 or 2
    # if a trade was actually executed.
    buys = df_history[df_history['action'] == 1]
    sells = df_history[df_history['action'] == 2]

    # Create 4 subplots, stacked vertically, sharing the x-axis
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            'Market Price & Executed Trades', # Title updated for clarity
            'Portfolio Value ($)',
            'Cash Account ($)',
            'Holdings Value ($)'
        ),
        row_heights=[0.5, 0.15, 0.15, 0.15] # Give more space to the main chart
    )

    # Plot 1: Price and Trades
    fig.add_trace(go.Scatter(x=df_history.index, y=df_history['price'], name='BTC Price',
                             line=dict(color='lightgrey')), row=1, col=1)
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys.index, y=buys['price'], name='Buy', mode='markers',
                                 marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells.index, y=sells['price'], name='Sell', mode='markers',
                                 marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)

    # Plot 2: Portfolio Value
    fig.add_trace(go.Scatter(x=df_history.index, y=df_history['portfolio_value'], name='Portfolio Value',
                             line=dict(color='blue', width=2)), row=2, col=1)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)

    # Plot 3: Cash
    fig.add_trace(go.Scatter(x=df_history.index, y=df_history['cash'], name='Cash',
                             line=dict(color='orange')), row=3, col=1)
    fig.update_yaxes(title_text="Value ($)", row=3, col=1)

    # Plot 4: Holdings
    fig.add_trace(go.Scatter(x=df_history.index, y=df_history['holdings'], name='Holdings',
                             line=dict(color='purple')), row=4, col=1)
    fig.update_yaxes(title_text="Value ($)", row=4, col=1)
    
    fig.update_layout(
        title_text=f'Agent Validation Performance - Episode {episode}',
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # The last x-axis title
    fig.update_xaxes(title_text="Time Step", row=4, col=1)
    fig.show()

def print_performance_stats(history, initial_capital, num_buy, num_sell):
    if not history:
        print("\n--- Validation Performance: No trades made or no history recorded. ---\n")
        return
    final_portfolio_value = history[-1]['portfolio_value']
    total_return_pct = ((final_portfolio_value - initial_capital) / initial_capital) * 100
    num_steps = len(history)
    total_days = num_steps / (60 * 24)
    if total_days > 0.001:
        annualized_return = ((1 + total_return_pct / 100) ** (365 / total_days) - 1) * 100
        monthly_return = ((1 + total_return_pct / 100) ** (30 / total_days) - 1) * 100
        weekly_return = ((1 + total_return_pct / 100) ** (7 / total_days) - 1) * 100
    else:
        annualized_return = monthly_return = weekly_return = 0
    print("\n--- Validation Performance ---")
    print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Number of Buys: {num_buy}")
    print(f"Number of Sells: {num_sell}")
    print("\nProjected Returns (based on this episode):")
    print(f"  - Weekly: {weekly_return:.2f}%")
    print(f"  - Monthly: {monthly_return:.2f}%")
    print(f"  - Annualized: {annualized_return:.2f}%")
    print("----------------------------\n")

# --- 6. Main Training Loop ---

if __name__ == '__main__':
    full_df = load_and_prepare_data(URL)
    n = len(full_df)
    train_end = int(n * TRAIN_PCT)
    val_end = int(n * (TRAIN_PCT + VALIDATION_PCT))
    train_df = full_df[:train_end].reset_index(drop=True)
    val_df = full_df[train_end:val_end].reset_index(drop=True)
    test_df = full_df[val_end:].reset_index(drop=True)
    print(f"Data split: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")

    train_env = TradingEnv(train_df, initial_capital=INITIAL_CAPITAL, episode_max_steps=EPISODE_MAX_STEPS)
    val_env = TradingEnv(val_df, initial_capital=INITIAL_CAPITAL, episode_max_steps=len(val_df) - 1)

    sample_state = train_env.reset()
    attention_input_dim = sample_state['attention_input'].shape[-1]
    portfolio_dim = sample_state['portfolio'].shape[-1]
    context_dim = sample_state['market_context'].shape[-1]
    num_actions = train_env.action_space_size
    agent = Agent(attention_input_dim, portfolio_dim, context_dim, num_actions)
    print(f"Starting training on {DEVICE}...")
    
    for episode in range(1, NUM_EPISODES + 1):
        state = train_env.reset()
        episode_reward = 0
        episode_loss = []

        # Use tqdm for the episode progress bar
        pbar = tqdm(range(train_env.episode_max_steps), desc=f"Episode {episode}/{NUM_EPISODES}")
        for t in pbar:
            action = agent.select_action(state)
            next_state, reward, done, _ = train_env.step(action.item())
            reward_tensor = torch.tensor([reward], device=DEVICE, dtype=torch.float32)
            agent.memory.push(state, action, reward_tensor, next_state, done)
            state = next_state
            episode_reward += reward
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)
            
            # Update tqdm progress bar description
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            pbar.set_description(f"Episode {episode}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Loss: {avg_loss:.4f}")

            if done:
                break
        pbar.close() # Close the progress bar for the episode
        
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_net()
            print(f"*** Target network updated at episode {episode} ***")

        # --- Validation Step ---
        print("Running validation...")
        state = val_env.reset(start_index=val_env.min_required_data_points)
        done = False
        while not done:
            with torch.no_grad():
                q_values = agent.policy_net(state)
                action = q_values.max(1)[1].view(1, 1)
            state, _, done, _ = val_env.step(action.item())
        
        print_performance_stats(val_env.history, val_env.initial_capital, val_env.buy_trades, val_env.sell_trades)
        plot_validation_results(val_env.history, episode)

    print("Training finished.")
