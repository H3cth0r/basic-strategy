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

# --- 1. Data Loading and Preprocessing ---

def get_data():
    """
    Downloads and preprocesses the Bitcoin 1-minute interval data.
    """
    print("Downloading and preprocessing data...")
    url = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"
    column_names = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']

    try:
        df = pd.read_csv(
            url,
            skiprows=[1, 2],
            header=0,
            names=column_names,
            parse_dates=['Datetime'],
            index_col='Datetime',
            dtype={'Volume': 'int64'},
            na_values=['NA', 'N/A', ''],
            keep_default_na=True
        )
        df.index = pd.to_datetime(df.index, utc=True)
    except Exception as e:
        print(f"Error reading data: {e}")
        return pd.DataFrame()

    df.ffill(inplace=True)
    df.dropna(inplace=True)

    print("Calculating technical indicators...")
    add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    original_close = df['Close'].copy()
    for col in df.columns:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-7)

    df['Original_Close'] = original_close
    df.dropna(inplace=True)
    print("Data loaded and preprocessed successfully.")
    print(df.head())
    print(f"Data shape: {df.shape}")
    return df

# --- 2. Attention Mechanism ---

class SharedSelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim, attention_heads=1, dropout_rate=0.1):
        super(SharedSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.attention_heads = attention_heads

        if self.attention_dim % self.attention_heads != 0:
            raise ValueError(f"Attention dim ({self.attention_dim}) must be divisible by the number of heads ({self.attention_heads}).")

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

        Q = Q.view(batch_size, seq_len, self.attention_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.attention_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.attention_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(energy, dim=-1)
        attention_weights = self.dropout(attention_weights)

        weighted_values = torch.matmul(attention_weights, V)
        weighted_values = weighted_values.permute(0, 2, 1, 3).contiguous()
        weighted_values = weighted_values.view(batch_size, seq_len, self.attention_dim)

        output = self.output_proj(weighted_values)
        return output.mean(dim=1)


# --- 3. Deep Q-Network (DQN) Agent ---

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
        super(QNetwork, self).__init__()
        self.attention = SharedSelfAttention(state_dim - 2, attention_dim, attention_heads)
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
    def __init__(self, state_dim, action_dim, attention_dim, attention_heads, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = QNetwork(state_dim, action_dim, attention_dim, attention_heads).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, attention_dim, attention_heads).to(self.device)
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
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# --- 4. Trading Environment ---

class TradingEnvironment:
    def __init__(self, data, initial_credit=10000, fee=0.001, window_size=180):
        self.data = data
        self.normalized_data = data.drop(columns=['Original_Close'])
        self.initial_credit = initial_credit
        self.fee = fee
        self.window_size = window_size
        self.n_features = self.normalized_data.shape[1] + 2
        self.action_space = [-.5, -.25, 0, .25, .5]
        self.n_actions = len(self.action_space)

    def reset(self, episode_start_index=0):
        self.credit = self.initial_credit
        self.holdings = 0
        self.average_buy_price = 0
        self.current_step = episode_start_index + self.window_size
        self.trades = []
        return self._get_state()

    def _get_state(self):
        start = self.current_step - self.window_size
        end = self.current_step
        market_data = self.normalized_data.iloc[start:end].values
        current_price = self.data['Original_Close'].iloc[self.current_step]
        portfolio_value = self.credit + self.holdings * current_price
        holdings_ratio = (self.holdings * current_price) / portfolio_value if portfolio_value > 0 else 0
        credit_ratio = self.credit / portfolio_value if portfolio_value > 0 else 0
        portfolio_state = np.array([[holdings_ratio, credit_ratio]] * self.window_size)
        return np.concatenate([market_data, portfolio_state], axis=1)

    def step(self, action_idx):
        action = self.action_space[action_idx]
        current_price = self.data['Original_Close'].iloc[self.current_step]
        reward = 0
        done = False

        if action < 0:
            sell_fraction = -action
            sell_amount = self.holdings * sell_fraction
            if sell_amount > 0:
                self.credit += sell_amount * current_price * (1 - self.fee)
                self.holdings -= sell_amount
                self.trades.append({'step': self.current_step, 'type': 'sell', 'price': current_price, 'amount': sell_amount})
                realized_pnl = (current_price - self.average_buy_price) * sell_amount
                reward += realized_pnl
                if self.holdings < 1e-6: self.average_buy_price = 0
        elif action > 0:
            buy_fraction = action
            investment = self.credit * buy_fraction
            if investment > 0:
                buy_amount = (investment / current_price) * (1 - self.fee)
                total_cost = (self.average_buy_price * self.holdings) + investment
                self.holdings += buy_amount
                self.credit -= investment
                self.average_buy_price = total_cost / self.holdings if self.holdings > 0 else 0
                self.trades.append({'step': self.current_step, 'type': 'buy', 'price': current_price, 'amount': buy_amount})

        if self.holdings > 0:
            unrealized_pnl = (current_price - self.average_buy_price) * self.holdings
            reward += unrealized_pnl * 0.01

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        
        next_state = self._get_state()
        next_price = self.data['Original_Close'].iloc[self.current_step] if not done else current_price
        portfolio_value = self.credit + self.holdings * next_price

        if done:
            self.credit += self.holdings * current_price * (1 - self.fee)
            self.holdings = 0
            portfolio_value = self.credit
            
        return next_state, reward, done, {'portfolio_value': portfolio_value, 'credit': self.credit, 'holdings': self.holdings, 'trades': self.trades}

# --- 5. Plotting ---

def plot_results(df, episode, portfolio_history, credit_history, holdings_history, trades, plot_title_prefix=""):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                          subplot_titles=(f'{plot_title_prefix} Price and Trades', 'Portfolio Value', 'Credit', 'Holdings Value'))

    plot_df = df.iloc[len(df) - len(portfolio_history):].copy()
    plot_df['portfolio_value'] = portfolio_history
    plot_df['credit'] = credit_history
    plot_df['holdings_value'] = holdings_history

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Original_Close'], mode='lines', name='Price', line=dict(color='lightgrey')), row=1, col=1)

    buy_trades = [trade for trade in trades if trade['type'] == 'buy']
    sell_trades = [trade for trade in trades if trade['type'] == 'sell']

    if buy_trades:
        fig.add_trace(go.Scatter(x=[df.index[t['step']] for t in buy_trades], y=[t['price'] for t in buy_trades],
                                   mode='markers', marker=dict(color='green', symbol='triangle-up', size=8), name='Buy'), row=1, col=1)
    if sell_trades:
        fig.add_trace(go.Scatter(x=[df.index[t['step']] for t in sell_trades], y=[t['price'] for t in sell_trades],
                                   mode='markers', marker=dict(color='red', symbol='triangle-down', size=8), name='Sell'), row=1, col=1)

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['portfolio_value'], mode='lines', name='Portfolio Value'), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['credit'], mode='lines', name='Credit'), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['holdings_value'], mode='lines', name='Holdings Value'), row=4, col=1)

    fig.update_layout(height=1000, title_text=f"{plot_title_prefix} Results (Episode {episode})", showlegend=False)
    fig.show()

# --- 6. Core Episode Runner ---

def run_episode(env, agent, data, batch_size, is_eval=False):
    state = env.reset()
    done = False
    
    portfolio_values, credits, holdings_values = [], [], []
    
    pbar_desc = "TESTING" if is_eval else "TRAINING"
    pbar = tqdm(total=len(data) - env.window_size - 1, desc=pbar_desc)
    
    while not done:
        action = agent.act(state, is_eval)
        next_state, reward, done, info = env.step(action)
        
        if not is_eval:
            agent.memory.push(state, action, reward, next_state, done)
            agent.learn(batch_size)
        
        state = next_state
        
        current_price = data['Original_Close'].iloc[env.current_step if not done else env.current_step - 1]
        portfolio_values.append(info['portfolio_value'])
        credits.append(info['credit'])
        holdings_values.append(info['holdings'] * current_price)
        
        pbar.update(1)

    pbar.close()
    return portfolio_values, credits, holdings_values, env.trades

# --- 7. Main Execution Block ---
def main():
    # --- Hyperparameters ---
    EPISODES = 50
    EPISODE_LENGTH_DAYS = 4
    BATCH_SIZE = 64
    ATTENTION_DIM = 32
    ATTENTION_HEADS = 4
    LEARNING_RATE = 0.0001
    TARGET_UPDATE = 5
    WINDOW_SIZE = 180

    # --- Setup ---
    full_data = get_data()
    if full_data.empty or len(full_data) < (WINDOW_SIZE * 2): return

    # --- Data Splitting (Train: 70%, Validation: 15%, Test: 15%) ---
    train_size = int(len(full_data) * 0.7)
    val_size = int(len(full_data) * 0.15)
    train_data = full_data[:train_size]
    val_data = full_data[train_size : train_size + val_size]
    test_data = full_data[train_size + val_size:] # The rest is for testing
    
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Test data size: {len(test_data)}")

    env_prototype = TradingEnvironment(full_data, window_size=WINDOW_SIZE)
    n_features = env_prototype.n_features
    n_actions = env_prototype.n_actions
    
    agent = DQNAgent(state_dim=n_features, action_dim=n_actions,
                     attention_dim=ATTENTION_DIM, attention_heads=ATTENTION_HEADS,
                     learning_rate=LEARNING_RATE)

    # --- Training Loop ---
    for e in range(EPISODES):
        print(f"\n--- Episode {e+1}/{EPISODES} ---")

        # --- Training Phase ---
        episode_length_minutes = EPISODE_LENGTH_DAYS * 24 * 60
        max_start_index = len(train_data) - episode_length_minutes - 1
        start_index = random.randint(0, max_start_index)
        episode_data = train_data.iloc[start_index : start_index + episode_length_minutes].copy()
        
        print(f"Starting Training Phase on data from {episode_data.index[0]} to {episode_data.index[-1]}...")
        train_env = TradingEnvironment(episode_data, window_size=WINDOW_SIZE)
        run_episode(train_env, agent, episode_data, BATCH_SIZE)
        agent.decay_epsilon()
        
        if (e + 1) % TARGET_UPDATE == 0:
            agent.update_target_network()
            print("Target network updated.")
            
        # --- Validation Phase ---
        print("\nStarting Validation Phase...")
        val_env = TradingEnvironment(val_data, window_size=WINDOW_SIZE)
        portfolio_values, credits, holdings_values, trades = run_episode(val_env, agent, val_data, BATCH_SIZE, is_eval=True)

        plot_results(val_data, e + 1, portfolio_values, credits, holdings_values, trades, plot_title_prefix="Validation")

        final_portfolio_value = portfolio_values[-1]
        returns = (final_portfolio_value - val_env.initial_credit) / val_env.initial_credit * 100
        num_buys = len([t for t in trades if t['type'] == 'buy'])
        num_sells = len([t for t in trades if t['type'] == 'sell'])

        print(f"\n--- Validation Results for Episode {e+1} ---")
        print(f"  Final Portfolio Value: ${final_portfolio_value:,.2f}")
        print(f"  Total Return: {returns:.2f}%")
        print(f"  Number of Buys: {num_buys}")
        print(f"  Number of Sells: {num_sells}")

        days_in_val = len(val_data) / (24 * 60)
        if days_in_val > 0 and returns > -100:
            daily_return = (1 + returns / 100)**(1 / days_in_val) - 1
            weekly_return = (1 + daily_return)**7 - 1
            monthly_return = (1 + daily_return)**30 - 1
            yearly_return = (1 + daily_return)**365 - 1
            print(f"  Projected Weekly Return: {weekly_return:.2%}")
            print(f"  Projected Monthly Return: {monthly_return:.2%}")
            print(f"  Projected Yearly Return: {yearly_return:.2%}")

    # --- Final Testing Phase (after all training is done) ---
    print("\n\n--- All training complete. Starting Final Test ---")
    test_env = TradingEnvironment(test_data, window_size=WINDOW_SIZE)
    portfolio_values, credits, holdings_values, trades = run_episode(test_env, agent, test_data, BATCH_SIZE, is_eval=True)
    
    plot_results(test_data, EPISODES, portfolio_values, credits, holdings_values, trades, plot_title_prefix="Final Test")

    final_portfolio_value = portfolio_values[-1]
    returns = (final_portfolio_value - test_env.initial_credit) / test_env.initial_credit * 100
    num_buys = len([t for t in trades if t['type'] == 'buy'])
    num_sells = len([t for t in trades if t['type'] == 'sell'])

    print(f"\n--- FINAL TEST RESULTS ---")
    print(f"  Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"  Total Return: {returns:.2f}%")
    print(f"  Number of Buys: {num_buys}")
    print(f"  Number of Sells: {num_sells}")

    days_in_test = len(test_data) / (24 * 60)
    if days_in_test > 0 and returns > -100:
        daily_return = (1 + returns / 100)**(1 / days_in_test) - 1
        weekly_return = (1 + daily_return)**7 - 1
        monthly_return = (1 + daily_return)**30 - 1
        yearly_return = (1 + daily_return)**365 - 1
        print(f"  Projected Weekly Return on Test Data: {weekly_return:.2%}")
        print(f"  Projected Monthly Return on Test Data: {monthly_return:.2%}")
        print(f"  Projected Yearly Return on Test Data: {yearly_return:.2%}")


if __name__ == "__main__":
    main()
