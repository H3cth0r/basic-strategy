import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import ta
import time
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Device Configuration ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

# #################################################################################
# ### CUSTOM TRADING ENVIRONMENT (WITH STRATEGIC REWARDS)
# #################################################################################

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, lookback_window=50, is_training=True, episode_duration_minutes=4320):
        super(TradingEnv, self).__init__()
        self.full_df = df.dropna().reset_index(drop=True)
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.is_training = is_training
        self.episode_duration = episode_duration_minutes
        self.df = pd.DataFrame()

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.full_df.columns) - 1 + 3,),
            dtype=np.float32
        )

    def _get_observation(self):
        obs_df = self.df.drop(columns=['Original_Close'])
        market_obs = obs_df.loc[self.current_step].values.astype(np.float32)

        portfolio_state = np.array([
            self.balance / self.portfolio_value,
            (self.crypto_held * self.df.loc[self.current_step, 'Original_Close']) / self.portfolio_value,
            self.portfolio_value / self.initial_balance
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

        self.balance = self.initial_balance
        self.crypto_held = 0
        self.portfolio_value = self.initial_balance
        self.current_step = self.lookback_window
        
        # --- NEW: Trade Counters ---
        self.buy_trades = 0
        self.sell_trades = 0
        
        return self._get_observation(), {}

    def step(self, action):
        action_val = action[0] if isinstance(action, np.ndarray) else action
        current_price = self.df.loc[self.current_step, 'Original_Close']
        prev_portfolio_value = self.portfolio_value
        transaction_fee = 0.001
        trade_executed = False

        # --- NEW: Penalty for taking action ---
        trade_penalty = 0.0001 # A small cost for making a trade

        if action_val > 0.01: # Buy threshold
            trade_amount_usd = self.balance * action_val
            if trade_amount_usd > 1: # Minimum trade size
                fee = trade_amount_usd * transaction_fee
                self.balance -= (trade_amount_usd + fee)
                self.crypto_held += trade_amount_usd / current_price
                self.buy_trades += 1
                trade_executed = True
        elif action_val < -0.01: # Sell threshold
            trade_amount_crypto = self.crypto_held * abs(action_val)
            if trade_amount_crypto > 1e-5: # Minimum trade size
                trade_amount_usd = trade_amount_crypto * current_price
                fee = trade_amount_usd * transaction_fee
                self.balance += trade_amount_usd - fee
                self.crypto_held -= trade_amount_crypto
                self.sell_trades += 1
                trade_executed = True
        
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1

        next_price = self.df.loc[self.current_step, 'Original_Close']
        self.portfolio_value = self.balance + (self.crypto_held * next_price)

        # --- REVISED REWARD LOGIC ---
        # 1. Base reward on portfolio change
        reward = np.log(self.portfolio_value / prev_portfolio_value) if prev_portfolio_value > 0 else 0
        
        # 2. Add penalty for trading to encourage fewer, better trades
        if trade_executed:
            reward -= trade_penalty

        # 3. If episode is over, give a large bonus for overall profit
        if terminated:
            if self.portfolio_value > self.initial_balance:
                reward += 1.0 # Large bonus for being profitable
            else:
                reward -= 1.0 # Large penalty for being unprofitable
            
        if self.portfolio_value < self.initial_balance * 0.5:
            terminated = True
            reward = -2 # Very large penalty for huge loss
        
        obs = self._get_observation()
        return obs, reward, terminated, False, {}

def load_and_prepare_data(url):
    print("Loading and preparing data...")
    df = pd.read_csv(url, skiprows=3, header=None)
    df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
    df.set_index(pd.to_datetime(df['Datetime']), inplace=True)
    df.drop('Datetime', axis=1, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df['Original_Close'] = df['Close']
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(close=df['Close']).macd_diff()
    df['Close'] = (df['Close'] - df['Close'].mean()) / df['Close'].std()
    df['High'] = (df['High'] - df['High'].mean()) / df['High'].std()
    df['Low'] = (df['Low'] - df['Low'].mean()) / df['Low'].std()
    df['Open'] = (df['Open'] - df['Open'].mean()) / df['Open'].std()
    df['Volume'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
    df['RSI'] = (df['RSI'] - 50) / 50
    df['MACD'] = (df['MACD'] - df['MACD'].mean()) / df['MACD'].std()
    df.dropna(inplace=True)
    print("Data preparation complete.")
    return df.reset_index(drop=True)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1, self.layer_2, self.layer_3 = nn.Linear(state_dim, 400), nn.Linear(400, 300), nn.Linear(300, action_dim)
        self.max_action = max_action
    def forward(self, x):
        return self.max_action * torch.tanh(self.layer_3(F.relu(self.layer_2(F.relu(self.layer_1(x))))))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1, self.layer_2, self.layer_3 = nn.Linear(state_dim + action_dim, 400), nn.Linear(400, 300), nn.Linear(300, 1)
    def forward(self, x, u):
        return self.layer_3(F.relu(self.layer_2(F.relu(self.layer_1(torch.cat([x, u], 1))))))

class ReplayBuffer:
    def __init__(self, max_size=1e6): self.storage, self.max_size, self.ptr = [], int(max_size), 0
    def add(self, data):
        if len(self.storage) == self.max_size: self.storage[self.ptr] = data
        else: self.storage.append(data)
        self.ptr = (self.ptr + 1) % self.max_size
    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = zip(*[self.storage[i] for i in ind])
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

class OUNoise:
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.1, min_sigma=0.05, decay_period=100000):
        self.mu, self.theta, self.max_sigma, self.min_sigma, self.decay_period = mu, theta, max_sigma, min_sigma, decay_period
        self.action_dim, self.low, self.high = action_space.shape[0], action_space.low, action_space.high
        self.reset()
    def reset(self): self.state, self.sigma = np.ones(self.action_dim) * self.mu, self.max_sigma
    def evolve_state(self):
        self.state += self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        return self.state
    def get_action(self, action, t=0):
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + self.evolve_state(), self.low, self.high)

class DDPG:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device); self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device); self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer()
    def select_action(self, state):
        return self.actor(torch.FloatTensor(state.reshape(1, -1)).to(self.device)).cpu().data.numpy().flatten()
    def update(self, batch_size, gamma=0.99, tau=0.005):
        x, y, u, r, d = self.replay_buffer.sample(batch_size)
        state, action, next_state, done, reward = torch.FloatTensor(x).to(self.device), torch.FloatTensor(u).to(self.device), torch.FloatTensor(y).to(self.device), torch.FloatTensor(1-d).to(self.device), torch.FloatTensor(r).to(self.device)
        with torch.no_grad(): target_Q = reward + (done * gamma * self.critic_target(next_state, self.actor_target(next_state)))
        current_Q = self.critic(state, action); critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
        for target, param in zip(self.critic_target.parameters(), self.critic.parameters()): target.data.copy_(tau * param.data + (1.0 - tau) * target.data)
        for target, param in zip(self.actor_target.parameters(), self.actor.parameters()): target.data.copy_(tau * param.data + (1.0 - tau) * target.data)

def plot_validation_results(history, episode_number):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=('Price and Executed Trades', 'Agent Actions', 'Portfolio Value ($)', 'Portfolio Composition ($)'))
    fig.add_trace(go.Scatter(x=history['steps'], y=history['price'], mode='lines', name='BTC Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=history['buy_steps'], y=history['buy_prices'], mode='markers', name='Buy', marker=dict(color='green', symbol='triangle-up', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=history['sell_steps'], y=history['sell_prices'], mode='markers', name='Sell', marker=dict(color='red', symbol='triangle-down', size=10)), row=1, col=1)
    action_colors = ['green' if a > 0 else 'red' for a in history['actions']]; fig.add_trace(go.Bar(x=history['steps'], y=history['actions'], name='Action Strength', marker_color=action_colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=history['steps'], y=history['portfolio_value'], mode='lines', name='Total Value', line=dict(color='purple')), row=3, col=1)
    fig.add_trace(go.Scatter(x=history['steps'], y=history['cash'], mode='lines', name='Cash (Credit)', stackgroup='one', line=dict(color='grey')), row=4, col=1)
    fig.add_trace(go.Scatter(x=history['steps'], y=history['crypto_value'], mode='lines', name='Crypto Value', stackgroup='one', line=dict(color='orange')), row=4, col=1)
    fig.update_layout(title_text=f"Validation Performance - After Episode {episode_number}", height=1200, showlegend=False)
    fig.show()

def evaluate_agent(agent, env, initial_balance):
    state, _ = env.reset()
    done = False
    history = {'steps': [], 'actions': [], 'portfolio_value': [], 'cash': [], 'crypto_value': [], 'price': [], 'buy_steps': [], 'buy_prices': [], 'sell_steps': [], 'sell_prices': []}
    while not done:
        action = agent.select_action(state)
        prev_balance, prev_crypto_held = env.balance, env.crypto_held
        
        next_state, _, terminated, _, _ = env.step(action)
        done = terminated
        
        # --- LOGGING FOR PLOT ---
        current_price = env.df.loc[env.current_step -1, 'Original_Close']
        history['steps'].append(env.current_step - 1)
        history['actions'].append(action[0])
        history['price'].append(current_price)
        history['portfolio_value'].append(env.portfolio_value)
        history['cash'].append(env.balance)
        history['crypto_value'].append(env.crypto_held * current_price)
        
        # Only log trades that were executed
        if env.balance < prev_balance: # A buy happened
            history['buy_steps'].append(env.current_step - 1)
            history['buy_prices'].append(current_price)
        elif env.crypto_held < prev_crypto_held: # A sell happened
            history['sell_steps'].append(env.current_step - 1)
            history['sell_prices'].append(current_price)
            
        state = next_state

    return env.portfolio_value, (env.portfolio_value / initial_balance - 1) * 100, history, (env.buy_trades, env.sell_trades)

if __name__ == "__main__":
    data_url = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"
    total_training_episodes, batch_size, initial_balance, validation_freq = 200, 128, 10000, 10
    EPISODE_DURATION_DAYS = 3; episode_duration_minutes = EPISODE_DURATION_DAYS * 24 * 60

    data_df = load_and_prepare_data(data_url)
    train_df, val_df = data_df.iloc[:int(len(data_df) * 0.8)], data_df.iloc[int(len(data_df) * 0.8):]
    print(f"Training data: {len(train_df)} points\nValidation data: {len(val_df)} points")

    train_env = TradingEnv(df=train_df, initial_balance=initial_balance, is_training=True, episode_duration_minutes=episode_duration_minutes)
    val_env = TradingEnv(df=val_df, initial_balance=initial_balance, is_training=False)

    state_dim, action_dim, max_action = train_env.observation_space.shape[0], train_env.action_space.shape[0], float(train_env.action_space.high[0])
    agent = DDPG(state_dim, action_dim, max_action, device)
    noise = OUNoise(train_env.action_space)

    for i in range(total_training_episodes):
        state, _ = train_env.reset(); noise.reset()
        episode_reward, terminated = 0, False
        num_steps = len(train_env.df) - train_env.lookback_window - 1
        
        with tqdm(total=num_steps, desc=f"Training Ep {i+1}/{total_training_episodes}") as pbar:
            for t in range(num_steps):
                action = noise.get_action(agent.select_action(state), t)
                next_state, reward, terminated, _, _ = train_env.step(action)
                agent.replay_buffer.add((state, next_state, action, reward, float(terminated)))
                if len(agent.replay_buffer.storage) > batch_size * 8: agent.update(batch_size)
                state, episode_reward = next_state, episode_reward + reward
                pbar.set_postfix(portfolio=f"${train_env.portfolio_value:,.2f}"); pbar.update(1)
                if terminated: break
        
        portfolio_perf = (train_env.portfolio_value / initial_balance - 1) * 100
        print(f"Ep {i+1} Finished | Reward: {episode_reward:.4f} | Portfolio: ${train_env.portfolio_value:,.2f} ({portfolio_perf:+.2f}%) | Trades (B/S): {train_env.buy_trades}/{train_env.sell_trades}")

        if (i + 1) % validation_freq == 0:
            val_portfolio, val_perf, history, trades = evaluate_agent(agent, val_env, initial_balance)
            print("------------------------------------------------------")
            print(f"VALIDATION after Ep {i+1} | Portfolio: ${val_portfolio:,.2f} ({val_perf:+.2f}%) | Trades (B/S): {trades[0]}/{trades[1]}")
            print("------------------------------------------------------")
            plot_validation_results(history, i + 1)

    train_env.close(); val_env.close()
    print("\n--- Training Finished ---")
