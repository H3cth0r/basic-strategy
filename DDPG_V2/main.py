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
# ### CUSTOM TRADING ENVIRONMENT
# #################################################################################

class TradingEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, lookback_window=50):
        super(TradingEnv, self).__init__()

        self.df = df.dropna().reset_index() # Drop rows with NaN from indicators
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window

        # Define the action space: 1 continuous action (buy/sell amount)
        # -1 means sell all, 1 means buy with all cash
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Define the observation space: market data + portfolio info
        # The shape is the number of indicator columns + 3 for portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.df.columns) - 1 + 3,), # -1 for 'Datetime' col
            dtype=np.float32
        )

        self.render_mode = "human" # for compatibility

    def _get_observation(self):
        # Get the market data for the current step
        market_obs = self.df.loc[self.current_step, self.df.columns != 'Datetime'].values.astype(np.float32)

        # Get the portfolio state
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalize balance
            self.crypto_held,
            self.portfolio_value / self.initial_balance # Normalize portfolio value
        ], dtype=np.float32)

        # Concatenate market and portfolio observations
        return np.concatenate((market_obs, portfolio_state))

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.crypto_held = 0
        self.portfolio_value = self.initial_balance
        self.current_step = self.lookback_window
        self.trades = []

        return self._get_observation(), {}

    def step(self, action):
        action = action[0] # Action is a single value
        current_price = self.df.loc[self.current_step, 'Close']
        prev_portfolio_value = self.portfolio_value

        # --- Execute Trade ---
        transaction_fee = 0.001 # 0.1% fee

        if action > 0: # Buy
            # Buy with a proportion of the current balance
            trade_amount = self.balance * action
            fee = trade_amount * transaction_fee
            self.balance -= (trade_amount + fee)
            self.crypto_held += trade_amount / current_price

        elif action < 0: # Sell
            # Sell a proportion of the crypto held
            trade_amount = self.crypto_held * abs(action)
            self.crypto_held -= trade_amount
            self.balance += (trade_amount * current_price) * (1 - transaction_fee)

        # --- Update Portfolio ---
        self.current_step += 1
        self.portfolio_value = self.balance + (self.crypto_held * current_price)

        # --- Calculate Reward ---
        reward = self.portfolio_value - prev_portfolio_value

        # --- Check for Termination ---
        terminated = self.portfolio_value <= 0 or self.current_step >= len(self.df) - 1
        truncated = False # Not using time limit truncation here

        # --- Get Next Observation ---
        obs = self._get_observation()

        return obs, reward, terminated, truncated, {}


# ===================================================================================
# +++ FIXED: Data Loading Function +++
# ===================================================================================
def load_and_prepare_data(url):
    """Loads data from URL and calculates technical indicators."""
    print("Loading and preparing data...")
    # Load data, skipping the first 3 metadata rows and assigning column names manually
    df = pd.read_csv(url, skiprows=3, header=None)
    df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']

    # Convert Datetime column and set it as the index
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    # Ensure all data columns are numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add technical indicators using the 'ta' library
    df.dropna(inplace=True)
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(close=df['Close']).macd_diff()
    bollinger = ta.volatility.BollingerBands(close=df['Close'])
    df['BB_HIGH'] = bollinger.bollinger_hband()
    df['BB_LOW'] = bollinger.bollinger_lband()

    # Normalize indicators to be roughly in the same scale
    df['Close'] = df['Close'] / df['Close'].iloc[0]
    df['High'] = df['High'] / df['High'].iloc[0]
    df['Low'] = df['Low'] / df['Low'].iloc[0]
    df['Open'] = df['Open'] / df['Open'].iloc[0]
    df['Volume'] = df['Volume'] / df['Volume'].mean()
    df['RSI'] = df['RSI'] / 100
    
    print("Data preparation complete.")
    return df


# #################################################################################
# ### DDPG AGENT (Provided Code)
# #################################################################################

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1    = nn.Linear(state_dim, 400)
        self.layer_2    = nn.Linear(400, 300)
        self.layer_3    = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x   = F.relu(self.layer_1(x))
        x   = F.relu(self.layer_2(x))
        x   = self.max_action * torch.tanh(self.layer_3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1    = nn.Linear(state_dim + action_dim, 400)
        self.layer_2    = nn.Linear(400, 300)
        self.layer_3    = nn.Linear(300, 1)

    def forward(self, x, u):
        x   = F.relu(self.layer_1(torch.cat([x, u], 1)))
        x   = F.relu(self.layer_2(x))
        x   = self.layer_3(x)
        return x

class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage    = []
        self.max_size   = int(max_size)
        self.ptr        = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.asarray(X, dtype=np.float32))
            y.append(np.asarray(Y, dtype=np.float32))
            u.append(np.asarray(U, dtype=np.float32))
            r.append(np.asarray(R, dtype=np.float32))
            d.append(np.asarray(D, dtype=np.float32))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu             = mu
        self.theta          = theta
        self.sigma          = max_sigma
        self.max_sigma      = max_sigma
        self.min_sigma      = min_sigma
        self.decay_period   = decay_period
        self.action_dim     = action_space.shape[0]
        self.low            = action_space.low
        self.high           = action_space.high
        self.reset()

    def reset(self):
        self.state  = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x           = self.state
        dx          = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state  = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer()

    def select_action(self, state):
       state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
       return self.actor(state).cpu().data.numpy().flatten()

    def update(self, batch_size, gamma=0.99, tau=0.005):
        x, y, u, r, d = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(x).to(self.device)
        action = torch.FloatTensor(u).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        done = torch.FloatTensor(1-d).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)

        # --- Critic Loss ---
        with torch.no_grad():
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * gamma * target_Q)
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Loss ---
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft Target Updates ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


# #################################################################################
# ### MAIN TRAINING LOOP
# #################################################################################
if __name__ == "__main__":
    # --- Parameters ---
    data_url = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"
    total_episodes = 50
    batch_size = 128
    initial_balance = 10000

    # --- Setup Environment ---
    data_df = load_and_prepare_data(data_url)
    env = TradingEnv(df=data_df, initial_balance=initial_balance)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # --- Initialize Agent and Noise ---
    agent = DDPG(state_dim, action_dim, max_action, device)
    noise = OUNoise(env.action_space)

    # --- Training ---
    for i in range(total_episodes):
        state, _ = env.reset()
        noise.reset()
        episode_reward = 0
        
        # We need a step counter for the entire episode
        t = 0
        
        while True:
            action = agent.select_action(state)
            action = noise.get_action(action, t)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add((state, next_state, action, reward, float(done)))

            # Start training only after collecting enough samples
            if len(agent.replay_buffer.storage) > batch_size * 2:
                agent.update(batch_size)

            state = next_state
            episode_reward += reward
            t += 1

            if done:
                break
        
        portfolio_perf = (env.portfolio_value / initial_balance - 1) * 100
        print(f"Episode: {i+1}, Total Reward: {episode_reward:.2f}, Final Portfolio: ${env.portfolio_value:,.2f} ({portfolio_perf:+.2f}%)")

    env.close()
    print("\n--- Training Finished ---")
