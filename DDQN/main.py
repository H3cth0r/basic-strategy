import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from collections import deque
import random
import ta
import sys

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
class Config:
    # Data
    DATA_URL = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"
    
    # RL Parameters
    SEQ_LEN = 40           # State lookback window
    GAMMA = 0.95           # Discount factor (Slightly lower to value immediate rewards)
    LR = 0.0005            # Learning rate
    BATCH_SIZE = 128
    MEMORY_SIZE = 50000
    MIN_MEMORY_SIZE = 1000
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.9995 # Slower decay per step
    TARGET_UPDATE = 500    # Steps between target net updates
    
    # Trading Environment
    INITIAL_BALANCE = 10000.0
    COMMISSION = 0.001     # 0.1% per trade
    
    # Walk-Forward Validation
    TRAIN_SIZE = 30000     # Larger training window
    TEST_SIZE = 5000       
    STEP_SIZE = 5000       
    EPOCHS_PER_FOLD = 2    # Keep low for speed, rely on step-by-step training

    # Device
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
def load_data() -> pd.DataFrame:
    print("Downloading data...")
    url = Config.DATA_URL
    column_names = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
    try:
        df = pd.read_csv(
            url, skiprows=[1, 2], header=0, names=column_names,
            parse_dates=["Datetime"], index_col="Datetime",
            dtype={"Volume": "int64"}, na_values=["NA", "N/A", ""],
            keep_default_na=True,
        )
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        # --- Feature Engineering ---
        # 1. Log Returns
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi() / 100.0 # Scale 0-1
        
        # 3. MACD Diff (Normalized later)
        macd = ta.trend.MACD(df['Close'])
        df['macd_diff'] = macd.macd_diff()
        
        # 4. Volatility for Reward
        df['rolling_std'] = df['log_ret'].rolling(window=20).std()
        
        # 5. ATR (Volatility)
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Clean NaNs
        df.dropna(inplace=True)
        
        print(f"Data loaded: {len(df)} samples.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

# ==========================================
# DUELING DEEP Q-NETWORK MODEL
# ==========================================
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Value Stream (V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream (A(s, a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

# ==========================================
# TRADING ENVIRONMENT
# ==========================================
class TradingEnv:
    def __init__(self, df, is_eval=False):
        self.df = df
        self.n_steps = len(df)
        self.is_eval = is_eval
        
        self.reset()
        
    def reset(self):
        self.current_step = Config.SEQ_LEN
        self.balance = Config.INITIAL_BALANCE
        self.shares_held = 0.0
        self.net_worth = Config.INITIAL_BALANCE
        self.initial_worth = Config.INITIAL_BALANCE
        self.trades = [] 
        self.history = []
        self.holding_duration = 0
        return self._get_state()
        
    def _get_state(self):
        # Window of data
        start = self.current_step - Config.SEQ_LEN
        end = self.current_step
        window = self.df.iloc[start:end]
        
        # Feature Extraction & Z-Score Normalization
        # We normalize based on the current window stats to be robust to trends
        log_ret = window['log_ret'].values
        rsi = window['rsi'].values
        macd = window['macd_diff'].values
        
        # Normalize MACD locally
        if np.std(macd) > 1e-5:
            macd_norm = (macd - np.mean(macd)) / np.std(macd)
        else:
            macd_norm = macd
            
        # Position Indicator (Agent needs to know if it's already in the market)
        pos_flag = np.full(Config.SEQ_LEN, 1.0 if self.shares_held > 0 else 0.0)
        
        # Concatenate features: Shape (4 * SEQ_LEN)
        state_features = np.concatenate([log_ret, rsi, macd_norm, pos_flag])
        return state_features

    def step(self, action):
        # Actions: 0=Hold, 1=Buy, 2=Sell
        current_data = self.df.iloc[self.current_step]
        current_price = current_data['Close']
        current_idx = self.df.index[self.current_step]
        
        prev_net_worth = self.net_worth
        trade_info = None
        reward = 0
        
        # --- EXECUTE ACTION ---
        if action == 1: # Buy
            if self.shares_held == 0:
                # All-in
                amount_to_invest = self.balance
                cost = amount_to_invest * Config.COMMISSION
                self.shares_held = (amount_to_invest - cost) / current_price
                self.balance = 0
                self.trades.append({'step': self.current_step, 'type': 'buy', 'price': current_price, 'idx': current_idx})
                self.holding_duration = 0
        
        elif action == 2: # Sell
            if self.shares_held > 0:
                revenue = self.shares_held * current_price
                cost = revenue * Config.COMMISSION
                self.balance = revenue - cost
                
                # Check win/loss
                last_buy = [t for t in self.trades if t['type'] == 'buy'][-1]
                buy_price = last_buy['price']
                is_win = current_price > buy_price
                
                self.shares_held = 0
                trade_info = {'step': self.current_step, 'type': 'sell', 'price': current_price, 'win': is_win, 'idx': current_idx}
                self.trades.append(trade_info)

        # --- UPDATE STATE ---
        self.net_worth = self.balance + (self.shares_held * current_price)
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # --- REWARD CALCULATION ---
        # Paper Strategy: Sharpe-based / Return-based
        # We calculate the Log Return of the Close price
        step_log_ret = current_data['log_ret']
        
        if self.shares_held > 0:
            # If holding, we get the return (positive or negative)
            # Scaled by volatility to approximate Sharpe
            vol = current_data['rolling_std'] if current_data['rolling_std'] > 1e-6 else 1.0
            reward = (step_log_ret / vol) 
        else:
            # If not holding, reward is 0 (or small penalty to encourage activity if needed)
            reward = 0 
            
            # Optional: Penalty if price went up and we didn't hold (FOMO penalty)
            # if step_log_ret > 0: reward -= 0.1 

        if self.is_eval:
            self.history.append({
                'date': current_idx,
                'price': current_price,
                'balance': self.balance,
                'net_worth': self.net_worth,
                'shares': self.shares_held
            })

        return self._get_state(), reward, done, trade_info

# ==========================================
# AGENT
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.epsilon = Config.EPSILON_START
        self.steps_done = 0
        
        self.policy_net = DuelingDQN(state_dim, action_dim).to(Config.DEVICE)
        self.target_net = DuelingDQN(state_dim, action_dim).to(Config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LR)
        self.loss_fn = nn.SmoothL1Loss() # Huber Loss (better for stability than MSE)

    def act(self, state, is_test=False):
        if not is_test and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < Config.MIN_MEMORY_SIZE:
            return None, None
        
        batch = random.sample(self.memory, Config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(Config.DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(Config.DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(Config.DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(Config.DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(Config.DEVICE)
        
        # DDQN
        # 1. Action selection from Policy Net
        next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        # 2. Action evaluation from Target Net
        next_q_values = self.target_net(next_states).gather(1, next_actions)
        
        expected_q_values = rewards + (Config.GAMMA * next_q_values * (1 - dones))
        curr_q_values = self.policy_net(states).gather(1, actions)
        
        loss = self.loss_fn(curr_q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps_done += 1
        if self.steps_done % Config.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item(), curr_q_values.mean().item()

    def update_epsilon(self):
        self.epsilon = max(Config.EPSILON_END, self.epsilon * Config.EPSILON_DECAY)

# ==========================================
# PLOTTING
# ==========================================
def plot_fold_results(history, trades, fold_idx, title="Results"):
    if not history:
        print("No history to plot.")
        return

    dates = [x['date'] for x in history]
    prices = [x['price'] for x in history]
    net_worths = [x['net_worth'] for x in history]
    balances = [x['balance'] for x in history]
    shares = [x['shares'] for x in history]
    
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        subplot_titles=(
            f"Fold {fold_idx}: Stock Price & Trades", 
            "Portfolio Value", 
            "Cash Balance", 
            "Holdings"
        ),
        row_heights=[0.5, 0.2, 0.15, 0.15]
    )

    # 1. Price
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price', line=dict(color='#636EFA', width=1)), row=1, col=1)
    
    # Buy Markers
    buy_x = [t['idx'] for t in trades if t['type'] == 'buy']
    buy_y = [t['price'] for t in trades if t['type'] == 'buy']
    fig.add_trace(go.Scatter(
        x=buy_x, y=buy_y, mode='markers', name='Buy',
        marker=dict(symbol='triangle-up', color='#00CC96', size=12, line=dict(width=1, color='black'))
    ), row=1, col=1)
    
    # Sell Win Markers
    win_x = [t['idx'] for t in trades if t['type'] == 'sell' and t['win']]
    win_y = [t['price'] for t in trades if t['type'] == 'sell' and t['win']]
    fig.add_trace(go.Scatter(
        x=win_x, y=win_y, mode='markers', name='Sell (Win)',
        marker=dict(symbol='triangle-down', color='#19D3F3', size=12, line=dict(width=1, color='black'))
    ), row=1, col=1)

    # Sell Loss Markers
    loss_x = [t['idx'] for t in trades if t['type'] == 'sell' and not t['win']]
    loss_y = [t['price'] for t in trades if t['type'] == 'sell' and not t['win']]
    fig.add_trace(go.Scatter(
        x=loss_x, y=loss_y, mode='markers', name='Sell (Loss)',
        marker=dict(symbol='triangle-down', color='#EF553B', size=12, line=dict(width=1, color='black'))
    ), row=1, col=1)

    # 2. Portfolio
    fig.add_trace(go.Scatter(x=dates, y=net_worths, mode='lines', name='Net Worth', line=dict(color='#AB63FA')), row=2, col=1)
    
    # 3. Cash
    fig.add_trace(go.Scatter(x=dates, y=balances, mode='lines', name='Cash', line=dict(color='#FFA15A')), row=3, col=1)
    
    # 4. Holdings
    fig.add_trace(go.Scatter(x=dates, y=shares, mode='lines', name='Holdings', line=dict(color='#19D3F3'), fill='tozeroy'), row=4, col=1)

    fig.update_layout(height=1000, title_text=title, template="plotly_dark")
    fig.show()

# ==========================================
# MAIN
# ==========================================
def main():
    df = load_data()
    
    # 4 features * SEQ_LEN
    state_dim = 4 * Config.SEQ_LEN
    action_dim = 3
    
    agent = DQNAgent(state_dim, action_dim)
    
    current_idx = 0
    fold_num = 1
    total_len = len(df)
    
    print("\n" + "="*50)
    print("STARTING WALK-FORWARD ANALYSIS")
    print("="*50)
    
    while current_idx + Config.TRAIN_SIZE + Config.TEST_SIZE <= total_len:
        # Define indices
        train_start = current_idx
        train_end = current_idx + Config.TRAIN_SIZE
        test_start = train_end
        test_end = test_start + Config.TEST_SIZE
        
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]
        
        date_train_str = f"{train_df.index[0].date()} -> {train_df.index[-1].date()}"
        date_test_str = f"{test_df.index[0].date()} -> {test_df.index[-1].date()}"
        
        print(f"\nFOLD {fold_num}")
        print(f"Training Period: {date_train_str} ({len(train_df)} steps)")
        print(f"Testing Period:  {date_test_str} ({len(test_df)} steps)")
        
        # --- TRAINING ---
        env_train = TradingEnv(train_df)
        agent.epsilon = Config.EPSILON_START # Reset exploration for new data distribution!
        
        for epoch in range(Config.EPOCHS_PER_FOLD):
            state = env_train.reset()
            done = False
            
            losses = []
            q_vals = []
            actions_count = {0:0, 1:0, 2:0}
            
            pbar = tqdm(total=env_train.n_steps, desc=f"Train Ep {epoch+1}/{Config.EPOCHS_PER_FOLD}", leave=False)
            
            while not done:
                action = agent.act(state)
                actions_count[action] += 1
                
                next_state, reward, done, _ = env_train.step(action)
                agent.memory.append((state, action, reward, next_state, done))
                
                state = next_state
                
                loss, mean_q = agent.learn()
                if loss is not None:
                    losses.append(loss)
                    q_vals.append(mean_q)
                    agent.update_epsilon()
                
                pbar.update(1)
            pbar.close()
            
            avg_loss = np.mean(losses) if losses else 0
            avg_q = np.mean(q_vals) if q_vals else 0
            print(f"  Ep {epoch+1}: Loss {avg_loss:.5f} | Avg Q {avg_q:.4f} | Eps {agent.epsilon:.2f} | Acts: H{actions_count[0]} B{actions_count[1]} S{actions_count[2]}")

        # --- TESTING ---
        env_test = TradingEnv(test_df, is_eval=True)
        state = env_test.reset()
        done = False
        
        while not done:
            action = agent.act(state, is_test=True) # Pure exploitation
            next_state, _, done, _ = env_test.step(action)
            state = next_state
            
        # Stats
        profit = ((env_test.net_worth - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE) * 100
        n_trades = len(env_test.trades)
        wins = len([t for t in env_test.trades if t['type']=='sell' and t['win']])
        losses = len([t for t in env_test.trades if t['type']=='sell' and not t['win']])
        
        print(f"  >> TEST RESULT: Profit {profit:.2f}% | Trades {n_trades} (W{wins}/L{losses})")
        
        # Plot
        plot_fold_results(env_test.history, env_test.trades, fold_num, title=f"Fold {fold_num} Analysis: {date_test_str}")
        
        # Advance Window
        current_idx += Config.STEP_SIZE
        fold_num += 1

if __name__ == "__main__":
    main()
