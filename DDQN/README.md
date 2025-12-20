Please write the python code of DRL model, based on the best resutls or model configuration , based on both papers.

 First write a general description of the code and then the code inside a block of code. Please keep all the code inside a  single file. 

 Please use plotply to make plots. The plots must show stock value, portfolio value, credit value over time. The plots must show buy trades and sell-loose and sell-win. Please add all the necessary metrics in the plots, to see what is happening at each or how did the bot did.
Please, a single tab for all the plots!

 Please use tqdm as much as possible, to see what is happening at each step. Please print the evalulation metrics in the terminal, with the results of the model. Please use something like forward moving. 

Please use this data. It contains about 130k samples of BTC-USD, one minute intervals. Separete the training and test dataset:

```
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
```

If you need to calculate any indicators, please use python module `ta`.

Please use mps if possible na d please make sure to follow the procedure and configuration from the paper.

Also write the pip install command to install the required dependencies.

Please make sure to have a separated figure per plot or line, beacause portfolio value, stock value and credit can have different scales. Also plot the holdings. The plots and figures must be in the same window, but in separated figures. 

Make sure to correctly identify if it was win or loose sell. 

Please make sure to use correctly the data and divide it into train and test.

Make sure to use correctly the best reward function described in the papers.

Please make sure this does not overfits. Then apply some walk-forward method.

Maybe add the lstm model. Also plot the predictions in the plots. Maybe please train the lstm, print and plot the results of this lstm training and then start using this trained model for the reinforcement learning. Make sure the lstm is well trained.

Maybe use smaller batch sizes, like 20_000 steps max? so that it doesnt take that long and it doesnt overfit and that it trains better. The intention is for intraday, then please consider batches or episode intraday size. This should also speed up training and will align to day trading behvaiour.

Maybe at the first stages, we should use some kind of syntetic data so that we can make the models learns the expected behaviour? This syntetic data should show the model to make trades that make money and that make the portfolio grow.

Then, also it will be a great idea to join reinforcement learning with supervised learning, so that it correctly learns the behaviours we want. so that it truly gets to understand where and when should it be trading. Not just the lstm should be supervised, but also the reinforcement learning model, to teach it where it should be executing trades.

This is some code version i have, but I cannot see a clear strategy or it is not making any money or profits. Make sure to print more information and to show more meaningfull plots that show the stratagy getting better each step:


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
import os

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Data
    DATA_URL = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"
    
    # Device (MPS for Mac, CUDA for Nvidia, CPU otherwise)
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Model Hyperparameters
    SEQ_LEN = 60           # Lookback window (1 hour if 1m candles)
    FEATURE_DIM = 7        # (LogRet, RSI, MACD, ATR, CCI, Vol_Change, Position)
    HIDDEN_DIM = 128       # LSTM Hidden Size
    GAMMA = 0.99           # Discount factor
    LR = 1e-4              # Learning Rate (combined)
    BATCH_SIZE = 64        # Smaller batch size for stability
    MEMORY_SIZE = 20000    # Replay buffer
    MIN_MEMORY_SIZE = 1000
    
    # Self-Rewarding Params (Paper 2)
    EXPERT_LOOKAHEAD = 5   # How many steps ahead to look for 'Expert' label
    
    # Exploration
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.9995 
    
    # Trading Constraints
    INITIAL_BALANCE = 10000.0
    COMMISSION = 0.001     # 0.1% per trade
    STOP_LOSS = -0.03      # 3% Stop loss
    TAKE_PROFIT = 0.06     # 6% Take profit
    
    # Walk-Forward Validation
    TRAIN_SIZE = 25000     
    TEST_SIZE = 5000       
    STEP_SIZE = 5000       
    EPOCHS = 3             # Epochs per fold (keep low to avoid overfitting on noise)

# ==========================================
# 1. DATA LOADING & PROCESSING
# ==========================================
def load_data() -> pd.DataFrame:
    print(">>> Downloading and processing data...")
    try:
        df = pd.read_csv(
            Config.DATA_URL, skiprows=[1, 2], header=0,
            names=["Datetime", "Close", "High", "Low", "Open", "Volume"],
            parse_dates=["Datetime"], index_col="Datetime",
            dtype={"Volume": "float32"}, na_values=["NA", "N/A", ""],
            keep_default_na=True
        )
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        # --- Feature Engineering ---
        # 1. Log Returns (Stationary Price)
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        
        # 2. RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi() / 100.0
        
        # 3. MACD (Normalized)
        macd = ta.trend.MACD(df['Close'])
        df['macd_diff'] = macd.macd_diff()
        # Rolling Z-Score for MACD to keep it in range
        df['macd_norm'] = (df['macd_diff'] - df['macd_diff'].rolling(200).mean()) / (df['macd_diff'].rolling(200).std() + 1e-6)
        
        # 4. ATR (Volatility)
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['atr_norm'] = df['atr'] / df['Close']
        
        # 5. CCI
        df['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        df['cci_norm'] = df['cci'] / 200.0 # Scale roughly to [-1, 1]
        
        # 6. Volume Change
        df['vol_chg'] = df['Volume'].pct_change().fillna(0)
        df['vol_norm'] = np.clip(df['vol_chg'], -1.0, 1.0)

        # Drop NaNs generated by indicators
        df.dropna(inplace=True)
        
        # Select Features
        features = ['log_ret', 'rsi', 'macd_norm', 'atr_norm', 'cci_norm', 'vol_norm']
        for c in features:
            df[c] = df[c].astype('float32')
            
        print(f">>> Data Loaded: {len(df)} candles. Features: {features}")
        return df, features
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

# ==========================================
# 2. MODEL ARCHITECTURE (LSTM + SR-DDQN)
# ==========================================

class SRDDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(SRDDQN, self).__init__()
        
        # Shared Feature Extractor (LSTM)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        
        # --- Q-Network Head (Dueling DQN) ---
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )
        
        # --- Self-Reward Head (Paper 2) ---
        # Predicts the "Expert Reward" for the current state
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        lstm_out, _ = self.lstm(x) 
        # Take last time step
        features = lstm_out[:, -1, :] 
        
        # Dueling Q-Learning
        adv = self.advantage(features)
        val = self.value(features)
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        
        # Predicted Reward (for self-correction)
        pred_reward = self.reward_predictor(features)
        
        return q_values, pred_reward

# ==========================================
# 3. TRADING ENVIRONMENT
# ==========================================
class TradingEnv:
    def __init__(self, df, feature_cols):
        self.df = df
        self.feature_cols = feature_cols
        self.n_steps = len(df)
        self.prices = df['Close'].values
        self.dates = df.index
        
        # Pre-convert features to numpy for speed
        self.data_matrix = df[feature_cols].values
        
        self.reset()
        
    def reset(self):
        self.current_step = Config.SEQ_LEN
        self.balance = Config.INITIAL_BALANCE
        self.shares = 0.0
        self.entry_price = 0.0
        self.net_worth = Config.INITIAL_BALANCE
        self.prev_net_worth = Config.INITIAL_BALANCE
        self.trades = []
        self.history = []
        return self._get_state()
    
    def _get_state(self):
        # 1. Get Window of features
        # Shape: (SEQ_LEN, Feature_Dim-1)
        window_data = self.data_matrix[self.current_step - Config.SEQ_LEN : self.current_step]
        
        # 2. Add Position Flag (Are we Long?)
        # We append this as an extra feature to every timestep in sequence
        pos_val = 1.0 if self.shares > 0 else 0.0
        pos_channel = np.full((Config.SEQ_LEN, 1), pos_val, dtype=np.float32)
        
        # Shape: (SEQ_LEN, Feature_Dim)
        state = np.hstack((window_data, pos_channel))
        return state

    def get_expert_reward(self):
        """
        Paper 2 Strategy: Look ahead to define the 'True' reward.
        If we Buy, and price goes up in k steps -> High Reward.
        """
        if self.current_step + Config.EXPERT_LOOKAHEAD >= self.n_steps:
            return 0.0
            
        curr_price = self.prices[self.current_step]
        future_price = self.prices[self.current_step + Config.EXPERT_LOOKAHEAD]
        
        pct_change = (future_price - curr_price) / curr_price
        
        # Logarithmic scaling of reward for stability
        return pct_change * 100.0

    def step(self, action):
        # Actions: 0=Hold, 1=Buy, 2=Sell
        current_price = self.prices[self.current_step]
        date = self.dates[self.current_step]
        
        done = False
        trade_info = None
        
        # --- Trading Logic ---
        if action == 1: # BUY
            if self.shares == 0:
                cost = self.balance * Config.COMMISSION
                self.shares = (self.balance - cost) / current_price
                self.balance = 0
                self.entry_price = current_price
                self.trades.append({'step': self.current_step, 'date': date, 'type': 'buy', 'price': current_price, 'profit': 0})
        
        elif action == 2: # SELL
            if self.shares > 0:
                revenue = self.shares * current_price
                cost = revenue * Config.COMMISSION
                profit_pct = (current_price - self.entry_price) / self.entry_price
                
                self.balance = revenue - cost
                self.shares = 0
                self.entry_price = 0
                
                is_win = profit_pct > 0
                trade_info = {'step': self.current_step, 'date': date, 'type': 'sell', 'price': current_price, 'win': is_win, 'profit': profit_pct}
                self.trades.append(trade_info)
        
        # --- Risk Management (Stop Loss / Take Profit) ---
        if self.shares > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            if unrealized_pnl < Config.STOP_LOSS or unrealized_pnl > Config.TAKE_PROFIT:
                # Force Close
                revenue = self.shares * current_price
                self.balance = revenue - (revenue * Config.COMMISSION)
                self.shares = 0
                trade_type = 'loss' if unrealized_pnl < 0 else 'win'
                self.trades.append({'step': self.current_step, 'date': date, 'type': 'sell', 'price': current_price, 'win': unrealized_pnl > 0, 'profit': unrealized_pnl})
        
        # --- Update Physics ---
        self.net_worth = self.balance + (self.shares * current_price)
        
        # --- Reward Calculation (Immediate + Expert) ---
        # 1. Immediate PnL change
        step_reward = ((self.net_worth - self.prev_net_worth) / self.prev_net_worth) * 100.0
        
        # 2. Get Expert Reward (Lookahead)
        expert_trend = self.get_expert_reward()
        
        # If we are Long, we want expert trend to be positive
        # If we are Flat, we want expert trend to be negative (avoided loss) or zero
        if self.shares > 0:
            expert_reward = expert_trend
        else:
            expert_reward = -expert_trend * 0.1 # Small reward for avoiding drops, small penalty for missing pumps
            
        self.prev_net_worth = self.net_worth
        
        # Recording History
        self.history.append({
            'date': date,
            'price': current_price,
            'net_worth': self.net_worth,
            'shares': self.shares,
            'expert_reward': expert_reward
        })

        self.current_step += 1
        if self.current_step >= self.n_steps - Config.EXPERT_LOOKAHEAD - 1:
            done = True
            
        return self._get_state(), step_reward, expert_reward, done, trade_info

# ==========================================
# 4. AGENT (TRAINING LOGIC)
# ==========================================
class Agent:
    def __init__(self, input_dim, action_dim):
        self.action_dim = action_dim
        
        self.policy_net = SRDDQN(input_dim, Config.HIDDEN_DIM, action_dim).to(Config.DEVICE)
        self.target_net = SRDDQN(input_dim, Config.HIDDEN_DIM, action_dim).to(Config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=Config.LR, amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss() # Huber Loss equivalent
        
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.epsilon = Config.EPSILON_START
        self.learn_step = 0

    def select_action(self, state, is_eval=False):
        if not is_eval and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            q_vals, _ = self.policy_net(state_t)
            return q_vals.argmax().item()

    def store(self, state, action, reward_immediate, reward_expert, next_state, done):
        self.memory.append((state, action, reward_immediate, reward_expert, next_state, done))

    def train_step(self):
        if len(self.memory) < Config.BATCH_SIZE:
            return None, None
        
        batch = random.sample(self.memory, Config.BATCH_SIZE)
        states, actions, r_imms, r_exps, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(Config.DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(Config.DEVICE)
        r_imms = torch.FloatTensor(r_imms).unsqueeze(1).to(Config.DEVICE)
        r_exps = torch.FloatTensor(r_exps).unsqueeze(1).to(Config.DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(Config.DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(Config.DEVICE)
        
        # --- 1. Forward Pass ---
        curr_q, pred_reward = self.policy_net(states)
        curr_q = curr_q.gather(1, actions)
        
        # --- 2. SR-DRL Logic (Paper 2) ---
        # Self-Rewarding Loss: Train the network to predict the Expert Reward
        sr_loss = self.loss_fn(pred_reward, r_exps)
        
        # Reward Alignment: Use the HIGHER of Expert or Predicted reward for Q-Learning
        # This allows the agent to be guided by expert hindsight during training, 
        # but smooths it out with its own predictions.
        # Note: We combine immediate PnL with the "Trend" reward
        combined_expert = r_imms + r_exps
        combined_pred = r_imms + pred_reward.detach()
        final_reward = torch.max(combined_expert, combined_pred)
        
        # --- 3. Double DQN Logic ---
        with torch.no_grad():
            next_q_online, _ = self.policy_net(next_states)
            next_action = next_q_online.argmax(1, keepdim=True)
            
            next_q_target, _ = self.target_net(next_states)
            next_q_val = next_q_target.gather(1, next_action)
            
            target_q = final_reward + (Config.GAMMA * next_q_val * (1 - dones))
            
        q_loss = self.loss_fn(curr_q, target_q)
        
        # Combined Backprop
        total_loss = q_loss + sr_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update Target Net
        self.learn_step += 1
        if self.learn_step % 200 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Decay Epsilon
        if self.epsilon > Config.EPSILON_END:
            self.epsilon *= Config.EPSILON_DECAY
            
        return q_loss.item(), sr_loss.item()

# ==========================================
# 5. VISUALIZATION
# ==========================================
def plot_results(history, trades, fold_id):
    df = pd.DataFrame(history)
    df.set_index('date', inplace=True)
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=(f"Price Action & Trades (Fold {fold_id})", "Portfolio Value", "Holdings", "Expert Rewards")
    )
    
    # --- Row 1: Price & Trades ---
    fig.add_trace(go.Scatter(x=df.index, y=df['price'], name='BTC', line=dict(color='blue', width=1)), row=1, col=1)
    
    # Process Trades
    buys = [t for t in trades if t['type'] == 'buy']
    wins = [t for t in trades if t['type'] == 'sell' and t['win']]
    losses = [t for t in trades if t['type'] == 'sell' and not t['win']]
    
    if buys:
        fig.add_trace(go.Scatter(
            x=[t['date'] for t in buys], y=[t['price'] for t in buys],
            mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=12, color='green')
        ), row=1, col=1)
    if wins:
        fig.add_trace(go.Scatter(
            x=[t['date'] for t in wins], y=[t['price'] for t in wins],
            mode='markers', name='Sell (Win)', marker=dict(symbol='triangle-down', size=12, color='lime')
        ), row=1, col=1)
    if losses:
        fig.add_trace(go.Scatter(
            x=[t['date'] for t in losses], y=[t['price'] for t in losses],
            mode='markers', name='Sell (Loss)', marker=dict(symbol='triangle-down', size=12, color='red')
        ), row=1, col=1)

    # --- Row 2: Portfolio ---
    fig.add_trace(go.Scatter(x=df.index, y=df['net_worth'], name='Bot Net Worth', line=dict(color='purple')), row=2, col=1)
    
    # Benchmark (Buy and Hold)
    initial_price = df['price'].iloc[0]
    initial_bal = df['net_worth'].iloc[0]
    df['bnh'] = (df['price'] / initial_price) * initial_bal
    fig.add_trace(go.Scatter(x=df.index, y=df['bnh'], name='Buy & Hold', line=dict(color='gray', dash='dot')), row=2, col=1)

    # --- Row 3: Holdings ---
    fig.add_trace(go.Scatter(x=df.index, y=df['shares'], name='BTC Position', fill='tozeroy', line=dict(color='orange')), row=3, col=1)

    # --- Row 4: Expert Rewards ---
    # Smooth line to see the trend of rewards
    fig.add_trace(go.Scatter(x=df.index, y=df['expert_reward'].rolling(20).mean(), name='Expert Reward Signal', line=dict(color='teal')), row=4, col=1)

    fig.update_layout(height=1200, template="plotly_dark", title=f"SR-DDQN Trading Results - Fold {fold_id}")
    fig.show()

# ==========================================
# 6. MAIN LOOP (WALK-FORWARD)
# ==========================================
def main():
    # Load Data
    full_df, features = load_data()
    
    # Initialize Agent
    # Feature dim is base features + 1 (for position flag)
    agent = Agent(input_dim=len(features) + 1, action_dim=3)
    
    total_len = len(full_df)
    current_idx = 0
    fold = 1
    
    while current_idx + Config.TRAIN_SIZE + Config.TEST_SIZE < total_len:
        print(f"\n{'='*60}")
        print(f"FOLD {fold} | Walk-Forward Analysis")
        print(f"{'='*60}")
        
        # Define Windows
        train_start = current_idx
        train_end = train_start + Config.TRAIN_SIZE
        test_start = train_end
        test_end = test_start + Config.TEST_SIZE
        
        train_df = full_df.iloc[train_start:train_end]
        test_df = full_df.iloc[test_start:test_end]
        
        print(f"Train: {train_df.index[0]} -> {train_df.index[-1]}")
        print(f"Test:  {test_df.index[0]} -> {test_df.index[-1]}")
        
        # --- Training ---
        print(">>> Training Phase...")
        env = TradingEnv(train_df, features)
        
        # Reset epsilon slightly to allow adaptation to new regime
        agent.epsilon = max(0.2, agent.epsilon) 
        
        for epoch in range(Config.EPOCHS):
            state = env.reset()
            done = False
            q_loss_track = []
            sr_loss_track = []
            
            pbar = tqdm(total=len(train_df), desc=f"Ep {epoch+1}/{Config.EPOCHS}", leave=False)
            
            while not done:
                action = agent.select_action(state)
                next_state, r_imm, r_exp, done, _ = env.step(action)
                
                agent.store(state, action, r_imm, r_exp, next_state, done)
                q_l, sr_l = agent.train_step()
                
                if q_l:
                    q_loss_track.append(q_l)
                    sr_loss_track.append(sr_l)
                    
                state = next_state
                pbar.update(1)
            pbar.close()
            
            print(f"   Avg Q-Loss: {np.mean(q_loss_track):.5f} | SR-Loss: {np.mean(sr_loss_track):.5f} | Final NW: {env.net_worth:.2f}")

        # --- Testing ---
        print(">>> Testing Phase...")
        test_env = TradingEnv(test_df, features)
        state = test_env.reset()
        done = False
        
        while not done:
            # Full exploitation
            action = agent.select_action(state, is_eval=True)
            state, _, _, done, _ = test_env.step(action)
            
        # --- Metrics ---
        profit_pct = ((test_env.net_worth - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE) * 100
        bnh_profit = ((test_df['Close'].iloc[-1] - test_df['Close'].iloc[0]) / test_df['Close'].iloc[0]) * 100
        
        sell_trades = [t for t in test_env.trades if t['type'] == 'sell']
        wins = [t for t in sell_trades if t['win']]
        win_rate = (len(wins) / len(sell_trades) * 100) if sell_trades else 0
        
        print(f"\nRESULTS FOLD {fold}:")
        print(f"Bot Profit: {profit_pct:.2f}%")
        print(f"Buy & Hold: {bnh_profit:.2f}%")
        print(f"Win Rate:   {win_rate:.2f}% ({len(wins)}/{len(sell_trades)})")
        print(f"Trades:     {len(test_env.trades)}")
        
        plot_results(test_env.history, test_env.trades, fold)
        
        # Advance Window
        current_idx += Config.STEP_SIZE
        fold += 1

if __name__ == "__main__":
    main()



Test:  2025-05-27 15:35:00+00:00 -> 2025-05-31 15:56:00+00:00
>>> Training Phase...
   Avg Q-Loss: 0.00900 | SR-Loss: 0.00616 | Final NW: 2059.94                                                                                   
   Avg Q-Loss: 0.01207 | SR-Loss: 0.00697 | Final NW: 4926.51                                                                                   
   Avg Q-Loss: 0.01991 | SR-Loss: 0.00659 | Final NW: 4044.79                                                                                   
>>> Testing Phase...

RESULTS FOLD 1:
Bot Profit: -9.45%
Buy & Hold: -4.69%
Win Rate:   41.67% (10/24)
Trades:     49

============================================================
FOLD 2 | Walk-Forward Analysis
============================================================
Train: 2025-05-12 03:35:00+00:00 -> 2025-05-31 15:56:00+00:00
Test:  2025-05-31 15:57:00+00:00 -> 2025-06-05 01:02:00+00:00
>>> Training Phase...
   Avg Q-Loss: 0.01689 | SR-Loss: 0.00671 | Final NW: 3824.42                                                                                   
   Avg Q-Loss: 0.01383 | SR-Loss: 0.00654 | Final NW: 3683.24                                                                                   
   Avg Q-Loss: 0.01497 | SR-Loss: 0.00642 | Final NW: 3621.80                                                                                   
>>> Testing Phase...

RESULTS FOLD 2:
Bot Profit: -5.75%
Buy & Hold: 0.40%
Win Rate:   56.25% (18/32)
Trades:     65

============================================================
FOLD 3 | Walk-Forward Analysis
============================================================
Train: 2025-05-16 07:37:00+00:00 -> 2025-06-05 01:02:00+00:00
Test:  2025-06-05 01:03:00+00:00 -> 2025-06-09 02:59:00+00:00
>>> Training Phase...
   Avg Q-Loss: 0.01627 | SR-Loss: 0.00619 | Final NW: 2806.39                                                                                   
   Avg Q-Loss: 0.01640 | SR-Loss: 0.00544 | Final NW: 3172.61                                                                                   
   Avg Q-Loss: 0.01454 | SR-Loss: 0.00528 | Final NW: 2939.51                                                                                   
>>> Testing Phase...

RESULTS FOLD 3:
Bot Profit: -9.44%
Buy & Hold: 0.59%
Win Rate:   63.46% (33/52)
Trades:     105

============================================================
FOLD 4 | Walk-Forward Analysis
============================================================
Train: 2025-05-19 23:08:00+00:00 -> 2025-06-09 02:59:00+00:00
Test:  2025-06-09 03:00:00+00:00 -> 2025-06-13 04:35:00+00:00
>>> Training Phase...
   Avg Q-Loss: 0.01967 | SR-Loss: 0.00533 | Final NW: 2760.72                                                                                   
   Avg Q-Loss: 0.01353 | SR-Loss: 0.00497 | Final NW: 3096.18                                                                                   
   Avg Q-Loss: 0.01834 | SR-Loss: 0.00476 | Final NW: 3429.65                                                                                   
>>> Testing Phase...

RESULTS FOLD 4:
Bot Profit: -10.52%
Buy & Hold: -1.32%
Win Rate:   56.52% (26/46)
Trades:     93


============================================================
FOLD 5 | Walk-Forward Analysis
============================================================
Train: 2025-05-23 18:44:00+00:00 -> 2025-06-13 04:35:00+00:00
Test:  2025-06-13 04:36:00+00:00 -> 2025-06-16 18:57:00+00:00
>>> Training Phase...
   Avg Q-Loss: 0.01048 | SR-Loss: 0.00381 | Final NW: 2779.38                                                                                   
   Avg Q-Loss: 0.00969 | SR-Loss: 0.00402 | Final NW: 3332.62                                                                                   
   Avg Q-Loss: 0.01010 | SR-Loss: 0.00400 | Final NW: 2917.35                                                                                   
>>> Testing Phase...

RESULTS FOLD 5:
Bot Profit: -7.65%
Buy & Hold: 3.79%
Win Rate:   59.26% (32/54)
Trades:     109


Please, to the first level of identation, add double tab. Just to the first level of identation, the other elements should look ok. Write the code inside a block of code.
