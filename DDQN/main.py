import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from collections import deque
import random
import ta
import sys
import copy

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # --- Data ---
    DATA_URL = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"
    
    # --- Hardware ---
    # Prioritize MPS (Mac), then CUDA (Nvidia), then CPU
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Hyperparameters ---
    SEQ_LEN = 60            # Lookback window (60 minutes)
    FEATURE_DIM = 8         # Number of input features
    HIDDEN_DIM = 128        # LSTM Hidden Dimension
    
    GAMMA = 0.99            # Discount Factor
    LR = 1e-4               # Learning Rate
    BATCH_SIZE = 64         # Mini-batch size
    MEMORY_SIZE = 50_000    # Replay Buffer Size
    MIN_MEMORY = 1_000      # Min samples before training
    TAU = 0.005             # Soft update parameter for Target Network
    
    # --- Self-Rewarding (Paper 2) ---
    EXPERT_LOOKAHEAD = 15   # Look 15 minutes into the future for "Expert" label
    REWARD_SCALING = 10.0   # Scale small returns to make gradients bigger
    BETA_SR = 0.5           # Weight of Self-Reward Loss in total loss
    
    # --- Exploration ---
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 0.99995      # Slower decay for 1m data
    
    # --- Trading Simulation ---
    INITIAL_CAPITAL = 10_000.0
    COMMISSION = 0.0005     # 0.05% per trade (standard crypto exchange fee)
    
    # --- Walk Forward Split ---
    TRAIN_SIZE = 30_000     # Train on ~20 days of minutes
    TEST_SIZE = 5_000       # Test on ~3.5 days
    STEP_SIZE = 5_000       # Slide forward
    EPOCHS_PER_FOLD = 2     # Keep low to avoid overfitting on noise

# ==========================================
# 2. DATA PROCESSING
# ==========================================
def load_data() -> pd.DataFrame:
    print(">>> [Data] Downloading and processing...")
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
        
        # --- Indicators (Using 'ta' lib) ---
        # 1. Log Returns (Stationary)
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. RSI (Momentum)
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi() / 100.0
        
        # 3. MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd_diff()
        
        # 4. ATR (Volatility)
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['atr_rel'] = df['atr'] / df['Close']
        
        # 5. CCI 
        df['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        
        # 6. Bollinger Band Width
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_width'] = bb.bollinger_wband()
        
        # 7. Volume Change
        df['vol_chg'] = df['Volume'].pct_change()

        # Clean NaNs
        df.dropna(inplace=True)
        
        # --- Z-Score Normalization (Robust) ---
        # We normalize columns using a rolling window to prevent lookahead bias in the data itself
        cols_to_norm = ['log_ret', 'macd', 'atr_rel', 'cci', 'bb_width', 'vol_chg']
        
        for c in cols_to_norm:
            # Rolling standardization
            roll = df[c].rolling(window=Config.TRAIN_SIZE, min_periods=Config.SEQ_LEN)
            df[c] = (df[c] - roll.mean()) / (roll.std() + 1e-8)
        
        # Clip outliers to stabilize LSTM
        df[cols_to_norm] = df[cols_to_norm].clip(-5, 5)
        
        # RSI is already 0-1, just center it
        df['rsi'] = (df['rsi'] - 0.5) * 2.0 
        
        # Drop initial NaN from rolling
        df.dropna(inplace=True)
        
        feature_cols = ['log_ret', 'rsi', 'macd', 'atr_rel', 'cci', 'bb_width', 'vol_chg']
        
        # Convert to float32
        for c in feature_cols:
            df[c] = df[c].astype('float32')
            
        print(f">>> [Data] Ready. Samples: {len(df)}. Features: {feature_cols}")
        return df, feature_cols
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

# ==========================================
# 3. MODEL: SR-DDQN (LSTM + Self-Reward)
# ==========================================
class SRDDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(SRDDQN, self).__init__()
        
        # 1. Feature Extractor (Deep LSTM - Paper 1)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        
        # 2. Dueling Heads (DDQN Standard)
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
        
        # 3. Self-Rewarding Head (Paper 2)
        # Predicts the expected 'Expert Reward' for the current state
        self.reward_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Features)
        lstm_out, _ = self.lstm(x)
        # Take the last hidden state of the sequence
        features = lstm_out[:, -1, :]
        
        # Dueling Logic
        val = self.value(features)
        adv = self.advantage(features)
        q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
        
        # Intrinsic Reward Prediction
        pred_reward = self.reward_net(features)
        
        return q_vals, pred_reward

# ==========================================
# 4. TRADING ENVIRONMENT
# ==========================================
class TradingEnv:
    def __init__(self, df, feature_cols):
        self.df = df
        self.feature_cols = feature_cols
        self.prices = df['Close'].values
        self.dates = df.index
        # Pre-convert features to numpy for speed
        self.features = df[feature_cols].values
        self.n_steps = len(df)
        
        self.reset()
        
    def reset(self):
        self.current_step = Config.SEQ_LEN
        self.balance = Config.INITIAL_CAPITAL
        self.shares = 0.0
        self.entry_price = 0.0
        self.net_worth = Config.INITIAL_CAPITAL
        self.prev_net_worth = Config.INITIAL_CAPITAL
        
        self.history = []
        self.trades = []
        return self._get_state()
        
    def _get_state(self):
        # Get sequence window
        window = self.features[self.current_step - Config.SEQ_LEN : self.current_step]
        
        # Augment with Position Info (Are we currently invested?)
        # 1.0 if Long, 0.0 if Cash.
        # We append this as a feature channel to the LSTM
        pos_feature = np.full((Config.SEQ_LEN, 1), 1.0 if self.shares > 0 else 0.0, dtype=np.float32)
        state = np.hstack((window, pos_feature)) # Shape: (Seq, Feat+1)
        return state
    
    def _get_expert_reward(self):
        """
        Paper 2: Expert Knowledge.
        Lookahead: If price is significantly higher in K steps, expert says 'Should be Long'.
        """
        if self.current_step + Config.EXPERT_LOOKAHEAD >= self.n_steps:
            return 0.0
        
        current_price = self.prices[self.current_step]
        future_price = self.prices[self.current_step + Config.EXPERT_LOOKAHEAD]
        
        # Future Return
        future_ret = (future_price - current_price) / current_price
        
        # Threshold to cover commissions
        threshold = Config.COMMISSION * 2
        
        # If we are Long
        if self.shares > 0:
            if future_ret > threshold: return future_ret * Config.REWARD_SCALING  # Good job
            if future_ret < -threshold: return future_ret * Config.REWARD_SCALING # Bad job
            return 0.0
        
        # If we are Cash (Neutral)
        else:
            if future_ret > threshold: return -future_ret * Config.REWARD_SCALING # Missed opportunity (Penalty)
            if future_ret < -threshold: return abs(future_ret) * Config.REWARD_SCALING # Good avoid
            return 0.0

    def step(self, action):
        # Action 0: HOLD (or Stay Cash)
        # Action 1: BUY / LONG
        # Action 2: SELL / CASH OUT
        
        current_price = self.prices[self.current_step]
        date = self.dates[self.current_step]
        reward = 0.0
        trade_info = None
        
        # Execute Trade
        if action == 1 and self.shares == 0: # BUY
            cost = self.balance * Config.COMMISSION
            self.shares = (self.balance - cost) / current_price
            self.balance = 0.0
            self.entry_price = current_price
            self.trades.append({'step': self.current_step, 'date': date, 'type': 'buy', 'price': current_price})
            
        elif action == 2 and self.shares > 0: # SELL
            revenue = self.shares * current_price
            cost = revenue * Config.COMMISSION
            self.balance = revenue - cost
            
            # Profit calc
            profit_pct = (current_price - self.entry_price) / self.entry_price
            is_win = profit_pct > 0
            
            self.shares = 0.0
            self.entry_price = 0.0
            
            trade_info = {'step': self.current_step, 'date': date, 'type': 'sell', 'price': current_price, 'win': is_win, 'profit': profit_pct}
            self.trades.append(trade_info)
            
        # Update Net Worth
        if self.shares > 0:
            self.net_worth = self.shares * current_price
        else:
            self.net_worth = self.balance
            
        # Calculate Rewards
        # 1. Immediate Reward: PnL Change (Scaled)
        pnl_change = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        immediate_reward = pnl_change * 100.0 # Scale up for stability
        
        # 2. Expert Reward (Future Lookahead)
        expert_reward = self._get_expert_reward()
        
        self.prev_net_worth = self.net_worth
        
        # Advance Step
        self.current_step += 1
        done = self.current_step >= self.n_steps - Config.EXPERT_LOOKAHEAD - 1
        
        # History
        self.history.append({
            'date': date,
            'price': current_price,
            'net_worth': self.net_worth,
            'shares': self.shares > 0,
            'expert_reward': expert_reward
        })
        
        return self._get_state(), immediate_reward, expert_reward, done, trade_info

# ==========================================
# 5. AGENT
# ==========================================
class Agent:
    def __init__(self, input_dim, action_dim):
        self.action_dim = action_dim
        
        # Networks
        self.policy_net = SRDDQN(input_dim, Config.HIDDEN_DIM, action_dim).to(Config.DEVICE)
        self.target_net = SRDDQN(input_dim, Config.HIDDEN_DIM, action_dim).to(Config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=Config.LR, weight_decay=1e-5)
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        
        self.epsilon = Config.EPS_START
        self.learn_steps = 0
        
    def select_action(self, state, is_eval=False):
        if not is_eval and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            q_vals, _ = self.policy_net(state_t)
            return q_vals.argmax().item()
            
    def store(self, state, action, r_imm, r_exp, next_state, done):
        self.memory.append((state, action, r_imm, r_exp, next_state, done))
        
    def update(self):
        if len(self.memory) < Config.MIN_MEMORY:
            return None
        
        batch = random.sample(self.memory, Config.BATCH_SIZE)
        state, action, r_imm, r_exp, next_state, done = zip(*batch)
        
        state = torch.FloatTensor(np.array(state)).to(Config.DEVICE)
        action = torch.LongTensor(action).unsqueeze(1).to(Config.DEVICE)
        r_imm = torch.FloatTensor(r_imm).unsqueeze(1).to(Config.DEVICE)
        r_exp = torch.FloatTensor(r_exp).unsqueeze(1).to(Config.DEVICE)
        next_state = torch.FloatTensor(np.array(next_state)).to(Config.DEVICE)
        done = torch.FloatTensor(done).unsqueeze(1).to(Config.DEVICE)
        
        # --- SR-DDQN Logic ---
        
        # 1. Forward Pass
        curr_q, pred_reward = self.policy_net(state)
        curr_q = curr_q.gather(1, action)
        
        # 2. Self-Reward Loss (Train auxiliary head to predict expert reward)
        loss_sr = F.smooth_l1_loss(pred_reward, r_exp)
        
        # 3. Determine Reward for Q-Learning (Max of Expert vs Predicted)
        # This is the core mechanism of Paper 2
        # Use detached pred_reward to not mess up gradients
        combined_expert = r_imm + r_exp
        combined_pred = r_imm + pred_reward.detach()
        final_reward = torch.max(combined_expert, combined_pred)
        
        # 4. Double DQN Target
        with torch.no_grad():
            next_q_online, _ = self.policy_net(next_state)
            next_action = next_q_online.argmax(1, keepdim=True)
            
            next_q_target, _ = self.target_net(next_state)
            next_q_val = next_q_target.gather(1, next_action)
            
            target = final_reward + (Config.GAMMA * next_q_val * (1 - done))
            
        loss_q = F.smooth_l1_loss(curr_q, target)
        
        # Total Loss
        total_loss = loss_q + (Config.BETA_SR * loss_sr)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft Update Target
        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(Config.TAU * param.data + (1 - Config.TAU) * target_param.data)
            
        self.learn_steps += 1
        if self.epsilon > Config.EPS_END:
            self.epsilon *= Config.EPS_DECAY
            
        return total_loss.item()

# ==========================================
# 6. VISUALIZATION
# ==========================================
def plot_results(history, trades, title):
    df = pd.DataFrame(history)
    df.set_index('date', inplace=True)
    
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=("Price & Trades", "Portfolio Value", "Holdings", "Reward Signal")
    )
    
    # 1. Price
    fig.add_trace(go.Scatter(x=df.index, y=df['price'], name='BTC', line=dict(color='blue', width=1)), row=1, col=1)
    
    # Trades
    buys = [t for t in trades if t['type'] == 'buy']
    wins = [t for t in trades if t['type'] == 'sell' and t['win']]
    losses = [t for t in trades if t['type'] == 'sell' and not t['win']]
    
    if buys:
        fig.add_trace(go.Scatter(
            x=[t['date'] for t in buys], y=[t['price'] for t in buys],
            mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=10, color='green')
        ), row=1, col=1)
    if wins:
        fig.add_trace(go.Scatter(
            x=[t['date'] for t in wins], y=[t['price'] for t in wins],
            mode='markers', name='Sell (Win)', marker=dict(symbol='triangle-down', size=10, color='lime')
        ), row=1, col=1)
    if losses:
        fig.add_trace(go.Scatter(
            x=[t['date'] for t in losses], y=[t['price'] for t in losses],
            mode='markers', name='Sell (Loss)', marker=dict(symbol='triangle-down', size=10, color='red')
        ), row=1, col=1)
        
    # 2. Portfolio
    fig.add_trace(go.Scatter(x=df.index, y=df['net_worth'], name='Net Worth', line=dict(color='purple')), row=2, col=1)
    
    # Benchmark
    start_price = df['price'].iloc[0]
    start_bal = df['net_worth'].iloc[0]
    bench = (df['price'] / start_price) * start_bal
    fig.add_trace(go.Scatter(x=df.index, y=bench, name='Buy & Hold', line=dict(color='gray', dash='dot')), row=2, col=1)
    
    # 3. Holdings
    # Convert boolean to 0/1 for plotting
    holdings = df['shares'].astype(int)
    fig.add_trace(go.Scatter(x=df.index, y=holdings, name='In Market', fill='tozeroy', line=dict(color='orange')), row=3, col=1)
    
    # 4. Expert Reward
    # Smooth it to visualize the trend signal
    fig.add_trace(go.Scatter(x=df.index, y=df['expert_reward'].rolling(30).mean(), name='Expert Signal (Avg)', line=dict(color='teal')), row=4, col=1)
    
    fig.update_layout(title=title, height=1000, template="plotly_dark")
    fig.show()

# ==========================================
# 7. MAIN WALK-FORWARD LOOP
# ==========================================
def main():
    # 1. Load Data
    full_df, feature_cols = load_data()
    # Input dim = features + 1 (position flag)
    input_dim = len(feature_cols) + 1
    
    # Initialize Agent
    agent = Agent(input_dim=input_dim, action_dim=3)
    print(f">>> [System] Device: {Config.DEVICE}")
    print(f">>> [System] SR-DDQN Model initialized.")
    
    total_len = len(full_df)
    current_idx = 0
    fold = 1
    
    # Walk-Forward Logic
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
        
        print(f"Train Period: {train_df.index[0]} -> {train_df.index[-1]}")
        print(f"Test Period:  {test_df.index[0]} -> {test_df.index[-1]}")
        
        # --- TRAIN PHASE ---
        train_env = TradingEnv(train_df, feature_cols)
        
        # Optional: Reset epsilon slightly to allow adaptation to new data
        agent.epsilon = max(agent.epsilon, 0.3)
        
        print(f">>> Training ({Config.EPOCHS_PER_FOLD} Epochs)...")
        for epoch in range(Config.EPOCHS_PER_FOLD):
            state = train_env.reset()
            done = False
            losses = []
            
            pbar = tqdm(total=Config.TRAIN_SIZE, desc=f"Ep {epoch+1}", leave=False)
            while not done:
                action = agent.select_action(state)
                next_state, r_imm, r_exp, done, _ = train_env.step(action)
                
                agent.store(state, action, r_imm, r_exp, next_state, done)
                loss = agent.update()
                
                if loss: losses.append(loss)
                state = next_state
                pbar.update(1)
            pbar.close()
            
            avg_loss = np.mean(losses) if losses else 0
            print(f"    Epoch {epoch+1} | Loss: {avg_loss:.5f} | Net Worth: ${train_env.net_worth:.2f} | Eps: {agent.epsilon:.3f}")
            
        # --- TEST PHASE ---
        print(">>> Testing...")
        test_env = TradingEnv(test_df, feature_cols)
        state = test_env.reset()
        done = False
        
        while not done:
            # Pure Exploitation
            action = agent.select_action(state, is_eval=True)
            state, _, _, done, _ = test_env.step(action)
            
        # --- METRICS ---
        final_bal = test_env.net_worth
        profit_pct = ((final_bal - Config.INITIAL_CAPITAL) / Config.INITIAL_CAPITAL) * 100
        
        # Benchmark
        start_price = test_df['Close'].iloc[0]
        end_price = test_df['Close'].iloc[-1]
        bh_profit = ((end_price - start_price) / start_price) * 100
        
        # Trade Stats
        sell_trades = [t for t in test_env.trades if t['type'] == 'sell']
        wins = [t for t in sell_trades if t['win']]
        win_rate = (len(wins) / len(sell_trades) * 100) if sell_trades else 0
        
        print(f"\n[RESULT FOLD {fold}]")
        print(f"Bot Profit:   {profit_pct:.2f}%")
        print(f"Buy & Hold:   {bh_profit:.2f}%")
        print(f"Win Rate:     {win_rate:.2f}% ({len(wins)}/{len(sell_trades)})")
        print(f"Total Trades: {len(test_env.trades)}")
        
        # Plot
        plot_results(test_env.history, test_env.trades, title=f"SR-DDQN Results - Fold {fold}")
        
        # Slide Window
        current_idx += Config.STEP_SIZE
        fold += 1

if __name__ == "__main__":
    main()
