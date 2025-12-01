import os
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import neat
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from datetime import datetime

# Filter warnings for clean output
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

DATA_URL = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"

# Training Parameters
GENERATIONS = 50       
POP_SIZE = 50          
EPISODE_STEPS = 10080  # ~7 days of 1-minute intervals
TRANSACTION_FEE = 0.001 
INITIAL_CAPITAL = 10000.0

# Neural Network Thresholds
DECISION_THRESHOLD = 0.5  
MIN_TRADE_INTERVAL = 5    

# Global variable to hold data to avoid passing it around constantly
TRAIN_DATA = None
TEST_DATA = None

# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================

def load_and_process_data():
    print("Downloading and processing data...")
    column_names = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
    
    try:
        df = pd.read_csv(
            DATA_URL, 
            skiprows=[1, 2], 
            header=0, 
            names=column_names,
            parse_dates=["Datetime"], 
            index_col="Datetime",
            dtype={"Volume": "float64"}, 
            na_values=["NA", "N/A", ""],
            keep_default_na=True,
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    df = df.sort_index().dropna()
    
    # --- Technical Indicators (Based on Paper) ---
    # Using 'ta' library to vectorized calculations
    
    # 1. SMA 5 & 10 (Normalized by Close Price as per paper)
    sma5 = ta.trend.SMAIndicator(df["Close"], window=5).sma_indicator()
    sma10 = ta.trend.SMAIndicator(df["Close"], window=10).sma_indicator()
    df["sma5_norm"] = (df["Close"] - sma5) / df["Close"]
    df["sma10_norm"] = (df["Close"] - sma10) / df["Close"]

    # 2. Stochastic Oscillator (K & D)
    stoch = ta.momentum.StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3)
    df["slow_k"] = stoch.stoch() / 100.0 # Normalize 0-1
    df["slow_d"] = stoch.stoch_signal() / 100.0

    # 3. Williams %R
    willr = ta.momentum.WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"])
    df["willr"] = (willr.williams_r() + 100) / 100.0 # Normalize 0-1 (Williams is usually -100 to 0)

    # 4. MACD Diff (Normalized by Close)
    macd = ta.trend.MACD(close=df["Close"])
    df["macd_diff"] = macd.macd_diff() / df["Close"]

    # 5. CCI (Normalized, usually oscillates +/- 100, we scale to approx -1 to 1)
    cci = ta.trend.CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"])
    df["cci"] = cci.cci() / 200.0 

    # 6. RSI (Normalized 0-1)
    rsi = ta.momentum.RSIIndicator(close=df["Close"])
    df["rsi"] = rsi.rsi() / 100.0

    # 7. Chaikin ADOSC (Paper suggests Tanh normalization)
    ad = ta.volume.AccDistIndexIndicator(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]).acc_dist_index()
    # ADOSC is traditionally EMA(3)-EMA(10) of AD line. 
    # We approximate normalization using tanh over a rolling mean volume to handle scale
    ema_fast = ta.trend.EMAIndicator(ad, window=3).ema_indicator()
    ema_slow = ta.trend.EMAIndicator(ad, window=10).ema_indicator()
    raw_adosc = ema_fast - ema_slow
    # Robust normalization
    df["adosc"] = np.tanh(raw_adosc / (df["Volume"].rolling(20).mean() * df["Close"] + 1e-9)) 

    # Drop NaNs created by indicators
    df.dropna(inplace=True)
    
    # Split Train/Test (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Data Loaded. Train: {len(train_df)} rows, Test: {len(test_df)} rows.")
    return train_df, test_df

# ==========================================
# 3. TRADING ENVIRONMENT
# ==========================================

class TradingEnv:
    def __init__(self, data):
        self.data = data
        self.reset()
        
    def reset(self):
        self.cash = INITIAL_CAPITAL
        self.holdings = 0.0
        self.trades = []  
        self.peak_equity = INITIAL_CAPITAL
        self.max_drawdown = 0.0
        self.last_trade_step = 0
        self.avg_entry_price = 0.0
        self.entry_tick = 0
        self.closed_trade_durations = []
        
        # Pre-calculate array access for speed
        self.close_prices = self.data["Close"].values
        self.dates = self.data.index
        # Inputs: SMA5, SMA10, SlowK, SlowD, WillR, MACD, CCI, RSI, ADOSC
        self.features = self.data[[
            "sma5_norm", "sma10_norm", "slow_k", "slow_d", 
            "willr", "macd_diff", "cci", "rsi", "adosc"
        ]].values

    def get_state(self, step):
        price = self.close_prices[step]
        equity = self.cash + (self.holdings * price)
        if equity <= 0: equity = 1e-9 
        
        # Paper Inputs: Long Pos, Short Pos (We simulate Long only for simplicity, Short=1-Long)
        long_ratio = (self.holdings * price) / equity
        short_ratio = 1.0 - long_ratio
        
        tech_ind = self.features[step]
        
        # Combine [Long, Short] + [9 Indicators] = 11 Inputs
        state = np.hstack(([long_ratio, short_ratio], tech_ind))
        
        # Clip state to prevent overflow in Neural Net
        return np.clip(state, -5.0, 5.0)

    def step(self, step, action):
        buy_sig, sell_sig, vol_sig = action
        
        price = self.close_prices[step]
        equity = self.cash + (self.holdings * price)
        
        # Track Drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd
            
        # Enforce minimum interval to prevent churning
        if step - self.last_trade_step < MIN_TRADE_INTERVAL:
            return equity

        # Logic from paper: Threshold > 0.5. Compare signals.
        decision = "hold"
        if buy_sig > DECISION_THRESHOLD and sell_sig > DECISION_THRESHOLD:
            decision = "buy" if buy_sig > sell_sig else "sell"
        elif buy_sig > DECISION_THRESHOLD:
            decision = "buy"
        elif sell_sig > DECISION_THRESHOLD:
            decision = "sell"
            
        vol_pct = np.clip(vol_sig, 0.0, 1.0)
        
        if decision == "buy" and self.cash > 10:
            # Buying
            amount_to_spend = self.cash * vol_pct
            if amount_to_spend > 10: # Minimum trade
                shares = (amount_to_spend / price) * (1 - TRANSACTION_FEE)
                cost = shares * price
                
                # Update Avg Entry
                total_val = (self.holdings * self.avg_entry_price) + cost
                total_shares = self.holdings + shares
                self.avg_entry_price = total_val / total_shares if total_shares > 0 else price
                
                if self.holdings == 0:
                    self.entry_tick = step # Start duration timer
                
                self.cash -= amount_to_spend
                self.holdings += shares
                self.last_trade_step = step
                self.trades.append({
                    'step': step, 'type': 'buy', 'price': price, 
                    'result': None, 'time': self.dates[step]
                })
            
        elif decision == "sell" and self.holdings > 0:
            # Selling
            shares_to_sell = self.holdings * vol_pct
            if (shares_to_sell * price) > 10:
                proceeds = (shares_to_sell * price) * (1 - TRANSACTION_FEE)
                
                # Determine win/loss logic based on average entry
                is_win = price > self.avg_entry_price
                
                self.cash += proceeds
                self.holdings -= shares_to_sell
                self.last_trade_step = step
                
                # If position closed (or almost closed), calculate duration
                if self.holdings < (0.0001 / price): # Basically zero
                    duration = step - self.entry_tick
                    self.closed_trade_durations.append(duration)
                    self.holdings = 0
                    self.avg_entry_price = 0
                
                self.trades.append({
                    'step': step, 'type': 'sell', 'price': price, 
                    'result': 'win' if is_win else 'loss', 'time': self.dates[step]
                })

        return equity

# ==========================================
# 4. REPORTING & PLOTTING
# ==========================================

class PeriodicReporter(neat.reporting.BaseReporter):
    """
    Evaluates the best genome on a fixed validation set every generation
    and plots results periodically.
    """
    def __init__(self, validation_data):
        self.val_data = validation_data
        self.generation = 0
        
    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        # Run the best genome on the fixed validation set
        net = neat.nn.RecurrentNetwork.create(best_genome, config)
        env = TradingEnv(self.val_data)
        
        equity_curve = []
        
        # Fast evaluation loop
        for i in range(len(self.val_data)):
            inputs = env.get_state(i)
            action = net.activate(inputs)
            eq = env.step(i, action)
            equity_curve.append(eq)
            
        final_equity = equity_curve[-1]
        pnl = final_equity - INITIAL_CAPITAL
        roi = (pnl / INITIAL_CAPITAL) * 100
        trades = len(env.trades)
        
        print(f"\n[{self.generation}] Validation Results (Best Genome):")
        print(f"   > Equity:   ${final_equity:,.2f} ({roi:+.2f}%)")
        print(f"   > Trades:   {trades}")
        print(f"   > Max DD:   {env.max_drawdown*100:.2f}%")
        
        # Plot every 10 generations
        if self.generation % 10 == 0:
            self.create_plot(env, equity_curve, title=f"Training_Gen_{self.generation}")

    def create_plot(self, env, equity_curve, title):
        df = self.val_data
        
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=("Price & Signals", "Portfolio vs Buy&Hold", "Holdings", "Cash")
        )

        # 1. Price
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color='gray', width=1)), row=1, col=1)
        
        buys = [t for t in env.trades if t['type'] == 'buy']
        sells_win = [t for t in env.trades if t['type'] == 'sell' and t['result'] == 'win']
        sells_loss = [t for t in env.trades if t['type'] == 'sell' and t['result'] == 'loss']
        
        if buys:
            fig.add_trace(go.Scatter(x=[t['time'] for t in buys], y=[t['price'] for t in buys], 
                                     mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name='Buy'), row=1, col=1)
        if sells_win:
            fig.add_trace(go.Scatter(x=[t['time'] for t in sells_win], y=[t['price'] for t in sells_win], 
                                     mode='markers', marker=dict(symbol='triangle-down', color='cyan', size=10), name='Win'), row=1, col=1)
        if sells_loss:
            fig.add_trace(go.Scatter(x=[t['time'] for t in sells_loss], y=[t['price'] for t in sells_loss], 
                                     mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Loss'), row=1, col=1)

        # 2. Equity vs Buy & Hold
        fig.add_trace(go.Scatter(x=df.index, y=equity_curve, name="Bot Equity", line=dict(color='purple')), row=2, col=1)
        
        # Buy & Hold baseline
        start_price = df.iloc[0]["Close"]
        bh_curve = (df["Close"] / start_price) * INITIAL_CAPITAL
        fig.add_trace(go.Scatter(x=df.index, y=bh_curve, name="Buy & Hold", line=dict(color='orange', dash='dot')), row=2, col=1)

        # 3. Holdings (Reconstructed roughly for viz)
        # Since we didn't store array, we can't plot line easily in validation, but we can plot final
        fig.add_annotation(text="Detailed holdings available in Test mode", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, row=3, col=1)

        fig.update_layout(title=title, template="plotly_dark", height=1000)
        filename = f"{title}.html"
        fig.write_html(filename)
        print(f"   > Plot saved: {filename}")

# ==========================================
# 5. FITNESS FUNCTION
# ==========================================

def eval_genomes(genomes, config):
    # --- Progressive / Stochastic Batching ---
    # Pick a random start point to train on a subset (Episode)
    # This prevents overfitting to a specific timeframe
    data_len = len(TRAIN_DATA)
    max_start = data_len - EPISODE_STEPS - 1
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + EPISODE_STEPS
    
    batch_data = TRAIN_DATA.iloc[start_idx:end_idx]
    
    # Calculate Buy & Hold Return for this specific batch
    start_price = batch_data.iloc[0]["Close"]
    end_price = batch_data.iloc[-1]["Close"]
    bh_return_pct = (end_price - start_price) / start_price
    
    # Evaluate every genome on THIS batch
    for genome_id, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        env = TradingEnv(batch_data)
        
        steps = len(batch_data)
        for i in range(steps):
            inputs = env.get_state(i)
            action = net.activate(inputs) # RNN keeps internal state
            env.step(i, action)
            
        final_equity = env.cash + (env.holdings * batch_data.iloc[-1]["Close"])
        pnl_pct = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
        
        pnl_relative = pnl_pct - bh_return_pct
        max_dd = env.max_drawdown
        num_trades = len(env.trades)
        
        # Average Duration (in minutes)
        if len(env.closed_trade_durations) > 0:
            avg_duration = sum(env.closed_trade_durations) / len(env.closed_trade_durations)
        else:
            avg_duration = steps # Penalize holding forever if no closes
        
        # --- FITNESS CALCULATION (Option 3 from Paper) ---
        # Formula: PnL + 1.5*RelPnL - 0.5*DD + Reward*Trades - Penalty*Duration
        
        # We scale inputs to make them roughly comparable
        # PnL of 10% becomes 10.0
        f_pnl = pnl_pct * 100
        f_rel = pnl_relative * 100
        f_dd = max_dd * 100
        
        # Small reward for being active (0.01 per trade)
        f_trades = 0.01 * num_trades 
        
        # Penalty for holding too long (Swing trading goal: < 4 hours? or < 1 day?)
        # 1 day = 1440 mins. If avg duration is 1440, penalty is small. 
        # Paper implies simple subtraction, we scale it.
        f_duration = avg_duration / 1000.0 
        
        fitness = f_pnl + (1.5 * f_rel) - (0.5 * f_dd) + f_trades - f_duration
                  
        # Bankrupt Penalty
        if final_equity < INITIAL_CAPITAL * 0.1:
            fitness -= 100 
            
        genome.fitness = fitness

# ==========================================
# 6. CONFIG GENERATION
# ==========================================

def create_config_file():
    # MANDATORY: Added 'no_fitness_termination' for newer NEAT versions
    # MANDATORY: Added 'single_structural_mutation' & 'structural_mutation_surer'
    # MANDATORY: Added 'enabled_rate_to_false_add' & 'enabled_rate_to_true_add'
    # MANDATORY: Explicit initialization types (gaussian)
    
    config_content = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 10000
pop_size              = 50
reset_on_extinction   = False
no_fitness_termination = False

[DefaultGenome]
# Node activation options
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh clamped relu sigmoid

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# Structural Mutation (Recurrent allowed)
feed_forward            = False

# Network Size
# Inputs: Long%, Short%, SMA5, SMA10, SlowK, SlowD, WillR, MACD, CCI, RSI, ADOSC
num_inputs              = 11
# Outputs: Buy, Sell, Volume
num_outputs             = 3
num_hidden              = 0

# Connection Initialization
initial_connection      = full_direct
conn_add_prob           = 0.5
conn_delete_prob        = 0.2

# Node Initialization
node_add_prob           = 0.2
node_delete_prob        = 0.2

# Structural Mutation Rules (Added for v0.92+)
single_structural_mutation = False
structural_mutation_surer  = default

# Parameters (Weights/Bias)
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_init_type          = gaussian
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_init_type        = gaussian
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

enabled_default         = True
enabled_mutate_rate     = 0.01
enabled_rate_to_false_add = 0.0
enabled_rate_to_true_add  = 0.0

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_init_type      = gaussian
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
    """
    with open('config-neat.txt', 'w') as f:
        f.write(config_content)
    return 'config-neat.txt'

# ==========================================
# 7. FINAL TEST & PLOT
# ==========================================

def run_final_test(winner_genome, config, test_data):
    print("\n" + "="*50)
    print("STARTING FINAL TEST ON UNSEEN DATA")
    print("="*50)
    
    net = neat.nn.RecurrentNetwork.create(winner_genome, config)
    env = TradingEnv(test_data)
    
    # Store history for plots
    history_equity = []
    history_cash = []
    history_holdings = []
    history_btc_price = []
    
    print("Running simulation forward...")
    for i in tqdm(range(len(test_data))):
        inputs = env.get_state(i)
        action = net.activate(inputs)
        eq = env.step(i, action)
        
        history_equity.append(eq)
        history_cash.append(env.cash)
        history_holdings.append(env.holdings)
        history_btc_price.append(test_data.iloc[i]["Close"])
        
    # --- Final Metrics ---
    final_pnl = history_equity[-1] - INITIAL_CAPITAL
    roi = (final_pnl / INITIAL_CAPITAL) * 100
    bh_roi = ((history_btc_price[-1] - history_btc_price[0]) / history_btc_price[0]) * 100
    
    print(f"\nFINAL TEST RESULTS:")
    print(f"ROI: {roi:.2f}% (Buy&Hold: {bh_roi:.2f}%)")
    print(f"Trades: {len(env.trades)}")
    print(f"Max DD: {env.max_drawdown*100:.2f}%")
        
    # --- PLOTTING ---
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=("Price & Trades", "Equity vs Buy & Hold", "Holdings (BTC)", "Cash ($)")
    )
    
    # 1. Price
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data["Close"], name="BTC Price", line=dict(color='rgb(50,50,50)')), row=1, col=1)
    
    buys = [t for t in env.trades if t['type'] == 'buy']
    sells_win = [t for t in env.trades if t['type'] == 'sell' and t['result'] == 'win']
    sells_loss = [t for t in env.trades if t['type'] == 'sell' and t['result'] == 'loss']
    
    if buys:
        fig.add_trace(go.Scatter(x=[t['time'] for t in buys], y=[t['price'] for t in buys], 
                                 mode='markers', marker=dict(symbol='triangle-up', color='lime', size=8), name='Buy'), row=1, col=1)
    if sells_win:
        fig.add_trace(go.Scatter(x=[t['time'] for t in sells_win], y=[t['price'] for t in sells_win], 
                                 mode='markers', marker=dict(symbol='triangle-down', color='cyan', size=8), name='Sell Win'), row=1, col=1)
    if sells_loss:
        fig.add_trace(go.Scatter(x=[t['time'] for t in sells_loss], y=[t['price'] for t in sells_loss], 
                                 mode='markers', marker=dict(symbol='triangle-down', color='magenta', size=8), name='Sell Loss'), row=1, col=1)

    # 2. Equity
    fig.add_trace(go.Scatter(x=test_data.index, y=history_equity, name="Strategy Equity", line=dict(color='purple', width=2)), row=2, col=1)
    
    bh_curve = (test_data["Close"] / test_data.iloc[0]["Close"]) * INITIAL_CAPITAL
    fig.add_trace(go.Scatter(x=test_data.index, y=bh_curve, name="Buy & Hold", line=dict(color='orange', dash='dot')), row=2, col=1)
    
    # 3. Holdings
    fig.add_trace(go.Scatter(x=test_data.index, y=history_holdings, name="Holdings", line=dict(color='cyan'), fill='tozeroy'), row=3, col=1)
    
    # 4. Cash
    fig.add_trace(go.Scatter(x=test_data.index, y=history_cash, name="Cash", line=dict(color='green')), row=4, col=1)
    
    fig.update_layout(title="Final Test Results: NEAT Trading Bot", template="plotly_dark", height=1200)
    fig.write_html("final_test_results.html")
    print("\nSaved 'final_test_results.html' - Open this file to view the interactive plots.")

# ==========================================
# 8. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    train, test = load_and_process_data()
    
    if train is not None:
        TRAIN_DATA = train
        TEST_DATA = test
        
        config_path = create_config_file()
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        p = neat.Population(config)
        
        # Add Standard Reporter
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.StatisticsReporter())
        
        # Add Custom Reporter (Validation on fixed subset)
        # Use the last chunk of training data as a fixed validation set for consistency
        val_slice = TRAIN_DATA.iloc[-EPISODE_STEPS:] 
        p.add_reporter(PeriodicReporter(val_slice))
        
        print(f"\nStarting Training for {GENERATIONS} generations...")
        print("Model will perform 'Progressive Random Batching' to avoid overfitting.")
        
        winner = p.run(eval_genomes, GENERATIONS)
        
        # Save Model
        with open("best_neat_model.pkl", "wb") as f:
            pickle.dump(winner, f)
        print("Best model saved to 'best_neat_model.pkl'")
            
        # Run Final Test
        run_final_test(winner, config, test)
