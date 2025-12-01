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

# Filter warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_URL = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"

# Training Settings
GENERATIONS = 100
POP_SIZE = 150
EPISODE_STEPS = 20160  # ~14 Days (2 weeks) to capture full market cycles
INITIAL_CAPITAL = 10000.0
TRANSACTION_FEE = 0.001 

# Trading Logic
MIN_TRADE_INTERVAL = 15 # Minimum 15 minutes between trades
DECISION_THRESHOLD = 0.6 

# Reward Weights (Srivastava et al. adapted)
W_RETURN  = 1.0
W_RISK    = 0.5  # Lower risk penalty to encourage trading
W_DIFF    = 3.0  # Massive bonus for beating Buy & Hold
W_TREYNOR = 1.0

# ==========================================
# 2. DATA PROCESSING
# ==========================================

def load_data():
    print("Downloading and processing data...")
    cols = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
    try:
        df = pd.read_csv(DATA_URL, skiprows=[1, 2], header=0, names=cols,
                         parse_dates=["Datetime"], index_col="Datetime",
                         dtype={"Volume": "float64"}, na_values=["NA", "N/A", ""],
                         keep_default_na=True)
    except Exception as e:
        print(f"Error: {e}")
        return None, None

    df = df.sort_index().dropna()

    # --- Feature Engineering ---
    
    # 1. Resample to 1 Hour (Macro Trend)
    df_1h = df.resample('1H').agg({'Close': 'last', 'High': 'max', 'Low': 'min'}).dropna()
    
    # 1H Trend (SMA 50 vs SMA 200)
    df_1h['sma50'] = ta.trend.SMAIndicator(df_1h['Close'], 50).sma_indicator()
    df_1h['sma200'] = ta.trend.SMAIndicator(df_1h['Close'], 200).sma_indicator()
    df_1h['trend_1h'] = (df_1h['sma50'] - df_1h['sma200']) / df_1h['sma200']
    
    # 1H RSI
    df_1h['rsi_1h'] = ta.momentum.RSIIndicator(df_1h['Close'], 14).rsi() / 100.0

    # 2. Resample to 15 Min (Intermediate)
    df_15m = df.resample('15T').agg({'Close': 'last'}).dropna()
    df_15m['rsi_15m'] = ta.momentum.RSIIndicator(df_15m['Close'], 14).rsi() / 100.0
    
    # Map back to 1m
    df = df.join(df_1h[['trend_1h', 'rsi_1h']].reindex(df.index, method='ffill'))
    df = df.join(df_15m[['rsi_15m']].reindex(df.index, method='ffill'))
    
    # 3. Micro Features (1m)
    # Volatility (Bollinger Width)
    bb = ta.volatility.BollingerBands(df['Close'], window=20)
    df['bb_width'] = bb.bollinger_wband()
    
    # ROC (Momentum)
    df['roc'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc()

    df.dropna(inplace=True)
    df.fillna(0, inplace=True)
    
    split = int(len(df) * 0.8)
    return df.iloc[:split], df.iloc[split:]

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
        self.entry_price = 0.0
        
        self.closes = self.data["Close"].values
        self.dates = self.data.index
        
        # Inputs: [Trend_1H, RSI_1H, RSI_15M, ROC_1M, BB_Width, Unrealized_PnL]
        self.feats = self.data[["trend_1h", "rsi_1h", "rsi_15m", "roc", "bb_width"]].values

    def get_state(self, step):
        price = self.closes[step]
        
        # Calculate Unrealized PnL % (Crucial for Profit Taking)
        if self.holdings > 0:
            unrealized_pnl = (price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = 0.0
            
        # Is Invested? (Binary 1.0 or -1.0)
        is_invested = 1.0 if self.holdings > 0 else -1.0
        
        # Market Feats
        mkt = self.feats[step]
        
        # Combine [Invested, Unr_PnL, 5 Market Indicators] = 7 Inputs
        state = np.hstack(([is_invested, unrealized_pnl], mkt))
        
        # Clean inputs
        return np.nan_to_num(np.clip(state, -5.0, 5.0))

    def step(self, step, action):
        # Action: [Buy, Sell] (Probabilities)
        buy_prob, sell_prob = action
        price = self.closes[step]
        
        # Calculate Equity
        equity = self.cash + (self.holdings * price)
        
        # Update Drawdown
        if equity > self.peak_equity: self.peak_equity = equity
        dd = (self.peak_equity - equity) / self.peak_equity
        if dd > self.max_drawdown: self.max_drawdown = dd
        
        # Interval Constraint
        if step - self.last_trade_step < MIN_TRADE_INTERVAL:
            return equity
            
        # --- LOGIC: All-in / All-out ---
        
        # Buy Signal
        if buy_prob > DECISION_THRESHOLD and buy_prob > sell_prob:
            if self.cash > 10: # If we have cash, go All-In
                cost = self.cash
                fee = cost * TRANSACTION_FEE
                self.holdings = (cost - fee) / price
                self.cash = 0.0
                self.entry_price = price
                self.last_trade_step = step
                self.trades.append({'step':step, 'type':'buy', 'price':price, 'time':self.dates[step]})
                
        # Sell Signal
        elif sell_prob > DECISION_THRESHOLD and sell_prob > buy_prob:
            if self.holdings > 0: # If we have stock, Sell All
                val = self.holdings * price
                fee = val * TRANSACTION_FEE
                self.cash = val - fee
                self.holdings = 0.0
                self.entry_price = 0.0
                self.last_trade_step = step
                self.trades.append({'step':step, 'type':'sell', 'price':price, 'time':self.dates[step]})
                
        return self.cash + (self.holdings * price)

# ==========================================
# 4. REWARD FUNCTION
# ==========================================

def calculate_fitness(env, equity_curve, market_prices):
    eq = np.array(equity_curve)
    
    # 1. Total Return
    total_ret = (eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # 2. Downside Risk (Only bad volatility)
    rets = np.diff(eq) / eq[:-1]
    neg_rets = rets[rets < 0]
    sigma_down = np.std(neg_rets) * np.sqrt(len(eq)) if len(neg_rets) > 0 else 0.0
    
    # 3. Market Beta
    mkt_rets = np.diff(market_prices) / market_prices[:-1]
    min_len = min(len(rets), len(mkt_rets))
    
    if min_len > 10 and np.var(mkt_rets[:min_len]) > 1e-9:
        cov = np.cov(rets[:min_len], mkt_rets[:min_len])[0,1]
        beta = cov / np.var(mkt_rets[:min_len])
    else:
        beta = 1.0
    beta = max(0.1, abs(beta)) # Avoid zero division
    
    # 4. Metrics
    market_ret = (market_prices[-1] - market_prices[0]) / market_prices[0]
    
    # Differential Return: (Strategy - Market) / Beta
    # "Did I beat the market given the risk I took?"
    diff_ret = (total_ret - market_ret) / beta
    
    # Treynor: Return / Beta
    treynor = total_ret / beta
    
    # --- Composite Score ---
    score = (W_RETURN * total_ret * 100) \
          - (W_RISK * sigma_down * 100) \
          + (W_DIFF * diff_ret * 100) \
          + (W_TREYNOR * treynor * 100)
          
    # --- Constraints ---
    
    # 1. Penalize Inactivity (Must make at least 2 trades in 2 weeks)
    if len(env.trades) < 2:
        score -= 50
        
    # 2. Penalize Churning (More than 40 trades in 2 weeks is likely scalping, which fails with fees)
    if len(env.trades) > 40:
        score -= (len(env.trades) - 40)
        
    # 3. Bankruptcy
    if eq[-1] < INITIAL_CAPITAL * 0.6:
        score -= 500
        
    return score

def eval_genomes(genomes, config):
    # Train on random 2-week episodes
    data_len = len(TRAIN_DATA)
    start = random.randint(0, data_len - EPISODE_STEPS - 1)
    batch = TRAIN_DATA.iloc[start : start+EPISODE_STEPS]
    mkt_prices = batch["Close"].values
    
    for _, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        env = TradingEnv(batch)
        
        eq_curve = []
        for i in range(len(batch)):
            inputs = env.get_state(i)
            action = net.activate(inputs)
            eq_curve.append(env.step(i, action))
            
        genome.fitness = calculate_fitness(env, eq_curve, mkt_prices)

# ==========================================
# 5. REPORTING
# ==========================================

class PeriodicReporter(neat.reporting.BaseReporter):
    def __init__(self, val_data):
        self.val_data = val_data
        self.gen = 0
    
    def start_generation(self, generation):
        self.gen = generation
        
    def post_evaluate(self, config, pop, species, best_genome):
        net = neat.nn.RecurrentNetwork.create(best_genome, config)
        env = TradingEnv(self.val_data)
        
        hist_eq = []
        hist_cash = []
        hist_holdings = []
        
        for i in range(len(self.val_data)):
            inputs = env.get_state(i)
            action = net.activate(inputs)
            eq = env.step(i, action)
            hist_eq.append(eq)
            hist_cash.append(env.cash)
            hist_holdings.append(env.holdings * env.closes[i])
            
        roi = (hist_eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        print(f"\n[{self.gen}] Best Genome Validation:")
        print(f"   > Equity:   ${hist_eq[-1]:,.2f} ({roi:+.2f}%)")
        print(f"   > Trades:   {len(env.trades)}")
        print(f"   > Max DD:   {env.max_drawdown*100:.2f}%")
        
        if self.gen % 5 == 0:
            self.plot(env, hist_eq, hist_cash, hist_holdings)
            
    def plot(self, env, eq, cash, holdings):
        df = self.val_data
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("Equity vs Buy&Hold", "Cash", "Holdings Value"))
        
        bh = (df["Close"] / df.iloc[0]["Close"]) * INITIAL_CAPITAL
        fig.add_trace(go.Scatter(x=df.index, y=bh, name="Buy & Hold", line=dict(color='gray', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=eq, name="Strategy", line=dict(color='purple')), row=1, col=1)
        
        # Trade Markers
        buys = [t for t in env.trades if t['type']=='buy']
        sells = [t for t in env.trades if t['type']=='sell']
        if buys:
            fig.add_trace(go.Scatter(x=[t['time'] for t in buys], y=[eq[t['step']] for t in buys],
                                     mode='markers', marker=dict(symbol='triangle-up', color='lime', size=8), name="Buy"), row=1, col=1)
        if sells:
            fig.add_trace(go.Scatter(x=[t['time'] for t in sells], y=[eq[t['step']] for t in sells],
                                     mode='markers', marker=dict(symbol='triangle-down', color='red', size=8), name="Sell"), row=1, col=1)
                                     
        fig.add_trace(go.Scatter(x=df.index, y=cash, name="Cash", line=dict(color='green'), fill='tozeroy'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=holdings, name="Holdings", line=dict(color='orange'), fill='tozeroy'), row=3, col=1)
        
        fig.update_layout(title=f"Generation {self.gen} Report", template="plotly_dark", height=1000)
        fig.write_html(f"Gen_{self.gen}_Report.html")
        print(f"   > Plot saved: Gen_{self.gen}_Report.html")

# ==========================================
# 6. CONFIG & MAIN
# ==========================================

def make_config():
    conf = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 100000
pop_size              = 150
reset_on_extinction   = False
no_fitness_termination = False

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh clamped relu sigmoid

aggregation_default     = sum
aggregation_mutate_rate = 0.1
aggregation_options     = sum

feed_forward            = False
num_inputs              = 7
num_outputs             = 2
num_hidden              = 1

initial_connection      = full_direct
conn_add_prob           = 0.5
conn_delete_prob        = 0.2
node_add_prob           = 0.2
node_delete_prob        = 0.2

single_structural_mutation = False
structural_mutation_surer  = default

bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_init_type = gaussian
bias_max_value = 10.0
bias_min_value = -10.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1

weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_init_type = gaussian
weight_max_value = 10.0
weight_min_value = -10.0
weight_mutate_power = 0.5
weight_mutate_rate = 0.7
weight_replace_rate = 0.1

enabled_default = True
enabled_mutate_rate = 0.05
enabled_rate_to_false_add = 0.0
enabled_rate_to_true_add = 0.0
response_init_mean = 1.0
response_init_stdev = 0.0
response_init_type = gaussian
response_max_value = 10.0
response_min_value = -10.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 3

[DefaultReproduction]
elitism            = 3
survival_threshold = 0.3
min_species_size   = 2
    """
    with open('config-neat.txt', 'w') as f: f.write(conf)
    return 'config-neat.txt'

if __name__ == "__main__":
    train, test = load_data()
    if train is not None:
        TRAIN_DATA = train
        
        # Validate on a stable 2-week period
        val_slice = train.iloc[-EPISODE_STEPS:]
        
        cfg = make_config()
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg)
        
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(PeriodicReporter(val_slice))
        
        print("Training Simplified Swing Trader...")
        winner = p.run(eval_genomes, GENERATIONS)
        
        with open("winner.pkl", "wb") as f:
            pickle.dump(winner, f)

        # Final Test
        print("Running Final Test...")
        net = neat.nn.RecurrentNetwork.create(winner, config)
        env = TradingEnv(test)
        
        hist_eq, hist_cash, hist_holdings = [], [], []
        for i in tqdm(range(len(test))):
            inputs = env.get_state(i)
            action = net.activate(inputs)
            eq = env.step(i, action)
            hist_eq.append(eq)
            hist_cash.append(env.cash)
            hist_holdings.append(env.holdings * env.closes[i])
            
        reporter = PeriodicReporter(test)
        reporter.plot(env, hist_eq, hist_cash, hist_holdings)
        print("Done.")
