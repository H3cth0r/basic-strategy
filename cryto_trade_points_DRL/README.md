# Crypto Trading Points with Deep Reinforcement Learning Approach

Please write the python code generate the model described in this paper. If possible write it inside a simple file. Please make plotly plots and show them. Download the data from here:
    data_url = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"

    data_df_raw = load_raw_data(data_url)

def load_raw_data(url):
    print("Loading and preparing data...")
    df = pd.read_csv(url, skiprows=3, header=None)
    df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)

    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['Original_Close'] = df['Close'].astype(float)

    df['log_close'] = np.log(df['Original_Close'])
    df['ret_1m'] = df['log_close'].diff(1)

    for w in [5, 15, 60, 240]:
        df[f'ret_{w}m'] = df['log_close'].diff(w)
        df[f'vol_{w}m_raw'] = df['ret_1m'].rolling(w).std()

    df['RSI'] = ta.momentum.RSIIndicator(close=df['Original_Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Original_Close'])
    df['MACD'] = macd.macd_diff()

    df['ema_12'] = df['Original_Close'].ewm(span=12, adjust=False).mean()
    df['ema_48'] = df['Original_Close'].ewm(span=48, adjust=False).mean()
    df['ema_ratio'] = df['ema_12'] / (df['ema_48'] + 1e-8) - 1.0

    atr = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Original_Close'], window=14
    ).average_true_range()
    df['atr_raw'] = atr
    df['atr_pct_raw'] = df['atr_raw'] / (df['Original_Close'] + 1e-8)

    vol_mean = df['Volume'].rolling(240).mean()
    vol_std = df['Volume'].rolling(240).std()
    df['vol_z_240'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)

    _add_time_features(df)
    df.dropna(inplace=True)

    df = df.reset_index(drop=True)
    df.drop(columns=['Datetime'], inplace=True, errors='ignore')

    print("Data preparation complete.")
    return df.reset_index(drop=True)


I have about 3 months data. Please 
Write the code inside a block of code
