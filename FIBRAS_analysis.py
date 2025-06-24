import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    return np, pd, plt, sns, yf


@app.cell
def _(time, yf):
    def get_fibra_data_robust(tickers, start_date, end_date):
        """
        Downloads historical stock data using the more robust yf.Ticker object method.
        This is the recommended way to avoid rate-limiting and state-related errors.

        Args:
            tickers (list): A list of stock ticker symbols.
            start_date (str): The start date for the data in 'YYYY-MM-DD' format.
            end_date (str): The end date for the data in 'YYYY-MM-DD' format.

        Returns:
            dict: A dictionary where keys are tickers and values are pandas DataFrames.
        """
        fibra_data = {}
        for ticker in tickers:
            try:
                print(f"Downloading data for {ticker}...")
            
                # Create a Ticker object for the specific stock
                ticker_obj = yf.Ticker(ticker)
            
                # Fetch the history for that Ticker object
                # auto_adjust=True by default, which is good. It adjusts for splits.
                # We will use back_adjust=True to also include dividends in the price,
                # but we also need the 'Dividends' column, so we'll download it separately.
            
                data = ticker_obj.history(
                    start=start_date, 
                    end=end_date, 
                    interval="1d",
                    auto_adjust=False # IMPORTANT: Set to False to get the 'Dividends' column
                )
            
                if data.empty:
                    print(f"  -> No data found for {ticker}. It might be delisted, have no data for the period, or be an incorrect ticker.")
                    continue
            
                fibra_data[ticker] = data
                print(f"  -> Success: Downloaded {len(data)} rows of data for {ticker}.")
            
                # Be polite to the API: wait before the next request
                time.sleep(1) 
            
            except Exception as e:
                print(f"  -> An error occurred for {ticker}: {e}")

        return fibra_data
    return (get_fibra_data_robust,)


@app.function
def calculate_total_return(df):
    """
    Calculates daily and cumulative total returns, including dividends.
    The total return is the sum of capital gain and dividend yield.

    Args:
        df (pd.DataFrame): DataFrame with historical data for one ticker.
                           Must contain 'Close' and 'Dividends' columns.

    Returns:
        pd.DataFrame: The original DataFrame with new columns for returns.
    """
    # To correctly calculate return with dividends, we use the prior day's close price.
    # Dividend Yield = Dividend / Price_before_dividend_payout
    df['Dividend Yield'] = df['Dividends'] / df['Close'].shift(1)
    
    # Price Return (Capital Gain)
    df['Price Return'] = df['Close'].pct_change()
    
    # Total Daily Return is the sum of price return and dividend yield.
    df['Total Daily Return'] = df['Price Return'] + df['Dividend Yield'].fillna(0)
    
    # Cumulative Total Return shows the growth of an initial investment of $1.
    df['Cumulative Total Return'] = (1 + df['Total Daily Return']).cumprod()
    
    return df


@app.cell
def _(np):
    def calculate_annualized_metrics(df):
        """
        Calculates annualized return and volatility from daily returns.

        Args:
            df (pd.DataFrame): DataFrame containing a 'Total Daily Return' column.

        Returns:
            tuple: A tuple containing (annualized_return, annualized_volatility).
        """
        daily_returns = df['Total Daily Return']
    
        # Trading days in a year is typically ~252
        trading_days = 252
    
        # Annualized Return (Geometric Mean)
        total_days = len(daily_returns.dropna())
        # The geometric mean is more accurate for investment returns over time
        annualized_return = (df['Cumulative Total Return'].iloc[-1]) ** (trading_days / total_days) - 1

        # Annualized Volatility (Standard Deviation)
        annualized_volatility = daily_returns.std() * np.sqrt(trading_days)
    
        return annualized_return, annualized_volatility
    return (calculate_annualized_metrics,)


@app.function
def get_monthly_and_yearly_returns(df):
    """
    Calculates the total monthly and yearly returns from daily return data.

    Args:
        df (pd.DataFrame): DataFrame with 'Total Daily Return' column.

    Returns:
        tuple: (monthly_returns_df, yearly_returns_df)
    """
    daily_returns = df['Total Daily Return']
    
    # Resample to get monthly returns by compounding daily returns
    monthly_returns = (1 + daily_returns).resample('M').prod() - 1
    
    # Resample to get yearly returns
    yearly_returns = (1 + daily_returns).resample('Y').prod() - 1

    return monthly_returns.to_frame(name="Return"), yearly_returns.to_frame(name="Return")


@app.cell
def _(plt, sns):
    def plot_cumulative_returns(cumulative_returns_df):
        """
        Plots the cumulative total returns for multiple FIBRAs.

        Args:
            cumulative_returns_df (pd.DataFrame): A DataFrame where each column is the
                                                  cumulative return series for a FIBRA.
        """
        plt.style.use('seaborn-v0_8-grid')
        fig, ax = plt.subplots(figsize=(14, 7))
    
        cumulative_returns_df.plot(ax=ax)
    
        ax.set_title('Growth of $1 Investment (Total Return)', fontsize=16)
        ax.set_ylabel('Cumulative Return')
        ax.set_xlabel('Date')
        ax.legend(title='FIBRA Ticker')
        plt.grid(True)
        plt.show()

    def plot_yearly_returns_heatmap(all_yearly_returns):
        """
        Creates a heatmap of yearly returns for direct comparison.

        Args:
            all_yearly_returns (pd.DataFrame): DataFrame with years as index, tickers as columns,
                                               and yearly returns as values.
        """
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            all_yearly_returns.T,          # Transpose to have tickers on Y-axis
            annot=True,                    # Show the return values
            fmt='.2%',                     # Format as percentage with 2 decimals
            cmap='RdYlGn',                 # Red-Yellow-Green colormap
            linewidths=.5
        )
        plt.title('Yearly Total Returns by FIBRA', fontsize=16)
        plt.xlabel('Year')
        plt.ylabel('FIBRA Ticker')
        plt.show()
    return plot_cumulative_returns, plot_yearly_returns_heatmap


@app.cell
def _(
    calculate_annualized_metrics,
    get_fibra_data_robust,
    pd,
    plot_cumulative_returns,
    plot_yearly_returns_heatmap,
):
    # Define the list of Mexican FIBRA tickers. 
    FIBRA_TICKERS = [
        'FIBRAMQ.MX',  # Fibra Macquarie México
        'FUNO11.MX',   # Fibra Uno
        'DANHOS13.MX', # Fibra Danhos
        'FMTY14.MX',   # Fibra MTY
        'TERRA13.MX'   # Fibra Terrafina
    ]

    START_DATE = '2018-01-01'
    END_DATE = pd.to_datetime('today').strftime('%Y-%m-%d')

    # Step 1: Get the data using the NEW, ROBUST function
    # THIS IS THE ONLY LINE THAT CHANGES IN THE MAIN BLOCK
    fibra_data_dict = get_fibra_data_robust(FIBRA_TICKERS, START_DATE, END_DATE)

    # --- The rest of the main block is identical to before ---

    # Prepare DataFrames to hold combined results for plotting
    all_cumulative_returns = pd.DataFrame()
    all_yearly_returns_list = []

    # Step 2: Process each FIBRA
    for ticker, df in fibra_data_dict.items():
        if df.empty:
            print(f"\nSkipping {ticker}, data is empty.")
            continue
        
        print(f"\n--- Analyzing: {ticker} ---")
    
        # Calculate returns
        df_processed = calculate_total_return(df)
    
        # Calculate annualized metrics
        ann_return, ann_volatility = calculate_annualized_metrics(df_processed)
        print(f"Annualized Return: {ann_return:.2%}")
        print(f"Annualized Volatility (Risk): {ann_volatility:.2%}")
    
        # Get monthly and yearly returns
        monthly_ret, yearly_ret = get_monthly_and_yearly_returns(df_processed)
        print("\nYearly Returns:")
        print((yearly_ret * 100).round(2).to_string(formatters={'Return': '{:,.2f}%'.format}))
    
        # Store results for combined plots
        all_cumulative_returns[ticker] = df_processed['Cumulative Total Return']
    
        yearly_ret['Ticker'] = ticker
        yearly_ret.index = yearly_ret.index.year 
        all_yearly_returns_list.append(yearly_ret)

    # Step 3: Generate comparison plots
    if not all_cumulative_returns.empty:
        plot_cumulative_returns(all_cumulative_returns.dropna())

        yearly_returns_df = pd.concat(all_yearly_returns_list)
        yearly_returns_pivot = yearly_returns_df.pivot_table(index=yearly_returns_df.index, columns='Ticker', values='Return')
        plot_yearly_returns_heatmap(yearly_returns_pivot)
    else:
        print("\nNo data was successfully processed. Unable to generate plots.")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
