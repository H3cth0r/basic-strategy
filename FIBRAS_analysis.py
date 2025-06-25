import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    return mo, pd, plt, sns, yf


@app.cell
def _(mo):
    mo.md(r"""## Configuration""")
    return


@app.cell
def _(pd):
    INITIAL_INVESTMENT_MXN = 100_000
    PROFIT_TAX_RATE = 0.1
    ANNUAL_BROKER_FEE_RATE = 0.0025
    RISK_FREE_RATE=0.1125

    FIBRA_TICKETS = [
        'FIBRAMQ12.MX', 
        'FUNO11.MX', 
        'DANHOS13.MX',
        'FMTY14.MX', 
        'TERRA13.MX', 
        'FIBRAUP18.MX', 
        'FINN13.MX' 
    ]

    START_DATE = '2019-01-01' 
    END_DATE = pd.to_datetime('today').strftime('%Y-%m-%d')
    NUM_PORTFOLIOS = 25000 
    return


@app.cell
def _(plt, sns):
    sns.set_style('whitegrid')
    plt.rc('figure', figsize=(12, 6))
    plt.rc('font', size=12)
    return


@app.cell
def _(mo):
    mo.md(r"""## Data Fetching and Processing""")
    return


@app.cell
def _(star_date, yf):
    def get_fibra_data(tickers, start_date, end_date):
        try:
            data = yf.download(
                tickers,
                start=star_date,
                end=end_date,
                auto_adjust=False,
                progress=True,
                actions=True
            )
            if data.empty:
                print("Error: No data downloaded. Check tickers or date range.")
                return None
            return data
        except Exception as e:
            print(f"An error occurred during download: {e}")
            return None
    return


if __name__ == "__main__":
    app.run()
