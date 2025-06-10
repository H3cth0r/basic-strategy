import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.float_format = "{:,.10f}".format

# Correct: skip 3 rows (Ticker, column labels, and "Datetime" row)
df = pd.read_csv(
    "/home/h3cth0r/Documents/stonks-data/data/CRYPTO/BTC-USD/data_0.csv",
    skiprows=3,
    names=["Datetime", "Close", "High", "Low", "Open", "Volume"]
)

# Parse Datetime column
df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S%z")

# Set Datetime as index
df.set_index("Datetime", inplace=True)

# Print first few rows to confirm
print(df.head())

# Plot the Close price
plt.figure(figsize=(10, 5))
plt.plot(df["Close"], label="Close Price (BTC-USD)", linestyle='-')
plt.title("BTC-USD Close Price Over Time")
plt.xlabel("Datetime")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
