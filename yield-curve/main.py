import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ^IRX: 13 Week Treasury Bill
# ^FVX: 5 Year Treasury Note
# ^TNX: 10 Year Treasury Note
# ^TYX: 30 Year Treasury Bond
yield_tickers = {
    "3M": "^IRX",
    "5Y": "^FVX",
    "10Y": "^TNX",
    "30Y": "^TYX"
}

# Corresponding maturities in years
maturities_in_years = {
        "3M": 0.25,
        "5Y": 5.0,
        "10Y": 10.0,
        "30Y": 30.0,
}

yields = []
maturities = []

for term, ticker_symbol in yield_tickers.items():
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            # close price for these tickers is the yield
            latest_yield = hist['Close'].iloc[-1]
            yields.append(latest_yield)
            maturities.append(maturities_in_years[term])
            print(f"{term} ({ticker_symbol}): {latest_yield:.2f}%")
        else:
            print(f"Could not fetch data for {term} {ticker_symbol}")
    except Exception as e:
        print(f"An error occurred while fetching data for {ticker_symbol}: {e}")


if maturities and yields:
    sorted_indices = np.argsort(maturities)
    maturities = np.array(maturities)[sorted_indices]
    yields = np.array(yields)[sorted_indices]
else:
    print("Could not fetch any yield data. Using sample data instead.")
    maturities = np.array([0.25, 5, 10, 30])
    yields = np.array([5.2, 4.3, 4.2, 4.4])

print(sorted_indices)
print(maturities)
print(yields)

def yield_curve_function(maturity):
    return np.interp(maturity, maturities, yields)

def calculate_riemann_sum(start_maturity, end_maturity, num_rectangles):
    width = (end_maturity - start_maturity) / num_rectangles
    total_area = 0

    for i in range(num_rectangles):
        mid_point = start_maturity + (i + 0.5) * width
        height = yield_curve_function(mid_point)
        area = width * height
        total_area += area
    return total_area

integration_start = 5
integration_end = 10
number_of_rectangles = 10

total_return_approximation = calculate_riemann_sum(integration_start, integration_end, number_of_rectangles)

print(f"\nApproximate total return between {integration_start} and {integration_end} years: {total_return_approximation:.4f}")


plt.figure(figsize=(12, 7))

plt.plot(maturities, yields, 'o-', label='Real-time Yield Curve')

plot_maturities = np.linspace(min(maturities), max(maturities), 200)
plot_yields = yield_curve_function(plot_maturities)
plt.plot(plot_maturities, plot_yields, '-', color='royalblue', alpha=0.5)

bar_starts = np.linspace(integration_start, integration_end, number_of_rectangles, endpoint=False)
bar_width = (integration_end - integration_start) / number_of_rectangles
bar_heights = yield_curve_function(bar_starts + bar_width/2)

plt.bar(bar_starts, bar_heights, width=bar_width, align='edge', alpha=0.5, color='skyblue', label='Riemann Sum Rectangles')

plt.title('Real-time U.S. Treasury Yield Curve and Integral Approximation')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.grid(True)
plt.legend()
plt.show()
