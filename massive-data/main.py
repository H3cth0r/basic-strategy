import os
import time
import requests
import argparse
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- Configuration ---
# Load variables from the .venv file
load_dotenv('.venv')
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found. Please ensure it is set in your .venv file.")

BASE_URL = "https://api.massive.com"

# 2 Years of data ending today (March 23, 2026)
END_DATE = datetime(2026, 3, 23)
START_DATE = END_DATE - timedelta(days=2*365)

def get_1min_data(ticker, start_date, end_date):
    """Fetches 1-minute aggregate data for a specific date range."""
    endpoint = f"/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}"
    url = f"{BASE_URL}{endpoint}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("results", [])
    else:
        print(f"Error fetching {ticker}: {response.status_code} - {response.text}")
        return []

def download_and_save_crypto_data(ticker):
    print(f"--- Starting download for {ticker} ---")
    all_results = []
    
    current_start = START_DATE
    api_call_count = 0  # Counter to track our API calls
    
    # Chunking the requests by 30-day windows
    while current_start < END_DATE:
        # Check if we need to sleep to respect the 5 calls/minute limit
        if api_call_count > 0 and api_call_count % 5 == 0:
            print("Rate limit buffer reached (5 calls). Waiting 65 seconds to respect API limits...")
            time.sleep(65)
        
        current_end = min(current_start + timedelta(days=30), END_DATE)
        
        str_start = current_start.strftime("%Y-%m-%d")
        str_end = current_end.strftime("%Y-%m-%d")
        
        print(f"Fetching {ticker} from {str_start} to {str_end}... (Call {api_call_count + 1})")
        
        data_chunk = get_1min_data(ticker, str_start, str_end)
        api_call_count += 1  # Increment the counter after every API call
        
        if data_chunk:
            all_results.extend(data_chunk)
            
        current_start = current_end + timedelta(days=1)
        
    # Convert to DataFrame and save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Convert the standard Unix timestamp (milliseconds) to readable datetime
        if 't' in df.columns:
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            
        # Rename columns to standard OHLCV
        df = df.rename(columns={
            'v': 'volume', 
            'vw': 'vwap', 
            'o': 'open', 
            'c': 'close', 
            'h': 'high', 
            'l': 'low', 
            't': 'unix_ms', 
            'n': 'transactions'
        })
        
        # Clean up filename
        filename = f"datavolume.{ticker.replace(':', '_')}_1min_2years.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} rows to {filename}\n")
    else:
        print(f"No data found for {ticker}.\n")

if __name__ == "__main__":
    # Setup argparse to take the ticker as a command-line argument
    parser = argparse.ArgumentParser(description="Download 2 years of 1-minute crypto data.")
    parser.add_argument(
        "ticker", 
        type=str, 
        help="The ticker symbol to download (e.g., X:BTCUSD)"
    )
    
    args = parser.parse_args()
    
    # Run the function with the passed argument
    download_and_save_crypto_data(args.ticker)