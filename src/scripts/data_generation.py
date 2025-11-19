# Packages
# --------

import yfinance as yf

from pathlib import Path

# Download data
# -------------

dj_tickers = [
    "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DOW", "XOM",
    "GS", "HD", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE",
    "PFE", "PG", "RTX", "TRV", "UNH", "VZ", "V", "WMT", "DIS",  # "WBA"
]

data = yf.download(
    tickers=" ".join(dj_tickers),
    start="2023-01-01",   # 2 years
    end="2025-10-01",
)

# Keep only Close prices
df = data["Close"]

# Compute Returns
# ---------------

returns = df.pct_change().dropna()

# Transform CSV
# -------------

path_prices = Path(
    r"C:\Users\Iker\Desktop\Warwick\Asset Pricing & Risk\Group Project\asset_pricing_project_group_2\src\data\full_data2_dj30.csv"
)

path_returns = Path(
    r"C:\Users\Iker\Desktop\Warwick\Asset Pricing & Risk\Group Project\asset_pricing_project_group_2\src\data\returns2_dj30.csv"
)

df.to_csv(path_prices)
returns.to_csv(path_returns)

print("Prices saved:", path_prices)
print("Returns saved:", path_returns)
