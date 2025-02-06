import yfinance as yf

# Download stock data
ticker = "AAPL"  # Example: Apple stock
data = yf.download(ticker, start="2010-01-01", end="2023-01-01")

# Save to CSV (optional)
data.to_csv(f"{ticker}_stock_data.csv")