import yfinance as yf
import pandas as pd
import os
import yaml

class StockDataFetcher:
    def __init__(self, config_path="config.yaml"):
        # Load configuration from YAML file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        self.start_date = config.get("start_date", "2005-01-01")
        self.interval = config.get("interval", "1d")
        self.output_dir = config.get("output_dir", "stock_data")
        self.stocks = config.get("stocks", [])
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fetch_stock_data(self, ticker):
        print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        data = stock.history(period="max", interval=self.interval, start=self.start_date)
        
        if data.empty:
            print(f"No data retrieved for {ticker}. Check Yahoo Finance limitations.")
            return None
        
        data.reset_index(inplace=True)
        data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        data.rename(columns={"Date": "timestamp"}, inplace=True)
        
        return data
    
    def save_to_csv(self, ticker, data):
        if data is not None:
            filepath = os.path.join(self.output_dir, f"{ticker}_daily.csv")
            data.to_csv(filepath, index=False)
            print(f"Saved {ticker} data to {filepath}")
    
    def fetch_and_save_all(self):
        dataset = {}
        for ticker in self.stocks:
            data = self.fetch_stock_data(ticker)
            if data is not None:
                dataset[ticker] = data
                self.save_to_csv(ticker, data)
        return dataset