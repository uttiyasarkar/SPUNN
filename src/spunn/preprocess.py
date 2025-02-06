import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockDataPreprocessor:
    def __init__(self, input_dir="stock_data", output_dir="processed_data", window_size=30):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self, ticker):
        filepath = os.path.join(self.input_dir, f"{ticker}_daily.csv")
        if os.path.exists(filepath):
            return pd.read_csv(filepath, parse_dates=["timestamp"])
        else:
            print(f"File {filepath} not found.")
            return None
    
    def preprocess(self, ticker):
        df = self.load_data(ticker)
        if df is None:
            return None
        
        # Handle missing values
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Normalize prices and volume
        df[["Open", "High", "Low", "Close", "Volume"]] = self.scaler.fit_transform(df[["Open", "High", "Low", "Close", "Volume"]])
        
        # Create sequences
        sequences = []
        for i in range(len(df) - self.window_size):
            seq = df.iloc[i:i + self.window_size].drop(columns=["timestamp"]).values
            sequences.append(seq)
        
        return np.array(sequences)
    
    def process_and_save_all(self, tickers):
        for ticker in tickers:
            sequences = self.preprocess(ticker)
            if sequences is not None:
                output_path = os.path.join(self.output_dir, f"{ticker}_processed.parquet")
                df = pd.DataFrame(sequences.reshape(sequences.shape[0], -1))
                df.to_parquet(output_path, engine="pyarrow", index=False)
                print(f"Saved processed data for {ticker} to {output_path}")
