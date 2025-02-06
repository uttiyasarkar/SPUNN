import os
import yaml
from normalize import StockDataFetcher
from preprocess import StockDataPreprocessor
from train import StockDataset

if __name__ == "__main__":
    # Load configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    stocks = config.get("stocks", [])
    
    # Fetch and save stock data
    fetcher = StockDataFetcher("config.yaml")
    dataset = fetcher.fetch_and_save_all()
    
    # Preprocess stock data
    preprocessor = StockDataPreprocessor()
    processed_data = preprocessor.process_and_save_all(stocks)

    # Train model
    StockDataset(processed_data)