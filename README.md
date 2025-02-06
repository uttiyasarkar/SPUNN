# My Python Project

This project is designed for stock price prediction using machine learning techniques. It fetches stock data, preprocesses it, and utilizes a transformer model for predictions.

## Project Structure

```
my-python-project
├── src
│   ├── preprocess.py       # Contains DataPreprocessor class for data normalization and sequence creation
│   ├── normalize.py        # Contains DataFetcher class for fetching stock data from Yahoo Finance
│   ├── create_sequences.py  # Contains StockPredictor class for defining and training the prediction model
│   └── main.py             # Main script to orchestrate the workflow
├── requirements.txt        # Lists project dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd my-python-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Modify the parameters in `src/main.py` to specify the stock ticker and date range for prediction.
2. Run the main script:
   ```
   python src/main.py
   ```

## Dependencies

This project requires the following Python packages:
- numpy
- pandas
- yfinance
- torch
- sklearn
- matplotlib

## License

This project is licensed under the MIT License.