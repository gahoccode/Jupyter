# Vietnamese Stock Portfolio Optimization

A Python application for fetching, analyzing, and optimizing Vietnamese stock portfolios using vnstock, PyPortfolioOpt, and pyfolio-reloaded.

## Features

- Fetch historical stock data from Vietnamese markets
- Combine OHLC data from multiple symbols
- Extract and process close prices for portfolio optimization
- Export data to CSV files for further analysis
- Portfolio optimization capabilities

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   ```
   # On Windows
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main application:

```python
python app.py
```

This will:
1. Fetch historical data for the specified stock symbols
2. Combine the data into unified datasets
3. Export the data to CSV files for further analysis

## Data Files

- `all_historical_data.csv`: Combined OHLC data for all symbols
- `combined_close_prices.csv`: Combined close prices for all symbols

## Notes

This project uses vnstock 3.x API for fetching Vietnamese stock data. Make sure to use compatible versions of dependencies as specified in the requirements.txt file.
