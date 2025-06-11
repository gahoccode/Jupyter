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

### Using UV (Recommended)

1. Clone this repository
2. Install UV if you haven't already:
   ```
   curl -fsSL https://astral.sh/uv/install.sh | bash
   ```
3. Sync dependencies using UV (no need to create a virtual environment):
   ```
   uv sync
   ```

Note: UV automatically manages isolated environments, so you don't need to create or activate a virtual environment manually.

### Using Traditional pip (Alternative)

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   ```
   # On macOS/Linux
   source .venv/bin/activate
   # On Windows
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running Jupyter Notebook/Lab

With UV:
```bash
uv run jupyter notebook
# or
uv run jupyter lab
```

With traditional virtual environment:
```bash
jupyter notebook
# or
jupyter lab
```

### Running the Application

This will:
1. Fetch historical data for the specified stock symbols
2. Combine the data into unified datasets
3. Export the data to CSV files for further analysis

## Data Files

- `all_historical_data.csv`: Combined OHLC data for all symbols
- `combined_close_prices.csv`: Combined close prices for all symbols

## Notes

- This project uses vnstock 3.x API for fetching Vietnamese stock data.
- Dependencies are managed through `pyproject.toml` and can be installed using UV.

- When using UV, all commands should be prefixed with `uv run` to ensure they use the correct environment.
