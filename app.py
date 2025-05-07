from vnstock import Quote
import pandas as pd
from rich import print

def combine_data(all_historical_data):
    """
    Combine OHLC data from multiple symbols into a single DataFrame.
    
    Parameters:
    -----------
    all_historical_data : dict
        Dictionary containing historical data for each symbol.
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all OHLC data for all symbols.
    """
    combined_data = pd.DataFrame()
    
    for symbol, data in all_historical_data.items():
        if not data.empty:
            # Make a copy of the data and rename columns to include symbol
            temp_df = data.copy()
            # Keep 'time' column as is for merging
            for col in temp_df.columns:
                if col != 'time':
                    temp_df.rename(columns={col: f'{symbol}_{col}'}, inplace=True)
            
            if combined_data.empty:
                combined_data = temp_df
            else:
                combined_data = pd.merge(combined_data, temp_df, on='time', how='outer')
    
    # Sort by time
    if not combined_data.empty:
        combined_data = combined_data.sort_values('time')
        
        # Display sample of combined data
        print("\nSample of combined data:")
        print(combined_data.head(3))
    
    return combined_data

def combine_prices(all_historical_data):
    """
    Combine close prices from multiple symbols into a single DataFrame.
    
    Parameters:
    -----------
    all_historical_data : dict
        Dictionary containing historical data for each symbol.
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with close prices for all symbols.
    """
    combined_prices = pd.DataFrame()
    
    for symbol, data in all_historical_data.items():
        if not data.empty:
            # Extract time and close price
            temp_df = data[['time', 'close']].copy()
            temp_df.rename(columns={'close': f'{symbol}_close'}, inplace=True)
            
            if combined_prices.empty:
                combined_prices = temp_df
            else:
                combined_prices = pd.merge(combined_prices, temp_df, on='time', how='outer')
    
    # Sort by time
    if not combined_prices.empty:
        combined_prices = combined_prices.sort_values('time')
    
    return combined_prices

def export_csv(dataframe, filename):
    """
    Export a DataFrame to a CSV file.
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        DataFrame to export.
    filename : str
        Name of the CSV file to create.
        
    Returns:
    --------
    bool
        True if export was successful, False otherwise.
    """
    if dataframe is None or dataframe.empty:
        print(f"Cannot export empty DataFrame to {filename}")
        return False
    
    try:
        dataframe.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"Data successfully exported to {filename}")
        return True
    except Exception as e:
        print(f"Error exporting data to {filename}: {e}")
        return False

# Main execution
if __name__ == "__main__":
    # Define the symbols you want to fetch data for
    symbols = ['REE', 'FMC', 'DHC']
    print(f"Fetching historical price data for: {symbols}")

    # Dictionary to store historical data for each symbol
    all_historical_data = {}

    # Set date range
    start_date = '2024-01-01'
    end_date = '2025-03-19'
    interval = '1D'

    # Fetch historical data for each symbol
    for symbol in symbols:
        try:
            print(f"\nProcessing {symbol}...")
            quote = Quote(symbol=symbol)
            
            # Fetch historical price data
            historical_data = quote.history(
                start=start_date,
                end=end_date,
                interval=interval,
                to_df=True
            )
            
            if not historical_data.empty:
                all_historical_data[symbol] = historical_data
                print(f"Successfully fetched {len(historical_data)} records for {symbol}")
            else:
                print(f"No historical data available for {symbol}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    # Process data if available
    if all_historical_data:
        # Combine OHLC data
        combined_data = combine_data(all_historical_data)
        
        # Export combined OHLC data
        export_csv(combined_data, 'all_historical_data.csv')
        
        # Combine close prices
        combined_prices = combine_prices(all_historical_data)
        
        # Export combined close prices
        export_csv(combined_prices, 'combined_close_prices.csv')
    else:
        print("No historical data was fetched for any symbol.")