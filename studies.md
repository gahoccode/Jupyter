```python

# # Loading stock prices with vnstock
```

```python


from vnstock import Quote
import pandas as pd

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

# Export all historical data to a single CSV file
if all_historical_data:
    # Create a combined DataFrame with all data
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
        
        # Export combined data to CSV
        combined_csv_filename = './outputs/all_historical_data.csv'
        combined_data.to_csv(combined_csv_filename, index=False, encoding='utf-8-sig')
        print(f"\nAll historical data exported to {combined_csv_filename}")
    
    # Also create a combined DataFrame for close prices only (for comparison purposes)
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
        
        # Export combined close prices to CSV
        combined_close_csv_filename = './outputs/combined_close_prices.csv'
        combined_prices.to_csv(combined_close_csv_filename, index=False, encoding='utf-8-sig')
        print(f"Combined close price data exported to {combined_close_csv_filename}")
else:
    print("No historical data was fetched for any symbol.")
```


```python


#print(combined_prices.head())
#print(combined_data.head())
```


```python


# Set the time column as index and ensure it's datetime format
combined_prices_indexed = combined_prices.copy()
combined_prices_indexed['time'] = pd.to_datetime(combined_prices_indexed['time'])
combined_prices_indexed.set_index('time', inplace=True)

# Calculate daily returns for each stock
returns_df = pd.DataFrame(index=combined_prices_indexed.index)
for symbol in symbols:
    column_name = f'{symbol}_close'
    returns_df[symbol] = combined_prices_indexed[column_name].pct_change()

# Drop the first row which will have NaN values due to pct_change()
returns_df = returns_df.dropna()
# Create an equal-weighted portfolio
portfolio_returns = returns_df.mean(axis=1)


# # Loading VNINDEX for benchmarking
```

```python


from vnstock import Vnstock
symbol='VCI'
source='VCI'
stock = Vnstock().stock(symbol=symbol, source=source)
stock.trading.price_board(['VNINDEX'])
vnindex_data=stock.quote.history(start=start_date, end=end_date)
print(vnindex_data.head())
```


```python


# Process VNINDEX data
vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
vnindex_data.set_index('time', inplace=True)
vnindex_data.sort_index(inplace=True)
    
# Calculate VNINDEX returns
benchmark_rets = vnindex_data['close'].pct_change().dropna()
    
# Align benchmark returns with portfolio returns (same dates)
benchmark_rets = benchmark_rets.reindex(portfolio_returns.index)
benchmark_rets = benchmark_rets.fillna(method='ffill')  # Forward fill any missing dates
```


```python


print(benchmark_rets.head())
```


```python


# 1. Make a copy of the combined_prices DataFrame
prices_df = combined_prices.copy()

# 2. Convert the 'time' column to datetime if it's not already
prices_df['time'] = pd.to_datetime(prices_df['time'])

# 3. Set the 'time' column as the index
prices_df.set_index('time', inplace=True)

# 4. Extract only the close price columns and rename them to just the symbol names
close_price_columns = [col for col in prices_df.columns if '_close' in col]
prices_df = prices_df[close_price_columns]
prices_df.columns = [col.replace('_close', '') for col in close_price_columns]

# 5. Make sure there are no NaN values
prices_df = prices_df.dropna()
print(prices_df.head())
```


```python


risk_free_rate=0.02
risk_aversion=1
```


```python


from pypfopt.expected_returns import returns_from_prices
log_returns=False
returns = returns_from_prices(prices_df, log_returns=log_returns)
returns.head()


# # Portfolio Optimization (Mean-Variance, Min-volatilty, Max Utility)

# ## Setting up returns and covariance variables
```

```python


from pypfopt import EfficientFrontier, risk_models, expected_returns, DiscreteAllocation
from pypfopt.exceptions import OptimizationError
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov #for covariance matrix, get more methods from risk_models
from pypfopt.efficient_frontier import EfficientFrontier


mu=mean_historical_return(prices_df, log_returns=log_returns ) #Optional: add log_returns=True
"""
For most portfolio optimization purposes, the default simple returns pct_change() are adequate, 
but logarithmic returns can provide more robust results in some cases, 
especially when dealing with volatile assets or longer time horizons.
"""
S=sample_cov(prices_df)


# ### Optional 
```

```python


"""
from pypfopt.risk_models import CovarianceShrinkage

# Assume 'prices' is a pandas DataFrame of historical asset prices
S = CovarianceShrinkage(prices_df).ledoit_wolf()
"""


# ## Create an instance of Efficient Frontier
```

```python


#Createa an instance 
ef=EfficientFrontier(mu,S,weight_bounds=(0, 1)) # Adding weight_bounds is optional, for clarifying short positions, 0 and 1 means weights will be positive


# ## Generate the EF with simulated portfolios and 3 optimized portfolios
```

```python


import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import plotting
import matplotlib.pyplot as plt

# Initialize Plotly for Jupyter notebook inline display
import plotly.io as pio
pio.renderers.default = "iframe" #"browser" if want to open in a browser tab

# Create a figure for the efficient frontier
fig = go.Figure()

# Generate points on the efficient frontier using the built-in function
ef_plot = EfficientFrontier(mu, S)
# We'll use matplotlib just to get the data points, but won't display the plot
fig_plt, ax = plt.subplots(figsize=(10, 7))
plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=False)

# Extract the efficient frontier line data from the matplotlib plot
ef_line = None
for line in ax.get_lines():
    if line.get_label() == 'Efficient frontier':
        ef_line = line
        break

if ef_line:
    ef_volatility = ef_line.get_xdata()
    ef_returns = ef_line.get_ydata()
    
    # Plot the efficient frontier with Plotly
    fig.add_trace(
        go.Scatter(
            x=ef_volatility,
            y=ef_returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='black', width=2)
        )
    )
plt.close(fig_plt)  # Close the matplotlib figure as we don't need to display it

# Rest of your code remains the same...
# Create a separate instance for max Sharpe ratio portfolio
ef_max_sharpe = EfficientFrontier(mu, S)
ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
weights_max_sharpe = ef_max_sharpe.clean_weights()
ret_tangent, std_tangent, sharpe = ef_max_sharpe.portfolio_performance(risk_free_rate=risk_free_rate)

# Create another separate instance for min volatility portfolio
ef_min_vol = EfficientFrontier(mu, S)
ef_min_vol.min_volatility()
weights_min_vol = ef_min_vol.clean_weights()
ret_min_vol, std_min_vol, sharpe_min_vol = ef_min_vol.portfolio_performance(risk_free_rate=risk_free_rate)

# Create another separate instance for max utility portfolio
ef_max_utility = EfficientFrontier(mu, S)
ef_max_utility.max_quadratic_utility(risk_aversion=risk_aversion, market_neutral=False)
weights_max_utility = ef_max_utility.clean_weights()
ret_utility, std_utility, sharpe_utility = ef_max_utility.portfolio_performance(risk_free_rate=risk_free_rate)

# Plot the optimal portfolios
fig.add_trace(
    go.Scatter(
        x=[std_tangent],
        y=[ret_tangent],
        mode='markers',
        name='Max Sharpe',
        marker=dict(color='red', size=15, symbol='star')
    )
)

fig.add_trace(
    go.Scatter(
        x=[std_min_vol],
        y=[ret_min_vol],
        mode='markers',
        name='Min Volatility',
        marker=dict(color='green', size=15, symbol='star')
    )
)

fig.add_trace(
    go.Scatter(
        x=[std_utility],
        y=[ret_utility],
        mode='markers',
        name='Max Utility',
        marker=dict(color='blue', size=15, symbol='star')
    )
)

# Generate random portfolios
n_samples = 10000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt(np.diag(w @ S @ w.T))
sharpes = rets / stds

# Create a colorscale for the Sharpe ratios
sharpe_colorscale = px.colors.sequential.Viridis_r

# Plot the random portfolios
fig.add_trace(
    go.Scatter(
        x=stds,
        y=rets,
        mode='markers',
        name='Random Portfolios',
        marker=dict(
            color=sharpes,
            colorscale=sharpe_colorscale,
            colorbar=dict(title='Sharpe Ratio'),
            size=5,
            opacity=0.7
        ),
        showlegend=False
    )
)

# Update the layout
fig.update_layout(
    title='Efficient Frontier with Random Portfolios',
    xaxis_title='Annual Volatility',
    yaxis_title='Expected Annual Return',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    width=1000,
    height=700,
    template='plotly_white'
)

# Display the figure inline in the notebook
fig.show()
```


```python


import matplotlib.pyplot as plt
from pypfopt import plotting
import numpy as np
# Create the plot
fig, ax = plt.subplots(figsize=(10, 7))

# Create a new instance for plotting the efficient frontier
ef_plot = EfficientFrontier(mu, S)
plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=False)

# Create a separate instance for max Sharpe ratio portfolio
ef_max_sharpe = EfficientFrontier(mu, S)
ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
weights_max_sharpe = ef_max_sharpe.clean_weights()
ret_tangent, std_tangent, sharpe = ef_max_sharpe.portfolio_performance(risk_free_rate=risk_free_rate)

# Create another separate instance for min volatility portfolio
ef_min_vol = EfficientFrontier(mu, S)
ef_min_vol.min_volatility()
weights_min_vol = ef_min_vol.clean_weights()
ret_min_vol, std_min_vol, sharpe_min_vol = ef_min_vol.portfolio_performance(risk_free_rate=risk_free_rate)

# Create another separate instance for max utility portfolio
ef_max_utility = EfficientFrontier(mu, S)
ef_max_utility.max_quadratic_utility(risk_aversion=risk_aversion, market_neutral=False)
weights_max_utility = ef_max_utility.clean_weights()
ret_utility, std_utility, sharpe_utility = ef_max_utility.portfolio_performance(risk_free_rate=risk_free_rate)

# Plot the tangency portfolio (max Sharpe)
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Plot the minimum volatility portfolio
ax.scatter(std_min_vol, ret_min_vol, marker="*", s=100, c="g", label="Min Volatility")

# Plot the maximum utility portfolio
ax.scatter(std_utility, ret_utility, marker="*", s=100, c="b", label="Max Utility")

# Generate random portfolios
n_samples = 10000
w = np.random.dirichlet(np.ones(ef_plot.n_assets), n_samples)
rets = w.dot(ef_plot.expected_returns)
stds = np.sqrt(np.diag(w @ ef_plot.cov_matrix @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig("./outputs/ef_scatter.png", dpi=200)
plt.show()

# In a separate cell, plot the weights for all three portfolios
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plotting.plot_weights(weights_max_sharpe)
plt.title("Max Sharpe Portfolio Weights")

plt.subplot(1, 3, 2)
plotting.plot_weights(weights_min_vol)
plt.title("Min Volatility Portfolio Weights")

plt.subplot(1, 3, 3)
plotting.plot_weights(weights_max_utility)
plt.title("Max Utility Portfolio Weights")

plt.tight_layout()
plt.show()

# Print the performance metrics for comparison
print("Maximum Sharpe Portfolio:")
print(f"Expected annual return: {ret_tangent:.4f}")
print(f"Annual volatility: {std_tangent:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")

print("\nMinimum Volatility Portfolio:")
print(f"Expected annual return: {ret_min_vol:.4f}")
print(f"Annual volatility: {std_min_vol:.4f}")
print(f"Sharpe Ratio: {sharpe_min_vol:.4f}")

print("\nMaximum Utility Portfolio:")
print(f"Expected annual return: {ret_utility:.4f}")
print(f"Annual volatility: {std_utility:.4f}")
print(f"Sharpe Ratio: {sharpe_utility:.4f}")
print(f"Risk Aversion Parameter: {risk_aversion}")
```


```python


import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import column, row
from bokeh.palettes import Category10, viridis
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper
from bokeh.transform import linear_cmap
from bokeh.models.widgets import Div
from bokeh.io import output_notebook
import os

# Enable Bokeh output in the notebook
output_notebook()

# Create the directory if it doesn't exist
os.makedirs('./outputs', exist_ok=True)

# Create Efficient Frontier plot
ef_fig = figure(
    width=600, height=500,
    title="Efficient Frontier with Random Portfolios",
    x_axis_label="Annual Volatility",
    y_axis_label="Expected Annual Return"
)

# Add random portfolios
color_mapper = LinearColorMapper(palette=viridis(256), low=min(sharpes), high=max(sharpes))
cbar = ColorBar(color_mapper=color_mapper, title="Sharpe Ratio")
ef_fig.add_layout(cbar, 'right')

random_source = ColumnDataSource(data=dict(
    x=stds,
    y=rets,
    sharpe=sharpes
))
ef_fig.scatter('x', 'y', source=random_source, size=5, 
              color=linear_cmap('sharpe', viridis(256), min(sharpes), max(sharpes)),
              alpha=0.5)

# Add the key portfolios
portfolio_names = ['Max Sharpe', 'Min Volatility', 'Max Utility']
portfolios_source = ColumnDataSource(data=dict(
    x=[std_tangent, std_min_vol, std_utility],
    y=[ret_tangent, ret_min_vol, ret_utility],
    port_name=portfolio_names,
    color=['red', 'green', 'blue'],
    ret=[f"{ret_tangent:.4f}", f"{ret_min_vol:.4f}", f"{ret_utility:.4f}"],
    std=[f"{std_tangent:.4f}", f"{std_min_vol:.4f}", f"{std_utility:.4f}"],
    sharpe=[f"{sharpe:.4f}", f"{sharpe_min_vol:.4f}", f"{sharpe_utility:.4f}"]
))

port_scatter = ef_fig.scatter('x', 'y', source=portfolios_source, size=15, 
                             color='color', line_color="black", line_width=2)

hover = HoverTool(renderers=[port_scatter], 
                 tooltips=[
                     ("Portfolio", "@port_name"),
                     ("Return", "@ret"),
                     ("Volatility", "@std"),
                     ("Sharpe", "@sharpe")
                 ])
ef_fig.add_tools(hover)

# Create Time Series plot
ts_fig = figure(
    width=600, height=300,
    title="Asset Prices Time Series",
    x_axis_type="datetime",
    x_axis_label="Date",
    y_axis_label="Price"
)

# Add a line for each asset in prices_df
colors = Category10[10][:len(prices_df.columns)]
for i, col in enumerate(prices_df.columns):
    ts_data = {
        'x': prices_df.index,
        'y': prices_df[col]
    }
    source = ColumnDataSource(data=ts_data)
    ts_fig.line('x', 'y', source=source, line_width=2, color=colors[i % len(colors)], 
                legend_label=str(col))

ts_fig.legend.location = "top_left"
ts_fig.legend.click_policy = "hide"

# Create pie charts for portfolio allocations
def create_pie_chart(weights, title, width=300, height=300):
    radius = 0.8
    
    # Convert weights dictionary to sorted lists
    assets = list(weights.keys())
    values = [weights[asset] for asset in assets]
    
    # Remove assets with zero weight
    filtered_assets = []
    filtered_values = []
    for a, v in zip(assets, values):
        if v > 0.0001:
            filtered_assets.append(str(a))
            filtered_values.append(v)
    
    # Calculate angles for pie chart
    total = sum(filtered_values)
    angles = [val/total * 2*np.pi for val in filtered_values]
    
    # Calculate start and end angles
    start_angles = [sum(angles[:i]) for i in range(len(angles))]
    end_angles = [sum(angles[:i+1]) for i in range(len(angles))]
    
    # Use colors based on number of filtered assets
    pie_colors = colors[:len(filtered_assets)]
    
    # Prepare data for plotting
    source = ColumnDataSource(data=dict(
        asset_name=filtered_assets,
        values=[f"{v:.2%}" for v in filtered_values],
        start_angle=start_angles,
        end_angle=end_angles,
        color=pie_colors,
    ))
    
    # Create figure
    fig = figure(width=width, height=height, title=title,
                tools="hover", tooltips=[("Asset", "@asset_name"), ("Weight", "@values")],
                x_range=(-1.1, 1.1), y_range=(-1.1, 1.1))
    
    # Add wedges for the pie chart
    fig.wedge(x=0, y=0, radius=radius,
             start_angle='start_angle', end_angle='end_angle',
             line_color="white", fill_color='color', source=source)
    
    # Remove axes and grid
    fig.axis.visible = False
    fig.grid.visible = False
    
    return fig

# Create pie charts
pie_max_sharpe = create_pie_chart(weights_max_sharpe, "Max Sharpe Portfolio Weights")
pie_min_vol = create_pie_chart(weights_min_vol, "Min Volatility Portfolio Weights")
pie_max_utility = create_pie_chart(weights_max_utility, "Max Utility Portfolio Weights")

# Add performance metrics as a header div
performance_html = f"""
<div style="padding: 10px; background: #f8f8f8; border: 1px solid #ddd;">
    <h3>Portfolio Performance Metrics</h3>
    <table style="width: 100%; border-collapse: collapse;">
        <tr style="border-bottom: 1px solid #ddd; font-weight: bold;">
            <td>Portfolio</td>
            <td>Expected Return</td>
            <td>Volatility</td>
            <td>Sharpe Ratio</td>
        </tr>
        <tr style="border-bottom: 1px solid #ddd;">
            <td>Max Sharpe</td>
            <td>{ret_tangent:.4f}</td>
            <td>{std_tangent:.4f}</td>
            <td>{sharpe:.4f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ddd;">
            <td>Min Volatility</td>
            <td>{ret_min_vol:.4f}</td>
            <td>{std_min_vol:.4f}</td>
            <td>{sharpe_min_vol:.4f}</td>
        </tr>
        <tr>
            <td>Max Utility (Risk Aversion: {risk_aversion})</td>
            <td>{ret_utility:.4f}</td>
            <td>{std_utility:.4f}</td>
            <td>{sharpe_utility:.4f}</td>
        </tr>
    </table>
</div>
"""
header = Div(text=performance_html, width=1200)

# Create layout with two columns
pie_row = row(pie_max_sharpe, pie_min_vol, pie_max_utility)
right_column = column(ts_fig, pie_row)
main_row = row(ef_fig, right_column)
layout = column(header, main_row)

# Display the visualization in the notebook
show(layout)

# Also save to HTML file
output_file('./outputs/index.html', title="Portfolio Optimization")
save(layout)
print(f"Portfolio visualization saved to './outputs/index.html'")


# ### More examples of optimization methods
```

```python


"""
#Optimizers, apply the optimizer of choice after instantiating EfficientFrontier(mu,S)
#Portfolio with max sharpe ratio
ef.max_sharpe(risk_free_rate=risk_free_rate)
#ef.min_volatility()
ef.portfolio_performance(risk_free_rate=risk_free_rate)

#Optimizer
#Maximise return for a target risk. The resulting portfolio will have a volatility less than the target (but not guaranteed to be equal).
target_volatility=0.15
conditional_volatility_portfolio=ef.efficient_risk(target_volatility=target_volatility,market_neutral=True)
print(conditional_volatility_portfolio)
#ef.portfolio_performance(risk_free_rate=risk_free_rate) #re-calculate portfolio performance on conditional_volatility_portfolio 

#Optimizer
#Calculate the ‘Markowitz portfolio’, minimising volatility for a given target return.
target_return=0.2
conditional_return_portfolio=ef.efficient_return(target_return=target_return, market_neutral=False) #market_neutral=False means no shorting allowed
print(conditional_return_portfolio)
ef.portfolio_performance(risk_free_rate=risk_free_rate) #re-calculate portfolio performance on conditional_return_portfolio

#Optimizer
#Maximise the given quadratic utility
max_utility_portfolio=ef.max_quadratic_utility(risk_aversion=risk_aversion, market_neutral=False)
print(max_utility_portfolio)
#ef.portfolio_performance(risk_free_rate=risk_free_rate)

#Optimizer 
# minimize semivariance
min_semivariance_portfolio=ef.min_semivariance(target_return=target_return, market_neutral=False)
print(min_semivariance_portfolio)
#ef.portfolio_performance(risk_free_rate=risk_free_rate)

#Optimizer 
# minimize CVaR.
min_cvar_portfolio=ef.min_cvar(target_return=target_return, market_neutral=False)
print(min_cvar_portfolio)
#ef.portfolio_performance(risk_free_rate=risk_free_rate)
"""


# ### List stocks by sectors and icb
```

```python


from vnstock import Listing
sectors=stock.listing.symbols_by_industries()
sectors
```


```python


#Check if a stock in the portfolio is on the list and get relavant information
sectors[sectors['symbol']== 'FMC']
```


```python


#Get a list of unique sectors with icb3
unique_sectors=sectors['icb_name3'].unique() #np array
pd.DataFrame(unique_sectors)


# # Create a sceener with a pre-defind criteria
```

```python


from vnstock import Screener
from vnstock import Vnstock
# Initialize Vnstock with a symbol and source
symbol = 'VCI'  # Any valid symbol
source = 'VCI'  # Same as symbol for simplicity
stock = Vnstock().stock(symbol=symbol, source=source)
params = {
            "exchangeName": "HOSE,HNX,UPCOM",
            "marketCap": (100, 1000),
            "dividendYield": (5, 10)
        }

full_params = {
    # General Information
    "exchangeName": "HOSE,HNX,UPCOM",
    "hasFinancialReport": 1,  # Has the latest financial report
    "industryName": "Banks,Technology,Real Estate",
    "marketCap": (500, 10000),  # Market cap between 500-10000 billion VND
    "priceNearRealtime": (10, 100),  # Price between 10-100 VND
    "foreignVolumePercent": (5, 100),  # At least 5% foreign trading volume
    "alpha": (0.1, None),  # Positive alpha (excess return vs market)
    "beta": (0.8, 1.2),  # Beta close to market
    "freeTransferRate": (30, None),  # At least 30% freely transferable shares
    
    # Growth Metrics
    "revenueGrowth1Year": (10, None),  # At least 10% revenue growth in past year
    "revenueGrowth5Year": (5, None),  # At least 5% average revenue growth over 5 years
    "epsGrowth1Year": (5, None),  # At least 5% EPS growth in past year
    "epsGrowth5Year": (3, None),  # At least 3% average EPS growth over 5 years
    "lastQuarterRevenueGrowth": (0, None),  # Positive revenue growth in last quarter
    "lastQuarterProfitGrowth": (0, None),  # Positive profit growth in last quarter
    
    # Financial Ratios
    "grossMargin": (20, None),  # At least 20% gross margin
    "netMargin": (10, None),  # At least 10% net margin
    "roe": (15, None),  # At least 15% return on equity
    "doe": (5, None),  # At least 5% dividend on equity
    "dividendYield": (3, None),  # At least 3% dividend yield
    "pe": (5, 20),  # P/E between 5 and 20
    "pb": (0.5, 3),  # P/B between 0.5 and 3
    "evEbitda": (None, 15),  # EV/EBITDA below 15
    "netCashPerMarketCap": (0.1, None),  # Net cash at least 10% of market cap
    
    # Price & Volume Movements
    "totalTradingValue": (1, None),  # At least 1 billion VND trading value today
    "avgTradingValue20Day": (0.5, None),  # At least 0.5 billion VND average trading value over 20 days
    "priceGrowth1Week": (-5, 5),  # Price change between -5% and +5% in last week
    "priceGrowth1Month": (-10, 10),  # Price change between -10% and +10% in last month
    "percent1YearFromPeak": (None, -20),  # At least 20% below 1-year peak
    "percent1YearFromBottom": (10, None),  # At least 10% above 1-year bottom
    
    # Technical Indicators
    "rsi14": (30, 70),  # RSI between 30 and 70
    "priceVsSMA20": "ABOVE",  # Price above 20-day SMA
    "priceVsSMA50": "ABOVE",  # Price above 50-day SMA
    "volumeVsVSma20": (1, None),  # Volume above 20-day volume SMA
    
    # Market Behavior
    "strongBuyPercentage": (50, None),  # At least 50% strong buy signals
    "foreignTransaction": "buyMoreThanSell",  # Foreign investors buying more than selling
    
    # TCBS Ratings
    "stockRating": (3, None),  # Stock rating at least 3 out of 5
    "businessModel": (3, None),  # Business model rating at least 3 out of 5
    "financialHealth": (3, None)  # Financial health rating at least 3 out of 5
}

df_filtered = stock.screener.stock(params = params, limit=1700) #drop_lang='vi' 
df_filtered
```


```python


#screener_df = stock.screener.stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=1700)
#screener_df


# # Hierarchical Risk Parity
```

```python


from pypfopt import HRPOpt
hrp = HRPOpt(returns=returns) #create the object
weights = hrp.optimize() #optimize the object
weights


# ### Round the weights and save to a json file for webview
```

```python


hrp.clean_weights()
#Saving weights to csv
hrp.save_weights_to_file("./outputs/hrp_weights.json")


# # Dollars Allocation
```

```python


# This extracts the last row of prices_df and converts it to a Series

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices = get_latest_prices(prices_df) #Alternatives: latest_prices = prices_df.iloc[-1]
# Assuming you already have optimized weights (e.g., from max_sharpe or min_volatility)
# And a total portfolio value
portfolio_value = 100000  # 1 million VND

# You can use either the Series or dict version of latest_prices
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portfolio_value)
allocation, leftover = da.greedy_portfolio()


# ### Plotting with the HRP portfolio with pypfopt
```

```python


from pypfopt import plotting
plotting.plot_dendrogram(hrp, ax=None, show_tickers=True)


# ### Plot the covariance
```

```python


# Plot the covariance
plotting.plot_covariance(S, plot_correlation=True, show_tickers=True)


# # Riskfolio
```

```python


import warnings
#warnings.filterwarnings("ignore")
import riskfolio as rp
"""
# Building the portfolio object
port = rp.Portfolio(returns=returns)

# Calculating optimal portfolio

# Select method and estimate input parameters:

method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov)
"""
# Estimate optimal portfolio:

model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
rm = 'MV' # Risk measure used, this time will be variance
obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True # Use historical scenarios for risk measures that depend on scenarios
rf = 0 # Risk free rate
l = 0 # Risk aversion factor, only useful when obj is 'Utility'

#w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
#display(w.T)



# ### Plot the weights on a pie chart
```

```python


# Plotting the composition of the portfolio
#Convert weights_max_sharpe variable to series or use w=w to use the wieghts generated by riskfolio

ax = rp.plot_pie(w=pd.Series(weights_max_sharpe), title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)


# ### Setting up risk measures variables
```

```python


# Risk Measures available:
#
# 'MV': Standard Deviation.
# 'MAD': Mean Absolute Deviation.
# 'MSV': Semi Standard Deviation.
# 'FLPM': First Lower Partial Moment (Omega Ratio).
# 'SLPM': Second Lower Partial Moment (Sortino Ratio).
# 'CVaR': Conditional Value at Risk.
# 'EVaR': Entropic Value at Risk.
# 'WR': Worst Realization (Minimax)
# 'MDD': Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio).
# 'ADD': Average Drawdown of uncompounded cumulative returns.
# 'CDaR': Conditional Drawdown at Risk of uncompounded cumulative returns.
# 'EDaR': Entropic Drawdown at Risk of uncompounded cumulative returns.
# 'UCI': Ulcer Index of uncompounded cumulative returns.

rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR',
       'EVaR', 'WR', 'MDD', 'ADD', 'CDaR', 'UCI', 'EDaR']
"""
w_s = pd.DataFrame([])

for i in rms:
    w = port.optimization(model=model, rm=i, obj=obj, rf=rf, l=l, hist=hist)
    w_s = pd.concat([w_s, w], axis=1)
    
w_s.columns = rms
"""
```


```python


#w_s.style.format("{:.2%}").background_gradient(cmap='YlGn')


# ### Plot the historical compounded cummulative returns
```

```python


ax = rp.plot_series(returns=returns,
                    w=pd.Series(weights_max_sharpe),
                    cmap='tab20',
                    height=6,
                    width=10,
                    ax=None)


# ### Plot risk contribution on a bar chart
```

```python


ax = rp.plot_risk_con(w=weights_max_sharpe,
                      cov=S, #using covariance from pypfopt lib, which in this case is sample_cov
                      returns=returns,
                      rm=rm,
                      rf=risk_free_rate,
                      alpha=0.05,
                      color="tab:blue",
                      height=6,
                      width=10,
                      t_factor=252,
                      ax=None)


# ### Plot a histogram of returns distribution
```

```python


ax = rp.plot_hist(returns=returns,
                  w=pd.Series(weights_max_sharpe),
                  alpha=0.05,
                  bins=50,
                  height=6,
                  width=10,
                  ax=None)
```


```python


ax = rp.plot_range(returns=returns,
                   w=pd.Series(weights_max_sharpe),
                   alpha=0.05,
                   a_sim=100,
                   beta=None,
                   b_sim=None,
                   bins=50,
                   height=6,
                   width=10,
                   ax=None)
```


```python


"""
ax = rp.jupyter_report(returns,
                       w,
                       rm='MV',
                       rf=0,
                       alpha=0.05,
                       height=6,
                       width=14,
                       others=0.05,
                       nrow=25)
"""                       


# ### Plot the full report
```

```python


ax = rp.plot_table(returns=returns,
                   w=pd.Series(weights_max_sharpe),
                   MAR=0,
                   alpha=0.05,
                   ax=None)


# ### Save a report to excel
```

```python


rp.excel_report(returns,
                w,
                rf=risk_free-rate, # could =0 or =risk_free-rate
                alpha=0.05,
                t_factor=252,
                ini_days=1,
                days_per_year=252,
                name="./outputs/report")
```


```python


ax = rp.plot_factor_risk_con(w=pd.Series(weights_max_sharpe),
                             cov=S,
                             returns=returns,
                             factors=returns,
                             B=None,
                             const=True,
                             rm=rm,
                             rf=0,
                             feature_selection="PCR", #Indicate the method used to estimate the loadings matrix, PCR or stepwise
                             n_components=0.95,
                             height=6,
                             width=10,
                             t_factor=252,
                             ax=None)
```


```python


ax = rp.plot_drawdown(returns=returns,
                      w=pd.Series(weights_max_sharpe), # or w=w
                      alpha=0.05,
                      height=8,
                      width=10,
                      ax=None)


# # Quantstats
```

```python


import quantstats as qs
qs.extend_pandas()
```


```python


#View a complete list of available metrics
[f for f in dir(qs.stats) if f[0] != '_'] 
```


```python


qs.stats.avg_loss(portfolio_returns)
```


```python


#View a complete list of available plots, some will not work in pandas 3.0 
[f for f in dir(qs.plots) if f[0] != '_']
```


```python


qs.plots.rolling_beta(portfolio_returns,benchmark_rets) #benchmark_rets should have been index fund rather than VNINDEX


# # Constraints Portfolio
```

```python


import pandas as pd

# Function to create a sector mapping dictionary from the grouped sectors
def create_sector_mapper(grouped_data):
    sector_mapper = {}
    
    for sector_name, group_df in grouped_data:
        # For each stock in this sector group, add to the dictionary
        for _, row in group_df.iterrows():
            sector_mapper[row['symbol']] = sector_name
    
    return sector_mapper

# Group by sector
grouped = sectors.groupby('icb_name3')

# Create the sector mapper dictionary
sector_mapper = create_sector_mapper(grouped)

print(f"Created sector mapper for {len(sector_mapper)} symbols")
```


```python


#sector_mapper
```


```python


sector_lower = {"Sản xuất & Phân phối Điện": 0.1, "Sản xuất thực phẩm": 0.05}  # Min percentages
sector_upper = {"Sản xuất & Phân phối Điện": 0.4, "Sản xuất thực phẩm/Gas": 0.2}      # Max percentages
```


```python


import json
import os
output_dir = "./outputs"
# Save the sector mapper to a JSON file in the specified directory
output_path = os.path.join(output_dir, "sector_map.json")
with open(output_path, 'w',encoding='utf-8') as json_file:
    json.dump(sector_mapper, json_file, indent=4, ensure_ascii=False) #ensure_ascii=False and encoding='utf-8' for Vietnamese language encoding
```


```python


from pypfopt import EfficientFrontier
#Instantiate an instance  
constraint_portfolio = EfficientFrontier(mu, S)

# Apply sector constraints
constraint_portfolio.add_sector_constraints(sector_mapper, sector_lower, sector_upper)



```

```python


# Optimize the portfolio
constraint_portfolio_weights = constraint_portfolio.max_sharpe(risk_free_rate=risk_free_rate)
constraint_portfolio_weights
```


```python


import os
# Convert the dictionary to a DataFrame for CSV export
sector_df = pd.DataFrame(list(sector_mapper.items()), columns=['symbol', 'sector'])

# Ensure the output directory exists
output_dir = "./outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the DataFrame to a CSV file with UTF-8 encoding
output_path = os.path.join(output_dir, "sector_map.csv")
sector_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"Sector mapper saved to '{output_path}' with UTF-8 encoding")


# # Tradingview
```

```python


!pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
#https://github.com/rongardF/tvdatafeed?tab=readme-ov-file

from tvDatafeed import TvDatafeed, Interval
import tvDatafeed
import datetime
#Getting stock data using nologin method
tv = TvDatafeed()

data = tv.get_hist('REE','HOSE', Interval.in_monthly,n_bars=120)
```


```python


#data
```


```python


"""
from tvDatafeed import TvDatafeed,Interval

username = 'YourTradingViewUsername'
password = 'YourTradingViewPassword'
    
tv=TvDatafeed(username, password, chromedriver_path=None)

"""


# # Visualization with vnstock_ezchart 
```

```python


from vnstock_ezchart import *
from vnstock import Vnstock
ezchart = MPlot() # Khởi tạo đối tượng
# Set date range
start_date = '2024-01-01'
end_date = '2025-03-19'
interval = '1D'
stock = Vnstock().stock(symbol='REE', source='VCI')
candle_df = stock.quote.history(start= start_date, end= end_date)
```


```python


"""
ezchart.combo_chart(candle_df['volume'] / 1000_000, candle_df['close']/1000,
                  left_ylabel='Volume (M)', right_ylabel='Price (K)',
                  color_palette='vnstock', palette_shuffle=True,
                  show_legend=False,
                  figsize=(10, 6),
                  title='Khối lượng giao dịch và giá đóng cửa theo thời gian',
                  title_fontsize=14
                  )
"""
candle_df.to_csv('./outputs/candle_df.csv')
```


```python


CashFlow = stock.finance.cash_flow(period='year', dropna=True)
#CashFlow.to_csv('./outputs/CashFlow.csv')
```


```python


# List the columns headers 
CashFlow.columns.tolist()


# ## Transpose the CF dataframe 
```

```python


#CashFlow
CashFlow_transposed = CashFlow.T
CashFlow_transposed.columns = CashFlow['yearReport']
# Drop the duplicate 'yearReport' row
CashFlow_transposed = CashFlow_transposed.drop('yearReport')
CashFlow_transposed.head()
```


```python


"""

ezchart.bar(CashFlow['Net cash inflows/outflows from operating activities'] / 1000_000_000, 
          color_palette='vnstock', palette_shuffle=False, 
          title='Biểu đồ cột', xlabel='Danh mục', ylabel='Giá trị', 
          grid=False, 
          data_labels=True,
          show_legend=False,
          legend_title='Chú thích',
          series_names=['Test'],
          figsize=(10, 6), 
          rot=45, 
          width=0.7,
          title_fontsize=15,
          label_fontsize=10,
          bar_edge_color='lightgrey'
          )
"""


# # Port the built-in visulization method to seaborn for a more polished look
```

```python


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming CashFlow is a DataFrame you already have
# Normalize values to billions
data = CashFlow['Net cash inflows/outflows from operating activities'] / 1000_000_000

# Create figure with specified size
plt.figure(figsize=(12, 6))

# Sort data by year to ensure proper ordering
sorted_indices = CashFlow['yearReport'].argsort()
years = CashFlow['yearReport'].iloc[sorted_indices]
sorted_data = data.iloc[sorted_indices]

# Create the vertical bar plot with sorted data
ax = sns.barplot(
    x=years,                # Sorted years on x-axis
    y=sorted_data,          # Sorted data values on y-axis
    edgecolor='lightgrey',
    width=0.7
)

# Set title and labels with specified font sizes
plt.title('Net Operating Cashflow', fontsize=15)
plt.xlabel('Years', fontsize=10)
plt.ylabel('Value', fontsize=10)

# Turn off grid
plt.grid(False)

# Add data labels with minimal gap above bars
for i, v in enumerate(sorted_data):
    # Much smaller offset - adjust the multiplier as needed
    offset = 0.01 * max(sorted_data)
    
    ax.text(
        i,                       # x position (bar index)
        v + offset,              # Position with minimal gap
        f'{v:.2f}',              # formatted value 
        ha='center',             # horizontal alignment
        va='bottom',             # vertical alignment
        fontsize=10
    )

# Adjust y-axis limit to make room for labels
ymax = max(sorted_data) * 1.05  # Add just 5% padding to y-axis
plt.ylim(0, ymax)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Save the seaborn figure
plt.savefig('./outputs/Net OCF.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
```


```python


"""
# For plotly, Add this at the end instead of fig.show()
fig.write_html("./outputs/cash_flow_chart.html")  # Saves as interactive HTML
# Or
fig.write_image("./outputs/cash_flow_chart.png")  # Requires kaleido package
"""


```

```python


Ratio = stock.finance.ratio(period='year', lang='vi', dropna=True)
#Ratio.to_csv('./outputs/ratios.csv')
Ratio.columns.to_list()


# ### Transpose the data frame to display on the web.
```

```python


Ratio_transposed = Ratio.T
Ratio_transposed.columns=Ratio_transposed.iloc[1]
Ratio_transposed = Ratio_transposed.iloc[3:]
```


```python


# Example: Select the 'ROE (%)' column under 'Chỉ tiêu khả năng sinh lợi'
roe = Ratio[('Chỉ tiêu khả năng sinh lợi', 'ROE (%)')]
roe
```


```python


import matplotlib.pyplot as plt
Ratio_plot = Ratio.copy()
# Select the two columns
col1 = ('Chỉ tiêu khả năng sinh lợi', 'Biên lợi nhuận gộp (%)')
col2 = ('Chỉ tiêu cơ cấu nguồn vốn', 'Nợ/VCSH')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(Ratio_plot.index, Ratio_plot[col1], marker='o', label='Biên lợi nhuận gộp (%)')
plt.plot(Ratio_plot.index, Ratio_plot[col2], marker='s', label='Nợ/VCSH')
plt.xlabel('Năm')
plt.ylabel('Giá trị (%)')
plt.title('So sánh Biên lợi nhuận gộp và Nợ/VCSH theo năm')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```


```python


import matplotlib.pyplot as plt
import seaborn as sns

# Select the columns
col_roe = ('Chỉ tiêu khả năng sinh lợi', 'ROE (%)')
col_debt_equity = ('Chỉ tiêu cơ cấu nguồn vốn', 'Nợ/VCSH')

# Prepare the DataFrame for plotting
df_plot = Ratio[[col_roe, col_debt_equity]].copy()
df_plot.columns = ['ROE (%)', 'Nợ/VCSH']

# Optional: add company or year info for further analysis
if ('Meta', 'CP') in Ratio.columns:
    df_plot['CP'] = Ratio[('Meta', 'CP')]
if ('Meta', 'Năm') in Ratio.columns:
    df_plot['Năm'] = Ratio[('Meta', 'Năm')]

# Scatter plot with regression line
plt.figure(figsize=(8, 6))
sns.regplot(data=df_plot, x='Nợ/VCSH', y='ROE (%)', scatter_kws={'alpha':0.7})
plt.title('Mối quan hệ giữa Đòn bẩy tài chính (Nợ/VCSH) và ROE (%)')
plt.xlabel('Nợ/VCSH')
plt.ylabel('ROE (%)')
plt.grid(True)
plt.tight_layout()
plt.show()
```


```python


import seaborn as sns
import matplotlib.pyplot as plt

# Define the five most meaningful metrics (six columns for all pairwise relationships)
selected_cols = [
    ('Chỉ tiêu khả năng sinh lợi', 'ROE (%)'),
    ('Chỉ tiêu cơ cấu nguồn vốn', 'Nợ/VCSH'),
    ('Chỉ tiêu hiệu quả hoạt động', 'Vòng quay tài sản'),
    ('Chỉ tiêu khả năng sinh lợi', 'Biên lợi nhuận ròng (%)'),
    ('Chỉ tiêu thanh khoản', 'Chỉ số thanh toán hiện thời'),
    ('Chỉ tiêu định giá', 'P/S'),
]

# Subset the DataFrame
df_pair = Ratio[selected_cols].copy()

# Use only the second part of each column tuple (the metric name)
df_pair.columns = [col[1] for col in df_pair.columns]

# Optional: Remove rows with missing values for these columns
#df_pair = df_pair.dropna()

# Create the pairplot
sns.pairplot(df_pair, diag_kind='kde', corner=True)
plt.suptitle('Pairplot of Key Financial Metrics', y=1.02)
plt.tight_layout()
plt.show()
```


```python


# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import numpy as np
# import pandas as pd

# # Define the five most meaningful metrics (six columns for all pairwise relationships)
# selected_cols = [
#     ('Chỉ tiêu khả năng sinh lợi', 'ROE (%)'),
#     ('Chỉ tiêu cơ cấu nguồn vốn', 'Nợ/VCSH'),
#     ('Chỉ tiêu hiệu quả hoạt động', 'Vòng quay tài sản'),
#     ('Chỉ tiêu khả năng sinh lợi', 'Biên lợi nhuận ròng (%)'),
#     ('Chỉ tiêu thanh khoản', 'Chỉ số thanh toán hiện thời'),
#     ('Chỉ tiêu định giá', 'P/S'),
# ]

# # Subset the DataFrame
# df_pair = Ratio[selected_cols].copy()

# # Use only the second part of each column tuple (the metric name)
# df_pair.columns = [col[1] for col in df_pair.columns]

# # Optional: Remove rows with missing values for these columns
# #df_pair = df_pair.dropna()

# # Create a plotly figure using px.scatter_matrix for the pairplot
# fig = px.scatter_matrix(
#     df_pair,
#     dimensions=df_pair.columns,
#     title="Pairplot of Key Financial Metrics",
#     labels={col: col for col in df_pair.columns},  # Use original column names as labels
#     color_discrete_sequence=['blue'],
#     opacity=0.6
# )

# # Update layout for better appearance
# fig.update_layout(
#     title={
#         'y':0.95,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'
#     },
#     dragmode='select',
#     width=1000,
#     height=900,
# )

# # Update traces for diagonal plots to show distributions
# for i, col in enumerate(df_pair.columns):
#     fig.update_traces(
#         diagonal_visible=True, 
#         showupperhalf=False,  # Only show lower half (like corner=True in seaborn)
#         selector=dict(dimensions=[col])
#     )

# # Update axes to include zero when appropriate
# fig.update_xaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
# fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)

# # Show the plot
# fig.show()

# # Save to HTML for sharing
# fig.write_html("./outputs/financial_metrics_pairplot.html")
# print("Interactive pairplot saved to financial_metrics_pairplot.html")
```


```python


# IncomeStatement.to_csv('./outputs/IncomeStatement.csv')
# BalanceSheet.to_csv('./outputs/BalanceSheet.csv') #index=False


# ### Transform the balance sheet from long format to wide format. 
```

```python


def BS_wide(stock=None):
    """
    Transform balance sheet data into a wide format with years as columns and metrics as rows.
    
    Parameters:
    -----------
    stock : object, default=None
        The stock ticker object containing financial data.
        If None, will create a default stock object for REE from VCI source.
    
    Returns:
    --------
    pandas.DataFrame
        Transformed balance sheet with years as columns and financial metrics as rows
    """
    
    # Create default stock object if not provided
    if stock is None:
        stock = Vnstock().stock(symbol='REE', source='VCI')
    
    # Get the balance sheet data
    BS = stock.finance.balance_sheet(period='year', lang='en', dropna=True)
    
    # Transpose the DataFrame
    BS_wide = BS.T
    
    # Promote header by setting column names using the second row (index 1)
    BS_wide.columns = BS_wide.iloc[1]
    
    # Keep only the data rows (skip the first 3 rows)
    BS_wide = BS_wide.iloc[3:]
    
    return BS_wide
```


```python


# BS_wide(stock)
```


```python


BalanceSheet = stock.finance.balance_sheet(period='year', lang='en', dropna=True)
BalanceSheet_Transposed = BalanceSheet.T
BalanceSheet_Transposed.columns = BalanceSheet_Transposed.iloc[1]
BalanceSheet_Transposed = BalanceSheet_Transposed.iloc[3:]
BalanceSheet_Transposed.head()


# ### Transform Income statement from long format to wide format. 
```

```python


IncomeStatement = stock.finance.income_statement(period='year', lang='en', dropna=True)
IncomeStatement_Transpose= IncomeStatement.T
IncomeStatement_Transpose.columns = IncomeStatement_Transpose.iloc[1]
IncomeStatement_Transpose = IncomeStatement_Transpose.iloc[3:]
IncomeStatement_Transpose.head()
```


```python


BalanceSheet_Transposed.to_csv('./outputs/REE_BalanceSheet_Transposed.csv')
IncomeStatement_Transpose.to_csv('./outputs/REE_IncomeStatement_Transpose.csv')
CashFlow_transposed.to_csv('./outputs/REE_CashFlow_transposed.csv')
```


```python


# import os
# import pandas as pd

# def save_financial_statements_to_csv(balance_sheet_df, income_statement_df, cashflow_df, output_dir='./outputs'):
#     """
#     Save financial statements DataFrames to CSV files in the specified output directory.
    
#     Parameters:
#     -----------
#     balance_sheet_df : pandas.DataFrame
#         Balance Sheet DataFrame
#     income_statement_df : pandas.DataFrame
#         Income Statement DataFrame
#     cashflow_df : pandas.DataFrame
#         Cash Flow Statement DataFrame
#     output_dir : str
#         Directory path where CSV files will be saved (default: './outputs')
#     """
    
#     # Create output directory if it doesn't exist
#     try:
#         os.makedirs(output_dir, exist_ok=True)
#         print(f"Output directory '{output_dir}' is ready.")
#     except Exception as e:
#         print(f"Error creating directory: {e}")
#         return
    
#     # Dictionary of DataFrames and their corresponding filenames
#     statements = {
#         'REE_BalanceSheet_Transposed.csv': balance_sheet_df,
#         'REE_IncomeStatement_Transpose.csv': income_statement_df,
#         'REE_CashFlow_transposed.csv': cashflow_df
#     }
    
#     # Save each DataFrame to CSV
#     for filename, df in statements.items():
#         try:
#             file_path = os.path.join(output_dir, filename)
#             df.to_csv(file_path)
#             print(f"Successfully saved {filename}")
#         except Exception as e:
#             print(f"Error saving {filename}: {e}")

# # Example usage:
# # save_financial_statements_to_csv(
# #     BalanceSheet_Transposed,
# #     IncomeStatement_Transpose,
# #     CashFlow_transposed
# # )


# # Holy Fama French batman
```

```python


factor3_url = 'https://raw.githubusercontent.com/gahoccode/Datasets/main/FamaFrench3FACTOR.csv'
factor5_url = 'https://raw.githubusercontent.com/gahoccode/Datasets/refs/heads/main/FamaFrench5FACTOR.csv'
fama_french_3f = pd.read_csv(factor3_url)
fama_french_5f= pd.read_csv(factor5_url)

# Convert the 'time' column to datetime and set as index
fama_french_3f['time'] = pd.to_datetime(fama_french_3f['time'])
fama_french_3f = fama_french_3f.set_index('time')
fama_french_5f['time'] = pd.to_datetime(fama_french_5f['time'])
fama_french_5f = fama_french_5f.set_index('time')
```


```python


fama_french_3f.head()
fama_french_5f.head()
```


```python


def three_factor(url=None):
    """
    Load and process the Fama-French 3-Factor model data.
    
    Parameters:
    ----------
    url : str, optional
        URL to the CSV file. If None, uses the default GitHub URL.
        
    Returns:
    -------
    pandas.DataFrame
        Processed Fama-French 3-Factor model data with datetime index.
    """
    if url is None:
        url = 'https://raw.githubusercontent.com/gahoccode/Datasets/main/FamaFrench3FACTOR.csv'
    
    # Load the data
    ff_3f = pd.read_csv(url)
    
    # Convert 'time' to datetime and set as index
    ff_3f['time'] = pd.to_datetime(ff_3f['time'])
    ff_3f = ff_3f.set_index('time')
    
    return ff_3f

def five_factor(url=None):
    """
    Load and process the Fama-French 5-Factor model data.
    
    Parameters:
    ----------
    url : str, optional
        URL to the CSV file. If None, uses the default GitHub URL.
        
    Returns:
    -------
    pandas.DataFrame
        Processed Fama-French 5-Factor model data with datetime index.
    """
    if url is None:
        url = 'https://raw.githubusercontent.com/gahoccode/Datasets/refs/heads/main/FamaFrench5FACTOR.csv'
    
    # Load the data
    ff_5f = pd.read_csv(url)
    
    # Convert 'time' to datetime and set as index
    ff_5f['time'] = pd.to_datetime(ff_5f['time'])
    ff_5f = ff_5f.set_index('time')
    
    return ff_5f

# # Import required libraries
# import pandas as pd

# # Load the factor data
# ff3 = three_factor()
# ff5 = five_factor()

# # Now you can work with the data
# print(ff3.head())
# print(ff5.head())  


# ### Add stock screener to get market cap info that only works from 9 to 5
```

```python


# from vnstock import Screener
# params = {
#     "exchangeName": "HOSE,HNX,UPCOM",
#     "ticker": "REE"  # Replace with your stock symbol
# }
# screener = Screener()
# df = screener.stock(params=params, limit=10)
# print(df[['ticker', 'marketCap']])

from vnstock import Vnstock
company = Vnstock().stock(symbol='REE', source='VCI').company
overview = company.overview()
overview.head()


```

```python


overview.columns.to_list()
```

