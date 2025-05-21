import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os



# Define the function to calculate DFL
def calculate_degree_of_financial_leverage(IncomeStatement):
    """
    Calculate Degree of Financial Leverage using percentage changes in Net Income and EBIT.
    
    DFL = % Change in Net Income / % Change in EBIT
    
    Parameters:
    -----------
    IncomeStatement : pandas DataFrame
        Income Statement data with columns including 'Operating Profit/Loss' and 'Attribute to parent company (Bn. VND)'
    
    Returns:
    --------
    pandas DataFrame
        DataFrame with DFL calculations
    """
    # Create a copy to avoid modifying the original dataframe
    financial_leverage_data = IncomeStatement.copy()
    
    # Rename for clarity
    financial_leverage_data = financial_leverage_data.rename(columns={
        'Operating Profit/Loss': 'EBIT (Bn. VND)',
        'Attribute to parent company (Bn. VND)': 'Net Income (Bn. VND)'
    })
    
    # Sort by ticker and year
    financial_leverage_data = financial_leverage_data.sort_values(['ticker', 'yearReport'])
    
    # Calculate year-over-year percentage changes for each ticker
    financial_leverage_data['EBIT % Change'] = financial_leverage_data.groupby('ticker')['EBIT (Bn. VND)'].pct_change() * 100
    financial_leverage_data['Net Income % Change'] = financial_leverage_data.groupby('ticker')['Net Income (Bn. VND)'].pct_change() * 100
    
    # Calculate DFL
    financial_leverage_data['DFL'] = financial_leverage_data['Net Income % Change'] / financial_leverage_data['EBIT % Change']
    
    # Handle infinite or NaN values (when EBIT % Change is near zero)
    financial_leverage_data['DFL'] = financial_leverage_data['DFL'].replace([np.inf, -np.inf], np.nan)
    
    # Select relevant columns
    dfl_results = financial_leverage_data[['ticker', 'yearReport', 
                                         'EBIT (Bn. VND)', 'Net Income (Bn. VND)', 
                                         'EBIT % Change', 'Net Income % Change', 
                                         'DFL']]
    
    return dfl_results

# Define the function to calculate ROCE and include ROE from Ratio dataframe
def calculate_roce_and_include_roe(BalanceSheet, IncomeStatement, Ratio):
    """
    Calculate ROCE and include ROE from Ratio dataframe with MultiIndex columns.
    """
    # Create a copy to avoid modifying the original dataframe
    BalanceSheet_copy = BalanceSheet.copy()
    
    # Step 1: Calculate Capital Employed
    BalanceSheet_copy['Capital Employed (Bn. VND)'] = (
        BalanceSheet_copy['Long-term borrowings (Bn. VND)'] + 
        BalanceSheet_copy['Short-term borrowings (Bn. VND)'] + 
        BalanceSheet_copy["OWNER'S EQUITY(Bn.VND)"]
    )
    
    # Step 2: Merge Balance Sheet and Income Statement
    merged_df = pd.merge(
        BalanceSheet_copy[['ticker', 'yearReport', 'Capital Employed (Bn. VND)']],
        IncomeStatement[['ticker', 'yearReport', 'Operating Profit/Loss']],
        on=['ticker', 'yearReport'],
        how='inner'
    )
    
    # Step 3: Calculate ROCE
    merged_df['ROCE'] = merged_df['Operating Profit/Loss'] / merged_df['Capital Employed (Bn. VND)']
    
    # Select columns for ROCE calculation
    ROCE_df = merged_df[['ticker', 'yearReport', 'Operating Profit/Loss', 
                         'Capital Employed (Bn. VND)', 'ROCE']]
    ROCE_df = ROCE_df.rename(columns={'Operating Profit/Loss': 'EBIT (Bn. VND)'})
    
    # Step 4: Create a simplified version of Ratio DataFrame for merging
    # Extract the ticker, year, and ROE columns
    ratio_simple = pd.DataFrame({
        'ticker': Ratio[('Meta', 'CP')],
        'yearReport': Ratio[('Meta', 'Năm')],
        'ROE': Ratio[('Chỉ tiêu khả năng sinh lợi', 'ROE (%)')]
    })
    
    # Step 5: Merge with simplified Ratio dataframe to include ROE
    ROCE_df = pd.merge(
        ROCE_df,
        ratio_simple[['ticker', 'yearReport', 'ROE']],
        on=['ticker', 'yearReport'],
        how='left'
    )
    
    return ROCE_df

# Define function for DuPont analysis
def create_dupont_analysis(IncomeStatement, BalanceSheet, CashFlow):
    """
    Create a 3-factor DuPont analysis based on the three financial statements.
    
    DuPont Analysis: ROE = Net Profit Margin × Asset Turnover × Financial Leverage
    
    Where:
    - Net Profit Margin = Net Income / Revenue
    - Asset Turnover = Revenue / Average Total Assets
    - Financial Leverage = Average Total Assets / Average Shareholders' Equity
    
    Returns:
    --------
    pandas DataFrame
        DataFrame with DuPont analysis results
    """
    # Step 1: Combine necessary data from all three statements
    # Start with Income Statement data for revenue and net income
    income_data = IncomeStatement[['ticker', 'yearReport', 'Revenue (Bn. VND)', 'Attribute to parent company (Bn. VND)']].copy()
    
    # Rename for clarity
    income_data = income_data.rename(columns={'Attribute to parent company (Bn. VND)': 'Net Income (Bn. VND)'})
    
    # Step 2: Add Balance Sheet data for assets and equity
    balance_data = BalanceSheet[['ticker', 'yearReport', 'TOTAL ASSETS (Bn. VND)', "OWNER'S EQUITY(Bn.VND)"]].copy()
    
    # Merge the dataframes
    dupont_df = pd.merge(income_data, balance_data, on=['ticker', 'yearReport'], how='inner')
    
    # Step 3: Group by ticker to calculate year-over-year values and averages
    # Sort by ticker and year
    dupont_df = dupont_df.sort_values(['ticker', 'yearReport'])
    
    # Calculate average total assets and equity for each year (current + previous year) / 2
    # First create shifted columns for previous year's values
    dupont_df['Prev_Assets'] = dupont_df.groupby('ticker')['TOTAL ASSETS (Bn. VND)'].shift(1)
    dupont_df['Prev_Equity'] = dupont_df.groupby('ticker')["OWNER'S EQUITY(Bn.VND)"].shift(1)
    
    # Calculate averages
    dupont_df['Average Total Assets (Bn. VND)'] = (dupont_df['TOTAL ASSETS (Bn. VND)'] + dupont_df['Prev_Assets']) / 2
    dupont_df['Average Equity (Bn. VND)'] = (dupont_df["OWNER'S EQUITY(Bn.VND)"] + dupont_df['Prev_Equity']) / 2
    
    # For the first year of each ticker, we don't have previous year data, so use current year
    dupont_df['Average Total Assets (Bn. VND)'] = dupont_df['Average Total Assets (Bn. VND)'].fillna(
        dupont_df['TOTAL ASSETS (Bn. VND)'])
    dupont_df['Average Equity (Bn. VND)'] = dupont_df['Average Equity (Bn. VND)'].fillna(
        dupont_df["OWNER'S EQUITY(Bn.VND)"])
    
    # Step 4: Calculate the 3 DuPont components
    # Net Profit Margin = Net Income / Revenue
    dupont_df['Net Profit Margin'] = dupont_df['Net Income (Bn. VND)'] / dupont_df['Revenue (Bn. VND)']
    
    # Asset Turnover = Revenue / Average Total Assets
    dupont_df['Asset Turnover'] = dupont_df['Revenue (Bn. VND)'] / dupont_df['Average Total Assets (Bn. VND)']
    
    # Financial Leverage = Average Total Assets / Average Equity
    dupont_df['Financial Leverage'] = dupont_df['Average Total Assets (Bn. VND)'] / dupont_df['Average Equity (Bn. VND)']
    
    # Step 5: Calculate ROE using DuPont formula
    dupont_df['ROE (DuPont)'] = dupont_df['Net Profit Margin'] * dupont_df['Asset Turnover'] * dupont_df['Financial Leverage']
    
    # Step 6: Calculate ROE directly for validation
    dupont_df['ROE (Direct)'] = dupont_df['Net Income (Bn. VND)'] / dupont_df['Average Equity (Bn. VND)']
    
    # Step 7: Clean up the DataFrame and select relevant columns
    dupont_analysis = dupont_df[[
        'ticker', 'yearReport', 
        'Net Income (Bn. VND)', 'Revenue (Bn. VND)',
        'Average Total Assets (Bn. VND)', 'Average Equity (Bn. VND)',
        'Net Profit Margin', 'Asset Turnover', 'Financial Leverage',
        'ROE (DuPont)', 'ROE (Direct)'
    ]]
    
    # Convert ratios to percentages for better readability
    dupont_analysis['Net Profit Margin'] = dupont_analysis['Net Profit Margin'] * 100
    dupont_analysis['ROE (DuPont)'] = dupont_analysis['ROE (DuPont)'] * 100
    dupont_analysis['ROE (Direct)'] = dupont_analysis['ROE (Direct)'] * 100
    
    # Round values for better display
    dupont_analysis = dupont_analysis.round({
        'Net Profit Margin': 2,
        'Asset Turnover': 2,
        'Financial Leverage': 2,
        'ROE (DuPont)': 2,
        'ROE (Direct)': 2
    })
    
    return dupont_analysis

# Create a new layout with 6 rows, added row for DFL only (removed EBIT vs Net Income)
fig = make_subplots(
    rows=6, 
    cols=2,
    subplot_titles=(
        'Net Operating Cashflow',                  # Row 1, Col 1
        'ROIC (%) vs WACC (%)', # Row 1, Col 2
        'Cashflow',                     # Row 2, Col 1
        'ROCE vs ROE Comparison',                 # Row 2, Col 2
        'DuPont Analysis: Net Profit Margin (%)',  # Row 3, Col 1
        'DuPont Analysis: Asset Turnover',         # Row 3, Col 2
        'DuPont Analysis: Financial Leverage',     # Row 4, Col 1
        'Degree of Financial Leverage (DFL)'       # Row 5, Col 1
    ),
    specs=[
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "waterfall"}, {"type": "xy", "secondary_y": True}],
        [{"type": "xy"}, {"type": "xy"}],  # Row 3: Net Profit Margin, Asset Turnover
        [{"type": "xy"}, None],  # Row 4: Financial Leverage only (removed ROE Comparison)
        [{"type": "xy"}, None],  # Row 5: DFL only
        [{"type": "xy", "colspan": 2}, None]  # Row 6: Empty row for spacing
    ],
    vertical_spacing=0.06,  # Reduced spacing for more compact layout
    horizontal_spacing=0.08
)

# Get unique years from CashFlow
years = sorted(CashFlow['yearReport'].unique())

# Ensure years are integers for comparison operations
years = [int(y) if isinstance(y, str) else y for y in years]

# 1. Net Operating Cashflow (Bar Chart) - Top Left
# For each year, calculate the average Net cash inflows/outflows from operating activities
ocf_data = CashFlow.groupby('yearReport')['Net cash inflows/outflows from operating activities'].mean() / 1000  # Convert to billions
ocf_years = ocf_data.index.tolist()
ocf_values = ocf_data.values.tolist()

# Add bar chart
fig.add_trace(
    go.Bar(
        x=ocf_years,
        y=ocf_values,
        text=[f'{v:.2f}B' for v in ocf_values],
        textposition='outside',
        marker_color='rgb(66, 135, 245)',
        marker_line_color='lightgrey',
        marker_line_width=1.5,
        opacity=0.8,
        name='Net Operating Cash Flow'
    ),
    row=1, col=1
)

# 2. ROIC vs WACC Chart with Economic Value Zone - Top Right
# Get the years from both DataFrames
ratio_years = Ratio[('Meta', 'Năm')].values
# Convert to int if they're strings
ratio_years = [int(y) if isinstance(y, str) else y for y in ratio_years]

result_years = result_df['yearReport'].values
# Convert to int if they're strings
result_years = [int(y) if isinstance(y, str) else y for y in result_years]

# Find common years
common_years = sorted(list(set(ratio_years).intersection(set(result_years))))

# Filter and sort both DataFrames by common years
# Ensure comparison works correctly by converting to the same type
ratio_filtered = Ratio[Ratio[('Meta', 'Năm')].apply(lambda x: int(x) if isinstance(x, str) else x).isin(common_years)]
ratio_filtered = ratio_filtered.sort_values(('Meta', 'Năm'))

result_filtered = result_df[result_df['yearReport'].apply(lambda x: int(x) if isinstance(x, str) else x).isin(common_years)]
result_filtered = result_filtered.sort_values('yearReport')

# Calculate values using the filtered and aligned data
wacc_mean = result_filtered['wacc_market_based'].mean() * 100
roic_values = ratio_filtered[('Chỉ tiêu khả năng sinh lợi', 'ROIC (%)')] * 100
wacc_values = result_filtered['wacc_market_based'] * 100

# Add ROIC line
fig.add_trace(
    go.Scatter(
        x=ratio_filtered[('Meta', 'Năm')],
        y=roic_values,
        name='ROIC (%)',
        line=dict(color='blue')
    ),
    row=1, col=2
)

# Add WACC line
fig.add_trace(
    go.Scatter(
        x=result_filtered['yearReport'],
        y=wacc_values,
        name='WACC (%)',
        line=dict(color='green')
    ),
    row=1, col=2
)

# Add horizontal line at WACC mean
fig.add_shape(
    type="line",
    x0=min(common_years),
    x1=max(common_years),
    y0=wacc_mean,
    y1=wacc_mean,
    line=dict(color="red", dash="dash"),
    row=1, col=2
)

# Add WACC mean annotation
fig.add_annotation(
    x=max(common_years),
    y=wacc_mean,
    text=f"WACC Mean: {wacc_mean:.2f}%",
    showarrow=False,
    xanchor="right",
    font=dict(color="red"),
    row=1, col=2
)

# Only add shaded area if we have data points
if len(roic_values) > 0:
    # Create x values for the shaded area
    x_combined = np.concatenate([ratio_filtered[('Meta', 'Năm')].values, ratio_filtered[('Meta', 'Năm')].values[::-1]])
    y_upper = np.where(roic_values > wacc_mean, roic_values, wacc_mean)
    y_lower = np.full_like(roic_values, wacc_mean)
    y_combined = np.concatenate([y_upper, y_lower[::-1]])

    # Add the shaded area
    fig.add_trace(
        go.Scatter(
            x=x_combined,
            y=y_combined,
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(width=0),
            showlegend=True,
            name='Economic Value',
            legendgroup='economic_value',
            hoverinfo='skip'
        ),
        row=1, col=2
    )

# Add Economic Value annotation in upper left corner of the ROIC vs WACC plot
fig.add_annotation(
    x=0.52,  # Adjusted for subplot positioning
    y=0.97,  # Top side
    xref="paper",  # Use paper coordinates (0-1)
    yref="paper",  # Use paper coordinates (0-1)
    text="<b>Economic Value</b>",
    showarrow=False,
    font=dict(
        size=12,
        color="blue"
    ),
    bgcolor='rgba(0, 100, 255, 0.2)',  # Same as fill color
    bordercolor="blue",
    borderwidth=1,
    borderpad=4,
    opacity=0.8,
    xanchor="left",  # Left-align the text
    yanchor="top"    # Anchor at the top
)

# 3. Cashflow Waterfall - 2nd Row, Left
# Get a representative ticker (or use one specified)
ticker = CashFlow['ticker'].iloc[0]
# Get the latest year
latest_year = max(years)

# Filter data for the specific ticker and latest year
waterfall_data = CashFlow[(CashFlow['ticker'] == ticker) & 
                          (CashFlow['yearReport'] == latest_year)]

if len(waterfall_data) > 0:
    data = waterfall_data.iloc[0]
    
    # Key cashflow components with updated labels
    measures = [
        'Net Profit',  # Changed from 'Initial Cash' to 'Net Profit'
        'Operating CF',
        'Investing CF',
        'Financing CF',
        'Final Cash'
    ]
    
    # Updated values using Net Profit as the initial value
    values = [
        data['Net Profit/Loss before tax'],  # Changed to Net Profit/Loss before tax
        data['Net cash inflows/outflows from operating activities'],
        data['Net Cash Flows from Investing Activities'],
        data['Cash flows from financial activities'],
        data['Cash and Cash Equivalents at the end of period']
    ]
    
    measure_types = ['absolute', 'relative', 'relative', 'relative', 'total']
    
    # Add waterfall chart
    fig.add_trace(
        go.Waterfall(
            name=f"Cashflow {latest_year}",
            orientation="v",
            measure=measure_types,
            x=measures,
            textposition="outside",  # Standard position for all labels
            text=[f"{x:,.1f}" for x in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ),
        row=2, col=1
    )
    
    # Create text position array - use 'none' only for Investing CF (index 2)
    text_positions = ['outside', 'outside', 'none', 'outside', 'outside']
    
    # Update the waterfall chart to use different text positions for each bar
    fig.data[-1].textposition = text_positions

# 4. ROCE vs ROE Comparison - 2nd Row, Right
# Calculate ROCE and ROE using the provided function
ROCE_df = calculate_roce_and_include_roe(BalanceSheet, IncomeStatement, Ratio)

# Filter for the specific ticker
ticker_data = ROCE_df[ROCE_df['ticker'] == ticker].sort_values('yearReport')

# Convert ROCE to percentage
ticker_data['ROCE_PCT'] = ticker_data['ROCE'] * 100

# Convert ROE to percentage if it's not already
# Check if ROE values are already in percentage (typically >1 if in %)
if ticker_data['ROE'].max() < 1:  # If ROE is in decimal form (e.g., 0.15 for 15%)
    ticker_data['ROE_PCT'] = ticker_data['ROE'] * 100
else:
    ticker_data['ROE_PCT'] = ticker_data['ROE']  # If already in percentage form

# Add ROCE trace (using percentage values) - REMOVED text labels
fig.add_trace(
    go.Scatter(
        x=ticker_data['yearReport'],
        y=ticker_data['ROCE_PCT'],
        name='ROCE (%)',
        mode='lines+markers',  # Removed 'text' from mode
        line=dict(color='rgba(0, 117, 210, 0.9)', width=3),
        marker=dict(
            size=10, 
            color='rgba(0, 117, 210, 0.9)',
            line=dict(color='white', width=2)
        )
    ),
    row=2, col=2,
    secondary_y=False
)

# Add shaded area under ROCE for emphasis
fig.add_trace(
    go.Scatter(
        x=ticker_data['yearReport'],
        y=ticker_data['ROCE_PCT'],
        name='ROCE Area',
        mode='none',
        fill='tozeroy',
        fillcolor='rgba(0, 117, 210, 0.1)',
        showlegend=False,
        hoverinfo='skip'
    ),
    row=2, col=2,
    secondary_y=False
)

# Add ROE trace using percentage values - REMOVED text labels
fig.add_trace(
    go.Scatter(
        x=ticker_data['yearReport'],
        y=ticker_data['ROE_PCT'],
        name='ROE (%)',
        mode='lines+markers',  # Removed 'text' from mode
        line=dict(color='rgba(220, 20, 60, 0.9)', width=3),
        marker=dict(
            size=10, 
            symbol='diamond',
            color='rgba(220, 20, 60, 0.9)',
            line=dict(color='white', width=2)
        )
    ),
    row=2, col=2,
    secondary_y=True
)

# Add shaded area under ROE for emphasis
fig.add_trace(
    go.Scatter(
        x=ticker_data['yearReport'],
        y=ticker_data['ROE_PCT'],
        name='ROE Area',
        mode='none',
        fill='tozeroy',
        fillcolor='rgba(220, 20, 60, 0.1)',
        showlegend=False,
        hoverinfo='skip'
    ),
    row=2, col=2,
    secondary_y=True
)

# 5. DuPont Analysis (Split into 3 separate charts)
# Calculate DuPont Analysis
dupont_analysis = create_dupont_analysis(IncomeStatement, BalanceSheet, CashFlow)

# Filter for the specific ticker
dupont_ticker_data = dupont_analysis[dupont_analysis['ticker'] == ticker].sort_values('yearReport')

# 5.1. Net Profit Margin - Row 3, Col 1
fig.add_trace(
    go.Scatter(
        x=dupont_ticker_data['yearReport'],
        y=dupont_ticker_data['Net Profit Margin'],
        mode='lines+markers',
        name='Net Profit Margin (%)',
        line=dict(color='rgba(66, 135, 245, 0.9)', width=3),
        marker=dict(
            size=10, 
            color='rgba(66, 135, 245, 0.9)',
            line=dict(color='white', width=2)
        ),
        showlegend=True
    ),
    row=3, col=1
)

# Add shaded area for Net Profit Margin
fig.add_trace(
    go.Scatter(
        x=dupont_ticker_data['yearReport'],
        y=dupont_ticker_data['Net Profit Margin'],
        name='NPM Area',
        mode='none',
        fill='tozeroy',
        fillcolor='rgba(66, 135, 245, 0.1)',
        showlegend=False,
        hoverinfo='skip'
    ),
    row=3, col=1
)

# 5.2. Asset Turnover - Row 3, Col 2
fig.add_trace(
    go.Scatter(
        x=dupont_ticker_data['yearReport'],
        y=dupont_ticker_data['Asset Turnover'],
        mode='lines+markers',
        name='Asset Turnover',
        line=dict(color='rgba(0, 200, 0, 0.9)', width=3),
        marker=dict(
            size=10, 
            color='rgba(0, 200, 0, 0.9)',
            line=dict(color='white', width=2)
        ),
        showlegend=True
    ),
    row=3, col=2
)

# Add shaded area for Asset Turnover
fig.add_trace(
    go.Scatter(
        x=dupont_ticker_data['yearReport'],
        y=dupont_ticker_data['Asset Turnover'],
        name='AT Area',
        mode='none',
        fill='tozeroy',
        fillcolor='rgba(0, 200, 0, 0.1)',
        showlegend=False,
        hoverinfo='skip'
    ),
    row=3, col=2
)

# 5.3. Financial Leverage - Row 4, Col 1
fig.add_trace(
    go.Scatter(
        x=dupont_ticker_data['yearReport'],
        y=dupont_ticker_data['Financial Leverage'],
        mode='lines+markers',
        name='Financial Leverage',
        line=dict(color='rgba(220, 20, 60, 0.9)', width=3),
        marker=dict(
            size=10, 
            color='rgba(220, 20, 60, 0.9)',
            line=dict(color='white', width=2)
        ),
        showlegend=True
    ),
    row=4, col=1
)

# Add shaded area for Financial Leverage
fig.add_trace(
    go.Scatter(
        x=dupont_ticker_data['yearReport'],
        y=dupont_ticker_data['Financial Leverage'],
        name='FL Area',
        mode='none',
        fill='tozeroy',
        fillcolor='rgba(220, 20, 60, 0.1)',
        showlegend=False,
        hoverinfo='skip'
    ),
    row=4, col=1
)

# ROE Comparison chart removed as requested

# 6. NEW: DFL Analysis - Row 5, Col 1
# Calculate DFL
dfl_results = calculate_degree_of_financial_leverage(IncomeStatement)

# Filter for the specific ticker and remove rows with NaN DFL
dfl_data = dfl_results[(dfl_results['ticker'] == ticker) & (~dfl_results['DFL'].isna())].sort_values('yearReport')

if len(dfl_data) > 0:
    # Add DFL line chart
    fig.add_trace(
        go.Scatter(
            x=dfl_data['yearReport'],
            y=dfl_data['DFL'],
            mode='lines+markers',
            name='DFL',
            line=dict(color='rgba(214, 39, 40, 0.9)', width=3),
            marker=dict(
                size=10, 
                color='rgba(214, 39, 40, 0.9)',
                line=dict(color='white', width=2)
            ),
            showlegend=True
        ),
        row=5, col=1
    )
    
    # Add shaded area under DFL for emphasis
    fig.add_trace(
        go.Scatter(
            x=dfl_data['yearReport'],
            y=dfl_data['DFL'],
            name='DFL Area',
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(214, 39, 40, 0.1)',
            showlegend=False,
            hoverinfo='skip'
        ),
        row=5, col=1
    )
    
    # Add horizontal reference line at DFL = 1
    fig.add_shape(
        type="line",
        x0=min(dfl_data['yearReport']),
        y0=1,
        x1=max(dfl_data['yearReport']),
        y1=1,
        line=dict(color="gray", width=1, dash="dash"),
        row=5, col=1
    )
    
    # Add annotation explaining DFL = 1
    fig.add_annotation(
        x=dfl_data['yearReport'].iloc[-1],
        y=1.1,
        text="DFL = 1 (Neutral)",
        showarrow=False,
        xanchor="right",
        font=dict(size=10, color="gray"),
        row=5, col=1
    )


# Update layout for better appearance
fig.update_layout(
    title={
        'text': f'Financial Analysis Dashboard for {ticker}',
        'y':0.98,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    template='plotly_white',
    height=1800,  # Increased height for 6 rows
    width=1200,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.10,  # Adjusted for new layout
        xanchor="center",
        x=0.5,
        bgcolor='rgba(255, 255, 255, 0.7)',
        bordercolor='rgba(0, 0, 0, 0.2)',
        borderwidth=1
    ),
    hovermode='x unified',
    margin=dict(t=100, b=150, l=80, r=80)  # Increased bottom margin
)

# DuPont formula annotation removed

# DFL formula annotation removed

# Update axes labels and settings for all plots
# Row 1
fig.update_xaxes(
    title_text="Year",
    tickangle=45,
    tickmode='array',
    tickvals=years,
    row=1, 
    col=1
)

fig.update_yaxes(title_text="Value (Billions VND)", row=1, col=1)

fig.update_xaxes(
    title_text="Year",
    tickangle=45,
    tickmode='array',
    tickvals=common_years,
    row=1, 
    col=2
)

fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)

# Row 2
fig.update_xaxes(title_text="Cash Flow Components", tickangle=0, row=2, col=1)
fig.update_yaxes(title_text="Amount (Bn. VND)", row=2, col=1)

fig.update_xaxes(
    title_text="Year",
    tickangle=45,
    tickmode='array',
    tickvals=ticker_data['yearReport'],
    row=2, 
    col=2
)

fig.update_yaxes(
    title_text="ROCE (%)",
    title_font=dict(color='rgba(0, 117, 210, 1)'),
    row=2, 
    col=2,
    secondary_y=False
)

fig.update_yaxes(
    title_text="ROE (%)",
    title_font=dict(color='rgba(220, 20, 60, 1)'),
    row=2, 
    col=2,
    secondary_y=True
)

# Row 3 - DuPont Analysis (Net Profit Margin, Asset Turnover)
fig.update_xaxes(
    title_text="Year",
    tickangle=45,
    tickmode='array',
    tickvals=dupont_ticker_data['yearReport'],
    row=3, 
    col=1
)

fig.update_yaxes(
    title_text="Net Profit Margin (%)",
    title_font=dict(color='rgba(66, 135, 245, 1)'),
    row=3, 
    col=1
)

fig.update_xaxes(
    title_text="Year",
    tickangle=45,
    tickmode='array',
    tickvals=dupont_ticker_data['yearReport'],
    row=3, 
    col=2
)

fig.update_yaxes(
    title_text="Asset Turnover (times)",
    title_font=dict(color='rgba(0, 200, 0, 1)'),
    row=3, 
    col=2
)

# Row 4 - DuPont Analysis (Financial Leverage, ROE Comparison)
fig.update_xaxes(
    title_text="Year",
    tickangle=45,
    tickmode='array',
    tickvals=dupont_ticker_data['yearReport'],
    row=4, 
    col=1
)

fig.update_yaxes(
    title_text="Financial Leverage (times)",
    title_font=dict(color='rgba(220, 20, 60, 1)'),
    row=4, 
    col=1
)

# Axis configuration for ROE Comparison chart removed

# Row 5 - NEW: DFL and EBIT vs Net Income
if len(dfl_data) > 0:
    fig.update_xaxes(
        title_text="Year",
        tickangle=45,
        tickmode='array',
        tickvals=dfl_data['yearReport'],
        row=5, 
        col=1
    )

    fig.update_yaxes(
        title_text="Degree of Financial Leverage",
        title_font=dict(color='rgba(214, 39, 40, 1)'),
        row=5, 
        col=1
    )

    fig.update_xaxes(
        title_text="Year",
        tickangle=45,
        tickmode='array',
        tickvals=dfl_data['yearReport'],
        row=5, 
        col=2
    )

    fig.update_yaxes(
        title_text="Amount (Bn. VND)",
        title_font=dict(color='rgba(55, 83, 109, 1)'),
        row=5, 
        col=2,
        secondary_y=False
    )

    fig.update_yaxes(
        title_text="Percent Change (%)",
        title_font=dict(color='rgba(26, 118, 255, 1)'),
        row=5, 
        col=2,
        secondary_y=True
    )

# Update subplot titles
fig.update_annotations(
    text=f"Cashflow Waterfall ({latest_year})",
    selector={"index": 2}
)

# Remove grid lines and keep only axis lines
for row in range(1, 6):  # All rows
    for col in range(1, 3):  # Both columns
        fig.update_xaxes(
            showgrid=False,  # No grid lines
            showline=True,   # Show axis line
            linewidth=1.5,
            linecolor='rgba(0, 0, 0, 0.3)',
            row=row, 
            col=col
        )
        
        fig.update_yaxes(
            showgrid=False,  # No grid lines
            showline=True,   # Show axis line
            linewidth=1.5,
            linecolor='rgba(0, 0, 0, 0.3)',
            row=row, 
            col=col
        )

# Create outputs directory if it doesn't exist
output_dir = './dist'  # Changed from './outputs' to './dist'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the figure as an HTML file
output_path = os.path.join(output_dir, 'index.html')
fig.write_html(output_path)

print(f"Financial dashboard saved to {output_path}")