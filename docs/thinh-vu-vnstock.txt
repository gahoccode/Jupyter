# Vnstock Library Methods for Financial Analysis

The vnstock library provides comprehensive access to financial data for Vietnamese companies. This document details the methods available for retrieving and analyzing company information, financial statements, and screening stocks.

## Setup and Initialization

There are two main approaches to initializing the classes and accessing their methods:

### Method 1: Using the Vnstock class (Recommended)

```python
from vnstock import Vnstock

# Initialize with a stock symbol and data source
stock = Vnstock().stock(symbol='VCI', source='VCI')

# Now access various components through stock objects
# For example: stock.finance, stock.company, stock.quote, etc.
```

### Method 2: Direct import of specific classes

```python
from vnstock import Finance, Company, Screener

# Initialize the classes with a symbol
finance = Finance(symbol='VCI')
company = Company(symbol='VCI')
screener = Screener()

# Call methods directly
```

## Finance Class Methods

The Finance class provides several key methods to access different types of financial statements. Each method supports various parameters to customize the data retrieval.

### 1. `balance_sheet()` - Balance Sheet Statements

Retrieves a company's balance sheet, showing assets, liabilities, and shareholders' equity.

#### Parameters:

- `period` (str): Time period of data, options:
  - `'year'`: Annual data (default)
  - `'quarter'`: Quarterly data
  
- `lang` (str): Language for column names and labels, options:
  - `'vi'`: Vietnamese (default)
  - `'en'`: English
  
- `dropna` (bool): Whether to remove rows with NA values
  - `True`: Remove rows with NA values (default)
  - `False`: Keep all rows

#### Example Usage:

**Method 1: Using the Vnstock class (Recommended)**
```python
from vnstock import Vnstock

# Initialize with stock symbol and source
stock = Vnstock().stock(symbol='VCI', source='VCI')

# Get annual balance sheet in Vietnamese
annual_balance_sheet = stock.finance.balance_sheet(period='year', lang='vi', dropna=True)
print(annual_balance_sheet)

# Get quarterly balance sheet in English
quarterly_balance_sheet = stock.finance.balance_sheet(period='quarter', lang='en', dropna=True)
print(quarterly_balance_sheet)
```

**Method 2: Direct import of Finance class**
```python
from vnstock import Finance

# Initialize Finance class directly
finance = Finance(symbol='VCI')

# Get annual balance sheet in Vietnamese
annual_balance_sheet = finance.balance_sheet(period='year', lang='vi', dropna=True)
print(annual_balance_sheet)

# Get quarterly balance sheet in English
quarterly_balance_sheet = finance.balance_sheet(period='quarter', lang='en', dropna=True)
print(quarterly_balance_sheet)
```

### 2. `income_statement()` - Income Statements

Retrieves a company's income statement (profit and loss statement), showing revenues, costs, and expenses during a specific period.

#### Parameters:

- `period` (str): Time period of data, options:
  - `'year'`: Annual data (default)
  - `'quarter'`: Quarterly data
  
- `lang` (str): Language for column names and labels, options:
  - `'vi'`: Vietnamese (default)
  - `'en'`: English
  
- `dropna` (bool): Whether to remove rows with NA values
  - `True`: Remove rows with NA values (default)
  - `False`: Keep all rows

#### Example Usage:

**Method 1: Using the Vnstock class (Recommended)**
```python
from vnstock import Vnstock

# Initialize with stock symbol and source
stock = Vnstock().stock(symbol='VCI', source='VCI')

# Get annual income statement in Vietnamese
annual_income = stock.finance.income_statement(period='year', lang='vi', dropna=True)
print(annual_income)

# Get quarterly income statement in English
quarterly_income = stock.finance.income_statement(period='quarter', lang='en', dropna=True)
print(quarterly_income)
```

**Method 2: Direct import of Finance class**
```python
from vnstock import Finance

# Initialize Finance class directly
finance = Finance(symbol='VCI')

# Get annual income statement in Vietnamese
annual_income = finance.income_statement(period='year', lang='vi', dropna=True)
print(annual_income)

# Get quarterly income statement in English
quarterly_income = finance.income_statement(period='quarter', lang='en', dropna=True)
print(quarterly_income)
```

### 3. `cash_flow()` - Cash Flow Statements

Retrieves a company's cash flow statement, showing how changes in balance sheet accounts and income affect cash and cash equivalents.

#### Parameters:

- `period` (str): Time period of data, options:
  - `'year'`: Annual data (default)
  - `'quarter'`: Quarterly data
  
- `lang` (str): Language for column names and labels, options:
  - `'vi'`: Vietnamese (default)
  - `'en'`: English
  
- `dropna` (bool): Whether to remove rows with NA values
  - `True`: Remove rows with NA values (default)
  - `False`: Keep all rows

#### Example Usage:

**Method 1: Using the Vnstock class (Recommended)**
```python
from vnstock import Vnstock

# Initialize with stock symbol and source
stock = Vnstock().stock(symbol='VCI', source='VCI')

# Get annual cash flow statement in Vietnamese
annual_cash_flow = stock.finance.cash_flow(period='year', lang='vi', dropna=True)
print(annual_cash_flow)

# Get quarterly cash flow statement in English
quarterly_cash_flow = stock.finance.cash_flow(period='quarter', lang='en', dropna=False)
print(quarterly_cash_flow)
```

**Method 2: Direct import of Finance class**
```python
from vnstock import Finance

# Initialize Finance class directly
finance = Finance(symbol='VCI')

# Get annual cash flow statement in Vietnamese
annual_cash_flow = finance.cash_flow(period='year', lang='vi', dropna=True)
print(annual_cash_flow)

# Get quarterly cash flow statement in English
quarterly_cash_flow = finance.cash_flow(period='quarter', lang='en', dropna=False)
print(quarterly_cash_flow)
```

### 4. `ratio()` - Financial Ratios

Retrieves calculated financial ratios for a company, including profitability, liquidity, and solvency metrics.

#### Parameters:

- `period` (str): Time period of data, options:
  - `'year'`: Annual data (default)
  - `'quarter'`: Quarterly data
  
- `lang` (str): Language for column names and labels, options:
  - `'vi'`: Vietnamese (default)
  - `'en'`: English
  
- `dropna` (bool): Whether to remove rows with NA values
  - `True`: Remove rows with NA values (default)
  - `False`: Keep all rows

#### Example Usage:

**Method 1: Using the Vnstock class (Recommended)**
```python
from vnstock import Vnstock

# Initialize with stock symbol and source
stock = Vnstock().stock(symbol='VCI', source='VCI')

# Get annual financial ratios in Vietnamese
annual_ratios = stock.finance.ratio(period='year', lang='vi', dropna=True)
print(annual_ratios)

# Get quarterly financial ratios in English
quarterly_ratios = stock.finance.ratio(period='quarter', lang='en', dropna=True)
print(quarterly_ratios)
```

**Method 2: Direct import of Finance class**
```python
from vnstock import Finance

# Initialize Finance class directly
finance = Finance(symbol='VCI')

# Get annual financial ratios in Vietnamese
annual_ratios = finance.ratio(period='year', lang='vi', dropna=True)
print(annual_ratios)

# Get quarterly financial ratios in English
quarterly_ratios = finance.ratio(period='quarter', lang='en', dropna=True)
print(quarterly_ratios)
```

## Complete Example

Here's a complete example demonstrating how to use all the Finance class methods to retrieve various financial statements for a company:

```python
from vnstock import Vnstock
import pandas as pd

# Initialize
stock = Vnstock().stock(symbol='VCI', source='VCI')

# Get financial statements
balance_sheet = stock.finance.balance_sheet(period='year', lang='en', dropna=True)
income_stmt = stock.finance.income_statement(period='year', lang='en', dropna=True)
cash_flow = stock.finance.cash_flow(period='year', lang='en', dropna=True)
ratios = stock.finance.ratio(period='year', lang='en', dropna=True)

# Display the first few rows of each
print("Balance Sheet:")
print(balance_sheet.head())

print("\nIncome Statement:")
print(income_stmt.head())

print("\nCash Flow Statement:")
print(cash_flow.head())

print("\nFinancial Ratios:")
print(ratios.head())

# Save to CSV files
balance_sheet.to_csv('balance_sheet.csv')
income_stmt.to_csv('income_statement.csv')
cash_flow.to_csv('cash_flow.csv')
ratios.to_csv('financial_ratios.csv')
```

## Practical Analysis Example

Here's an example of how to analyze financial data using the vnstock Finance class methods:

```python
from vnstock import Vnstock
import pandas as pd
import matplotlib.pyplot as plt

# Initialize with two different companies for comparison
company1 = Vnstock().stock(symbol='VCI', source='VCI')
company2 = Vnstock().stock(symbol='SSI', source='VCI')

# Get income statements for both companies
income1 = company1.finance.income_statement(period='year', lang='en', dropna=True)
income2 = company2.finance.income_statement(period='year', lang='en', dropna=True)

# Get relevant rows (may vary based on actual data structure)
revenue1 = income1.loc[income1.index.str.contains('Revenue', case=False)]
revenue2 = income2.loc[income2.index.str.contains('Revenue', case=False)]

# Create a plot
plt.figure(figsize=(12, 6))
plt.plot(revenue1.columns, revenue1.iloc[0], 'o-', label='VCI Revenue')
plt.plot(revenue2.columns, revenue2.iloc[0], 's-', label='SSI Revenue')
plt.title('Revenue Comparison: VCI vs SSI')
plt.xlabel('Year')
plt.ylabel('Revenue (VND)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('revenue_comparison.png')
plt.show()
```

## Company Class Methods

The `Company` class provides methods to access company profile and information.

### 1. `profile()` - Company Profile Information

Retrieves detailed company profile information including business description, sector, industry, and key details.

#### Parameters:

- `lang` (str): Language for column names and labels, options:
  - `'vi'`: Vietnamese (default)
  - `'en'`: English

#### Example Usage:

**Method 1: Using the Vnstock class (Recommended)**
```python
from vnstock import Vnstock

# Initialize with stock symbol and source
stock = Vnstock().stock(symbol='VCI', source='VCI')

# Get company profile information
company_profile = stock.company.profile(lang='en')
print(company_profile)
```

**Method 2: Direct import of Company class**
```python
from vnstock import Company

# Initialize Company class directly
company = Company(symbol='VCI')

# Get company profile information
company_profile = company.profile(lang='en')
print(company_profile)
```

### 2. `officers()` - Company Officers/Management

Retrieves information about company officers and management team.

#### Parameters:

- `lang` (str): Language for column names and labels, options:
  - `'vi'`: Vietnamese (default)
  - `'en'`: English

#### Example Usage:

**Method 1: Using the Vnstock class (Recommended)**
```python
from vnstock import Vnstock

# Initialize with stock symbol and source
stock = Vnstock().stock(symbol='VCI', source='VCI')

# Get company officers information
management_team = stock.company.officers(lang='en')
print(management_team)
```

**Method 2: Direct import of Company class**
```python
from vnstock import Company

# Initialize Company class directly
company = Company(symbol='VCI')

# Get company officers information
management_team = company.officers(lang='en')
print(management_team)
```

### 3. `major_holders()` - Major Shareholders

Retrieves information about major shareholders of the company.

#### Parameters:

- `lang` (str): Language for column names and labels, options:
  - `'vi'`: Vietnamese (default)
  - `'en'`: English

#### Example Usage:

**Method 1: Using the Vnstock class (Recommended)**
```python
from vnstock import Vnstock

# Initialize with stock symbol and source
stock = Vnstock().stock(symbol='VCI', source='VCI')

# Get major shareholders information
major_shareholders = stock.company.major_holders(lang='en')
print(major_shareholders)
```

**Method 2: Direct import of Company class**
```python
from vnstock import Company

# Initialize Company class directly
company = Company(symbol='VCI')

# Get major shareholders information
major_shareholders = company.major_holders(lang='en')
print(major_shareholders)
```

## Stock Screener Functionality

The `Screener` class allows you to filter stocks based on various financial, technical, and market criteria.

### Using the Stock Screener

The main method of the Screener class is the `stock()` method, which allows filtering based on custom parameters.

#### Method: `stock(params, limit)`

- `params` (dict): Dictionary containing filtering parameters
- `limit` (int): Maximum number of results to return (default: 1700)

#### Example Usage:

**Method 1: Using the Vnstock class (Recommended)**
```python
from vnstock import Vnstock

# Initialize Vnstock
stock = Vnstock()

# Basic usage with exchange filter
df = stock.screener.stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=1700)
print(df.head())

# Advanced filtering with multiple parameters
params = {
    "exchangeName": "HOSE,HNX", 
    "marketCap": (100, 1000),  # Market cap between 100-1000 billion VND
    "dividendYield": (5, 10),  # Dividend yield between 5-10%
    "roe": (15, 50)            # ROE between 15-50%
}
filtered_stocks = stock.screener.stock(params=params, limit=500)
print(filtered_stocks)
```

**Method 2: Direct import of Screener class**
```python
from vnstock import Screener

# Initialize Screener class directly
screener = Screener()

# Basic usage with exchange filter
df = screener.stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=1700)
print(df.head())

# Advanced filtering with multiple parameters
params = {
    "exchangeName": "HOSE,HNX", 
    "marketCap": (100, 1000),  # Market cap between 100-1000 billion VND
    "dividendYield": (5, 10),  # Dividend yield between 5-10%
    "roe": (15, 50)            # ROE between 15-50%
}
filtered_stocks = screener.stock(params=params, limit=500)
print(filtered_stocks)
```

### Common Screening Parameters

The following parameters can be used in the `params` dictionary to filter stocks:

1. **Market Parameters**:
   - `exchangeName`: Stock exchange(s) - "HOSE", "HNX", "UPCOM" (use commas for multiple)
   - `marketCap`: Market capitalization range in billions VND - tuple (min, max)
   - `industry`: Industry name or code
   - `hasFinancialReport`: 1 for companies with recent financial reports, 0 otherwise

2. **Financial Metrics**:
   - `roe`: Return on Equity range - tuple (min, max)
   - `pe`: Price-to-Earnings ratio range - tuple (min, max)
   - `pb`: Price-to-Book ratio range - tuple (min, max)
   - `dividendYield`: Dividend yield percentage range - tuple (min, max)
   - `epsGrowth1Year`: EPS growth in the last year range - tuple (min, max)
   - `lastQuarterProfitGrowth`: Last quarter profit growth range - tuple (min, max)

3. **Technical Indicators**:
   - `avgTradingValue20Day`: 20-day average trading value range - tuple (min, max)
   - `relativeStrength1Month`: 1-month relative strength range - tuple (min, max)
   - `priceToSma20`: Price to 20-day Simple Moving Average ratio range - tuple (min, max)

4. **Ratings and Analysis**:
   - `stockRating`: Stock rating score (1-5, with 5 being best)
   - `businessModel`: Business model rating (1-5)
   - `businessOperation`: Business operation rating (1-5)
   - `financialHealth`: Financial health rating (1-5)

### Example Use Case: CANSLIM Stock Screening

The CANSLIM investment strategy focuses on growth stocks. Here's how to implement it with vnstock:

**Method 1: Using the Vnstock class (Recommended)**
```python
from vnstock import Vnstock
import pandas as pd

# Initialize
stock = Vnstock()

# CANSLIM parameters
params = {
    "exchangeName": "HOSE,HNX,UPCOM",
    "epsGrowth1Year": (25, 999),            # Strong earnings growth
    "lastQuarterProfitGrowth": (20, 999),   # Recent quarterly growth
    "roe": (15, 999),                       # High return on equity
    "avgTradingValue20Day": (5000000, 999999999999),  # High trading volume
    "relativeStrength1Month": (10, 999)     # Strong relative strength
}

# Get CANSLIM stocks
canslim_stocks = stock.screener.stock(params=params, limit=100)

# Sort by EPS growth for best candidates
canslim_stocks = canslim_stocks.sort_values(by='epsGrowth1Year', ascending=False)

print("Top 10 CANSLIM Stock Candidates:")
print(canslim_stocks.head(10))

# Export to CSV
canslim_stocks.to_csv('canslim_stocks.csv')
```

**Method 2: Direct import of Screener class**
```python
from vnstock import Screener
import pandas as pd

# Initialize
screener = Screener()

# CANSLIM parameters
params = {
    "exchangeName": "HOSE,HNX,UPCOM",
    "epsGrowth1Year": (25, 999),            # Strong earnings growth
    "lastQuarterProfitGrowth": (20, 999),   # Recent quarterly growth
    "roe": (15, 999),                       # High return on equity
    "avgTradingValue20Day": (5000000, 999999999999),  # High trading volume
    "relativeStrength1Month": (10, 999)     # Strong relative strength
}

# Get CANSLIM stocks
canslim_stocks = screener.stock(params=params, limit=100)

# Sort by EPS growth for best candidates
canslim_stocks = canslim_stocks.sort_values(by='epsGrowth1Year', ascending=False)

print("Top 10 CANSLIM Stock Candidates:")
print(canslim_stocks.head(10))

# Export to CSV
canslim_stocks.to_csv('canslim_stocks.csv')
```

## Complete Example: Multi-Class Analysis

Here's a complete example demonstrating how to use all the classes together to perform a comprehensive analysis:

```python
from vnstock import Vnstock
import pandas as pd
import matplotlib.pyplot as plt

# Initialize with two different companies for comparison
stock1 = Vnstock().stock(symbol='VCB', source='VCI')
stock2 = Vnstock().stock(symbol='TCB', source='VCI')

# 1. Get company profile information
profile1 = stock1.company.profile(lang='en')
profile2 = stock2.company.profile(lang='en')
print("Company Profiles:")
print(f"VCB: {profile1['organShortName'].values[0]}")
print(f"TCB: {profile2['organShortName'].values[0]}")

# 2. Get financial data
income1 = stock1.finance.income_statement(period='year', lang='en', dropna=True)
income2 = stock2.finance.income_statement(period='year', lang='en', dropna=True)

ratios1 = stock1.finance.ratio(period='year', lang='en', dropna=True)
ratios2 = stock2.finance.ratio(period='year', lang='en', dropna=True)

# 3. Perform screening for similar companies in the banking sector
screener = Vnstock().screener
bank_params = {
    "exchangeName": "HOSE,HNX,UPCOM",
    "industryName": "Banking",
    "marketCap": (50000, 999999999),  # Large banks only
    "roe": (10, 999)                  # Good profitability
}
banks = screener.stock(params=bank_params, limit=10)

# 4. Compare ROE of the banks
plt.figure(figsize=(12, 6))
plt.bar(banks['ticker'], banks['roe'])
plt.title('ROE Comparison of Vietnamese Banks')
plt.xlabel('Bank Ticker')
plt.ylabel('Return on Equity (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('bank_roe_comparison.png')

# 5. Create final analysis report
print("\nBank Analysis Results:")
print(f"Total banks analyzed: {len(banks)}")
print(f"Highest ROE Bank: {banks.loc[banks['roe'].idxmax(), 'ticker']} with ROE of {banks['roe'].max():.2f}%")
print(f"Average Banking Sector ROE: {banks['roe'].mean():.2f}%")
print(f"VCB ROE: {banks.loc[banks['ticker'] == 'VCB', 'roe'].values[0]:.2f}%")
print(f"TCB ROE: {banks.loc[banks['ticker'] == 'TCB', 'roe'].values[0]:.2f}%")

# Export results
banks.to_csv('vietnamese_banks_analysis.csv')
```
# Example Use Cases for vnstock Library

Here are practical example use cases demonstrating how to use the vnstock library for Vietnamese stock market analysis:

## 1. Setting Up and Basic Usage

```python
from vnstock import Vnstock
# Initialize with a default stock symbol and data source
stock = Vnstock().stock(symbol='ACB', source='VCI')
```

## 2. Market Overview Analysis

**Get list of all VN30 stocks for market analysis:**
```python
vn30_stocks = stock.listing.symbols_by_group('VN30')
print(f"There are {len(vn30_stocks)} stocks in the VN30 index")
```

**Create a market dashboard for top banking stocks:**
```python
banking_board = stock.trading.price_board(['VCB','ACB','TCB','BID','MBB'])
# Format as a dashboard with key metrics
banking_board_display = banking_board[['listing.symbol', 'match.match_price', 
                                       'match.price_change_ratio', 'match.total_volume']]
```

## 3. Technical Analysis

**Get historical data for moving average analysis:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Get 1-year historical data for VNIndex
vnindex_hist = stock.quote.history(symbol='VNINDEX', start='2024-01-01', end='2025-03-19', interval='1D')

# Calculate moving averages
vnindex_hist['MA20'] = vnindex_hist['close'].rolling(window=20).mean()
vnindex_hist['MA50'] = vnindex_hist['close'].rolling(window=50).mean()
vnindex_hist['MA200'] = vnindex_hist['close'].rolling(window=200).mean()

# Plot the chart
plt.figure(figsize=(12, 6))
plt.plot(vnindex_hist['time'], vnindex_hist['close'], label='VNINDEX')
plt.plot(vnindex_hist['time'], vnindex_hist['MA20'], label='MA20')
plt.plot(vnindex_hist['time'], vnindex_hist['MA50'], label='MA50')
plt.plot(vnindex_hist['time'], vnindex_hist['MA200'], label='MA200')
plt.title('VNINDEX with Moving Averages')
plt.legend()
plt.grid(True)
```

**Analyze intraday trading pattern:**
```python
# Get intraday data to analyze trading volume by time of day
intraday = stock.quote.intraday(symbol='ACB', page_size=10000)

# Group data by hour to see time-based trading patterns
intraday['hour'] = pd.to_datetime(intraday['time']).dt.hour
hourly_volume = intraday.groupby('hour')['volume'].sum()
```

## 4. Fundamental Analysis

**Evaluate company financial health:**
```python
# Get balance sheet and income statement data
balance_sheet = stock.finance.balance_sheet(period='year', lang='en')
income_stmt = stock.finance.income_statement(period='year', lang='en')

# Calculate basic financial metrics year-over-year
recent_years = balance_sheet['yearReport'].unique()[-5:]  # Last 5 years
filtered_bs = balance_sheet[balance_sheet['yearReport'].isin(recent_years)]
filtered_is = income_stmt[income_stmt['yearReport'].isin(recent_years)]

# Calculate debt-to-equity trend
filtered_bs['debt_to_equity'] = (filtered_bs['TOTAL ASSETS (Bn. VND)'] - filtered_bs['OWNER\'S EQUITY(Bn.VND)']) / filtered_bs['OWNER\'S EQUITY(Bn.VND)']

# Plot financial metrics trend
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(filtered_is['yearReport'], filtered_is['Revenue (Bn. VND)'], marker='o', label='Revenue')
plt.plot(filtered_is['yearReport'], filtered_is['Attribute to parent company (Bn. VND)'], marker='o', label='Net Profit')
plt.title('Revenue and Profit Trend')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(filtered_bs['yearReport'], filtered_bs['debt_to_equity'], marker='o')
plt.title('Debt-to-Equity Ratio')
```

## 5. Competitor Analysis

```python
# Compare financial ratios across banking sector
banks = ['VCB', 'ACB', 'TCB', 'BID', 'MBB']
bank_ratios = {}

for bank in banks:
    bank_stock = Vnstock().stock(symbol=bank, source='VCI')
    ratios = bank_stock.finance.ratio(period='year', lang='en').iloc[0]  # Most recent year
    bank_ratios[bank] = {
        'ROE': ratios['ROE (%)'],
        'ROA': ratios['ROA (%)'],
        'P/E': ratios['P/E'],
        'P/B': ratios['P/B']
    }

bank_comparison = pd.DataFrame(bank_ratios).T
print("Banking Sector Comparison:")
print(bank_comparison)
```

## 6. Investment Screening

```python
# Screen for high-ROE stocks with low P/B ratio
screener = stock.screener.stock(params={"exchangeName": "HOSE"}, limit=100)
filtered_stocks = screener[(screener['roe'] > 15) & 
                          (screener['price_vs_sma100'] == 'Giá nằm trên SMA(100)') &
                          (screener['last_quarter_profit_growth'] > 0)]

# Sort by market cap to find the largest companies meeting criteria
investment_candidates = filtered_stocks.sort_values('market_cap', ascending=False).head(10)
print("Top 10 Investment Candidates:")
print(investment_candidates[['ticker', 'market_cap', 'roe', 'last_quarter_profit_growth']])
```

## 7. Ownership Analysis

```python
# Analyze foreign ownership in a company
company_info = stock.company
ownership = company_info.shareholders()

# Calculate total percent owned by foreigners
foreign_owners = ownership[ownership['share_holder'].str.contains('Limited|Ltd|LLC|Fund|Corp', case=False, na=False)]
foreign_ownership_pct = foreign_owners['share_own_percent'].sum()

print(f"Foreign ownership percentage: {foreign_ownership_pct:.2%}")
```

## 8. Event Impact Analysis

```python
# Analyze stock performance around dividend events
events = stock.company.events()
dividend_events = events[events['event_list_name'] == 'Trả cổ tức bằng tiền mặt']

for index, event in dividend_events.iterrows():
    # Get stock prices around the ex-dividend date
    exright_date = event['exright_date']
    start_date = pd.to_datetime(exright_date) - pd.Timedelta(days=30)
    end_date = pd.to_datetime(exright_date) + pd.Timedelta(days=30)
    
    # Convert dates to the required string format 'YYYY-MM-DD'
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Get price history around the event
    event_prices = stock.quote.history(start=start_str, end=end_str, interval='1D')
    
    # Additional analysis on price behavior before/after ex-dividend date
    # ...
```

## 9. Seasonal Performance Analysis

```python
# Analyze quarterly performance patterns over multiple years
quarterly_data = stock.finance.income_statement(period='quarter', lang='en')

# Group by quarter to analyze seasonal patterns
quarterly_data['quarter'] = quarterly_data['lengthReport']
quarterly_perf = quarterly_data.groupby(['yearReport', 'quarter'])['Revenue YoY (%)'].mean().unstack()

# Plot seasonal performance by quarter
quarterly_perf.mean().plot(kind='bar', figsize=(10, 6), title='Average Quarterly Performance')
```

## 10. News Sentiment Analysis

```python
# Basic news sentiment analysis
import re
from textblob import TextBlob  # You'll need to install this package

# Get recent news
recent_news = stock.company.news().head(20)

# Simple sentiment analysis function
def analyze_sentiment(text):
    if isinstance(text, str):
        # Clean text
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        return TextBlob(text).sentiment.polarity  # Return polarity (-1 to 1)
    return 0

# Apply sentiment analysis to news titles
recent_news['sentiment'] = recent_news['news_title'].apply(analyze_sentiment)

# Classify sentiment
recent_news['sentiment_category'] = recent_news['sentiment'].apply(
    lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

# Count by sentiment
sentiment_count = recent_news['sentiment_category'].value_counts()
print("News Sentiment Analysis:")
print(sentiment_count)
```

These examples demonstrate the versatility of the vnstock library for various types of stock market analysis in the Vietnamese context, from technical and fundamental analysis to screening, competitor comparison, and event impact studies.