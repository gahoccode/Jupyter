import pandas as pd
from vnstock_ezchart import *
from vnstock import Vnstock

# Set date range
start_date = '2024-01-01'
end_date = '2025-03-19'
interval = '1D'
stock = Vnstock().stock(symbol='REE', source='VCI')

CashFlow = stock.finance.cash_flow(period='year')

BalanceSheet = stock.finance.balance_sheet(period='year', lang='en', dropna=True)

IncomeStatement = stock.finance.income_statement(period='year', lang='en', dropna=True)


from vnstock import Vnstock
import warnings
warnings.filterwarnings("ignore")
company = Vnstock().stock(symbol='REE', source='TCBS').company
overview = company.overview()
overview.head()

CashFlow['Dividends paid']

# Levered Free Cash Flow (accounts for debt repayments/receipts)
CashFlow['Levered Free Cash Flow'] = (
    CashFlow['Net cash inflows/outflows from operating activities'] 
    - CashFlow['Purchase of fixed assets']
    + CashFlow['Proceeds from disposal of fixed assets']
    - (CashFlow['Repayment of borrowings'] - CashFlow['Proceeds from borrowings'])
)

CashFlow['Levered Free Cash Flow']

dividend_coverage_ratio = CashFlow['Levered Free Cash Flow'] / CashFlow['Dividends paid'].abs()

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from datetime import datetime

# Check if API key is already set as an environment variable
if "OPENAI_API_KEY" not in os.environ:
    # If not set, you may want to load from .env file
    # from dotenv import load_dotenv
    # load_dotenv()
    
    # Or set it directly (not recommended for production code)
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize the ChatGPT model
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Function to format DataFrame as a string for the prompt
def format_dataframe(df):
    return df.to_string()

# Create a prompt template for comprehensive financial analysis
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a financial analyst expert. Analyze the following financial statements and ratios:
    
    BALANCE SHEET:
    {balance_sheet}
    
    INCOME STATEMENT:
    {income_statement}
    
    CASH FLOW STATEMENT:
    {cash_flow}
    
    KEY FINANCIAL RATIOS:
    {financial_ratios}
    
    Provide the following comprehensive analysis:
    1. Analysis of asset composition, liabilities and equity structure
    2. Profitability analysis (margins, ROCE, ROE, ROIC)
    3. Cash flow quality and trends
    4. Evaluation of liquidity and solvency ratios
    5. Assessment of working capital management
    6. Year-over-year changes in key financial items
    7. Integrated analysis showing relationships between the three statements
    8. Dividend sustainability analysis based on the dividend coverage ratio
    
    Be specific in your analysis and provide actionable insights.""")
])

# Analyze function using all three financial statements
def analyze_financial_statements(balance_sheet_df, income_statement_df, cash_flow_df):
    # Calculate key financial ratios
    financial_ratios = pd.DataFrame({
        'Dividend Coverage Ratio': cash_flow_df['Levered Free Cash Flow'] / cash_flow_df['Dividends paid'].abs()
        # Add other ratios as needed, maintaining the same index structure
    })
    
    # Format the DataFrames as strings
    balance_sheet_string = format_dataframe(balance_sheet_df)
    income_statement_string = format_dataframe(income_statement_df)
    cash_flow_string = format_dataframe(cash_flow_df)
    financial_ratios_string = format_dataframe(financial_ratios)
    
    # Create a chain using the pipe operator
    chain = prompt_template | llm
    
    # Run the chain with all inputs
    result = chain.invoke({
        "balance_sheet": balance_sheet_string,
        "income_statement": income_statement_string,
        "cash_flow": cash_flow_string,
        "financial_ratios": financial_ratios_string
    })
    
    return result.content

# Now use your specific DataFrames
# Assuming these DataFrames are already defined
analysis = analyze_financial_statements(
    BalanceSheet, 
    IncomeStatement,  
    CashFlow  
)

# Print analysis to notebook
print(analysis)

# Create markdown formatted content
markdown_content = f"""# Comprehensive Financial Analysis

## Analysis
{analysis}

---
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

# Ensure output directory exists
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Save the output as a markdown file in the output directory
output_filename = f"{output_dir}/financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
with open(output_filename, 'w', encoding='utf-8') as md_file:
    md_file.write(markdown_content)

print(f"\nAnalysis saved to {output_filename}")