uv import pandas as pd
import numpy as np
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.io import output_file, show, save
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

# Load the data or create a similar DataFrame as in the original code
# In the original code, there was a Ratio DataFrame with selected columns
# Here we'll recreate a similar structure or load it from a file if available

try:
    # Try to load the Ratio DataFrame if it exists
    Ratio = pd.read_csv('ratio_data.csv')
except FileNotFoundError:
    # If the file doesn't exist, we'll create a mock DataFrame for demonstration
    print("Ratio data file not found. Creating mock data for demonstration.")
    np.random.seed(42)
    mock_data = {
        ('Chỉ tiêu khả năng sinh lợi', 'ROE (%)'): np.random.normal(15, 5, 100),
        ('Chỉ tiêu cơ cấu nguồn vốn', 'Nợ/VCSH'): np.random.normal(1.5, 0.5, 100),
        ('Chỉ tiêu hiệu quả hoạt động', 'Vòng quay tài sản'): np.random.normal(0.8, 0.2, 100),
        ('Chỉ tiêu khả năng sinh lợi', 'Biên lợi nhuận ròng (%)'): np.random.normal(10, 3, 100),
        ('Chỉ tiêu thanh khoản', 'Chỉ số thanh toán hiện thời'): np.random.normal(2, 0.5, 100),
        ('Chỉ tiêu định giá', 'P/S'): np.random.normal(1.2, 0.3, 100),
    }
    Ratio = pd.DataFrame(mock_data)

# Define the selected columns as in the original code
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
df_pair = df_pair.dropna()

# Convert the pandas DataFrame to an ArviZ InferenceData object
# ArviZ requires a specific format, so we'll create a dict of the data first
data_dict = {}
for col in df_pair.columns:
    data_dict[col] = df_pair[col].values

# Create an ArviZ InferenceData object
inference_data = az.convert_to_inference_data({"posterior": data_dict})

# Create the pairplot using ArviZ with Bokeh backend
print("Creating ArviZ pairplot with Bokeh backend...")
az.style.use("arviz-darkgrid")
az.rcParams["plot.backend"] = "bokeh"

# Generate the pairplot
pair_plot = az.plot_pair(
    inference_data,
    var_names=list(data_dict.keys()),
    kind="scatter",
    marginals=True,
    figsize=(12, 10),
    textsize=12,
    plot_kwargs={"alpha": 0.6, "color": "blue"},
)

# Save the plot to an HTML file
output_file("financial_metrics_pairplot.html")
save(pair_plot)
print("Pairplot saved to 'financial_metrics_pairplot.html'")

# Display a preview of the data
print("\nPreview of the data used for the pairplot:")
print(df_pair.head())

print("\nTo view the interactive pairplot, open 'financial_metrics_pairplot.html' in a web browser.")
print("You can also integrate this into a larger Bokeh layout for your application.")
