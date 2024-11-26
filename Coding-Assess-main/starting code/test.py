import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Part 1: API Pulling, Data Wrangling, and Visualizations


# Part 1.1: API Setup
def fetch_fred_yield(series_id, start_date="2023-01-01", end_date="2023-12-31", api_key="your_api_key"):
    # Define the API URL and parameters here
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }

    # Make an API request and handle errors
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    # Convert response JSON into a DataFrame, setting 'date' as index
    data = response.json().get("observations", [])
    df = pd.DataFrame(data)

    # Convert the 'value' column to numeric and rename it based on series_id
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.rename(columns={"value": series_id}, inplace=True)

    return df[[series_id]]  # Final DataFrame with date index and yield column

# Part 1.2: Data Pulling


# List of Treasury yield series IDs on FRED
tenor_series_ids = [
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1",  # Short-term yields
    "DGS2", "DGS3", "DGS5",  # Medium-term yields
    "DGS7", "DGS10", "DGS20", "DGS30"  # Long-term yields
]

# Initialize API key
api_key = "015eaf80d4964ba2d2c184346cd4a567"

# Fetch data for each tenor, store in a dictionary of DataFrames
yield_data = {}
for series_id in tenor_series_ids:
    yield_data[series_id] = fetch_fred_yield(series_id, api_key=api_key)

# Part 1.3: Data Storage
# Combine all DataFrames into a single DataFrame, joining on the date index
combined_yield_data = pd.concat(yield_data.values(), axis=1)

# Print the number of rows in the final DataFrame
print(f"Number of rows in the final DataFrame: {len(combined_yield_data)}")
print(combined_yield_data.head())

latest_treasury_yields = combined_yield_data.iloc[-1].dropna()
latest_treasury_yields.index = [float(maturity.replace("DGS", "").replace("MO", "").replace("Y", ""))
                                for maturity in latest_treasury_yields.index]

treasury_curve = pd.DataFrame({
    "Maturity": latest_treasury_yields.index,
    "Yield": latest_treasury_yields.values
})

# Part 1.4: Spread Calculation
bonds_yields_path = '/Users/allenwang/Downloads/Part 1. bonds_yields.xlsx'
loan_data_path = '/Users/allenwang/Downloads/Part 2. loan_data_final.csv'

# Load Excel file (bonds yields)
bonds_yields = pd.read_excel(bonds_yields_path)

bonds_yields.rename(columns={"WAL (years)": "WAL", "Yield (%)": "Yield"}, inplace=True)

# Load CSV file (loan data)
loan_data = pd.read_csv(loan_data_path)


def interpolate_treasury_yield(wal, treasury_df):
    maturities = treasury_df["Maturity"].values
    yields = treasury_df["Yield"].values
    return np.interp(wal, maturities, yields)


def calculate_spread(row, treasury_df):
    interpolated_yield = interpolate_treasury_yield(row["WAL"], treasury_df)
    return row["Yield"] - interpolated_yield

# Apply spread calculation
bonds_yields["Spread"] = bonds_yields.apply(
    lambda row: calculate_spread(row, treasury_df=treasury_curve), axis=1
)

# Display the bonds data with spreads
print(bonds_yields)


# Part 1.5: Visualizations

# Spreads vs. Sector Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=bonds_yields, x="Sector", y="Spread", palette="Set2")
plt.title("Spread Distribution by Sector", fontsize=16)
plt.xlabel("Sector", fontsize=14)
plt.ylabel("Spread (bps)", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Spreads vs. WAL Scatterplot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=bonds_yields, x="WAL", y="Spread", hue="Sector", palette="Set1", s=100, edgecolor="k")
plt.title("Spread vs Weighted Average Life (WAL)", fontsize=16)
plt.xlabel("Weighted Average Life (Years)", fontsize=14)
plt.ylabel("Spread (bps)", fontsize=14)
plt.legend(title="Sector", fontsize=10, loc="upper right")
plt.grid(axis="both", linestyle="--", alpha=0.7)
plt.show()

# Average Spread vs. Sector Barchart
plt.figure(figsize=(12, 6))
sns.barplot(data=bonds_yields, x="Sector", y="Spread", ci=None, palette="viridis")
plt.title("Average Spread by Sector", fontsize=16)
plt.xlabel("Sector", fontsize=14)
plt.ylabel("Average Spread (bps)", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

#  Spreads Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=bonds_yields, x="Sector", y="Spread", palette="coolwarm", inner="quartile", scale="width")
plt.title("Spread Distribution by Sector (Violin Plot)", fontsize=16)
plt.xlabel("Sector", fontsize=14)
plt.ylabel("Spread (bps)", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()