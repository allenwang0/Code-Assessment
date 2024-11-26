import requests
import pandas as pd


def fetch_fred_yield(series_id, start_date="2023-01-01", end_date="2023-12-31", api_key="your_api_key"):
    # Define the API URL and parameters here


    # Make an API request and handle errors

    # Convert response JSON into a DataFrame, setting 'date' as index
    # Convert the 'value' column to numeric and rename it based on series_id

    return df[[series_id]]  # Final DataFrame with date index and yield column


# List of Treasury yield series IDs on FRED
tenor_series_ids = [
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1",  # Short-term yields
    "DGS2", "DGS3", "DGS5",  # Medium-term yields
    "DGS7", "DGS10", "DGS20", "DGS30"  # Long-term yields
]

# Initialize API key

# Fetch data for each tenor, store in a dictionary of DataFrames

# Combine all DataFrames into a single DataFrame, joining on the date index

# Print the number of rows in the final DataFrame