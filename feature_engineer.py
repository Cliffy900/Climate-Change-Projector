import pandas as pd
import os
from data_loader import load_and_process_data # Assuming data_loader.py is available

# =============================================================================
# Feature Engineering for Time Series Analysis
#
# This script creates a critical feature, 'TimeIndex', which is essential
# for training a linear regression model on time series data.
# =============================================================================

def create_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a 'TimeIndex' feature for each distinct time series in the dataset.

    The linear regression model requires the independent variable (time) to be
    a continuous, zero-indexed integer series, rather than the raw year (1961, 1962, ...).

    Args:
        df (pd.DataFrame): The cleaned DataFrame from the data loader,
                           containing ['Area', 'Months', 'Year', 'TempChange'].

    Returns:
        pd.DataFrame: The DataFrame with the added 'TimeIndex' column.
    """
    if df.empty:
        print("Input DataFrame is empty. Cannot perform feature engineering.")
        return df

    print("Creating 'TimeIndex' feature...")

    # Group the data by Area and Months (e.g., 'World' and 'Annual')
    # and apply a cumulative count to the Year within each group.
    # The minimum value in the Year column (1961) will map to index 0.
    df['TimeIndex'] = df.groupby(['Area', 'Months'])['Year'].rank(method='min').astype(int) - 1

    print("Feature engineering complete. TimeIndex created.")
    return df

# =============================================================================
# Demonstration of Usage
# =============================================================================
if __name__ == "__main__":
    print("--- Executing Feature Engineer ---")

    # NOTE: In a real environment, you would import and use load_and_process_data()
    # For demonstration, we create a mock DataFrame similar to the loader's output:
    try:
        data_cleaned = load_and_process_data()
    except NameError:
        print("Note: data_loader.py not found or configured. Using mock data for demonstration.")
        data_cleaned = pd.DataFrame({
            'Area': ['World', 'World', 'World', 'World', 'Australia', 'Australia'],
            'Months': ['Annual', 'Annual', 'Annual', 'Annual', 'Annual', 'Annual'],
            'Year': [1961, 1962, 1963, 1964, 1961, 1962],
            'TempChange': [0.01, 0.05, 0.12, 0.20, -0.10, -0.05]
        })

    # Apply the feature engineering function
    data_features = create_time_index(data_cleaned)

    if not data_features.empty:
        print("\nProcessed Data Head (Showing TimeIndex):")
        print(data_features[['Area', 'Year', 'TempChange', 'TimeIndex']].head(10))

        # Verification check: The first year (1961 in this data) should be TimeIndex 0
        if not data_features.empty and data_features.iloc[0]['TimeIndex'] == 0:
             print("\nVerification: TimeIndex successfully started at 0 for the first record.")
        else:
             print("\nVerification: TimeIndex check failed or DataFrame is empty.")