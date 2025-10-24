import pandas as pd
import numpy as np

# =============================================================================
# Configuration and Constants
# =============================================================================
# The window size for the rolling mean calculation (e.g., 5-year smoothing)
SMOOTHING_WINDOW_SIZE = 5


def smooth_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a rolling mean filter to the 'TempChange' column for each unique
    time series (Area and Months combination) to smooth out year-to-year noise.

    A centered rolling window is used for better representation, requiring a
    window size that is odd (e.g., 5, 7, 9) so the current year is centered.

    Args:
        df (pd.DataFrame): The DataFrame containing ['Area', 'Months', 'TempChange'].

    Returns:
        pd.DataFrame: The original DataFrame with a new column,
                      'SmoothedTempChange', added.
    """
    if df.empty:
        print("Input DataFrame is empty. Cannot perform smoothing.")
        return df

    if 'TempChange' not in df.columns:
        print("ERROR: 'TempChange' column not found. Ensure data_loader was executed.")
        return df

    print(f"Applying {SMOOTHING_WINDOW_SIZE}-year centered rolling mean smoothing...")

    # Group the data by Area and Months (the unique time series)
    # and apply the rolling mean within each group.
    # The 'center=True' argument ensures the smoothing window is centered on the current row.
    df['SmoothedTempChange'] = df.groupby(['Area', 'Months'])['TempChange'] \
        .rolling(window=SMOOTHING_WINDOW_SIZE, center=True, min_periods=1) \
        .mean() \
        .reset_index(level=[0, 1], drop=True)

    # Note: Rolling mean introduces NaNs at the beginning/end for window centering.
    # The model training script expects the data to be clean, but for visualization
    # and robust analysis, we keep the NaNs, and the model training will handle
    # dropping them if necessary, but the pipeline usually handles full datasets.
    # Since we use min_periods=1, there should be no NaNs, only the edge data is
    # smoothed over a smaller window.

    print("Smoothing completed. 'SmoothedTempChange' column added.")
    return df


# =============================================================================
# Demonstration of Usage
# =============================================================================
if __name__ == "__main__":
    # Mock data structure to demonstrate functionality
    mock_data = pd.DataFrame({
        'Area': ['World'] * 10,
        'Months': ['Annual'] * 10,
        'Year': range(1961, 1971),
        'TempChange': [0.01, 0.05, 0.12, 0.20, 0.15, 0.10, 0.05, 0.03, 0.06, 0.11]
    })

    print("--- Executing Data Smoother on Mock Data ---")
    data_smoothed = smooth_time_series(mock_data)

    if not data_smoothed.empty:
        print("\nProcessed Data Head (Showing SmoothedTrend):")
        print(data_smoothed[['Year', 'TempChange', 'SmoothedTempChange']].head(10))

        # Verify the calculation for the center point (1963 should be avg of 1961-1965)
        expected_1963_avg = (0.01 + 0.05 + 0.12 + 0.20 + 0.15) / 5
        print(
            f"\nVerification (1963 Smoothed): Expected {expected_1963_avg:.4f}, Actual {data_smoothed.loc[2, 'SmoothedTempChange']:.4f}")

