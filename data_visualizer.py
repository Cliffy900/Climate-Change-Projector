import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import r2_score

# Imports for data preparation (reusing the established pipeline)
from data_loader import load_and_process_data
from feature_engineer import create_time_index
from data_smoother import smooth_time_series

# =============================================================================
# Configuration and Constants
# =============================================================================
COEFFICIENTS_FILE = "linear_model_coefficients.json"
PLOT_OUTPUT_FILE = "world_annual_temperature_trend.png"
BASE_YEAR = 1961
# Default area and period to plot
DEFAULT_AREA = "World"
DEFAULT_MONTHS = "Annual"


# =============================================================================
# Data Loading and Preparation (Reusing Pipeline)
# =============================================================================

def load_and_prepare_data() -> pd.DataFrame:
    """
    Loads raw data, melts, cleans, creates the TimeIndex, and applies smoothing.
    Replicates the data pipeline used for training and evaluation.
    """
    print("1. Loading raw data...")
    df_cleaned = load_and_process_data()

    print("2. Creating TimeIndex feature...")
    df_indexed = create_time_index(df_cleaned)

    print("3. Applying 5-year smoothing...")
    df_smoothed = smooth_time_series(df_indexed)

    return df_smoothed


def load_coefficients() -> dict:
    """
    Loads the linear regression coefficients (intercept and slope) from the JSON file.
    """
    try:
        with open(COEFFICIENTS_FILE, 'r') as f:
            coefficients = json.load(f)
        print(f"Successfully loaded coefficients from {COEFFICIENTS_FILE}.")
        return coefficients
    except FileNotFoundError:
        print(f"ERROR: Model coefficients file not found at '{COEFFICIENTS_FILE}'.")
        print("Please ensure 'train_test_split.py' was run successfully.")
        return {}
    except Exception as e:
        print(f"An error occurred while loading coefficients: {e}")
        return {}


# =============================================================================
# Core Visualization Function
# =============================================================================

def plot_trend(  # Renamed function from plot_model_and_data to plot_trend
        df_data: pd.DataFrame,
        coefficients: dict,
        area: str,
        months: str,
        output_path: str
):
    """
    Generates and saves a visualization for a specific time series, showing
    raw data, smoothed data, and the linear regression line.
    """
    key = f"{area}|{months}"

    if key not in coefficients:
        print(f"ERROR: Coefficients for {key} not found. Cannot plot.")
        return

    # Filter data for the specific area and month
    df_subset = df_data[(df_data['Area'] == area) & (df_data['Months'] == months)]

    if df_subset.empty:
        print(f"ERROR: No data found for {area} ({months}). Cannot plot.")
        return

    # Extract coefficients
    intercept = coefficients[key]['intercept']
    slope = coefficients[key]['slope']

    # --- Prepare Data for Plotting ---

    # 1. Linear Regression Line (Prediction based on TimeIndex)
    # The TimeIndex for prediction spans the whole data range
    min_time = df_subset['TimeIndex'].min()
    max_time = df_subset['TimeIndex'].max()

    # Create prediction points for the line
    x_line = np.arange(min_time, max_time + 1)
    y_line = intercept + slope * x_line

    # Convert TimeIndex back to Year for the X-axis
    x_years = x_line + BASE_YEAR

    # 2. Calculate R-squared for display (R-squared is assumed to be training R2
    # as the raw training data is not strictly isolated here, but we can compute
    # it easily for the full range for display purposes.)

    # Compute R-squared on the Smoothed Data vs. Model Fit
    y_smoothed = df_subset['SmoothedTempChange'].values
    y_model_fit = intercept + slope * df_subset['TimeIndex'].values
    r_squared = r2_score(y_smoothed, y_model_fit)

    # --- Start Plotting ---
    plt.figure(figsize=(12, 6))

    # Scatter plot of the raw data (TempChange)
    # Use .tolist() to convert Series/Arrays to standard lists to satisfy strict type checkers
    plt.scatter(
        df_subset['Year'].tolist(),
        df_subset['TempChange'].tolist(),
        label='Raw Annual Data',
        color='#1f77b4',  # Muted blue
        alpha=0.5,
        marker='o',
        s=20
    )

    # Line plot of the smoothed data (SmoothedTempChange)
    # Use .tolist() to convert Series/Arrays to standard lists
    plt.plot(
        df_subset['Year'].tolist(),
        df_subset['SmoothedTempChange'].tolist(),
        label='5-Year Smoothed Trend',
        color='#ff7f0e',  # Orange
        linewidth=2.5,
        linestyle='--'
    )

    # Plot the Linear Regression Trend Line
    # Use .tolist() to convert NumPy arrays to standard lists
    plt.plot(
        x_years.tolist(),
        y_line.tolist(),
        label=f'Linear Model (Slope: {slope:.4f}°C/yr)',
        color='#2ca02c',  # Green
        linewidth=3.5
    )

    # Annotate the warming rate (Slope)
    rate_text = f"Warming Rate:\n{slope:.4f}°C / Year"
    plt.annotate(
        rate_text,
        xy=(x_years[-1], y_line[-1]),
        xytext=(x_years[-1] + 5, y_line[-1]),
        arrowprops=dict(facecolor='#2ca02c', shrink=0.05),
        fontsize=10,
        color='#2ca02c',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7)
    )

    # Title and Labels
    title = f'Temperature Change Trend: {area} ({months})'
    subtitle = f'Baseline: {BASE_YEAR}. Model R-squared (vs. smoothed data): {r_squared:.4f}.'
    plt.title(title, fontsize=16, fontweight='bold')
    plt.suptitle(subtitle, fontsize=10, y=0.92)
    plt.xlabel(f'Year (Data starts {BASE_YEAR})', fontsize=12)
    plt.ylabel('Temperature Change (°C)', fontsize=12)

    # Legend and Grid
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle='--')

    # Add a horizontal line at 0.0 for visual baseline
    plt.axhline(0.0, color='gray', linestyle='-', linewidth=0.8)

    # Final save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"\n4. Visualization saved to: '{output_path}'")
    plt.close()  # Close the plot figure to free up memory


# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    print("=" * 90)
    print("              CLIMATE TREND ANALYSIS: DATA VISUALIZER                 ")
    print("=" * 90)

    # 1. Load and prepare the data (including smoothing)
    data_for_plot = load_and_prepare_data()

    # 2. Load the coefficients
    model_coefficients = load_coefficients()

    if not data_for_plot.empty and model_coefficients:
        # 3. Plot the representative trend (World|Annual)
        print(f"\n--- Plotting {DEFAULT_AREA} ({DEFAULT_MONTHS}) Trend ---")
        plot_trend(  # Updated function call
            df_data=data_for_plot,
            coefficients=model_coefficients,
            area=DEFAULT_AREA,
            months=DEFAULT_MONTHS,
            output_path=PLOT_OUTPUT_FILE
        )
    else:
        print("\nCould not proceed with visualization due to missing data or coefficients.")

    print("\n" + "=" * 90)

