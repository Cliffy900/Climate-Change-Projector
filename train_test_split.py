import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import json
from feature_engineer import create_time_index
from data_loader import load_and_process_data
from data_smoother import smooth_time_series  # <-- NEW: Import for data smoothing

# =============================================================================
# Configuration Constants
# =============================================================================
MODEL_OUTPUT_FILE = "linear_model_coefficients.json"
# We define the target column explicitly based on the intended pipeline
TARGET_COLUMN = 'SmoothedTempChange'  # <-- Use the smoothed data as the target

# =============================================================================
# Core Model Training and Saving Functions
# =============================================================================

def train_linear_models(df_features: pd.DataFrame) -> dict:
    """
    Trains a separate Linear Regression model for every unique time series
    (Area and Months combination) in the dataset.

    The model uses 'TimeIndex' as the independent variable (X) and
    'SmoothedTempChange' as the dependent variable (y) for a robust trend fit.

    Args:
        df_features (pd.DataFrame): The DataFrame with the 'TimeIndex' feature
                                    and the 'SmoothedTempChange' target column.

    Returns:
        dict: A dictionary storing the training results (intercept and slope)
              for each unique time series key.
    """
    if df_features.empty:
        print("Input DataFrame is empty. Cannot train models.")
        return {}

    # Check if the required target column exists
    if TARGET_COLUMN not in df_features.columns:
        print(f"ERROR: Required target column '{TARGET_COLUMN}' not found. Please check data_smoother.py.")
        return {}

    print(f"Starting Linear Regression model training on '{TARGET_COLUMN}'...")

    # Define the dictionary to store all model results
    model_results = {}

    # Drop any row that still has a NaN in the TimeIndex or the Target column
    # This handles edge cases, particularly NaNs from the centered rolling window (though min_periods=1 is used)
    df_clean = df_features.dropna(subset=['TimeIndex', TARGET_COLUMN]).copy()

    # Iterate over all unique time series defined by Area and Months
    for key, group in df_clean.groupby(['Area', 'Months']):
        # Create a unique string key for the model result
        model_key = f"{key[0]}|{key[1]}"

        # Reshape X to be a 2D array (required by sklearn)
        X = group[['TimeIndex']].values
        y = group[TARGET_COLUMN].values

        # Skip training if there are too few data points (requires at least 2 for a line)
        if len(y) < 2:
            print(f"Skipping {model_key}: Only {len(y)} data points available.")
            continue

        # Initialize and fit the Linear Regression Model
        model = LinearRegression()
        model.fit(X, y)

        # Store the results
        model_results[model_key] = {
            'intercept': model.intercept_,
            'slope': model.coef_[0]
        }

    print(f"Model training complete. Trained {len(model_results)} unique models.")
    return model_results


def save_model_coefficients(coefficients: dict) -> None:
    """
    Saves the trained model coefficients to a JSON file.

    Args:
        coefficients (dict): The dictionary of model results.
    """
    try:
        with open(MODEL_OUTPUT_FILE, 'w') as f:
            json.dump(coefficients, f, indent=4)
        print(f"Successfully saved coefficients to '{MODEL_OUTPUT_FILE}'.")
    except Exception as e:
        print(f"ERROR: Failed to save model coefficients to JSON: {e}")


# =============================================================================
# Main Execution Block for Testing/Running
# =============================================================================
def run_training_pipeline():
    """Executes the full data loading, feature engineering, and model training pipeline."""
    print("=" * 90)
    print("        CLIMATE TREND ANALYSIS: MODEL TRAINING PIPELINE STARTING        ")
    print("=" * 90)

    # 1. Data Loading
    print("\n--- 1. Data Loading & Cleaning ---")
    data_cleaned = load_and_process_data()

    # 2. Feature Engineering
    print("\n--- 2. Feature Engineering (TimeIndex) ---")
    data_features = create_time_index(data_cleaned)

    # 3. Data Smoothing <--- CRITICAL STEP INSERTED HERE
    print("\n--- 3. Data Smoothing ---")
    # This call generates the 'SmoothedTempChange' column required by the model
    data_smoothed = smooth_time_series(data_features)

    if data_smoothed.empty:
        print("Stopping execution: Smoothed data is empty.")
    else:
        # 4. Train the Models
        print("\n--- 4. Model Training ---")
        # Pass the data *after* smoothing to the training function
        trained_coefficients = train_linear_models(data_smoothed)

        if trained_coefficients:
            # 5. Save the Coefficients
            print("\n--- 5. Saving Results ---")
            save_model_coefficients(trained_coefficients)

            # Example of a key result
            world_annual_key = "World|Annual"
            if world_annual_key in trained_coefficients:
                print(f"\nExample result for World Annual:")
                print(
                    f"  Intercept (T_change at start year): {trained_coefficients[world_annual_key]['intercept']:.4f}°C")
                print(
                    f"  Slope (Warming rate per year):     {trained_coefficients[world_annual_key]['slope']:.4f}°C/year")
            else:
                print(f"\nERROR: Key '{world_annual_key}' was NOT found in the trained models.")
                # For debugging, print the keys that contain 'World' or 'Annual'
                world_keys = [k for k in trained_coefficients if 'World' in k]
                annual_keys = [k for k in trained_coefficients if 'Annual' in k]
                if world_keys or annual_keys:
                    print(f"Keys containing 'World' (first 5): {world_keys[:5]}")
                    print(f"Keys containing 'Annual' (first 5): {annual_keys[:5]}")
                else:
                    print("No related keys found. Check data loading and filtering steps.")

        print("\n[PIPELINE COMPLETE]")


if __name__ == "__main__":
    run_training_pipeline()

