import pandas as pd
import numpy as np
import json
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Imports for data preparation (assuming these files exist and are correctly configured):
from data_loader import load_and_process_data
from feature_engineer import create_time_index
from data_smoother import smooth_time_series

# =============================================================================
# Configuration and Constants
# =============================================================================
COEFFICIENTS_FILE = "linear_model_coefficients.json"
BASE_YEAR = 1961
TEST_SIZE = 0.20  # Replicates the 80% train / 20% test split from training
TARGET_COLUMN = 'SmoothedTempChange'  # <-- Explicitly use the smoothed column


# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_and_prepare_data() -> pd.DataFrame:
    """
    Loads raw data, melts, cleans, creates the TimeIndex, and applies smoothing.
    Replicates the data pipeline used for training.
    """
    print("1. Loading raw data...")
    df_cleaned = load_and_process_data()

    print("2. Creating TimeIndex feature...")
    df_indexed = create_time_index(df_cleaned)

    print("3. Applying 5-year smoothing...")
    # This ensures the evaluation is done on the same 'SmoothedTempChange' target
    df_smoothed = smooth_time_series(df_indexed)

    # Filter out any rows that became NaN due to the smoothing window at the start
    df_final = df_smoothed.dropna(subset=[TARGET_COLUMN]).copy()
    print(f"Data ready for evaluation. Total records: {len(df_final)}")
    return df_final


def load_coefficients() -> dict:
    """Loads the linear regression coefficients (intercept and slope) from the JSON file."""
    try:
        with open(COEFFICIENTS_FILE, 'r') as f:
            coefficients = json.load(f)
        print(f"Coefficients successfully loaded from {COEFFICIENTS_FILE}.")
        return coefficients
    except FileNotFoundError:
        print(f"ERROR: Coefficients file not found at '{COEFFICIENTS_FILE}'. Run 'train_test_split.py' first.")
        return {}
    except Exception as e:
        print(f"An error occurred while loading coefficients: {e}")
        return {}


# =============================================================================
# Core Evaluation Function
# =============================================================================

def evaluate_models(df_full: pd.DataFrame, coefficients: dict) -> pd.DataFrame:
    """
    Evaluates the performance of the trained linear models using the test data set.

    Args:
        df_full (pd.DataFrame): The full DataFrame containing all features and targets.
        coefficients (dict): The dictionary of trained model coefficients.

    Returns:
        pd.DataFrame: A summary DataFrame of R-squared and RMSE for each time series.
    """
    if df_full.empty or not coefficients:
        print("Cannot evaluate models: Data or coefficients are missing.")
        return pd.DataFrame()

    print("\nStarting Model Evaluation...")

    results_list = []

    # Iterate over each unique time series (Area|Months)
    for key, model_params in coefficients.items():
        area, months = key.split('|')

        # 1. Filter the data for the current time series
        df_series = df_full[
            (df_full['Area'] == area) &
            (df_full['Months'] == months)
            ].copy()

        if len(df_series) < 2:
            continue  # Skip if not enough data points

        # 2. Re-create the train/test split on the full data for this series
        # We use the full range and then apply the split to get the test set.
        X = df_series[['TimeIndex']]
        y = df_series[TARGET_COLUMN]

        # Splitting the data for evaluation (we only need the test set)
        # Note: stratify=None as TimeIndex is continuous
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, shuffle=False, random_state=42
        )

        # 3. Use the trained coefficients to predict on the test set
        intercept = model_params['intercept']
        slope = model_params['slope']

        # The prediction uses the linear equation: y_pred = intercept + slope * X
        y_pred_test = intercept + slope * X_test['TimeIndex']

        # 4. Calculate R-squared and RMSE for the test set
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        results_list.append({
            'Area': area,
            'Months': months,
            'R2_Test': r2_test,
            'RMSE_Test': rmse_test,
            'Slope': slope,  # Include slope for quick comparison
        })

    print("Evaluation complete.")
    return pd.DataFrame(results_list)


# =============================================================================
# Main Execution Block
# =============================================================================
def run_evaluation_pipeline():
    """Executes the full model evaluation pipeline."""
    print("=" * 90)
    print("         CLIMATE TREND ANALYSIS: MODEL EVALUATION PIPELINE            ")
    print("=" * 90)

    # 1. Load and prepare the data (including smoothing)
    data_for_eval = load_and_prepare_data()

    # 2. Load the coefficients
    model_coefficients = load_coefficients()

    df_results = pd.DataFrame()  # Initialize empty DataFrame

    if not data_for_eval.empty and model_coefficients:
        # 3. Evaluate the models
        df_results = evaluate_models(data_for_eval, model_coefficients)

    if not df_results.empty:
        # 4. Print Summary
        world_annual_results = df_results[
            (df_results['Area'] == 'World') & (df_results['Months'] == 'Annual')
            ]

        if not world_annual_results.empty:
            r2_wa = world_annual_results['R2_Test'].iloc[0]
            test_rmse_wa = world_annual_results['RMSE_Test'].iloc[0]

            print(f"\n--- Key Model Performance: World|Annual ---")
            print(f"  R-squared: {r2_wa:.4f} (Closer to 1.0 is better fit)")
            print(f"  RMSE:      {test_rmse_wa:.4f}°C (Lower is better prediction error)\n")

        # Overall Summary
        print(f"Total Models Evaluated: {len(df_results)}")

        avg_r2 = df_results['R2_Test'].mean()
        print(f"Average Test R-squared across all models: {avg_r2:.4f}")

        print("\nTop 5 Best Performing Models (Highest R-squared):")
        top_r2 = df_results.sort_values(by='R2_Test', ascending=False).head(5)

        print(top_r2[['Area', 'Months', 'R2_Test', 'RMSE_Test', 'Slope']].to_string(index=False))

        print("\nTop 5 Poorest Performing Models (Lowest R-squared):")
        bottom_r2 = df_results.sort_values(by='R2_Test', ascending=True).head(5)

        print(bottom_r2[['Area', 'Months', 'R2_Test', 'RMSE_Test', 'Slope']].to_string(index=False))

    print("\n[EVALUATION PIPELINE COMPLETE]")
    print("=" * 90)


if __name__ == "__main__":
    run_evaluation_pipeline()
