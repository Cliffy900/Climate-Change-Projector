import json
import pandas as pd
import numpy as np
from config import START_YEAR, TARGET_YEAR, DEFAULT_AREA, DEFAULT_MONTHS_PERIOD  # All constants imported

# =============================================================================
# Configuration Constants
# =============================================================================
MODEL_INPUT_FILE = "linear_model_coefficients.json"


# START_YEAR and TARGET_YEAR are imported from config.py

# =============================================================================
# Core Functions
# =============================================================================

def load_coefficients() -> dict:
    """
    Loads the linear regression coefficients (intercept and slope) from the JSON file.

    Returns:
        dict: A dictionary containing the coefficients for each time series.
    """
    try:
        with open(MODEL_INPUT_FILE, 'r') as f:
            coefficients = json.load(f)
        print(f"Successfully loaded coefficients from {MODEL_INPUT_FILE}.")
        return coefficients
    except FileNotFoundError:
        print(f"ERROR: Model coefficients file not found at '{MODEL_INPUT_FILE}'.")
        print("Please ensure 'train_test_split.py' was run successfully.")
        return {}
    except Exception as e:
        print(f"An error occurred while loading coefficients: {e}")
        return {}


def predict_future_change(coefficients: dict, prediction_year: int, base_year: int) -> pd.DataFrame:
    """
    Uses the trained coefficients to project the temperature change for a
    specified future year.

    Args:
        coefficients (dict): Dictionary of model coefficients (slope and intercept).
        prediction_year (int): The target year for prediction (e.g., 2050).
        base_year (int): The reference year (e.g., 1961) used to calculate TimeIndex=0.

    Returns:
        pd.DataFrame: A DataFrame containing the projected temperature change
                      for all time series, sorted by warming magnitude.
    """
    if not coefficients:
        return pd.DataFrame()

    print(f"Projecting temperature changes up to the year {prediction_year}...")

    # Calculate the TimeIndex for the prediction year
    time_index_future = prediction_year - base_year

    predictions = []

    for key, params in coefficients.items():
        area, months = key.split('|')

        intercept = params['intercept']
        slope = params['slope']

        # Prediction: TempChange = intercept + (slope * TimeIndex)
        projected_change = intercept + (slope * time_index_future)

        predictions.append({
            'Area': area,
            'Months': months,
            'Projected TempChange': projected_change
        })

    df_predictions = pd.DataFrame(predictions)

    # Sort by the magnitude of projected warming (descending)
    df_predictions = df_predictions.sort_values(
        by='Projected TempChange',
        ascending=False
    ).reset_index(drop=True)

    return df_predictions


def display_projections(df_predictions: pd.DataFrame):
    """
    Prints the key projection results.

    Args:
        df_predictions (pd.DataFrame): The DataFrame containing projection results.
    """
    if df_predictions.empty:
        print("No predictions to display.")
        return

    # --- 1. Key Result: World (Annual) ---
    world_annual_key = f"{DEFAULT_AREA}|{DEFAULT_MONTHS_PERIOD}"

    world_annual = df_predictions[
        (df_predictions['Area'] == DEFAULT_AREA) &
        (df_predictions['Months'] == DEFAULT_MONTHS_PERIOD)
        ]

    if not world_annual.empty:
        temp_change = world_annual['Projected TempChange'].iloc[0]
        print(
            f"\n-> Key Result: {DEFAULT_AREA} ({DEFAULT_MONTHS_PERIOD}) projected change in {TARGET_YEAR}: {temp_change:.4f}°C")

        # Print for the user to see the baseline year used
        print(f"  (Baseline: {START_YEAR})")

    # --- 2. Top 3 Fastest Warming Regions/Months (largest positive change) ---
    print("\nTop 3 Projected Warming Trends:")
    top_warming = df_predictions.head(3)
    for index, row in top_warming.iterrows():
        print(f"  {row['Projected TempChange']:>8.3f}°C  | {row['Area']:<30} ({row['Months']})")

    # --- 3. Top 3 Fastest Cooling Regions/Months (largest negative change) ---\
    # We sort by the smallest (most negative) change
    top_cooling = df_predictions.sort_values(by='Projected TempChange', ascending=True).head(3)

    print("\nTop 3 Projected Cooling Trends (Decrease in Temp Change):")
    for index, row in top_cooling.iterrows():
        print(f"  {row['Projected TempChange']:>8.3f}°C  | {row['Area']:<30} ({row['Months']})")

    print("=" * 80)


# =============================================================================
# Main Execution Block
# =============================================================================
def run_prediction_pipeline():
    """Executes the full prediction pipeline."""
    # 1. Load the trained coefficients
    coefficients = load_coefficients()

    if coefficients:
        # 2. Predict future changes using the imported START_YEAR and TARGET_YEAR
        df_predictions = predict_future_change(coefficients, TARGET_YEAR, START_YEAR)

        # 3. Display the results
        display_projections(df_predictions)
    else:
        print("Prediction halted due to missing coefficients.")


if __name__ == "__main__":
    run_prediction_pipeline()
