# =============================================================================
# Import Core Functions and Constants from Modules
# =============================================================================

# Configuration Constants
# We import specific constants used throughout the pipeline
from config import START_YEAR, DEFAULT_AREA, DEFAULT_MONTHS_PERIOD, TARGET_YEAR

# Pipeline Constants for reporting purposes
MODEL_COEFFICIENTS_FILE = "linear_model_coefficients.json"
PLOT_OUTPUT_FILE = "world_annual_temperature_trend.png"

# Data Pipeline Steps
from data_loader import load_and_process_data
from feature_engineer import create_time_index
from data_smoother import smooth_time_series  # <-- NEW: Data Smoothing
from train_test_split import train_linear_models, save_model_coefficients

# Pipeline Execution Wrappers
# We use the core functions and wrappers defined in the downstream modules
from model_evaluator import evaluate_models
from predictor import run_prediction_pipeline
from data_visualizer import plot_trend


def run_analysis_pipeline():
    """
    Orchestrates the complete end-to-end climate trend analysis pipeline.

    The pipeline includes:
    1. Data Loading and Cleaning.
    2. Feature Engineering (TimeIndex creation).
    3. Data Smoothing (5-year rolling mean for a robust trend line).
    4. Model Training (Linear Regression) and Saving Coefficients.
    5. Model Evaluation, Prediction, and Visualization.
    """
    print("=" * 90)
    print("      CLIMATE TREND ANALYSIS PIPELINE STARTING (Final Run)      ")
    print("=" * 90)

    # ----------------------------------------
    # 1. DATA LOADING AND CLEANING
    # ----------------------------------------
    print("\n[STEP 1/7] Data Loading and Cleaning...")
    data_cleaned = load_and_process_data()

    if data_cleaned.empty:
        print("Stopping execution: Data loading failed.")
        return

    # ----------------------------------------
    # 2. FEATURE ENGINEERING
    # ----------------------------------------
    print("\n[STEP 2/7] Feature Engineering (Creating Time Index)...")
    data_features = create_time_index(data_cleaned)

    # ----------------------------------------
    # 3. DATA SMOOTHING
    # ----------------------------------------
    print("\n[STEP 3/7] Data Smoothing (5-year rolling mean)...")
    # This creates the 'SmoothedTempChange' column, which becomes the model's target.
    data_smoothed = smooth_time_series(data_features)

    if data_smoothed.empty:
        print("Stopping execution: Data smoothing failed or resulted in an empty DataFrame.")
        return

    # ----------------------------------------
    # 4. MODEL TRAINING AND SAVING
    # ----------------------------------------
    print("\n[STEP 4/7] Model Training (Linear Regression) and Saving Coefficients...")
    # Train models using the smoothed data as the target variable
    trained_coefficients = train_linear_models(data_smoothed)
    save_model_coefficients(trained_coefficients)

    if not trained_coefficients:
        print("Stopping execution: Model training failed.")
        return

    # ----------------------------------------
    # 5. MODEL EVALUATION
    # ----------------------------------------
    print("\n[STEP 5/7] Running Model Evaluation...")
    # The evaluation function is designed to run the test split and print results
    evaluation_results = evaluate_models(data_smoothed, trained_coefficients)

    # ----------------------------------------
    # 6. PREDICTION
    # ----------------------------------------
    print(f"\n[STEP 6/7] Running Future Predictions for Year {TARGET_YEAR}...")
    # This wrapper function loads the coefficients, predicts, and prints the summary
    run_prediction_pipeline()

    # ----------------------------------------
    # 7. VISUALIZATION
    # ----------------------------------------
    print(f"\n[STEP 7/7] Generating Key Visualization: {DEFAULT_AREA} ({DEFAULT_MONTHS_PERIOD})...")
    # We pass the pre-prepared data and coefficients to avoid redundant loading
    plot_trend(
        df_data=data_smoothed,
        coefficients=trained_coefficients,
        area=DEFAULT_AREA,
        months=DEFAULT_MONTHS_PERIOD,
        output_path=PLOT_OUTPUT_FILE,
    )

    print(f"\n[PIPELINE COMPLETE] Check '{PLOT_OUTPUT_FILE}' for the visualization and '{MODEL_COEFFICIENTS_FILE}' for model results.")
    print("=" * 90)


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    try:
        run_analysis_pipeline()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed during execution: {e}")