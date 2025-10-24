import pandas as pd
import os
import re  # Import the regular expression module
from config import CSV_FETCH_ID, TARGET_ELEMENT, YEAR_COLUMN_PREFIX, START_YEAR, END_YEAR


# =============================================================================
# Core Data Loading and Processing Functions
# =============================================================================

def load_and_process_data() -> pd.DataFrame:
    """
    Loads the raw temperature data, filters it, standardizes key columns, and
    melts (unpivots) the time series data into a long, clean format suitable
    for analysis.

    Returns:
        pd.DataFrame: A processed DataFrame with columns:
                      ['Area', 'Months', 'Year', 'TempChange'].
    """
    print(f"Attempting to load data from: {CSV_FETCH_ID}")

    try:
        # Load the CSV using latin1 encoding to handle special characters (like °C)
        df_raw = pd.read_csv(CSV_FETCH_ID, encoding='latin1')

    except FileNotFoundError:
        print(f"ERROR: Failed to load CSV file. File not found at '{CSV_FETCH_ID}'.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return pd.DataFrame()

    print("Initial data loaded. Filtering and standardizing data...")

    # --- 1. FILTERING ---
    df_filtered = df_raw[
        df_raw['Element'].str.lower().str.contains(TARGET_ELEMENT.lower(), na=False)
    ].copy()

    if df_filtered.empty:
        print(f"ERROR: Filtered DataFrame is empty. Check if TARGET_ELEMENT ('{TARGET_ELEMENT}') is correct.")
        return pd.DataFrame()

    year_columns = [col for col in df_filtered.columns if col.startswith(YEAR_COLUMN_PREFIX)]

    # --- 2. STANDARDIZATION (ULTIMATE CLEANING FIX) ---

    # 1. Clean 'Area' column: Strip whitespace, ensure title case, and replace common synonyms with 'World'.
    df_filtered['Area'] = df_filtered['Area'].astype(str).str.strip().str.title()
    df_filtered['Area'] = df_filtered['Area'].replace(
        {'World': 'World', 'Global': 'World', 'All': 'World'}, regex=True
    )

    # 2. Clean 'Months' column: This is the most problematic column.
    df_filtered['Months'] = df_filtered['Months'].astype(str)  # Convert to string first

    # Remove all non-alphanumeric characters (except standard spaces) and strip
    df_filtered['Months'] = df_filtered['Months'].apply(
        lambda x: re.sub(r'[^\w\s]', '', x).strip()
    ).str.title()

    # Force the key for Annual data to 'Annual'
    # CRITICAL: Added 'Meteorological Year' to explicitly map the raw data value
    df_filtered['Months'] = df_filtered['Months'].replace(
        ['Annual', 'Annuel', 'Yearly', 'Annuel', 'Meteorological Year'], 'Annual', regex=True
    )

    # --- 3. MELTING (Unpivoting) ---
    id_vars = [col for col in df_filtered.columns if col not in year_columns]

    df_melted = df_filtered.melt(
        id_vars=id_vars,
        value_vars=year_columns,
        var_name='Year',
        value_name='TempChange'
    )

    # --- 4. FINAL CLEANUP ---
    df_melted['Year'] = df_melted['Year'].str.replace(YEAR_COLUMN_PREFIX, '').astype(int)
    df_final = df_melted.dropna(subset=['TempChange'])
    df_final = df_final[
        (df_final['Year'] >= START_YEAR) & (df_final['Year'] <= END_YEAR)
        ].copy()

    df_final = df_final[['Area', 'Months', 'Year', 'TempChange']]

    print(f"Data loading and processing successful. Final records for analysis: {len(df_final)}")
    return df_final


# =============================================================================
# Demonstration of Usage
# =============================================================================
if __name__ == "__main__":
    print("--- Executing Data Loader ---\n")

    # Load the processed data
    data = load_and_process_data()

    if not data.empty:
        print("\nProcessed Data Head (First 5 Rows):")
        print(data.head())

        print("\nData Structure Summary:")
        data.info()

        # Example: Check how many unique areas and periods are available
        unique_areas = data['Area'].nunique()
        unique_months = data['Months'].nunique()
        print(f"\nFound {unique_areas} unique areas and {unique_months} time periods.")

        # Example Query: Global Annual Temperature Change (CRITICAL CHECK)
        world_annual = data[(data['Area'] == 'World') & (data['Months'] == 'Annual')]
        if not world_annual.empty:
            print("\nWorld Annual Temperature Change Statistics:")
            print(f"Total World Annual Data Points: {len(world_annual)}")
            print(f"Temperature Change Range: {world_annual['Year'].min()} - {world_annual['Year'].max()}")
            print(f"Mean Temperature Change: {world_annual['TempChange'].mean():.4f}°C")
        else:
            print("\nCRITICAL CHECK FAILED: 'World|Annual' key is still missing after standardization.")
