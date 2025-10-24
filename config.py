# =============================================================================
# Configuration Settings for Climate Change Data Analysis
#
# This file centralizes all constants, file paths, and default parameters
# for processing and projecting the environmental temperature change data.
# =============================================================================

# --- FILE PATHS & IDENTIFIERS ---

# MANDATORY: The file ID of the uploaded CSV containing temperature data.
# This variable is used to locate the data source within the environment.
CSV_FETCH_ID = "Environment_Temperature_change_E_All_Data_NOFLAG.csv"

# --- DATA SCHEMA CONSTANTS ---

# List of columns that define the data entry (metadata, not time series)
METADATA_COLUMNS = [
    "Area Code",
    "Area",
    "Months Code",
    "Months",
    "Element Code",
    "Element",
    "Unit",
]

# The element value to filter for (Temperature change is the key element)
TARGET_ELEMENT = "Temperature change"

# The column name prefix for time series data
YEAR_COLUMN_PREFIX = "Y"

# --- TIME SERIES WINDOW ---

# The first year of data available in the CSV
START_YEAR = 1961

# The last year of data available in the CSV
END_YEAR = 2019

# --- DEFAULT ANALYSIS PARAMETERS ---

# The default area to select when the application loads (e.g., 'World', 'Australia')
DEFAULT_AREA = "World"

# The default time period to analyze (e.g., 'Annual', 'January', 'Summer')
DEFAULT_MONTHS_PERIOD = "Annual"

# The default future year for which we want to project the temperature change
TARGET_YEAR = 2050

# The percentage of data to reserve for testing the model's performance
TEST_SIZE = 0.20





