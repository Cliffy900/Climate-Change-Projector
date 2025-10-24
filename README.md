Global Climate Trend Projector

Project Overview

This repository hosts a comprehensive Python data pipeline and an interactive Streamlit dashboard designed to analyze and project long-term global temperature trends.

The core analysis involves segmenting the historical data by geographical region and time period, smoothing the time series to identify underlying trends, and applying a simple Linear Regression model to forecast temperature change up to the year 2100.

The final product is a highly interactive web application that allows users to explore specific regional warming rates, view historical data alongside the projected trend line, and compare the most extreme projected warming and cooling trends globally.

Key Features

Modular Pipeline: A robust, multi-step pipeline built across multiple Python scripts for data loading, feature engineering, smoothing, training, and prediction.

Interactive Dashboard: A Streamlit application (streamlit_app.py) for real-time visualization and exploration.

Linear Trend Projection: Utilizes Linear Regression to project temperature changes relative to the 1961 baseline.

5-Year Smoothing: Applies a 5-year rolling mean to remove year-to-year noise and fit the model to the robust, long-term trend.

Performance Caching: Uses Streamlit's data caching to ensure the computationally intensive model training only runs once.

🔬 Methodology and Data

Data Source

Data: Temperature Change data (relative to a long-term average).

Time Period: 1961 to 2019.

Granularity: Global, regional, and national data, broken down by Annual and Monthly periods.

Baseline: All temperature change values are measured relative to the historical 1961 average. A reading of 0.0 represents no change from that baseline.

Analysis Pipeline

Data Preparation: Raw data is loaded, filtered to include only "Temperature change," and unpivoted (melted) from wide format (years as columns) to long format (single Year column).

Feature Engineering: A TimeIndex feature is created, mapping the start year (1961) to 0, which serves as the independent variable ($X$) for the regression model.

Data Smoothing: A 5-year centered rolling mean is applied to the time series to create the SmoothedTempChange target variable ($Y$). This focuses the model on the climate trend rather than annual weather volatility.

Model Training: A separate Linear Regression model ($Y = mX + c$) is trained for every unique combination of Area and Months. The coefficients (slope $m$ and intercept $c$) are saved.

Projection: The trained slope ($m$, or warming rate) is used to project the temperature change to any target year selected by the user, up to 2100.

🚀 How to Run the Dashboard

This project is best viewed as a Streamlit web application.

Prerequisites

You need Python (3.9+) and the following packages:

# Ensure you have your requirements installed
pip install -r requirements.txt 


Files Needed

Ensure all the following files are in your project directory:

streamlit_app.py (The main dashboard interface)

config.py

data_loader.py

feature_engineer.py

data_smoother.py

train_test_split.py

predictor.py

Environment_Temperature_change_E_All_Data_NOFLAG.csv (The source data)

Launching the App

To start the interactive dashboard, navigate to your project directory in the terminal and execute the Streamlit command:

streamlit run streamlit_app.py


This command will start a local web server and automatically open the application in your default web browser (usually at http://localhost:8501).

🛠️ Repository Structure

File

Description

streamlit_app.py

Main Dashboard. The single entry point for the interactive web application, importing and running all logic.

config.py

Centralized constants for the entire pipeline (years, default settings, file paths).

data_loader.py

Handles CSV loading, filtering for the target element, and unpivoting (melting) the time series.

feature_engineer.py

Creates the TimeIndex column required for regression analysis.

data_smoother.py

Applies the 5-year rolling mean to the data for robust trend analysis.

train_test_split.py

Contains the function to train a linear model for every time series and save the coefficients.

predictor.py

Uses the saved coefficients to calculate and summarize future temperature projections.

main.py

Legacy: The original command-line script for sequential pipeline execution (no longer needed for the Streamlit dashboard).