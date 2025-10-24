import   streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Import all core functions and constants from the existing pipeline files ---
# IMPORTANT: All these files (config.py, data_loader.py, etc.) MUST be in the same directory.
from config import START_YEAR, END_YEAR, TARGET_YEAR, DEFAULT_AREA, DEFAULT_MONTHS_PERIOD
from data_loader import load_and_process_data
from feature_engineer import create_time_index
from data_smoother import smooth_time_series
from train_test_split import train_linear_models
from predictor import predict_future_change


# =============================================================================
# Streamlit Caching and Core Pipeline Execution
# =============================================================================

@st.cache_data(show_spinner="Loading and preparing raw data...")
def load_data_pipeline():
    """Runs the data loading, cleaning, time-index creation, and smoothing steps."""
    print("Caching: Running full data loading and prep pipeline...")
    df_cleaned = load_and_process_data()
    if df_cleaned.empty:
        st.error("Failed to load or clean the data. Please check CSV_FETCH_ID in config.py.")
        return pd.DataFrame()

    df_indexed = create_time_index(df_cleaned)
    df_smoothed = smooth_time_series(df_indexed)

    return df_smoothed


@st.cache_data(show_spinner="Training models for all regions/periods...")
def train_and_cache_models(df):
    """Runs the training process and caches the resulting coefficients."""
    print("Caching: Training all linear models...")
    # This calls the function from train_test_split.py which returns the coefficients dict
    # Streamlit handles the caching, making this fast on subsequent runs.
    return train_linear_models(df)


# =============================================================================
# Utility Functions
# =============================================================================

def get_prediction(area, months, target_year, coefficients, start_year):
    """Calculates the single prediction, slope, and R-squared for the selected area/months."""
    key = f"{area}|{months}"
    if key not in coefficients:
        # Return None for all metrics if the key is missing
        return None, None, 0.0

    # Load the specific coefficients
    slope = coefficients[key]['slope']
    intercept = coefficients[key]['intercept']
    # R-squared is optional and defaults to 0.0 if not saved in the coefficients dictionary
    r_squared = coefficients[key].get('r_squared', 0.0)

    # Calculate TimeIndex for the target year
    target_time_index = target_year - start_year

    # Predict value: Y = mX + c
    predicted_value = slope * target_time_index + intercept
    return predicted_value, slope, r_squared


def generate_interactive_plot(df_data, area, months, coefficients, target_year):
    """Generates an interactive Plotly visualization for the selected trend."""

    df_series = df_data[(df_data['Area'] == area) & (df_data['Months'] == months)].copy()

    if df_series.empty:
        return go.Figure().update_layout(
            template="plotly_dark",
            annotations=[dict(
                text=f"No data available for {area} ({months})",
                showarrow=False, font=dict(size=20)
            )]
        )

    key = f"{area}|{months}"
    if key not in coefficients:
        return go.Figure().update_layout(
            template="plotly_dark",
            annotations=[dict(
                text="Model coefficients not found for this series.",
                showarrow=False, font=dict(size=20)
            )]
        )

    # --- 1. Get Model Parameters and Projection ---
    slope = coefficients[key]['slope']
    intercept = coefficients[key]['intercept']

    # Generate the linear fit line for historical data
    df_series['ModelFit'] = slope * df_series['TimeIndex'] + intercept

    # Create projection points
    projection_df = pd.DataFrame({
        'Year': range(df_series['Year'].max() + 1, target_year + 1)
    })

    # Handle the case where projection starts immediately after max year
    if projection_df.empty:
        projection_df = pd.DataFrame({'Year': [target_year]})

    # Ensure TimeIndex for projection years is calculated correctly
    projection_df['TimeIndex'] = projection_df['Year'] - START_YEAR
    projection_df['ModelFit'] = slope * projection_df['TimeIndex'] + intercept

    # Get the last known point for continuity
    last_known_point = df_series[['Year', 'ModelFit']].iloc[-1].to_dict()

    # --- 2. Create Plotly Figure ---
    fig = go.Figure()

    # Trace 1: Raw Temperature Change (Scatter/Dotted)
    fig.add_trace(go.Scatter(
        x=df_series['Year'], y=df_series['TempChange'],
        mode='lines',
        name='Raw Temp. Change',
        line=dict(color='#888', width=1, dash='dot'),
        opacity=0.5,
        hovertemplate='Year: %{x}<br>Raw Change: %{y:.3f}°C<extra></extra>'
    ))

    # Trace 2: Smoothed Temperature Change (Thick Line)
    fig.add_trace(go.Scatter(
        x=df_series['Year'], y=df_series['SmoothedTempChange'],
        mode='lines',
        name='5-Year Smoothed Trend',
        line=dict(color='#4c78a8', width=3),
        hovertemplate='Year: %{x}<br>Smoothed Change: %{y:.3f}°C<extra></extra>'
    ))

    # Trace 3: Linear Regression Fit (Historical)
    fig.add_trace(go.Scatter(
        x=df_series['Year'], y=df_series['ModelFit'],
        mode='lines',
        name='Linear Trend Fit',
        line=dict(color='orange', width=2, dash='solid'),
        hovertemplate='Year: %{x}<br>Fit: %{y:.3f}°C<extra></extra>'
    ))

    # Trace 4: Projection (Dashed Line)
    # Combine the last historical point with the projection points for continuity
    projection_x = [last_known_point['Year']] + projection_df['Year'].tolist()
    projection_y = [last_known_point['ModelFit']] + projection_df['ModelFit'].tolist()

    fig.add_trace(go.Scatter(
        x=projection_x, y=projection_y,
        mode='lines',
        name=f'Projection to {target_year}',
        line=dict(color='#ff006e', width=2, dash='dash'),
        hovertemplate='Year: %{x}<br>Projection: %{y:.3f}°C<extra></extra>'
    ))

    # Add a marker at the final projected point
    final_projection = projection_df.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[final_projection['Year']], y=[final_projection['ModelFit']],
        mode='markers',
        name=f'Target {target_year}',
        marker=dict(size=10, color='#ff006e', symbol='circle-dot'),
        hovertemplate=f'Target {target_year}<br>Change: {final_projection["ModelFit"]:.3f}°C<extra></extra>'
    ))

    # Add 0.0 line for baseline context
    fig.add_hline(y=0.0, line_dash="solid", line_color="#444", line_width=1,
                  annotation_text="Pre-Industrial Baseline (0.0°C)",
                  annotation_position="bottom right",
                  annotation_font_color="#aaa")

    # --- 3. Update Layout (Dark Theme) ---
    fig.update_layout(
        title={
            'text': f'Temperature Change Trend: {area} ({months})',
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=24, color='white')
        },
        xaxis_title="Year",
        yaxis_title="Temperature Change (°C)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(showgrid=True, gridcolor='#333')
    fig.update_yaxes(showgrid=True, gridcolor='#333')

    return fig


# =============================================================================
# Streamlit App Structure
# =============================================================================

def main():
    """The main function to run the Streamlit application."""
    st.set_page_config(
        page_title="Global Warming Trend Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Force dark theme for a "cool look" using Streamlit markdown injection
    st.markdown("""
        <style>
        .stApp {
            background-color: #0d1117;
            color: #c9d1d9;
        }
        .stSidebar {
            background-color: #161b22;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🌍 Climate Trend Analysis Dashboard")
    st.markdown("---")

    # --- 1. RUN CORE PIPELINE (Cached) ---
    # These steps only re-run when the function's inputs or code changes.
    df_data = load_data_pipeline()
    if df_data.empty:
        return  # Stop execution if data loading failed

    coefficients = train_and_cache_models(df_data)
    if not coefficients:
        st.error("Model training failed. Cannot proceed with visualization.")
        return

    # --- 2. SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("Select Time Series & Projection")

        # Get unique values for selectors
        unique_areas = sorted(df_data['Area'].unique().tolist())
        unique_months = sorted(df_data['Months'].unique().tolist())

        # Safely find default indices
        default_area_index = unique_areas.index(DEFAULT_AREA) if DEFAULT_AREA in unique_areas else 0
        default_months_index = unique_months.index(
            DEFAULT_MONTHS_PERIOD) if DEFAULT_MONTHS_PERIOD in unique_months else 0

        # Create selectors
        selected_area = st.selectbox(
            "Select Area/Region:",
            options=unique_areas,
            index=default_area_index
        )
        selected_months = st.selectbox(
            "Select Time Period:",
            options=unique_months,
            index=default_months_index
        )

        # Create slider for projection
        selected_target_year = st.slider(
            "Target Projection Year:",
            min_value=END_YEAR,
            max_value=2100,
            value=TARGET_YEAR,
            step=5
        )

        st.markdown("---")
        st.markdown(f"**Data Span:** {START_YEAR} - {END_YEAR}")
        st.markdown(f"**Total Models Trained:** {len(coefficients)}")

    # --- 3. KEY METRICS HEADER ---
    col1, col2, col3 = st.columns(3)

    # Get metrics for the selected series
    prediction_value, warming_rate, r_squared = get_prediction(
        selected_area, selected_months, selected_target_year, coefficients, START_YEAR
    )

    def format_metric(value, unit, style):
        """Formats the metric card content."""
        if value is None:
            return "N/A"
        return f'<p style="font-size:32px; font-weight:700; color:{style}; margin-bottom: 0px;">{value:.3f}{unit}</p>'

    rate_color = 'red' if warming_rate and warming_rate > 0 else 'blue' if warming_rate and warming_rate < 0 else 'gray'
    pred_color = 'red' if prediction_value and prediction_value > 0 else 'blue' if prediction_value and prediction_value < 0 else 'gray'

    with col1:
        st.markdown(
            f"""
            <div style="border: 1px solid #30363d; border-radius: 8px; padding: 10px; text-align: center;">
                <p style="font-size:16px; margin-bottom: 5px;">Projected Change by {selected_target_year}</p>
                {format_metric(prediction_value, '°C', pred_color)}
            </div>
            """, unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="border: 1px solid #30363d; border-radius: 8px; padding: 10px; text-align: center;">
                <p style="font-size:16px; margin-bottom: 5px;">Warming Rate (Slope)</p>
                {format_metric(warming_rate, '°C/Year', rate_color)}
            </div>
            """, unsafe_allow_html=True
        )

    with col3:
        # Display R-squared value
        r_squared_display = f'<p style="font-size:32px; font-weight:700; color:#4E8DFF; margin-bottom: 0px;">{r_squared:.4f}</p>' if r_squared != 0.0 else 'N/A'

        st.markdown(
            f"""
            <div style="border: 1px solid #30363d; border-radius: 8px; padding: 10px; text-align: center;">
                <p style="font-size:16px; margin-bottom: 5px;">Model Fit (R-Squared)</p>
                {r_squared_display}
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("---")

    # --- 4. MAIN VISUALIZATION ---
    st.subheader("Temperature Trend and Future Projection")
    fig = generate_interactive_plot(df_data, selected_area, selected_months, coefficients, selected_target_year)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- 5. GLOBAL SUMMARY LEADERBOARD ---
    st.subheader(f"Global Trends Summary (Projection to {TARGET_YEAR})")

    # We use the existing function from predictor.py to get the full prediction DF
    # NOTE: This uses the fixed TARGET_YEAR from config.py for the static summary
    df_predictions = predict_future_change(coefficients, TARGET_YEAR, START_YEAR)

    if not df_predictions.empty:
        col_warm, col_cool = st.columns(2)

        # Top 5 Warming
        top_warming = df_predictions.sort_values(
            by='Projected TempChange', ascending=False
        ).head(5)[['Area', 'Months', 'Projected TempChange']]
        top_warming.columns = ['Area', 'Period', f'Change by {TARGET_YEAR} (°C)']
        top_warming = top_warming.reset_index(drop=True)
        top_warming.index += 1  # 1-based index

        with col_warm:
            st.markdown("#### Hottest Projected Warming Trends")
            st.dataframe(top_warming, use_container_width=True, hide_index=False)

        # Top 5 Cooling (Decrease in Temp Change)
        top_cooling = df_predictions.sort_values(
            by='Projected TempChange', ascending=True
        ).head(5)[['Area', 'Months', 'Projected TempChange']]
        top_cooling.columns = ['Area', 'Period', f'Change by {TARGET_YEAR} (°C)']
        top_cooling = top_cooling.reset_index(drop=True)
        top_cooling.index += 1  # 1-based index

        with col_cool:
            st.markdown("#### Fastest Projected Cooling Trends")
            st.dataframe(top_cooling, use_container_width=True, hide_index=False)

    else:
        st.warning("Could not generate global prediction summary.")


if __name__ == '__main__':
    main()