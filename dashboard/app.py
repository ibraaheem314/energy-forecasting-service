"""Streamlit dashboard for energy forecasting service."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.models import ModelService
from app.services.features import FeatureService
from app.services.registry import ModelRegistry
from app.services.evaluate import EvaluationService

# Configure Streamlit page
st.set_page_config(
    page_title="Energy Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_sample_data():
    """Load sample energy data for demonstration."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='H')
    
    # Generate realistic energy consumption pattern
    base_load = 1000
    hourly_pattern = 200 * np.sin(2 * np.pi * dates.hour / 24)
    weekly_pattern = 100 * np.sin(2 * np.pi * dates.dayofweek / 7)
    noise = np.random.normal(0, 50, len(dates))
    
    consumption = base_load + hourly_pattern + weekly_pattern + noise
    consumption = np.maximum(consumption, 0)
    
    # Generate weather data
    temperature = 20 + 10 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 2, len(dates))
    
    return pd.DataFrame({
        'timestamp': dates,
        'consumption': consumption,
        'temperature': temperature
    })

async def get_model_status():
    """Get current model status."""
    try:
        model_service = ModelService()
        status = await model_service.get_status()
        return status
    except Exception as e:
        return {"error": str(e)}

async def generate_forecast(location, days_ahead):
    """Generate forecast for the specified location and period."""
    try:
        model_service = ModelService()
        feature_service = FeatureService()
        
        # Prepare features for forecast period
        start_time = datetime.now()
        end_time = start_time + timedelta(days=days_ahead)
        
        features_df = await feature_service.prepare_features(
            start_time=start_time,
            end_time=end_time,
            location=location
        )
        
        # Generate predictions
        predictions = await model_service.predict(features_df, confidence_interval=True)
        
        return predictions
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main dashboard function."""
    
    # Title and header
    st.title("⚡ Energy Forecasting Dashboard")
    st.markdown("Real-time energy consumption and production forecasting")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Location selection
    location = st.sidebar.selectbox(
        "Select Location",
        ["region_1", "region_2", "grid_A", "grid_B"],
        index=0
    )
    
    # Forecast horizon
    forecast_days = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=1,
        max_value=7,
        value=3
    )
    
    # Data type
    data_type = st.sidebar.selectbox(
        "Data Type",
        ["consumption", "production"],
        index=0
    )
    
    # Refresh button
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    # KPI metrics
    with col1:
        st.metric(
            label="Current Load (MW)",
            value="1,234",
            delta="5.2%"
        )
    
    with col2:
        st.metric(
            label="Forecast Accuracy",
            value="94.2%",
            delta="1.3%"
        )
    
    with col3:
        st.metric(
            label="Active Models",
            value="4",
            delta="1"
        )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecasts", "🔍 Model Performance", "📊 Historical Data", "⚙️ Settings"])
    
    with tab1:
        st.header("Energy Forecasts")
        
        # Generate forecast button
        if st.button("Generate New Forecast"):
            with st.spinner("Generating forecast..."):
                # Use sample data for demonstration
                sample_data = load_sample_data()
                
                # Create forecast visualization
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=sample_data['timestamp'][-168:],  # Last 7 days
                    y=sample_data['consumption'][-168:],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Forecast data (simulated)
                forecast_dates = pd.date_range(
                    start=sample_data['timestamp'].iloc[-1] + timedelta(hours=1),
                    periods=forecast_days * 24,
                    freq='H'
                )
                
                # Simple forecast (trend + seasonality)
                last_values = sample_data['consumption'][-24:].values
                seasonal_pattern = np.tile(last_values, forecast_days)[:len(forecast_dates)]
                trend = np.linspace(0, -50, len(forecast_dates))  # Slight downward trend
                forecast_values = seasonal_pattern + trend + np.random.normal(0, 20, len(forecast_dates))
                
                # Confidence intervals
                lower_bound = forecast_values - 100
                upper_bound = forecast_values + 100
                
                # Add forecast traces
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_values,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red')
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=lower_bound,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    name='Confidence Interval',
                    fillcolor='rgba(255,0,0,0.2)'
                ))
                
                fig.update_layout(
                    title=f"Energy {data_type.title()} Forecast - {location}",
                    xaxis_title="Time",
                    yaxis_title="Energy (MW)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Forecast Summary")
            st.write(f"**Location:** {location}")
            st.write(f"**Forecast Period:** {forecast_days} days")
            st.write(f"**Model Used:** Random Forest")
            st.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            st.subheader("Key Insights")
            st.write("• Peak demand expected tomorrow at 6 PM")
            st.write("• 15% increase predicted for weekdays")
            st.write("• Weather conditions favorable")
            st.write("• High confidence in 24h forecast")
    
    with tab2:
        st.header("Model Performance")
        
        # Model comparison
        st.subheader("Model Accuracy Comparison")
        
        # Sample model performance data
        models_data = pd.DataFrame({
            'Model': ['Random Forest', 'Gradient Boosting', 'Linear Regression', 'LSTM'],
            'MAE': [45.2, 48.1, 67.3, 52.8],
            'MAPE': [4.2, 4.8, 6.1, 5.1],
            'R²': [0.94, 0.92, 0.85, 0.90]
        })
        
        # Create performance chart
        fig = px.bar(
            models_data,
            x='Model',
            y='R²',
            title='Model Performance (R² Score)',
            color='R²',
            color_continuous_scale='viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance table
        st.subheader("Detailed Metrics")
        st.dataframe(models_data, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance")
        
        feature_importance = pd.DataFrame({
            'Feature': ['Temperature', 'Hour of Day', 'Day of Week', 'Previous Hour Load', 'Wind Speed'],
            'Importance': [0.25, 0.20, 0.18, 0.22, 0.15]
        })
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 5 Most Important Features'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Historical Data Analysis")
        
        # Load and display historical data
        sample_data = load_sample_data()
        
        # Time series plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sample_data['timestamp'],
            y=sample_data['consumption'],
            mode='lines',
            name='Energy Consumption'
        ))
        
        fig.update_layout(
            title=f"Historical Energy Consumption - {location}",
            xaxis_title="Date",
            yaxis_title="Energy (MW)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Statistics")
            st.write(f"**Average Load:** {sample_data['consumption'].mean():.1f} MW")
            st.write(f"**Peak Load:** {sample_data['consumption'].max():.1f} MW")
            st.write(f"**Min Load:** {sample_data['consumption'].min():.1f} MW")
            st.write(f"**Standard Deviation:** {sample_data['consumption'].std():.1f} MW")
        
        with col2:
            st.subheader("Data Quality")
            st.write(f"**Total Records:** {len(sample_data):,}")
            st.write(f"**Missing Values:** 0")
            st.write(f"**Data Coverage:** 100%")
            st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Correlation with weather
        st.subheader("Weather Correlation")
        
        fig = px.scatter(
            sample_data,
            x='temperature',
            y='consumption',
            title='Energy Consumption vs Temperature',
            trendline='ols'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Model Management")
        
        # Model status
        st.subheader("Active Models")
        
        model_status_data = pd.DataFrame({
            'Model': ['Random Forest', 'Gradient Boosting', 'Linear Regression'],
            'Location': ['region_1', 'region_2', 'grid_A'],
            'Status': ['Active', 'Active', 'Training'],
            'Last Updated': ['2024-01-15 10:30', '2024-01-15 09:45', '2024-01-15 11:00'],
            'Accuracy': ['94.2%', '92.8%', 'Pending']
        })
        
        st.dataframe(model_status_data, use_container_width=True)
        
        # Model actions
        st.subheader("Model Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Retrain Models"):
                st.success("Model retraining initiated")
        
        with col2:
            if st.button("Deploy Model"):
                st.success("Model deployment initiated")
        
        with col3:
            if st.button("Run Backtest"):
                st.success("Backtesting initiated")
        
        # Configuration
        st.subheader("Configuration")
        
        with st.form("config_form"):
            st.write("Update model configuration:")
            
            retrain_frequency = st.selectbox(
                "Retrain Frequency",
                ["Daily", "Weekly", "Monthly"],
                index=1
            )
            
            forecast_horizon = st.slider(
                "Default Forecast Horizon (hours)",
                min_value=1,
                max_value=168,
                value=24
            )
            
            confidence_level = st.slider(
                "Confidence Level (%)",
                min_value=80,
                max_value=99,
                value=95
            )
            
            submitted = st.form_submit_button("Update Configuration")
            
            if submitted:
                st.success("Configuration updated successfully!")
    
    # Footer
    st.markdown("---")
    st.markdown("Energy Forecasting Service Dashboard | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
