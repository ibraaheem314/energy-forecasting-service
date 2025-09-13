"""Tests for the feature engineering service."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

# Import the feature service
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.features import FeatureService

class TestFeatureService:
    """Test cases for the FeatureService class."""
    
    @pytest.fixture
    def feature_service(self):
        """Create a FeatureService instance for testing."""
        return FeatureService()
    
    @pytest.fixture
    def sample_energy_data(self):
        """Create sample energy data for testing."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
        values = 1000 + 200 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 50, len(dates))
        
        return pd.DataFrame({
            'timestamp': dates,
            'value': np.maximum(values, 0),
            'location': 'test_region',
            'data_type': 'consumption'
        })
    
    @pytest.mark.asyncio
    async def test_prepare_features_basic(self, feature_service):
        """Test basic feature preparation."""
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        location = "test_region"
        
        features_df = await feature_service.prepare_features(
            start_time=start_time,
            end_time=end_time,
            location=location,
            include_weather=False,
            include_external=False
        )
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0
        assert 'timestamp' in features_df.columns
        assert 'value' in features_df.columns
    
    def test_create_temporal_features(self, feature_service, sample_energy_data):
        """Test temporal feature creation."""
        features_df = feature_service._create_temporal_features(sample_energy_data)
        
        # Check that temporal features are created
        expected_features = [
            'hour', 'day_of_week', 'day_of_month', 'day_of_year',
            'week_of_year', 'month', 'quarter',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'is_weekend', 'is_business_hour', 'is_night'
        ]
        
        for feature in expected_features:
            assert feature in features_df.columns, f"Missing feature: {feature}"
        
        # Check value ranges
        assert features_df['hour'].min() >= 0
        assert features_df['hour'].max() <= 23
        assert features_df['day_of_week'].min() >= 0
        assert features_df['day_of_week'].max() <= 6
        assert features_df['month'].min() >= 1
        assert features_df['month'].max() <= 12
        
        # Check cyclical features are normalized
        assert features_df['hour_sin'].min() >= -1
        assert features_df['hour_sin'].max() <= 1
        assert features_df['hour_cos'].min() >= -1
        assert features_df['hour_cos'].max() <= 1
        
        # Check binary features
        assert set(features_df['is_weekend'].unique()).issubset({0, 1})
        assert set(features_df['is_business_hour'].unique()).issubset({0, 1})
        assert set(features_df['is_night'].unique()).issubset({0, 1})
    
    def test_create_lag_features(self, feature_service, sample_energy_data):
        """Test lag feature creation."""
        features_df = feature_service._create_lag_features(sample_energy_data)
        
        # Check that lag features are created
        expected_lags = [1, 2, 3, 6, 12, 24]
        for lag in expected_lags:
            lag_col = f'value_lag_{lag}'
            assert lag_col in features_df.columns, f"Missing lag feature: {lag_col}"
            
            # Check that lag values are correctly shifted
            # (allowing for NaN values at the beginning)
            non_nan_indices = ~features_df[lag_col].isna()
            if non_nan_indices.any():
                original_values = features_df.loc[non_nan_indices, 'value'].values
                lag_values = features_df.loc[non_nan_indices, lag_col].values
                
                # The lag values should match original values shifted by lag periods
                # (We'll check a few values to account for any edge cases)
                if len(original_values) > lag:
                    assert np.allclose(
                        original_values[lag:lag+5], 
                        lag_values[lag:lag+5] if len(lag_values) > lag+5 else lag_values[lag:],
                        rtol=1e-10
                    )
    
    def test_create_rolling_features(self, feature_service, sample_energy_data):
        """Test rolling statistics feature creation."""
        features_df = feature_service._create_rolling_features(sample_energy_data)
        
        # Check that rolling features are created
        windows = [3, 6, 12, 24]
        stats = ['mean', 'std', 'min', 'max']
        
        for window in windows:
            for stat in stats:
                feature_name = f'value_rolling_{stat}_{window}'
                assert feature_name in features_df.columns, f"Missing rolling feature: {feature_name}"
        
        # Check EWMA features
        assert 'value_ewma_alpha_0.1' in features_df.columns
        assert 'value_ewma_alpha_0.3' in features_df.columns
        
        # Validate rolling statistics make sense
        for window in windows:
            rolling_mean = features_df[f'value_rolling_mean_{window}']
            rolling_min = features_df[f'value_rolling_min_{window}']
            rolling_max = features_df[f'value_rolling_max_{window}']
            
            # Min should be <= mean <= max (where not NaN)
            valid_indices = ~(rolling_mean.isna() | rolling_min.isna() | rolling_max.isna())
            if valid_indices.any():
                assert (rolling_min[valid_indices] <= rolling_mean[valid_indices]).all()
                assert (rolling_mean[valid_indices] <= rolling_max[valid_indices]).all()
    
    @pytest.mark.asyncio
    async def test_create_weather_features(self, feature_service):
        """Test weather feature creation."""
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()
        location = "test_region"
        
        weather_features = await feature_service._create_weather_features(
            start_time, end_time, location
        )
        
        assert isinstance(weather_features, pd.DataFrame)
        assert len(weather_features) > 0
        
        # Check required weather features
        expected_features = [
            'timestamp', 'temperature', 'humidity', 'wind_speed', 'cloud_cover',
            'apparent_temperature', 'heat_index', 'wind_chill', 'temp_category', 'wind_category'
        ]
        
        for feature in expected_features:
            assert feature in weather_features.columns, f"Missing weather feature: {feature}"
        
        # Check value ranges
        assert weather_features['humidity'].min() >= 0
        assert weather_features['humidity'].max() <= 100
        assert weather_features['cloud_cover'].min() >= 0
        assert weather_features['cloud_cover'].max() <= 1
        assert weather_features['wind_speed'].min() >= 0
    
    @pytest.mark.asyncio
    async def test_create_external_features(self, feature_service):
        """Test external factor feature creation."""
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()
        
        external_features = await feature_service._create_external_features(
            start_time, end_time
        )
        
        assert isinstance(external_features, pd.DataFrame)
        assert len(external_features) > 0
        assert 'timestamp' in external_features.columns
        
        # Check for holiday and economic indicator features
        assert 'is_holiday' in external_features.columns
        assert 'economic_index' in external_features.columns
        
        # Holiday should be binary
        assert set(external_features['is_holiday'].unique()).issubset({True, False})
    
    def test_handle_missing_values(self, feature_service):
        """Test missing value handling."""
        # Create data with missing values
        data = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now(), periods=10, freq='H'),
            'value': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
            'feature1': [1, np.nan, 3, 4, np.nan, 6, 7, 8, np.nan, 10],
            'feature2': [np.nan, 2, 3, np.nan, 5, 6, np.nan, 8, 9, 10]
        })
        
        result_df = feature_service._handle_missing_values(data)
        
        # Check that missing values are handled
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'timestamp':
                assert not result_df[col].isna().any(), f"Missing values remain in {col}"
    
    def test_scale_features(self, feature_service, sample_energy_data):
        """Test feature scaling."""
        # Add some features to scale
        sample_energy_data['feature1'] = np.random.normal(100, 20, len(sample_energy_data))
        sample_energy_data['feature2'] = np.random.normal(0, 1, len(sample_energy_data))
        
        result_df = feature_service._scale_features(sample_energy_data)
        
        # Check that features are scaled (should have mean ~0, std ~1)
        # Note: value should not be scaled as it's the target
        assert 'value' in result_df.columns
        assert not np.allclose(result_df['value'].mean(), 0, atol=0.1)  # Value should not be scaled
        
        # Other features should be scaled
        if 'feature1' in result_df.columns:
            # After scaling, should have approximately mean 0 and std 1
            assert abs(result_df['feature1'].mean()) < 0.1
            assert abs(result_df['feature1'].std() - 1.0) < 0.1
    
    def test_get_feature_importance(self, feature_service):
        """Test feature importance calculation."""
        feature_names = ['temp', 'humidity', 'lag_1', 'rolling_mean_24', 'hour_sin']
        
        importance_scores = feature_service.get_feature_importance(feature_names)
        
        assert isinstance(importance_scores, dict)
        assert len(importance_scores) == len(feature_names)
        
        for name in feature_names:
            assert name in importance_scores
            assert 0 <= importance_scores[name] <= 1  # Assuming scores are normalized
    
    @pytest.mark.asyncio
    async def test_prepare_features_with_weather_and_external(self, feature_service):
        """Test feature preparation with weather and external factors."""
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()
        location = "test_region"
        
        features_df = await feature_service.prepare_features(
            start_time=start_time,
            end_time=end_time,
            location=location,
            include_weather=True,
            include_external=True
        )
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0
        
        # Should have base energy features
        assert 'timestamp' in features_df.columns
        assert 'value' in features_df.columns
        
        # Should have temporal features
        assert 'hour' in features_df.columns
        assert 'day_of_week' in features_df.columns
        
        # Should have weather features
        assert 'temperature' in features_df.columns
        assert 'humidity' in features_df.columns
        
        # Should have external features
        assert 'is_holiday' in features_df.columns
    
    def test_feature_service_consistency(self, feature_service):
        """Test that feature service produces consistent results."""
        # Create identical input data
        data1 = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime(2023, 1, 1), periods=48, freq='H'),
            'value': np.sin(np.arange(48) * 2 * np.pi / 24) * 100 + 1000,
            'location': 'test',
            'data_type': 'consumption'
        })
        
        data2 = data1.copy()
        
        # Process both datasets
        result1 = feature_service._create_temporal_features(data1)
        result2 = feature_service._create_temporal_features(data2)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

class TestFeatureServiceEdgeCases:
    """Test edge cases for the feature service."""
    
    @pytest.fixture
    def feature_service(self):
        return FeatureService()
    
    def test_empty_dataframe(self, feature_service):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame(columns=['timestamp', 'value', 'location', 'data_type'])
        
        result = feature_service._create_temporal_features(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_single_row_dataframe(self, feature_service):
        """Test handling of single-row dataframes."""
        single_row_df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'value': [1000],
            'location': ['test'],
            'data_type': ['consumption']
        })
        
        result = feature_service._create_temporal_features(single_row_df)
        assert len(result) == 1
        assert 'hour' in result.columns
        assert not result['hour'].isna().any()
    
    def test_extreme_values(self, feature_service):
        """Test handling of extreme values."""
        extreme_df = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now(), periods=5, freq='H'),
            'value': [0, 1e6, -1000, np.inf, -np.inf],
            'location': ['test'] * 5,
            'data_type': ['consumption'] * 5
        })
        
        # Should handle extreme values gracefully
        result = feature_service._handle_missing_values(extreme_df)
        assert isinstance(result, pd.DataFrame)

if __name__ == "__main__":
    pytest.main([__file__])
