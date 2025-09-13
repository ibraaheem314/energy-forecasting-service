"""Feature engineering for energy forecasting models."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

from .loader import DataLoader

logger = logging.getLogger(__name__)

class FeatureService:
    """Service for feature engineering and preparation."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
    
    async def prepare_features(
        self,
        start_time: datetime,
        end_time: datetime,
        location: str,
        include_weather: bool = True,
        include_external: bool = True
    ) -> pd.DataFrame:
        """Prepare features for model training or prediction.
        
        Args:
            start_time: Start time for feature preparation
            end_time: End time for feature preparation
            location: Location identifier
            include_weather: Whether to include weather features
            include_external: Whether to include external factors
            
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info(f"Preparing features for {location} from {start_time} to {end_time}")
            
            # Load base energy data
            energy_data = await self.data_loader.load_historical_data(
                start_time, end_time, location, "consumption"
            )
            
            # Create temporal features
            features_df = self._create_temporal_features(energy_data)
            
            # Add lag features
            features_df = self._create_lag_features(features_df)
            
            # Add rolling statistics
            features_df = self._create_rolling_features(features_df)
            
            # Add weather features if requested
            if include_weather:
                weather_features = await self._create_weather_features(
                    start_time, end_time, location
                )
                features_df = pd.merge(
                    features_df, weather_features, on='timestamp', how='left'
                )
            
            # Add external factor features if requested
            if include_external:
                external_features = await self._create_external_features(
                    start_time, end_time
                )
                features_df = pd.merge(
                    features_df, external_features, on='timestamp', how='left'
                )
            
            # Handle missing values
            features_df = self._handle_missing_values(features_df)
            
            # Scale numerical features
            features_df = self._scale_features(features_df)
            
            logger.info(f"Features prepared: {features_df.shape[1]} features, {features_df.shape[0]} samples")
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to prepare features: {e}")
            raise
    
    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract temporal components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Create cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame, max_lag: int = 24) -> pd.DataFrame:
        """Create lag features for time series data."""
        result_df = df.copy()
        
        for lag in [1, 2, 3, 6, 12, 24]:
            if lag <= max_lag:
                result_df[f'value_lag_{lag}'] = result_df['value'].shift(lag)
        
        return result_df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistics features."""
        result_df = df.copy()
        
        # Rolling means
        for window in [3, 6, 12, 24]:
            result_df[f'value_rolling_mean_{window}'] = (
                result_df['value'].rolling(window=window, min_periods=1).mean()
            )
            result_df[f'value_rolling_std_{window}'] = (
                result_df['value'].rolling(window=window, min_periods=1).std()
            )
            result_df[f'value_rolling_min_{window}'] = (
                result_df['value'].rolling(window=window, min_periods=1).min()
            )
            result_df[f'value_rolling_max_{window}'] = (
                result_df['value'].rolling(window=window, min_periods=1).max()
            )
        
        # Exponential weighted moving average
        result_df['value_ewma_alpha_0.1'] = (
            result_df['value'].ewm(alpha=0.1).mean()
        )
        result_df['value_ewma_alpha_0.3'] = (
            result_df['value'].ewm(alpha=0.3).mean()
        )
        
        return result_df
    
    async def _create_weather_features(
        self,
        start_time: datetime,
        end_time: datetime,
        location: str
    ) -> pd.DataFrame:
        """Create weather-based features."""
        weather_data = await self.data_loader.load_weather_data(
            start_time, end_time, location
        )
        
        df = weather_data.copy()
        
        # Derived weather features
        df['apparent_temperature'] = (
            df['temperature'] - 0.4 * (df['temperature'] - 10) * (1 - df['humidity'] / 100)
        )
        df['heat_index'] = df['temperature'] + 0.5 * (df['humidity'] - 50)
        df['wind_chill'] = (
            df['temperature'] - 2 * df['wind_speed']
        )
        
        # Weather categories
        df['temp_category'] = pd.cut(
            df['temperature'],
            bins=[-float('inf'), 5, 15, 25, float('inf')],
            labels=['cold', 'cool', 'warm', 'hot']
        )
        df['wind_category'] = pd.cut(
            df['wind_speed'],
            bins=[-float('inf'), 5, 15, 25, float('inf')],
            labels=['calm', 'light', 'moderate', 'strong']
        )
        
        # Encode categorical features
        for col in ['temp_category', 'wind_category']:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.encoders[col].transform(df[col].astype(str))
        
        return df[['timestamp', 'temperature', 'humidity', 'wind_speed', 
                  'cloud_cover', 'apparent_temperature', 'heat_index', 
                  'wind_chill', 'temp_category', 'wind_category']]
    
    async def _create_external_features(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Create external factor features."""
        external_data = await self.data_loader.load_external_factors(
            start_time, end_time, ['holidays', 'economic_indicators']
        )
        
        return external_data
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        result_df = df.copy()
        
        # Forward fill for most features
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'timestamp':
                result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Fill remaining NaN with median
        for col in numeric_cols:
            if col != 'timestamp':
                result_df[col] = result_df[col].fillna(result_df[col].median())
        
        return result_df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        result_df = df.copy()
        
        # Identify numeric columns to scale (exclude target and categorical)
        exclude_cols = ['timestamp', 'value', 'location', 'data_type']
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in cols_to_scale:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
                result_df[col] = self.scalers[col].fit_transform(
                    result_df[col].values.reshape(-1, 1)
                ).flatten()
            else:
                result_df[col] = self.scalers[col].transform(
                    result_df[col].values.reshape(-1, 1)
                ).flatten()
        
        return result_df
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance scores (placeholder implementation)."""
        # This would typically come from a trained model
        importance_scores = {}
        for name in feature_names:
            if 'lag' in name:
                importance_scores[name] = np.random.uniform(0.1, 0.3)
            elif 'rolling' in name:
                importance_scores[name] = np.random.uniform(0.05, 0.2)
            elif any(temp in name for temp in ['temperature', 'weather']):
                importance_scores[name] = np.random.uniform(0.1, 0.25)
            else:
                importance_scores[name] = np.random.uniform(0.01, 0.1)
        
        return importance_scores
