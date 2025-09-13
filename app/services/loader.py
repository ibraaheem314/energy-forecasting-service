"""Data loading utilities for energy forecasting."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import logging

from ..config import DATA_DIR

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader for energy consumption and production data."""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    async def load_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        location: Optional[str] = None,
        data_type: str = "consumption"
    ) -> pd.DataFrame:
        """Load historical energy data for the specified period.
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
            location: Location identifier (optional)
            data_type: Type of data ('consumption', 'production', 'weather')
            
        Returns:
            DataFrame with historical data
        """
        try:
            # This would typically connect to a database or API
            # For now, create sample data
            logger.info(f"Loading {data_type} data from {start_date} to {end_date}")
            
            return self._generate_sample_data(start_date, end_date, location, data_type)
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            raise
    
    async def load_weather_data(
        self,
        start_date: datetime,
        end_date: datetime,
        location: Optional[str] = None
    ) -> pd.DataFrame:
        """Load weather data for the specified period.
        
        Args:
            start_date: Start date for weather data
            end_date: End date for weather data
            location: Location identifier
            
        Returns:
            DataFrame with weather data
        """
        try:
            logger.info(f"Loading weather data from {start_date} to {end_date}")
            return self._generate_sample_weather_data(start_date, end_date, location)
            
        except Exception as e:
            logger.error(f"Failed to load weather data: {e}")
            raise
    
    async def load_external_factors(
        self,
        start_date: datetime,
        end_date: datetime,
        factors: List[str] = None
    ) -> pd.DataFrame:
        """Load external factors data (holidays, economic indicators, etc.).
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            factors: List of factor types to load
            
        Returns:
            DataFrame with external factors
        """
        if factors is None:
            factors = ["holidays", "economic_indicators"]
        
        try:
            logger.info(f"Loading external factors: {factors}")
            return self._generate_sample_external_data(start_date, end_date, factors)
            
        except Exception as e:
            logger.error(f"Failed to load external factors: {e}")
            raise
    
    def _generate_sample_data(
        self,
        start_date: datetime,
        end_date: datetime,
        location: Optional[str],
        data_type: str
    ) -> pd.DataFrame:
        """Generate sample energy data for demonstration."""
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate realistic energy consumption patterns
        base_load = 1000 if data_type == "consumption" else 800
        seasonal_pattern = 200 * np.sin(2 * np.pi * date_range.hour / 24)
        weekly_pattern = 100 * np.sin(2 * np.pi * date_range.dayofweek / 7)
        noise = np.random.normal(0, 50, len(date_range))
        
        values = base_load + seasonal_pattern + weekly_pattern + noise
        values = np.maximum(values, 0)  # Ensure non-negative values
        
        return pd.DataFrame({
            'timestamp': date_range,
            'value': values,
            'location': location or 'default',
            'data_type': data_type
        })
    
    def _generate_sample_weather_data(
        self,
        start_date: datetime,
        end_date: datetime,
        location: Optional[str]
    ) -> pd.DataFrame:
        """Generate sample weather data."""
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate realistic weather patterns
        temperature = 20 + 10 * np.sin(2 * np.pi * date_range.hour / 24) + np.random.normal(0, 2, len(date_range))
        humidity = 50 + 20 * np.sin(2 * np.pi * date_range.hour / 24 + np.pi/4) + np.random.normal(0, 5, len(date_range))
        wind_speed = 5 + 3 * np.abs(np.random.normal(0, 1, len(date_range)))
        cloud_cover = np.random.uniform(0, 1, len(date_range))
        
        return pd.DataFrame({
            'timestamp': date_range,
            'temperature': temperature,
            'humidity': np.clip(humidity, 0, 100),
            'wind_speed': np.maximum(wind_speed, 0),
            'cloud_cover': cloud_cover,
            'location': location or 'default'
        })
    
    def _generate_sample_external_data(
        self,
        start_date: datetime,
        end_date: datetime,
        factors: List[str]
    ) -> pd.DataFrame:
        """Generate sample external factors data."""
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        data = {'timestamp': date_range}
        
        if 'holidays' in factors:
            # Simple holiday indicator (weekends as holidays)
            data['is_holiday'] = date_range.weekday >= 5
        
        if 'economic_indicators' in factors:
            # Simple economic indicator
            data['economic_index'] = 100 + np.random.normal(0, 5, len(date_range))
        
        return pd.DataFrame(data)
    
    async def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save data to local storage.
        
        Args:
            data: DataFrame to save
            filename: Name of the file to save
        """
        try:
            filepath = self.data_dir / filename
            data.to_parquet(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise
    
    async def load_cached_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load cached data from local storage.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            DataFrame if file exists, None otherwise
        """
        try:
            filepath = self.data_dir / filename
            if filepath.exists():
                return pd.read_parquet(filepath)
            return None
            
        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")
            return None
