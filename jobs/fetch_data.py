"""Data fetching job for energy forecasting service."""
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import argparse

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.loader import DataLoader
from app.config import DATA_REFRESH_INTERVAL_HOURS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetchJob:
    """Job for fetching and updating energy data."""
    
    def __init__(self):
        self.data_loader = DataLoader()
    
    async def run(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        locations: list = None,
        data_types: list = None
    ):
        """Run the data fetching job.
        
        Args:
            start_date: Start date for data fetching (default: 7 days ago)
            end_date: End date for data fetching (default: now)
            locations: List of locations to fetch data for
            data_types: List of data types to fetch
        """
        try:
            logger.info("Starting data fetch job")
            
            # Set default parameters
            if end_date is None:
                end_date = datetime.now()
            
            if start_date is None:
                start_date = end_date - timedelta(days=7)
            
            if locations is None:
                locations = ["region_1", "region_2", "grid_A", "grid_B"]
            
            if data_types is None:
                data_types = ["consumption", "production"]
            
            logger.info(f"Fetching data from {start_date} to {end_date}")
            logger.info(f"Locations: {locations}")
            logger.info(f"Data types: {data_types}")
            
            # Fetch data for each combination
            for location in locations:
                for data_type in data_types:
                    try:
                        logger.info(f"Fetching {data_type} data for {location}")
                        
                        # Load historical data
                        data = await self.data_loader.load_historical_data(
                            start_date=start_date,
                            end_date=end_date,
                            location=location,
                            data_type=data_type
                        )
                        
                        # Save to cache
                        filename = f"{data_type}_{location}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
                        await self.data_loader.save_data(data, filename)
                        
                        logger.info(f"Saved {len(data)} records for {location} {data_type}")
                        
                    except Exception as e:
                        logger.error(f"Failed to fetch {data_type} data for {location}: {e}")
                        continue
            
            # Fetch weather data
            logger.info("Fetching weather data")
            for location in locations:
                try:
                    weather_data = await self.data_loader.load_weather_data(
                        start_date=start_date,
                        end_date=end_date,
                        location=location
                    )
                    
                    # Save weather data
                    filename = f"weather_{location}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
                    await self.data_loader.save_data(weather_data, filename)
                    
                    logger.info(f"Saved {len(weather_data)} weather records for {location}")
                    
                except Exception as e:
                    logger.error(f"Failed to fetch weather data for {location}: {e}")
                    continue
            
            # Fetch external factors
            logger.info("Fetching external factors")
            try:
                external_data = await self.data_loader.load_external_factors(
                    start_date=start_date,
                    end_date=end_date,
                    factors=["holidays", "economic_indicators"]
                )
                
                filename = f"external_factors_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
                await self.data_loader.save_data(external_data, filename)
                
                logger.info(f"Saved {len(external_data)} external factor records")
                
            except Exception as e:
                logger.error(f"Failed to fetch external factors: {e}")
            
            logger.info("Data fetch job completed successfully")
            
        except Exception as e:
            logger.error(f"Data fetch job failed: {e}")
            raise

async def main():
    """Main function for running the data fetch job."""
    parser = argparse.ArgumentParser(description="Fetch energy data")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--locations", nargs="+", help="List of locations")
    parser.add_argument("--data-types", nargs="+", help="List of data types")
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Run the job
    job = DataFetchJob()
    await job.run(
        start_date=start_date,
        end_date=end_date,
        locations=args.locations,
        data_types=args.data_types
    )

if __name__ == "__main__":
    asyncio.run(main())
