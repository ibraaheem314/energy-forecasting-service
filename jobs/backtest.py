"""Backtesting job for energy forecasting models."""
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import argparse
import json
import pandas as pd
import numpy as np

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.models import ModelService
from app.services.features import FeatureService
from app.services.evaluate import EvaluationService
from app.services.registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestJob:
    """Job for backtesting energy forecasting models."""
    
    def __init__(self):
        self.model_service = ModelService()
        self.feature_service = FeatureService()
        self.evaluation_service = EvaluationService()
        self.registry = ModelRegistry()
    
    async def run(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        locations: list = None,
        model_names: list = None,
        forecast_horizons: list = None,
        backtest_type: str = "rolling_window"
    ):
        """Run the backtesting job.
        
        Args:
            start_date: Start date for backtesting period
            end_date: End date for backtesting period
            locations: List of locations to backtest
            model_names: List of model names to backtest
            forecast_horizons: List of forecast horizons (in hours)
            backtest_type: Type of backtesting ('rolling_window', 'expanding_window')
        """
        try:
            logger.info("Starting backtesting job")
            
            # Set default parameters
            if end_date is None:
                end_date = datetime.now()
            
            if start_date is None:
                start_date = end_date - timedelta(days=30)  # Default to last 30 days
            
            if locations is None:
                locations = ["region_1", "region_2"]
            
            if forecast_horizons is None:
                forecast_horizons = [1, 6, 12, 24]  # 1h, 6h, 12h, 24h ahead
            
            logger.info(f"Backtesting period: {start_date} to {end_date}")
            logger.info(f"Locations: {locations}")
            logger.info(f"Forecast horizons: {forecast_horizons}")
            logger.info(f"Backtest type: {backtest_type}")
            
            # Get available models if not specified
            if model_names is None:
                models_list = await self.registry.list_models()
                model_names = [model["model_name"] for model in models_list if model["active_version"]]
                logger.info(f"Using active models: {model_names}")
            
            backtest_results = {}
            
            # Run backtests for each combination
            for location in locations:
                location_results = {}
                
                for model_name in model_names:
                    try:
                        logger.info(f"Backtesting {model_name} for {location}")
                        
                        # Run backtest for different horizons
                        model_results = {}
                        for horizon in forecast_horizons:
                            try:
                                logger.info(f"Testing {horizon}h horizon")
                                
                                if backtest_type == "rolling_window":
                                    horizon_results = await self._rolling_window_backtest(
                                        model_name, location, start_date, end_date, horizon
                                    )
                                elif backtest_type == "expanding_window":
                                    horizon_results = await self._expanding_window_backtest(
                                        model_name, location, start_date, end_date, horizon
                                    )
                                else:
                                    raise ValueError(f"Unsupported backtest type: {backtest_type}")
                                
                                model_results[f"{horizon}h"] = horizon_results
                                
                            except Exception as e:
                                logger.error(f"Failed to backtest {horizon}h horizon: {e}")
                                model_results[f"{horizon}h"] = {"error": str(e)}
                        
                        location_results[model_name] = model_results
                        
                    except Exception as e:
                        logger.error(f"Failed to backtest {model_name} for {location}: {e}")
                        location_results[model_name] = {"error": str(e)}
                
                backtest_results[location] = location_results
            
            # Generate summary report
            summary = self._generate_backtest_summary(backtest_results)
            
            # Save results
            results_file = Path("data") / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "backtest_period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat()
                    },
                    "parameters": {
                        "locations": locations,
                        "model_names": model_names,
                        "forecast_horizons": forecast_horizons,
                        "backtest_type": backtest_type
                    },
                    "results": backtest_results,
                    "summary": summary
                }, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to {results_file}")
            logger.info("Backtesting job completed")
            
        except Exception as e:
            logger.error(f"Backtesting job failed: {e}")
            raise
    
    async def _rolling_window_backtest(
        self,
        model_name: str,
        location: str,
        start_date: datetime,
        end_date: datetime,
        horizon_hours: int
    ) -> dict:
        """Perform rolling window backtesting."""
        window_size_days = 7  # Use 7 days of history for each prediction
        step_hours = 24  # Move forward by 24 hours each time
        
        results = []
        current_date = start_date + timedelta(days=window_size_days)
        
        while current_date + timedelta(hours=horizon_hours) <= end_date:
            try:
                # Define training window
                train_start = current_date - timedelta(days=window_size_days)
                train_end = current_date
                
                # Define prediction target
                prediction_time = current_date + timedelta(hours=horizon_hours)
                
                # Prepare features for the prediction time
                features_df = await self.feature_service.prepare_features(
                    start_time=prediction_time - timedelta(hours=1),
                    end_time=prediction_time,
                    location=location
                )
                
                if len(features_df) == 0:
                    logger.warning(f"No features available for {prediction_time}")
                    current_date += timedelta(hours=step_hours)
                    continue
                
                # Generate prediction
                predictions = await self.model_service.predict(
                    features_df, model_name, confidence_interval=True
                )
                
                if predictions:
                    pred = predictions[0]
                    
                    # Get actual value (simulate by generating it)
                    actual_features = await self.feature_service.prepare_features(
                        start_time=prediction_time,
                        end_time=prediction_time + timedelta(hours=1),
                        location=location
                    )
                    
                    if len(actual_features) > 0:
                        actual_value = actual_features['value'].iloc[0]
                        
                        result = {
                            "prediction_time": prediction_time.isoformat(),
                            "predicted_value": pred["value"],
                            "actual_value": actual_value,
                            "error": abs(pred["value"] - actual_value),
                            "relative_error": abs(pred["value"] - actual_value) / actual_value * 100,
                            "confidence_interval": {
                                "lower": pred.get("lower_bound"),
                                "upper": pred.get("upper_bound")
                            } if pred.get("lower_bound") is not None else None
                        }
                        
                        results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed prediction for {current_date}: {e}")
            
            current_date += timedelta(hours=step_hours)
        
        # Calculate aggregate metrics
        if results:
            errors = [r["error"] for r in results]
            relative_errors = [r["relative_error"] for r in results]
            
            metrics = {
                "mae": np.mean(errors),
                "rmse": np.sqrt(np.mean([e**2 for e in errors])),
                "mape": np.mean(relative_errors),
                "max_error": np.max(errors),
                "median_error": np.median(errors),
                "prediction_count": len(results),
                "coverage_ratio": self._calculate_coverage_ratio(results)
            }
        else:
            metrics = {"error": "No valid predictions generated"}
        
        return {
            "predictions": results[-10:],  # Keep last 10 for inspection
            "metrics": metrics,
            "total_predictions": len(results)
        }
    
    async def _expanding_window_backtest(
        self,
        model_name: str,
        location: str,
        start_date: datetime,
        end_date: datetime,
        horizon_hours: int
    ) -> dict:
        """Perform expanding window backtesting."""
        # Similar to rolling window but with expanding training set
        # This is a simplified version - in practice you'd retrain the model
        return await self._rolling_window_backtest(
            model_name, location, start_date, end_date, horizon_hours
        )
    
    def _calculate_coverage_ratio(self, results: list) -> float:
        """Calculate the coverage ratio for confidence intervals."""
        if not results:
            return 0.0
        
        covered = 0
        total_with_ci = 0
        
        for result in results:
            ci = result.get("confidence_interval")
            if ci and ci.get("lower") is not None and ci.get("upper") is not None:
                total_with_ci += 1
                actual = result["actual_value"]
                if ci["lower"] <= actual <= ci["upper"]:
                    covered += 1
        
        return covered / total_with_ci if total_with_ci > 0 else 0.0
    
    def _generate_backtest_summary(self, backtest_results: dict) -> dict:
        """Generate summary of backtest results."""
        summary = {
            "total_model_location_combinations": 0,
            "successful_backtests": 0,
            "failed_backtests": 0,
            "average_metrics_by_horizon": {},
            "best_performing_combinations": [],
            "worst_performing_combinations": []
        }
        
        all_combinations = []
        
        for location, location_results in backtest_results.items():
            for model_name, model_results in location_results.items():
                summary["total_model_location_combinations"] += 1
                
                if "error" in model_results:
                    summary["failed_backtests"] += 1
                    continue
                
                summary["successful_backtests"] += 1
                
                for horizon, horizon_results in model_results.items():
                    if "error" not in horizon_results and "metrics" in horizon_results:
                        metrics = horizon_results["metrics"]
                        
                        combination = {
                            "location": location,
                            "model_name": model_name,
                            "horizon": horizon,
                            "mae": metrics.get("mae", float('inf')),
                            "mape": metrics.get("mape", float('inf')),
                            "rmse": metrics.get("rmse", float('inf')),
                            "prediction_count": metrics.get("prediction_count", 0)
                        }
                        
                        all_combinations.append(combination)
        
        # Calculate average metrics by horizon
        horizons = set(combo["horizon"] for combo in all_combinations)
        for horizon in horizons:
            horizon_combos = [c for c in all_combinations if c["horizon"] == horizon]
            if horizon_combos:
                summary["average_metrics_by_horizon"][horizon] = {
                    "average_mae": np.mean([c["mae"] for c in horizon_combos]),
                    "average_mape": np.mean([c["mape"] for c in horizon_combos]),
                    "average_rmse": np.mean([c["rmse"] for c in horizon_combos]),
                    "total_predictions": sum(c["prediction_count"] for c in horizon_combos)
                }
        
        # Get best and worst performing combinations
        if all_combinations:
            sorted_by_mae = sorted(all_combinations, key=lambda x: x["mae"])
            summary["best_performing_combinations"] = sorted_by_mae[:5]
            summary["worst_performing_combinations"] = sorted_by_mae[-5:]
        
        return summary

async def main():
    """Main function for running the backtest job."""
    parser = argparse.ArgumentParser(description="Backtest energy forecasting models")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--locations", nargs="+", help="List of locations")
    parser.add_argument("--model-names", nargs="+", help="List of model names")
    parser.add_argument("--forecast-horizons", nargs="+", type=int, help="Forecast horizons in hours")
    parser.add_argument("--backtest-type", choices=["rolling_window", "expanding_window"], 
                       default="rolling_window", help="Type of backtesting")
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Run the job
    job = BacktestJob()
    await job.run(
        start_date=start_date,
        end_date=end_date,
        locations=args.locations,
        model_names=args.model_names,
        forecast_horizons=args.forecast_horizons,
        backtest_type=args.backtest_type
    )

if __name__ == "__main__":
    asyncio.run(main())
