"""Model retraining job for energy forecasting service."""
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import argparse
import json

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.models import ModelService
from app.services.features import FeatureService
from app.services.registry import ModelRegistry
from app.services.evaluate import EvaluationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrainingJob:
    """Job for retraining energy forecasting models."""
    
    def __init__(self):
        self.model_service = ModelService()
        self.feature_service = FeatureService()
        self.registry = ModelRegistry()
        self.evaluation_service = EvaluationService()
    
    async def run(
        self,
        model_types: list = None,
        locations: list = None,
        training_days: int = 90,
        auto_deploy: bool = False,
        performance_threshold: float = 0.8
    ):
        """Run the model retraining job.
        
        Args:
            model_types: List of model types to retrain
            locations: List of locations to train models for
            training_days: Number of days of historical data to use for training
            auto_deploy: Whether to automatically deploy models that meet performance criteria
            performance_threshold: Minimum R² score for auto-deployment
        """
        try:
            logger.info("Starting model retraining job")
            
            # Set default parameters
            if model_types is None:
                model_types = ["random_forest", "gradient_boosting", "ridge"]
            
            if locations is None:
                locations = ["region_1", "region_2", "grid_A", "grid_B"]
            
            # Calculate training period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=training_days)
            
            logger.info(f"Training period: {start_date} to {end_date}")
            logger.info(f"Model types: {model_types}")
            logger.info(f"Locations: {locations}")
            
            training_results = {}
            
            # Train models for each location and type
            for location in locations:
                location_results = {}
                
                for model_type in model_types:
                    try:
                        logger.info(f"Training {model_type} model for {location}")
                        
                        # Train the model
                        training_result = await self.model_service.train_model(
                            model_type=model_type,
                            start_date=start_date,
                            end_date=end_date,
                            location=location,
                            test_size=0.2
                        )
                        
                        model_name = training_result["model_name"]
                        metrics = training_result["metrics"]
                        
                        logger.info(f"Model {model_name} trained. R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.2f}")
                        
                        # Register the model
                        version_id = await self.registry.register_model(
                            model_name=f"{model_type}_{location}",
                            run_id=training_result["run_id"],
                            model_type=model_type,
                            location=location,
                            metrics=metrics,
                            description=f"Retrained {model_type} model for {location}"
                        )
                        
                        # Evaluate the model
                        eval_start = end_date - timedelta(days=30)  # Use last 30 days for evaluation
                        evaluation_result = await self.evaluation_service.evaluate_model_performance(
                            model_name=model_name,
                            start_date=eval_start,
                            end_date=end_date,
                            location=location,
                            evaluation_type="holdout"
                        )
                        
                        result = {
                            "model_name": model_name,
                            "version_id": version_id,
                            "training_metrics": metrics,
                            "evaluation_metrics": evaluation_result.get("metrics", {}),
                            "training_completed": True,
                            "auto_deploy_eligible": metrics["r2"] >= performance_threshold
                        }
                        
                        # Auto-deploy if conditions are met
                        if auto_deploy and result["auto_deploy_eligible"]:
                            try:
                                deployment = await self.registry.deploy_model(
                                    model_name=f"{model_type}_{location}",
                                    version_id=version_id,
                                    deployment_target="production"
                                )
                                result["deployment"] = deployment
                                result["auto_deployed"] = True
                                logger.info(f"Model {model_name} auto-deployed to production")
                            except Exception as e:
                                logger.error(f"Failed to auto-deploy model {model_name}: {e}")
                                result["auto_deployed"] = False
                                result["deployment_error"] = str(e)
                        else:
                            result["auto_deployed"] = False
                        
                        location_results[model_type] = result
                        
                    except Exception as e:
                        logger.error(f"Failed to train {model_type} for {location}: {e}")
                        location_results[model_type] = {
                            "training_completed": False,
                            "error": str(e)
                        }
                
                training_results[location] = location_results
            
            # Generate summary report
            summary = self._generate_training_summary(training_results)
            logger.info("Retraining job completed")
            logger.info(f"Summary: {summary}")
            
            # Save results
            results_file = Path("data") / f"retraining_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "training_period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat()
                    },
                    "parameters": {
                        "model_types": model_types,
                        "locations": locations,
                        "training_days": training_days,
                        "auto_deploy": auto_deploy,
                        "performance_threshold": performance_threshold
                    },
                    "results": training_results,
                    "summary": summary
                }, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Retraining job failed: {e}")
            raise
    
    def _generate_training_summary(self, training_results: dict) -> dict:
        """Generate a summary of training results."""
        summary = {
            "total_models_trained": 0,
            "successful_trainings": 0,
            "failed_trainings": 0,
            "auto_deployed_models": 0,
            "average_r2_score": 0,
            "best_performing_models": [],
            "locations_processed": len(training_results),
            "model_types_processed": 0
        }
        
        all_r2_scores = []
        best_models = []
        
        for location, location_results in training_results.items():
            summary["model_types_processed"] = max(
                summary["model_types_processed"], 
                len(location_results)
            )
            
            for model_type, result in location_results.items():
                summary["total_models_trained"] += 1
                
                if result.get("training_completed", False):
                    summary["successful_trainings"] += 1
                    
                    metrics = result.get("training_metrics", {})
                    r2_score = metrics.get("r2", 0)
                    all_r2_scores.append(r2_score)
                    
                    if result.get("auto_deployed", False):
                        summary["auto_deployed_models"] += 1
                    
                    best_models.append({
                        "model_name": result.get("model_name"),
                        "location": location,
                        "model_type": model_type,
                        "r2_score": r2_score,
                        "mae": metrics.get("mae", 0)
                    })
                else:
                    summary["failed_trainings"] += 1
        
        # Calculate average R² score
        if all_r2_scores:
            summary["average_r2_score"] = sum(all_r2_scores) / len(all_r2_scores)
        
        # Get top 5 best performing models
        best_models.sort(key=lambda x: x["r2_score"], reverse=True)
        summary["best_performing_models"] = best_models[:5]
        
        return summary

async def main():
    """Main function for running the retraining job."""
    parser = argparse.ArgumentParser(description="Retrain energy forecasting models")
    parser.add_argument("--model-types", nargs="+", help="List of model types to train")
    parser.add_argument("--locations", nargs="+", help="List of locations")
    parser.add_argument("--training-days", type=int, default=90, help="Number of days of training data")
    parser.add_argument("--auto-deploy", action="store_true", help="Auto-deploy models meeting performance criteria")
    parser.add_argument("--performance-threshold", type=float, default=0.8, help="Minimum R² for auto-deployment")
    
    args = parser.parse_args()
    
    # Run the job
    job = RetrainingJob()
    await job.run(
        model_types=args.model_types,
        locations=args.locations,
        training_days=args.training_days,
        auto_deploy=args.auto_deploy,
        performance_threshold=args.performance_threshold
    )

if __name__ == "__main__":
    asyncio.run(main())
