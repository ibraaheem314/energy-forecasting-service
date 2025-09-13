"""Model training and prediction services for energy forecasting."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit

from .features import FeatureService
from .loader import DataLoader
from ..config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, BASE_DIR

logger = logging.getLogger(__name__)

class ModelService:
    """Service for model training, evaluation, and prediction."""
    
    def __init__(self):
        self.feature_service = FeatureService()
        self.data_loader = DataLoader()
        self.models = {}
        self.model_metadata = {}
        
        # Setup MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'class': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'linear_regression': {
                'class': LinearRegression,
                'params': {}
            },
            'ridge': {
                'class': Ridge,
                'params': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            }
        }
    
    async def train_model(
        self,
        model_type: str,
        start_date: datetime,
        end_date: datetime,
        location: str,
        test_size: float = 0.2,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Train a forecasting model.
        
        Args:
            model_type: Type of model to train
            start_date: Start date for training data
            end_date: End date for training data
            location: Location identifier
            test_size: Fraction of data to use for testing
            hyperparameters: Custom hyperparameters
            
        Returns:
            Training results and metrics
        """
        try:
            logger.info(f"Training {model_type} model for {location}")
            
            with mlflow.start_run() as run:
                # Prepare features
                features_df = await self.feature_service.prepare_features(
                    start_date, end_date, location
                )
                
                # Prepare training data
                X, y = self._prepare_training_data(features_df)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, shuffle=False, random_state=42
                )
                
                # Initialize model
                model = self._create_model(model_type, hyperparameters)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                metrics = self._evaluate_model(model, X_test, y_test)
                
                # Log to MLflow
                mlflow.log_params({
                    'model_type': model_type,
                    'location': location,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features_count': X.shape[1]
                })
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")
                
                # Store model
                model_name = f"{model_type}_{location}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.models[model_name] = model
                self.model_metadata[model_name] = {
                    'type': model_type,
                    'location': location,
                    'trained_at': datetime.now(),
                    'metrics': metrics,
                    'run_id': run.info.run_id,
                    'feature_names': X.columns.tolist()
                }
                
                logger.info(f"Model {model_name} trained successfully. MAE: {metrics['mae']:.2f}")
                
                return {
                    'model_name': model_name,
                    'metrics': metrics,
                    'run_id': run.info.run_id,
                    'feature_importance': self._get_feature_importance(model, X.columns)
                }
                
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise
    
    async def predict(
        self,
        features_df: pd.DataFrame,
        model_name: Optional[str] = None,
        confidence_interval: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate predictions using a trained model.
        
        Args:
            features_df: Features dataframe
            model_name: Name of the model to use for prediction
            confidence_interval: Whether to include confidence intervals
            
        Returns:
            List of prediction results
        """
        try:
            # Use default model if none specified
            if model_name is None:
                model_name = self._get_default_model()
            
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Prepare features for prediction
            X = self._prepare_prediction_features(features_df, model_name)
            
            # Generate predictions
            predictions = model.predict(X)
            
            # Generate confidence intervals if requested
            if confidence_interval:
                lower_bounds, upper_bounds = self._calculate_confidence_intervals(
                    model, X, predictions
                )
            else:
                lower_bounds = upper_bounds = None
            
            # Format results
            results = []
            timestamps = features_df['timestamp'].tolist()
            
            for i, (timestamp, pred) in enumerate(zip(timestamps, predictions)):
                result = {
                    'timestamp': timestamp,
                    'value': float(pred),
                    'confidence': 0.95 if confidence_interval else None
                }
                
                if confidence_interval:
                    result['lower_bound'] = float(lower_bounds[i])
                    result['upper_bound'] = float(upper_bounds[i])
                
                results.append(result)
            
            logger.info(f"Generated {len(results)} predictions using {model_name}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")
            raise
    
    async def get_status(self) -> Dict[str, Any]:
        """Get model service status and information."""
        try:
            active_models = list(self.models.keys())
            
            status = {
                'active_models': active_models,
                'total_models': len(active_models),
                'mlflow_tracking_uri': MLFLOW_TRACKING_URI,
                'experiment_name': MLFLOW_EXPERIMENT_NAME,
                'last_updated': datetime.now().isoformat()
            }
            
            if active_models:
                # Get information about the most recent model
                latest_model = max(
                    self.model_metadata.keys(),
                    key=lambda x: self.model_metadata[x]['trained_at']
                )
                status['latest_model'] = {
                    'name': latest_model,
                    'type': self.model_metadata[latest_model]['type'],
                    'trained_at': self.model_metadata[latest_model]['trained_at'].isoformat(),
                    'metrics': self.model_metadata[latest_model]['metrics']
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get model status: {e}")
            raise
    
    def _create_model(self, model_type: str, hyperparameters: Optional[Dict[str, Any]]):
        """Create a model instance with specified configuration."""
        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        config = self.model_configs[model_type].copy()
        
        # Override with custom hyperparameters if provided
        if hyperparameters:
            config['params'].update(hyperparameters)
        
        return config['class'](**config['params'])
    
    def _prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        # Remove non-feature columns
        exclude_cols = ['timestamp', 'location', 'data_type']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols].drop('value', axis=1)
        y = features_df['value']
        
        return X, y
    
    def _prepare_prediction_features(self, features_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Prepare features for prediction."""
        # Get expected feature names from model metadata
        expected_features = self.model_metadata[model_name]['feature_names']
        
        # Select and order features according to training
        available_features = [col for col in expected_features if col in features_df.columns]
        missing_features = [col for col in expected_features if col not in features_df.columns]
        
        if missing_features:
            logger.warning(f"Missing features for prediction: {missing_features}")
        
        X = features_df[available_features]
        
        # Fill missing features with zeros or median values
        for feature in missing_features:
            X[feature] = 0  # or use appropriate default value
        
        # Ensure correct column order
        X = X.reindex(columns=expected_features, fill_value=0)
        
        return X
    
    def _evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }
        
        return metrics
    
    def _calculate_confidence_intervals(
        self,
        model,
        X: pd.DataFrame,
        predictions: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence intervals for predictions."""
        # Simple approach using prediction variance
        # For ensemble models, this could use prediction variance
        # For now, use a simple percentage-based approach
        
        std_factor = 1.96  # 95% confidence interval
        prediction_std = np.std(predictions) * 0.1  # Assume 10% of prediction std
        
        margin = std_factor * prediction_std
        lower_bounds = predictions - margin
        upper_bounds = predictions + margin
        
        return lower_bounds, upper_bounds
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                return dict(zip(feature_names, importance_scores.tolist()))
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                importance_scores = np.abs(model.coef_)
                return dict(zip(feature_names, importance_scores.tolist()))
            else:
                # Fallback: equal importance
                return dict(zip(feature_names, [1.0 / len(feature_names)] * len(feature_names)))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}
    
    def _get_default_model(self) -> str:
        """Get the name of the default model to use for predictions."""
        if not self.models:
            raise ValueError("No trained models available")
        
        # Return the most recently trained model
        return max(
            self.model_metadata.keys(),
            key=lambda x: self.model_metadata[x]['trained_at']
        )
    
    async def save_model(self, model_name: str, filepath: str) -> None:
        """Save a trained model to disk."""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model_data = {
                'model': self.models[model_name],
                'metadata': self.model_metadata[model_name]
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    async def load_model(self, filepath: str) -> str:
        """Load a trained model from disk."""
        try:
            model_data = joblib.load(filepath)
            
            model_name = f"loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.models[model_name] = model_data['model']
            self.model_metadata[model_name] = model_data['metadata']
            
            logger.info(f"Model loaded as {model_name}")
            return model_name
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
