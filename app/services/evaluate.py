"""Model evaluation and performance monitoring services."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from .models import ModelService
from .features import FeatureService
from .loader import DataLoader

logger = logging.getLogger(__name__)

class EvaluationService:
    """Service for comprehensive model evaluation and monitoring."""
    
    def __init__(self):
        self.model_service = ModelService()
        self.feature_service = FeatureService()
        self.data_loader = DataLoader()
    
    async def evaluate_model_performance(
        self,
        model_name: str,
        start_date: datetime,
        end_date: datetime,
        location: str,
        evaluation_type: str = "holdout"
    ) -> Dict[str, Any]:
        """Comprehensive model performance evaluation.
        
        Args:
            model_name: Name of the model to evaluate
            start_date: Start date for evaluation data
            end_date: End date for evaluation data
            location: Location identifier
            evaluation_type: Type of evaluation ('holdout', 'time_series_cv', 'walk_forward')
            
        Returns:
            Detailed evaluation results
        """
        try:
            logger.info(f"Evaluating model {model_name} with {evaluation_type} validation")
            
            # Prepare evaluation data
            features_df = await self.feature_service.prepare_features(
                start_date, end_date, location
            )
            
            if evaluation_type == "holdout":
                results = await self._holdout_evaluation(model_name, features_df)
            elif evaluation_type == "time_series_cv":
                results = await self._time_series_cv_evaluation(model_name, features_df)
            elif evaluation_type == "walk_forward":
                results = await self._walk_forward_evaluation(model_name, features_df)
            else:
                raise ValueError(f"Unsupported evaluation type: {evaluation_type}")
            
            # Add model metadata
            results['model_name'] = model_name
            results['evaluation_type'] = evaluation_type
            results['location'] = location
            results['evaluation_period'] = {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
            results['evaluated_at'] = datetime.now().isoformat()
            
            logger.info(f"Model evaluation completed. Overall MAE: {results.get('overall_mae', 'N/A')}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            raise
    
    async def _holdout_evaluation(self, model_name: str, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform holdout validation."""
        # Split data (80% train, 20% test)
        split_idx = int(len(features_df) * 0.8)
        train_df = features_df[:split_idx]
        test_df = features_df[split_idx:]
        
        # Generate predictions for test set
        predictions = await self.model_service.predict(test_df, model_name, confidence_interval=True)
        
        # Extract actual values and predictions
        y_true = test_df['value'].values
        y_pred = np.array([p['value'] for p in predictions])
        
        # Calculate metrics
        metrics = self._calculate_detailed_metrics(y_true, y_pred)
        
        # Calculate residual statistics
        residuals = y_true - y_pred
        residual_stats = self._calculate_residual_statistics(residuals)
        
        return {
            'overall_mae': metrics['mae'],
            'overall_mse': metrics['mse'],
            'overall_rmse': metrics['rmse'],
            'overall_r2': metrics['r2'],
            'overall_mape': metrics['mape'],
            'metrics': metrics,
            'residual_statistics': residual_stats,
            'predictions': predictions[:100],  # Limit for response size
            'sample_size': len(test_df)
        }
    
    async def _time_series_cv_evaluation(self, model_name: str, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=5)
        cv_results = []
        
        X, y = self._prepare_features_target(features_df)
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Note: This is a simplified version. In practice, you'd retrain the model
            # For now, we'll use the existing model for prediction
            test_features_df = features_df.iloc[test_idx]
            predictions = await self.model_service.predict(test_features_df, model_name, confidence_interval=False)
            
            y_pred = np.array([p['value'] for p in predictions])
            metrics = self._calculate_detailed_metrics(y_test.values, y_pred)
            
            cv_results.append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'metrics': metrics
            })
        
        # Aggregate CV results
        avg_metrics = self._aggregate_cv_metrics(cv_results)
        
        return {
            'cv_results': cv_results,
            'average_metrics': avg_metrics,
            'n_folds': len(cv_results)
        }
    
    async def _walk_forward_evaluation(self, model_name: str, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform walk-forward validation."""
        window_size = max(100, len(features_df) // 10)  # Use 10% as minimum window
        step_size = max(24, window_size // 10)  # Step by 10% of window
        
        results = []
        
        for start_idx in range(0, len(features_df) - window_size, step_size):
            end_idx = start_idx + window_size
            window_df = features_df[start_idx:end_idx]
            
            # Use last 20% as test set
            test_start = int(len(window_df) * 0.8)
            test_df = window_df[test_start:]
            
            predictions = await self.model_service.predict(test_df, model_name, confidence_interval=False)
            
            y_true = test_df['value'].values
            y_pred = np.array([p['value'] for p in predictions])
            
            metrics = self._calculate_detailed_metrics(y_true, y_pred)
            
            results.append({
                'window_start': start_idx,
                'window_end': end_idx,
                'test_size': len(test_df),
                'metrics': metrics
            })
        
        # Aggregate results
        avg_metrics = self._aggregate_walk_forward_metrics(results)
        
        return {
            'walk_forward_results': results,
            'average_metrics': avg_metrics,
            'n_windows': len(results)
        }
    
    def _prepare_features_target(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables."""
        exclude_cols = ['timestamp', 'location', 'data_type']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols].drop('value', axis=1)
        y = features_df['value']
        
        return X, y
    
    def _calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'max_error': np.max(np.abs(y_true - y_pred)),
            'median_absolute_error': np.median(np.abs(y_true - y_pred)),
            'explained_variance': 1 - np.var(y_true - y_pred) / np.var(y_true)
        }
        
        # Add directional accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics['directional_accuracy'] = np.mean(true_direction == pred_direction) * 100
        
        return metrics
    
    def _calculate_residual_statistics(self, residuals: np.ndarray) -> Dict[str, float]:
        """Calculate residual statistics for model diagnostics."""
        return {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'q25': np.percentile(residuals, 25),
            'median': np.median(residuals),
            'q75': np.percentile(residuals, 75),
            'skewness': self._calculate_skewness(residuals),
            'kurtosis': self._calculate_kurtosis(residuals)
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0
    
    def _aggregate_cv_metrics(self, cv_results: List[Dict]) -> Dict[str, float]:
        """Aggregate cross-validation metrics."""
        metrics_keys = cv_results[0]['metrics'].keys()
        aggregated = {}
        
        for key in metrics_keys:
            values = [result['metrics'][key] for result in cv_results]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    def _aggregate_walk_forward_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate walk-forward validation metrics."""
        metrics_keys = results[0]['metrics'].keys()
        aggregated = {}
        
        for key in metrics_keys:
            values = [result['metrics'][key] for result in results]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_trend'] = self._calculate_trend(values)
        
        return aggregated
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in metric values over time."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Return slope
    
    async def compare_models(
        self,
        model_names: List[str],
        start_date: datetime,
        end_date: datetime,
        location: str
    ) -> Dict[str, Any]:
        """Compare performance of multiple models.
        
        Args:
            model_names: List of model names to compare
            start_date: Start date for comparison data
            end_date: End date for comparison data
            location: Location identifier
            
        Returns:
            Comparison results
        """
        try:
            logger.info(f"Comparing models: {model_names}")
            
            # Prepare test data
            features_df = await self.feature_service.prepare_features(
                start_date, end_date, location
            )
            
            # Use last 30% as test set for comparison
            test_start = int(len(features_df) * 0.7)
            test_df = features_df[test_start:]
            
            comparison_results = {}
            
            for model_name in model_names:
                try:
                    predictions = await self.model_service.predict(
                        test_df, model_name, confidence_interval=False
                    )
                    
                    y_true = test_df['value'].values
                    y_pred = np.array([p['value'] for p in predictions])
                    
                    metrics = self._calculate_detailed_metrics(y_true, y_pred)
                    
                    comparison_results[model_name] = {
                        'metrics': metrics,
                        'prediction_count': len(predictions)
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate model {model_name}: {e}")
                    comparison_results[model_name] = {
                        'error': str(e),
                        'metrics': None
                    }
            
            # Rank models by MAE (lower is better)
            valid_models = {k: v for k, v in comparison_results.items() 
                          if v.get('metrics') is not None}
            
            if valid_models:
                ranked_models = sorted(
                    valid_models.keys(),
                    key=lambda x: valid_models[x]['metrics']['mae']
                )
            else:
                ranked_models = []
            
            return {
                'comparison_results': comparison_results,
                'ranked_models': ranked_models,
                'best_model': ranked_models[0] if ranked_models else None,
                'comparison_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'test_samples': len(test_df)
            }
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise
    
    async def generate_evaluation_report(
        self,
        model_name: str,
        evaluation_results: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            model_name: Name of the evaluated model
            evaluation_results: Results from model evaluation
            
        Returns:
            Formatted evaluation report
        """
        try:
            report_lines = [
                f"Model Evaluation Report",
                f"=" * 50,
                f"Model Name: {model_name}",
                f"Evaluation Date: {evaluation_results.get('evaluated_at', 'Unknown')}",
                f"Evaluation Type: {evaluation_results.get('evaluation_type', 'Unknown')}",
                f"Location: {evaluation_results.get('location', 'Unknown')}",
                "",
                "Performance Metrics:",
                f"- Mean Absolute Error (MAE): {evaluation_results.get('overall_mae', 'N/A'):.4f}",
                f"- Root Mean Square Error (RMSE): {evaluation_results.get('overall_rmse', 'N/A'):.4f}",
                f"- R² Score: {evaluation_results.get('overall_r2', 'N/A'):.4f}",
                f"- Mean Absolute Percentage Error (MAPE): {evaluation_results.get('overall_mape', 'N/A'):.2f}%",
                ""
            ]
            
            # Add residual statistics if available
            if 'residual_statistics' in evaluation_results:
                residual_stats = evaluation_results['residual_statistics']
                report_lines.extend([
                    "Residual Statistics:",
                    f"- Mean: {residual_stats.get('mean', 'N/A'):.4f}",
                    f"- Standard Deviation: {residual_stats.get('std', 'N/A'):.4f}",
                    f"- Skewness: {residual_stats.get('skewness', 'N/A'):.4f}",
                    f"- Kurtosis: {residual_stats.get('kurtosis', 'N/A'):.4f}",
                    ""
                ])
            
            # Add CV results if available
            if 'cv_results' in evaluation_results:
                avg_metrics = evaluation_results.get('average_metrics', {})
                report_lines.extend([
                    "Cross-Validation Results:",
                    f"- Average MAE: {avg_metrics.get('mae_mean', 'N/A'):.4f} ± {avg_metrics.get('mae_std', 'N/A'):.4f}",
                    f"- Average RMSE: {avg_metrics.get('rmse_mean', 'N/A'):.4f} ± {avg_metrics.get('rmse_std', 'N/A'):.4f}",
                    f"- Average R²: {avg_metrics.get('r2_mean', 'N/A'):.4f} ± {avg_metrics.get('r2_std', 'N/A'):.4f}",
                    ""
                ])
            
            report_lines.extend([
                "Evaluation Summary:",
                f"- Sample Size: {evaluation_results.get('sample_size', 'N/A')}",
                f"- Model Status: {'Good' if evaluation_results.get('overall_r2', 0) > 0.7 else 'Needs Improvement'}",
                ""
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Failed to generate evaluation report: {e}")
            return f"Error generating report: {str(e)}"
