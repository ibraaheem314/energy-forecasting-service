"""Tests for the model training and prediction services."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from unittest.mock import Mock, patch

# Import the model service
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.models import ModelService

class TestModelService:
    """Test cases for the ModelService class."""
    
    @pytest.fixture
    def model_service(self):
        """Create a ModelService instance for testing."""
        return ModelService()
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='H')
        n_samples = len(dates)
        
        # Create features
        features_data = {
            'timestamp': dates,
            'value': 1000 + 200 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 50, n_samples),
            'temperature': 20 + 10 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 2, n_samples),
            'humidity': 50 + 20 * np.sin(2 * np.pi * dates.hour / 24 + np.pi/4) + np.random.normal(0, 5, n_samples),
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'location': ['test_region'] * n_samples,
            'data_type': ['consumption'] * n_samples
        }
        
        return pd.DataFrame(features_data)
    
    def test_model_service_initialization(self, model_service):
        """Test model service initialization."""
        assert isinstance(model_service.models, dict)
        assert isinstance(model_service.model_metadata, dict)
        assert isinstance(model_service.model_configs, dict)
        
        # Check that model configurations are available
        expected_models = ['random_forest', 'gradient_boosting', 'linear_regression', 'ridge']
        for model_type in expected_models:
            assert model_type in model_service.model_configs
    
    def test_create_model(self, model_service):
        """Test model creation with different configurations."""
        # Test random forest creation
        rf_model = model_service._create_model('random_forest', None)
        assert rf_model is not None
        assert hasattr(rf_model, 'fit')
        assert hasattr(rf_model, 'predict')
        
        # Test with custom hyperparameters
        custom_params = {'n_estimators': 50, 'max_depth': 5}
        rf_custom = model_service._create_model('random_forest', custom_params)
        assert rf_custom.n_estimators == 50
        assert rf_custom.max_depth == 5
        
        # Test invalid model type
        with pytest.raises(ValueError):
            model_service._create_model('invalid_model', None)
    
    def test_prepare_training_data(self, model_service, sample_training_data):
        """Test training data preparation."""
        X, y = model_service._prepare_training_data(sample_training_data)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) == len(sample_training_data)
        
        # Check that target variable is excluded from features
        assert 'value' not in X.columns
        assert 'timestamp' not in X.columns
        assert 'location' not in X.columns
        assert 'data_type' not in X.columns
        
        # Check that features are included
        assert 'temperature' in X.columns
        assert 'humidity' in X.columns
        assert 'hour' in X.columns
    
    def test_evaluate_model(self, model_service, sample_training_data):
        """Test model evaluation."""
        # Prepare data
        X, y = model_service._prepare_training_data(sample_training_data)
        
        # Train a simple model
        model = model_service._create_model('linear_regression', None)
        model.fit(X, y)
        
        # Evaluate
        metrics = model_service._evaluate_model(model, X, y)
        
        assert isinstance(metrics, dict)
        expected_metrics = ['mae', 'mse', 'rmse', 'r2', 'mape']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])
        
        # Check metric ranges
        assert metrics['r2'] <= 1.0
        assert metrics['mae'] >= 0
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mape'] >= 0
    
    @pytest.mark.asyncio
    async def test_train_model(self, model_service):
        """Test complete model training workflow."""
        # Mock the feature service to avoid dependency
        with patch('app.services.models.FeatureService') as mock_feature_service:
            # Create mock training data
            dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
            mock_features = pd.DataFrame({
                'timestamp': dates,
                'value': 1000 + np.random.normal(0, 100, len(dates)),
                'temperature': 20 + np.random.normal(0, 5, len(dates)),
                'hour': dates.hour,
                'day_of_week': dates.dayofweek,
                'location': ['test_region'] * len(dates),
                'data_type': ['consumption'] * len(dates)
            })
            
            mock_feature_service.return_value.prepare_features.return_value = mock_features
            
            # Train model
            result = await model_service.train_model(
                model_type='linear_regression',
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now(),
                location='test_region',
                test_size=0.2
            )
            
            assert isinstance(result, dict)
            assert 'model_name' in result
            assert 'metrics' in result
            assert 'run_id' in result
            assert 'feature_importance' in result
            
            # Check that model was stored
            model_name = result['model_name']
            assert model_name in model_service.models
            assert model_name in model_service.model_metadata
    
    @pytest.mark.asyncio 
    async def test_predict(self, model_service, sample_training_data):
        """Test model prediction."""
        # First train a model
        X, y = model_service._prepare_training_data(sample_training_data)
        model = model_service._create_model('linear_regression', None)
        model.fit(X, y)
        
        # Store model in service
        model_name = 'test_model'
        model_service.models[model_name] = model
        model_service.model_metadata[model_name] = {
            'type': 'linear_regression',
            'location': 'test_region',
            'trained_at': datetime.now(),
            'metrics': {'mae': 50.0, 'r2': 0.8},
            'feature_names': X.columns.tolist()
        }
        
        # Prepare prediction data (subset of training data)
        prediction_data = sample_training_data.head(24)  # First 24 hours
        
        # Generate predictions
        predictions = await model_service.predict(
            prediction_data, 
            model_name=model_name, 
            confidence_interval=True
        )
        
        assert isinstance(predictions, list)
        assert len(predictions) == len(prediction_data)
        
        for pred in predictions:
            assert isinstance(pred, dict)
            assert 'timestamp' in pred
            assert 'value' in pred
            assert 'confidence' in pred
            assert 'lower_bound' in pred
            assert 'upper_bound' in pred
            
            assert isinstance(pred['value'], float)
            assert pred['confidence'] == 0.95
    
    def test_prepare_prediction_features(self, model_service, sample_training_data):
        """Test prediction feature preparation."""
        # Set up model metadata
        expected_features = ['temperature', 'humidity', 'hour', 'day_of_week', 'is_weekend']
        model_name = 'test_model'
        model_service.model_metadata[model_name] = {
            'feature_names': expected_features
        }
        
        # Test with complete features
        X = model_service._prepare_prediction_features(sample_training_data, model_name)
        
        assert isinstance(X, pd.DataFrame)
        assert list(X.columns) == expected_features
        assert len(X) == len(sample_training_data)
        
        # Test with missing features
        incomplete_data = sample_training_data[['timestamp', 'temperature', 'hour']].copy()
        X_incomplete = model_service._prepare_prediction_features(incomplete_data, model_name)
        
        assert list(X_incomplete.columns) == expected_features
        assert len(X_incomplete) == len(incomplete_data)
        # Missing features should be filled with 0
        assert X_incomplete['humidity'].iloc[0] == 0
    
    def test_calculate_confidence_intervals(self, model_service):
        """Test confidence interval calculation."""
        # Create mock model and predictions
        mock_model = Mock()
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        predictions = np.array([100, 200, 300])
        
        lower_bounds, upper_bounds = model_service._calculate_confidence_intervals(
            mock_model, X, predictions
        )
        
        assert isinstance(lower_bounds, np.ndarray)
        assert isinstance(upper_bounds, np.ndarray)
        assert len(lower_bounds) == len(predictions)
        assert len(upper_bounds) == len(predictions)
        
        # Check that bounds make sense
        assert (lower_bounds <= predictions).all()
        assert (predictions <= upper_bounds).all()
    
    def test_get_feature_importance(self, model_service):
        """Test feature importance extraction."""
        feature_names = ['temp', 'humidity', 'hour', 'day']
        
        # Test with model that has feature_importances_
        mock_model_rf = Mock()
        mock_model_rf.feature_importances_ = np.array([0.4, 0.3, 0.2, 0.1])
        
        importance = model_service._get_feature_importance(mock_model_rf, feature_names)
        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)
        assert importance['temp'] == 0.4
        
        # Test with linear model (coef_)
        mock_model_linear = Mock()
        mock_model_linear.coef_ = np.array([0.5, -0.3, 0.2, 0.1])
        del mock_model_linear.feature_importances_  # Ensure it doesn't have feature_importances_
        
        importance_linear = model_service._get_feature_importance(mock_model_linear, feature_names)
        assert isinstance(importance_linear, dict)
        assert importance_linear['temp'] == 0.5
        assert importance_linear['humidity'] == 0.3  # Should be absolute value
    
    @pytest.mark.asyncio
    async def test_get_status(self, model_service):
        """Test model service status retrieval."""
        # Add some mock models
        model_service.models['model1'] = Mock()
        model_service.models['model2'] = Mock()
        model_service.model_metadata['model1'] = {
            'type': 'random_forest',
            'trained_at': datetime.now() - timedelta(hours=1),
            'metrics': {'mae': 45.2, 'r2': 0.94}
        }
        model_service.model_metadata['model2'] = {
            'type': 'linear_regression', 
            'trained_at': datetime.now(),
            'metrics': {'mae': 52.1, 'r2': 0.88}
        }
        
        status = await model_service.get_status()
        
        assert isinstance(status, dict)
        assert 'active_models' in status
        assert 'total_models' in status
        assert 'latest_model' in status
        
        assert len(status['active_models']) == 2
        assert status['total_models'] == 2
        assert status['latest_model']['name'] == 'model2'  # Most recent
    
    def test_get_default_model(self, model_service):
        """Test default model selection."""
        # Test with no models
        with pytest.raises(ValueError):
            model_service._get_default_model()
        
        # Add models with different training times
        model_service.model_metadata['old_model'] = {
            'trained_at': datetime.now() - timedelta(hours=2)
        }
        model_service.model_metadata['new_model'] = {
            'trained_at': datetime.now()
        }
        
        default_model = model_service._get_default_model()
        assert default_model == 'new_model'
    
    @pytest.mark.asyncio
    async def test_save_and_load_model(self, model_service, tmp_path):
        """Test model saving and loading."""
        # Create and train a simple model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        model.fit(X, y)
        
        # Store in service
        model_name = 'test_model'
        model_service.models[model_name] = model
        model_service.model_metadata[model_name] = {
            'type': 'linear_regression',
            'trained_at': datetime.now(),
            'metrics': {'mae': 0.1, 'r2': 0.99}
        }
        
        # Save model
        filepath = tmp_path / "test_model.joblib"
        await model_service.save_model(model_name, str(filepath))
        
        assert filepath.exists()
        
        # Clear service and load model
        model_service.models.clear()
        model_service.model_metadata.clear()
        
        loaded_name = await model_service.load_model(str(filepath))
        
        assert loaded_name in model_service.models
        assert loaded_name in model_service.model_metadata
        
        # Test that loaded model works
        loaded_model = model_service.models[loaded_name]
        prediction = loaded_model.predict([[7, 8]])
        assert isinstance(prediction, np.ndarray)

class TestModelServiceIntegration:
    """Integration tests for model service."""
    
    @pytest.mark.asyncio
    async def test_full_training_pipeline(self):
        """Test the complete training pipeline."""
        # This would test the full pipeline from data loading to model registration
        # For now, we'll keep it as a placeholder
        pass
    
    @pytest.mark.asyncio 
    async def test_prediction_pipeline(self):
        """Test the complete prediction pipeline."""
        # This would test the full pipeline from feature preparation to prediction
        pass

class TestModelServicePerformance:
    """Performance tests for model service."""
    
    def test_training_performance(self, model_service):
        """Test that training completes in reasonable time."""
        import time
        
        # Create larger dataset
        n_samples = 1000
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples)
        })
        y = X['feature1'] + X['feature2'] * 0.5 + np.random.normal(0, 0.1, n_samples)
        
        # Train model and measure time
        start_time = time.time()
        model = model_service._create_model('linear_regression', None)
        model.fit(X, y)
        end_time = time.time()
        
        # Should complete quickly for linear regression
        assert (end_time - start_time) < 1.0  # Less than 1 second
    
    def test_prediction_performance(self, model_service):
        """Test prediction performance."""
        import time
        
        # Train a model
        X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        y_train = X_train['feature1'] + X_train['feature2'] + np.random.normal(0, 0.1, 100)
        
        model = model_service._create_model('linear_regression', None)
        model.fit(X_train, y_train)
        
        # Test prediction time
        X_test = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000)
        })
        
        start_time = time.time()
        predictions = model.predict(X_test)
        end_time = time.time()
        
        assert len(predictions) == 1000
        assert (end_time - start_time) < 0.1  # Should be very fast

if __name__ == "__main__":
    pytest.main([__file__])
