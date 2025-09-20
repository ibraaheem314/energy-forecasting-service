"""Tests pour les modèles."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Ajouter le chemin src
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import create_model, LinearRegressionModel, RandomForestModel


class TestModelCreation:
    """Tests pour la création des modèles."""
    
    def test_create_linear_model(self):
        """Test création modèle linéaire."""
        model = create_model("linear")
        assert isinstance(model, LinearRegressionModel)
        assert model.name == "Linear Regression"
        
    def test_create_random_forest_model(self):
        """Test création modèle Random Forest."""
        model = create_model("random_forest")
        assert isinstance(model, RandomForestModel)
        assert model.name == "Random Forest"
        
    def test_invalid_model_type(self):
        """Test avec type de modèle invalide."""
        with pytest.raises(ValueError):
            create_model("invalid_model")


class TestModelTraining:
    """Tests pour l'entraînement des modèles."""
    
    def setup_method(self):
        """Préparer les données de test."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='h'
        )
        
        # Données synthétiques simples
        y_values = 100 + np.sin(2 * np.pi * np.arange(len(dates)) / 24) * 10 + np.random.normal(0, 5, len(dates))
        
        self.df = pd.DataFrame({
            'y': y_values,
            'feature1': np.random.normal(0, 1, len(dates)),
            'feature2': np.random.normal(0, 1, len(dates))
        }, index=dates)
        
        self.X = self.df[['feature1', 'feature2']]
        self.y = self.df['y']
    
    def test_linear_model_training(self):
        """Test entraînement modèle linéaire."""
        model = create_model("linear")
        model.fit(self.X, self.y)
        
        assert model.is_fitted
        assert model.feature_names == ['feature1', 'feature2']
        
        # Test prédiction
        predictions = model.predict(self.X.head(10))
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)
    
    def test_model_evaluation(self):
        """Test évaluation du modèle."""
        model = create_model("linear")
        model.fit(self.X, self.y)
        
        metrics = model.evaluate(self.X.head(100), self.y.head(100))
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        
        assert isinstance(metrics['mae'], float)
        assert isinstance(metrics['rmse'], float)
        assert isinstance(metrics['r2'], float)
        assert isinstance(metrics['mape'], float)


class TestModelSaveLoad:
    """Tests pour la sauvegarde et le chargement des modèles."""
    
    def test_model_save_load(self, tmp_path):
        """Test sauvegarde et chargement."""
        # Données d'entraînement
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = pd.Series([10, 20, 30])
        
        # Entraîner et sauvegarder
        model = create_model("linear")
        model.fit(X, y)
        
        save_path = tmp_path / "test_model.joblib"
        model.save(str(save_path))
        
        # Charger et tester
        new_model = create_model("linear")
        new_model.load(str(save_path))
        
        assert new_model.is_fitted
        assert new_model.name == "Linear Regression"
        assert new_model.feature_names == ['feature1', 'feature2']
        
        # Vérifier que les prédictions sont identiques
        pred1 = model.predict(X)
        pred2 = new_model.predict(X)
        np.testing.assert_array_almost_equal(pred1, pred2)