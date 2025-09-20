"""Modèles de prévision énergétique : SARIMAX, LightGBM, quantile, expectile."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class BaseModel:
    """Classe de base pour tous les modèles."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entraîner le modèle."""
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faire des prédictions."""
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        return self.model.predict(X)
        
    def save(self, path: str):
        """Sauvegarder le modèle."""
        model_info = {
            'model': self.model,
            'name': self.name,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_info, path)
        
    def load(self, path: str):
        """Charger le modèle."""
        model_info = joblib.load(path)
        self.model = model_info['model']
        self.name = model_info['name']
        self.feature_names = model_info['feature_names']
        self.is_fitted = model_info['is_fitted']
        
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Évaluer le modèle."""
        y_pred = self.predict(X)
        
        return {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100
        }


class LinearRegressionModel(BaseModel):
    """Modèle de régression linéaire."""
    
    def __init__(self):
        super().__init__("Linear Regression")
        self.model = LinearRegression()


class RandomForestModel(BaseModel):
    """Modèle Random Forest."""
    
    def __init__(self, n_estimators=100, random_state=42):
        super().__init__("Random Forest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )


class LightGBMModel(BaseModel):
    """Modèle LightGBM."""
    
    def __init__(self, **params):
        super().__init__("LightGBM")
        if HAS_LIGHTGBM:
            default_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            default_params.update(params)
            self.model = lgb.LGBMRegressor(**default_params)
        else:
            # Fallback vers GradientBoostingRegressor
            print("LightGBM non disponible, utilisation de GradientBoostingRegressor")
            self.name = "Gradient Boosting (fallback)"
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )


class GradientBoostingQuantileModel(BaseModel):
    """Modèle Gradient Boosting pour prédictions quantiles."""
    
    def __init__(self, quantiles=None, **params):
        super().__init__("Gradient Boosting Quantile")
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.models = {}
        
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42
        }
        default_params.update(params)
        self.params = default_params
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entraîner les modèles pour chaque quantile."""
        self.feature_names = X.columns.tolist()
        
        for quantile in self.quantiles:
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=quantile,
                **self.params
            )
            model.fit(X, y)
            self.models[quantile] = model
            
        self.is_fitted = True
        return self
        
    def predict(self, X: pd.DataFrame) -> dict:
        """Faire des prédictions quantiles."""
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
            
        predictions = {}
        for quantile, model in self.models.items():
            predictions[f'q{int(quantile*100)}'] = model.predict(X)
            
        return predictions
        
    def predict_median(self, X: pd.DataFrame) -> np.ndarray:
        """Prédiction médiane (quantile 0.5)."""
        if 0.5 not in self.models:
            raise ValueError("Quantile 0.5 non disponible")
        return self.models[0.5].predict(X)
        
    def save(self, path: str):
        """Sauvegarder le modèle."""
        model_info = {
            'models': self.models,
            'name': self.name,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'quantiles': self.quantiles,
            'params': self.params
        }
        joblib.dump(model_info, path)
        
    def load(self, path: str):
        """Charger le modèle."""
        model_info = joblib.load(path)
        self.models = model_info['models']
        self.name = model_info['name']
        self.feature_names = model_info['feature_names']
        self.is_fitted = model_info['is_fitted']
        self.quantiles = model_info['quantiles']
        self.params = model_info['params']


def create_model(model_type: str, **params):
    """Factory pour créer les modèles."""
    if model_type == "linear":
        return LinearRegressionModel()
    elif model_type == "random_forest":
        return RandomForestModel(**params)
    elif model_type == "lightgbm":
        return LightGBMModel(**params)
    elif model_type == "gradient_boosting_quantile":
        return GradientBoostingQuantileModel(**params)
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")
