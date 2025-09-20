"""Modèles améliorés avec hyperparameter tuning et features avancées."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, Optional
import joblib
from .models import BaseModel

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class EnhancedLinearModel(BaseModel):
    """Modèle linéaire amélioré avec régularisation."""
    
    def __init__(self, regularization='ridge'):
        super().__init__(f"Enhanced {regularization.title()}")
        self.regularization = regularization
        self.best_params = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entraîner avec hyperparameter tuning."""
        if self.regularization == 'ridge':
            model = Ridge()
            param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        elif self.regularization == 'elastic':
            model = ElasticNet()
            param_grid = {
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            }
        else:
            raise ValueError("Regularization must be 'ridge' or 'elastic'")
        
        # Time series split pour éviter le data leakage
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        
        return self


class EnhancedRandomForestModel(BaseModel):
    """Random Forest avec hyperparameter tuning."""
    
    def __init__(self):
        super().__init__("Enhanced Random Forest")
        self.best_params = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entraîner avec hyperparameter tuning."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        
        return self


class EnhancedLightGBMModel(BaseModel):
    """LightGBM avec hyperparameter tuning."""
    
    def __init__(self):
        super().__init__("Enhanced LightGBM")
        self.best_params = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entraîner avec hyperparameter tuning."""
        if HAS_LIGHTGBM:
            param_grid = {
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.05, 0.1, 0.2],
                'feature_fraction': [0.8, 0.9, 1.0],
                'n_estimators': [100, 200, 300]
            }
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            grid_search = GridSearchCV(
                lgb.LGBMRegressor(
                    objective='regression',
                    metric='rmse',
                    verbose=-1,
                    random_state=42
                ),
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
        else:
            # Fallback vers Gradient Boosting
            self.name = "Enhanced Gradient Boosting"
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            grid_search = GridSearchCV(
                GradientBoostingRegressor(random_state=42),
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
        
        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        
        return self


class EnsembleModel(BaseModel):
    """Modèle d'ensemble combinant plusieurs algorithmes."""
    
    def __init__(self, models=None):
        super().__init__("Ensemble Model")
        self.models = models or [
            EnhancedLinearModel('ridge'),
            EnhancedRandomForestModel(),
            EnhancedLightGBMModel()
        ]
        self.weights = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entraîner tous les modèles de l'ensemble."""
        print(f"Entraînement de {len(self.models)} modèles pour l'ensemble...")
        
        for i, model in enumerate(self.models):
            print(f"  Modèle {i+1}/{len(self.models)}: {model.name}")
            model.fit(X, y)
        
        # Calculer les poids basés sur les performances en validation
        self._calculate_weights(X, y)
        
        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        
        return self
    
    def _calculate_weights(self, X: pd.DataFrame, y: pd.Series):
        """Calculer les poids optimaux pour chaque modèle."""
        # Split pour validation
        split_idx = int(len(X) * 0.8)
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        
        scores = []
        for model in self.models:
            try:
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                scores.append(max(score, 0))  # Score minimum de 0
            except:
                scores.append(0)
        
        # Normaliser les scores en poids
        total_score = sum(scores)
        if total_score > 0:
            self.weights = [s / total_score for s in scores]
        else:
            self.weights = [1/len(self.models)] * len(self.models)
        
        print(f"  Poids calculés: {[f'{w:.3f}' for w in self.weights]}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédiction par ensemble pondéré."""
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        predictions = []
        for model, weight in zip(self.models, self.weights):
            try:
                pred = model.predict(X)
                predictions.append(pred * weight)
            except:
                predictions.append(np.zeros(len(X)))
        
        return np.sum(predictions, axis=0)
    
    def get_model_contributions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Obtenir les contributions individuelles de chaque modèle."""
        contributions = {}
        for model, weight in zip(self.models, self.weights):
            try:
                pred = model.predict(X)
                contributions[model.name] = pred * weight
            except:
                contributions[model.name] = np.zeros(len(X))
        
        return contributions


def create_enhanced_model(model_type: str, **params):
    """Factory pour créer les modèles améliorés."""
    if model_type == "enhanced_linear":
        return EnhancedLinearModel(params.get('regularization', 'ridge'))
    elif model_type == "enhanced_random_forest":
        return EnhancedRandomForestModel()
    elif model_type == "enhanced_lightgbm":
        return EnhancedLightGBMModel()
    elif model_type == "ensemble":
        return EnsembleModel()
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")


# Mise à jour de la fonction create_model pour inclure les modèles améliorés
def create_model_enhanced(model_type: str, **params):
    """Factory étendue incluant les modèles améliorés."""
    # Modèles améliorés
    if model_type.startswith("enhanced_") or model_type == "ensemble":
        return create_enhanced_model(model_type, **params)
    
    # Modèles standards (import depuis models.py)
    from .models import create_model
    return create_model(model_type, **params)
