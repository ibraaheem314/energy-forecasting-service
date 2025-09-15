"""Classe de base pour tous les modèles individuels."""
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.train_simple import create_features


class BaseModel:
    """Classe de base pour tous les modèles."""
    
    def __init__(self, model_name: str, model_class, model_params: dict = None):
        self.model_name = model_name
        self.model_class = model_class
        self.model_params = model_params or {}
        self.model = None
        self.feature_names = None
        self.metrics = {}
        
    def load_data(self, data_source="odre", force_reload=False):
        """Charger les données et créer les features (avec cache)."""
        # Import du cache ici pour éviter les imports circulaires
        sys.path.append(str(Path(__file__).parent.parent))
        from scripts.data_cache import get_cached_data
        
        print(f"Chargement données {data_source}...")
        df_features = get_cached_data(data_source, force_reload=force_reload)
        
        print(f"Données prêtes: {len(df_features)} échantillons")
        return df_features
    
    def prepare_data(self, df, test_size=0.2):
        """Préparer les données train/test."""
        feature_cols = [col for col in df.columns if col != 'y']
        X = df[feature_cols]
        y = df['y']
        
        # Split train/test
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.feature_names = X.columns.tolist()
        
        print(f"Train: {len(X_train)} échantillons")
        print(f"Test: {len(X_test)} échantillons")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """Entraîner le modèle."""
        print(f"Entraînement {self.model_name}...")
        
        self.model = self.model_class(**self.model_params)
        self.model.fit(X_train, y_train)
        
        print(f"{self.model_name} entraîné")
    
    def evaluate(self, X_test, y_test):
        """Évaluer le modèle."""
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        print(f"Métriques {self.model_name}:")
        print(f"   - MAE: {self.metrics['mae']:.2f}")
        print(f"   - RMSE: {self.metrics['rmse']:.2f}")
        print(f"   - R²: {self.metrics['r2']:.4f}")
        print(f"   - MAPE: {self.metrics['mape']:.2f}%")
        
        return self.metrics
    
    def save(self, models_dir="models"):
        """Sauvegarder le modèle."""
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.model_name}_{timestamp}.joblib"
        filepath = models_path / filename
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'model_name': self.model_name,
            'trained_at': datetime.now().isoformat(),
            'model_params': self.model_params
        }
        
        joblib.dump(model_data, filepath)
        print(f"Sauvegardé: {filepath}")
        
        return filepath
    
    def run_full_pipeline(self, data_source="odre", test_size=0.2):
        """Exécuter le pipeline complet."""
        print(f"Pipeline {self.model_name}")
        print("=" * 50)
        
        # 1. Charger données
        df = self.load_data(data_source)
        
        # 2. Préparer train/test
        X_train, X_test, y_train, y_test = self.prepare_data(df, test_size)
        
        # 3. Entraîner
        self.train(X_train, y_train)
        
        # 4. Évaluer
        metrics = self.evaluate(X_test, y_test)
        
        # 5. Sauvegarder
        filepath = self.save()
        
        print(f"{self.model_name} terminé !")
        return metrics, filepath
    
    def run_pipeline_with_data(self, df, test_size=0.2):
        """Exécuter le pipeline avec des données déjà chargées (plus efficace)."""
        print(f"Pipeline {self.model_name} (données partagées)")
        print("=" * 50)
        
        # 1. Préparer train/test (données déjà chargées et features créées)
        X_train, X_test, y_train, y_test = self.prepare_data(df, test_size)
        
        # 2. Entraîner
        self.train(X_train, y_train)
        
        # 3. Évaluer
        metrics = self.evaluate(X_test, y_test)
        
        # 4. Sauvegarder
        filepath = self.save()
        
        print(f"{self.model_name} terminé !")
        return metrics, filepath
