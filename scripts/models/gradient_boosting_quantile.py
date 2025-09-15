"""Modèle Gradient Boosting avec quantiles pour prévision énergétique."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from base_model import BaseModel


class GradientBoostingQuantileModel(BaseModel):
    """Modèle Gradient Boosting avec prédictions par quantiles."""
    
    def __init__(self):
        super().__init__(
            model_name="gradient_boosting_quantile",
            model_class=GradientBoostingRegressor,
            model_params={
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'loss': 'quantile',  # Utilise la perte quantile
                'alpha': 0.5  # Médiane par défaut
            }
        )
        self.quantile_models = {}
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]  # Quantiles à prédire
    
    def train(self, X_train, y_train):
        """Entraîner plusieurs modèles pour différents quantiles."""
        print(f"🤖 Entraînement {self.model_name} avec quantiles {self.quantiles}...")
        
        for quantile in self.quantiles:
            print(f"   📊 Entraînement quantile {quantile}...")
            
            # Créer un modèle pour ce quantile
            model_params = self.model_params.copy()
            model_params['alpha'] = quantile
            
            model = self.model_class(**model_params)
            model.fit(X_train, y_train)
            
            self.quantile_models[quantile] = model
        
        # Le modèle principal est celui de la médiane (0.5)
        self.model = self.quantile_models[0.5]
        
        print(f"✅ {self.model_name} entraîné pour {len(self.quantiles)} quantiles")
    
    def predict_quantiles(self, X, quantiles=None):
        """Prédire avec plusieurs quantiles."""
        if quantiles is None:
            quantiles = self.quantiles
        
        predictions = {}
        for quantile in quantiles:
            if quantile in self.quantile_models:
                predictions[f'q{int(quantile*100)}'] = self.quantile_models[quantile].predict(X)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Évaluer le modèle avec métriques de quantiles."""
        # Évaluation standard avec médiane
        y_pred = self.model.predict(X_test)
        
        # Métriques standard
        self.metrics = {
            'mae': np.mean(np.abs(y_test - y_pred)),
            'mse': np.mean((y_test - y_pred) ** 2),
            'rmse': np.sqrt(np.mean((y_test - y_pred) ** 2)),
            'r2': 1 - np.var(y_test - y_pred) / np.var(y_test),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        # Prédictions par quantiles
        quantile_preds = self.predict_quantiles(X_test)
        
        # Métriques spécifiques aux quantiles
        q10 = quantile_preds.get('q10', y_pred)
        q90 = quantile_preds.get('q90', y_pred)
        
        # Coverage (pourcentage de vraies valeurs dans l'intervalle 80%)
        coverage_80 = np.mean((y_test >= q10) & (y_test <= q90)) * 100
        
        # Width de l'intervalle de prédiction moyen
        interval_width = np.mean(q90 - q10)
        
        self.metrics.update({
            'coverage_80': coverage_80,
            'interval_width': interval_width,
            'quantiles_available': list(quantile_preds.keys())
        })
        
        print(f"📊 Métriques {self.model_name}:")
        print(f"   - MAE: {self.metrics['mae']:.2f}")
        print(f"   - RMSE: {self.metrics['rmse']:.2f}")
        print(f"   - R²: {self.metrics['r2']:.4f}")
        print(f"   - MAPE: {self.metrics['mape']:.2f}%")
        print(f"   - Coverage 80%: {coverage_80:.1f}%")
        print(f"   - Largeur intervalle: {interval_width:.0f} MW")
        print(f"   - Quantiles: {list(quantile_preds.keys())}")
        
        return self.metrics
    
    def save(self, models_dir="models"):
        """Sauvegarder tous les modèles quantiles."""
        from pathlib import Path
        import joblib
        from datetime import datetime
        
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.model_name}_{timestamp}.joblib"
        filepath = models_path / filename
        
        model_data = {
            'quantile_models': self.quantile_models,
            'model': self.model,  # Modèle médiane
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'model_name': self.model_name,
            'trained_at': datetime.now().isoformat(),
            'model_params': self.model_params,
            'quantiles': self.quantiles
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Sauvegardé: {filepath} (tous quantiles)")
        
        return filepath


def main():
    """Entraîner le modèle Gradient Boosting Quantile."""
    model = GradientBoostingQuantileModel()
    metrics, filepath = model.run_full_pipeline()
    return metrics, filepath


if __name__ == "__main__":
    main()
