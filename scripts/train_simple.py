"""Script d'entraînement simple pour les modèles de prévision énergétique."""
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))
from app.services.loader import load_timeseries

def create_features(df):
    """Créer des features simples pour l'entraînement."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Features lag
    for lag in [1, 24, 168]:  # 1h, 1 jour, 1 semaine
        df[f'y_lag_{lag}'] = df['y'].shift(lag)
    
    # Rolling means
    df['y_rolling_24'] = df['y'].rolling(24, min_periods=1).mean()
    df['y_rolling_168'] = df['y'].rolling(168, min_periods=1).mean()
    
    return df.dropna()

def prepare_train_data(df):
    """Préparer les données pour l'entraînement."""
    feature_cols = [col for col in df.columns if col != 'y']
    X = df[feature_cols]
    y = df['y']
    return X, y

def train_models(df):
    """Entraîner différents modèles."""
    # Créer les features
    df_features = create_features(df)
    
    # Préparer les données
    X, y = prepare_train_data(df_features)
    
    # Split train/test (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"📊 Données d'entraînement: {len(X_train)} échantillons")
    print(f"📊 Données de test: {len(X_test)} échantillons")
    
    # Modèles à entraîner
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    for name, model in models.items():
        print(f"\n🤖 Entraînement du modèle: {name}")
        
        # Entraîner
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Métriques
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        results[name] = metrics
        
        # Sauvegarder le modèle
        model_path = models_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump({
            'model': model,
            'feature_names': X.columns.tolist(),
            'metrics': metrics,
            'trained_at': datetime.now().isoformat()
        }, model_path)
        
        print(f"✅ {name}:")
        print(f"   - MAE: {metrics['mae']:.2f}")
        print(f"   - RMSE: {metrics['rmse']:.2f}")
        print(f"   - R²: {metrics['r2']:.4f}")
        print(f"   - MAPE: {metrics['mape']:.2f}%")
        print(f"   - Sauvegardé: {model_path}")
    
    # Meilleur modèle
    best_model = min(results.items(), key=lambda x: x[1]['mae'])
    print(f"\n🏆 Meilleur modèle (MAE): {best_model[0]}")
    print(f"   MAE: {best_model[1]['mae']:.2f}")
    
    return results

def main():
    """Fonction principale."""
    print("🚀 Démarrage de l'entraînement des modèles simples")
    
    try:
        # Charger les données
        df = load_timeseries(os.getenv("CITY", "Paris"))
        print(f"📊 Données chargées: {len(df)} échantillons")
        
        # Entraîner les modèles
        results = train_models(df)
        
        # Sauvegarder un résumé
        import json
        summary = {
            'training_date': datetime.now().isoformat(),
            'data_shape': len(df),
            'models': results
        }
        
        with open("models/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📄 Résumé sauvegardé: models/training_summary.json")
        print("🎉 Entraînement terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        raise

if __name__ == "__main__":
    main()
