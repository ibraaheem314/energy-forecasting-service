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
    """Créer des features simples pour l'entraînement - Version ODRÉ simplifiée."""
    # Normaliser le nom de la colonne cible
    target_candidates = ["y", "consommation", "consommation_mw", "consommation__mw_"]
    target_col = None
    for candidate in target_candidates:
        if candidate in df.columns:
            target_col = candidate
            break
    
    if target_col is None:
        raise ValueError("Aucune colonne cible trouvée")
    
    # Créer un nouveau DataFrame avec toutes les données
    df_clean = df.copy()
    df_clean = df_clean.rename(columns={target_col: "y"})
    
    print(f"Données totales: {len(df_clean)} échantillons")
    print(f"NaN dans cible: {df_clean['y'].isnull().sum()}")
    
    # Features temporelles basées sur l'index
    df_clean['hour'] = df_clean.index.hour
    df_clean['day_of_week'] = df_clean.index.dayofweek
    df_clean['month'] = df_clean.index.month
    df_clean['is_weekend'] = (df_clean.index.dayofweek >= 5).astype(int)
    
    # Garder seulement quelques colonnes d'énergie principales qui ont le moins de NaN
    energy_cols = ['y', 'nucleaire_mw', 'eolien_mw', 'gaz_mw', 'hydraulique_mw']
    keep_cols = ['hour', 'day_of_week', 'month', 'is_weekend']
    
    # Ajouter les colonnes d'énergie qui existent
    for col in energy_cols:
        if col in df_clean.columns:
            keep_cols.append(col)
    
    df_final = df_clean[keep_cols].copy()
    
    # VRAIE imputation par médiane pour TOUTES les colonnes avec NaN
    print("Imputation par médiane en cours...")
    for col in df_final.columns:
        if df_final[col].dtype in ['float64', 'int64'] and df_final[col].isnull().any():
            median_val = df_final[col].median()
            if not pd.isna(median_val):
                nan_count_before = df_final[col].isnull().sum()
                df_final[col] = df_final[col].fillna(median_val)
                nan_count_after = df_final[col].isnull().sum()
                print(f"{col}: {nan_count_before} → {nan_count_after} NaN (médiane={median_val:.1f})")
            else:
                # Si médiane est NaN, utiliser 0 comme fallback
                nan_count = df_final[col].isnull().sum()
                df_final[col] = df_final[col].fillna(0)
                print(f"{col}: {nan_count} → 0 NaN (fallback=0)")
    
    print(f"Imputation terminée. Shape finale: {df_final.shape}")
    print(f"Total NaN restants: {df_final.isnull().sum().sum()}")
    
    return df_final

def prepare_train_data(df):
    """Préparer les données pour l'entraînement."""
    print(f"Données préparées: {len(df)} échantillons")
    
    # Nettoyage final : supprimer toute ligne avec NaN
    df_clean = df.dropna()
    print(f"Après nettoyage final: {len(df_clean)} échantillons")
    
    feature_cols = [col for col in df_clean.columns if col != 'y']
    X = df_clean[feature_cols]
    y = df_clean['y']
    
    # Vérification finale
    if X.isnull().any().any():
        print("Warning: NaN détectés dans X")
    if y.isnull().any():
        print("Warning: NaN détectés dans y")
    
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
    
    print(f"Données d'entraînement: {len(X_train)} échantillons")
    print(f"Données de test: {len(X_test)} échantillons")
    
    # Modèles à entraîner
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    for name, model in models.items():
        print(f"\nEntraînement du modèle: {name}")
        
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
        
        print(f"{name}:")
        print(f"   - MAE: {metrics['mae']:.2f}")
        print(f"   - RMSE: {metrics['rmse']:.2f}")
        print(f"   - R²: {metrics['r2']:.4f}")
        print(f"   - MAPE: {metrics['mape']:.2f}%")
        print(f"   - Sauvegardé: {model_path}")
    
    # Meilleur modèle
    best_model = min(results.items(), key=lambda x: x[1]['mae'])
    print(f"\nMeilleur modèle (MAE): {best_model[0]}")
    print(f"   MAE: {best_model[1]['mae']:.2f}")
    
    return results

def main():
    """Fonction principale."""
    print("Démarrage de l'entraînement des modèles simples")
    
    try:
        # Charger les données
        df = load_timeseries(os.getenv("CITY", "Paris"))
        print(f"Données chargées: {len(df)} échantillons")
        
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
        
        print(f"\nRésumé sauvegardé: models/training_summary.json")
        print("Entraînement terminé avec succès!")
        
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {e}")
        raise

if __name__ == "__main__":
    main()