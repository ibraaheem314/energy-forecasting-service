"""Script d'évaluation simple des modèles de prévision énergétique."""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))
from app.services.loader import load_timeseries

def create_features(df):
    """Créer des features simples (même fonction que train_simple.py)."""
    df = df.copy()
    
    # Normaliser le nom de la colonne cible
    if "consommation" in df.columns:
        df = df.rename(columns={"consommation": "y"})
    elif len(df.columns) > 0 and "y" not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df = df.rename(columns={numeric_cols[0]: "y"})
    
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

def evaluate_model(model_path, df_test):
    """Évaluer un modèle sur des données de test."""
    # Charger le modèle
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Préparer les features de test
    df_features = create_features(df_test)
    
    # S'assurer qu'on a toutes les features nécessaires
    X_test = df_features[feature_names]
    y_test = df_features['y']
    
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
    
    return metrics, y_test, y_pred, df_features.index

def plot_predictions(y_test, y_pred, timestamps, model_name):
    """Créer un graphique des prédictions."""
    plt.figure(figsize=(12, 6))
    
    # Limiter à 7 jours pour la lisibilité
    n_show = min(168, len(y_test))  # 7 jours maximum
    
    plt.plot(timestamps[-n_show:], y_test.iloc[-n_show:], 
             label='Réel', linewidth=2, alpha=0.8)
    plt.plot(timestamps[-n_show:], y_pred[-n_show:], 
             label='Prédit', linewidth=2, alpha=0.8)
    
    plt.title(f'Prédictions vs Réel - {model_name}')
    plt.xlabel('Temps')
    plt.ylabel('Consommation (MW)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Sauvegarder
    output_path = f"models/predictions_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    """Fonction principale d'évaluation."""
    print("🔍 Démarrage de l'évaluation des modèles")
    
    try:
        # Vérifier les modèles disponibles
        models_dir = Path("models")
        if not models_dir.exists():
            print("❌ Dossier 'models' introuvable. Exécutez d'abord train_simple.py")
            return
        
        model_files = list(models_dir.glob("*.joblib"))
        if not model_files:
            print("❌ Aucun modèle trouvé. Exécutez d'abord train_simple.py")
            return
        
        print(f"📊 {len(model_files)} modèles trouvés")
        
        # Charger des données récentes pour l'évaluation
        df = load_timeseries(os.getenv("CITY", "Paris"))
        
        # Utiliser les 20% les plus récentes comme données de test
        test_size = int(len(df) * 0.2)
        df_test = df.iloc[-test_size:]
        
        print(f"📊 Données d'évaluation: {len(df_test)} échantillons")
        
        evaluation_results = {}
        
        for model_file in model_files:
            model_name = model_file.stem.split('_')[0]  # Extraire le nom du modèle
            print(f"\n🤖 Évaluation du modèle: {model_name}")
            
            try:
                # Évaluer le modèle
                metrics, y_test, y_pred, timestamps = evaluate_model(model_file, df_test)
                
                evaluation_results[model_name] = {
                    'file': str(model_file),
                    'metrics': metrics
                }
                
                print(f"✅ Métriques:")
                print(f"   - MAE: {metrics['mae']:.2f}")
                print(f"   - RMSE: {metrics['rmse']:.2f}")
                print(f"   - R²: {metrics['r2']:.4f}")
                print(f"   - MAPE: {metrics['mape']:.2f}%")
                
                # Créer le graphique
                plot_path = plot_predictions(y_test, y_pred, timestamps, model_name)
                print(f"📊 Graphique sauvegardé: {plot_path}")
                
            except Exception as e:
                print(f"❌ Erreur avec le modèle {model_name}: {e}")
                continue
        
        if not evaluation_results:
            print("❌ Aucun modèle n'a pu être évalué")
            return
        
        # Comparaison des modèles
        print("\n📊 Comparaison des modèles:")
        print("-" * 60)
        print(f"{'Modèle':<15} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10}")
        print("-" * 60)
        
        for model_name, result in evaluation_results.items():
            metrics = result['metrics']
            print(f"{model_name:<15} {metrics['mae']:<10.2f} "
                  f"{metrics['rmse']:<10.2f} {metrics['r2']:<10.4f} {metrics['mape']:<10.2f}")
        
        # Meilleur modèle
        best_model = min(evaluation_results.items(), key=lambda x: x[1]['metrics']['mae'])
        print(f"\n🏆 Meilleur modèle: {best_model[0]} (MAE: {best_model[1]['metrics']['mae']:.2f})")
        
        # Sauvegarder les résultats
        import json
        evaluation_summary = {
            'evaluation_date': datetime.now().isoformat(),
            'test_samples': len(df_test),
            'models_evaluated': len(evaluation_results),
            'results': evaluation_results,
            'best_model': {
                'name': best_model[0],
                'metrics': best_model[1]['metrics']
            }
        }
        
        with open("models/evaluation_summary.json", "w") as f:
            json.dump(evaluation_summary, f, indent=2, default=str)
        
        print(f"📄 Résultats sauvegardés: models/evaluation_summary.json")
        print("🎉 Évaluation terminée avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'évaluation: {e}")
        raise

if __name__ == "__main__":
    main()
