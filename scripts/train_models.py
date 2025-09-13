"""Script d'entraînement des modèles de prévision énergétique."""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from app.services.loader import DataLoader
from app.services.features import FeatureService
from app.services.models import ModelService

async def main():
    """Fonction principale d'entraînement des modèles."""
    print("🚀 Démarrage de l'entraînement des modèles")
    
    # Initialiser les services
    data_loader = DataLoader()
    feature_service = FeatureService()
    model_service = ModelService()
    
    # Définir la période d'entraînement
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 mois de données
    location = "region_1"
    
    print(f"📅 Période d'entraînement: {start_date.date()} à {end_date.date()}")
    print(f"📍 Location: {location}")
    
    try:
        # 1. Charger et préparer les données
        print("\n📊 Préparation des features...")
        features_df = await feature_service.prepare_features(
            start_time=start_date,
            end_time=end_date,
            location=location,
            include_weather=True,
            include_external=False  # Simplifié
        )
        
        print(f"✅ Features préparées: {features_df.shape[0]} échantillons, {features_df.shape[1]} features")
        
        # 2. Préparer les données d'entraînement
        X, y = model_service._prepare_training_data(features_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        print(f"📈 Données d'entraînement: {len(X_train)} échantillons")
        print(f"📊 Données de test: {len(X_test)} échantillons")
        
        # 3. Entraîner différents modèles
        model_types = ['linear_regression', 'random_forest', 'gradient_boosting']
        results = {}
        
        for model_type in model_types:
            print(f"\n🤖 Entraînement du modèle: {model_type}")
            
            # Créer et entraîner le modèle
            model = model_service._create_model(model_type, None)
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
            
            # Stocker le modèle
            model_name = f"{model_type}_{location}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_service.models[model_name] = model
            model_service.model_metadata[model_name] = {
                'type': model_type,
                'location': location,
                'trained_at': datetime.now(),
                'metrics': metrics,
                'feature_names': X.columns.tolist()
            }
            
            results[model_type] = {
                'model_name': model_name,
                'metrics': metrics
            }
            
            # Sauvegarder le modèle
            model_path = Path("models") / f"{model_name}.joblib"
            model_path.parent.mkdir(exist_ok=True)
            await model_service.save_model(model_name, str(model_path))
            
            print(f"✅ {model_type}:")
            print(f"   - MAE: {metrics['mae']:.2f}")
            print(f"   - RMSE: {metrics['rmse']:.2f}")
            print(f"   - R²: {metrics['r2']:.4f}")
            print(f"   - MAPE: {metrics['mape']:.2f}%")
            print(f"   - Sauvegardé: {model_path}")
        
        # 4. Résumé des résultats
        print("\n🏆 Résumé des performances:")
        print("-" * 50)
        best_model = min(results.items(), key=lambda x: x[1]['metrics']['mae'])
        print(f"🥇 Meilleur modèle (MAE): {best_model[0]}")
        print(f"   MAE: {best_model[1]['metrics']['mae']:.2f}")
        print(f"   R²: {best_model[1]['metrics']['r2']:.4f}")
        
        # 5. Sauvegarder un résumé
        summary = {
            'training_date': datetime.now().isoformat(),
            'training_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'location': location,
            'data_shape': {
                'total_samples': len(X),
                'features': len(X.columns),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            },
            'models': results,
            'best_model': {
                'name': best_model[0],
                'metrics': best_model[1]['metrics']
            }
        }
        
        import json
        with open("models/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📄 Résumé sauvegardé: models/training_summary.json")
        print("🎉 Entraînement terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
