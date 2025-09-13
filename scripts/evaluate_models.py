"""Script d'évaluation des modèles de prévision énergétique."""
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from app.services.loader import DataLoader
from app.services.features import FeatureService
from app.services.models import ModelService

async def main():
    """Fonction principale d'évaluation des modèles."""
    print("🔍 Démarrage de l'évaluation des modèles")
    
    # Vérifier si des modèles existent
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Aucun dossier 'models' trouvé. Exécutez d'abord train_models.py")
        return
    
    model_files = list(models_dir.glob("*.joblib"))
    if not model_files:
        print("❌ Aucun modèle trouvé. Exécutez d'abord train_models.py")
        return
    
    print(f"📊 {len(model_files)} modèles trouvés")
    
    # Initialiser les services
    data_loader = DataLoader()
    feature_service = FeatureService()
    model_service = ModelService()
    
    # Période d'évaluation (différente de l'entraînement)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)  # 2 semaines pour évaluation
    location = "region_1"
    
    print(f"📅 Période d'évaluation: {start_date.date()} à {end_date.date()}")
    
    try:
        # Préparer les données d'évaluation
        print("\n📊 Préparation des données d'évaluation...")
        features_df = await feature_service.prepare_features(
            start_time=start_date,
            end_time=end_date,
            location=location,
            include_weather=True,
            include_external=False
        )
        
        X, y = model_service._prepare_training_data(features_df)
        print(f"✅ Données d'évaluation: {len(X)} échantillons")
        
        # Évaluer chaque modèle
        evaluation_results = {}
        predictions = {}
        
        for model_file in model_files:
            model_name = model_file.stem
            print(f"\n🤖 Évaluation du modèle: {model_name}")
            
            try:
                # Charger le modèle
                model_data = joblib.load(model_file)
                if isinstance(model_data, dict):
                    model = model_data['model']
                    metadata = model_data.get('metadata', {})
                else:
                    model = model_data
                    metadata = {}
                
                # Vérifier la compatibilité des features
                if 'feature_names' in metadata:
                    expected_features = metadata['feature_names']
                    missing_features = set(expected_features) - set(X.columns)
                    if missing_features:
                        print(f"⚠️  Features manquantes: {missing_features}")
                        # Ajouter les features manquantes avec des valeurs par défaut
                        for feature in missing_features:
                            X[feature] = 0
                    
                    # Réorganiser les colonnes dans le bon ordre
                    X_ordered = X.reindex(columns=expected_features, fill_value=0)
                else:
                    X_ordered = X
                
                # Prédictions
                y_pred = model.predict(X_ordered)
                
                # Métriques
                metrics = {
                    'mae': mean_absolute_error(y, y_pred),
                    'mse': mean_squared_error(y, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                    'r2': r2_score(y, y_pred),
                    'mape': np.mean(np.abs((y - y_pred) / y)) * 100
                }
                
                evaluation_results[model_name] = {
                    'metrics': metrics,
                    'model_type': metadata.get('type', 'unknown'),
                    'trained_at': metadata.get('trained_at', 'unknown')
                }
                
                predictions[model_name] = {
                    'actual': y.values,
                    'predicted': y_pred,
                    'timestamps': features_df['timestamp'].values
                }
                
                print(f"✅ Métriques:")
                print(f"   - MAE: {metrics['mae']:.2f}")
                print(f"   - RMSE: {metrics['rmse']:.2f}")
                print(f"   - R²: {metrics['r2']:.4f}")
                print(f"   - MAPE: {metrics['mape']:.2f}%")
                
            except Exception as e:
                print(f"❌ Erreur avec le modèle {model_name}: {e}")
                continue
        
        if not evaluation_results:
            print("❌ Aucun modèle n'a pu être évalué")
            return
        
        # Comparaison des modèles
        print("\n📊 Comparaison des modèles:")
        print("-" * 80)
        print(f"{'Modèle':<30} {'Type':<15} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10}")
        print("-" * 80)
        
        for model_name, result in evaluation_results.items():
            metrics = result['metrics']
            model_type = result['model_type']
            print(f"{model_name:<30} {model_type:<15} {metrics['mae']:<10.2f} "
                  f"{metrics['rmse']:<10.2f} {metrics['r2']:<10.4f} {metrics['mape']:<10.2f}")
        
        # Meilleur modèle
        best_model = min(evaluation_results.items(), key=lambda x: x[1]['metrics']['mae'])
        print(f"\n🏆 Meilleur modèle: {best_model[0]} (MAE: {best_model[1]['metrics']['mae']:.2f})")
        
        # Générer des graphiques
        print("\n📈 Génération des graphiques...")
        create_evaluation_plots(predictions, evaluation_results)
        
        # Sauvegarder les résultats
        evaluation_summary = {
            'evaluation_date': datetime.now().isoformat(),
            'evaluation_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'location': location,
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

def create_evaluation_plots(predictions, evaluation_results):
    """Créer des graphiques d'évaluation."""
    if not predictions:
        return
    
    # Configuration matplotlib
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Évaluation des Modèles de Prévision Énergétique', fontsize=16)
    
    # 1. Comparaison des métriques
    ax1 = axes[0, 0]
    model_names = list(evaluation_results.keys())
    mae_values = [evaluation_results[name]['metrics']['mae'] for name in model_names]
    r2_values = [evaluation_results[name]['metrics']['r2'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1_twin = ax1.twinx()
    bars1 = ax1.bar(x - width/2, mae_values, width, label='MAE', alpha=0.7, color='red')
    bars2 = ax1_twin.bar(x + width/2, r2_values, width, label='R²', alpha=0.7, color='blue')
    
    ax1.set_xlabel('Modèles')
    ax1.set_ylabel('MAE', color='red')
    ax1_twin.set_ylabel('R²', color='blue')
    ax1.set_title('Comparaison des Métriques')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.split('_')[0] for name in model_names], rotation=45)
    
    # 2. Prédictions vs Réel (meilleur modèle)
    ax2 = axes[0, 1]
    best_model_name = min(evaluation_results.items(), key=lambda x: x[1]['metrics']['mae'])[0]
    best_pred = predictions[best_model_name]
    
    ax2.scatter(best_pred['actual'], best_pred['predicted'], alpha=0.6)
    ax2.plot([best_pred['actual'].min(), best_pred['actual'].max()], 
             [best_pred['actual'].min(), best_pred['actual'].max()], 'r--', lw=2)
    ax2.set_xlabel('Valeurs Réelles')
    ax2.set_ylabel('Prédictions')
    ax2.set_title(f'Prédictions vs Réel - {best_model_name.split("_")[0]}')
    
    # 3. Série temporelle (meilleur modèle)
    ax3 = axes[1, 0]
    timestamps = pd.to_datetime(best_pred['timestamps'])
    ax3.plot(timestamps, best_pred['actual'], label='Réel', linewidth=2)
    ax3.plot(timestamps, best_pred['predicted'], label='Prédit', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Temps')
    ax3.set_ylabel('Consommation Énergétique')
    ax3.set_title('Série Temporelle - Réel vs Prédit')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Distribution des erreurs (meilleur modèle)
    ax4 = axes[1, 1]
    errors = best_pred['actual'] - best_pred['predicted']
    ax4.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Erreurs (Réel - Prédit)')
    ax4.set_ylabel('Fréquence')
    ax4.set_title('Distribution des Erreurs')
    
    plt.tight_layout()
    plt.savefig('models/evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Graphiques sauvegardés: models/evaluation_plots.png")

if __name__ == "__main__":
    asyncio.run(main())
