"""Script pour entraîner et comparer tous les modèles."""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

# Import des modèles
from scripts.models.linear_regression import LinearRegressionModel
from scripts.models.random_forest import RandomForestModel
from scripts.models.gradient_boosting_quantile import GradientBoostingQuantileModel
from scripts.models.lightgbm_model import LightGBMModel


def run_all_models(data_source="odre"):
    """Entraîner tous les modèles et comparer les résultats."""
    print("Entraînement de tous les modèles")
    print("=" * 60)
    
    # CHARGEMENT UNIQUE DES DONNÉES
    print(f"\nChargement unique des données {data_source}...")
    from scripts.data_cache import get_cached_data
    shared_data = get_cached_data(data_source)
    print(f"Données partagées prêtes: {len(shared_data)} échantillons")
    
    models = [
        LinearRegressionModel(),
        RandomForestModel(), 
        LightGBMModel(),
        GradientBoostingQuantileModel()
    ]
    
    results = {}
    
    for model in models:
        try:
            print(f"\n{'='*20} {model.model_name.upper()} {'='*20}")
            # Utiliser les données partagées au lieu de les recharger
            metrics, filepath = model.run_pipeline_with_data(shared_data)
            results[model.model_name] = {
                'metrics': metrics,
                'filepath': filepath,
                'status': 'success'
            }
        except Exception as e:
            print(f"Erreur avec {model.model_name}: {e}")
            results[model.model_name] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Comparaison des résultats
    print(f"\n{'='*60}")
    print("COMPARAISON DES MODÈLES")
    print("=" * 60)
    
    # Préparer le tableau de comparaison
    comparison_data = []
    for model_name, result in results.items():
        if result['status'] == 'success':
            metrics = result['metrics']
            comparison_data.append({
                'Modèle': model_name,
                'MAE': f"{metrics['mae']:.2f}",
                'RMSE': f"{metrics['rmse']:.2f}",
                'R²': f"{metrics['r2']:.4f}",
                'MAPE': f"{metrics['mape']:.2f}%",
                'Status': 'OK'
            })
        else:
            comparison_data.append({
                'Modèle': model_name,
                'MAE': 'ERROR',
                'RMSE': 'ERROR', 
                'R²': 'ERROR',
                'MAPE': 'ERROR',
                'Status': 'ERROR'
            })
    
    # Afficher le tableau
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Trouver le meilleur modèle (plus petit MAE)
    successful_models = [r for r in results.values() if r['status'] == 'success']
    if successful_models:
        best_model = min(successful_models, key=lambda x: x['metrics']['mae'])
        best_name = [name for name, result in results.items() if result == best_model][0]
        
        print(f"\nMEILLEUR MODÈLE: {best_name}")
        print(f"   MAE: {best_model['metrics']['mae']:.2f}")
        print(f"   R²: {best_model['metrics']['r2']:.4f}")
        print(f"   Fichier: {best_model['filepath']}")
    
    # Sauvegarder la comparaison
    comparison_file = Path("models") / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    comparison_file.parent.mkdir(exist_ok=True)
    
    import json
    with open(comparison_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'data_source': data_source,
            'results': results,
            'comparison': comparison_data
        }, f, indent=2, default=str)
    
    print(f"\nComparaison sauvegardée: {comparison_file}")
    
    return results


def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraîner tous les modèles")
    parser.add_argument("--data", choices=["odre", "synthetic"], default="odre",
                       help="Source de données (default: odre)")
    
    args = parser.parse_args()
    
    results = run_all_models(args.data)
    print(f"\nEntraînement terminé pour {len(results)} modèles !")


if __name__ == "__main__":
    main()