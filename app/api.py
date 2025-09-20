"""API FastAPI pour la prévision énergétique."""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Ajouter le chemin src au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import create_model, BaseModel as MLBaseModel
from features import create_features
from evaluation import ModelEvaluator

# Configuration
DATA_SOURCE = os.getenv("DATA_SOURCE", "synthetic")
MODEL_DIR = Path("models")

app = FastAPI(
    title="Energy Forecasting API",
    description="API de prévision de consommation énergétique",
    version="1.0.0"
)


class ForecastRequest(BaseModel):
    horizon: int = 24
    city: str = "Paris"
    with_intervals: bool = False
    model_type: str = "linear"
    data_source: str = "synthetic"


class ForecastResponse(BaseModel):
    timestamps: List[str]
    yhat: List[float]
    yhat_lower: Optional[List[float]] = None
    yhat_upper: Optional[List[float]] = None
    model_name: str
    model_version: str


@app.get("/")
async def root():
    """Point d'entrée principal de l'API."""
    return {
        "message": "Energy Forecasting API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "forecast_url": "/forecast"
    }


@app.get("/health")
async def health():
    """Point de santé de l'API."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_source": DATA_SOURCE
    }


@app.get("/models")
async def get_available_models():
    """Obtenir la liste des modèles disponibles."""
    models = {
        "linear": {
            "name": "Linear Regression",
            "description": "Modèle de régression linéaire rapide et simple",
            "supports_intervals": False
        },
        "random_forest": {
            "name": "Random Forest",
            "description": "Modèle d'ensemble robuste avec bonnes performances",
            "supports_intervals": False
        },
        "lightgbm": {
            "name": "LightGBM",
            "description": "Gradient boosting efficace et performant",
            "supports_intervals": False
        },
        "gradient_boosting_quantile": {
            "name": "Gradient Boosting Quantile",
            "description": "Modèle avancé avec intervalles de confiance",
            "supports_intervals": True
        }
    }
    return {
        "models": models,
        "default": "linear"
    }


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """Endpoint principal pour les prévisions."""
    try:
        # Charger et préparer les données
        if DATA_SOURCE == "synthetic":
            df = generate_synthetic_data()
        else:
            df = load_odre_data()
        
        # Créer les features
        df_features = create_features(df)
        
        # Utiliser le modèle demandé ou chercher un modèle sauvegardé
        model_path = get_latest_model(model_type=request.model_type)
        if not model_path:
            # Créer et entraîner le modèle demandé
            model = create_model(request.model_type)
            
            # Entraînement rapide sur les dernières données
            feature_cols = [col for col in df_features.columns if col != 'y']
            X = df_features[feature_cols].fillna(0)
            y = df_features['y']
            
            # Utiliser seulement les 1000 derniers points pour un entraînement rapide
            if len(X) > 1000:
                X = X.tail(1000)
                y = y.tail(1000)
            
            model.fit(X, y)
        else:
            # Charger le modèle sauvegardé
            model = load_model(model_path)
        
        # Générer les prédictions
        predictions = generate_predictions(
            model, df_features, request.horizon, request.with_intervals
        )
        
        return ForecastResponse(**predictions)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prévision: {str(e)}")


def generate_synthetic_data() -> pd.DataFrame:
    """Générer des données synthétiques pour les tests."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='h'
    )
    
    # Patterns journaliers et hebdomadaires
    hourly_pattern = np.sin(2 * np.pi * np.arange(len(dates)) / 24) * 20
    weekly_pattern = np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 7)) * 10
    noise = np.random.normal(0, 5, len(dates))
    
    consumption = 100 + hourly_pattern + weekly_pattern + noise
    
    return pd.DataFrame({
        'y': consumption
    }, index=dates)


def load_odre_data() -> pd.DataFrame:
    """Charger les vraies données RTE ODRÉ depuis l'API officielle."""
    try:
        from src.odre_loader import load_odre_data_cached
        print("🏭 Chargement des vraies données RTE ODRÉ...")
        df = load_odre_data_cached(cache_hours=1, max_records=1000)
        
        if df.empty:
            print("⚠️ Aucune donnée ODRÉ disponible, basculement vers données synthétiques")
            return generate_synthetic_data()
            
        print(f"✅ Données ODRÉ chargées: {len(df)} enregistrements")
        return df
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement ODRÉ: {e}")
        print("🔄 Basculement vers données synthétiques")
        return generate_synthetic_data()


def get_latest_model(model_type: str = None) -> Optional[Path]:
    """Obtenir le chemin du modèle le plus récent."""
    if not MODEL_DIR.exists():
        return None
    
    if model_type:
        # Chercher un modèle spécifique
        pattern = f"*{model_type}*.joblib"
        model_files = list(MODEL_DIR.glob(pattern))
    else:
        # Tous les modèles
        model_files = list(MODEL_DIR.glob("*.joblib"))
    
    if not model_files:
        return None
    
    # Retourner le plus récent (basé sur la date de modification)
    latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
    return latest_model


def load_model(model_path: Path) -> MLBaseModel:
    """Charger un modèle sauvegardé."""
    import joblib
    
    try:
        model_info = joblib.load(model_path)
        
        # Créer une instance du bon type de modèle
        if "linear" in model_path.name.lower():
            model = create_model("linear")
        elif "random_forest" in model_path.name.lower():
            model = create_model("random_forest")
        elif "lightgbm" in model_path.name.lower():
            model = create_model("lightgbm")
        elif "gradient_boosting" in model_path.name.lower():
            model = create_model("gradient_boosting_quantile")
        else:
            model = create_model("linear")  # fallback
        
        model.load(model_path)
        return model
        
    except Exception as e:
        print(f"Erreur lors du chargement du modèle {model_path}: {e}")
        # Retourner un modèle simple en fallback
        return create_model("linear")


def generate_predictions(model: MLBaseModel, df_features: pd.DataFrame, 
                        horizon: int, with_intervals: bool) -> Dict[str, Any]:
    """Générer les prédictions."""
    # Utiliser les dernières données comme base pour la prédiction
    last_features = df_features.tail(1).copy()
    
    # Générer les timestamps futurs
    last_timestamp = df_features.index[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=horizon,
        freq='h'
    )
    
    predictions = []
    timestamps = []
    
    # Pour chaque point futur
    for i, timestamp in enumerate(future_timestamps):
        # Mettre à jour les features temporelles
        features = last_features.copy()
        features.index = [timestamp]
        features = create_temporal_features_for_prediction(features, timestamp)
        
        # Faire la prédiction
        feature_cols = [col for col in features.columns if col != 'y']
        X = features[feature_cols].fillna(0)
        
        try:
            if hasattr(model, 'predict_median'):
                # Modèle quantile
                pred = model.predict_median(X)[0]
            else:
                pred = model.predict(X)[0]
        except Exception:
            # Fallback: prédiction simple basée sur la dernière valeur
            pred = df_features['y'].iloc[-1] + np.random.normal(0, 5)
        
        predictions.append(float(pred))
        timestamps.append(timestamp.isoformat())
    
    result = {
        'timestamps': timestamps,
        'yhat': predictions,
        'model_name': getattr(model, 'name', 'Unknown'),
        'model_version': '1.0'
    }
    
    if with_intervals:
        # Ajouter des intervalles de confiance simples (±10%)
        yhat_lower = [p * 0.9 for p in predictions]
        yhat_upper = [p * 1.1 for p in predictions]
        result['yhat_lower'] = yhat_lower
        result['yhat_upper'] = yhat_upper
    
    return result


def create_temporal_features_for_prediction(df: pd.DataFrame, timestamp: datetime) -> pd.DataFrame:
    """Créer des features temporelles pour une prédiction."""
    df = df.copy()
    
    df['hour'] = timestamp.hour
    df['day_of_week'] = timestamp.weekday()
    df['day_of_month'] = timestamp.day
    df['month'] = timestamp.month
    df['quarter'] = (timestamp.month - 1) // 3 + 1
    
    # Features cycliques
    df['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * timestamp.weekday() / 7)
    df['dow_cos'] = np.cos(2 * np.pi * timestamp.weekday() / 7)
    df['month_sin'] = np.sin(2 * np.pi * timestamp.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * timestamp.month / 12)
    
    # Features booléennes
    df['is_weekend'] = int(timestamp.weekday() >= 5)
    df['is_business_hour'] = int(8 <= timestamp.hour <= 18)
    df['is_peak_hour'] = int(timestamp.hour in [8, 9, 10, 18, 19, 20])
    df['is_night'] = int(timestamp.hour >= 22 or timestamp.hour <= 6)
    
    return df


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
