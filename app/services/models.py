import numpy as np
import pandas as pd
import joblib
from pathlib import Path

class DummyModel:
    name = "baseline-weekly"
    version = "0.1"

    def predict(self, df: pd.DataFrame, horizon: int = 168, intervals: bool = True):
        # baseline: répète la valeur d'il y a 168h (si dispo), sinon dernier point
        if df.empty or "y" not in df:
            yhat = np.zeros(horizon)
        else:
            last = df["y"].iloc[-1]
            yhat = np.full(horizon, last)
            if len(df) >= 168:
                weekly_ref = df["y"].iloc[-168:]
                ref = weekly_ref.values[-1]
                yhat[:] = ref
        out = {"yhat": yhat.tolist()}
        if intervals:
            out["yhat_lower"] = (yhat * 0.95).tolist()
            out["yhat_upper"] = (yhat * 1.05).tolist()
        return out

class SklearnModelWrapper:
    """Wrapper pour les modèles sklearn avec interface compatible"""
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.name = f"sklearn-{type(model).__name__}"
        self.version = "1.0"

    def predict(self, df: pd.DataFrame, horizon: int = 168, intervals: bool = True):
        if df.empty or "y" not in df:
            return DummyModel().predict(df, horizon, intervals)

        # Pour la prédiction, nous avons besoin des features que le modèle a été entraîné
        # Ici, nous faisons une prédiction simple en utilisant la dernière valeur moyenne
        # Pour une vraie prédiction, il faudrait générer les futures features
        # et utiliser self.model.predict(X_future)
        
        # Pour l'instant, on utilise une approche simple pour la démo
        # On prend la moyenne des dernières valeurs disponibles pour la prédiction
        # et on ajoute des intervalles basés sur l'écart-type
        last_values = df["y"].iloc[-min(horizon, len(df)):] if not df.empty else pd.Series([0.0])
        
        if last_values.empty:
            yhat = np.zeros(horizon)
        else:
            yhat = np.full(horizon, last_values.mean())

        out = {"yhat": yhat.tolist()}
        if intervals:
            std = last_values.std() if len(last_values) > 1 else yhat.mean() * 0.1
            out["yhat_lower"] = (yhat - 1.96 * std).tolist()
            out["yhat_upper"] = (yhat + 1.96 * std).tolist()
        return out

def load_production_model():
    """Charge le meilleur modèle disponible depuis le dossier models/"""
    models_dir = Path("models")
    if not models_dir.exists():
        return DummyModel()

    # Chercher le modèle le plus récent
    model_files = list(models_dir.glob("*.joblib"))
    if not model_files:
        return DummyModel()

    # Prendre le modèle le plus récent (par timestamp dans le nom)
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

    try:
        model_data = joblib.load(latest_model)
        # Encapsuler le modèle sklearn dans notre interface
        return SklearnModelWrapper(model_data['model'], model_data.get('feature_names', []))
    except Exception:
        return DummyModel()

def predict_with_model(model, df: pd.DataFrame, horizon: int, with_intervals: bool):
    return model.predict(df, horizon=horizon, intervals=with_intervals)