import numpy as np
import pandas as pd

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

def load_production_model():
    # TODO: remplacer par un vrai chargement de modèle (pickle/MLflow, etc.)
    return DummyModel()

def predict_with_model(model, df: pd.DataFrame, horizon: int, with_intervals: bool):
    return model.predict(df, horizon=horizon, intervals=with_intervals)
