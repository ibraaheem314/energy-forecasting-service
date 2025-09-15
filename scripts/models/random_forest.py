"""Modèle Random Forest pour prévision énergétique."""
from sklearn.ensemble import RandomForestRegressor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from base_model import BaseModel


class RandomForestModel(BaseModel):
    """Modèle Random Forest."""
    
    def __init__(self):
        super().__init__(
            model_name="random_forest",
            model_class=RandomForestRegressor,
            model_params={
                'n_estimators': 100,
                'random_state': 42,
                'n_jobs': -1,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        )


def main():
    """Entraîner le modèle Random Forest."""
    model = RandomForestModel()
    metrics, filepath = model.run_full_pipeline()
    return metrics, filepath


if __name__ == "__main__":
    main()
