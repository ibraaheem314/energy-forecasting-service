"""Modèle LightGBM pour prévision énergétique."""
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM non installé. Utilisation de GradientBoostingRegressor comme fallback.")
    from sklearn.ensemble import GradientBoostingRegressor as lgb_fallback

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from base_model import BaseModel


class LightGBMModel(BaseModel):
    """Modèle LightGBM (ou Gradient Boosting en fallback)."""
    
    def __init__(self):
        if LIGHTGBM_AVAILABLE:
            model_class = lgb.LGBMRegressor
            model_params = {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 8,
                'num_leaves': 31,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1
            }
        else:
            model_class = lgb_fallback
            model_params = {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 8,
                'random_state': 42
            }
        
        super().__init__(
            model_name="lightgbm" if LIGHTGBM_AVAILABLE else "gradient_boosting_fallback",
            model_class=model_class,
            model_params=model_params
        )


def main():
    """Entraîner le modèle LightGBM."""
    model = LightGBMModel()
    metrics, filepath = model.run_full_pipeline()
    return metrics, filepath


if __name__ == "__main__":
    main()
