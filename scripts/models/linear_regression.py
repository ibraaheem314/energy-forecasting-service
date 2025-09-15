"""Modèle Linear Regression pour prévision énergétique."""
from sklearn.linear_model import LinearRegression
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from base_model import BaseModel


class LinearRegressionModel(BaseModel):
    """Modèle Linear Regression."""
    
    def __init__(self):
        super().__init__(
            model_name="linear_regression",
            model_class=LinearRegression,
            model_params={}
        )


def main():
    """Entraîner le modèle Linear Regression."""
    model = LinearRegressionModel()
    metrics, filepath = model.run_full_pipeline()
    return metrics, filepath


if __name__ == "__main__":
    main()
