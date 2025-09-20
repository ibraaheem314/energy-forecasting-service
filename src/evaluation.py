"""Métriques d'évaluation : RMSE, MAPE, Pinball Loss, CRPS, Coverage."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient de détermination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Pinball Loss pour les prédictions quantiles."""
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))


def coverage_probability(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Probabilité de couverture pour les intervalles de prédiction."""
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    return np.mean(covered)


def interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Largeur moyenne des intervalles de prédiction."""
    return np.mean(y_upper - y_lower)


def crps_gaussian(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> float:
    """Continuous Ranked Probability Score pour distribution gaussienne."""
    # Normalisation
    z = (y_true - y_pred) / y_std
    
    # CRPS pour distribution normale standard
    crps = y_std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    
    return np.mean(crps)


def evaluate_point_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Évaluation complète pour prédictions ponctuelles."""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def evaluate_quantile_forecast(y_true: np.ndarray, quantile_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Évaluation pour prédictions quantiles."""
    metrics = {}
    
    for quantile_name, y_pred in quantile_predictions.items():
        # Extraire la valeur du quantile depuis le nom (ex: 'q10' -> 0.1)
        if quantile_name.startswith('q'):
            quantile_value = float(quantile_name[1:]) / 100
            metrics[f'pinball_loss_{quantile_name}'] = pinball_loss(y_true, y_pred, quantile_value)
    
    return metrics


def evaluate_interval_forecast(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, 
                             confidence_level: float = 0.8) -> Dict[str, float]:
    """Évaluation pour intervalles de prédiction."""
    return {
        'coverage_probability': coverage_probability(y_true, y_lower, y_upper),
        'average_interval_width': interval_width(y_lower, y_upper),
        'expected_coverage': confidence_level,
        'coverage_deviation': abs(coverage_probability(y_true, y_lower, y_upper) - confidence_level)
    }


def evaluate_fairness_by_groups(y_true: np.ndarray, y_pred: np.ndarray, 
                               groups: np.ndarray, metric_func=mean_absolute_error) -> Dict[str, float]:
    """Évaluation de la fairness par sous-groupes."""
    unique_groups = np.unique(groups)
    fairness_metrics = {}
    
    for group in unique_groups:
        mask = groups == group
        if np.sum(mask) > 0:
            group_metric = metric_func(y_true[mask], y_pred[mask])
            fairness_metrics[f'metric_group_{group}'] = group_metric
    
    # Calcul de la variance inter-groupes
    group_values = list(fairness_metrics.values())
    if len(group_values) > 1:
        fairness_metrics['inter_group_variance'] = np.var(group_values)
        fairness_metrics['max_group_difference'] = np.max(group_values) - np.min(group_values)
    
    return fairness_metrics


class ModelEvaluator:
    """Classe pour évaluation complète des modèles."""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model_name: str, y_true: np.ndarray, y_pred: Union[np.ndarray, Dict], 
                      groups: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Évaluation complète d'un modèle."""
        metrics = {}
        
        # Prédictions ponctuelles
        if isinstance(y_pred, np.ndarray):
            metrics.update(evaluate_point_forecast(y_true, y_pred))
            
            # Fairness si groupes fournis
            if groups is not None:
                fairness = evaluate_fairness_by_groups(y_true, y_pred, groups)
                metrics.update(fairness)
                
        # Prédictions quantiles
        elif isinstance(y_pred, dict):
            metrics.update(evaluate_quantile_forecast(y_true, y_pred))
            
            # Si quantiles 10% et 90% disponibles, évaluer l'intervalle
            if 'q10' in y_pred and 'q90' in y_pred:
                interval_metrics = evaluate_interval_forecast(
                    y_true, y_pred['q10'], y_pred['q90'], 0.8
                )
                metrics.update(interval_metrics)
        
        self.results[model_name] = metrics
        return metrics
        
    def compare_models(self) -> pd.DataFrame:
        """Comparer les performances des modèles."""
        if not self.results:
            return pd.DataFrame()
            
        return pd.DataFrame(self.results).T
        
    def get_best_model(self, metric: str = 'rmse', ascending: bool = True) -> str:
        """Obtenir le meilleur modèle selon une métrique."""
        if not self.results:
            return None
            
        df = self.compare_models()
        if metric not in df.columns:
            raise ValueError(f"Métrique '{metric}' non disponible")
            
        best_idx = df[metric].idxmin() if ascending else df[metric].idxmax()
        return best_idx


# Import pour CRPS (optionnel)
try:
    from scipy.stats import norm
except ImportError:
    def crps_gaussian(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> float:
        """Version simplifiée du CRPS sans scipy."""
        return np.mean(np.abs(y_true - y_pred))  # Fallback vers MAE
