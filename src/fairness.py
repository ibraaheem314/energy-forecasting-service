"""Métriques de fairness : couverture par sous-groupes."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from .evaluation import mean_absolute_error, root_mean_squared_error, coverage_probability


def create_fairness_groups(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Créer des groupes pour l'analyse de fairness."""
    groups = {}
    
    # Groupes temporels
    if hasattr(df.index, 'hour'):
        groups['time_of_day'] = np.where(
            (df.index.hour >= 8) & (df.index.hour <= 18), 
            'business_hours', 
            'off_hours'
        )
        
        groups['peak_hours'] = np.where(
            df.index.hour.isin([8, 9, 10, 18, 19, 20]),
            'peak',
            'off_peak'
        )
    
    # Groupes saisonniers
    if hasattr(df.index, 'month'):
        groups['season'] = np.where(
            df.index.month.isin([12, 1, 2]), 'winter',
            np.where(df.index.month.isin([3, 4, 5]), 'spring',
            np.where(df.index.month.isin([6, 7, 8]), 'summer', 'autumn'))
        )
        
        groups['is_winter'] = np.where(
            df.index.month.isin([12, 1, 2]), 
            'winter', 
            'non_winter'
        )
    
    # Groupes de weekend
    if hasattr(df.index, 'dayofweek'):
        groups['weekend'] = np.where(
            df.index.dayofweek >= 5, 
            'weekend', 
            'weekday'
        )
    
    return groups


def calculate_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          groups: np.ndarray, group_name: str) -> Dict[str, float]:
    """Calculer les métriques pour chaque groupe."""
    unique_groups = np.unique(groups)
    metrics = {}
    
    for group in unique_groups:
        mask = groups == group
        n_samples = np.sum(mask)
        
        if n_samples > 0:
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            metrics[f'{group_name}_{group}_mae'] = mean_absolute_error(group_y_true, group_y_pred)
            metrics[f'{group_name}_{group}_rmse'] = root_mean_squared_error(group_y_true, group_y_pred)
            metrics[f'{group_name}_{group}_count'] = n_samples
    
    return metrics


def calculate_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             groups_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculer toutes les métriques de fairness."""
    fairness_metrics = {}
    
    for group_name, groups in groups_dict.items():
        # Métriques par groupe
        group_metrics = calculate_group_metrics(y_true, y_pred, groups, group_name)
        fairness_metrics.update(group_metrics)
        
        # Analyse de l'équité
        unique_groups = np.unique(groups)
        group_maes = []
        group_rmses = []
        
        for group in unique_groups:
            mask = groups == group
            if np.sum(mask) > 0:
                group_y_true = y_true[mask]
                group_y_pred = y_pred[mask]
                group_maes.append(mean_absolute_error(group_y_true, group_y_pred))
                group_rmses.append(root_mean_squared_error(group_y_true, group_y_pred))
        
        if len(group_maes) > 1:
            # Variance inter-groupes
            fairness_metrics[f'{group_name}_mae_variance'] = np.var(group_maes)
            fairness_metrics[f'{group_name}_rmse_variance'] = np.var(group_rmses)
            
            # Différence max-min
            fairness_metrics[f'{group_name}_mae_range'] = np.max(group_maes) - np.min(group_maes)
            fairness_metrics[f'{group_name}_rmse_range'] = np.max(group_rmses) - np.min(group_rmses)
            
            # Coefficient de variation
            fairness_metrics[f'{group_name}_mae_cv'] = np.std(group_maes) / np.mean(group_maes) if np.mean(group_maes) > 0 else 0
            fairness_metrics[f'{group_name}_rmse_cv'] = np.std(group_rmses) / np.mean(group_rmses) if np.mean(group_rmses) > 0 else 0
    
    return fairness_metrics


def calculate_coverage_fairness(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray,
                              groups_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculer la fairness pour la couverture des intervalles."""
    coverage_metrics = {}
    
    for group_name, groups in groups_dict.items():
        unique_groups = np.unique(groups)
        group_coverages = []
        
        for group in unique_groups:
            mask = groups == group
            if np.sum(mask) > 0:
                group_coverage = coverage_probability(
                    y_true[mask], y_lower[mask], y_upper[mask]
                )
                coverage_metrics[f'{group_name}_{group}_coverage'] = group_coverage
                group_coverages.append(group_coverage)
        
        if len(group_coverages) > 1:
            # Variance de couverture inter-groupes
            coverage_metrics[f'{group_name}_coverage_variance'] = np.var(group_coverages)
            coverage_metrics[f'{group_name}_coverage_range'] = np.max(group_coverages) - np.min(group_coverages)
    
    return coverage_metrics


class FairnessEvaluator:
    """Évaluateur de fairness pour les modèles de prévision."""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_fairness(self, model_name: str, df: pd.DataFrame, y_true: np.ndarray, 
                         y_pred: Union[np.ndarray, Dict], 
                         custom_groups: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """Évaluation complète de fairness."""
        # Créer les groupes
        groups_dict = create_fairness_groups(df)
        if custom_groups:
            groups_dict.update(custom_groups)
        
        fairness_metrics = {}
        
        # Prédictions ponctuelles
        if isinstance(y_pred, np.ndarray):
            fairness_metrics.update(
                calculate_fairness_metrics(y_true, y_pred, groups_dict)
            )
            
        # Prédictions avec intervalles
        elif isinstance(y_pred, dict) and 'q10' in y_pred and 'q90' in y_pred:
            # Métriques sur la médiane
            if 'q50' in y_pred:
                fairness_metrics.update(
                    calculate_fairness_metrics(y_true, y_pred['q50'], groups_dict)
                )
            
            # Métriques de couverture
            coverage_metrics = calculate_coverage_fairness(
                y_true, y_pred['q10'], y_pred['q90'], groups_dict
            )
            fairness_metrics.update(coverage_metrics)
        
        self.results[model_name] = fairness_metrics
        return fairness_metrics
        
    def compare_fairness(self) -> pd.DataFrame:
        """Comparer la fairness entre modèles."""
        if not self.results:
            return pd.DataFrame()
            
        return pd.DataFrame(self.results).T
        
    def get_fairness_summary(self) -> Dict[str, Dict]:
        """Résumé de fairness par modèle."""
        summary = {}
        
        for model_name, metrics in self.results.items():
            model_summary = {}
            
            # Grouper les métriques par type
            for metric_name, value in metrics.items():
                if '_variance' in metric_name:
                    group_type = metric_name.split('_variance')[0]
                    if group_type not in model_summary:
                        model_summary[group_type] = {}
                    model_summary[group_type]['variance'] = value
                elif '_range' in metric_name:
                    group_type = metric_name.split('_range')[0]
                    if group_type not in model_summary:
                        model_summary[group_type] = {}
                    model_summary[group_type]['range'] = value
                elif '_cv' in metric_name:
                    group_type = metric_name.split('_cv')[0]
                    if group_type not in model_summary:
                        model_summary[group_type] = {}
                    model_summary[group_type]['coefficient_variation'] = value
            
            summary[model_name] = model_summary
            
        return summary
        
    def get_most_fair_model(self, criterion: str = 'variance') -> str:
        """Obtenir le modèle le plus équitable selon un critère."""
        if not self.results:
            return None
            
        if criterion == 'variance':
            # Somme des variances (plus faible = plus équitable)
            model_scores = {}
            for model_name, metrics in self.results.items():
                variance_sum = sum(
                    value for key, value in metrics.items() 
                    if '_variance' in key and isinstance(value, (int, float))
                )
                model_scores[model_name] = variance_sum
                
            return min(model_scores, key=model_scores.get) if model_scores else None
            
        elif criterion == 'range':
            # Somme des ranges (plus faible = plus équitable)
            model_scores = {}
            for model_name, metrics in self.results.items():
                range_sum = sum(
                    value for key, value in metrics.items() 
                    if '_range' in key and isinstance(value, (int, float))
                )
                model_scores[model_name] = range_sum
                
            return min(model_scores, key=model_scores.get) if model_scores else None
            
        else:
            raise ValueError(f"Critère non supporté: {criterion}")
