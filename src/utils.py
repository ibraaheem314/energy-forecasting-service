"""Fonctions utilitaires génériques pour le projet de prévision énergétique."""

import os
import json
import pickle
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union


def ensure_dir(directory: Union[str, Path]) -> Path:
    """Créer un répertoire s'il n'existe pas."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, filepath: Union[str, Path]) -> None:
    """Sauvegarder des données en JSON."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: Union[str, Path]) -> Dict:
    """Charger des données depuis JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """Sauvegarder un objet avec pickle."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Charger un objet depuis pickle."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def generate_cache_key(*args, **kwargs) -> str:
    """Générer une clé de cache basée sur les arguments."""
    # Convertir tous les arguments en string
    key_parts = []
    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(json.dumps(arg, sort_keys=True, default=str))
        else:
            key_parts.append(str(arg))
    
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (dict, list)):
            key_parts.append(f"{key}:{json.dumps(value, sort_keys=True, default=str)}")
        else:
            key_parts.append(f"{key}:{str(value)}")
    
    # Hasher pour une clé compacte
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def format_timestamp(timestamp: datetime, format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Formater un timestamp."""
    return timestamp.strftime(format_str)


def get_timestamp_now(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Obtenir le timestamp actuel formaté."""
    return format_timestamp(datetime.now(), format_str)


def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None, 
                      min_rows: int = 1) -> bool:
    """Valider un DataFrame."""
    if df is None or df.empty:
        return False
        
    if len(df) < min_rows:
        return False
        
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            print(f"Colonnes manquantes: {missing_cols}")
            return False
            
    return True


def detect_data_frequency(df: pd.DataFrame, timestamp_col: str = None) -> str:
    """Détecter la fréquence des données temporelles."""
    if timestamp_col and timestamp_col in df.columns:
        timestamps = pd.to_datetime(df[timestamp_col])
    elif hasattr(df.index, 'freq') or isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index
    else:
        return "unknown"
    
    if len(timestamps) < 2:
        return "unknown"
    
    # Calculer les différences
    diffs = timestamps.diff().dropna()
    most_common_diff = diffs.mode()[0] if not diffs.empty else None
    
    if most_common_diff is None:
        return "unknown"
    
    # Mapper vers des fréquences standard
    if most_common_diff == timedelta(minutes=15):
        return "15min"
    elif most_common_diff == timedelta(hours=1):
        return "1h"
    elif most_common_diff == timedelta(days=1):
        return "1d"
    else:
        return f"custom_{most_common_diff}"


def resample_timeseries(df: pd.DataFrame, target_freq: str = "1h", 
                       agg_method: str = "mean") -> pd.DataFrame:
    """Ré-échantillonner une série temporelle."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("L'index doit être de type DatetimeIndex")
    
    agg_func = {
        "mean": "mean",
        "sum": "sum",
        "first": "first",
        "last": "last",
        "min": "min",
        "max": "max"
    }.get(agg_method, "mean")
    
    return df.resample(target_freq).agg(agg_func)


def split_train_test(df: pd.DataFrame, test_size: float = 0.2, 
                    time_split: bool = True) -> tuple:
    """Diviser les données en train/test."""
    if time_split:
        # Split temporel (plus réaliste pour les séries temporelles)
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
    else:
        # Split aléatoire
        test_df = df.sample(frac=test_size, random_state=42)
        train_df = df.drop(test_df.index)
    
    return train_df, test_df


def calculate_data_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculer des statistiques sur les données."""
    stats = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    # Statistiques pour colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # Fréquence temporelle si index datetime
    if isinstance(df.index, pd.DatetimeIndex):
        stats['time_frequency'] = detect_data_frequency(df)
        stats['time_range'] = {
            'start': str(df.index.min()),
            'end': str(df.index.max()),
            'duration_days': (df.index.max() - df.index.min()).days
        }
    
    return stats


def log_execution_time(func):
    """Décorateur pour mesurer le temps d'exécution."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Fonction '{func.__name__}' exécutée en {execution_time:.2f} secondes")
        return result
    
    return wrapper


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Configurer le logging."""
    import logging
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class ConfigManager:
    """Gestionnaire de configuration."""
    
    def __init__(self, config_path: Union[str, Path] = None):
        self.config_path = Path(config_path) if config_path else Path("config.json")
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Charger la configuration."""
        if self.config_path.exists():
            return load_json(self.config_path)
        else:
            return self.get_default_config()
    
    def save_config(self):
        """Sauvegarder la configuration."""
        save_json(self.config, self.config_path)
    
    def get(self, key: str, default=None):
        """Obtenir une valeur de configuration."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Définir une valeur de configuration."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()
    
    def get_default_config(self) -> Dict:
        """Configuration par défaut."""
        return {
            'data': {
                'source': 'odre',
                'window_days': 365,
                'resample_hourly': False
            },
            'models': {
                'test_size': 0.2,
                'random_state': 42
            },
            'api': {
                'host': '127.0.0.1',
                'port': 8000
            },
            'cache': {
                'enabled': True,
                'max_size_mb': 1000
            }
        }
