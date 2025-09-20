"""Features avancées pour améliorer les performances des modèles."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler
from .features import create_features


def create_advanced_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Créer des features temporelles avancées."""
    df = df.copy()
    
    # Features de calendrier avancées
    df['is_business_day'] = (df.index.dayofweek < 5).astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    
    # Semaine de l'année normalisée
    df['week_of_year_norm'] = (df.index.isocalendar().week - 1) / 51
    
    # Features de vacances françaises (approximation)
    df['is_holiday_period'] = (
        ((df.index.month == 7) | (df.index.month == 8)) |  # Vacances d'été
        ((df.index.month == 12) & (df.index.day >= 20)) |  # Noël
        ((df.index.month == 1) & (df.index.day <= 10))     # Nouvel An
    ).astype(int)
    
    # Features d'interaction temporelle
    df['hour_weekday_interaction'] = df.index.hour * (df.index.dayofweek + 1)
    df['month_hour_interaction'] = df.index.month * df.index.hour
    
    return df


def create_lag_and_rolling_advanced(df: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
    """Créer des lags et rolling features avancés."""
    df = df.copy()
    
    # Lags multiples avec différentes fenêtres
    lag_periods = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h à 1 semaine
    for lag in lag_periods:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling features avec différentes fenêtres
    windows = [3, 6, 12, 24, 48, 168]  # 3h à 1 semaine
    
    for window in windows:
        # Statistiques de base
        rolling = df[target_col].rolling(window=window, min_periods=1)
        df[f'{target_col}_rolling_mean_{window}'] = rolling.mean()
        df[f'{target_col}_rolling_std_{window}'] = rolling.std()
        df[f'{target_col}_rolling_min_{window}'] = rolling.min()
        df[f'{target_col}_rolling_max_{window}'] = rolling.max()
        df[f'{target_col}_rolling_median_{window}'] = rolling.median()
        
        # Features dérivées
        df[f'{target_col}_rolling_range_{window}'] = (
            df[f'{target_col}_rolling_max_{window}'] - df[f'{target_col}_rolling_min_{window}']
        )
        
        # Z-score par rapport à la fenêtre
        mean_col = f'{target_col}_rolling_mean_{window}'
        std_col = f'{target_col}_rolling_std_{window}'
        df[f'{target_col}_zscore_{window}'] = (
            (df[target_col] - df[mean_col]) / (df[std_col] + 1e-8)
        )
    
    # EWMA avec différents alphas
    alphas = [0.1, 0.3, 0.5, 0.7]
    for alpha in alphas:
        df[f'{target_col}_ewma_{str(alpha).replace(".", "_")}'] = (
            df[target_col].ewm(alpha=alpha).mean()
        )
    
    # Différences et tendances
    df[f'{target_col}_diff_1'] = df[target_col].diff(1)
    df[f'{target_col}_diff_24'] = df[target_col].diff(24)
    df[f'{target_col}_pct_change_1'] = df[target_col].pct_change(1)
    df[f'{target_col}_pct_change_24'] = df[target_col].pct_change(24)
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Créer des features d'interaction."""
    df = df.copy()
    
    # Interactions temporelles importantes
    if 'hour' in df.columns and 'day_of_week' in df.columns:
        df['hour_dow_peak'] = (
            ((df['hour'].between(8, 10) | df['hour'].between(18, 20)) & 
             (df['day_of_week'] < 5))
        ).astype(int)
    
    if 'is_weekend' in df.columns and 'hour' in df.columns:
        df['weekend_hour_low'] = (
            (df['is_weekend'] == 1) & df['hour'].between(2, 6)
        ).astype(int)
    
    # Interactions avec lags
    if 'y_lag_1' in df.columns and 'hour' in df.columns:
        df['lag1_hour_interaction'] = df['y_lag_1'] * df['hour']
    
    if 'y_lag_24' in df.columns and 'day_of_week' in df.columns:
        df['lag24_dow_interaction'] = df['y_lag_24'] * df['day_of_week']
    
    return df


def create_fourier_features(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """Créer des features de Fourier pour capturer la cyclicité."""
    if periods is None:
        periods = [24, 168, 8760]  # Jour, semaine, année (en heures)
    
    df = df.copy()
    
    # Créer un index temporel en heures depuis le début
    time_index = np.arange(len(df))
    
    for period in periods:
        for k in range(1, 4):  # 3 harmoniques par période
            df[f'fourier_cos_{period}_{k}'] = np.cos(2 * np.pi * k * time_index / period)
            df[f'fourier_sin_{period}_{k}'] = np.sin(2 * np.pi * k * time_index / period)
    
    return df


def create_statistical_features(df: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
    """Créer des features statistiques avancées."""
    df = df.copy()
    
    # Features de distribution
    windows = [24, 168]  # Jour et semaine
    
    for window in windows:
        rolling = df[target_col].rolling(window=window, min_periods=1)
        
        # Quantiles
        df[f'{target_col}_q25_{window}'] = rolling.quantile(0.25)
        df[f'{target_col}_q75_{window}'] = rolling.quantile(0.75)
        df[f'{target_col}_iqr_{window}'] = (
            df[f'{target_col}_q75_{window}'] - df[f'{target_col}_q25_{window}']
        )
        
        # Skewness et Kurtosis (approximations)
        mean = df[f'{target_col}_rolling_mean_{window}'] if f'{target_col}_rolling_mean_{window}' in df.columns else rolling.mean()
        std = df[f'{target_col}_rolling_std_{window}'] if f'{target_col}_rolling_std_{window}' in df.columns else rolling.std()
        
        # Position relative dans la distribution
        df[f'{target_col}_percentile_{window}'] = (
            (df[target_col] - mean) / (std + 1e-8)
        )
    
    return df


def create_comprehensive_features(df: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
    """Pipeline complet de création de features avancées."""
    # Commencer par les features de base
    df = create_features(df, target_col)
    
    # Ajouter les features avancées
    df = create_advanced_temporal_features(df)
    df = create_lag_and_rolling_advanced(df, target_col)
    df = create_interaction_features(df)
    df = create_fourier_features(df)
    df = create_statistical_features(df, target_col)
    
    # Gestion finale des NaN
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df


def feature_selection_advanced(df: pd.DataFrame, target_col: str = 'y', max_features: int = 50) -> pd.DataFrame:
    """Sélection intelligente des features les plus importantes."""
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.ensemble import RandomForestRegressor
    
    # Séparer features et target
    X = df[[col for col in df.columns if col != target_col]]
    y = df[target_col]
    
    # Supprimer les features avec trop de NaN ou variance nulle
    X = X.loc[:, X.isnull().sum() < len(X) * 0.5]  # Moins de 50% de NaN
    X = X.loc[:, X.var() > 1e-8]  # Variance non nulle
    
    if len(X.columns) <= max_features:
        return df
    
    # Méthode 1: F-test
    selector_f = SelectKBest(score_func=f_regression, k=min(max_features, len(X.columns)))
    X_selected_f = selector_f.fit_transform(X.fillna(0), y)
    selected_features_f = X.columns[selector_f.get_support()].tolist()
    
    # Méthode 2: Feature importance Random Forest
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X.fillna(0), y)
    
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
    top_features_rf = feature_importance.nlargest(max_features).index.tolist()
    
    # Combiner les deux méthodes
    selected_features = list(set(selected_features_f + top_features_rf))[:max_features]
    selected_features.append(target_col)  # Garder la target
    
    print(f"Feature selection: {len(X.columns)} → {len(selected_features)-1} features")
    
    return df[selected_features]


def scale_features_robust(df: pd.DataFrame, target_col: str = 'y', scaler_type: str = 'robust') -> pd.DataFrame:
    """Scaling robuste des features."""
    df = df.copy()
    
    feature_cols = [col for col in df.columns if col != target_col]
    
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df
