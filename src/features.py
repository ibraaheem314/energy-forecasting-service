"""Extraction et engineering des features pour la prévision énergétique."""

import numpy as np
import pandas as pd
from typing import List, Optional


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Créer les features temporelles (heure, jour, mois, etc.)."""
    df = df.copy()
    
    # Features temporelles de base
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    
    # Features cycliques
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Features booléennes
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    df['is_peak_hour'] = df['hour'].isin([8, 9, 10, 18, 19, 20]).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    return df


def create_lag_features(df: pd.DataFrame, target_col: str = 'y', lags: List[int] = None) -> pd.DataFrame:
    """Créer les features de lag."""
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24]
    
    df = df.copy()
    
    for lag in lags:
        if target_col in df.columns:
            # Utiliser .loc pour s'assurer qu'on récupère une Series
            df[f'{target_col}_lag_{lag}'] = df.loc[:, target_col].shift(lag)
        else:
            print(f"⚠️ Colonne '{target_col}' introuvable pour les lags")
    
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str = 'y', windows: List[int] = None) -> pd.DataFrame:
    """Créer les features de rolling (moyennes mobiles, etc.)."""
    if windows is None:
        windows = [3, 6, 12, 24, 168]
    
    df = df.copy()
    
    for window in windows:
        if target_col in df.columns:
            df[f'{target_col}_rolling_mean_{window}'] = df.loc[:, target_col].rolling(
                window=window, min_periods=1
            ).mean()
            df[f'{target_col}_rolling_std_{window}'] = df.loc[:, target_col].rolling(
                window=window, min_periods=1
            ).std()
        else:
            print(f"⚠️ Colonne '{target_col}' introuvable pour les rolling features")
    
    # Features EWMA
    if target_col in df.columns:
        df[f'{target_col}_ewma_0_1'] = df.loc[:, target_col].ewm(alpha=0.1).mean()
        df[f'{target_col}_ewma_0_3'] = df.loc[:, target_col].ewm(alpha=0.3).mean()
    else:
        print(f"⚠️ Colonne '{target_col}' introuvable pour les EWMA features")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Gestion des valeurs manquantes."""
    df = df.copy()
    
    # Pour les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            # Forward fill puis backward fill
            df[col] = df[col].ffill().bfill()
            # Si encore des NaN, utiliser la médiane
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    
    return df


def create_features(df: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
    """Pipeline complet de création des features."""
    # Identifier et renommer la colonne cible
    target_candidates = ['consommation_mw', 'consommation', 'y']
    actual_target = None
    
    for candidate in target_candidates:
        if candidate in df.columns:
            actual_target = candidate
            break
    
    if actual_target and actual_target != target_col:
        # Si la colonne target_col existe déjà, la supprimer pour éviter les doublons
        if target_col in df.columns:
            df = df.drop(columns=[target_col])
        df = df.rename(columns={actual_target: target_col})
    
    # Créer toutes les features
    df = create_temporal_features(df)
    df = create_lag_features(df, target_col)
    df = create_rolling_features(df, target_col)
    df = handle_missing_values(df)
    
    # Supprimer les lignes où la cible est NaN
    df = df.dropna(subset=[target_col])
    
    return df
