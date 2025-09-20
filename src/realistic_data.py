"""Générateur de données énergétiques réalistes pour éviter l'overfitting."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple


def generate_realistic_energy_data(
    start_date: datetime = None,
    days: int = 365,
    freq: str = 'h',
    base_load: float = 50000,  # MW - charge de base réaliste
    add_realistic_noise: bool = True
) -> pd.DataFrame:
    """
    Générer des données de consommation énergétique réalistes.
    
    Basé sur les patterns réels des réseaux électriques français.
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    dates = pd.date_range(start=start_date, periods=days*24 if freq=='h' else days*24*4, freq=freq)
    
    # 1. CHARGE DE BASE (variations lentes)
    # Tendance saisonnière réaliste (hiver = plus de consommation)
    seasonal_trend = np.sin(2 * np.pi * (dates.dayofyear - 1) / 365) * (-8000) + 2000
    
    # Croissance économique lente
    growth_trend = np.linspace(0, 1000, len(dates))
    
    base_consumption = base_load + seasonal_trend + growth_trend
    
    # 2. PATTERNS JOURNALIERS RÉALISTES
    # Pas parfaitement sinusoïdaux !
    hour_of_day = dates.hour
    
    # Profil de consommation français type
    daily_profile = np.zeros_like(hour_of_day, dtype=float)
    
    # Nuit (0-6h) : Consommation basse
    night_mask = (hour_of_day >= 0) & (hour_of_day < 6)
    daily_profile[night_mask] = -12000 + np.random.normal(0, 1000, np.sum(night_mask))
    
    # Matin (6-10h) : Montée progressive 
    morning_mask = (hour_of_day >= 6) & (hour_of_day < 10)
    morning_values = np.linspace(-8000, 8000, 4)
    daily_profile[morning_mask] = morning_values[hour_of_day[morning_mask] - 6] + np.random.normal(0, 1500, np.sum(morning_mask))
    
    # Journée (10-17h) : Plateau élevé avec variations
    day_mask = (hour_of_day >= 10) & (hour_of_day < 17)
    daily_profile[day_mask] = 6000 + np.random.normal(0, 2000, np.sum(day_mask))
    
    # Pic du soir (17-21h) : Maximum de consommation
    evening_mask = (hour_of_day >= 17) & (hour_of_day < 21)
    daily_profile[evening_mask] = 10000 + np.random.normal(0, 2500, np.sum(evening_mask))
    
    # Soirée (21-24h) : Décroissance
    late_mask = (hour_of_day >= 21)
    late_values = np.linspace(5000, -8000, 3)
    daily_profile[late_mask] = late_values[hour_of_day[late_mask] - 21] + np.random.normal(0, 1500, np.sum(late_mask))
    
    # 3. EFFET WEEKEND (consommation différente)
    weekend_effect = np.where(dates.dayofweek >= 5, -5000 + np.random.normal(0, 1000, len(dates)), 0)
    
    # 4. ÉVÉNEMENTS MÉTÉO ET EXCEPTIONNELS
    # Vagues de froid/chaleur (5% de probabilité)
    weather_events = np.random.choice(
        [0, 15000, -8000], 
        len(dates), 
        p=[0.95, 0.03, 0.02]  # Normal, froid extrême, temps très doux
    )
    
    # Pannes et maintenance (1% de probabilité)
    maintenance_events = np.random.choice(
        [0, -20000], 
        len(dates), 
        p=[0.99, 0.01]
    )
    
    # 5. BRUIT RÉALISTE (autocorrélé)
    if add_realistic_noise:
        # Bruit avec mémoire (consommation dépend de l'heure précédente)
        noise = np.random.normal(0, 3000, len(dates))
        for i in range(1, len(noise)):
            noise[i] += 0.2 * noise[i-1]  # Autocorrélation
    else:
        noise = np.random.normal(0, 1000, len(dates))
    
    # 6. ASSEMBLAGE FINAL
    total_consumption = (
        base_consumption + 
        daily_profile + 
        weekend_effect + 
        weather_events + 
        maintenance_events + 
        noise
    )
    
    # Contraintes réalistes
    total_consumption = np.clip(total_consumption, 20000, 100000)  # Limites physiques
    
    return pd.DataFrame({
        'consommation_mw': total_consumption
    }, index=dates)


def create_realistic_features(df: pd.DataFrame, max_features: int = 15) -> pd.DataFrame:
    """
    Créer un ensemble limité de features non-overfittantes.
    
    MAXIMUM 15 features pour éviter l'overfitting !
    """
    # Identifier et renommer la colonne cible
    target_candidates = ['consommation_mw', 'consommation', 'y']
    actual_target = None
    
    for candidate in target_candidates:
        if candidate in df.columns:
            actual_target = candidate
            break
    
    if actual_target and actual_target != 'y':
        df = df.rename(columns={actual_target: 'y'})
    
    # Features temporelles de base SEULEMENT
    df_features = df.copy()
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)
    
    # MAXIMUM 3 lags (au lieu de 6+)
    df_features['y_lag_1'] = df_features['y'].shift(1)
    df_features['y_lag_24'] = df_features['y'].shift(24)
    df_features['y_lag_168'] = df_features['y'].shift(168)  # 1 semaine
    
    # MAXIMUM 3 rolling features
    df_features['y_rolling_mean_24'] = df_features['y'].rolling(24, min_periods=1).mean()
    df_features['y_rolling_mean_168'] = df_features['y'].rolling(168, min_periods=1).mean()
    df_features['y_rolling_std_24'] = df_features['y'].rolling(24, min_periods=1).std()
    
    # Features cycliques (SEULEMENT les plus importantes)
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    
    # Features d'interaction (LIMITÉES)
    df_features['is_peak_hour'] = df_features['hour'].isin([8, 9, 18, 19, 20]).astype(int)
    df_features['weekend_evening'] = (df_features['is_weekend'] * (df_features['hour'] > 18)).astype(int)
    
    # Gestion des NaN CONSERVATIVE
    df_features = df_features.fillna(method='ffill').fillna(0)
    
    # Supprimer les lignes avec des NaN dans la cible
    df_features = df_features.dropna(subset=['y'])
    
    # VÉRIFICATION : Pas plus de max_features features
    feature_cols = [c for c in df_features.columns if c != 'y']
    if len(feature_cols) > max_features:
        print(f"Trop de features ({len(feature_cols)} > {max_features}), sélection des {max_features} premières")
        df_features = df_features[['y'] + feature_cols[:max_features]]
    
    print(f"Features créées: {len([c for c in df_features.columns if c != 'y'])}")
    
    return df_features


def temporal_split_with_gap(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    gap_hours: int = 168  # 1 semaine de gap
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporel avec gap pour éviter le data leakage.
    
    Train: [début] -----> [fin train]
    Gap:                          [gap de 1 semaine]
    Test:                                    [début test] -> [fin]
    """
    total_size = len(df)
    test_size_points = int(total_size * test_size)
    gap_size_points = gap_hours
    
    # Calculer les indices
    train_end = total_size - test_size_points - gap_size_points
    test_start = train_end + gap_size_points
    
    train_data = df.iloc[:train_end].copy()
    test_data = df.iloc[test_start:].copy()
    
    print(f" Split temporel avec gap:")
    print(f"   Train: {train_data.index.min()} → {train_data.index.max()} ({len(train_data)} points)")
    print(f"   Gap: {gap_hours}h ({train_data.index.max()} → {test_data.index.min()})")
    print(f"   Test: {test_data.index.min()} → {test_data.index.max()} ({len(test_data)} points)")
    
    return train_data, test_data


def validate_realistic_performance(y_true, y_pred, model_name="Model"):
    """
    Valider que les performances sont dans les ranges réalistes.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n PERFORMANCE DE {model_name}:")
    print(f"   MAE: {mae:.0f} MW")
    print(f"   RMSE: {rmse:.0f} MW")
    print(f"   R²: {r2:.3f}")
    print(f"   MAPE: {mape:.1f}%")
    
    # Validation des ranges réalistes
    warnings = []
    
    if r2 > 0.90:
        warnings.append(f" R² trop élevé ({r2:.3f}) - Probable overfitting")
    if mape < 3.0:
        warnings.append(f" MAPE trop faible ({mape:.1f}%) - Probable overfitting")
    if r2 < 0.50:
        warnings.append(f" R² faible ({r2:.3f}) - Modèle peu performant")
    if mape > 20.0:
        warnings.append(f" MAPE élevé ({mape:.1f}%) - Prédictions imprécises")
    
    if warnings:
        print("   ALERTES:")
        for warning in warnings:
            print(f"      {warning}")
    else:
        print("Performances dans les ranges réalistes")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'is_realistic': len(warnings) == 0
    }
