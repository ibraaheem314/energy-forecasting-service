"""Tests pour l'engineering des features."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Ajouter le chemin src
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features import create_temporal_features, create_features


def test_create_temporal_features():
    """Test création des features temporelles."""
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=24, freq='h')
    df = pd.DataFrame({'value': range(24)}, index=dates)
    
    result = create_temporal_features(df)
    
    # Vérifier les features de base
    assert 'hour' in result.columns
    assert 'day_of_week' in result.columns
    assert 'month' in result.columns
    
    # Vérifier les features cycliques
    assert 'hour_sin' in result.columns
    assert 'hour_cos' in result.columns
    
    # Vérifier les features booléennes
    assert 'is_weekend' in result.columns
    assert 'is_business_hour' in result.columns


def test_create_features():
    """Test du pipeline complet."""
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=50, freq='h')
    df = pd.DataFrame({
        'consommation_mw': 100 + np.sin(2 * np.pi * np.arange(50) / 24) * 10
    }, index=dates)
    
    result = create_features(df)
    
    # Vérifier que la colonne cible a été renommée
    assert 'y' in result.columns
    assert 'consommation_mw' not in result.columns
    
    # Vérifier la présence des features
    assert 'hour' in result.columns
    assert 'y_lag_1' in result.columns
    assert 'y_rolling_mean_3' in result.columns
    
    # Vérifier qu'il n'y a pas de NaN dans la colonne cible
    assert not result['y'].isnull().any()
