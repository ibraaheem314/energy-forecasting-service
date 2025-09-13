import os
import io
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_SOURCE = os.getenv("DATA_SOURCE", "synthetic").lower()  # synthetic | odre
ODRE_BASE_URL = os.getenv("ODRE_BASE_URL", "https://odre.opendatasoft.com")
ODRE_DATASET = os.getenv("ODRE_DATASET", "eco2mix-national-cons-def")

def _now_utc():
    return datetime.now(timezone.utc)

def _to_hourly_index(start, periods):
    idx = pd.date_range(start=start, periods=periods, freq="h", tz="UTC")
    return idx

def _load_odre():
    """
    Chargement simple depuis Opendatasoft (ODRÉ), dataset par défaut:
      eco2mix-national-cons-def  (consommation électrique nationale)
    On récupère un CSV récent et on le met au format index horaire / colonne 'y'.
    """
    # Fenêtre de 120 jours par défaut (suffisant pour un modèle simple)
    end = _now_utc()
    start = end - timedelta(days=120)

    # Endpoint CSV "download" d'Opendatasoft (simple et sans OAuth)
    # NB: suivant le dataset, les colonnes peuvent varier ; on essaie champs courants.
    url = (
        f"{ODRE_BASE_URL}/explore/dataset/{ODRE_DATASET}/download/"
        "?format=csv&timezone=UTC&lang=fr&use_labels_for_header=true"
    )

    logger.info("Fetching ODRÉ CSV from %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(io.BytesIO(resp.content))
    # Tentative de détection de colonnes usuelles:
    # date_heure / consommation (MW)
    date_cols = [c for c in df.columns if "date" in c.lower() or "heure" in c.lower()]
    value_cols = [c for c in df.columns if "consommation" in c.lower() or "cons" in c.lower() or "load" in c.lower()]
    if not date_cols or not value_cols:
        # fallback: essayer 'datetime' et 'value'
        date_cols = date_cols or [c for c in df.columns if "time" in c.lower()]
        value_cols = value_cols or [c for c in df.columns if "value" in c.lower()]
    if not date_cols or not value_cols:
        raise RuntimeError("Colonnes date/valeur introuvables dans le CSV ODRÉ")

    ts_col = date_cols[0]
    y_col = value_cols[0]

    df = df[[ts_col, y_col]].copy()
    # Parse dates (UTC)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    df = df.set_index(ts_col).rename(columns={y_col: "y"})

    # Filtrer fenêtre récente
    df = df.loc[(df.index >= start) & (df.index <= end)]
    # Agg/Resample horaire (si besoin)
    df = df.resample("h").mean().interpolate(limit=3)

    # Nettoyage de base
    df = df[df["y"].notna()]
    return df

def _load_synthetic():
    """ Génère une série horaire synthétique avec saisonnalités journalière/hebdo. """
    end = _now_utc().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=120)
    idx = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    hours = np.arange(len(idx))
    # composantes
    daily = 30000 + 5000 * np.sin(2 * np.pi * (hours % 24) / 24.0)
    weekly = 2000 * np.sin(2 * np.pi * (hours % (24*7)) / (24*7))
    noise = np.random.normal(0, 800, size=len(idx))
    y = daily + weekly + noise
    df = pd.DataFrame({"y": y}, index=idx)
    return df

def load_timeseries(city: str = "Paris") -> pd.DataFrame:
    """Retourne une série horaire avec index UTC et colonne 'y' (MW)."""
    try:
        if DATA_SOURCE == "odre":
            return _load_odre()
        return _load_synthetic()
    except Exception as e:
        logger.exception("Erreur de chargement données: %s", e)
        # fallback synthétique
        return _load_synthetic()
