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

DATA_SOURCE = os.getenv("DATA_SOURCE", "synthetic").lower()  # "synthetic" | "odre"
ODRE_BASE_URL = os.getenv("ODRE_BASE_URL", "https://odre.opendatasoft.com")
ODRE_DATASET = os.getenv("ODRE_DATASET", "eco2mix-national-cons-def")
WINDOW_DAYS = int(os.getenv("WINDOW_DAYS", "1095"))  # 3 ans pour avoir plus de données
RESAMPLE_HOURLY = os.getenv("RESAMPLE_HOURLY", "false").lower() == "true"  # Garder 15min natif

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _clean_colname(c: str) -> str:
    # uniformise accents/espaces et passe en snake_case léger
    s = str(c).strip().lower()
    s = (s.replace("\xa0", " ").replace("\u202f", " ")
           .replace("’", "'").replace("é", "e").replace("è", "e")
           .replace("ê", "e").replace("à", "a").replace("ç", "c"))
    s = " ".join(s.split())            # squeeze multiples spaces
    s = s.replace(" - ", " ").replace("-", " ").replace("/", " ")
    s = s.replace("(", "").replace(")", "")
    s = s.replace(".", " ").replace(",", " ")
    s = "_".join(s.split())            # to snake_case minimal
    return s

def _to_float(x):
    # convertit '63 949', '1 894', '-2 497', '0', "1894", "" -> float (ou NaN)
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace("\u202f", "").replace("\xa0", "").replace(" ", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def _fetch_odre_csv() -> pd.DataFrame:
    url = (
        f"{ODRE_BASE_URL}/explore/dataset/{ODRE_DATASET}/download/"
        "?format=csv&timezone=UTC&lang=fr&use_labels_for_header=true"
    )
    logger.info("Fetching ODRÉ CSV from %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    # CSV français utilise le séparateur point-virgule
    return pd.read_csv(io.BytesIO(resp.content), sep=';')

def _load_odre() -> pd.DataFrame:
    """
    Charge ODRÉ, normalise les noms/valeurs, indexe par timestamp UTC
    et applique une fenêtre temporelle. Aucune sélection/renommage métier ici.
    """
    end = _now_utc()
    start = end - timedelta(days=WINDOW_DAYS)

    df = _fetch_odre_csv()

    # normalise les noms de colonnes
    df.columns = [_clean_colname(c) for c in df.columns]

    # détecte une colonne temporelle plausible
    # (préférence pour 'date_heure' ou 'date_heure_utc' selon exports)
    ts_candidates = [c for c in df.columns if "date" in c or "heure" in c or "time" in c]
    if not ts_candidates:
        raise RuntimeError("Aucune colonne temporelle détectée dans le CSV ODRÉ.")
    ts_col = ts_candidates[0]

    # parse datetime UTC
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)
    df.index.name = "timestamp"

    # conversion numérique sur toutes les colonnes non-datetime
    for c in df.columns:
        df[c] = df[c].apply(_to_float) if df[c].dtype == object else df[c]

    # fenêtre temporelle
    df = df.loc[(df.index >= start) & (df.index <= end)]

    # resample horaire optionnel (sinon on garde 15 min natif)
    if RESAMPLE_HOURLY:
        df = df.resample("h").mean()

    return df

def _load_synthetic() -> pd.DataFrame:
    """Série synthétique multi-variables minimaliste (pour fallback)."""
    end = _now_utc().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=WINDOW_DAYS)
    idx = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    hours = np.arange(len(idx))
    base = 30000 + 5000 * np.sin(2 * np.pi * (hours % 24) / 24.0) + \
           2000 * np.sin(2 * np.pi * (hours % (24*7)) / (24*7))
    noise = np.random.normal(0, 800, size=len(idx))
    df = pd.DataFrame({
        "consommation": base + noise,
        "prevision_j1": base * 0.998,
        "prevision_j":  base * 1.001,
        "nucleaire": 34000 + 500*np.sin(2*np.pi*(hours%24)/24.0),
        "eolien": 3000 + 1500*np.sin(2*np.pi*(hours%168)/168.0),
        "gaz": 1000 + 200*np.cos(2*np.pi*(hours%24)/24.0),
        "taux_co2": 20 + 3*np.random.randn(len(idx)),
    }, index=idx)
    df.index.name = "timestamp"
    return df

def load_timeseries(location: str = "France") -> pd.DataFrame:
    """
    Retourne un DataFrame indexé UTC.
    """
    try:
        if DATA_SOURCE == "odre":
            return _load_odre()
        return _load_synthetic()
    except Exception as e:
        logger.exception("Erreur de chargement données: %s", e)
        return _load_synthetic()

