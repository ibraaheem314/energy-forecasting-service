"""Module pour charger les vraies données RTE ODRÉ depuis l'API officielle."""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# Configuration de l'API ODRÉ
ODRE_BASE_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/records"
ODRE_TIMEOUT = 30  # secondes
MAX_RECORDS_PER_REQUEST = 100  # Limite API


def fetch_odre_data(
    max_records: int = 1000,
    timeout: int = ODRE_TIMEOUT
) -> pd.DataFrame:
    """
    Récupérer les vraies données RTE ODRÉ depuis l'API officielle.
    
    Args:
        max_records: Nombre maximum d'enregistrements à récupérer
        timeout: Timeout pour les requêtes HTTP
    
    Returns:
        DataFrame avec les données de consommation électrique française
    """
    print(f"Récupération des données RTE ODRÉ (derniers {max_records} enregistrements)...")
    
    all_records = []
    offset = 0
    
    try:
        while len(all_records) < max_records:
            # Construire les paramètres de la requête (seulement les données non-nulles)
            params = {
                'limit': min(MAX_RECORDS_PER_REQUEST, max_records - len(all_records)),
                'offset': offset,
                'refine': 'nature:"Données temps réel"',
                'where': 'consommation IS NOT NULL',  # Seulement les données existantes
                'order_by': 'date_heure DESC'  # Les plus récents d'abord
            }
            
            print(f"Requête API: offset={offset}, limit={params['limit']}")
            
            # Faire la requête
            response = requests.get(ODRE_BASE_URL, params=params, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            records = data.get('results', [])
            
            if not records:
                print("Plus de données disponibles")
                break
                
            all_records.extend(records)
            offset += len(records)
            
            print(f"Récupéré {len(records)} enregistrements (total: {len(all_records)})")
            
            # Si on a moins d'enregistrements que demandé, on a tout récupéré
            if len(records) < params['limit']:
                break
                
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la récupération des données ODRÉ: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        return pd.DataFrame()
    
    if not all_records:
        print("Aucune donnée récupérée")
        return pd.DataFrame()
    
    print(f"Total récupéré: {len(all_records)} enregistrements")
    
    # Convertir en DataFrame
    df = _process_odre_records(all_records)
    
    return df


def _process_odre_records(records: list) -> pd.DataFrame:
    """
    Traiter les enregistrements bruts de l'API ODRÉ.
    
    Args:
        records: Liste des enregistrements JSON de l'API
        
    Returns:
        DataFrame nettoyé et indexé par datetime
    """
    print("Traitement des données ODRÉ...")
    
    processed_data = []
    
    for record in records:
        try:
            # Extraire les champs principaux
            row = {
                'date_heure': pd.to_datetime(record.get('date_heure')),
                'consommation': record.get('consommation'),
                'prevision_j1': record.get('prevision_j1'),
                'prevision_j': record.get('prevision_j'),
                'fioul': record.get('fioul', 0),
                'charbon': record.get('charbon', 0),
                'gaz': record.get('gaz', 0),
                'nucleaire': record.get('nucleaire', 0),
                'eolien': record.get('eolien', 0),
                'solaire': record.get('solaire', 0),
                'hydraulique': record.get('hydraulique', 0),
                'pompage': record.get('pompage', 0),
                'bioenergies': record.get('bioenergies', 0),
                'ech_physiques': record.get('ech_physiques', 0),
                'stockage_batterie': record.get('stockage_batterie', 0),
                'destockage_batterie': record.get('destockage_batterie', 0),
                'taux_co2': record.get('taux_co2'),
                'ech_comm_angleterre': record.get('ech_comm_angleterre', 0),
                'ech_comm_espagne': record.get('ech_comm_espagne', 0),
                'ech_comm_italie': record.get('ech_comm_italie', 0),
                'ech_comm_suisse': record.get('ech_comm_suisse', 0),
                'ech_comm_allemagne_belgique': record.get('ech_comm_allemagne_belgique', 0)
            }
            
            processed_data.append(row)
            
        except Exception as e:
            print(f"Erreur lors du traitement d'un enregistrement: {e}")
            continue
    
    if not processed_data:
        print("Aucune donnée traitée avec succès")
        return pd.DataFrame()
    
    # Créer le DataFrame
    df = pd.DataFrame(processed_data)
    
    # Vérifier que nous avons la colonne datetime
    if 'date_heure' not in df.columns:
        print("Colonne date_heure manquante")
        return pd.DataFrame()
    
    # Supprimer les lignes avec des timestamps invalides
    df = df.dropna(subset=['date_heure'])
    
    # Trier par date
    df = df.sort_values('date_heure')
    
    # Définir l'index temporel
    df.set_index('date_heure', inplace=True)
    
    # Convertir les colonnes numériques
    numeric_columns = [
        'consommation', 'prevision_j1', 'prevision_j', 'fioul', 'charbon', 'gaz',
        'nucleaire', 'eolien', 'solaire', 'hydraulique', 'pompage', 'bioenergies',
        'ech_physiques', 'stockage_batterie', 'destockage_batterie', 'taux_co2',
        'ech_comm_angleterre', 'ech_comm_espagne', 'ech_comm_italie', 
        'ech_comm_suisse', 'ech_comm_allemagne_belgique'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Renommer la colonne principale pour la compatibilité
    if 'consommation' in df.columns:
        df['y'] = df['consommation']
    
    print(f"Données traitées: {len(df)} lignes, période {df.index.min()} à {df.index.max()}")
    print(f"Colonnes disponibles: {list(df.columns)}")
    
    # Afficher quelques statistiques
    if 'consommation' in df.columns and not df['consommation'].isna().all():
        print(f"Consommation - Min: {df['consommation'].min():.0f} MW, "
              f"Max: {df['consommation'].max():.0f} MW, "
              f"Moyenne: {df['consommation'].mean():.0f} MW")
    
    return df


def get_odre_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Générer un résumé des données ODRÉ récupérées.
    
    Args:
        df: DataFrame avec les données ODRÉ
        
    Returns:
        Dictionnaire avec les statistiques des données
    """
    if df.empty:
        return {"status": "empty", "message": "Aucune donnée disponible"}
    
    summary = {
        "status": "success",
        "records_count": len(df),
        "date_range": {
            "start": df.index.min().isoformat() if not df.index.empty else None,
            "end": df.index.max().isoformat() if not df.index.empty else None,
            "duration_hours": len(df)
        },
        "columns": list(df.columns),
        "missing_data": df.isnull().sum().to_dict(),
        "data_quality": {
            "completeness": (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
    }
    
    # Statistiques de consommation si disponible
    if 'consommation' in df.columns and not df['consommation'].isna().all():
        summary["consumption_stats"] = {
            "min_mw": df['consommation'].min(),
            "max_mw": df['consommation'].max(),
            "mean_mw": df['consommation'].mean(),
            "std_mw": df['consommation'].std()
        }
    
    return summary


def load_odre_data_cached(cache_hours: int = 1, max_records: int = 1000) -> pd.DataFrame:
    """
    Charger les données ODRÉ avec cache simple en mémoire.
    
    Args:
        cache_hours: Durée de validité du cache en heures
        max_records: Nombre maximum d'enregistrements à récupérer
        
    Returns:
        DataFrame avec les données ODRÉ
    """
    # Cache simple en variable globale (pour cette session)
    if not hasattr(load_odre_data_cached, '_cache'):
        load_odre_data_cached._cache = {}
    
    cache_key = "odre_data"
    now = datetime.now()
    
    # Vérifier si le cache est valide
    if (cache_key in load_odre_data_cached._cache and
        'timestamp' in load_odre_data_cached._cache[cache_key] and
        'data' in load_odre_data_cached._cache[cache_key]):
        
        cache_time = load_odre_data_cached._cache[cache_key]['timestamp']
        if (now - cache_time).total_seconds() < cache_hours * 3600:
            print(f"Utilisation du cache ODRÉ (âge: {(now - cache_time).total_seconds()/60:.1f} min)")
            return load_odre_data_cached._cache[cache_key]['data']
    
    # Récupérer de nouvelles données
    print("Cache expiré, récupération de nouvelles données ODRÉ...")
    df = fetch_odre_data(max_records=max_records)
    
    # Mettre en cache
    load_odre_data_cached._cache[cache_key] = {
        'timestamp': now,
        'data': df
    }
    
    return df


if __name__ == "__main__":
    # Test du module
    print("Test du module de chargement ODRÉ")
    
    # Récupérer les données
    df = fetch_odre_data(max_records=200)
    
    if not df.empty:
        print("\n Aperçu des données:")
        print(df.head())
        
        print("\n Résumé:")
        summary = get_odre_data_summary(df)
        print(summary)
    else:
        print("Aucune donnée récupérée")
