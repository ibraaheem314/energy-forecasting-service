"""Cache intelligent pour éviter de recharger les données plusieurs fois."""
import os
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import sys

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))


class DataCache:
    """Gestionnaire de cache pour les données de time series."""
    
    def __init__(self, cache_dir="data/cache", cache_duration_hours=1):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self._memory_cache = {}
    
    def _get_cache_key(self, data_source, location="France"):
        """Générer une clé de cache basée sur les paramètres."""
        key_data = f"{data_source}_{location}_{os.getenv('WINDOW_DAYS', '1095')}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key):
        """Obtenir le chemin du fichier cache."""
        return self.cache_dir / f"data_{cache_key}.pkl"
    
    def _is_cache_valid(self, cache_path):
        """Vérifier si le cache est encore valide."""
        if not cache_path.exists():
            return False
        
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - cache_time < self.cache_duration
    
    def get_data(self, data_source="odre", location="France", force_reload=False):
        """
        Obtenir les données, soit du cache, soit en les rechargeant.
        
        Args:
            data_source: Source des données ('odre' ou 'synthetic')
            location: Localisation (pour compatibilité future)
            force_reload: Forcer le rechargement même si cache valide
        
        Returns:
            DataFrame avec données et features créées
        """
        cache_key = self._get_cache_key(data_source, location)
        
        # 1. Vérifier le cache mémoire d'abord (plus rapide)
        if not force_reload and cache_key in self._memory_cache:
            print(f"Utilisation cache mémoire pour {data_source}")
            return self._memory_cache[cache_key].copy()
        
        # 2. Vérifier le cache disque
        cache_path = self._get_cache_path(cache_key)
        if not force_reload and self._is_cache_valid(cache_path):
            print(f"Chargement depuis cache disque: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self._memory_cache[cache_key] = data
                    return data.copy()
            except Exception as e:
                print(f"Erreur lecture cache: {e}, rechargement...")
        
        # 3. Recharger les données
        print(f"Rechargement données {data_source}...")
        data = self._load_fresh_data(data_source, location)
        
        # 4. Sauvegarder en cache
        self._save_to_cache(data, cache_path, cache_key)
        
        return data.copy()
    
    def _load_fresh_data(self, data_source, location):
        """Charger des données fraîches et créer les features."""
        # Import ici pour éviter les imports circulaires
        from app.services.loader import load_timeseries
        from scripts.train_simple import create_features
        
        # Configurer la source de données
        old_source = os.getenv("DATA_SOURCE")
        os.environ["DATA_SOURCE"] = data_source
        
        try:
            # Charger et traiter
            df_raw = load_timeseries(location)
            df_features = create_features(df_raw)
            
            print(f"Données fraîches: {len(df_features)} échantillons")
            return df_features
        
        finally:
            # Restaurer l'ancienne config
            if old_source:
                os.environ["DATA_SOURCE"] = old_source
    
    def _save_to_cache(self, data, cache_path, cache_key):
        """Sauvegarder les données en cache."""
        try:
            # Cache disque
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Cache mémoire
            self._memory_cache[cache_key] = data
            
            print(f"Données sauvées en cache: {cache_path}")
            
        except Exception as e:
            print(f"Erreur sauvegarde cache: {e}")
    
    def clear_cache(self, data_source=None):
        """Vider le cache (tout ou pour une source spécifique)."""
        if data_source:
            cache_key = self._get_cache_key(data_source)
            # Supprimer du cache mémoire
            self._memory_cache.pop(cache_key, None)
            # Supprimer du cache disque
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
                print(f"Cache {data_source} supprimé")
        else:
            # Vider tout
            self._memory_cache.clear()
            for cache_file in self.cache_dir.glob("data_*.pkl"):
                cache_file.unlink()
            print("Tout le cache supprimé")
    
    def get_cache_info(self):
        """Obtenir des infos sur le cache."""
        memory_count = len(self._memory_cache)
        disk_files = list(self.cache_dir.glob("data_*.pkl"))
        disk_count = len(disk_files)
        
        total_size = sum(f.stat().st_size for f in disk_files) / (1024 * 1024)  # MB
        
        print(f"Cache Info:")
        print(f"   - Mémoire: {memory_count} datasets")
        print(f"   - Disque: {disk_count} fichiers ({total_size:.1f} MB)")
        
        for file in disk_files:
            age = datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)
            print(f"   - {file.name}: {age.total_seconds()/3600:.1f}h")


# Instance globale du cache
_global_cache = DataCache()

def get_cached_data(data_source="odre", location="France", force_reload=False):
    """Interface simple pour obtenir des données cachées."""
    return _global_cache.get_data(data_source, location, force_reload)

def clear_data_cache(data_source=None):
    """Interface simple pour vider le cache."""
    _global_cache.clear_cache(data_source)

def cache_info():
    """Interface simple pour les infos du cache."""
    return _global_cache.get_cache_info()


if __name__ == "__main__":
    # Test du cache
    print("Test du système de cache")
    
    # Premier chargement (devrait recharger)
    print("\n=== Premier chargement ===")
    data1 = get_cached_data("odre")
    print(f"Shape: {data1.shape}")
    
    # Deuxième chargement (devrait utiliser cache)
    print("\n=== Deuxième chargement ===")
    data2 = get_cached_data("odre")
    print(f"Shape: {data2.shape}")
    
    # Vérifier que c'est identique
    print(f"Données identiques: {data1.equals(data2)}")
    
    # Info cache
    print("\n=== Info Cache ===")
    cache_info()
