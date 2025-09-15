"""Utilitaire pour gérer le cache des données."""
import argparse
import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from scripts.data_cache import cache_info, clear_data_cache, get_cached_data


def main():
    """Interface en ligne de commande pour gérer le cache."""
    parser = argparse.ArgumentParser(description="Gestionnaire de cache des données")
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande info
    info_parser = subparsers.add_parser('info', help='Afficher les informations du cache')
    
    # Commande clear
    clear_parser = subparsers.add_parser('clear', help='Vider le cache')
    clear_parser.add_argument('--source', choices=['odre', 'synthetic'], 
                             help='Source spécifique à vider (par défaut: tout)')
    
    # Commande preload
    preload_parser = subparsers.add_parser('preload', help='Précharger les données en cache')
    preload_parser.add_argument('--source', choices=['odre', 'synthetic'], 
                               default='odre', help='Source à précharger')
    preload_parser.add_argument('--force', action='store_true', 
                               help='Forcer le rechargement même si cache valide')
    
    args = parser.parse_args()
    
    if args.command == 'info':
        print("Informations du cache:")
        cache_info()
        
    elif args.command == 'clear':
        print(f"Nettoyage du cache...")
        clear_data_cache(args.source)
        print("Cache nettoyé")
        
    elif args.command == 'preload':
        print(f"Préchargement des données {args.source}...")
        data = get_cached_data(args.source, force_reload=args.force)
        print(f"Données préchargées: {len(data)} échantillons")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
