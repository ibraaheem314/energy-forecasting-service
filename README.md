# Energy Forecasting Service

Service de prévision de consommation énergétique utilisant des techniques de Machine Learning.

## Structure du Projet

```
energy-forecasting-service/
├── README.md              # Documentation principale
├── requirements.txt       # Dépendances (versionnées proprement)
├── .gitignore            # Fichiers à ignorer par Git
├── Makefile              # Commandes automatisées
├── pyproject.toml        # Configuration du projet Python
│
├── data/                 # Jeux de données
│   ├── raw/             # Données brutes / externes
│   ├── processed/       # Données nettoyées prêtes à l'usage
│   └── cache/           # Cache des données traitées
│
├── notebooks/           # Exploration, prototypes
│   └── 01_exploration_donnees.ipynb
│
├── src/                 # Code source principal
│   ├── __init__.py
│   ├── features.py      # Extraction de features (lags, rolling, calendaires)
│   ├── models.py        # SARIMAX, LightGBM, quantile, expectile
│   ├── evaluation.py    # RMSE, MAPE, Pinball Loss, CRPS, Coverage
│   ├── fairness.py      # Couverture par sous-groupes (fairness metrics)
│   └── utils.py         # Fonctions utilitaires génériques
│
├── app/                 # Application légère
│   ├── api.py          # FastAPI (endpoints /forecast etc.)
│   └── dashboard.py    # Streamlit (visualisation résultats, fairness)
│
├── models/              # Modèles entraînés sauvegardés
│
└── tests/               # Tests unitaires simples
    ├── test_features.py
    ├── test_models.py
    └── test_api.py
```

## Installation

1. **Cloner le repository**
   ```bash
   git clone <repository-url>
   cd energy-forecasting-service
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration optionnelle**
   ```bash
   # Pour utiliser les données RTE ODRÉ au lieu des données synthétiques
   export DATA_SOURCE=odre
   ```

## Utilisation

### Démarrage rapide

1. **Lancer l'API**
   ```bash
   make run
   # ou directement :
   python -c "from app.api import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8000)"
   ```

2. **Lancer le dashboard**
   ```bash
   make dashboard
   # ou directement :
   streamlit run app/dashboard.py
   ```

3. **Tester l'API**
   ```bash
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/forecast -H "Content-Type: application/json" -d '{"horizon": 24, "city": "Paris", "with_intervals": true}'
   ```

### Commandes Makefile

- `make help` - Afficher l'aide
- `make install` - Installer les dépendances
- `make test` - Lancer les tests
- `make lint` - Vérifier le code
- `make run` - Lancer l'API
- `make dashboard` - Lancer le dashboard
- `make train` - Entraîner un modèle simple
- `make clean` - Nettoyer les fichiers temporaires

## Fonctionnalités

### API FastAPI

L'API expose les endpoints suivants :

- `GET /` - Information sur l'API
- `GET /health` - Statut de santé
- `POST /forecast` - Prévisions énergétiques

Exemple de requête :
```json
{
  "horizon": 24,
  "city": "Paris",
  "with_intervals": true
}
```

### Dashboard Streamlit

Interface web interactive pour :
- Configurer les paramètres de prévision
- Visualiser les résultats
- Analyser les intervalles de confiance

### Modèles Disponibles

- **Linear Regression** - Modèle de base rapide
- **Random Forest** - Modèle d'ensemble robuste
- **LightGBM** - Gradient boosting efficace
- **Gradient Boosting Quantile** - Prédictions avec intervalles de confiance

### Engineering des Features

- **Features temporelles** : heure, jour, mois, saison
- **Features cycliques** : sin/cos pour capturer la cyclicité
- **Features de lag** : valeurs passées (1h, 2h, 3h, etc.)
- **Features de rolling** : moyennes mobiles, écarts-types
- **Features booléennes** : weekend, heures de pointe, etc.

### Métriques d'Évaluation

- **Prédictions ponctuelles** : MAE, RMSE, R², MAPE
- **Prédictions quantiles** : Pinball Loss, Coverage
- **Fairness** : Métriques par sous-groupes temporels

## Développement

### Tests

```bash
# Tous les tests
make test

# Tests spécifiques
python -m pytest tests/test_api.py -v
python -m pytest tests/test_models.py -v
python -m pytest tests/test_features.py -v
```

### Linting et Formatage

```bash
# Vérifier le code
make lint

# Formater le code
make format
```

### Ajout de Nouveaux Modèles

1. Créer une nouvelle classe dans `src/models.py` qui hérite de `BaseModel`
2. Implémenter les méthodes `fit`, `predict`, `save`, `load`
3. Ajouter le modèle dans la fonction `create_model`
4. Ajouter des tests dans `tests/test_models.py`

## Configuration

### Variables d'Environnement

- `DATA_SOURCE` - Source des données (`synthetic` ou `odre`)
- `API_HOST` - Host de l'API (défaut: `127.0.0.1`)
- `API_PORT` - Port de l'API (défaut: `8000`)

### Données

Le service peut utiliser :
- **Données synthétiques** - Pour le développement et les tests
- **Données RTE ODRÉ** - Données réelles de consommation française

## Performance

- **API** : ~100-200ms par prédiction
- **Entraînement** : 1-5 minutes selon le modèle
- **Cache** : Réduction de 90% du temps de chargement des données

## Troubleshooting

### Problèmes Courants

1. **ModuleNotFoundError** - Vérifier que les dépendances sont installées
2. **API ne répond pas** - Vérifier que le port n'est pas utilisé
3. **Données manquantes** - Vérifier la configuration `DATA_SOURCE`

### Debug

```bash
# Vérifier la structure
make structure

# Nettoyer les caches
make clean

# Logs détaillés
python -c "from src.utils import setup_logging; setup_logging('DEBUG')"
```

## Contribution

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.