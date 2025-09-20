# Notes de Migration - Nouvelle Structure

## Résumé des Changements

Le projet a été entièrement restructuré selon une architecture plus académique et professionnelle.

### Ancienne Structure → Nouvelle Structure

```
AVANT:                          APRÈS:
app/                           app/
├── api/main.py               ├── api.py
├── services/                 └── dashboard.py
│   ├── features.py           
│   ├── loader.py             src/
│   └── models.py             ├── features.py
dashboard/app.py              ├── models.py
scripts/                      ├── evaluation.py
├── train_simple.py           ├── fairness.py
├── evaluate_simple.py        └── utils.py
├── models/
└── data_cache.py             data/
                              ├── raw/
                              ├── processed/
                              └── cache/
```

## Fichiers Supprimés

- `scripts/train_simple.py` - Remplacé par `src/models.py`
- `scripts/evaluate_simple.py` - Remplacé par `src/evaluation.py`
- `scripts/models/` - Logique intégrée dans `src/models.py`
- `scripts/data_cache.py` - Fonctionnalité dans `src/utils.py`
- `app/services/` - Migré vers `src/`
- `app/api/main.py` - Remplacé par `app/api.py`
- `dashboard/` - Intégré dans `app/dashboard.py`
- Rapports temporaires (PROJECT_COHERENCE_REPORT.md, etc.)

## Nouveaux Fichiers

- `src/` - Package principal avec toute la logique métier
- `requirements.txt` - Dépendances extraites de pyproject.toml
- `data/raw/` et `data/processed/` - Séparation des données
- `.gitignore` - Fichiers à ignorer
- `MIGRATION_NOTES.md` - Ce fichier

## Changements Fonctionnels

### API (`app/api.py`)
- Endpoint principal consolidé
- Import simplifié depuis `src/`
- Gestion d'erreur améliorée
- Support pour différents types de modèles

### Modèles (`src/models.py`)
- Classe `BaseModel` unifiée
- Factory pattern pour création de modèles
- Support des modèles quantiles
- Sauvegarde/chargement standardisés

### Features (`src/features.py`)
- Pipeline de features modulaire
- Gestion robuste des valeurs manquantes
- Features temporelles et cycliques
- Lags et rolling windows configurables

### Évaluation (`src/evaluation.py`)
- Métriques complètes (MAE, RMSE, R², MAPE)
- Support des prédictions quantiles
- Pinball loss et coverage
- Comparaison de modèles

### Fairness (`src/fairness.py`)
- Analyse par sous-groupes temporels
- Métriques d'équité
- Couverture des intervalles
- Variance inter-groupes

### Utilitaires (`src/utils.py`)
- Gestion des fichiers et cache
- Configuration centralisée
- Logging et debugging
- Helpers pour données temporelles

## Tests Mis à Jour

- `tests/test_api.py` - Adapté à la nouvelle API
- `tests/test_models.py` - Tests pour tous les modèles
- `tests/test_features.py` - Tests du pipeline de features

## Makefile Simplifié

Nouvelles commandes :
- `make run` - Lancer l'API
- `make dashboard` - Lancer Streamlit
- `make train` - Entraîner un modèle simple
- `make test` - Tous les tests
- `make lint` - Vérification code

## Migration des Données

- Cache existant dans `data/cache/` préservé
- Modèles entraînés dans `models/` conservés
- Notebooks dans `notebooks/` maintenus

## Compatibilité

✅ **Conservé :**
- API endpoints (`/health`, `/forecast`)
- Format des prédictions
- Métriques d'évaluation
- Cache des données

❌ **Changé :**
- Imports Python (maintenant depuis `src/`)
- Structure des fichiers
- Scripts d'entraînement (maintenant via `src/models.py`)

## Avantages de la Nouvelle Structure

1. **Académique** - Structure standard pour projets ML
2. **Modulaire** - Séparation claire des responsabilités
3. **Testable** - Meilleure couverture de tests
4. **Scalable** - Plus facile d'ajouter de nouveaux modèles
5. **Documenté** - README complet et structure claire
6. **Standard** - Suit les conventions Python/ML

## Tests de Validation

Tous les tests passent :
```
============================= 10 passed in 10.87s =============================
```

API fonctionnelle :
```
GET /health → 200 OK
POST /forecast → 200 OK
```

Modèles opérationnels :
- Linear Regression ✅
- Random Forest ✅
- LightGBM ✅
- Gradient Boosting Quantile ✅
