# ⚡ Energy Forecasting - Projet Data Science

**Prévision de consommation énergétique avec Machine Learning**

> Projet junior Data Science pour la prévision de consommation énergétique utilisant les données RTE et des techniques de ML.

## 📋 Table des matières

- [🎯 Objectifs](#-objectifs)
- [🔧 Structure du projet](#-structure-du-projet)
- [🚀 Démarrage rapide](#-démarrage-rapide)
- [📊 Données](#-données)
- [🤖 Modèles](#-modèles)
- [📈 Résultats](#-résultats)
- [🧪 Utilisation](#-utilisation)

## 🎯 Objectifs

Ce projet Data Science implémente un système de **prévision de consommation énergétique** pour anticiper la demande sur les prochains jours.

**Objectifs pédagogiques :**
- **Analyse exploratoire** des données énergétiques
- **Feature engineering** pour les séries temporelles
- **Modélisation ML** avec différents algorithmes
- **Évaluation** et comparaison des performances
- **API simple** pour servir les prédictions
- **Dashboard** de visualisation des résultats

## 🔧 Structure du projet

```
energy-forecasting/
├─ app/                          # Application
│  ├─ api/                       # API FastAPI
│  │  ├─ main.py                 # Endpoints
│  │  └─ schemas.py              # Modèles de données
│  ├─ services/                  # Services ML
│  │  ├─ loader.py               # Chargement données
│  │  ├─ features.py             # Feature engineering
│  │  └─ models.py               # Modèles ML
│  └─ config.py                  # Configuration
├─ scripts/                      # Scripts d'entraînement
│  ├─ train_models.py            # Entraînement
│  └─ evaluate_models.py         # Évaluation
├─ notebooks/                    # Notebooks Jupyter
│  └─ 01_exploration_donnees.ipynb
├─ dashboard/                    # Dashboard Streamlit
│  └─ app.py                     # Visualisations
├─ jobs/                         # Jobs de données
│  └─ fetch_data.py              # Récupération
├─ tests/                        # Tests
├─ data/                         # Données (gitignored)
├─ models/                       # Modèles sauvegardés
└─ README.md
```

## 🚀 Démarrage rapide

### Prérequis

- Python 3.9+
- Docker & Docker Compose
- Git

### Installation

1. **Cloner le repository**
```bash
git clone https://github.com/your-username/energy-forecasting-service.git
cd energy-forecasting-service
```

2. **Démarrage avec Docker (recommandé)**
```bash
# Démarrage complet avec une seule commande
make quick-start

# Ou manuellement
docker-compose up -d
```

3. **Installation locale pour développement**
```bash
# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
make install-dev

# Configurer l'environnement
make setup-env
# Éditer le fichier .env avec vos paramètres
```

### Services disponibles

Après le démarrage, les services suivants sont accessibles :

- 📊 **Dashboard** : http://localhost:8501
- 🔌 **API** : http://localhost:8000 (docs: /docs)
- 📈 **MLflow** : http://localhost:5000
- 💾 **Database** : localhost:5432
- 📉 **Grafana** : http://localhost:3000
- 🔍 **Prometheus** : http://localhost:9090

## 📊 Sources de données

### Données RTE (Réseau de Transport d'Électricité)

Le service utilise les APIs ouvertes RTE pour récupérer :

- **Consommation réalisée** par région
- **Production par filière** (nucléaire, éolien, solaire, etc.)
- **Données de marché** (prix spot, capacités)
- **Prévisions météorologiques** impactant la consommation

```bash
# Configuration RTE dans .env
RTE_API_KEY=your-rte-api-key
RTE_API_BASE_URL=https://digital.iservices.rte-france.com
```

### Sources complémentaires

- **Météo** : OpenWeatherMap API
- **Jours fériés** : Algorithme de calcul français
- **Indicateurs économiques** : Intégration optionnelle

### Récupération des données

```bash
# Récupération manuelle
make fetch-data

# Récupération automatique (configurée dans docker-compose)
# Toutes les 6 heures via le scheduler
```

## 🤖 Modèles implémentés

### Modèles de séries temporelles

1. **ARIMA** : Modèle statistique classique
   - Auto-détection des paramètres (p,d,q)
   - Gestion de la saisonnalité

2. **Prophet** : Développé par Facebook
   - Robuste aux valeurs manquantes
   - Modélisation des jours fériés
   - Tendances non-linéaires

3. **Ensemble Methods**
   - Random Forest Regressor
   - Gradient Boosting
   - XGBoost avec optimisation bayésienne

4. **Deep Learning** (optionnel)
   - LSTM networks
   - Transformer architecture
   - Conv1D pour patterns locaux

### Feature Engineering

- **Features temporelles** : heure, jour, mois, saison
- **Features cycliques** : encodage sinusoïdal
- **Lags** : valeurs historiques (1h à 7 jours)
- **Rolling statistics** : moyennes mobiles, écarts-types
- **Features météo** : température, humidité, vent
- **Features calendaires** : weekends, jours fériés

### Entraînement et évaluation

```bash
# Entraînement de tous les modèles
make train-models

# Backtest sur données historiques
make backtest

# Évaluation comparée
python -m app.services.evaluate compare-models
```

## 📈 Dashboard et visualisations

Le dashboard Streamlit fournit :

### 📊 Vues principales

1. **Forecasts** : Prédictions avec intervalles de confiance
2. **Model Performance** : Comparaison des métriques (MAE, MAPE, R²)
3. **Historical Data** : Analyse des données historiques
4. **Settings** : Gestion des modèles et configuration

### 🎯 Métriques business

- **Précision de prévision** : MAPE < 5% sur 24h
- **Fiabilité** : R² > 0.90 pour les modèles principaux
- **Latence** : Prédictions < 1s pour 24h ahead
- **Couverture** : Intervalles de confiance à 95%

### 🖼️ Visualisations

- **Séries temporelles** interactives (Plotly)
- **Heatmaps** de corrélation
- **Distributions** des erreurs de prédiction
- **Feature importance** par modèle
- **Monitoring** temps réel des performances

## 🔄 Pipeline MLOps

### Workflow automatisé

1. **Data Pipeline**
   ```
   Fetch Data → Validation → Feature Engineering → Storage
   ```

2. **Model Pipeline**
   ```
   Training → Validation → Registry → A/B Testing → Deployment
   ```

3. **Monitoring Pipeline**
   ```
   Predictions → Metrics → Alerts → Retraining Triggers
   ```

### CI/CD avec GitHub Actions

- **Tests automatisés** : Unitaires, intégration, performance
- **Quality Gates** : Linting, sécurité, couverture
- **Déploiement** : Staging automatique, production manuelle
- **Monitoring** : Alertes Slack, métriques Prometheus

### MLflow Integration

- **Experiment Tracking** : Hyperparamètres, métriques, artifacts
- **Model Registry** : Versioning, staging, production
- **Model Serving** : Déploiement via REST API
- **Lineage** : Traçabilité complète des modèles

## 📚 Documentation

### API Documentation

- **FastAPI Docs** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc
- **OpenAPI Spec** : http://localhost:8000/openapi.json

### Code Documentation

```bash
# Générer la documentation
make docs

# Servir localement
make docs-serve  # http://localhost:8080
```

### Exemples d'usage

```python
# Client API Python
import requests

# Prédiction sur 24h
response = requests.post("http://localhost:8000/forecast", json={
    "location": "region_1",
    "start_time": "2024-01-15T00:00:00",
    "end_time": "2024-01-16T00:00:00",
    "forecast_type": "consumption",
    "confidence_interval": True
})

predictions = response.json()["predictions"]
```

```bash
# Client cURL
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "region_1",
    "start_time": "2024-01-15T00:00:00",
    "end_time": "2024-01-16T00:00:00",
    "forecast_type": "consumption"
  }'
```

## 🧪 Tests

### Types de tests

```bash
# Tests unitaires
make test

# Tests rapides (sans couverture)
make test-fast

# Tests d'intégration
make test-integration

# Tests de performance
make benchmark

# Tests de charge
make load-test
```

### Couverture de code

- **Cible** : >80% de couverture
- **Reports** : HTML, XML, terminal
- **CI/CD** : Blocage si < 80%

### Stratégie de tests

- **Unit Tests** : Logique métier, feature engineering
- **Integration Tests** : API endpoints, database
- **E2E Tests** : Workflow complet forecast
- **Performance Tests** : Latence, throughput
- **Security Tests** : Vulnérabilités, injections

## 🚢 Déploiement

### Environnements

1. **Development** : Docker Compose local
2. **Staging** : Kubernetes (auto-deploy depuis `develop`)
3. **Production** : Kubernetes (deploy manuel depuis `main`)

### Configuration par environnement

```bash
# Development
cp .env.example .env.dev
# Edit with dev settings

# Production
cp .env.example .env.prod
# Edit with production settings
```

### Commandes de déploiement

```bash
# Staging
make deploy-staging

# Production
make deploy-prod

# Rollback
kubectl rollout undo deployment/energy-forecasting-api
```

### Health Checks

- **API** : `/health` endpoint
- **Database** : Connection pooling
- **Redis** : Cache availability
- **MLflow** : Model registry access

## 🏆 Livrables Portfolio

Ce projet fournit des livrables concrets pour votre CV/Portfolio :

### 📊 Rapport Comparatif des Modèles

- **Benchmark** sur données RTE réelles
- **Métriques** : MAE, MAPE, R², directional accuracy
- **Analyse** des forces/faiblesses par modèle
- **Recommandations** business

### 📈 Dashboard Interactif

- **Démo live** : Prédictions temps réel
- **Visualisations** professionnelles
- **UX/UI** moderne avec Streamlit
- **Responsive** design

### 🔧 Architecture Technique

- **Microservices** avec Docker
- **API REST** documentée
- **CI/CD** avec GitHub Actions
- **MLOps** avec MLflow

### 💼 Use Case Business

- **Optimisation** de la production énergétique
- **Anticipation** des pics de consommation
- **Gestion** des risques pour assureurs
- **Smart Grid** applications

### 📝 Documentation Technique

- **README** complet avec guides
- **Architecture** diagrammes
- **API** documentation
- **Deployment** guides

## 🤝 Contribution

Contributions welcomes ! Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour les guidelines.

## 📄 License

MIT License - voir [LICENSE](LICENSE) pour détails.

## 📞 Contact

- **Email** : team@energy-forecasting.com
- **Issues** : [GitHub Issues](https://github.com/your-username/energy-forecasting-service/issues)
- **Discussions** : [GitHub Discussions](https://github.com/your-username/energy-forecasting-service/discussions)

---

**⚡ Développé avec passion pour l'énergie et l'IA ⚡**

*Ce projet constitue un portfolio complet pour démontrer vos compétences en Data Science, MLOps et développement d'applications ML en production.*
