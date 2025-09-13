# ⚡ Energy Forecasting - Projet Data Science

**Prévision de consommation énergétique avec Machine Learning (exécution locale sans Docker)**

> Projet exécutable **en local** avec `venv` + `Makefile`.
> Pas de Docker, pas de MLflow/Postgres/Grafana/Prometheus/Kubernetes dans cette version.

## Table des matières

* [ Objectifs](#-objectifs)
* [ Structure du projet](#-structure-du-projet)
* [ Démarrage rapide](#-démarrage-rapide)
* [ Données](#-données)
* [ Modèles](#-modèles)
* [ Utilisation API](#-utilisation-api)
* [ Dashboard](#-dashboard)
* [ Tests](#-tests)
* [ Livrables Portfolio](#-livrables-portfolio)
* [ Contribution](#-contribution)
* [ License](#-license)
* [ Contact](#-contact)

---

## Objectifs

Implémenter un système de **prévision de la consommation énergétique** sur les prochains jours, avec :

* **Feature engineering** pour séries temporelles
* **Modélisation** (baseline/SARIMAX/LightGBM)
* **Évaluation** (RMSE/MAPE)
* **API FastAPI** pour servir les prédictions
* **Dashboard Streamlit** pour visualiser les résultats

---

## Structure du projet

```
energy-forecasting-service/
├─ app/
│  ├─ api/                 # API FastAPI (endpoints /health, /forecast)
│  │  ├─ main.py
│  │  └─ schemas.py
│  ├─ services/            # Services data/ML
│  │  ├─ loader.py         # Données (synthétiques par défaut ou ODRÉ)
│  │  ├─ features.py       # Lags, rolling, calendaires
│  │  └─ models.py         # Baseline, SARIMAX, LGBM
│  └─ config.py
├─ dashboard/
│  └─ app.py               # Streamlit (consomme l'API)
├─ scripts/
│  ├─ fetch_data.py        # Récupération/formatage
│  ├─ train_models.py      # Entraînement
│  └─ evaluate_models.py   # Backtests & métriques
├─ tests/
├─ data/                   # Données locales (gitignored)
├─ models/                 # Artefacts modèles (gitignored)
├─ .env.example
├─ Makefile
├─ requirements.txt
└─ README.md
```

> Les **notebooks** sont optionnels (uniquement pour explorations rapides). Ils ne sont pas nécessaires pour exécuter le projet.

---

## Démarrage rapide

### Prérequis

* **Python 3.11+**
* **Git**
* macOS / Linux / Windows (PowerShell)

### Installation

```bash
# 1) Cloner
git clone https://github.com/ibraaheem314/energy-forecasting-service.git
cd energy-forecasting-service

# 2) Variables d'environnement
cp .env.example .env

# 3) Installer (crée .venv + installe requirements)
make install
# (optionnel) outils dev: pytest, ruff, black
make install-dev
```

### Lancer l’API

```bash
make run
# Swagger: http://127.0.0.1:8000/docs
```

### Lancer le Dashboard

```bash
make dashboard
# http://127.0.0.1:8501
```

> Sur Windows PowerShell, active le venv si besoin : `.\.venv\Scripts\Activate.ps1`

---

## Données

* **Par défaut** : `app/services/loader.py` génère **des données synthétiques** pour tester l’API et le dashboard immédiatement.
* **Option recommandée (Open Data)** : brancher **ODRÉ (OpenDataSoft / RTE Open Data)** dans `loader.py` pour récupérer de la conso réelle sans OAuth (plus simple).
* **Option avancée (plus tard)** : **RTE iservices** (OAuth2/client secret) si tu veux des APIs nécessitant authentification.

Configuration minimale (`.env`) :

```ini
API_HOST=127.0.0.1
API_PORT=8000
DASHBOARD_PORT=8501
DATA_DIR=./data
CITY=Paris
TIMEZONE=Europe/Paris
```

---

## Modèles

* **Baselines** : persistance (y\[t] = y\[t-168]), moyennes mobiles 24/168h.
* **SARIMAX** : exogènes calendaires/météo si disponibles.
* **LightGBM** : lags (1, 24, 168), rolling (mean\_24, mean\_168), variables calendaires.
* **Métriques** : **RMSE** (principale), **MAPE** (secondaire).
* **Sélection** : promotion du meilleur modèle “prod” (flag simple interne ; pas de MLflow dans cette option).

### Commandes utiles

```bash
make fetch-data   # récupère/prepare les données (synthétiques ou ODRÉ si configuré)
make train        # entraîne les modèles
make evaluate     # exécute backtests RMSE/MAPE
```

---

## Utilisation API

### Endpoints

* **Santé** : `GET /health` → `{"status": "ok"}`
* **Prévision** : `POST /forecast`

**Exemple de requête** :

```json
{
  "horizon": 168,
  "city": "Paris",
  "with_intervals": true
}
```

**Exemple de réponse (extrait)** :

```json
{
  "timestamps": ["2025-09-14T00:00:00Z", "..."],
  "yhat": [31245.1, "..."],
  "yhat_lower": [29800.5, "..."],
  "yhat_upper": [32790.2, "..."],
  "model_name": "lightgbm",
  "model_version": "1.0.0"
}
```

**cURL** :

```bash
curl -X POST "http://127.0.0.1:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{"horizon":168,"city":"Paris","with_intervals":true}'
```

---

## Dashboard

Le dashboard Streamlit consomme l’API `/forecast` et propose :

* **Forecasts** : prédictions 7 jours + intervalles
* **Model Performance** : RMSE/MAPE des backtests
* **Historical Data** : exploration des historiques

Lancer :

```bash
make dashboard
# http://127.0.0.1:8501
```

---

## Tests

```bash
make test      # tests unitaires
make lint      # ruff + black --check
```

**Couverture visée** : ≥ 80% sur la logique de features et endpoints principaux.

---

## Livrables Portfolio

* **API locale** : endpoint `/forecast` documenté (Swagger)
* **Dashboard** : visualisations claires des prédictions
* **Rapport comparatif** (README/notes) : RMSE/MAPE, choix du modèle, limites & next steps
* **Code propre** : Makefile, tests unitaires, structure claire

---

## Contribution

Contributions bienvenues ! Ouvre une **issue** ou une **PR**.

---

## 📄 License

MIT License — voir [LICENSE](LICENSE).

---

## Contact

* **Issues** : [https://github.com/ibraaheem314/energy-forecasting-service/issues](https://github.com/ibraaheem314/energy-forecasting-service/issues)
* **Discussions** : [https://github.com/ibraaheem314/energy-forecasting-service/discussions](https://github.com/ibraaheem314/energy-forecasting-service/discussions)

---

**⚡ Développé pour apprendre et démontrer une mise en production simple (sans Docker) ⚡**
