# Makefile pour Energy Forecasting Project

.PHONY: help install install-dev test lint clean run dashboard train evaluate

# Configuration
PYTHON := python
PIP := pip
PORT := 8000
DASHBOARD_PORT := 8501

help: ## Afficher l'aide
	@echo "Commandes disponibles :"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Installer les dépendances
	$(PIP) install -r requirements.txt

install-dev: install ## Installer les dépendances de développement
	$(PIP) install -e .

test: ## Lancer les tests
	$(PYTHON) -m pytest tests/ -v

lint: ## Vérifier le code avec les linters
	$(PYTHON) -m black --check src/ app/ tests/
	$(PYTHON) -m isort --check-only src/ app/ tests/
	$(PYTHON) -m flake8 src/ app/ tests/

format: ## Formater le code
	$(PYTHON) -m black src/ app/ tests/
	$(PYTHON) -m isort src/ app/ tests/

run: ## Lancer l'API
	$(PYTHON) -c "from app.api import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=$(PORT))"

dashboard: ## Lancer le dashboard Streamlit
	streamlit run app/dashboard.py --server.port $(DASHBOARD_PORT)

train: ## Entraîner un modèle
	$(PYTHON) -c "from src.models import create_model; from src.features import create_features; import pandas as pd; import numpy as np; from datetime import datetime, timedelta; print('Entraînement d\\'un modèle simple...'); dates = pd.date_range(start=datetime.now()-timedelta(days=30), end=datetime.now(), freq='h'); data = pd.DataFrame({'consommation_mw': 100 + np.random.normal(0, 10, len(dates))}, index=dates); df_features = create_features(data); X = df_features[[c for c in df_features.columns if c != 'y']].fillna(0); y = df_features['y']; model = create_model('linear'); model.fit(X, y); model.save('models/linear_latest.joblib'); print('Modèle sauvegardé dans models/linear_latest.joblib')"

evaluate: ## Évaluer les modèles
	$(PYTHON) -c "from src.evaluation import ModelEvaluator; print('Évaluation des modèles...')"

clean: ## Nettoyer les fichiers temporaires
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

structure: ## Afficher la structure du projet
	@echo "Structure du projet :"
	@tree -I '__pycache__|*.pyc|*.pyo|*.egg-info|.git' . || ls -la