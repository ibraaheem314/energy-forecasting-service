
# Energy Forecasting - Data Science Project (Option B: local sans Docker)
.PHONY: help install install-dev test run dashboard fetch-data train evaluate train-models evaluate-models jupyter clean lint

PYTHON ?= python
VENV   ?= .venv
ACTIVATE = . $(VENV)/bin/activate

help: ## Show this help message
	@echo "Energy Forecasting - Data Science Project"
	@echo "========================================"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install runtime dependencies
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install -U pip
	@if [ -f requirements.txt ]; then \
		$(ACTIVATE) && pip install -r requirements.txt ; \
	else \
		$(ACTIVATE) && pip install -e . ; \
	fi

install-dev: install ## Install dev tools
	$(ACTIVATE) && pip install -U ruff black pytest pytest-cov streamlit jupyterlab

test: ## Run tests with coverage
	$(ACTIVATE) && pytest tests/ --cov=app --cov-report=term-missing --cov-report=html

lint: ## Lint & format check
	$(ACTIVATE) && ruff check . && black --check .

fetch-data: ## Fetch/prepare data (synthetic or ODRÉ if configured)
	@if [ -f scripts/fetch_data.py ]; then \
		$(ACTIVATE) && $(PYTHON) scripts/fetch_data.py ; \
	elif [ -f jobs/fetch_data.py ]; then \
		$(ACTIVATE) && $(PYTHON) jobs/fetch_data.py ; \
	else \
		echo "No fetch_data.py found in scripts/ or jobs/"; exit 1; \
	fi

train: ## Train models
	@if [ -f scripts/train_models.py ]; then \
		$(ACTIVATE) && $(PYTHON) scripts/train_models.py ; \
	elif [ -f scripts/train.py ]; then \
		$(ACTIVATE) && $(PYTHON) scripts/train.py ; \
	else \
		echo "No training script found (scripts/train_models.py or scripts/train.py)"; exit 1; \
	fi

evaluate: ## Evaluate models (backtests RMSE/MAPE)
	@if [ -f scripts/evaluate_models.py ]; then \
		$(ACTIVATE) && $(PYTHON) scripts/evaluate_models.py ; \
	elif [ -f scripts/evaluate.py ]; then \
		$(ACTIVATE) && $(PYTHON) scripts/evaluate.py ; \
	else \
		echo "No evaluation script found (scripts/evaluate_models.py or scripts/evaluate.py)"; exit 1; \
	fi

# Backward-compat aliases
train-models: train        ## Alias of 'train'
evaluate-models: evaluate  ## Alias of 'evaluate'

run: ## Run the API server
	$(ACTIVATE) && uvicorn app.api.main:app --host $${API_HOST:-127.0.0.1} --port $${API_PORT:-8000} --reload

dashboard: ## Run the Streamlit dashboard
	$(ACTIVATE) && streamlit run dashboard/app.py --server.port $${DASHBOARD_PORT:-8501}

jupyter: ## Start JupyterLab (optional)
	$(ACTIVATE) && jupyter lab

clean: ## Clean build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

# === MODÈLES INDIVIDUELS ===
train-linear: ## Train Linear Regression model
	$(ACTIVATE) && $(PYTHON) scripts/models/linear_regression.py

train-rf: ## Train Random Forest model
	$(ACTIVATE) && $(PYTHON) scripts/models/random_forest.py

train-lgb: ## Train LightGBM model
	$(ACTIVATE) && $(PYTHON) scripts/models/lightgbm_model.py

train-gbq: ## Train Gradient Boosting Quantile model
	$(ACTIVATE) && $(PYTHON) scripts/models/gradient_boosting_quantile.py

train-all: ## Train all models with optimized data loading
	$(ACTIVATE) && $(PYTHON) scripts/run_all_models.py --data odre

# === CACHE MANAGEMENT ===
cache-info: ## Show cache information
	$(ACTIVATE) && $(PYTHON) scripts/manage_cache.py info

cache-clear: ## Clear all cache
	$(ACTIVATE) && $(PYTHON) scripts/manage_cache.py clear

cache-preload: ## Preload ODRÉ data to cache
	$(ACTIVATE) && $(PYTHON) scripts/manage_cache.py preload --source odre
