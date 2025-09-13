# Energy Forecasting - Data Science Project (local sans Docker)
.PHONY: help install install-dev test run dashboard fetch-data train evaluate train-models evaluate-models jupyter clean

# -------- Config --------
PYTHON ?= python
VENV   ?= .venv

# Helpers
ACTIVATE = . $(VENV)/bin/activate

# -------- Help --------
help: ## Show this help message
	@echo "Energy Forecasting - Data Science Project"
	@echo "========================================"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# -------- Install --------
install: ## Create venv and install runtime dependencies
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install -U pip
	# If requirements.txt exists use it, else use editable install (pyproject)
	@if [ -f requirements.txt ]; then \
		$(ACTIVATE) && pip install -r requirements.txt ; \
	else \
		$(ACTIVATE) && pip install -e . ; \
	fi

install-dev: install ## Install dev tools (ruff, black, pytest, pytest-cov, streamlit, jupyterlab)
	$(ACTIVATE) && pip install -U ruff black pytest pytest-cov streamlit jupyterlab

# -------- Testing --------
test: ## Run tests with coverage
	$(ACTIVATE) && pytest tests/ --cov=app --cov-report=term-missing --cov-report=html

# -------- Data & Modeling --------
fetch-data: ## Fetch/prepare data (tries scripts/, falls back to jobs/)
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

# -------- Run services --------
run: ## Run the API server
	$(ACTIVATE) && uvicorn app.api.main:app --host $${API_HOST:-127.0.0.1} --port $${API_PORT:-8000} --reload

dashboard: ## Run the Streamlit dashboard
	$(ACTIVATE) && streamlit run dashboard/app.py --server.port $${DASHBOARD_PORT:-8501}

# -------- Jupyter (optional) --------
jupyter: ## Start JupyterLab (optional)
	$(ACTIVATE) && jupyter lab

# -------- Clean --------
clean: ## Clean build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/
