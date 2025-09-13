# Energy Forecasting - Data Science Project
.PHONY: help install install-dev test run dashboard clean

# Default target
help: ## Show this help message
	@echo "Energy Forecasting - Data Science Project"
	@echo "========================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development setup
install: ## Install dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev,dashboard]"

# Testing
test: ## Run tests
	pytest tests/ --cov=app --cov-report=html

# Data science workflow
fetch-data: ## Fetch training data
	python jobs/fetch_data.py

train-models: ## Train models
	python scripts/train_models.py

evaluate-models: ## Evaluate models
	python scripts/evaluate_models.py

# Run services
run: ## Run the API server
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Run the Streamlit dashboard
	streamlit run dashboard/app.py --server.port 8501

# Jupyter
jupyter: ## Start Jupyter Lab
	jupyter lab

# Clean up
clean: ## Clean up build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
