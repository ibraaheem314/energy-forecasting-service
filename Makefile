# Energy Forecasting Service Makefile
.PHONY: help install install-dev test lint format clean build run docker-build docker-up docker-down deploy

# Default target
help: ## Show this help message
	@echo "Energy Forecasting Service"
	@echo "=========================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development setup
install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

# Code quality
lint: ## Run linting tools
	black --check app/ tests/ jobs/ dashboard/
	isort --check-only app/ tests/ jobs/ dashboard/
	flake8 app/ tests/ jobs/ dashboard/
	mypy app/

format: ## Format code with black and isort
	black app/ tests/ jobs/ dashboard/
	isort app/ tests/ jobs/ dashboard/

# Testing
test: ## Run all tests
	pytest tests/ --cov=app --cov-report=html --cov-report=term

test-fast: ## Run tests without coverage
	pytest tests/ -x -v

test-integration: ## Run integration tests only
	pytest tests/ -m integration

# Data and jobs
fetch-data: ## Fetch fresh training data
	python jobs/fetch_data.py --start-date $(shell date -d '30 days ago' +%Y-%m-%d) --end-date $(shell date +%Y-%m-%d)

train-models: ## Train all models
	python jobs/retrain.py --model-types random_forest gradient_boosting --auto-deploy

backtest: ## Run backtesting
	python jobs/backtest.py --start-date $(shell date -d '7 days ago' +%Y-%m-%d)

# API and dashboard
run-api: ## Run the API server
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard: ## Run the Streamlit dashboard
	streamlit run dashboard/app.py --server.port 8501

# Docker operations
docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start all services with Docker Compose
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

docker-logs: ## Show logs from all services
	docker-compose logs -f

docker-clean: ## Clean up Docker resources
	docker-compose down -v
	docker system prune -f

# MLflow
mlflow-ui: ## Start MLflow UI
	mlflow ui --host 0.0.0.0 --port 5000

# Database operations
db-migrate: ## Run database migrations
	alembic upgrade head

db-reset: ## Reset database (WARNING: destructive)
	docker-compose exec db psql -U postgres -d energy_forecasting -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

# Monitoring
prometheus: ## Start Prometheus monitoring
	docker run -d -p 9090:9090 -v $(PWD)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

grafana: ## Start Grafana dashboard
	docker run -d -p 3000:3000 grafana/grafana

# Deployment
deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging..."
	# Add your staging deployment commands here

deploy-prod: ## Deploy to production environment
	@echo "Deploying to production..."
	# Add your production deployment commands here

# Performance testing
benchmark: ## Run performance benchmarks
	python -m pytest tests/ -m benchmark

load-test: ## Run load testing
	locust -f tests/load_test.py --host=http://localhost:8000

# Data quality
validate-data: ## Validate data quality
	python -c "from app.services.loader import DataLoader; import asyncio; asyncio.run(DataLoader().validate_data_quality())"

# Security
security-scan: ## Run security scans
	bandit -r app/ -f json -o security-report.json
	safety check

# Documentation
docs: ## Generate documentation
	sphinx-build -b html docs/ docs/_build/html

docs-serve: ## Serve documentation locally
	python -m http.server 8080 -d docs/_build/html

# Clean up
clean: ## Clean up build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Build and package
build: ## Build package
	python -m build

release: ## Create a new release
	@echo "Creating release..."
	python -m build
	twine check dist/*
	@echo "Release built successfully!"

# Environment setup
setup-env: ## Set up environment variables
	cp .env.example .env
	@echo "Please edit .env file with your configuration"

# Quick start
quick-start: install-dev setup-env docker-up ## Quick start for development
	@echo "🚀 Energy Forecasting Service is starting up!"
	@echo "📊 Dashboard: http://localhost:8501"
	@echo "🔌 API: http://localhost:8000"
	@echo "📈 MLflow: http://localhost:5000"
	@echo "💾 Database: localhost:5432"

# Production deployment
production-deploy: test lint security-scan build ## Full production deployment pipeline
	@echo "🏭 Running production deployment pipeline..."
	docker-compose -f docker-compose.prod.yml up -d

# Development workflow
dev-workflow: format lint test ## Complete development workflow
	@echo "✅ Development workflow completed successfully!"

# Check system requirements
check-requirements: ## Check if system requirements are met
	@python --version
	@docker --version
	@docker-compose --version
	@echo "✅ System requirements check passed"
