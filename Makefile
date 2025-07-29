.PHONY: help install install-dev test test-cov lint format type-check security clean build publish docker docker-dev docs serve-docs compliance audit
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "RLHF Audit Trail - Development Commands"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation and setup
install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev,testing,docs]"
	pre-commit install
	@echo "✅ Development environment ready!"

# Testing
test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=html --cov-report=term --cov-report=xml
	@echo "📊 Coverage report generated in htmlcov/"

test-integration: ## Run integration tests
	pytest tests/ -m integration -v

test-compliance: ## Run compliance tests
	pytest tests/ -m compliance -v
	python -m rlhf_audit_trail.compliance.verify

# Code quality
lint: ## Run linting
	ruff check src tests
	bandit -r src/

format: ## Format code
	black src tests
	ruff format src tests
	isort src tests

type-check: ## Run type checking
	mypy src

# Security
security: ## Run security checks
	bandit -r src/ -f json -o reports/bandit-report.json
	safety check --json --output reports/safety-report.json
	detect-secrets scan --baseline .secrets.baseline

audit: ## Full security audit
	@mkdir -p reports
	bandit -r src/ -f json -o reports/bandit-report.json || true
	safety check --json --output reports/safety-report.json || true
	detect-secrets scan --baseline .secrets.baseline
	@echo "🔒 Security reports generated in reports/"

# Quality gates
check-all: lint type-check security test-cov ## Run all quality checks
	@echo "✅ All quality checks passed!"

# Build and publish
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	python -m build

publish-test: build ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	twine upload dist/*

# Development environment
docker: ## Build production Docker image
	docker build -t rlhf-audit-trail:latest .

docker-dev: ## Build development Docker image
	docker build -f Dockerfile.dev -t rlhf-audit-trail:dev .

docker-run: ## Run in Docker
	docker run -it --rm -p 8501:8501 rlhf-audit-trail:latest

# Documentation
docs: ## Build documentation
	cd docs && make html
	@echo "📚 Documentation built in docs/_build/html/"

serve-docs: docs ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

# Compliance and monitoring
compliance: ## Check compliance requirements
	python -m rlhf_audit_trail.compliance.check_eu_ai_act
	python -m rlhf_audit_trail.compliance.check_nist_requirements
	@echo "⚖️ Compliance checks completed"

dashboard: ## Start monitoring dashboard
	streamlit run src/rlhf_audit_trail/dashboard/app.py --server.port 8501

# Benchmarking
benchmark: ## Run performance benchmarks
	python benchmarks/run_benchmarks.py
	@echo "🏃 Benchmarks completed. Check benchmarks/results/"

# Database
db-migrate: ## Run database migrations
	alembic upgrade head

db-reset: ## Reset database (WARNING: Destructive)
	@echo "⚠️  This will delete all data. Are you sure? [y/N]" && read ans && [ $${ans:-N} = y ]
	alembic downgrade base
	alembic upgrade head

# Development utilities
deps-update: ## Update dependencies
	pip-compile requirements.in
	pip-compile requirements-dev.in
	@echo "📦 Dependencies updated"

pre-commit-all: ## Run pre-commit on all files
	pre-commit run --all-files

init-project: install-dev ## Initialize project for development
	@echo "🚀 Initializing RLHF Audit Trail for development..."
	python -c "from rlhf_audit_trail.setup import init_dev_environment; init_dev_environment()"
	@echo "✅ Project initialized!"

# Monitoring and maintenance
health-check: ## Check system health
	python -m rlhf_audit_trail.health_check
	@echo "💚 Health check completed"

logs: ## Show recent logs
	docker-compose logs -f --tail=100

backup: ## Backup audit data
	python -m rlhf_audit_trail.backup --output backups/$(shell date +%Y%m%d_%H%M%S)
	@echo "💾 Backup completed"