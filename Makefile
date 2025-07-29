.PHONY: help install test lint format type-check clean build docs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in development mode
	pip install -e ".[dev]"
	pre-commit install

install-all: ## Install with all optional dependencies
	pip install -e ".[aws,gcp,azure,ml,ui,dev]"

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run linting
	ruff check .

format: ## Format code
	black .
	ruff check . --fix

type-check: ## Run type checking
	mypy src/

quality: format lint type-check ## Run all quality checks

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	python -m build

docs: ## Build documentation (placeholder)
	@echo "Documentation build not yet implemented"

security: ## Run security checks
	bandit -r src/
	safety check

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

dev-setup: install ## Complete development setup
	@echo "Development environment ready!"
	@echo "Run 'make help' to see available commands"