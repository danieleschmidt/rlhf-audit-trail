#!/bin/bash

# Post-create script for RLHF Audit Trail development container
set -e

echo "🚀 Setting up RLHF Audit Trail development environment..."

# Update package lists
sudo apt-get update

# Install additional system dependencies
sudo apt-get install -y \
    curl \
    wget \
    unzip \
    jq \
    tree \
    htop \
    postgresql-client \
    redis-tools

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev,testing,docs]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p {audit_logs,checkpoints,model_cards,privacy_reports,compliance_exports}

# Set up database (if PostgreSQL is running)
echo "🗄️ Setting up database..."
if pg_isready -h db -p 5432 -U postgres; then
    echo "PostgreSQL is available, running migrations..."
    alembic upgrade head || echo "⚠️ Migration failed - database may not be ready"
else
    echo "⚠️ PostgreSQL not available - skipping database setup"
fi

# Set up Git hooks for consistent commits
echo "🔀 Setting up Git configuration..."
git config --global core.autocrlf false
git config --global pull.rebase true
git config --global init.defaultBranch main

# Install development tools
echo "🛠️ Installing additional development tools..."
pip install --upgrade \
    ipython \
    jupyter \
    jupyterlab \
    notebook

# Set up shell environment
echo "🐚 Setting up shell environment..."
cat >> ~/.bashrc << 'EOF'

# RLHF Audit Trail development aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias pytest-cov='pytest --cov=src --cov-report=html --cov-report=term'
alias lint='ruff check . && black --check . && mypy src'
alias format='black . && ruff check --fix .'
alias audit-logs='tail -f audit_logs/*.log'

# Environment variables
export PYTHONPATH="/workspace/src:$PYTHONPATH"
export RLHF_AUDIT_ENV="development"
export RLHF_AUDIT_DEBUG="true"
EOF

# Display setup completion message
echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick commands:"
echo "  make dev        - Start development servers"
echo "  make test       - Run test suite"
echo "  make lint       - Run linting"
echo "  make format     - Format code"
echo "  make docs       - Build documentation"
echo ""
echo "📚 Documentation: http://localhost:8000/docs"
echo "📊 Dashboard: http://localhost:8501"
echo ""
echo "Happy coding! 🎉"