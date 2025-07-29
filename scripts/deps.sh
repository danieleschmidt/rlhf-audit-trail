#!/bin/bash
# Simple dependency update script for RLHF Audit Trail

set -euo pipefail

echo "🔄 Updating dependencies..."

# Create backup
BACKUP_DIR="deps_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup existing lock files
for file in requirements.lock requirements-dev.lock; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/"
        echo "📦 Backed up $file"
    fi
done

# Install pip-tools if not available
if ! command -v pip-compile &> /dev/null; then
    echo "📥 Installing pip-tools..."
    pip install pip-tools
fi

# Compile requirements
echo "🔨 Compiling production requirements..."
pip-compile --output-file=requirements.lock requirements.in

echo "🔨 Compiling development requirements..."
pip-compile --output-file=requirements-dev.lock requirements-dev.in

# Run security check if safety is available
if command -v safety &> /dev/null; then
    echo "🔒 Running security check..."
    safety check --file requirements.lock || echo "⚠️ Security issues found"
else
    echo "⚠️ Safety not installed, skipping security check"
fi

echo "✅ Dependencies updated successfully!"
echo "📁 Backup saved to: $BACKUP_DIR"
echo ""
echo "To install updated dependencies:"
echo "  pip install -r requirements.lock -r requirements-dev.lock"