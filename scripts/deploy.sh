#!/bin/bash

# Quantum Task Planner Deployment Script

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="${IMAGE_NAME:-quantum-task-planner}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-docker-compose}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Main deployment function
main() {
    log_info "Starting Quantum Task Planner deployment"
    log_info "Deployment type: $DEPLOYMENT_TYPE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Image: $IMAGE_NAME:$IMAGE_TAG"
    
    # Build Docker image
    log_info "Building Docker image..."
    cd "$PROJECT_ROOT"
    docker build -f docker/Dockerfile -t "$IMAGE_NAME:$IMAGE_TAG" .
    
    # Deploy with Docker Compose
    if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        log_info "Deploying with Docker Compose..."
        export IMAGE_NAME IMAGE_TAG ENVIRONMENT
        docker-compose up -d
        log_info "Deployment completed!"
        log_info "Access the application at: http://localhost:8000"
    fi
}

main "$@"