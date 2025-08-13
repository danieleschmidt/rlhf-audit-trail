#!/bin/bash
set -euo pipefail

# RLHF Audit Trail - Production Deployment Script
# This script deploys the system to production with all safety checks

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_DIR="${PROJECT_ROOT}/deploy"
COMPOSE_FILE="${DEPLOY_DIR}/production-docker-compose.yml"
ENV_FILE="${DEPLOY_DIR}/.env"

# Default values
ENVIRONMENT="production"
BACKUP_ENABLED=true
DRY_RUN=false
SKIP_TESTS=false
FORCE_DEPLOY=false

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

# Help function
show_help() {
    cat << EOF
RLHF Audit Trail - Production Deployment Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV    Target environment (default: production)
    -n, --dry-run           Perform a dry run without making changes
    -t, --skip-tests        Skip pre-deployment tests
    -f, --force             Force deployment even if checks fail
    -h, --help              Show this help message

Examples:
    $0                      # Standard production deployment
    $0 --dry-run           # Preview what would be deployed
    $0 --skip-tests        # Deploy without running tests
    $0 --environment staging  # Deploy to staging environment

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -t|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -f|--force)
                FORCE_DEPLOY=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use --help for usage information."
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "python3" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "$cmd is required but not installed"
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        warn "Environment file not found: $ENV_FILE"
        warn "Please copy production-env-template to .env and configure it"
        
        if [[ "$FORCE_DEPLOY" != "true" ]]; then
            error "Environment file is required for deployment"
        fi
    fi
    
    success "Prerequisites check passed"
}

# Validate configuration
validate_config() {
    log "Validating configuration..."
    
    if [[ -f "$ENV_FILE" ]]; then
        # Source environment file
        set -a
        source "$ENV_FILE"
        set +a
        
        # Check required variables
        local required_vars=("DB_PASSWORD" "GRAFANA_PASSWORD")
        for var in "${required_vars[@]}"; do
            if [[ -z "${!var:-}" ]]; then
                error "Required environment variable $var is not set"
            fi
        done
        
        # Validate storage backend
        case "${STORAGE_BACKEND:-local}" in
            s3)
                if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] || [[ -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
                    error "AWS credentials required for S3 storage backend"
                fi
                ;;
            gcs)
                if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
                    error "Google Cloud credentials required for GCS storage backend"
                fi
                ;;
            azure)
                if [[ -z "${AZURE_STORAGE_ACCOUNT:-}" ]] || [[ -z "${AZURE_STORAGE_KEY:-}" ]]; then
                    error "Azure credentials required for Azure Blob storage backend"
                fi
                ;;
        esac
    fi
    
    success "Configuration validation passed"
}

# Run pre-deployment tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warn "Skipping tests (--skip-tests flag used)"
        return
    fi
    
    log "Running pre-deployment tests..."
    
    cd "$PROJECT_ROOT"
    
    # Create test virtual environment
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    # Install test dependencies
    pip install -q -r requirements-minimal.txt pytest pytest-cov
    
    # Run tests
    log "Running unit tests..."
    if ! PYTHONPATH=src python3 -m pytest tests/ -q --tb=short; then
        if [[ "$FORCE_DEPLOY" != "true" ]]; then
            error "Tests failed. Use --force to deploy anyway or fix the tests."
        else
            warn "Tests failed but continuing due to --force flag"
        fi
    fi
    
    success "Tests passed"
}

# Create backup of current deployment
create_backup() {
    if [[ "$BACKUP_ENABLED" != "true" ]]; then
        return
    fi
    
    log "Creating deployment backup..."
    
    local backup_dir="${DEPLOY_DIR}/backups/$(date +'%Y%m%d_%H%M%S')"
    mkdir -p "$backup_dir"
    
    # Backup configuration
    if [[ -f "$ENV_FILE" ]]; then
        cp "$ENV_FILE" "$backup_dir/"
    fi
    
    # Backup database if running
    if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up"; then
        log "Backing up database..."
        docker-compose -f "$COMPOSE_FILE" exec -T postgres \
            pg_dump -U audit_user rlhf_audit | gzip > "$backup_dir/database_backup.sql.gz"
    fi
    
    success "Backup created at $backup_dir"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would build production Docker image"
        return
    fi
    
    # Build production image
    docker build -f docker/Dockerfile.production -t rlhf-audit-trail:latest .
    
    success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    cd "$DEPLOY_DIR"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would deploy with docker-compose"
        docker-compose -f production-docker-compose.yml config
        return
    fi
    
    # Pull external images
    log "Pulling external Docker images..."
    docker-compose -f production-docker-compose.yml pull postgres redis nginx prometheus grafana loki promtail
    
    # Start services
    log "Starting services..."
    docker-compose -f production-docker-compose.yml up -d
    
    success "Services deployed"
}

# Health checks
run_health_checks() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would run health checks"
        return
    fi
    
    log "Running health checks..."
    
    local max_attempts=30
    local attempt=0
    
    cd "$DEPLOY_DIR"
    
    # Check service health
    while [[ $attempt -lt $max_attempts ]]; do
        log "Health check attempt $((attempt + 1))/$max_attempts"
        
        # Check if all services are healthy
        local unhealthy_services
        unhealthy_services=$(docker-compose -f production-docker-compose.yml ps --filter "health=unhealthy" -q | wc -l)
        
        if [[ $unhealthy_services -eq 0 ]]; then
            success "All services are healthy"
            return
        fi
        
        sleep 10
        ((attempt++))
    done
    
    error "Health checks failed after $max_attempts attempts"
}

# Verify deployment
verify_deployment() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would verify deployment"
        return
    fi
    
    log "Verifying deployment..."
    
    # Test API endpoints
    local api_url="http://localhost/health"
    
    log "Testing health endpoint..."
    if curl -f -s "$api_url" > /dev/null; then
        success "API health endpoint is responding"
    else
        error "API health endpoint is not responding"
    fi
    
    # Test monitoring endpoints
    log "Testing monitoring endpoints..."
    if curl -f -s "http://localhost:9090/api/v1/label/__name__/values" > /dev/null; then
        success "Prometheus is responding"
    else
        warn "Prometheus may not be fully ready yet"
    fi
    
    success "Deployment verification completed"
}

# Show deployment summary
show_summary() {
    log "Deployment Summary"
    echo "=================="
    echo "Environment: $ENVIRONMENT"
    echo "Compose File: $COMPOSE_FILE"
    echo "Dry Run: $DRY_RUN"
    echo
    
    if [[ "$DRY_RUN" != "true" ]]; then
        cd "$DEPLOY_DIR"
        echo "Service Status:"
        docker-compose -f production-docker-compose.yml ps
        echo
        echo "Service URLs:"
        echo "  - Application: http://localhost"
        echo "  - Prometheus: http://localhost:9090"
        echo "  - Grafana: http://localhost:3000"
        echo "  - Health Check: http://localhost/health"
    fi
}

# Cleanup function
cleanup() {
    log "Performing cleanup..."
    # Add any cleanup tasks here
}

# Trap for cleanup on exit
trap cleanup EXIT

# Main function
main() {
    echo "RLHF Audit Trail - Production Deployment"
    echo "========================================"
    echo
    
    parse_args "$@"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        warn "DRY RUN MODE - No changes will be made"
    fi
    
    check_prerequisites
    validate_config
    run_tests
    create_backup
    build_images
    deploy_services
    run_health_checks
    verify_deployment
    show_summary
    
    success "Deployment completed successfully!"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        echo
        log "Next steps:"
        echo "1. Monitor the application logs: docker-compose -f $COMPOSE_FILE logs -f"
        echo "2. Set up SSL certificates for production use"
        echo "3. Configure external monitoring and alerting"
        echo "4. Run full integration tests"
        echo "5. Update DNS records to point to the new deployment"
    fi
}

# Run main function
main "$@"