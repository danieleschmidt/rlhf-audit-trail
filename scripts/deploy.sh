#!/bin/bash
# Deployment script for RLHF Audit Trail
# Supports multiple environments and deployment strategies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_ENVIRONMENT="staging"
DEFAULT_STRATEGY="rolling"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

show_help() {
    cat << EOF
RLHF Audit Trail Deployment Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV     Target environment (development, staging, production)
    -s, --strategy STRATEGY   Deployment strategy (rolling, blue-green, canary)
    -v, --version VERSION     Version to deploy (default: latest)
    -c, --config FILE         Custom configuration file
    -d, --dry-run            Show what would be deployed without executing
    --skip-tests             Skip pre-deployment tests
    --skip-backup            Skip database backup
    --force                  Force deployment even with warnings
    --rollback VERSION       Rollback to specified version
    -h, --help               Show this help message

Examples:
    $0 -e staging                           # Deploy to staging
    $0 -e production -s blue-green         # Blue-green deployment to production
    $0 --rollback v1.2.3                  # Rollback to version 1.2.3
    $0 -e production --dry-run             # Show what would be deployed

Environments:
    development     Local development environment
    staging         Staging environment for testing
    production      Production environment

Deployment Strategies:
    rolling         Rolling update (default)
    blue-green      Blue-green deployment
    canary          Canary deployment
EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    local required_tools=("docker" "docker-compose" "git" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir &> /dev/null; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Check for clean working directory (production only)
    if [[ "$ENVIRONMENT" == "production" ]] && [[ -n "$(git status --porcelain)" ]]; then
        log_error "Working directory is not clean. Commit or stash changes before production deployment."
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

load_configuration() {
    local config_file="${CONFIG_FILE:-$PROJECT_ROOT/deploy/config/$ENVIRONMENT.env}"
    
    if [[ -f "$config_file" ]]; then
        log_info "Loading configuration from $config_file"
        # shellcheck source=/dev/null
        source "$config_file"
    else
        log_warn "Configuration file $config_file not found, using defaults"
    fi
    
    # Set defaults
    export VERSION="${VERSION:-latest}"
    export DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
    export NAMESPACE="${NAMESPACE:-rlhf-audit-trail}"
    export REPLICAS="${REPLICAS:-2}"
    export HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
}

validate_environment() {
    log_info "Validating environment configuration..."
    
    case "$ENVIRONMENT" in
        development)
            log_info "Deploying to development environment"
            ;;
        staging)
            log_info "Deploying to staging environment"
            ;;
        production)
            log_info "Deploying to production environment"
            if [[ "$FORCE" != "true" ]]; then
                read -p "Are you sure you want to deploy to production? (y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    log_info "Deployment cancelled"
                    exit 0
                fi
            fi
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
}

run_pre_deployment_checks() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warn "Skipping pre-deployment tests"
        return 0
    fi
    
    log_info "Running pre-deployment checks..."
    
    # Run tests
    log_info "Running test suite..."
    if ! make test; then
        log_error "Tests failed"
        exit 1
    fi
    
    # Run security checks
    log_info "Running security checks..."
    if ! make security; then
        log_error "Security checks failed"
        exit 1
    fi
    
    # Run compliance checks
    log_info "Running compliance validation..."
    if ! python compliance/compliance-validator.py --format json > /tmp/compliance-report.json; then
        log_error "Compliance validation failed"
        if [[ "$ENVIRONMENT" == "production" ]]; then
            exit 1
        else
            log_warn "Continuing with non-production deployment despite compliance issues"
        fi
    fi
    
    log_info "Pre-deployment checks passed"
}

backup_database() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        log_warn "Skipping database backup"
        return 0
    fi
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_info "Creating database backup..."
        
        local backup_name="backup-$(date +%Y%m%d_%H%M%S)"
        
        # Create backup using pg_dump
        if docker-compose exec -T postgres pg_dump -U postgres rlhf_audit > "backups/${backup_name}.sql"; then
            log_info "Database backup created: backups/${backup_name}.sql"
        else
            log_error "Database backup failed"
            if [[ "$FORCE" != "true" ]]; then
                exit 1
            fi
        fi
    fi
}

build_images() {
    log_info "Building Docker images..."
    
    # Build application image
    local image_tag="${DOCKER_REGISTRY}rlhf-audit-trail:${VERSION}"
    
    log_info "Building image: $image_tag"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build: $image_tag"
        return 0
    fi
    
    docker build \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --build-arg VERSION="$VERSION" \
        -t "$image_tag" \
        "$PROJECT_ROOT"
    
    # Push to registry if configured
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        log_info "Pushing image to registry..."
        docker push "$image_tag"
    fi
}

deploy_rolling() {
    log_info "Performing rolling deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform rolling deployment"
        return 0
    fi
    
    # Update docker-compose with new image
    export IMAGE_TAG="${DOCKER_REGISTRY}rlhf-audit-trail:${VERSION}"
    
    # Rolling update
    docker-compose up -d --scale app="$REPLICAS"
    
    # Wait for health checks
    wait_for_health_check
}

deploy_blue_green() {
    log_info "Performing blue-green deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform blue-green deployment"
        return 0
    fi
    
    # Deploy to green environment
    log_info "Deploying to green environment..."
    export COLOR="green"
    docker-compose -f docker-compose.yml -f docker-compose.blue-green.yml up -d
    
    # Wait for green environment to be healthy
    wait_for_health_check
    
    # Switch traffic to green
    log_info "Switching traffic to green environment..."
    # Implementation would depend on load balancer configuration
    
    # Stop blue environment
    log_info "Stopping blue environment..."
    export COLOR="blue"
    docker-compose -f docker-compose.yml -f docker-compose.blue-green.yml down
}

deploy_canary() {
    log_info "Performing canary deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform canary deployment"
        return 0
    fi
    
    # Deploy canary version (10% of traffic)
    log_info "Deploying canary version..."
    export CANARY_PERCENTAGE=10
    docker-compose -f docker-compose.yml -f docker-compose.canary.yml up -d
    
    # Monitor canary for issues
    log_info "Monitoring canary deployment..."
    sleep 300  # Wait 5 minutes
    
    # Check error rates, response times, etc.
    if check_canary_health; then
        log_info "Canary deployment successful, rolling out to 100%"
        export CANARY_PERCENTAGE=100
        docker-compose -f docker-compose.yml -f docker-compose.canary.yml up -d
    else
        log_error "Canary deployment failed, rolling back"
        rollback_deployment
        exit 1
    fi
}

wait_for_health_check() {
    log_info "Waiting for application health check..."
    
    local timeout="$HEALTH_CHECK_TIMEOUT"
    local count=0
    
    while [[ $count -lt $timeout ]]; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            log_info "Application is healthy"
            return 0
        fi
        
        log_debug "Health check failed, retrying in 5 seconds..."
        sleep 5
        ((count+=5))
    done
    
    log_error "Health check timeout after ${timeout}s"
    return 1
}

check_canary_health() {
    log_info "Checking canary deployment health..."
    
    # Check error rate
    local error_rate
    error_rate=$(curl -s http://localhost:9090/api/v1/query?query='rate(http_requests_total{status=~"5.."}[5m])' | jq -r '.data.result[0].value[1] // "0"')
    
    if (( $(echo "$error_rate > 0.05" | bc -l) )); then
        log_error "Canary error rate too high: $error_rate"
        return 1
    fi
    
    # Check response time
    local response_time
    response_time=$(curl -s http://localhost:9090/api/v1/query?query='histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))' | jq -r '.data.result[0].value[1] // "0"')
    
    if (( $(echo "$response_time > 2.0" | bc -l) )); then
        log_error "Canary response time too high: ${response_time}s"
        return 1
    fi
    
    log_info "Canary deployment health check passed"
    return 0
}

rollback_deployment() {
    local rollback_version="$1"
    
    log_info "Rolling back to version: $rollback_version"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would rollback to version: $rollback_version"
        return 0
    fi
    
    # Update image tag for rollback
    export IMAGE_TAG="${DOCKER_REGISTRY}rlhf-audit-trail:${rollback_version}"
    
    # Perform rollback
    docker-compose up -d
    
    # Wait for health check
    wait_for_health_check
    
    log_info "Rollback completed successfully"
}

run_post_deployment_checks() {
    log_info "Running post-deployment checks..."
    
    # Verify application is responding
    if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_error "Application health check failed"
        return 1
    fi
    
    # Run smoke tests
    log_info "Running smoke tests..."
    if ! make test-smoke; then
        log_error "Smoke tests failed"
        return 1
    fi
    
    # Check compliance after deployment
    log_info "Verifying post-deployment compliance..."
    if ! python compliance/compliance-validator.py --format json > /tmp/post-deployment-compliance.json; then
        log_warn "Post-deployment compliance check failed"
    fi
    
    # Update monitoring
    log_info "Updating monitoring dashboards..."
    # This would typically update Grafana dashboards, alert rules, etc.
    
    log_info "Post-deployment checks completed"
}

cleanup() {
    log_info "Cleaning up old images and containers..."
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Remove old images (keep last 3 versions)
        docker images --format "table {{.Repository}}:{{.Tag}}\t{{.CreatedAt}}" | \
        grep "rlhf-audit-trail" | \
        sort -k2 -r | \
        tail -n +4 | \
        cut -f1 | \
        xargs -r docker rmi
        
        # Clean up unused containers and networks
        docker container prune -f
        docker network prune -f
    fi
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--strategy)
                STRATEGY="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            --skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP="true"
                shift
                ;;
            --force)
                FORCE="true"
                shift
                ;;
            --rollback)
                ROLLBACK_VERSION="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Set defaults
    ENVIRONMENT="${ENVIRONMENT:-$DEFAULT_ENVIRONMENT}"
    STRATEGY="${STRATEGY:-$DEFAULT_STRATEGY}"
    DRY_RUN="${DRY_RUN:-false}"
    SKIP_TESTS="${SKIP_TESTS:-false}"
    SKIP_BACKUP="${SKIP_BACKUP:-false}"
    FORCE="${FORCE:-false}"
    
    log_info "Starting RLHF Audit Trail deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Strategy: $STRATEGY"
    log_info "Version: ${VERSION:-latest}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "DRY RUN MODE - No actual changes will be made"
    fi
    
    # Handle rollback
    if [[ -n "$ROLLBACK_VERSION" ]]; then
        rollback_deployment "$ROLLBACK_VERSION"
        exit 0
    fi
    
    # Main deployment flow
    check_prerequisites
    load_configuration
    validate_environment
    run_pre_deployment_checks
    backup_database
    build_images
    
    # Execute deployment strategy
    case "$STRATEGY" in
        rolling)
            deploy_rolling
            ;;
        blue-green)
            deploy_blue_green
            ;;
        canary)
            deploy_canary
            ;;
        *)
            log_error "Unknown deployment strategy: $STRATEGY"
            exit 1
            ;;
    esac
    
    run_post_deployment_checks
    cleanup
    
    log_info "Deployment completed successfully!"
}

# Execute main function
main "$@"