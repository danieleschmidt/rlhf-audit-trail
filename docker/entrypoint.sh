#!/bin/bash
# Production entrypoint script for RLHF Audit Trail container
# Handles initialization, configuration, and graceful startup

set -e

# Color codes for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_debug() {
    if [[ "${RLHF_AUDIT_LOG_LEVEL}" == "DEBUG" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Environment validation
validate_environment() {
    log_info "Validating environment configuration..."
    
    # Check required environment variables
    local required_vars=(
        "RLHF_AUDIT_ENV"
        "RLHF_AUDIT_PORT"
        "RLHF_AUDIT_HOST"
    )
    
    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        exit 1
    fi
    
    # Validate port
    if ! [[ "$RLHF_AUDIT_PORT" =~ ^[0-9]+$ ]] || [[ "$RLHF_AUDIT_PORT" -lt 1 ]] || [[ "$RLHF_AUDIT_PORT" -gt 65535 ]]; then
        log_error "Invalid port number: $RLHF_AUDIT_PORT"
        exit 1
    fi
    
    # Validate log level
    local valid_log_levels=("DEBUG" "INFO" "WARNING" "ERROR" "CRITICAL")
    if [[ ! " ${valid_log_levels[@]} " =~ " ${RLHF_AUDIT_LOG_LEVEL} " ]]; then
        log_warn "Invalid log level '${RLHF_AUDIT_LOG_LEVEL}', defaulting to INFO"
        export RLHF_AUDIT_LOG_LEVEL="INFO"
    fi
    
    log_info "Environment validation completed successfully"
}

# Database initialization and migration
init_database() {
    log_info "Initializing database..."
    
    # Wait for database to be ready if using external database
    if [[ "${RLHF_AUDIT_DATABASE_URL:-}" =~ ^postgresql:// ]] || [[ "${RLHF_AUDIT_DATABASE_URL:-}" =~ ^mysql:// ]]; then
        log_info "Waiting for external database to be ready..."
        
        local max_attempts=30
        local attempt=1
        
        while [[ $attempt -le $max_attempts ]]; do
            log_debug "Database connection attempt $attempt/$max_attempts"
            
            if python3 -c "
import sys
import os
sys.path.insert(0, '/app/src')
from rlhf_audit_trail.database import DatabaseManager
try:
    db = DatabaseManager(os.environ.get('RLHF_AUDIT_DATABASE_URL'))
    import asyncio
    result = asyncio.run(db.health_check())
    if result['status'] == 'healthy':
        print('Database connection successful')
        sys.exit(0)
    else:
        sys.exit(1)
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
"; then
                log_info "Database connection established"
                break
            fi
            
            if [[ $attempt -eq $max_attempts ]]; then
                log_error "Failed to connect to database after $max_attempts attempts"
                exit 1
            fi
            
            sleep 2
            ((attempt++))
        done
    fi
    
    # Run database initialization
    log_info "Running database initialization..."
    python3 -c "
import sys
import os
sys.path.insert(0, '/app/src')
from rlhf_audit_trail.database import DatabaseManager
try:
    db = DatabaseManager(os.environ.get('RLHF_AUDIT_DATABASE_URL'))
    print('Database initialized successfully')
except Exception as e:
    print(f'Database initialization failed: {e}')
    sys.exit(1)
"
    
    log_info "Database initialization completed"
}

# Load secrets from files
load_secrets() {
    log_info "Loading secrets from mounted files..."
    
    # Load secret key
    if [[ -f "${RLHF_AUDIT_SECRET_KEY_FILE:-}" ]]; then
        export RLHF_AUDIT_SECRET_KEY=$(cat "$RLHF_AUDIT_SECRET_KEY_FILE")
        log_debug "Secret key loaded from file"
    else
        log_warn "Secret key file not found, using environment variable"
    fi
    
    # Load database password
    if [[ -f "${RLHF_AUDIT_DB_PASSWORD_FILE:-}" ]]; then
        export RLHF_AUDIT_DB_PASSWORD=$(cat "$RLHF_AUDIT_DB_PASSWORD_FILE")
        log_debug "Database password loaded from file"
    else
        log_debug "Database password file not found, using environment variable"
    fi
}

# Health check setup
setup_health_checks() {
    log_info "Setting up health check endpoints..."
    
    # Ensure health check script is executable
    if [[ -f "/app/healthcheck.py" ]]; then
        chmod +x /app/healthcheck.py
        log_debug "Health check script made executable"
    else
        log_warn "Health check script not found"
    fi
}

# Graceful shutdown handler
cleanup() {
    local exit_code=$?
    log_info "Received shutdown signal, initiating graceful shutdown..."
    
    # Send SIGTERM to all child processes
    if [[ -n "${app_pid:-}" ]]; then
        log_info "Stopping application (PID: $app_pid)..."
        kill -TERM "$app_pid" 2>/dev/null || true
        
        # Wait for graceful shutdown
        local timeout=30
        local elapsed=0
        while kill -0 "$app_pid" 2>/dev/null && [[ $elapsed -lt $timeout ]]; do
            sleep 1
            ((elapsed++))
        done
        
        # Force kill if still running
        if kill -0 "$app_pid" 2>/dev/null; then
            log_warn "Application didn't shutdown gracefully, forcing termination"
            kill -KILL "$app_pid" 2>/dev/null || true
        else
            log_info "Application shutdown gracefully"
        fi
    fi
    
    log_info "Shutdown complete"
    exit $exit_code
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check Python version
    local python_version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_info "Python version: $python_version"
    
    # Check available disk space
    local available_space
    available_space=$(df /app | awk 'NR==2 {print $4}')
    log_info "Available disk space: ${available_space}KB"
    
    # Minimum space check (100MB)
    if [[ $available_space -lt 102400 ]]; then
        log_warn "Low disk space available: ${available_space}KB"
    fi
    
    # Check memory
    local available_memory
    available_memory=$(free -m | awk 'NR==2{print $7}')
    log_info "Available memory: ${available_memory}MB"
    
    # Check if required directories exist
    local required_dirs=("/app/data" "/app/logs" "/app/audit_data")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_warn "Required directory missing: $dir"
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    # Check permissions
    local test_file="/app/logs/startup_test.log"
    if echo "test" > "$test_file" 2>/dev/null; then
        rm -f "$test_file"
        log_debug "Write permissions OK"
    else
        log_error "No write permissions for logs directory"
        exit 1
    fi
    
    log_info "Pre-flight checks completed"
}

# Application startup
start_application() {
    log_info "Starting RLHF Audit Trail application..."
    
    # Change to application directory
    cd /app
    
    # Set Python path
    export PYTHONPATH="/app/src:$PYTHONPATH"
    
    # Choose startup method based on environment
    case "${RLHF_AUDIT_ENV}" in
        "production")
            log_info "Starting in production mode with Gunicorn"
            exec gunicorn \
                --config /app/gunicorn.conf.py \
                --bind "${RLHF_AUDIT_HOST}:${RLHF_AUDIT_PORT}" \
                --workers "${RLHF_AUDIT_WORKERS}" \
                --worker-class uvicorn.workers.UvicornWorker \
                --access-logfile /app/logs/access.log \
                --error-logfile /app/logs/error.log \
                --capture-output \
                --enable-stdio-inheritance \
                "src.rlhf_audit_trail.main:app" &
            app_pid=$!
            ;;
        "development")
            log_info "Starting in development mode with Uvicorn"
            exec uvicorn \
                --host "${RLHF_AUDIT_HOST}" \
                --port "${RLHF_AUDIT_PORT}" \
                --reload \
                --log-level "${RLHF_AUDIT_LOG_LEVEL,,}" \
                "src.rlhf_audit_trail.main:app" &
            app_pid=$!
            ;;
        "testing")
            log_info "Starting in testing mode"
            exec python3 -m pytest \
                --verbose \
                --cov=src/rlhf_audit_trail \
                --cov-report=html:/app/logs/coverage \
                tests/
            ;;
        *)
            log_error "Unknown environment: ${RLHF_AUDIT_ENV}"
            exit 1
            ;;
    esac
    
    # Wait for application to start
    if [[ "${RLHF_AUDIT_ENV}" != "testing" ]]; then
        log_info "Application started with PID: $app_pid"
        
        # Wait for the application to be ready
        local max_wait=60
        local elapsed=0
        while [[ $elapsed -lt $max_wait ]]; do
            if curl -f -s "http://${RLHF_AUDIT_HOST}:${RLHF_AUDIT_PORT}/health" > /dev/null 2>&1; then
                log_info "Application is ready and responding to health checks"
                break
            fi
            sleep 1
            ((elapsed++))
        done
        
        if [[ $elapsed -eq $max_wait ]]; then
            log_warn "Application may not be fully ready after ${max_wait} seconds"
        fi
        
        # Keep container running and wait for application
        wait $app_pid
    fi
}

# Main execution flow
main() {
    log_info "=== RLHF Audit Trail Container Starting ==="
    log_info "Environment: ${RLHF_AUDIT_ENV}"
    log_info "Version: $(cat /app/VERSION 2>/dev/null || echo 'unknown')"
    log_info "Host: ${RLHF_AUDIT_HOST}:${RLHF_AUDIT_PORT}"
    log_info "================================================"
    
    # Run startup sequence
    validate_environment
    load_secrets
    preflight_checks
    setup_health_checks
    init_database
    start_application
    
    log_info "=== Container Shutdown ==="
}

# Execute main function
main "$@"