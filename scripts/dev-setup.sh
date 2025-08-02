#!/bin/bash
# Development Environment Setup Script for RLHF Audit Trail
# This script sets up a complete development environment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed. Please install it first."
        return 1
    fi
    return 0
}

# Main setup function
main() {
    log_info "Starting RLHF Audit Trail development environment setup..."
    
    # Check prerequisites
    log_info "Checking prerequisites..."
    check_command "python3" || exit 1
    check_command "git" || exit 1
    check_command "docker" || exit 1
    check_command "docker-compose" || exit 1
    
    # Check Python version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version < 3.10" | bc -l) -eq 1 ]]; then
        log_error "Python 3.10+ is required. Current version: $python_version"
        exit 1
    fi
    log_success "Python version $python_version meets requirements"
    
    # Create virtual environment
    log_info "Setting up Python virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    log_success "Virtual environment activated"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install development dependencies
    log_info "Installing development dependencies..."
    pip install -e ".[dev,testing,docs]"
    log_success "Development dependencies installed"
    
    # Install pre-commit hooks
    log_info "Setting up pre-commit hooks..."
    pre-commit install
    pre-commit install --hook-type commit-msg
    log_success "Pre-commit hooks installed"
    
    # Setup environment file
    log_info "Setting up environment configuration..."
    if [ ! -f ".env" ]; then
        cp .env.example .env
        log_success "Environment file created from template"
        log_warning "Please edit .env file with your specific configuration"
    else
        log_warning ".env file already exists"
    fi
    
    # Create necessary directories
    log_info "Creating necessary directories..."
    mkdir -p logs
    mkdir -p storage
    mkdir -p cache
    mkdir -p keys
    log_success "Directories created"
    
    # Generate development keys
    log_info "Generating development cryptographic keys..."
    if [ ! -f "keys/signature_private.pem" ]; then
        python3 -c "
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import os

# Generate private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=4096,
)

# Save private key
with open('keys/signature_private.pem', 'wb') as f:
    f.write(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ))

# Save public key
public_key = private_key.public_key()
with open('keys/signature_public.pem', 'wb') as f:
    f.write(public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ))

print('Cryptographic keys generated successfully')
"
        log_success "Development keys generated"
    else
        log_warning "Development keys already exist"
    fi
    
    # Setup database containers
    log_info "Starting development database containers..."
    if docker-compose -f docker-compose.dev.yml ps postgres | grep -q "Up"; then
        log_warning "Database containers already running"
    else
        docker-compose -f docker-compose.dev.yml up -d postgres redis
        log_success "Database containers started"
    fi
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    timeout 30 bash -c '
        until docker-compose -f docker-compose.dev.yml exec -T postgres pg_isready -U postgres; do
            echo "Waiting for postgres..."
            sleep 2
        done
    '
    log_success "Database is ready"
    
    # Run database migrations
    log_info "Running database migrations..."
    alembic upgrade head
    log_success "Database migrations completed"
    
    # Run initial tests
    log_info "Running initial test suite..."
    pytest tests/ -v --tb=short -x
    log_success "Initial tests passed"
    
    # Setup git hooks
    log_info "Setting up additional git hooks..."
    cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
# Pre-push hook to run tests and linting

echo "Running pre-push checks..."

# Run linting
echo "Running linting..."
pre-commit run --all-files || exit 1

# Run tests
echo "Running tests..."
pytest tests/ -x || exit 1

echo "Pre-push checks passed!"
EOF
    chmod +x .git/hooks/pre-push
    log_success "Git hooks configured"
    
    # Create development scripts
    log_info "Creating development helper scripts..."
    
    # Quick test runner
    cat > scripts/test-quick.sh << 'EOF'
#!/bin/bash
# Quick test runner for development
set -e
pytest tests/unit/ -v --tb=short "$@"
EOF
    chmod +x scripts/test-quick.sh
    
    # Full test runner
    cat > scripts/test-full.sh << 'EOF'
#!/bin/bash
# Full test suite runner
set -e
pytest tests/ -v --cov=src --cov-report=html --cov-report=term "$@"
EOF
    chmod +x scripts/test-full.sh
    
    # Development server starter
    cat > scripts/dev-server.sh << 'EOF'
#!/bin/bash
# Start development server with hot reload
set -e
source venv/bin/activate
export ENVIRONMENT=development
export DEBUG=true
uvicorn src.rlhf_audit_trail.main:app --host 0.0.0.0 --port 8000 --reload
EOF
    chmod +x scripts/dev-server.sh
    
    log_success "Development helper scripts created"
    
    # Final verification
    log_info "Running final verification..."
    python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    import rlhf_audit_trail
    print('✓ Package imports successfully')
except ImportError as e:
    print(f'✗ Package import failed: {e}')
    sys.exit(1)
"
    
    log_success "Setup verification completed"
    
    # Display summary
    echo ""
    log_success "Development environment setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Edit .env file with your configuration"
    echo "3. Run quick tests: ./scripts/test-quick.sh"
    echo "4. Start development server: ./scripts/dev-server.sh"
    echo "5. Open Streamlit dashboard: streamlit run src/rlhf_audit_trail/dashboard.py"
    echo ""
    echo "Available commands:"
    echo "- make test          # Run test suite"
    echo "- make lint          # Run linting"
    echo "- make format        # Format code"
    echo "- make security      # Run security scans"
    echo "- make docs          # Build documentation"
    echo ""
    log_info "Happy coding! 🚀"
}

# Run main function
main "$@"