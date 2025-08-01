#!/bin/bash

# RLHF Audit Trail - Build Script
# Automated build process with security scanning and multi-architecture support

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default values
BUILD_TARGET="production"
PUSH_IMAGE="false"
SCAN_SECURITY="true"
MULTI_ARCH="false"
REGISTRY=""
IMAGE_NAME="rlhf-audit-trail"
VERSION="${VERSION:-$(cat VERSION 2>/dev/null || echo '0.1.0')}"
BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
VCS_REF="${GITHUB_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Help function
show_help() {
    cat << EOF
RLHF Audit Trail Build Script

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --target TARGET     Build target (production, development) [default: production]
    -p, --push             Push image to registry after build
    -r, --registry REGISTRY Registry URL (e.g., docker.io, ghcr.io)
    -n, --name NAME        Image name [default: rlhf-audit-trail]
    -v, --version VERSION  Image version [default: from VERSION file or 0.1.0]
    --no-scan             Skip security scanning
    --multi-arch          Build for multiple architectures (linux/amd64,linux/arm64)
    --cache-from IMAGE    Use image as cache source
    --cache-to IMAGE      Push cache to image
    -h, --help            Show this help message

EXAMPLES:
    # Build production image
    $0

    # Build and push development image
    $0 -t development -p -r ghcr.io/terragonlabs

    # Build with security scanning disabled
    $0 --no-scan

    # Build multi-architecture image
    $0 --multi-arch -p -r docker.io/terragonlabs

ENVIRONMENT VARIABLES:
    VERSION       Image version (overrides --version)
    GITHUB_SHA    Git commit SHA (used for VCS_REF)
    REGISTRY      Default registry (overrides --registry)
    PUSH_IMAGE    Push after build (true/false)
    SCAN_SECURITY Security scanning (true/false)
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -p|--push)
            PUSH_IMAGE="true"
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --no-scan)
            SCAN_SECURITY="false"
            shift
            ;;
        --multi-arch)
            MULTI_ARCH="true"
            shift
            ;;
        --cache-from)
            CACHE_FROM="$2"
            shift 2
            ;;
        --cache-to)
            CACHE_TO="$2"
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

# Validate build target
if [[ "$BUILD_TARGET" != "production" && "$BUILD_TARGET" != "development" ]]; then
    log_error "Invalid build target: $BUILD_TARGET. Must be 'production' or 'development'"
    exit 1
fi

# Build full image name
if [[ -n "$REGISTRY" ]]; then
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}"
else
    FULL_IMAGE_NAME="$IMAGE_NAME"
fi

log_info "Starting build process..."
log_info "Target: $BUILD_TARGET"
log_info "Image: ${FULL_IMAGE_NAME}:${VERSION}"
log_info "Build Date: $BUILD_DATE"
log_info "VCS Ref: $VCS_REF"

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Buildx for multi-arch builds
    if [[ "$MULTI_ARCH" == "true" ]]; then
        if ! docker buildx version &> /dev/null; then
            log_error "Docker Buildx is required for multi-architecture builds"
            exit 1
        fi
    fi
    
    # Check security scanners if enabled
    if [[ "$SCAN_SECURITY" == "true" ]]; then
        if ! command -v trivy &> /dev/null && ! command -v grype &> /dev/null; then
            log_warning "Neither Trivy nor Grype found. Security scanning will be skipped."
            SCAN_SECURITY="false"
        fi
    fi
    
    log_success "Prerequisites check completed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    # Choose Dockerfile
    if [[ "$BUILD_TARGET" == "development" ]]; then
        DOCKERFILE="Dockerfile.dev"
    else
        DOCKERFILE="Dockerfile"
    fi
    
    # Build arguments
    BUILD_ARGS=(
        --build-arg BUILD_DATE="$BUILD_DATE"
        --build-arg VCS_REF="$VCS_REF"
        --build-arg VERSION="$VERSION"
        --target "$BUILD_TARGET"
        --tag "${FULL_IMAGE_NAME}:${VERSION}"
        --tag "${FULL_IMAGE_NAME}:latest"
    )
    
    # Add cache options if specified
    if [[ -n "${CACHE_FROM:-}" ]]; then
        BUILD_ARGS+=(--cache-from "$CACHE_FROM")
    fi
    
    if [[ -n "${CACHE_TO:-}" ]]; then
        BUILD_ARGS+=(--cache-to "$CACHE_TO")
    fi
    
    # Multi-architecture build
    if [[ "$MULTI_ARCH" == "true" ]]; then
        log_info "Building multi-architecture image (linux/amd64,linux/arm64)"
        
        # Create and use buildx builder if it doesn't exist
        if ! docker buildx ls | grep -q "rlhf-builder"; then
            docker buildx create --name rlhf-builder --use
        else
            docker buildx use rlhf-builder
        fi
        
        BUILD_ARGS+=(
            --platform linux/amd64,linux/arm64
            --push  # Multi-arch builds require push
        )
        
        docker buildx build "${BUILD_ARGS[@]}" -f "$DOCKERFILE" .
        
    else
        # Single architecture build
        docker build "${BUILD_ARGS[@]}" -f "$DOCKERFILE" .
    fi
    
    log_success "Image built successfully"
}

# Security scanning
scan_security() {
    if [[ "$SCAN_SECURITY" != "true" ]]; then
        log_info "Security scanning disabled"
        return 0
    fi
    
    log_info "Running security scans..."
    
    # Create reports directory
    mkdir -p reports
    
    # Trivy scan
    if command -v trivy &> /dev/null; then
        log_info "Running Trivy vulnerability scan..."
        
        trivy image \
            --format json \
            --output "reports/trivy-report.json" \
            --severity HIGH,CRITICAL \
            "${FULL_IMAGE_NAME}:${VERSION}" || {
            log_warning "Trivy scan completed with findings"
        }
        
        # Generate human-readable report
        trivy image \
            --format table \
            --output "reports/trivy-report.txt" \
            --severity HIGH,CRITICAL \
            "${FULL_IMAGE_NAME}:${VERSION}" || true
        
        log_info "Trivy report saved to reports/trivy-report.json"
    fi
    
    # Grype scan (alternative to Trivy)
    if command -v grype &> /dev/null && ! command -v trivy &> /dev/null; then
        log_info "Running Grype vulnerability scan..."
        
        grype "${FULL_IMAGE_NAME}:${VERSION}" \
            --output json \
            --file "reports/grype-report.json" || {
            log_warning "Grype scan completed with findings"
        }
        
        log_info "Grype report saved to reports/grype-report.json"
    fi
    
    log_success "Security scanning completed"
}

# Push image to registry
push_image() {
    if [[ "$PUSH_IMAGE" != "true" ]]; then
        log_info "Image push disabled"
        return 0
    fi
    
    if [[ "$MULTI_ARCH" == "true" ]]; then
        log_info "Multi-architecture image already pushed during build"
        return 0
    fi
    
    log_info "Pushing image to registry..."
    
    docker push "${FULL_IMAGE_NAME}:${VERSION}"
    docker push "${FULL_IMAGE_NAME}:latest"
    
    log_success "Image pushed successfully"
}

# Generate build metadata
generate_metadata() {
    log_info "Generating build metadata..."
    
    # Get image information
    IMAGE_ID=$(docker inspect --format='{{.Id}}' "${FULL_IMAGE_NAME}:${VERSION}")
    IMAGE_SIZE=$(docker inspect --format='{{.Size}}' "${FULL_IMAGE_NAME}:${VERSION}")
    
    # Create metadata file
    cat > "reports/build-metadata.json" << EOF
{
  "image": {
    "name": "${FULL_IMAGE_NAME}",
    "version": "${VERSION}",
    "id": "${IMAGE_ID}",
    "size": ${IMAGE_SIZE},
    "target": "${BUILD_TARGET}",
    "multi_arch": ${MULTI_ARCH}
  },
  "build": {
    "date": "${BUILD_DATE}",
    "vcs_ref": "${VCS_REF}",
    "builder": "$(whoami)@$(hostname)",
    "docker_version": "$(docker --version)",
    "buildx_version": "$(docker buildx version 2>/dev/null || echo 'N/A')"
  },
  "security": {
    "scanned": ${SCAN_SECURITY},
    "scan_date": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  }
}
EOF
    
    log_success "Build metadata saved to reports/build-metadata.json"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Remove temporary files if any
    # (Currently no temp files to clean)
    
    # Clean up buildx builder for multi-arch builds
    if [[ "$MULTI_ARCH" == "true" ]]; then
        docker buildx use default || true
    fi
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    # Set up trap for cleanup
    trap cleanup EXIT
    
    # Execute build steps
    check_prerequisites
    build_image
    scan_security
    push_image
    generate_metadata
    
    log_success "Build process completed successfully!"
    log_info "Image: ${FULL_IMAGE_NAME}:${VERSION}"
    
    # Show image size
    if [[ "$MULTI_ARCH" != "true" ]]; then
        IMAGE_SIZE=$(docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep "${FULL_IMAGE_NAME}:${VERSION}" | awk '{print $2}' || echo "Unknown")
        log_info "Image size: $IMAGE_SIZE"
    fi
    
    # Show next steps
    echo
    log_info "Next steps:"
    echo "  • Run the image: docker run -p 8501:8501 ${FULL_IMAGE_NAME}:${VERSION}"
    echo "  • View security reports: ls -la reports/"
    if [[ "$PUSH_IMAGE" == "true" ]]; then
        echo "  • Pull from registry: docker pull ${FULL_IMAGE_NAME}:${VERSION}"
    fi
}

# Execute main function
main "$@"