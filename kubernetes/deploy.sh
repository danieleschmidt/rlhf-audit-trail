#!/bin/bash
# Kubernetes deployment script for RLHF Audit Trail
# Production-ready deployment with validation and rollback capabilities

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="rlhf-audit-trail"
APP_NAME="rlhf-audit-trail"
KUSTOMIZE_VERSION="v5.0.0"
KUBECTL_MIN_VERSION="1.25"
TIMEOUT="600s"
DRY_RUN="${DRY_RUN:-false}"
SKIP_VALIDATION="${SKIP_VALIDATION:-false}"
FORCE_DEPLOY="${FORCE_DEPLOY:-false}"

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
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Help function
show_help() {
    cat << EOF
RLHF Audit Trail Kubernetes Deployment Script

Usage: $0 [OPTIONS] [COMMAND]

COMMANDS:
    deploy       Deploy the application (default)
    update       Update existing deployment
    rollback     Rollback to previous deployment
    status       Show deployment status
    logs         Show application logs
    delete       Delete the deployment
    validate     Validate manifests only

OPTIONS:
    -n, --namespace NAME    Kubernetes namespace (default: rlhf-audit-trail)
    -i, --image TAG         Docker image tag (default: 1.0.0)
    -r, --replicas COUNT    Number of replicas (default: 3)
    -e, --env ENVIRONMENT   Environment (dev/staging/prod, default: prod)
    --dry-run              Perform a dry run without applying changes
    --skip-validation      Skip manifest validation
    --force                Force deployment even if validation fails
    --timeout DURATION     Deployment timeout (default: 600s)
    --debug                Enable debug logging
    -h, --help             Show this help message

EXAMPLES:
    $0 deploy                                    # Deploy with defaults
    $0 deploy --env staging --replicas 2        # Deploy to staging with 2 replicas
    $0 update --image 1.1.0                     # Update to new image version
    $0 rollback                                  # Rollback to previous version
    $0 status                                    # Check deployment status
    $0 --dry-run deploy                          # Preview deployment changes

ENVIRONMENT VARIABLES:
    KUBECONFIG              Path to kubeconfig file
    DOCKER_REGISTRY         Docker registry URL
    DRY_RUN                 Set to 'true' for dry run mode
    SKIP_VALIDATION         Set to 'true' to skip validation
    FORCE_DEPLOY            Set to 'true' to force deployment
    DEBUG                   Set to 'true' for debug output

EOF
}

# Utility functions
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        return 1
    fi
    
    local kubectl_version
    kubectl_version=$(kubectl version --client --short 2>/dev/null | grep -o 'v[0-9]\+\.[0-9]\+' | head -1)
    log_debug "kubectl version: $kubectl_version"
    
    # Check kustomize
    if ! command -v kustomize &> /dev/null; then
        log_warn "kustomize not found, installing..."
        install_kustomize
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        log_error "Please check your KUBECONFIG and cluster connectivity"
        return 1
    fi
    
    log_info "Prerequisites check passed"
}

install_kustomize() {
    local install_dir="/tmp/kustomize-install"
    mkdir -p "$install_dir"
    
    log_info "Installing kustomize $KUSTOMIZE_VERSION..."
    curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash -s "$KUSTOMIZE_VERSION" "$install_dir"
    
    sudo mv "$install_dir/kustomize" /usr/local/bin/
    rm -rf "$install_dir"
    
    log_info "kustomize installed successfully"
}

validate_manifests() {
    if [[ "$SKIP_VALIDATION" == "true" ]]; then
        log_warn "Skipping manifest validation"
        return 0
    fi
    
    log_info "Validating Kubernetes manifests..."
    
    # Build manifests with kustomize
    if ! kustomize build "$SCRIPT_DIR" > /tmp/manifests.yaml; then
        log_error "Failed to build manifests with kustomize"
        return 1
    fi
    
    # Validate with kubectl dry-run
    if ! kubectl apply --dry-run=server -f /tmp/manifests.yaml &> /tmp/validation.log; then
        log_error "Manifest validation failed:"
        cat /tmp/validation.log
        return 1
    fi
    
    # Check for common issues
    local issues=()
    
    # Check for resource limits
    if ! grep -q "resources:" /tmp/manifests.yaml; then
        issues+=("No resource limits/requests defined")
    fi
    
    # Check for security context
    if ! grep -q "securityContext:" /tmp/manifests.yaml; then
        issues+=("No security context defined")
    fi
    
    # Check for health checks
    if ! grep -q "livenessProbe:" /tmp/manifests.yaml; then
        issues+=("No liveness probes defined")
    fi
    
    if [[ ${#issues[@]} -gt 0 ]]; then
        log_warn "Validation warnings found:"
        for issue in "${issues[@]}"; do
            log_warn "  - $issue"
        done
        
        if [[ "$FORCE_DEPLOY" != "true" ]]; then
            log_error "Use --force to deploy despite warnings"
            return 1
        fi
    fi
    
    log_info "Manifest validation passed"
    rm -f /tmp/manifests.yaml /tmp/validation.log
}

deploy_application() {
    log_info "Deploying RLHF Audit Trail to Kubernetes..."
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Apply manifests
    local apply_cmd="kubectl apply"
    if [[ "$DRY_RUN" == "true" ]]; then
        apply_cmd="$apply_cmd --dry-run=server"
        log_info "Performing dry run deployment..."
    fi
    
    # Use kustomize to build and apply
    kustomize build "$SCRIPT_DIR" | $apply_cmd -f -
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run completed successfully"
        return 0
    fi
    
    # Wait for rollout to complete
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/"$APP_NAME" -n "$NAMESPACE" --timeout="$TIMEOUT"
    
    # Verify pods are running
    local ready_pods
    ready_pods=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=$APP_NAME" --field-selector=status.phase=Running --no-headers | wc -l)
    
    if [[ "$ready_pods" -eq 0 ]]; then
        log_error "No pods are running after deployment"
        return 1
    fi
    
    log_info "Deployment completed successfully"
    log_info "Ready pods: $ready_pods"
    
    # Show service endpoints
    show_endpoints
}

update_deployment() {
    log_info "Updating RLHF Audit Trail deployment..."
    
    # Check if deployment exists
    if ! kubectl get deployment "$APP_NAME" -n "$NAMESPACE" &> /dev/null; then
        log_error "Deployment $APP_NAME not found in namespace $NAMESPACE"
        log_info "Use 'deploy' command for initial deployment"
        return 1
    fi
    
    # Perform rolling update
    deploy_application
}

rollback_deployment() {
    log_info "Rolling back RLHF Audit Trail deployment..."
    
    # Check rollout history
    if ! kubectl rollout history deployment/"$APP_NAME" -n "$NAMESPACE" &> /dev/null; then
        log_error "No rollout history found for deployment $APP_NAME"
        return 1
    fi
    
    # Show rollout history
    log_info "Rollout history:"
    kubectl rollout history deployment/"$APP_NAME" -n "$NAMESPACE"
    
    # Perform rollback
    kubectl rollout undo deployment/"$APP_NAME" -n "$NAMESPACE"
    
    # Wait for rollback to complete
    log_info "Waiting for rollback to complete..."
    kubectl rollout status deployment/"$APP_NAME" -n "$NAMESPACE" --timeout="$TIMEOUT"
    
    log_info "Rollback completed successfully"
}

show_status() {
    log_info "RLHF Audit Trail Deployment Status"
    echo "================================="
    
    # Deployment status
    echo -e "\n${BLUE}Deployment:${NC}"
    kubectl get deployment "$APP_NAME" -n "$NAMESPACE" -o wide 2>/dev/null || log_warn "Deployment not found"
    
    # Pod status
    echo -e "\n${BLUE}Pods:${NC}"
    kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=$APP_NAME" -o wide 2>/dev/null || log_warn "No pods found"
    
    # Service status
    echo -e "\n${BLUE}Services:${NC}"
    kubectl get services -n "$NAMESPACE" -l "app.kubernetes.io/name=$APP_NAME" 2>/dev/null || log_warn "No services found"
    
    # Ingress status
    echo -e "\n${BLUE}Ingress:${NC}"
    kubectl get ingress -n "$NAMESPACE" -l "app.kubernetes.io/name=$APP_NAME" 2>/dev/null || log_warn "No ingress found"
    
    # HPA status
    echo -e "\n${BLUE}HPA:${NC}"
    kubectl get hpa -n "$NAMESPACE" -l "app.kubernetes.io/name=$APP_NAME" 2>/dev/null || log_warn "No HPA found"
    
    # Recent events
    echo -e "\n${BLUE}Recent Events:${NC}"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' --field-selector=involvedObject.name="$APP_NAME" 2>/dev/null | tail -5 || log_warn "No recent events"
}

show_logs() {
    log_info "Showing RLHF Audit Trail application logs..."
    
    local pod_name
    pod_name=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=$APP_NAME" --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [[ -z "$pod_name" ]]; then
        log_error "No running pods found"
        return 1
    fi
    
    log_info "Showing logs for pod: $pod_name"
    kubectl logs -n "$NAMESPACE" "$pod_name" -c "$APP_NAME" --tail=50 -f
}

delete_deployment() {
    log_warn "This will delete the entire RLHF Audit Trail deployment"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deletion cancelled"
        return 0
    fi
    
    log_info "Deleting RLHF Audit Trail deployment..."
    
    # Delete using kustomize
    kustomize build "$SCRIPT_DIR" | kubectl delete -f - --ignore-not-found=true
    
    # Delete namespace if requested
    read -p "Delete namespace $NAMESPACE as well? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
        log_info "Namespace $NAMESPACE deleted"
    fi
    
    log_info "Deployment deletion completed"
}

show_endpoints() {
    log_info "Application endpoints:"
    
    # Get ingress endpoints
    local ingress_hosts
    ingress_hosts=$(kubectl get ingress -n "$NAMESPACE" -l "app.kubernetes.io/name=$APP_NAME" -o jsonpath='{.items[*].spec.rules[*].host}' 2>/dev/null)
    
    if [[ -n "$ingress_hosts" ]]; then
        for host in $ingress_hosts; do
            echo "  - https://$host"
        done
    else
        # Get service endpoints
        local service_ip
        service_ip=$(kubectl get service "${APP_NAME}-service" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
        
        if [[ -n "$service_ip" ]]; then
            echo "  - http://$service_ip:8000"
        else
            log_info "Use port-forward for local access:"
            echo "  kubectl port-forward -n $NAMESPACE service/${APP_NAME}-service 8000:8000"
        fi
    fi
}

# Parse command line arguments
COMMAND="deploy"
IMAGE_TAG="1.0.0"
REPLICAS="3"
ENVIRONMENT="prod"

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--replicas)
            REPLICAS="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION="true"
            shift
            ;;
        --force)
            FORCE_DEPLOY="true"
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        deploy|update|rollback|status|logs|delete|validate)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Export variables for kustomize
export IMAGE_TAG
export REPLICAS
export ENVIRONMENT

# Main execution
main() {
    log_info "RLHF Audit Trail Kubernetes Deployment"
    log_info "Namespace: $NAMESPACE"
    log_info "Command: $COMMAND"
    log_info "Environment: $ENVIRONMENT"
    
    # Check prerequisites
    check_prerequisites
    
    # Execute command
    case $COMMAND in
        deploy)
            validate_manifests
            deploy_application
            ;;
        update)
            validate_manifests
            update_deployment
            ;;
        rollback)
            rollback_deployment
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        delete)
            delete_deployment
            ;;
        validate)
            validate_manifests
            log_info "Validation completed successfully"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"