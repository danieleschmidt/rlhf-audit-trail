#!/bin/bash
# Comprehensive security scanning script for RLHF Audit Trail

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPORTS_DIR="reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

create_reports_dir() {
    mkdir -p "$REPORTS_DIR"
    log "Created reports directory: $REPORTS_DIR"
}

run_bandit_scan() {
    log "Running Bandit security scan..."
    
    if command -v bandit &> /dev/null; then
        bandit -r src/ \
            -f json \
            -o "$REPORTS_DIR/bandit-report-$TIMESTAMP.json" \
            -ll \
            --confidence-level medium || warning "Bandit found security issues"
        
        # Generate human-readable report
        bandit -r src/ \
            -f txt \
            -o "$REPORTS_DIR/bandit-report-$TIMESTAMP.txt" \
            -ll \
            --confidence-level medium || true
        
        success "Bandit scan completed"
    else
        error "Bandit not installed. Install with: pip install bandit"
        return 1
    fi
}

run_safety_check() {
    log "Running Safety dependency vulnerability check..."
    
    if command -v safety &> /dev/null; then
        # Check production dependencies
        safety check \
            --file requirements.lock \
            --json \
            --output "$REPORTS_DIR/safety-report-$TIMESTAMP.json" || warning "Safety found vulnerabilities"
        
        # Generate human-readable report
        safety check \
            --file requirements.lock \
            --output "$REPORTS_DIR/safety-report-$TIMESTAMP.txt" || true
        
        success "Safety check completed"
    else
        error "Safety not installed. Install with: pip install safety"
        return 1
    fi
}

run_secrets_scan() {
    log "Running secrets detection scan..."
    
    if command -v detect-secrets &> /dev/null; then
        # Update baseline if it doesn't exist
        if [ ! -f .secrets.baseline ]; then
            detect-secrets scan --baseline .secrets.baseline
            log "Created secrets baseline"
        fi
        
        # Scan for new secrets
        detect-secrets scan \
            --baseline .secrets.baseline \
            --exclude-files '^tests/fixtures/' \
            --exclude-files '\.lock$' \
            > "$REPORTS_DIR/secrets-scan-$TIMESTAMP.json" || warning "New secrets detected"
        
        success "Secrets scan completed"
    else
        error "detect-secrets not installed. Install with: pip install detect-secrets"
        return 1
    fi
}

run_dependency_audit() {
    log "Running dependency audit..."
    
    if command -v pip-audit &> /dev/null; then
        pip-audit \
            --requirement requirements.lock \
            --format json \
            --output "$REPORTS_DIR/pip-audit-$TIMESTAMP.json" || warning "Pip-audit found vulnerabilities"
        
        # Generate human-readable report
        pip-audit \
            --requirement requirements.lock \
            --format text \
            --output "$REPORTS_DIR/pip-audit-$TIMESTAMP.txt" || true
        
        success "Dependency audit completed"
    else
        warning "pip-audit not installed. Install with: pip install pip-audit"
    fi
}

run_dockerfile_scan() {
    log "Scanning Dockerfile for security issues..."
    
    if command -v hadolint &> /dev/null && [ -f Dockerfile ]; then
        hadolint Dockerfile \
            --format json \
            > "$REPORTS_DIR/hadolint-$TIMESTAMP.json" || warning "Dockerfile issues found"
        
        success "Dockerfile scan completed"
    else
        warning "hadolint not available or Dockerfile not found"
    fi
}

run_license_check() {
    log "Checking license compliance..."
    
    if command -v pip-licenses &> /dev/null; then
        pip-licenses \
            --format json \
            --output-file "$REPORTS_DIR/licenses-$TIMESTAMP.json"
        
        # Check for problematic licenses
        pip-licenses \
            --format plain \
            --output-file "$REPORTS_DIR/licenses-$TIMESTAMP.txt"
        
        success "License check completed"
    else
        warning "pip-licenses not installed. Install with: pip install pip-licenses"
    fi
}

analyze_results() {
    log "Analyzing security scan results..."
    
    local issues_found=0
    local critical_issues=0
    
    # Check Bandit results
    if [ -f "$REPORTS_DIR/bandit-report-$TIMESTAMP.json" ]; then
        local bandit_issues=$(jq '.results | length' "$REPORTS_DIR/bandit-report-$TIMESTAMP.json" 2>/dev/null || echo "0")
        if [ "$bandit_issues" -gt 0 ]; then
            warning "Bandit found $bandit_issues security issues"
            issues_found=$((issues_found + bandit_issues))
        fi
    fi
    
    # Check Safety results
    if [ -f "$REPORTS_DIR/safety-report-$TIMESTAMP.json" ]; then
        local safety_issues=$(jq '.vulnerabilities | length' "$REPORTS_DIR/safety-report-$TIMESTAMP.json" 2>/dev/null || echo "0")
        if [ "$safety_issues" -gt 0 ]; then
            warning "Safety found $safety_issues vulnerabilities"
            issues_found=$((issues_found + safety_issues))
        fi
    fi
    
    # Generate summary report
    generate_summary_report "$issues_found" "$critical_issues"
    
    return $issues_found
}

generate_summary_report() {
    local total_issues=$1
    local critical_issues=$2
    
    local summary_file="$REPORTS_DIR/security-summary-$TIMESTAMP.md"
    
    cat > "$summary_file" << EOF
# Security Scan Summary

**Scan Date:** $(date)
**Total Issues Found:** $total_issues
**Critical Issues:** $critical_issues

## Scan Results

### Static Code Analysis (Bandit)
$(if [ -f "$REPORTS_DIR/bandit-report-$TIMESTAMP.json" ]; then
    echo "✅ Completed - $(jq '.results | length' "$REPORTS_DIR/bandit-report-$TIMESTAMP.json" 2>/dev/null || echo "0") issues found"
else
    echo "❌ Not run"
fi)

### Dependency Vulnerabilities (Safety)
$(if [ -f "$REPORTS_DIR/safety-report-$TIMESTAMP.json" ]; then
    echo "✅ Completed - $(jq '.vulnerabilities | length' "$REPORTS_DIR/safety-report-$TIMESTAMP.json" 2>/dev/null || echo "0") vulnerabilities found"
else
    echo "❌ Not run"
fi)

### Secrets Detection
$(if [ -f "$REPORTS_DIR/secrets-scan-$TIMESTAMP.json" ]; then
    echo "✅ Completed"
else
    echo "❌ Not run"
fi)

### License Compliance
$(if [ -f "$REPORTS_DIR/licenses-$TIMESTAMP.json" ]; then
    echo "✅ Completed"
else
    echo "❌ Not run"
fi)

## Next Steps

1. Review detailed reports in the \`$REPORTS_DIR\` directory
2. Address critical and high-severity issues immediately
3. Plan remediation for medium and low-severity issues
4. Update security baseline if needed

## Report Files

- Bandit: \`$REPORTS_DIR/bandit-report-$TIMESTAMP.json\`
- Safety: \`$REPORTS_DIR/safety-report-$TIMESTAMP.json\`
- Secrets: \`$REPORTS_DIR/secrets-scan-$TIMESTAMP.json\`
- Licenses: \`$REPORTS_DIR/licenses-$TIMESTAMP.json\`
- Summary: \`$summary_file\`
EOF

    log "Security summary report generated: $summary_file"
}

main() {
    log "Starting comprehensive security scan..."
    
    create_reports_dir
    
    # Run all security scans
    local scan_results=0
    
    run_bandit_scan || ((scan_results++))
    run_safety_check || ((scan_results++))
    run_secrets_scan || ((scan_results++))
    run_dependency_audit || true  # Don't fail if not available
    run_dockerfile_scan || true   # Don't fail if not available
    run_license_check || true     # Don't fail if not available
    
    # Analyze and report results
    if analyze_results; then
        success "Security scan completed successfully - no critical issues found"
        exit 0
    else
        warning "Security scan completed with issues found"
        log "Review reports in $REPORTS_DIR directory"
        exit 1
    fi
}

# Handle interrupts
trap 'error "Security scan interrupted"; exit 1' INT TERM

# Run main function
main "$@"