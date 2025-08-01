# RLHF Audit Trail - GitHub Workflows Setup Guide

## 🚨 IMPORTANT NOTICE

**This repository contains workflow documentation and templates that must be manually configured due to GitHub App permission limitations.**

The RLHF Audit Trail project includes comprehensive CI/CD workflows for:
- Continuous Integration (CI)
- Continuous Deployment (CD) 
- Security Scanning
- Dependency Management
- Performance Monitoring
- Compliance Validation

## Manual Setup Required

Repository maintainers must manually copy the workflow templates from `docs/workflows/examples/` to `.github/workflows/` and configure the required secrets and settings.

## Quick Setup Checklist

### 1. Create Workflow Files

Copy the template files to the correct location:

```bash
# Create .github/workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/examples/ci.yml .github/workflows/ci.yml
cp docs/workflows/examples/cd.yml .github/workflows/cd.yml
cp docs/workflows/examples/security-scan.yml .github/workflows/security-scan.yml
cp docs/workflows/examples/dependency-update.yml .github/workflows/dependency-update.yml

# Commit the workflow files
git add .github/workflows/
git commit -m "feat: add GitHub workflows for CI/CD and security"
git push
```

### 2. Configure Repository Secrets

Navigate to **Settings > Secrets and variables > Actions** and add:

#### Registry and Deployment Secrets
```
GITHUB_TOKEN                    # Automatically provided
STAGING_KUBECONFIG             # Base64 encoded staging kubeconfig
PRODUCTION_KUBECONFIG          # Base64 encoded production kubeconfig
STAGING_API_KEY                # API key for staging environment
PRODUCTION_API_KEY             # API key for production environment
```

#### Security and Compliance Secrets
```
SONAR_TOKEN                    # SonarCloud token
SNYK_TOKEN                     # Snyk vulnerability scanning token
SECURITY_SLACK_WEBHOOK_URL     # Slack webhook for security alerts
```

#### Notification Secrets
```
SLACK_WEBHOOK_URL              # Slack webhook for general notifications
STATUS_PAGE_TOKEN              # Status page API token
DEPLOYMENT_TRACKER_TOKEN       # Deployment tracking API token
```

### 3. Repository Settings Configuration

#### Branch Protection Rules

Navigate to **Settings > Branches** and configure protection for `main` branch:

- ✅ Require a pull request before merging
- ✅ Require approvals: 2
- ✅ Dismiss stale PR approvals when new commits are pushed
- ✅ Require review from code owners
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Require conversation resolution before merging
- ✅ Include administrators

**Required Status Checks:**
- `Code Quality & Security`
- `Test Suite`
- `Compliance & Privacy Tests`
- `Documentation`
- `Container Security Scan`

#### Security Settings

Navigate to **Settings > Security & analysis** and enable:

- ✅ Dependency graph
- ✅ Dependabot alerts
- ✅ Dependabot security updates
- ✅ Code scanning (CodeQL)
- ✅ Secret scanning
- ✅ Secret scanning push protection

### 4. Environment Configuration

#### Staging Environment

Navigate to **Settings > Environments** and create `staging`:

- **Deployment branches:** Selected branches → `main`
- **Environment secrets:**
  - `STAGING_API_URL`: `https://staging.rlhf-audit-trail.example.com`
  - `STAGING_DATABASE_URL`: Staging database connection string
- **Reviewers:** (Optional) Add required reviewers

#### Production Environment

Navigate to **Settings > Environments** and create `production`:

- **Deployment branches:** Selected branches → `main`
- **Required reviewers:** Add production deployment approvers
- **Wait timer:** 10 minutes (allows for final checks)
- **Environment secrets:**
  - `PRODUCTION_API_URL`: `https://rlhf-audit-trail.example.com`
  - `PRODUCTION_DATABASE_URL`: Production database connection string

#### Production Approval Environment

Create `production-approval` environment:

- **Required reviewers:** Add senior engineers and compliance team
- **Deployment branches:** Selected branches → `main`

## Workflow Details

### Continuous Integration (ci.yml)

**Triggers:**
- Push to `main` and `develop` branches
- Pull requests to `main` and `develop`
- Manual workflow dispatch

**Jobs:**
1. **Code Quality & Security**
   - Black code formatting check
   - Ruff linting
   - MyPy type checking
   - Bandit security scan
   - Safety dependency scan
   - SonarCloud analysis

2. **Test Suite** (Matrix: Python 3.10, 3.11, 3.12)
   - Unit tests with coverage
   - Integration tests
   - Test result upload
   - Codecov integration

3. **Compliance & Privacy Tests**
   - EU AI Act compliance validation
   - GDPR compliance checks
   - Privacy protection tests
   - Security requirement validation

4. **Documentation**
   - Sphinx documentation build
   - Link validation
   - Documentation artifact upload

5. **Container Security Scan**
   - Docker image vulnerability scanning
   - SARIF result upload
   - Security report generation

6. **Performance Benchmarks** (main branch only)
   - Performance regression testing
   - Benchmark result tracking
   - Performance alerting

### Continuous Deployment (cd.yml)

**Triggers:**
- Push to `main` branch (staging deployment)
- Tag creation `v*` (production deployment)
- Manual workflow dispatch

**Jobs:**
1. **Setup**
   - Environment determination
   - Configuration setup

2. **Build & Push Images**
   - Multi-architecture Docker builds
   - Container registry push
   - Vulnerability scanning
   - SBOM generation

3. **Deploy to Staging**
   - Kubernetes deployment
   - Smoke tests
   - Integration tests
   - Slack notifications

4. **Production Approval**
   - Manual approval gate
   - Stakeholder review

5. **Deploy to Production**
   - Database backup
   - Blue-green deployment
   - Traffic switching
   - Post-deployment validation

### Security Scanning (security-scan.yml)

**Triggers:**
- Daily schedule (2 AM UTC)
- Push to `main` branch
- Pull requests
- Manual workflow dispatch

**Jobs:**
1. **SAST (Static Application Security Testing)**
   - Bandit security analysis
   - Semgrep security patterns
   - CodeQL analysis
   - Secret scanning

2. **Dependency Vulnerability Check**
   - Safety vulnerability scanning
   - pip-audit analysis
   - Snyk dependency scan
   - License compliance check

3. **Container Security Scan**
   - Trivy vulnerability scanning
   - Grype security analysis
   - Docker Scout scanning
   - SBOM generation

4. **Infrastructure Security**
   - Checkov IaC analysis
   - Terrascan security scan
   - Kubernetes security validation

5. **Compliance Security**
   - Privacy security tests
   - Compliance validation
   - Audit trail integrity
   - Encryption verification

## Compliance Features

### EU AI Act Compliance

- **Automated Validation:** CI pipeline validates EU AI Act requirements
- **Documentation Generation:** Automatic compliance documentation
- **Audit Trail Integrity:** Cryptographic verification in every build
- **Risk Assessment:** Automated risk level evaluation
- **Human Oversight:** Required approvals for production deployments

### GDPR Compliance

- **Privacy Impact Assessment:** Automated PIA generation
- **Data Protection:** Privacy test validation
- **Consent Management:** Consent mechanism testing
- **Data Minimization:** Automated data usage validation
- **Right to be Forgotten:** Deletion mechanism testing

### Security Standards

- **SLSA Compliance:** Supply chain security validation
- **NIST Framework:** Security control implementation
- **ISO 27001:** Information security management
- **SOC 2:** Service organization control validation

## Monitoring and Alerting

### Slack Integration

Configure Slack webhooks for:
- **Deployment notifications:** Success/failure alerts
- **Security alerts:** Vulnerability and scan failures
- **Compliance alerts:** Regulatory requirement failures
- **Performance alerts:** SLA violations

### Status Page Integration

Automated status page updates for:
- **Deployment status:** Real-time deployment tracking
- **Service health:** System availability monitoring
- **Incident management:** Automated incident creation
- **Maintenance windows:** Scheduled maintenance notifications

## Troubleshooting

### Common Issues

#### Workflow Permissions

**Error:** `Resource not accessible by integration`

**Solution:** Ensure GITHUB_TOKEN has sufficient permissions:
```yaml
permissions:
  contents: read
  packages: write
  security-events: write
```

#### Missing Secrets

**Error:** `Secret not found`

**Solution:** Verify all required secrets are configured in repository settings.

#### Environment Access

**Error:** `Environment protection rules`

**Solution:** Ensure proper reviewers are configured for protected environments.

### Debug Mode

Enable workflow debugging by setting repository secrets:
```
ACTIONS_STEP_DEBUG=true
ACTIONS_RUNNER_DEBUG=true
```

### Workflow Validation

Validate workflow syntax before committing:
```bash
# Install act for local testing
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Test workflows locally
act -n  # Dry run
act pull_request  # Test PR workflow
```

## Advanced Configuration

### Custom Runners

For improved performance and security, configure self-hosted runners:

1. **Setup:** Follow GitHub's self-hosted runner setup guide
2. **Labels:** Use custom labels for specialized workloads
3. **Security:** Implement proper network isolation
4. **Monitoring:** Add runner health monitoring

### Matrix Strategies

Customize test matrices for different scenarios:

```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
    os: [ubuntu-latest, windows-latest, macos-latest]
    include:
      - python-version: '3.10'
        compliance-test: true
```

### Parallel Deployments

Configure parallel deployments for different regions:

```yaml
strategy:
  matrix:
    region: [us-east-1, eu-west-1, ap-southeast-1]
```

## Maintenance

### Regular Updates

- **Monthly:** Review and update workflow dependencies
- **Quarterly:** Security scan tool updates
- **Annually:** Comprehensive workflow review and optimization

### Performance Optimization

- **Cache Management:** Optimize build caches
- **Parallel Execution:** Maximize job parallelization
- **Resource Allocation:** Monitor runner resource usage

### Security Reviews

- **Access Review:** Quarterly secret and permission audit
- **Vulnerability Assessment:** Regular security tool updates
- **Compliance Updates:** Track regulatory requirement changes

## Support

### Getting Help

1. **GitHub Docs:** [GitHub Actions Documentation](https://docs.github.com/en/actions)
2. **Community:** GitHub Community Forum
3. **Enterprise Support:** GitHub Enterprise support channels

### Escalation Process

1. **Level 1:** Check workflow logs and status
2. **Level 2:** Review security alerts and compliance status
3. **Level 3:** Engage DevOps and security teams
4. **Level 4:** Contact GitHub support for platform issues

---

## Summary

This guide provides comprehensive instructions for setting up the RLHF Audit Trail GitHub workflows. The workflows ensure:

- ✅ **Code Quality:** Automated formatting, linting, and type checking
- ✅ **Security:** Comprehensive vulnerability scanning and SAST
- ✅ **Compliance:** EU AI Act and GDPR validation
- ✅ **Testing:** Unit, integration, and performance tests
- ✅ **Deployment:** Automated staging and production deployments
- ✅ **Monitoring:** Real-time alerts and status updates

Following these setup instructions will establish a robust, secure, and compliant CI/CD pipeline for the RLHF Audit Trail project.