# Manual Setup Required

Due to GitHub App permission limitations, the following setup steps must be performed manually by repository maintainers after implementing the SDLC checkpoints.

## Required GitHub Workflows

The following workflow files need to be manually copied from `docs/workflows/examples/` to `.github/workflows/`:

### Core CI/CD Workflows
```bash
cp docs/workflows/examples/ci.yml .github/workflows/ci.yml
cp docs/workflows/examples/cd.yml .github/workflows/cd.yml
cp docs/workflows/examples/dependency-update.yml .github/workflows/dependency-update.yml
```

### Security & Compliance Workflows  
```bash
cp docs/workflows/examples/security-scan.yml .github/workflows/security-scan.yml
cp docs/workflows/examples/slsa-provenance.yml .github/workflows/slsa-provenance.yml
cp docs/workflows/examples/compliance-audit.yml .github/workflows/compliance-audit.yml
```

## Repository Settings Configuration

### Branch Protection Rules
Configure the following branch protection rules for the `main` branch:

#### Required Status Checks
- [ ] `build` - Build and test pipeline
- [ ] `security-scan` - Security vulnerability scanning
- [ ] `compliance-check` - Regulatory compliance validation
- [ ] `code-quality` - Code quality and linting checks

#### Additional Protection Settings
- [ ] **Require pull request reviews before merging** (minimum 2 reviewers)
- [ ] **Require review from code owners** (uses CODEOWNERS file)
- [ ] **Dismiss stale PR approvals when new commits are pushed**
- [ ] **Require status checks to pass before merging**
- [ ] **Require branches to be up to date before merging**
- [ ] **Require signed commits**
- [ ] **Include administrators** (enforce rules for admins)
- [ ] **Restrict pushes that create files larger than 100MB**

### Required Repository Secrets

Configure the following secrets in **Settings > Secrets and variables > Actions**:

#### Container Registry
- `GHCR_TOKEN` - GitHub Container Registry token for image publishing

#### Cloud Provider Secrets (choose one)
**AWS:**
- `AWS_ACCESS_KEY_ID` - AWS access key for S3/infrastructure
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `AWS_REGION` - Default AWS region (e.g., `us-east-1`)

**Google Cloud:**
- `GCP_SA_KEY` - Service account key JSON (base64 encoded)
- `GCP_PROJECT_ID` - Google Cloud project ID

**Azure:**
- `AZURE_CLIENT_ID` - Azure service principal client ID
- `AZURE_CLIENT_SECRET` - Azure service principal secret
- `AZURE_TENANT_ID` - Azure tenant ID

#### Security & Compliance
- `GPG_PRIVATE_KEY` - GPG private key for artifact signing
- `GPG_PASSPHRASE` - GPG key passphrase
- `SECURITY_SCAN_TOKEN` - Token for security scanning services
- `SLACK_WEBHOOK` - Slack webhook for notifications (optional)

#### Database & Infrastructure
- `DB_PASSWORD` - Production database password
- `REDIS_PASSWORD` - Redis password (if using authentication)
- `ENCRYPTION_KEY` - Key for encrypting sensitive data at rest

### Repository Variables

Configure these variables in **Settings > Secrets and variables > Actions > Variables**:

#### General
- `PYTHON_VERSION` - Python version to use (default: `3.10`)
- `NODE_VERSION` - Node.js version for tooling (default: `18`)

#### Infrastructure
- `REGISTRY_URL` - Container registry URL (default: `ghcr.io`)
- `K8S_CLUSTER_NAME` - Kubernetes cluster name
- `ENVIRONMENT_PREFIX` - Environment prefix for resource naming

#### Compliance
- `COMPLIANCE_FRAMEWORKS` - Comma-separated list (e.g., `eu_ai_act,gdpr,nist_ai_rmf`)
- `AUDIT_LOG_RETENTION_DAYS` - Log retention period (default: `2555` for EU AI Act)

## Code Owners Configuration

Create or update the `.github/CODEOWNERS` file with appropriate reviewers:

```
# Global code owners
* @security-team @compliance-team

# Core application code
/src/ @engineering-team @security-team

# Infrastructure and deployment
/k8s/ @devops-team @security-team
/docker/ @devops-team
/.github/workflows/ @devops-team @security-team

# Compliance and security
/compliance/ @compliance-team @legal-team
/security/ @security-team
/docs/compliance/ @compliance-team

# Documentation
/docs/ @technical-writers @compliance-team

# Configuration files
pyproject.toml @engineering-lead
requirements*.txt @engineering-lead
Dockerfile @devops-team
```

## Dependabot Configuration

Enable Dependabot by creating `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    target-branch: "develop"
    reviewers:
      - "engineering-team"
    labels:
      - "dependencies"
      - "automated"
    
  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    reviewers:
      - "devops-team"
    labels:
      - "github-actions"
      - "automated"

  # Docker
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "devops-team"
    labels:
      - "docker"
      - "automated"
```

## Issue and Pull Request Templates

### Issue Templates
Create `.github/ISSUE_TEMPLATE/` directory with:

#### Bug Report (`bug_report.yml`)
```yaml
name: Bug Report
description: Report a bug or unexpected behavior
title: "[BUG] "
labels: ["bug", "triage"]
body:
  - type: textarea
    attributes:
      label: Bug Description
      description: Clear description of the bug
    validations:
      required: true
      
  - type: textarea
    attributes:
      label: Steps to Reproduce
      description: Step-by-step instructions to reproduce
    validations:
      required: true
      
  - type: textarea
    attributes:
      label: Expected Behavior
      description: What you expected to happen
      
  - type: textarea
    attributes:
      label: Compliance Impact
      description: Does this affect regulatory compliance?
      
  - type: checkboxes
    attributes:
      label: Security Impact
      options:
        - label: This bug has security implications
        - label: This bug affects audit trail integrity
        - label: This bug impacts privacy protection
```

#### Security Issue (`security_issue.yml`)
```yaml
name: Security Issue
description: Report a security vulnerability (use private disclosure for critical issues)
title: "[SECURITY] "
labels: ["security", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        ⚠️ **For critical security vulnerabilities, please use private disclosure.**
        
  - type: textarea
    attributes:
      label: Vulnerability Description
      description: Detailed description of the security issue
    validations:
      required: true
      
  - type: dropdown
    attributes:
      label: Severity
      options:
        - Low
        - Medium
        - High
        - Critical
    validations:
      required: true
```

### Pull Request Template
Create `.github/pull_request_template.md`:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)  
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Security enhancement
- [ ] Compliance improvement

## Compliance Checklist
- [ ] Changes maintain EU AI Act compliance
- [ ] Privacy impact assessed (GDPR compliance)
- [ ] Security review completed
- [ ] Audit trail integrity preserved
- [ ] Documentation updated

## Testing Checklist
- [ ] Unit tests added/updated and passing
- [ ] Integration tests passing
- [ ] Security tests passing  
- [ ] Compliance tests passing
- [ ] Performance impact assessed

## Security Checklist
- [ ] No hardcoded secrets or credentials
- [ ] Input validation implemented
- [ ] Authorization checks implemented
- [ ] Audit logging included
- [ ] Error handling doesn't leak sensitive information

## Review Checklist
- [ ] Code follows project standards
- [ ] Documentation is clear and complete
- [ ] Breaking changes are documented
- [ ] Migration path provided for breaking changes
```

## Monitoring and Alerting Setup

### Required External Integrations

#### Status Page (optional)
If using a public status page service:
- Configure webhook endpoints for automated status updates
- Set up monitoring dashboards for public visibility
- Configure escalation procedures for incidents

#### Notification Services
Configure notification channels:
- **Slack:** Create dedicated channels for alerts (`#security-alerts`, `#compliance-alerts`)
- **Email:** Set up distribution lists for different alert types
- **PagerDuty/OpsGenie:** Configure escalation policies for critical alerts

### Monitoring Dashboards
Set up the following dashboards in your monitoring system:

#### Security Dashboard
- Failed authentication attempts
- Unusual access patterns
- Privacy budget utilization
- Audit trail integrity status
- Certificate expiration dates

#### Compliance Dashboard  
- Compliance score trends
- Failed compliance checks
- Data retention metrics
- Privacy impact assessments
- Regulatory reporting status

#### Operational Dashboard
- System health metrics
- Performance indicators
- Error rates and response times
- Resource utilization
- Backup status and integrity

## Compliance Documentation Updates

### Required Legal Reviews
Schedule reviews with legal team for:
- Data Processing Agreements (DPA)
- Privacy Impact Assessments (PIA)
- Terms of Service updates
- Cookie and privacy policies

### Regulatory Notifications
Prepare notifications for relevant regulatory bodies:
- Data Protection Authorities (for GDPR compliance)
- AI regulatory bodies (for EU AI Act compliance)
- Industry-specific regulators

## Final Verification Steps

After completing all setup steps:

1. **Test all workflows** by creating a test pull request
2. **Verify security scans** are running and reporting correctly  
3. **Confirm compliance checks** are executing and generating reports
4. **Test notification systems** with dummy alerts
5. **Validate backup and recovery** procedures
6. **Run full compliance audit** to establish baseline
7. **Document any customizations** made to the templates

## Support and Documentation

- **Internal Documentation:** Update internal runbooks with any custom configurations
- **Training:** Schedule training sessions for team members on new workflows
- **Support Contacts:** Establish clear escalation paths for workflow issues
- **Regular Reviews:** Schedule quarterly reviews of repository settings and workflows

---

**Note:** This setup must be completed by users with appropriate repository administration privileges. Some steps may require organization-level permissions for enterprise features.