# GitHub Actions Workflows

This directory contains documentation for recommended GitHub Actions workflows for this repository. These workflows need to be manually created by a repository administrator with appropriate permissions.

## Required Setup

1. **Repository Settings**: Ensure GitHub Actions are enabled
2. **Permissions**: Administrator must have `workflows` permission to create these files
3. **Secrets**: Configure required repository secrets (see each workflow documentation)

## Available Workflows

### 1. CI/CD Pipeline (`ci.yml`)
- **Purpose**: Continuous integration and deployment
- **Triggers**: Push to main, pull requests
- **Actions**: Test, lint, build, deploy
- **Location**: `.github/workflows/ci.yml`
- **Documentation**: [ci-workflow.md](./ci-workflow.md)

### 2. Security Scanning (`security.yml`)
- **Purpose**: Automated security vulnerability scanning
- **Triggers**: Push, scheduled daily
- **Actions**: Dependency check, code analysis, secret scanning
- **Location**: `.github/workflows/security.yml`
- **Documentation**: [security-workflow.md](./security-workflow.md)

### 3. Compliance Validation (`compliance.yml`)
- **Purpose**: EU AI Act and regulatory compliance checks
- **Triggers**: Push to main, release
- **Actions**: Compliance validation, audit trail generation
- **Location**: `.github/workflows/compliance.yml`
- **Documentation**: [compliance-workflow.md](./compliance-workflow.md)

### 4. Release Automation (`release.yml`)
- **Purpose**: Automated release management and SBOM generation
- **Triggers**: Version tags, manual dispatch
- **Actions**: Build, test, package, create release
- **Location**: `.github/workflows/release.yml`
- **Documentation**: [release-workflow.md](./release-workflow.md)

### 5. Dependency Management (`dependabot-auto-approve.yml`)
- **Purpose**: Automated dependency updates with intelligent approval
- **Triggers**: Dependabot PRs
- **Actions**: Auto-approve safe dependency updates
- **Location**: `.github/workflows/dependabot-auto-approve.yml`
- **Documentation**: [dependabot-workflow.md](./dependabot-workflow.md)

## Manual Setup Instructions

1. **Create Workflow Directory**:
   ```bash
   mkdir -p .github/workflows
   ```

2. **Copy Workflow Files**:
   - Follow the documentation in each workflow file
   - Customize configurations for your specific needs
   - Test workflows in a development branch first

3. **Configure Secrets**:
   - Go to Repository Settings → Secrets and variables → Actions
   - Add required secrets as documented in each workflow

4. **Configure Dependabot** (Optional):
   - Create `.github/dependabot.yml`
   - Follow [dependabot-config.md](./dependabot-config.md)

## Deployment Strategies

The workflows support multiple deployment strategies:

### Rolling Deployment
- Zero-downtime updates
- Gradual instance replacement
- Quick rollback capability

### Blue-Green Deployment
- Complete environment switch
- Immediate rollback
- Higher resource requirements

### Canary Deployment
- Traffic splitting
- Risk mitigation
- Gradual rollout

See [deployment-strategies.md](./deployment-strategies.md) for detailed configuration.

## Monitoring and Alerts

All workflows integrate with:
- **Prometheus**: Metrics collection
- **Grafana**: Dashboard visualization
- **AlertManager**: Incident notifications

Configuration files are in the `monitoring/` directory.

## Troubleshooting

Common issues and solutions:

1. **Permission Denied**: Ensure admin has `workflows` permission
2. **Secret Not Found**: Check repository secrets configuration
3. **Workflow Failed**: Review logs and dependency requirements

See [troubleshooting.md](./troubleshooting.md) for detailed solutions.