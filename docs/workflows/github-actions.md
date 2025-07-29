# GitHub Actions Workflows

This document describes the recommended GitHub Actions workflows for the RLHF Audit Trail project.

## Required Workflows

### 1. CI/CD Pipeline (`.github/workflows/ci.yml`)

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run tests
      run: pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.10
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run linting
      run: |
        black --check .
        ruff check .
        mypy src/

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.10
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety
    
    - name: Run security checks
      run: |
        bandit -r src/
        safety check
```

### 2. Security Scanning (`.github/workflows/security.yml`)

```yaml
name: Security Scan

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  push:
    branches: [ main ]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Dependency Review
      uses: actions/dependency-review-action@v3
```

### 3. Release Automation (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.10
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Security Configuration

### Branch Protection Rules

Configure the following branch protection rules for `main`:

- Require pull request reviews before merging
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Include administrators in restrictions
- Restrict pushes that create files matching `*.key`, `*.pem`, `secrets.*`

### Required Status Checks

- `test (3.10)`
- `test (3.11)` 
- `test (3.12)`
- `lint`
- `security`

### Secrets Configuration

Required repository secrets:

- `PYPI_API_TOKEN`: For automated releases
- `CODECOV_TOKEN`: For coverage reporting

## Workflow Integration

### Pre-commit Integration

Ensure all workflows run the same checks as pre-commit hooks:

1. Install pre-commit: `pre-commit install`
2. Run locally: `pre-commit run --all-files`
3. Workflows use identical tool versions

### Manual Workflow Triggers

Some workflows support manual triggering:

```bash
# Trigger security scan
gh workflow run security.yml

# Trigger full CI pipeline
gh workflow run ci.yml
```

## Monitoring and Notifications

### Slack Integration

Configure Slack notifications for:
- Failed CI builds
- Security vulnerability discoveries
- Successful releases

### Email Notifications

Set up email alerts for:
- Dependency vulnerabilities
- Failed security scans
- Release completions

## Compliance Automation

### EU AI Act Compliance

Add workflow steps for:
- Model card validation
- Audit trail verification
- Compliance report generation

### Privacy Testing

Include privacy-specific tests:
- Differential privacy parameter validation
- Anonymization effectiveness testing
- Data leakage detection

## Performance Testing

### Benchmark Workflows

Regular performance testing:
- RLHF training overhead measurement
- Storage backend performance
- Cryptographic operation timing

## Documentation Updates

Automatic documentation:
- API documentation generation
- Compliance report updates
- Security posture documentation