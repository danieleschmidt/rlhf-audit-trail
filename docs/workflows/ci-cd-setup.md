# CI/CD Workflow Setup Guide

This document provides templates and setup instructions for GitHub Actions workflows for the RLHF Audit Trail project.

## Required Workflows

### 1. Continuous Integration (`.github/workflows/ci.yml`)

```yaml
name: CI

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
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,testing]"
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml --cov-report=html
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run linting
      run: |
        black --check .
        ruff check .
        mypy src

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run security checks
      run: |
        pip install bandit safety
        bandit -r src/
        safety check
```

### 2. Security Scanning (`.github/workflows/security.yml`)

```yaml
name: Security

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  dependency-review:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    - name: Dependency Review
      uses: actions/dependency-review-action@v3
```

### 3. Release Workflow (`.github/workflows/release.yml`)

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
        python-version: "3.10"
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false
```

## Setup Instructions

### Repository Secrets

Add these secrets in GitHub Settings > Secrets and variables > Actions:

- `PYPI_API_TOKEN`: PyPI API token for package publishing
- `CODECOV_TOKEN`: Codecov token for coverage reporting

### Branch Protection Rules

Enable branch protection for `main` branch with:

- Require pull request reviews
- Require status checks to pass before merging:
  - `test (3.10)`
  - `test (3.11)` 
  - `test (3.12)`
  - `lint`
  - `security / codeql`
- Allow force pushes: disabled
- Allow deletions: disabled

### GitHub Advanced Security

Enable in repository settings:

- Dependency graph
- Dependabot alerts
- Dependabot security updates
- Code scanning (CodeQL)
- Secret scanning
- Private vulnerability reporting

## Compliance Considerations

### EU AI Act Requirements

- Automated compliance testing in CI
- Model card validation
- Privacy budget monitoring
- Audit trail integrity checks

### Security Standards

- SAST (Static Application Security Testing)
- Dependency vulnerability scanning
- Container security scanning (if applicable)
- License compliance checking

## Manual Setup Steps

1. **Create workflow files**: Copy the YAML templates to `.github/workflows/`
2. **Configure secrets**: Add required tokens and API keys
3. **Enable branch protection**: Configure protection rules for main branch
4. **Enable security features**: Turn on GitHub Advanced Security features
5. **Test workflows**: Push a commit to trigger initial workflow runs

## Monitoring and Maintenance

- Review Dependabot PRs weekly
- Update workflow dependencies monthly
- Monitor security alerts and address promptly
- Review and update compliance checks quarterly
