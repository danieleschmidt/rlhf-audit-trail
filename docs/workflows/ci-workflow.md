# CI/CD Workflow Configuration

Create `.github/workflows/ci.yml` with the following content:

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
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: make test
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: make lint

  deploy:
    needs: [test, lint]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    - name: Deploy to production
      run: ./scripts/deploy.sh production
      env:
        DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
```

## Required Secrets

Add these secrets in Repository Settings → Secrets:

- `DEPLOY_KEY`: SSH key for deployment access
- `CODECOV_TOKEN`: Code coverage reporting token

## Configuration Options

### Test Matrix
Customize Python versions in the matrix strategy based on your requirements.

### Deployment Targets
Modify deployment steps for your specific infrastructure:
- Kubernetes
- Docker containers
- Cloud platforms (AWS, GCP, Azure)

### Integration Options
Add additional steps for:
- Database migrations
- Cache invalidation
- CDN updates
- Notification services