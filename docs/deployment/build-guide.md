# RLHF Audit Trail - Build and Deployment Guide

## Overview

This guide covers building, containerizing, and deploying the RLHF Audit Trail system. The project uses Docker for containerization and provides automated scripts for building and releasing.

## Prerequisites

### Required Tools

- **Docker** (20.10+) with BuildKit enabled
- **Docker Compose** (2.0+) for multi-service orchestration
- **Git** for version control
- **Make** for build automation
- **bash** shell for running scripts

### Optional Tools

- **Docker Buildx** for multi-architecture builds
- **Trivy** or **Grype** for security scanning
- **GitHub CLI** (`gh`) for automated releases

### System Requirements

- **CPU**: 4+ cores recommended for build performance
- **RAM**: 8GB+ (16GB+ for multi-architecture builds)
- **Storage**: 10GB+ free space for Docker images and build cache
- **Network**: Internet access for downloading dependencies

## Build Targets

### Production Build

Optimized for production deployment with minimal size and security hardening:

```bash
# Quick build
docker build -t rlhf-audit-trail:latest .

# Using build script (recommended)
./scripts/build.sh

# With custom registry
./scripts/build.sh -r ghcr.io/terragonlabs -p
```

**Features:**
- Multi-stage build for minimal size
- Non-root user for security
- Health checks included
- Optimized for runtime performance

### Development Build

Includes development tools and debugging capabilities:

```bash
# Using Dockerfile.dev
docker build -f Dockerfile.dev -t rlhf-audit-trail:dev .

# Using build script
./scripts/build.sh -t development

# Using Docker Compose
docker-compose -f docker-compose.dev.yml up --build
```

**Features:**
- Development tools (vim, htop, tree)
- Pre-commit hooks
- Jupyter notebook support
- Hot reloading enabled

## Build Scripts

### Build Script (`scripts/build.sh`)

Comprehensive build automation with security scanning and multi-architecture support.

#### Basic Usage

```bash
# Production build
./scripts/build.sh

# Development build
./scripts/build.sh -t development

# Build and push to registry
./scripts/build.sh -p -r docker.io/terragonlabs
```

#### Advanced Options

```bash
# Multi-architecture build
./scripts/build.sh --multi-arch -p -r ghcr.io/terragonlabs

# Skip security scanning
./scripts/build.sh --no-scan

# Use build cache
./scripts/build.sh --cache-from ghcr.io/terragonlabs/rlhf-audit-trail:cache

# Custom version
./scripts/build.sh -v 1.2.3
```

#### Environment Variables

```bash
# Set default registry
export REGISTRY=ghcr.io/terragonlabs

# Enable pushing by default
export PUSH_IMAGE=true

# Disable security scanning
export SCAN_SECURITY=false
```

### Release Script (`scripts/release.sh`)

Automated release process with semantic versioning.

#### Basic Usage

```bash
# Patch release (1.0.0 -> 1.0.1)
./scripts/release.sh

# Minor release (1.0.0 -> 1.1.0)
./scripts/release.sh -t minor

# Major release (1.0.0 -> 2.0.0)
./scripts/release.sh -t major
```

#### Advanced Options

```bash
# Dry run to preview changes
./scripts/release.sh -t minor --dry-run

# Skip tests and build
./scripts/release.sh --skip-tests --skip-build

# Create GitHub release
./scripts/release.sh --github-release

# Custom registry
./scripts/release.sh -r ghcr.io/terragonlabs
```

## Docker Compose Deployments

### Development Environment

Complete development environment with all services:

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop environment
docker-compose -f docker-compose.dev.yml down
```

**Services included:**
- Main application (development mode)
- PostgreSQL database
- Redis cache
- Grafana monitoring
- Prometheus metrics

### Production Environment

Production-ready deployment with monitoring:

```bash
# Start production environment
docker-compose up -d

# Scale workers
docker-compose up -d --scale worker=3

# Update services
docker-compose pull
docker-compose up -d
```

**Services included:**
- Main application
- Celery workers
- Celery scheduler
- PostgreSQL database
- Redis cache
- Monitoring stack (Grafana, Prometheus)
- Nginx reverse proxy (optional)

### Environment-Specific Overrides

```bash
# Staging environment
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d

# Production with proxy
docker-compose --profile with-proxy up -d
```

## Build Optimization

### Multi-Stage Builds

The Dockerfile uses multi-stage builds for optimization:

1. **Builder Stage**: Compiles dependencies and installs packages
2. **Production Stage**: Minimal runtime image with only necessary components
3. **Development Stage**: Extended with development tools

### Build Cache

Leverage Docker build cache for faster builds:

```bash
# Use registry cache
./scripts/build.sh --cache-from ghcr.io/terragonlabs/rlhf-audit-trail:cache

# Save cache to registry
./scripts/build.sh --cache-to ghcr.io/terragonlabs/rlhf-audit-trail:cache
```

### Multi-Architecture Builds

Build for multiple CPU architectures:

```bash
# Build for AMD64 and ARM64
./scripts/build.sh --multi-arch

# Supported platforms
docker buildx ls
```

## Security Scanning

### Automated Scanning

Security scanning is integrated into the build process:

```bash
# Build with security scanning (default)
./scripts/build.sh

# Skip security scanning
./scripts/build.sh --no-scan
```

### Manual Scanning

Run security scans manually:

```bash
# Trivy scan
trivy image rlhf-audit-trail:latest

# Grype scan
grype rlhf-audit-trail:latest

# Docker Scout scan
docker scout cves rlhf-audit-trail:latest
```

### Security Reports

Security scan reports are saved to the `reports/` directory:

- `trivy-report.json` - Trivy vulnerability report
- `trivy-report.txt` - Human-readable Trivy report
- `grype-report.json` - Grype vulnerability report

## Deployment Strategies

### Single Node Deployment

For development and small-scale deployments:

```bash
# Using Docker Compose
docker-compose up -d

# Using Docker Swarm
docker stack deploy -c docker-compose.yml rlhf-stack
```

### Kubernetes Deployment

For production and scalable deployments:

```bash
# Apply Kubernetes manifests
kubectl apply -f deploy/kubernetes/

# Using Helm
helm install rlhf-audit-trail ./charts/rlhf-audit-trail
```

### Cloud Deployments

#### AWS ECS

```bash
# Build and push to ECR
./scripts/build.sh -r 123456789012.dkr.ecr.us-west-2.amazonaws.com -p

# Deploy to ECS
aws ecs update-service --cluster rlhf-cluster --service rlhf-service --force-new-deployment
```

#### Google Cloud Run

```bash
# Build and push to GCR
./scripts/build.sh -r gcr.io/project-id -p

# Deploy to Cloud Run
gcloud run deploy rlhf-audit-trail --image gcr.io/project-id/rlhf-audit-trail:latest
```

#### Azure Container Instances

```bash
# Build and push to ACR
./scripts/build.sh -r myregistry.azurecr.io -p

# Deploy to ACI
az container create --resource-group myResourceGroup --name rlhf-audit-trail --image myregistry.azurecr.io/rlhf-audit-trail:latest
```

## Monitoring and Observability

### Built-in Monitoring

The deployment includes comprehensive monitoring:

- **Grafana**: Visualization dashboards (port 3000)
- **Prometheus**: Metrics collection (port 9090)
- **Health Checks**: Application health monitoring
- **Structured Logging**: JSON-formatted logs

### Custom Metrics

The application exposes custom metrics:

```bash
# View metrics endpoint
curl http://localhost:8000/metrics

# Prometheus format
curl http://localhost:9090/metrics
```

### Log Aggregation

Logs are structured and can be aggregated using:

- ELK Stack (Elasticsearch, Logstash, Kibana)
- Grafana Loki
- Fluentd/Fluent Bit
- Cloud logging services (CloudWatch, Stackdriver, Azure Monitor)

## Troubleshooting

### Common Build Issues

#### Out of Memory During Build

```bash
# Increase Docker memory limit
# Or build with fewer parallel jobs
docker build --memory=4g --cpus=2 .
```

#### Build Context Too Large

```bash
# Check .dockerignore file
# Clean up unnecessary files
docker system prune -a
```

#### Permission Issues

```bash
# Fix file permissions
chmod +x scripts/*.sh

# Check Docker daemon permissions
sudo usermod -aG docker $USER
```

### Runtime Issues

#### Container Won't Start

```bash
# Check logs
docker logs rlhf-audit-trail-app

# Inspect container
docker inspect rlhf-audit-trail-app

# Check health status
docker ps --filter health=unhealthy
```

#### Database Connection Issues

```bash
# Check database status
docker-compose logs postgres

# Test connection
docker-compose exec app python -c "import psycopg2; print('DB OK')"
```

#### Performance Issues

```bash
# Monitor resource usage
docker stats

# Check system resources
docker system df
docker system events
```

### Debugging Tools

#### Development Container Access

```bash
# Access development container
docker-compose -f docker-compose.dev.yml exec dev bash

# Run interactive Python
docker-compose -f docker-compose.dev.yml exec dev python

# Access Jupyter notebook
open http://localhost:8888
```

#### Production Container Debugging

```bash
# Access running container
docker exec -it rlhf-audit-trail-app bash

# Run health check manually
docker exec rlhf-audit-trail-app python -c "import rlhf_audit_trail; print('OK')"
```

## Best Practices

### Build Best Practices

1. **Use .dockerignore**: Exclude unnecessary files from build context
2. **Multi-stage builds**: Separate build and runtime stages
3. **Layer caching**: Order instructions for optimal caching
4. **Security scanning**: Always scan images for vulnerabilities
5. **Non-root users**: Run containers as non-root users
6. **Health checks**: Include health check endpoints

### Deployment Best Practices

1. **Environment separation**: Use different configurations for dev/staging/prod
2. **Secret management**: Use proper secret management solutions
3. **Resource limits**: Set appropriate CPU and memory limits
4. **Monitoring**: Implement comprehensive monitoring and alerting
5. **Backup strategy**: Regular backups of data and configuration
6. **Rolling updates**: Use rolling deployment strategies

### Security Best Practices

1. **Regular updates**: Keep base images and dependencies updated
2. **Vulnerability scanning**: Regular security scans
3. **Network security**: Use network policies and firewalls
4. **Access control**: Implement proper authentication and authorization
5. **Audit logging**: Enable comprehensive audit logging
6. **Compliance**: Ensure regulatory compliance (EU AI Act, GDPR)

## CI/CD Integration

### GitHub Actions

```yaml
name: Build and Deploy
on:
  push:
    branches: [main]
    tags: [v*]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and push
        run: |
          ./scripts/build.sh -r ghcr.io/terragonlabs -p
        env:
          REGISTRY: ghcr.io/terragonlabs
```

### GitLab CI

```yaml
build:
  stage: build
  script:
    - ./scripts/build.sh -r $CI_REGISTRY -p
  variables:
    REGISTRY: $CI_REGISTRY_IMAGE
```

### Jenkins

```groovy
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh './scripts/build.sh -r ${DOCKER_REGISTRY} -p'
            }
        }
    }
}
```

## Support

For build and deployment issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review Docker and system logs
3. Consult the [development documentation](../DEVELOPMENT.md)
4. Open an issue in the GitHub repository

For production deployments, consider:

- Load testing before deployment
- Gradual rollout strategies
- Monitoring and alerting setup
- Disaster recovery planning
- Compliance validation