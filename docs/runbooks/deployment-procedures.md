# RLHF Audit Trail - Deployment Procedures

## Overview

This runbook provides step-by-step procedures for deploying the RLHF Audit Trail system across different environments.

## Pre-Deployment Checklist

### Code Quality Gates
```bash
# Run all tests
make test-all

# Security scanning
make security-scan

# Compliance validation
make compliance-check

# Build verification
make build

# Container security scan
make docker-security-scan
```

### Environment Preparation
- [ ] Infrastructure provisioned
- [ ] Secrets configured
- [ ] DNS records updated
- [ ] SSL certificates valid
- [ ] Monitoring configured
- [ ] Backup systems ready

## Development Environment

### Local Development Setup
```bash
# Clone repository
git clone https://github.com/danieleschmidt/rlhf-audit-trail.git
cd rlhf-audit-trail

# Setup environment
make dev-setup

# Start local stack
docker-compose -f docker-compose.dev.yml up -d

# Run migrations
make migrate

# Verify deployment
curl http://localhost:8000/health
```

### Development Deploy
```bash
# Build and deploy
make deploy-dev

# Verify services
make verify-dev

# Run smoke tests
make test-smoke
```

## Staging Environment

### Pre-Staging Validation
```bash
# Validate configuration
kubectl apply --dry-run=client -f k8s/staging/

# Check resource quotas
kubectl describe quota -n rlhf-staging

# Verify secrets
kubectl get secrets -n rlhf-staging
```

### Staging Deployment
```bash
# Deploy infrastructure
kubectl apply -f k8s/staging/namespace.yaml
kubectl apply -f k8s/staging/configmaps.yaml
kubectl apply -f k8s/staging/secrets.yaml

# Deploy database
kubectl apply -f k8s/staging/postgresql.yaml
kubectl wait --for=condition=Ready pod -l app=postgresql -n rlhf-staging --timeout=300s

# Run database migrations
kubectl exec -it postgresql-0 -n rlhf-staging -- python manage.py migrate

# Deploy application
kubectl apply -f k8s/staging/deployment.yaml
kubectl apply -f k8s/staging/service.yaml
kubectl apply -f k8s/staging/ingress.yaml

# Wait for rollout
kubectl rollout status deployment/rlhf-audit-trail -n rlhf-staging

# Verify deployment
kubectl get pods -n rlhf-staging
```

### Staging Verification
```bash
# Health check
curl -f https://staging.rlhf-audit-trail.example.com/health

# API verification
curl -f https://staging.rlhf-audit-trail.example.com/api/v1/status

# Database connectivity
kubectl exec -it postgresql-0 -n rlhf-staging -- pg_isready

# Run integration tests
make test-integration-staging
```

## Production Environment

### Pre-Production Gates
- [ ] Code reviewed and approved
- [ ] Security scan passed
- [ ] Compliance validation passed
- [ ] Staging deployment successful
- [ ] Integration tests passed
- [ ] Performance benchmarks met
- [ ] Change approval received
- [ ] Rollback plan prepared

### Blue-Green Deployment Strategy

**Phase 1: Prepare Green Environment**
```bash
# Create green namespace
kubectl apply -f k8s/production/namespace-green.yaml

# Deploy green infrastructure
kubectl apply -f k8s/production/configmaps-green.yaml
kubectl apply -f k8s/production/secrets-green.yaml

# Deploy green database (if schema changes)
kubectl apply -f k8s/production/postgresql-green.yaml

# Deploy green application
kubectl apply -f k8s/production/deployment-green.yaml

# Wait for green deployment
kubectl rollout status deployment/rlhf-audit-trail-green -n rlhf-production
```

**Phase 2: Green Environment Verification**
```bash
# Internal health check
kubectl port-forward service/rlhf-audit-trail-green 8080:8000 -n rlhf-production &
curl -f http://localhost:8080/health

# Database migration test (if applicable)
kubectl exec -it rlhf-audit-trail-green-pod -n rlhf-production -- python manage.py migrate --check

# Run smoke tests against green
make test-smoke-green
```

**Phase 3: Traffic Switch**
```bash
# Update ingress to point to green
kubectl patch ingress rlhf-audit-trail -n rlhf-production -p '{"spec":{"rules":[{"host":"api.rlhf-audit-trail.com","http":{"paths":[{"path":"/","pathType":"Prefix","backend":{"service":{"name":"rlhf-audit-trail-green","port":{"number":8000}}}}]}}]}}'

# Monitor metrics immediately
watch -n 5 'kubectl top pods -n rlhf-production'

# Monitor error rates
curl -s http://prometheus:9090/api/v1/query?query='rate(http_requests_total{status=~"5.."}[5m])'
```

**Phase 4: Validation and Cleanup**
```bash
# Monitor for 10 minutes
sleep 600

# Check error rates and performance
make verify-production-metrics

# If successful, cleanup blue environment
kubectl delete deployment rlhf-audit-trail-blue -n rlhf-production

# Update labels for next deployment
kubectl label deployment rlhf-audit-trail-green deployment-color=blue -n rlhf-production --overwrite
```

### Rolling Deployment Strategy (Alternative)

```bash
# Update deployment with rolling update
kubectl set image deployment/rlhf-audit-trail rlhf-audit-trail=rlhf-audit-trail:v1.2.3 -n rlhf-production

# Monitor rolling update
kubectl rollout status deployment/rlhf-audit-trail -n rlhf-production

# Verify new pods
kubectl get pods -l app=rlhf-audit-trail -n rlhf-production
```

### Production Verification
```bash
# Health checks
curl -f https://api.rlhf-audit-trail.com/health
curl -f https://api.rlhf-audit-trail.com/api/v1/status

# Audit trail integrity
curl -f https://api.rlhf-audit-trail.com/api/v1/audit/integrity/verify

# Privacy engine status  
curl -f https://api.rlhf-audit-trail.com/api/v1/privacy/status

# Compliance status
curl -f https://api.rlhf-audit-trail.com/api/v1/compliance/status

# Performance metrics
curl -f https://api.rlhf-audit-trail.com/metrics
```

## Database Migration Procedures

### Schema Migrations
```bash
# Generate migration
python manage.py makemigrations --name="descriptive_name"

# Review migration file
cat migrations/XXXX_descriptive_name.py

# Test migration on copy of production data
python manage.py migrate --database=test_db

# Apply to staging
kubectl exec -it postgresql-0 -n rlhf-staging -- python manage.py migrate

# Apply to production (during maintenance window)
kubectl exec -it postgresql-0 -n rlhf-production -- python manage.py migrate
```

### Data Migrations
```bash
# Create data migration
python manage.py makemigrations --empty --name="data_migration_name" app_name

# Test data migration
python manage.py migrate --database=test_db

# Backup before migration
kubectl exec -it postgresql-0 -n rlhf-production -- pg_dump -U postgres rlhf_audit > backup_pre_migration.sql

# Apply data migration
kubectl exec -it postgresql-0 -n rlhf-production -- python manage.py migrate
```

## Rollback Procedures

### Application Rollback
```bash
# Check deployment history
kubectl rollout history deployment/rlhf-audit-trail -n rlhf-production

# Rollback to previous version
kubectl rollout undo deployment/rlhf-audit-trail -n rlhf-production

# Rollback to specific revision
kubectl rollout undo deployment/rlhf-audit-trail --to-revision=2 -n rlhf-production

# Monitor rollback
kubectl rollout status deployment/rlhf-audit-trail -n rlhf-production
```

### Database Rollback
```bash
# Stop application (prevent new writes)
kubectl scale deployment/rlhf-audit-trail --replicas=0 -n rlhf-production

# Restore from backup
kubectl exec -it postgresql-0 -n rlhf-production -- psql -U postgres -c "DROP DATABASE IF EXISTS rlhf_audit;"
kubectl exec -it postgresql-0 -n rlhf-production -- psql -U postgres -c "CREATE DATABASE rlhf_audit;"
kubectl exec -i postgresql-0 -n rlhf-production -- psql -U postgres rlhf_audit < backup_pre_migration.sql

# Restart application
kubectl scale deployment/rlhf-audit-trail --replicas=3 -n rlhf-production
```

## Monitoring During Deployment

### Key Metrics to Monitor
```bash
# Application metrics
curl -s http://prometheus:9090/api/v1/query?query='rate(http_requests_total[5m])'
curl -s http://prometheus:9090/api/v1/query?query='http_request_duration_seconds_bucket'
curl -s http://prometheus:9090/api/v1/query?query='up{job="rlhf-audit-trail"}'

# System metrics
kubectl top nodes
kubectl top pods -n rlhf-production

# Business metrics
curl -s http://localhost:8000/api/v1/metrics/business
```

### Alert Thresholds During Deployment
- Error rate > 1%
- Response time > 500ms
- CPU usage > 80%
- Memory usage > 85%
- Database connections > 90% of pool

## Post-Deployment Tasks

### Immediate (0-30 minutes)
- [ ] Verify all services healthy
- [ ] Check error rates
- [ ] Monitor performance metrics
- [ ] Validate core functionality
- [ ] Update status page

### Short-term (30 minutes - 2 hours)
- [ ] Run full test suite
- [ ] Monitor business metrics
- [ ] Check audit trail integrity
- [ ] Validate compliance status
- [ ] Review logs for errors

### Medium-term (2-24 hours)
- [ ] Performance analysis
- [ ] User feedback collection
- [ ] Resource utilization review
- [ ] Security scan results
- [ ] Compliance verification

## Emergency Procedures

### Deployment Failure
1. **Stop deployment:** `kubectl rollout pause deployment/rlhf-audit-trail`
2. **Assess impact:** Check monitoring dashboards
3. **Quick fix or rollback:** Depending on issue severity
4. **Communication:** Update stakeholders
5. **Investigation:** Root cause analysis

### Security Incident During Deployment
1. **Halt deployment immediately**
2. **Isolate affected systems**  
3. **Notify security team**
4. **Preserve evidence**
5. **Incident response procedures**

## Deployment Checklist Template

```markdown
## Deployment Checklist - v[VERSION] - [DATE]

### Pre-Deployment
- [ ] Code review complete
- [ ] Tests passing
- [ ] Security scan clean
- [ ] Change approval received
- [ ] Rollback plan prepared
- [ ] Team notified

### Deployment
- [ ] Staging deployment successful
- [ ] Production deployment started
- [ ] Health checks passing
- [ ] Metrics looking normal
- [ ] Core functionality verified

### Post-Deployment  
- [ ] Performance stable
- [ ] Error rates normal
- [ ] Business metrics healthy
- [ ] Team notified of success
- [ ] Documentation updated

### Issues Encountered
[Document any issues and resolutions]

### Lessons Learned
[Document improvements for next deployment]
```

## Environment Configuration

### Development
- Replicas: 1
- Resource limits: Low
- Debug logging: Enabled
- External services: Mocked

### Staging  
- Replicas: 2
- Resource limits: Medium
- Debug logging: Enabled
- External services: Staging equivalents

### Production
- Replicas: 3+ (auto-scaling)
- Resource limits: Production values
- Debug logging: Disabled
- External services: Production instances

## Troubleshooting Common Issues

### Pod Won't Start
```bash
# Check pod events
kubectl describe pod <pod-name> -n <namespace>

# Check logs
kubectl logs <pod-name> -n <namespace>

# Check resource constraints
kubectl top pod <pod-name> -n <namespace>
```

### Service Unreachable
```bash
# Check service
kubectl get svc -n <namespace>

# Check endpoints
kubectl get endpoints -n <namespace>

# Test connectivity
kubectl exec -it <pod-name> -- nslookup <service-name>
```

### Database Connection Issues
```bash
# Check database pod
kubectl get pods -l app=postgresql -n <namespace>

# Check service
kubectl get svc postgresql -n <namespace>

# Test connection
kubectl exec -it <app-pod> -- pg_isready -h postgresql -p 5432
```