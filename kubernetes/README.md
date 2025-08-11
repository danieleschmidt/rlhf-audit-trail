# RLHF Audit Trail Kubernetes Deployment

This directory contains comprehensive Kubernetes manifests for deploying the RLHF Audit Trail system in production environments.

## Overview

The deployment includes:

- **Production-ready application deployment** with security hardening
- **PostgreSQL database** with persistent storage and monitoring
- **Auto-scaling** (HPA/VPA) based on CPU, memory, and custom metrics
- **Monitoring integration** with Prometheus and Grafana
- **Security policies** including NetworkPolicies and RBAC
- **Multi-environment support** through Kustomize overlays

## Quick Start

### Prerequisites

- Kubernetes cluster (v1.25+)
- kubectl configured
- kustomize (v5.0+)
- Helm (optional, for monitoring stack)

### Basic Deployment

```bash
# Deploy with default settings
./deploy.sh deploy

# Deploy to staging environment
./deploy.sh deploy --env staging --replicas 2

# Check deployment status
./deploy.sh status

# View application logs
./deploy.sh logs
```

### Custom Configuration

```bash
# Deploy with custom image and settings
./deploy.sh deploy \
  --image 1.1.0 \
  --replicas 5 \
  --env production \
  --namespace rlhf-prod
```

## Architecture

### Components

1. **Application Pods**
   - RLHF Audit Trail main application
   - Gunicorn WSGI server with Uvicorn workers
   - Health checks and graceful shutdown
   - Resource limits and security contexts

2. **Database**
   - PostgreSQL 15 with persistent storage
   - Connection pooling and performance tuning
   - Backup and monitoring integration
   - SSL/TLS encryption

3. **Storage**
   - Persistent volumes for application data
   - Audit data storage with compliance retention
   - Log storage with rotation

4. **Networking**
   - Service mesh ready (Istio compatible)
   - Ingress with TLS termination
   - Network policies for security
   - Load balancing across pods

### Security Features

- **Container Security**
  - Non-root user execution
  - Read-only root filesystem
  - Dropped Linux capabilities
  - Security contexts and AppArmor/SELinux profiles

- **Network Security**
  - Network policies restricting pod-to-pod communication
  - TLS encryption for all external traffic
  - Internal service mesh for microservice communication

- **Secrets Management**
  - Kubernetes secrets for sensitive data
  - External secret management integration ready
  - Automatic secret rotation capabilities

## File Structure

```
kubernetes/
├── namespace.yaml          # Namespace and resource quotas
├── secrets.yaml           # Application secrets (template)
├── configmaps.yaml        # Configuration data
├── postgresql.yaml        # Database deployment
├── deployment.yaml        # Main application deployment
├── autoscaling.yaml       # HPA, VPA, and monitoring
├── kustomization.yaml     # Kustomize configuration
├── deploy.sh              # Deployment script
└── README.md              # This file
```

## Configuration

### Environment Variables

The application is configured through environment variables defined in `configmaps.yaml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `RLHF_AUDIT_ENV` | `production` | Application environment |
| `RLHF_AUDIT_LOG_LEVEL` | `INFO` | Logging level |
| `RLHF_AUDIT_WORKERS` | `4` | Number of worker processes |
| `RLHF_AUDIT_COMPLIANCE_MODE` | `eu_ai_act` | Compliance framework |
| `RLHF_AUDIT_REGION` | `EU` | Deployment region |

### Secrets

Sensitive configuration is stored in Kubernetes secrets. Update `secrets.yaml` with actual values:

```yaml
stringData:
  db-password: "your-secure-database-password"
  secret-key: "your-application-secret-key"
  encryption-key: "your-encryption-key"
```

### Resource Requirements

**Application Container:**
- Requests: 200m CPU, 512Mi memory
- Limits: 2000m CPU, 4Gi memory

**Database Container:**
- Requests: 200m CPU, 512Mi memory  
- Limits: 1000m CPU, 2Gi memory

**Storage:**
- Application data: 20Gi
- Audit data: 50Gi
- Logs: 10Gi
- Database: 20Gi

## Deployment Environments

### Development

```bash
./deploy.sh deploy --env dev --replicas 1 --namespace rlhf-dev
```

**Characteristics:**
- Single replica
- Reduced resource limits
- Debug logging enabled
- Hot reload for development

### Staging

```bash
./deploy.sh deploy --env staging --replicas 2 --namespace rlhf-staging
```

**Characteristics:**
- 2 replicas for availability testing
- Production-like configuration
- Synthetic data for testing
- Monitoring enabled

### Production

```bash
./deploy.sh deploy --env production --replicas 3 --namespace rlhf-production
```

**Characteristics:**
- 3+ replicas with anti-affinity
- Full resource allocation
- Enhanced security policies
- Comprehensive monitoring and alerting

## Auto-scaling Configuration

### Horizontal Pod Autoscaler (HPA)

The HPA scales pods based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- Custom metrics (requests/second, queue length)

**Scaling Behavior:**
- Min replicas: 2
- Max replicas: 20
- Scale up: 50% increase, max 4 pods per minute
- Scale down: 10% decrease, max 2 pods per minute

### Vertical Pod Autoscaler (VPA)

The VPA adjusts resource requests/limits based on actual usage:
- CPU: 100m - 4000m
- Memory: 256Mi - 8Gi
- Update mode: Auto (with rolling updates)

## Monitoring and Observability

### Metrics Collection

The deployment includes Prometheus ServiceMonitor for metrics collection:

```yaml
endpoints:
- port: metrics
  interval: 30s
  path: /metrics
```

**Application Metrics:**
- HTTP request metrics (rate, latency, errors)
- Privacy budget consumption
- Audit trail integrity metrics
- Database connection health

### Health Checks

**Liveness Probe:** `/health`
- Initial delay: 30s
- Period: 30s
- Timeout: 10s
- Failure threshold: 3

**Readiness Probe:** `/health/ready`
- Initial delay: 10s
- Period: 5s
- Timeout: 5s
- Failure threshold: 3

**Startup Probe:** `/health`
- Initial delay: 10s
- Period: 10s
- Failure threshold: 30

### Alerting Rules

Critical alerts configured in PrometheusRule:

1. **High Error Rate** (>10% for 5min)
2. **High Latency** (95th percentile >2s)
3. **Pod Crash Looping**
4. **Database Connection Failure**
5. **Privacy Budget Low** (<10% remaining)
6. **Audit Integrity Violations**
7. **Disk Space Low** (<10% remaining)

## Backup and Recovery

### Database Backup

Automated backup script included in ConfigMap:

```bash
# Manual backup
kubectl exec -n rlhf-audit-trail deployment/postgresql -- \
  bash /scripts/backup.sh
```

### Application Data Backup

Persistent volumes should be backed up regularly:

```bash
# Create volume snapshot
kubectl create -f - <<EOF
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: rlhf-audit-data-snapshot
  namespace: rlhf-audit-trail
spec:
  source:
    persistentVolumeClaimName: rlhf-audit-trail-audit-pvc
  volumeSnapshotClassName: csi-hostpath-snapclass
EOF
```

### Disaster Recovery

For complete disaster recovery:

1. **Backup Kubernetes manifests** to version control
2. **Export secrets** to secure location
3. **Snapshot persistent volumes** regularly
4. **Document recovery procedures**
5. **Test recovery process** periodically

## Troubleshooting

### Common Issues

#### Pod Not Starting

```bash
# Check pod status and events
kubectl describe pod -n rlhf-audit-trail -l app.kubernetes.io/name=rlhf-audit-trail

# Check logs
kubectl logs -n rlhf-audit-trail deployment/rlhf-audit-trail --tail=100
```

#### Database Connection Issues

```bash
# Check database pod status
kubectl get pods -n rlhf-audit-trail -l app.kubernetes.io/name=postgresql

# Test database connectivity
kubectl exec -n rlhf-audit-trail deployment/rlhf-audit-trail -- \
  python3 -c "
import os, sys
sys.path.insert(0, '/app/src')
from rlhf_audit_trail.database import DatabaseManager
import asyncio
result = asyncio.run(DatabaseManager().health_check())
print(result)
"
```

#### Performance Issues

```bash
# Check resource usage
kubectl top pods -n rlhf-audit-trail

# Check HPA status
kubectl get hpa -n rlhf-audit-trail

# Check metrics
kubectl get --raw /apis/metrics.k8s.io/v1beta1/namespaces/rlhf-audit-trail/pods
```

#### Network Issues

```bash
# Check network policies
kubectl get networkpolicy -n rlhf-audit-trail

# Test service connectivity
kubectl exec -n rlhf-audit-trail deployment/rlhf-audit-trail -- \
  curl -f http://rlhf-audit-trail-service:8000/health
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Update configmap for debug logging
kubectl patch configmap rlhf-audit-trail-config -n rlhf-audit-trail -p \
  '{"data":{"RLHF_AUDIT_LOG_LEVEL":"DEBUG"}}'

# Restart deployment
kubectl rollout restart deployment/rlhf-audit-trail -n rlhf-audit-trail
```

## Upgrade Procedures

### Application Updates

```bash
# Update to new version
./deploy.sh update --image 1.1.0

# Monitor rollout
kubectl rollout status deployment/rlhf-audit-trail -n rlhf-audit-trail

# Rollback if needed
./deploy.sh rollback
```

### Database Migrations

```bash
# Run database migrations
kubectl create job migration-$(date +%s) \
  --from=deployment/rlhf-audit-trail \
  -n rlhf-audit-trail -- \
  python3 -m alembic upgrade head
```

### Configuration Updates

```bash
# Update configuration
kubectl apply -f configmaps.yaml

# Rolling restart to pick up changes
kubectl rollout restart deployment/rlhf-audit-trail -n rlhf-audit-trail
```

## Security Considerations

### Network Security

- All traffic between pods is encrypted using service mesh
- External traffic uses TLS 1.3 with strong cipher suites
- Network policies restrict unnecessary pod-to-pod communication
- Database connections use SSL/TLS encryption

### Data Security

- All sensitive data stored in Kubernetes secrets
- Audit data encrypted at rest using application-level encryption
- Regular security scanning of container images
- Compliance with EU AI Act and GDPR requirements

### Access Control

- RBAC configured with least-privilege principle
- Service accounts for each component
- Regular rotation of secrets and credentials
- Multi-factor authentication for cluster access

## Maintenance

### Regular Tasks

1. **Weekly:**
   - Review monitoring dashboards
   - Check backup integrity
   - Update security patches

2. **Monthly:**
   - Performance review and tuning
   - Capacity planning review
   - Security audit

3. **Quarterly:**
   - Disaster recovery testing
   - Dependency updates
   - Compliance review

### Maintenance Windows

Schedule maintenance during low-traffic periods:

```bash
# Scale down during maintenance
kubectl scale deployment rlhf-audit-trail --replicas=1 -n rlhf-audit-trail

# Perform maintenance tasks
# ...

# Scale back up
kubectl scale deployment rlhf-audit-trail --replicas=3 -n rlhf-audit-trail
```

## Support

### Documentation

- [Application Documentation](../docs/)
- [API Documentation](../docs/api/)
- [Security Documentation](../SECURITY.md)

### Monitoring Dashboards

- Application metrics: Grafana dashboard
- Infrastructure metrics: Kubernetes dashboard
- Security metrics: Security dashboard

### Contact Information

For deployment issues:
- **Development Team**: dev-team@yourdomain.com
- **DevOps Team**: devops@yourdomain.com
- **Security Team**: security@yourdomain.com

### Emergency Procedures

For critical issues:
1. Check monitoring dashboards
2. Review recent deployment changes
3. Check application and infrastructure logs
4. Escalate to on-call engineer if needed
5. Follow incident response procedures