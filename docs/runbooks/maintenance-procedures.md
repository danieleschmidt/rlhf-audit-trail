# RLHF Audit Trail - Maintenance & Operations Runbook

## Overview

This runbook covers routine maintenance tasks, backup procedures, and operational tasks for the RLHF Audit Trail system.

## Daily Maintenance Tasks

### System Health Checks (Automated)
```bash
#!/bin/bash
# Daily health check script (runs via cron at 6 AM)

echo "=== Daily Health Check - $(date) ==="

# Check service health
curl -sf http://localhost:8000/health || echo "❌ API health check failed"

# Check database connectivity
kubectl exec -it postgresql-0 -n rlhf-production -- pg_isready || echo "❌ Database health check failed"

# Check Redis connectivity
kubectl exec -it redis-0 -n rlhf-production -- redis-cli ping || echo "❌ Redis health check failed"

# Check disk space
kubectl exec -it postgresql-0 -n rlhf-production -- df -h | grep -E '[8-9][0-9]%|100%' && echo "⚠️ High disk usage detected"

# Check audit trail integrity (sample)
curl -sf http://localhost:8000/api/v1/audit/integrity/sample-check || echo "❌ Audit integrity check failed"

echo "✅ Daily health check complete"
```

### Metric Collection
- Monitor error rates (target: <0.1%)
- Track response times (target: <200ms)
- Monitor resource usage
- Check audit trail growth rate
- Validate privacy budget consumption

## Weekly Maintenance Tasks

### Certificate Renewal Check
```bash
# Check certificate expiration
kubectl get certificates -n rlhf-production

# Check specific certificate details
kubectl describe certificate rlhf-audit-trail-tls -n rlhf-production

# Force renewal if needed (30 days before expiry)
kubectl annotate certificate rlhf-audit-trail-tls cert-manager.io/issue-temporary-certificate="true" -n rlhf-production
```

### Security Updates
```bash
# Check for security updates
kubectl exec -it rlhf-audit-trail-pod -- apt list --upgradable | grep -i security

# Update container images (if security patches available)
kubectl set image deployment/rlhf-audit-trail rlhf-audit-trail=rlhf-audit-trail:v1.2.3-security-patch

# Monitor deployment
kubectl rollout status deployment/rlhf-audit-trail -n rlhf-production
```

### Compliance Report Generation
```bash
# Generate weekly compliance report
curl -X POST http://localhost:8000/api/v1/compliance/reports/generate \
  -H "Content-Type: application/json" \
  -d '{"period": "weekly", "frameworks": ["eu_ai_act", "gdpr"]}'

# Export to secure storage
curl -s http://localhost:8000/api/v1/compliance/reports/latest/export \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  > compliance_report_$(date +%Y%m%d).json

# Notify compliance team
python scripts/notify_compliance_team.py --report compliance_report_$(date +%Y%m%d).json
```

## Monthly Maintenance Tasks

### Database Maintenance
```bash
# Update database statistics
kubectl exec -it postgresql-0 -n rlhf-production -- psql -U postgres -c "ANALYZE;"

# Vacuum full (during maintenance window)
kubectl exec -it postgresql-0 -n rlhf-production -- psql -U postgres -c "VACUUM FULL;"

# Reindex critical tables
kubectl exec -it postgresql-0 -n rlhf-production -- psql -U postgres -c "REINDEX TABLE audit_records;"
kubectl exec -it postgresql-0 -n rlhf-production -- psql -U postgres -c "REINDEX TABLE privacy_records;"

# Check table sizes
kubectl exec -it postgresql-0 -n rlhf-production -- psql -U postgres -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables 
WHERE schemaname='public' 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"
```

### Log Rotation and Cleanup
```bash
# Rotate application logs
kubectl exec -it rlhf-audit-trail-pod -- logrotate /etc/logrotate.d/rlhf-audit-trail

# Clean up old log files (older than 90 days)
find /var/log/rlhf-audit-trail -name "*.log.*" -mtime +90 -delete

# Clean up temporary files
kubectl exec -it rlhf-audit-trail-pod -- find /tmp -name "rlhf_*" -mtime +7 -delete
```

### Dependency Updates
```bash
# Check for dependency updates
pip list --outdated

# Update non-critical dependencies
pip install --upgrade requests urllib3 certifi

# Update critical dependencies (with testing)
# 1. Update in staging first
# 2. Run full test suite  
# 3. Deploy to production during maintenance window
```

## Backup Procedures

### Daily Automated Backups

**Database Backup:**
```bash
#!/bin/bash
# Database backup script (runs daily at 2 AM)

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="rlhf_audit_backup_$BACKUP_DATE.sql"

# Create database dump
kubectl exec -it postgresql-0 -n rlhf-production -- pg_dump -U postgres -Fc rlhf_audit > $BACKUP_FILE

# Compress and encrypt
gpg --cipher-algo AES256 --compress-algo 1 --s2k-mode 3 --s2k-digest-algo SHA512 --s2k-count 65536 --encrypt -r backup@company.com $BACKUP_FILE

# Upload to secure storage
aws s3 cp ${BACKUP_FILE}.gpg s3://rlhf-backups/database/daily/

# Cleanup local files
rm $BACKUP_FILE ${BACKUP_FILE}.gpg

# Verify backup integrity
aws s3api head-object --bucket rlhf-backups --key database/daily/${BACKUP_FILE}.gpg

echo "✅ Database backup completed: $BACKUP_FILE"
```

**Audit Logs Backup:**
```bash
#!/bin/bash
# Audit logs backup script

BACKUP_DATE=$(date +%Y%m%d)

# Sync audit logs to backup storage
aws s3 sync /app/audit_logs/ s3://rlhf-backups/audit-logs/daily/$BACKUP_DATE/

# Create integrity manifest
find /app/audit_logs -type f -name "*.json" -exec sha256sum {} \; > audit_manifest_$BACKUP_DATE.txt
aws s3 cp audit_manifest_$BACKUP_DATE.txt s3://rlhf-backups/audit-logs/manifests/

echo "✅ Audit logs backup completed for $BACKUP_DATE"
```

### Weekly Full Backups
```bash
# Full system backup (runs Sundays at 1 AM)
#!/bin/bash

BACKUP_DATE=$(date +%Y%m%d)

# Database full backup
kubectl exec -it postgresql-0 -n rlhf-production -- pg_dumpall -U postgres > full_backup_$BACKUP_DATE.sql

# Configuration backup
kubectl get configmaps -n rlhf-production -o yaml > configmaps_backup_$BACKUP_DATE.yaml
kubectl get secrets -n rlhf-production -o yaml > secrets_backup_$BACKUP_DATE.yaml

# Application state backup
tar -czf app_state_$BACKUP_DATE.tar.gz /app/data /app/checkpoints

# Encrypt and store
for file in full_backup_$BACKUP_DATE.sql configmaps_backup_$BACKUP_DATE.yaml app_state_$BACKUP_DATE.tar.gz; do
    gpg --encrypt -r backup@company.com $file
    aws s3 cp ${file}.gpg s3://rlhf-backups/weekly/
    rm $file ${file}.gpg
done

echo "✅ Weekly full backup completed for $BACKUP_DATE"
```

### Backup Restoration Testing
```bash
# Monthly backup restoration test
#!/bin/bash

# Create test environment
kubectl create namespace rlhf-backup-test

# Deploy test database
kubectl apply -f k8s/backup-test/postgresql.yaml -n rlhf-backup-test

# Download and decrypt latest backup
LATEST_BACKUP=$(aws s3 ls s3://rlhf-backups/database/daily/ | sort | tail -n 1 | awk '{print $4}')
aws s3 cp s3://rlhf-backups/database/daily/$LATEST_BACKUP ./
gpg --decrypt $LATEST_BACKUP > restored_backup.sql

# Restore to test database
kubectl exec -it postgresql-test-0 -n rlhf-backup-test -- pg_restore -U postgres -d rlhf_audit_test restored_backup.sql

# Verify data integrity
kubectl exec -it postgresql-test-0 -n rlhf-backup-test -- psql -U postgres -c "SELECT COUNT(*) FROM audit_records;"

# Cleanup test environment
kubectl delete namespace rlhf-backup-test
rm restored_backup.sql $LATEST_BACKUP

echo "✅ Backup restoration test completed successfully"
```

## Privacy Budget Management

### Daily Budget Monitoring
```bash
# Check current privacy budget status
curl -s http://localhost:8000/api/v1/privacy/budget | jq '
{
  total_epsilon: .total_epsilon,
  consumed_epsilon: .consumed_epsilon,
  remaining_epsilon: .remaining_epsilon,
  days_until_reset: .days_until_reset
}'

# Alert if budget usage > 80%
USAGE=$(curl -s http://localhost:8000/api/v1/privacy/budget | jq '.consumed_epsilon / .total_epsilon * 100')
if (( $(echo "$USAGE > 80" | bc -l) )); then
    echo "⚠️ Privacy budget usage at ${USAGE}% - consider reducing epsilon or resetting budget"
fi
```

### Weekly Budget Analysis
```bash
# Generate privacy usage report
curl -s http://localhost:8000/api/v1/privacy/reports/weekly | jq '
{
  period: .period,
  total_queries: .total_queries,
  epsilon_per_query: .epsilon_per_query,
  anonymization_stats: .anonymization_stats,
  compliance_status: .compliance_status
}' > privacy_report_$(date +%Y%m%d).json

# Check for unusual patterns
python scripts/analyze_privacy_usage.py privacy_report_$(date +%Y%m%d).json
```

## Audit Trail Maintenance

### Daily Integrity Verification
```bash
# Verify audit trail integrity (sample verification)
curl -s http://localhost:8000/api/v1/audit/integrity/verify?sample_size=1000 | jq '
{
  verification_status: .status,
  records_verified: .records_verified,
  integrity_score: .integrity_score,
  anomalies_detected: .anomalies_detected
}'

# Full verification (weekly)
if [ $(date +%u) -eq 7 ]; then  # Sunday
    curl -s http://localhost:8000/api/v1/audit/integrity/verify?full=true
fi
```

### Merkle Tree Maintenance
```bash
# Rebuild merkle tree if needed
INTEGRITY_SCORE=$(curl -s http://localhost:8000/api/v1/audit/integrity/verify | jq '.integrity_score')
if (( $(echo "$INTEGRITY_SCORE < 0.95" | bc -l) )); then
    echo "🔧 Rebuilding merkle tree due to low integrity score: $INTEGRITY_SCORE"
    curl -X POST http://localhost:8000/api/v1/audit/merkle/rebuild
fi
```

## Performance Optimization

### Database Query Optimization
```bash
# Identify slow queries
kubectl exec -it postgresql-0 -n rlhf-production -- psql -U postgres -c "
SELECT query, calls, total_time, mean_time, rows
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
"

# Update statistics for query planner
kubectl exec -it postgresql-0 -n rlhf-production -- psql -U postgres -c "
UPDATE pg_stat_statements SET calls = 0, total_time = 0;
ANALYZE;
"
```

### Cache Optimization
```bash
# Redis memory analysis
kubectl exec -it redis-0 -n rlhf-production -- redis-cli INFO memory

# Clear expired keys
kubectl exec -it redis-0 -n rlhf-production -- redis-cli EVAL "
local keys = redis.call('keys', ARGV[1])
for i=1,#keys,5000 do
  redis.call('del', unpack(keys, i, math.min(i+4999, #keys)))
end
return #keys" 0 "expired:*"

# Analyze cache hit rates  
kubectl exec -it redis-0 -n rlhf-production -- redis-cli INFO stats | grep -E "(hits|misses)"
```

## Capacity Planning

### Monthly Resource Analysis
```bash
# CPU and memory trends
kubectl top nodes
kubectl top pods -n rlhf-production

# Database growth analysis
kubectl exec -it postgresql-0 -n rlhf-production -- psql -U postgres -c "
SELECT 
    date_trunc('month', created_at) as month,
    COUNT(*) as records,
    pg_size_pretty(SUM(octet_length(event_data::text))) as data_size
FROM audit_records 
WHERE created_at >= NOW() - INTERVAL '12 months'
GROUP BY date_trunc('month', created_at)
ORDER BY month;
"

# Storage utilization forecast
python scripts/capacity_forecast.py --lookback=90 --forecast=180
```

### Scaling Recommendations
```bash
# Auto-scaling policy review
kubectl get hpa -n rlhf-production

# Resource utilization analysis
kubectl exec -it monitoring-prometheus-0 -- promtool query instant \
  'rate(container_cpu_usage_seconds_total[5m]) * 100'

# Generate scaling recommendations
python scripts/scaling_recommendations.py --environment=production
```

## Compliance Maintenance

### EU AI Act Compliance Tasks
```bash
# Generate monthly risk assessment
curl -X POST http://localhost:8000/api/v1/compliance/risk-assessment \
  -H "Content-Type: application/json" \
  -d '{"framework": "eu_ai_act", "period": "monthly"}'

# Update model cards
curl -X POST http://localhost:8000/api/v1/model-cards/refresh-all

# Validate transparency requirements
curl -s http://localhost:8000/api/v1/compliance/transparency/validate | jq '.status'
```

### GDPR Compliance Tasks
```bash
# Data retention policy enforcement
curl -X POST http://localhost:8000/api/v1/privacy/data-retention/enforce

# Subject access request processing
curl -s http://localhost:8000/api/v1/privacy/subject-requests/pending | jq '.count'

# Right to erasure implementation
python scripts/process_erasure_requests.py --dry-run=false
```

## Monitoring and Alerting Maintenance

### Alert Rule Updates
```bash
# Update Prometheus alert rules
kubectl apply -f monitoring/prometheus/rules/rlhf-audit-trail.yml -n monitoring

# Test alert rules
kubectl exec -it prometheus-0 -n monitoring -- promtool check rules /etc/prometheus/rules/*.yml

# Update Grafana dashboards
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d @monitoring/grafana/dashboards/rlhf-audit-trail-overview.json
```

### Log Aggregation Maintenance
```bash
# Loki log retention cleanup
kubectl exec -it loki-0 -n monitoring -- /usr/bin/loki-canary -addr=http://localhost:3100

# Update log parsing rules
kubectl apply -f monitoring/loki/loki-config.yaml -n monitoring

# Verify log ingestion rates
curl -s http://loki:3100/metrics | grep loki_ingester_received_chunks
```

## Emergency Maintenance Procedures

### Emergency Shutdown
```bash
# Graceful shutdown sequence
kubectl scale deployment/rlhf-audit-trail --replicas=0 -n rlhf-production
kubectl scale deployment/celery-workers --replicas=0 -n rlhf-production

# Wait for connections to drain
sleep 30

# Stop database if needed
kubectl scale statefulset/postgresql --replicas=0 -n rlhf-production

# Update status page
curl -X POST https://status.company.com/api/incidents \
  -H "Authorization: Bearer $STATUS_API_KEY" \
  -d '{"name": "Scheduled Maintenance", "status": "maintenance", "message": "Emergency maintenance in progress"}'
```

### Emergency Startup
```bash
# Start database first
kubectl scale statefulset/postgresql --replicas=1 -n rlhf-production
kubectl wait --for=condition=Ready pod -l app=postgresql -n rlhf-production --timeout=300s

# Start application services
kubectl scale deployment/rlhf-audit-trail --replicas=3 -n rlhf-production
kubectl scale deployment/celery-workers --replicas=2 -n rlhf-production

# Verify all services
kubectl get pods -n rlhf-production

# Update status page
curl -X PATCH https://status.company.com/api/incidents/latest \
  -H "Authorization: Bearer $STATUS_API_KEY" \
  -d '{"status": "resolved", "message": "All systems operational"}'
```

## Maintenance Windows

### Weekly Maintenance (Sundays 2-4 AM UTC)
- Database maintenance (VACUUM, ANALYZE)
- Security updates
- Certificate renewals
- Log rotation
- Cache cleanup

### Monthly Maintenance (First Sunday 1-5 AM UTC)
- Full system backup
- Dependency updates
- Performance optimization
- Capacity planning review
- Compliance audit

### Quarterly Maintenance (Seasonal, 6-hour window)
- Major version updates
- Infrastructure upgrades
- Security audit
- Disaster recovery testing
- Performance benchmarking

Remember: Always communicate maintenance windows in advance and have rollback procedures ready.