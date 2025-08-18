# RLHF Audit Trail - Incident Response Runbook

## Overview

This runbook provides step-by-step procedures for responding to incidents in the RLHF Audit Trail system.

## Incident Classification

### Severity Levels

**P0 - Critical (Response: Immediate)**
- Complete system outage
- Data corruption or loss
- Security breach
- Privacy violation
- Compliance audit failure

**P1 - High (Response: <1 hour)**
- Performance degradation >50%
- Authentication failures
- API errors affecting multiple users
- Audit trail integrity issues

**P2 - Medium (Response: <4 hours)**
- Single component failure with workaround
- Non-critical feature unavailable
- Performance degradation <50%

**P3 - Low (Response: <24 hours)**
- Minor bugs
- Documentation issues
- Enhancement requests

## Immediate Response (First 15 minutes)

### 1. Initial Assessment
```bash
# Check system health
curl -f http://localhost:8000/health

# Check database connectivity
kubectl exec -it postgres-pod -- pg_isready

# Check Redis connectivity
kubectl exec -it redis-pod -- redis-cli ping

# Check Prometheus alerts
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing")'
```

### 2. Communication
- Create incident channel: `#incident-YYYY-MM-DD-HHMM`
- Notify on-call engineer
- Update status page if public-facing

### 3. Initial Triage
- Identify affected components
- Estimate user impact
- Check recent deployments
- Review system logs

## Component-Specific Procedures

### API Service Issues

**Symptoms:**
- High error rates
- Slow response times
- Authentication failures

**Diagnosis:**
```bash
# Check API logs
kubectl logs -l app=rlhf-audit-trail --tail=100

# Check resource usage
kubectl top pods -l app=rlhf-audit-trail

# Check recent requests
curl -s http://localhost:9090/api/v1/query?query='rate(http_requests_total[5m])' | jq
```

**Resolution:**
1. Scale up if resource constrained
2. Restart unhealthy pods
3. Check database connections
4. Review recent configuration changes

### Database Issues

**Symptoms:**
- Connection timeouts
- Slow queries
- Lock contentions
- Storage space issues

**Diagnosis:**
```bash
# Check database status
kubectl exec -it postgres-pod -- psql -U postgres -c "\l"

# Check connections
kubectl exec -it postgres-pod -- psql -U postgres -c "SELECT * FROM pg_stat_activity;"

# Check locks
kubectl exec -it postgres-pod -- psql -U postgres -c "SELECT * FROM pg_locks WHERE NOT granted;"

# Check disk usage
kubectl exec -it postgres-pod -- df -h
```

**Resolution:**
1. Terminate long-running queries if needed
2. Scale storage if space issue
3. Restart database if necessary
4. Check backup integrity

### Privacy Engine Issues

**Symptoms:**
- Privacy budget violations
- Noise generation failures
- Anonymization errors

**Diagnosis:**
```bash
# Check privacy metrics
curl -s http://localhost:9090/api/v1/query?query='privacy_budget_remaining' | jq

# Check privacy engine logs
kubectl logs -l component=privacy-engine --tail=50

# Verify differential privacy parameters
curl -s http://localhost:8000/api/v1/privacy/config | jq
```

**Resolution:**
1. Reset privacy budgets if needed
2. Verify epsilon/delta parameters
3. Check noise generation algorithms
4. Validate anonymization procedures

### Audit Trail Issues

**Symptoms:**
- Integrity verification failures
- Missing audit records
- Merkle tree corruption

**Diagnosis:**
```bash
# Check audit trail integrity
curl -s http://localhost:8000/api/v1/audit/integrity/verify | jq

# Check recent audit records
curl -s http://localhost:8000/api/v1/audit/records?limit=10 | jq

# Verify merkle tree
curl -s http://localhost:8000/api/v1/audit/merkle/verify | jq
```

**Resolution:**
1. Rebuild merkle tree if corrupted
2. Restore from backup if data loss
3. Regenerate cryptographic proofs
4. Notify compliance team

## Recovery Procedures

### Data Recovery

**Database Recovery:**
```bash
# List available backups
kubectl exec -it postgres-pod -- ls -la /backups/

# Restore from backup (point-in-time)
kubectl exec -it postgres-pod -- pg_restore -U postgres -d rlhf_audit /backups/backup_YYYY-MM-DD.sql

# Verify data integrity
kubectl exec -it postgres-pod -- psql -U postgres -c "SELECT COUNT(*) FROM audit_records;"
```

**Audit Log Recovery:**
```bash
# Check S3/cloud storage backups
aws s3 ls s3://rlhf-audit-trail-backups/audit-logs/

# Restore audit logs
aws s3 sync s3://rlhf-audit-trail-backups/audit-logs/ ./restored-logs/

# Verify integrity
python scripts/verify_audit_integrity.py ./restored-logs/
```

### System Recovery

**Rolling Back Deployment:**
```bash
# Check deployment history
kubectl rollout history deployment/rlhf-audit-trail

# Rollback to previous version
kubectl rollout undo deployment/rlhf-audit-trail

# Monitor rollback progress
kubectl rollout status deployment/rlhf-audit-trail
```

**Scaling Resources:**
```bash
# Scale API pods
kubectl scale deployment/rlhf-audit-trail --replicas=5

# Scale workers
kubectl scale deployment/celery-workers --replicas=3

# Monitor resource usage
kubectl top pods
```

## Security Incident Response

### Suspected Breach

**Immediate Actions:**
1. **STOP** - Don't modify anything
2. **PRESERVE** - Take snapshots/backups
3. **ISOLATE** - Block suspicious traffic
4. **NOTIFY** - Contact security team

**Investigation Steps:**
```bash
# Check access logs
kubectl logs -l app=rlhf-audit-trail | grep -E "(401|403|suspicious_activity)"

# Check audit trail for unauthorized access
curl -s http://localhost:8000/api/v1/audit/records?event_type=access_attempt | jq

# Review user activity
curl -s http://localhost:8000/api/v1/users/activity/suspicious | jq

# Check system integrity
curl -s http://localhost:8000/api/v1/system/integrity/check | jq
```

**Evidence Collection:**
1. Export audit logs
2. Capture network traffic
3. Document timeline
4. Preserve system state

### Data Privacy Incident

**GDPR/Privacy Violation Response:**
1. Assess scope of data exposure
2. Document affected individuals
3. Notify DPO within 1 hour
4. Prepare regulatory notifications
5. Implement containment measures

## Post-Incident Procedures

### 1. Root Cause Analysis
- Document timeline
- Identify contributing factors
- Determine root cause
- Review detection mechanisms

### 2. Action Items
- Create preventive measures
- Update monitoring/alerting
- Improve documentation
- Schedule follow-up reviews

### 3. Communication
- Update stakeholders
- Document lessons learned
- Update runbooks
- Conduct post-mortem

### 4. Compliance Reporting
- Generate incident report
- Update compliance status
- Notify auditors if required
- Document remediation efforts

## Emergency Contacts

### Internal
- **Incident Commander:** [On-call rotation]
- **Engineering Lead:** [Contact info]
- **Security Team:** [24/7 contact]
- **Compliance Officer:** [Contact info]

### External
- **Cloud Provider Support:** [Support number]
- **Security Vendor:** [Emergency contact]
- **Legal Counsel:** [Emergency contact]
- **Regulatory Bodies:** [As required]

## Tools and Resources

### Monitoring Dashboards
- **Grafana:** http://localhost:3000/d/rlhf-overview
- **Prometheus:** http://localhost:9090/alerts
- **Alertmanager:** http://localhost:9093

### Log Analysis
- **Loki:** http://localhost:3100
- **Kibana/ElasticSearch:** [If configured]

### Communication
- **Slack:** #incidents channel
- **Status Page:** [If public]
- **Incident Management:** [Tool link]

## Escalation Matrix

| Time | Action |
|------|--------|
| 0-15 min | Initial response, basic triage |
| 15-30 min | Escalate to engineering lead |
| 30-60 min | Escalate to incident commander |
| 1-2 hours | Escalate to executive team |
| 2+ hours | External support engagement |

Remember: **When in doubt, escalate early and communicate frequently.**