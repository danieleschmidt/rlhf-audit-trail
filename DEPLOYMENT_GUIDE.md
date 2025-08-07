# Quantum Task Planner - Deployment Guide

## Overview

This guide covers deploying the Quantum Task Planner in production environments using Docker Compose or Kubernetes.

## Prerequisites

### System Requirements

- **CPU**: Minimum 2 cores, recommended 4+ cores
- **Memory**: Minimum 4GB RAM, recommended 8GB+ RAM  
- **Storage**: Minimum 20GB, recommended 50GB+ for logs and data
- **Network**: Stable internet connection for image pulls

### Software Requirements

- Docker 20.10+
- Docker Compose 1.29+ (for Docker deployment)
- Kubernetes 1.20+ (for Kubernetes deployment)
- kubectl configured (for Kubernetes deployment)

## Quick Start

### Docker Compose Deployment (Recommended for Development/Testing)

1. **Clone and prepare the repository:**
   ```bash
   git clone https://github.com/your-org/quantum-task-planner.git
   cd quantum-task-planner
   ```

2. **Run the deployment script:**
   ```bash
   ./scripts/deploy.sh
   ```

3. **Access the application:**
   - API: http://localhost:8000
   - Health check: http://localhost:8000/health
   - Metrics: http://localhost:8000/metrics

### Manual Docker Compose Deployment

1. **Build the application:**
   ```bash
   docker build -f docker/Dockerfile -t quantum-task-planner:latest .
   ```

2. **Start services:**
   ```bash
   docker-compose up -d
   ```

3. **Verify deployment:**
   ```bash
   docker-compose ps
   curl http://localhost:8000/health
   ```

## Production Deployment

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Application Configuration
QUANTUM_PLANNER_ENV=production
QUANTUM_PLANNER_LOG_LEVEL=INFO
QUANTUM_PLANNER_API_HOST=0.0.0.0
QUANTUM_PLANNER_API_PORT=8000

# Database Configuration
DATABASE_URL=postgresql://quantum:secure_password@postgres:5432/quantum_planner
REDIS_URL=redis://redis:6379/0

# Security Configuration
SECRET_KEY=your-super-secure-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here

# Performance Configuration
MAX_WORKERS=4
ENABLE_CACHING=true
CACHE_TTL=3600

# Monitoring Configuration
ENABLE_METRICS=true
ENABLE_HEALTH_CHECKS=true
LOG_FORMAT=json
```

### Kubernetes Deployment

1. **Create namespace:**
   ```bash
   kubectl create namespace quantum-planner
   ```

2. **Apply secrets:**
   ```bash
   kubectl create secret generic quantum-secrets \
     --from-literal=database-url="postgresql://user:pass@host:5432/db" \
     --from-literal=redis-url="redis://host:6379/0" \
     --from-literal=secret-key="your-secret-key" \
     -n quantum-planner
   ```

3. **Deploy application:**
   ```bash
   kubectl apply -f kubernetes/ -n quantum-planner
   ```

4. **Check deployment:**
   ```bash
   kubectl get all -n quantum-planner
   kubectl logs -f deployment/quantum-task-planner -n quantum-planner
   ```

## Configuration

### Application Configuration

The application supports configuration through:

1. **Environment variables** (highest priority)
2. **Configuration files** (config/config.yaml)
3. **Default values** (lowest priority)

Key configuration options:

```yaml
quantum_planner:
  # Core settings
  coherence_preservation: true
  entanglement_enabled: true
  interference_threshold: 0.3
  
  # Performance settings
  performance:
    enable_caching: true
    cache_size: 1000
    max_worker_threads: 4
    batch_size: 10
    
  # Security settings
  security:
    max_concurrent_tasks: 100
    max_total_tasks: 10000
    rate_limiting:
      max_tasks_per_minute: 1000
```

### Database Configuration

The application supports PostgreSQL as the primary database:

```bash
# Connection string format
DATABASE_URL=postgresql://username:password@hostname:port/database_name

# Example
DATABASE_URL=postgresql://quantum:mypassword@localhost:5432/quantum_planner
```

### Redis Configuration

Redis is used for caching and message queuing:

```bash
# Connection string format  
REDIS_URL=redis://hostname:port/database_number

# Example with authentication
REDIS_URL=redis://:password@hostname:port/0
```

## Monitoring and Observability

### Health Checks

The application provides comprehensive health checks:

- **Liveness probe**: `GET /health/live`
- **Readiness probe**: `GET /health/ready`  
- **Detailed health**: `GET /health`

### Metrics

Prometheus metrics are available at `/metrics`:

- Task execution metrics
- Quantum state metrics  
- Performance metrics
- System resource metrics

### Logging

Structured logging is available in multiple formats:

- **Development**: Human-readable format
- **Production**: JSON format for log aggregation

Log levels: `DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL`

## Security

### Production Security Checklist

- [ ] Change all default passwords
- [ ] Use strong, unique secret keys
- [ ] Enable HTTPS/TLS
- [ ] Configure rate limiting
- [ ] Set up proper authentication
- [ ] Enable audit logging
- [ ] Configure network security groups
- [ ] Regular security updates

### Authentication

The application supports multiple authentication methods:

1. **API Keys** (for service-to-service)
2. **JWT Tokens** (for user authentication)
3. **OAuth 2.0** (for external providers)

### Network Security

- Use HTTPS in production
- Configure proper firewall rules
- Use private networks for internal communication
- Enable request rate limiting

## Scaling

### Horizontal Scaling

The application supports horizontal scaling:

1. **Stateless design** - No local state dependencies
2. **Load balancer ready** - Health checks and graceful shutdown
3. **Database clustering** - PostgreSQL read replicas
4. **Cache clustering** - Redis cluster support

### Vertical Scaling

Resource recommendations by load:

| Load Level | CPU | Memory | Storage |
|------------|-----|--------|---------|
| Light      | 2 cores | 4GB | 20GB |
| Medium     | 4 cores | 8GB | 50GB |
| Heavy      | 8 cores | 16GB | 100GB |

### Auto-scaling Configuration

For Kubernetes deployments:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: quantum-task-planner-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: quantum-task-planner
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource  
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Backup and Recovery

### Database Backup

Automated PostgreSQL backup:

```bash
# Create backup
pg_dump -h localhost -U quantum quantum_planner > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
psql -h localhost -U quantum quantum_planner < backup_file.sql
```

### Data Persistence

Important directories to backup:

- `/app/data` - Application data
- `/app/logs` - Application logs
- Database data directory
- Redis persistence files (if enabled)

### Disaster Recovery

1. **Database replication** - Set up PostgreSQL streaming replication
2. **Regular backups** - Automated daily/weekly backups
3. **Configuration backup** - Version control all configuration
4. **Monitoring** - Alert on backup failures

## Troubleshooting

### Common Issues

1. **Application won't start**
   ```bash
   # Check logs
   docker-compose logs quantum-planner
   
   # Check configuration
   docker-compose config
   ```

2. **Database connection failed**
   ```bash
   # Test database connectivity
   docker-compose exec quantum-planner python -c "
   import psycopg2
   conn = psycopg2.connect('$DATABASE_URL')
   print('Database connection successful')
   "
   ```

3. **High memory usage**
   ```bash
   # Check memory usage
   docker stats
   
   # Reduce cache size in configuration
   CACHE_SIZE=500
   ```

4. **Performance issues**
   ```bash
   # Check metrics
   curl http://localhost:8000/metrics
   
   # Check quantum system state  
   curl http://localhost:8000/system/state
   ```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
QUANTUM_PLANNER_LOG_LEVEL=DEBUG
QUANTUM_PLANNER_DEBUG=true
```

### Performance Profiling

Enable performance profiling:

```bash
# Enable profiler
ENABLE_PROFILER=true

# Access profiling data
curl http://localhost:8000/debug/profile
```

## Maintenance

### Regular Maintenance Tasks

1. **Update dependencies** - Monthly security updates
2. **Database maintenance** - Weekly VACUUM and ANALYZE
3. **Log rotation** - Daily log cleanup  
4. **Backup verification** - Weekly restore tests
5. **Performance monitoring** - Continuous monitoring
6. **Security audits** - Quarterly security reviews

### Upgrade Process

1. **Backup current deployment**
2. **Test upgrade in staging**
3. **Deploy new version with rolling update**
4. **Verify functionality**
5. **Monitor for issues**

## Support and Resources

### Documentation

- [API Documentation](./API.md)
- [Configuration Reference](./CONFIGURATION.md)
- [Security Guide](./SECURITY.md)
- [Performance Tuning](./PERFORMANCE.md)

### Support Channels

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and tutorials
- **Community**: Discussion forums and chat

### Monitoring Dashboards

- **Application Metrics**: Grafana dashboard for application metrics
- **Infrastructure**: System resource monitoring
- **Logs**: Centralized log analysis and alerting

---

For additional help, please refer to the project documentation or open an issue on GitHub.