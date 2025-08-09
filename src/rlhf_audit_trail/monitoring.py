"""Comprehensive monitoring and observability for RLHF audit trail system."""

import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
import json
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .exceptions import AuditTrailError

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used: float  # MB
    disk_usage_percent: float
    disk_free: float  # GB
    network_sent: float  # MB
    network_recv: float  # MB


@dataclass
class AuditMetrics:
    """Audit trail specific metrics."""
    timestamp: datetime
    session_count: int
    total_events: int
    events_per_second: float
    avg_event_size: float  # bytes
    privacy_budget_utilization: float  # percentage
    compliance_score: float
    verification_failures: int
    storage_usage: float  # MB


@dataclass
class PerformanceMetrics:
    """Performance timing metrics."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def duration_ms(self) -> float:
        return self.duration * 1000


class MetricsCollector:
    """Collects and aggregates system and application metrics."""
    
    def __init__(self, collection_interval: int = 60, retention_hours: int = 24):
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.max_samples = (retention_hours * 3600) // collection_interval
        
        # Metric storage
        self.system_metrics: deque = deque(maxlen=self.max_samples)
        self.audit_metrics: deque = deque(maxlen=self.max_samples)
        self.performance_metrics: deque = deque(maxlen=self.max_samples * 10)  # More samples
        
        # Operation counters
        self.operation_counters = defaultdict(int)
        self.error_counters = defaultdict(int)
        self.response_times = defaultdict(list)
        
        # Collection state
        self._collecting = False
        self._collection_thread = None
        self._lock = threading.Lock()
        
        logger.info(f"Initialized metrics collector with {collection_interval}s interval")
    
    def start_collection(self):
        """Start background metrics collection."""
        if self._collecting:
            logger.warning("Metrics collection already running")
            return
            
        self._collecting = True
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        logger.info("Started background metrics collection")
    
    def stop_collection(self):
        """Stop background metrics collection."""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Background collection loop."""
        while self._collecting:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        if not PSUTIL_AVAILABLE:
            return
            
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used=memory.used / (1024 * 1024),  # MB
                disk_usage_percent=disk.percent,
                disk_free=disk.free / (1024 * 1024 * 1024),  # GB
                network_sent=network.bytes_sent / (1024 * 1024),  # MB
                network_recv=network.bytes_recv / (1024 * 1024)  # MB
            )
            
            with self._lock:
                self.system_metrics.append(metrics)
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def record_audit_metrics(self, metrics: AuditMetrics):
        """Record audit trail specific metrics."""
        with self._lock:
            self.audit_metrics.append(metrics)
    
    def record_performance_metric(self, metric: PerformanceMetrics):
        """Record performance timing metric."""
        with self._lock:
            self.performance_metrics.append(metric)
            self.operation_counters[metric.operation] += 1
            
            if metric.error:
                self.error_counters[f"{metric.operation}_error"] += 1
            
            # Track response times
            if len(self.response_times[metric.operation]) > 100:
                self.response_times[metric.operation].pop(0)
            self.response_times[metric.operation].append(metric.duration)
    
    @contextmanager
    def time_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        error = None
        success = True
        
        try:
            yield
        except Exception as e:
            error = str(e)
            success = False
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            metric = PerformanceMetrics(
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=success,
                error=error,
                metadata=metadata
            )
            
            self.record_performance_metric(metric)
    
    def get_system_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of system metrics."""
        if not self.system_metrics:
            return {}
        
        with self._lock:
            metrics = list(self.system_metrics)
        
        if not metrics:
            return {}
        
        latest = metrics[-1]
        
        # Calculate averages over last hour
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_metrics = [m for m in metrics if m.timestamp >= hour_ago]
        
        if recent_metrics:
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        else:
            avg_cpu = latest.cpu_percent
            avg_memory = latest.memory_percent
        
        return {
            'current': {
                'cpu_percent': latest.cpu_percent,
                'memory_percent': latest.memory_percent,
                'memory_used_mb': latest.memory_used,
                'disk_usage_percent': latest.disk_usage_percent,
                'disk_free_gb': latest.disk_free,
                'network_sent_mb': latest.network_sent,
                'network_recv_mb': latest.network_recv,
                'timestamp': latest.timestamp
            },
            'averages_1h': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory
            },
            'samples_count': len(metrics)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        with self._lock:
            operations = dict(self.operation_counters)
            errors = dict(self.error_counters)
            response_times = {op: list(times) for op, times in self.response_times.items()}
        
        summary = {}
        
        for operation, count in operations.items():
            times = response_times.get(operation, [])
            error_count = errors.get(f"{operation}_error", 0)
            
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                p95_time = sorted(times)[int(len(times) * 0.95)] if len(times) > 0 else 0
            else:
                avg_time = min_time = max_time = p95_time = 0
            
            summary[operation] = {
                'total_calls': count,
                'error_count': error_count,
                'success_rate': ((count - error_count) / count * 100) if count > 0 else 0,
                'avg_response_time_ms': avg_time * 1000,
                'min_response_time_ms': min_time * 1000,
                'max_response_time_ms': max_time * 1000,
                'p95_response_time_ms': p95_time * 1000
            }
        
        return summary
    
    def get_audit_metrics_summary(self) -> Dict[str, Any]:
        """Get audit metrics summary."""
        if not self.audit_metrics:
            return {}
        
        with self._lock:
            metrics = list(self.audit_metrics)
        
        if not metrics:
            return {}
        
        latest = metrics[-1]
        
        return {
            'current': {
                'session_count': latest.session_count,
                'total_events': latest.total_events,
                'events_per_second': latest.events_per_second,
                'avg_event_size': latest.avg_event_size,
                'privacy_budget_utilization': latest.privacy_budget_utilization,
                'compliance_score': latest.compliance_score,
                'verification_failures': latest.verification_failures,
                'storage_usage_mb': latest.storage_usage,
                'timestamp': latest.timestamp
            },
            'samples_count': len(metrics)
        }


class AlertManager:
    """Manages alerts and notifications for system monitoring."""
    
    def __init__(self):
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_handlers = []
        self.notification_cooldown = {}  # Prevent spam
        
    def add_alert_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool], 
                       severity: str = 'warning', cooldown_minutes: int = 15):
        """Add an alert rule.
        
        Args:
            name: Alert name
            condition: Function that returns True if alert should fire
            severity: Alert severity ('info', 'warning', 'critical')
            cooldown_minutes: Minutes to wait before re-alerting
        """
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'severity': severity,
            'cooldown_minutes': cooldown_minutes
        })
        logger.info(f"Added alert rule: {name} ({severity})")
    
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        for rule in self.alert_rules:
            try:
                if rule['condition'](metrics):
                    self._trigger_alert(rule, metrics)
                else:
                    # Clear alert if it was active
                    if rule['name'] in self.active_alerts:
                        self._clear_alert(rule)
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    def _trigger_alert(self, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Trigger an alert."""
        now = datetime.utcnow()
        alert_name = rule['name']
        
        # Check cooldown
        if alert_name in self.notification_cooldown:
            last_sent = self.notification_cooldown[alert_name]
            if now - last_sent < timedelta(minutes=rule['cooldown_minutes']):
                return
        
        # Create alert
        alert = {
            'name': alert_name,
            'severity': rule['severity'],
            'timestamp': now,
            'message': f"Alert: {alert_name}",
            'metrics': metrics,
            'status': 'active'
        }
        
        self.active_alerts[alert_name] = alert
        self.notification_cooldown[alert_name] = now
        
        # Send notifications
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        logger.warning(f"Alert triggered: {alert_name} ({rule['severity']})")
    
    def _clear_alert(self, rule: Dict[str, Any]):
        """Clear an active alert."""
        alert_name = rule['name']
        if alert_name in self.active_alerts:
            alert = self.active_alerts[alert_name]
            alert['status'] = 'resolved'
            alert['resolved_at'] = datetime.utcnow()
            
            # Send clear notification
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert clear handler: {e}")
            
            del self.active_alerts[alert_name]
            logger.info(f"Alert cleared: {alert_name}")


class HealthChecker:
    """Performs comprehensive health checks."""
    
    def __init__(self):
        self.health_checks = {}
        
    def register_health_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {
            'overall_status': 'healthy',
            'timestamp': datetime.utcnow(),
            'checks': {}
        }
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results['checks'][name] = result
                
                # Update overall status
                if result.get('status') == 'unhealthy':
                    results['overall_status'] = 'unhealthy'
                elif result.get('status') == 'degraded' and results['overall_status'] == 'healthy':
                    results['overall_status'] = 'degraded'
                    
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results['checks'][name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.utcnow()
                }
                results['overall_status'] = 'unhealthy'
        
        return results


class MonitoringManager:
    """Main monitoring manager that orchestrates all monitoring components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            collection_interval=self.config.get('collection_interval', 60),
            retention_hours=self.config.get('retention_hours', 24)
        )
        
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Setup default alert handlers
        if self.config.get('log_alerts', True):
            self.alert_manager.add_alert_handler(self._log_alert_handler)
        
        logger.info("Initialized monitoring manager")
    
    def start(self):
        """Start monitoring."""
        self.metrics_collector.start_collection()
        logger.info("Monitoring started")
    
    def stop(self):
        """Stop monitoring."""
        self.metrics_collector.stop_collection()
        logger.info("Monitoring stopped")
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # High CPU usage
        self.alert_manager.add_alert_rule(
            'high_cpu_usage',
            lambda m: m.get('system', {}).get('current', {}).get('cpu_percent', 0) > 80,
            'warning'
        )
        
        # High memory usage
        self.alert_manager.add_alert_rule(
            'high_memory_usage',
            lambda m: m.get('system', {}).get('current', {}).get('memory_percent', 0) > 85,
            'warning'
        )
        
        # Low disk space
        self.alert_manager.add_alert_rule(
            'low_disk_space',
            lambda m: m.get('system', {}).get('current', {}).get('disk_usage_percent', 0) > 90,
            'critical'
        )
        
        # High error rate
        def high_error_rate(metrics):
            perf = metrics.get('performance', {})
            for operation, stats in perf.items():
                if stats.get('success_rate', 100) < 90:
                    return True
            return False
        
        self.alert_manager.add_alert_rule(
            'high_error_rate',
            high_error_rate,
            'critical'
        )
    
    def _log_alert_handler(self, alert: Dict[str, Any]):
        """Log alert to standard logging."""
        severity = alert['severity']
        message = f"ALERT [{severity.upper()}] {alert['name']}: {alert['message']}"
        
        if severity == 'critical':
            logger.critical(message)
        elif severity == 'warning':
            logger.warning(message)
        else:
            logger.info(message)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        system_metrics = self.metrics_collector.get_system_metrics_summary()
        performance_metrics = self.metrics_collector.get_performance_summary()
        audit_metrics = self.metrics_collector.get_audit_metrics_summary()
        
        status = {
            'system': system_metrics,
            'performance': performance_metrics,
            'audit': audit_metrics,
            'active_alerts': list(self.alert_manager.active_alerts.values()),
            'timestamp': datetime.utcnow()
        }
        
        # Check alerts
        self.alert_manager.check_alerts(status)
        
        return status
    
    def record_audit_event(self, session_count: int, total_events: int, 
                          events_per_second: float, avg_event_size: float,
                          privacy_budget_utilization: float, compliance_score: float,
                          verification_failures: int, storage_usage: float):
        """Record audit trail metrics."""
        metrics = AuditMetrics(
            timestamp=datetime.utcnow(),
            session_count=session_count,
            total_events=total_events,
            events_per_second=events_per_second,
            avg_event_size=avg_event_size,
            privacy_budget_utilization=privacy_budget_utilization,
            compliance_score=compliance_score,
            verification_failures=verification_failures,
            storage_usage=storage_usage
        )
        
        self.metrics_collector.record_audit_metrics(metrics)


# Global monitoring instance
_global_monitor: Optional[MonitoringManager] = None

def get_monitor() -> MonitoringManager:
    """Get global monitoring instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MonitoringManager()
    return _global_monitor

def time_operation(operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for timing operations using global monitor."""
    return get_monitor().metrics_collector.time_operation(operation, metadata)