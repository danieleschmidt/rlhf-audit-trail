"""
Advanced health monitoring and observability for RLHF audit trail system.
Generation 2: Robust monitoring, logging, and health checks.
"""

import asyncio
import logging
import psutil
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import os

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .exceptions import MonitoringError


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    unit: str
    status: HealthStatus
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemAlert:
    """System alert."""
    id: str
    level: AlertLevel
    title: str
    message: str
    component: str
    timestamp: datetime
    resolved: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    name: str
    status: HealthStatus
    metrics: List[HealthMetric]
    alerts: List[SystemAlert]
    last_check: datetime
    uptime_seconds: float
    error_count: int = 0


class MetricsCollector:
    """Collects various system and application metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history: Dict[str, List[HealthMetric]] = {}
        self.max_history = 1000
        
    def collect_system_metrics(self) -> List[HealthMetric]:
        """Collect system-level metrics."""
        metrics = []
        now = datetime.utcnow()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(HealthMetric(
                name="cpu_usage_percent",
                value=cpu_percent,
                unit="%",
                status=self._get_status(cpu_percent, 70, 90),
                timestamp=now,
                threshold_warning=70.0,
                threshold_critical=90.0
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            metrics.append(HealthMetric(
                name="memory_usage_percent",
                value=memory_percent,
                unit="%",
                status=self._get_status(memory_percent, 80, 95),
                timestamp=now,
                threshold_warning=80.0,
                threshold_critical=95.0
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            metrics.append(HealthMetric(
                name="disk_usage_percent",
                value=disk_percent,
                unit="%",
                status=self._get_status(disk_percent, 85, 95),
                timestamp=now,
                threshold_warning=85.0,
                threshold_critical=95.0
            ))
            
            # System load
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
            cpu_count = psutil.cpu_count()
            load_percent = (load_avg / cpu_count) * 100
            metrics.append(HealthMetric(
                name="system_load_percent",
                value=load_percent,
                unit="%",
                status=self._get_status(load_percent, 80, 100),
                timestamp=now,
                threshold_warning=80.0,
                threshold_critical=100.0
            ))
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            metrics.append(HealthMetric(
                name="system_metrics_error",
                value=1,
                unit="error",
                status=HealthStatus.CRITICAL,
                timestamp=now,
                metadata={"error": str(e)}
            ))
        
        return metrics
    
    def collect_application_metrics(self, auditor_instances: List[Any]) -> List[HealthMetric]:
        """Collect application-specific metrics."""
        metrics = []
        now = datetime.utcnow()
        
        try:
            # Count active sessions
            active_sessions = sum(1 for auditor in auditor_instances if auditor.current_session)
            metrics.append(HealthMetric(
                name="active_sessions",
                value=active_sessions,
                unit="count",
                status=HealthStatus.HEALTHY if active_sessions < 100 else HealthStatus.DEGRADED,
                timestamp=now
            ))
            
            # Privacy budget utilization
            total_budget_used = 0
            total_budget_available = 0
            
            for auditor in auditor_instances:
                if hasattr(auditor, 'privacy_budget'):
                    total_budget_used += auditor.privacy_budget.total_spent_epsilon
                    total_budget_available += auditor.privacy_config.epsilon
            
            if total_budget_available > 0:
                budget_utilization = (total_budget_used / total_budget_available) * 100
                metrics.append(HealthMetric(
                    name="privacy_budget_utilization",
                    value=budget_utilization,
                    unit="%",
                    status=self._get_status(budget_utilization, 80, 95),
                    timestamp=now,
                    threshold_warning=80.0,
                    threshold_critical=95.0
                ))
            
            # Error rate (mock - would integrate with actual error tracking)
            error_rate = 0.5  # Placeholder
            metrics.append(HealthMetric(
                name="error_rate_percent",
                value=error_rate,
                unit="%",
                status=self._get_status(error_rate, 2, 5),
                timestamp=now,
                threshold_warning=2.0,
                threshold_critical=5.0
            ))
            
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            metrics.append(HealthMetric(
                name="application_metrics_error",
                value=1,
                unit="error",
                status=HealthStatus.CRITICAL,
                timestamp=now,
                metadata={"error": str(e)}
            ))
        
        return metrics
    
    def collect_database_metrics(self, db_manager: Optional[Any] = None) -> List[HealthMetric]:
        """Collect database-related metrics."""
        metrics = []
        now = datetime.utcnow()
        
        try:
            if db_manager:
                # Database connection health (mock)
                connection_health = True  # Would check actual connection
                metrics.append(HealthMetric(
                    name="database_connection",
                    value=1 if connection_health else 0,
                    unit="boolean",
                    status=HealthStatus.HEALTHY if connection_health else HealthStatus.CRITICAL,
                    timestamp=now
                ))
                
                # Query response time (mock)
                response_time_ms = 45.2  # Placeholder
                metrics.append(HealthMetric(
                    name="database_response_time",
                    value=response_time_ms,
                    unit="ms",
                    status=self._get_status(response_time_ms, 100, 500),
                    timestamp=now,
                    threshold_warning=100.0,
                    threshold_critical=500.0
                ))
            else:
                metrics.append(HealthMetric(
                    name="database_not_configured",
                    value=0,
                    unit="boolean",
                    status=HealthStatus.DEGRADED,
                    timestamp=now
                ))
                
        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")
            metrics.append(HealthMetric(
                name="database_metrics_error",
                value=1,
                unit="error",
                status=HealthStatus.CRITICAL,
                timestamp=now,
                metadata={"error": str(e)}
            ))
        
        return metrics
    
    def collect_storage_metrics(self, storage_backend: Optional[Any] = None) -> List[HealthMetric]:
        """Collect storage-related metrics."""
        metrics = []
        now = datetime.utcnow()
        
        try:
            if storage_backend:
                # Storage availability (mock)
                storage_available = True  # Would check actual storage
                metrics.append(HealthMetric(
                    name="storage_availability",
                    value=1 if storage_available else 0,
                    unit="boolean",
                    status=HealthStatus.HEALTHY if storage_available else HealthStatus.CRITICAL,
                    timestamp=now
                ))
                
                # Storage response time (mock)
                storage_latency = 25.8  # Placeholder
                metrics.append(HealthMetric(
                    name="storage_latency",
                    value=storage_latency,
                    unit="ms",
                    status=self._get_status(storage_latency, 50, 200),
                    timestamp=now,
                    threshold_warning=50.0,
                    threshold_critical=200.0
                ))
            else:
                metrics.append(HealthMetric(
                    name="storage_not_configured",
                    value=0,
                    unit="boolean",
                    status=HealthStatus.DEGRADED,
                    timestamp=now
                ))
                
        except Exception as e:
            logger.error(f"Failed to collect storage metrics: {e}")
            metrics.append(HealthMetric(
                name="storage_metrics_error",
                value=1,
                unit="error",
                status=HealthStatus.CRITICAL,
                timestamp=now,
                metadata={"error": str(e)}
            ))
        
        return metrics
    
    def _get_status(self, value: float, warning_threshold: float, critical_threshold: float) -> HealthStatus:
        """Determine health status based on thresholds."""
        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def store_metrics(self, metrics: List[HealthMetric]):
        """Store metrics in history."""
        for metric in metrics:
            if metric.name not in self.metrics_history:
                self.metrics_history[metric.name] = []
            
            self.metrics_history[metric.name].append(metric)
            
            # Trim history
            if len(self.metrics_history[metric.name]) > self.max_history:
                self.metrics_history[metric.name] = self.metrics_history[metric.name][-self.max_history:]
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[HealthMetric]:
        """Get metric history for specified time period."""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            metric for metric in self.metrics_history[metric_name]
            if metric.timestamp >= cutoff_time
        ]


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self.alerts: Dict[str, SystemAlert] = {}
        self.alert_handlers: List[Callable[[SystemAlert], None]] = []
        
    def add_alert_handler(self, handler: Callable[[SystemAlert], None]):
        """Add alert notification handler."""
        self.alert_handlers.append(handler)
    
    def create_alert(
        self,
        title: str,
        message: str,
        component: str,
        level: AlertLevel = AlertLevel.WARNING,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemAlert:
        """Create and process a new alert."""
        alert_id = f"{component}_{int(time.time())}"
        
        alert = SystemAlert(
            id=alert_id,
            level=level,
            title=title,
            message=message,
            component=component,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.alerts[alert_id] = alert
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"Alert created: {alert.title} - {alert.message}")
        return alert
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(self) -> List[SystemAlert]:
        """Get all unresolved alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def cleanup_old_alerts(self, max_age_hours: int = 72):
        """Remove old resolved alerts."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        to_remove = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.resolved and alert.timestamp < cutoff_time
        ]
        
        for alert_id in to_remove:
            del self.alerts[alert_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old alerts")


class HealthMonitor:
    """Main health monitoring system."""
    
    def __init__(self, check_interval: int = 60):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Health check interval in seconds
        """
        self.check_interval = check_interval
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.components: Dict[str, ComponentHealth] = {}
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # External dependencies
        self.auditor_instances: List[Any] = []
        self.db_manager: Optional[Any] = None
        self.storage_backend: Optional[Any] = None
        
        # Setup default alert handlers
        self.alert_manager.add_alert_handler(self._log_alert_handler)
    
    def register_auditor(self, auditor: Any):
        """Register an auditor instance for monitoring."""
        self.auditor_instances.append(auditor)
    
    def set_database_manager(self, db_manager: Any):
        """Set database manager for monitoring."""
        self.db_manager = db_manager
    
    def set_storage_backend(self, storage: Any):
        """Set storage backend for monitoring."""
        self.storage_backend = storage
    
    def start_monitoring(self):
        """Start the monitoring loop."""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                self._perform_health_check()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                time.sleep(10)  # Short delay before retrying
    
    def _perform_health_check(self):
        """Perform comprehensive health check."""
        try:
            # Collect all metrics
            all_metrics = []
            all_metrics.extend(self.metrics_collector.collect_system_metrics())
            all_metrics.extend(self.metrics_collector.collect_application_metrics(self.auditor_instances))
            all_metrics.extend(self.metrics_collector.collect_database_metrics(self.db_manager))
            all_metrics.extend(self.metrics_collector.collect_storage_metrics(self.storage_backend))
            
            # Store metrics
            self.metrics_collector.store_metrics(all_metrics)
            
            # Update component health
            self._update_component_health(all_metrics)
            
            # Generate alerts for critical metrics
            self._check_for_alerts(all_metrics)
            
            logger.debug(f"Health check completed with {len(all_metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            self.alert_manager.create_alert(
                title="Health Check Failed",
                message=f"Health monitoring system error: {str(e)}",
                component="health_monitor",
                level=AlertLevel.CRITICAL
            )
    
    def _update_component_health(self, metrics: List[HealthMetric]):
        """Update component health based on metrics."""
        component_metrics = {}
        
        # Group metrics by component
        for metric in metrics:
            component = self._get_component_for_metric(metric.name)
            if component not in component_metrics:
                component_metrics[component] = []
            component_metrics[component].append(metric)
        
        # Update each component
        for component, component_metric_list in component_metrics.items():
            overall_status = self._calculate_overall_status(component_metric_list)
            
            self.components[component] = ComponentHealth(
                name=component,
                status=overall_status,
                metrics=component_metric_list,
                alerts=[],  # Would be populated with component-specific alerts
                last_check=datetime.utcnow(),
                uptime_seconds=time.time() - self.metrics_collector.start_time
            )
    
    def _get_component_for_metric(self, metric_name: str) -> str:
        """Determine component name from metric name."""
        if metric_name.startswith(('cpu_', 'memory_', 'disk_', 'system_')):
            return 'system'
        elif metric_name.startswith(('database_', 'db_')):
            return 'database'
        elif metric_name.startswith(('storage_',)):
            return 'storage'
        elif metric_name.startswith(('privacy_', 'active_sessions', 'error_rate')):
            return 'application'
        else:
            return 'other'
    
    def _calculate_overall_status(self, metrics: List[HealthMetric]) -> HealthStatus:
        """Calculate overall status from multiple metrics."""
        statuses = [metric.status for metric in metrics]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _check_for_alerts(self, metrics: List[HealthMetric]):
        """Check metrics for alert conditions."""
        for metric in metrics:
            if metric.status == HealthStatus.CRITICAL:
                self.alert_manager.create_alert(
                    title=f"Critical Metric: {metric.name}",
                    message=f"{metric.name} is at critical level: {metric.value} {metric.unit}",
                    component=self._get_component_for_metric(metric.name),
                    level=AlertLevel.CRITICAL,
                    metadata={"metric": asdict(metric)}
                )
            elif metric.status == HealthStatus.DEGRADED:
                self.alert_manager.create_alert(
                    title=f"Warning: {metric.name}",
                    message=f"{metric.name} is degraded: {metric.value} {metric.unit}",
                    component=self._get_component_for_metric(metric.name),
                    level=AlertLevel.WARNING,
                    metadata={"metric": asdict(metric)}
                )
    
    def _log_alert_handler(self, alert: SystemAlert):
        """Default alert handler that logs alerts."""
        level_map = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        
        log_level = level_map.get(alert.level, logging.WARNING)
        logger.log(log_level, f"ALERT [{alert.component}]: {alert.title} - {alert.message}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        active_alerts = self.alert_manager.get_active_alerts()
        
        overall_status = HealthStatus.HEALTHY
        if any(alert.level == AlertLevel.CRITICAL for alert in active_alerts):
            overall_status = HealthStatus.CRITICAL
        elif any(component.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY] 
                for component in self.components.values()):
            overall_status = HealthStatus.UNHEALTHY
        elif any(component.status == HealthStatus.DEGRADED 
                for component in self.components.values()):
            overall_status = HealthStatus.DEGRADED
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.metrics_collector.start_time,
            "components": {
                name: {
                    "status": component.status.value,
                    "last_check": component.last_check.isoformat(),
                    "metrics_count": len(component.metrics),
                    "error_count": component.error_count
                }
                for name, component in self.components.items()
            },
            "active_alerts": [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "title": alert.title,
                    "component": alert.component,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in active_alerts
            ],
            "metrics_summary": self._get_metrics_summary()
        }
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        summary = {}
        
        for metric_name, metric_history in self.metrics_collector.metrics_history.items():
            if metric_history:
                latest = metric_history[-1]
                summary[metric_name] = {
                    "current_value": latest.value,
                    "unit": latest.unit,
                    "status": latest.status.value,
                    "timestamp": latest.timestamp.isoformat()
                }
        
        return summary


# Global health monitor instance
_global_monitor: Optional[HealthMonitor] = None

def get_health_monitor(create_if_missing: bool = True) -> Optional[HealthMonitor]:
    """Get global health monitor instance."""
    global _global_monitor
    
    if _global_monitor is None and create_if_missing:
        _global_monitor = HealthMonitor()
    
    return _global_monitor

def start_monitoring():
    """Start global health monitoring."""
    monitor = get_health_monitor()
    if monitor:
        monitor.start_monitoring()

def stop_monitoring():
    """Stop global health monitoring."""
    monitor = get_health_monitor(create_if_missing=False)
    if monitor:
        monitor.stop_monitoring()