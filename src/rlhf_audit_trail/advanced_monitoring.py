"""Advanced monitoring and observability system for RLHF audit trail.

This module provides comprehensive monitoring, metrics collection, alerting,
and observability features for production deployment.
"""

import time
import logging
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import queue
import contextlib
import functools

from .exceptions import MonitoringError, get_error_details
from .health_system import HealthStatus


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class Alert:
    """System alert definition."""
    id: str
    name: str
    message: str
    severity: AlertSeverity
    timestamp: float
    source: str
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self, max_points: int = 10000):
        """Initialize metrics collector.
        
        Args:
            max_points: Maximum number of metric points to retain
        """
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.aggregates: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metric_type: MetricType = MetricType.GAUGE
    ):
        """Record a metric data point."""
        with self.lock:
            point = MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=metric_type
            )
            
            self.metrics[name].append(point)
            self._update_aggregates(name, value, metric_type)
            
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self.lock:
            current_value = self.aggregates[name].get("total", 0.0)
            self.record_metric(name, current_value + value, tags, MetricType.COUNTER)
            
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        self.record_metric(name, value, tags, MetricType.GAUGE)
        
    def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric."""
        self.record_metric(name, duration, tags, MetricType.TIMER)
        
    def _update_aggregates(self, name: str, value: float, metric_type: MetricType):
        """Update aggregate statistics for a metric."""
        if name not in self.aggregates:
            self.aggregates[name] = {
                "count": 0,
                "total": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
                "avg": 0.0
            }
            
        agg = self.aggregates[name]
        
        if metric_type == MetricType.COUNTER:
            agg["total"] = value  # Counters track absolute value
            agg["count"] += 1
        else:
            agg["count"] += 1
            agg["total"] += value
            agg["min"] = min(agg["min"], value)
            agg["max"] = max(agg["max"], value)
            agg["avg"] = agg["total"] / agg["count"]
            
    def get_metric_summary(self, name: str, time_window: float = 300.0) -> Dict[str, Any]:
        """Get summary statistics for a metric within a time window."""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - time_window
            
            if name not in self.metrics:
                return {}
                
            recent_points = [
                point for point in self.metrics[name]
                if point.timestamp >= cutoff_time
            ]
            
            if not recent_points:
                return {}
                
            values = [point.value for point in recent_points]
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "total": sum(values),
                "latest": values[-1] if values else 0,
                "time_window": time_window,
                "points": len(recent_points)
            }
            
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metric summaries."""
        with self.lock:
            return {
                name: self.get_metric_summary(name)
                for name in self.metrics.keys()
            }


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, max_alerts: int = 1000):
        """Initialize alert manager.
        
        Args:
            max_alerts: Maximum number of alerts to retain
        """
        self.max_alerts = max_alerts
        self.alerts: deque = deque(maxlen=max_alerts)
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.suppressed_alerts: Dict[str, float] = {}  # name -> last_sent_time
        self.suppression_window = 300.0  # 5 minutes
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert notification handler."""
        self.alert_handlers.append(handler)
        
    def fire_alert(
        self,
        name: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        source: str = "system",
        tags: Optional[Dict[str, str]] = None,
        suppress_duplicates: bool = True
    ) -> Optional[Alert]:
        """Fire an alert."""
        current_time = time.time()
        
        # Check suppression
        if suppress_duplicates:
            last_sent = self.suppressed_alerts.get(name, 0)
            if current_time - last_sent < self.suppression_window:
                return None  # Suppressed
                
        with self.lock:
            alert = Alert(
                id=f"{name}_{int(current_time)}",
                name=name,
                message=message,
                severity=severity,
                timestamp=current_time,
                source=source,
                tags=tags or {}
            )
            
            self.alerts.append(alert)
            
            # Update suppression tracking
            if suppress_duplicates:
                self.suppressed_alerts[name] = current_time
                
            # Send to handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")
                    
            self.logger.warning(f"Alert fired: {name} - {message}")
            return alert
            
    def resolve_alert(self, alert_name: str) -> bool:
        """Mark an alert as resolved."""
        with self.lock:
            for alert in reversed(self.alerts):
                if alert.name == alert_name and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = time.time()
                    self.logger.info(f"Alert resolved: {alert_name}")
                    return True
            return False
            
    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts."""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]
            
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        with self.lock:
            active_alerts = self.get_active_alerts()
            
            severity_counts = defaultdict(int)
            for alert in active_alerts:
                severity_counts[alert.severity.value] += 1
                
            return {
                "total_alerts": len(self.alerts),
                "active_alerts": len(active_alerts),
                "resolved_alerts": len(self.alerts) - len(active_alerts),
                "severity_breakdown": dict(severity_counts),
                "oldest_active": min(
                    (alert.timestamp for alert in active_alerts),
                    default=0
                ),
                "latest_alert": max(
                    (alert.timestamp for alert in self.alerts),
                    default=0
                ) if self.alerts else 0
            }


class PerformanceMonitor:
    """Monitors system performance and identifies bottlenecks."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance monitor.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics = metrics_collector
        self.active_timers: Dict[str, float] = {}
        self.lock = threading.RLock()
        
    @contextlib.contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics.record_timing(name, duration, tags)
            
    def start_timer(self, name: str) -> str:
        """Start a named timer."""
        timer_id = f"{name}_{int(time.time() * 1000)}"
        with self.lock:
            self.active_timers[timer_id] = time.time()
        return timer_id
        
    def stop_timer(self, timer_id: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Stop a named timer and record the duration."""
        with self.lock:
            if timer_id not in self.active_timers:
                raise MonitoringError(f"Timer {timer_id} not found")
                
            start_time = self.active_timers.pop(timer_id)
            duration = time.time() - start_time
            
            # Extract operation name from timer_id
            operation_name = timer_id.rsplit("_", 1)[0]
            self.metrics.record_timing(f"{operation_name}.duration", duration, tags)
            
            return duration
            
    def monitor_function_performance(self, func: Callable) -> Callable:
        """Decorator to monitor function performance."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.timer(f"function.{func.__name__}"):
                try:
                    result = func(*args, **kwargs)
                    self.metrics.increment_counter(f"function.{func.__name__}.success")
                    return result
                except Exception as e:
                    self.metrics.increment_counter(f"function.{func.__name__}.error")
                    raise
                    
        return wrapper
        
    def get_performance_summary(self, time_window: float = 300.0) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        
        for metric_name in self.metrics.metrics.keys():
            if ".duration" in metric_name:
                stats = self.metrics.get_metric_summary(metric_name, time_window)
                if stats:
                    summary[metric_name] = stats
                    
        return summary


class ObservabilityManager:
    """Central observability management system."""
    
    def __init__(self):
        """Initialize observability manager."""
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.performance_monitor = PerformanceMonitor(self.metrics_collector)
        
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.check_interval = 30.0
        
        self.logger = logging.getLogger(__name__)
        
        # Register default alert handlers
        self._setup_default_handlers()
        
    def _setup_default_handlers(self):
        """Setup default alert handlers."""
        def log_alert(alert: Alert):
            level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.CRITICAL: logging.ERROR,
                AlertSeverity.EMERGENCY: logging.CRITICAL
            }[alert.severity]
            
            self.logger.log(level, f"ALERT: {alert.name} - {alert.message}", 
                          extra={"alert_data": alert.to_dict()})
            
        self.alert_manager.add_alert_handler(log_alert)
        
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Observability monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Observability monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_system_health()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(min(self.check_interval, 60.0))
                
    def _check_system_health(self):
        """Check overall system health and fire alerts."""
        # Get recent metrics
        metrics_summary = self.metrics_collector.get_all_metrics()
        
        # Check for performance issues
        for metric_name, stats in metrics_summary.items():
            if not stats:
                continue
                
            # Check error rates
            if "error" in metric_name and stats.get("total", 0) > 10:
                self.alert_manager.fire_alert(
                    name=f"high_error_rate_{metric_name}",
                    message=f"High error rate detected for {metric_name}: {stats['total']} errors",
                    severity=AlertSeverity.WARNING,
                    source="metrics_monitor"
                )
                
            # Check response times
            if "duration" in metric_name and stats.get("avg", 0) > 5.0:
                self.alert_manager.fire_alert(
                    name=f"slow_response_{metric_name}",
                    message=f"Slow response time for {metric_name}: {stats['avg']:.2f}s average",
                    severity=AlertSeverity.WARNING,
                    source="metrics_monitor"
                )
                
    def record_business_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a business/domain-specific metric."""
        self.metrics_collector.record_metric(f"business.{name}", value, tags)
        
    def record_audit_event(self, event_type: str, details: Optional[Dict[str, Any]] = None):
        """Record an audit trail event for observability."""
        self.metrics_collector.increment_counter(f"audit.{event_type}", tags=details)
        
        # Log significant events
        if event_type in ["privacy_violation", "integrity_failure", "compliance_issue"]:
            self.alert_manager.fire_alert(
                name=f"audit_event_{event_type}",
                message=f"Audit event: {event_type}",
                severity=AlertSeverity.CRITICAL,
                source="audit_system",
                tags=details or {}
            )
            
    def get_observability_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive observability dashboard data."""
        return {
            "timestamp": time.time(),
            "metrics_summary": self.metrics_collector.get_all_metrics(),
            "alert_summary": self.alert_manager.get_alert_summary(),
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "active_alerts": [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
            "system_health": "healthy"  # Would integrate with health monitoring
        }
        
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        dashboard_data = self.get_observability_dashboard()
        
        if format_type == "json":
            return json.dumps(dashboard_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


# Global observability instance
global_observability = ObservabilityManager()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    return global_observability.metrics_collector


def get_alert_manager() -> AlertManager:
    """Get global alert manager."""
    return global_observability.alert_manager


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    return global_observability.performance_monitor


@contextlib.contextmanager
def monitor_operation(name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager to monitor an operation."""
    with global_observability.performance_monitor.timer(name, tags):
        try:
            yield
        except Exception as e:
            global_observability.metrics_collector.increment_counter(
                f"{name}.error",
                tags={**(tags or {}), "error_type": type(e).__name__}
            )
            raise


def monitor_function(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    return global_observability.performance_monitor.monitor_function_performance(func)


def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Record a metric point."""
    global_observability.metrics_collector.record_metric(name, value, tags)


def fire_alert(
    name: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    source: str = "application",
    tags: Optional[Dict[str, str]] = None
):
    """Fire a system alert."""
    global_observability.alert_manager.fire_alert(
        name=name,
        message=message,
        severity=severity,
        source=source,
        tags=tags
    )