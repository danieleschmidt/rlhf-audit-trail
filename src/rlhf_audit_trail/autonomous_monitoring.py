"""Autonomous Monitoring System for RLHF Audit Trail.

Implements self-monitoring, adaptive alerting, and intelligent
system health management with predictive analytics.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Set
import logging
from collections import deque, defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy for basic functionality
    class MockNumpy:
        def mean(self, data): return sum(data) / len(data) if data else 0
        def std(self, data): 
            if not data: return 0
            mean_val = self.mean(data)
            return (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5
        def percentile(self, data, p): 
            if not data: return 0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p / 100
            return sorted_data[int(k)]
    np = MockNumpy()


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class MetricData:
    """Represents a system metric."""
    name: str
    value: float
    timestamp: float
    unit: str
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    """Represents a system alert."""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    metric_name: str
    threshold: float
    actual_value: float
    timestamp: float
    resolved: bool = False
    acknowledged: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class HealthCheck:
    """Represents a health check result."""
    component: str
    status: HealthStatus
    message: str
    timestamp: float
    duration: float
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class AutonomousMonitor:
    """Autonomous Monitoring System.
    
    Provides intelligent monitoring with adaptive thresholds,
    predictive alerting, and self-healing capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize autonomous monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or {}
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = {}
        self.health_checks = {}
        self.alert_rules = {}
        self.adaptive_thresholds = {}
        
        # Monitoring state
        self.is_running = False
        self.monitoring_tasks = []
        self.last_health_check = 0
        self.system_health = HealthStatus.HEALTHY
        
        self.logger = logging.getLogger(__name__)
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default monitoring rules."""
        # Response time alerts
        self.add_alert_rule(
            metric_name="response_time_ms",
            threshold=200.0,
            alert_level=AlertLevel.WARNING,
            title="High Response Time",
            comparison="greater"
        )
        
        self.add_alert_rule(
            metric_name="response_time_ms",
            threshold=500.0,
            alert_level=AlertLevel.ERROR,
            title="Very High Response Time",
            comparison="greater"
        )
        
        # Memory usage alerts
        self.add_alert_rule(
            metric_name="memory_usage_mb",
            threshold=512.0,
            alert_level=AlertLevel.WARNING,
            title="High Memory Usage",
            comparison="greater"
        )
        
        self.add_alert_rule(
            metric_name="memory_usage_mb",
            threshold=1024.0,
            alert_level=AlertLevel.CRITICAL,
            title="Critical Memory Usage",
            comparison="greater"
        )
        
        # Error rate alerts
        self.add_alert_rule(
            metric_name="error_rate_percent",
            threshold=5.0,
            alert_level=AlertLevel.WARNING,
            title="Elevated Error Rate",
            comparison="greater"
        )
        
        self.add_alert_rule(
            metric_name="error_rate_percent",
            threshold=15.0,
            alert_level=AlertLevel.CRITICAL,
            title="Critical Error Rate",
            comparison="greater"
        )
        
        # Privacy budget alerts
        self.add_alert_rule(
            metric_name="privacy_budget_used_percent",
            threshold=80.0,
            alert_level=AlertLevel.WARNING,
            title="High Privacy Budget Usage",
            comparison="greater"
        )
        
        self.add_alert_rule(
            metric_name="privacy_budget_used_percent",
            threshold=95.0,
            alert_level=AlertLevel.CRITICAL,
            title="Critical Privacy Budget Usage",
            comparison="greater"
        )
    
    def add_alert_rule(
        self,
        metric_name: str,
        threshold: float,
        alert_level: AlertLevel,
        title: str,
        comparison: str = "greater",
        window_minutes: int = 5
    ):
        """Add an alert rule.
        
        Args:
            metric_name: Name of metric to monitor
            threshold: Alert threshold value
            alert_level: Alert severity level
            title: Alert title
            comparison: Comparison operator ('greater', 'less', 'equal')
            window_minutes: Time window for evaluation
        """
        rule_id = f"{metric_name}_{threshold}_{comparison}"
        self.alert_rules[rule_id] = {
            'metric_name': metric_name,
            'threshold': threshold,
            'alert_level': alert_level,
            'title': title,
            'comparison': comparison,
            'window_minutes': window_minutes,
            'last_triggered': 0,
            'cooldown_minutes': 15
        }
        
        self.logger.info(f"Added alert rule: {title} for {metric_name}")
    
    def record_metric(self, metric: MetricData):
        """Record a metric value.
        
        Args:
            metric: Metric data to record
        """
        self.metrics_buffer[metric.name].append(metric)
        
        # Update adaptive thresholds
        self._update_adaptive_threshold(metric.name)
        
        # Check for alerts
        self._evaluate_alerts(metric)
    
    def _update_adaptive_threshold(self, metric_name: str):
        """Update adaptive thresholds based on historical data.
        
        Args:
            metric_name: Name of metric to update threshold for
        """
        if len(self.metrics_buffer[metric_name]) < 50:
            return  # Need sufficient data
        
        recent_values = [m.value for m in list(self.metrics_buffer[metric_name])[-50:]]
        
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        
        # Update adaptive threshold (mean + 2 * std for anomaly detection)
        self.adaptive_thresholds[metric_name] = {
            'mean': mean_val,
            'std': std_val,
            'anomaly_threshold': mean_val + 2 * std_val,
            'updated_at': time.time()
        }
    
    def _evaluate_alerts(self, metric: MetricData):
        """Evaluate alert rules for a metric.
        
        Args:
            metric: Metric to evaluate
        """
        current_time = time.time()
        
        for rule_id, rule in self.alert_rules.items():
            if rule['metric_name'] != metric.name:
                continue
            
            # Check cooldown
            if current_time - rule['last_triggered'] < rule['cooldown_minutes'] * 60:
                continue
            
            # Evaluate condition
            triggered = False
            if rule['comparison'] == 'greater':
                triggered = metric.value > rule['threshold']
            elif rule['comparison'] == 'less':
                triggered = metric.value < rule['threshold']
            elif rule['comparison'] == 'equal':
                triggered = abs(metric.value - rule['threshold']) < 0.001
            
            if triggered:
                self._trigger_alert(rule, metric)
                rule['last_triggered'] = current_time
    
    def _trigger_alert(self, rule: Dict[str, Any], metric: MetricData):
        """Trigger an alert.
        
        Args:
            rule: Alert rule that was triggered
            metric: Metric that triggered the alert
        """
        alert_id = str(uuid.uuid4())
        alert = Alert(
            alert_id=alert_id,
            level=rule['alert_level'],
            title=rule['title'],
            message=f"{rule['title']}: {metric.name} = {metric.value} {metric.unit} (threshold: {rule['threshold']})",
            metric_name=metric.name,
            threshold=rule['threshold'],
            actual_value=metric.value,
            timestamp=metric.timestamp,
            metadata={
                'rule_id': f"{rule['metric_name']}_{rule['threshold']}_{rule['comparison']}",
                'comparison': rule['comparison'],
                'tags': metric.tags
            }
        )
        
        self.alerts[alert_id] = alert
        self.logger.warning(f"Alert triggered: {alert.title} - {alert.message}")
        
        # Update system health based on alert level
        if alert.level == AlertLevel.CRITICAL:
            self.system_health = HealthStatus.CRITICAL
        elif alert.level == AlertLevel.ERROR and self.system_health == HealthStatus.HEALTHY:
            self.system_health = HealthStatus.UNHEALTHY
        elif alert.level == AlertLevel.WARNING and self.system_health == HealthStatus.HEALTHY:
            self.system_health = HealthStatus.DEGRADED
    
    async def start_monitoring(self):
        """Start autonomous monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._monitor_system_health()),
            asyncio.create_task(self._monitor_metrics_collection()),
            asyncio.create_task(self._monitor_alert_resolution()),
            asyncio.create_task(self._generate_insights())
        ]
        
        self.logger.info("Started autonomous monitoring")
    
    async def stop_monitoring(self):
        """Stop autonomous monitoring."""
        self.is_running = False
        
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks = []
        
        self.logger.info("Stopped autonomous monitoring")
    
    async def _monitor_system_health(self):
        """Monitor overall system health."""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Perform health checks
                health_results = await self._perform_health_checks()
                
                # Determine overall health
                overall_health = self._calculate_overall_health(health_results)
                self.system_health = overall_health
                
                # Record health metric
                self.record_metric(MetricData(
                    name="system_health_score",
                    value=self._health_to_score(overall_health),
                    timestamp=current_time,
                    unit="score",
                    tags={"component": "system"}
                ))
                
                self.last_health_check = current_time
                
            except Exception as e:
                self.logger.error(f"Error in system health monitoring: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _monitor_metrics_collection(self):
        """Monitor metrics collection and generate system metrics."""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Generate system metrics
                await self._collect_system_metrics(current_time)
                
                # Clean up old metrics
                self._cleanup_old_metrics(current_time)
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
            
            await asyncio.sleep(10)  # Collect every 10 seconds
    
    async def _monitor_alert_resolution(self):
        """Monitor and auto-resolve alerts."""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check for alerts that can be auto-resolved
                for alert_id, alert in list(self.alerts.items()):
                    if alert.resolved:
                        continue
                    
                    # Check if condition no longer exists
                    if await self._check_alert_resolution(alert):
                        alert.resolved = True
                        alert.metadata['resolved_at'] = current_time
                        alert.metadata['auto_resolved'] = True
                        
                        self.logger.info(f"Auto-resolved alert: {alert.title}")
                
                # Update system health based on active alerts
                self._update_system_health_from_alerts()
                
            except Exception as e:
                self.logger.error(f"Error in alert resolution monitoring: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _generate_insights(self):
        """Generate predictive insights and recommendations."""
        while self.is_running:
            try:
                # Analyze trends and generate insights
                insights = await self._analyze_trends()
                
                if insights:
                    self.logger.info(f"Generated {len(insights)} system insights")
                
            except Exception as e:
                self.logger.error(f"Error in insights generation: {e}")
            
            await asyncio.sleep(300)  # Generate every 5 minutes
    
    async def _perform_health_checks(self) -> Dict[str, HealthCheck]:
        """Perform comprehensive health checks.
        
        Returns:
            Dictionary of health check results
        """
        checks = {}
        current_time = time.time()
        
        # Core system check
        start_time = time.time()
        try:
            from .core import AuditableRLHF
            auditor = AuditableRLHF(model_name="health-check")
            privacy_report = auditor.get_privacy_report()
            
            checks['core_system'] = HealthCheck(
                component="core_system",
                status=HealthStatus.HEALTHY,
                message="Core system operational",
                timestamp=current_time,
                duration=time.time() - start_time,
                details={'privacy_report': 'available'}
            )
        except Exception as e:
            checks['core_system'] = HealthCheck(
                component="core_system",
                status=HealthStatus.CRITICAL,
                message=f"Core system error: {str(e)}",
                timestamp=current_time,
                duration=time.time() - start_time,
                details={'error': str(e)}
            )
        
        # Storage system check
        start_time = time.time()
        try:
            from .storage import LocalStorage
            storage = LocalStorage()
            
            # Test basic operations
            await storage.store("health-check.json", {"status": "ok"})
            data = await storage.retrieve("health-check.json")
            
            checks['storage_system'] = HealthCheck(
                component="storage_system",
                status=HealthStatus.HEALTHY,
                message="Storage system operational",
                timestamp=current_time,
                duration=time.time() - start_time
            )
        except Exception as e:
            checks['storage_system'] = HealthCheck(
                component="storage_system",
                status=HealthStatus.DEGRADED,
                message=f"Storage system issues: {str(e)}",
                timestamp=current_time,
                duration=time.time() - start_time,
                details={'error': str(e)}
            )
        
        # Privacy engine check
        start_time = time.time()
        try:
            from .privacy import DifferentialPrivacyEngine
            from .config import PrivacyConfig
            
            config = PrivacyConfig()
            engine = DifferentialPrivacyEngine(config)
            cost = engine.estimate_epsilon_cost(10)
            
            checks['privacy_engine'] = HealthCheck(
                component="privacy_engine",
                status=HealthStatus.HEALTHY,
                message="Privacy engine operational",
                timestamp=current_time,
                duration=time.time() - start_time,
                details={'epsilon_cost_estimation': cost}
            )
        except Exception as e:
            checks['privacy_engine'] = HealthCheck(
                component="privacy_engine",
                status=HealthStatus.DEGRADED,
                message=f"Privacy engine issues: {str(e)}",
                timestamp=current_time,
                duration=time.time() - start_time,
                details={'error': str(e)}
            )
        
        # Store health check results
        self.health_checks.update(checks)
        return checks
    
    def _calculate_overall_health(self, health_results: Dict[str, HealthCheck]) -> HealthStatus:
        """Calculate overall system health from individual checks.
        
        Args:
            health_results: Individual health check results
            
        Returns:
            Overall health status
        """
        if not health_results:
            return HealthStatus.UNHEALTHY
        
        status_counts = defaultdict(int)
        for check in health_results.values():
            status_counts[check.status] += 1
        
        total_checks = len(health_results)
        
        # Determine overall health
        if status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > total_checks * 0.3:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _health_to_score(self, health: HealthStatus) -> float:
        """Convert health status to numeric score.
        
        Args:
            health: Health status
            
        Returns:
            Numeric health score (0-1)
        """
        mapping = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.7,
            HealthStatus.UNHEALTHY: 0.3,
            HealthStatus.CRITICAL: 0.0
        }
        return mapping.get(health, 0.0)
    
    async def _collect_system_metrics(self, current_time: float):
        """Collect system performance metrics.
        
        Args:
            current_time: Current timestamp
        """
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Memory usage
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.record_metric(MetricData(
                name="memory_usage_mb",
                value=memory_mb,
                timestamp=current_time,
                unit="MB",
                tags={"component": "system"}
            ))
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            self.record_metric(MetricData(
                name="cpu_usage_percent",
                value=cpu_percent,
                timestamp=current_time,
                unit="%",
                tags={"component": "system"}
            ))
            
        except ImportError:
            # psutil not available, record placeholder metrics
            self.record_metric(MetricData(
                name="memory_usage_mb",
                value=100.0,  # Placeholder
                timestamp=current_time,
                unit="MB",
                tags={"component": "system", "placeholder": "true"}
            ))
    
    def _cleanup_old_metrics(self, current_time: float):
        """Clean up old metrics data.
        
        Args:
            current_time: Current timestamp
        """
        cutoff_time = current_time - (24 * 60 * 60)  # Keep 24 hours
        
        for metric_name in self.metrics_buffer:
            buffer = self.metrics_buffer[metric_name]
            while buffer and buffer[0].timestamp < cutoff_time:
                buffer.popleft()
    
    async def _check_alert_resolution(self, alert: Alert) -> bool:
        """Check if an alert condition has been resolved.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert can be resolved
        """
        if alert.metric_name not in self.metrics_buffer:
            return False
        
        recent_metrics = [
            m for m in self.metrics_buffer[alert.metric_name]
            if m.timestamp > alert.timestamp
        ]
        
        if len(recent_metrics) < 3:
            return False  # Need more data points
        
        # Check if recent values are below threshold
        recent_values = [m.value for m in recent_metrics[-5:]]
        
        if alert.metadata.get('comparison') == 'greater':
            return all(v < alert.threshold for v in recent_values)
        elif alert.metadata.get('comparison') == 'less':
            return all(v > alert.threshold for v in recent_values)
        
        return False
    
    def _update_system_health_from_alerts(self):
        """Update system health based on active alerts."""
        active_alerts = [a for a in self.alerts.values() if not a.resolved]
        
        if not active_alerts:
            self.system_health = HealthStatus.HEALTHY
            return
        
        critical_count = sum(1 for a in active_alerts if a.level == AlertLevel.CRITICAL)
        error_count = sum(1 for a in active_alerts if a.level == AlertLevel.ERROR)
        warning_count = sum(1 for a in active_alerts if a.level == AlertLevel.WARNING)
        
        if critical_count > 0:
            self.system_health = HealthStatus.CRITICAL
        elif error_count > 0:
            self.system_health = HealthStatus.UNHEALTHY
        elif warning_count > 2:
            self.system_health = HealthStatus.DEGRADED
        else:
            self.system_health = HealthStatus.DEGRADED
    
    async def _analyze_trends(self) -> List[Dict[str, Any]]:
        """Analyze metrics trends and generate insights.
        
        Returns:
            List of insights and recommendations
        """
        insights = []
        
        for metric_name, buffer in self.metrics_buffer.items():
            if len(buffer) < 20:
                continue
            
            recent_values = [m.value for m in list(buffer)[-20:]]
            older_values = [m.value for m in list(buffer)[-40:-20]] if len(buffer) >= 40 else []
            
            if not older_values:
                continue
            
            recent_mean = np.mean(recent_values)
            older_mean = np.mean(older_values)
            
            # Detect significant trends
            if recent_mean > older_mean * 1.2:
                insights.append({
                    'type': 'increasing_trend',
                    'metric': metric_name,
                    'increase_percent': ((recent_mean - older_mean) / older_mean) * 100,
                    'recommendation': f'Monitor {metric_name} closely - showing upward trend',
                    'severity': 'warning' if recent_mean > older_mean * 1.5 else 'info'
                })
            elif recent_mean < older_mean * 0.8:
                insights.append({
                    'type': 'decreasing_trend',
                    'metric': metric_name,
                    'decrease_percent': ((older_mean - recent_mean) / older_mean) * 100,
                    'recommendation': f'{metric_name} showing improvement',
                    'severity': 'info'
                })
        
        return insights
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status.
        
        Returns:
            Dictionary with current system status
        """
        current_time = time.time()
        active_alerts = [a for a in self.alerts.values() if not a.resolved]
        
        return {
            'system_health': self.system_health.value,
            'monitoring_active': self.is_running,
            'last_health_check': self.last_health_check,
            'active_alerts': len(active_alerts),
            'total_alerts': len(self.alerts),
            'metrics_tracked': len(self.metrics_buffer),
            'health_checks': {
                name: {
                    'status': check.status.value,
                    'message': check.message,
                    'last_check': check.timestamp
                }
                for name, check in self.health_checks.items()
            },
            'alert_summary': {
                level.value: sum(
                    1 for a in active_alerts 
                    if a.level == level
                )
                for level in AlertLevel
            }
        }
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for specified time period.
        
        Args:
            hours: Hours of historical data to include
            
        Returns:
            Dictionary with metrics summary
        """
        cutoff_time = time.time() - (hours * 60 * 60)
        summary = {}
        
        for metric_name, buffer in self.metrics_buffer.items():
            recent_metrics = [m for m in buffer if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                continue
            
            values = [m.value for m in recent_metrics]
            summary[metric_name] = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99),
                'unit': recent_metrics[0].unit if recent_metrics else 'unknown'
            }
        
        return summary
    
    def export_monitoring_data(self, output_path: Path) -> None:
        """Export monitoring data to file.
        
        Args:
            output_path: Path to save monitoring data
        """
        monitoring_data = {
            'status': self.get_current_status(),
            'metrics_summary': self.get_metrics_summary(24),  # Last 24 hours
            'alerts': {k: asdict(v) for k, v in self.alerts.items()},
            'health_checks': {k: asdict(v) for k, v in self.health_checks.items()},
            'alert_rules': self.alert_rules,
            'adaptive_thresholds': self.adaptive_thresholds,
            'exported_at': time.time()
        }
        
        output_path.write_text(json.dumps(monitoring_data, indent=2, default=str))
        self.logger.info(f"Monitoring data exported to {output_path}")