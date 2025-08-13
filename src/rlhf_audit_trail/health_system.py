"""Advanced health monitoring and error recovery system for RLHF audit trail.

This module provides comprehensive system health monitoring, automatic error
recovery, circuit breaker patterns, and self-healing capabilities.
"""

import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
import queue
import contextlib
from pathlib import Path

from .exceptions import (
    MonitoringError, 
    ResourceExhaustedError, 
    NetworkError,
    AuditTrailError,
    get_error_details
)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class HealthMetric:
    """Individual health metric data."""
    name: str
    value: float
    threshold: float
    status: HealthStatus
    timestamp: float
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SystemHealth:
    """Overall system health state."""
    overall_status: HealthStatus
    metrics: Dict[str, HealthMetric]
    last_updated: float
    uptime_seconds: float
    error_count: int
    recovery_attempts: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_status": self.overall_status.value,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "last_updated": self.last_updated,
            "uptime_seconds": self.uptime_seconds,
            "error_count": self.error_count,
            "recovery_attempts": self.recovery_attempts
        }


class CircuitBreaker:
    """Circuit breaker for fault tolerance and graceful degradation."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exceptions: tuple = (Exception,)
    ):
        """Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Number of failures before opening
            recovery_timeout: Time before attempting recovery
            expected_exceptions: Exceptions to catch
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0
        self.success_count = 0
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.logger.info(f"Circuit breaker {self.name} transitioning to half-open")
            else:
                raise NetworkError(
                    f"Circuit breaker {self.name} is open",
                    endpoint=self.name,
                    timeout=True
                )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
            
        except self.expected_exceptions as e:
            self._record_failure()
            raise
            
    def _record_success(self):
        """Record successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Recovery threshold
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info(f"Circuit breaker {self.name} recovered")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
            
    def _record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.success_count = 0
            self.logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
            
    def reset(self):
        """Reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.logger.info(f"Circuit breaker {self.name} manually reset")


class HealthMonitor:
    """Comprehensive system health monitoring with automatic recovery."""
    
    def __init__(self, check_interval: float = 30.0):
        """Initialize health monitor.
        
        Args:
            check_interval: Health check interval in seconds
        """
        self.check_interval = check_interval
        self.start_time = time.time()
        
        # Health metrics and thresholds
        self.metrics: Dict[str, HealthMetric] = {}
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 0.05,
            "response_time": 5.0,
            "privacy_budget_usage": 90.0,
            "storage_connectivity": 1.0
        }
        
        # Error tracking
        self.error_history: deque = deque(maxlen=1000)
        self.error_count = 0
        self.recovery_attempts = 0
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Alerting
        self.alert_handlers: List[Callable] = []
        
        self.logger = logging.getLogger(__name__)
        
    def add_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ) -> CircuitBreaker:
        """Add a circuit breaker for a component."""
        breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        self.circuit_breakers[name] = breaker
        return breaker
        
    def add_recovery_strategy(self, component: str, strategy: Callable):
        """Add automatic recovery strategy for a component."""
        self.recovery_strategies[component] = strategy
        
    def add_alert_handler(self, handler: Callable):
        """Add alert notification handler."""
        self.alert_handlers.append(handler)
        
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Health monitoring started")
        
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self.check_health()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(min(self.check_interval, 60.0))
                
    def check_health(self) -> SystemHealth:
        """Perform comprehensive health check."""
        current_time = time.time()
        
        # Update metrics
        self._check_system_resources()
        self._check_error_rates()
        self._check_component_health()
        
        # Determine overall status
        overall_status = self._calculate_overall_status()
        
        # Create health report
        health = SystemHealth(
            overall_status=overall_status,
            metrics=self.metrics.copy(),
            last_updated=current_time,
            uptime_seconds=current_time - self.start_time,
            error_count=self.error_count,
            recovery_attempts=self.recovery_attempts
        )
        
        # Handle unhealthy conditions
        if overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            self._handle_unhealthy_system(health)
            
        # Send alerts if needed
        self._send_alerts(health)
        
        return health
        
    def _check_system_resources(self):
        """Check system resource usage."""
        try:
            # CPU usage (simplified - would use psutil in production)
            cpu_usage = self._get_cpu_usage()
            self.metrics["cpu_usage"] = HealthMetric(
                name="cpu_usage",
                value=cpu_usage,
                threshold=self.thresholds["cpu_usage"],
                status=HealthStatus.HEALTHY if cpu_usage < self.thresholds["cpu_usage"] else HealthStatus.DEGRADED,
                timestamp=time.time()
            )
            
            # Memory usage
            memory_usage = self._get_memory_usage()
            self.metrics["memory_usage"] = HealthMetric(
                name="memory_usage",
                value=memory_usage,
                threshold=self.thresholds["memory_usage"],
                status=HealthStatus.HEALTHY if memory_usage < self.thresholds["memory_usage"] else HealthStatus.UNHEALTHY,
                timestamp=time.time()
            )
            
            # Disk usage
            disk_usage = self._get_disk_usage()
            self.metrics["disk_usage"] = HealthMetric(
                name="disk_usage",
                value=disk_usage,
                threshold=self.thresholds["disk_usage"],
                status=HealthStatus.HEALTHY if disk_usage < self.thresholds["disk_usage"] else HealthStatus.CRITICAL,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Resource check failed: {e}")
            
    def _check_error_rates(self):
        """Check system error rates."""
        current_time = time.time()
        recent_errors = [
            err for err in self.error_history
            if current_time - err.get("timestamp", 0) < 300  # Last 5 minutes
        ]
        
        error_rate = len(recent_errors) / 300.0  # Errors per second
        
        self.metrics["error_rate"] = HealthMetric(
            name="error_rate",
            value=error_rate,
            threshold=self.thresholds["error_rate"],
            status=HealthStatus.HEALTHY if error_rate < self.thresholds["error_rate"] else HealthStatus.DEGRADED,
            timestamp=current_time,
            details={"recent_errors": len(recent_errors)}
        )
        
    def _check_component_health(self):
        """Check health of individual components."""
        # Check circuit breaker states
        unhealthy_breakers = [
            name for name, breaker in self.circuit_breakers.items()
            if breaker.state != CircuitState.CLOSED
        ]
        
        if unhealthy_breakers:
            self.metrics["circuit_breakers"] = HealthMetric(
                name="circuit_breakers",
                value=len(unhealthy_breakers),
                threshold=0.0,
                status=HealthStatus.DEGRADED if len(unhealthy_breakers) < 3 else HealthStatus.UNHEALTHY,
                timestamp=time.time(),
                details={"open_breakers": unhealthy_breakers}
            )
        else:
            self.metrics["circuit_breakers"] = HealthMetric(
                name="circuit_breakers",
                value=0.0,
                threshold=0.0,
                status=HealthStatus.HEALTHY,
                timestamp=time.time()
            )
            
    def _calculate_overall_status(self) -> HealthStatus:
        """Calculate overall system health status."""
        if not self.metrics:
            return HealthStatus.UNKNOWN
            
        critical_count = sum(1 for m in self.metrics.values() if m.status == HealthStatus.CRITICAL)
        unhealthy_count = sum(1 for m in self.metrics.values() if m.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for m in self.metrics.values() if m.status == HealthStatus.DEGRADED)
        
        if critical_count > 0:
            return HealthStatus.CRITICAL
        elif unhealthy_count > 2:
            return HealthStatus.CRITICAL
        elif unhealthy_count > 0 or degraded_count > 3:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
            
    def _handle_unhealthy_system(self, health: SystemHealth):
        """Handle unhealthy system conditions with recovery attempts."""
        self.logger.warning(f"System unhealthy: {health.overall_status.value}")
        
        # Attempt automatic recovery
        for metric_name, metric in health.metrics.items():
            if metric.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                self._attempt_recovery(metric_name, metric)
                
    def _attempt_recovery(self, component: str, metric: HealthMetric):
        """Attempt automatic recovery for a component."""
        if component in self.recovery_strategies:
            try:
                self.logger.info(f"Attempting recovery for {component}")
                self.recovery_strategies[component](metric)
                self.recovery_attempts += 1
                
            except Exception as e:
                self.logger.error(f"Recovery failed for {component}: {e}")
                self._record_error(e, context={"recovery_component": component})
                
    def _send_alerts(self, health: SystemHealth):
        """Send health alerts to registered handlers."""
        if health.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            for handler in self.alert_handlers:
                try:
                    handler(health)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")
                    
    def record_error(self, error: Exception, context: Optional[Dict] = None):
        """Record an error for health monitoring."""
        self._record_error(error, context)
        
    def _record_error(self, error: Exception, context: Optional[Dict] = None):
        """Internal error recording."""
        error_record = {
            "timestamp": time.time(),
            "error": get_error_details(error),
            "context": context or {}
        }
        
        self.error_history.append(error_record)
        self.error_count += 1
        
        # Log error
        self.logger.error(
            f"Error recorded: {error}",
            extra={"error_context": context}
        )
        
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        health = self.check_health()
        return health.to_dict()
        
    def get_component_status(self, component: str) -> Optional[HealthStatus]:
        """Get health status of specific component."""
        metric = self.metrics.get(component)
        return metric.status if metric else None
        
    # Simplified system resource methods (would use psutil in production)
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            with open("/proc/loadavg", "r") as f:
                load_avg = float(f.read().split()[0])
            return min(load_avg * 25.0, 100.0)  # Simplified conversion
        except:
            return 0.0
            
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            
            mem_total = 0
            mem_available = 0
            
            for line in lines:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1])
                    
            if mem_total > 0:
                return ((mem_total - mem_available) / mem_total) * 100.0
        except:
            pass
        return 0.0
        
    def _get_disk_usage(self) -> float:
        """Get disk usage percentage."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            return (used / total) * 100.0
        except:
            return 0.0


class SelfHealingSystem:
    """Self-healing system that automatically recovers from failures."""
    
    def __init__(self):
        """Initialize self-healing system."""
        self.health_monitor = HealthMonitor()
        self.healing_strategies: Dict[str, List[Callable]] = defaultdict(list)
        self.healing_history: List[Dict] = []
        
        self.logger = logging.getLogger(__name__)
        
        # Register default healing strategies
        self._register_default_strategies()
        
    def _register_default_strategies(self):
        """Register default self-healing strategies."""
        # Memory usage recovery
        self.add_healing_strategy("memory_usage", self._heal_memory_usage)
        
        # Disk usage recovery
        self.add_healing_strategy("disk_usage", self._heal_disk_usage)
        
        # Error rate recovery
        self.add_healing_strategy("error_rate", self._heal_error_rate)
        
        # Circuit breaker recovery
        self.add_healing_strategy("circuit_breakers", self._heal_circuit_breakers)
        
    def add_healing_strategy(self, component: str, strategy: Callable):
        """Add a healing strategy for a component."""
        self.healing_strategies[component].append(strategy)
        
    def start(self):
        """Start the self-healing system."""
        # Add recovery strategies to health monitor
        for component, strategies in self.healing_strategies.items():
            if strategies:
                self.health_monitor.add_recovery_strategy(
                    component, 
                    lambda metric, strats=strategies: self._execute_healing_strategies(strats, metric)
                )
                
        self.health_monitor.start_monitoring()
        self.logger.info("Self-healing system started")
        
    def stop(self):
        """Stop the self-healing system."""
        self.health_monitor.stop_monitoring()
        self.logger.info("Self-healing system stopped")
        
    def _execute_healing_strategies(self, strategies: List[Callable], metric: HealthMetric):
        """Execute healing strategies for a component."""
        for strategy in strategies:
            try:
                strategy(metric)
                # Record successful healing
                self.healing_history.append({
                    "timestamp": time.time(),
                    "component": metric.name,
                    "strategy": strategy.__name__,
                    "status": "success",
                    "metric_value": metric.value
                })
                return  # Stop on first successful strategy
                
            except Exception as e:
                self.logger.warning(f"Healing strategy {strategy.__name__} failed: {e}")
                # Record failed healing attempt
                self.healing_history.append({
                    "timestamp": time.time(),
                    "component": metric.name,
                    "strategy": strategy.__name__,
                    "status": "failed",
                    "error": str(e),
                    "metric_value": metric.value
                })
                
    def _heal_memory_usage(self, metric: HealthMetric):
        """Healing strategy for high memory usage."""
        if metric.value > 85.0:
            import gc
            # Force garbage collection
            collected = gc.collect()
            self.logger.info(f"Garbage collection freed {collected} objects")
            
            # Additional memory optimization could be added here
            
    def _heal_disk_usage(self, metric: HealthMetric):
        """Healing strategy for high disk usage."""
        if metric.value > 90.0:
            # Clean up temporary files
            import tempfile
            import shutil
            
            temp_dir = Path(tempfile.gettempdir())
            if temp_dir.exists():
                for temp_file in temp_dir.glob("*audit*"):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file)
                    except:
                        pass
                        
            self.logger.info("Cleaned up temporary files")
            
    def _heal_error_rate(self, metric: HealthMetric):
        """Healing strategy for high error rates."""
        if metric.value > 0.05:
            # Reset circuit breakers to give components a chance to recover
            for breaker in self.health_monitor.circuit_breakers.values():
                if breaker.state == CircuitState.OPEN:
                    breaker.reset()
                    
            self.logger.info("Reset circuit breakers to reduce error rates")
            
    def _heal_circuit_breakers(self, metric: HealthMetric):
        """Healing strategy for open circuit breakers."""
        if metric.value > 0:
            # Attempt to reset circuit breakers after a cooldown period
            current_time = time.time()
            
            for breaker in self.health_monitor.circuit_breakers.values():
                if (breaker.state == CircuitState.OPEN and 
                    current_time - breaker.last_failure_time > breaker.recovery_timeout * 2):
                    breaker.reset()
                    self.logger.info(f"Force reset circuit breaker: {breaker.name}")


# Global health monitoring instance
global_health_monitor = HealthMonitor()
global_self_healing = SelfHealingSystem()


@contextlib.contextmanager
def health_monitoring():
    """Context manager for health monitoring."""
    global_self_healing.start()
    try:
        yield global_health_monitor
    finally:
        global_self_healing.stop()


def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    return global_health_monitor.get_health_report()


def record_system_error(error: Exception, context: Optional[Dict] = None):
    """Record an error in the global health monitoring system."""
    global_health_monitor.record_error(error, context)