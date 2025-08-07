"""Monitoring, logging and health check system for quantum task planner."""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque

from .core import Task, TaskState, QuantumPriority, QuantumTaskPlanner
from .exceptions import QuantumTaskPlannerError


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['status'] = self.status.value
        return result


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    
    # Task metrics
    total_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int
    pending_tasks: int
    
    # Quantum metrics
    coherent_tasks: int
    entangled_pairs: int
    average_amplitude: float
    total_collapses: int
    
    # Performance metrics
    tasks_per_second: float
    average_execution_time: float
    success_rate: float
    
    # System metrics
    memory_usage_mb: float
    cpu_usage_percent: float
    active_schedulers: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class MetricsCollector:
    """Collects and aggregates system metrics."""
    
    def __init__(self, collection_interval: float = 30.0, history_size: int = 1000):
        """Initialize metrics collector.
        
        Args:
            collection_interval: Seconds between metric collections
            history_size: Maximum number of metric snapshots to keep
        """
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.task_execution_times: deque = deque(maxlen=1000)
        self.task_completion_times: deque = deque(maxlen=100)
        self.is_running = False
        self.collection_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger("quantum_planner.metrics")
    
    async def start(self, planner: QuantumTaskPlanner):
        """Start metrics collection."""
        if self.is_running:
            return
        
        self.is_running = True
        self.planner = planner
        self.collection_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Metrics collection started")
    
    async def stop(self):
        """Stop metrics collection."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        try:
            while self.is_running:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Log periodic summary
                if len(self.metrics_history) % 10 == 0:
                    self.logger.info(f"Metrics: {metrics.total_tasks} tasks, "
                                   f"{metrics.success_rate:.1%} success rate, "
                                   f"{metrics.tasks_per_second:.2f} tasks/sec")
                
                await asyncio.sleep(self.collection_interval)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in metrics collection: {e}")
    
    async def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        now = datetime.now()
        
        # Get system state
        system_state = self.planner.get_system_state()
        tasks_by_state = system_state["tasks_by_state"]
        quantum_metrics = system_state["quantum_metrics"]
        execution_metrics = system_state["execution_metrics"]
        
        # Calculate performance metrics
        tasks_per_second = self._calculate_throughput()
        avg_execution_time = self._calculate_average_execution_time()
        success_rate = execution_metrics.get("success_rate", 0.0)
        
        # System resource metrics (simplified - would integrate with actual monitoring)
        memory_usage = self._estimate_memory_usage()
        cpu_usage = self._estimate_cpu_usage()
        
        return SystemMetrics(
            timestamp=now,
            total_tasks=system_state["total_tasks"],
            running_tasks=tasks_by_state.get("running", 0),
            completed_tasks=tasks_by_state.get("completed", 0),
            failed_tasks=tasks_by_state.get("failed", 0),
            pending_tasks=tasks_by_state.get("pending", 0),
            coherent_tasks=quantum_metrics["coherent_tasks"],
            entangled_pairs=quantum_metrics["active_entanglements"],
            average_amplitude=quantum_metrics["average_amplitude"],
            total_collapses=quantum_metrics["total_collapses"],
            tasks_per_second=tasks_per_second,
            average_execution_time=avg_execution_time,
            success_rate=success_rate,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            active_schedulers=1  # Simplified
        )
    
    def record_task_execution_start(self, task_id: str):
        """Record start of task execution."""
        self.task_execution_times.append((task_id, time.time(), "start"))
    
    def record_task_execution_end(self, task_id: str, success: bool):
        """Record end of task execution."""
        self.task_execution_times.append((task_id, time.time(), "end"))
        if success:
            self.task_completion_times.append(time.time())
    
    def _calculate_throughput(self) -> float:
        """Calculate tasks completed per second."""
        if len(self.task_completion_times) < 2:
            return 0.0
        
        # Use recent completions (last 5 minutes)
        cutoff_time = time.time() - 300
        recent_completions = [t for t in self.task_completion_times if t > cutoff_time]
        
        if len(recent_completions) < 2:
            return 0.0
        
        time_span = recent_completions[-1] - recent_completions[0]
        if time_span > 0:
            return len(recent_completions) / time_span
        return 0.0
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average task execution time."""
        execution_pairs = {}
        
        # Match start and end times
        for task_id, timestamp, event_type in self.task_execution_times:
            if task_id not in execution_pairs:
                execution_pairs[task_id] = {}
            execution_pairs[task_id][event_type] = timestamp
        
        # Calculate durations
        durations = []
        for task_id, events in execution_pairs.items():
            if "start" in events and "end" in events:
                duration = events["end"] - events["start"]
                if duration > 0:
                    durations.append(duration)
        
        return sum(durations) / len(durations) if durations else 0.0
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Simplified estimation based on task count
        task_count = len(self.planner.tasks)
        base_usage = 50  # Base system usage
        task_usage = task_count * 0.1  # Assume 0.1 MB per task
        return base_usage + task_usage
    
    def _estimate_cpu_usage(self) -> float:
        """Estimate CPU usage percentage."""
        # Simplified estimation based on running tasks
        running_count = len(self.planner.running_tasks)
        return min(100.0, running_count * 10)  # 10% per running task
    
    def get_recent_metrics(self, count: int = 10) -> List[SystemMetrics]:
        """Get recent metrics snapshots."""
        return list(self.metrics_history)[-count:]
    
    def get_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get aggregated metrics summary for specified duration."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate aggregates
        total_tasks = [m.total_tasks for m in recent_metrics]
        success_rates = [m.success_rate for m in recent_metrics]
        throughput = [m.tasks_per_second for m in recent_metrics]
        execution_times = [m.average_execution_time for m in recent_metrics]
        
        return {
            "duration_minutes": duration_minutes,
            "sample_count": len(recent_metrics),
            "tasks": {
                "min": min(total_tasks) if total_tasks else 0,
                "max": max(total_tasks) if total_tasks else 0,
                "avg": sum(total_tasks) / len(total_tasks) if total_tasks else 0
            },
            "success_rate": {
                "min": min(success_rates) if success_rates else 0,
                "max": max(success_rates) if success_rates else 0,
                "avg": sum(success_rates) / len(success_rates) if success_rates else 0
            },
            "throughput": {
                "min": min(throughput) if throughput else 0,
                "max": max(throughput) if throughput else 0,
                "avg": sum(throughput) / len(throughput) if throughput else 0
            },
            "execution_time": {
                "min": min(execution_times) if execution_times else 0,
                "max": max(execution_times) if execution_times else 0,
                "avg": sum(execution_times) / len(execution_times) if execution_times else 0
            }
        }


class HealthMonitor:
    """System health monitoring and alerting."""
    
    def __init__(
        self, 
        check_interval: float = 60.0,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        """Initialize health monitor.
        
        Args:
            check_interval: Seconds between health checks
            alert_callbacks: Functions to call when alerts are triggered
        """
        self.check_interval = check_interval
        self.alert_callbacks = alert_callbacks or []
        self.health_history: deque = deque(maxlen=100)
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger("quantum_planner.health")
        
        # Health check thresholds
        self.thresholds = {
            "max_failed_tasks_ratio": 0.1,  # 10% failure rate triggers warning
            "max_failed_tasks_critical": 0.25,  # 25% failure rate triggers critical
            "min_coherent_tasks_ratio": 0.8,  # Below 80% coherence triggers warning
            "max_execution_time": 300.0,  # 5 minutes max execution time
            "max_memory_usage_mb": 1000.0,  # 1GB memory limit
            "max_cpu_usage_percent": 90.0  # 90% CPU usage limit
        }
    
    async def start(self, planner: QuantumTaskPlanner, metrics_collector: MetricsCollector):
        """Start health monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.planner = planner
        self.metrics_collector = metrics_collector
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Health monitoring started")
    
    async def stop(self):
        """Stop health monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main health monitoring loop."""
        try:
            while self.is_running:
                health_checks = await self._perform_health_checks()
                
                # Store health status
                overall_status = self._determine_overall_status(health_checks)
                self.health_history.append((datetime.now(), overall_status, health_checks))
                
                # Trigger alerts if needed
                await self._process_alerts(overall_status, health_checks)
                
                # Log health status
                critical_count = sum(1 for check in health_checks if check.status == HealthStatus.CRITICAL)
                warning_count = sum(1 for check in health_checks if check.status == HealthStatus.WARNING)
                
                if critical_count > 0 or warning_count > 0:
                    self.logger.warning(f"Health status: {overall_status.value}, "
                                      f"Critical: {critical_count}, Warning: {warning_count}")
                else:
                    self.logger.debug(f"Health status: {overall_status.value}")
                
                await asyncio.sleep(self.check_interval)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in health monitoring: {e}")
    
    async def _perform_health_checks(self) -> List[HealthCheck]:
        """Perform all health checks."""
        checks = []
        
        try:
            # Get current metrics
            recent_metrics = self.metrics_collector.get_recent_metrics(1)
            current_metrics = recent_metrics[0] if recent_metrics else None
            
            # Task failure rate check
            checks.append(await self._check_task_failure_rate(current_metrics))
            
            # Quantum coherence check
            checks.append(await self._check_quantum_coherence(current_metrics))
            
            # Performance checks
            checks.append(await self._check_execution_performance(current_metrics))
            checks.append(await self._check_resource_usage(current_metrics))
            
            # Dependency integrity check
            checks.append(await self._check_dependency_integrity())
            
            # System responsiveness check
            checks.append(await self._check_system_responsiveness())
            
        except Exception as e:
            checks.append(HealthCheck(
                name="health_check_system",
                status=HealthStatus.CRITICAL,
                message=f"Health check system error: {e}",
                timestamp=datetime.now()
            ))
        
        return checks
    
    async def _check_task_failure_rate(self, metrics: Optional[SystemMetrics]) -> HealthCheck:
        """Check task failure rate."""
        if not metrics or metrics.total_tasks == 0:
            return HealthCheck(
                name="task_failure_rate",
                status=HealthStatus.UNKNOWN,
                message="No task data available",
                timestamp=datetime.now()
            )
        
        failure_rate = metrics.failed_tasks / metrics.total_tasks
        
        if failure_rate >= self.thresholds["max_failed_tasks_critical"]:
            status = HealthStatus.CRITICAL
            message = f"Critical failure rate: {failure_rate:.1%}"
        elif failure_rate >= self.thresholds["max_failed_tasks_ratio"]:
            status = HealthStatus.WARNING
            message = f"High failure rate: {failure_rate:.1%}"
        else:
            status = HealthStatus.HEALTHY
            message = f"Normal failure rate: {failure_rate:.1%}"
        
        return HealthCheck(
            name="task_failure_rate",
            status=status,
            message=message,
            timestamp=datetime.now(),
            details={"failure_rate": failure_rate, "failed_tasks": metrics.failed_tasks}
        )
    
    async def _check_quantum_coherence(self, metrics: Optional[SystemMetrics]) -> HealthCheck:
        """Check quantum coherence levels."""
        if not metrics or metrics.total_tasks == 0:
            return HealthCheck(
                name="quantum_coherence",
                status=HealthStatus.UNKNOWN,
                message="No quantum data available",
                timestamp=datetime.now()
            )
        
        coherence_ratio = metrics.coherent_tasks / metrics.total_tasks
        
        if coherence_ratio < self.thresholds["min_coherent_tasks_ratio"]:
            status = HealthStatus.WARNING
            message = f"Low coherence ratio: {coherence_ratio:.1%}"
        else:
            status = HealthStatus.HEALTHY
            message = f"Good coherence ratio: {coherence_ratio:.1%}"
        
        return HealthCheck(
            name="quantum_coherence",
            status=status,
            message=message,
            timestamp=datetime.now(),
            details={
                "coherence_ratio": coherence_ratio,
                "coherent_tasks": metrics.coherent_tasks,
                "entangled_pairs": metrics.entangled_pairs
            }
        )
    
    async def _check_execution_performance(self, metrics: Optional[SystemMetrics]) -> HealthCheck:
        """Check task execution performance."""
        if not metrics:
            return HealthCheck(
                name="execution_performance",
                status=HealthStatus.UNKNOWN,
                message="No performance data available",
                timestamp=datetime.now()
            )
        
        if metrics.average_execution_time > self.thresholds["max_execution_time"]:
            status = HealthStatus.WARNING
            message = f"Slow execution: {metrics.average_execution_time:.1f}s average"
        elif metrics.tasks_per_second < 0.01:  # Less than 1 task per 100 seconds
            status = HealthStatus.WARNING
            message = f"Low throughput: {metrics.tasks_per_second:.3f} tasks/sec"
        else:
            status = HealthStatus.HEALTHY
            message = f"Good performance: {metrics.tasks_per_second:.2f} tasks/sec"
        
        return HealthCheck(
            name="execution_performance",
            status=status,
            message=message,
            timestamp=datetime.now(),
            details={
                "average_execution_time": metrics.average_execution_time,
                "tasks_per_second": metrics.tasks_per_second,
                "success_rate": metrics.success_rate
            }
        )
    
    async def _check_resource_usage(self, metrics: Optional[SystemMetrics]) -> HealthCheck:
        """Check system resource usage."""
        if not metrics:
            return HealthCheck(
                name="resource_usage",
                status=HealthStatus.UNKNOWN,
                message="No resource data available",
                timestamp=datetime.now()
            )
        
        status = HealthStatus.HEALTHY
        issues = []
        
        if metrics.memory_usage_mb > self.thresholds["max_memory_usage_mb"]:
            status = HealthStatus.WARNING
            issues.append(f"High memory usage: {metrics.memory_usage_mb:.1f}MB")
        
        if metrics.cpu_usage_percent > self.thresholds["max_cpu_usage_percent"]:
            status = HealthStatus.CRITICAL
            issues.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        message = "; ".join(issues) if issues else "Resource usage normal"
        
        return HealthCheck(
            name="resource_usage",
            status=status,
            message=message,
            timestamp=datetime.now(),
            details={
                "memory_usage_mb": metrics.memory_usage_mb,
                "cpu_usage_percent": metrics.cpu_usage_percent
            }
        )
    
    async def _check_dependency_integrity(self) -> HealthCheck:
        """Check task dependency integrity."""
        try:
            from .validation import DependencyValidator
            
            validator = DependencyValidator(self.planner.tasks)
            result = validator.validate_dependency_graph()
            
            if result.errors:
                status = HealthStatus.CRITICAL
                message = f"Dependency errors: {len(result.errors)} issues"
            elif result.warnings:
                status = HealthStatus.WARNING
                message = f"Dependency warnings: {len(result.warnings)} issues"
            else:
                status = HealthStatus.HEALTHY
                message = "Dependencies valid"
            
            return HealthCheck(
                name="dependency_integrity",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={"errors": result.errors, "warnings": result.warnings}
            )
        
        except Exception as e:
            return HealthCheck(
                name="dependency_integrity",
                status=HealthStatus.CRITICAL,
                message=f"Dependency check failed: {e}",
                timestamp=datetime.now()
            )
    
    async def _check_system_responsiveness(self) -> HealthCheck:
        """Check system responsiveness."""
        try:
            start_time = time.time()
            
            # Simple responsiveness test - get system state
            state = self.planner.get_system_state()
            
            response_time = time.time() - start_time
            
            if response_time > 5.0:  # 5 seconds is too slow
                status = HealthStatus.CRITICAL
                message = f"System unresponsive: {response_time:.2f}s"
            elif response_time > 1.0:  # 1 second is slow
                status = HealthStatus.WARNING
                message = f"System slow: {response_time:.2f}s"
            else:
                status = HealthStatus.HEALTHY
                message = f"System responsive: {response_time:.3f}s"
            
            return HealthCheck(
                name="system_responsiveness",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={"response_time": response_time}
            )
        
        except Exception as e:
            return HealthCheck(
                name="system_responsiveness",
                status=HealthStatus.CRITICAL,
                message=f"Responsiveness check failed: {e}",
                timestamp=datetime.now()
            )
    
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall system health status."""
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            return HealthStatus.CRITICAL
        elif any(check.status == HealthStatus.WARNING for check in checks):
            return HealthStatus.WARNING
        elif any(check.status == HealthStatus.UNKNOWN for check in checks):
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    async def _process_alerts(self, overall_status: HealthStatus, checks: List[HealthCheck]):
        """Process health alerts and notifications."""
        if overall_status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            alert_data = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": overall_status.value,
                "checks": [check.to_dict() for check in checks]
            }
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert_data)
                    else:
                        callback(alert_data)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        if not self.health_history:
            return {"status": "unknown", "message": "No health data available"}
        
        timestamp, status, checks = self.health_history[-1]
        
        return {
            "timestamp": timestamp.isoformat(),
            "overall_status": status.value,
            "checks": [check.to_dict() for check in checks]
        }
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)