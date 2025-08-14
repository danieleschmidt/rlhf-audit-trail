"""Enhanced Reliability System for RLHF Audit Trail.

Implements advanced error handling, circuit breakers, retry mechanisms,
and self-healing capabilities with comprehensive monitoring.
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
from contextlib import asynccontextmanager
from functools import wraps
import traceback

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class FailureType(Enum):
    """Types of system failures."""
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    VALIDATION = "validation"
    RESOURCE = "resource"
    PRIVACY = "privacy"
    COMPLIANCE = "compliance"
    UNKNOWN = "unknown"


@dataclass
class FailureRecord:
    """Records a system failure."""
    failure_id: str
    failure_type: FailureType
    component: str
    message: str
    timestamp: float
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0
    recovery_timeout: float = 30.0
    max_failures_per_minute: int = 10


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        """Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.failures_in_window = []
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def __call__(self, func):
        """Decorator to wrap functions with circuit breaker."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func, *args, **kwargs):
        """Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        current_time = time.time()
        
        # Clean old failures
        self._clean_failure_window(current_time)
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if current_time - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            # Call the function
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            else:
                result = func(*args, **kwargs)
            
            # Handle success
            await self._handle_success()
            return result
            
        except Exception as e:
            # Handle failure
            await self._handle_failure(e, current_time)
            raise
    
    async def _handle_success(self):
        """Handle successful function call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info(f"Circuit breaker {self.name} recovered to CLOSED")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
    
    async def _handle_failure(self, exception: Exception, current_time: float):
        """Handle function failure."""
        self.failure_count += 1
        self.last_failure_time = current_time
        self.failures_in_window.append(current_time)
        
        self.logger.warning(f"Circuit breaker {self.name} recorded failure #{self.failure_count}: {exception}")
        
        # Check if we should open the circuit
        if (self.failure_count >= self.config.failure_threshold or 
            len(self.failures_in_window) >= self.config.max_failures_per_minute):
            
            self.state = CircuitState.OPEN
            self.success_count = 0
            self.logger.error(f"Circuit breaker {self.name} opened due to failures")
    
    def _clean_failure_window(self, current_time: float):
        """Clean failures outside the time window."""
        window_start = current_time - 60.0  # 1 minute window
        self.failures_in_window = [t for t in self.failures_in_window if t > window_start]
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'failures_in_window': len(self.failures_in_window)
        }


class RetryStrategy:
    """Advanced retry strategy with exponential backoff."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """Initialize retry strategy.
        
        Args:
            max_attempts: Maximum retry attempts
            initial_delay: Initial delay between retries
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func):
        """Decorator to wrap functions with retry logic."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.retry(func, *args, **kwargs)
        return wrapper
    
    async def retry(self, func, *args, **kwargs):
        """Retry function with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.max_attempts} attempts failed")
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter and NUMPY_AVAILABLE:
            # Add jitter to prevent thundering herd
            jitter_factor = np.random.uniform(0.5, 1.5)
            delay *= jitter_factor
        
        return delay


class HealthMonitor:
    """Advanced health monitoring with predictive alerts."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.health_checks = {}
        self.failure_records = {}
        self.circuit_breakers = {}
        self.monitoring_active = False
        
        self.logger = logging.getLogger(__name__)
    
    def register_circuit_breaker(self, circuit_breaker: CircuitBreaker):
        """Register a circuit breaker for monitoring.
        
        Args:
            circuit_breaker: Circuit breaker to monitor
        """
        self.circuit_breakers[circuit_breaker.name] = circuit_breaker
        self.logger.info(f"Registered circuit breaker: {circuit_breaker.name}")
    
    def record_failure(
        self,
        component: str,
        failure_type: FailureType,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ) -> str:
        """Record a system failure.
        
        Args:
            component: Component that failed
            failure_type: Type of failure
            message: Failure message
            context: Additional context
            exception: Exception that caused the failure
            
        Returns:
            Failure record ID
        """
        failure_id = str(uuid.uuid4())
        
        record = FailureRecord(
            failure_id=failure_id,
            failure_type=failure_type,
            component=component,
            message=message,
            timestamp=time.time(),
            context=context or {},
            stack_trace=traceback.format_exc() if exception else None
        )
        
        self.failure_records[failure_id] = record
        self.logger.error(f"Recorded failure {failure_id} in {component}: {message}")
        
        return failure_id
    
    def resolve_failure(self, failure_id: str) -> bool:
        """Mark a failure as resolved.
        
        Args:
            failure_id: ID of failure to resolve
            
        Returns:
            True if failure was found and resolved
        """
        if failure_id in self.failure_records:
            record = self.failure_records[failure_id]
            record.resolved = True
            record.resolution_time = time.time()
            
            self.logger.info(f"Resolved failure {failure_id}")
            return True
        
        return False
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_circuit_breakers())
        asyncio.create_task(self._monitor_system_health())
        
        self.logger.info("Started health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        self.logger.info("Stopped health monitoring")
    
    async def _monitor_circuit_breakers(self):
        """Monitor circuit breaker states."""
        while self.monitoring_active:
            try:
                open_circuits = [
                    name for name, cb in self.circuit_breakers.items()
                    if cb.state == CircuitState.OPEN
                ]
                
                if open_circuits:
                    self.logger.warning(f"Open circuit breakers: {open_circuits}")
                
            except Exception as e:
                self.logger.error(f"Error monitoring circuit breakers: {e}")
            
            await asyncio.sleep(30)
    
    async def _monitor_system_health(self):
        """Monitor overall system health."""
        while self.monitoring_active:
            try:
                # Check for failure patterns
                recent_failures = [
                    record for record in self.failure_records.values()
                    if not record.resolved and time.time() - record.timestamp < 300
                ]
                
                if len(recent_failures) > 5:
                    self.logger.warning(f"High failure rate: {len(recent_failures)} failures in 5 minutes")
                
                # Check for repeated failures
                failure_counts = {}
                for record in recent_failures:
                    key = (record.component, record.failure_type)
                    failure_counts[key] = failure_counts.get(key, 0) + 1
                
                for (component, failure_type), count in failure_counts.items():
                    if count >= 3:
                        self.logger.error(
                            f"Repeated failures in {component} ({failure_type.value}): {count} times"
                        )
                
            except Exception as e:
                self.logger.error(f"Error monitoring system health: {e}")
            
            await asyncio.sleep(60)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get system health summary.
        
        Returns:
            Dictionary with health information
        """
        current_time = time.time()
        
        # Recent failures (last hour)
        recent_failures = [
            record for record in self.failure_records.values()
            if current_time - record.timestamp < 3600
        ]
        
        # Unresolved failures
        unresolved_failures = [
            record for record in self.failure_records.values()
            if not record.resolved
        ]
        
        # Circuit breaker statuses
        circuit_statuses = {
            name: cb.get_status()
            for name, cb in self.circuit_breakers.items()
        }
        
        return {
            'monitoring_active': self.monitoring_active,
            'total_failures': len(self.failure_records),
            'recent_failures': len(recent_failures),
            'unresolved_failures': len(unresolved_failures),
            'circuit_breakers': circuit_statuses,
            'failure_types': {
                failure_type.value: len([
                    r for r in recent_failures
                    if r.failure_type == failure_type
                ])
                for failure_type in FailureType
            }
        }


class SelfHealingManager:
    """Self-healing system manager."""
    
    def __init__(self, health_monitor: HealthMonitor):
        """Initialize self-healing manager.
        
        Args:
            health_monitor: Health monitor instance
        """
        self.health_monitor = health_monitor
        self.healing_strategies = {}
        self.active_healings = {}
        
        self.logger = logging.getLogger(__name__)
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default healing strategies."""
        self.register_strategy(
            "circuit_breaker_reset",
            self._reset_circuit_breaker,
            ["circuit_breaker_open"]
        )
        
        self.register_strategy(
            "resource_cleanup",
            self._cleanup_resources,
            ["memory_high", "disk_full"]
        )
        
        self.register_strategy(
            "connection_refresh",
            self._refresh_connections,
            ["connection_timeout", "network_error"]
        )
    
    def register_strategy(
        self,
        name: str,
        handler: Callable,
        triggers: List[str]
    ):
        """Register a self-healing strategy.
        
        Args:
            name: Strategy name
            handler: Healing handler function
            triggers: List of trigger conditions
        """
        self.healing_strategies[name] = {
            'handler': handler,
            'triggers': triggers
        }
        
        self.logger.info(f"Registered healing strategy: {name}")
    
    async def attempt_healing(self, failure_record: FailureRecord) -> bool:
        """Attempt to heal a system failure.
        
        Args:
            failure_record: Failure to attempt healing
            
        Returns:
            True if healing was attempted
        """
        healing_id = str(uuid.uuid4())
        trigger = f"{failure_record.component}_{failure_record.failure_type.value}"
        
        # Find applicable strategies
        applicable_strategies = [
            name for name, strategy in self.healing_strategies.items()
            if any(t in trigger for t in strategy['triggers'])
        ]
        
        if not applicable_strategies:
            self.logger.info(f"No healing strategies for {trigger}")
            return False
        
        self.active_healings[healing_id] = {
            'failure_id': failure_record.failure_id,
            'strategies': applicable_strategies,
            'start_time': time.time(),
            'status': 'in_progress'
        }
        
        try:
            for strategy_name in applicable_strategies:
                strategy = self.healing_strategies[strategy_name]
                
                self.logger.info(f"Attempting healing with strategy: {strategy_name}")
                
                success = await strategy['handler'](failure_record)
                
                if success:
                    self.health_monitor.resolve_failure(failure_record.failure_id)
                    self.active_healings[healing_id]['status'] = 'success'
                    self.logger.info(f"Healing successful with strategy: {strategy_name}")
                    return True
            
            self.active_healings[healing_id]['status'] = 'failed'
            self.logger.warning(f"All healing strategies failed for {trigger}")
            return False
            
        except Exception as e:
            self.active_healings[healing_id]['status'] = 'error'
            self.logger.error(f"Error during healing: {e}")
            return False
    
    async def _reset_circuit_breaker(self, failure_record: FailureRecord) -> bool:
        """Reset circuit breaker healing strategy."""
        circuit_name = failure_record.context.get('circuit_name')
        
        if circuit_name and circuit_name in self.health_monitor.circuit_breakers:
            circuit = self.health_monitor.circuit_breakers[circuit_name]
            
            # Wait for circuit to naturally transition
            await asyncio.sleep(circuit.config.recovery_timeout)
            
            # Force reset if still open
            if circuit.state == CircuitState.OPEN:
                circuit.state = CircuitState.HALF_OPEN
                circuit.failure_count = 0
                self.logger.info(f"Force reset circuit breaker: {circuit_name}")
            
            return True
        
        return False
    
    async def _cleanup_resources(self, failure_record: FailureRecord) -> bool:
        """Resource cleanup healing strategy."""
        try:
            import gc
            
            # Force garbage collection
            collected = gc.collect()
            self.logger.info(f"Collected {collected} objects during cleanup")
            
            # Additional cleanup logic would go here
            return True
            
        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")
            return False
    
    async def _refresh_connections(self, failure_record: FailureRecord) -> bool:
        """Connection refresh healing strategy."""
        try:
            # Simulated connection refresh
            await asyncio.sleep(1.0)
            
            self.logger.info("Refreshed connections")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection refresh failed: {e}")
            return False


class EnhancedReliabilitySystem:
    """Comprehensive reliability system with advanced fault tolerance."""
    
    def __init__(self):
        """Initialize enhanced reliability system."""
        self.health_monitor = HealthMonitor()
        self.self_healing = SelfHealingManager(self.health_monitor)
        self.circuit_breakers = {}
        self.retry_strategies = {}
        
        self.logger = logging.getLogger(__name__)
        self._setup_default_configurations()
    
    def _setup_default_configurations(self):
        """Setup default reliability configurations."""
        # Default circuit breakers
        self.create_circuit_breaker(
            "storage_operations",
            CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout=30.0,
                recovery_timeout=60.0
            )
        )
        
        self.create_circuit_breaker(
            "privacy_engine",
            CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=3,
                timeout=10.0,
                recovery_timeout=30.0
            )
        )
        
        self.create_circuit_breaker(
            "compliance_validation",
            CircuitBreakerConfig(
                failure_threshold=2,
                success_threshold=1,
                timeout=60.0,
                recovery_timeout=120.0
            )
        )
        
        # Default retry strategies
        self.retry_strategies['default'] = RetryStrategy()
        self.retry_strategies['critical'] = RetryStrategy(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=30.0
        )
        self.retry_strategies['background'] = RetryStrategy(
            max_attempts=10,
            initial_delay=5.0,
            max_delay=300.0
        )
    
    def create_circuit_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig
    ) -> CircuitBreaker:
        """Create and register a circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration
            
        Returns:
            Created circuit breaker
        """
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        self.health_monitor.register_circuit_breaker(circuit_breaker)
        
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name.
        
        Args:
            name: Circuit breaker name
            
        Returns:
            Circuit breaker or None if not found
        """
        return self.circuit_breakers.get(name)
    
    def get_retry_strategy(self, name: str = 'default') -> RetryStrategy:
        """Get retry strategy by name.
        
        Args:
            name: Retry strategy name
            
        Returns:
            Retry strategy
        """
        return self.retry_strategies.get(name, self.retry_strategies['default'])
    
    @asynccontextmanager
    async def reliable_operation(
        self,
        operation_name: str,
        circuit_breaker_name: Optional[str] = None,
        retry_strategy_name: str = 'default',
        timeout: Optional[float] = None
    ):
        """Context manager for reliable operations.
        
        Args:
            operation_name: Name of operation
            circuit_breaker_name: Circuit breaker to use
            retry_strategy_name: Retry strategy to use
            timeout: Operation timeout
        """
        start_time = time.time()
        operation_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Starting reliable operation: {operation_name} ({operation_id})")
            
            # Setup circuit breaker if specified
            circuit_breaker = None
            if circuit_breaker_name:
                circuit_breaker = self.get_circuit_breaker(circuit_breaker_name)
                if not circuit_breaker:
                    self.logger.warning(f"Circuit breaker not found: {circuit_breaker_name}")
            
            # Setup retry strategy
            retry_strategy = self.get_retry_strategy(retry_strategy_name)
            
            yield {
                'operation_id': operation_id,
                'circuit_breaker': circuit_breaker,
                'retry_strategy': retry_strategy,
                'health_monitor': self.health_monitor
            }
            
        except Exception as e:
            # Record failure
            failure_id = self.health_monitor.record_failure(
                component=operation_name,
                failure_type=FailureType.UNKNOWN,
                message=str(e),
                context={
                    'operation_id': operation_id,
                    'circuit_breaker': circuit_breaker_name,
                    'retry_strategy': retry_strategy_name
                },
                exception=e
            )
            
            # Attempt self-healing
            if failure_id in self.health_monitor.failure_records:
                failure_record = self.health_monitor.failure_records[failure_id]
                await self.self_healing.attempt_healing(failure_record)
            
            raise
            
        finally:
            duration = time.time() - start_time
            self.logger.info(f"Completed reliable operation: {operation_name} in {duration:.2f}s")
    
    async def start_monitoring(self):
        """Start reliability monitoring."""
        await self.health_monitor.start_monitoring()
        self.logger.info("Started enhanced reliability monitoring")
    
    async def stop_monitoring(self):
        """Stop reliability monitoring."""
        await self.health_monitor.stop_monitoring()
        self.logger.info("Stopped enhanced reliability monitoring")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status.
        
        Returns:
            Dictionary with system status
        """
        return {
            'health_summary': self.health_monitor.get_health_summary(),
            'circuit_breakers': {
                name: cb.get_status()
                for name, cb in self.circuit_breakers.items()
            },
            'active_healings': len(self.self_healing.active_healings),
            'retry_strategies': list(self.retry_strategies.keys())
        }
    
    def export_reliability_metrics(self, output_path: Path) -> None:
        """Export reliability metrics to file.
        
        Args:
            output_path: Path to save metrics
        """
        metrics = {
            'system_status': self.get_system_status(),
            'failure_records': {
                k: asdict(v) for k, v in self.health_monitor.failure_records.items()
            },
            'healing_history': self.self_healing.active_healings,
            'exported_at': time.time()
        }
        
        output_path.write_text(json.dumps(metrics, indent=2, default=str))
        self.logger.info(f"Reliability metrics exported to {output_path}")