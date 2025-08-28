"""Enhanced reliability features for robust operation.

This module provides circuit breakers, retry logic, health monitoring,
and auto-recovery mechanisms to ensure system reliability.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps
import random

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failure state, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5      # Failures before opening
    recovery_timeout: float = 60.0  # Seconds before attempting recovery
    success_threshold: int = 3      # Successes to close circuit
    timeout: float = 30.0           # Request timeout
    
    
@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opened_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            config: Configuration settings
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_failure_time = 0.0
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        self.stats.total_requests += 1
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time < self.config.recovery_timeout:
                raise CircuitBreakerOpenError(f"Circuit {self.name} is open")
            else:
                # Transition to half-open for testing
                self.state = CircuitState.HALF_OPEN
                self.consecutive_successes = 0
                logger.info(f"Circuit {self.name} transitioning to half-open")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_function(func, *args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            self._record_success()
            return result
            
        except asyncio.TimeoutError:
            self.stats.timeouts += 1
            self._record_failure()
            raise CircuitBreakerTimeoutError(f"Circuit {self.name} timeout")
            
        except Exception as e:
            self._record_failure()
            raise
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function, handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _record_success(self) -> None:
        """Record successful execution."""
        self.stats.successful_requests += 1
        self.stats.last_success_time = time.time()
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        
        # Close circuit if enough successes in half-open state
        if (self.state == CircuitState.HALF_OPEN and 
            self.consecutive_successes >= self.config.success_threshold):
            self.state = CircuitState.CLOSED
            logger.info(f"Circuit {self.name} closed after recovery")
    
    def _record_failure(self) -> None:
        """Record failed execution."""
        self.stats.failed_requests += 1
        self.stats.last_failure_time = time.time()
        self.last_failure_time = time.time()
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        
        # Open circuit if threshold exceeded
        if (self.state == CircuitState.CLOSED and 
            self.consecutive_failures >= self.config.failure_threshold):
            self.state = CircuitState.OPEN
            self.stats.circuit_opened_count += 1
            logger.error(f"Circuit {self.name} opened due to failures")
        
        # Return to open if failure in half-open state
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name} returned to open state")


@dataclass 
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[type] = field(default_factory=lambda: [Exception])


class RetryHandler:
    """Retry logic with exponential backoff and jitter."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize retry handler.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        self.retry_stats = {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_permanently': 0,
            'average_attempts': 0.0
        }
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to function."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        self.retry_stats['total_attempts'] += 1
        
        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Update success stats
                if attempt > 0:
                    self.retry_stats['successful_retries'] += 1
                self._update_average_attempts(attempt + 1)
                
                return result
                    
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not any(isinstance(e, exc_type) for exc_type in self.config.retryable_exceptions):
                    logger.warning(f"Non-retryable exception: {e}")
                    self.retry_stats['failed_permanently'] += 1
                    raise
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    break
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        self.retry_stats['failed_permanently'] += 1
        self._update_average_attempts(self.config.max_attempts)
        logger.error(f"All {self.config.max_attempts} attempts failed. Last error: {last_exception}")
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry attempt."""
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter = random.uniform(0.1, 0.9)
            delay = delay * jitter
            
        return delay
    
    def _update_average_attempts(self, attempts: int):
        """Update average attempts statistic."""
        total = self.retry_stats['total_attempts']
        current_avg = self.retry_stats['average_attempts']
        self.retry_stats['average_attempts'] = ((current_avg * (total - 1)) + attempts) / total
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get retry handler statistics."""
        return dict(self.retry_stats)


class HealthMonitor:
    """Health monitoring system for components."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.components: Dict[str, Dict[str, Any]] = {}
        self.health_checks: Dict[str, Callable] = {}
        
    def register_component(self, name: str, health_check: Callable, 
                         check_interval: float = 60.0) -> None:
        """Register a component for health monitoring.
        
        Args:
            name: Component name
            health_check: Function that returns health status
            check_interval: How often to check health (seconds)
        """
        self.components[name] = {
            "status": "unknown",
            "last_check": 0.0,
            "check_interval": check_interval,
            "consecutive_failures": 0,
            "total_checks": 0,
            "failure_count": 0
        }
        self.health_checks[name] = health_check
        logger.info(f"Registered health monitoring for component: {name}")
    
    async def check_health(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """Check health of component(s).
        
        Args:
            component_name: Specific component to check, or None for all
            
        Returns:
            Health status dictionary
        """
        if component_name:
            components_to_check = [component_name]
        else:
            components_to_check = list(self.components.keys())
        
        results = {}
        
        for name in components_to_check:
            if name not in self.components:
                continue
                
            component = self.components[name]
            current_time = time.time()
            
            # Skip if checked recently
            if (current_time - component["last_check"]) < component["check_interval"]:
                results[name] = {
                    "status": component["status"],
                    "cached": True,
                    "last_check": component["last_check"]
                }
                continue
            
            # Perform health check
            try:
                health_check = self.health_checks[name]
                if asyncio.iscoroutinefunction(health_check):
                    status = await health_check()
                else:
                    status = health_check()
                
                component["status"] = "healthy" if status else "unhealthy"
                component["consecutive_failures"] = 0 if status else component["consecutive_failures"] + 1
                if not status:
                    component["failure_count"] += 1
                    
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                component["status"] = "error"
                component["consecutive_failures"] += 1
                component["failure_count"] += 1
            
            component["last_check"] = current_time
            component["total_checks"] += 1
            
            results[name] = {
                "status": component["status"],
                "consecutive_failures": component["consecutive_failures"],
                "failure_rate": component["failure_count"] / component["total_checks"],
                "last_check": component["last_check"],
                "cached": False
            }
        
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        healthy_count = 0
        total_count = len(self.components)
        
        for component in self.components.values():
            if component["status"] == "healthy":
                healthy_count += 1
        
        health_percentage = (healthy_count / total_count * 100) if total_count > 0 else 0
        
        return {
            "overall_status": "healthy" if health_percentage >= 80 else "degraded" if health_percentage >= 50 else "unhealthy",
            "health_percentage": health_percentage,
            "healthy_components": healthy_count,
            "total_components": total_count,
            "component_details": dict(self.components)
        }


# Custom exceptions for reliability features
class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""
    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreakerTimeoutError(CircuitBreakerError):
    """Raised when circuit breaker times out."""
    pass


# Global instances for system-wide use
system_health_monitor = HealthMonitor()


# Convenience decorators
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Convenience decorator for circuit breaker."""
    breaker = CircuitBreaker(name, config)
    return breaker


def retry(config: Optional[RetryConfig] = None):
    """Convenience decorator for retry logic."""
    retry_handler = RetryHandler(config)
    return retry_handler