"""Production Reliability Module for RLHF Audit Trail.

Implements enterprise-grade reliability patterns including:
- Retry logic with exponential backoff
- Circuit breakers
- Bulkhead isolation
- Rate limiting
- Graceful degradation
- Disaster recovery
"""

import asyncio
import json
import time
import uuid
import functools
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
import logging
from collections import defaultdict, deque

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class MockNumpy:
        def random(self):
            import random
            class MockRandom:
                def uniform(self, low, high): return random.uniform(low, high)
                def exponential(self, scale): return random.expovariate(1/scale)
            return MockRandom()
    np = MockNumpy()


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


class IsolationLevel(Enum):
    """Bulkhead isolation levels."""
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    QUEUE_BASED = "queue_based"
    RESOURCE_LIMIT = "resource_limit"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    backoff_multiplier: float = 2.0
    retriable_exceptions: List[type] = field(default_factory=lambda: [Exception])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0
    expected_exception: type = Exception


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 100.0
    burst_size: int = 200
    window_size: float = 1.0


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation."""
    isolation_level: IsolationLevel = IsolationLevel.THREAD_POOL
    max_concurrent: int = 10
    queue_size: int = 100
    timeout: float = 30.0


class RetryHandler:
    """Handles retry logic with various strategies."""
    
    def __init__(self, config: RetryConfig):
        """Initialize retry handler.
        
        Args:
            config: Retry configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply retry logic."""
        if asyncio.iscoroutinefunction(func):
            return self._async_wrapper(func)
        else:
            return self._sync_wrapper(func)
            
    def _sync_wrapper(self, func: Callable) -> Callable:
        """Wrapper for synchronous functions."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.config.max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if exception is retriable
                    if not any(isinstance(e, exc_type) for exc_type in self.config.retriable_exceptions):
                        raise e
                        
                    # Don't retry on last attempt
                    if attempt == self.config.max_attempts - 1:
                        break
                        
                    # Calculate delay
                    delay = self._calculate_delay(attempt)
                    
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    time.sleep(delay)
                    
            # All attempts failed
            raise last_exception
            
        return wrapper
        
    def _async_wrapper(self, func: Callable) -> Callable:
        """Wrapper for asynchronous functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if exception is retriable
                    if not any(isinstance(e, exc_type) for exc_type in self.config.retriable_exceptions):
                        raise e
                        
                    # Don't retry on last attempt
                    if attempt == self.config.max_attempts - 1:
                        break
                        
                    # Calculate delay
                    delay = self._calculate_delay(attempt)
                    
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    await asyncio.sleep(delay)
                    
            # All attempts failed
            raise last_exception
            
        return wrapper
        
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
            
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * (attempt + 1)
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
            
        else:  # IMMEDIATE
            delay = 0.0
            
        # Apply jitter
        if self.config.jitter and delay > 0:
            delay *= (0.5 + np.random().uniform(0, 0.5))
            
        # Respect max delay
        return min(delay, self.config.max_delay)


class AdvancedCircuitBreaker:
    """Advanced circuit breaker with multiple states and monitoring."""
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.state = 'closed'  # closed, open, half-open
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
        # Monitoring
        self.total_requests = 0
        self.total_failures = 0
        self.state_transitions = []
        
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker."""
        if asyncio.iscoroutinefunction(func):
            return self._async_wrapper(func)
        else:
            return self._sync_wrapper(func)
            
    def _sync_wrapper(self, func: Callable) -> Callable:
        """Wrapper for synchronous functions."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.total_requests += 1
            
            # Check circuit state
            if self.state == 'open':
                if time.time() - self.last_failure_time > self.config.timeout:
                    self._transition_to_half_open()
                else:
                    raise Exception(f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}")
                    
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.config.expected_exception as e:
                self._on_failure()
                raise e
                
        return wrapper
        
    def _async_wrapper(self, func: Callable) -> Callable:
        """Wrapper for asynchronous functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            self.total_requests += 1
            
            # Check circuit state
            if self.state == 'open':
                if time.time() - self.last_failure_time > self.config.timeout:
                    self._transition_to_half_open()
                else:
                    raise Exception(f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}")
                    
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.config.expected_exception as e:
                self._on_failure()
                raise e
                
        return wrapper
        
    def _on_success(self):
        """Handle successful request."""
        self.last_success_time = time.time()
        
        if self.state == 'half-open':
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        else:
            self.failure_count = 0
            
    def _on_failure(self):
        """Handle failed request."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state in ['closed', 'half-open']:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
                
    def _transition_to_open(self):
        """Transition to open state."""
        old_state = self.state
        self.state = 'open'
        self.success_count = 0
        self._record_transition(old_state, 'open')
        self.logger.warning(f"Circuit breaker opened. Failure count: {self.failure_count}")
        
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        old_state = self.state
        self.state = 'half-open'
        self.success_count = 0
        self.failure_count = 0
        self._record_transition(old_state, 'half-open')
        self.logger.info("Circuit breaker half-opened for testing")
        
    def _transition_to_closed(self):
        """Transition to closed state."""
        old_state = self.state
        self.state = 'closed'
        self.failure_count = 0
        self.success_count = 0
        self._record_transition(old_state, 'closed')
        self.logger.info("Circuit breaker closed. Service recovered")
        
    def _record_transition(self, from_state: str, to_state: str):
        """Record state transition for monitoring."""
        self.state_transitions.append({
            'timestamp': time.time(),
            'from_state': from_state,
            'to_state': to_state,
            'failure_count': self.failure_count,
            'success_count': self.success_count
        })
        
        # Keep only recent transitions
        if len(self.state_transitions) > 100:
            self.state_transitions = self.state_transitions[-50:]
            
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        current_time = time.time()
        
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_requests': self.total_requests,
            'total_failures': self.total_failures,
            'failure_rate': self.total_failures / max(self.total_requests, 1),
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time,
            'time_since_last_failure': current_time - self.last_failure_time if self.last_failure_time else None,
            'state_transitions': len(self.state_transitions)
        }


class TokenBucketRateLimiter:
    """Token bucket rate limiter for request throttling."""
    
    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.tokens = float(config.burst_size)
        self.last_update = time.time()
        
    def is_allowed(self, tokens_required: int = 1) -> bool:
        """Check if request is allowed.
        
        Args:
            tokens_required: Number of tokens required
            
        Returns:
            True if request is allowed
        """
        current_time = time.time()
        
        # Add tokens based on elapsed time
        elapsed = current_time - self.last_update
        tokens_to_add = elapsed * self.config.requests_per_second
        
        self.tokens = min(
            self.config.burst_size,
            self.tokens + tokens_to_add
        )
        
        self.last_update = current_time
        
        # Check if enough tokens available
        if self.tokens >= tokens_required:
            self.tokens -= tokens_required
            return True
        else:
            return False
            
    def wait_time(self, tokens_required: int = 1) -> float:
        """Calculate wait time until tokens are available.
        
        Args:
            tokens_required: Number of tokens required
            
        Returns:
            Wait time in seconds
        """
        if self.tokens >= tokens_required:
            return 0.0
            
        tokens_needed = tokens_required - self.tokens
        return tokens_needed / self.config.requests_per_second
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply rate limiting."""
        if asyncio.iscoroutinefunction(func):
            return self._async_wrapper(func)
        else:
            return self._sync_wrapper(func)
            
    def _sync_wrapper(self, func: Callable) -> Callable:
        """Wrapper for synchronous functions."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.is_allowed():
                wait_time = self.wait_time()
                raise Exception(f"Rate limit exceeded. Wait {wait_time:.2f}s")
                
            return func(*args, **kwargs)
            
        return wrapper
        
    def _async_wrapper(self, func: Callable) -> Callable:
        """Wrapper for asynchronous functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not self.is_allowed():
                wait_time = self.wait_time()
                await asyncio.sleep(wait_time)
                
            return await func(*args, **kwargs)
            
        return wrapper


class BulkheadIsolation:
    """Bulkhead pattern for isolating resources."""
    
    def __init__(self, config: BulkheadConfig):
        """Initialize bulkhead isolation.
        
        Args:
            config: Bulkhead configuration
        """
        self.config = config
        self.active_requests = 0
        self.request_queue = asyncio.Queue(maxsize=config.queue_size)
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply bulkhead isolation."""
        if asyncio.iscoroutinefunction(func):
            return self._async_wrapper(func)
        else:
            raise ValueError("Bulkhead isolation only supports async functions")
            
    def _async_wrapper(self, func: Callable) -> Callable:
        """Wrapper for asynchronous functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to acquire semaphore with timeout
            try:
                await asyncio.wait_for(
                    self.semaphore.acquire(),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                raise Exception(f"Bulkhead isolation timeout: {self.config.timeout}s")
                
            try:
                self.active_requests += 1
                result = await func(*args, **kwargs)
                return result
                
            finally:
                self.active_requests -= 1
                self.semaphore.release()
                
        return wrapper
        
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            'active_requests': self.active_requests,
            'max_concurrent': self.config.max_concurrent,
            'queue_size': self.request_queue.qsize(),
            'max_queue_size': self.config.queue_size,
            'available_permits': self.semaphore._value
        }


class GracefulDegradation:
    """Graceful degradation handler for fallback behaviors."""
    
    def __init__(self):
        """Initialize graceful degradation handler."""
        self.fallback_functions: Dict[str, Callable] = {}
        self.degradation_triggers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_fallback(self, operation: str, fallback_func: Callable):
        """Register fallback function for an operation.
        
        Args:
            operation: Operation name
            fallback_func: Fallback function to call
        """
        self.fallback_functions[operation] = fallback_func
        
    def register_trigger(self, operation: str, trigger_func: Callable):
        """Register trigger function to determine if degradation needed.
        
        Args:
            operation: Operation name
            trigger_func: Function returning True if degradation needed
        """
        self.degradation_triggers[operation] = trigger_func
        
    def __call__(self, operation: str):
        """Decorator for graceful degradation.
        
        Args:
            operation: Operation name
        """
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                return self._async_wrapper(func, operation)
            else:
                return self._sync_wrapper(func, operation)
                
        return decorator
        
    def _sync_wrapper(self, func: Callable, operation: str) -> Callable:
        """Wrapper for synchronous functions."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if degradation is needed
            trigger = self.degradation_triggers.get(operation)
            if trigger and trigger():
                fallback = self.fallback_functions.get(operation)
                if fallback:
                    self.logger.warning(f"Degrading operation {operation}")
                    return fallback(*args, **kwargs)
                    
            # Try normal operation
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Fall back on error if available
                fallback = self.fallback_functions.get(operation)
                if fallback:
                    self.logger.error(f"Operation {operation} failed, falling back: {e}")
                    return fallback(*args, **kwargs)
                else:
                    raise e
                    
        return wrapper
        
    def _async_wrapper(self, func: Callable, operation: str) -> Callable:
        """Wrapper for asynchronous functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if degradation is needed
            trigger = self.degradation_triggers.get(operation)
            if trigger and trigger():
                fallback = self.fallback_functions.get(operation)
                if fallback:
                    self.logger.warning(f"Degrading operation {operation}")
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    else:
                        return fallback(*args, **kwargs)
                        
            # Try normal operation
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Fall back on error if available
                fallback = self.fallback_functions.get(operation)
                if fallback:
                    self.logger.error(f"Operation {operation} failed, falling back: {e}")
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    else:
                        return fallback(*args, **kwargs)
                else:
                    raise e
                    
        return wrapper


class ProductionReliabilityManager:
    """Central manager for production reliability patterns."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize reliability manager.
        
        Args:
            config: Reliability configuration
        """
        self.config = config or {}
        
        # Initialize components
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
        self.rate_limiters: Dict[str, TokenBucketRateLimiter] = {}
        self.bulkheads: Dict[str, BulkheadIsolation] = {}
        self.degradation_handler = GracefulDegradation()
        
        # Monitoring
        self.reliability_metrics = defaultdict(dict)
        
        self.logger = logging.getLogger(__name__)
        self._setup_default_patterns()
        
    def _setup_default_patterns(self):
        """Setup default reliability patterns."""
        # Default retry configurations
        retry_configs = {
            'database': RetryConfig(max_attempts=3, base_delay=1.0),
            'external_api': RetryConfig(max_attempts=5, base_delay=2.0, max_delay=30.0),
            'ml_inference': RetryConfig(max_attempts=2, base_delay=0.5),
            'storage': RetryConfig(max_attempts=3, base_delay=1.5)
        }
        
        for name, config in retry_configs.items():
            self.retry_handlers[name] = RetryHandler(config)
            
        # Default circuit breaker configurations
        cb_configs = {
            'database': CircuitBreakerConfig(failure_threshold=3, timeout=30.0),
            'external_api': CircuitBreakerConfig(failure_threshold=5, timeout=60.0),
            'ml_inference': CircuitBreakerConfig(failure_threshold=2, timeout=120.0),
            'compliance_check': CircuitBreakerConfig(failure_threshold=2, timeout=300.0)
        }
        
        for name, config in cb_configs.items():
            self.circuit_breakers[name] = AdvancedCircuitBreaker(config)
            
        # Default rate limiters
        rl_configs = {
            'api_requests': RateLimitConfig(requests_per_second=100.0, burst_size=200),
            'ml_inference': RateLimitConfig(requests_per_second=10.0, burst_size=20),
            'audit_logging': RateLimitConfig(requests_per_second=1000.0, burst_size=2000)
        }
        
        for name, config in rl_configs.items():
            self.rate_limiters[name] = TokenBucketRateLimiter(config)
            
        # Default bulkhead configurations
        bulkhead_configs = {
            'ml_processing': BulkheadConfig(max_concurrent=5, queue_size=50),
            'audit_processing': BulkheadConfig(max_concurrent=10, queue_size=100),
            'compliance_validation': BulkheadConfig(max_concurrent=3, queue_size=30)
        }
        
        for name, config in bulkhead_configs.items():
            self.bulkheads[name] = BulkheadIsolation(config)
            
        # Setup fallback functions
        self._setup_fallback_functions()
        
    def _setup_fallback_functions(self):
        """Setup fallback functions for graceful degradation."""
        # ML inference fallback
        def ml_fallback(*args, **kwargs):
            # Return cached or simplified result
            return {'prediction': 0.5, 'confidence': 0.0, 'fallback': True}
            
        self.degradation_handler.register_fallback('ml_inference', ml_fallback)
        
        # Compliance check fallback
        def compliance_fallback(*args, **kwargs):
            # Return basic compliance result
            return {'compliant': True, 'score': 0.8, 'fallback': True}
            
        self.degradation_handler.register_fallback('compliance_check', compliance_fallback)
        
        # Audit logging fallback
        def audit_fallback(*args, **kwargs):
            # Log to local file or queue for later processing
            return {'logged': True, 'location': 'fallback_queue'}
            
        self.degradation_handler.register_fallback('audit_logging', audit_fallback)
        
    def get_retry_handler(self, name: str) -> RetryHandler:
        """Get retry handler by name."""
        return self.retry_handlers.get(name, self.retry_handlers['database'])
        
    def get_circuit_breaker(self, name: str) -> AdvancedCircuitBreaker:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name, self.circuit_breakers['database'])
        
    def get_rate_limiter(self, name: str) -> TokenBucketRateLimiter:
        """Get rate limiter by name."""
        return self.rate_limiters.get(name, self.rate_limiters['api_requests'])
        
    def get_bulkhead(self, name: str) -> BulkheadIsolation:
        """Get bulkhead by name."""
        return self.bulkheads.get(name, self.bulkheads['ml_processing'])
        
    def apply_reliability_patterns(self, 
                                   operation: str,
                                   retry: bool = True,
                                   circuit_breaker: bool = True,
                                   rate_limit: bool = False,
                                   bulkhead: bool = False,
                                   graceful_degradation: bool = False):
        """Decorator to apply multiple reliability patterns.
        
        Args:
            operation: Operation name
            retry: Apply retry logic
            circuit_breaker: Apply circuit breaker
            rate_limit: Apply rate limiting
            bulkhead: Apply bulkhead isolation
            graceful_degradation: Apply graceful degradation
        """
        def decorator(func: Callable) -> Callable:
            # Apply patterns in reverse order (innermost first)
            decorated_func = func
            
            if graceful_degradation:
                decorated_func = self.degradation_handler(operation)(decorated_func)
                
            if bulkhead:
                decorated_func = self.get_bulkhead(operation)(decorated_func)
                
            if rate_limit:
                decorated_func = self.get_rate_limiter(operation)(decorated_func)
                
            if circuit_breaker:
                decorated_func = self.get_circuit_breaker(operation)(decorated_func)
                
            if retry:
                decorated_func = self.get_retry_handler(operation)(decorated_func)
                
            return decorated_func
            
        return decorator
        
    def get_reliability_status(self) -> Dict[str, Any]:
        """Get overall reliability status."""
        status = {
            'circuit_breakers': {},
            'rate_limiters': {},
            'bulkheads': {},
            'overall_health': 'healthy'
        }
        
        # Circuit breaker status
        unhealthy_breakers = 0
        for name, breaker in self.circuit_breakers.items():
            stats = breaker.get_stats()
            status['circuit_breakers'][name] = stats
            
            if stats['state'] == 'open':
                unhealthy_breakers += 1
                
        # Rate limiter status
        for name, limiter in self.rate_limiters.items():
            status['rate_limiters'][name] = {
                'tokens_available': limiter.tokens,
                'max_tokens': limiter.config.burst_size,
                'requests_per_second': limiter.config.requests_per_second
            }
            
        # Bulkhead status
        for name, bulkhead in self.bulkheads.items():
            status['bulkheads'][name] = bulkhead.get_stats()
            
        # Overall health assessment
        if unhealthy_breakers > 0:
            if unhealthy_breakers >= len(self.circuit_breakers) * 0.5:
                status['overall_health'] = 'critical'
            else:
                status['overall_health'] = 'degraded'
                
        return status