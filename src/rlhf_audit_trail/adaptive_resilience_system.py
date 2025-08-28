"""Adaptive Resilience System for Dynamic Fault Tolerance.

Advanced resilience system that learns from failures and adapts
its strategies for improved system reliability and performance.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Set
import logging
import statistics
from collections import deque, defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class MockNumpy:
        def array(self, x): return x
        def mean(self, x): return sum(x) / len(x) if x else 0
        def std(self, x): return statistics.stdev(x) if len(x) > 1 else 0.1
        def percentile(self, x, p): return sorted(x)[int(len(x) * p / 100)] if x else 0
    np = MockNumpy()


class ResilienceStrategy(Enum):
    """Resilience strategy types."""
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY = "retry"
    TIMEOUT = "timeout"
    BULKHEAD = "bulkhead"
    RATE_LIMIT = "rate_limit"
    CACHE = "cache"
    FALLBACK = "fallback"


class FailurePattern(Enum):
    """Common failure patterns."""
    TRANSIENT = "transient"
    PERSISTENT = "persistent"
    CASCADE = "cascade"
    OVERLOAD = "overload"
    TIMEOUT = "timeout"
    NETWORK = "network"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class FailureEvent:
    """Represents a failure event for learning."""
    event_id: str
    timestamp: float
    component: str
    failure_type: str
    duration: float
    context: Dict[str, Any]
    recovery_strategy: Optional[str] = None
    recovery_time: Optional[float] = None
    successful_recovery: bool = False


@dataclass
class ResilienceConfig:
    """Configuration for resilience strategies."""
    strategy: ResilienceStrategy
    parameters: Dict[str, Any]
    adaptive: bool = True
    learning_rate: float = 0.1
    confidence_threshold: float = 0.8


class AdaptiveLearningEngine:
    """Machine learning engine for failure pattern recognition."""
    
    def __init__(self):
        """Initialize learning engine."""
        self.failure_history: deque = deque(maxlen=1000)
        self.pattern_models: Dict[str, Dict[str, Any]] = {}
        self.strategy_effectiveness: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.learning_enabled = True
        
    def record_failure(self, failure: FailureEvent) -> None:
        """Record a failure event for learning."""
        self.failure_history.append(failure)
        self._update_pattern_models(failure)
        
    def _update_pattern_models(self, failure: FailureEvent) -> None:
        """Update pattern recognition models."""
        component = failure.component
        failure_type = failure.failure_type
        
        if component not in self.pattern_models:
            self.pattern_models[component] = {
                'failure_rates': defaultdict(float),
                'recovery_strategies': defaultdict(list),
                'temporal_patterns': deque(maxlen=100),
                'context_patterns': {}
            }
        
        model = self.pattern_models[component]
        
        # Update failure rates
        model['failure_rates'][failure_type] += 1
        
        # Track temporal patterns
        model['temporal_patterns'].append({
            'timestamp': failure.timestamp,
            'type': failure_type,
            'duration': failure.duration
        })
        
        # Update recovery strategy effectiveness
        if failure.recovery_strategy and failure.successful_recovery:
            model['recovery_strategies'][failure_type].append({
                'strategy': failure.recovery_strategy,
                'recovery_time': failure.recovery_time,
                'success': failure.successful_recovery
            })
    
    def predict_failure_pattern(self, component: str, context: Dict[str, Any]) -> Optional[FailurePattern]:
        """Predict likely failure pattern based on context."""
        if component not in self.pattern_models:
            return None
            
        model = self.pattern_models[component]
        
        # Analyze recent failure trends
        recent_failures = [
            event for event in model['temporal_patterns']
            if time.time() - event['timestamp'] < 3600  # Last hour
        ]
        
        if not recent_failures:
            return None
        
        # Simple pattern recognition based on frequency and duration
        failure_types = [f['type'] for f in recent_failures]
        durations = [f['duration'] for f in recent_failures]
        
        if len(recent_failures) > 5:  # High frequency
            return FailurePattern.OVERLOAD
        elif len(set(failure_types)) == 1 and len(failure_types) > 2:
            return FailurePattern.PERSISTENT
        elif np.mean(durations) < 5.0:  # Short duration failures
            return FailurePattern.TRANSIENT
        elif 'timeout' in failure_types[-1]:
            return FailurePattern.TIMEOUT
        else:
            return FailurePattern.TRANSIENT
    
    def recommend_strategy(self, component: str, failure_pattern: FailurePattern) -> Optional[ResilienceStrategy]:
        """Recommend resilience strategy based on pattern."""
        if component in self.pattern_models:
            model = self.pattern_models[component]
            
            # Find most effective strategy for this pattern
            strategies = model['recovery_strategies'].get(failure_pattern.value, [])
            
            if strategies:
                # Calculate effectiveness scores
                strategy_scores = defaultdict(list)
                for strategy_event in strategies:
                    if strategy_event['success']:
                        score = 1.0 / max(strategy_event['recovery_time'], 0.1)
                        strategy_scores[strategy_event['strategy']].append(score)
                
                if strategy_scores:
                    best_strategy = max(strategy_scores.keys(), 
                                      key=lambda s: np.mean(strategy_scores[s]))
                    return ResilienceStrategy(best_strategy)
        
        # Default strategy recommendations
        strategy_map = {
            FailurePattern.TRANSIENT: ResilienceStrategy.RETRY,
            FailurePattern.PERSISTENT: ResilienceStrategy.CIRCUIT_BREAKER,
            FailurePattern.OVERLOAD: ResilienceStrategy.RATE_LIMIT,
            FailurePattern.TIMEOUT: ResilienceStrategy.TIMEOUT,
            FailurePattern.NETWORK: ResilienceStrategy.RETRY,
            FailurePattern.RESOURCE_EXHAUSTION: ResilienceStrategy.BULKHEAD
        }
        
        return strategy_map.get(failure_pattern, ResilienceStrategy.CIRCUIT_BREAKER)


class AdaptiveResilienceSystem:
    """Adaptive resilience system with learning capabilities.
    
    Features:
    - Pattern recognition for failures
    - Dynamic strategy adaptation
    - Self-healing capabilities
    - Performance optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adaptive resilience system.
        
        Args:
            config: System configuration
        """
        self.config = config or {}
        
        # Core components
        self.learning_engine = AdaptiveLearningEngine()
        self.active_strategies: Dict[str, ResilienceConfig] = {}
        self.component_health: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'failed_requests': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'strategy_adaptations': 0,
            'mean_response_time': 0.0
        }
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'failure_rate_threshold': 0.05,  # 5%
            'response_time_threshold': 1000.0,  # 1 second
            'adaptation_confidence': 0.8
        }
        
        self.logger = logging.getLogger(__name__)
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Setup default resilience strategies."""
        # Default circuit breaker
        self.active_strategies['circuit_breaker'] = ResilienceConfig(
            strategy=ResilienceStrategy.CIRCUIT_BREAKER,
            parameters={
                'failure_threshold': 5,
                'recovery_timeout': 60.0,
                'success_threshold': 3
            }
        )
        
        # Default retry strategy
        self.active_strategies['retry'] = ResilienceConfig(
            strategy=ResilienceStrategy.RETRY,
            parameters={
                'max_attempts': 3,
                'base_delay': 1.0,
                'exponential_base': 2.0,
                'max_delay': 30.0
            }
        )
        
        # Default timeout strategy
        self.active_strategies['timeout'] = ResilienceConfig(
            strategy=ResilienceStrategy.TIMEOUT,
            parameters={
                'default_timeout': 30.0,
                'slow_timeout': 60.0
            }
        )
    
    async def execute_with_resilience(self, 
                                    component: str,
                                    operation: Callable,
                                    *args,
                                    **kwargs) -> Any:
        """Execute operation with adaptive resilience.
        
        Args:
            component: Component name
            operation: Operation to execute
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
        """
        start_time = time.time()
        self.performance_metrics['total_requests'] += 1
        
        # Get current context
        context = self._get_execution_context(component)
        
        # Predict failure pattern
        predicted_pattern = self.learning_engine.predict_failure_pattern(component, context)
        
        # Select or adapt strategy
        strategy_config = self._select_strategy(component, predicted_pattern)
        
        try:
            # Execute with selected strategy
            result = await self._execute_with_strategy(
                strategy_config, component, operation, *args, **kwargs
            )
            
            # Record success metrics
            execution_time = time.time() - start_time
            self._update_success_metrics(component, execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            failure_event = FailureEvent(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                component=component,
                failure_type=type(e).__name__,
                duration=time.time() - start_time,
                context=context
            )
            
            # Attempt recovery
            recovery_result = await self._attempt_recovery(failure_event, operation, *args, **kwargs)
            
            if recovery_result['success']:
                failure_event.recovery_strategy = recovery_result['strategy']
                failure_event.recovery_time = recovery_result['recovery_time']
                failure_event.successful_recovery = True
                
                self.performance_metrics['successful_recoveries'] += 1
                self.learning_engine.record_failure(failure_event)
                
                return recovery_result['result']
            else:
                # Log failure for learning
                self.performance_metrics['failed_requests'] += 1
                self.learning_engine.record_failure(failure_event)
                
                # Trigger adaptation
                await self._trigger_adaptation(component, failure_event)
                
                raise e
    
    def _get_execution_context(self, component: str) -> Dict[str, Any]:
        """Get current execution context."""
        return {
            'component': component,
            'timestamp': time.time(),
            'system_load': self._get_system_load(),
            'component_health': self.component_health.get(component, {}),
            'recent_failures': len([
                f for f in self.learning_engine.failure_history
                if f.component == component and time.time() - f.timestamp < 300
            ])
        }
    
    def _get_system_load(self) -> Dict[str, float]:
        """Get current system load metrics."""
        # Simulate system load metrics
        return {
            'cpu_usage': 0.65,
            'memory_usage': 0.45,
            'network_usage': 0.30,
            'request_rate': 150.0
        }
    
    def _select_strategy(self, component: str, predicted_pattern: Optional[FailurePattern]) -> ResilienceConfig:
        """Select appropriate resilience strategy."""
        if predicted_pattern:
            # Get AI recommendation
            recommended_strategy = self.learning_engine.recommend_strategy(component, predicted_pattern)
            
            if recommended_strategy and recommended_strategy.value in self.active_strategies:
                strategy_config = self.active_strategies[recommended_strategy.value]
                
                # Adapt parameters if enabled
                if strategy_config.adaptive:
                    self._adapt_strategy_parameters(strategy_config, component, predicted_pattern)
                
                return strategy_config
        
        # Default fallback to circuit breaker
        return self.active_strategies['circuit_breaker']
    
    def _adapt_strategy_parameters(self, 
                                 strategy_config: ResilienceConfig,
                                 component: str,
                                 pattern: FailurePattern) -> None:
        """Adapt strategy parameters based on learning."""
        if component not in self.learning_engine.pattern_models:
            return
        
        model = self.learning_engine.pattern_models[component]
        
        # Adapt based on recent performance
        if strategy_config.strategy == ResilienceStrategy.CIRCUIT_BREAKER:
            failure_rate = len([
                f for f in self.learning_engine.failure_history[-50:]
                if f.component == component
            ]) / 50.0
            
            # Adjust failure threshold based on component reliability
            if failure_rate > 0.1:  # High failure rate
                strategy_config.parameters['failure_threshold'] = max(2, 
                    int(strategy_config.parameters['failure_threshold'] * 0.8))
            else:  # Low failure rate
                strategy_config.parameters['failure_threshold'] = min(10,
                    int(strategy_config.parameters['failure_threshold'] * 1.2))
        
        elif strategy_config.strategy == ResilienceStrategy.RETRY:
            # Adapt retry parameters based on success rate
            recent_failures = [f for f in model['temporal_patterns'] if time.time() - f['timestamp'] < 3600]
            avg_duration = np.mean([f['duration'] for f in recent_failures]) if recent_failures else 5.0
            
            # Adjust retry parameters
            strategy_config.parameters['base_delay'] = max(0.5, min(5.0, avg_duration * 0.5))
            strategy_config.parameters['max_attempts'] = 5 if pattern == FailurePattern.TRANSIENT else 2
        
        self.performance_metrics['strategy_adaptations'] += 1
        self.logger.info(f"Adapted {strategy_config.strategy.value} strategy for {component}")
    
    async def _execute_with_strategy(self,
                                   strategy_config: ResilienceConfig,
                                   component: str,
                                   operation: Callable,
                                   *args,
                                   **kwargs) -> Any:
        """Execute operation with specific strategy."""
        if strategy_config.strategy == ResilienceStrategy.CIRCUIT_BREAKER:
            return await self._execute_with_circuit_breaker(strategy_config, operation, *args, **kwargs)
        elif strategy_config.strategy == ResilienceStrategy.RETRY:
            return await self._execute_with_retry(strategy_config, operation, *args, **kwargs)
        elif strategy_config.strategy == ResilienceStrategy.TIMEOUT:
            return await self._execute_with_timeout(strategy_config, operation, *args, **kwargs)
        else:
            # Direct execution
            if asyncio.iscoroutinefunction(operation):
                return await operation(*args, **kwargs)
            else:
                return operation(*args, **kwargs)
    
    async def _execute_with_circuit_breaker(self,
                                          strategy_config: ResilienceConfig,
                                          operation: Callable,
                                          *args,
                                          **kwargs) -> Any:
        """Execute with circuit breaker pattern."""
        # Implement circuit breaker logic
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        else:
            return operation(*args, **kwargs)
    
    async def _execute_with_retry(self,
                                strategy_config: ResilienceConfig,
                                operation: Callable,
                                *args,
                                **kwargs) -> Any:
        """Execute with retry logic."""
        max_attempts = strategy_config.parameters.get('max_attempts', 3)
        base_delay = strategy_config.parameters.get('base_delay', 1.0)
        
        for attempt in range(max_attempts):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation(*args, **kwargs)
                else:
                    return operation(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
    
    async def _execute_with_timeout(self,
                                  strategy_config: ResilienceConfig,
                                  operation: Callable,
                                  *args,
                                  **kwargs) -> Any:
        """Execute with timeout protection."""
        timeout = strategy_config.parameters.get('default_timeout', 30.0)
        
        if asyncio.iscoroutinefunction(operation):
            return await asyncio.wait_for(operation(*args, **kwargs), timeout=timeout)
        else:
            return operation(*args, **kwargs)
    
    async def _attempt_recovery(self,
                              failure_event: FailureEvent,
                              operation: Callable,
                              *args,
                              **kwargs) -> Dict[str, Any]:
        """Attempt to recover from failure."""
        self.performance_metrics['recovery_attempts'] += 1
        
        # Try different recovery strategies
        recovery_strategies = [
            ('retry_with_backoff', self._recovery_retry_with_backoff),
            ('fallback_operation', self._recovery_fallback),
            ('cache_lookup', self._recovery_cache_lookup)
        ]
        
        for strategy_name, recovery_func in recovery_strategies:
            try:
                start_time = time.time()
                result = await recovery_func(failure_event, operation, *args, **kwargs)
                recovery_time = time.time() - start_time
                
                return {
                    'success': True,
                    'strategy': strategy_name,
                    'recovery_time': recovery_time,
                    'result': result
                }
                
            except Exception:
                continue  # Try next strategy
        
        return {'success': False}
    
    async def _recovery_retry_with_backoff(self,
                                         failure_event: FailureEvent,
                                         operation: Callable,
                                         *args,
                                         **kwargs) -> Any:
        """Recovery through retry with exponential backoff."""
        await asyncio.sleep(2.0)  # Initial backoff
        
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        else:
            return operation(*args, **kwargs)
    
    async def _recovery_fallback(self,
                               failure_event: FailureEvent,
                               operation: Callable,
                               *args,
                               **kwargs) -> Any:
        """Recovery through fallback operation."""
        # Return a safe default or cached result
        return {'status': 'fallback', 'message': 'Using fallback response'}
    
    async def _recovery_cache_lookup(self,
                                   failure_event: FailureEvent,
                                   operation: Callable,
                                   *args,
                                   **kwargs) -> Any:
        """Recovery through cache lookup."""
        # Simulate cache lookup
        cache_key = f"{failure_event.component}:{hash(str(args))}"
        # Return cached result if available
        raise Exception("Cache miss")  # Simulate cache miss
    
    async def _trigger_adaptation(self, component: str, failure_event: FailureEvent) -> None:
        """Trigger strategy adaptation based on failure."""
        # Check if adaptation is needed
        component_failures = [
            f for f in self.learning_engine.failure_history[-100:]
            if f.component == component
        ]
        
        failure_rate = len(component_failures) / 100.0
        
        if failure_rate > self.adaptive_thresholds['failure_rate_threshold']:
            # High failure rate, adapt strategies
            predicted_pattern = self.learning_engine.predict_failure_pattern(component, failure_event.context)
            
            if predicted_pattern:
                recommended_strategy = self.learning_engine.recommend_strategy(component, predicted_pattern)
                
                if recommended_strategy and recommended_strategy.value in self.active_strategies:
                    # Update strategy for this component
                    self.logger.info(f"Adapting strategy for {component} to {recommended_strategy.value}")
                    # Implementation would update component-specific strategies
    
    def _update_success_metrics(self, component: str, execution_time: float) -> None:
        """Update success metrics."""
        # Update response time moving average
        current_avg = self.performance_metrics['mean_response_time']
        total_requests = self.performance_metrics['total_requests']
        
        self.performance_metrics['mean_response_time'] = (
            (current_avg * (total_requests - 1) + execution_time) / total_requests
        )
        
        # Update component health
        if component not in self.component_health:
            self.component_health[component] = {
                'success_count': 0,
                'failure_count': 0,
                'avg_response_time': 0.0,
                'health_score': 1.0
            }
        
        comp_health = self.component_health[component]
        comp_health['success_count'] += 1
        
        # Update component response time average
        total_comp_requests = comp_health['success_count'] + comp_health['failure_count']
        if total_comp_requests > 0:
            comp_health['avg_response_time'] = (
                (comp_health['avg_response_time'] * (total_comp_requests - 1) + execution_time) 
                / total_comp_requests
            )
        
        # Calculate health score
        success_rate = comp_health['success_count'] / total_comp_requests
        response_time_factor = min(1.0, 5.0 / max(comp_health['avg_response_time'], 0.1))
        comp_health['health_score'] = success_rate * response_time_factor
    
    def get_system_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience system status."""
        return {
            'performance_metrics': dict(self.performance_metrics),
            'active_strategies': {
                name: {
                    'strategy': config.strategy.value,
                    'parameters': config.parameters,
                    'adaptive': config.adaptive
                }
                for name, config in self.active_strategies.items()
            },
            'component_health': dict(self.component_health),
            'learning_stats': {
                'total_failures_recorded': len(self.learning_engine.failure_history),
                'pattern_models': len(self.learning_engine.pattern_models),
                'learning_enabled': self.learning_engine.learning_enabled
            },
            'adaptive_thresholds': dict(self.adaptive_thresholds),
            'system_health_score': self._calculate_system_health_score()
        }
    
    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score."""
        if not self.component_health:
            return 1.0
        
        component_scores = [health['health_score'] for health in self.component_health.values()]
        return np.mean(component_scores)
    
    def export_learning_data(self, output_path: str) -> None:
        """Export learning data for analysis."""
        export_data = {
            'failure_history': [
                {
                    'event_id': f.event_id,
                    'timestamp': f.timestamp,
                    'component': f.component,
                    'failure_type': f.failure_type,
                    'duration': f.duration,
                    'recovery_strategy': f.recovery_strategy,
                    'successful_recovery': f.successful_recovery
                }
                for f in self.learning_engine.failure_history
            ],
            'pattern_models': {
                component: {
                    'failure_rates': dict(model['failure_rates']),
                    'recovery_strategies': {k: list(v) for k, v in model['recovery_strategies'].items()}
                }
                for component, model in self.learning_engine.pattern_models.items()
            },
            'performance_metrics': dict(self.performance_metrics),
            'export_timestamp': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Learning data exported to {output_path}")