"""Advanced auto-scaling and load balancing system for RLHF audit trail.

This module provides intelligent scaling decisions, load balancing algorithms,
and resource optimization for production-grade deployments.
"""

import asyncio
import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import statistics
import logging
import weakref

from .exceptions import ScalingError, LoadBalancingError, ResourceExhaustedError
from .advanced_monitoring import record_metric, fire_alert, AlertSeverity


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


class LoadBalancingStrategy(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    ADAPTIVE = "adaptive"


@dataclass
class InstanceMetrics:
    """Metrics for a single instance."""
    id: str
    cpu_usage: float
    memory_usage: float
    active_connections: int
    request_queue_size: int
    response_time_avg: float
    error_rate: float
    last_updated: float
    health_status: str = "healthy"
    
    @property
    def load_score(self) -> float:
        """Calculate composite load score (0.0 to 1.0)."""
        weights = {
            "cpu": 0.3,
            "memory": 0.25,
            "connections": 0.2,
            "queue": 0.15,
            "response_time": 0.1
        }
        
        # Normalize metrics to 0-1 scale
        cpu_score = min(self.cpu_usage / 100.0, 1.0)
        memory_score = min(self.memory_usage / 100.0, 1.0)
        connections_score = min(self.active_connections / 100.0, 1.0)  # Assume 100 max
        queue_score = min(self.request_queue_size / 50.0, 1.0)  # Assume 50 max
        response_score = min(self.response_time_avg / 5.0, 1.0)  # Assume 5s max
        
        total_score = (
            cpu_score * weights["cpu"] +
            memory_score * weights["memory"] +
            connections_score * weights["connections"] +
            queue_score * weights["queue"] +
            response_score * weights["response_time"]
        )
        
        return total_score
    
    def is_healthy(self) -> bool:
        """Check if instance is healthy."""
        return (
            self.health_status == "healthy" and
            self.cpu_usage < 90.0 and
            self.memory_usage < 85.0 and
            self.error_rate < 0.1 and
            time.time() - self.last_updated < 60.0  # Recent data
        )
    
    def can_handle_load(self) -> bool:
        """Check if instance can handle additional load."""
        return (
            self.is_healthy() and
            self.load_score < 0.8 and
            self.request_queue_size < 30
        )


@dataclass
class ScalingDecision:
    """Represents a scaling decision."""
    action: str  # "scale_up", "scale_down", "no_change"
    target_instances: int
    current_instances: int
    confidence: float
    reasoning: str
    metrics_snapshot: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "action": self.action,
            "target_instances": self.target_instances,
            "current_instances": self.current_instances,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp
        }


class PredictiveScaler:
    """Predictive scaling based on historical patterns and ML."""
    
    def __init__(self, prediction_window: int = 300):
        """Initialize predictive scaler.
        
        Args:
            prediction_window: Prediction window in seconds
        """
        self.prediction_window = prediction_window
        self.load_history: deque = deque(maxlen=2000)  # ~33 minutes at 1s intervals
        self.scaling_history: List[ScalingDecision] = []
        
        # Pattern detection
        self.daily_patterns: Dict[int, List[float]] = defaultdict(list)  # hour -> loads
        self.weekly_patterns: Dict[int, List[float]] = defaultdict(list)  # day -> loads
        
        self.logger = logging.getLogger(__name__)
        
    def add_load_sample(self, load: float):
        """Add a load sample for pattern learning."""
        timestamp = time.time()
        self.load_history.append((timestamp, load))
        
        # Extract time patterns
        dt = datetime.fromtimestamp(timestamp)
        self.daily_patterns[dt.hour].append(load)
        self.weekly_patterns[dt.weekday()].append(load)
        
        # Keep patterns manageable
        for hour_patterns in self.daily_patterns.values():
            if len(hour_patterns) > 100:
                hour_patterns[:] = hour_patterns[-50:]
        
        for day_patterns in self.weekly_patterns.values():
            if len(day_patterns) > 100:
                day_patterns[:] = day_patterns[-50:]
                
    def predict_load(self, minutes_ahead: int = 5) -> Tuple[float, float]:
        """Predict future load.
        
        Args:
            minutes_ahead: Minutes to predict ahead
            
        Returns:
            Tuple of (predicted_load, confidence)
        """
        if len(self.load_history) < 10:
            return 0.5, 0.0  # No data, assume medium load with no confidence
        
        current_time = time.time()
        future_time = current_time + (minutes_ahead * 60)
        future_dt = datetime.fromtimestamp(future_time)
        
        predictions = []
        confidences = []
        
        # Trend-based prediction
        recent_loads = [load for ts, load in self.load_history if current_time - ts < 600]
        if len(recent_loads) >= 5:
            # Simple linear trend
            x = list(range(len(recent_loads)))
            y = recent_loads
            
            # Calculate slope
            n = len(recent_loads)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] * x[i] for i in range(n))
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                intercept = (sum_y - slope * sum_x) / n
                
                trend_prediction = intercept + slope * (n + minutes_ahead)
                trend_prediction = max(0.0, min(1.0, trend_prediction))
                
                predictions.append(trend_prediction)
                confidences.append(0.6)
        
        # Pattern-based prediction
        hour_pattern = self.daily_patterns.get(future_dt.hour, [])
        if len(hour_pattern) >= 3:
            hour_prediction = statistics.median(hour_pattern)
            predictions.append(hour_prediction)
            confidences.append(0.7)
        
        day_pattern = self.weekly_patterns.get(future_dt.weekday(), [])
        if len(day_pattern) >= 3:
            day_prediction = statistics.median(day_pattern)
            predictions.append(day_prediction)
            confidences.append(0.5)
        
        # Seasonal adjustment (simplified)
        if len(self.load_history) >= 60:  # At least 1 minute of data
            same_time_samples = []
            target_hour = future_dt.hour
            
            for ts, load in self.load_history:
                sample_dt = datetime.fromtimestamp(ts)
                if abs(sample_dt.hour - target_hour) <= 1:
                    same_time_samples.append(load)
            
            if len(same_time_samples) >= 3:
                seasonal_prediction = statistics.median(same_time_samples)
                predictions.append(seasonal_prediction)
                confidences.append(0.4)
        
        if not predictions:
            # Fallback to current load
            current_load = self.load_history[-1][1] if self.load_history else 0.5
            return current_load, 0.1
        
        # Weighted average of predictions
        total_weight = sum(confidences)
        if total_weight > 0:
            weighted_prediction = sum(
                pred * conf for pred, conf in zip(predictions, confidences)
            ) / total_weight
            avg_confidence = statistics.mean(confidences)
        else:
            weighted_prediction = statistics.mean(predictions)
            avg_confidence = 0.3
        
        return max(0.0, min(1.0, weighted_prediction)), avg_confidence
    
    def should_scale_predictively(
        self,
        current_instances: int,
        min_instances: int,
        max_instances: int
    ) -> Optional[ScalingDecision]:
        """Make predictive scaling decision."""
        predicted_load, confidence = self.predict_load(minutes_ahead=5)
        
        if confidence < 0.5:
            return None  # Not confident enough
        
        current_time = time.time()
        current_load = self.load_history[-1][1] if self.load_history else 0.5
        
        # Scaling thresholds with hysteresis
        scale_up_threshold = 0.75
        scale_down_threshold = 0.35
        
        action = "no_change"
        target_instances = current_instances
        reasoning = f"Predicted load: {predicted_load:.2f}, confidence: {confidence:.2f}"
        
        if predicted_load > scale_up_threshold and current_instances < max_instances:
            target_instances = min(current_instances + 1, max_instances)
            action = "scale_up"
            reasoning += f". Load will exceed {scale_up_threshold}"
            
        elif predicted_load < scale_down_threshold and current_instances > min_instances:
            # Additional safety check - don't scale down if current load is high
            if current_load < 0.5:
                target_instances = max(current_instances - 1, min_instances)
                action = "scale_down"
                reasoning += f". Load will drop below {scale_down_threshold}"
        
        decision = ScalingDecision(
            action=action,
            target_instances=target_instances,
            current_instances=current_instances,
            confidence=confidence,
            reasoning=reasoning,
            metrics_snapshot={
                "predicted_load": predicted_load,
                "current_load": current_load,
                "prediction_confidence": confidence
            },
            timestamp=current_time
        )
        
        if action != "no_change":
            self.scaling_history.append(decision)
            
        return decision


class LoadBalancer:
    """Intelligent load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        """Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
        """
        self.strategy = strategy
        self.instances: Dict[str, InstanceMetrics] = {}
        self.request_history: deque = deque(maxlen=10000)
        
        # Round-robin state
        self._round_robin_index = 0
        
        # Adaptive learning
        self._strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self._current_adaptive_strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
        
        self.logger = logging.getLogger(__name__)
        
    def add_instance(self, instance: InstanceMetrics):
        """Add or update an instance."""
        self.instances[instance.id] = instance
        record_metric("load_balancer.instances", len(self.instances))
        
    def remove_instance(self, instance_id: str):
        """Remove an instance."""
        if instance_id in self.instances:
            del self.instances[instance_id]
            record_metric("load_balancer.instances", len(self.instances))
            
    def get_healthy_instances(self) -> List[InstanceMetrics]:
        """Get list of healthy instances."""
        return [
            instance for instance in self.instances.values()
            if instance.is_healthy()
        ]
    
    def select_instance(self, request_metadata: Optional[Dict] = None) -> Optional[str]:
        """Select the best instance for a request."""
        healthy_instances = self.get_healthy_instances()
        
        if not healthy_instances:
            fire_alert(
                "no_healthy_instances",
                "No healthy instances available for load balancing",
                AlertSeverity.CRITICAL,
                "load_balancer"
            )
            return None
        
        if len(healthy_instances) == 1:
            return healthy_instances[0].id
        
        # Apply strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected = self._round_robin_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected = self._least_connections_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
            selected = self._weighted_response_time_select(healthy_instances)
        else:  # ADAPTIVE
            selected = self._adaptive_select(healthy_instances)
        
        # Record selection for learning
        self._record_selection(selected, request_metadata)
        
        return selected.id if selected else None
    
    def _round_robin_select(self, instances: List[InstanceMetrics]) -> InstanceMetrics:
        """Round-robin selection."""
        selected = instances[self._round_robin_index % len(instances)]
        self._round_robin_index += 1
        return selected
    
    def _least_connections_select(self, instances: List[InstanceMetrics]) -> InstanceMetrics:
        """Select instance with least connections."""
        return min(instances, key=lambda i: i.active_connections)
    
    def _weighted_response_time_select(self, instances: List[InstanceMetrics]) -> InstanceMetrics:
        """Select based on weighted response time."""
        # Calculate weights (inverse of response time + small constant)
        weights = []
        for instance in instances:
            weight = 1.0 / (instance.response_time_avg + 0.1)
            # Adjust for load
            weight *= (1.0 - instance.load_score)
            weights.append(weight)
        
        # Select instance with highest weight
        best_index = weights.index(max(weights))
        return instances[best_index]
    
    def _adaptive_select(self, instances: List[InstanceMetrics]) -> InstanceMetrics:
        """Adaptive selection using learned performance."""
        # Try different strategies and learn from results
        current_time = time.time()
        
        # Switch strategies periodically to learn
        strategies = [
            LoadBalancingStrategy.LEAST_CONNECTIONS,
            LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME,
            LoadBalancingStrategy.ROUND_ROBIN
        ]
        
        # Evaluate strategy performance
        best_strategy = self._current_adaptive_strategy
        best_score = 0.0
        
        for strategy in strategies:
            recent_performance = self._strategy_performance[strategy.value]
            if len(recent_performance) >= 5:
                # Calculate average response time (lower is better)
                avg_response = sum(recent_performance[-10:]) / min(10, len(recent_performance))
                score = 1.0 / (avg_response + 0.1)  # Inverse for scoring
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
        
        self._current_adaptive_strategy = best_strategy
        
        # Apply the best strategy
        if best_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(instances)
        elif best_strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
            return self._weighted_response_time_select(instances)
        else:
            return self._round_robin_select(instances)
    
    def _record_selection(self, selected: InstanceMetrics, request_metadata: Optional[Dict]):
        """Record selection for learning."""
        self.request_history.append({
            "timestamp": time.time(),
            "instance_id": selected.id,
            "strategy": self.strategy.value,
            "load_score": selected.load_score,
            "response_time": selected.response_time_avg,
            "metadata": request_metadata or {}
        })
    
    def record_request_result(
        self,
        instance_id: str,
        response_time: float,
        success: bool
    ):
        """Record the result of a request for learning."""
        if instance_id in self.instances:
            instance = self.instances[instance_id]
            
            # Update instance metrics
            instance.response_time_avg = (
                instance.response_time_avg * 0.9 + response_time * 0.1
            )
            
            if not success:
                instance.error_rate = instance.error_rate * 0.9 + 0.1
            else:
                instance.error_rate = instance.error_rate * 0.95
            
            # Record for adaptive learning
            if hasattr(self, '_current_adaptive_strategy'):
                strategy_name = self._current_adaptive_strategy.value
                self._strategy_performance[strategy_name].append(response_time)
                
                # Keep recent history
                if len(self._strategy_performance[strategy_name]) > 100:
                    self._strategy_performance[strategy_name] = \
                        self._strategy_performance[strategy_name][-50:]
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across instances."""
        if not self.instances:
            return {}
        
        total_connections = sum(i.active_connections for i in self.instances.values())
        total_load = sum(i.load_score for i in self.instances.values())
        
        distribution = {}
        for instance_id, instance in self.instances.items():
            distribution[instance_id] = {
                "load_score": instance.load_score,
                "connections": instance.active_connections,
                "connection_percentage": (
                    instance.active_connections / total_connections * 100
                    if total_connections > 0 else 0
                ),
                "load_percentage": (
                    instance.load_score / total_load * 100
                    if total_load > 0 else 0
                ),
                "health_status": instance.health_status
            }
        
        return {
            "instances": distribution,
            "total_instances": len(self.instances),
            "healthy_instances": len(self.get_healthy_instances()),
            "total_connections": total_connections,
            "average_load": total_load / len(self.instances),
            "strategy": self.strategy.value
        }


class AdvancedAutoScaler:
    """Advanced auto-scaling system with multiple strategies."""
    
    def __init__(
        self,
        min_instances: int = 2,
        max_instances: int = 50,
        strategy: ScalingStrategy = ScalingStrategy.HYBRID
    ):
        """Initialize advanced auto-scaler.
        
        Args:
            min_instances: Minimum instances to maintain
            max_instances: Maximum instances allowed
            strategy: Scaling strategy to use
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.strategy = strategy
        self.current_instances = min_instances
        
        # Scaling components
        self.predictive_scaler = PredictiveScaler()
        self.load_balancer = LoadBalancer(LoadBalancingStrategy.ADAPTIVE)
        
        # Scaling state
        self.last_scale_action = time.time()
        self.scale_cooldown = 180.0  # 3 minutes
        self.scaling_history: List[ScalingDecision] = []
        
        # Metrics tracking
        self.system_metrics: Dict[str, Any] = {}
        self.scaling_events: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)
        
    def update_system_metrics(self, metrics: Dict[str, Any]):
        """Update system metrics for scaling decisions."""
        self.system_metrics = metrics
        
        # Extract load for predictive scaling
        if "overall_load" in metrics:
            self.predictive_scaler.add_load_sample(metrics["overall_load"])
        
        # Update instance metrics in load balancer
        if "instances" in metrics:
            for instance_id, instance_data in metrics["instances"].items():
                instance_metrics = InstanceMetrics(
                    id=instance_id,
                    cpu_usage=instance_data.get("cpu_usage", 0.0),
                    memory_usage=instance_data.get("memory_usage", 0.0),
                    active_connections=instance_data.get("active_connections", 0),
                    request_queue_size=instance_data.get("queue_size", 0),
                    response_time_avg=instance_data.get("response_time", 0.0),
                    error_rate=instance_data.get("error_rate", 0.0),
                    last_updated=time.time(),
                    health_status=instance_data.get("health_status", "healthy")
                )
                self.load_balancer.add_instance(instance_metrics)
                
    def make_scaling_decision(self) -> Optional[ScalingDecision]:
        """Make an intelligent scaling decision."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scale_action < self.scale_cooldown:
            return None
        
        decisions = []
        
        # Reactive scaling based on current metrics
        reactive_decision = self._make_reactive_decision()
        if reactive_decision:
            decisions.append(reactive_decision)
        
        # Predictive scaling
        if self.strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
            predictive_decision = self.predictive_scaler.should_scale_predictively(
                self.current_instances,
                self.min_instances,
                self.max_instances
            )
            if predictive_decision:
                decisions.append(predictive_decision)
        
        # Combine decisions if multiple
        if not decisions:
            return None
        
        if len(decisions) == 1:
            final_decision = decisions[0]
        else:
            final_decision = self._combine_decisions(decisions)
        
        # Execute if confidence is high enough
        if final_decision.confidence >= 0.6 and final_decision.action != "no_change":
            self._execute_scaling_decision(final_decision)
            
        return final_decision
    
    def _make_reactive_decision(self) -> Optional[ScalingDecision]:
        """Make reactive scaling decision based on current state."""
        if not self.system_metrics:
            return None
        
        current_load = self.system_metrics.get("overall_load", 0.5)
        healthy_instances = len(self.load_balancer.get_healthy_instances())
        
        # Scale up conditions
        if (current_load > 0.8 and 
            healthy_instances < self.max_instances and
            healthy_instances == self.current_instances):  # All instances healthy
            
            return ScalingDecision(
                action="scale_up",
                target_instances=min(self.current_instances + 1, self.max_instances),
                current_instances=self.current_instances,
                confidence=0.8,
                reasoning=f"High load ({current_load:.2f}) requires more capacity",
                metrics_snapshot=self.system_metrics.copy(),
                timestamp=time.time()
            )
        
        # Scale down conditions
        elif (current_load < 0.3 and 
              self.current_instances > self.min_instances and
              healthy_instances >= self.current_instances * 0.8):  # Most instances healthy
            
            return ScalingDecision(
                action="scale_down",
                target_instances=max(self.current_instances - 1, self.min_instances),
                current_instances=self.current_instances,
                confidence=0.7,
                reasoning=f"Low load ({current_load:.2f}) allows capacity reduction",
                metrics_snapshot=self.system_metrics.copy(),
                timestamp=time.time()
            )
        
        return None
    
    def _combine_decisions(self, decisions: List[ScalingDecision]) -> ScalingDecision:
        """Combine multiple scaling decisions intelligently."""
        # Priority: safety first, then performance
        scale_up_decisions = [d for d in decisions if d.action == "scale_up"]
        scale_down_decisions = [d for d in decisions if d.action == "scale_down"]
        
        if scale_up_decisions:
            # Take the most conservative scale-up decision
            best_decision = max(scale_up_decisions, key=lambda d: d.confidence)
            best_decision.reasoning += " (combined with other signals)"
            return best_decision
            
        elif scale_down_decisions:
            # Take the most conservative scale-down decision
            best_decision = max(scale_down_decisions, key=lambda d: d.confidence)
            best_decision.reasoning += " (combined with other signals)"
            return best_decision
        
        # No clear decision
        return ScalingDecision(
            action="no_change",
            target_instances=self.current_instances,
            current_instances=self.current_instances,
            confidence=0.5,
            reasoning="Conflicting signals, maintaining current capacity",
            metrics_snapshot=self.system_metrics.copy(),
            timestamp=time.time()
        )
    
    def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision."""
        try:
            old_instances = self.current_instances
            self.current_instances = decision.target_instances
            self.last_scale_action = time.time()
            
            # Record the decision
            self.scaling_history.append(decision)
            
            # Fire appropriate alerts
            if decision.action == "scale_up":
                fire_alert(
                    "auto_scale_up",
                    f"Scaling up from {old_instances} to {self.current_instances} instances",
                    AlertSeverity.INFO,
                    "autoscaler",
                    {"reasoning": decision.reasoning}
                )
                record_metric("autoscaler.scale_up_events", 1)
                
            elif decision.action == "scale_down":
                fire_alert(
                    "auto_scale_down",
                    f"Scaling down from {old_instances} to {self.current_instances} instances",
                    AlertSeverity.INFO,
                    "autoscaler",
                    {"reasoning": decision.reasoning}
                )
                record_metric("autoscaler.scale_down_events", 1)
            
            # Log the scaling action
            self.logger.info(f"Scaling action executed: {decision.to_dict()}")
            
            # Update metrics
            record_metric("autoscaler.current_instances", self.current_instances)
            
        except Exception as e:
            raise ScalingError(
                f"Failed to execute scaling decision: {e}",
                scaling_action=decision.action,
                target_instances=decision.target_instances,
                current_instances=self.current_instances
            )
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status."""
        recent_decisions = self.scaling_history[-10:] if self.scaling_history else []
        
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "strategy": self.strategy.value,
            "last_scale_action": self.last_scale_action,
            "cooldown_remaining": max(0, self.scale_cooldown - (time.time() - self.last_scale_action)),
            "recent_decisions": [d.to_dict() for d in recent_decisions],
            "load_balancer_stats": self.load_balancer.get_load_distribution(),
            "scaling_events_count": len(self.scaling_history)
        }