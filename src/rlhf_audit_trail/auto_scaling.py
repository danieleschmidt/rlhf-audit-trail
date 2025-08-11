"""
Auto-scaling and load balancing system for RLHF audit trail.
Generation 3: Auto-scaling triggers, load balancing, and resource management.
"""

import asyncio
import time
import threading
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import json
import os

try:
    import kubernetes
    from kubernetes import client, config
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

from .exceptions import ScalingError, LoadBalancingError


logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_BASED = "resource_based"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    request_rate: float
    error_rate: float
    response_time_ms: float


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""
    name: str
    metric_name: str
    threshold_up: float
    threshold_down: float
    duration_seconds: int
    cooldown_seconds: int
    min_instances: int
    max_instances: int
    scale_factor: float = 1.5
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceInstance:
    """Service instance information."""
    instance_id: str
    host: str
    port: int
    status: str
    cpu_usage: float
    memory_usage: float
    active_connections: int
    last_health_check: datetime
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects system and application metrics for scaling decisions."""
    
    def __init__(self, collection_interval: int = 30):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: Metrics collection interval in seconds
        """
        self.collection_interval = collection_interval
        self.metrics_history: deque = deque(maxlen=1000)
        self.custom_metrics: Dict[str, Callable[[], float]] = {}
        self.is_collecting = False
        self.collector_thread: Optional[threading.Thread] = None
        
        # Application-specific metrics
        self.request_count = 0
        self.error_count = 0
        self.response_times: deque = deque(maxlen=1000)
        self.active_sessions = 0
    
    def start_collection(self):
        """Start metrics collection."""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def add_custom_metric(self, name: str, collector_func: Callable[[], float]):
        """Add custom metric collector."""
        self.custom_metrics[name] = collector_func
    
    def record_request(self, response_time_ms: float, error: bool = False):
        """Record application request metrics."""
        self.request_count += 1
        if error:
            self.error_count += 1
        self.response_times.append(response_time_ms)
    
    def set_active_sessions(self, count: int):
        """Set current active sessions count."""
        self.active_sessions = count
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current system metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Application metrics
            current_time = time.time()
            recent_requests = [t for t in self.response_times if current_time - t < 60]  # Last minute
            request_rate = len(recent_requests) / 60 if recent_requests else 0
            
            error_rate = (self.error_count / max(self.request_count, 1)) * 100 if self.request_count > 0 else 0
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            
            return ResourceMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_bytes_sent=network.bytes_sent if network else 0,
                network_bytes_recv=network.bytes_recv if network else 0,
                active_connections=self.active_sessions,
                request_rate=request_rate,
                error_rate=error_rate,
                response_time_ms=avg_response_time
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            # Return zero metrics on failure
            return ResourceMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=0.0, memory_percent=0.0, disk_percent=0.0,
                network_bytes_sent=0, network_bytes_recv=0,
                active_connections=0, request_rate=0.0,
                error_rate=0.0, response_time_ms=0.0
            )
    
    def get_metric_value(self, metric_name: str, time_window_seconds: int = 300) -> float:
        """Get aggregated metric value over time window."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_window_seconds)
        
        # Filter recent metrics
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return 0.0
        
        # Calculate aggregate based on metric type
        if metric_name == "cpu_percent":
            return sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        elif metric_name == "memory_percent":
            return sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        elif metric_name == "disk_percent":
            return sum(m.disk_percent for m in recent_metrics) / len(recent_metrics)
        elif metric_name == "request_rate":
            return sum(m.request_rate for m in recent_metrics) / len(recent_metrics)
        elif metric_name == "error_rate":
            return sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        elif metric_name == "response_time_ms":
            return sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
        elif metric_name == "active_connections":
            return max(m.active_connections for m in recent_metrics)  # Peak connections
        elif metric_name in self.custom_metrics:
            try:
                return self.custom_metrics[metric_name]()
            except Exception as e:
                logger.error(f"Custom metric {metric_name} collection failed: {e}")
                return 0.0
        else:
            return 0.0
    
    def _collection_loop(self):
        """Main metrics collection loop."""
        while self.is_collecting:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(5)  # Short delay on error


class AutoScaler:
    """Auto-scaling engine based on resource metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize auto-scaler.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self.scaling_rules: List[ScalingRule] = []
        self.current_instances = 1
        self.last_scaling_time: Dict[str, datetime] = {}
        self.scaling_history: List[Dict[str, Any]] = []
        
        self.is_enabled = False
        self.scaling_thread: Optional[threading.Thread] = None
        
        # Scaling callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        
        # Default scaling rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default scaling rules."""
        self.scaling_rules = [
            ScalingRule(
                name="cpu_scaling",
                metric_name="cpu_percent",
                threshold_up=70.0,
                threshold_down=30.0,
                duration_seconds=300,  # 5 minutes
                cooldown_seconds=600,  # 10 minutes
                min_instances=1,
                max_instances=10,
                scale_factor=2.0
            ),
            ScalingRule(
                name="memory_scaling",
                metric_name="memory_percent",
                threshold_up=80.0,
                threshold_down=40.0,
                duration_seconds=180,  # 3 minutes
                cooldown_seconds=300,  # 5 minutes
                min_instances=1,
                max_instances=8
            ),
            ScalingRule(
                name="request_rate_scaling",
                metric_name="request_rate",
                threshold_up=100.0,  # 100 requests per minute
                threshold_down=20.0,   # 20 requests per minute
                duration_seconds=120,  # 2 minutes
                cooldown_seconds=240,  # 4 minutes
                min_instances=1,
                max_instances=15
            ),
            ScalingRule(
                name="response_time_scaling",
                metric_name="response_time_ms",
                threshold_up=1000.0,  # 1 second
                threshold_down=200.0,  # 200ms
                duration_seconds=180,  # 3 minutes
                cooldown_seconds=300,  # 5 minutes
                min_instances=1,
                max_instances=12
            )
        ]
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add custom scaling rule."""
        self.scaling_rules.append(rule)
    
    def set_scale_callbacks(self, 
                          scale_up_callback: Callable[[int], bool],
                          scale_down_callback: Callable[[int], bool]):
        """Set scaling callback functions."""
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback
    
    def start_autoscaling(self):
        """Start auto-scaling monitoring."""
        if self.is_enabled:
            return
        
        self.is_enabled = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        logger.info("Auto-scaling started")
    
    def stop_autoscaling(self):
        """Stop auto-scaling monitoring."""
        self.is_enabled = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5)
        logger.info("Auto-scaling stopped")
    
    def evaluate_scaling_decision(self) -> Tuple[ScalingDirection, int, str]:
        """Evaluate whether scaling is needed."""
        scaling_decisions = []
        
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            decision = self._evaluate_rule(rule)
            if decision[0] != ScalingDirection.NONE:
                scaling_decisions.append((decision, rule))
        
        if not scaling_decisions:
            return ScalingDirection.NONE, self.current_instances, "No scaling needed"
        
        # Prioritize scale-up decisions over scale-down
        up_decisions = [d for d in scaling_decisions if d[0][0] == ScalingDirection.UP]
        down_decisions = [d for d in scaling_decisions if d[0][0] == ScalingDirection.DOWN]
        
        if up_decisions:
            # Take the most aggressive scale-up decision
            decision, rule = max(up_decisions, key=lambda x: x[0][1])
            return decision[0], decision[1], f"Scale up triggered by rule: {rule.name}"
        elif down_decisions:
            # Take the most conservative scale-down decision
            decision, rule = min(down_decisions, key=lambda x: x[0][1])
            return decision[0], decision[1], f"Scale down triggered by rule: {rule.name}"
        
        return ScalingDirection.NONE, self.current_instances, "No scaling needed"
    
    def _evaluate_rule(self, rule: ScalingRule) -> Tuple[ScalingDirection, int]:
        """Evaluate a single scaling rule."""
        # Check cooldown
        last_scaling = self.last_scaling_time.get(rule.name)
        if last_scaling and (datetime.utcnow() - last_scaling).total_seconds() < rule.cooldown_seconds:
            return ScalingDirection.NONE, self.current_instances
        
        # Get metric value over the rule duration
        metric_value = self.metrics_collector.get_metric_value(rule.metric_name, rule.duration_seconds)
        
        # Scale up decision
        if metric_value > rule.threshold_up and self.current_instances < rule.max_instances:
            target_instances = min(
                int(self.current_instances * rule.scale_factor),
                rule.max_instances
            )
            return ScalingDirection.UP, target_instances
        
        # Scale down decision
        elif metric_value < rule.threshold_down and self.current_instances > rule.min_instances:
            target_instances = max(
                int(self.current_instances / rule.scale_factor),
                rule.min_instances
            )
            return ScalingDirection.DOWN, target_instances
        
        return ScalingDirection.NONE, self.current_instances
    
    def _scaling_loop(self):
        """Main auto-scaling loop."""
        while self.is_enabled:
            try:
                direction, target_instances, reason = self.evaluate_scaling_decision()
                
                if direction != ScalingDirection.NONE:
                    success = self._execute_scaling(direction, target_instances)
                    
                    # Record scaling event
                    self.scaling_history.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "direction": direction.value,
                        "from_instances": self.current_instances,
                        "to_instances": target_instances,
                        "reason": reason,
                        "success": success
                    })
                    
                    # Limit history
                    if len(self.scaling_history) > 1000:
                        self.scaling_history = self.scaling_history[-500:]
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                time.sleep(30)  # Shorter delay on error
    
    def _execute_scaling(self, direction: ScalingDirection, target_instances: int) -> bool:
        """Execute scaling operation."""
        try:
            if direction == ScalingDirection.UP and self.scale_up_callback:
                success = self.scale_up_callback(target_instances)
            elif direction == ScalingDirection.DOWN and self.scale_down_callback:
                success = self.scale_down_callback(target_instances)
            else:
                logger.warning(f"No callback available for scaling {direction.value}")
                return False
            
            if success:
                self.current_instances = target_instances
                self.last_scaling_time[f"global_{direction.value}"] = datetime.utcnow()
                logger.info(f"Scaling {direction.value} to {target_instances} instances successful")
            else:
                logger.error(f"Scaling {direction.value} to {target_instances} instances failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Scaling execution error: {e}")
            return False
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        recent_scalings = [
            event for event in self.scaling_history
            if datetime.fromisoformat(event["timestamp"]) > datetime.utcnow() - timedelta(hours=24)
        ]
        
        return {
            "enabled": self.is_enabled,
            "current_instances": self.current_instances,
            "rules_count": len([r for r in self.scaling_rules if r.enabled]),
            "recent_scalings_24h": len(recent_scalings),
            "last_scaling": self.scaling_history[-1] if self.scaling_history else None,
            "scaling_success_rate": (
                sum(1 for e in recent_scalings if e["success"]) / len(recent_scalings) * 100
                if recent_scalings else 100
            )
        }


class LoadBalancer:
    """Load balancer for distributing requests across service instances."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
        """
        self.strategy = strategy
        self.instances: List[ServiceInstance] = []
        self.current_index = 0
        self.request_counts: Dict[str, int] = {}
        self.lock = threading.RLock()
        
        # Health checking
        self.health_check_interval = 30
        self.health_check_enabled = True
        self.health_check_thread: Optional[threading.Thread] = None
    
    def add_instance(self, instance: ServiceInstance):
        """Add service instance to load balancer."""
        with self.lock:
            if instance.instance_id not in [i.instance_id for i in self.instances]:
                self.instances.append(instance)
                self.request_counts[instance.instance_id] = 0
                logger.info(f"Added instance {instance.instance_id} to load balancer")
    
    def remove_instance(self, instance_id: str):
        """Remove service instance from load balancer."""
        with self.lock:
            self.instances = [i for i in self.instances if i.instance_id != instance_id]
            if instance_id in self.request_counts:
                del self.request_counts[instance_id]
            logger.info(f"Removed instance {instance_id} from load balancer")
    
    def get_next_instance(self) -> Optional[ServiceInstance]:
        """Get next instance based on load balancing strategy."""
        with self.lock:
            healthy_instances = [i for i in self.instances if i.status == "healthy"]
            
            if not healthy_instances:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                instance = self._round_robin_select(healthy_instances)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                instance = self._least_connections_select(healthy_instances)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                instance = self._weighted_round_robin_select(healthy_instances)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                instance = self._resource_based_select(healthy_instances)
            else:
                instance = healthy_instances[0]  # Default
            
            if instance:
                self.request_counts[instance.instance_id] += 1
            
            return instance
    
    def record_request_completion(self, instance_id: str, success: bool, response_time_ms: float):
        """Record request completion for load balancing decisions."""
        with self.lock:
            # Update instance metrics
            for instance in self.instances:
                if instance.instance_id == instance_id:
                    # Update connection count (simplified)
                    if success:
                        instance.active_connections = max(0, instance.active_connections - 1)
                    break
    
    def start_health_checks(self):
        """Start health checking for instances."""
        if not self.health_check_enabled or self.health_check_thread:
            return
        
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        logger.info("Load balancer health checks started")
    
    def stop_health_checks(self):
        """Stop health checking."""
        self.health_check_enabled = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        logger.info("Load balancer health checks stopped")
    
    def _round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin selection."""
        instance = instances[self.current_index % len(instances)]
        self.current_index = (self.current_index + 1) % len(instances)
        return instance
    
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least active connections."""
        return min(instances, key=lambda i: i.active_connections)
    
    def _weighted_round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round robin selection based on instance weights."""
        total_weight = sum(i.weight for i in instances)
        if total_weight == 0:
            return self._round_robin_select(instances)
        
        # Simple weighted selection (could be optimized)
        weighted_instances = []
        for instance in instances:
            count = int(instance.weight * 10)  # Scale weights
            weighted_instances.extend([instance] * count)
        
        if weighted_instances:
            instance = weighted_instances[self.current_index % len(weighted_instances)]
            self.current_index = (self.current_index + 1) % len(weighted_instances)
            return instance
        
        return self._round_robin_select(instances)
    
    def _resource_based_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance based on resource utilization."""
        # Calculate composite score (lower is better)
        def resource_score(instance: ServiceInstance) -> float:
            cpu_score = instance.cpu_usage / 100.0
            memory_score = instance.memory_usage / 100.0
            connection_score = instance.active_connections / 100.0  # Normalized
            return (cpu_score + memory_score + connection_score) / 3.0
        
        return min(instances, key=resource_score)
    
    def _health_check_loop(self):
        """Health check loop for service instances."""
        while self.health_check_enabled:
            try:
                with self.lock:
                    for instance in self.instances:
                        # Simplified health check (would implement actual health check)
                        is_healthy = self._perform_health_check(instance)
                        instance.status = "healthy" if is_healthy else "unhealthy"
                        instance.last_health_check = datetime.utcnow()
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(10)
    
    def _perform_health_check(self, instance: ServiceInstance) -> bool:
        """Perform health check on instance."""
        try:
            # Simplified health check - in production, would make HTTP request
            # or use other health check mechanisms
            return instance.cpu_usage < 95 and instance.memory_usage < 95
        except Exception as e:
            logger.error(f"Health check failed for {instance.instance_id}: {e}")
            return False
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.lock:
            total_requests = sum(self.request_counts.values())
            healthy_instances = len([i for i in self.instances if i.status == "healthy"])
            
            return {
                "strategy": self.strategy.value,
                "total_instances": len(self.instances),
                "healthy_instances": healthy_instances,
                "total_requests": total_requests,
                "request_distribution": dict(self.request_counts),
                "instances": [
                    {
                        "id": i.instance_id,
                        "host": i.host,
                        "port": i.port,
                        "status": i.status,
                        "cpu_usage": i.cpu_usage,
                        "memory_usage": i.memory_usage,
                        "active_connections": i.active_connections,
                        "requests_handled": self.request_counts.get(i.instance_id, 0)
                    }
                    for i in self.instances
                ]
            }


class ScalingManager:
    """Main scaling and load balancing coordinator."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.auto_scaler = AutoScaler(self.metrics_collector)
        self.load_balancer = LoadBalancer()
        
        # Container orchestration (if available)
        self.k8s_client = None
        self.docker_client = None
        
        self._initialize_orchestration()
    
    def _initialize_orchestration(self):
        """Initialize container orchestration clients."""
        if K8S_AVAILABLE:
            try:
                if os.path.exists(os.path.expanduser("~/.kube/config")):
                    config.load_kube_config()
                else:
                    config.load_incluster_config()
                self.k8s_client = client.AppsV1Api()
                logger.info("Kubernetes client initialized")
            except Exception as e:
                logger.warning(f"Kubernetes initialization failed: {e}")
        
        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized")
            except Exception as e:
                logger.warning(f"Docker initialization failed: {e}")
    
    def start_all(self):
        """Start all scaling and load balancing components."""
        self.metrics_collector.start_collection()
        self.auto_scaler.start_autoscaling()
        self.load_balancer.start_health_checks()
        
        # Set up scaling callbacks
        self.auto_scaler.set_scale_callbacks(
            scale_up_callback=self._scale_up_instances,
            scale_down_callback=self._scale_down_instances
        )
        
        logger.info("Scaling manager started")
    
    def stop_all(self):
        """Stop all scaling and load balancing components."""
        self.metrics_collector.stop_collection()
        self.auto_scaler.stop_autoscaling()
        self.load_balancer.stop_health_checks()
        logger.info("Scaling manager stopped")
    
    def _scale_up_instances(self, target_count: int) -> bool:
        """Scale up service instances."""
        try:
            current_count = len(self.load_balancer.instances)
            instances_to_add = target_count - current_count
            
            logger.info(f"Scaling up: adding {instances_to_add} instances")
            
            # Use Kubernetes if available
            if self.k8s_client:
                return self._k8s_scale(target_count)
            
            # Use Docker if available
            elif self.docker_client:
                return self._docker_scale_up(instances_to_add)
            
            # Fallback to mock scaling
            else:
                return self._mock_scale_up(instances_to_add)
                
        except Exception as e:
            logger.error(f"Scale up failed: {e}")
            return False
    
    def _scale_down_instances(self, target_count: int) -> bool:
        """Scale down service instances."""
        try:
            current_count = len(self.load_balancer.instances)
            instances_to_remove = current_count - target_count
            
            logger.info(f"Scaling down: removing {instances_to_remove} instances")
            
            # Use Kubernetes if available
            if self.k8s_client:
                return self._k8s_scale(target_count)
            
            # Use Docker if available
            elif self.docker_client:
                return self._docker_scale_down(instances_to_remove)
            
            # Fallback to mock scaling
            else:
                return self._mock_scale_down(instances_to_remove)
                
        except Exception as e:
            logger.error(f"Scale down failed: {e}")
            return False
    
    def _k8s_scale(self, replicas: int) -> bool:
        """Scale using Kubernetes."""
        try:
            deployment_name = os.environ.get("K8S_DEPLOYMENT_NAME", "rlhf-audit-trail")
            namespace = os.environ.get("K8S_NAMESPACE", "default")
            
            # Update deployment replicas
            body = {"spec": {"replicas": replicas}}
            self.k8s_client.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body=body
            )
            
            logger.info(f"Kubernetes deployment {deployment_name} scaled to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes scaling failed: {e}")
            return False
    
    def _docker_scale_up(self, count: int) -> bool:
        """Scale up using Docker."""
        try:
            image_name = os.environ.get("DOCKER_IMAGE", "rlhf-audit-trail:latest")
            
            for i in range(count):
                container = self.docker_client.containers.run(
                    image=image_name,
                    detach=True,
                    name=f"rlhf-audit-trail-{int(time.time())}-{i}",
                    environment={
                        "INSTANCE_ID": f"docker-{int(time.time())}-{i}"
                    }
                )
                
                # Add to load balancer (simplified)
                instance = ServiceInstance(
                    instance_id=container.id[:12],
                    host="localhost",  # Would get actual host
                    port=8000 + i,   # Would get actual port
                    status="healthy",
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    active_connections=0,
                    last_health_check=datetime.utcnow()
                )
                self.load_balancer.add_instance(instance)
            
            return True
            
        except Exception as e:
            logger.error(f"Docker scale up failed: {e}")
            return False
    
    def _docker_scale_down(self, count: int) -> bool:
        """Scale down using Docker."""
        try:
            # Get containers to remove (least loaded)
            instances_to_remove = sorted(
                self.load_balancer.instances,
                key=lambda i: i.active_connections
            )[:count]
            
            for instance in instances_to_remove:
                try:
                    container = self.docker_client.containers.get(instance.instance_id)
                    container.stop()
                    container.remove()
                    self.load_balancer.remove_instance(instance.instance_id)
                except Exception as e:
                    logger.error(f"Failed to remove container {instance.instance_id}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Docker scale down failed: {e}")
            return False
    
    def _mock_scale_up(self, count: int) -> bool:
        """Mock scale up for testing."""
        for i in range(count):
            instance = ServiceInstance(
                instance_id=f"mock-{int(time.time())}-{i}",
                host="localhost",
                port=8000 + len(self.load_balancer.instances) + i,
                status="healthy",
                cpu_usage=20.0,
                memory_usage=30.0,
                active_connections=0,
                last_health_check=datetime.utcnow()
            )
            self.load_balancer.add_instance(instance)
        
        return True
    
    def _mock_scale_down(self, count: int) -> bool:
        """Mock scale down for testing."""
        instances_to_remove = self.load_balancer.instances[-count:]
        for instance in instances_to_remove:
            self.load_balancer.remove_instance(instance.instance_id)
        
        return True
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling system status."""
        return {
            "metrics_collector": {
                "is_collecting": self.metrics_collector.is_collecting,
                "metrics_count": len(self.metrics_collector.metrics_history),
                "current_metrics": self.metrics_collector.get_current_metrics().__dict__
            },
            "auto_scaler": self.auto_scaler.get_scaling_stats(),
            "load_balancer": self.load_balancer.get_load_balancer_stats(),
            "orchestration": {
                "kubernetes_available": self.k8s_client is not None,
                "docker_available": self.docker_client is not None
            }
        }


# Global scaling manager
_global_scaling_manager: Optional[ScalingManager] = None

def get_scaling_manager() -> ScalingManager:
    """Get global scaling manager."""
    global _global_scaling_manager
    if _global_scaling_manager is None:
        _global_scaling_manager = ScalingManager()
    return _global_scaling_manager