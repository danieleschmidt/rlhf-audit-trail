"""Performance optimization features for quantum task planner."""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
import weakref
from collections import defaultdict
import functools
import multiprocessing as mp

from .core import Task, TaskState, QuantumPriority
from .exceptions import QuantumTaskPlannerError


class CacheStrategy(Enum):
    """Caching strategies."""
    LRU = "lru"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: float = 300.0
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    
    # Concurrency
    max_worker_threads: int = 4
    max_worker_processes: int = 2
    enable_parallel_scheduling: bool = True
    enable_parallel_optimization: bool = True
    
    # Resource management
    memory_limit_mb: int = 512
    cpu_usage_threshold: float = 0.8
    auto_scaling_enabled: bool = True
    
    # Batch processing
    batch_size: int = 10
    batch_timeout_seconds: float = 1.0
    enable_batch_optimization: bool = True
    
    # Quantum-specific optimizations
    lazy_decoherence: bool = True
    interference_caching: bool = True
    entanglement_pooling: bool = True


class QuantumTaskCache:
    """High-performance caching system for quantum operations."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize cache with configuration.
        
        Args:
            config: Performance configuration
        """
        self.config = config
        self.logger = logging.getLogger("quantum_planner.cache")
        
        # Cache storage
        if config.cache_strategy == CacheStrategy.LRU:
            from functools import lru_cache
            self._cache: Dict[str, Any] = {}
            self._lru_order: List[str] = []
        else:
            self._cache: Dict[str, Tuple[Any, float]] = {}  # (value, timestamp)
        
        # Cache statistics
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if self.config.cache_strategy == CacheStrategy.LRU:
                if key in self._cache:
                    # Move to end (most recently used)
                    self._lru_order.remove(key)
                    self._lru_order.append(key)
                    self.hit_count += 1
                    return self._cache[key]
            else:
                if key in self._cache:
                    value, timestamp = self._cache[key]
                    
                    # Check TTL
                    if time.time() - timestamp < self.config.cache_ttl_seconds:
                        self.hit_count += 1
                        return value
                    else:
                        # Expired
                        del self._cache[key]
            
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if self.config.cache_strategy == CacheStrategy.LRU:
                # Remove oldest if at capacity
                if len(self._cache) >= self.config.cache_size:
                    oldest_key = self._lru_order.pop(0)
                    del self._cache[oldest_key]
                    self.eviction_count += 1
                
                self._cache[key] = value
                if key not in self._lru_order:
                    self._lru_order.append(key)
            else:
                # TTL or adaptive strategy
                if len(self._cache) >= self.config.cache_size:
                    # Remove expired or oldest entries
                    self._cleanup_expired()
                    
                    if len(self._cache) >= self.config.cache_size:
                        # Remove oldest entry
                        oldest_key = min(self._cache.keys(), 
                                       key=lambda k: self._cache[k][1])
                        del self._cache[oldest_key]
                        self.eviction_count += 1
                
                self._cache[key] = (value, time.time())
    
    def invalidate(self, key: str) -> None:
        """Remove key from cache.
        
        Args:
            key: Key to remove
        """
        with self._lock:
            if self.config.cache_strategy == CacheStrategy.LRU:
                if key in self._cache:
                    del self._cache[key]
                    self._lru_order.remove(key)
            else:
                if key in self._cache:
                    del self._cache[key]
    
    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            if hasattr(self, '_lru_order'):
                self._lru_order.clear()
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp >= self.config.cache_ttl_seconds
        ]
        
        for key in expired_keys:
            del self._cache[key]
            self.eviction_count += len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self.eviction_count,
            "cache_size": len(self._cache),
            "max_size": self.config.cache_size
        }


class BatchProcessor:
    """Batch processing for improved performance."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize batch processor.
        
        Args:
            config: Performance configuration
        """
        self.config = config
        self.logger = logging.getLogger("quantum_planner.batch")
        
        # Batch queues
        self.pending_tasks: List[Task] = []
        self.batch_timers: Dict[str, float] = {}
        
        # Thread safety
        self._lock = threading.Lock()
    
    async def add_task_to_batch(
        self, 
        task: Task, 
        processor: Callable[[List[Task]], Any]
    ) -> Any:
        """Add task to batch for processing.
        
        Args:
            task: Task to add to batch
            processor: Function to process batch
            
        Returns:
            Processing result
        """
        with self._lock:
            self.pending_tasks.append(task)
        
        # Check if batch is ready
        if len(self.pending_tasks) >= self.config.batch_size:
            return await self._process_batch(processor)
        
        # Start timer if this is first task in batch
        batch_key = f"batch_{int(time.time())}"
        if batch_key not in self.batch_timers:
            self.batch_timers[batch_key] = time.time()
            
            # Schedule timeout processing
            asyncio.create_task(self._timeout_batch(processor, batch_key))
        
        return None
    
    async def _process_batch(self, processor: Callable[[List[Task]], Any]) -> Any:
        """Process current batch.
        
        Args:
            processor: Function to process batch
            
        Returns:
            Processing result
        """
        with self._lock:
            if not self.pending_tasks:
                return None
            
            batch = self.pending_tasks.copy()
            self.pending_tasks.clear()
        
        self.logger.debug(f"Processing batch of {len(batch)} tasks")
        
        try:
            return await processor(batch)
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise
    
    async def _timeout_batch(self, processor: Callable, batch_key: str):
        """Process batch after timeout.
        
        Args:
            processor: Function to process batch
            batch_key: Batch identifier
        """
        await asyncio.sleep(self.config.batch_timeout_seconds)
        
        # Check if batch still needs processing
        if (batch_key in self.batch_timers and 
            time.time() - self.batch_timers[batch_key] >= self.config.batch_timeout_seconds):
            
            await self._process_batch(processor)
            del self.batch_timers[batch_key]


class ParallelExecutor:
    """Parallel execution engine for quantum operations."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize parallel executor.
        
        Args:
            config: Performance configuration
        """
        self.config = config
        self.logger = logging.getLogger("quantum_planner.parallel")
        
        # Thread and process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_worker_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_worker_processes)
        
        # Resource monitoring
        self.active_threads = 0
        self.active_processes = 0
        self._resource_lock = threading.Lock()
    
    async def execute_parallel_tasks(
        self, 
        tasks: List[Task],
        executor_func: Callable[[Task], Any],
        use_processes: bool = False
    ) -> List[Any]:
        """Execute tasks in parallel.
        
        Args:
            tasks: Tasks to execute
            executor_func: Function to execute each task
            use_processes: Whether to use process pool instead of thread pool
            
        Returns:
            List of execution results
        """
        if not tasks:
            return []
        
        pool = self.process_pool if use_processes else self.thread_pool
        loop = asyncio.get_event_loop()
        
        # Submit all tasks
        futures = []
        for task in tasks:
            future = loop.run_in_executor(pool, executor_func, task)
            futures.append(future)
        
        # Wait for completion
        try:
            results = await asyncio.gather(*futures, return_exceptions=True)
            
            # Log any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Task {tasks[i].id} failed: {result}")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            raise
    
    async def execute_parallel_scheduling(
        self,
        task_groups: List[List[Task]], 
        scheduler_func: Callable[[List[Task]], Any]
    ) -> List[Any]:
        """Execute scheduling operations in parallel.
        
        Args:
            task_groups: Groups of tasks to schedule
            scheduler_func: Scheduling function
            
        Returns:
            List of scheduling results
        """
        if not self.config.enable_parallel_scheduling:
            # Fall back to sequential processing
            results = []
            for group in task_groups:
                result = await scheduler_func(group)
                results.append(result)
            return results
        
        # Process groups in parallel
        loop = asyncio.get_event_loop()
        futures = []
        
        for group in task_groups:
            future = loop.run_in_executor(self.thread_pool, scheduler_func, group)
            futures.append(future)
        
        try:
            return await asyncio.gather(*futures)
        except Exception as e:
            self.logger.error(f"Parallel scheduling failed: {e}")
            raise
    
    def shutdown(self):
        """Shutdown executor pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class QuantumOptimizationEngine:
    """Optimized quantum operations for better performance."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize optimization engine.
        
        Args:
            config: Performance configuration
        """
        self.config = config
        self.logger = logging.getLogger("quantum_planner.optimization")
        
        # Caches for expensive operations
        self.interference_cache = QuantumTaskCache(config) if config.interference_caching else None
        self.entanglement_cache = QuantumTaskCache(config) if config.entanglement_pooling else None
        
        # Lazy evaluation tracking
        self.decoherence_pending: Set[str] = set()
        self.last_decoherence_check = time.time()
    
    def calculate_interference_optimized(self, task1: Task, task2: Task) -> float:
        """Calculate quantum interference with caching.
        
        Args:
            task1: First task
            task2: Second task
            
        Returns:
            Interference value
        """
        if not self.interference_cache:
            return self._calculate_interference_direct(task1, task2)
        
        # Create cache key
        cache_key = f"interference_{min(task1.id, task2.id)}_{max(task1.id, task2.id)}"
        
        # Check cache
        cached_result = self.interference_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Calculate and cache
        result = self._calculate_interference_direct(task1, task2)
        self.interference_cache.put(cache_key, result)
        
        return result
    
    def _calculate_interference_direct(self, task1: Task, task2: Task) -> float:
        """Direct interference calculation."""
        import math
        
        # Calculate phase difference
        phase_diff = abs(task1.phase - task2.phase)
        if phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff
        
        # Interference pattern
        interference = math.cos(phase_diff)
        
        # Scale by amplitudes
        return task1.amplitude * task2.amplitude * interference
    
    def process_decoherence_lazy(self, tasks: List[Task]) -> List[Task]:
        """Process decoherence using lazy evaluation.
        
        Args:
            tasks: Tasks to check for decoherence
            
        Returns:
            Tasks that have decohered
        """
        if not self.config.lazy_decoherence:
            return self._process_decoherence_immediate(tasks)
        
        current_time = time.time()
        
        # Only check decoherence periodically
        if current_time - self.last_decoherence_check < 10.0:  # 10 second interval
            return []
        
        self.last_decoherence_check = current_time
        
        decohered_tasks = []
        
        for task in tasks:
            if task.id in self.decoherence_pending or not task.is_coherent:
                task.decohere()
                decohered_tasks.append(task)
                self.decoherence_pending.discard(task.id)
            elif not task.is_coherent:
                # Schedule for next decoherence check
                self.decoherence_pending.add(task.id)
        
        return decohered_tasks
    
    def _process_decoherence_immediate(self, tasks: List[Task]) -> List[Task]:
        """Process decoherence immediately."""
        decohered_tasks = []
        
        for task in tasks:
            if not task.is_coherent:
                task.decohere()
                decohered_tasks.append(task)
        
        return decohered_tasks
    
    def optimize_entanglement_creation(
        self, 
        task1: Task, 
        task2: Task,
        strength_threshold: float = 0.3
    ) -> bool:
        """Optimize entanglement creation with pooling.
        
        Args:
            task1: First task
            task2: Second task
            strength_threshold: Minimum strength for entanglement
            
        Returns:
            True if entanglement was created
        """
        if not self.entanglement_cache:
            return self._create_entanglement_direct(task1, task2, strength_threshold)
        
        # Check if similar entanglement exists in pool
        cache_key = f"entanglement_pool_{task1.priority.value}_{task2.priority.value}"
        cached_entanglement = self.entanglement_cache.get(cache_key)
        
        if cached_entanglement:
            # Reuse existing entanglement pattern
            task1.phase = cached_entanglement["phase1"]
            task2.phase = cached_entanglement["phase2"]
            task1.entangle_with(task2)
            return True
        
        # Create new entanglement
        success = self._create_entanglement_direct(task1, task2, strength_threshold)
        
        if success:
            # Cache the entanglement pattern
            self.entanglement_cache.put(cache_key, {
                "phase1": task1.phase,
                "phase2": task2.phase,
                "strength": self.calculate_interference_optimized(task1, task2)
            })
        
        return success
    
    def _create_entanglement_direct(
        self, 
        task1: Task, 
        task2: Task,
        strength_threshold: float
    ) -> bool:
        """Direct entanglement creation."""
        # Calculate interference strength
        interference = self.calculate_interference_optimized(task1, task2)
        
        if abs(interference) < strength_threshold:
            return False
        
        # Create entanglement
        task1.entangle_with(task2)
        return True


class ResourceMonitor:
    """Monitor and manage system resources."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize resource monitor.
        
        Args:
            config: Performance configuration
        """
        self.config = config
        self.logger = logging.getLogger("quantum_planner.resources")
        
        # Resource tracking
        self.memory_usage_mb = 0.0
        self.cpu_usage_percent = 0.0
        self.active_tasks = 0
        
        # Auto-scaling
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.last_scale_action = time.time()
        self.min_scale_interval = 60.0  # 1 minute between scaling actions
    
    def check_resource_constraints(self) -> bool:
        """Check if system is within resource constraints.
        
        Returns:
            True if within constraints
        """
        # Estimate current resource usage
        self._update_resource_estimates()
        
        # Check memory limit
        if self.memory_usage_mb > self.config.memory_limit_mb:
            self.logger.warning(f"Memory usage ({self.memory_usage_mb:.1f}MB) "
                              f"exceeds limit ({self.config.memory_limit_mb}MB)")
            return False
        
        # Check CPU threshold
        if self.cpu_usage_percent > self.config.cpu_usage_threshold * 100:
            self.logger.warning(f"CPU usage ({self.cpu_usage_percent:.1f}%) "
                              f"exceeds threshold ({self.config.cpu_usage_threshold * 100:.1f}%)")
            return False
        
        return True
    
    def _update_resource_estimates(self):
        """Update resource usage estimates."""
        # Simple estimation based on active tasks
        # In production, would use actual system monitoring
        self.memory_usage_mb = 50 + (self.active_tasks * 2)  # Base + 2MB per active task
        self.cpu_usage_percent = min(100, self.active_tasks * 10)  # 10% per active task
    
    def suggest_scaling_action(self) -> Optional[str]:
        """Suggest scaling action based on resource usage.
        
        Returns:
            Scaling action suggestion or None
        """
        if not self.config.auto_scaling_enabled:
            return None
        
        current_time = time.time()
        if current_time - self.last_scale_action < self.min_scale_interval:
            return None  # Too soon for another scaling action
        
        self._update_resource_estimates()
        
        if self.cpu_usage_percent > self.scale_up_threshold * 100:
            return "scale_up"
        elif self.cpu_usage_percent < self.scale_down_threshold * 100:
            return "scale_down"
        
        return None
    
    def apply_scaling_action(self, action: str) -> bool:
        """Apply suggested scaling action.
        
        Args:
            action: Scaling action to apply
            
        Returns:
            True if action was applied
        """
        if action == "scale_up":
            # Increase resource allocation
            self.config.max_worker_threads = min(8, self.config.max_worker_threads + 1)
            self.config.batch_size = min(20, self.config.batch_size + 2)
            self.logger.info(f"Scaled up: threads={self.config.max_worker_threads}, "
                           f"batch_size={self.config.batch_size}")
            
        elif action == "scale_down":
            # Decrease resource allocation
            self.config.max_worker_threads = max(2, self.config.max_worker_threads - 1)
            self.config.batch_size = max(5, self.config.batch_size - 2)
            self.logger.info(f"Scaled down: threads={self.config.max_worker_threads}, "
                           f"batch_size={self.config.batch_size}")
        
        self.last_scale_action = time.time()
        return True


class PerformanceManager:
    """Main performance management system."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """Initialize performance manager.
        
        Args:
            config: Performance configuration
        """
        self.config = config or PerformanceConfig()
        self.logger = logging.getLogger("quantum_planner.performance")
        
        # Components
        self.cache = QuantumTaskCache(self.config)
        self.batch_processor = BatchProcessor(self.config)
        self.parallel_executor = ParallelExecutor(self.config)
        self.optimization_engine = QuantumOptimizationEngine(self.config)
        self.resource_monitor = ResourceMonitor(self.config)
        
        # Performance metrics
        self.start_time = time.time()
        self.operation_count = 0
        self.total_execution_time = 0.0
    
    async def optimize_task_batch(self, tasks: List[Task]) -> List[Task]:
        """Optimize a batch of tasks for better performance.
        
        Args:
            tasks: Tasks to optimize
            
        Returns:
            Optimized tasks
        """
        if not tasks:
            return []
        
        start_time = time.time()
        
        # Process decoherence lazily
        decohered_tasks = self.optimization_engine.process_decoherence_lazy(tasks)
        
        # Optimize entanglements in parallel if enabled
        if self.config.enable_parallel_optimization and len(tasks) > 4:
            optimized_tasks = await self._optimize_tasks_parallel(tasks)
        else:
            optimized_tasks = self._optimize_tasks_sequential(tasks)
        
        # Update performance metrics
        execution_time = time.time() - start_time
        self.total_execution_time += execution_time
        self.operation_count += 1
        
        self.logger.debug(f"Optimized {len(tasks)} tasks in {execution_time:.3f}s")
        
        return optimized_tasks
    
    async def _optimize_tasks_parallel(self, tasks: List[Task]) -> List[Task]:
        """Optimize tasks using parallel processing."""
        # Split tasks into groups for parallel processing
        group_size = max(1, len(tasks) // self.config.max_worker_threads)
        task_groups = [tasks[i:i + group_size] for i in range(0, len(tasks), group_size)]
        
        # Process groups in parallel
        results = await self.parallel_executor.execute_parallel_scheduling(
            task_groups,
            self._optimize_task_group
        )
        
        # Flatten results
        optimized_tasks = []
        for group_result in results:
            if isinstance(group_result, list):
                optimized_tasks.extend(group_result)
        
        return optimized_tasks
    
    def _optimize_task_group(self, task_group: List[Task]) -> List[Task]:
        """Optimize a group of tasks."""
        return self._optimize_tasks_sequential(task_group)
    
    def _optimize_tasks_sequential(self, tasks: List[Task]) -> List[Task]:
        """Optimize tasks sequentially."""
        optimized_tasks = []
        
        for task in tasks:
            # Apply quantum optimizations
            if task.is_coherent:
                # Optimize quantum properties
                if task.amplitude < 0.1:
                    task.amplitude = 0.1  # Minimum viable amplitude
                
                # Normalize phase
                import math
                task.phase = task.phase % (2 * math.pi)
            
            optimized_tasks.append(task)
        
        return optimized_tasks
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        uptime = time.time() - self.start_time
        avg_execution_time = (self.total_execution_time / self.operation_count 
                            if self.operation_count > 0 else 0.0)
        
        return {
            "uptime_seconds": uptime,
            "operation_count": self.operation_count,
            "average_execution_time": avg_execution_time,
            "cache_stats": self.cache.get_stats(),
            "resource_usage": {
                "memory_mb": self.resource_monitor.memory_usage_mb,
                "cpu_percent": self.resource_monitor.cpu_usage_percent,
                "active_tasks": self.resource_monitor.active_tasks
            },
            "configuration": {
                "caching_enabled": self.config.enable_caching,
                "parallel_scheduling": self.config.enable_parallel_scheduling,
                "batch_processing": self.config.enable_batch_optimization,
                "auto_scaling": self.config.auto_scaling_enabled
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform performance health check.
        
        Returns:
            Health check results
        """
        health_status = {
            "overall_status": "healthy",
            "issues": [],
            "recommendations": []
        }
        
        # Check resource constraints
        if not self.resource_monitor.check_resource_constraints():
            health_status["overall_status"] = "warning"
            health_status["issues"].append("Resource constraints exceeded")
            health_status["recommendations"].append("Consider scaling up resources")
        
        # Check cache performance
        cache_stats = self.cache.get_stats()
        if cache_stats["hit_rate"] < 0.5:
            health_status["issues"].append(f"Low cache hit rate: {cache_stats['hit_rate']:.1%}")
            health_status["recommendations"].append("Review caching strategy")
        
        # Check auto-scaling
        scaling_action = self.resource_monitor.suggest_scaling_action()
        if scaling_action:
            health_status["recommendations"].append(f"Consider {scaling_action}")
        
        return health_status
    
    def shutdown(self):
        """Shutdown performance manager."""
        self.parallel_executor.shutdown()
        self.logger.info("Performance manager shutdown complete")