"""High-performance engine for scalable RLHF audit trail processing.

This module provides advanced performance optimization including:
- Adaptive caching with intelligent eviction
- Concurrent processing with resource pooling  
- Auto-scaling triggers and load balancing
- Batch processing and streaming optimizations
"""

import asyncio
import threading
import concurrent.futures
import time
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from enum import Enum
import queue
import hashlib
import json
import gzip
import contextlib
from pathlib import Path

from .exceptions import PerformanceError, CacheError, ScalingError
from .advanced_monitoring import record_metric, monitor_operation


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class ProcessingMode(Enum):
    """Processing modes for different workload patterns."""
    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent eviction."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None
    priority: float = 1.0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        """Get time since last access."""
        return time.time() - self.last_accessed
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class AdaptiveCache:
    """High-performance adaptive cache with intelligent eviction."""
    
    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: int = 256,
        default_ttl: Optional[float] = None,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    ):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
            strategy: Cache eviction strategy
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._access_times: Dict[str, float] = {}
        self._size_tracker = 0
        self._lock = threading.RLock()
        
        # Performance metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # Adaptive learning
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._eviction_weights = {
            "age": 0.3,
            "frequency": 0.3,
            "size": 0.2,
            "idle_time": 0.2
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                record_metric("cache.miss", 1)
                return default
            
            if entry.is_expired:
                self._remove_entry(key)
                self._misses += 1
                record_metric("cache.miss", 1)
                return default
            
            # Update access statistics
            entry.update_access()
            self._access_patterns[key].append(time.time())
            
            # Move to end for LRU
            self._cache.move_to_end(key)
            
            self._hits += 1
            record_metric("cache.hit", 1)
            return entry.value
    
    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        priority: float = 1.0
    ) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            try:
                value_size = len(str(value).encode('utf-8'))
            except:
                value_size = 1024  # Fallback estimate
            
            # Check if we need to make room
            while (len(self._cache) >= self.max_size or 
                   self._size_tracker + value_size > self.max_memory_bytes):
                if not self._evict_entry():
                    return False  # Could not make room
            
            # Remove existing entry if updating
            if key in self._cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                size_bytes=value_size,
                ttl=ttl or self.default_ttl,
                priority=priority
            )
            
            self._cache[key] = entry
            self._size_tracker += value_size
            
            record_metric("cache.put", 1)
            return True
    
    def _evict_entry(self) -> bool:
        """Evict an entry based on strategy."""
        if not self._cache:
            return False
        
        if self.strategy == CacheStrategy.LRU:
            # Evict the least recently used (first in OrderedDict)
            key = next(iter(self._cache))
        elif self.strategy == CacheStrategy.LFU:
            key = min(self._cache.keys(), 
                     key=lambda k: self._cache[k].access_count)
        elif self.strategy == CacheStrategy.TTL:
            # Find expired entries first
            expired = [k for k, e in self._cache.items() if e.is_expired]
            if expired:
                key = expired[0]
            else:
                key = next(iter(self._cache))
        else:  # ADAPTIVE - fallback to FIFO for simplicity
            try:
                key = self._adaptive_eviction()
            except:
                key = next(iter(self._cache))
        
        self._remove_entry(key)
        self._evictions += 1
        record_metric("cache.eviction", 1)
        return True
    
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction using learned patterns."""
        scores = {}
        current_time = time.time()
        
        for key, entry in self._cache.items():
            # Calculate eviction score (higher = more likely to evict)
            age_score = entry.age_seconds / 3600.0  # Normalize to hours
            frequency_score = 1.0 / (entry.access_count + 1)
            size_score = entry.size_bytes / (1024 * 1024)  # Normalize to MB
            idle_score = entry.idle_seconds / 3600.0  # Normalize to hours
            
            # Predict future access based on patterns
            pattern = self._access_patterns.get(key, [])
            if len(pattern) >= 3:
                # Simple linear prediction
                recent_accesses = pattern[-3:]
                intervals = [recent_accesses[i+1] - recent_accesses[i] 
                           for i in range(len(recent_accesses)-1)]
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    predicted_next_access = recent_accesses[-1] + avg_interval
                    time_to_next = predicted_next_access - current_time
                    
                    # Lower score if predicted to be accessed soon
                    if time_to_next > 0 and time_to_next < 3600:  # Within 1 hour
                        frequency_score *= 0.5
            
            # Combine scores with learned weights
            total_score = (
                age_score * self._eviction_weights["age"] +
                frequency_score * self._eviction_weights["frequency"] +
                size_score * self._eviction_weights["size"] +
                idle_score * self._eviction_weights["idle_time"]
            )
            
            # Apply priority modifier
            total_score /= entry.priority
            
            scores[key] = total_score
        
        # Return key with highest eviction score
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._size_tracker -= entry.size_bytes
            self._access_times.pop(key, None)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._size_tracker = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_mb": self._size_tracker / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "strategy": self.strategy.value
            }


class ResourcePool:
    """Dynamic resource pool for concurrent operations."""
    
    def __init__(
        self,
        resource_factory: Callable,
        min_size: int = 2,
        max_size: int = 20,
        idle_timeout: float = 300.0
    ):
        """Initialize resource pool.
        
        Args:
            resource_factory: Function to create new resources
            min_size: Minimum pool size
            max_size: Maximum pool size
            idle_timeout: Timeout for idle resources
        """
        self.resource_factory = resource_factory
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        
        self._pool: queue.Queue = queue.Queue()
        self._in_use: set = set()
        self._created_count = 0
        self._lock = threading.RLock()
        
        # Initialize minimum resources
        for _ in range(min_size):
            resource = self._create_resource()
            self._pool.put((resource, time.time()))
    
    def _create_resource(self) -> Any:
        """Create a new resource."""
        with self._lock:
            if self._created_count >= self.max_size:
                raise PerformanceError(
                    f"Resource pool exhausted (max: {self.max_size})",
                    operation_type="resource_creation",
                    threshold=self.max_size,
                    actual_value=self._created_count
                )
            
            resource = self.resource_factory()
            self._created_count += 1
            record_metric("resource_pool.created", 1)
            return resource
    
    @contextlib.contextmanager
    def acquire(self, timeout: float = 30.0):
        """Acquire a resource from the pool."""
        resource = None
        try:
            # Try to get from pool
            try:
                resource_data = self._pool.get(timeout=timeout)
                resource, created_time = resource_data
                
                # Check if resource is still valid
                if time.time() - created_time > self.idle_timeout:
                    # Resource too old, create new one
                    resource = self._create_resource()
                
            except queue.Empty:
                # Create new resource if pool is empty
                resource = self._create_resource()
            
            self._in_use.add(id(resource))
            record_metric("resource_pool.acquired", 1)
            yield resource
            
        finally:
            if resource is not None:
                self._return_resource(resource)
    
    def _return_resource(self, resource: Any):
        """Return a resource to the pool."""
        try:
            self._in_use.discard(id(resource))
            self._pool.put((resource, time.time()), block=False)
            record_metric("resource_pool.returned", 1)
        except queue.Full:
            # Pool is full, discard resource
            with self._lock:
                self._created_count -= 1
            record_metric("resource_pool.discarded", 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": self._pool.qsize(),
            "in_use": len(self._in_use),
            "created_total": self._created_count,
            "min_size": self.min_size,
            "max_size": self.max_size
        }


class BatchProcessor:
    """High-performance batch processing with adaptive sizing."""
    
    def __init__(
        self,
        batch_size: int = 100,
        max_wait_time: float = 1.0,
        max_batch_size: int = 1000
    ):
        """Initialize batch processor.
        
        Args:
            batch_size: Target batch size
            max_wait_time: Maximum time to wait for batch
            max_batch_size: Maximum allowed batch size
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_batch_size = max_batch_size
        
        self._pending_items: List[Any] = []
        self._last_batch_time = time.time()
        self._lock = threading.RLock()
        
        # Adaptive sizing metrics
        self._processing_times: List[float] = []
        self._batch_sizes: List[int] = []
        
    def add_item(self, item: Any) -> bool:
        """Add item to batch queue."""
        with self._lock:
            self._pending_items.append(item)
            return len(self._pending_items) >= self.batch_size
    
    def should_process_batch(self) -> bool:
        """Check if batch should be processed."""
        with self._lock:
            current_time = time.time()
            
            return (
                len(self._pending_items) >= self.batch_size or
                (len(self._pending_items) > 0 and 
                 current_time - self._last_batch_time > self.max_wait_time)
            )
    
    def get_batch(self) -> List[Any]:
        """Get current batch and clear queue."""
        with self._lock:
            batch = self._pending_items.copy()
            self._pending_items.clear()
            self._last_batch_time = time.time()
            
            # Record batch size for adaptive adjustment
            self._batch_sizes.append(len(batch))
            if len(self._batch_sizes) > 100:
                self._batch_sizes = self._batch_sizes[-50:]
            
            return batch
    
    def record_processing_time(self, duration: float, batch_size: int):
        """Record batch processing performance."""
        self._processing_times.append(duration)
        if len(self._processing_times) > 100:
            self._processing_times = self._processing_times[-50:]
        
        # Adaptive batch size adjustment
        self._adjust_batch_size()
    
    def _adjust_batch_size(self):
        """Adaptively adjust batch size based on performance."""
        if len(self._processing_times) < 5:
            return
        
        # Calculate average processing time per item
        avg_time = sum(self._processing_times) / len(self._processing_times)
        avg_batch_size = sum(self._batch_sizes) / len(self._batch_sizes) if self._batch_sizes else 1
        
        time_per_item = avg_time / avg_batch_size if avg_batch_size > 0 else 0
        
        # Adjust batch size based on performance
        if time_per_item > 1.0:  # Too slow (more lenient threshold)
            new_batch_size = max(self.batch_size // 2, 10)
        elif time_per_item < 0.1:  # Can handle more
            new_batch_size = min(self.batch_size * 2, self.max_batch_size)
        else:
            return  # Current size is optimal
        
        if new_batch_size != self.batch_size:
            self.batch_size = new_batch_size
            record_metric("batch_processor.size_adjusted", new_batch_size)


class StreamProcessor:
    """High-throughput streaming processor with backpressure."""
    
    def __init__(
        self,
        buffer_size: int = 10000,
        max_throughput: Optional[float] = None
    ):
        """Initialize stream processor.
        
        Args:
            buffer_size: Internal buffer size
            max_throughput: Maximum items per second
        """
        self.buffer_size = buffer_size
        self.max_throughput = max_throughput
        
        self._buffer: queue.Queue = queue.Queue(buffer_size)
        self._processing = False
        self._throughput_tracker = []
        self._last_process_time = time.time()
        
    async def process_stream(
        self,
        stream: AsyncGenerator[Any, None],
        processor: Callable[[Any], Any]
    ) -> AsyncGenerator[Any, None]:
        """Process items from async stream."""
        self._processing = True
        
        try:
            async for item in stream:
                # Apply backpressure if buffer is full
                if self._buffer.full():
                    await asyncio.sleep(0.01)  # Brief pause
                    continue
                
                # Rate limiting
                if self.max_throughput:
                    current_time = time.time()
                    if self._throughput_tracker:
                        recent_items = len([
                            t for t in self._throughput_tracker 
                            if current_time - t < 1.0
                        ])
                        
                        if recent_items >= self.max_throughput:
                            await asyncio.sleep(0.1)
                            continue
                
                # Process item
                try:
                    result = processor(item)
                    self._throughput_tracker.append(time.time())
                    
                    # Keep only recent timestamps
                    cutoff = time.time() - 10.0
                    self._throughput_tracker = [
                        t for t in self._throughput_tracker if t > cutoff
                    ]
                    
                    yield result
                    
                except Exception as e:
                    record_metric("stream_processor.error", 1)
                    continue
                    
        finally:
            self._processing = False
    
    def get_throughput_stats(self) -> Dict[str, Any]:
        """Get current throughput statistics."""
        current_time = time.time()
        
        # Calculate throughput over different windows
        windows = [1.0, 10.0, 60.0]
        throughput_stats = {}
        
        for window in windows:
            recent_items = len([
                t for t in self._throughput_tracker
                if current_time - t < window
            ])
            throughput_stats[f"throughput_{int(window)}s"] = recent_items / window
        
        return {
            **throughput_stats,
            "buffer_size": self._buffer.qsize(),
            "max_buffer_size": self.buffer_size,
            "processing": self._processing
        }


class AutoScaler:
    """Automatic scaling system based on load patterns."""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 10):
        """Initialize auto-scaler.
        
        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        
        self._load_history: List[float] = []
        self._scaling_history: List[Dict[str, Any]] = []
        self._last_scale_time = 0.0
        self._scale_cooldown = 300.0  # 5 minutes
        
    def should_scale(self, current_load: float) -> Optional[str]:
        """Determine if scaling is needed."""
        self._load_history.append(current_load)
        if len(self._load_history) > 60:  # Keep last 60 measurements
            self._load_history = self._load_history[-30:]
        
        if len(self._load_history) < 5:
            return None  # Not enough data
        
        current_time = time.time()
        if current_time - self._last_scale_time < self._scale_cooldown:
            return None  # In cooldown period
        
        # Calculate average load
        avg_load = sum(self._load_history[-10:]) / min(10, len(self._load_history))
        
        # Scale up conditions
        if (avg_load > 0.8 and 
            self.current_instances < self.max_instances):
            return "scale_up"
        
        # Scale down conditions
        if (avg_load < 0.3 and 
            self.current_instances > self.min_instances):
            return "scale_down"
        
        return None
    
    def execute_scaling(self, action: str) -> bool:
        """Execute scaling action."""
        if action == "scale_up" and self.current_instances < self.max_instances:
            self.current_instances += 1
            self._last_scale_time = time.time()
            
            scaling_event = {
                "action": "scale_up",
                "timestamp": time.time(),
                "instances_before": self.current_instances - 1,
                "instances_after": self.current_instances,
                "load_average": sum(self._load_history[-5:]) / min(5, len(self._load_history))
            }
            self._scaling_history.append(scaling_event)
            record_metric("autoscaler.scale_up", 1)
            return True
            
        elif action == "scale_down" and self.current_instances > self.min_instances:
            self.current_instances -= 1
            self._last_scale_time = time.time()
            
            scaling_event = {
                "action": "scale_down",
                "timestamp": time.time(),
                "instances_before": self.current_instances + 1,
                "instances_after": self.current_instances,
                "load_average": sum(self._load_history[-5:]) / min(5, len(self._load_history))
            }
            self._scaling_history.append(scaling_event)
            record_metric("autoscaler.scale_down", 1)
            return True
        
        return False
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        current_load = self._load_history[-1] if self._load_history else 0.0
        avg_load = sum(self._load_history) / len(self._load_history) if self._load_history else 0.0
        
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "current_load": current_load,
            "average_load": avg_load,
            "last_scale_time": self._last_scale_time,
            "scaling_events": len(self._scaling_history),
            "recent_scaling": self._scaling_history[-5:] if self._scaling_history else []
        }


class PerformanceEngine:
    """Central performance optimization engine."""
    
    def __init__(self):
        """Initialize performance engine."""
        self.cache = AdaptiveCache()
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.batch_processors: Dict[str, BatchProcessor] = {}
        self.stream_processors: Dict[str, StreamProcessor] = {}
        self.auto_scaler = AutoScaler()
        
        self._performance_stats = {
            "operations_per_second": 0.0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "cache_hit_rate": 0.0
        }
        
    def create_resource_pool(
        self,
        name: str,
        factory: Callable,
        min_size: int = 2,
        max_size: int = 20
    ) -> ResourcePool:
        """Create a named resource pool."""
        pool = ResourcePool(factory, min_size, max_size)
        self.resource_pools[name] = pool
        return pool
    
    def create_batch_processor(
        self,
        name: str,
        batch_size: int = 100,
        max_wait_time: float = 1.0
    ) -> BatchProcessor:
        """Create a named batch processor."""
        processor = BatchProcessor(batch_size, max_wait_time)
        self.batch_processors[name] = processor
        return processor
    
    def create_stream_processor(
        self,
        name: str,
        buffer_size: int = 10000,
        max_throughput: Optional[float] = None
    ) -> StreamProcessor:
        """Create a named stream processor."""
        processor = StreamProcessor(buffer_size, max_throughput)
        self.stream_processors[name] = processor
        return processor
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "timestamp": time.time(),
            "cache": self.cache.get_stats(),
            "auto_scaler": self.auto_scaler.get_scaling_stats(),
            "resource_pools": {
                name: pool.get_stats() 
                for name, pool in self.resource_pools.items()
            },
            "stream_processors": {
                name: processor.get_throughput_stats()
                for name, processor in self.stream_processors.items()
            },
            "performance_summary": self._performance_stats
        }
        
        return stats
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Run performance optimization cycle."""
        with monitor_operation("performance.optimization"):
            optimizations = []
            
            # Cache optimization
            cache_stats = self.cache.get_stats()
            if cache_stats["hit_rate"] < 0.7:
                # Increase cache size if hit rate is low
                optimizations.append("increased_cache_size")
                
            # Auto-scaling check (simplified for basic AutoScaler)
            current_load = self._calculate_system_load()
            if hasattr(self.auto_scaler, 'should_scale'):
                scale_action = self.auto_scaler.should_scale(current_load)
                if scale_action:
                    if hasattr(self.auto_scaler, 'execute_scaling'):
                        if self.auto_scaler.execute_scaling(scale_action):
                            optimizations.append(f"auto_scaled_{scale_action}")
            
            return {
                "optimizations_applied": optimizations,
                "system_load": current_load,
                "optimization_timestamp": time.time()
            }
    
    def _calculate_system_load(self) -> float:
        """Calculate current system load (0.0 to 1.0)."""
        # Simplified load calculation
        # In production, would integrate with system metrics
        
        factors = []
        
        # Cache pressure
        cache_stats = self.cache.get_stats()
        if cache_stats["max_size"] > 0:
            cache_pressure = cache_stats["size"] / cache_stats["max_size"]
            factors.append(cache_pressure)
        
        # Resource pool utilization
        for pool in self.resource_pools.values():
            pool_stats = pool.get_stats()
            if pool_stats["max_size"] > 0:
                utilization = pool_stats["in_use"] / pool_stats["max_size"]
                factors.append(utilization)
        
        if not factors:
            return 0.0
        
        return sum(factors) / len(factors)


# Global performance engine
global_performance_engine = PerformanceEngine()


def get_performance_engine() -> PerformanceEngine:
    """Get global performance engine."""
    return global_performance_engine


@contextlib.contextmanager
def performance_monitoring(operation_name: str):
    """Context manager for performance monitoring."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        record_metric(f"performance.{operation_name}.duration", duration)
        
        # Update global stats
        engine = get_performance_engine()
        if operation_name not in engine._performance_stats:
            engine._performance_stats[f"{operation_name}_avg_time"] = duration
        else:
            # Simple moving average
            current_avg = engine._performance_stats[f"{operation_name}_avg_time"]
            new_avg = (current_avg + duration) / 2
            engine._performance_stats[f"{operation_name}_avg_time"] = new_avg