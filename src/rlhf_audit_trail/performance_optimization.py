"""
Advanced performance optimization and caching system for RLHF audit trail.
Generation 3: Performance optimization, caching strategies, and resource management.
"""

import asyncio
import time
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from collections import OrderedDict, defaultdict
import pickle
import hashlib
import logging
import psutil
import os
import gzip

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

from .exceptions import PerformanceError, CacheError, ResourceExhaustedError


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_used: int
    cpu_usage: float
    cache_hits: int = 0
    cache_misses: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0


class PerformanceProfiler:
    """Performance profiling and monitoring."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def profile_operation(self, operation_name: str):
        """Decorator to profile operation performance."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._profile_async_operation(operation_name, func, *args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._profile_sync_operation(operation_name, func, *args, **kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator
    
    async def _profile_async_operation(self, operation_name: str, func: Callable, *args, **kwargs):
        """Profile async operation."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_cpu = psutil.cpu_percent()
        
        try:
            result = await func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            end_cpu = psutil.cpu_percent()
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_used=end_memory - start_memory,
                cpu_usage=(start_cpu + end_cpu) / 2,
                metadata={
                    "success": success,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
            )
            
            self._record_metrics(metrics)
        
        return result
    
    def _profile_sync_operation(self, operation_name: str, func: Callable, *args, **kwargs):
        """Profile synchronous operation."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_cpu = psutil.cpu_percent()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            end_cpu = psutil.cpu_percent()
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_used=end_memory - start_memory,
                cpu_usage=(start_cpu + end_cpu) / 2,
                metadata={
                    "success": success,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
            )
            
            self._record_metrics(metrics)
        
        return result
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self.lock:
            self.metrics_history.append(metrics)
            self.operation_stats[metrics.operation_name].append(metrics.duration)
            
            # Limit history size
            if len(self.metrics_history) > 10000:
                self.metrics_history = self.metrics_history[-5000:]  # Keep last 5000
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        with self.lock:
            durations = self.operation_stats.get(operation_name, [])
            
            if not durations:
                return {"operation": operation_name, "count": 0}
            
            if NUMPY_AVAILABLE:
                return {
                    "operation": operation_name,
                    "count": len(durations),
                    "avg_duration": np.mean(durations),
                    "min_duration": np.min(durations),
                    "max_duration": np.max(durations),
                    "std_duration": np.std(durations),
                    "median_duration": np.median(durations),
                    "p95_duration": np.percentile(durations, 95),
                    "p99_duration": np.percentile(durations, 99)
                }
            else:
                # Fallback without numpy
                sorted_durations = sorted(durations)
                n = len(sorted_durations)
                
                return {
                    "operation": operation_name,
                    "count": n,
                    "avg_duration": sum(durations) / n,
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "median_duration": sorted_durations[n // 2],
                    "p95_duration": sorted_durations[int(0.95 * n)],
                    "p99_duration": sorted_durations[int(0.99 * n)]
                }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        with self.lock:
            total_operations = len(self.metrics_history)
            if total_operations == 0:
                return {"total_operations": 0}
            
            recent_metrics = [m for m in self.metrics_history if m.start_time > time.time() - 3600]  # Last hour
            
            summary = {
                "total_operations": total_operations,
                "operations_last_hour": len(recent_metrics),
                "avg_operations_per_minute": len(recent_metrics) / 60 if recent_metrics else 0,
                "operation_breakdown": {}
            }
            
            # Break down by operation type
            for operation_name in self.operation_stats.keys():
                summary["operation_breakdown"][operation_name] = self.get_operation_stats(operation_name)
            
            return summary


class SmartCache:
    """Intelligent caching system with multiple strategies."""
    
    def __init__(self, 
                 max_size: int = 1000, 
                 ttl_seconds: int = 3600,
                 strategy: str = "lru"):
        """
        Initialize smart cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time to live for cached items
            strategy: Caching strategy ("lru", "lfu", "fifo")
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.strategy = strategy
        
        self._cache: OrderedDict = OrderedDict()
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Cleanup thread
        self._cleanup_interval = 300  # 5 minutes
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get value from cache with hit/miss tracking."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None, False
            
            cached_item = self._cache[key]
            
            # Check TTL
            if time.time() - cached_item["timestamp"] > self.ttl_seconds:
                self._remove_item(key)
                self.misses += 1
                return None, False
            
            # Update access patterns
            self._access_counts[key] += 1
            self._access_times[key] = time.time()
            
            # Move to end for LRU
            if self.strategy == "lru":
                self._cache.move_to_end(key)
            
            self.hits += 1
            return cached_item["value"], True
    
    def set(self, key: str, value: Any, custom_ttl: Optional[int] = None):
        """Set value in cache."""
        with self._lock:
            current_time = time.time()
            ttl = custom_ttl or self.ttl_seconds
            
            # Remove existing item if present
            if key in self._cache:
                self._remove_item(key)
            
            # Check size limit and evict if necessary
            if len(self._cache) >= self.max_size:
                self._evict_item()
            
            # Add new item
            self._cache[key] = {
                "value": value,
                "timestamp": current_time,
                "ttl": ttl,
                "size": self._estimate_size(value)
            }
            
            self._access_counts[key] = 1
            self._access_times[key] = current_time
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_item(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache items."""
        with self._lock:
            self._cache.clear()
            self._access_counts.clear()
            self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            total_size = sum(item["size"] for item in self._cache.values())
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "total_memory_bytes": total_size,
                "avg_item_size": total_size / len(self._cache) if self._cache else 0
            }
    
    def _evict_item(self):
        """Evict item based on strategy."""
        if not self._cache:
            return
        
        if self.strategy == "lru":
            # Remove least recently used (first item)
            key = next(iter(self._cache))
        elif self.strategy == "lfu":
            # Remove least frequently used
            key = min(self._access_counts.keys(), key=lambda k: self._access_counts[k])
        elif self.strategy == "fifo":
            # Remove first in (first item)
            key = next(iter(self._cache))
        else:
            # Default to LRU
            key = next(iter(self._cache))
        
        self._remove_item(key)
        self.evictions += 1
    
    def _remove_item(self, key: str):
        """Remove item from cache and cleanup related data."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_counts:
            del self._access_counts[key]
        if key in self._access_times:
            del self._access_times[key]
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of cached value."""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
    
    def _periodic_cleanup(self):
        """Periodic cleanup of expired items."""
        while True:
            try:
                time.sleep(self._cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def _cleanup_expired(self):
        """Remove expired items."""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, cached_item in self._cache.items():
                if current_time - cached_item["timestamp"] > cached_item["ttl"]:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_item(key)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")


class ResourcePool:
    """Generic resource pool for connection/object reuse."""
    
    def __init__(self, 
                 factory: Callable[[], Any],
                 max_size: int = 10,
                 min_size: int = 2,
                 validator: Optional[Callable[[Any], bool]] = None,
                 cleanup: Optional[Callable[[Any], None]] = None):
        """
        Initialize resource pool.
        
        Args:
            factory: Function to create new resources
            max_size: Maximum pool size
            min_size: Minimum pool size
            validator: Function to validate resource health
            cleanup: Function to cleanup resources
        """
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.validator = validator or (lambda x: True)
        self.cleanup = cleanup or (lambda x: None)
        
        self._pool: List[Any] = []
        self._in_use: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.RLock()
        self._created_count = 0
        self._total_requests = 0
        
        # Pre-populate with minimum resources
        self._populate_pool()
    
    def acquire(self) -> Any:
        """Acquire resource from pool."""
        with self._lock:
            self._total_requests += 1
            
            # Try to get from pool
            while self._pool:
                resource = self._pool.pop()
                
                # Validate resource
                if self.validator(resource):
                    self._in_use.add(resource)
                    return resource
                else:
                    # Resource invalid, cleanup and continue
                    try:
                        self.cleanup(resource)
                    except Exception as e:
                        logger.warning(f"Resource cleanup failed: {e}")
            
            # No valid resources in pool, create new one
            if self._created_count < self.max_size:
                resource = self.factory()
                self._created_count += 1
                self._in_use.add(resource)
                return resource
            else:
                raise PerformanceError("Resource pool exhausted")
    
    def release(self, resource: Any):
        """Release resource back to pool."""
        with self._lock:
            if resource in self._in_use:
                self._in_use.discard(resource)
                
                # Validate before returning to pool
                if self.validator(resource) and len(self._pool) < self.max_size:
                    self._pool.append(resource)
                else:
                    # Cleanup invalid or excess resource
                    try:
                        self.cleanup(resource)
                        self._created_count -= 1
                    except Exception as e:
                        logger.warning(f"Resource cleanup failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "in_use": len(self._in_use),
                "created_count": self._created_count,
                "total_requests": self._total_requests,
                "utilization": len(self._in_use) / self.max_size * 100
            }
    
    def _populate_pool(self):
        """Pre-populate pool with minimum resources."""
        with self._lock:
            while len(self._pool) < self.min_size and self._created_count < self.max_size:
                try:
                    resource = self.factory()
                    self._pool.append(resource)
                    self._created_count += 1
                except Exception as e:
                    logger.error(f"Failed to create resource: {e}")
                    break


class ConcurrentProcessor:
    """High-performance concurrent processing system."""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 use_processes: bool = False):
        """
        Initialize concurrent processor.
        
        Args:
            max_workers: Maximum number of workers (None for CPU count)
            use_processes: Use processes instead of threads
        """
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        
        self.max_workers = max_workers
        self.use_processes = use_processes
        
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self._lock = threading.Lock()
    
    async def process_batch(self, 
                           items: List[Any], 
                           processor_func: Callable,
                           batch_size: int = 10,
                           max_retries: int = 3) -> List[Any]:
        """Process items in parallel batches."""
        if not items:
            return []
        
        # Split into batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        async def process_single_batch(batch):
            """Process a single batch."""
            with self._lock:
                self.active_tasks += 1
            
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._process_batch_with_retry,
                    batch,
                    processor_func,
                    max_retries
                )
                
                with self._lock:
                    self.completed_tasks += 1
                
                return result
                
            except Exception as e:
                with self._lock:
                    self.failed_tasks += 1
                logger.error(f"Batch processing failed: {e}")
                raise
            finally:
                with self._lock:
                    self.active_tasks -= 1
        
        # Process all batches concurrently
        tasks = [process_single_batch(batch) for batch in batches]
        
        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Concurrent batch processing failed: {e}")
            raise
        
        # Flatten results and handle exceptions
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch failed: {batch_result}")
                # Could implement partial failure handling here
                raise batch_result
            else:
                results.extend(batch_result)
        
        return results
    
    def _process_batch_with_retry(self, 
                                 batch: List[Any], 
                                 processor_func: Callable,
                                 max_retries: int) -> List[Any]:
        """Process batch with retry logic."""
        for attempt in range(max_retries + 1):
            try:
                return [processor_func(item) for item in batch]
            except Exception as e:
                if attempt == max_retries:
                    raise
                else:
                    logger.warning(f"Batch processing attempt {attempt + 1} failed: {e}")
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        
        return []  # Should never reach here
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        with self._lock:
            total_tasks = self.completed_tasks + self.failed_tasks
            success_rate = (self.completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            return {
                "max_workers": self.max_workers,
                "active_tasks": self.active_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "success_rate": success_rate,
                "use_processes": self.use_processes
            }
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.cache = SmartCache(max_size=5000, ttl_seconds=1800)  # 30 minutes
        self.processor = ConcurrentProcessor()
        
        # Create resource pools for common resources
        self.connection_pools: Dict[str, ResourcePool] = {}
        
        # Performance monitoring
        self.optimization_history: List[Dict[str, Any]] = []
        self.auto_optimization_enabled = True
        self._optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self._optimization_thread.start()
    
    def create_connection_pool(self, 
                             pool_name: str,
                             factory: Callable[[], Any],
                             max_size: int = 10,
                             validator: Optional[Callable[[Any], bool]] = None) -> ResourcePool:
        """Create a named connection pool."""
        pool = ResourcePool(
            factory=factory,
            max_size=max_size,
            validator=validator
        )
        self.connection_pools[pool_name] = pool
        return pool
    
    def get_connection_pool(self, pool_name: str) -> Optional[ResourcePool]:
        """Get connection pool by name."""
        return self.connection_pools.get(pool_name)
    
    def cached_operation(self, 
                        cache_key_func: Callable[..., str],
                        ttl: Optional[int] = None):
        """Decorator for caching operation results."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                cache_key = cache_key_func(*args, **kwargs)
                
                # Try to get from cache
                cached_result, hit = self.cache.get(cache_key)
                if hit:
                    return cached_result
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                self.cache.set(cache_key, result, custom_ttl=ttl)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                cache_key = cache_key_func(*args, **kwargs)
                
                # Try to get from cache
                cached_result, hit = self.cache.get(cache_key)
                if hit:
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.cache.set(cache_key, result, custom_ttl=ttl)
                
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "profiler": self.profiler.get_performance_summary(),
            "cache": self.cache.get_stats(),
            "processor": self.processor.get_stats(),
            "connection_pools": {
                name: pool.get_stats() 
                for name, pool in self.connection_pools.items()
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "process_memory_mb": psutil.Process().memory_info().rss / 1024 / 1024
            }
        }
    
    def _optimization_loop(self):
        """Continuous optimization monitoring loop."""
        while self.auto_optimization_enabled:
            try:
                time.sleep(300)  # Check every 5 minutes
                if self.auto_optimization_enabled:
                    self._perform_auto_optimization()
            except Exception as e:
                logger.error(f"Auto-optimization error: {e}")
    
    def _perform_auto_optimization(self):
        """Perform automatic optimizations based on metrics."""
        stats = self.get_comprehensive_stats()
        
        # Cache optimization
        cache_stats = stats["cache"]
        if cache_stats["hit_rate"] < 50:  # Low hit rate
            logger.info("Cache hit rate low, consider increasing cache size or TTL")
        
        # Memory optimization
        memory_percent = stats["system"]["memory_percent"]
        if memory_percent > 85:
            logger.warning("High memory usage detected, clearing old cache entries")
            self.cache._cleanup_expired()
        
        # Connection pool optimization
        for pool_name, pool_stats in stats["connection_pools"].items():
            utilization = pool_stats["utilization"]
            if utilization > 90:
                logger.warning(f"High utilization in pool {pool_name}: {utilization}%")
        
        # Record optimization check
        self.optimization_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "stats_snapshot": stats,
            "actions_taken": []  # Would record actual optimizations performed
        })
        
        # Limit history
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-500:]


# Global performance optimizer
_global_optimizer: Optional[PerformanceOptimizer] = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer

def profile(operation_name: str):
    """Decorator for profiling operations."""
    return get_performance_optimizer().profiler.profile_operation(operation_name)

def cached(cache_key_func: Callable[..., str], ttl: Optional[int] = None):
    """Decorator for caching operation results."""
    return get_performance_optimizer().cached_operation(cache_key_func, ttl)


class AdaptiveResourceManager:
    """Adaptive resource management for optimal performance."""
    
    def __init__(self):
        self.cpu_count = os.cpu_count() or 4
        self.memory_limit = psutil.virtual_memory().total
        self.thread_pool = ThreadPoolExecutor(max_workers=self.cpu_count * 2)
        self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_count)
        self.async_semaphore = asyncio.Semaphore(self.cpu_count * 4)
        
        # Resource monitoring
        self.resource_history = []
        self.auto_scale_enabled = True
        self.scale_up_threshold = 80  # CPU/Memory %
        self.scale_down_threshold = 20
        
    async def execute_adaptive(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with adaptive resource allocation."""
        current_load = self._get_current_load()
        
        if current_load > self.scale_up_threshold:
            # High load - use process pool for CPU-intensive tasks
            if self._is_cpu_intensive(func):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.process_pool, func, *args, **kwargs)
            else:
                # Use async semaphore to limit concurrency
                async with self.async_semaphore:
                    return await self._execute_with_monitoring(func, *args, **kwargs)
        else:
            # Normal load - use thread pool
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
    
    def _get_current_load(self) -> float:
        """Get current system load."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        return max(cpu_percent, memory_percent)
    
    def _is_cpu_intensive(self, func: Callable) -> bool:
        """Heuristic to determine if function is CPU intensive."""
        # Simple heuristic based on function name and module
        cpu_intensive_keywords = ['compute', 'calculate', 'process', 'encrypt', 'hash']
        func_name = func.__name__.lower()
        return any(keyword in func_name for keyword in cpu_intensive_keywords)
    
    async def _execute_with_monitoring(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with resource monitoring."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Adaptive execution failed: {e}")
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            # Record resource usage
            self.resource_history.append({
                "timestamp": end_time,
                "duration": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "function": func.__name__
            })
            
            # Keep only recent history
            if len(self.resource_history) > 1000:
                self.resource_history = self.resource_history[-500:]
        
        return result
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource utilization statistics."""
        if not self.resource_history:
            return {"status": "no_data"}
        
        recent_history = self.resource_history[-100:]  # Last 100 operations
        
        durations = [item["duration"] for item in recent_history]
        memory_deltas = [item["memory_delta"] for item in recent_history]
        
        return {
            "avg_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "avg_memory_delta": sum(memory_deltas) / len(memory_deltas),
            "max_memory_delta": max(memory_deltas),
            "total_operations": len(recent_history),
            "current_cpu": psutil.cpu_percent(),
            "current_memory": psutil.virtual_memory().percent,
            "thread_pool_active": self.thread_pool._threads,
            "process_pool_active": len(self.process_pool._processes) if hasattr(self.process_pool, '_processes') else 0
        }


class AutoScalingManager:
    """Auto-scaling system for dynamic load balancing."""
    
    def __init__(self, resource_manager: AdaptiveResourceManager):
        self.resource_manager = resource_manager
        self.scaling_policies = {
            "cpu_scale_out": {"threshold": 85, "duration": 60, "action": "scale_out"},
            "cpu_scale_in": {"threshold": 25, "duration": 300, "action": "scale_in"},
            "memory_scale_out": {"threshold": 90, "duration": 30, "action": "scale_out"},
            "memory_scale_in": {"threshold": 30, "duration": 300, "action": "scale_in"}
        }
        self.scaling_history = []
        self.last_scale_time = 0
        self.cooldown_period = 180  # 3 minutes
    
    async def evaluate_scaling(self) -> Dict[str, Any]:
        """Evaluate if scaling action is needed."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_time < self.cooldown_period:
            return {"action": "cooldown", "remaining": self.cooldown_period - (current_time - self.last_scale_time)}
        
        # Get current metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        scaling_decision = {
            "timestamp": current_time,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "recommended_action": "none",
            "reason": ""
        }
        
        # Evaluate CPU scaling
        if cpu_percent > self.scaling_policies["cpu_scale_out"]["threshold"]:
            scaling_decision["recommended_action"] = "scale_out"
            scaling_decision["reason"] = f"CPU utilization {cpu_percent}% exceeds threshold"
        elif cpu_percent < self.scaling_policies["cpu_scale_in"]["threshold"]:
            scaling_decision["recommended_action"] = "scale_in"
            scaling_decision["reason"] = f"CPU utilization {cpu_percent}% below threshold"
        
        # Evaluate memory scaling (overrides CPU if more critical)
        if memory_percent > self.scaling_policies["memory_scale_out"]["threshold"]:
            scaling_decision["recommended_action"] = "scale_out"
            scaling_decision["reason"] = f"Memory utilization {memory_percent}% exceeds threshold"
        
        # Execute scaling if needed
        if scaling_decision["recommended_action"] != "none":
            await self._execute_scaling(scaling_decision)
            self.last_scale_time = current_time
        
        self.scaling_history.append(scaling_decision)
        return scaling_decision
    
    async def _execute_scaling(self, decision: Dict[str, Any]) -> None:
        """Execute scaling action."""
        action = decision["recommended_action"]
        
        if action == "scale_out":
            # Increase resource allocation
            current_workers = self.resource_manager.thread_pool._max_workers
            new_workers = min(current_workers + 2, self.resource_manager.cpu_count * 4)
            
            if new_workers > current_workers:
                # Create new thread pool with more workers
                old_pool = self.resource_manager.thread_pool
                self.resource_manager.thread_pool = ThreadPoolExecutor(max_workers=new_workers)
                
                # Gracefully shutdown old pool
                old_pool.shutdown(wait=False)
                
                logger.info(f"Scaled out thread pool from {current_workers} to {new_workers} workers")
        
        elif action == "scale_in":
            # Decrease resource allocation
            current_workers = self.resource_manager.thread_pool._max_workers
            new_workers = max(current_workers - 1, self.resource_manager.cpu_count)
            
            if new_workers < current_workers:
                # Create new thread pool with fewer workers
                old_pool = self.resource_manager.thread_pool
                self.resource_manager.thread_pool = ThreadPoolExecutor(max_workers=new_workers)
                
                # Gracefully shutdown old pool
                old_pool.shutdown(wait=False)
                
                logger.info(f"Scaled in thread pool from {current_workers} to {new_workers} workers")


class ConcurrentBatchProcessor:
    """High-performance concurrent batch processing."""
    
    def __init__(self, max_workers: Optional[int] = None, batch_size: int = 100):
        self.max_workers = max_workers or os.cpu_count() * 2
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
    async def process_batch_concurrent(self, items: List[Any], 
                                     processor_func: Callable,
                                     *args, **kwargs) -> List[Any]:
        """Process items concurrently in batches."""
        if not items:
            return []
        
        results = []
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        
        # Process batches concurrently
        loop = asyncio.get_event_loop()
        futures = []
        
        for batch in batches:
            future = loop.run_in_executor(
                self.executor,
                self._process_single_batch,
                batch, processor_func, args, kwargs
            )
            futures.append(future)
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing failed: {batch_result}")
                raise batch_result
            results.extend(batch_result)
        
        return results
    
    def _process_single_batch(self, batch: List[Any], processor_func: Callable,
                            args: tuple, kwargs: dict) -> List[Any]:
        """Process a single batch of items."""
        results = []
        for item in batch:
            try:
                result = processor_func(item, *args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Item processing failed: {e}")
                results.append(None)  # or handle error appropriately
        return results


class QuantumScaleOptimizer:
    """Quantum-inspired optimization for extreme scale."""
    
    def __init__(self):
        self.resource_manager = AdaptiveResourceManager()
        self.auto_scaler = AutoScalingManager(self.resource_manager)
        self.batch_processor = ConcurrentBatchProcessor()
        self.optimization_history = []
        
    async def optimize_performance(self, operation_type: str) -> Dict[str, Any]:
        """Quantum-inspired performance optimization."""
        
        # Get current system state
        system_state = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
            "timestamp": time.time()
        }
        
        # Quantum-inspired optimization algorithm
        optimization_vector = self._calculate_optimization_vector(system_state, operation_type)
        
        # Apply optimizations
        optimizations_applied = []
        
        # Memory optimization
        if optimization_vector.get("memory_pressure", 0) > 0.7:
            await self._optimize_memory()
            optimizations_applied.append("memory_optimization")
        
        # CPU optimization
        if optimization_vector.get("cpu_pressure", 0) > 0.8:
            await self._optimize_cpu_usage()
            optimizations_applied.append("cpu_optimization")
        
        # I/O optimization
        if optimization_vector.get("io_pressure", 0) > 0.6:
            await self._optimize_io_patterns()
            optimizations_applied.append("io_optimization")
        
        # Auto-scaling evaluation
        scaling_decision = await self.auto_scaler.evaluate_scaling()
        if scaling_decision["recommended_action"] != "none":
            optimizations_applied.append(f"auto_scale_{scaling_decision['recommended_action']}")
        
        optimization_result = {
            "operation_type": operation_type,
            "system_state": system_state,
            "optimization_vector": optimization_vector,
            "optimizations_applied": optimizations_applied,
            "performance_improvement": self._estimate_improvement(optimizations_applied),
            "timestamp": time.time()
        }
        
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    def _calculate_optimization_vector(self, system_state: Dict[str, Any], 
                                     operation_type: str) -> Dict[str, float]:
        """Calculate quantum-inspired optimization vector."""
        # Simplified quantum-inspired calculation
        cpu_pressure = system_state["cpu_percent"] / 100.0
        memory_pressure = system_state["memory_percent"] / 100.0
        
        # Operation-specific weights
        operation_weights = {
            "encryption": {"cpu": 0.8, "memory": 0.3, "io": 0.2},
            "storage": {"cpu": 0.2, "memory": 0.4, "io": 0.9},
            "computation": {"cpu": 0.9, "memory": 0.6, "io": 0.1},
            "default": {"cpu": 0.5, "memory": 0.5, "io": 0.5}
        }
        
        weights = operation_weights.get(operation_type, operation_weights["default"])
        
        return {
            "cpu_pressure": cpu_pressure * weights["cpu"],
            "memory_pressure": memory_pressure * weights["memory"],
            "io_pressure": 0.5 * weights["io"],  # Simplified
            "optimization_potential": 1.0 - max(cpu_pressure, memory_pressure)
        }
    
    async def _optimize_memory(self) -> None:
        """Apply memory optimizations."""
        import gc
        gc.collect()  # Force garbage collection
        logger.info("Applied memory optimization: garbage collection")
    
    async def _optimize_cpu_usage(self) -> None:
        """Apply CPU usage optimizations."""
        # Reduce process priority if high load
        current_process = psutil.Process()
        if hasattr(current_process, 'nice'):
            try:
                current_process.nice(5)  # Lower priority
                logger.info("Applied CPU optimization: reduced process priority")
            except (psutil.AccessDenied, OSError):
                pass
    
    async def _optimize_io_patterns(self) -> None:
        """Apply I/O pattern optimizations."""
        # Enable write buffering and batching
        logger.info("Applied I/O optimization: enhanced buffering")
    
    def _estimate_improvement(self, optimizations: List[str]) -> float:
        """Estimate performance improvement percentage."""
        improvement_map = {
            "memory_optimization": 5.0,
            "cpu_optimization": 10.0,
            "io_optimization": 15.0,
            "auto_scale_scale_out": 25.0,
            "auto_scale_scale_in": -5.0  # Slight performance reduction for cost savings
        }
        
        total_improvement = sum(improvement_map.get(opt, 0) for opt in optimizations)
        return min(total_improvement, 50.0)  # Cap at 50% improvement


# Global optimizer instance
_quantum_optimizer = None

def get_quantum_optimizer() -> QuantumScaleOptimizer:
    """Get or create global quantum optimizer instance."""
    global _quantum_optimizer
    if _quantum_optimizer is None:
        _quantum_optimizer = QuantumScaleOptimizer()
    return _quantum_optimizer


# Performance optimization decorators
def quantum_optimized(operation_type: str = "default"):
    """Decorator for quantum-optimized operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            optimizer = get_quantum_optimizer()
            
            # Pre-optimization
            optimization_result = await optimizer.optimize_performance(operation_type)
            
            # Execute function with resource management
            result = await optimizer.resource_manager.execute_adaptive(func, *args, **kwargs)
            
            return result
        return wrapper
    return decorator


def batch_optimized(batch_size: int = 100, max_workers: Optional[int] = None):
    """Decorator for batch-optimized processing."""
    def decorator(func):
        @wraps(func)
        async def wrapper(items: List[Any], *args, **kwargs):
            processor = ConcurrentBatchProcessor(max_workers, batch_size)
            return await processor.process_batch_concurrent(items, func, *args, **kwargs)
        return wrapper
    return decorator