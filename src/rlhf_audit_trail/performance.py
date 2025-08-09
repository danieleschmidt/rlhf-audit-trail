"""Performance optimization and caching system for RLHF audit trail."""

import asyncio
import hashlib
import logging
import pickle
import time
import threading
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
import json

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .exceptions import AuditTrailError
from .monitoring import time_operation

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    enabled: bool = True
    ttl_seconds: int = 3600  # 1 hour
    max_memory_items: int = 1000
    redis_url: Optional[str] = None
    redis_db: int = 0
    compression_enabled: bool = True
    async_writes: bool = True


@dataclass 
class PerformanceConfig:
    """Performance optimization configuration."""
    cache_config: CacheConfig
    batch_size: int = 100
    max_concurrent_operations: int = 50
    connection_pool_size: int = 20
    async_processing: bool = True
    metrics_collection: bool = True


class MemoryCache:
    """In-memory LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expires': 0
        }
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
                
            # Check TTL
            if self._is_expired(key):
                self._remove(key)
                self.stats['expires'] += 1
                self.stats['misses'] += 1
                return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.stats['hits'] += 1
            return value
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        with self._lock:
            # Remove if already exists
            if key in self.cache:
                self.cache.pop(key)
            
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = time.time()
            
            # Evict oldest if over max size
            if len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                self._remove(oldest_key)
                self.stats['evictions'] += 1
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        with self._lock:
            if key in self.cache:
                self._remove(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache item is expired."""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl_seconds
    
    def _remove(self, key: str) -> None:
        """Remove item from cache."""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                **self.stats
            }


class RedisCache:
    """Redis-based distributed cache."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 db: int = 0, ttl_seconds: int = 3600):
        if not REDIS_AVAILABLE:
            raise AuditTrailError("Redis not available. Install with: pip install redis")
        
        self.ttl_seconds = ttl_seconds
        self.stats = defaultdict(int)
        
        try:
            self.redis_client = redis.from_url(redis_url, db=db, decode_responses=False)
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise AuditTrailError(f"Redis connection failed: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache."""
        try:
            value = self.redis_client.get(key)
            if value is None:
                self.stats['misses'] += 1
                return None
            
            self.stats['hits'] += 1
            return pickle.loads(value)
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats['errors'] += 1
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """Set item in Redis cache."""
        try:
            serialized = pickle.dumps(value)
            result = self.redis_client.setex(key, self.ttl_seconds, serialized)
            self.stats['sets'] += 1
            return result
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self.stats['errors'] += 1
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from Redis cache."""
        try:
            result = self.redis_client.delete(key)
            self.stats['deletes'] += 1
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            self.stats['errors'] += 1
            return False
    
    def clear(self) -> None:
        """Clear all items from cache."""
        try:
            self.redis_client.flushdb()
            self.stats['clears'] += 1
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            self.stats['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        try:
            info = self.redis_client.info('memory')
            memory_usage = info.get('used_memory_human', 'Unknown')
        except:
            memory_usage = 'Unknown'
        
        return {
            'hit_rate': hit_rate,
            'memory_usage': memory_usage,
            **dict(self.stats)
        }


class MultiLevelCache:
    """Multi-level cache with memory and Redis backends."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Always have memory cache
        self.memory_cache = MemoryCache(
            max_size=config.max_memory_items,
            ttl_seconds=config.ttl_seconds
        )
        
        # Optional Redis cache
        self.redis_cache = None
        if config.redis_url:
            try:
                self.redis_cache = RedisCache(
                    redis_url=config.redis_url,
                    db=config.redis_db,
                    ttl_seconds=config.ttl_seconds
                )
            except Exception as e:
                logger.warning(f"Failed to setup Redis cache: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache (memory first, then Redis)."""
        if not self.config.enabled:
            return None
        
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache
        if self.redis_cache:
            value = self.redis_cache.get(key)
            if value is not None:
                # Populate memory cache
                self.memory_cache.set(key, value)
                return value
        
        return None
    
    async def set(self, key: str, value: Any) -> bool:
        """Set item in cache."""
        if not self.config.enabled:
            return True
        
        success = True
        
        # Set in memory cache
        self.memory_cache.set(key, value)
        
        # Set in Redis cache if available
        if self.redis_cache:
            if self.config.async_writes:
                # Async write to Redis
                asyncio.create_task(self._async_redis_set(key, value))
            else:
                success &= self.redis_cache.set(key, value)
        
        return success
    
    async def _async_redis_set(self, key: str, value: Any):
        """Asynchronous Redis write."""
        try:
            self.redis_cache.set(key, value)
        except Exception as e:
            logger.error(f"Async Redis write failed: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        if not self.config.enabled:
            return True
        
        success = True
        success &= self.memory_cache.delete(key)
        
        if self.redis_cache:
            success &= self.redis_cache.delete(key)
        
        return success
    
    async def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        if self.redis_cache:
            self.redis_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'memory_cache': self.memory_cache.get_stats()
        }
        
        if self.redis_cache:
            stats['redis_cache'] = self.redis_cache.get_stats()
        
        return stats


class BatchProcessor:
    """Batch processing for improved performance."""
    
    def __init__(self, batch_size: int = 100, max_concurrent: int = 10):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def process_batch(self, items: List[Any], 
                           processor: Callable[[List[Any]], Any]) -> List[Any]:
        """Process items in batches."""
        if not items:
            return []
        
        # Split into batches
        batches = [
            items[i:i + self.batch_size] 
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Process batches concurrently
        tasks = [
            self._process_single_batch(batch, processor)
            for batch in batches
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        flattened = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
                continue
            if isinstance(result, list):
                flattened.extend(result)
            else:
                flattened.append(result)
        
        return flattened
    
    async def _process_single_batch(self, batch: List[Any], 
                                   processor: Callable[[List[Any]], Any]) -> Any:
        """Process a single batch with concurrency control."""
        async with self.semaphore:
            try:
                return await processor(batch)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                raise


class ConnectionPool:
    """Connection pooling for database and external services."""
    
    def __init__(self, factory: Callable, max_size: int = 20, 
                 max_overflow: int = 10, timeout: float = 30.0):
        self.factory = factory
        self.max_size = max_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        
        self._pool = asyncio.Queue(maxsize=max_size)
        self._overflow = 0
        self._created = 0
        self._stats = {
            'created': 0,
            'reused': 0,
            'timeouts': 0,
            'errors': 0
        }
        
        # Pre-populate pool
        asyncio.create_task(self._initialize_pool())
    
    async def _initialize_pool(self):
        """Initialize connection pool."""
        for _ in range(self.max_size // 2):  # Start with half capacity
            try:
                connection = await self.factory()
                await self._pool.put(connection)
                self._created += 1
                self._stats['created'] += 1
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
                break
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool."""
        connection = None
        is_overflow = False
        
        try:
            # Try to get from pool
            try:
                connection = await asyncio.wait_for(
                    self._pool.get(), 
                    timeout=self.timeout
                )
                self._stats['reused'] += 1
            except asyncio.TimeoutError:
                self._stats['timeouts'] += 1
                
                # Create overflow connection if allowed
                if self._overflow < self.max_overflow:
                    connection = await self.factory()
                    self._overflow += 1
                    is_overflow = True
                    self._stats['created'] += 1
                else:
                    raise AuditTrailError("Connection pool exhausted")
            
            yield connection
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Connection pool error: {e}")
            raise
        finally:
            # Return connection to pool
            if connection:
                if is_overflow:
                    # Close overflow connections
                    self._overflow -= 1
                    await self._close_connection(connection)
                else:
                    # Return to pool
                    try:
                        await self._pool.put(connection)
                    except asyncio.QueueFull:
                        # Pool full, close connection
                        await self._close_connection(connection)
    
    async def _close_connection(self, connection):
        """Close a connection."""
        try:
            if hasattr(connection, 'close'):
                await connection.close()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'pool_size': self._pool.qsize(),
            'overflow_connections': self._overflow,
            'total_created': self._created,
            **self._stats
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        
        # Initialize components
        self.cache = MultiLevelCache(config.cache_config)
        self.batch_processor = BatchProcessor(
            batch_size=config.batch_size,
            max_concurrent_operations=config.max_concurrent_operations
        )
        
        # Performance metrics
        self.metrics = {
            'cache_operations': 0,
            'batch_operations': 0,
            'optimization_saves_ms': 0
        }
        
        logger.info("Performance optimizer initialized")
    
    @asynccontextmanager
    async def cached_operation(self, cache_key: str, ttl_override: Optional[int] = None):
        """Context manager for cached operations."""
        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            self.metrics['cache_operations'] += 1
            yield cached_result
            return
        
        # Cache miss - compute and cache result
        start_time = time.time()
        result = None
        
        try:
            yield lambda r: setattr(result, 'value', r) or r  # Capture result
        finally:
            if result is not None and hasattr(result, 'value'):
                # Cache the result
                await self.cache.set(cache_key, result.value)
                
                # Track performance gain
                elapsed = (time.time() - start_time) * 1000
                self.metrics['optimization_saves_ms'] += elapsed
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': kwargs,
            'timestamp': int(time.time() / 300)  # 5-minute buckets
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def batch_process(self, items: List[Any], 
                           processor: Callable[[List[Any]], Any]) -> List[Any]:
        """Process items in optimized batches."""
        with time_operation('batch_process'):
            result = await self.batch_processor.process_batch(items, processor)
            self.metrics['batch_operations'] += 1
            return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'cache_stats': self.cache.get_stats(),
            'optimization_metrics': self.metrics,
            'config': {
                'batch_size': self.config.batch_size,
                'max_concurrent': self.config.max_concurrent_operations,
                'cache_enabled': self.config.cache_config.enabled
            }
        }
    
    async def warm_cache(self, warmup_data: Dict[str, Any]):
        """Warm up cache with frequently accessed data."""
        logger.info("Starting cache warmup")
        
        for key, value in warmup_data.items():
            await self.cache.set(f"warmup_{key}", value)
        
        logger.info(f"Cache warmed up with {len(warmup_data)} items")
    
    async def optimize_storage_access(self, access_pattern: str, 
                                     data_size: int) -> Dict[str, Any]:
        """Optimize storage access based on patterns."""
        optimizations = {
            'use_compression': data_size > 1024 * 1024,  # 1MB threshold
            'use_async_writes': access_pattern == 'write_heavy',
            'batch_reads': access_pattern == 'read_heavy',
            'cache_duration': 3600 if access_pattern == 'frequent' else 300
        }
        
        return optimizations
    
    async def health_check(self) -> Dict[str, Any]:
        """Performance system health check."""
        cache_stats = self.cache.get_stats()
        
        # Determine health status
        memory_cache_hit_rate = cache_stats.get('memory_cache', {}).get('hit_rate', 0)
        
        status = 'healthy'
        if memory_cache_hit_rate < 50:
            status = 'degraded'
        if memory_cache_hit_rate < 20:
            status = 'unhealthy'
        
        return {
            'status': status,
            'cache_hit_rate': memory_cache_hit_rate,
            'cache_stats': cache_stats,
            'metrics': self.metrics,
            'timestamp': datetime.utcnow()
        }


# Global performance optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None

def get_performance_optimizer(config: Optional[PerformanceConfig] = None) -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer(config or PerformanceConfig(
            cache_config=CacheConfig()
        ))
    return _global_optimizer

def cached_operation(cache_key: str):
    """Decorator for caching operation results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            key = f"{func.__name__}_{cache_key}_{optimizer.cache_key(*args, **kwargs)}"
            
            async with optimizer.cached_operation(key) as cache_setter:
                if callable(cache_setter):
                    # Cache miss - compute result
                    result = await func(*args, **kwargs)
                    cache_setter(result)
                    return result
                else:
                    # Cache hit
                    return cache_setter
        return wrapper
    return decorator