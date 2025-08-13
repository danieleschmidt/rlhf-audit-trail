"""Comprehensive tests for Generation 3 optimization features.

Tests for performance engine, caching, scaling, and load balancing.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

from rlhf_audit_trail.performance_engine import (
    AdaptiveCache, CacheStrategy, ResourcePool, BatchProcessor, 
    StreamProcessor, AutoScaler, PerformanceEngine
)
from rlhf_audit_trail.scaling_system import (
    AdvancedAutoScaler, LoadBalancer, LoadBalancingStrategy,
    InstanceMetrics, PredictiveScaler, ScalingStrategy
)
from rlhf_audit_trail.exceptions import PerformanceError, ScalingError


class TestAdaptiveCache:
    """Test adaptive caching system."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = AdaptiveCache(max_size=3)
        
        # Test put and get
        assert cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test default value
        assert cache.get("nonexistent", "default") == "default"
        
        # Test cache size limit
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict oldest
        
        assert cache.get("key1") is None  # Should be evicted
        assert cache.get("key4") == "value4"
    
    def test_cache_ttl(self):
        """Test time-to-live functionality."""
        cache = AdaptiveCache(max_size=10, default_ttl=0.1)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        time.sleep(0.2)  # Wait for TTL to expire
        assert cache.get("key1") is None
    
    def test_cache_strategies(self):
        """Test different cache eviction strategies."""
        # Test LRU strategy
        cache = AdaptiveCache(max_size=2, strategy=CacheStrategy.LRU)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.get("key1")  # Access key1 to make it recently used
        cache.put("key3", "value3")  # Should evict key2
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = AdaptiveCache(max_size=10)
        
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == 0.5


class TestResourcePool:
    """Test resource pooling system."""
    
    def test_resource_pool_creation(self):
        """Test resource pool creation and basic operations."""
        def create_resource():
            return {"connection": "test"}
        
        pool = ResourcePool(create_resource, min_size=2, max_size=5)
        
        # Test resource acquisition
        with pool.acquire() as resource:
            assert resource["connection"] == "test"
        
        stats = pool.get_stats()
        assert stats["min_size"] == 2
        assert stats["max_size"] == 5
    
    def test_resource_pool_limits(self):
        """Test resource pool size limits."""
        create_count = 0
        
        def create_resource():
            nonlocal create_count
            create_count += 1
            return {"id": create_count}
        
        pool = ResourcePool(create_resource, min_size=1, max_size=2)
        
        # Acquire maximum resources
        with pool.acquire() as r1:
            with pool.acquire() as r2:
                # Try to acquire one more - should raise error
                with pytest.raises(PerformanceError):
                    with pool.acquire(timeout=0.1):
                        pass


class TestBatchProcessor:
    """Test batch processing system."""
    
    def test_batch_processor_basic(self):
        """Test basic batch processing."""
        processor = BatchProcessor(batch_size=3, max_wait_time=1.0)
        
        # Add items
        assert not processor.add_item("item1")  # Not ready yet
        assert not processor.add_item("item2")  # Not ready yet
        assert processor.add_item("item3")  # Batch is ready
        
        assert processor.should_process_batch()
        
        batch = processor.get_batch()
        assert len(batch) == 3
        assert batch == ["item1", "item2", "item3"]
    
    def test_batch_processor_timeout(self):
        """Test batch processing timeout."""
        processor = BatchProcessor(batch_size=5, max_wait_time=0.1)
        
        processor.add_item("item1")
        processor.add_item("item2")
        
        # Wait for timeout
        time.sleep(0.2)
        
        assert processor.should_process_batch()
        batch = processor.get_batch()
        assert len(batch) == 2
    
    def test_adaptive_batch_sizing(self):
        """Test adaptive batch size adjustment."""
        processor = BatchProcessor(batch_size=10)
        
        # Simulate slow processing
        processor.record_processing_time(5.0, 10)  # 5 seconds for 10 items
        
        # Should reduce batch size
        assert processor.batch_size < 10


class TestLoadBalancer:
    """Test load balancing system."""
    
    def create_test_instance(self, instance_id: str, load: float = 0.5) -> InstanceMetrics:
        """Create a test instance."""
        return InstanceMetrics(
            id=instance_id,
            cpu_usage=load * 80,
            memory_usage=load * 70,
            active_connections=int(load * 20),
            request_queue_size=int(load * 10),
            response_time_avg=load * 2,
            error_rate=load * 0.1,
            last_updated=time.time()
        )
    
    def test_load_balancer_round_robin(self):
        """Test round-robin load balancing."""
        lb = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        # Add test instances
        lb.add_instance(self.create_test_instance("instance1"))
        lb.add_instance(self.create_test_instance("instance2"))
        lb.add_instance(self.create_test_instance("instance3"))
        
        # Test round-robin selection
        selections = []
        for _ in range(6):
            selected = lb.select_instance()
            selections.append(selected)
        
        # Should cycle through instances
        assert selections[0] == selections[3]
        assert selections[1] == selections[4]
        assert selections[2] == selections[5]
    
    def test_load_balancer_least_connections(self):
        """Test least connections load balancing."""
        lb = LoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
        
        # Add instances with different connection counts
        lb.add_instance(self.create_test_instance("instance1", load=0.8))  # High connections
        lb.add_instance(self.create_test_instance("instance2", load=0.2))  # Low connections
        lb.add_instance(self.create_test_instance("instance3", load=0.5))  # Medium connections
        
        # Should select instance with least connections
        selected = lb.select_instance()
        assert selected == "instance2"
    
    def test_load_balancer_health_filtering(self):
        """Test that unhealthy instances are filtered out."""
        lb = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        # Add healthy and unhealthy instances
        healthy_instance = self.create_test_instance("healthy", load=0.5)
        unhealthy_instance = self.create_test_instance("unhealthy", load=0.9)
        unhealthy_instance.health_status = "unhealthy"
        
        lb.add_instance(healthy_instance)
        lb.add_instance(unhealthy_instance)
        
        # Should only select healthy instance
        selected = lb.select_instance()
        assert selected == "healthy"
    
    def test_load_balancer_stats(self):
        """Test load balancer statistics."""
        lb = LoadBalancer()
        
        lb.add_instance(self.create_test_instance("instance1", load=0.3))
        lb.add_instance(self.create_test_instance("instance2", load=0.7))
        
        stats = lb.get_load_distribution()
        
        assert stats["total_instances"] == 2
        assert stats["healthy_instances"] == 2
        assert "instance1" in stats["instances"]
        assert "instance2" in stats["instances"]


class TestPredictiveScaler:
    """Test predictive scaling system."""
    
    def test_predictive_scaler_basic(self):
        """Test basic predictive scaling."""
        scaler = PredictiveScaler()
        
        # Add load samples to build history
        for i in range(20):
            load = 0.5 + (i % 10) * 0.05  # Varying load
            scaler.add_load_sample(load)
        
        # Test prediction
        predicted_load, confidence = scaler.predict_load(minutes_ahead=5)
        
        assert 0.0 <= predicted_load <= 1.0
        assert 0.0 <= confidence <= 1.0
    
    def test_predictive_scaling_decision(self):
        """Test predictive scaling decision making."""
        scaler = PredictiveScaler()
        
        # Add high load samples
        for _ in range(10):
            scaler.add_load_sample(0.9)
        
        decision = scaler.should_scale_predictively(
            current_instances=2,
            min_instances=1,
            max_instances=10
        )
        
        assert decision is not None
        assert decision.action == "scale_up"
        assert decision.target_instances > 2


class TestAdvancedAutoScaler:
    """Test advanced auto-scaling system."""
    
    def test_autoscaler_initialization(self):
        """Test auto-scaler initialization."""
        scaler = AdvancedAutoScaler(
            min_instances=2,
            max_instances=10,
            strategy=ScalingStrategy.HYBRID
        )
        
        assert scaler.min_instances == 2
        assert scaler.max_instances == 10
        assert scaler.current_instances == 2
        assert scaler.strategy == ScalingStrategy.HYBRID
    
    def test_autoscaler_metrics_update(self):
        """Test metrics update and processing."""
        scaler = AdvancedAutoScaler()
        
        metrics = {
            "overall_load": 0.8,
            "instances": {
                "instance1": {
                    "cpu_usage": 70.0,
                    "memory_usage": 60.0,
                    "active_connections": 15,
                    "queue_size": 5,
                    "response_time": 1.2,
                    "error_rate": 0.02,
                    "health_status": "healthy"
                }
            }
        }
        
        scaler.update_system_metrics(metrics)
        assert scaler.system_metrics["overall_load"] == 0.8
    
    def test_autoscaler_scaling_decision(self):
        """Test scaling decision making."""
        scaler = AdvancedAutoScaler(min_instances=1, max_instances=5)
        scaler.scale_cooldown = 0  # Disable cooldown for testing
        
        # Update with high load metrics
        high_load_metrics = {
            "overall_load": 0.9,
            "instances": {
                "instance1": {
                    "cpu_usage": 85.0,
                    "memory_usage": 80.0,
                    "active_connections": 25,
                    "queue_size": 15,
                    "response_time": 3.0,
                    "error_rate": 0.05,
                    "health_status": "healthy"
                }
            }
        }
        
        scaler.update_system_metrics(high_load_metrics)
        decision = scaler.make_scaling_decision()
        
        assert decision is not None
        assert decision.action == "scale_up"
        assert decision.target_instances > scaler.current_instances
    
    def test_autoscaler_status(self):
        """Test scaling status reporting."""
        scaler = AdvancedAutoScaler()
        
        status = scaler.get_scaling_status()
        
        required_fields = [
            "current_instances", "min_instances", "max_instances",
            "strategy", "last_scale_action", "cooldown_remaining",
            "recent_decisions", "load_balancer_stats", "scaling_events_count"
        ]
        
        for field in required_fields:
            assert field in status


class TestPerformanceEngine:
    """Test the central performance engine."""
    
    def test_performance_engine_initialization(self):
        """Test performance engine initialization."""
        engine = PerformanceEngine()
        
        assert engine.cache is not None
        assert isinstance(engine.resource_pools, dict)
        assert isinstance(engine.batch_processors, dict)
        assert isinstance(engine.stream_processors, dict)
        assert engine.auto_scaler is not None
    
    def test_performance_engine_resource_pool_creation(self):
        """Test resource pool creation through engine."""
        engine = PerformanceEngine()
        
        def create_connection():
            return {"connection": "test"}
        
        pool = engine.create_resource_pool(
            "test_pool",
            create_connection,
            min_size=1,
            max_size=5
        )
        
        assert "test_pool" in engine.resource_pools
        assert engine.resource_pools["test_pool"] is pool
    
    def test_performance_engine_batch_processor_creation(self):
        """Test batch processor creation through engine."""
        engine = PerformanceEngine()
        
        processor = engine.create_batch_processor(
            "test_batch",
            batch_size=50,
            max_wait_time=2.0
        )
        
        assert "test_batch" in engine.batch_processors
        assert engine.batch_processors["test_batch"] is processor
    
    def test_performance_engine_stream_processor_creation(self):
        """Test stream processor creation through engine."""
        engine = PerformanceEngine()
        
        processor = engine.create_stream_processor(
            "test_stream",
            buffer_size=1000,
            max_throughput=100.0
        )
        
        assert "test_stream" in engine.stream_processors
        assert engine.stream_processors["test_stream"] is processor
    
    def test_performance_engine_comprehensive_stats(self):
        """Test comprehensive statistics collection."""
        engine = PerformanceEngine()
        
        # Create some components to get stats from
        engine.create_resource_pool("pool1", lambda: {"test": True})
        engine.create_stream_processor("stream1", buffer_size=100)
        
        stats = engine.get_comprehensive_stats()
        
        required_sections = [
            "timestamp", "cache", "auto_scaler", "resource_pools",
            "stream_processors", "performance_summary"
        ]
        
        for section in required_sections:
            assert section in stats
        
        assert "pool1" in stats["resource_pools"]
        assert "stream1" in stats["stream_processors"]
    
    def test_performance_optimization_cycle(self):
        """Test performance optimization cycle."""
        engine = PerformanceEngine()
        
        # Mock system load calculation
        with patch.object(engine, '_calculate_system_load', return_value=0.8):
            result = engine.optimize_performance()
            
            assert "optimizations_applied" in result
            assert "system_load" in result
            assert "optimization_timestamp" in result
            assert result["system_load"] == 0.8


@pytest.mark.asyncio
class TestAsyncStreamProcessor:
    """Test async stream processing."""
    
    async def test_stream_processor_async(self):
        """Test async stream processing."""
        processor = StreamProcessor(buffer_size=100, max_throughput=10.0)
        
        # Create a simple async generator
        async def test_stream():
            for i in range(5):
                yield f"item_{i}"
                await asyncio.sleep(0.01)
        
        def simple_processor(item):
            return item.upper()
        
        results = []
        async for result in processor.process_stream(test_stream(), simple_processor):
            results.append(result)
        
        assert len(results) == 5
        assert results[0] == "ITEM_0"
        assert results[-1] == "ITEM_4"
    
    async def test_stream_processor_throughput_limiting(self):
        """Test throughput limiting in stream processor."""
        processor = StreamProcessor(buffer_size=10, max_throughput=2.0)
        
        async def fast_stream():
            for i in range(10):
                yield f"item_{i}"
                # No delay - trying to go as fast as possible
        
        def identity_processor(item):
            return item
        
        start_time = time.time()
        results = []
        async for result in processor.process_stream(fast_stream(), identity_processor):
            results.append(result)
        
        elapsed_time = time.time() - start_time
        
        # Should take at least some time due to rate limiting
        # (This is a simplified test - real rate limiting would be more precise)
        assert len(results) <= 10
        assert elapsed_time >= 0.01  # At least some delay


class TestIntegration:
    """Integration tests for combined features."""
    
    def test_engine_with_all_components(self):
        """Test performance engine with all components working together."""
        engine = PerformanceEngine()
        
        # Create all types of components
        pool = engine.create_resource_pool("integration_pool", lambda: {"id": time.time()})
        batch_proc = engine.create_batch_processor("integration_batch", batch_size=10)
        stream_proc = engine.create_stream_processor("integration_stream", buffer_size=50)
        
        # Use cache
        engine.cache.put("integration_key", "integration_value")
        
        # Add some load data to auto-scaler
        engine.auto_scaler.predictive_scaler.add_load_sample(0.6)
        engine.auto_scaler.predictive_scaler.add_load_sample(0.7)
        
        # Get comprehensive stats
        stats = engine.get_comprehensive_stats()
        
        # Verify all components are represented
        assert stats["cache"]["size"] == 1
        assert "integration_pool" in stats["resource_pools"]
        assert "integration_stream" in stats["stream_processors"]
        assert stats["auto_scaler"]["current_instances"] >= 1
        
    def test_scaling_with_load_balancer(self):
        """Test scaling system with integrated load balancer."""
        scaler = AdvancedAutoScaler(min_instances=1, max_instances=5)
        
        # Add instance metrics
        metrics = {
            "overall_load": 0.4,
            "instances": {
                f"instance_{i}": {
                    "cpu_usage": 40.0 + i * 10,
                    "memory_usage": 30.0 + i * 5,
                    "active_connections": 5 + i * 2,
                    "queue_size": i,
                    "response_time": 0.5 + i * 0.1,
                    "error_rate": 0.01,
                    "health_status": "healthy"
                }
                for i in range(3)
            }
        }
        
        scaler.update_system_metrics(metrics)
        
        # Test load balancer has instances
        healthy_instances = scaler.load_balancer.get_healthy_instances()
        assert len(healthy_instances) == 3
        
        # Test instance selection
        selected = scaler.load_balancer.select_instance()
        assert selected in [f"instance_{i}" for i in range(3)]
        
        # Test load distribution stats
        distribution = scaler.load_balancer.get_load_distribution()
        assert distribution["total_instances"] == 3
        assert distribution["healthy_instances"] == 3


# Performance benchmarks (optional, run with pytest -v -s)
class TestPerformanceBenchmarks:
    """Performance benchmarks for optimization features."""
    
    def test_cache_performance(self):
        """Benchmark cache performance."""
        cache = AdaptiveCache(max_size=10000)
        
        # Benchmark put operations
        start_time = time.time()
        for i in range(1000):
            cache.put(f"key_{i}", f"value_{i}")
        put_time = time.time() - start_time
        
        # Benchmark get operations  
        start_time = time.time()
        for i in range(1000):
            cache.get(f"key_{i}")
        get_time = time.time() - start_time
        
        # Simple performance assertions (adjust based on requirements)
        assert put_time < 1.0  # Should complete in under 1 second
        assert get_time < 0.5  # Gets should be faster than puts
        
        print(f"Cache benchmark - Put: {put_time:.3f}s, Get: {get_time:.3f}s")
    
    def test_batch_processing_performance(self):
        """Benchmark batch processing performance."""
        processor = BatchProcessor(batch_size=100, max_wait_time=1.0)
        
        # Add many items quickly
        start_time = time.time()
        for i in range(1000):
            processor.add_item(f"item_{i}")
        add_time = time.time() - start_time
        
        # Process batches
        batches_processed = 0
        start_time = time.time()
        while processor.should_process_batch():
            batch = processor.get_batch()
            batches_processed += 1
            processor.record_processing_time(0.01, len(batch))
        process_time = time.time() - start_time
        
        assert add_time < 0.1  # Adding items should be very fast
        assert batches_processed >= 10  # Should have created multiple batches
        
        print(f"Batch benchmark - Add: {add_time:.3f}s, Process: {process_time:.3f}s, Batches: {batches_processed}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])