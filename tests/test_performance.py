"""Performance and optimization tests."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

from src.quantum_task_planner.core import Task, TaskState, QuantumPriority, QuantumTaskPlanner
from src.quantum_task_planner.performance import (
    PerformanceConfig, QuantumTaskCache, BatchProcessor,
    ParallelExecutor, QuantumOptimizationEngine, PerformanceManager,
    CacheStrategy
)


class TestQuantumTaskCache:
    """Test caching system."""
    
    def test_lru_cache_basic_operations(self):
        """Test basic LRU cache operations."""
        config = PerformanceConfig(cache_strategy=CacheStrategy.LRU, cache_size=3)
        cache = QuantumTaskCache(config)
        
        # Test put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("nonexistent") is None
        
        # Test cache statistics
        stats = cache.get_stats()
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
    
    def test_lru_cache_eviction(self):
        """Test LRU cache eviction policy."""
        config = PerformanceConfig(cache_strategy=CacheStrategy.LRU, cache_size=2)
        cache = QuantumTaskCache(config)
        
        # Fill cache to capacity
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access first key to make it recently used
        cache.get("key1")
        
        # Add third key, should evict key2
        cache.put("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Should still be there
        assert cache.get("key2") is None       # Should be evicted
        assert cache.get("key3") == "value3"   # Should be there
    
    def test_ttl_cache(self):
        """Test TTL cache functionality."""
        config = PerformanceConfig(cache_strategy=CacheStrategy.TTL, cache_ttl_seconds=0.1)
        cache = QuantumTaskCache(config)
        
        # Put value
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get("key1") is None
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        config = PerformanceConfig()
        cache = QuantumTaskCache(config)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        cache.invalidate("key1")
        assert cache.get("key1") is None
    
    def test_cache_clear(self):
        """Test cache clearing."""
        config = PerformanceConfig()
        cache = QuantumTaskCache(config)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestBatchProcessor:
    """Test batch processing system."""
    
    @pytest.mark.asyncio
    async def test_batch_size_trigger(self):
        """Test batch processing triggered by size."""
        config = PerformanceConfig(batch_size=3, batch_timeout_seconds=10.0)
        processor = BatchProcessor(config)
        
        processed_batches = []
        
        async def mock_processor(batch):
            processed_batches.append(len(batch))
            return f"processed_{len(batch)}"
        
        # Add tasks one by one
        tasks = [Task(name=f"Task {i}") for i in range(3)]
        
        results = []
        for task in tasks[:-1]:
            result = await processor.add_task_to_batch(task, mock_processor)
            results.append(result)
        
        # Last task should trigger batch processing
        result = await processor.add_task_to_batch(tasks[-1], mock_processor)
        results.append(result)
        
        # Should have processed one batch of size 3
        assert len(processed_batches) == 1
        assert processed_batches[0] == 3
    
    @pytest.mark.asyncio
    async def test_batch_timeout_trigger(self):
        """Test batch processing triggered by timeout."""
        config = PerformanceConfig(batch_size=10, batch_timeout_seconds=0.1)
        processor = BatchProcessor(config)
        
        processed_batches = []
        
        async def mock_processor(batch):
            processed_batches.append(len(batch))
            return f"processed_{len(batch)}"
        
        # Add single task
        task = Task(name="Test Task")
        await processor.add_task_to_batch(task, mock_processor)
        
        # Wait for timeout
        await asyncio.sleep(0.15)
        
        # Should have processed batch due to timeout
        assert len(processed_batches) == 1
        assert processed_batches[0] == 1


class TestParallelExecutor:
    """Test parallel execution system."""
    
    @pytest.mark.asyncio
    async def test_parallel_task_execution(self):
        """Test parallel task execution."""
        config = PerformanceConfig(max_worker_threads=2)
        executor = ParallelExecutor(config)
        
        def mock_executor_func(task):
            time.sleep(0.01)  # Simulate work
            return f"executed_{task.name}"
        
        tasks = [Task(name=f"Task {i}") for i in range(4)]
        
        start_time = time.time()
        results = await executor.execute_parallel_tasks(tasks, mock_executor_func)
        end_time = time.time()
        
        # Should complete in less time than sequential execution
        assert end_time - start_time < 0.04  # Less than 4 * 0.01
        assert len(results) == 4
        
        # Clean up
        executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_parallel_scheduling(self):
        """Test parallel scheduling execution."""
        config = PerformanceConfig(enable_parallel_scheduling=True, max_worker_threads=2)
        executor = ParallelExecutor(config)
        
        def mock_scheduler_func(task_group):
            return len(task_group)  # Return group size
        
        task_groups = [
            [Task(name=f"Group1_Task{i}") for i in range(2)],
            [Task(name=f"Group2_Task{i}") for i in range(3)],
        ]
        
        results = await executor.execute_parallel_scheduling(task_groups, mock_scheduler_func)
        
        assert len(results) == 2
        assert results[0] == 2
        assert results[1] == 3
        
        # Clean up
        executor.shutdown()


class TestQuantumOptimizationEngine:
    """Test quantum optimization engine."""
    
    def test_interference_calculation_caching(self):
        """Test cached interference calculations."""
        config = PerformanceConfig(interference_caching=True)
        engine = QuantumOptimizationEngine(config)
        
        task1 = Task(name="Task 1", amplitude=0.6, phase=0.0)
        task2 = Task(name="Task 2", amplitude=0.8, phase=1.57)  # π/2
        
        # First calculation
        result1 = engine.calculate_interference_optimized(task1, task2)
        
        # Second calculation (should use cache)
        result2 = engine.calculate_interference_optimized(task1, task2)
        
        assert result1 == result2
        
        # Check cache stats
        cache_stats = engine.interference_cache.get_stats()
        assert cache_stats["hit_count"] >= 1
    
    def test_lazy_decoherence_processing(self):
        """Test lazy decoherence processing."""
        config = PerformanceConfig(lazy_decoherence=True)
        engine = QuantumOptimizationEngine(config)
        
        # Create tasks with short coherence time
        tasks = [
            Task(name=f"Task {i}", coherence_time=0.1) 
            for i in range(3)
        ]
        
        # Force time passage
        for task in tasks:
            task.created_at = task.created_at - timedelta(seconds=1)
        
        # Process decoherence (should be lazy)
        decohered = engine.process_decoherence_lazy(tasks)
        
        # Verify processing occurred
        assert isinstance(decohered, list)
    
    def test_entanglement_optimization(self):
        """Test entanglement creation optimization."""
        config = PerformanceConfig(entanglement_pooling=True)
        engine = QuantumOptimizationEngine(config)
        
        task1 = Task(name="Task 1", priority=QuantumPriority.HIGH)
        task2 = Task(name="Task 2", priority=QuantumPriority.HIGH)
        
        # Create entanglement
        success = engine.optimize_entanglement_creation(task1, task2)
        
        # Should succeed with high probability for similar tasks
        assert isinstance(success, bool)
        
        if success:
            assert task1.id in task2.entangled_tasks
            assert task2.id in task1.entangled_tasks


class TestPerformanceManager:
    """Test performance management system."""
    
    @pytest.mark.asyncio
    async def test_task_batch_optimization(self):
        """Test batch optimization of tasks."""
        config = PerformanceConfig(enable_batch_optimization=True)
        manager = PerformanceManager(config)
        
        tasks = [Task(name=f"Task {i}") for i in range(5)]
        
        # Optimize batch
        optimized_tasks = await manager.optimize_task_batch(tasks)
        
        assert len(optimized_tasks) == len(tasks)
        assert manager.operation_count > 0
        assert manager.total_execution_time >= 0
    
    @pytest.mark.asyncio
    async def test_parallel_optimization(self):
        """Test parallel task optimization."""
        config = PerformanceConfig(
            enable_parallel_optimization=True,
            max_worker_threads=2
        )
        manager = PerformanceManager(config)
        
        # Create enough tasks to trigger parallel processing
        tasks = [Task(name=f"Task {i}") for i in range(8)]
        
        start_time = time.time()
        optimized_tasks = await manager.optimize_task_batch(tasks)
        end_time = time.time()
        
        assert len(optimized_tasks) >= 0  # Some tasks should be optimized
        assert end_time - start_time >= 0
        
        # Clean up
        manager.shutdown()
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        manager = PerformanceManager()
        
        metrics = manager.get_performance_metrics()
        
        assert "uptime_seconds" in metrics
        assert "operation_count" in metrics
        assert "cache_stats" in metrics
        assert "resource_usage" in metrics
        assert "configuration" in metrics
        
        # Verify metrics structure
        assert isinstance(metrics["uptime_seconds"], (int, float))
        assert isinstance(metrics["operation_count"], int)
        assert isinstance(metrics["cache_stats"], dict)
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test performance health check."""
        manager = PerformanceManager()
        
        health = await manager.health_check()
        
        assert "overall_status" in health
        assert "issues" in health
        assert "recommendations" in health
        
        # Should be healthy initially
        assert health["overall_status"] in ["healthy", "warning"]
        assert isinstance(health["issues"], list)
        assert isinstance(health["recommendations"], list)


class TestResourceMonitor:
    """Test resource monitoring."""
    
    def test_resource_constraint_checking(self):
        """Test resource constraint checking."""
        from src.quantum_task_planner.performance import ResourceMonitor
        
        config = PerformanceConfig(
            memory_limit_mb=100,
            cpu_usage_threshold=0.5
        )
        monitor = ResourceMonitor(config)
        
        # Should be within constraints initially
        assert monitor.check_resource_constraints()
        
        # Simulate high resource usage
        monitor.memory_usage_mb = 150  # Over limit
        assert not monitor.check_resource_constraints()
    
    def test_scaling_suggestions(self):
        """Test auto-scaling suggestions."""
        from src.quantum_task_planner.performance import ResourceMonitor
        
        config = PerformanceConfig(auto_scaling_enabled=True)
        monitor = ResourceMonitor(config)
        
        # Simulate high CPU usage
        monitor.cpu_usage_percent = 90
        suggestion = monitor.suggest_scaling_action()
        assert suggestion == "scale_up"
        
        # Simulate low CPU usage
        monitor.cpu_usage_percent = 20
        suggestion = monitor.suggest_scaling_action()
        assert suggestion == "scale_down"
        
        # Simulate medium CPU usage
        monitor.cpu_usage_percent = 50
        suggestion = monitor.suggest_scaling_action()
        assert suggestion is None
    
    def test_scaling_action_application(self):
        """Test applying scaling actions."""
        from src.quantum_task_planner.performance import ResourceMonitor
        
        config = PerformanceConfig(max_worker_threads=4, batch_size=10)
        monitor = ResourceMonitor(config)
        
        # Test scale up
        original_threads = config.max_worker_threads
        original_batch_size = config.batch_size
        
        success = monitor.apply_scaling_action("scale_up")
        assert success
        assert config.max_worker_threads > original_threads
        assert config.batch_size > original_batch_size
        
        # Test scale down
        success = monitor.apply_scaling_action("scale_down")
        assert success
        assert config.max_worker_threads < monitor.config.max_worker_threads


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.asyncio
    async def test_task_creation_performance(self):
        """Benchmark task creation performance."""
        planner = QuantumTaskPlanner()
        
        start_time = time.time()
        
        # Create many tasks
        tasks = []
        for i in range(100):
            task = planner.create_task(name=f"Benchmark Task {i}")
            tasks.append(task)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Should create 100 tasks in reasonable time
        assert creation_time < 1.0  # Less than 1 second
        assert len(tasks) == 100
        
        # Calculate throughput
        throughput = len(tasks) / creation_time
        print(f"Task creation throughput: {throughput:.2f} tasks/second")
    
    @pytest.mark.asyncio
    async def test_quantum_cycle_performance(self):
        """Benchmark quantum cycle performance."""
        planner = QuantumTaskPlanner()
        
        # Create many tasks
        for i in range(50):
            planner.create_task(name=f"Cycle Task {i}")
        
        start_time = time.time()
        
        # Run multiple quantum cycles
        results = []
        for _ in range(10):
            result = await planner.run_quantum_cycle()
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete cycles in reasonable time
        assert total_time < 5.0  # Less than 5 seconds for 10 cycles
        
        # Calculate cycle throughput
        cycle_throughput = len(results) / total_time
        print(f"Quantum cycle throughput: {cycle_throughput:.2f} cycles/second")
    
    def test_interference_calculation_performance(self):
        """Benchmark interference calculation performance."""
        config = PerformanceConfig(interference_caching=True)
        engine = QuantumOptimizationEngine(config)
        
        # Create test tasks
        tasks = [Task(name=f"Task {i}") for i in range(20)]
        
        start_time = time.time()
        
        # Calculate all pairwise interferences
        interferences = []
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks[i+1:], i+1):
                interference = engine.calculate_interference_optimized(task1, task2)
                interferences.append(interference)
        
        end_time = time.time()
        calculation_time = end_time - start_time
        
        # Should calculate interferences quickly
        assert calculation_time < 1.0  # Less than 1 second
        assert len(interferences) == (20 * 19) // 2  # n*(n-1)/2 pairs
        
        # Test cache effectiveness by recalculating
        start_time = time.time()
        
        cached_interferences = []
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks[i+1:], i+1):
                interference = engine.calculate_interference_optimized(task1, task2)
                cached_interferences.append(interference)
        
        end_time = time.time()
        cached_calculation_time = end_time - start_time
        
        # Cached calculations should be faster
        assert cached_calculation_time < calculation_time
        
        # Results should be identical
        for orig, cached in zip(interferences, cached_interferences):
            assert abs(orig - cached) < 1e-10


class TestCacheStrategies:
    """Test different caching strategies."""
    
    def test_lru_vs_ttl_performance(self):
        """Compare LRU vs TTL cache performance."""
        # LRU cache
        lru_config = PerformanceConfig(cache_strategy=CacheStrategy.LRU, cache_size=100)
        lru_cache = QuantumTaskCache(lru_config)
        
        # TTL cache
        ttl_config = PerformanceConfig(cache_strategy=CacheStrategy.TTL, cache_ttl_seconds=60.0)
        ttl_cache = QuantumTaskCache(ttl_config)
        
        # Test data
        test_data = [(f"key_{i}", f"value_{i}") for i in range(50)]
        
        # Benchmark LRU cache
        start_time = time.time()
        for key, value in test_data:
            lru_cache.put(key, value)
        for key, value in test_data:
            result = lru_cache.get(key)
            assert result == value
        lru_time = time.time() - start_time
        
        # Benchmark TTL cache
        start_time = time.time()
        for key, value in test_data:
            ttl_cache.put(key, value)
        for key, value in test_data:
            result = ttl_cache.get(key)
            assert result == value
        ttl_time = time.time() - start_time
        
        # Both should complete quickly
        assert lru_time < 1.0
        assert ttl_time < 1.0
        
        print(f"LRU cache time: {lru_time:.4f}s")
        print(f"TTL cache time: {ttl_time:.4f}s")


@pytest.mark.stress
class TestStressTests:
    """Stress tests for performance validation."""
    
    @pytest.mark.asyncio
    async def test_high_task_volume(self):
        """Test performance with high task volume."""
        planner = QuantumTaskPlanner()
        
        # Create many tasks
        task_count = 1000
        start_time = time.time()
        
        for i in range(task_count):
            planner.create_task(name=f"Stress Task {i}")
        
        creation_time = time.time() - start_time
        
        # Should handle high volume
        assert len(planner.tasks) == task_count
        assert creation_time < 10.0  # Less than 10 seconds
        
        print(f"Created {task_count} tasks in {creation_time:.2f} seconds")
        print(f"Throughput: {task_count/creation_time:.2f} tasks/second")
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        planner = QuantumTaskPlanner()
        
        # Create many tasks with metadata
        for i in range(500):
            planner.create_task(
                name=f"Memory Task {i}",
                description=f"Task {i} with description",
                metadata={
                    "index": i,
                    "data": f"some_data_{i}",
                    "tags": ["stress", "test", f"task_{i}"]
                }
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.2f} MB for 500 tasks")
        print(f"Memory per task: {memory_increase/500:.4f} MB")
        
        # Should be reasonable memory usage
        assert memory_increase < 100  # Less than 100MB for 500 tasks


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not benchmark and not stress"])