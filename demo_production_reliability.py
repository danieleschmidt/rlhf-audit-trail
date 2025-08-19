#!/usr/bin/env python3
"""Demo: Production Reliability Patterns.

Demonstrates enterprise-grade reliability patterns including:
- Retry logic with exponential backoff
- Circuit breakers with monitoring
- Rate limiting with token bucket
- Bulkhead isolation
- Graceful degradation
- Real-time monitoring
"""

import asyncio
import json
import time
import random
from pathlib import Path

# Import our reliability modules
import sys
sys.path.append('src')

from rlhf_audit_trail.production_reliability import (
    ProductionReliabilityManager, RetryConfig, CircuitBreakerConfig,
    RateLimitConfig, BulkheadConfig
)
from rlhf_audit_trail.robust_monitoring_system import (
    RobustMonitoringSystem, AlertLevel, MetricType
)


class MockDatabaseService:
    """Mock database service for testing reliability patterns."""
    
    def __init__(self, failure_rate: float = 0.3):
        """Initialize mock service.
        
        Args:
            failure_rate: Probability of failure (0.0 to 1.0)
        """
        self.failure_rate = failure_rate
        self.call_count = 0
        
    async def query(self, query: str) -> dict:
        """Simulate database query."""
        self.call_count += 1
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Simulate failures
        if random.random() < self.failure_rate:
            raise Exception(f"Database connection failed (call #{self.call_count})")
            
        return {
            'query': query,
            'results': [{'id': i, 'data': f'row_{i}'} for i in range(5)],
            'call_count': self.call_count
        }


class MockMLService:
    """Mock ML service for testing reliability patterns."""
    
    def __init__(self, failure_rate: float = 0.2):
        """Initialize mock ML service.
        
        Args:
            failure_rate: Probability of failure
        """
        self.failure_rate = failure_rate
        self.model_loaded = True
        self.call_count = 0
        
    async def predict(self, data: dict) -> dict:
        """Simulate ML prediction."""
        self.call_count += 1
        
        # Simulate longer processing time
        await asyncio.sleep(random.uniform(0.5, 1.0))
        
        # Simulate model loading failures
        if not self.model_loaded or random.random() < self.failure_rate:
            raise Exception(f"ML model prediction failed (call #{self.call_count})")
            
        prediction = random.uniform(0.0, 1.0)
        confidence = random.uniform(0.7, 0.95)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'model_version': '1.0.0',
            'call_count': self.call_count
        }


async def demo_retry_patterns():
    """Demo retry patterns with different strategies."""
    print("🔄 Retry Patterns Demo")
    print("=" * 40)
    
    reliability_manager = ProductionReliabilityManager()
    monitoring = RobustMonitoringSystem()
    monitoring.start_monitoring()
    
    # Create mock service with high failure rate
    db_service = MockDatabaseService(failure_rate=0.7)
    
    # Apply retry pattern
    @reliability_manager.get_retry_handler('database')
    async def reliable_query(query: str):
        """Database query with retry logic."""
        monitoring.record_metric('database.request', 1, MetricType.COUNTER)
        
        try:
            result = await db_service.query(query)
            monitoring.record_metric('database.success', 1, MetricType.COUNTER)
            return result
        except Exception as e:
            monitoring.record_metric('database.error', 1, MetricType.COUNTER)
            raise e
    
    # Test retry behavior
    for i in range(3):
        print(f"\n📝 Test {i + 1}: Retrying database query")
        start_time = time.time()
        
        try:
            result = await reliable_query(f"SELECT * FROM table_{i}")
            duration = (time.time() - start_time) * 1000
            
            print(f"  ✅ Success after {result['call_count']} database calls")
            print(f"  📊 Duration: {duration:.0f}ms")
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            print(f"  ❌ Failed after all retries: {e}")
            print(f"  📊 Duration: {duration:.0f}ms")
            
    # Show monitoring metrics
    metrics_summary = monitoring.get_metrics_summary()
    print(f"\n📈 Metrics Summary:")
    
    for metric_name, data in metrics_summary.items():
        if 'database' in metric_name:
            current_val = data.get('current', 'N/A')
            count = data.get('count', 0)
            print(f"  {metric_name}: {current_val} (total: {count})")
    
    monitoring.stop_monitoring()


async def demo_circuit_breaker():
    """Demo circuit breaker pattern."""
    print("\n⚡ Circuit Breaker Demo")
    print("=" * 40)
    
    reliability_manager = ProductionReliabilityManager()
    
    # Create ML service with failures
    ml_service = MockMLService(failure_rate=0.8)  # High failure rate
    
    # Apply circuit breaker
    @reliability_manager.get_circuit_breaker('ml_inference')
    async def protected_predict(data: dict):
        """ML prediction with circuit breaker."""
        return await ml_service.predict(data)
    
    # Test circuit breaker behavior
    for i in range(15):
        print(f"\n🔮 Test {i + 1}: ML prediction")
        
        try:
            start_time = time.time()
            result = await protected_predict({'input': f'data_{i}'})
            duration = (time.time() - start_time) * 1000
            
            print(f"  ✅ Prediction: {result['prediction']:.3f} "
                  f"(confidence: {result['confidence']:.3f})")
            print(f"  📊 Duration: {duration:.0f}ms")
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            print(f"  ❌ Failed: {str(e)[:60]}...")
            print(f"  📊 Duration: {duration:.0f}ms")
            
        # Check circuit breaker status
        breaker = reliability_manager.get_circuit_breaker('ml_inference')
        stats = breaker.get_stats()
        print(f"  🔌 Circuit State: {stats['state']} "
              f"(failures: {stats['failure_count']}, "
              f"successes: {stats['success_count']})")
        
        # Brief pause between requests
        await asyncio.sleep(0.5)


async def demo_rate_limiting():
    """Demo rate limiting with token bucket."""
    print("\n🚦 Rate Limiting Demo")
    print("=" * 40)
    
    reliability_manager = ProductionReliabilityManager()
    
    # Create rate-limited function
    @reliability_manager.get_rate_limiter('api_requests')
    async def rate_limited_api_call(request_id: int):
        """API call with rate limiting."""
        await asyncio.sleep(0.1)  # Simulate API call
        return {'request_id': request_id, 'timestamp': time.time()}
    
    # Test rate limiting
    start_time = time.time()
    successful_requests = 0
    rate_limited_requests = 0
    
    # Send burst of requests
    for i in range(250):  # More than burst size
        try:
            result = await rate_limited_api_call(i)
            successful_requests += 1
            
            if i % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  📈 Request {i}: Success (elapsed: {elapsed:.1f}s)")
                
        except Exception as e:
            rate_limited_requests += 1
            
            if "Rate limit" in str(e):
                if rate_limited_requests % 10 == 1:
                    elapsed = time.time() - start_time
                    print(f"  🚫 Request {i}: Rate limited (elapsed: {elapsed:.1f}s)")
    
    total_time = time.time() - start_time
    
    print(f"\n📊 Rate Limiting Results:")
    print(f"  Successful requests: {successful_requests}")
    print(f"  Rate limited requests: {rate_limited_requests}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Effective rate: {successful_requests / total_time:.1f} req/s")


async def demo_bulkhead_isolation():
    """Demo bulkhead isolation pattern."""
    print("\n🛡️  Bulkhead Isolation Demo")
    print("=" * 40)
    
    reliability_manager = ProductionReliabilityManager()
    
    # Create bulkhead-isolated function
    @reliability_manager.get_bulkhead('ml_processing')
    async def isolated_ml_task(task_id: int):
        """ML task with bulkhead isolation."""
        processing_time = random.uniform(1.0, 3.0)
        await asyncio.sleep(processing_time)
        
        return {
            'task_id': task_id,
            'processing_time': processing_time,
            'result': f'processed_data_{task_id}'
        }
    
    # Launch many concurrent tasks
    print("🚀 Launching 20 concurrent ML tasks...")
    
    tasks = []
    start_time = time.time()
    
    for i in range(20):
        task = asyncio.create_task(isolated_ml_task(i))
        tasks.append(task)
        
    # Monitor progress
    completed = 0
    failed = 0
    
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
            completed += 1
            elapsed = time.time() - start_time
            
            print(f"  ✅ Task {result['task_id']} completed in "
                  f"{result['processing_time']:.1f}s (total elapsed: {elapsed:.1f}s)")
                  
        except Exception as e:
            failed += 1
            elapsed = time.time() - start_time
            print(f"  ❌ Task failed: {str(e)[:50]}... (elapsed: {elapsed:.1f}s)")
    
    total_time = time.time() - start_time
    
    print(f"\n📊 Bulkhead Isolation Results:")
    print(f"  Completed tasks: {completed}")
    print(f"  Failed tasks: {failed}")
    print(f"  Total time: {total_time:.1f}s")
    
    # Show bulkhead stats
    bulkhead = reliability_manager.get_bulkhead('ml_processing')
    stats = bulkhead.get_stats()
    print(f"  Max concurrent: {stats['max_concurrent']}")
    print(f"  Final active requests: {stats['active_requests']}")


async def demo_graceful_degradation():
    """Demo graceful degradation pattern."""
    print("\n🎭 Graceful Degradation Demo")
    print("=" * 40)
    
    reliability_manager = ProductionReliabilityManager()
    
    # Setup degradation trigger (simulating high load)
    high_load = False
    
    def check_system_load():
        return high_load
    
    reliability_manager.degradation_handler.register_trigger(
        'ml_inference', check_system_load
    )
    
    # Create service that can degrade
    ml_service = MockMLService(failure_rate=0.0)  # No failures initially
    
    @reliability_manager.degradation_handler('ml_inference')
    async def degradable_predict(data: dict):
        """ML prediction that can degrade gracefully."""
        return await ml_service.predict(data)
    
    # Test normal operation
    print("🔵 Normal Operation:")
    for i in range(3):
        result = await degradable_predict({'input': f'data_{i}'})
        is_fallback = result.get('fallback', False)
        status = "🔄 Fallback" if is_fallback else "✅ Normal"
        
        print(f"  {status} Prediction {i + 1}: {result['prediction']:.3f}")
    
    # Trigger degradation
    print("\n🟡 High Load - Degraded Operation:")
    high_load = True
    
    for i in range(3):
        result = await degradable_predict({'input': f'data_{i + 3}'})
        is_fallback = result.get('fallback', False)
        status = "🔄 Fallback" if is_fallback else "✅ Normal"
        
        print(f"  {status} Prediction {i + 4}: {result['prediction']:.3f}")
    
    # Return to normal
    print("\n🟢 Load Reduced - Normal Operation:")
    high_load = False
    
    for i in range(3):
        result = await degradable_predict({'input': f'data_{i + 6}'})
        is_fallback = result.get('fallback', False)
        status = "🔄 Fallback" if is_fallback else "✅ Normal"
        
        print(f"  {status} Prediction {i + 7}: {result['prediction']:.3f}")


async def demo_combined_patterns():
    """Demo multiple reliability patterns working together."""
    print("\n🎪 Combined Patterns Demo")
    print("=" * 40)
    
    reliability_manager = ProductionReliabilityManager()
    monitoring = RobustMonitoringSystem()
    monitoring.start_monitoring()
    
    # Create unreliable service
    db_service = MockDatabaseService(failure_rate=0.4)
    
    # Apply multiple patterns
    @reliability_manager.apply_reliability_patterns(
        'database',
        retry=True,
        circuit_breaker=True,
        rate_limit=True,
        graceful_degradation=True
    )
    async def robust_database_query(query: str):
        """Database query with all reliability patterns."""
        monitoring.record_metric('robust_db.request', 1, MetricType.COUNTER)
        
        try:
            result = await db_service.query(query)
            monitoring.record_metric('robust_db.success', 1, MetricType.COUNTER)
            return result
        except Exception as e:
            monitoring.record_metric('robust_db.error', 1, MetricType.COUNTER)
            raise e
    
    # Test combined patterns
    successful_queries = 0
    failed_queries = 0
    
    print("🚀 Running 15 database queries with combined patterns...")
    
    for i in range(15):
        try:
            start_time = time.time()
            result = await robust_database_query(f"SELECT * FROM products WHERE id = {i}")
            duration = (time.time() - start_time) * 1000
            
            successful_queries += 1
            
            if i % 3 == 0:
                print(f"  ✅ Query {i + 1}: Success in {duration:.0f}ms")
                
        except Exception as e:
            failed_queries += 1
            print(f"  ❌ Query {i + 1}: {str(e)[:50]}...")
            
        # Brief pause
        await asyncio.sleep(0.3)
    
    print(f"\n📊 Combined Patterns Results:")
    print(f"  Successful queries: {successful_queries}")
    print(f"  Failed queries: {failed_queries}")
    print(f"  Success rate: {successful_queries / 15:.1%}")
    
    # Show reliability status
    reliability_status = reliability_manager.get_reliability_status()
    print(f"\n🔍 Reliability Status:")
    print(f"  Overall health: {reliability_status['overall_health']}")
    
    for name, stats in reliability_status['circuit_breakers'].items():
        if 'database' in name:
            print(f"  Circuit breaker ({name}): {stats['state']} "
                  f"(failures: {stats['failure_count']})")
    
    monitoring.stop_monitoring()


async def demo_monitoring_integration():
    """Demo monitoring integration with reliability patterns."""
    print("\n📊 Monitoring Integration Demo")
    print("=" * 40)
    
    monitoring = RobustMonitoringSystem()
    monitoring.start_monitoring()
    
    # Add custom alert callback
    def alert_handler(alert):
        level_icons = {
            'debug': '🔍',
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }
        
        icon = level_icons.get(alert.level.value, '📢')
        print(f"  {icon} ALERT: {alert.title}")
        if alert.current_value is not None:
            print(f"     Value: {alert.current_value}")
    
    monitoring.add_alert_callback(alert_handler)
    
    # Simulate some metrics that will trigger alerts
    print("📈 Generating metrics to trigger alerts...")
    
    # Normal metrics
    for i in range(5):
        monitoring.record_metric('response_time', 150 + i * 10, MetricType.TIMER)
        await asyncio.sleep(0.2)
    
    # Spike in response time (should trigger warning)
    monitoring.record_metric('response_time', 1200, MetricType.TIMER)
    await asyncio.sleep(0.5)
    
    # Critical spike
    monitoring.record_metric('response_time', 6000, MetricType.TIMER)
    await asyncio.sleep(0.5)
    
    # High error rate
    for i in range(10):
        monitoring.record_metric('error_rate', 0.08 + i * 0.01, MetricType.GAUGE)
        await asyncio.sleep(0.1)
    
    # Return to normal
    for i in range(3):
        monitoring.record_metric('response_time', 180, MetricType.TIMER)
        monitoring.record_metric('error_rate', 0.02, MetricType.GAUGE)
        await asyncio.sleep(0.2)
    
    # Wait for monitoring to process
    await asyncio.sleep(2.0)
    
    # Show system health
    health = monitoring.get_system_health()
    print(f"\n🏥 System Health:")
    print(f"  Overall status: {health['overall_status']}")
    print(f"  Active alerts: {health['active_alerts']}")
    print(f"  Resolved alerts: {health['resolved_alerts']}")
    
    monitoring.stop_monitoring()


if __name__ == "__main__":
    async def main():
        print("🛡️  Production Reliability Patterns Demo")
        print("Demonstrating enterprise-grade reliability patterns\n")
        
        try:
            # Run all demos
            await demo_retry_patterns()
            await demo_circuit_breaker()
            await demo_rate_limiting()
            await demo_bulkhead_isolation()
            await demo_graceful_degradation()
            await demo_combined_patterns()
            await demo_monitoring_integration()
            
            print("\n🎉 Demo completed successfully!")
            print("\nKey Patterns Demonstrated:")
            print("✅ Retry with exponential backoff")
            print("✅ Circuit breaker with state monitoring")
            print("✅ Token bucket rate limiting")
            print("✅ Bulkhead isolation for resource protection")
            print("✅ Graceful degradation with fallbacks")
            print("✅ Combined patterns working together")
            print("✅ Real-time monitoring and alerting")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the demo
    asyncio.run(main())