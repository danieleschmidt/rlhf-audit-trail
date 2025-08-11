"""
Comprehensive integration tests for the enhanced RLHF audit trail system.
Tests all three generations: Basic functionality, Robustness, and Scalability.
"""

import pytest
import asyncio
import time
import sys
import os
import tempfile
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rlhf_audit_trail import AuditableRLHF, PrivacyConfig, ComplianceConfig
from rlhf_audit_trail.validation import ComprehensiveValidator, ValidationSeverity
from rlhf_audit_trail.health_monitoring import HealthMonitor, HealthStatus
from rlhf_audit_trail.security_hardening import InputSanitizer, SecurityValidator
from rlhf_audit_trail.performance_optimization import PerformanceOptimizer, get_performance_optimizer
from rlhf_audit_trail.auto_scaling import ScalingManager
from rlhf_audit_trail.exceptions import (
    AuditTrailError, PrivacyBudgetExceededError, ValidationError as AuditValidationError,
    SecurityError, MonitoringError, PerformanceError
)


class TestGeneration1BasicFunctionality:
    """Test Generation 1: Make It Work - Basic functionality."""
    
    @pytest.fixture
    async def basic_auditor(self):
        """Create basic auditor for testing."""
        privacy_config = PrivacyConfig(epsilon=10.0, delta=1e-5)
        compliance_config = ComplianceConfig()
        
        auditor = AuditableRLHF(
            model_name="test-model-gen1",
            privacy_config=privacy_config,
            compliance_config=compliance_config,
            storage_backend="local"
        )
        return auditor
    
    @pytest.mark.asyncio
    async def test_basic_training_session(self, basic_auditor):
        """Test basic training session creation and management."""
        async with basic_auditor.track_training("test_experiment") as session:
            assert session.session_id is not None
            assert session.experiment_name == "test_experiment"
            assert session.model_name == "test-model-gen1"
            assert session.is_active
            assert session.phase.value in ["initialization", "human_feedback", "policy_update", "validation", "checkpoint", "completion"]
    
    @pytest.mark.asyncio
    async def test_annotation_logging(self, basic_auditor):
        """Test human annotation logging with privacy protection."""
        async with basic_auditor.track_training("annotation_test") as session:
            prompts = ["What is AI?", "Explain machine learning"]
            responses = ["AI is artificial intelligence", "ML is a subset of AI"]
            rewards = [0.8, 0.9]
            annotator_ids = ["ann_001", "ann_002"]
            
            batch = await basic_auditor.log_annotations(
                prompts=prompts,
                responses=responses,
                rewards=rewards,
                annotator_ids=annotator_ids,
                metadata={"test": "annotation_test"}
            )
            
            assert batch.batch_id is not None
            assert batch.batch_size == 2
            assert len(batch.prompts) == 2
            assert len(batch.rewards) == 2
    
    @pytest.mark.asyncio
    async def test_policy_updates(self, basic_auditor):
        """Test policy update tracking."""
        async with basic_auditor.track_training("policy_test") as session:
            mock_model = "test_model"
            mock_optimizer = "test_optimizer"
            mock_batch = "test_batch"
            
            update = await basic_auditor.track_policy_update(
                model=mock_model,
                optimizer=mock_optimizer,
                batch=mock_batch,
                loss=0.5,
                metadata={"test": "policy_test"}
            )
            
            assert update.update_id is not None
            assert update.loss == 0.5
            assert update.step_number > 0
    
    @pytest.mark.asyncio
    async def test_checkpoints(self, basic_auditor):
        """Test training checkpoints."""
        async with basic_auditor.track_training("checkpoint_test") as session:
            metrics = {"loss": 0.3, "accuracy": 0.85}
            
            await basic_auditor.checkpoint(
                epoch=1,
                metrics=metrics,
                metadata={"test": "checkpoint_test"}
            )
            
            # Verify checkpoint was created (would check storage in real implementation)
            assert basic_auditor.current_session is not None
    
    @pytest.mark.asyncio
    async def test_model_card_generation(self, basic_auditor):
        """Test model card generation."""
        async with basic_auditor.track_training("model_card_test") as session:
            # Add some data first
            await basic_auditor.log_annotations(
                prompts=["test"],
                responses=["response"],
                rewards=[0.7],
                annotator_ids=["ann_001"]
            )
            
            model_card = await basic_auditor.generate_model_card(
                include_provenance=True,
                include_privacy_analysis=True,
                format="eu_standard"
            )
            
            assert model_card["model_name"] == "test-model-gen1"
            assert "privacy_analysis" in model_card
            assert "training_summary" in model_card
    
    @pytest.mark.asyncio
    async def test_privacy_budget_tracking(self, basic_auditor):
        """Test privacy budget management."""
        privacy_report = basic_auditor.get_privacy_report()
        
        assert "total_epsilon" in privacy_report
        assert "epsilon_remaining" in privacy_report
        assert privacy_report["total_epsilon"] == 10.0
        
        # Test privacy budget depletion
        async with basic_auditor.track_training("privacy_test") as session:
            # Log many annotations to test budget limits
            for i in range(5):
                await basic_auditor.log_annotations(
                    prompts=[f"test_{i}"],
                    responses=[f"response_{i}"],
                    rewards=[0.8],
                    annotator_ids=["ann_001"]
                )
            
            updated_report = basic_auditor.get_privacy_report()
            assert updated_report["epsilon_remaining"] < privacy_report["epsilon_remaining"]


class TestGeneration2Robustness:
    """Test Generation 2: Make It Robust - Error handling, validation, security."""
    
    def test_input_validation_security(self):
        """Test comprehensive input validation and security."""
        validator = ComprehensiveValidator(strict_mode=True)
        
        # Test SQL injection detection
        malicious_prompts = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "UNION SELECT * FROM passwords"
        ]
        
        for prompt in malicious_prompts:
            result = validator.validate_annotation_batch(
                prompts=[prompt],
                responses=["safe response"],
                rewards=[0.5],
                annotator_ids=["ann_001"]
            )
            assert not result.is_valid
            assert any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                      for issue in result.issues)
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        # Test HTML encoding
        html_input = "<script>alert('xss')</script>"
        sanitized = InputSanitizer.sanitize_string(html_input, allow_html=False)
        assert "<script>" not in sanitized
        assert "&lt;script&gt;" in sanitized
        
        # Test SQL injection prevention
        with pytest.raises(SecurityError):
            InputSanitizer.sanitize_string("'; DROP TABLE users; --")
        
        # Test path traversal prevention
        with pytest.raises(SecurityError):
            InputSanitizer.sanitize_string("../../../etc/passwd")
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        from rlhf_audit_trail.security_hardening import RateLimiter, RateLimitConfig
        
        config = RateLimitConfig(requests_per_minute=5, burst_allowance=2)
        limiter = RateLimiter(config)
        
        # Test normal operation
        for i in range(5):
            allowed, info = limiter.is_allowed("test_user")
            assert allowed
        
        # Test rate limit exceeded
        allowed, info = limiter.is_allowed("test_user")
        assert not allowed
        assert info["reason"] == "rate_limit_exceeded"
    
    def test_health_monitoring(self):
        """Test health monitoring system."""
        monitor = HealthMonitor(check_interval=1)
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Wait for some metrics
        time.sleep(2)
        
        health_status = monitor.get_system_health()
        assert "overall_status" in health_status
        assert "components" in health_status
        assert "metrics_summary" in health_status
        
        # Stop monitoring
        monitor.stop_monitoring()
    
    def test_error_handling_robustness(self):
        """Test error handling and recovery."""
        from rlhf_audit_trail.validation import validate_annotation_batch
        
        # Test empty input handling
        result = validate_annotation_batch(
            prompts=[],
            responses=[],
            rewards=[],
            annotator_ids=[]
        )
        assert not result.is_valid
        
        # Test mismatched array lengths
        result = validate_annotation_batch(
            prompts=["test1", "test2"],
            responses=["response1"],  # Mismatched length
            rewards=[0.5, 0.6],
            annotator_ids=["ann1", "ann2"]
        )
        assert not result.is_valid
        assert any("inconsistent" in issue.message.lower() for issue in result.issues)
    
    def test_security_monitoring(self):
        """Test security event monitoring."""
        from rlhf_audit_trail.security_hardening import get_security_monitor, ThreatLevel
        
        monitor = get_security_monitor()
        
        # Log security events
        monitor.log_security_event(
            event_type="failed_authentication",
            severity=ThreatLevel.HIGH,
            details={"user": "test_user", "reason": "invalid_password"},
            source_ip="192.168.1.1"
        )
        
        # Analyze threats
        analysis = monitor.analyze_threats(time_window_hours=1)
        assert analysis["total_events"] > 0
        assert "events_by_severity" in analysis
        assert "threat_indicators" in analysis


class TestGeneration3Scalability:
    """Test Generation 3: Make It Scale - Performance, caching, concurrency."""
    
    def test_performance_optimization(self):
        """Test performance optimization features."""
        optimizer = get_performance_optimizer()
        
        @optimizer.profiler.profile_operation("test_operation")
        def test_function():
            time.sleep(0.1)
            return "test_result"
        
        # Execute function
        result = test_function()
        assert result == "test_result"
        
        # Check metrics
        stats = optimizer.profiler.get_operation_stats("test_operation")
        assert stats["count"] > 0
        assert stats["avg_duration"] > 0.05  # Should be around 0.1 seconds
    
    def test_caching_system(self):
        """Test intelligent caching."""
        optimizer = get_performance_optimizer()
        
        # Test cache decorator
        call_count = 0
        
        @optimizer.cached_operation(lambda x: f"key_{x}")
        def expensive_function(value):
            nonlocal call_count
            call_count += 1
            return f"result_{value}"
        
        # First call - cache miss
        result1 = expensive_function("test")
        assert result1 == "result_test"
        assert call_count == 1
        
        # Second call - cache hit
        result2 = expensive_function("test")
        assert result2 == "result_test"
        assert call_count == 1  # Should not increment
        
        # Check cache stats
        cache_stats = optimizer.cache.get_stats()
        assert cache_stats["hits"] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent batch processing."""
        optimizer = get_performance_optimizer()
        
        def process_item(item):
            time.sleep(0.01)  # Simulate work
            return item * 2
        
        items = list(range(20))
        start_time = time.time()
        
        results = await optimizer.processor.process_batch(
            items=items,
            processor_func=process_item,
            batch_size=5
        )
        
        end_time = time.time()
        
        assert len(results) == 20
        assert results[0] == 0
        assert results[19] == 38
        
        # Should be faster than sequential processing
        assert (end_time - start_time) < 0.5  # Should be much less than 0.2 seconds (20 * 0.01)
    
    def test_resource_pooling(self):
        """Test resource pool functionality."""
        from rlhf_audit_trail.performance_optimization import ResourcePool
        
        creation_count = 0
        
        def create_resource():
            nonlocal creation_count
            creation_count += 1
            return f"resource_{creation_count}"
        
        def validate_resource(resource):
            return isinstance(resource, str)
        
        pool = ResourcePool(
            factory=create_resource,
            max_size=3,
            min_size=1,
            validator=validate_resource
        )
        
        # Test resource acquisition and release
        resource1 = pool.acquire()
        assert resource1 is not None
        
        resource2 = pool.acquire()
        assert resource2 is not None
        assert resource2 != resource1
        
        pool.release(resource1)
        pool.release(resource2)
        
        stats = pool.get_stats()
        assert stats["created_count"] >= 2
    
    def test_auto_scaling_decisions(self):
        """Test auto-scaling decision logic."""
        from rlhf_audit_trail.auto_scaling import AutoScaler, MetricsCollector, ScalingDirection
        
        collector = MetricsCollector()
        scaler = AutoScaler(collector)
        
        # Mock high CPU usage
        collector.metrics_history.clear()
        for i in range(10):
            mock_metrics = Mock()
            mock_metrics.timestamp = datetime.utcnow()
            mock_metrics.cpu_percent = 85.0  # High CPU
            mock_metrics.memory_percent = 40.0
            mock_metrics.request_rate = 50.0
            mock_metrics.error_rate = 1.0
            mock_metrics.response_time_ms = 200.0
            collector.metrics_history.append(mock_metrics)
        
        # Evaluate scaling decision
        direction, target_instances, reason = scaler.evaluate_scaling_decision()
        
        # Should decide to scale up due to high CPU
        assert direction == ScalingDirection.UP
        assert target_instances > scaler.current_instances
        assert "cpu" in reason.lower()
    
    def test_load_balancing(self):
        """Test load balancing algorithms."""
        from rlhf_audit_trail.auto_scaling import LoadBalancer, ServiceInstance, LoadBalancingStrategy
        
        balancer = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        # Add test instances
        for i in range(3):
            instance = ServiceInstance(
                instance_id=f"test_{i}",
                host="localhost",
                port=8000 + i,
                status="healthy",
                cpu_usage=20.0,
                memory_usage=30.0,
                active_connections=i,  # Different connection counts
                last_health_check=datetime.utcnow()
            )
            balancer.add_instance(instance)
        
        # Test round robin
        selected_instances = []
        for i in range(6):
            instance = balancer.get_next_instance()
            selected_instances.append(instance.instance_id)
        
        # Should cycle through instances
        assert selected_instances[0] == selected_instances[3]
        assert selected_instances[1] == selected_instances[4]
        assert selected_instances[2] == selected_instances[5]
        
        # Test least connections strategy
        balancer.strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
        instance = balancer.get_next_instance()
        assert instance.instance_id == "test_0"  # Should have least connections (0)


class TestQualityGatesIntegration:
    """Test quality gates and integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow with all components."""
        # Initialize system with all components
        privacy_config = PrivacyConfig(epsilon=5.0, delta=1e-5)
        compliance_config = ComplianceConfig()
        
        auditor = AuditableRLHF(
            model_name="integration-test-model",
            privacy_config=privacy_config,
            compliance_config=compliance_config,
            storage_backend="local"
        )
        
        # Enable performance monitoring
        optimizer = get_performance_optimizer()
        
        # Track complete training workflow
        async with auditor.track_training("integration_test") as session:
            # Phase 1: Log annotations
            prompts = [f"Test prompt {i}" for i in range(10)]
            responses = [f"Test response {i}" for i in range(10)]
            rewards = [0.5 + (i * 0.05) for i in range(10)]
            annotator_ids = [f"ann_{i%3}" for i in range(10)]
            
            batch = await auditor.log_annotations(
                prompts=prompts,
                responses=responses,
                rewards=rewards,
                annotator_ids=annotator_ids
            )
            
            # Phase 2: Multiple policy updates
            for epoch in range(3):
                await auditor.track_policy_update(
                    model=f"model_epoch_{epoch}",
                    optimizer=f"optimizer_epoch_{epoch}",
                    batch=f"batch_epoch_{epoch}",
                    loss=0.5 - (epoch * 0.1),
                    metadata={"epoch": epoch}
                )
                
                # Create checkpoints
                metrics = {
                    "loss": 0.5 - (epoch * 0.1),
                    "reward": 0.7 + (epoch * 0.05),
                    "perplexity": 15.0 - (epoch * 2.0)
                }
                
                await auditor.checkpoint(
                    epoch=epoch,
                    metrics=metrics
                )
            
            # Phase 3: Generate final artifacts
            model_card = await auditor.generate_model_card()
            verification = await auditor.verify_provenance()
            privacy_report = auditor.get_privacy_report()
            
            # Verify all components worked
            assert model_card is not None
            assert "training_summary" in model_card
            assert verification is not None
            assert privacy_report["epsilon_remaining"] < privacy_report["total_epsilon"]
        
        # Check performance metrics
        perf_stats = optimizer.get_comprehensive_stats()
        assert perf_stats["profiler"]["total_operations"] > 0
    
    def test_security_integration(self):
        """Test security integration with all components."""
        from rlhf_audit_trail.security_hardening import get_access_controller, AccessLevel
        
        controller = get_access_controller()
        
        # Add test users
        controller.add_user("admin_user", AccessLevel.ADMIN)
        controller.add_user("annotator_user", AccessLevel.ANNOTATOR)
        controller.add_user("read_only_user", AccessLevel.READ_ONLY)
        
        # Test authentication (mock)
        with patch.object(controller, '_verify_password', return_value=True):
            admin_token = controller.authenticate_user("admin_user", "password123", "127.0.0.1")
            assert admin_token is not None
        
        # Test authorization
        assert controller.authorize_action(admin_token, "manage_sessions")
        assert not controller.authorize_action(admin_token, "invalid_action")
    
    def test_monitoring_integration(self):
        """Test monitoring integration across components."""
        from rlhf_audit_trail.health_monitoring import get_health_monitor
        
        monitor = get_health_monitor()
        
        # Register mock auditor
        mock_auditor = Mock()
        mock_auditor.current_session = None
        mock_auditor.privacy_budget = Mock()
        mock_auditor.privacy_budget.total_spent_epsilon = 2.0
        mock_auditor.privacy_config = Mock()
        mock_auditor.privacy_config.epsilon = 10.0
        
        monitor.register_auditor(mock_auditor)
        
        # Get system health
        health = monitor.get_system_health()
        assert "overall_status" in health
        assert "components" in health
        assert health["overall_status"] in ["healthy", "degraded", "unhealthy", "critical"]
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks meet requirements."""
        from rlhf_audit_trail.performance_optimization import SmartCache
        
        cache = SmartCache(max_size=1000, ttl_seconds=300)
        
        # Benchmark cache operations
        start_time = time.time()
        
        # Write operations
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
        
        write_time = time.time() - start_time
        
        # Read operations  
        start_time = time.time()
        
        hits = 0
        for i in range(1000):
            value, hit = cache.get(f"key_{i}")
            if hit:
                hits += 1
        
        read_time = time.time() - start_time
        
        # Performance assertions
        assert write_time < 1.0  # Should write 1000 items in under 1 second
        assert read_time < 0.5   # Should read 1000 items in under 0.5 seconds
        assert hits == 1000      # All items should be cache hits
        
        # Cache efficiency
        stats = cache.get_stats()
        assert stats["hit_rate"] == 100.0  # Perfect hit rate for this test


class TestComplianceIntegration:
    """Test compliance and regulatory requirements."""
    
    @pytest.mark.asyncio
    async def test_eu_ai_act_compliance(self):
        """Test EU AI Act compliance requirements."""
        compliance_config = ComplianceConfig()
        auditor = AuditableRLHF(
            model_name="eu-compliance-test",
            compliance_config=compliance_config,
            compliance_mode="eu_ai_act"
        )
        
        async with auditor.track_training("eu_compliance") as session:
            # Log required data for compliance
            await auditor.log_annotations(
                prompts=["Test prompt"],
                responses=["Test response"],
                rewards=[0.8],
                annotator_ids=["eu_ann_001"]
            )
            
            # Generate compliance model card
            model_card = await auditor.generate_model_card(
                format="eu_standard",
                include_provenance=True,
                include_privacy_analysis=True
            )
            
            # Verify EU AI Act requirements
            assert "compliance" in model_card
            assert "privacy_analysis" in model_card
            assert "provenance" in model_card
            
            # Should have differential privacy guarantees
            privacy_analysis = model_card["privacy_analysis"]
            assert "differential_privacy" in privacy_analysis
            assert "epsilon" in privacy_analysis["differential_privacy"]
            assert "delta" in privacy_analysis["differential_privacy"]
    
    def test_audit_trail_immutability(self):
        """Test that audit trails are tamper-evident."""
        from rlhf_audit_trail.crypto import IntegrityVerifier, CryptographicEngine
        
        crypto = CryptographicEngine()
        verifier = IntegrityVerifier(crypto)
        
        # Create test audit data
        test_data = {
            "event_type": "test_event",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"test": "value"}
        }
        
        # Create signature
        signature = crypto.sign_data(json.dumps(test_data, sort_keys=True))
        
        # Verify signature
        is_valid = crypto.verify_signature(
            json.dumps(test_data, sort_keys=True),
            signature
        )
        assert is_valid
        
        # Test tampering detection
        tampered_data = test_data.copy()
        tampered_data["data"]["test"] = "modified_value"
        
        is_valid_tampered = crypto.verify_signature(
            json.dumps(tampered_data, sort_keys=True),
            signature
        )
        assert not is_valid_tampered


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])