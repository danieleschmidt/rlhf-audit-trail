"""Comprehensive integration tests for the RLHF audit trail system."""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlhf_audit_trail.core import AuditableRLHF, TrainingPhase
from rlhf_audit_trail.config import PrivacyConfig, SecurityConfig, ComplianceConfig
from rlhf_audit_trail.crypto import CryptographicEngine, MerkleTree
from rlhf_audit_trail.storage import LocalStorage, create_storage_backend
from rlhf_audit_trail.monitoring import MetricsCollector, MonitoringManager
from rlhf_audit_trail.performance import PerformanceOptimizer, PerformanceConfig, CacheConfig
from rlhf_audit_trail.model_card import ModelCardGenerator, ModelCardFormat, ModelMetadata, TrainingProvenance, RegulatoryCompliance
from rlhf_audit_trail.integrations import ModelExtractor
from rlhf_audit_trail.exceptions import AuditTrailError, PrivacyBudgetExceededError


class TestCryptographicEngine:
    """Test cryptographic operations."""
    
    def test_key_generation(self):
        """Test RSA key generation."""
        crypto = CryptographicEngine()
        assert crypto._private_key is not None
        assert crypto._public_key is not None
    
    def test_data_hashing(self):
        """Test data hashing."""
        crypto = CryptographicEngine()
        
        # Test string hashing
        hash1 = crypto.hash_data("test data")
        hash2 = crypto.hash_data("test data")
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
        
        # Test dict hashing (should be deterministic)
        data = {"key": "value", "number": 42}
        hash3 = crypto.hash_data(data)
        hash4 = crypto.hash_data(data)
        assert hash3 == hash4
    
    def test_signing_and_verification(self):
        """Test digital signatures."""
        crypto = CryptographicEngine()
        data = "test message for signing"
        
        # Sign data
        signature = crypto.sign_data(data)
        assert signature is not None
        assert len(signature) > 0
        
        # Verify signature
        assert crypto.verify_signature(data, signature) is True
        
        # Verify with wrong data should fail
        assert crypto.verify_signature("wrong data", signature) is False
        
        # Verify with corrupted signature should fail
        corrupted_sig = signature[:-10] + "0" * 10
        assert crypto.verify_signature(data, corrupted_sig) is False
    
    def test_encryption_decryption(self):
        """Test data encryption and decryption."""
        crypto = CryptographicEngine()
        data = "sensitive test data"
        password = "test_password_123"
        
        # Encrypt data
        encrypted_bundle, _ = crypto.encrypt_data(data, password)
        assert encrypted_bundle != data.encode()
        assert len(encrypted_bundle) > len(data)
        
        # Decrypt data
        decrypted = crypto.decrypt_data(encrypted_bundle, password)
        assert decrypted.decode() == data
        
        # Wrong password should fail
        with pytest.raises(Exception):
            crypto.decrypt_data(encrypted_bundle, "wrong_password")


class TestMerkleTree:
    """Test Merkle tree implementation."""
    
    def test_empty_tree(self):
        """Test empty Merkle tree."""
        crypto = CryptographicEngine()
        tree = MerkleTree(crypto)
        assert tree.get_root_hash() is None
    
    def test_single_leaf(self):
        """Test tree with single leaf."""
        crypto = CryptographicEngine()
        tree = MerkleTree(crypto)
        
        leaf_hash = tree.add_leaf("test data")
        assert leaf_hash is not None
        assert tree.get_root_hash() == leaf_hash
    
    def test_multiple_leaves(self):
        """Test tree with multiple leaves."""
        crypto = CryptographicEngine()
        tree = MerkleTree(crypto)
        
        data = ["leaf1", "leaf2", "leaf3", "leaf4"]
        leaf_hashes = []
        
        for item in data:
            leaf_hash = tree.add_leaf(item)
            leaf_hashes.append(leaf_hash)
        
        # Root should be computed
        root = tree.get_root_hash()
        assert root is not None
        
        # Generate and verify proofs
        for leaf_hash in leaf_hashes:
            proof = tree.generate_proof(leaf_hash)
            assert proof is not None
            assert tree.verify_proof(proof) is True


class TestStorageBackends:
    """Test storage backend implementations."""
    
    def test_local_storage(self):
        """Test local filesystem storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(base_path=tmpdir)
            
            # Test storing data
            test_data = {"key": "value", "number": 42}
            assert storage.store("test_key", test_data) is True
            
            # Test retrieving data
            retrieved = storage.retrieve("test_key")
            assert retrieved is not None
            assert json.loads(retrieved) == test_data
            
            # Test existence check
            assert storage.exists("test_key") is True
            assert storage.exists("nonexistent") is False
            
            # Test listing keys
            keys = storage.list_keys()
            assert "test_key" in keys
            
            # Test deletion
            assert storage.delete("test_key") is True
            assert storage.exists("test_key") is False
    
    def test_storage_factory(self):
        """Test storage backend factory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = create_storage_backend("local", base_path=tmpdir)
            assert isinstance(storage, LocalStorage)
            
            # Test invalid backend
            with pytest.raises(ValueError):
                create_storage_backend("invalid_backend")


class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    def test_memory_cache(self):
        """Test memory cache functionality."""
        from rlhf_audit_trail.performance import MemoryCache
        
        cache = MemoryCache(max_size=3, ttl_seconds=1)
        
        # Test basic operations
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
        
        # Test eviction
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        
        # Test stats
        stats = cache.get_stats()
        assert stats['size'] == 3
        assert stats['hits'] > 0
        assert stats['misses'] > 0
    
    @pytest.mark.asyncio
    async def test_performance_optimizer(self):
        """Test performance optimizer."""
        config = PerformanceConfig(
            cache_config=CacheConfig(enabled=True, redis_url=None),
            batch_size=10
        )
        
        optimizer = PerformanceOptimizer(config)
        
        # Test cache operations
        cache_key = optimizer.cache_key("test", key="value")
        assert len(cache_key) == 64  # SHA-256 hex
        
        # Test caching
        await optimizer.cache.set("test_key", {"data": "value"})
        cached = await optimizer.cache.get("test_key")
        assert cached == {"data": "value"}
        
        # Test stats
        stats = optimizer.get_performance_stats()
        assert "cache_stats" in stats
        assert "optimization_metrics" in stats


class TestMonitoring:
    """Test monitoring and metrics collection."""
    
    def test_metrics_collector(self):
        """Test metrics collection."""
        collector = MetricsCollector(collection_interval=1, retention_hours=1)
        
        # Test operation timing
        with collector.time_operation("test_operation", {"param": "value"}):
            pass  # Simulate work
        
        # Check performance metrics
        summary = collector.get_performance_summary()
        assert "test_operation" in summary
        assert summary["test_operation"]["total_calls"] == 1
        assert summary["test_operation"]["success_rate"] == 100.0
    
    def test_monitoring_manager(self):
        """Test monitoring manager."""
        manager = MonitoringManager()
        
        # Test status
        status = manager.get_comprehensive_status()
        assert "system" in status or status == {}  # May be empty if psutil not available
        assert "performance" in status
        assert "timestamp" in status


class TestModelCardGeneration:
    """Test model card generation."""
    
    def test_model_card_creation(self):
        """Test model card generation."""
        generator = ModelCardGenerator()
        
        # Create test metadata
        model_metadata = ModelMetadata(
            name="test-model",
            version="1.0",
            architecture="transformer",
            parameters=7000000,
            training_data={"source": "custom", "size": "10k"},
            intended_use="Research and testing",
            limitations=["Not for production", "Limited domain"],
            performance_metrics={"accuracy": 0.85, "safety_score": 0.92},
            ethical_considerations=["Bias mitigation applied", "Privacy preserved"],
            creation_date=datetime.now(),
            last_updated=datetime.now()
        )
        
        training_provenance = TrainingProvenance(
            training_duration=3600.0,
            total_annotations=1000,
            annotator_count=5,
            policy_updates=50,
            checkpoint_count=10,
            privacy_budget_used=8.5,
            compliance_score=95.2,
            verification_status=True
        )
        
        compliance_info = RegulatoryCompliance(
            frameworks=["eu_ai_act"],
            requirements_met={"risk_management": True, "transparency": True},
            compliance_scores={"eu_ai_act": 95.2},
            risk_assessment={"classification": "high-risk"},
            mitigation_measures=["Human oversight", "Continuous monitoring"],
            audit_trail_location="/audit/session_123"
        )
        
        # Generate model card
        model_card = generator.generate_model_card(
            model_metadata=model_metadata,
            training_provenance=training_provenance,
            compliance_info=compliance_info,
            format_type=ModelCardFormat.EU_AI_ACT,
            output_format="markdown"
        )
        
        assert isinstance(model_card, str)
        assert "test-model" in model_card
        assert "EU AI Act" in model_card
        assert "95.2%" in model_card
        
        # Test validation
        validation = generator.validate_model_card(model_card, ModelCardFormat.EU_AI_ACT)
        assert validation["is_valid"] is True or len(validation["missing_sections"]) < 3


class TestIntegrations:
    """Test ML library integrations."""
    
    def test_model_extractor(self):
        """Test model metadata extraction."""
        # Mock PyTorch model
        class MockModel:
            def __init__(self):
                self.param_count = 1000
            
            def parameters(self):
                return []
        
        model = MockModel()
        metadata = ModelExtractor.extract_model_metadata(model)
        
        assert "model_type" in metadata
        assert metadata["model_type"] == "MockModel"
        assert "timestamp" in metadata


@pytest.mark.asyncio
class TestAuditableRLHF:
    """Test main AuditableRLHF system integration."""
    
    async def test_basic_initialization(self):
        """Test basic system initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            privacy_config = PrivacyConfig(epsilon=10.0, delta=1e-5)
            
            auditor = AuditableRLHF(
                model_name="test-model",
                privacy_config=privacy_config,
                storage_backend="local",
                storage_config={"base_path": tmpdir}
            )
            
            assert auditor.model_name == "test-model"
            assert auditor.privacy_config.epsilon == 10.0
            assert auditor.current_session is None
    
    async def test_training_session_lifecycle(self):
        """Test complete training session lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            privacy_config = PrivacyConfig(epsilon=10.0, delta=1e-5)
            
            # Mock the components that require external dependencies
            with patch('rlhf_audit_trail.audit.AuditLogger'), \
                 patch('rlhf_audit_trail.privacy.DifferentialPrivacyEngine'), \
                 patch('rlhf_audit_trail.compliance.ComplianceValidator'):
                
                auditor = AuditableRLHF(
                    model_name="test-model",
                    privacy_config=privacy_config,
                    storage_backend="local",
                    storage_config={"base_path": tmpdir}
                )
                
                # Mock the dependencies
                auditor.audit_logger = MagicMock()
                auditor.audit_logger.log_event = AsyncMock()
                auditor.privacy_engine = MagicMock()
                auditor.privacy_engine.estimate_epsilon_cost.return_value = 0.1
                auditor.privacy_engine.add_noise_to_rewards.return_value = [0.1, 0.2, 0.3]
                auditor.privacy_budget = MagicMock()
                auditor.privacy_budget.can_spend.return_value = True
                auditor.privacy_budget.remaining_epsilon = 9.9
                auditor.compliance_validator = MagicMock()
                auditor.compliance_validator.validate_session = AsyncMock(return_value={})
                auditor.compliance_validator.validate_checkpoint = AsyncMock(return_value=MagicMock(is_compliant=True))
                auditor.compliance_validator.generate_final_report = AsyncMock(return_value={})
                auditor.verifier = MagicMock()
                auditor.verifier.verify_session_integrity = AsyncMock(return_value={'is_valid': True})
                auditor.storage = MagicMock()
                auditor.storage.store_encrypted = AsyncMock()
                
                # Test training session
                async with auditor.track_training("test-experiment") as session:
                    assert session.experiment_name == "test-experiment"
                    assert session.model_name == "test-model"
                    assert session.phase == TrainingPhase.INITIALIZATION
                    assert auditor.current_session == session
                    
                    # Test logging annotations
                    batch = await auditor.log_annotations(
                        prompts=["prompt1", "prompt2"],
                        responses=["response1", "response2"],
                        rewards=[0.8, 0.9],
                        annotator_ids=["annotator1", "annotator2"]
                    )
                    
                    assert batch.batch_size == 2
                    assert len(batch.prompts) == 2
                    
                    # Test policy update tracking
                    update = await auditor.track_policy_update(
                        model="mock_model",
                        optimizer="mock_optimizer", 
                        batch="mock_batch",
                        loss=0.5
                    )
                    
                    assert update.loss == 0.5
                    assert update.step_number == 1
                    
                    # Test checkpointing
                    await auditor.checkpoint(
                        epoch=1,
                        metrics={"loss": 0.5, "accuracy": 0.85}
                    )
                    
                    # Test model card generation
                    model_card = await auditor.generate_model_card()
                    assert "test-model" in model_card
                    assert "test-experiment" in model_card
                    
                    # Test provenance verification
                    verification = await auditor.verify_provenance()
                    assert verification["is_valid"] is True
                
                # Session should be completed
                assert auditor.current_session is None
    
    async def test_privacy_budget_enforcement(self):
        """Test privacy budget enforcement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            privacy_config = PrivacyConfig(epsilon=1.0, delta=1e-5)  # Small budget
            
            with patch('rlhf_audit_trail.audit.AuditLogger'), \
                 patch('rlhf_audit_trail.privacy.DifferentialPrivacyEngine'), \
                 patch('rlhf_audit_trail.compliance.ComplianceValidator'):
                
                auditor = AuditableRLHF(
                    model_name="test-model",
                    privacy_config=privacy_config,
                    storage_backend="local",
                    storage_config={"base_path": tmpdir}
                )
                
                # Mock budget exhaustion
                auditor.privacy_budget = MagicMock()
                auditor.privacy_budget.can_spend.return_value = False
                
                async with auditor.track_training("test-experiment"):
                    # This should raise privacy budget exceeded error
                    with pytest.raises(PrivacyBudgetExceededError):
                        await auditor.log_annotations(
                            prompts=["prompt1"],
                            responses=["response1"],
                            rewards=[0.8],
                            annotator_ids=["annotator1"]
                        )


def test_comprehensive_system_integration():
    """Test high-level system integration."""
    # This test ensures all major components can be imported and initialized
    try:
        from rlhf_audit_trail.core import AuditableRLHF
        from rlhf_audit_trail.config import PrivacyConfig
        from rlhf_audit_trail.crypto import CryptographicEngine
        from rlhf_audit_trail.storage import LocalStorage
        from rlhf_audit_trail.monitoring import get_monitor
        from rlhf_audit_trail.performance import get_performance_optimizer
        from rlhf_audit_trail.model_card import ModelCardGenerator
        
        # Test basic initialization
        crypto = CryptographicEngine()
        storage = LocalStorage()
        monitor = get_monitor()
        optimizer = get_performance_optimizer()
        generator = ModelCardGenerator()
        
        assert crypto is not None
        assert storage is not None
        assert monitor is not None
        assert optimizer is not None
        assert generator is not None
        
    except ImportError as e:
        pytest.fail(f"Import error in system integration: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])