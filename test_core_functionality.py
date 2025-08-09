#!/usr/bin/env python3
"""Quick validation test for core RLHF audit trail functionality."""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test that core modules can be imported."""
    print("🧪 Testing basic imports...")
    
    try:
        from rlhf_audit_trail.crypto import CryptographicEngine
        from rlhf_audit_trail.storage import LocalStorage
        print("✅ Core crypto and storage modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_crypto_functionality():
    """Test basic cryptographic functionality."""
    print("\n🔒 Testing cryptographic functionality...")
    
    try:
        from rlhf_audit_trail.crypto import CryptographicEngine
        
        crypto = CryptographicEngine()
        
        # Test hashing
        hash1 = crypto.hash_data("test data")
        hash2 = crypto.hash_data("test data")
        assert hash1 == hash2, "Hash should be deterministic"
        assert len(hash1) == 64, "SHA-256 hash should be 64 hex chars"
        
        # Test signing
        signature = crypto.sign_data("test message")
        verified = crypto.verify_signature("test message", signature)
        assert verified, "Signature verification should pass"
        
        # Test encryption
        encrypted_data, _ = crypto.encrypt_data("secret", "password123")
        decrypted = crypto.decrypt_data(encrypted_data, "password123")
        assert decrypted.decode() == "secret", "Decryption should return original data"
        
        print("✅ Cryptographic functions working correctly")
        return True
    except Exception as e:
        print(f"❌ Crypto test failed: {e}")
        traceback.print_exc()
        return False

def test_storage_functionality():
    """Test basic storage functionality."""
    print("\n💾 Testing storage functionality...")
    
    try:
        from rlhf_audit_trail.storage import LocalStorage
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(base_path=tmpdir)
            
            # Test store and retrieve
            test_data = {"key": "value", "number": 42}
            assert storage.store("test_key", test_data), "Store should succeed"
            
            retrieved = storage.retrieve("test_key")
            assert retrieved is not None, "Retrieve should return data"
            
            # Test existence
            assert storage.exists("test_key"), "Key should exist"
            assert not storage.exists("nonexistent"), "Nonexistent key should not exist"
            
            # Test delete
            assert storage.delete("test_key"), "Delete should succeed"
            assert not storage.exists("test_key"), "Key should not exist after deletion"
        
        print("✅ Storage functions working correctly")
        return True
    except Exception as e:
        print(f"❌ Storage test failed: {e}")
        traceback.print_exc()
        return False

def test_model_card_generation():
    """Test model card generation."""
    print("\n📋 Testing model card generation...")
    
    try:
        from rlhf_audit_trail.model_card import (
            ModelCardGenerator, ModelCardFormat, ModelMetadata, 
            TrainingProvenance, RegulatoryCompliance
        )
        from datetime import datetime
        
        generator = ModelCardGenerator()
        
        # Create test metadata
        model_metadata = ModelMetadata(
            name="test-model",
            version="1.0",
            architecture="transformer",
            parameters=7000000,
            training_data={"source": "test", "size": "1k"},
            intended_use="Testing",
            limitations=["Test only"],
            performance_metrics={"accuracy": 0.85},
            ethical_considerations=["Test consideration"],
            creation_date=datetime.now(),
            last_updated=datetime.now()
        )
        
        training_provenance = TrainingProvenance(
            training_duration=3600.0,
            total_annotations=100,
            annotator_count=5,
            policy_updates=10,
            checkpoint_count=5,
            privacy_budget_used=8.5,
            compliance_score=95.0,
            verification_status=True
        )
        
        compliance_info = RegulatoryCompliance(
            frameworks=["eu_ai_act"],
            requirements_met={"test": True},
            compliance_scores={"eu_ai_act": 95.0},
            risk_assessment={"level": "high"},
            mitigation_measures=["Testing"],
            audit_trail_location="/test/path"
        )
        
        # Generate model card
        model_card = generator.generate_model_card(
            model_metadata=model_metadata,
            training_provenance=training_provenance,
            compliance_info=compliance_info,
            format_type=ModelCardFormat.EU_AI_ACT
        )
        
        assert "test-model" in model_card, "Model name should be in card"
        assert "EU AI Act" in model_card, "Framework should be mentioned"
        
        print("✅ Model card generation working correctly")
        return True
    except Exception as e:
        print(f"❌ Model card test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration classes."""
    print("\n⚙️  Testing configuration...")
    
    try:
        from rlhf_audit_trail.config import PrivacyConfig, SecurityConfig, ComplianceConfig
        
        privacy_config = PrivacyConfig(epsilon=10.0, delta=1e-5)
        security_config = SecurityConfig()
        compliance_config = ComplianceConfig()
        
        assert privacy_config.epsilon == 10.0, "Privacy config should store epsilon"
        assert hasattr(security_config, 'key_size'), "Security config should have key_size"
        assert hasattr(compliance_config, 'frameworks'), "Compliance config should have frameworks"
        
        print("✅ Configuration classes working correctly")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("🚀 RLHF Audit Trail - Core Functionality Validation")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_crypto_functionality,
        test_storage_functionality,
        test_model_card_generation,
        test_configuration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core functionality tests PASSED!")
        return 0
    else:
        print("⚠️  Some tests failed - check logs above")
        return 1

if __name__ == "__main__":
    sys.exit(main())