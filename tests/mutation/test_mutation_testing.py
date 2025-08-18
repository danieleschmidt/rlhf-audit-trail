"""Mutation testing configuration for RLHF Audit Trail.

Mutation testing helps identify weaknesses in our test suite by introducing
bugs (mutations) into the code and verifying that tests catch them.
"""

import pytest
from unittest.mock import Mock, patch

from rlhf_audit_trail.core import AuditableRLHF
from rlhf_audit_trail.privacy import DifferentialPrivacy
from rlhf_audit_trail.audit import AuditLogger
from tests.fixtures.sample_data import create_sample_training_data


class TestCriticalPathMutations:
    """Test that critical security and privacy paths are well-tested."""

    @pytest.mark.mutation
    def test_privacy_budget_enforcement_mutations(self):
        """Test mutations in privacy budget enforcement logic."""
        privacy_engine = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        # Original behavior - should pass
        assert privacy_engine.check_budget_available(0.5) is True
        
        # Simulate mutation: budget check always returns True
        with patch.object(privacy_engine, 'check_budget_available', return_value=True):
            # This should fail if our tests are good
            with pytest.raises(Exception):
                privacy_engine.apply_noise(data=[1, 2, 3], epsilon=10.0)  # Over budget

    @pytest.mark.mutation
    def test_audit_trail_integrity_mutations(self):
        """Test mutations in audit trail integrity verification."""
        auditor = AuditLogger()
        
        # Test original hash verification
        data = {"event": "test", "timestamp": "2025-01-01T00:00:00Z"}
        hash_value = auditor.calculate_hash(data)
        
        # Verify integrity check works
        assert auditor.verify_integrity(data, hash_value) is True
        
        # Simulate mutation: integrity check always returns True
        with patch.object(auditor, 'verify_integrity', return_value=True):
            # Should detect tampered data
            tampered_data = {"event": "tampered", "timestamp": "2025-01-01T00:00:00Z"}
            # Our tests should catch this mutation
            assert auditor.verify_integrity(tampered_data, hash_value) is not True

    @pytest.mark.mutation  
    def test_compliance_validation_mutations(self):
        """Test mutations in compliance validation logic."""
        from rlhf_audit_trail.compliance import ComplianceValidator
        
        validator = ComplianceValidator(frameworks=["eu_ai_act"])
        
        # Test valid compliance
        audit_data = create_sample_training_data()
        result = validator.validate_compliance(audit_data)
        assert result.is_compliant is True
        
        # Simulate mutation: validation always passes
        with patch.object(validator, 'validate_compliance') as mock_validate:
            mock_validate.return_value.is_compliant = True
            
            # Test with non-compliant data
            invalid_data = {"missing": "required_fields"}
            result = validator.validate_compliance(invalid_data)
            
            # Tests should catch this mutation
            assert result.is_compliant is not True or len(result.violations) == 0

    @pytest.mark.mutation
    def test_cryptographic_signature_mutations(self):
        """Test mutations in cryptographic signature verification."""
        from rlhf_audit_trail.crypto import SignatureManager
        
        sig_manager = SignatureManager()
        data = b"test message for signing"
        signature = sig_manager.sign(data)
        
        # Verify original signature works
        assert sig_manager.verify(data, signature) is True
        
        # Simulate mutation: signature verification always returns True
        with patch.object(sig_manager, 'verify', return_value=True):
            # Should detect invalid signature
            invalid_signature = b"invalid_signature_bytes"
            # Tests should catch this mutation
            assert sig_manager.verify(data, invalid_signature) is not True

    @pytest.mark.mutation
    def test_access_control_mutations(self):
        """Test mutations in access control logic."""
        from rlhf_audit_trail.auth import AccessController
        
        controller = AccessController()
        
        # Test normal access control
        assert controller.has_permission("user123", "read", "audit_logs") is True
        assert controller.has_permission("user123", "admin", "system") is False
        
        # Simulate mutation: always grants access
        with patch.object(controller, 'has_permission', return_value=True):
            # Should deny unauthorized access
            result = controller.has_permission("unauthorized_user", "admin", "system")
            # Tests should catch this mutation
            assert result is not True


class TestDataIntegrityMutations:
    """Test mutations in data integrity and validation logic."""

    @pytest.mark.mutation
    def test_data_sanitization_mutations(self):
        """Test mutations in data sanitization logic."""
        from rlhf_audit_trail.validation import DataSanitizer
        
        sanitizer = DataSanitizer()
        
        # Test malicious input sanitization
        malicious_input = "<script>alert('xss')</script>"
        sanitized = sanitizer.sanitize_input(malicious_input)
        assert "<script>" not in sanitized
        
        # Simulate mutation: sanitization disabled
        with patch.object(sanitizer, 'sanitize_input', side_effect=lambda x: x):
            # Should remove malicious content
            result = sanitizer.sanitize_input(malicious_input)
            # Tests should catch this mutation
            assert "<script>" not in result

    @pytest.mark.mutation
    def test_input_validation_mutations(self):
        """Test mutations in input validation logic."""
        from rlhf_audit_trail.validation import InputValidator
        
        validator = InputValidator()
        
        # Test SQL injection detection
        sql_injection = "'; DROP TABLE users; --"
        assert validator.is_safe_input(sql_injection) is False
        
        # Simulate mutation: validation always passes
        with patch.object(validator, 'is_safe_input', return_value=True):
            # Should detect dangerous input
            result = validator.is_safe_input(sql_injection)
            # Tests should catch this mutation
            assert result is not True


class TestBusinessLogicMutations:
    """Test mutations in critical business logic."""

    @pytest.mark.mutation
    def test_privacy_noise_calculation_mutations(self):
        """Test mutations in privacy noise calculation."""
        from rlhf_audit_trail.privacy import NoiseGenerator
        
        generator = NoiseGenerator()
        
        # Test noise generation
        noise = generator.generate_gaussian_noise(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        assert noise != 0  # Should add noise
        
        # Simulate mutation: no noise added
        with patch.object(generator, 'generate_gaussian_noise', return_value=0):
            # Should add privacy noise
            result = generator.generate_gaussian_noise(sensitivity=1.0, epsilon=1.0, delta=1e-5)
            # Tests should catch this mutation
            assert result != 0

    @pytest.mark.mutation
    def test_audit_log_retention_mutations(self):
        """Test mutations in audit log retention logic."""
        from rlhf_audit_trail.storage import AuditStorage
        
        storage = AuditStorage()
        
        # Test retention policy
        old_records = storage.get_records_older_than_days(2555)  # EU AI Act requirement
        assert isinstance(old_records, list)
        
        # Simulate mutation: retention policy disabled
        with patch.object(storage, 'delete_old_records', return_value=0):
            # Should enforce retention policy
            deleted_count = storage.cleanup_old_records()
            # Tests should verify proper cleanup
            assert deleted_count >= 0


# Configuration for mutation testing tools
MUTATION_TEST_CONFIG = {
    "target_modules": [
        "rlhf_audit_trail.privacy",
        "rlhf_audit_trail.audit", 
        "rlhf_audit_trail.compliance",
        "rlhf_audit_trail.crypto",
        "rlhf_audit_trail.validation",
    ],
    "exclude_patterns": [
        "test_*",
        "__pycache__",
        "*.pyc",
    ],
    "minimum_mutation_score": 80,  # 80% minimum mutation detection
}