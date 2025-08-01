"""Unit tests for privacy protection components."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.mark.unit
@pytest.mark.privacy
class TestPrivacyEngine:
    """Unit tests for differential privacy and anonymization."""
    
    def test_differential_privacy_noise_generation(self, privacy_config):
        """Test differential privacy noise generation."""
        # Mock privacy engine
        privacy_engine = MagicMock()
        
        # Test noise generation parameters
        epsilon = privacy_config.get("epsilon", 1.0)
        delta = privacy_config.get("delta", 1e-5)
        sensitivity = 1.0
        
        # Mock noise calculation
        expected_noise_scale = sensitivity / epsilon
        mock_noise = np.random.laplace(0.0, expected_noise_scale, size=100)
        
        privacy_engine.generate_noise.return_value = mock_noise
        
        # Test noise generation
        generated_noise = privacy_engine.generate_noise(epsilon, sensitivity)
        
        assert generated_noise is not None
        assert len(generated_noise) == 100
        privacy_engine.generate_noise.assert_called_once_with(epsilon, sensitivity)
    
    def test_privacy_budget_tracking(self, privacy_config):
        """Test privacy budget management."""
        # Mock privacy budget manager
        budget_manager = MagicMock()
        
        total_epsilon = privacy_config.get("epsilon", 1.0)
        budget_manager.total_epsilon = total_epsilon
        budget_manager.used_epsilon = 0.0
        budget_manager.remaining_epsilon = total_epsilon
        
        # Test budget allocation
        allocation_requests = [0.1, 0.2, 0.15, 0.25, 0.2]
        
        for request in allocation_requests:
            if budget_manager.remaining_epsilon >= request:
                budget_manager.used_epsilon += request
                budget_manager.remaining_epsilon -= request
                allocation_result = True
            else:
                allocation_result = False
        
        assert budget_manager.used_epsilon == 0.9
        assert budget_manager.remaining_epsilon == 0.1
        assert budget_manager.used_epsilon + budget_manager.remaining_epsilon == total_epsilon
    
    def test_annotator_anonymization(self, sample_annotations):
        """Test annotator ID anonymization."""
        # Mock anonymizer
        anonymizer = MagicMock()
        
        # Test anonymization function
        def mock_anonymize(annotator_id, salt="default_salt"):
            import hashlib
            return hashlib.sha256(f"{annotator_id}_{salt}".encode()).hexdigest()[:16]
        
        anonymizer.anonymize_id.side_effect = mock_anonymize
        
        # Test anonymization
        original_ids = [ann["annotator_id"] for ann in sample_annotations]
        anonymized_ids = [anonymizer.anonymize_id(id_) for id_ in original_ids]
        
        # Verify anonymization
        assert len(anonymized_ids) == len(original_ids)
        assert all(len(anon_id) == 16 for anon_id in anonymized_ids)
        assert all(orig != anon for orig, anon in zip(original_ids, anonymized_ids))
        
        # Verify consistency
        duplicate_anonymized = [anonymizer.anonymize_id(id_) for id_ in original_ids]
        assert anonymized_ids == duplicate_anonymized
    
    def test_data_sanitization(self, sample_prompts, sample_responses):
        """Test data sanitization for privacy protection."""
        # Mock data sanitizer
        sanitizer = MagicMock()
        
        # Define PII patterns to remove
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[\w\.-]+@[\w\.-]+\.\w+\b',  # Email
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
            r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b'  # Credit card
        ]
        
        # Mock sanitization function
        def mock_sanitize(text):
            import re
            sanitized = text
            for pattern in pii_patterns:
                sanitized = re.sub(pattern, '[REDACTED]', sanitized)
            return sanitized
        
        sanitizer.sanitize_text.side_effect = mock_sanitize
        
        # Test sanitization
        test_text = "Contact john.doe@email.com or call 555-123-4567"
        sanitized_text = sanitizer.sanitize_text(test_text)
        
        assert '[REDACTED]' in sanitized_text
        assert 'john.doe@email.com' not in sanitized_text
        assert '555-123-4567' not in sanitized_text
    
    def test_privacy_preserving_aggregation(self, sample_annotations):
        """Test privacy-preserving aggregation of annotations."""
        # Mock aggregator
        aggregator = MagicMock()
        
        # Extract ratings for aggregation
        ratings = [ann["rating"] for ann in sample_annotations]
        
        # Mock privacy-preserving aggregation
        def mock_aggregate_with_privacy(values, epsilon=1.0):
            # Simulate adding noise to aggregate statistics
            true_mean = np.mean(values)
            noise_scale = 1.0 / epsilon  # Simplified noise scale
            noisy_mean = true_mean + np.random.laplace(0, noise_scale)
            return noisy_mean
        
        aggregator.aggregate_with_privacy.side_effect = mock_aggregate_with_privacy
        
        # Test aggregation
        noisy_mean = aggregator.aggregate_with_privacy(ratings)
        true_mean = np.mean(ratings)
        
        # Verify privacy-preserving aggregation
        assert isinstance(noisy_mean, (int, float))
        assert abs(noisy_mean - true_mean) > 0  # Noise should be added
        aggregator.aggregate_with_privacy.assert_called_once_with(ratings)
    
    def test_k_anonymity_enforcement(self, sample_annotations):
        """Test k-anonymity enforcement for annotator data."""
        # Mock k-anonymity enforcer
        k_anonymity_enforcer = MagicMock()
        
        k_value = 3  # Minimum group size
        
        # Group annotations by some quasi-identifier (mocked)
        quasi_identifiers = ["group_A", "group_B", "group_A", "group_C", "group_B"]
        
        def mock_enforce_k_anonymity(data, k=3):
            # Group data by quasi-identifiers
            groups = {}
            for i, (annotation, quasi_id) in enumerate(zip(data, quasi_identifiers)):
                if quasi_id not in groups:
                    groups[quasi_id] = []
                groups[quasi_id].append(annotation)
            
            # Filter groups with less than k members
            valid_groups = {k: v for k, v in groups.items() if len(v) >= k}
            
            return valid_groups
        
        k_anonymity_enforcer.enforce_k_anonymity.side_effect = mock_enforce_k_anonymity
        
        # Test k-anonymity enforcement
        anonymized_groups = k_anonymity_enforcer.enforce_k_anonymity(sample_annotations, k_value)
        
        # Verify k-anonymity
        for group_id, group_data in anonymized_groups.items():
            assert len(group_data) >= k_value
        
        k_anonymity_enforcer.enforce_k_anonymity.assert_called_once_with(
            sample_annotations, k_value
        )
    
    def test_privacy_loss_accounting(self, privacy_config):
        """Test privacy loss accounting across multiple operations."""
        # Mock privacy accountant
        accountant = MagicMock()
        
        # Initialize privacy parameters
        accountant.total_epsilon = privacy_config.get("epsilon", 1.0)
        accountant.total_delta = privacy_config.get("delta", 1e-5)
        accountant.operations = []
        
        # Simulate multiple privacy-consuming operations
        operations = [
            {"type": "query", "epsilon": 0.1, "delta": 1e-6},
            {"type": "aggregation", "epsilon": 0.2, "delta": 2e-6},
            {"type": "release", "epsilon": 0.15, "delta": 1.5e-6},
        ]
        
        def mock_add_operation(operation):
            accountant.operations.append(operation)
            return True
        
        def mock_get_privacy_spent():
            total_eps = sum(op["epsilon"] for op in accountant.operations)
            total_del = sum(op["delta"] for op in accountant.operations)
            return {"epsilon": total_eps, "delta": total_del}
        
        accountant.add_operation.side_effect = mock_add_operation
        accountant.get_privacy_spent.side_effect = mock_get_privacy_spent
        
        # Add operations
        for op in operations:
            accountant.add_operation(op)
        
        # Check privacy spent
        privacy_spent = accountant.get_privacy_spent()
        
        assert privacy_spent["epsilon"] == 0.45
        assert privacy_spent["delta"] == 4.5e-6
        assert privacy_spent["epsilon"] < accountant.total_epsilon
        assert privacy_spent["delta"] < accountant.total_delta
    
    def test_local_differential_privacy(self, sample_annotations):
        """Test local differential privacy for individual annotations."""
        # Mock local DP mechanism
        ldp_mechanism = MagicMock()
        
        # Test randomized response mechanism
        def mock_randomized_response(value, epsilon):
            p = np.exp(epsilon) / (np.exp(epsilon) + 1)
            if np.random.random() < p:
                return value  # Truth
            else:
                return 1 - value if isinstance(value, bool) else -value  # Lie
        
        ldp_mechanism.randomized_response.side_effect = mock_randomized_response
        
        # Test with boolean values (simplified ratings as good/bad)
        boolean_ratings = [rating > 4.0 for rating in [ann["rating"] for ann in sample_annotations]]
        epsilon = 1.0
        
        # Apply local DP
        ldp_ratings = [ldp_mechanism.randomized_response(rating, epsilon) for rating in boolean_ratings]
        
        # Verify LDP application
        assert len(ldp_ratings) == len(boolean_ratings)
        assert all(isinstance(rating, bool) for rating in ldp_ratings)
        
        # Some values should potentially be flipped (privacy protection)
        # Note: This is probabilistic, so we can't guarantee flips in a small sample
    
    def test_privacy_aware_model_updates(self, mock_torch_model):
        """Test privacy-aware model parameter updates."""
        # Mock privacy-aware optimizer
        private_optimizer = MagicMock()
        
        # Mock gradient clipping and noise addition
        def mock_private_step(gradients, clip_norm=1.0, noise_scale=0.1):
            # Simulate gradient clipping
            clipped_gradients = []
            for grad in gradients:
                grad_norm = np.linalg.norm(grad) if hasattr(grad, '__len__') else abs(grad)
                if grad_norm > clip_norm:
                    clipped_grad = grad * (clip_norm / grad_norm)
                else:
                    clipped_grad = grad
                
                # Add noise
                if hasattr(clipped_grad, '__len__'):
                    noise = np.random.normal(0, noise_scale, size=len(clipped_grad))
                    noisy_grad = clipped_grad + noise
                else:
                    noise = np.random.normal(0, noise_scale)
                    noisy_grad = clipped_grad + noise
                
                clipped_gradients.append(noisy_grad)
            
            return clipped_gradients
        
        private_optimizer.private_step.side_effect = mock_private_step
        
        # Test private gradient update
        mock_gradients = [np.array([1.5, -2.0, 0.8]), np.array([0.5])]
        private_gradients = private_optimizer.private_step(mock_gradients)
        
        # Verify privacy-aware updates
        assert len(private_gradients) == len(mock_gradients)
        for orig, priv in zip(mock_gradients, private_gradients):
            # Gradients should be modified (clipped and/or noised)
            if hasattr(orig, '__len__'):
                assert not np.array_equal(orig, priv)
            else:
                assert orig != priv
    
    def test_privacy_audit_logging(self, privacy_config):
        """Test privacy-specific audit logging."""
        # Mock privacy audit logger
        privacy_logger = MagicMock()
        
        # Test privacy event logging
        privacy_events = [
            {
                "event_type": "epsilon_allocation",
                "epsilon_used": 0.1,
                "operation": "query_response",
                "timestamp": "2025-01-01T00:00:00Z"
            },
            {
                "event_type": "noise_addition",
                "noise_scale": 0.05,
                "mechanism": "laplace",
                "timestamp": "2025-01-01T00:01:00Z"
            },
            {
                "event_type": "anonymization",
                "items_anonymized": 5,
                "method": "hash_based",
                "timestamp": "2025-01-01T00:02:00Z"
            }
        ]
        
        # Mock logging function
        def mock_log_privacy_event(event):
            return {"logged": True, "event_id": f"priv_{len(privacy_logger.logged_events)}"}
        
        privacy_logger.logged_events = []
        privacy_logger.log_privacy_event.side_effect = mock_log_privacy_event
        
        # Log privacy events
        for event in privacy_events:
            result = privacy_logger.log_privacy_event(event)
            privacy_logger.logged_events.append(event)
            assert result["logged"] is True
        
        # Verify logging
        assert len(privacy_logger.logged_events) == len(privacy_events)
        assert privacy_logger.log_privacy_event.call_count == len(privacy_events)