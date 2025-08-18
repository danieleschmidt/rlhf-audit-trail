"""API contract tests for RLHF Audit Trail.

These tests verify that API contracts are maintained across versions
and ensure backward compatibility for integrations.
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rlhf_audit_trail.api import app
from tests.fixtures.sample_data import create_sample_audit_record, create_sample_training_session

client = TestClient(app)


class TestAuditTrailAPIContracts:
    """Test contracts for audit trail API endpoints."""

    @pytest.mark.contract
    def test_create_audit_record_contract(self):
        """Test that audit record creation follows expected contract."""
        audit_data = create_sample_audit_record()
        response = client.post("/api/v1/audit/records", json=audit_data)
        
        assert response.status_code == 201
        response_data = response.json()
        
        # Contract requirements
        assert "id" in response_data
        assert "timestamp" in response_data
        assert "event_type" in response_data
        assert "merkle_proof" in response_data
        assert "signature" in response_data
        
        # Verify data types
        assert isinstance(response_data["id"], str)
        assert isinstance(response_data["timestamp"], str)
        assert isinstance(response_data["event_type"], str)

    @pytest.mark.contract
    def test_training_session_contract(self):
        """Test that training session API follows expected contract."""
        session_data = create_sample_training_session()
        response = client.post("/api/v1/training/sessions", json=session_data)
        
        assert response.status_code == 201
        response_data = response.json()
        
        # Contract requirements
        assert "session_id" in response_data
        assert "experiment_name" in response_data
        assert "privacy_config" in response_data
        assert "compliance_mode" in response_data
        assert "status" in response_data

    @pytest.mark.contract
    def test_compliance_report_contract(self):
        """Test that compliance report API follows expected contract."""
        response = client.get("/api/v1/compliance/reports/latest")
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Contract requirements
            assert "report_id" in response_data
            assert "generated_at" in response_data
            assert "compliance_frameworks" in response_data
            assert "violations" in response_data
            assert "recommendations" in response_data

    @pytest.mark.contract
    def test_privacy_budget_contract(self):
        """Test that privacy budget API follows expected contract."""
        response = client.get("/api/v1/privacy/budget")
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Contract requirements
        assert "total_epsilon" in response_data
        assert "consumed_epsilon" in response_data
        assert "remaining_epsilon" in response_data
        assert "budget_reset_time" in response_data

    @pytest.mark.contract
    def test_error_response_contract(self):
        """Test that error responses follow expected contract."""
        response = client.get("/api/v1/nonexistent/endpoint")
        
        assert response.status_code == 404
        response_data = response.json()
        
        # Error contract requirements
        assert "detail" in response_data or "message" in response_data

    @pytest.mark.contract
    def test_health_check_contract(self):
        """Test that health check follows expected contract."""
        response = client.get("/health")
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Health check contract
        assert "status" in response_data
        assert "timestamp" in response_data
        assert "version" in response_data
        assert response_data["status"] in ["healthy", "unhealthy", "degraded"]


class TestAPIVersionCompatibility:
    """Test API version compatibility and backward compatibility."""

    @pytest.mark.contract
    def test_api_version_headers(self):
        """Test that API version headers are properly handled."""
        headers = {"Accept": "application/vnd.api+json;version=1"}
        response = client.get("/api/v1/health", headers=headers)
        
        assert response.status_code == 200
        assert "X-API-Version" in response.headers

    @pytest.mark.contract  
    def test_deprecated_endpoints_still_work(self):
        """Test that deprecated endpoints still function for backward compatibility."""
        # This would test any deprecated but still supported endpoints
        pass

    @pytest.mark.contract
    def test_request_validation_contract(self):
        """Test that request validation errors follow consistent format."""
        invalid_data = {"invalid_field": "invalid_value"}
        response = client.post("/api/v1/audit/records", json=invalid_data)
        
        assert response.status_code == 422
        response_data = response.json()
        
        # Validation error contract
        assert "detail" in response_data
        if isinstance(response_data["detail"], list):
            for error in response_data["detail"]:
                assert "loc" in error
                assert "msg" in error
                assert "type" in error


class TestDataSchemaContracts:
    """Test that data schemas are maintained across versions."""

    @pytest.mark.contract
    def test_audit_record_schema_stability(self):
        """Test that audit record schema remains stable."""
        from rlhf_audit_trail.models import AuditRecord
        
        # Test that required fields haven't changed
        required_fields = {
            "timestamp", "event_type", "event_data", 
            "merkle_proof", "signature"
        }
        
        model_fields = set(AuditRecord.__annotations__.keys())
        assert required_fields.issubset(model_fields)

    @pytest.mark.contract
    def test_privacy_config_schema_stability(self):
        """Test that privacy config schema remains stable."""
        from rlhf_audit_trail.models import PrivacyConfig
        
        required_fields = {"epsilon", "delta", "clip_norm"}
        model_fields = set(PrivacyConfig.__annotations__.keys())
        assert required_fields.issubset(model_fields)

    @pytest.mark.contract
    def test_model_card_schema_stability(self):
        """Test that model card schema remains stable."""
        from rlhf_audit_trail.models import ModelCard
        
        required_fields = {
            "model_name", "version", "training_data", 
            "privacy_analysis", "compliance_status"
        }
        
        model_fields = set(ModelCard.__annotations__.keys())
        assert required_fields.issubset(model_fields)


@pytest.fixture
def api_client():
    """Provide API client for contract tests."""
    return TestClient(app)


@pytest.fixture
def contract_test_data():
    """Provide test data that conforms to expected contracts."""
    return {
        "audit_record": create_sample_audit_record(),
        "training_session": create_sample_training_session(),
    }