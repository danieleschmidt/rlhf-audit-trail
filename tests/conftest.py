"""Pytest configuration and shared fixtures for RLHF Audit Trail tests."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

import pytest
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["TESTING"] = "true"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="session")
def test_database_url(temp_dir: Path) -> str:
    """Create a test database URL."""
    db_path = temp_dir / "test.db"
    return f"sqlite:///{db_path}"


@pytest.fixture(scope="session")
def test_redis_url() -> str:
    """Get test Redis URL."""
    return os.getenv("REDIS_URL", "redis://localhost:6379/1")


@pytest.fixture
def mock_redis():
    """Mock Redis client for tests that don't need real Redis."""
    mock_redis = MagicMock(spec=redis.Redis)
    mock_redis.ping.return_value = True
    mock_redis.set.return_value = True
    mock_redis.get.return_value = None
    mock_redis.delete.return_value = 1
    mock_redis.exists.return_value = False
    return mock_redis


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "How do neural networks work?",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis.",
    ]


@pytest.fixture
def sample_responses():
    """Sample responses for testing."""
    return [
        "The capital of France is Paris.",
        "Quantum computing uses quantum mechanics principles...",
        "Neural networks are computational models inspired by the brain...",
        "Renewable energy benefits include reduced emissions...",
        "Photosynthesis is the process by which plants convert light energy...",
    ]


@pytest.fixture
def sample_annotations():
    """Sample human annotations for testing."""
    return [
        {"annotator_id": "annotator_001", "rating": 4.5, "feedback": "Good response"},
        {"annotator_id": "annotator_002", "rating": 3.8, "feedback": "Could be more detailed"},
        {"annotator_id": "annotator_003", "rating": 4.2, "feedback": "Accurate and clear"},
        {"annotator_id": "annotator_004", "rating": 3.9, "feedback": "Helpful response"},
        {"annotator_id": "annotator_005", "rating": 4.1, "feedback": "Well explained"},
    ]


@pytest.fixture
def privacy_config():
    """Sample privacy configuration for testing."""
    from rlhf_audit_trail.config import PrivacyConfig
    
    return PrivacyConfig(
        epsilon=1.0,
        delta=1e-5,
        clip_norm=1.0,
        noise_multiplier=1.1,
        annotator_privacy_mode="moderate"
    )


@pytest.fixture
def security_config():
    """Sample security configuration for testing."""
    from rlhf_audit_trail.config import SecurityConfig
    
    return SecurityConfig(
        enable_encryption=True,
        key_rotation_interval=3600,
        audit_log_retention_days=365,
        require_digital_signatures=True
    )


@pytest.fixture
def compliance_config():
    """Sample compliance configuration for testing."""
    from rlhf_audit_trail.config import ComplianceConfig
    
    return ComplianceConfig(
        mode="eu_ai_act",
        enable_audit_trail=True,
        require_human_oversight=True,
        data_retention_years=7
    )


@pytest.fixture
async def audit_trail_mock():
    """Mock AuditableRLHF instance for testing."""
    from unittest.mock import AsyncMock
    
    mock = AsyncMock()
    mock.log_annotations.return_value = {"logged": True}
    mock.track_policy_update.return_value = {"tracked": True}
    mock.checkpoint.return_value = {"checkpoint": "test_checkpoint"}
    mock.generate_model_card.return_value = {"model_card": "test_card"}
    mock.verify_provenance.return_value = {"is_valid": True}
    
    return mock


@pytest.fixture
def real_redis_client(test_redis_url: str):
    """Real Redis client for integration tests."""
    try:
        client = redis.from_url(test_redis_url)
        client.ping()  # Test connection
        yield client
        # Cleanup
        client.flushdb()
    except redis.ConnectionError:
        pytest.skip("Redis not available for integration tests")


@pytest.fixture
def test_model_config():
    """Test model configuration."""
    return {
        "model_name": "test-model",
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "batch_size": 4
    }


@pytest.fixture
def mock_torch_model():
    """Mock PyTorch model for testing."""
    from unittest.mock import MagicMock
    
    model = MagicMock()
    model.eval.return_value = model
    model.train.return_value = model
    model.parameters.return_value = []
    model.state_dict.return_value = {}
    model.load_state_dict.return_value = None
    
    return model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    from unittest.mock import MagicMock
    
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test output"
    tokenizer.vocab_size = 50000
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    
    return tokenizer


@pytest.fixture(autouse=True)
def cleanup_test_files(temp_dir: Path):
    """Automatically cleanup test files after each test."""
    yield
    # Cleanup is handled by temp_dir fixture


@pytest.fixture
def capture_logs(caplog):
    """Capture logs for testing."""
    import logging
    caplog.set_level(logging.INFO)
    return caplog


# Markers for different test categories
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "compliance: mark test as compliance verification"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security-related"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to tests that take a long time
        if "slow" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "integration" in item.name or item.fspath.basename.startswith("test_integration"):
            item.add_marker(pytest.mark.integration)
        
        # Add compliance marker to compliance tests
        if "compliance" in item.name or "regulatory" in item.name:
            item.add_marker(pytest.mark.compliance)
        
        # Add security marker to security tests
        if "security" in item.name or "crypto" in item.name:
            item.add_marker(pytest.mark.security)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Set environment variables for testing
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    os.environ.setdefault("DISABLE_TELEMETRY", "true")
    
    yield
    
    # Cleanup environment
    test_vars = ["LOG_LEVEL", "DISABLE_TELEMETRY"]
    for var in test_vars:
        os.environ.pop(var, None)