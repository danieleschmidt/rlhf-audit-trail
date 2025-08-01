# RLHF Audit Trail - Testing Guide

## Overview

This guide provides comprehensive information about the testing infrastructure and practices for the RLHF Audit Trail project.

## Test Structure

### Test Categories

The test suite is organized into several categories:

```
tests/
├── unit/                 # Unit tests for individual components
├── integration/          # Integration tests for component interactions
├── e2e/                 # End-to-end tests for complete workflows
├── performance/         # Performance and benchmark tests
├── fixtures/            # Test data and utilities
└── docs/               # Testing documentation
```

### Test Markers

Use pytest markers to categorize and run specific test types:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (moderate speed)
- `@pytest.mark.e2e` - End-to-end tests (slow, comprehensive)
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.fast` - Quick tests for CI
- `@pytest.mark.compliance` - Compliance validation tests
- `@pytest.mark.security` - Security-focused tests
- `@pytest.mark.privacy` - Privacy protection tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.database` - Tests requiring database
- `@pytest.mark.redis` - Tests requiring Redis
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.network` - Tests requiring network access

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m "not slow"             # Skip slow tests
pytest -m "integration or e2e"   # Integration and E2E tests

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run tests in parallel
pytest -n auto                   # Auto-detect CPU cores
pytest -n 4                      # Use 4 processes

# Run specific test files
pytest tests/unit/test_privacy_engine.py
pytest tests/integration/test_audit_trail_integration.py

# Run tests matching pattern
pytest -k "privacy"              # Tests with "privacy" in name
pytest -k "not slow"             # Skip tests with "slow" in name
```

### Environment-Specific Testing

```bash
# Development environment
pytest --env=development

# CI environment (faster, fewer resources)
pytest -m "not slow and not gpu" --maxfail=5

# Production validation
pytest -m "integration or e2e" --strict-markers
```

## Test Configuration

### pytest.ini Configuration

The project uses comprehensive pytest configuration:

- **Coverage**: 85% minimum coverage requirement
- **Timeouts**: 300 seconds maximum per test
- **Logging**: Detailed logging for debugging
- **Warnings**: Strict warning handling
- **Markers**: Custom markers for test categorization

### Environment Variables

Set these environment variables for testing:

```bash
# Required for tests
export ENVIRONMENT=test
export TESTING=true
export LOG_LEVEL=DEBUG

# Optional for integration tests
export DATABASE_URL=postgresql://postgres:password@localhost:5432/test_db
export REDIS_URL=redis://localhost:6379/1
```

## Test Data and Fixtures

### Using Sample Data

```python
from tests.fixtures import SampleDataGenerator

# Generate test data
prompts = SampleDataGenerator.generate_sample_prompts(10)
responses = SampleDataGenerator.generate_sample_responses(prompts)
annotations = SampleDataGenerator.generate_sample_annotations(10)
```

### Common Fixtures

Available fixtures in `conftest.py`:

- `temp_dir` - Temporary directory for file operations
- `sample_prompts` - Sample training prompts
- `sample_responses` - Sample model responses
- `sample_annotations` - Sample human annotations
- `privacy_config` - Privacy configuration for testing
- `compliance_config` - Compliance settings
- `mock_redis` - Mocked Redis client
- `audit_trail_mock` - Mocked audit trail system

### Custom Fixtures

Create test-specific fixtures:

```python
@pytest.fixture
def custom_training_data():
    """Custom training data for specific tests."""
    return {
        "prompts": ["Test prompt 1", "Test prompt 2"],
        "responses": ["Test response 1", "Test response 2"],
        "quality_scores": [4.5, 3.8]
    }
```

## Writing Tests

### Unit Test Example

```python
import pytest
from unittest.mock import MagicMock

@pytest.mark.unit
@pytest.mark.privacy
class TestPrivacyEngine:
    def test_differential_privacy_noise(self, privacy_config):
        """Test DP noise generation."""
        privacy_engine = MagicMock()
        
        # Test implementation
        noise = privacy_engine.generate_noise(
            epsilon=privacy_config["epsilon"],
            sensitivity=1.0
        )
        
        assert noise is not None
        privacy_engine.generate_noise.assert_called_once()
```

### Integration Test Example

```python
@pytest.mark.integration
@pytest.mark.slow
async def test_audit_trail_integration(self, audit_system, sample_data):
    """Test complete audit trail workflow."""
    # Setup
    session = await audit_system.start_session()
    
    # Execute workflow
    for data_point in sample_data:
        result = await audit_system.log_event(data_point)
        assert result.success is True
    
    # Verify results
    audit_log = await audit_system.get_audit_log(session.id)
    assert len(audit_log.events) == len(sample_data)
```

### End-to-End Test Example

```python
@pytest.mark.e2e
@pytest.mark.slow
async def test_complete_rlhf_workflow(self, complete_system):
    """Test full RLHF training with audit trail."""
    # Initialize training
    session = await complete_system.start_training({
        "model": "test-model",
        "privacy_enabled": True,
        "compliance_mode": "eu_ai_act"
    })
    
    # Execute training phases
    await complete_system.run_training(session.id)
    
    # Validate results
    report = await complete_system.generate_final_report(session.id)
    assert report.compliance_status == "compliant"
    assert report.privacy_violations == 0
```

## Testing Best Practices

### Test Organization

1. **One test class per component**: Group related tests together
2. **Descriptive test names**: Use clear, descriptive test method names
3. **Arrange-Act-Assert**: Structure tests with clear phases
4. **Independent tests**: Each test should be able to run independently

### Mocking Guidelines

```python
# Mock external dependencies
@patch('rlhf_audit_trail.external_service')
def test_with_external_service(self, mock_service):
    mock_service.call_api.return_value = {"status": "success"}
    # Test implementation

# Use MagicMock for complex objects
mock_model = MagicMock()
mock_model.train.return_value = {"loss": 0.5}
```

### Async Testing

```python
@pytest.mark.asyncio
async def test_async_function(self):
    """Test async functionality."""
    result = await async_function()
    assert result is not None
```

## Performance Testing

### Benchmark Tests

```python
@pytest.mark.performance
def test_audit_logging_performance(self, benchmark):
    """Benchmark audit logging performance."""
    def audit_operation():
        return audit_logger.log_event(sample_event)
    
    result = benchmark(audit_operation)
    assert result.success is True
```

### Memory Testing

```python
import psutil
import os

def test_memory_usage(self):
    """Test memory usage doesn't exceed limits."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Run memory-intensive operation
    large_operation()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Assert memory increase is reasonable
    assert memory_increase < 100 * 1024 * 1024  # 100MB limit
```

## Compliance and Security Testing

### Compliance Tests

```python
@pytest.mark.compliance
def test_eu_ai_act_compliance(self, compliance_validator):
    """Test EU AI Act compliance requirements."""
    result = compliance_validator.validate_eu_ai_act(training_session)
    
    assert result.compliant is True
    assert result.score > 0.9
    assert len(result.violations) == 0
```

### Security Tests

```python
@pytest.mark.security
def test_cryptographic_integrity(self, audit_trail):
    """Test cryptographic integrity of audit trail."""
    # Create audit entries
    events = [create_test_event() for _ in range(10)]
    
    # Verify integrity
    integrity_result = audit_trail.verify_integrity(events)
    
    assert integrity_result.valid is True
    assert integrity_result.tamper_detected is False
```

## Continuous Integration

### CI Test Strategy

1. **Fast feedback**: Run unit tests on every commit
2. **Integration testing**: Run integration tests on PR
3. **Full testing**: Run all tests including E2E on main branch
4. **Nightly testing**: Run performance and long-running tests

### GitHub Actions Example

```yaml
- name: Run fast tests
  run: pytest -m "not slow" --maxfail=5
  
- name: Run integration tests
  run: pytest -m integration --cov=src
  
- name: Upload coverage
  uses: codecov/codecov-action@v1
```

## Debugging Tests

### Debugging Commands

```bash
# Run with verbose output
pytest -v

# Drop to debugger on failure
pytest --pdb

# Debug specific test
pytest --pdb tests/unit/test_privacy_engine.py::test_specific_function

# Capture stdout/stderr
pytest -s

# Show local variables in tracebacks
pytest --tb=long
```

### Logging in Tests

```python
import logging

def test_with_logging(self, caplog):
    """Test with log capture."""
    with caplog.at_level(logging.INFO):
        function_that_logs()
    
    assert "Expected log message" in caplog.text
```

## Test Data Management

### Test Database

```python
@pytest.fixture(scope="session")
def test_database():
    """Create test database."""
    # Setup
    engine = create_test_engine()
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Teardown
    Base.metadata.drop_all(engine)
```

### Data Cleanup

```python
@pytest.fixture(autouse=True)
def cleanup_test_data(self, temp_dir):
    """Automatically cleanup test data."""
    yield
    # Cleanup performed after each test
    shutil.rmtree(temp_dir, ignore_errors=True)
```

## Common Testing Patterns

### Testing Exceptions

```python
def test_invalid_input_raises_exception(self):
    """Test that invalid input raises appropriate exception."""
    with pytest.raises(ValueError, match="Invalid input"):
        function_with_validation(invalid_input)
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input_value,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double_function(self, input_value, expected):
    """Test function with multiple inputs."""
    result = double(input_value)
    assert result == expected
```

### Testing Configuration

```python
def test_configuration_loading(self, temp_dir):
    """Test configuration loading from file."""
    config_file = temp_dir / "config.yaml"
    config_file.write_text("setting: value")
    
    config = load_config(str(config_file))
    assert config["setting"] == "value"
```

## Maintenance and Updates

### Test Maintenance

1. **Regular review**: Review and update tests quarterly
2. **Remove obsolete tests**: Clean up tests for removed features
3. **Update fixtures**: Keep test data current and relevant
4. **Performance monitoring**: Monitor test execution times

### Documentation Updates

Keep this guide updated when:
- Adding new test categories or markers
- Changing test infrastructure
- Adding new testing tools or practices
- Updating CI/CD pipeline

For questions about testing, please refer to the development team or create an issue in the project repository.