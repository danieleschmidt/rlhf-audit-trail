# Development Guide - RLHF Audit Trail

This guide provides comprehensive information for developers working on the RLHF Audit Trail project.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Debugging and Troubleshooting](#debugging-and-troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Security Considerations](#security-considerations)
- [Compliance Development](#compliance-development)

## Development Environment Setup

### Prerequisites

Ensure you have the following installed:

- **Python 3.10+**: Primary development language
- **Docker & Docker Compose**: For containerized development
- **Git**: Version control
- **Node.js 16+**: For frontend tooling (if applicable)
- **Make**: For build automation

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/terragonlabs/rlhf-audit-trail.git
   cd rlhf-audit-trail
   ```

2. **Set up development environment:**
   ```bash
   # Using Make (recommended)
   make init-project
   
   # Or manually
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   pre-commit install
   ```

3. **Start development services:**
   ```bash
   # Using Docker Compose
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
   
   # Or individual services
   make dev
   ```

### Development with Docker

For a consistent development environment:

```bash
# Build development image
docker-compose -f docker-compose.dev.yml build

# Start development environment
docker-compose -f docker-compose.dev.yml up

# Access development container
docker-compose -f docker-compose.dev.yml exec dev bash
```

### IDE Configuration

#### VS Code Setup

1. Install recommended extensions:
   - Python
   - Pylance
   - Black Formatter
   - Ruff
   - Docker
   - GitLens

2. Configure settings (`.vscode/settings.json`):
   ```json
   {
     "python.defaultInterpreterPath": "./venv/bin/python",
     "python.linting.enabled": true,
     "python.linting.ruffEnabled": true,
     "python.formatting.provider": "black",
     "python.testing.pytestEnabled": true,
     "python.testing.pytestArgs": ["tests/"],
     "files.exclude": {
       "**/__pycache__": true,
       "**/*.pyc": true
     }
   }
   ```

#### PyCharm Setup

1. Configure Python interpreter to use virtual environment
2. Set up code style to use Black and Ruff
3. Configure test runner to use pytest
4. Enable type checking with mypy

## Project Structure

```
rlhf-audit-trail/
├── src/rlhf_audit_trail/          # Main application code
│   ├── core/                      # Core RLHF functionality
│   ├── audit/                     # Audit trail management
│   ├── privacy/                   # Privacy protection
│   ├── compliance/                # Compliance validation
│   ├── api/                       # REST API endpoints
│   ├── dashboard/                 # Streamlit dashboard
│   └── cli/                       # Command-line interface
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   ├── compliance/                # Compliance tests
│   └── fixtures/                  # Test fixtures
├── docs/                          # Documentation
├── scripts/                       # Utility scripts
├── deploy/                        # Deployment configurations
├── monitoring/                    # Monitoring configurations
├── compliance/                    # Compliance artifacts
└── benchmarks/                    # Performance benchmarks
```

### Key Directories

- **`src/rlhf_audit_trail/`**: All application source code
- **`tests/`**: Comprehensive test suite with different categories
- **`docs/`**: Documentation including API docs and guides
- **`scripts/`**: Development and deployment scripts
- **`compliance/`**: Compliance-related files and validators

## Development Workflow

### Branch Strategy

We use GitFlow with the following branches:

- **`main`**: Production-ready code
- **`develop`**: Integration branch for features
- **`feature/*`**: Feature development branches
- **`hotfix/*`**: Emergency fixes for production
- **`release/*`**: Release preparation branches

### Feature Development Process

1. **Create feature branch:**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Develop and test:**
   ```bash
   # Make your changes
   # Run tests locally
   make test
   make lint
   make type-check
   ```

3. **Commit changes:**
   ```bash
   # Stage changes
   git add .
   
   # Commit with conventional commit format
   git commit -m "feat: add new compliance validation feature"
   ```

4. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request via GitHub UI
   ```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(privacy): implement differential privacy budget tracking
fix(audit): resolve merkle tree validation issue
docs: update API documentation for compliance endpoints
test: add integration tests for RLHF core functionality
```

## Code Standards

### Python Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **Ruff**: Linting and code analysis
- **MyPy**: Type checking
- **isort**: Import sorting

### Configuration Files

All tools are configured in `pyproject.toml`:

```toml
[tool.black]
target-version = ['py310']
line-length = 88

[tool.ruff]
target-version = "py310"
line-length = 88
select = ["E", "F", "W", "I", "N", "UP", "S", "B", "A", "C4", "DTZ", "T10", "EM", "EXE", "ISC", "ICN", "G", "INP", "PIE", "T20", "PYI", "PT", "Q", "RSE", "RET", "SLF", "SIM", "TID", "TCH", "ARG", "PTH", "ERA", "PD", "PGH", "PL", "TRY", "NPY", "RUF"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
disallow_untyped_defs = true
```

### Code Quality Commands

```bash
# Format code
make format
# or
black src tests
isort src tests

# Lint code
make lint
# or
ruff check src tests

# Type check
make type-check
# or
mypy src

# Run all quality checks
make check-all
```

### Documentation Standards

- **Docstrings**: Use Google-style docstrings
- **Type hints**: All functions must have type hints
- **Comments**: Explain complex business logic
- **README**: Keep project README up to date

**Example:**
```python
def calculate_privacy_budget(
    epsilon: float, 
    delta: float, 
    num_queries: int
) -> PrivacyBudget:
    """Calculate remaining privacy budget for differential privacy.
    
    Args:
        epsilon: Privacy parameter controlling noise level
        delta: Probability of privacy breach
        num_queries: Number of queries already executed
        
    Returns:
        PrivacyBudget object with remaining budget information
        
    Raises:
        PrivacyBudgetExceededError: If budget is already exhausted
    """
    # Implementation here
```

## Testing Guidelines

### Test Structure

We use pytest with the following test categories:

```bash
# Run all tests
make test

# Run specific test categories
pytest -m unit           # Unit tests
pytest -m integration    # Integration tests
pytest -m compliance     # Compliance tests
pytest -m security       # Security tests
```

### Test Organization

```
tests/
├── unit/                     # Fast, isolated tests
│   ├── test_privacy.py
│   ├── test_audit.py
│   └── test_compliance.py
├── integration/              # Tests with external dependencies
│   ├── test_database.py
│   └── test_api.py
├── compliance/               # Regulatory compliance tests
│   └── test_eu_ai_act.py
└── conftest.py              # Shared fixtures
```

### Writing Tests

#### Unit Test Example

```python
import pytest
from unittest.mock import Mock, patch
from rlhf_audit_trail.privacy import DifferentialPrivacy

class TestDifferentialPrivacy:
    def test_calculate_noise_multiplier(self):
        """Test noise multiplier calculation."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        noise_multiplier = dp.calculate_noise_multiplier()
        
        assert isinstance(noise_multiplier, float)
        assert noise_multiplier > 0
    
    def test_privacy_budget_exceeded(self):
        """Test privacy budget exceeded exception."""
        dp = DifferentialPrivacy(epsilon=0.1, delta=1e-5)
        
        with pytest.raises(PrivacyBudgetExceededError):
            dp.consume_budget(0.2)
```

#### Integration Test Example

```python
@pytest.mark.integration
class TestAuditTrailIntegration:
    def test_end_to_end_audit_flow(self, test_database):
        """Test complete audit trail flow."""
        # Setup
        auditor = AuditTrail(database=test_database)
        
        # Execute
        session_id = auditor.start_session("test_session")
        auditor.log_event(session_id, "training_step", {"epoch": 1})
        auditor.end_session(session_id)
        
        # Verify
        events = auditor.get_session_events(session_id)
        assert len(events) == 2
        assert events[0]["event_type"] == "session_start"
        assert events[1]["event_type"] == "training_step"
```

### Test Coverage

Maintain >90% test coverage:

```bash
# Run tests with coverage
pytest --cov=src --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html
```

## Debugging and Troubleshooting

### Local Debugging

#### Using Python Debugger

```python
import pdb; pdb.set_trace()  # Basic debugger
import ipdb; ipdb.set_trace()  # Enhanced debugger (if installed)
```

#### VS Code Debugging

Configure `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": ["rlhf_audit_trail.api.main:app", "--reload"],
            "jinja": true,
            "console": "integratedTerminal"
        },
        {
            "name": "Python: Pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

### Docker Debugging

```bash
# View logs
docker-compose logs -f app

# Access container shell
docker-compose exec app bash

# Debug specific service
docker-compose exec app python -m pdb your_script.py
```

### Common Issues

#### Database Connection Issues

```python
# Check database connection
from rlhf_audit_trail.database import get_database
db = get_database()
db.execute("SELECT 1")
```

#### Redis Connection Issues

```python
# Check Redis connection
import redis
r = redis.from_url("redis://localhost:6379")
r.ping()
```

#### Import Issues

```python
# Verify Python path
import sys
print(sys.path)

# Check package installation
import pkg_resources
pkg_resources.get_distribution("rlhf-audit-trail")
```

## Performance Optimization

### Profiling

#### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
@profile
def memory_intensive_function():
    # Your code here
    pass

# Run with profiler
python -m memory_profiler your_script.py
```

#### CPU Profiling

```bash
# Install py-spy
pip install py-spy

# Profile running application
py-spy record -o profile.svg -- python your_script.py

# Top-like profiling
py-spy top --pid <pid>
```

### Performance Best Practices

1. **Database Optimization**
   ```python
   # Use connection pooling
   # Add database indexes
   # Optimize queries
   # Use async operations where possible
   ```

2. **Caching Strategy**
   ```python
   # Use Redis for session data
   # Cache expensive computations
   # Implement cache invalidation
   ```

3. **Async Programming**
   ```python
   # Use async/await for I/O operations
   async def fetch_data():
       async with httpx.AsyncClient() as client:
           response = await client.get(url)
           return response.json()
   ```

### Benchmarking

```bash
# Run performance benchmarks
python benchmarks/run_benchmarks.py

# Pytest benchmark
pytest benchmarks/ --benchmark-only
```

## Security Considerations

### Secure Development Practices

1. **Input Validation**
   ```python
   from pydantic import BaseModel, validator
   
   class UserInput(BaseModel):
       email: str
       password: str
       
       @validator('email')
       def validate_email(cls, v):
           # Email validation logic
           return v
   ```

2. **SQL Injection Prevention**
   ```python
   # Use parameterized queries
   query = "SELECT * FROM users WHERE id = %s"
   cursor.execute(query, (user_id,))
   ```

3. **Secret Management**
   ```python
   import os
   from cryptography.fernet import Fernet
   
   # Never hardcode secrets
   secret_key = os.environ.get('SECRET_KEY')
   if not secret_key:
       raise ValueError("SECRET_KEY environment variable not set")
   ```

### Security Testing

```bash
# Run security scans
make security

# Bandit security scan
bandit -r src/

# Safety dependency scan
safety check

# Secrets detection
detect-secrets scan --baseline .secrets.baseline
```

## Compliance Development

### EU AI Act Compliance

When developing features that affect compliance:

1. **Document Changes**
   ```python
   def new_feature():
       """New feature implementation.
       
       EU AI Act Compliance Notes:
       - Implements Article X requirement for Y
       - Maintains audit trail for all operations
       - Preserves data privacy through anonymization
       """
   ```

2. **Add Compliance Tests**
   ```python
   @pytest.mark.compliance
   def test_eu_ai_act_article_12_compliance():
       """Test compliance with EU AI Act Article 12 - Record Keeping."""
       # Test implementation
   ```

3. **Update Compliance Validator**
   ```python
   # Add new compliance checks to compliance/compliance-validator.py
   def _check_new_requirement(self):
       """Check new regulatory requirement."""
       # Implementation
   ```

### Privacy by Design

```python
class PrivacyAwareFeature:
    def __init__(self, privacy_config: PrivacyConfig):
        self.privacy = DifferentialPrivacy(privacy_config)
    
    def process_sensitive_data(self, data: List[str]) -> List[str]:
        """Process data with privacy protection."""
        # Apply differential privacy
        noisy_data = self.privacy.add_noise(data)
        
        # Log privacy budget consumption
        self.privacy.consume_budget(len(data))
        
        return noisy_data
```

### Compliance Documentation

Always update compliance documentation when making changes:

```bash
# Update compliance checklist
vim compliance/eu-ai-act-checklist.yml

# Run compliance validation
python compliance/compliance-validator.py

# Update documentation
vim docs/compliance.md
```

## Development Tools and Utilities

### Useful Make Commands

```bash
make help              # Show all available commands
make clean             # Clean build artifacts
make install           # Install dependencies
make install-dev       # Install development dependencies
make test              # Run tests
make test-cov          # Run tests with coverage
make lint              # Run linting
make format            # Format code
make type-check        # Run type checking
make security          # Run security checks
make docs              # Build documentation
make docker            # Build Docker image
make docker-dev        # Build development Docker image
```

### Development Scripts

```bash
# Update dependencies
./scripts/update-deps.sh

# Deploy to staging
./scripts/deploy.sh -e staging

# Generate compliance report
python compliance/compliance-validator.py --format text
```

### Environment Variables

For development, create a `.env` file:

```bash
# Development environment variables
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://postgres:password@localhost:5432/rlhf_audit_dev
REDIS_URL=redis://localhost:6379/0
```

## Contributing Guidelines

1. **Follow Code Standards**: Use pre-commit hooks
2. **Write Tests**: Maintain high test coverage
3. **Document Changes**: Update relevant documentation
4. **Compliance Awareness**: Consider regulatory implications
5. **Security First**: Follow secure development practices

For more detailed contributing guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Getting Help

- **Documentation**: Check existing docs in `/docs`
- **Issues**: Create GitHub issues for bugs/features
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Request reviews from team members

This development guide should help you get started with contributing to the RLHF Audit Trail project. For specific questions or issues, don't hesitate to reach out to the development team!