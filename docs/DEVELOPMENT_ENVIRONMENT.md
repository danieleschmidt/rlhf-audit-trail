# Development Environment Guide

This guide provides comprehensive instructions for setting up and working with the RLHF Audit Trail development environment.

## Quick Start

### Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/danieleschmidt/rlhf-audit-trail.git
cd rlhf-audit-trail

# Run automated setup
./scripts/dev-setup.sh
```

The automated setup script will:
- Check prerequisites
- Create Python virtual environment
- Install all dependencies
- Setup pre-commit hooks
- Configure environment files
- Generate development keys
- Start database containers
- Run initial tests

### Manual Setup

If you prefer manual setup or need to troubleshoot:

1. **Prerequisites**
   ```bash
   # Ensure you have these installed:
   python3.10+
   git
   docker
   docker-compose
   make
   ```

2. **Python Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -e ".[dev,testing,docs]"
   ```

3. **Pre-commit Hooks**
   ```bash
   pre-commit install
   pre-commit install --hook-type commit-msg
   ```

4. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Database Setup**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d postgres redis
   alembic upgrade head
   ```

## Development Tools

### Code Quality Tools

The project uses multiple tools to maintain code quality:

#### Formatting
- **Black**: Python code formatter
- **isort**: Import statement organizer
- **Ruff**: Fast Python linter and formatter

```bash
# Format code
make format

# Check formatting
black --check src tests
ruff format --check src tests
```

#### Linting
- **Ruff**: Comprehensive Python linting
- **Bandit**: Security vulnerability scanner
- **MyPy**: Static type checking

```bash
# Run linting
make lint

# Run type checking
make type-check

# Run security scanning
make security
```

#### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit` and include:
- Code formatting (Black, isort, Ruff)
- Linting (Ruff, MyPy)
- Security scanning (Bandit, Safety)
- Documentation checks (pydocstyle)
- YAML/JSON validation
- Secret detection
- Trailing whitespace removal

```bash
# Run pre-commit on all files
pre-commit run --all-files

# Update pre-commit hooks
pre-commit autoupdate
```

### Testing Framework

#### Test Structure
```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
├── performance/   # Performance benchmarks
├── fixtures/      # Test data and fixtures
└── conftest.py    # Test configuration
```

#### Running Tests

```bash
# Quick unit tests
./scripts/test-quick.sh

# Full test suite with coverage
make test-cov

# Integration tests only
make test-integration

# Compliance tests
make test-compliance

# Performance benchmarks
make benchmark
```

#### Test Configuration

Tests are configured via `pytest.ini` and `pyproject.toml`:
- Markers for different test types
- Coverage reporting
- Test discovery patterns
- Performance benchmarking setup

### IDE Configuration

#### Visual Studio Code

The project includes comprehensive VS Code configuration:

**Extensions (automatically installed in DevContainer):**
- Python support (ms-python.python)
- Code formatting (Black, Ruff)
- Type checking (MyPy)
- Testing integration
- Docker support
- YAML/JSON support
- Spell checking

**Settings:**
- Python interpreter path
- Linting and formatting configuration
- Test discovery
- File associations
- Code completion

#### DevContainer Support

For consistent development environments:

```bash
# Open in VS Code with DevContainer
code .
# VS Code will prompt to reopen in container
```

The DevContainer includes:
- Python 3.10+ with all dependencies
- Database services (PostgreSQL, Redis)
- Development tools
- VS Code extensions
- Port forwarding for services

### Database Development

#### Local Database Setup

Development uses PostgreSQL and Redis via Docker:

```bash
# Start services
docker-compose -f docker-compose.dev.yml up -d

# Check service status
docker-compose -f docker-compose.dev.yml ps

# View logs
docker-compose -f docker-compose.dev.yml logs -f
```

#### Database Migrations

Using Alembic for database schema management:

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback migrations
alembic downgrade -1

# Reset database (destructive!)
make db-reset
```

#### Database Access

```bash
# Connect to development database
docker-compose -f docker-compose.dev.yml exec postgres psql -U postgres -d rlhf_audit_trail

# Connect to Redis
docker-compose -f docker-compose.dev.yml exec redis redis-cli
```

### Development Workflows

#### Feature Development

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Development Cycle**
   ```bash
   # Make changes
   # Run quick tests
   ./scripts/test-quick.sh
   
   # Run linting
   make lint
   
   # Commit changes (pre-commit hooks run automatically)
   git commit -m "feat: your feature description"
   ```

3. **Before Pushing**
   ```bash
   # Run full quality checks
   make check-all
   
   # Push changes
   git push origin feature/your-feature-name
   ```

#### Bug Fixing

1. **Create Bug Fix Branch**
   ```bash
   git checkout -b fix/bug-description
   ```

2. **Write Failing Test**
   ```python
   # Add test that reproduces the bug
   def test_bug_reproduction():
       # Test that currently fails
       pass
   ```

3. **Fix and Verify**
   ```bash
   # Fix the bug
   # Verify test passes
   pytest tests/unit/test_specific_module.py::test_bug_reproduction
   
   # Run full test suite
   make test
   ```

#### Compliance Verification

```bash
# Check EU AI Act compliance
python -m rlhf_audit_trail.compliance.check_eu_ai_act

# Check NIST requirements
python -m rlhf_audit_trail.compliance.check_nist_requirements

# Generate compliance report
python -m rlhf_audit_trail.compliance.generate_report
```

### Performance Monitoring

#### Development Metrics

```bash
# Run performance benchmarks
make benchmark

# Monitor system health
make health-check

# Check application metrics
curl http://localhost:8000/metrics
```

#### Profiling

```python
# CPU profiling
import cProfile
import pstats

def profile_function():
    # Your code here
    pass

cProfile.run('profile_function()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

### Debugging

#### Debug Configuration

VS Code debug configurations are provided in `.vscode/launch.json`:
- FastAPI application debugging
- Test debugging
- Streamlit dashboard debugging

#### Common Debug Tasks

```bash
# Debug with verbose logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with debugger
python -m debugpy --listen 5678 --wait-for-client src/rlhf_audit_trail/main.py

# Debug tests
python -m pytest --pdb tests/unit/test_module.py::test_function
```

### Security Development

#### Security Scanning

```bash
# Full security audit
make audit

# Dependency vulnerability scanning
safety check

# Secret detection
detect-secrets scan
```

#### Secure Coding Practices

- Use parameterized queries for database access
- Validate all input data
- Implement proper authentication and authorization
- Use cryptographic libraries correctly
- Regular dependency updates

### Documentation

#### Building Documentation

```bash
# Build documentation
make docs

# Serve documentation locally
make serve-docs
# Open http://localhost:8000
```

#### Documentation Standards

- Use Google-style docstrings
- Include type hints
- Provide usage examples
- Document compliance requirements
- Update API documentation automatically

### Troubleshooting

#### Common Issues

1. **Database Connection Issues**
   ```bash
   # Check if services are running
   docker-compose -f docker-compose.dev.yml ps
   
   # Restart services
   docker-compose -f docker-compose.dev.yml restart
   ```

2. **Import Errors**
   ```bash
   # Ensure PYTHONPATH is set
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   
   # Or use development installation
   pip install -e .
   ```

3. **Test Failures**
   ```bash
   # Run specific test with verbose output
   pytest -vvv tests/unit/test_module.py::test_function
   
   # Debug test
   pytest --pdb tests/unit/test_module.py::test_function
   ```

4. **Pre-commit Hook Failures**
   ```bash
   # Fix formatting issues
   make format
   
   # Run specific hook
   pre-commit run black --all-files
   
   # Skip hooks (not recommended)
   git commit --no-verify
   ```

#### Getting Help

- Check the [troubleshooting guide](troubleshooting.md)
- Review [GitHub Issues](https://github.com/danieleschmidt/rlhf-audit-trail/issues)
- Ask questions in [GitHub Discussions](https://github.com/danieleschmidt/rlhf-audit-trail/discussions)

### Development Scripts

The `scripts/` directory contains helpful development utilities:

- `dev-setup.sh`: Automated environment setup
- `test-quick.sh`: Quick unit test runner
- `test-full.sh`: Full test suite with coverage
- `dev-server.sh`: Development server launcher

### Best Practices

1. **Code Style**
   - Follow PEP 8 with Black formatting
   - Use type hints consistently
   - Write descriptive commit messages
   - Keep functions small and focused

2. **Testing**
   - Write tests before implementing features (TDD)
   - Maintain >90% code coverage
   - Use descriptive test names
   - Mock external dependencies

3. **Security**
   - Never commit secrets or credentials
   - Use environment variables for configuration
   - Validate all inputs
   - Regular security scanning

4. **Performance**
   - Profile code regularly
   - Monitor memory usage
   - Use async/await for I/O operations
   - Cache expensive computations

5. **Documentation**
   - Document public APIs
   - Include usage examples
   - Keep documentation up to date
   - Use clear, concise language