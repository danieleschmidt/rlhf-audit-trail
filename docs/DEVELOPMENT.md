# Development Guide

## Development Environment Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rlhf-audit-trail.git
cd rlhf-audit-trail

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

The `dev` extra includes:
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **black**: Code formatting
- **ruff**: Fast linting
- **mypy**: Type checking
- **pre-commit**: Git hooks

## Project Structure

```
rlhf-audit-trail/
├── src/rlhf_audit_trail/     # Main package
├── tests/                    # Test suite
├── docs/                     # Documentation
├── examples/                 # Usage examples
├── scripts/                  # Utility scripts
├── pyproject.toml           # Project configuration
└── README.md               # Main documentation
```

## Development Workflow

### 1. Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy src/

# Run all quality checks
pre-commit run --all-files
```

### 2. Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_audit.py

# Run with verbose output
pytest -v
```

### 3. Building and Installation

```bash
# Build package
python -m build

# Install locally
pip install -e .

# Install with extras
pip install -e ".[aws,dev]"
```

## Debugging and Profiling

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling

```bash
# Profile with cProfile
python -m cProfile -o profile.stats your_script.py

# Analyze results
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## Security Testing

### Dependency Scanning

```bash
# Check for known vulnerabilities
pip-audit

# Check with safety
safety check
```

### Privacy Testing

```bash
# Run privacy analysis tests
pytest tests/privacy/ -v

# Generate privacy report
python -m rlhf_audit_trail.analyze_privacy --config tests/fixtures/privacy_config.yaml
```

## Documentation

### Building Docs

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation (when implemented)
# sphinx-build -b html docs/ docs/_build/
```

### Code Documentation

- Use Google-style docstrings
- Include type hints for all public functions
- Document security considerations clearly
- Provide usage examples in docstrings

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release tag
5. Build and publish to PyPI

## Troubleshooting

### Common Issues

**Import errors**: Ensure you've installed the package with `pip install -e .`

**Test failures**: Check that all dependencies are installed and virtual environment is activated

**Pre-commit failures**: Run `pre-commit run --all-files` to fix formatting issues

### Getting Help

- Check existing issues on GitHub
- Read the documentation thoroughly
- Ask questions in discussions