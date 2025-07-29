# Contributing to RLHF Audit Trail

We welcome contributions to improve RLHF transparency and regulatory compliance!

## Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and test
4. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/rlhf-audit-trail.git
cd rlhf-audit-trail

# Set up development environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all public functions
- Add docstrings for modules, classes, and functions
- Write tests for new functionality
- Run linting: `ruff check .`
- Format code: `black .`
- Type check: `mypy src/`

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_audit.py
```

## Security Considerations

- Never commit API keys, secrets, or private data
- Validate all inputs in security-critical functions
- Follow differential privacy best practices
- Document cryptographic assumptions clearly

## Pull Request Process

1. Update documentation for any API changes
2. Ensure all tests pass and coverage is maintained
3. Follow conventional commit format
4. Request review from core maintainers

## Compliance Focus Areas

We especially welcome contributions in:
- EU AI Act compliance features
- NIST framework alignment
- Privacy-preserving techniques
- Cryptographic verification methods
- Additional storage backends

## Questions?

Open an issue or contact the maintainers directly.