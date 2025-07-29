# Contributing to RLHF Audit Trail

Thank you for your interest in contributing to the RLHF Audit Trail project! This document provides guidelines for contributing to this EU AI Act and NIST compliance-focused tool.

## 🚀 Quick Start

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create** a feature branch: `git checkout -b feature/your-feature-name`
4. **Install** development dependencies: `pip install -e ".[dev]"`
5. **Set up** pre-commit hooks: `pre-commit install`
6. **Make** your changes
7. **Test** your changes: `pytest`
8. **Submit** a pull request

## 🛠️ Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/rlhf-audit-trail.git
cd rlhf-audit-trail

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 📋 Development Guidelines

### Code Style
- **Black** for code formatting
- **Ruff** for linting and code quality
- **MyPy** for type checking
- Follow **PEP 8** and **PEP 484** standards
- Maximum line length: **88 characters**

### Testing
- Write tests for all new functionality
- Maintain **90%+ test coverage**
- Use **pytest** for testing framework
- Include compliance-specific tests when applicable

### Documentation
- Update docstrings using **Google style**
- Update README.md for new features
- Add examples to `examples/` directory
- Document compliance implications

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m compliance    # Compliance tests only

# Run tests for specific Python versions
tox
```

## 🔒 Security Considerations

This project handles sensitive data and compliance requirements:

- **Never commit** API keys, credentials, or personal data
- **Test privacy** features thoroughly
- **Validate compliance** with EU AI Act requirements
- **Document security** implications of changes
- **Review cryptographic** implementations carefully

## 📝 Pull Request Process

1. **Update documentation** for any new features
2. **Add tests** covering the new functionality
3. **Ensure CI passes** (tests, linting, type checking)
4. **Update CHANGELOG.md** with your changes
5. **Request review** from maintainers
6. **Address feedback** promptly

### PR Title Format
```
feat: add differential privacy noise calibration
fix: resolve memory leak in audit log storage
docs: update compliance documentation
test: add integration tests for merkle verification
```

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `compliance`: EU AI Act or NIST related
- `security`: Security-related issue
- `privacy`: Privacy-related feature
- `documentation`: Documentation improvements
- `good-first-issue`: Good for newcomers
- `help-wanted`: Extra attention needed

## 🤝 Community Guidelines

- Be **respectful** and **inclusive**
- **Ask questions** if requirements are unclear
- **Share knowledge** about compliance and privacy
- **Help others** in discussions and reviews
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)

## 📚 Key Areas for Contribution

### High Priority
- **Compliance frameworks** (additional regulatory support)
- **Privacy-preserving techniques** (advanced DP methods)
- **Storage backends** (additional cloud providers)
- **Visualization improvements** (better dashboards)

### Medium Priority
- **Performance optimizations**
- **Additional RLHF library integrations**
- **Documentation enhancements**
- **Example applications**

### Compliance-Specific Contributions
- **EU AI Act** requirement implementations
- **NIST framework** alignment features
- **Audit report** generation improvements
- **Privacy analysis** tooling

## 🔧 Development Tools

### Pre-commit Hooks
We use pre-commit hooks to ensure code quality:
```bash
pre-commit run --all-files  # Run on all files
pre-commit run <hook-name>  # Run specific hook
```

### Useful Commands
```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy src

# Update dependencies
pip-compile requirements.in
```

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🆘 Getting Help

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Email**: compliance@terragonlabs.com for sensitive issues

## 🙏 Recognition

Contributors are recognized in:
- **AUTHORS.md** file
- **Release notes** for significant contributions
- **README.md** acknowledgments section

Thank you for helping make RLHF more transparent and compliant! 🚀
