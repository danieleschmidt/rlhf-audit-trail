# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly:

**DO NOT** open a public GitHub issue for security vulnerabilities.

### Private Reporting

1. **Email**: Send details to `security@example.com`
2. **Include**: 
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Target 30 days for critical issues

## Security Considerations

### Cryptographic Components

- Audit trail integrity depends on merkle tree implementations
- Private key management is user responsibility
- Differential privacy parameters must be carefully tuned

### Data Protection

- Never log raw annotator data without privacy protection
- Ensure proper access controls on audit databases
- Implement secure storage for cryptographic keys

### Dependencies

- Regularly update dependencies for security patches
- Monitor for vulnerabilities in ML/crypto libraries
- Use pinned versions in production deployments

## Compliance Standards

This project aims to meet:
- EU AI Act transparency requirements
- NIST AI Risk Management Framework
- SOC 2 Type II controls (where applicable)

## Security Best Practices

When contributing:
- Use parameterized queries for database operations
- Validate all inputs, especially cryptographic parameters
- Follow principle of least privilege
- Document security assumptions clearly

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help improve our security posture.