"""RLHF Audit Trail - Verifiable provenance for RLHF with EU AI Act compliance."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

# Main API exports
from .core import AuditableRLHF
from .config import PrivacyConfig, SecurityConfig, ComplianceConfig
from .exceptions import AuditTrailError, PrivacyBudgetExceededError

__all__ = [
    "AuditableRLHF",
    "PrivacyConfig",
    "SecurityConfig", 
    "ComplianceConfig",
    "AuditTrailError",
    "PrivacyBudgetExceededError",
]
