"""Basic tests for RLHF Audit Trail package."""

import pytest
from rlhf_audit_trail import __version__


def test_version():
    """Test that version is defined."""
    assert __version__ == "0.1.0"


def test_package_imports():
    """Test that package can be imported."""
    import rlhf_audit_trail
    assert hasattr(rlhf_audit_trail, '__version__')


# Placeholder for future tests
@pytest.mark.skip(reason="Implementation pending")
def test_auditable_rlhf():
    """Test AuditableRLHF functionality."""
    pass


@pytest.mark.skip(reason="Implementation pending") 
def test_privacy_config():
    """Test PrivacyConfig functionality."""
    pass