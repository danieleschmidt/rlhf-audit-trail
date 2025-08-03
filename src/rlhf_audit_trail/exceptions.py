"""Custom exceptions for RLHF Audit Trail system.

This module defines all custom exceptions used throughout the audit trail
system, providing clear error handling and informative error messages.
"""

from typing import Dict, List, Optional, Any


class AuditTrailError(Exception):
    """Base exception for all audit trail related errors.
    
    This is the parent class for all custom exceptions in the system,
    providing common functionality for error handling and reporting.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize audit trail error.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "AUDIT_TRAIL_ERROR"
        self.details = details or {}
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "original_error": str(self.original_error) if self.original_error else None
        }


class PrivacyBudgetExceededError(AuditTrailError):
    """Raised when differential privacy budget would be exceeded.
    
    This exception is thrown when an operation would consume more privacy
    budget than is available, preventing potential privacy violations.
    """
    
    def __init__(
        self,
        message: str,
        requested_epsilon: Optional[float] = None,
        available_epsilon: Optional[float] = None,
        session_id: Optional[str] = None
    ):
        """Initialize privacy budget exceeded error.
        
        Args:
            message: Error message
            requested_epsilon: Epsilon amount requested
            available_epsilon: Epsilon amount available
            session_id: Training session ID
        """
        details = {}
        if requested_epsilon is not None:
            details["requested_epsilon"] = requested_epsilon
        if available_epsilon is not None:
            details["available_epsilon"] = available_epsilon
        if session_id is not None:
            details["session_id"] = session_id
            
        super().__init__(
            message=message,
            error_code="PRIVACY_BUDGET_EXCEEDED",
            details=details
        )
        
        self.requested_epsilon = requested_epsilon
        self.available_epsilon = available_epsilon
        self.session_id = session_id


class ComplianceViolationError(AuditTrailError):
    """Raised when a compliance violation is detected.
    
    This exception indicates that an operation or system state violates
    regulatory compliance requirements.
    """
    
    def __init__(
        self,
        message: str,
        framework: Optional[str] = None,
        violations: Optional[List[str]] = None,
        severity: str = "high"
    ):
        """Initialize compliance violation error.
        
        Args:
            message: Error message
            framework: Compliance framework (e.g., "EU_AI_ACT")
            violations: List of specific violations
            severity: Violation severity ("low", "medium", "high", "critical")
        """
        details = {
            "severity": severity
        }
        if framework:
            details["framework"] = framework
        if violations:
            details["violations"] = violations
            
        super().__init__(
            message=message,
            error_code="COMPLIANCE_VIOLATION",
            details=details
        )
        
        self.framework = framework
        self.violations = violations or []
        self.severity = severity


class CryptographicError(AuditTrailError):
    """Raised when cryptographic operations fail.
    
    This exception covers errors in encryption, decryption, hashing,
    digital signatures, and integrity verification.
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        algorithm: Optional[str] = None
    ):
        """Initialize cryptographic error.
        
        Args:
            message: Error message
            operation: Cryptographic operation that failed
            algorithm: Algorithm being used
        """
        details = {}
        if operation:
            details["operation"] = operation
        if algorithm:
            details["algorithm"] = algorithm
            
        super().__init__(
            message=message,
            error_code="CRYPTOGRAPHIC_ERROR",
            details=details
        )
        
        self.operation = operation
        self.algorithm = algorithm


class StorageError(AuditTrailError):
    """Raised when storage operations fail.
    
    This exception covers errors in reading, writing, or managing
    audit trail data in various storage backends.
    """
    
    def __init__(
        self,
        message: str,
        storage_backend: Optional[str] = None,
        operation: Optional[str] = None,
        path: Optional[str] = None
    ):
        """Initialize storage error.
        
        Args:
            message: Error message
            storage_backend: Storage backend type
            operation: Storage operation that failed
            path: File/object path involved
        """
        details = {}
        if storage_backend:
            details["storage_backend"] = storage_backend
        if operation:
            details["operation"] = operation
        if path:
            details["path"] = path
            
        super().__init__(
            message=message,
            error_code="STORAGE_ERROR",
            details=details
        )
        
        self.storage_backend = storage_backend
        self.operation = operation
        self.path = path


class ValidationError(AuditTrailError):
    """Raised when data validation fails.
    
    This exception is thrown when input data fails validation checks,
    including schema validation, business rule validation, etc.
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        validation_rules: Optional[List[str]] = None
    ):
        """Initialize validation error.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
            validation_rules: Validation rules that were violated
        """
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        if validation_rules:
            details["validation_rules"] = validation_rules
            
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )
        
        self.field = field
        self.value = value
        self.validation_rules = validation_rules or []


class SessionError(AuditTrailError):
    """Raised when training session operations fail.
    
    This exception covers errors related to training session management,
    including session state issues, invalid operations, etc.
    """
    
    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        session_state: Optional[str] = None,
        operation: Optional[str] = None
    ):
        """Initialize session error.
        
        Args:
            message: Error message
            session_id: Training session ID
            session_state: Current session state
            operation: Operation that failed
        """
        details = {}
        if session_id:
            details["session_id"] = session_id
        if session_state:
            details["session_state"] = session_state
        if operation:
            details["operation"] = operation
            
        super().__init__(
            message=message,
            error_code="SESSION_ERROR",
            details=details
        )
        
        self.session_id = session_id
        self.session_state = session_state
        self.operation = operation


class IntegrityError(AuditTrailError):
    """Raised when data integrity verification fails.
    
    This exception indicates that stored data has been tampered with
    or corrupted, based on cryptographic verification.
    """
    
    def __init__(
        self,
        message: str,
        data_path: Optional[str] = None,
        expected_hash: Optional[str] = None,
        actual_hash: Optional[str] = None,
        verification_method: Optional[str] = None
    ):
        """Initialize integrity error.
        
        Args:
            message: Error message
            data_path: Path to compromised data
            expected_hash: Expected hash value
            actual_hash: Actual hash value
            verification_method: Verification method used
        """
        details = {}
        if data_path:
            details["data_path"] = data_path
        if expected_hash:
            details["expected_hash"] = expected_hash
        if actual_hash:
            details["actual_hash"] = actual_hash
        if verification_method:
            details["verification_method"] = verification_method
            
        super().__init__(
            message=message,
            error_code="INTEGRITY_ERROR",
            details=details
        )
        
        self.data_path = data_path
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        self.verification_method = verification_method


class ConfigurationError(AuditTrailError):
    """Raised when system configuration is invalid.
    
    This exception is thrown when the system configuration contains
    invalid values or conflicting settings.
    """
    
    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        config_field: Optional[str] = None,
        invalid_value: Optional[Any] = None
    ):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_section: Configuration section with error
            config_field: Specific field with invalid value
            invalid_value: The invalid value
        """
        details = {}
        if config_section:
            details["config_section"] = config_section
        if config_field:
            details["config_field"] = config_field
        if invalid_value is not None:
            details["invalid_value"] = str(invalid_value)
            
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details
        )
        
        self.config_section = config_section
        self.config_field = config_field
        self.invalid_value = invalid_value


class MLIntegrationError(AuditTrailError):
    """Raised when ML library integration fails.
    
    This exception covers errors when integrating with external ML
    libraries like TRL, transformers, PyTorch, etc.
    """
    
    def __init__(
        self,
        message: str,
        library: Optional[str] = None,
        library_version: Optional[str] = None,
        operation: Optional[str] = None
    ):
        """Initialize ML integration error.
        
        Args:
            message: Error message
            library: ML library name
            library_version: Library version
            operation: Integration operation that failed
        """
        details = {}
        if library:
            details["library"] = library
        if library_version:
            details["library_version"] = library_version
        if operation:
            details["operation"] = operation
            
        super().__init__(
            message=message,
            error_code="ML_INTEGRATION_ERROR",
            details=details
        )
        
        self.library = library
        self.library_version = library_version
        self.operation = operation


class AuthenticationError(AuditTrailError):
    """Raised when authentication fails.
    
    This exception is thrown when user authentication or authorization
    fails in the audit trail system.
    """
    
    def __init__(
        self,
        message: str,
        auth_method: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Initialize authentication error.
        
        Args:
            message: Error message
            auth_method: Authentication method used
            user_id: User ID that failed authentication
        """
        details = {}
        if auth_method:
            details["auth_method"] = auth_method
        if user_id:
            details["user_id"] = user_id
            
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            details=details
        )
        
        self.auth_method = auth_method
        self.user_id = user_id


class NetworkError(AuditTrailError):
    """Raised when network operations fail.
    
    This exception covers network-related errors including connection
    failures, timeouts, and API communication issues.
    """
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        timeout: Optional[bool] = False
    ):
        """Initialize network error.
        
        Args:
            message: Error message
            endpoint: Network endpoint that failed
            status_code: HTTP status code
            timeout: Whether error was due to timeout
        """
        details = {}
        if endpoint:
            details["endpoint"] = endpoint
        if status_code:
            details["status_code"] = status_code
        if timeout:
            details["timeout"] = timeout
            
        super().__init__(
            message=message,
            error_code="NETWORK_ERROR",
            details=details
        )
        
        self.endpoint = endpoint
        self.status_code = status_code
        self.timeout = timeout


class ResourceExhaustedError(AuditTrailError):
    """Raised when system resources are exhausted.
    
    This exception is thrown when the system runs out of critical
    resources like memory, disk space, or compute capacity.
    """
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None
    ):
        """Initialize resource exhausted error.
        
        Args:
            message: Error message
            resource_type: Type of resource exhausted
            current_usage: Current resource usage
            limit: Resource limit
        """
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if current_usage is not None:
            details["current_usage"] = current_usage
        if limit is not None:
            details["limit"] = limit
            
        super().__init__(
            message=message,
            error_code="RESOURCE_EXHAUSTED",
            details=details
        )
        
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


# Exception hierarchy summary for documentation
EXCEPTION_HIERARCHY = {
    "AuditTrailError": {
        "description": "Base exception for all audit trail errors",
        "children": [
            "PrivacyBudgetExceededError",
            "ComplianceViolationError", 
            "CryptographicError",
            "StorageError",
            "ValidationError",
            "SessionError",
            "IntegrityError",
            "ConfigurationError",
            "MLIntegrationError",
            "AuthenticationError",
            "NetworkError",
            "ResourceExhaustedError"
        ]
    }
}


def get_error_details(error: Exception) -> Dict[str, Any]:
    """Extract error details from any exception for logging.
    
    Args:
        error: Exception to extract details from
        
    Returns:
        Dictionary containing error details
    """
    if isinstance(error, AuditTrailError):
        return error.to_dict()
    else:
        return {
            "error_type": type(error).__name__,
            "message": str(error),
            "is_audit_trail_error": False
        }