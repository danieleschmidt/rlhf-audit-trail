"""
Comprehensive validation system for RLHF audit trail inputs and outputs.
Generation 2: Robust validation, error handling, and security measures.
"""

import re
import hashlib
import uuid
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from pydantic import BaseModel, Field, validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    ValidationError = Exception

from .exceptions import ValidationError as AuditValidationError


logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error" 
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    field: str
    message: str
    code: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int = 0
    errors_count: int = 0
    
    def __post_init__(self):
        self.warnings_count = sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)
        self.errors_count = sum(1 for issue in self.issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
    
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return self.errors_count > 0
    
    def get_error_messages(self) -> List[str]:
        """Get all error messages."""
        return [
            issue.message for issue in self.issues 
            if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
        ]


class SecurityValidator:
    """Security-focused input validation."""
    
    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = {
        'sql_injection': [
            r"(?i)(union|select|insert|update|delete|drop|create|alter)\s+",
            r"(?i)(\bor\b|\band\b)\s+\d+\s*=\s*\d+",
            r"(?i)--\s*$",
            r"(?i)/\*.*?\*/"
        ],
        'script_injection': [
            r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
            r"javascript:",
            r"on\w+\s*=",
            r"eval\s*\(",
            r"setTimeout\s*\(",
            r"setInterval\s*\("
        ],
        'path_traversal': [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e%5c"
        ],
        'command_injection': [
            r"[;&|`]",
            r"\$\(",
            r"`.*`",
            r"\|\s*\w+"
        ]
    }
    
    @classmethod
    def validate_string_security(cls, value: str, field_name: str) -> List[ValidationIssue]:
        """Validate string for security vulnerabilities."""
        issues = []
        
        for category, patterns in cls.DANGEROUS_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, value):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        field=field_name,
                        message=f"Potential {category.replace('_', ' ')} detected",
                        code=f"SECURITY_{category.upper()}",
                        suggestion="Sanitize input or use parameterized queries"
                    ))
        
        # Check for excessive length
        if len(value) > 10000:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field=field_name,
                message="String length exceeds recommended limit (10K characters)",
                code="LENGTH_WARNING"
            ))
        
        return issues


class DataValidator:
    """Data format and content validation."""
    
    @staticmethod
    def validate_uuid(value: str, field_name: str) -> List[ValidationIssue]:
        """Validate UUID format."""
        issues = []
        try:
            uuid.UUID(value)
        except (ValueError, TypeError):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field=field_name,
                message="Invalid UUID format",
                code="INVALID_UUID",
                suggestion="Use valid UUID v4 format"
            ))
        return issues
    
    @staticmethod
    def validate_email(value: str, field_name: str) -> List[ValidationIssue]:
        """Validate email format."""
        issues = []
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, value):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field=field_name,
                message="Invalid email format",
                code="INVALID_EMAIL"
            ))
        return issues
    
    @staticmethod
    def validate_model_name(value: str, field_name: str) -> List[ValidationIssue]:
        """Validate model name format."""
        issues = []
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9._/-]+$', value):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field=field_name,
                message="Model name contains invalid characters",
                code="INVALID_MODEL_NAME",
                suggestion="Use only alphanumeric characters, dots, hyphens, and slashes"
            ))
        
        # Check length
        if len(value) < 3 or len(value) > 128:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field=field_name,
                message="Model name must be 3-128 characters long",
                code="INVALID_LENGTH"
            ))
        
        return issues
    
    @staticmethod
    def validate_rewards(values: List[float], field_name: str) -> List[ValidationIssue]:
        """Validate reward values."""
        issues = []
        
        for i, reward in enumerate(values):
            if not isinstance(reward, (int, float)):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field=f"{field_name}[{i}]",
                    message="Reward must be a number",
                    code="INVALID_TYPE"
                ))
            elif reward < -10.0 or reward > 10.0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=f"{field_name}[{i}]",
                    message="Reward outside typical range [-10, 10]",
                    code="UNUSUAL_RANGE"
                ))
        
        return issues
    
    @staticmethod
    def validate_privacy_config(config: Dict[str, Any], field_name: str) -> List[ValidationIssue]:
        """Validate privacy configuration."""
        issues = []
        
        # Validate epsilon
        epsilon = config.get('epsilon')
        if epsilon is not None:
            if not isinstance(epsilon, (int, float)) or epsilon <= 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field=f"{field_name}.epsilon",
                    message="Epsilon must be a positive number",
                    code="INVALID_EPSILON"
                ))
            elif epsilon > 50.0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=f"{field_name}.epsilon",
                    message="Very high epsilon value may compromise privacy",
                    code="HIGH_EPSILON_WARNING"
                ))
        
        # Validate delta
        delta = config.get('delta')
        if delta is not None:
            if not isinstance(delta, (int, float)) or delta <= 0 or delta >= 1:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field=f"{field_name}.delta",
                    message="Delta must be between 0 and 1 (exclusive)",
                    code="INVALID_DELTA"
                ))
        
        return issues


class BusinessLogicValidator:
    """Business logic and domain-specific validation."""
    
    @staticmethod
    def validate_batch_consistency(
        prompts: List[str], 
        responses: List[str], 
        rewards: List[float], 
        annotator_ids: List[str]
    ) -> List[ValidationIssue]:
        """Validate annotation batch consistency."""
        issues = []
        
        lengths = {
            'prompts': len(prompts),
            'responses': len(responses),
            'rewards': len(rewards),
            'annotator_ids': len(annotator_ids)
        }
        
        # Check if all lists have same length
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="batch",
                message=f"Inconsistent batch sizes: {lengths}",
                code="INCONSISTENT_BATCH_SIZE",
                suggestion="Ensure all input lists have the same length"
            ))
        
        # Check minimum batch size
        min_length = min(lengths.values())
        if min_length == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field="batch",
                message="Empty batch not allowed",
                code="EMPTY_BATCH"
            ))
        elif min_length > 1000:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="batch",
                message="Large batch size may impact performance",
                code="LARGE_BATCH_WARNING"
            ))
        
        return issues
    
    @staticmethod
    def validate_session_lifecycle(
        session_start: datetime,
        current_time: Optional[datetime] = None
    ) -> List[ValidationIssue]:
        """Validate training session lifecycle."""
        issues = []
        
        if current_time is None:
            current_time = datetime.utcnow()
        
        # Check if session is too old
        session_age = current_time - session_start
        if session_age > timedelta(days=7):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="session",
                message="Training session is older than 7 days",
                code="OLD_SESSION_WARNING"
            ))
        elif session_age > timedelta(days=30):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field="session",
                message="Training session is older than 30 days (expired)",
                code="EXPIRED_SESSION"
            ))
        
        return issues


class ComprehensiveValidator:
    """Main validation orchestrator."""
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
        self.security_validator = SecurityValidator()
        self.data_validator = DataValidator()
        self.business_validator = BusinessLogicValidator()
        
    def validate_annotation_batch(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
        annotator_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Comprehensive validation of annotation batch."""
        all_issues = []
        
        # Business logic validation
        all_issues.extend(
            self.business_validator.validate_batch_consistency(
                prompts, responses, rewards, annotator_ids
            )
        )
        
        # Security validation for prompts and responses
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                all_issues.extend(
                    self.security_validator.validate_string_security(prompt, f"prompts[{i}]")
                )
        
        for i, response in enumerate(responses):
            if isinstance(response, str):
                all_issues.extend(
                    self.security_validator.validate_string_security(response, f"responses[{i}]")
                )
        
        # Data format validation
        all_issues.extend(
            self.data_validator.validate_rewards(rewards, "rewards")
        )
        
        # Validate annotator IDs
        for i, annotator_id in enumerate(annotator_ids):
            if not isinstance(annotator_id, str) or len(annotator_id) < 3:
                all_issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field=f"annotator_ids[{i}]",
                    message="Annotator ID must be a string with at least 3 characters",
                    code="INVALID_ANNOTATOR_ID"
                ))
        
        # Validate metadata if provided
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    all_issues.extend(
                        self.security_validator.validate_string_security(value, f"metadata.{key}")
                    )
        
        return self._build_result(all_issues)
    
    def validate_training_session(
        self,
        experiment_name: str,
        model_name: str,
        privacy_config: Optional[Dict[str, Any]] = None,
        session_start: Optional[datetime] = None
    ) -> ValidationResult:
        """Validate training session parameters."""
        all_issues = []
        
        # Validate experiment name
        all_issues.extend(
            self.security_validator.validate_string_security(experiment_name, "experiment_name")
        )
        
        if not experiment_name.strip():
            all_issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field="experiment_name",
                message="Experiment name cannot be empty",
                code="EMPTY_FIELD"
            ))
        
        # Validate model name
        all_issues.extend(
            self.data_validator.validate_model_name(model_name, "model_name")
        )
        
        # Validate privacy config
        if privacy_config:
            all_issues.extend(
                self.data_validator.validate_privacy_config(privacy_config, "privacy_config")
            )
        
        # Validate session lifecycle
        if session_start:
            all_issues.extend(
                self.business_validator.validate_session_lifecycle(session_start)
            )
        
        return self._build_result(all_issues)
    
    def validate_policy_update(
        self,
        loss: float,
        learning_rate: float,
        gradient_norm: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate policy update parameters."""
        all_issues = []
        
        # Validate loss
        if not isinstance(loss, (int, float)):
            all_issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field="loss",
                message="Loss must be a number",
                code="INVALID_TYPE"
            ))
        elif loss < 0:
            all_issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="loss",
                message="Negative loss value is unusual",
                code="UNUSUAL_VALUE"
            ))
        elif loss > 100:
            all_issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="loss",
                message="Very high loss value",
                code="HIGH_LOSS"
            ))
        
        # Validate learning rate
        if not isinstance(learning_rate, (int, float)):
            all_issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field="learning_rate",
                message="Learning rate must be a number",
                code="INVALID_TYPE"
            ))
        elif learning_rate <= 0 or learning_rate > 1:
            all_issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="learning_rate",
                message="Learning rate outside typical range (0, 1]",
                code="UNUSUAL_LEARNING_RATE"
            ))
        
        # Validate gradient norm
        if not isinstance(gradient_norm, (int, float)):
            all_issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field="gradient_norm",
                message="Gradient norm must be a number",
                code="INVALID_TYPE"
            ))
        elif gradient_norm < 0:
            all_issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field="gradient_norm",
                message="Gradient norm cannot be negative",
                code="INVALID_GRADIENT_NORM"
            ))
        elif gradient_norm > 1000:
            all_issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="gradient_norm",
                message="Very large gradient norm may indicate instability",
                code="LARGE_GRADIENT"
            ))
        
        # Validate metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    all_issues.extend(
                        self.security_validator.validate_string_security(value, f"metadata.{key}")
                    )
        
        return self._build_result(all_issues)
    
    def _build_result(self, issues: List[ValidationIssue]) -> ValidationResult:
        """Build validation result from issues."""
        errors = [issue for issue in issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        warnings = [issue for issue in issues if issue.severity == ValidationSeverity.WARNING]
        
        is_valid = len(errors) == 0
        if self.strict_mode:
            is_valid = is_valid and len(warnings) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues
        )


# Global validator instance
_default_validator = ComprehensiveValidator(strict_mode=False)

def validate_annotation_batch(*args, **kwargs) -> ValidationResult:
    """Convenience function for annotation batch validation."""
    return _default_validator.validate_annotation_batch(*args, **kwargs)

def validate_training_session(*args, **kwargs) -> ValidationResult:
    """Convenience function for training session validation."""
    return _default_validator.validate_training_session(*args, **kwargs)

def validate_policy_update(*args, **kwargs) -> ValidationResult:
    """Convenience function for policy update validation."""
    return _default_validator.validate_policy_update(*args, **kwargs)