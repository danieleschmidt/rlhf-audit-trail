"""Security and safety features for quantum task planner."""

import hashlib
import hmac
import secrets
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

from .core import Task, TaskState, QuantumPriority
from .exceptions import QuantumTaskPlannerError, ValidationError


class SecurityLevel(Enum):
    """Security levels for operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityConfig:
    """Security configuration for quantum task planner."""
    
    # Input validation
    max_task_name_length: int = 200
    max_description_length: int = 2000
    max_metadata_size: int = 10000
    allowed_metadata_keys: Optional[Set[str]] = None
    
    # Rate limiting
    max_tasks_per_minute: int = 100
    max_tasks_per_hour: int = 1000
    
    # Resource limits
    max_concurrent_tasks: int = 50
    max_total_tasks: int = 10000
    max_task_duration_hours: float = 24.0
    
    # Quantum safety
    max_entanglements_per_task: int = 10
    min_coherence_time_seconds: float = 1.0
    max_coherence_time_seconds: float = 86400.0
    
    # Logging and monitoring
    log_all_operations: bool = True
    audit_trail_enabled: bool = True
    sensitive_data_masking: bool = True


class SecurityManager:
    """Manages security policies and validation."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize security manager.
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.logger = logging.getLogger("quantum_planner.security")
        
        # Rate limiting tracking
        self.operation_history: List[float] = []
        
        # Operation counters
        self.task_creation_count = 0
        self.failed_operations = 0
        
    def validate_task_security(self, task: Task) -> bool:
        """Validate task meets security requirements.
        
        Args:
            task: Task to validate
            
        Returns:
            True if task is secure
            
        Raises:
            ValidationError: If task violates security policies
        """
        # Check task name length
        if len(task.name) > self.config.max_task_name_length:
            raise ValidationError(
                field="name",
                value=task.name,
                constraint=f"Name exceeds maximum length ({self.config.max_task_name_length})"
            )
        
        # Check description length
        if len(task.description) > self.config.max_description_length:
            raise ValidationError(
                field="description", 
                value=len(task.description),
                constraint=f"Description exceeds maximum length ({self.config.max_description_length})"
            )
        
        # Check metadata size
        if task.metadata:
            import json
            metadata_size = len(json.dumps(task.metadata, default=str))
            if metadata_size > self.config.max_metadata_size:
                raise ValidationError(
                    field="metadata",
                    value=metadata_size,
                    constraint=f"Metadata exceeds maximum size ({self.config.max_metadata_size})"
                )
            
            # Check allowed metadata keys
            if self.config.allowed_metadata_keys:
                invalid_keys = set(task.metadata.keys()) - self.config.allowed_metadata_keys
                if invalid_keys:
                    raise ValidationError(
                        field="metadata",
                        value=list(invalid_keys),
                        constraint=f"Metadata contains disallowed keys: {invalid_keys}"
                    )
        
        # Check task duration
        if task.estimated_duration > self.config.max_task_duration_hours:
            raise ValidationError(
                field="estimated_duration",
                value=task.estimated_duration,
                constraint=f"Duration exceeds maximum ({self.config.max_task_duration_hours} hours)"
            )
        
        # Check quantum properties
        if task.coherence_time < self.config.min_coherence_time_seconds:
            raise ValidationError(
                field="coherence_time",
                value=task.coherence_time,
                constraint=f"Coherence time below minimum ({self.config.min_coherence_time_seconds}s)"
            )
        
        if task.coherence_time > self.config.max_coherence_time_seconds:
            raise ValidationError(
                field="coherence_time",
                value=task.coherence_time,
                constraint=f"Coherence time exceeds maximum ({self.config.max_coherence_time_seconds}s)"
            )
        
        return True
    
    def check_rate_limits(self, operation: str) -> bool:
        """Check if operation is within rate limits.
        
        Args:
            operation: Type of operation being performed
            
        Returns:
            True if within limits
            
        Raises:
            QuantumTaskPlannerError: If rate limit exceeded
        """
        current_time = time.time()
        
        # Clean old entries (older than 1 hour)
        cutoff_time = current_time - 3600
        self.operation_history = [t for t in self.operation_history if t > cutoff_time]
        
        # Check minute rate limit
        minute_cutoff = current_time - 60
        recent_minute_ops = [t for t in self.operation_history if t > minute_cutoff]
        
        if len(recent_minute_ops) >= self.config.max_tasks_per_minute:
            raise QuantumTaskPlannerError(
                f"Rate limit exceeded: {len(recent_minute_ops)} operations in last minute "
                f"(limit: {self.config.max_tasks_per_minute})",
                error_code="RATE_LIMIT_EXCEEDED"
            )
        
        # Check hour rate limit
        if len(self.operation_history) >= self.config.max_tasks_per_hour:
            raise QuantumTaskPlannerError(
                f"Rate limit exceeded: {len(self.operation_history)} operations in last hour "
                f"(limit: {self.config.max_tasks_per_hour})",
                error_code="RATE_LIMIT_EXCEEDED"
            )
        
        # Record operation
        self.operation_history.append(current_time)
        return True
    
    def validate_resource_limits(
        self, 
        total_tasks: int,
        running_tasks: int,
        new_task_count: int = 1
    ) -> bool:
        """Validate resource usage limits.
        
        Args:
            total_tasks: Current total task count
            running_tasks: Current running task count
            new_task_count: Number of new tasks being added
            
        Returns:
            True if within limits
            
        Raises:
            QuantumTaskPlannerError: If resource limit would be exceeded
        """
        # Check total task limit
        if total_tasks + new_task_count > self.config.max_total_tasks:
            raise QuantumTaskPlannerError(
                f"Total task limit exceeded: {total_tasks + new_task_count} > "
                f"{self.config.max_total_tasks}",
                error_code="RESOURCE_LIMIT_EXCEEDED"
            )
        
        # Check concurrent task limit  
        if running_tasks >= self.config.max_concurrent_tasks:
            raise QuantumTaskPlannerError(
                f"Concurrent task limit exceeded: {running_tasks} >= "
                f"{self.config.max_concurrent_tasks}",
                error_code="RESOURCE_LIMIT_EXCEEDED"
            )
        
        return True
    
    def validate_entanglement_security(self, task1: Task, task2: Task) -> bool:
        """Validate entanglement meets security requirements.
        
        Args:
            task1: First task in entanglement
            task2: Second task in entanglement
            
        Returns:
            True if entanglement is secure
            
        Raises:
            ValidationError: If entanglement violates security policies
        """
        # Check entanglement limits
        if len(task1.entangled_tasks) >= self.config.max_entanglements_per_task:
            raise ValidationError(
                field="entangled_tasks",
                value=len(task1.entangled_tasks),
                constraint=f"Task {task1.id} exceeds maximum entanglements "
                           f"({self.config.max_entanglements_per_task})"
            )
        
        if len(task2.entangled_tasks) >= self.config.max_entanglements_per_task:
            raise ValidationError(
                field="entangled_tasks", 
                value=len(task2.entangled_tasks),
                constraint=f"Task {task2.id} exceeds maximum entanglements "
                           f"({self.config.max_entanglements_per_task})"
            )
        
        # Check priority compatibility (security policy)
        priority_values = {p: i for i, p in enumerate(QuantumPriority)}
        priority_diff = abs(priority_values[task1.priority] - priority_values[task2.priority])
        
        # Don't allow entanglement between critical and low priority tasks
        if priority_diff > 2:
            raise ValidationError(
                field="priority_compatibility",
                value=(task1.priority.value, task2.priority.value),
                constraint="Cannot entangle tasks with large priority differences"
            )
        
        return True
    
    def sanitize_input(self, input_data: Any, data_type: str) -> Any:
        """Sanitize input data to prevent injection attacks.
        
        Args:
            input_data: Data to sanitize
            data_type: Type of data being sanitized
            
        Returns:
            Sanitized data
        """
        if input_data is None:
            return None
        
        if data_type == "string":
            if isinstance(input_data, str):
                # Remove potentially dangerous characters
                dangerous_chars = ['<', '>', '"', "'", '&', '\x00', '\n', '\r']
                sanitized = input_data
                for char in dangerous_chars:
                    sanitized = sanitized.replace(char, '')
                return sanitized.strip()
            else:
                return str(input_data).strip()
        
        elif data_type == "metadata":
            if isinstance(input_data, dict):
                sanitized_metadata = {}
                for key, value in input_data.items():
                    # Sanitize keys and values
                    clean_key = self.sanitize_input(key, "string")[:50]  # Limit key length
                    clean_value = self.sanitize_input(value, "string")
                    
                    if clean_key and clean_value is not None:
                        sanitized_metadata[clean_key] = clean_value
                
                return sanitized_metadata
            else:
                return {}
        
        return input_data
    
    def mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data for logging.
        
        Args:
            data: Data dictionary that may contain sensitive information
            
        Returns:
            Data with sensitive fields masked
        """
        if not self.config.sensitive_data_masking:
            return data
        
        masked_data = data.copy()
        sensitive_keys = ['password', 'token', 'secret', 'key', 'auth', 'credential']
        
        def mask_value(key: str, value: Any) -> Any:
            if isinstance(key, str):
                key_lower = key.lower()
                if any(sensitive_key in key_lower for sensitive_key in sensitive_keys):
                    if isinstance(value, str) and len(value) > 4:
                        return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"
                    else:
                        return "***"
            
            if isinstance(value, dict):
                return {k: mask_value(k, v) for k, v in value.items()}
            elif isinstance(value, list):
                return [mask_value("", item) for item in value]
            
            return value
        
        for key, value in masked_data.items():
            masked_data[key] = mask_value(key, value)
        
        return masked_data
    
    def log_security_event(
        self, 
        event_type: str, 
        details: Dict[str, Any],
        severity: SecurityLevel = SecurityLevel.MEDIUM
    ):
        """Log security-related events.
        
        Args:
            event_type: Type of security event
            details: Event details
            severity: Security level of the event
        """
        if not self.config.log_all_operations:
            return
        
        # Mask sensitive data
        safe_details = self.mask_sensitive_data(details)
        
        log_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "severity": severity.value,
            "details": safe_details
        }
        
        if severity == SecurityLevel.CRITICAL:
            self.logger.critical(f"SECURITY EVENT: {event_type} - {safe_details}")
        elif severity == SecurityLevel.HIGH:
            self.logger.error(f"Security event: {event_type} - {safe_details}")
        elif severity == SecurityLevel.MEDIUM:
            self.logger.warning(f"Security event: {event_type} - {safe_details}")
        else:
            self.logger.info(f"Security event: {event_type} - {safe_details}")
    
    def generate_secure_id(self) -> str:
        """Generate cryptographically secure task ID.
        
        Returns:
            Secure random ID
        """
        return secrets.token_hex(16)
    
    def validate_task_permissions(
        self, 
        operation: str, 
        task: Task, 
        user_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Validate user has permissions for task operation.
        
        Args:
            operation: Operation being performed
            task: Task being operated on
            user_context: User context/credentials
            
        Returns:
            True if authorized
            
        Raises:
            QuantumTaskPlannerError: If not authorized
        """
        # Simplified permission check - in production would integrate with auth system
        if user_context is None:
            user_context = {"role": "user"}  # Default role
        
        user_role = user_context.get("role", "user")
        
        # Define operation permissions
        role_permissions = {
            "admin": ["create", "read", "update", "delete", "execute", "entangle"],
            "operator": ["create", "read", "update", "execute", "entangle"],
            "user": ["create", "read", "execute"],
            "viewer": ["read"]
        }
        
        allowed_operations = role_permissions.get(user_role, [])
        
        if operation not in allowed_operations:
            self.log_security_event(
                "unauthorized_operation",
                {
                    "operation": operation,
                    "user_role": user_role,
                    "task_id": task.id,
                    "task_name": task.name
                },
                SecurityLevel.HIGH
            )
            
            raise QuantumTaskPlannerError(
                f"User role '{user_role}' not authorized for operation '{operation}'",
                error_code="UNAUTHORIZED_OPERATION"
            )
        
        return True