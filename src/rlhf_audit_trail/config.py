"""Configuration classes for RLHF Audit Trail system.

This module defines configuration dataclasses for privacy, security, and
compliance settings used throughout the audit trail system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class PrivacyMode(Enum):
    """Privacy protection modes."""
    MINIMAL = "minimal"      # Basic anonymization only
    MODERATE = "moderate"    # Standard differential privacy
    STRONG = "strong"        # High privacy with strict bounds


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    EU_AI_ACT = "eu_ai_act"
    NIST_DRAFT = "nist_draft"
    GDPR = "gdpr"
    CCPA = "ccpa"


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy and annotator protection.
    
    This configuration controls how the system applies differential privacy
    to protect annotator identities and sensitive training data.
    """
    
    # Core differential privacy parameters
    epsilon: float = 1.0                    # Total privacy budget
    delta: float = 1e-5                     # Delta parameter for (ε,δ)-DP
    epsilon_per_round: float = 0.1           # Per-round epsilon allocation
    noise_multiplier: float = 1.1           # Noise scaling factor
    clip_norm: float = 1.0                  # Gradient clipping norm
    
    # Privacy mode and advanced settings
    privacy_mode: PrivacyMode = PrivacyMode.MODERATE
    annotator_privacy_mode: str = "strong"  # Annotator-specific privacy level
    
    # Composition and budget management
    max_rounds: int = 100                   # Maximum training rounds
    budget_alert_threshold: float = 0.8     # Alert when budget 80% used
    auto_budget_management: bool = True     # Automatically manage budget allocation
    
    # Advanced privacy techniques
    use_amplification: bool = True          # Use privacy amplification by sampling
    sampling_rate: float = 0.01             # Subsampling rate for amplification
    use_moments_accountant: bool = True     # Use moments accountant for tighter bounds
    
    # Annotator anonymization
    annotator_k_anonymity: int = 5          # Minimum k-anonymity for annotators
    annotator_id_rotation: bool = True      # Rotate annotator IDs periodically
    rotation_frequency: int = 50            # Rotation frequency (batches)
    
    def validate(self) -> List[str]:
        """Validate privacy configuration and return any issues."""
        issues = []
        
        if self.epsilon <= 0:
            issues.append("epsilon must be positive")
        if self.delta <= 0 or self.delta >= 1:
            issues.append("delta must be in (0, 1)")
        if self.epsilon_per_round > self.epsilon:
            issues.append("epsilon_per_round cannot exceed total epsilon")
        if self.clip_norm <= 0:
            issues.append("clip_norm must be positive")
        if self.sampling_rate <= 0 or self.sampling_rate > 1:
            issues.append("sampling_rate must be in (0, 1]")
        if self.annotator_k_anonymity < 2:
            issues.append("annotator_k_anonymity must be at least 2")
            
        return issues


@dataclass  
class SecurityConfig:
    """Configuration for cryptographic security and data protection.
    
    This configuration controls encryption, hashing, and digital signatures
    used to ensure data integrity and authenticity.
    """
    
    # Encryption settings
    encryption_algorithm: str = "AES-256"   # Encryption algorithm
    key_size: int = 256                     # Encryption key size in bits
    key_rotation_days: int = 90             # Key rotation frequency
    use_hsm: bool = False                   # Use Hardware Security Module
    
    # Hashing and integrity
    hash_algorithm: str = "SHA-256"         # Hash algorithm for integrity
    merkle_tree_hash: str = "SHA-256"       # Hash for merkle tree construction
    salt_length: int = 32                   # Salt length for hashing
    
    # Digital signatures
    signature_algorithm: str = "RSA-PSS"    # Digital signature algorithm
    signature_key_size: int = 4096          # Signature key size in bits
    signature_hash: str = "SHA-256"         # Hash for signatures
    
    # Key management
    key_derivation_iterations: int = 100000  # PBKDF2 iterations
    key_storage_backend: str = "local"      # Key storage ("local", "vault", "hsm")
    key_backup_enabled: bool = True         # Enable key backup
    
    # Transport security
    tls_version: str = "1.3"                # Minimum TLS version
    certificate_validation: bool = True     # Validate TLS certificates
    
    # Audit log protection
    log_encryption_enabled: bool = True     # Encrypt audit logs
    log_signature_enabled: bool = True      # Sign audit log entries
    tamper_detection: bool = True           # Enable tamper detection
    
    def validate(self) -> List[str]:
        """Validate security configuration and return any issues."""
        issues = []
        
        valid_encryption = ["AES-128", "AES-192", "AES-256", "ChaCha20"]
        if self.encryption_algorithm not in valid_encryption:
            issues.append(f"Invalid encryption algorithm: {self.encryption_algorithm}")
            
        valid_hash = ["SHA-256", "SHA-384", "SHA-512", "SHA-3-256", "BLAKE2b"]
        if self.hash_algorithm not in valid_hash:
            issues.append(f"Invalid hash algorithm: {self.hash_algorithm}")
            
        if self.key_size not in [128, 192, 256]:
            issues.append(f"Invalid key size: {self.key_size}")
            
        if self.signature_key_size < 2048:
            issues.append("Signature key size should be at least 2048 bits")
            
        if self.key_rotation_days < 1:
            issues.append("Key rotation days must be positive")
            
        return issues


@dataclass
class ComplianceConfig:
    """Configuration for regulatory compliance validation.
    
    This configuration controls how the system validates compliance with
    various regulatory frameworks like EU AI Act, NIST, GDPR, etc.
    """
    
    # Supported frameworks
    frameworks: List[ComplianceFramework] = field(default_factory=lambda: [
        ComplianceFramework.EU_AI_ACT,
        ComplianceFramework.GDPR
    ])
    
    # EU AI Act specific settings
    eu_ai_act_risk_category: str = "high"   # Risk category: "minimal", "limited", "high", "unacceptable"
    eu_conformity_assessment: bool = True    # Require conformity assessment
    eu_ce_marking: bool = False             # CE marking required
    eu_fundamental_rights_assessment: bool = True  # Fundamental rights impact assessment
    
    # NIST AI Risk Management Framework
    nist_risk_tier: str = "tier3"           # Risk management tier (tier1-tier4)
    nist_trustworthiness_characteristics: List[str] = field(default_factory=lambda: [
        "reliability", "safety", "fairness", "explainability", "accountability", "privacy", "security"
    ])
    
    # Data governance requirements
    data_retention_days: int = 2555         # 7 years (EU AI Act requirement)
    audit_log_retention_days: int = 3650    # 10 years for audit logs
    data_subject_rights: bool = True        # Support GDPR data subject rights
    right_to_explanation: bool = True       # Provide algorithmic explanations
    
    # Documentation requirements
    technical_documentation: bool = True     # Maintain technical documentation
    risk_management_system: bool = True     # Implement risk management system
    quality_management_system: bool = True  # Quality management system
    post_market_monitoring: bool = True     # Post-market monitoring system
    
    # Transparency and reporting
    transparency_reporting: bool = True     # Generate transparency reports
    algorithmic_impact_assessment: bool = True  # Conduct algorithmic impact assessments
    bias_monitoring: bool = True            # Monitor for algorithmic bias
    fairness_metrics: List[str] = field(default_factory=lambda: [
        "demographic_parity", "equalized_odds", "calibration"
    ])
    
    # Human oversight requirements
    human_oversight_required: bool = True   # Require human oversight
    human_override_capability: bool = True  # Enable human override
    oversight_competency_requirements: bool = True  # Competency requirements for oversight
    
    # Validation and testing
    conformity_assessment_required: bool = True  # Third-party conformity assessment
    validation_dataset_requirements: bool = True  # Validation dataset requirements
    robustness_testing: bool = True         # Adversarial robustness testing
    performance_monitoring: bool = True     # Continuous performance monitoring
    
    # Notification and registration
    regulatory_notification: bool = False   # Notify regulatory authorities
    system_registration: bool = False       # Register system with authorities
    incident_reporting: bool = True         # Report serious incidents
    
    def validate(self) -> List[str]:
        """Validate compliance configuration and return any issues."""
        issues = []
        
        if not self.frameworks:
            issues.append("At least one compliance framework must be specified")
            
        if self.eu_ai_act_risk_category not in ["minimal", "limited", "high", "unacceptable"]:
            issues.append(f"Invalid EU AI Act risk category: {self.eu_ai_act_risk_category}")
            
        if self.nist_risk_tier not in ["tier1", "tier2", "tier3", "tier4"]:
            issues.append(f"Invalid NIST risk tier: {self.nist_risk_tier}")
            
        if self.data_retention_days < 1:
            issues.append("Data retention days must be positive")
            
        if self.audit_log_retention_days < self.data_retention_days:
            issues.append("Audit log retention should be at least as long as data retention")
            
        return issues


@dataclass
class MonitoringConfig:
    """Configuration for system monitoring and observability."""
    
    # Metrics collection
    metrics_enabled: bool = True
    metrics_endpoint: str = "/metrics"
    metrics_port: int = 9090
    
    # Health checks
    health_check_enabled: bool = True
    health_check_interval: int = 30         # seconds
    health_check_timeout: int = 10          # seconds
    
    # Performance monitoring
    performance_monitoring: bool = True
    latency_percentiles: List[float] = field(default_factory=lambda: [0.5, 0.9, 0.95, 0.99])
    slow_query_threshold: float = 1.0       # seconds
    
    # Alerting
    alerting_enabled: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    
    # Alert thresholds
    error_rate_threshold: float = 0.05      # 5% error rate
    latency_threshold: float = 2.0          # 2 seconds
    memory_threshold: float = 0.85          # 85% memory usage
    disk_threshold: float = 0.90            # 90% disk usage
    
    # Logging
    log_level: str = "INFO"
    structured_logging: bool = True
    log_retention_days: int = 90
    
    def validate(self) -> List[str]:
        """Validate monitoring configuration."""
        issues = []
        
        if self.health_check_interval <= 0:
            issues.append("Health check interval must be positive")
            
        if self.health_check_timeout <= 0:
            issues.append("Health check timeout must be positive")
            
        if not (0 <= self.error_rate_threshold <= 1):
            issues.append("Error rate threshold must be between 0 and 1")
            
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            issues.append(f"Invalid log level: {self.log_level}")
            
        return issues


@dataclass
class SystemConfig:
    """Main system configuration combining all component configs."""
    
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # System-wide settings
    environment: str = "development"        # "development", "staging", "production"
    debug_mode: bool = False
    max_concurrent_sessions: int = 10
    session_timeout_hours: int = 24
    
    # Storage configuration
    storage_backend: str = "local"          # "local", "s3", "gcp", "azure"
    storage_encryption: bool = True
    storage_compression: bool = True
    
    # API configuration
    api_rate_limit: int = 1000              # requests per hour
    api_max_request_size: int = 10485760    # 10MB
    api_timeout: int = 30                   # seconds
    
    def validate(self) -> List[str]:
        """Validate entire system configuration."""
        issues = []
        
        # Validate component configs
        issues.extend(self.privacy.validate())
        issues.extend(self.security.validate())
        issues.extend(self.compliance.validate())
        issues.extend(self.monitoring.validate())
        
        # System-wide validation
        if self.environment not in ["development", "staging", "production"]:
            issues.append(f"Invalid environment: {self.environment}")
            
        if self.max_concurrent_sessions <= 0:
            issues.append("Max concurrent sessions must be positive")
            
        if self.session_timeout_hours <= 0:
            issues.append("Session timeout must be positive")
            
        if self.api_rate_limit <= 0:
            issues.append("API rate limit must be positive")
            
        return issues
    
    @classmethod
    def for_environment(cls, env: str) -> "SystemConfig":
        """Create configuration optimized for specific environment."""
        config = cls()
        config.environment = env
        
        if env == "production":
            # Production settings
            config.debug_mode = False
            config.privacy.privacy_mode = PrivacyMode.STRONG
            config.security.key_rotation_days = 30
            config.compliance.conformity_assessment_required = True
            config.monitoring.health_check_interval = 15
            
        elif env == "staging":
            # Staging settings
            config.debug_mode = False
            config.privacy.privacy_mode = PrivacyMode.MODERATE
            config.security.key_rotation_days = 60
            config.monitoring.health_check_interval = 30
            
        else:  # development
            # Development settings
            config.debug_mode = True
            config.privacy.privacy_mode = PrivacyMode.MINIMAL
            config.security.key_rotation_days = 365
            config.compliance.conformity_assessment_required = False
            config.monitoring.health_check_interval = 60
            
        return config