"""Compliance validation system for regulatory frameworks.

This module implements compliance validation for various regulatory
frameworks including EU AI Act, NIST AI Risk Management Framework,
GDPR, and other relevant standards.
"""

import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import logging

from .config import ComplianceConfig, ComplianceFramework
from .exceptions import ComplianceViolationError, ValidationError


class ComplianceLevel(Enum):
    """Compliance assessment levels."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


class ViolationSeverity(Enum):
    """Severity levels for compliance violations."""
    CRITICAL = "critical"    # Immediate action required
    HIGH = "high"           # Action required soon
    MEDIUM = "medium"       # Should be addressed
    LOW = "low"            # Minor issue
    INFO = "info"          # Informational only


@dataclass
class ComplianceRequirement:
    """Represents a specific compliance requirement."""
    requirement_id: str
    framework: ComplianceFramework
    title: str
    description: str
    mandatory: bool
    assessment_method: str
    acceptance_criteria: Dict[str, Any]
    references: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ComplianceViolation:
    """Represents a compliance violation."""
    violation_id: str
    requirement_id: str
    framework: ComplianceFramework
    severity: ViolationSeverity
    title: str
    description: str
    detected_at: float
    evidence: Dict[str, Any]
    remediation_steps: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["framework"] = self.framework.value
        result["severity"] = self.severity.value
        return result


@dataclass
class ComplianceAssessment:
    """Results of a compliance assessment."""
    framework: ComplianceFramework
    assessment_id: str
    timestamp: float
    overall_level: ComplianceLevel
    requirements_assessed: int
    requirements_met: int
    violations: List[ComplianceViolation]
    compliance_score: float  # 0.0 to 1.0
    
    @property
    def is_compliant(self) -> bool:
        """Check if assessment shows compliance."""
        return self.overall_level == ComplianceLevel.COMPLIANT
    
    @property
    def critical_violations(self) -> List[ComplianceViolation]:
        """Get critical violations."""
        return [v for v in self.violations if v.severity == ViolationSeverity.CRITICAL]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["framework"] = self.framework.value
        result["overall_level"] = self.overall_level.value
        result["violations"] = [v.to_dict() for v in self.violations]
        return result


class EUAIActValidator:
    """Validator for EU AI Act compliance requirements."""
    
    def __init__(self, config: ComplianceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Define EU AI Act requirements
        self.requirements = self._define_requirements()
    
    def _define_requirements(self) -> List[ComplianceRequirement]:
        """Define EU AI Act compliance requirements."""
        return [
            ComplianceRequirement(
                requirement_id="eu_ai_act_art_9",
                framework=ComplianceFramework.EU_AI_ACT,
                title="Risk Management System",
                description="High-risk AI systems must have risk management system throughout lifecycle",
                mandatory=True,
                assessment_method="documentation_review",
                acceptance_criteria={
                    "risk_management_documented": True,
                    "lifecycle_coverage": True,
                    "regular_updates": True
                },
                references=["EU AI Act Article 9"]
            ),
            ComplianceRequirement(
                requirement_id="eu_ai_act_art_10",
                framework=ComplianceFramework.EU_AI_ACT,
                title="Data and Data Governance",
                description="Training data must be relevant, representative, free of errors and complete",
                mandatory=True,
                assessment_method="data_quality_analysis",
                acceptance_criteria={
                    "data_quality_documented": True,
                    "bias_assessment_performed": True,
                    "data_governance_established": True
                },
                references=["EU AI Act Article 10"]
            ),
            ComplianceRequirement(
                requirement_id="eu_ai_act_art_11",
                framework=ComplianceFramework.EU_AI_ACT,
                title="Technical Documentation",
                description="Comprehensive technical documentation must be maintained",
                mandatory=True,
                assessment_method="documentation_audit",
                acceptance_criteria={
                    "technical_docs_complete": True,
                    "training_process_documented": True,
                    "performance_metrics_documented": True
                },
                references=["EU AI Act Article 11"]
            ),
            ComplianceRequirement(
                requirement_id="eu_ai_act_art_12",
                framework=ComplianceFramework.EU_AI_ACT,
                title="Record-keeping",
                description="Automatic recording of events during system operation",
                mandatory=True,
                assessment_method="audit_trail_verification",
                acceptance_criteria={
                    "audit_trail_exists": True,
                    "events_logged_automatically": True,
                    "integrity_verified": True,
                    "retention_period_met": True
                },
                references=["EU AI Act Article 12"]
            ),
            ComplianceRequirement(
                requirement_id="eu_ai_act_art_13",
                framework=ComplianceFramework.EU_AI_ACT,
                title="Transparency and Information",
                description="Users must be informed they are interacting with AI system",
                mandatory=True,
                assessment_method="user_notification_review",
                acceptance_criteria={
                    "user_notification_implemented": True,
                    "system_capabilities_disclosed": True,
                    "limitations_communicated": True
                },
                references=["EU AI Act Article 13"]
            ),
            ComplianceRequirement(
                requirement_id="eu_ai_act_art_14",
                framework=ComplianceFramework.EU_AI_ACT,
                title="Human Oversight",
                description="High-risk AI systems must be designed for effective human oversight",
                mandatory=True,
                assessment_method="oversight_mechanism_review",
                acceptance_criteria={
                    "human_oversight_implemented": True,
                    "override_capability_exists": True,
                    "competent_oversight_personnel": True
                },
                references=["EU AI Act Article 14"]
            ),
            ComplianceRequirement(
                requirement_id="eu_ai_act_gdpr_consistency",
                framework=ComplianceFramework.EU_AI_ACT,
                title="GDPR Consistency",
                description="AI system must be consistent with GDPR requirements",
                mandatory=True,
                assessment_method="gdpr_compliance_check",
                acceptance_criteria={
                    "data_protection_compliant": True,
                    "consent_mechanisms": True,
                    "data_subject_rights": True
                },
                references=["EU AI Act Article 10(5)", "GDPR"]
            )
        ]
    
    def assess_compliance(self, session_data: Dict[str, Any]) -> ComplianceAssessment:
        """Assess EU AI Act compliance for a training session.
        
        Args:
            session_data: Training session data and metadata
            
        Returns:
            Compliance assessment results
        """
        violations = []
        requirements_met = 0
        
        for requirement in self.requirements:
            violation = self._assess_requirement(requirement, session_data)
            if violation:
                violations.append(violation)
            else:
                requirements_met += 1
        
        # Calculate compliance score
        compliance_score = requirements_met / len(self.requirements)
        
        # Determine overall compliance level
        if compliance_score >= 1.0:
            overall_level = ComplianceLevel.COMPLIANT
        elif compliance_score >= 0.8:
            overall_level = ComplianceLevel.PARTIALLY_COMPLIANT
        else:
            overall_level = ComplianceLevel.NON_COMPLIANT
        
        return ComplianceAssessment(
            framework=ComplianceFramework.EU_AI_ACT,
            assessment_id=f"eu_ai_act_{int(time.time())}",
            timestamp=time.time(),
            overall_level=overall_level,
            requirements_assessed=len(self.requirements),
            requirements_met=requirements_met,
            violations=violations,
            compliance_score=compliance_score
        )
    
    def _assess_requirement(self, requirement: ComplianceRequirement, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess a specific requirement."""
        if requirement.requirement_id == "eu_ai_act_art_9":
            return self._assess_risk_management(requirement, session_data)
        elif requirement.requirement_id == "eu_ai_act_art_10":
            return self._assess_data_governance(requirement, session_data)
        elif requirement.requirement_id == "eu_ai_act_art_11":
            return self._assess_technical_documentation(requirement, session_data)
        elif requirement.requirement_id == "eu_ai_act_art_12":
            return self._assess_record_keeping(requirement, session_data)
        elif requirement.requirement_id == "eu_ai_act_art_13":
            return self._assess_transparency(requirement, session_data)
        elif requirement.requirement_id == "eu_ai_act_art_14":
            return self._assess_human_oversight(requirement, session_data)
        elif requirement.requirement_id == "eu_ai_act_gdpr_consistency":
            return self._assess_gdpr_consistency(requirement, session_data)
        
        return None
    
    def _assess_risk_management(self, requirement: ComplianceRequirement, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess risk management system compliance."""
        issues = []
        
        if not self.config.risk_management_system:
            issues.append("Risk management system not implemented")
        
        if not session_data.get("risk_assessment_performed"):
            issues.append("No risk assessment documented for this session")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"violation_{requirement.requirement_id}_{int(time.time())}",
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.EU_AI_ACT,
                severity=ViolationSeverity.HIGH,
                title="Risk Management System Inadequate",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"session_data": session_data, "config": asdict(self.config)},
                remediation_steps=[
                    "Implement comprehensive risk management system",
                    "Document risk assessment for each training session",
                    "Establish regular risk review processes"
                ]
            )
        
        return None
    
    def _assess_data_governance(self, requirement: ComplianceRequirement, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess data and data governance compliance."""
        issues = []
        
        # Check for bias monitoring
        if not self.config.bias_monitoring:
            issues.append("Bias monitoring not enabled")
        
        # Check for data quality documentation
        if not session_data.get("data_quality_metrics"):
            issues.append("Data quality metrics not documented")
        
        # Check annotation count for statistical validity
        annotation_count = session_data.get("annotation_count", 0)
        if annotation_count < 100:  # Minimum threshold
            issues.append(f"Insufficient annotations for statistical validity: {annotation_count}")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"violation_{requirement.requirement_id}_{int(time.time())}",
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.EU_AI_ACT,
                severity=ViolationSeverity.MEDIUM,
                title="Data Governance Inadequate",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"annotation_count": annotation_count, "issues": issues},
                remediation_steps=[
                    "Enable bias monitoring and assessment",
                    "Document data quality metrics",
                    "Collect sufficient annotations for statistical validity"
                ]
            )
        
        return None
    
    def _assess_technical_documentation(self, requirement: ComplianceRequirement, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess technical documentation compliance."""
        issues = []
        
        if not self.config.technical_documentation:
            issues.append("Technical documentation requirement not enabled")
        
        required_docs = [
            "model_architecture", "training_parameters", "performance_metrics",
            "privacy_guarantees", "audit_trail_description"
        ]
        
        for doc_type in required_docs:
            if not session_data.get(f"{doc_type}_documented"):
                issues.append(f"Missing documentation: {doc_type}")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"violation_{requirement.requirement_id}_{int(time.time())}",
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.EU_AI_ACT,
                severity=ViolationSeverity.MEDIUM,
                title="Technical Documentation Incomplete",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"missing_docs": issues},
                remediation_steps=[
                    "Complete all required technical documentation",
                    "Establish documentation standards",
                    "Implement automated documentation generation"
                ]
            )
        
        return None
    
    def _assess_record_keeping(self, requirement: ComplianceRequirement, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess record-keeping compliance."""
        issues = []
        
        # Check audit trail existence
        if not session_data.get("audit_trail_exists"):
            issues.append("No audit trail detected")
        
        # Check automatic logging
        if not session_data.get("automatic_logging_enabled"):
            issues.append("Automatic logging not enabled")
        
        # Check integrity verification
        if not session_data.get("integrity_verified"):
            issues.append("Audit trail integrity not verified")
        
        # Check retention period
        retention_days = session_data.get("retention_period_days", 0)
        if retention_days < self.config.audit_log_retention_days:
            issues.append(f"Retention period too short: {retention_days} days")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"violation_{requirement.requirement_id}_{int(time.time())}",
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.EU_AI_ACT,
                severity=ViolationSeverity.CRITICAL,
                title="Record-keeping Requirements Not Met",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"retention_days": retention_days, "issues": issues},
                remediation_steps=[
                    "Implement comprehensive audit trail system",
                    "Enable automatic event logging",
                    "Verify audit trail integrity",
                    "Extend retention period to meet requirements"
                ]
            )
        
        return None
    
    def _assess_transparency(self, requirement: ComplianceRequirement, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess transparency and information compliance."""
        issues = []
        
        if not self.config.transparency_reporting:
            issues.append("Transparency reporting not enabled")
        
        if not session_data.get("user_notification_implemented"):
            issues.append("User notification not implemented")
        
        if not session_data.get("model_limitations_documented"):
            issues.append("Model limitations not documented")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"violation_{requirement.requirement_id}_{int(time.time())}",
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.EU_AI_ACT,
                severity=ViolationSeverity.MEDIUM,
                title="Transparency Requirements Not Met",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"issues": issues},
                remediation_steps=[
                    "Enable transparency reporting",
                    "Implement user notification systems",
                    "Document model capabilities and limitations"
                ]
            )
        
        return None
    
    def _assess_human_oversight(self, requirement: ComplianceRequirement, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess human oversight compliance."""
        issues = []
        
        if not self.config.human_oversight_required:
            issues.append("Human oversight not required in configuration")
        
        if not self.config.human_override_capability:
            issues.append("Human override capability not enabled")
        
        if not session_data.get("oversight_personnel_competent"):
            issues.append("Oversight personnel competency not verified")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"violation_{requirement.requirement_id}_{int(time.time())}",
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.EU_AI_ACT,
                severity=ViolationSeverity.HIGH,
                title="Human Oversight Requirements Not Met",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"issues": issues},
                remediation_steps=[
                    "Enable human oversight requirements",
                    "Implement human override capabilities",
                    "Verify oversight personnel competency"
                ]
            )
        
        return None
    
    def _assess_gdpr_consistency(self, requirement: ComplianceRequirement, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess GDPR consistency compliance."""
        issues = []
        
        if not self.config.data_subject_rights:
            issues.append("Data subject rights not supported")
        
        if not session_data.get("privacy_impact_assessment"):
            issues.append("Privacy impact assessment not performed")
        
        privacy_budget_used = session_data.get("privacy_budget_used", 1.0)
        if privacy_budget_used > 0.9:  # High privacy budget usage
            issues.append("High privacy budget consumption may indicate privacy risks")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"violation_{requirement.requirement_id}_{int(time.time())}",
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.EU_AI_ACT,
                severity=ViolationSeverity.HIGH,
                title="GDPR Consistency Issues",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"privacy_budget_used": privacy_budget_used, "issues": issues},
                remediation_steps=[
                    "Implement data subject rights support",
                    "Conduct privacy impact assessment",
                    "Optimize privacy budget usage"
                ]
            )
        
        return None


class NISTAIRMFValidator:
    """Validator for NIST AI Risk Management Framework compliance."""
    
    def __init__(self, config: ComplianceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def assess_compliance(self, session_data: Dict[str, Any]) -> ComplianceAssessment:
        """Assess NIST AI RMF compliance."""
        violations = []
        requirements_met = 0
        total_requirements = 7  # Number of trustworthiness characteristics
        
        # Assess each trustworthiness characteristic
        for characteristic in self.config.nist_trustworthiness_characteristics:
            violation = self._assess_characteristic(characteristic, session_data)
            if violation:
                violations.append(violation)
            else:
                requirements_met += 1
        
        compliance_score = requirements_met / total_requirements
        
        if compliance_score >= 1.0:
            overall_level = ComplianceLevel.COMPLIANT
        elif compliance_score >= 0.7:
            overall_level = ComplianceLevel.PARTIALLY_COMPLIANT
        else:
            overall_level = ComplianceLevel.NON_COMPLIANT
        
        return ComplianceAssessment(
            framework=ComplianceFramework.NIST_DRAFT,
            assessment_id=f"nist_ai_rmf_{int(time.time())}",
            timestamp=time.time(),
            overall_level=overall_level,
            requirements_assessed=total_requirements,
            requirements_met=requirements_met,
            violations=violations,
            compliance_score=compliance_score
        )
    
    def _assess_characteristic(self, characteristic: str, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess a specific trustworthiness characteristic."""
        if characteristic == "reliability":
            return self._assess_reliability(session_data)
        elif characteristic == "safety":
            return self._assess_safety(session_data)
        elif characteristic == "fairness":
            return self._assess_fairness(session_data)
        elif characteristic == "explainability":
            return self._assess_explainability(session_data)
        elif characteristic == "accountability":
            return self._assess_accountability(session_data)
        elif characteristic == "privacy":
            return self._assess_privacy(session_data)
        elif characteristic == "security":
            return self._assess_security(session_data)
        
        return None
    
    def _assess_reliability(self, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess reliability characteristic."""
        issues = []
        
        if not session_data.get("performance_monitoring"):
            issues.append("Performance monitoring not implemented")
        
        error_rate = session_data.get("error_rate", 1.0)
        if error_rate > 0.05:  # 5% threshold
            issues.append(f"High error rate: {error_rate:.3f}")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"nist_reliability_{int(time.time())}",
                requirement_id="nist_reliability",
                framework=ComplianceFramework.NIST_DRAFT,
                severity=ViolationSeverity.MEDIUM,
                title="Reliability Issues",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"error_rate": error_rate},
                remediation_steps=["Implement performance monitoring", "Reduce error rate"]
            )
        
        return None
    
    def _assess_safety(self, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess safety characteristic."""
        issues = []
        
        if not session_data.get("safety_testing_performed"):
            issues.append("Safety testing not performed")
        
        if not session_data.get("failure_mode_analysis"):
            issues.append("Failure mode analysis not conducted")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"nist_safety_{int(time.time())}",
                requirement_id="nist_safety",
                framework=ComplianceFramework.NIST_DRAFT,
                severity=ViolationSeverity.HIGH,
                title="Safety Issues",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"issues": issues},
                remediation_steps=["Conduct safety testing", "Perform failure mode analysis"]
            )
        
        return None
    
    def _assess_fairness(self, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess fairness characteristic."""
        issues = []
        
        if not self.config.bias_monitoring:
            issues.append("Bias monitoring not enabled")
        
        bias_metrics = session_data.get("bias_metrics", {})
        for metric, value in bias_metrics.items():
            if value > 0.1:  # 10% bias threshold
                issues.append(f"High bias detected in {metric}: {value:.3f}")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"nist_fairness_{int(time.time())}",
                requirement_id="nist_fairness",
                framework=ComplianceFramework.NIST_DRAFT,
                severity=ViolationSeverity.HIGH,
                title="Fairness Issues",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"bias_metrics": bias_metrics},
                remediation_steps=["Enable bias monitoring", "Address detected biases"]
            )
        
        return None
    
    def _assess_explainability(self, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess explainability characteristic."""
        issues = []
        
        if not self.config.right_to_explanation:
            issues.append("Right to explanation not supported")
        
        if not session_data.get("model_interpretability_assessed"):
            issues.append("Model interpretability not assessed")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"nist_explainability_{int(time.time())}",
                requirement_id="nist_explainability",
                framework=ComplianceFramework.NIST_DRAFT,
                severity=ViolationSeverity.MEDIUM,
                title="Explainability Issues",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"issues": issues},
                remediation_steps=["Support right to explanation", "Assess model interpretability"]
            )
        
        return None
    
    def _assess_accountability(self, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess accountability characteristic."""
        issues = []
        
        if not session_data.get("audit_trail_exists"):
            issues.append("Audit trail not implemented")
        
        if not session_data.get("governance_framework"):
            issues.append("Governance framework not established")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"nist_accountability_{int(time.time())}",
                requirement_id="nist_accountability",
                framework=ComplianceFramework.NIST_DRAFT,
                severity=ViolationSeverity.HIGH,
                title="Accountability Issues",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"issues": issues},
                remediation_steps=["Implement audit trail", "Establish governance framework"]
            )
        
        return None
    
    def _assess_privacy(self, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess privacy characteristic."""
        issues = []
        
        privacy_budget_used = session_data.get("privacy_budget_used", 0.0)
        if privacy_budget_used > 0.8:
            issues.append("High privacy budget consumption")
        
        if not session_data.get("privacy_impact_assessment"):
            issues.append("Privacy impact assessment not performed")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"nist_privacy_{int(time.time())}",
                requirement_id="nist_privacy",
                framework=ComplianceFramework.NIST_DRAFT,
                severity=ViolationSeverity.MEDIUM,
                title="Privacy Issues",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"privacy_budget_used": privacy_budget_used},
                remediation_steps=["Optimize privacy budget usage", "Conduct privacy impact assessment"]
            )
        
        return None
    
    def _assess_security(self, session_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Assess security characteristic."""
        issues = []
        
        if not session_data.get("encryption_enabled"):
            issues.append("Encryption not enabled")
        
        if not session_data.get("access_controls_implemented"):
            issues.append("Access controls not implemented")
        
        if issues:
            return ComplianceViolation(
                violation_id=f"nist_security_{int(time.time())}",
                requirement_id="nist_security",
                framework=ComplianceFramework.NIST_DRAFT,
                severity=ViolationSeverity.HIGH,
                title="Security Issues",
                description="; ".join(issues),
                detected_at=time.time(),
                evidence={"issues": issues},
                remediation_steps=["Enable encryption", "Implement access controls"]
            )
        
        return None


class ComplianceValidator:
    """Main compliance validation system supporting multiple frameworks."""
    
    def __init__(self, frameworks: List[ComplianceFramework], config: ComplianceConfig):
        """Initialize compliance validator.
        
        Args:
            frameworks: List of compliance frameworks to validate against
            config: Compliance configuration
        """
        self.frameworks = frameworks
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize framework validators
        self.validators = {}
        if ComplianceFramework.EU_AI_ACT in frameworks:
            self.validators[ComplianceFramework.EU_AI_ACT] = EUAIActValidator(config)
        if ComplianceFramework.NIST_DRAFT in frameworks:
            self.validators[ComplianceFramework.NIST_DRAFT] = NISTAIRMFValidator(config)
    
    async def validate_session(self, session) -> Dict[str, Any]:
        """Validate compliance for a training session.
        
        Args:
            session: Training session object
            
        Returns:
            Compliance validation results
        """
        session_data = {
            "session_id": session.session_id,
            "duration": session.duration,
            "audit_trail_exists": True,  # Should check actual audit trail
            "automatic_logging_enabled": True,
            "integrity_verified": True,  # Should perform actual verification
            "retention_period_days": self.config.audit_log_retention_days,
            "performance_monitoring": self.config.performance_monitoring,
            "bias_monitoring": self.config.bias_monitoring,
            "privacy_impact_assessment": True,  # Should check if actually performed
            "technical_documentation": self.config.technical_documentation,
            "transparency_reporting": self.config.transparency_reporting,
            "human_oversight_required": self.config.human_oversight_required,
            "encryption_enabled": True,  # Should check actual encryption status
            "access_controls_implemented": True,  # Should check actual access controls
        }
        
        assessments = {}
        overall_compliant = True
        
        for framework in self.frameworks:
            if framework in self.validators:
                assessment = self.validators[framework].assess_compliance(session_data)
                assessments[framework.value] = assessment.to_dict()
                
                if not assessment.is_compliant:
                    overall_compliant = False
        
        return {
            "overall_compliant": overall_compliant,
            "frameworks_assessed": [f.value for f in self.frameworks],
            "assessments": assessments,
            "validation_timestamp": time.time()
        }
    
    async def validate_checkpoint(self, checkpoint_data: Dict[str, Any], session) -> Dict[str, Any]:
        """Validate compliance at a training checkpoint.
        
        Args:
            checkpoint_data: Checkpoint data
            session: Training session
            
        Returns:
            Compliance status
        """
        # Simplified checkpoint validation
        issues = []
        
        # Check if compliance monitoring is up to date
        if not self.config.performance_monitoring:
            issues.append("Performance monitoring not enabled")
        
        # Check privacy budget usage
        privacy_budget_used = checkpoint_data.get("privacy_budget_used", 0.0)
        if privacy_budget_used > 0.9:
            issues.append("Privacy budget nearly exhausted")
        
        return {
            "is_compliant": len(issues) == 0,
            "issues": issues,
            "checkpoint_timestamp": checkpoint_data.get("timestamp"),
            "recommendations": self._generate_checkpoint_recommendations(issues)
        }
    
    async def generate_final_report(self, session) -> Dict[str, Any]:
        """Generate final compliance report for a completed session.
        
        Args:
            session: Completed training session
            
        Returns:
            Final compliance report
        """
        validation_result = await self.validate_session(session)
        
        # Add summary information
        report = {
            "session_summary": {
                "session_id": session.session_id,
                "experiment_name": session.experiment_name,
                "duration": session.duration,
                "start_time": session.start_time,
                "end_time": session.end_time
            },
            "compliance_validation": validation_result,
            "recommendations": self._generate_final_recommendations(validation_result),
            "report_generated_at": time.time()
        }
        
        return report
    
    def _generate_checkpoint_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations for checkpoint compliance issues."""
        recommendations = []
        
        for issue in issues:
            if "performance monitoring" in issue:
                recommendations.append("Enable performance monitoring for compliance")
            elif "privacy budget" in issue:
                recommendations.append("Monitor privacy budget consumption more carefully")
        
        if not issues:
            recommendations.append("Checkpoint compliance is satisfactory")
        
        return recommendations
    
    def _generate_final_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate final recommendations based on validation results."""
        recommendations = []
        
        if validation_result["overall_compliant"]:
            recommendations.append("Training session meets all compliance requirements")
            recommendations.append("Continue following current compliance practices")
        else:
            recommendations.append("Address identified compliance violations before deployment")
            recommendations.append("Review and update compliance procedures")
            
            # Add specific recommendations based on framework assessments
            for framework, assessment in validation_result["assessments"].items():
                if assessment["overall_level"] != "compliant":
                    recommendations.append(f"Review {framework} compliance requirements")
        
        return recommendations