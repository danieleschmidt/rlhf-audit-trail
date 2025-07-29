#!/usr/bin/env python3
"""
Automated Compliance Validator for RLHF Audit Trail
Validates compliance with EU AI Act and other regulatory frameworks
"""

import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ComplianceStatus(Enum):
    """Compliance check status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class ComplianceResult:
    """Result of a compliance check."""
    check_id: str
    title: str
    status: ComplianceStatus
    level: ComplianceLevel
    message: str
    evidence: Optional[str] = None
    remediation: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class ComplianceValidator:
    """Main compliance validation engine."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the compliance validator."""
        self.config_path = config_path or Path(__file__).parent / "eu-ai-act-checklist.yml"
        self.results: List[ComplianceResult] = []
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load compliance configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded compliance configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load compliance config: {e}")
            self.config = {}
    
    def run_all_checks(self) -> List[ComplianceResult]:
        """Run all compliance checks."""
        logger.info("Starting comprehensive compliance validation")
        
        self.results = []
        
        # Run different categories of checks
        self._check_risk_management()
        self._check_data_governance()
        self._check_technical_documentation()
        self._check_record_keeping()
        self._check_transparency()
        self._check_human_oversight()
        self._check_accuracy_robustness()
        self._check_privacy_data_protection()
        self._check_quality_management()
        self._check_post_market_monitoring()
        
        logger.info(f"Completed compliance validation with {len(self.results)} checks")
        return self.results
    
    def _check_risk_management(self) -> None:
        """Check risk management compliance (Article 9)."""
        category = "risk_management"
        
        # Check if risk management system documentation exists
        risk_doc_path = Path("docs/risk-management-system.md")
        if risk_doc_path.exists():
            self.results.append(ComplianceResult(
                check_id="RM001",
                title="Risk Management System Documentation",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.HIGH,
                message="Risk management system documentation found",
                evidence=str(risk_doc_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="RM001",
                title="Risk Management System Documentation",
                status=ComplianceStatus.FAIL,
                level=ComplianceLevel.CRITICAL,
                message="Risk management system documentation missing",
                remediation="Create comprehensive risk management documentation"
            ))
        
        # Check monitoring implementation
        monitoring_path = Path("src/rlhf_audit_trail/monitoring")
        if monitoring_path.exists():
            self.results.append(ComplianceResult(
                check_id="RM002",
                title="Continuous Risk Monitoring",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.HIGH,
                message="Risk monitoring implementation found",
                evidence=str(monitoring_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="RM002",
                title="Continuous Risk Monitoring",
                status=ComplianceStatus.FAIL,
                level=ComplianceLevel.HIGH,
                message="Risk monitoring implementation missing",
                remediation="Implement continuous risk monitoring system"
            ))
    
    def _check_data_governance(self) -> None:
        """Check data governance compliance (Article 10)."""
        # Check data validation implementation
        data_validation_path = Path("src/rlhf_audit_trail/data_validation")
        if data_validation_path.exists():
            self.results.append(ComplianceResult(
                check_id="DG001",
                title="Data Quality and Validation",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.HIGH,
                message="Data validation implementation found",
                evidence=str(data_validation_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="DG001",
                title="Data Quality and Validation",
                status=ComplianceStatus.WARNING,
                level=ComplianceLevel.MEDIUM,
                message="Data validation implementation not found",
                remediation="Implement comprehensive data quality validation"
            ))
        
        # Check bias detection
        bias_detection_path = Path("src/rlhf_audit_trail/bias_detection")
        if bias_detection_path.exists():
            self.results.append(ComplianceResult(
                check_id="DG003",
                title="Bias Detection and Mitigation",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.HIGH,
                message="Bias detection implementation found",
                evidence=str(bias_detection_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="DG003",
                title="Bias Detection and Mitigation",
                status=ComplianceStatus.WARNING,
                level=ComplianceLevel.MEDIUM,
                message="Bias detection implementation not found",
                remediation="Implement bias detection and mitigation measures"
            ))
    
    def _check_technical_documentation(self) -> None:
        """Check technical documentation compliance (Article 11)."""
        # Check architecture documentation
        arch_doc_path = Path("docs/architecture.md")
        if arch_doc_path.exists():
            self.results.append(ComplianceResult(
                check_id="TD002",
                title="System Architecture Documentation",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.MEDIUM,
                message="Architecture documentation found",
                evidence=str(arch_doc_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="TD002",
                title="System Architecture Documentation",
                status=ComplianceStatus.FAIL,
                level=ComplianceLevel.HIGH,
                message="Architecture documentation missing",
                remediation="Create comprehensive architecture documentation"
            ))
        
        # Check README completeness
        readme_path = Path("README.md")
        if readme_path.exists():
            content = readme_path.read_text()
            if all(section in content.lower() for section in 
                   ["installation", "usage", "requirements", "license"]):
                self.results.append(ComplianceResult(
                    check_id="TD001",
                    title="Technical Documentation Completeness",
                    status=ComplianceStatus.PASS,
                    level=ComplianceLevel.MEDIUM,
                    message="README.md contains required sections",
                    evidence=str(readme_path)
                ))
            else:
                self.results.append(ComplianceResult(
                    check_id="TD001",
                    title="Technical Documentation Completeness",
                    status=ComplianceStatus.WARNING,
                    level=ComplianceLevel.MEDIUM,
                    message="README.md missing some required sections",
                    remediation="Ensure README includes installation, usage, requirements, and license"
                ))
    
    def _check_record_keeping(self) -> None:
        """Check record-keeping compliance (Article 12)."""
        # Check logging implementation
        logging_path = Path("src/rlhf_audit_trail/logging")
        if logging_path.exists():
            self.results.append(ComplianceResult(
                check_id="RK001",
                title="Automatic Logging Implementation",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.CRITICAL,
                message="Logging implementation found",
                evidence=str(logging_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="RK001",
                title="Automatic Logging Implementation",
                status=ComplianceStatus.FAIL,
                level=ComplianceLevel.CRITICAL,
                message="Logging implementation missing",
                remediation="Implement comprehensive audit logging system"
            ))
        
        # Check cryptographic implementation
        crypto_path = Path("src/rlhf_audit_trail/crypto")
        if crypto_path.exists():
            self.results.append(ComplianceResult(
                check_id="RK003",
                title="Immutable Audit Trail",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.CRITICAL,
                message="Cryptographic implementation found",
                evidence=str(crypto_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="RK003",
                title="Immutable Audit Trail",
                status=ComplianceStatus.WARNING,
                level=ComplianceLevel.HIGH,
                message="Cryptographic implementation not found",
                remediation="Implement cryptographically secure audit trails"
            ))
    
    def _check_transparency(self) -> None:
        """Check transparency compliance (Article 13)."""
        # Check user information documentation
        user_info_paths = [
            Path("docs/user-information.md"),
            Path("docs/intended-use.md"),
            Path("README.md")
        ]
        
        found_docs = [p for p in user_info_paths if p.exists()]
        if found_docs:
            self.results.append(ComplianceResult(
                check_id="TR001",
                title="User Information and Transparency",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.MEDIUM,
                message=f"User information documentation found: {', '.join(str(p) for p in found_docs)}",
                evidence=str(found_docs[0])
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="TR001",
                title="User Information and Transparency",
                status=ComplianceStatus.FAIL,
                level=ComplianceLevel.HIGH,
                message="User information documentation missing",
                remediation="Create clear user information and system purpose documentation"
            ))
    
    def _check_human_oversight(self) -> None:
        """Check human oversight compliance (Article 14)."""
        oversight_path = Path("src/rlhf_audit_trail/human_oversight")
        if oversight_path.exists():
            self.results.append(ComplianceResult(
                check_id="HO001",
                title="Human Oversight Implementation",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.HIGH,
                message="Human oversight implementation found",
                evidence=str(oversight_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="HO001",
                title="Human Oversight Implementation",
                status=ComplianceStatus.WARNING,
                level=ComplianceLevel.MEDIUM,
                message="Human oversight implementation not found",
                remediation="Implement human oversight mechanisms"
            ))
    
    def _check_accuracy_robustness(self) -> None:
        """Check accuracy and robustness compliance (Article 15)."""
        # Check quality assurance
        qa_path = Path("src/rlhf_audit_trail/quality_assurance")
        if qa_path.exists():
            self.results.append(ComplianceResult(
                check_id="AR001",
                title="Quality Assurance Measures",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.HIGH,
                message="Quality assurance implementation found",
                evidence=str(qa_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="AR001",
                title="Quality Assurance Measures",
                status=ComplianceStatus.WARNING,
                level=ComplianceLevel.MEDIUM,
                message="Quality assurance implementation not found",
                remediation="Implement quality assurance and accuracy measures"
            ))
        
        # Check robustness tests
        robustness_test_path = Path("tests/robustness")
        if robustness_test_path.exists():
            self.results.append(ComplianceResult(
                check_id="AR002",
                title="Robustness Testing",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.HIGH,
                message="Robustness tests found",
                evidence=str(robustness_test_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="AR002",
                title="Robustness Testing",
                status=ComplianceStatus.WARNING,
                level=ComplianceLevel.MEDIUM,
                message="Robustness tests not found",
                remediation="Implement comprehensive robustness testing"
            ))
        
        # Check security measures
        security_path = Path("security")
        if security_path.exists():
            self.results.append(ComplianceResult(
                check_id="AR003",
                title="Cybersecurity Measures",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.HIGH,
                message="Security measures documentation found",
                evidence=str(security_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="AR003",
                title="Cybersecurity Measures",
                status=ComplianceStatus.FAIL,
                level=ComplianceLevel.CRITICAL,
                message="Security measures documentation missing",
                remediation="Implement and document cybersecurity measures"
            ))
    
    def _check_privacy_data_protection(self) -> None:
        """Check privacy and data protection compliance."""
        # Check privacy implementation
        privacy_path = Path("src/rlhf_audit_trail/privacy")
        if privacy_path.exists():
            self.results.append(ComplianceResult(
                check_id="PDP001",
                title="Privacy Protection Implementation",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.CRITICAL,
                message="Privacy implementation found",
                evidence=str(privacy_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="PDP001",
                title="Privacy Protection Implementation",
                status=ComplianceStatus.WARNING,
                level=ComplianceLevel.HIGH,
                message="Privacy implementation not found",
                remediation="Implement comprehensive privacy protection measures"
            ))
        
        # Check differential privacy
        dp_path = Path("src/rlhf_audit_trail/differential_privacy")
        if dp_path.exists():
            self.results.append(ComplianceResult(
                check_id="PDP003",
                title="Differential Privacy Implementation",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.HIGH,
                message="Differential privacy implementation found",
                evidence=str(dp_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="PDP003",
                title="Differential Privacy Implementation",
                status=ComplianceStatus.WARNING,
                level=ComplianceLevel.MEDIUM,
                message="Differential privacy implementation not found",
                remediation="Implement differential privacy for sensitive data"
            ))
    
    def _check_quality_management(self) -> None:
        """Check quality management system compliance."""
        qms_doc_path = Path("docs/quality-management-system.md")
        if qms_doc_path.exists():
            self.results.append(ComplianceResult(
                check_id="QM001",
                title="Quality Management System",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.MEDIUM,
                message="Quality management system documentation found",
                evidence=str(qms_doc_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="QM001",
                title="Quality Management System",
                status=ComplianceStatus.FAIL,
                level=ComplianceLevel.HIGH,
                message="Quality management system documentation missing",
                remediation="Create quality management system documentation"
            ))
    
    def _check_post_market_monitoring(self) -> None:
        """Check post-market monitoring compliance."""
        monitoring_path = Path("src/rlhf_audit_trail/post_market_monitoring")
        if monitoring_path.exists():
            self.results.append(ComplianceResult(
                check_id="PMM001",
                title="Post-Market Monitoring System",
                status=ComplianceStatus.PASS,
                level=ComplianceLevel.HIGH,
                message="Post-market monitoring implementation found",
                evidence=str(monitoring_path)
            ))
        else:
            self.results.append(ComplianceResult(
                check_id="PMM001",
                title="Post-Market Monitoring System",
                status=ComplianceStatus.WARNING,
                level=ComplianceLevel.MEDIUM,
                message="Post-market monitoring implementation not found",
                remediation="Implement post-market monitoring system"
            ))
    
    def generate_report(self, format_type: str = "json") -> str:
        """Generate compliance report."""
        summary = self._generate_summary()
        
        report_data = {
            "compliance_report": {
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "validator_version": "1.0.0",
                    "framework": "EU AI Act 2024/1689",
                    "total_checks": len(self.results)
                },
                "summary": summary,
                "results": [
                    {
                        "check_id": result.check_id,
                        "title": result.title,
                        "status": result.status.value,
                        "level": result.level.value,
                        "message": result.message,
                        "evidence": result.evidence,
                        "remediation": result.remediation,
                        "timestamp": result.timestamp.isoformat() if result.timestamp else None
                    }
                    for result in self.results
                ]
            }
        }
        
        if format_type.lower() == "json":
            return json.dumps(report_data, indent=2)
        elif format_type.lower() == "yaml":
            return yaml.dump(report_data, default_flow_style=False)
        else:
            return self._generate_text_report(report_data)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate compliance summary statistics."""
        total = len(self.results)
        passed = len([r for r in self.results if r.status == ComplianceStatus.PASS])
        failed = len([r for r in self.results if r.status == ComplianceStatus.FAIL])
        warnings = len([r for r in self.results if r.status == ComplianceStatus.WARNING])
        
        critical_failures = len([r for r in self.results 
                               if r.status == ComplianceStatus.FAIL and r.level == ComplianceLevel.CRITICAL])
        
        compliance_score = (passed / total * 100) if total > 0 else 0
        
        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "critical_failures": critical_failures,
            "compliance_score": round(compliance_score, 2),
            "overall_status": "COMPLIANT" if critical_failures == 0 and failed == 0 else "NON_COMPLIANT"
        }
    
    def _generate_text_report(self, report_data: Dict[str, Any]) -> str:
        """Generate human-readable text report."""
        report = []
        report.append("=" * 60)
        report.append("EU AI ACT COMPLIANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        metadata = report_data["compliance_report"]["metadata"]
        report.append(f"Generated: {metadata['generated_at']}")
        report.append(f"Framework: {metadata['framework']}")
        report.append(f"Total Checks: {metadata['total_checks']}")
        report.append("")
        
        summary = report_data["compliance_report"]["summary"]
        report.append("SUMMARY")
        report.append("-" * 20)
        report.append(f"Overall Status: {summary['overall_status']}")
        report.append(f"Compliance Score: {summary['compliance_score']}%")
        report.append(f"Passed: {summary['passed']}")
        report.append(f"Failed: {summary['failed']}")
        report.append(f"Warnings: {summary['warnings']}")
        report.append(f"Critical Failures: {summary['critical_failures']}")
        report.append("")
        
        report.append("DETAILED RESULTS")
        report.append("-" * 20)
        
        for result in report_data["compliance_report"]["results"]:
            status_symbol = {
                "pass": "✅",
                "fail": "❌",
                "warning": "⚠️",
                "skip": "⏭️",
                "error": "💥"
            }.get(result["status"], "❓")
            
            report.append(f"{status_symbol} {result['check_id']}: {result['title']}")
            report.append(f"   Status: {result['status'].upper()}")
            report.append(f"   Level: {result['level'].upper()}")
            report.append(f"   Message: {result['message']}")
            
            if result["evidence"]:
                report.append(f"   Evidence: {result['evidence']}")
            
            if result["remediation"]:
                report.append(f"   Remediation: {result['remediation']}")
            
            report.append("")
        
        return "\n".join(report)


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RLHF Audit Trail Compliance Validator")
    parser.add_argument("--config", type=Path, help="Path to compliance configuration file")
    parser.add_argument("--format", choices=["json", "yaml", "text"], default="text",
                       help="Output format for the report")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Run compliance validation
    validator = ComplianceValidator(args.config)
    results = validator.run_all_checks()
    
    # Generate report
    report = validator.generate_report(args.format)
    
    # Output report
    if args.output:
        args.output.write_text(report)
        print(f"Compliance report written to {args.output}")
    else:
        print(report)
    
    # Exit with appropriate code
    summary = validator._generate_summary()
    exit_code = 0 if summary["overall_status"] == "COMPLIANT" else 1
    exit(exit_code)


if __name__ == "__main__":
    main()