#!/usr/bin/env python3
"""
Supply Chain Security Automation for RLHF Audit Trail
Advanced security scanning and SBOM generation for high-maturity repositories
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SupplyChainSecurityScanner:
    """Advanced supply chain security scanner for Python AI/ML projects."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "reports" / "security"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_sbom(self, formats: List[str] = None) -> Dict[str, str]:
        """Generate Software Bill of Materials in multiple formats."""
        if formats is None:
            formats = ["spdx-json", "cyclonedx-json"]
            
        sbom_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("Generating SBOM...")
        
        for fmt in formats:
            try:
                if fmt == "spdx-json":
                    output_file = self.reports_dir / f"sbom_spdx_{timestamp}.json"
                    cmd = [
                        "syft", "packages", str(self.project_root),
                        "-o", f"spdx-json={output_file}"
                    ]
                elif fmt == "cyclonedx-json":
                    output_file = self.reports_dir / f"sbom_cyclonedx_{timestamp}.json"
                    cmd = [
                        "cyclonedx-py", "--format", "json", 
                        "--output", str(output_file),
                        str(self.project_root)
                    ]
                else:
                    logger.warning(f"Unsupported SBOM format: {fmt}")
                    continue
                    
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    sbom_files[fmt] = str(output_file)
                    logger.info(f"Generated {fmt} SBOM: {output_file}")
                else:
                    logger.error(f"Failed to generate {fmt} SBOM: {result.stderr}")
                    
            except FileNotFoundError:
                logger.warning(f"Tool not found for {fmt} SBOM generation")
                
        return sbom_files
    
    def scan_vulnerabilities(self) -> Dict[str, str]:
        """Run comprehensive vulnerability scanning."""
        scan_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # OSV Scanner
        try:
            osv_output = self.reports_dir / f"osv_scan_{timestamp}.json"
            cmd = [
                "osv-scanner", "--format=json", 
                f"--output={osv_output}",
                str(self.project_root)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                scan_results["osv"] = str(osv_output)
                logger.info(f"OSV scan completed: {osv_output}")
            else:
                logger.warning(f"OSV scan failed: {result.stderr}")
        except FileNotFoundError:
            logger.warning("OSV scanner not found")
            
        # Trivy Scanner
        try:
            trivy_output = self.reports_dir / f"trivy_scan_{timestamp}.json"
            cmd = [
                "trivy", "fs", "--format", "json",
                "-o", str(trivy_output),
                str(self.project_root)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                scan_results["trivy"] = str(trivy_output)
                logger.info(f"Trivy scan completed: {trivy_output}")
            else:
                logger.warning(f"Trivy scan failed: {result.stderr}")
        except FileNotFoundError:
            logger.warning("Trivy scanner not found")
            
        # Grype Scanner
        try:
            grype_output = self.reports_dir / f"grype_scan_{timestamp}.json"
            cmd = [
                "grype", str(self.project_root),
                "-o", "json",
                "--file", str(grype_output)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                scan_results["grype"] = str(grype_output)
                logger.info(f"Grype scan completed: {grype_output}")
            else:
                logger.warning(f"Grype scan failed: {result.stderr}")
        except FileNotFoundError:
            logger.warning("Grype scanner not found")
            
        return scan_results
    
    def check_license_compliance(self) -> Dict[str, any]:
        """Check license compliance for all dependencies."""
        logger.info("Checking license compliance...")
        
        try:
            # Generate pip-licenses report
            result = subprocess.run([
                "pip-licenses", "--format=json", "--with-urls"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"License scan failed: {result.stderr}")
                return {}
                
            licenses_data = json.loads(result.stdout)
            
            # Load allowed/forbidden licenses from config
            config_file = self.project_root / "sbom.yaml"
            if config_file.exists():
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                    
                approved = config.get("automation", {}).get("license_scanning", {}).get("approved_licenses", [])
                forbidden = config.get("automation", {}).get("license_scanning", {}).get("forbidden_licenses", [])
            else:
                # Default license policy
                approved = ["Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause"]
                forbidden = ["GPL-3.0", "AGPL-3.0", "SSPL-1.0"]
            
            compliance_report = {
                "compliant": [],
                "non_compliant": [],
                "unknown": [],
                "summary": {
                    "total_packages": len(licenses_data),
                    "compliant_count": 0,
                    "non_compliant_count": 0,
                    "unknown_count": 0
                }
            }
            
            for package in licenses_data:
                package_name = package.get("Name", "Unknown")
                license_name = package.get("License", "Unknown")
                
                if license_name in approved:
                    compliance_report["compliant"].append({
                        "package": package_name,
                        "license": license_name
                    })
                    compliance_report["summary"]["compliant_count"] += 1
                elif license_name in forbidden:
                    compliance_report["non_compliant"].append({
                        "package": package_name,
                        "license": license_name,
                        "reason": "Forbidden license"
                    })
                    compliance_report["summary"]["non_compliant_count"] += 1
                else:
                    compliance_report["unknown"].append({
                        "package": package_name,
                        "license": license_name,
                        "reason": "License not in approved list"
                    })
                    compliance_report["summary"]["unknown_count"] += 1
            
            # Save compliance report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.reports_dir / f"license_compliance_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(compliance_report, f, indent=2)
                
            logger.info(f"License compliance report saved: {report_file}")
            return compliance_report
            
        except FileNotFoundError:
            logger.warning("pip-licenses not found")
            return {}
        except Exception as e:
            logger.error(f"License compliance check failed: {e}")
            return {}
    
    def generate_provenance(self) -> Optional[str]:
        """Generate SLSA provenance information."""
        logger.info("Generating provenance information...")
        
        provenance = {
            "buildDefinition": {
                "buildType": "https://github.com/terragonlabs/rlhf-audit-trail",
                "externalParameters": {
                    "repository": "https://github.com/terragonlabs/rlhf-audit-trail",
                    "ref": "refs/heads/main"
                },
                "internalParameters": {
                    "buildTimestamp": datetime.now().isoformat(),
                    "buildEnvironment": "GitHub Actions"
                }
            },
            "runDetails": {
                "builder": {
                    "id": "https://github.com/actions/runner"
                },
                "metadata": {
                    "invocationId": f"build-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "startedOn": datetime.now().isoformat()
                }
            }
        }
        
        # Save provenance
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        provenance_file = self.reports_dir / f"provenance_{timestamp}.json"
        with open(provenance_file, 'w') as f:
            json.dump(provenance, f, indent=2)
            
        logger.info(f"Provenance generated: {provenance_file}")
        return str(provenance_file)
    
    def run_full_scan(self) -> Dict[str, any]:
        """Run complete supply chain security scan."""
        logger.info("Starting full supply chain security scan...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "sbom_files": self.generate_sbom(),
            "vulnerability_scans": self.scan_vulnerabilities(),
            "license_compliance": self.check_license_compliance(),
            "provenance_file": self.generate_provenance()
        }
        
        # Save comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"supply_chain_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Full supply chain security scan completed: {report_file}")
        return results


def main():
    """Main entry point for supply chain security scanning."""
    project_root = Path(__file__).parent.parent
    scanner = SupplyChainSecurityScanner(project_root)
    
    try:
        results = scanner.run_full_scan()
        
        # Print summary
        print("\n" + "="*60)
        print("SUPPLY CHAIN SECURITY SCAN SUMMARY")
        print("="*60)
        print(f"Scan completed at: {results['timestamp']}")
        print(f"SBOM files generated: {len(results['sbom_files'])}")
        print(f"Vulnerability scans: {len(results['vulnerability_scans'])}")
        
        license_summary = results.get('license_compliance', {}).get('summary', {})
        if license_summary:
            print(f"License compliance:")
            print(f"  - Compliant: {license_summary.get('compliant_count', 0)}")
            print(f"  - Non-compliant: {license_summary.get('non_compliant_count', 0)}")
            print(f"  - Unknown: {license_summary.get('unknown_count', 0)}")
        
        print(f"Reports saved in: {scanner.reports_dir}")
        print("="*60)
        
        # Exit with error code if vulnerabilities or license issues found
        vuln_count = sum(1 for scan in results['vulnerability_scans'].values())
        license_issues = license_summary.get('non_compliant_count', 0)
        
        if vuln_count > 0 or license_issues > 0:
            logger.warning("Security issues detected - review reports")
            sys.exit(1)
        else:
            logger.info("No critical security issues detected")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Supply chain security scan failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()