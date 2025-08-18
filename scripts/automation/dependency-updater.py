#!/usr/bin/env python3
"""
Automated Dependency Updater for RLHF Audit Trail

This script manages dependency updates with security, compliance, and compatibility checks.
Integrates with security scanning and compliance validation to ensure safe updates.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import packaging.version as version
import requests


class DependencyUpdater:
    """Manages automated dependency updates with safety checks."""

    def __init__(self, repo_path: str = "."):
        """Initialize the dependency updater."""
        self.repo_path = Path(repo_path)
        self.requirements_files = [
            "requirements.txt",
            "requirements-dev.txt", 
            "requirements.in",
            "requirements-dev.in"
        ]
        self.pyproject_file = "pyproject.toml"
        
        # Security and compliance constraints
        self.forbidden_licenses = ['GPL-3.0', 'AGPL-3.0', 'SSPL-1.0']
        self.critical_packages = [
            'cryptography',
            'fastapi', 
            'sqlalchemy',
            'torch',
            'transformers',
            'opacus',
            'pydantic'
        ]
        
        self.update_policy = {
            'security': 'auto',      # Auto-update security fixes
            'patch': 'auto',         # Auto-update patch versions  
            'minor': 'manual',       # Manual review for minor versions
            'major': 'manual'        # Manual review for major versions
        }

    def run_command(self, command: List[str], **kwargs) -> Tuple[int, str, str]:
        """Run shell command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                **kwargs
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)

    def get_current_dependencies(self) -> Dict[str, str]:
        """Get currently installed dependencies and their versions."""
        code, stdout, stderr = self.run_command(["pip", "list", "--format=json"])
        
        if code != 0:
            print(f"Error getting dependencies: {stderr}")
            return {}
        
        dependencies = {}
        for package in json.loads(stdout):
            dependencies[package["name"]] = package["version"]
        
        return dependencies

    def get_outdated_packages(self) -> Dict[str, Dict]:
        """Get list of outdated packages with version information."""
        code, stdout, stderr = self.run_command(["pip", "list", "--outdated", "--format=json"])
        
        if code != 0:
            print(f"Error getting outdated packages: {stderr}")
            return {}
        
        outdated = {}
        for package in json.loads(stdout):
            outdated[package["name"]] = {
                "current": package["version"],
                "latest": package["latest_version"],
                "type": package.get("latest_filetype", "wheel")
            }
        
        return outdated

    def check_security_advisories(self, package_name: str, current_version: str) -> List[Dict]:
        """Check for security advisories for a specific package."""
        advisories = []
        
        try:
            # Use PyPA Advisory Database via PyPI API
            url = f"https://pypi.org/pypi/{package_name}/json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for security-related information in description
                description = data.get("info", {}).get("description", "").lower()
                if any(keyword in description for keyword in ["security", "vulnerability", "cve"]):
                    advisories.append({
                        "type": "security_mention",
                        "description": "Package mentions security fixes in description"
                    })
        
        except Exception as e:
            print(f"Warning: Could not check advisories for {package_name}: {e}")
        
        # Run safety check for specific package
        try:
            code, stdout, stderr = self.run_command([
                "safety", "check", "--json", f"--packages={package_name}=={current_version}"
            ])
            
            if code != 0 and stdout:
                safety_data = json.loads(stdout)
                for vuln in safety_data:
                    advisories.append({
                        "type": "safety_vulnerability",
                        "id": vuln.get("id"),
                        "cve": vuln.get("cve"),
                        "description": vuln.get("advisory"),
                        "severity": vuln.get("severity", "unknown")
                    })
        
        except Exception as e:
            print(f"Warning: Safety check failed for {package_name}: {e}")
        
        return advisories

    def check_license_compatibility(self, package_name: str) -> Optional[str]:
        """Check if package license is compatible with project requirements."""
        try:
            # Get package license information
            code, stdout, stderr = self.run_command([
                "pip-licenses", "--format=json", "--packages", package_name
            ])
            
            if code == 0 and stdout:
                licenses = json.loads(stdout)
                for pkg in licenses:
                    if pkg["Name"].lower() == package_name.lower():
                        license_name = pkg.get("License", "Unknown")
                        if license_name in self.forbidden_licenses:
                            return license_name
                        return None
        
        except Exception as e:
            print(f"Warning: Could not check license for {package_name}: {e}")
        
        return None

    def categorize_update_type(self, current: str, latest: str) -> str:
        """Categorize the type of update (patch, minor, major, security)."""
        try:
            current_ver = version.parse(current)
            latest_ver = version.parse(latest)
            
            # Compare version components
            if (current_ver.major != latest_ver.major):
                return "major"
            elif (current_ver.minor != latest_ver.minor):
                return "minor"
            elif (current_ver.micro != latest_ver.micro):
                return "patch"
            else:
                return "build"  # Pre-release or build metadata change
        
        except Exception:
            # If version parsing fails, treat as major change
            return "major"

    def analyze_package_update(self, package_name: str, current_version: str, latest_version: str) -> Dict:
        """Analyze a potential package update for safety and compatibility."""
        print(f"  Analyzing {package_name}: {current_version} -> {latest_version}")
        
        analysis = {
            "package": package_name,
            "current_version": current_version,
            "latest_version": latest_version,
            "update_type": self.categorize_update_type(current_version, latest_version),
            "is_critical": package_name.lower() in [p.lower() for p in self.critical_packages],
            "security_advisories": [],
            "license_issues": None,
            "recommendation": "unknown",
            "risk_level": "unknown"
        }
        
        # Check for security advisories
        advisories = self.check_security_advisories(package_name, current_version)
        analysis["security_advisories"] = advisories
        
        # Check license compatibility
        license_issue = self.check_license_compatibility(package_name)
        analysis["license_issues"] = license_issue
        
        # Determine recommendation based on policy and findings
        has_security_issues = len(advisories) > 0
        has_license_issues = license_issue is not None
        update_type = analysis["update_type"]
        is_critical = analysis["is_critical"]
        
        # Risk assessment
        if has_license_issues:
            analysis["risk_level"] = "high"
            analysis["recommendation"] = "reject"
        elif has_security_issues:
            analysis["risk_level"] = "high" 
            analysis["recommendation"] = "approve_immediately"
        elif update_type == "major":
            analysis["risk_level"] = "high" if is_critical else "medium"
            analysis["recommendation"] = "manual_review"
        elif update_type == "minor":
            analysis["risk_level"] = "medium" if is_critical else "low"
            analysis["recommendation"] = "approve_with_testing"
        elif update_type == "patch":
            analysis["risk_level"] = "low"
            analysis["recommendation"] = "approve"
        else:
            analysis["risk_level"] = "low"
            analysis["recommendation"] = "approve"
        
        return analysis

    def create_test_environment(self) -> str:
        """Create isolated test environment for dependency updates."""
        print("🧪 Creating test environment...")
        
        # Create temporary virtual environment
        temp_dir = tempfile.mkdtemp(prefix="dep_test_")
        venv_path = os.path.join(temp_dir, "venv")
        
        # Create virtual environment
        code, _, stderr = self.run_command([
            sys.executable, "-m", "venv", venv_path
        ])
        
        if code != 0:
            raise RuntimeError(f"Failed to create virtual environment: {stderr}")
        
        return temp_dir

    def test_dependency_update(self, package_name: str, new_version: str, test_env_path: str) -> Dict:
        """Test a dependency update in isolated environment."""
        print(f"  Testing {package_name}=={new_version} in isolated environment...")
        
        venv_path = os.path.join(test_env_path, "venv")
        pip_path = os.path.join(venv_path, "bin", "pip")
        python_path = os.path.join(venv_path, "bin", "python")
        
        test_results = {
            "package": package_name,
            "version": new_version,
            "install_success": False,
            "test_success": False,
            "import_success": False,
            "errors": []
        }
        
        try:
            # Install current requirements in test environment
            code, stdout, stderr = self.run_command([
                pip_path, "install", "-r", "requirements.txt"
            ])
            
            if code != 0:
                test_results["errors"].append(f"Failed to install base requirements: {stderr}")
                return test_results
            
            # Update the specific package
            code, stdout, stderr = self.run_command([
                pip_path, "install", f"{package_name}=={new_version}"
            ])
            
            if code != 0:
                test_results["errors"].append(f"Failed to install {package_name}: {stderr}")
                return test_results
            
            test_results["install_success"] = True
            
            # Test imports
            import_test_code = f"""
import sys
try:
    import {package_name.replace('-', '_')}
    print("SUCCESS: Import successful")
except Exception as e:
    print(f"ERROR: Import failed: {{e}}")
    sys.exit(1)
"""
            
            with open(os.path.join(test_env_path, "import_test.py"), "w") as f:
                f.write(import_test_code)
            
            code, stdout, stderr = self.run_command([
                python_path, os.path.join(test_env_path, "import_test.py")
            ])
            
            if code == 0 and "SUCCESS" in stdout:
                test_results["import_success"] = True
            else:
                test_results["errors"].append(f"Import test failed: {stderr}")
            
            # Run basic tests if they exist
            if (self.repo_path / "tests").exists():
                code, stdout, stderr = self.run_command([
                    python_path, "-m", "pytest", "-x", "--tb=short"
                ], cwd=self.repo_path, env={**os.environ, "VIRTUAL_ENV": venv_path})
                
                if code == 0:
                    test_results["test_success"] = True
                else:
                    test_results["errors"].append(f"Tests failed: {stderr}")
        
        except Exception as e:
            test_results["errors"].append(f"Test execution failed: {str(e)}")
        
        return test_results

    def update_requirements_file(self, file_path: str, package_updates: Dict[str, str]) -> bool:
        """Update requirements file with new package versions."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            modified_content = content
            
            for package, new_version in package_updates.items():
                # Find and replace version specifications
                patterns = [
                    rf'^{re.escape(package)}==.*$',          # exact version
                    rf'^{re.escape(package)}>=.*$',          # minimum version
                    rf'^{re.escape(package)}\[.*\]==.*$',    # with extras
                ]
                
                for pattern in patterns:
                    modified_content = re.sub(
                        pattern,
                        f"{package}=={new_version}",
                        modified_content,
                        flags=re.MULTILINE | re.IGNORECASE
                    )
            
            # Write updated content
            with open(file_path, 'w') as f:
                f.write(modified_content)
            
            return True
        
        except Exception as e:
            print(f"Error updating {file_path}: {e}")
            return False

    def generate_update_report(self, analyses: List[Dict], test_results: List[Dict]) -> Dict:
        """Generate comprehensive update report."""
        report = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_packages_analyzed": len(analyses),
            "recommendations": {
                "approve": [],
                "approve_with_testing": [],
                "approve_immediately": [],
                "manual_review": [],
                "reject": []
            },
            "risk_summary": {
                "low": 0,
                "medium": 0, 
                "high": 0
            },
            "security_updates": [],
            "license_issues": [],
            "test_failures": [],
            "update_summary": {
                "patch": 0,
                "minor": 0,
                "major": 0,
                "security": 0
            }
        }
        
        # Analyze recommendations and risks
        for analysis in analyses:
            recommendation = analysis["recommendation"]
            risk_level = analysis["risk_level"]
            update_type = analysis["update_type"]
            
            report["recommendations"][recommendation].append({
                "package": analysis["package"],
                "current": analysis["current_version"],
                "latest": analysis["latest_version"],
                "risk_level": risk_level,
                "update_type": update_type
            })
            
            report["risk_summary"][risk_level] += 1
            report["update_summary"][update_type] += 1
            
            # Collect security issues
            if analysis["security_advisories"]:
                report["security_updates"].append({
                    "package": analysis["package"],
                    "advisories": analysis["security_advisories"]
                })
            
            # Collect license issues
            if analysis["license_issues"]:
                report["license_issues"].append({
                    "package": analysis["package"],
                    "license": analysis["license_issues"]
                })
        
        # Analyze test results
        for test in test_results:
            if not test["test_success"] or test["errors"]:
                report["test_failures"].append(test)
        
        return report

    def run_dependency_update(self, auto_approve: bool = False, test_updates: bool = True) -> Dict:
        """Run comprehensive dependency update process."""
        print("🔄 Starting dependency update process...")
        
        # Get current state
        current_deps = self.get_current_dependencies()
        outdated_packages = self.get_outdated_packages()
        
        if not outdated_packages:
            print("✅ All dependencies are up to date!")
            return {"status": "up_to_date", "message": "No updates needed"}
        
        print(f"📦 Found {len(outdated_packages)} outdated packages")
        
        # Analyze each potential update
        print("🔍 Analyzing potential updates...")
        analyses = []
        
        for package_name, version_info in outdated_packages.items():
            analysis = self.analyze_package_update(
                package_name,
                version_info["current"],
                version_info["latest"]
            )
            analyses.append(analysis)
        
        # Filter packages for testing based on recommendations
        packages_to_test = []
        for analysis in analyses:
            if analysis["recommendation"] in ["approve", "approve_with_testing", "approve_immediately"]:
                packages_to_test.append(analysis)
        
        test_results = []
        if test_updates and packages_to_test:
            print(f"🧪 Testing {len(packages_to_test)} package updates...")
            
            # Create test environment
            test_env = self.create_test_environment()
            
            try:
                for analysis in packages_to_test:
                    test_result = self.test_dependency_update(
                        analysis["package"],
                        analysis["latest_version"],
                        test_env
                    )
                    test_results.append(test_result)
            finally:
                # Cleanup test environment
                import shutil
                shutil.rmtree(test_env, ignore_errors=True)
        
        # Generate comprehensive report
        report = self.generate_update_report(analyses, test_results)
        
        # Apply automatic updates if enabled
        if auto_approve:
            approved_updates = {}
            
            for analysis in analyses:
                if analysis["recommendation"] == "approve":
                    # Check if tests passed for this package
                    test_passed = True
                    for test_result in test_results:
                        if (test_result["package"] == analysis["package"] and 
                            not test_result["test_success"]):
                            test_passed = False
                            break
                    
                    if test_passed:
                        approved_updates[analysis["package"]] = analysis["latest_version"]
            
            if approved_updates:
                print(f"✅ Auto-approving {len(approved_updates)} safe updates...")
                
                # Update requirements files
                for req_file in self.requirements_files:
                    file_path = self.repo_path / req_file
                    if file_path.exists():
                        self.update_requirements_file(str(file_path), approved_updates)
                
                report["auto_applied_updates"] = approved_updates
        
        return report

    def create_pull_request_content(self, report: Dict) -> Tuple[str, str]:
        """Create pull request title and body for dependency updates."""
        auto_applied = report.get("auto_applied_updates", {})
        security_updates = len(report.get("security_updates", []))
        
        # Title
        if security_updates > 0:
            title = f"🔒 Security dependency updates ({security_updates} packages)"
        elif auto_applied:
            title = f"📦 Automated dependency updates ({len(auto_applied)} packages)"
        else:
            title = "📦 Dependency update analysis"
        
        # Body
        body = f"""## Dependency Update Report

This PR contains dependency updates analyzed and tested automatically.

### Summary
- **Packages analyzed**: {report['total_packages_analyzed']}
- **Security updates**: {len(report['security_updates'])}
- **License issues**: {len(report['license_issues'])}
- **Test failures**: {len(report['test_failures'])}

### Risk Assessment
- 🟢 Low risk: {report['risk_summary']['low']}
- 🟡 Medium risk: {report['risk_summary']['medium']}
- 🔴 High risk: {report['risk_summary']['high']}

### Update Types
- Patch: {report['update_summary']['patch']}
- Minor: {report['update_summary']['minor']}  
- Major: {report['update_summary']['major']}

### Recommendations

"""
        
        # Add recommendations sections
        for category, packages in report['recommendations'].items():
            if packages:
                emoji = {
                    "approve": "✅",
                    "approve_with_testing": "🧪",
                    "approve_immediately": "🚨",
                    "manual_review": "👀",
                    "reject": "❌"
                }
                
                body += f"#### {emoji.get(category, '📋')} {category.replace('_', ' ').title()}\n"
                for pkg in packages:
                    body += f"- **{pkg['package']}**: {pkg['current']} → {pkg['latest']} ({pkg['update_type']}, {pkg['risk_level']} risk)\n"
                body += "\n"
        
        # Add security section if relevant
        if report['security_updates']:
            body += "### 🔒 Security Updates\n"
            for sec_update in report['security_updates']:
                body += f"- **{sec_update['package']}**: {len(sec_update['advisories'])} security advisories\n"
            body += "\n"
        
        # Add test results section
        if report['test_failures']:
            body += "### ❌ Test Failures\n"
            for failure in report['test_failures']:
                body += f"- **{failure['package']}**: {', '.join(failure['errors'])}\n"
            body += "\n"
        
        if auto_applied:
            body += "### ✅ Auto-Applied Updates\n"
            for pkg, version in auto_applied.items():
                body += f"- {pkg} → {version}\n"
            body += "\n"
        
        body += """### Next Steps
1. Review the recommendations above
2. Test critical package updates manually if needed
3. Address any security advisories immediately
4. Merge after approval

Generated by automated dependency updater 🤖"""
        
        return title, body


def main():
    """Main function to run dependency updater."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Dependency Updater")
    parser.add_argument("--auto-approve", action="store_true",
                       help="Automatically apply safe updates")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip testing updates in isolated environment")
    parser.add_argument("--output", "-o", help="Output report file")
    parser.add_argument("--format", choices=["json", "markdown"], default="json",
                       help="Output format")
    parser.add_argument("--create-pr", action="store_true",
                       help="Output PR title and body")
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = DependencyUpdater()
    
    # Run update process
    report = updater.run_dependency_update(
        auto_approve=args.auto_approve,
        test_updates=not args.skip_tests
    )
    
    # Output results
    if args.create_pr:
        title, body = updater.create_pull_request_content(report)
        print(f"PR_TITLE={title}")
        print(f"PR_BODY<<EOF")
        print(body)
        print("EOF")
    else:
        if args.format == "json":
            output = json.dumps(report, indent=2)
        else:
            # Format as markdown (simplified)
            output = f"# Dependency Update Report\n\n{json.dumps(report, indent=2)}"
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"📄 Report saved to {args.output}")
        else:
            print(output)
    
    # Exit with appropriate code
    security_updates = len(report.get("security_updates", []))
    high_risk_updates = report.get("risk_summary", {}).get("high", 0)
    
    if security_updates > 0:
        print(f"\n🚨 {security_updates} security updates available!")
        sys.exit(2)  # Security updates needed
    elif high_risk_updates > 0:
        print(f"\n⚠️  {high_risk_updates} high-risk updates require manual review")
        sys.exit(1)  # Manual review needed
    else:
        print("\n✅ Dependency update analysis completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()