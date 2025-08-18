#!/usr/bin/env python3
"""
Repository Health Monitoring Script

Automated repository health checks and maintenance tasks for RLHF Audit Trail.
Monitors code quality, security, compliance, and operational metrics.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests
import yaml


class RepositoryHealthMonitor:
    """Monitors and maintains repository health metrics."""

    def __init__(self, repo_path: str = ".", config_path: str = ".github/project-metrics.json"):
        """Initialize the health monitor."""
        self.repo_path = Path(repo_path)
        self.config_path = Path(config_path)
        self.metrics = {}
        self.alerts = []
        self.load_metrics()

    def load_metrics(self) -> None:
        """Load current metrics from configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                self.metrics = json.load(f)
        except FileNotFoundError:
            print(f"Metrics file not found: {self.config_path}")
            self.metrics = {"project": "rlhf-audit-trail", "metrics": {}}

    def save_metrics(self) -> None:
        """Save updated metrics to configuration file."""
        self.metrics["last_updated"] = datetime.utcnow().isoformat() + "Z"
        with open(self.config_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def run_git_command(self, command: List[str]) -> str:
        """Run git command and return output."""
        try:
            result = subprocess.run(
                ["git"] + command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {e}")
            return ""

    def check_repository_stats(self) -> Dict:
        """Check basic repository statistics."""
        print("📊 Checking repository statistics...")
        
        stats = {
            "commits": 0,
            "branches": 0,
            "contributors": 0,
            "files": 0,
            "lines_of_code": 0
        }

        # Count commits
        commits = self.run_git_command(["rev-list", "--all", "--count"])
        if commits:
            stats["commits"] = int(commits)

        # Count branches
        branches = self.run_git_command(["branch", "-a"])
        if branches:
            stats["branches"] = len([b for b in branches.split('\n') if b.strip()])

        # Count contributors
        contributors = self.run_git_command(["log", "--format=%ae"])
        if contributors:
            stats["contributors"] = len(set(contributors.split('\n')))

        # Count Python files and lines
        try:
            result = subprocess.run(
                ["find", ".", "-name", "*.py", "-type", "f"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            py_files = [f for f in result.stdout.strip().split('\n') if f]
            stats["files"] = len(py_files)

            # Count lines of code
            if py_files:
                wc_result = subprocess.run(
                    ["wc", "-l"] + py_files,
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                if wc_result.stdout:
                    last_line = wc_result.stdout.strip().split('\n')[-1]
                    stats["lines_of_code"] = int(last_line.split()[0])

        except Exception as e:
            print(f"Error counting files: {e}")

        return stats

    def check_test_coverage(self) -> Dict:
        """Check test coverage metrics."""
        print("🧪 Checking test coverage...")
        
        coverage_data = {
            "coverage_percentage": 0.0,
            "total_tests": 0,
            "passing_tests": 0,
            "failing_tests": 0
        }

        try:
            # Run pytest with coverage
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=json", "-q"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            # Parse coverage report
            coverage_file = self.repo_path / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage = json.load(f)
                coverage_data["coverage_percentage"] = coverage.get("totals", {}).get("percent_covered", 0.0)

            # Count tests
            if "collected" in result.stdout:
                for line in result.stdout.split('\n'):
                    if "collected" in line:
                        coverage_data["total_tests"] = int(line.split()[0])
                        break

            # Parse test results
            if "failed" in result.stdout:
                for line in result.stdout.split('\n'):
                    if "failed" in line and "passed" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "failed,":
                                coverage_data["failing_tests"] = int(parts[i-1])
                            if part == "passed":
                                coverage_data["passing_tests"] = int(parts[i-1])
                        break

        except Exception as e:
            print(f"Error checking test coverage: {e}")

        return coverage_data

    def check_security_vulnerabilities(self) -> Dict:
        """Check for security vulnerabilities."""
        print("🔒 Checking security vulnerabilities...")
        
        security_data = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "total": 0,
            "last_scan": datetime.utcnow().isoformat() + "Z"
        }

        try:
            # Run safety check
            result = subprocess.run(
                ["safety", "check", "--json"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            if result.stdout:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    severity = vuln.get("severity", "unknown").lower()
                    if severity in security_data:
                        security_data[severity] += 1
                        security_data["total"] += 1

            # Run bandit security scan
            bandit_result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            if bandit_result.stdout:
                bandit_data = json.loads(bandit_result.stdout)
                for issue in bandit_data.get("results", []):
                    severity = issue.get("issue_severity", "").lower()
                    if severity in security_data:
                        security_data[severity] += 1
                        security_data["total"] += 1

        except Exception as e:
            print(f"Error checking security: {e}")

        return security_data

    def check_compliance_status(self) -> Dict:
        """Check compliance framework status."""
        print("⚖️ Checking compliance status...")
        
        compliance_data = {
            "eu_ai_act": {"score": 0.0, "status": "unknown"},
            "gdpr": {"score": 0.0, "status": "unknown"},
            "nist_ai_rmf": {"score": 0.0, "status": "unknown"}
        }

        try:
            # Run compliance validator
            for framework in compliance_data.keys():
                result = subprocess.run(
                    ["python", "-m", f"compliance.{framework}.validate"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    # Parse compliance score from output
                    for line in result.stdout.split('\n'):
                        if "compliance score" in line.lower():
                            try:
                                score = float(line.split(':')[-1].strip().rstrip('%'))
                                compliance_data[framework]["score"] = score / 100
                                compliance_data[framework]["status"] = "passing" if score >= 90 else "needs_attention"
                            except ValueError:
                                pass
                else:
                    compliance_data[framework]["status"] = "failing"

        except Exception as e:
            print(f"Error checking compliance: {e}")

        return compliance_data

    def check_performance_metrics(self) -> Dict:
        """Check performance and resource utilization."""
        print("⚡ Checking performance metrics...")
        
        performance_data = {
            "build_time": "unknown",
            "test_time": "unknown",
            "memory_usage": "unknown",
            "cpu_usage": "unknown"
        }

        try:
            # Measure build time
            start_time = datetime.now()
            build_result = subprocess.run(
                ["python", "-m", "build"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if build_result.returncode == 0:
                build_duration = datetime.now() - start_time
                performance_data["build_time"] = f"{build_duration.total_seconds():.1f}s"

            # Measure test execution time
            start_time = datetime.now()
            test_result = subprocess.run(
                ["python", "-m", "pytest", "-x"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if test_result.returncode == 0:
                test_duration = datetime.now() - start_time
                performance_data["test_time"] = f"{test_duration.total_seconds():.1f}s"

        except Exception as e:
            print(f"Error checking performance: {e}")

        return performance_data

    def check_dependency_health(self) -> Dict:
        """Check dependency health and updates."""
        print("📦 Checking dependency health...")
        
        dep_data = {
            "total_dependencies": 0,
            "outdated_dependencies": 0,
            "vulnerable_dependencies": 0,
            "license_issues": 0
        }

        try:
            # Check for outdated dependencies
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True
            )

            if result.stdout:
                outdated = json.loads(result.stdout)
                dep_data["outdated_dependencies"] = len(outdated)

            # Check total dependencies
            total_result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True
            )

            if total_result.stdout:
                total = json.loads(total_result.stdout)
                dep_data["total_dependencies"] = len(total)

            # Check for license issues with pip-licenses
            license_result = subprocess.run(
                ["pip-licenses", "--format=json"],
                capture_output=True,
                text=True
            )

            if license_result.stdout:
                licenses = json.loads(license_result.stdout)
                forbidden_licenses = ['GPL-3.0', 'AGPL-3.0', 'SSPL-1.0']
                for pkg in licenses:
                    if pkg.get('License', '') in forbidden_licenses:
                        dep_data["license_issues"] += 1

        except Exception as e:
            print(f"Error checking dependencies: {e}")

        return dep_data

    def generate_health_alerts(self) -> List[Dict]:
        """Generate health alerts based on metrics."""
        alerts = []

        # Check test coverage
        coverage = self.metrics.get("metrics", {}).get("testing", {}).get("test_coverage", 0)
        if coverage < 85:
            alerts.append({
                "severity": "medium",
                "category": "quality",
                "message": f"Test coverage is {coverage}% (target: 85%+)",
                "action": "Add more unit and integration tests"
            })

        # Check security vulnerabilities
        security = self.metrics.get("metrics", {}).get("security", {}).get("vulnerability_count", {})
        critical = security.get("critical", 0)
        high = security.get("high", 0)

        if critical > 0:
            alerts.append({
                "severity": "critical",
                "category": "security",
                "message": f"{critical} critical security vulnerabilities detected",
                "action": "Immediately address critical vulnerabilities"
            })

        if high > 0:
            alerts.append({
                "severity": "high",
                "category": "security",
                "message": f"{high} high severity security vulnerabilities detected",
                "action": "Address high severity vulnerabilities within 7 days"
            })

        # Check compliance scores
        compliance = self.metrics.get("metrics", {}).get("compliance", {})
        for framework, data in compliance.items():
            score = data.get("compliance_score", 0)
            if score < 0.9:
                alerts.append({
                    "severity": "high",
                    "category": "compliance",
                    "message": f"{framework} compliance score is {score:.1%} (target: 90%+)",
                    "action": f"Review and address {framework} compliance gaps"
                })

        return alerts

    def run_health_check(self) -> Dict:
        """Run comprehensive repository health check."""
        print("🏥 Running comprehensive repository health check...")
        
        # Collect all metrics
        repo_stats = self.check_repository_stats()
        test_coverage = self.check_test_coverage()
        security_data = self.check_security_vulnerabilities()
        compliance_data = self.check_compliance_status()
        performance_data = self.check_performance_metrics()
        dependency_data = self.check_dependency_health()

        # Update metrics
        if "metrics" not in self.metrics:
            self.metrics["metrics"] = {}

        self.metrics["metrics"].update({
            "repository": repo_stats,
            "testing": test_coverage,
            "security": {
                "vulnerability_count": {
                    "critical": security_data["critical"],
                    "high": security_data["high"],
                    "medium": security_data["medium"],
                    "low": security_data["low"]
                },
                "last_scan": security_data["last_scan"]
            },
            "compliance": compliance_data,
            "performance": performance_data,
            "dependencies": dependency_data
        })

        # Generate alerts
        self.alerts = self.generate_health_alerts()
        self.metrics["alerts"] = {
            "active": self.alerts,
            "count": len(self.alerts),
            "last_check": datetime.utcnow().isoformat() + "Z"
        }

        # Calculate overall health score
        health_score = self.calculate_health_score()
        self.metrics["health_score"] = health_score

        return self.metrics

    def calculate_health_score(self) -> float:
        """Calculate overall repository health score (0-100)."""
        scores = []
        
        # Test coverage score (0-25 points)
        coverage = self.metrics.get("metrics", {}).get("testing", {}).get("coverage_percentage", 0)
        scores.append(min(25, coverage / 4))  # 100% coverage = 25 points

        # Security score (0-25 points)
        security = self.metrics.get("metrics", {}).get("security", {}).get("vulnerability_count", {})
        critical = security.get("critical", 0)
        high = security.get("high", 0)
        medium = security.get("medium", 0)
        
        security_score = 25
        security_score -= critical * 10  # -10 per critical
        security_score -= high * 5       # -5 per high  
        security_score -= medium * 2     # -2 per medium
        scores.append(max(0, security_score))

        # Compliance score (0-25 points)
        compliance = self.metrics.get("metrics", {}).get("compliance", {})
        compliance_scores = []
        for framework_data in compliance.values():
            if isinstance(framework_data, dict) and "score" in framework_data:
                compliance_scores.append(framework_data["score"] * 25)
        
        if compliance_scores:
            scores.append(sum(compliance_scores) / len(compliance_scores))
        else:
            scores.append(0)

        # Code quality score (0-25 points)
        # Based on various factors like dependency health, performance, etc.
        deps = self.metrics.get("metrics", {}).get("dependencies", {})
        total_deps = deps.get("total_dependencies", 1)
        outdated_ratio = deps.get("outdated_dependencies", 0) / total_deps
        license_issues = deps.get("license_issues", 0)
        
        quality_score = 25
        quality_score -= outdated_ratio * 10  # Penalty for outdated deps
        quality_score -= license_issues * 5   # Penalty for license issues
        scores.append(max(0, quality_score))

        return round(sum(scores), 1)

    def generate_report(self, output_format: str = "json") -> str:
        """Generate health report in specified format."""
        if output_format == "json":
            return json.dumps(self.metrics, indent=2)
        
        elif output_format == "markdown":
            report = f"""# Repository Health Report

Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Overall Health Score: {self.metrics.get('health_score', 0)}/100

"""
            
            # Add alerts section
            if self.alerts:
                report += "## 🚨 Active Alerts\n\n"
                for alert in self.alerts:
                    emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(alert["severity"], "ℹ️")
                    report += f"- {emoji} **{alert['category'].title()}**: {alert['message']}\n"
                    report += f"  *Action:* {alert['action']}\n\n"
            else:
                report += "## ✅ No Active Alerts\n\n"

            # Add metrics sections
            metrics = self.metrics.get("metrics", {})
            
            if "testing" in metrics:
                test = metrics["testing"]
                report += f"""## 🧪 Testing Metrics
- Coverage: {test.get('coverage_percentage', 0):.1f}%
- Total Tests: {test.get('total_tests', 0)}
- Passing: {test.get('passing_tests', 0)}
- Failing: {test.get('failing_tests', 0)}

"""

            if "security" in metrics:
                sec = metrics["security"]["vulnerability_count"]
                report += f"""## 🔒 Security Metrics
- Critical: {sec.get('critical', 0)}
- High: {sec.get('high', 0)}
- Medium: {sec.get('medium', 0)}
- Low: {sec.get('low', 0)}

"""

            if "compliance" in metrics:
                comp = metrics["compliance"]
                report += "## ⚖️ Compliance Metrics\n"
                for framework, data in comp.items():
                    score = data.get('score', 0) * 100 if isinstance(data, dict) else 0
                    report += f"- {framework.replace('_', ' ').title()}: {score:.1f}%\n"
                report += "\n"

            return report
        
        else:
            raise ValueError(f"Unsupported format: {output_format}")


def main():
    """Main function to run repository health check."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository Health Monitor")
    parser.add_argument("--format", choices=["json", "markdown"], default="json",
                       help="Output format for the report")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--save-metrics", action="store_true",
                       help="Save updated metrics to project-metrics.json")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = RepositoryHealthMonitor()
    
    # Run health check
    results = monitor.run_health_check()
    
    # Save metrics if requested
    if args.save_metrics:
        monitor.save_metrics()
        print(f"✅ Metrics saved to {monitor.config_path}")
    
    # Generate report
    report = monitor.generate_report(args.format)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to {args.output}")
    else:
        print(report)
    
    # Exit with non-zero code if there are critical alerts
    critical_alerts = [a for a in monitor.alerts if a["severity"] == "critical"]
    if critical_alerts:
        print(f"\n❌ {len(critical_alerts)} critical alerts detected!")
        sys.exit(1)
    else:
        print(f"\n✅ Repository health check completed. Score: {results.get('health_score', 0)}/100")
        sys.exit(0)


if __name__ == "__main__":
    main()