#!/usr/bin/env python3
"""
Comprehensive test runner for RLHF Audit Trail.

This script provides various testing options including unit tests, integration tests,
compliance validation, performance benchmarks, and security testing.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Colors for console output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def log_info(message: str) -> None:
    """Log info message."""
    print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} {message}")


def log_success(message: str) -> None:
    """Log success message."""
    print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {message}")


def log_warning(message: str) -> None:
    """Log warning message."""
    print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {message}")


def log_error(message: str) -> None:
    """Log error message."""
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")


def run_command(cmd: List[str], description: str, capture_output: bool = False) -> Optional[subprocess.CompletedProcess]:
    """Run a command with error handling."""
    log_info(f"Running: {description}")
    log_info(f"Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=True
        )
        duration = time.time() - start_time
        log_success(f"Completed {description} in {duration:.2f}s")
        return result
    except subprocess.CalledProcessError as e:
        log_error(f"Failed: {description}")
        log_error(f"Exit code: {e.returncode}")
        if e.stdout:
            log_error(f"Stdout: {e.stdout}")
        if e.stderr:
            log_error(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        log_error(f"Command not found: {cmd[0]}")
        return None


class TestSuite:
    """Test suite configuration and execution."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.test_dir = root_dir / "tests"
        self.coverage_dir = root_dir / "htmlcov"
        self.reports_dir = root_dir / "reports"
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(exist_ok=True)
    
    def run_unit_tests(self, verbose: bool = False, coverage: bool = True) -> bool:
        """Run unit tests."""
        cmd = ["pytest", "tests/unit/", "-m", "unit"]
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html",
                "--cov-report=xml:reports/coverage.xml"
            ])
        
        cmd.extend([
            "--junitxml=reports/unit-tests.xml",
            "--tb=short"
        ])
        
        result = run_command(cmd, "Unit tests")
        return result is not None and result.returncode == 0
    
    def run_integration_tests(self, verbose: bool = False) -> bool:
        """Run integration tests."""
        cmd = ["pytest", "tests/integration/", "-m", "integration"]
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        cmd.extend([
            "--junitxml=reports/integration-tests.xml",
            "--tb=short"
        ])
        
        result = run_command(cmd, "Integration tests")
        return result is not None and result.returncode == 0
    
    def run_e2e_tests(self, verbose: bool = False) -> bool:
        """Run end-to-end tests."""
        cmd = ["pytest", "tests/e2e/", "-m", "e2e"]
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        cmd.extend([
            "--junitxml=reports/e2e-tests.xml",
            "--tb=short"
        ])
        
        result = run_command(cmd, "End-to-end tests")
        return result is not None and result.returncode == 0
    
    def run_compliance_tests(self, framework: Optional[str] = None, verbose: bool = False) -> bool:
        """Run compliance tests."""
        cmd = ["pytest", "tests/", "-m", "compliance"]
        
        if framework:
            if framework == "eu_ai_act":
                cmd.extend(["-k", "eu_ai_act"])
            elif framework == "nist":
                cmd.extend(["-k", "nist"])
            elif framework == "gdpr":
                cmd.extend(["-k", "gdpr"])
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        cmd.extend([
            "--junitxml=reports/compliance-tests.xml",
            "--tb=short"
        ])
        
        result = run_command(cmd, f"Compliance tests ({framework or 'all frameworks'})")
        return result is not None and result.returncode == 0
    
    def run_security_tests(self, verbose: bool = False) -> bool:
        """Run security-focused tests."""
        cmd = ["pytest", "tests/", "-m", "security"]
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        cmd.extend([
            "--junitxml=reports/security-tests.xml",
            "--tb=short"
        ])
        
        result = run_command(cmd, "Security tests")
        return result is not None and result.returncode == 0
    
    def run_privacy_tests(self, verbose: bool = False) -> bool:
        """Run privacy-focused tests."""
        cmd = ["pytest", "tests/", "-m", "privacy"]
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        cmd.extend([
            "--junitxml=reports/privacy-tests.xml",
            "--tb=short"
        ])
        
        result = run_command(cmd, "Privacy tests")
        return result is not None and result.returncode == 0
    
    def run_performance_tests(self, verbose: bool = False) -> bool:
        """Run performance benchmark tests."""
        cmd = ["pytest", "tests/performance/", "-m", "performance", "--benchmark-only"]
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        cmd.extend([
            "--benchmark-json=reports/benchmark.json",
            "--junitxml=reports/performance-tests.xml"
        ])
        
        result = run_command(cmd, "Performance tests")
        return result is not None and result.returncode == 0
    
    def run_smoke_tests(self, verbose: bool = False) -> bool:
        """Run smoke tests for basic functionality."""
        cmd = ["pytest", "tests/", "-m", "smoke", "--maxfail=5"]
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        cmd.extend([
            "--junitxml=reports/smoke-tests.xml",
            "--tb=short"
        ])
        
        result = run_command(cmd, "Smoke tests")
        return result is not None and result.returncode == 0
    
    def run_fast_tests(self, verbose: bool = False) -> bool:
        """Run fast tests for quick feedback."""
        cmd = ["pytest", "tests/", "-m", "fast or (unit and not slow)", "--maxfail=10"]
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        cmd.extend([
            "--junitxml=reports/fast-tests.xml",
            "--tb=short"
        ])
        
        result = run_command(cmd, "Fast tests")
        return result is not None and result.returncode == 0
    
    def run_all_tests(self, verbose: bool = False, include_slow: bool = True) -> Dict[str, bool]:
        """Run complete test suite."""
        results = {}
        
        log_info("Starting comprehensive test suite")
        
        # Run different test categories
        test_categories = [
            ("unit", self.run_unit_tests),
            ("integration", self.run_integration_tests),
            ("compliance", self.run_compliance_tests),
            ("security", self.run_security_tests),
            ("privacy", self.run_privacy_tests),
        ]
        
        if include_slow:
            test_categories.extend([
                ("e2e", self.run_e2e_tests),
                ("performance", self.run_performance_tests),
            ])
        
        for category, test_func in test_categories:
            log_info(f"Running {category} tests...")
            results[category] = test_func(verbose=verbose)
            
            if not results[category]:
                log_error(f"{category} tests failed!")
            else:
                log_success(f"{category} tests passed!")
        
        return results
    
    def run_quality_checks(self) -> Dict[str, bool]:
        """Run code quality checks."""
        results = {}
        
        log_info("Running code quality checks")
        
        # Linting
        log_info("Running linting checks...")
        lint_result = run_command(
            ["ruff", "check", "src", "tests"],
            "Linting with Ruff"
        )
        results["linting"] = lint_result is not None
        
        # Type checking
        log_info("Running type checking...")
        mypy_result = run_command(
            ["mypy", "src"],
            "Type checking with MyPy"
        )
        results["type_checking"] = mypy_result is not None
        
        # Security scanning
        log_info("Running security scanning...")
        bandit_result = run_command(
            ["bandit", "-r", "src/", "-f", "json", "-o", "reports/bandit-report.json"],
            "Security scanning with Bandit"
        )
        results["security_scanning"] = bandit_result is not None
        
        # Dependency vulnerability check
        log_info("Checking dependency vulnerabilities...")
        safety_result = run_command(
            ["safety", "check", "--json", "--output", "reports/safety-report.json"],
            "Dependency vulnerability check"
        )
        results["dependency_check"] = safety_result is not None
        
        return results
    
    def generate_test_report(self, results: Dict[str, bool]) -> None:
        """Generate a test summary report."""
        log_info("Generating test report...")
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"\n{Colors.HEADER}=== TEST SUMMARY ==={Colors.ENDC}")
        print(f"Total test categories: {total_tests}")
        print(f"Passed: {Colors.OKGREEN}{passed_tests}{Colors.ENDC}")
        print(f"Failed: {Colors.FAIL}{failed_tests}{Colors.ENDC}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        print(f"\n{Colors.HEADER}=== DETAILED RESULTS ==={Colors.ENDC}")
        for category, passed in results.items():
            status = f"{Colors.OKGREEN}PASS{Colors.ENDC}" if passed else f"{Colors.FAIL}FAIL{Colors.ENDC}"
            print(f"{category.replace('_', ' ').title()}: {status}")
        
        # Coverage report
        if (self.coverage_dir / "index.html").exists():
            print(f"\n{Colors.OKBLUE}Coverage report available at: {self.coverage_dir / 'index.html'}{Colors.ENDC}")
        
        # XML reports
        xml_reports = list(self.reports_dir.glob("*.xml"))
        if xml_reports:
            print(f"\n{Colors.OKBLUE}JUnit XML reports:{Colors.ENDC}")
            for report in xml_reports:
                print(f"  - {report}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RLHF Audit Trail Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s unit                    # Run unit tests only
  %(prog)s all                     # Run all tests
  %(prog)s compliance --framework eu_ai_act  # Run EU AI Act compliance tests
  %(prog)s fast                    # Run fast tests for quick feedback
  %(prog)s quality                 # Run code quality checks
        """
    )
    
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "e2e", "compliance", "security", "privacy", 
                "performance", "smoke", "fast", "all", "quality"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true", 
        help="Skip coverage reporting"
    )
    
    parser.add_argument(
        "--framework",
        choices=["eu_ai_act", "nist", "gdpr"],
        help="Specific compliance framework to test"
    )
    
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow tests when running 'all'"
    )
    
    args = parser.parse_args()
    
    # Set up test environment
    root_dir = Path(__file__).parent.parent
    test_suite = TestSuite(root_dir)
    
    # Ensure we're in the right directory
    os.chdir(root_dir)
    
    print(f"{Colors.HEADER}RLHF Audit Trail Test Runner{Colors.ENDC}")
    print(f"Root directory: {root_dir}")
    print(f"Test type: {args.test_type}")
    
    results = {}
    success = True
    
    try:
        if args.test_type == "unit":
            success = test_suite.run_unit_tests(
                verbose=args.verbose, 
                coverage=not args.no_coverage
            )
            results["unit"] = success
            
        elif args.test_type == "integration":
            success = test_suite.run_integration_tests(verbose=args.verbose)
            results["integration"] = success
            
        elif args.test_type == "e2e":
            success = test_suite.run_e2e_tests(verbose=args.verbose)
            results["e2e"] = success
            
        elif args.test_type == "compliance":
            success = test_suite.run_compliance_tests(
                framework=args.framework,
                verbose=args.verbose
            )
            results["compliance"] = success
            
        elif args.test_type == "security":
            success = test_suite.run_security_tests(verbose=args.verbose)
            results["security"] = success
            
        elif args.test_type == "privacy":
            success = test_suite.run_privacy_tests(verbose=args.verbose)
            results["privacy"] = success
            
        elif args.test_type == "performance":
            success = test_suite.run_performance_tests(verbose=args.verbose)
            results["performance"] = success
            
        elif args.test_type == "smoke":
            success = test_suite.run_smoke_tests(verbose=args.verbose)
            results["smoke"] = success
            
        elif args.test_type == "fast":
            success = test_suite.run_fast_tests(verbose=args.verbose)
            results["fast"] = success
            
        elif args.test_type == "all":
            results = test_suite.run_all_tests(
                verbose=args.verbose,
                include_slow=args.include_slow
            )
            success = all(results.values())
            
        elif args.test_type == "quality":
            results = test_suite.run_quality_checks()
            success = all(results.values())
        
        # Generate report
        if results:
            test_suite.generate_test_report(results)
        
        if success:
            log_success("All tests completed successfully!")
            sys.exit(0)
        else:
            log_error("Some tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        log_warning("Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()