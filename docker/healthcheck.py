#!/usr/bin/env python3
"""
Health check script for RLHF Audit Trail container.
Performs comprehensive health checks and returns appropriate exit codes.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, '/app/src')

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    import urllib.request
    import urllib.error

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class HealthChecker:
    """Comprehensive health checker for RLHF Audit Trail."""
    
    def __init__(self):
        self.host = os.environ.get('RLHF_AUDIT_HOST', '0.0.0.0')
        self.port = int(os.environ.get('RLHF_AUDIT_PORT', '8000'))
        self.timeout = int(os.environ.get('HEALTH_CHECK_TIMEOUT', '10'))
        self.environment = os.environ.get('RLHF_AUDIT_ENV', 'production')
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Health check results
        self.results: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'unknown',
            'checks': {},
            'errors': []
        }
    
    def check_http_endpoint(self) -> bool:
        """Check if HTTP endpoint is responding."""
        try:
            url = f"http://{self.host}:{self.port}/health"
            
            if REQUESTS_AVAILABLE:
                response = requests.get(url, timeout=self.timeout)
                success = response.status_code == 200
                
                if success:
                    try:
                        data = response.json()
                        self.results['checks']['http_endpoint'] = {
                            'status': 'healthy',
                            'response_data': data,
                            'response_time_ms': response.elapsed.total_seconds() * 1000
                        }
                    except json.JSONDecodeError:
                        self.results['checks']['http_endpoint'] = {
                            'status': 'healthy',
                            'response_data': 'non-json response',
                            'response_time_ms': response.elapsed.total_seconds() * 1000
                        }
                else:
                    self.results['checks']['http_endpoint'] = {
                        'status': 'unhealthy',
                        'error': f'HTTP {response.status_code}',
                        'response_text': response.text[:200]
                    }
                    
            else:
                # Fallback using urllib
                request = urllib.request.Request(url)
                request.add_header('User-Agent', 'HealthCheck/1.0')
                
                start_time = time.time()
                response = urllib.request.urlopen(request, timeout=self.timeout)
                response_time = (time.time() - start_time) * 1000
                
                success = response.status == 200
                
                if success:
                    self.results['checks']['http_endpoint'] = {
                        'status': 'healthy',
                        'response_code': response.status,
                        'response_time_ms': response_time
                    }
                else:
                    self.results['checks']['http_endpoint'] = {
                        'status': 'unhealthy',
                        'error': f'HTTP {response.status}'
                    }
            
            return success
            
        except Exception as e:
            self.logger.error(f"HTTP endpoint check failed: {e}")
            self.results['checks']['http_endpoint'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            self.results['errors'].append(f"HTTP endpoint: {e}")
            return False
    
    def check_database_connection(self) -> bool:
        """Check database connection health."""
        try:
            from rlhf_audit_trail.database import DatabaseManager
            
            database_url = os.environ.get('RLHF_AUDIT_DATABASE_URL')
            if not database_url:
                self.results['checks']['database'] = {
                    'status': 'skipped',
                    'reason': 'No database URL configured'
                }
                return True
            
            # Create database manager and test connection
            db_manager = DatabaseManager(database_url)
            
            # Run async health check
            result = asyncio.run(db_manager.health_check())
            
            success = result.get('status') == 'healthy'
            
            self.results['checks']['database'] = {
                'status': 'healthy' if success else 'unhealthy',
                'connection_test': result.get('connection_test'),
                'total_sessions': result.get('total_sessions'),
                'total_events': result.get('total_events')
            }
            
            if not success:
                self.results['errors'].append(f"Database: {result.get('error', 'Unknown error')}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Database check failed: {e}")
            self.results['checks']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            self.results['errors'].append(f"Database: {e}")
            return False
    
    def check_file_system(self) -> bool:
        """Check file system health and permissions."""
        try:
            checks = {}
            overall_success = True
            
            # Check required directories
            required_dirs = ['/app/data', '/app/logs', '/app/audit_data']
            
            for dir_path in required_dirs:
                path = Path(dir_path)
                
                if not path.exists():
                    checks[f'dir_{path.name}'] = {
                        'status': 'unhealthy',
                        'error': 'Directory does not exist'
                    }
                    overall_success = False
                    continue
                
                # Check write permissions
                test_file = path / '.health_check_test'
                try:
                    test_file.write_text('test')
                    test_file.unlink()
                    
                    checks[f'dir_{path.name}'] = {
                        'status': 'healthy',
                        'writable': True
                    }
                    
                except Exception as e:
                    checks[f'dir_{path.name}'] = {
                        'status': 'unhealthy',
                        'error': f'Not writable: {e}'
                    }
                    overall_success = False
            
            # Check disk space
            try:
                if PSUTIL_AVAILABLE:
                    disk_usage = psutil.disk_usage('/app')
                    free_space_mb = disk_usage.free / (1024 * 1024)
                    
                    checks['disk_space'] = {
                        'status': 'healthy' if free_space_mb > 100 else 'warning',
                        'free_space_mb': round(free_space_mb, 2),
                        'total_space_mb': round(disk_usage.total / (1024 * 1024), 2),
                        'usage_percent': round((disk_usage.used / disk_usage.total) * 100, 2)
                    }
                    
                    if free_space_mb <= 50:  # Less than 50MB
                        overall_success = False
                        self.results['errors'].append(f"Low disk space: {free_space_mb:.1f}MB free")
                
            except Exception as e:
                checks['disk_space'] = {
                    'status': 'unknown',
                    'error': str(e)
                }
            
            self.results['checks']['filesystem'] = {
                'status': 'healthy' if overall_success else 'unhealthy',
                'details': checks
            }
            
            return overall_success
            
        except Exception as e:
            self.logger.error(f"Filesystem check failed: {e}")
            self.results['checks']['filesystem'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            self.results['errors'].append(f"Filesystem: {e}")
            return False
    
    def check_system_resources(self) -> bool:
        """Check system resource utilization."""
        try:
            if not PSUTIL_AVAILABLE:
                self.results['checks']['system_resources'] = {
                    'status': 'skipped',
                    'reason': 'psutil not available'
                }
                return True
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Load average (Linux only)
            load_avg = None
            try:
                load_avg = os.getloadavg()
            except (OSError, AttributeError):
                pass  # Not available on all systems
            
            # Determine health status
            warnings = []
            critical = []
            
            if cpu_percent > 90:
                critical.append(f"CPU usage critical: {cpu_percent}%")
            elif cpu_percent > 75:
                warnings.append(f"CPU usage high: {cpu_percent}%")
            
            if memory.percent > 90:
                critical.append(f"Memory usage critical: {memory.percent}%")
            elif memory.percent > 75:
                warnings.append(f"Memory usage high: {memory.percent}%")
            
            status = 'healthy'
            if critical:
                status = 'critical'
            elif warnings:
                status = 'warning'
            
            self.results['checks']['system_resources'] = {
                'status': status,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': round(memory.available / (1024 * 1024), 2),
                'load_average': load_avg,
                'warnings': warnings,
                'critical_issues': critical
            }
            
            if critical:
                self.results['errors'].extend(critical)
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"System resources check failed: {e}")
            self.results['checks']['system_resources'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            return True  # Non-critical failure
    
    def check_application_specific(self) -> bool:
        """Check application-specific health indicators."""
        try:
            checks = {}
            
            # Check if core modules can be imported
            try:
                from rlhf_audit_trail.core import AuditableRLHF
                checks['core_import'] = {'status': 'healthy'}
            except Exception as e:
                checks['core_import'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
            
            # Check configuration
            try:
                from rlhf_audit_trail.config import get_config
                config = get_config()
                checks['config'] = {
                    'status': 'healthy',
                    'privacy_mode': getattr(config, 'privacy_mode', 'unknown')
                }
            except Exception as e:
                checks['config'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
            
            # Check if any critical imports fail
            critical_failures = [
                check for check in checks.values()
                if check['status'] == 'unhealthy' and 'core' in str(check)
            ]
            
            self.results['checks']['application'] = {
                'status': 'healthy' if not critical_failures else 'unhealthy',
                'details': checks
            }
            
            if critical_failures:
                self.results['errors'].extend([
                    f"Application: {check.get('error', 'Unknown error')}"
                    for check in critical_failures
                ])
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Application check failed: {e}")
            self.results['checks']['application'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            self.results['errors'].append(f"Application: {e}")
            return False
    
    def run_all_checks(self) -> bool:
        """Run all health checks and return overall status."""
        self.logger.info("Starting health checks...")
        
        checks = [
            ('HTTP endpoint', self.check_http_endpoint),
            ('Database connection', self.check_database_connection),
            ('File system', self.check_file_system),
            ('System resources', self.check_system_resources),
            ('Application components', self.check_application_specific)
        ]
        
        results = []
        
        for check_name, check_func in checks:
            try:
                self.logger.info(f"Running {check_name} check...")
                result = check_func()
                results.append(result)
                
                status = "✓" if result else "✗"
                self.logger.info(f"{status} {check_name}: {'PASS' if result else 'FAIL'}")
                
            except Exception as e:
                self.logger.error(f"✗ {check_name}: ERROR - {e}")
                results.append(False)
                self.results['errors'].append(f"{check_name}: {e}")
        
        # Determine overall status
        if all(results):
            self.results['overall_status'] = 'healthy'
            overall_success = True
        elif any(results):
            self.results['overall_status'] = 'degraded'
            overall_success = False
        else:
            self.results['overall_status'] = 'unhealthy'
            overall_success = False
        
        self.logger.info(f"Health check completed: {self.results['overall_status'].upper()}")
        
        return overall_success
    
    def print_results(self):
        """Print health check results in a readable format."""
        print(json.dumps(self.results, indent=2))
        
        # Summary
        print(f"\n=== Health Check Summary ===")
        print(f"Overall Status: {self.results['overall_status'].upper()}")
        print(f"Timestamp: {self.results['timestamp']}")
        
        if self.results['errors']:
            print(f"\nErrors ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                print(f"  - {error}")
        else:
            print("\nNo errors detected.")
        
        print(f"\nCheck Details:")
        for check_name, check_data in self.results['checks'].items():
            status_symbol = {
                'healthy': '✓',
                'warning': '⚠',
                'unhealthy': '✗',
                'critical': '⚠',
                'skipped': '-',
                'unknown': '?'
            }.get(check_data.get('status', 'unknown'), '?')
            
            print(f"  {status_symbol} {check_name}: {check_data.get('status', 'unknown').upper()}")


def main():
    """Main health check execution."""
    try:
        checker = HealthChecker()
        success = checker.run_all_checks()
        
        # Print results if verbose mode or if unhealthy
        verbose = os.environ.get('HEALTH_CHECK_VERBOSE', '').lower() in ('true', '1', 'yes')
        if verbose or not success:
            checker.print_results()
        
        # Exit with appropriate code
        if success:
            sys.exit(0)  # Healthy
        else:
            sys.exit(1)  # Unhealthy
            
    except KeyboardInterrupt:
        print("\nHealth check interrupted")
        sys.exit(2)
        
    except Exception as e:
        print(f"Health check failed with unexpected error: {e}")
        sys.exit(3)


if __name__ == '__main__':
    main()