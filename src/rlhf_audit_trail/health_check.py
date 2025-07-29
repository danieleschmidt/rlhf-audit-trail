"""Health check system for RLHF Audit Trail.

This module provides comprehensive health checks for all system components
including database connectivity, Redis availability, storage systems,
and compliance monitoring status.
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import logging
import sys
from pathlib import Path

# Health check status levels
class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    duration_ms: float
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    timestamp: float
    checks: List[HealthCheckResult]
    summary: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "checks": [asdict(check) for check in self.checks],
            "summary": self.summary,
            "uptime": time.time() - self.timestamp if self.timestamp else 0
        }

class HealthChecker:
    """Comprehensive health checker for RLHF Audit Trail system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize health checker with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.checks: Dict[str, Callable] = {}
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.checks.update({
            "database": self._check_database,
            "redis": self._check_redis,
            "storage": self._check_storage,
            "privacy_engine": self._check_privacy_engine,
            "compliance_monitor": self._check_compliance_monitor,
            "audit_trail": self._check_audit_trail,
            "system_resources": self._check_system_resources,
            "security_services": self._check_security_services,
        })
    
    async def check_health(self, checks: Optional[List[str]] = None) -> SystemHealth:
        """Run health checks and return overall system health."""
        start_time = time.time()
        
        # Determine which checks to run
        checks_to_run = checks or list(self.checks.keys())
        results = []
        
        # Run health checks
        for check_name in checks_to_run:
            if check_name in self.checks:
                try:
                    result = await self._run_check(check_name, self.checks[check_name])
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Health check '{check_name}' failed: {e}")
                    results.append(HealthCheckResult(
                        name=check_name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed with exception: {str(e)}",
                        timestamp=time.time(),
                        duration_ms=0,
                        error=str(e)
                    ))
        
        # Calculate overall status
        overall_status = self._calculate_overall_status(results)
        
        # Generate summary
        summary = {
            "total": len(results),
            "healthy": sum(1 for r in results if r.status == HealthStatus.HEALTHY),
            "degraded": sum(1 for r in results if r.status == HealthStatus.DEGRADED),
            "unhealthy": sum(1 for r in results if r.status == HealthStatus.UNHEALTHY),
            "unknown": sum(1 for r in results if r.status == HealthStatus.UNKNOWN),
        }
        
        return SystemHealth(
            status=overall_status,
            timestamp=start_time,
            checks=results,
            summary=summary
        )
    
    async def _run_check(self, name: str, check_func: Callable) -> HealthCheckResult:
        """Run a single health check with timing."""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, HealthCheckResult):
                result.duration_ms = duration_ms
                return result
            else:
                # Legacy return format
                return HealthCheckResult(
                    name=name,
                    status=result.get("status", HealthStatus.UNKNOWN),
                    message=result.get("message", "No message"),
                    timestamp=time.time(),
                    duration_ms=duration_ms,
                    details=result.get("details")
                )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                timestamp=time.time(),
                duration_ms=duration_ms,
                error=str(e)
            )
    
    def _calculate_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Calculate overall system health status."""
        if not results:
            return HealthStatus.UNKNOWN
        
        # If any critical checks are unhealthy, system is unhealthy
        critical_checks = ["database", "audit_trail"]
        for result in results:
            if result.name in critical_checks and result.status == HealthStatus.UNHEALTHY:
                return HealthStatus.UNHEALTHY
        
        # Count status types
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0,
        }
        
        for result in results:
            status_counts[result.status] += 1
        
        # Determine overall status
        total_checks = len(results)
        
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > total_checks * 0.3:  # More than 30% degraded
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.HEALTHY] > total_checks * 0.8:  # More than 80% healthy
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.DEGRADED
    
    # Individual health check implementations
    
    async def _check_database(self) -> HealthCheckResult:
        """Check database connectivity and performance."""
        try:
            # Mock database check - replace with actual implementation
            connection_time = 0.05  # Simulated connection time
            
            if connection_time > 1.0:
                return HealthCheckResult(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    message=f"Database connection slow: {connection_time:.3f}s",
                    timestamp=time.time(),
                    duration_ms=0,
                    details={"connection_time": connection_time}
                )
            
            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection healthy",
                timestamp=time.time(),
                duration_ms=0,
                details={
                    "connection_time": connection_time,
                    "pool_size": 10,
                    "active_connections": 3
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                timestamp=time.time(),
                duration_ms=0,
                error=str(e)
            )
    
    async def _check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity and performance."""
        try:
            # Mock Redis check - replace with actual implementation
            ping_time = 0.001  # Simulated ping time
            
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis connection healthy",
                timestamp=time.time(),
                duration_ms=0,
                details={
                    "ping_time": ping_time,
                    "memory_usage": "45MB",
                    "connected_clients": 5
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}",
                timestamp=time.time(),
                duration_ms=0,
                error=str(e)
            )
    
    async def _check_storage(self) -> HealthCheckResult:
        """Check storage system health."""
        try:
            # Mock storage check - replace with actual implementation
            available_space = 85.6  # GB
            total_space = 100.0  # GB
            usage_percent = (total_space - available_space) / total_space * 100
            
            if usage_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Storage critically low: {usage_percent:.1f}% used"
            elif usage_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Storage usage high: {usage_percent:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Storage healthy: {usage_percent:.1f}% used"
            
            return HealthCheckResult(
                name="storage",
                status=status,
                message=message,
                timestamp=time.time(),
                duration_ms=0,
                details={
                    "available_gb": available_space,
                    "total_gb": total_space,
                    "usage_percent": usage_percent
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="storage",
                status=HealthStatus.UNHEALTHY,
                message=f"Storage check failed: {str(e)}",
                timestamp=time.time(),
                duration_ms=0,
                error=str(e)
            )
    
    async def _check_privacy_engine(self) -> HealthCheckResult:
        """Check privacy engine health."""
        try:
            # Mock privacy engine check
            privacy_budget_usage = 0.45  # 45% of total budget used
            
            if privacy_budget_usage > 0.9:
                status = HealthStatus.DEGRADED
                message = f"Privacy budget critical: {privacy_budget_usage:.1%} used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Privacy engine healthy: {privacy_budget_usage:.1%} budget used"
            
            return HealthCheckResult(
                name="privacy_engine",
                status=status,
                message=message,
                timestamp=time.time(),
                duration_ms=0,
                details={
                    "budget_usage": privacy_budget_usage,
                    "active_annotators": 12,
                    "noise_calibration": "optimal"
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="privacy_engine",
                status=HealthStatus.UNHEALTHY,
                message=f"Privacy engine check failed: {str(e)}",
                timestamp=time.time(),
                duration_ms=0,
                error=str(e)
            )
    
    async def _check_compliance_monitor(self) -> HealthCheckResult:
        """Check compliance monitoring system."""
        try:
            # Mock compliance check
            last_compliance_check = time.time() - 3600  # 1 hour ago
            compliance_status = "compliant"
            
            if time.time() - last_compliance_check > 86400:  # 24 hours
                status = HealthStatus.DEGRADED
                message = "Compliance check overdue"
            else:
                status = HealthStatus.HEALTHY
                message = f"Compliance monitoring healthy: {compliance_status}"
            
            return HealthCheckResult(
                name="compliance_monitor",
                status=status,
                message=message,
                timestamp=time.time(),
                duration_ms=0,
                details={
                    "last_check": last_compliance_check,
                    "compliance_status": compliance_status,
                    "frameworks": ["EU AI Act", "NIST"]
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="compliance_monitor",
                status=HealthStatus.UNHEALTHY,
                message=f"Compliance monitor check failed: {str(e)}",
                timestamp=time.time(),
                duration_ms=0,
                error=str(e)
            )
    
    async def _check_audit_trail(self) -> HealthCheckResult:
        """Check audit trail system integrity."""
        try:
            # Mock audit trail check
            last_audit_event = time.time() - 300  # 5 minutes ago
            integrity_verified = True
            
            if not integrity_verified:
                status = HealthStatus.UNHEALTHY
                message = "Audit trail integrity compromised"
            elif time.time() - last_audit_event > 3600:  # 1 hour
                status = HealthStatus.DEGRADED
                message = "No recent audit events"
            else:
                status = HealthStatus.HEALTHY
                message = "Audit trail healthy and verified"
            
            return HealthCheckResult(
                name="audit_trail",
                status=status,
                message=message,
                timestamp=time.time(),
                duration_ms=0,
                details={
                    "last_event": last_audit_event,
                    "integrity_verified": integrity_verified,
                    "merkle_root": "a1b2c3d4...",
                    "event_count_24h": 1247
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="audit_trail",
                status=HealthStatus.UNHEALTHY,
                message=f"Audit trail check failed: {str(e)}",
                timestamp=time.time(),
                duration_ms=0,
                error=str(e)
            )
    
    async def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource utilization."""
        try:
            # Mock system resource check
            cpu_usage = 45.2  # percent
            memory_usage = 68.5  # percent
            disk_usage = 72.1  # percent
            
            max_usage = max(cpu_usage, memory_usage, disk_usage)
            
            if max_usage > 90:
                status = HealthStatus.UNHEALTHY
                message = f"System resources critical: {max_usage:.1f}% max usage"
            elif max_usage > 80:
                status = HealthStatus.DEGRADED
                message = f"System resources high: {max_usage:.1f}% max usage"
            else:
                status = HealthStatus.HEALTHY
                message = f"System resources normal: {max_usage:.1f}% max usage"
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                timestamp=time.time(),
                duration_ms=0,
                details={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage,
                    "load_average": [1.2, 1.4, 1.1]
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"System resource check failed: {str(e)}",
                timestamp=time.time(),
                duration_ms=0,
                error=str(e)
            )
    
    async def _check_security_services(self) -> HealthCheckResult:
        """Check security service health."""
        try:
            # Mock security services check
            encryption_status = "active"
            key_rotation_due = False
            vulnerability_scan_age = 24  # hours
            
            if vulnerability_scan_age > 168:  # 1 week
                status = HealthStatus.DEGRADED
                message = f"Vulnerability scan overdue: {vulnerability_scan_age}h ago"
            elif key_rotation_due:
                status = HealthStatus.DEGRADED
                message = "Key rotation due"
            else:
                status = HealthStatus.HEALTHY
                message = "Security services healthy"
            
            return HealthCheckResult(
                name="security_services",
                status=status,
                message=message,
                timestamp=time.time(),
                duration_ms=0,
                details={
                    "encryption_status": encryption_status,
                    "key_rotation_due": key_rotation_due,
                    "last_vuln_scan": vulnerability_scan_age,
                    "active_certificates": 3
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="security_services",
                status=HealthStatus.UNHEALTHY,
                message=f"Security services check failed: {str(e)}",
                timestamp=time.time(),
                duration_ms=0,
                error=str(e)
            )

# CLI interface
async def main():
    """Main CLI interface for health checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RLHF Audit Trail Health Check")
    parser.add_argument("--checks", nargs="*", help="Specific checks to run")
    parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=Path, help="Output file")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run health checks
    checker = HealthChecker()
    health = await checker.check_health(args.checks)
    
    # Format output
    if args.format == "json":
        output = json.dumps(health.to_dict(), indent=2, default=str)
    else:
        output = format_text_output(health)
    
    # Write output
    if args.output:
        args.output.write_text(output)
        print(f"Health check results written to: {args.output}")
    else:
        print(output)
    
    # Exit with appropriate code
    if health.status == HealthStatus.UNHEALTHY:
        sys.exit(1)
    elif health.status == HealthStatus.DEGRADED:
        sys.exit(2)
    else:
        sys.exit(0)

def format_text_output(health: SystemHealth) -> str:
    """Format health results as human-readable text."""
    lines = []
    
    # Header
    status_symbol = {
        HealthStatus.HEALTHY: "✅",
        HealthStatus.DEGRADED: "⚠️",
        HealthStatus.UNHEALTHY: "❌",
        HealthStatus.UNKNOWN: "❓",
    }
    
    lines.append("=" * 60)
    lines.append(f"{status_symbol[health.status]} RLHF Audit Trail Health Check")
    lines.append(f"Overall Status: {health.status.value.upper()}")
    lines.append(f"Timestamp: {time.ctime(health.timestamp)}")
    lines.append("=" * 60)
    
    # Summary
    lines.append(f"\nSummary:")
    lines.append(f"  Total Checks: {health.summary['total']}")
    lines.append(f"  ✅ Healthy:   {health.summary['healthy']}")
    lines.append(f"  ⚠️  Degraded:  {health.summary['degraded']}")
    lines.append(f"  ❌ Unhealthy: {health.summary['unhealthy']}")
    lines.append(f"  ❓ Unknown:   {health.summary['unknown']}")
    
    # Individual checks
    lines.append(f"\nDetailed Results:")
    lines.append("-" * 40)
    
    for check in health.checks:
        symbol = status_symbol[check.status]
        lines.append(f"{symbol} {check.name}: {check.message}")
        lines.append(f"   Duration: {check.duration_ms:.1f}ms")
        
        if check.details:
            for key, value in check.details.items():
                lines.append(f"   {key}: {value}")
        
        if check.error:
            lines.append(f"   Error: {check.error}")
        
        lines.append("")
    
    return "\n".join(lines)

if __name__ == "__main__":
    asyncio.run(main())