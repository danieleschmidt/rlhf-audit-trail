#!/usr/bin/env python3
"""Quality Gates Execution Script.

Executes comprehensive quality gates for the RLHF Audit Trail system.
This script runs all quality gates and generates detailed reports.
"""

import asyncio
import json
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rlhf_audit_trail.progressive_quality_gates import ProgressiveQualityGates, QualityGateType
from rlhf_audit_trail.autonomous_monitoring import AutonomousMonitor
from rlhf_audit_trail.enhanced_reliability import EnhancedReliabilitySystem
from rlhf_audit_trail.adaptive_security import AdaptiveSecuritySystem


async def main():
    """Execute comprehensive quality gates."""
    print("🚀 Starting TERRAGON SDLC Quality Gates Execution")
    print("=" * 60)
    
    # Initialize systems
    quality_gates = ProgressiveQualityGates()
    monitoring = AutonomousMonitor()
    reliability = EnhancedReliabilitySystem()
    security = AdaptiveSecuritySystem()
    
    start_time = time.time()
    
    try:
        # Start monitoring systems
        print("📊 Starting monitoring systems...")
        await monitoring.start_monitoring()
        await reliability.start_monitoring()
        
        # Record some sample metrics
        from rlhf_audit_trail.autonomous_monitoring import MetricData, AlertLevel
        monitoring.record_metric(MetricData(
            name="response_time_ms",
            value=45.0,
            timestamp=time.time(),
            unit="ms",
            tags={"component": "core_system"}
        ))
        
        monitoring.record_metric(MetricData(
            name="memory_usage_mb",
            value=256.0,
            timestamp=time.time(),
            unit="MB",
            tags={"component": "core_system"}
        ))
        
        # Execute quality gates by type
        gate_types = [
            QualityGateType.FUNCTIONAL,
            QualityGateType.PERFORMANCE,
            QualityGateType.SECURITY,
            QualityGateType.COMPLIANCE,
            QualityGateType.RELIABILITY,
            QualityGateType.SCALABILITY
        ]
        
        all_results = {}
        total_gates = 0
        passed_gates = 0
        
        for gate_type in gate_types:
            print(f"\\n🔍 Executing {gate_type.value.upper()} Quality Gates...")
            
            results = await quality_gates.execute_gates(
                gate_types=[gate_type],
                fail_fast=False
            )
            
            all_results.update(results)
            
            type_passed = len([r for r in results.values() if r.passed])
            type_total = len(results)
            
            total_gates += type_total
            passed_gates += type_passed
            
            print(f"✅ {gate_type.value}: {type_passed}/{type_total} gates passed")
            
            # Show failed gates
            failed = [r for r in results.values() if not r.passed]
            for failure in failed:
                print(f"   ❌ {failure.gate_id}: {failure.details.get('error', 'Failed')}")
        
        # Security system analysis
        print("\\n🔒 Running Security Analysis...")
        security_event = {
            "component": "test_component",
            "request_count": 50,
            "source_ip": "192.168.1.100",
            "user_agent": "test_agent"
        }
        
        incident = await security.analyze_security_event(
            "test_component",
            security_event,
            {"cpu_usage": 45.0, "memory_usage": 256.0}
        )
        
        if incident:
            print(f"   ⚠️  Security incident detected: {incident.threat_level.value}")
        else:
            print("   ✅ No security incidents detected")
        
        # Generate comprehensive report
        print("\\n📋 Generating Quality Gates Report...")
        
        report = {
            "execution_summary": {
                "total_gates": total_gates,
                "passed_gates": passed_gates,
                "success_rate": passed_gates / total_gates if total_gates > 0 else 0,
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            },
            "gate_results": {
                gate_id: {
                    "passed": result.passed,
                    "score": result.score,
                    "execution_time": result.execution_time,
                    "details": result.details,
                    "recommendations": result.recommendations
                }
                for gate_id, result in all_results.items()
            },
            "monitoring_status": monitoring.get_current_status(),
            "reliability_status": reliability.get_system_status(),
            "security_dashboard": security.get_security_dashboard(),
            "gate_statistics": quality_gates.get_gate_statistics()
        }
        
        # Save report
        output_path = Path("quality_gates_results.json")
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display results
        print("\\n" + "=" * 60)
        print("📊 QUALITY GATES EXECUTION RESULTS")
        print("=" * 60)
        print(f"✅ Gates Passed: {passed_gates}/{total_gates} ({passed_gates/total_gates*100:.1f}%)")
        print(f"⏱️  Execution Time: {time.time() - start_time:.2f} seconds")
        print(f"📁 Report saved to: {output_path.absolute()}")
        
        # Success criteria
        success_rate = passed_gates / total_gates if total_gates > 0 else 0
        if success_rate >= 0.85:
            print("\\n🎉 QUALITY GATES: PASSED (85%+ success rate achieved)")
            return 0
        else:
            print(f"\\n❌ QUALITY GATES: FAILED ({success_rate*100:.1f}% success rate, need 85%)")
            return 1
            
    except Exception as e:
        print(f"\\n💥 Quality Gates Execution Failed: {e}")
        return 1
    
    finally:
        # Cleanup
        await monitoring.stop_monitoring()
        await reliability.stop_monitoring()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))