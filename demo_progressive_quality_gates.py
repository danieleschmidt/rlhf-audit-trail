#!/usr/bin/env python3
"""Demo: Progressive Quality Gates with ML-driven Adaptation.

Demonstrates the enhanced progressive quality gates system with:
- Autonomous ML-driven adaptation
- Real-time risk assessment
- Dynamic threshold optimization
- Performance-based evolution
"""

import asyncio
import json
import time
import random
from pathlib import Path

# Import our enhanced progressive gates
import sys
sys.path.append('src')

from rlhf_audit_trail.enhanced_progressive_gates import (
    EnhancedProgressiveGates, AdaptiveConfig, AdaptiveStrategy
)
from rlhf_audit_trail.progressive_quality_gates import GateStatus


def generate_sample_data(scenario: str = "normal") -> dict:
    """Generate sample data for different testing scenarios."""
    base_data = {
        'test_pass_rate': 0.85,
        'test_coverage': 0.80,
        'code_complexity': 5,
        'response_time': 200,
        'throughput': 1000,
        'memory_usage': 512,
        'cpu_usage': 50,
        'vulnerability_count': 0,
        'auth_strength': 0.9,
        'encryption_level': 0.95,
        'access_control_score': 0.85,
        'gdpr_compliance': 0.9,
        'audit_trail_completeness': 0.95,
        'data_retention_compliance': 0.88,
        'privacy_protection_level': 0.92,
        'uptime_percentage': 99.5,
        'error_rate': 0.01,
        'mean_recovery_time': 60,
        'current_requests_per_second': 500,
        'target_requests_per_second': 1000,
        'response_time_degradation': 1.2,
        'resource_utilization_at_load': 0.8
    }
    
    if scenario == "high_risk":
        base_data.update({
            'test_pass_rate': 0.65,
            'vulnerability_count': 3,
            'code_complexity': 12,
            'response_time': 800,
            'error_rate': 0.05,
            'cpu_usage': 85
        })
    elif scenario == "low_performance":
        base_data.update({
            'response_time': 1500,
            'throughput': 200,
            'memory_usage': 1500,
            'cpu_usage': 95,
            'current_requests_per_second': 100
        })
    elif scenario == "security_concern":
        base_data.update({
            'vulnerability_count': 5,
            'auth_strength': 0.6,
            'encryption_level': 0.7,
            'access_control_score': 0.5
        })
    elif scenario == "compliance_issues":
        base_data.update({
            'gdpr_compliance': 0.6,
            'audit_trail_completeness': 0.7,
            'privacy_protection_level': 0.5
        })
        
    return base_data


async def run_quality_gates_demo():
    """Run comprehensive demo of enhanced progressive quality gates."""
    print("🚀 Enhanced Progressive Quality Gates Demo")
    print("=" * 60)
    
    # Initialize enhanced progressive gates with different strategies
    configs = [
        ("Conservative", AdaptiveConfig(strategy=AdaptiveStrategy.CONSERVATIVE)),
        ("Balanced", AdaptiveConfig(strategy=AdaptiveStrategy.BALANCED)),
        ("ML-Driven", AdaptiveConfig(strategy=AdaptiveStrategy.ML_DRIVEN))
    ]
    
    for strategy_name, config in configs:
        print(f"\n🧠 Testing {strategy_name} Strategy")
        print("-" * 40)
        
        gates_system = EnhancedProgressiveGates(adaptive_config=config)
        
        # Test different scenarios
        scenarios = ["normal", "high_risk", "low_performance", "security_concern", "compliance_issues"]
        
        for scenario in scenarios:
            print(f"\n📊 Scenario: {scenario.replace('_', ' ').title()}")
            
            # Generate test data
            test_data = generate_sample_data(scenario)
            
            # Run quality gates
            results = await run_all_gates(gates_system, test_data)
            
            # Display results
            display_results(results, scenario)
            
            # Simulate learning from results
            await simulate_learning(gates_system, results, test_data)
            
        # Show evolution report
        evolution_report = await gates_system.get_evolution_report()
        display_evolution_report(evolution_report, strategy_name)


async def run_all_gates(gates_system, test_data):
    """Run all quality gates and collect results."""
    results = {}
    
    # Run adaptive gates
    adaptive_gates = [
        "adaptive_functional",
        "adaptive_performance", 
        "adaptive_security",
        "adaptive_compliance",
        "adaptive_reliability",
        "adaptive_scalability"
    ]
    
    for gate_id in adaptive_gates:
        if gate_id in gates_system.gates:
            gate = gates_system.gates[gate_id]
            try:
                result = await gate.validator(test_data)
                results[gate_id] = result
            except Exception as e:
                print(f"Error running {gate_id}: {e}")
                
    return results


def display_results(results, scenario):
    """Display quality gate results in a formatted way."""
    passed_count = sum(1 for r in results.values() if r.passed)
    total_count = len(results)
    
    print(f"  Results: {passed_count}/{total_count} gates passed")
    
    for gate_id, result in results.items():
        status_icon = "✅" if result.passed else "❌"
        gate_name = gate_id.replace("adaptive_", "").replace("_", " ").title()
        
        threshold_val = result.details.get('adaptive_threshold', 'N/A')
        threshold_str = f"{threshold_val:.3f}" if isinstance(threshold_val, (int, float)) else str(threshold_val)
        print(f"  {status_icon} {gate_name}: {result.score:.3f} "
              f"(threshold: {threshold_str}) "
              f"[{result.execution_time:.3f}s]")
        
        # Show key recommendations
        if result.recommendations:
            for rec in result.recommendations[:2]:  # Show first 2 recommendations
                print(f"    💡 {rec}")


async def simulate_learning(gates_system, results, test_data):
    """Simulate learning from gate execution results."""
    # Create mock feedback based on results
    predictions = {}
    actual_outcomes = {}
    
    for gate_id, result in results.items():
        predictions[f"{gate_id}_score"] = result.score
        # Simulate actual outcome with some variance
        actual_outcomes[f"{gate_id}_score"] = result.score + random.uniform(-0.1, 0.1)
        
    # Learn from feedback
    await gates_system.ml_engine.learn_from_feedback(
        predictions, actual_outcomes, test_data
    )


def display_evolution_report(report, strategy_name):
    """Display evolution report."""
    print(f"\n📈 Evolution Report - {strategy_name} Strategy")
    print("-" * 50)
    print(f"Total Adaptations: {report['total_adaptations']}")
    print(f"ML Confidence: {report['average_ml_confidence']:.3f}")
    print(f"Success Rate: {report['average_success_rate']:.3f}")
    print(f"Quality Improvement: {report['average_quality_improvement']:.3f}")
    print(f"System Health: {report['system_health_score']:.3f}")
    print(f"Risk Level: {report['current_risk_level']:.3f}")
    
    print("\nGate Evolution Status:")
    for gate_id, evolution in report['gate_evolutions'].items():
        print(f"  {gate_id}: {evolution['adaptation_count']} adaptations, "
              f"confidence: {evolution['ml_confidence']:.3f}")


async def demo_autonomous_adaptation():
    """Demo autonomous adaptation over time."""
    print("\n🤖 Autonomous Adaptation Demo")
    print("=" * 40)
    
    gates_system = EnhancedProgressiveGates(
        adaptive_config=AdaptiveConfig(strategy=AdaptiveStrategy.ML_DRIVEN)
    )
    
    # Simulate multiple iterations with evolving conditions
    for iteration in range(5):
        print(f"\n🔄 Iteration {iteration + 1}")
        
        # Generate evolving test data
        test_data = generate_evolving_data(iteration)
        
        # Run gates
        results = await run_all_gates(gates_system, test_data)
        
        # Create performance data for evolution
        performance_data = {
            'success_rate': sum(1 for r in results.values() if r.passed) / len(results),
            'avg_execution_time': sum(r.execution_time for r in results.values()) / len(results),
            'cost_efficiency': 0.75 + random.uniform(-0.1, 0.1),
            'risk_reduction': 0.1 + random.uniform(-0.05, 0.05),
            'user_satisfaction': 0.8 + random.uniform(-0.1, 0.1)
        }
        
        # Trigger evolution
        await gates_system.evolve_gates(performance_data)
        
        # Show adaptation
        print(f"  Success Rate: {performance_data['success_rate']:.3f}")
        print(f"  Avg Execution Time: {performance_data['avg_execution_time']:.3f}s")
        
        # Show threshold evolution for key gates
        for gate_id in ['adaptive_performance', 'adaptive_security']:
            if gate_id in gates_system.gates:
                threshold = gates_system.gates[gate_id].threshold
                adaptations = gates_system.gate_evolutions[gate_id].adaptation_count
                print(f"  {gate_id} threshold: {threshold:.3f} (adaptations: {adaptations})")


def generate_evolving_data(iteration):
    """Generate test data that evolves over iterations."""
    base_data = generate_sample_data("normal")
    
    # Simulate improving conditions over time
    improvement_factor = 1 + (iteration * 0.1)
    degradation_factor = 1 - (iteration * 0.05)
    
    base_data.update({
        'test_pass_rate': min(0.98, base_data['test_pass_rate'] * improvement_factor),
        'response_time': max(50, base_data['response_time'] * degradation_factor),
        'vulnerability_count': max(0, base_data['vulnerability_count'] - iteration),
        'test_coverage': min(0.95, base_data['test_coverage'] + iteration * 0.02)
    })
    
    return base_data


async def demo_risk_based_gating():
    """Demo risk-based dynamic gating."""
    print("\n⚠️  Risk-Based Dynamic Gating Demo")
    print("=" * 40)
    
    gates_system = EnhancedProgressiveGates()
    
    # Test scenarios with different risk profiles
    risk_scenarios = [
        ("Low Risk - Standard Flow", "normal"),
        ("Medium Risk - Enhanced Checks", "low_performance"), 
        ("High Risk - Maximum Validation", "high_risk"),
        ("Critical Security Risk", "security_concern")
    ]
    
    for scenario_name, scenario_type in risk_scenarios:
        print(f"\n🎯 {scenario_name}")
        
        test_data = generate_sample_data(scenario_type)
        
        # Extract features and predict risk
        features = await gates_system.ml_engine.extract_features(test_data)
        risk_predictions = await gates_system.ml_engine.predict_risk(features)
        
        print(f"  Overall Risk: {risk_predictions['overall_risk']:.3f}")
        print(f"  Security Risk: {risk_predictions['security_risk']:.3f}")
        print(f"  Performance Risk: {risk_predictions['performance_risk']:.3f}")
        
        # Run gates with risk-aware execution
        results = await run_all_gates(gates_system, test_data)
        
        # Show how gates adapted to risk level
        security_result = results.get('adaptive_security')
        if security_result:
            scan_depth = security_result.details.get('scan_depth', 'unknown')
            print(f"  Security Scan Depth: {scan_depth}")
            
        performance_result = results.get('adaptive_performance')
        if performance_result:
            threshold = performance_result.details.get('adaptive_threshold', 'unknown')
            print(f"  Performance Threshold: {threshold}")


if __name__ == "__main__":
    async def main():
        print("🎭 Progressive Quality Gates ML Demo")
        print("Demonstrating autonomous, ML-driven quality gates\n")
        
        try:
            # Run main demo
            await run_quality_gates_demo()
            
            # Demo autonomous adaptation
            await demo_autonomous_adaptation()
            
            # Demo risk-based gating
            await demo_risk_based_gating()
            
            print("\n🎉 Demo completed successfully!")
            print("\nKey Features Demonstrated:")
            print("✅ ML-driven adaptive thresholds")
            print("✅ Risk-based gate selection")
            print("✅ Autonomous evolution and learning")
            print("✅ Real-time performance optimization")
            print("✅ Dynamic compliance validation")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the demo
    asyncio.run(main())