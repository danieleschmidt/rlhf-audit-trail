#!/usr/bin/env python3
"""
Demonstration of basic functionality for RLHF Audit Trail and Quantum Task Planner.
This script validates that both core systems are operational.
"""

import asyncio
import time
from pathlib import Path

from src.rlhf_audit_trail.core import AuditableRLHF
from src.rlhf_audit_trail.config import PrivacyConfig
from src.quantum_task_planner.core import QuantumTaskPlanner, QuantumPriority


async def demo_rlhf_audit():
    """Demonstrate RLHF audit trail functionality."""
    print("🔬 RLHF Audit Trail Demo")
    print("=" * 50)
    
    # Initialize auditable RLHF system
    privacy_config = PrivacyConfig(epsilon=1.0, delta=1e-5, clip_norm=1.0)
    auditor = AuditableRLHF(
        model_name="demo-llama-7b",
        privacy_config=privacy_config,
        storage_backend="local",
        compliance_mode="eu_ai_act"
    )
    
    # Track a training session
    async with auditor.track_training("safety_alignment_demo") as session:
        print(f"📊 Started session: {session.session_id[:8]}...")
        
        # Log sample annotations
        await auditor.log_annotations(
            prompts=["What is AI safety?", "How does RLHF work?"],
            responses=["AI safety focuses on...", "RLHF trains models using..."],
            rewards=[0.85, 0.92],
            annotator_ids=["annotator_001", "annotator_002"]
        )
        
        # Track policy update
        await auditor.track_policy_update(
            model={"dummy": "model"},
            optimizer={"dummy": "optimizer"},
            batch={"dummy": "batch"},
            loss=0.234
        )
        
        # Create checkpoint
        await auditor.checkpoint(
            epoch=1,
            metrics={"loss": 0.234, "reward": 0.885}
        )
        
        print(f"✅ Privacy budget remaining: {auditor.get_privacy_report()['epsilon_remaining']:.3f}")
        
        # Generate model card (inside session context)
        model_card = await auditor.generate_model_card()
        print(f"📋 Generated model card with {len(model_card)} sections")
    
    print("✅ RLHF Audit Trail demo completed successfully\n")


def demo_quantum_planner():
    """Demonstrate quantum task planner functionality."""
    print("🔮 Quantum Task Planner Demo")
    print("=" * 50)
    
    # Initialize quantum planner
    planner = QuantumTaskPlanner("DemoPlanner")
    
    # Create quantum tasks
    task1 = planner.create_task(
        name="Critical Analysis",
        description="Analyze quantum entanglement patterns",
        priority=QuantumPriority.HIGH,
        estimated_duration=2.0
    )
    
    task2 = planner.create_task(
        name="Data Processing", 
        description="Process experimental data",
        priority=QuantumPriority.MEDIUM,
        estimated_duration=1.5
    )
    
    task3 = planner.create_task(
        name="Report Generation",
        description="Generate final analysis report",
        priority=QuantumPriority.LOW,
        estimated_duration=3.0,
        dependencies=[task1.id, task2.id]
    )
    
    print(f"📝 Created {len(planner.tasks)} quantum tasks")
    print(f"🎲 Task1 probability: {task1.probability:.2f}")
    print(f"🎲 Task2 probability: {task2.probability:.2f}")
    print(f"🎲 Task3 probability: {task3.probability:.2f}")
    
    # Collapse superposition tasks
    collapsed = planner.collapse_superposition_tasks()
    print(f"⚡ Collapsed {len(collapsed)} superposition tasks")
    
    # Show executable tasks
    executable = planner.get_executable_tasks()
    print(f"🚀 {len(executable)} tasks ready for execution")
    
    # Get system state
    state = planner.get_system_state()
    print(f"📈 System metrics: {state['quantum_metrics']['average_amplitude']:.3f} avg amplitude")
    print("✅ Quantum Task Planner demo completed successfully\n")


async def main():
    """Run both demonstrations."""
    print("🚀 TERRAGON LABS - AUTONOMOUS SDLC EXECUTION")
    print("=" * 80)
    print("Demonstrating Generation 1: MAKE IT WORK functionality\n")
    
    start_time = time.time()
    
    # Run RLHF demo
    await demo_rlhf_audit()
    
    # Run Quantum planner demo
    demo_quantum_planner()
    
    elapsed = time.time() - start_time
    print(f"⏱️  Total demo time: {elapsed:.2f} seconds")
    print("\n🎉 GENERATION 1 SUCCESSFULLY VALIDATED!")
    print("Both core systems are operational and ready for enhancement.")


if __name__ == "__main__":
    asyncio.run(main())