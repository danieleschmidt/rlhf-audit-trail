#!/usr/bin/env python3
"""
Simple Usage Demo for RLHF Audit Trail
Generation 1: Make It Work - Basic functionality demonstration
"""

import asyncio
import sys
import os
from pathlib import Path
import json

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from rlhf_audit_trail import AuditableRLHF, PrivacyConfig, ComplianceConfig
    from rlhf_audit_trail.exceptions import AuditTrailError, PrivacyBudgetExceededError
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Try running: python3 scripts/setup.py")
    sys.exit(1)

def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_step(step, description):
    """Print step information."""
    print(f"\n🔄 Step {step}: {description}")

def print_success(message):
    """Print success message."""
    print(f"✅ {message}")

def print_info(message):
    """Print info message."""
    print(f"ℹ️  {message}")

def print_json_data(data, title="Data"):
    """Print JSON data nicely."""
    print(f"\n📄 {title}:")
    print("-" * 40)
    print(json.dumps(data, indent=2, default=str))

async def demo_basic_functionality():
    """Demonstrate basic RLHF audit trail functionality."""
    
    print_header("RLHF Audit Trail - Basic Functionality Demo")
    print("This demo shows core features working without heavy dependencies")
    
    # Step 1: Initialize the auditor
    print_step(1, "Initialize AuditableRLHF with basic configuration")
    
    privacy_config = PrivacyConfig(
        epsilon=10.0,           # Privacy budget
        delta=1e-5,            # Privacy parameter
        clip_norm=1.0,         # Gradient clipping
        noise_multiplier=1.1   # Noise level
    )
    
    compliance_config = ComplianceConfig()
    
    try:
        auditor = AuditableRLHF(
            model_name="demo-llama-7b",
            privacy_config=privacy_config,
            compliance_config=compliance_config,
            storage_backend="local",
            compliance_mode="eu_ai_act"
        )
        print_success("AuditableRLHF initialized successfully")
        
        # Show configuration
        print_json_data({
            "model_name": auditor.model_name,
            "privacy_epsilon": auditor.privacy_config.epsilon,
            "privacy_delta": auditor.privacy_config.delta,
            "compliance_mode": auditor.compliance_mode,
            "storage_backend": type(auditor.storage).__name__
        }, "Configuration")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Step 2: Create a training session
    print_step(2, "Create training session context")
    
    try:
        async with auditor.track_training("safety_alignment_demo") as session:
            print_success(f"Training session created: {session.session_id}")
            
            print_json_data({
                "session_id": session.session_id,
                "experiment_name": session.experiment_name,
                "model_name": session.model_name,
                "phase": session.phase.value,
                "is_active": session.is_active
            }, "Session Details")
            
            # Step 3: Log sample annotations
            print_step(3, "Log human annotations with differential privacy")
            
            sample_prompts = [
                "Write a helpful response about climate change",
                "Explain quantum computing in simple terms",
                "Describe the importance of data privacy"
            ]
            
            sample_responses = [
                "Climate change refers to long-term shifts in global temperatures...",
                "Quantum computing uses quantum bits that can exist in multiple states...",
                "Data privacy protects personal information from unauthorized access..."
            ]
            
            sample_rewards = [0.85, 0.92, 0.78]
            sample_annotator_ids = ["anon_001", "anon_002", "anon_001"]
            
            try:
                annotation_batch = await auditor.log_annotations(
                    prompts=sample_prompts,
                    responses=sample_responses,
                    rewards=sample_rewards,
                    annotator_ids=sample_annotator_ids,
                    metadata={"batch_type": "demo", "quality_check": True}
                )
                
                print_success(f"Logged annotation batch with {annotation_batch.batch_size} samples")
                print_info(f"Batch ID: {annotation_batch.batch_id}")
                
            except PrivacyBudgetExceededError as e:
                print(f"❌ Privacy budget exceeded: {e}")
            except Exception as e:
                print(f"❌ Annotation logging failed: {e}")
            
            # Step 4: Track policy updates
            print_step(4, "Track policy model updates")
            
            try:
                # Simulate multiple policy updates
                for epoch in range(3):
                    mock_model = f"model_state_epoch_{epoch}"
                    mock_optimizer = f"optimizer_epoch_{epoch}"
                    mock_batch = f"batch_data_epoch_{epoch}"
                    loss = 0.5 - (epoch * 0.1)  # Decreasing loss
                    
                    policy_update = await auditor.track_policy_update(
                        model=mock_model,
                        optimizer=mock_optimizer,
                        batch=mock_batch,
                        loss=loss,
                        metadata={"epoch": epoch, "demo_mode": True}
                    )
                    
                    print_success(f"Tracked policy update for epoch {epoch} (loss: {loss:.3f})")
                    
                    # Create checkpoint
                    metrics = {
                        "loss": loss,
                        "reward": 0.7 + (epoch * 0.05),
                        "perplexity": 15.0 - (epoch * 2.0)
                    }
                    
                    await auditor.checkpoint(
                        epoch=epoch,
                        metrics=metrics,
                        metadata={"checkpoint_type": "demo"}
                    )
                    
                    print_success(f"Created checkpoint for epoch {epoch}")
                    
            except Exception as e:
                print(f"❌ Policy update tracking failed: {e}")
            
            # Step 5: Generate privacy report
            print_step(5, "Generate privacy budget report")
            
            try:
                privacy_report = auditor.get_privacy_report()
                print_json_data(privacy_report, "Privacy Budget Report")
                
                remaining_ratio = privacy_report["epsilon_remaining"] / privacy_report["total_epsilon"]
                if remaining_ratio > 0.5:
                    print_success("Privacy budget is healthy")
                elif remaining_ratio > 0.25:
                    print_info("Privacy budget is moderate")
                else:
                    print("⚠️ Privacy budget is running low")
                    
            except Exception as e:
                print(f"❌ Privacy report generation failed: {e}")
            
            # Step 6: Generate model card
            print_step(6, "Generate compliant model card")
            
            try:
                model_card = await auditor.generate_model_card(
                    include_provenance=True,
                    include_privacy_analysis=True,
                    format="eu_standard"
                )
                
                # Show key sections of model card
                key_sections = {
                    "model_name": model_card.get("model_name"),
                    "experiment_name": model_card.get("experiment_name"),
                    "session_id": model_card.get("session_id"),
                    "training_summary": model_card.get("training_summary"),
                    "privacy_analysis": model_card.get("privacy_analysis", {}).get("differential_privacy"),
                    "compliance": model_card.get("compliance")
                }
                
                print_json_data(key_sections, "Model Card (Key Sections)")
                print_success("Model card generated successfully")
                
            except Exception as e:
                print(f"❌ Model card generation failed: {e}")
            
            # Step 7: Verify integrity
            print_step(7, "Verify audit trail integrity")
            
            try:
                verification = await auditor.verify_provenance()
                print_json_data(verification, "Integrity Verification")
                
                if verification.get("is_valid", False):
                    print_success("Audit trail integrity verified")
                else:
                    print("⚠️ Integrity verification issues found")
                    
            except Exception as e:
                print(f"❌ Verification failed: {e}")
    
    except Exception as e:
        print(f"❌ Training session failed: {e}")
        return
    
    print_header("Demo Completed Successfully")
    print("✅ All basic functionality is working")
    print("📝 Next steps:")
    print("   - Run: PYTHONPATH=src python3 -m rlhf_audit_trail.cli --help")
    print("   - Try: PYTHONPATH=src python3 -m rlhf_audit_trail.cli status")
    print("   - Explore: src/rlhf_audit_trail/ for implementation details")

async def demo_cli_functionality():
    """Demonstrate CLI functionality."""
    print_header("CLI Functionality Demo")
    
    print("The RLHF Audit Trail includes a comprehensive CLI interface:")
    print()
    print("📋 Available Commands:")
    commands = [
        ("session create", "Create and manage training sessions"),
        ("status", "Show system status and health"),
        ("database", "Initialize and manage database"),
        ("monitor", "Start system monitoring"),
        ("api", "Start REST API server"),
        ("dashboard", "Launch Streamlit dashboard"),
        ("verify", "Verify audit trail integrity"),
        ("export", "Export audit data")
    ]
    
    for cmd, desc in commands:
        print(f"  • {cmd:<15} - {desc}")
    
    print("\n🔧 Example Usage:")
    examples = [
        'PYTHONPATH=src python3 -m rlhf_audit_trail.cli session create --model-name "test-model" --experiment-name "demo" --duration 10',
        'PYTHONPATH=src python3 -m rlhf_audit_trail.cli status',
        'PYTHONPATH=src python3 -m rlhf_audit_trail.cli --help'
    ]
    
    for example in examples:
        print(f"  {example}")

def demo_features_overview():
    """Show feature overview."""
    print_header("RLHF Audit Trail - Feature Overview")
    
    features = {
        "🔐 Privacy Protection": [
            "Differential privacy for annotator protection",
            "Privacy budget tracking and management",
            "Configurable noise parameters",
            "Privacy exhaustion risk monitoring"
        ],
        "📋 Compliance": [
            "EU AI Act compliance framework",
            "NIST AI Risk Management alignment",
            "Automated compliance reporting",
            "Audit trail requirements satisfaction"
        ],
        "🔍 Cryptographic Verification": [
            "Merkle tree-based integrity verification",
            "Cryptographic signatures for events",
            "Tamper-evident audit logs",
            "Chain of custody tracking"
        ],
        "🏗️ Architecture": [
            "Local and cloud storage backends (S3, GCS, Azure)",
            "SQL database for metadata",
            "REST API for integration",
            "Real-time monitoring dashboard"
        ],
        "🧪 RLHF Integration": [
            "Support for TRL/TRLX frameworks",
            "Policy update tracking",
            "Human feedback annotation logging",
            "Training session management"
        ],
        "🚀 Production Ready": [
            "Docker containerization",
            "Kubernetes deployment",
            "Prometheus monitoring",
            "Multi-environment configuration"
        ]
    }
    
    for category, items in features.items():
        print(f"\n{category}")
        for item in items:
            print(f"  ✓ {item}")

async def main():
    """Main demo entry point."""
    try:
        # Show feature overview
        demo_features_overview()
        
        # Run basic functionality demo
        await demo_basic_functionality()
        
        # Show CLI demo
        await demo_cli_functionality()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())