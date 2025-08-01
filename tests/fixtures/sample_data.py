"""Test fixtures and sample data for RLHF Audit Trail tests."""

import json
from datetime import datetime, timezone
from typing import Dict, List, Any


class SampleDataGenerator:
    """Generate sample data for testing."""
    
    @staticmethod
    def generate_sample_prompts(count: int = 10) -> List[str]:
        """Generate sample prompts for testing."""
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "How do neural networks work?",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "What is the theory of relativity?",
            "How does machine learning differ from traditional programming?",
            "What are the main causes of climate change?",
            "Explain the concept of blockchain technology.",
            "What is the difference between AI and machine learning?",
            "How do vaccines work?",
            "What is the significance of DNA in genetics?",
            "Explain the water cycle.",
            "What are the principles of sustainable development?",
            "How does the internet work?",
        ]
        return prompts[:count]
    
    @staticmethod
    def generate_sample_responses(prompts: List[str]) -> List[str]:
        """Generate sample responses corresponding to prompts."""
        responses = [
            "The capital of France is Paris, a beautiful city known for its art, culture, and history.",
            "Quantum computing uses quantum mechanics principles like superposition and entanglement to process information in ways that could solve certain problems much faster than classical computers.",
            "Neural networks are computational models inspired by the brain's structure, consisting of interconnected nodes (neurons) that process and transmit information through weighted connections.",
            "Renewable energy benefits include reduced greenhouse gas emissions, energy independence, job creation, and long-term cost savings compared to fossil fuels.",
            "Photosynthesis is the process by which plants convert light energy, carbon dioxide, and water into glucose and oxygen, providing energy for plant growth and oxygen for other organisms.",
            "Einstein's theory of relativity consists of special and general relativity, describing how space, time, and gravity are interconnected and how they behave at high speeds and in strong gravitational fields.",
            "Machine learning enables computers to learn patterns from data without explicit programming, while traditional programming requires developers to write specific instructions for every task.",
            "Main causes of climate change include greenhouse gas emissions from burning fossil fuels, deforestation, industrial processes, and agricultural practices that increase atmospheric CO2 levels.",
            "Blockchain is a distributed ledger technology that maintains a continuously growing list of records (blocks) linked and secured using cryptography, enabling decentralized and transparent transactions.",
            "AI is a broader field focused on creating intelligent machines, while machine learning is a subset of AI that uses algorithms to learn patterns from data and make predictions or decisions.",
            "Vaccines work by introducing harmless versions of pathogens or their components to the immune system, training it to recognize and fight the actual disease-causing organism.",
            "DNA carries genetic information in the form of sequences of nucleotides, determining an organism's traits and serving as instructions for cellular processes and inheritance.",
            "The water cycle describes how water moves through Earth's systems via evaporation, condensation, precipitation, and collection, continuously recycling water resources.",
            "Sustainable development principles include meeting present needs without compromising future generations' ability to meet their needs, balancing economic, social, and environmental considerations.",
            "The internet works through a global network of interconnected computers that communicate using standardized protocols, routing data packets through multiple pathways to reach their destinations.",
        ]
        return responses[:len(prompts)]
    
    @staticmethod
    def generate_sample_annotations(count: int = 10) -> List[Dict[str, Any]]:
        """Generate sample human annotations."""
        annotations = []
        for i in range(count):
            annotation = {
                "annotator_id": f"annotator_{i+1:03d}",
                "rating": round(3.5 + (i % 3) * 0.5 + (i % 2) * 0.3, 1),
                "feedback": [
                    "Excellent response, very informative",
                    "Good explanation, could be more detailed",
                    "Accurate and well-structured",
                    "Helpful but could include examples",
                    "Clear and concise explanation",
                    "Good coverage of the topic",
                    "Well-explained with proper context",
                    "Informative and easy to understand",
                    "Comprehensive response",
                    "Good but could be more engaging"
                ][i % 10],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "annotation_time_seconds": 45 + (i % 30),
                "confidence": round(0.7 + (i % 4) * 0.075, 2),
                "categories": {
                    "helpfulness": round(3.8 + (i % 3) * 0.4, 1),
                    "accuracy": round(4.0 + (i % 2) * 0.5, 1),
                    "clarity": round(3.9 + (i % 4) * 0.3, 1),
                    "completeness": round(3.7 + (i % 3) * 0.3, 1)
                }
            }
            annotations.append(annotation)
        return annotations
    
    @staticmethod
    def generate_privacy_config() -> Dict[str, Any]:
        """Generate sample privacy configuration."""
        return {
            "epsilon": 1.0,
            "delta": 1e-5,
            "clip_norm": 1.0,
            "noise_multiplier": 1.1,
            "annotator_privacy_mode": "moderate",
            "k_anonymity_level": 5,
            "privacy_budget_reset_interval": 3600,
            "local_dp_enabled": True,
            "anonymization_method": "hash_based"
        }
    
    @staticmethod
    def generate_compliance_config() -> Dict[str, Any]:
        """Generate sample compliance configuration."""
        return {
            "frameworks": ["eu_ai_act", "gdpr", "nist"],
            "audit_level": "comprehensive",
            "retention_period_years": 7,
            "human_oversight_required": True,
            "transparency_requirements": {
                "model_cards": True,
                "training_data_documentation": True,
                "decision_explanations": True
            },
            "risk_assessment": {
                "automated": True,
                "frequency": "continuous",
                "thresholds": {
                    "high_risk": 0.8,
                    "medium_risk": 0.5,
                    "low_risk": 0.2
                }
            }
        }
    
    @staticmethod
    def generate_audit_events(count: int = 50) -> List[Dict[str, Any]]:
        """Generate sample audit events."""
        event_types = [
            "training_start", "training_step", "training_end",
            "annotation_received", "model_update", "checkpoint_saved",
            "privacy_budget_allocated", "compliance_check", "audit_log_created"
        ]
        
        events = []
        for i in range(count):
            event = {
                "event_id": f"event_{i+1:04d}",
                "event_type": event_types[i % len(event_types)],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": f"session_{(i // 10) + 1:03d}",
                "data": {
                    "step": i,
                    "loss": round(1.0 - (i * 0.01), 3),
                    "learning_rate": 0.001,
                    "batch_size": 32
                },
                "hash": f"sha256:hash_{i+1:04d}",
                "signature": f"signature_{i+1:04d}",
                "privacy_applied": i % 3 == 0,
                "compliance_validated": i % 5 == 0
            }
            events.append(event)
        return events
    
    @staticmethod
    def generate_model_checkpoint() -> Dict[str, Any]:
        """Generate sample model checkpoint data."""
        return {
            "checkpoint_id": "checkpoint_001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_state": {
                "architecture": "transformer",
                "parameters": 7_000_000_000,
                "layers": 32,
                "hidden_size": 4096,
                "attention_heads": 32
            },
            "training_state": {
                "epoch": 10,
                "step": 1000,
                "global_step": 10000,
                "loss": 0.25,
                "learning_rate": 0.0001
            },
            "optimizer_state": {
                "type": "AdamW",
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.01
            },
            "metrics": {
                "train_loss": 0.25,
                "eval_loss": 0.28,
                "perplexity": 15.2,
                "bleu_score": 0.68,
                "human_eval_score": 4.2
            },
            "hashes": {
                "model_weights": "sha256:model_weights_hash",
                "optimizer_state": "sha256:optimizer_state_hash",
                "config": "sha256:config_hash"
            }
        }
    
    @staticmethod
    def generate_model_card() -> Dict[str, Any]:
        """Generate sample model card."""
        return {
            "model_name": "rlhf-audit-trail-model-v1",
            "model_version": "1.0.0",
            "creation_date": datetime.now(timezone.utc).isoformat(),
            "model_description": "A language model trained using RLHF with comprehensive audit trail",
            "intended_use": {
                "primary_use": "Conversational AI assistant",
                "primary_users": "Researchers and developers",
                "out_of_scope": "High-stakes decision making, medical diagnosis"
            },
            "training_data": {
                "dataset_name": "Custom RLHF dataset",
                "size": "10,000 examples",
                "collection_period": "2024-Q4",
                "languages": ["English"],
                "privacy_measures": ["Differential privacy", "Anonymization"]
            },
            "model_architecture": {
                "type": "Transformer",
                "parameters": "7B",
                "layers": 32,
                "context_length": 4096
            },
            "training_procedure": {
                "framework": "PyTorch",
                "rlhf_method": "PPO",
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "privacy_budget": 1.0
            },
            "evaluation": {
                "metrics": {
                    "perplexity": 15.2,
                    "bleu_score": 0.68,
                    "human_eval": 4.2,
                    "safety_score": 0.95
                },
                "datasets": ["Custom eval set", "Human preference benchmark"]
            },
            "ethical_considerations": {
                "bias_mitigation": "Applied during training and evaluation",
                "fairness_assessment": "Conducted across demographic groups",
                "privacy_protection": "Differential privacy with ε=1.0"
            },
            "limitations": {
                "technical": ["Limited context window", "Language-specific"],
                "ethical": ["May reflect training data biases", "Not suitable for high-stakes decisions"]
            },
            "compliance": {
                "eu_ai_act": {
                    "compliant": True,
                    "risk_category": "Limited risk",
                    "requirements_met": ["Transparency", "Human oversight", "Accuracy"]
                },
                "gdpr": {
                    "compliant": True,
                    "privacy_measures": ["Data minimization", "Anonymization", "Consent"]
                }
            },
            "audit_trail": {
                "available": True,
                "integrity_verified": True,
                "retention_period": "7 years"
            }
        }
    
    @staticmethod
    def generate_compliance_report() -> Dict[str, Any]:
        """Generate sample compliance report."""
        return {
            "report_id": "compliance_report_001",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reporting_period": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-12-31T23:59:59Z"
            },
            "frameworks_assessed": ["eu_ai_act", "gdpr", "nist_ai_rmf"],
            "overall_compliance_score": 0.94,
            "eu_ai_act": {
                "compliance_status": "Compliant",
                "score": 0.96,
                "requirements": {
                    "risk_management": {"status": "Met", "score": 0.95},
                    "data_governance": {"status": "Met", "score": 0.97},
                    "transparency": {"status": "Met", "score": 0.94},
                    "human_oversight": {"status": "Met", "score": 0.98},
                    "accuracy_robustness": {"status": "Met", "score": 0.93},
                    "record_keeping": {"status": "Met", "score": 0.99}
                },
                "findings": {
                    "critical": 0,
                    "major": 0,
                    "minor": 2,
                    "observations": 5
                }
            },
            "gdpr": {
                "compliance_status": "Compliant",
                "score": 0.92,
                "principles": {
                    "lawfulness": {"status": "Met", "score": 0.95},
                    "purpose_limitation": {"status": "Met", "score": 0.90},
                    "data_minimization": {"status": "Met", "score": 0.94},
                    "accuracy": {"status": "Met", "score": 0.91},
                    "storage_limitation": {"status": "Met", "score": 0.89},
                    "security": {"status": "Met", "score": 0.96}
                }
            },
            "recommendations": [
                "Enhance automated compliance monitoring",
                "Implement additional bias detection measures",
                "Expand documentation for edge cases",
                "Regular compliance training for team members"
            ],
            "next_assessment_date": "2025-12-31T00:00:00Z"
        }
    
    @staticmethod
    def save_sample_data_to_files(output_dir: str) -> None:
        """Save all sample data to JSON files for use in tests."""
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate all sample data
        prompts = SampleDataGenerator.generate_sample_prompts(15)
        responses = SampleDataGenerator.generate_sample_responses(prompts)
        annotations = SampleDataGenerator.generate_sample_annotations(15)
        
        data_files = {
            "prompts.json": prompts,
            "responses.json": responses,
            "annotations.json": annotations,
            "privacy_config.json": SampleDataGenerator.generate_privacy_config(),
            "compliance_config.json": SampleDataGenerator.generate_compliance_config(),
            "audit_events.json": SampleDataGenerator.generate_audit_events(100),
            "model_checkpoint.json": SampleDataGenerator.generate_model_checkpoint(),
            "model_card.json": SampleDataGenerator.generate_model_card(),
            "compliance_report.json": SampleDataGenerator.generate_compliance_report()
        }
        
        # Save to files
        for filename, data in data_files.items():
            file_path = output_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Sample data saved to {output_path}")
        return data_files