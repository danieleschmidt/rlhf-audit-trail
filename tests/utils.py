"""Testing utilities for RLHF Audit Trail."""

import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import torch


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def generate_prompts(count: int = 10) -> List[str]:
        """Generate test prompts."""
        templates = [
            "What is the best way to {}?",
            "How do I solve the problem of {}?",
            "Explain the concept of {} in simple terms.",
            "What are the benefits of {}?",
            "Compare {} and {}.",
            "Describe the process of {}.",
            "What factors should I consider when {}?",
            "How has {} evolved over time?",
            "What are the challenges of {}?",
            "Provide examples of {}."
        ]
        
        topics = [
            "machine learning", "climate change", "quantum computing",
            "artificial intelligence", "renewable energy", "space exploration",
            "biotechnology", "cybersecurity", "blockchain", "robotics",
            "data science", "cloud computing", "virtual reality",
            "sustainable development", "neural networks"
        ]
        
        prompts = []
        for i in range(count):
            template = random.choice(templates)
            if "{}" in template and template.count("{}") == 1:
                topic = random.choice(topics)
                prompts.append(template.format(topic))
            elif template.count("{}") == 2:
                topic1, topic2 = random.sample(topics, 2)
                prompts.append(template.format(topic1, topic2))
            else:
                prompts.append(template)
        
        return prompts
    
    @staticmethod
    def generate_responses(prompts: List[str]) -> List[str]:
        """Generate test responses for prompts."""
        response_starters = [
            "The best approach to this is",
            "This is a complex topic that involves",
            "To understand this concept, consider",
            "The key benefits include",
            "When comparing these options",
            "The process typically involves",
            "Important factors to consider are",
            "Over time, this has evolved through",
            "The main challenges include",
            "Here are some examples:"
        ]
        
        responses = []
        for prompt in prompts:
            starter = random.choice(response_starters)
            # Generate a response of 50-200 words
            word_count = random.randint(50, 200)
            words = ["and", "the", "of", "to", "a", "in", "for", "is", "on", "that", "by", "this", "with", "as", "it", "be", "or", "an", "are", "from", "at", "has", "been", "have", "will", "their", "but", "not", "they", "we", "all", "can", "had", "her", "was", "one", "our", "out", "day", "get", "may", "use", "water", "than", "many", "first", "would", "way", "like", "its", "who", "oil", "sit", "now", "find", "long", "down", "part", "came", "made", "time", "very", "when", "where", "much", "how", "some", "each", "which", "what", "said", "there", "about", "could", "these", "were", "him", "see", "into", "up", "do", "if", "she", "my", "both", "go", "other", "even", "more", "two", "come", "before", "back", "any", "did", "new", "help", "make", "most", "over", "know", "should", "just", "being", "after", "also", "such", "through", "does", "take", "need", "want", "still", "people", "used", "work", "well", "here", "life", "think", "great", "good", "little", "right", "different", "between", "own", "while", "old", "important", "might", "under", "number", "system", "too", "set", "world", "never", "look", "same", "those", "things", "another", "without", "place", "end", "small", "around", "three", "years", "always", "often", "every", "hand", "house", "large", "group", "point", "turn", "order", "possible", "something", "school", "move", "try", "kind", "against", "side", "form", "fact", "far", "high", "interest", "public", "state", "keep", "start", "course", "example", "information", "probably", "early", "government", "company", "business", "case", "development", "week", "question", "put", "money", "provide", "service", "however", "change", "study", "program", "problem", "social", "room", "power", "policy", "since", "often", "local", "book", "though", "community", "policy", "research", "health", "level", "process", "economic", "million", "market", "education", "country", "individual", "several", "student", "person", "today", "available", "national", "report", "special", "really", "include", "certainly", "law", "particularly", "usually", "almost", "areas", "understand", "within", "history", "perhaps", "plan", "control", "view", "increase", "learn", "simple", "general", "technology", "science", "natural", "approach", "development", "knowledge", "digital", "effective", "method", "significant", "create", "build", "analysis", "sustainable", "innovative", "comprehensive"]
            
            # Create a more coherent response
            response_words = [starter] + random.sample(words, min(word_count - 1, len(words)))
            response = " ".join(response_words) + "."
            responses.append(response)
        
        return responses
    
    @staticmethod
    def generate_annotations(count: int = 10) -> List[Dict[str, Any]]:
        """Generate test annotations."""
        annotations = []
        
        for i in range(count):
            annotation = {
                "annotator_id": f"annotator_{i:03d}",
                "rating": round(random.uniform(1.0, 5.0), 1),
                "feedback": random.choice([
                    "Clear and helpful response",
                    "Could use more detail",
                    "Very informative",
                    "Good explanation",
                    "Needs improvement",
                    "Excellent coverage of the topic",
                    "Well structured",
                    "Missing key points",
                    "Comprehensive answer",
                    "Good but could be clearer"
                ]),
                "timestamp": datetime.now() - timedelta(minutes=random.randint(1, 1440)),
                "session_id": f"session_{random.randint(1000, 9999)}",
                "metadata": {
                    "annotation_time_seconds": random.randint(30, 300),
                    "annotator_expertise": random.choice(["novice", "intermediate", "expert"]),
                    "annotation_confidence": round(random.uniform(0.5, 1.0), 2)
                }
            }
            annotations.append(annotation)
        
        return annotations
    
    @staticmethod
    def generate_privacy_noise(shape: tuple, noise_scale: float = 1.0) -> np.ndarray:
        """Generate differential privacy noise."""
        return np.random.laplace(0, noise_scale, shape)
    
    @staticmethod
    def generate_model_weights(layer_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """Generate mock model weights."""
        weights = {}
        
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            weights[f"layer_{i}.weight"] = torch.randn(out_size, in_size)
            weights[f"layer_{i}.bias"] = torch.randn(out_size)
        
        return weights
    
    @staticmethod
    def generate_audit_events(count: int = 20) -> List[Dict[str, Any]]:
        """Generate test audit events."""
        event_types = [
            "annotation_logged", "policy_updated", "checkpoint_created",
            "verification_performed", "compliance_checked", "privacy_applied"
        ]
        
        events = []
        base_time = datetime.now() - timedelta(hours=24)
        
        for i in range(count):
            event = {
                "event_id": f"evt_{i:06d}",
                "event_type": random.choice(event_types),
                "timestamp": base_time + timedelta(minutes=i * 5),
                "session_id": f"session_{random.randint(1000, 9999)}",
                "user_id": f"user_{random.randint(100, 999)}",
                "metadata": {
                    "version": "1.0.0",
                    "environment": "test",
                    "source": "unit_test"
                },
                "data": {
                    "operation_duration_ms": random.randint(10, 1000),
                    "success": random.choice([True, True, True, False]),  # 75% success rate
                    "resource_usage": {
                        "cpu_percent": round(random.uniform(10, 90), 1),
                        "memory_mb": random.randint(100, 2000)
                    }
                }
            }
            events.append(event)
        
        return events


class MockBuilder:
    """Helper class to build various mock objects."""
    
    @staticmethod
    def build_audit_logger() -> MagicMock:
        """Build a mock audit logger."""
        mock = MagicMock()
        mock.log_event.return_value = {"logged": True, "event_id": "test_event"}
        mock.get_events.return_value = TestDataGenerator.generate_audit_events()
        mock.verify_integrity.return_value = {"is_valid": True, "merkle_root": "abc123"}
        return mock
    
    @staticmethod
    def build_privacy_engine() -> MagicMock:
        """Build a mock privacy engine."""
        mock = MagicMock()
        mock.apply_privacy.return_value = {"data": "anonymized", "noise_added": True}
        mock.check_budget.return_value = {"remaining": 5.0, "used": 3.0}
        mock.get_privacy_report.return_value = {
            "total_epsilon": 8.0,
            "remaining_budget": 5.0,
            "privacy_guarantees": "strong"
        }
        return mock
    
    @staticmethod
    def build_compliance_validator() -> MagicMock:
        """Build a mock compliance validator."""
        mock = MagicMock()
        mock.validate_eu_ai_act.return_value = {"compliant": True, "score": 0.95}
        mock.validate_nist_framework.return_value = {"compliant": True, "score": 0.92}
        mock.generate_report.return_value = {"report_id": "test_report", "status": "passed"}
        return mock
    
    @staticmethod
    def build_model_checkpoint() -> MagicMock:
        """Build a mock model checkpoint."""
        mock = MagicMock()
        mock.save.return_value = {"path": "/tmp/checkpoint.pt", "size_mb": 150}
        mock.load.return_value = {"loaded": True, "model_state": "restored"}
        mock.verify.return_value = {"verified": True, "checksum": "def456"}
        return mock


class AssertionHelpers:
    """Helper methods for common test assertions."""
    
    @staticmethod
    def assert_valid_audit_event(event: Dict[str, Any]) -> None:
        """Assert that an event has valid audit structure."""
        required_fields = ["event_id", "event_type", "timestamp", "metadata"]
        for field in required_fields:
            assert field in event, f"Missing required field: {field}"
        
        assert isinstance(event["timestamp"], (str, datetime))
        assert isinstance(event["metadata"], dict)
        assert len(event["event_id"]) > 0
    
    @staticmethod
    def assert_privacy_preserved(original_data: Any, processed_data: Any) -> None:
        """Assert that privacy has been preserved in processed data."""
        # Check that data has been modified (noise added or anonymized)
        if isinstance(original_data, (int, float)):
            assert original_data != processed_data, "Numeric data should be modified by privacy mechanism"
        elif isinstance(original_data, str):
            assert original_data != processed_data or "anon" in processed_data.lower(), "String data should be anonymized"
        elif isinstance(original_data, dict):
            # For dictionaries, check that sensitive fields are modified
            sensitive_keys = ["annotator_id", "user_id", "email", "name"]
            for key in sensitive_keys:
                if key in original_data and key in processed_data:
                    assert original_data[key] != processed_data[key], f"Sensitive field {key} should be anonymized"
    
    @staticmethod
    def assert_compliance_metadata(data: Dict[str, Any], framework: str = "eu_ai_act") -> None:
        """Assert that data contains required compliance metadata."""
        if framework == "eu_ai_act":
            required_fields = ["risk_level", "human_oversight", "data_governance"]
            for field in required_fields:
                assert field in data.get("compliance", {}), f"Missing EU AI Act field: {field}"
        elif framework == "nist":
            required_fields = ["risk_management", "bias_evaluation", "explainability"]
            for field in required_fields:
                assert field in data.get("compliance", {}), f"Missing NIST field: {field}"
    
    @staticmethod
    def assert_cryptographic_integrity(data: Dict[str, Any]) -> None:
        """Assert that data has cryptographic integrity features."""
        assert "signature" in data or "hash" in data, "Data should have cryptographic integrity proof"
        
        if "signature" in data:
            assert len(data["signature"]) > 20, "Signature should be substantial length"
        
        if "hash" in data:
            assert len(data["hash"]) >= 32, "Hash should be at least 32 characters"
    
    @staticmethod
    def assert_performance_acceptable(duration_ms: float, max_ms: float = 1000) -> None:
        """Assert that performance is within acceptable limits."""
        assert duration_ms <= max_ms, f"Operation took {duration_ms}ms, exceeding limit of {max_ms}ms"
        assert duration_ms > 0, "Duration should be positive"


class TestFileManager:
    """Manage test files and cleanup."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.created_files: List[Path] = []
    
    def create_test_file(self, filename: str, content: str = "") -> Path:
        """Create a test file and track it for cleanup."""
        file_path = self.base_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w") as f:
            f.write(content)
        
        self.created_files.append(file_path)
        return file_path
    
    def create_json_file(self, filename: str, data: Dict[str, Any]) -> Path:
        """Create a JSON test file."""
        content = json.dumps(data, indent=2, default=str)
        return self.create_test_file(filename, content)
    
    def cleanup(self) -> None:
        """Clean up all created test files."""
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()
        self.created_files.clear()


class PerformanceTimer:
    """Context manager for timing test operations."""
    
    def __init__(self, max_duration_ms: float = 1000):
        self.max_duration_ms = max_duration_ms
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        
        if self.duration_ms > self.max_duration_ms:
            raise AssertionError(
                f"Operation took {self.duration_ms:.2f}ms, "
                f"exceeding limit of {self.max_duration_ms}ms"
            )


def generate_random_string(length: int = 10) -> str:
    """Generate a random string for testing."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_test_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate a test configuration with optional overrides."""
    base_config = {
        "environment": "test",
        "debug": True,
        "database_url": "sqlite:///:memory:",
        "redis_url": "redis://localhost:6379/1",
        "privacy": {
            "epsilon": 1.0,
            "delta": 1e-5,
            "clip_norm": 1.0
        },
        "compliance": {
            "mode": "eu_ai_act",
            "enable_audit_trail": True,
            "require_human_oversight": True
        },
        "security": {
            "enable_encryption": True,
            "require_digital_signatures": True
        }
    }
    
    if overrides:
        base_config.update(overrides)
    
    return base_config


# Export commonly used utilities
__all__ = [
    "TestDataGenerator",
    "MockBuilder", 
    "AssertionHelpers",
    "TestFileManager",
    "PerformanceTimer",
    "generate_random_string",
    "generate_test_config"
]