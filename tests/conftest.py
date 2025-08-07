"""Test configuration and fixtures for quantum task planner tests."""

import pytest
import asyncio
import logging
from unittest.mock import Mock

from src.quantum_task_planner.core import Task, TaskState, QuantumPriority, QuantumTaskPlanner
from src.quantum_task_planner.validation import TaskValidator


# Configure test logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("quantum_planner").setLevel(logging.ERROR)


@pytest.fixture
def simple_task():
    """Create a simple test task."""
    return Task(
        name="Simple Test Task",
        description="A simple task for testing",
        priority=QuantumPriority.MEDIUM,
        estimated_duration=1.0
    )


@pytest.fixture  
def basic_planner():
    """Create a basic quantum task planner."""
    return QuantumTaskPlanner(
        name="test-planner",
        coherence_preservation=True,
        entanglement_enabled=True
    )


@pytest.fixture
def task_validator():
    """Create task validator for testing."""
    return TaskValidator()


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "benchmark: Benchmark tests")
    config.addinivalue_line("markers", "quantum: Quantum-specific tests")