"""Tests for validation and error handling."""

import pytest
import json
from datetime import datetime, timedelta

from src.quantum_task_planner.core import Task, TaskState, QuantumPriority
from src.quantum_task_planner.validation import (
    TaskValidator, DependencyValidator, ValidationResult,
    sanitize_input
)
from src.quantum_task_planner.exceptions import (
    ValidationError, DependencyError
)


class TestTaskValidator:
    """Test TaskValidator functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.validator = TaskValidator()
    
    def test_valid_task_creation(self):
        """Test validation of valid task parameters."""
        result = self.validator.validate_task_creation(
            name="Valid Task",
            description="A valid test task",
            priority=QuantumPriority.MEDIUM,
            estimated_duration=2.0,
            dependencies=["dep1", "dep2"],
            metadata={"type": "test", "priority": 1}
        )
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.sanitized_data["name"] == "Valid Task"
        assert result.sanitized_data["priority"] == QuantumPriority.MEDIUM
    
    def test_invalid_task_name(self):
        """Test validation of invalid task names."""
        # Empty name
        result = self.validator.validate_task_creation(name="")
        assert not result.is_valid
        assert any("Name too short" in error for error in result.errors)
        
        # Too long name
        long_name = "a" * 300
        result = self.validator.validate_task_creation(name=long_name)
        assert not result.is_valid
        assert any("Name too long" in error for error in result.errors)
        
        # Invalid characters
        result = self.validator.validate_task_creation(name="Task<script>")
        assert not result.is_valid
        assert any("invalid characters" in error for error in result.errors)
    
    def test_description_validation(self):
        """Test description validation."""
        # Too long description
        long_desc = "a" * 3000
        result = self.validator.validate_task_creation(
            name="Test",
            description=long_desc
        )
        assert not result.is_valid
        assert any("Description too long" in error for error in result.errors)
        
        # Whitespace trimming
        result = self.validator.validate_task_creation(
            name="Test",
            description="  Description with whitespace  "
        )
        assert result.is_valid
        assert result.sanitized_data["description"] == "Description with whitespace"
        assert any("whitespace has been trimmed" in warning for warning in result.warnings)
    
    def test_priority_validation(self):
        """Test priority validation."""
        # String priority
        result = self.validator.validate_task_creation(
            name="Test",
            priority="high"
        )
        assert result.is_valid
        assert result.sanitized_data["priority"] == QuantumPriority.HIGH
        
        # Invalid priority
        result = self.validator.validate_task_creation(
            name="Test", 
            priority="invalid"
        )
        assert not result.is_valid
        assert any("Invalid priority" in error for error in result.errors)
    
    def test_duration_validation(self):
        """Test duration validation."""
        # Negative duration
        result = self.validator.validate_task_creation(
            name="Test",
            estimated_duration=-1.0
        )
        assert not result.is_valid
        assert any("cannot be negative" in error for error in result.errors)
        
        # Excessive duration
        result = self.validator.validate_task_creation(
            name="Test",
            estimated_duration=10000.0
        )
        assert not result.is_valid
        assert any("Duration too large" in error for error in result.errors)
        
        # Zero duration warning
        result = self.validator.validate_task_creation(
            name="Test",
            estimated_duration=0.0
        )
        assert result.is_valid
        assert any("Zero duration" in warning for warning in result.warnings)
    
    def test_dependencies_validation(self):
        """Test dependencies validation."""
        # Valid UUIDs
        valid_uuid = "12345678-1234-1234-1234-123456789abc"
        result = self.validator.validate_task_creation(
            name="Test",
            dependencies=[valid_uuid]
        )
        assert result.is_valid
        assert valid_uuid in result.sanitized_data["dependencies"]
        
        # Invalid UUID format
        result = self.validator.validate_task_creation(
            name="Test",
            dependencies=["not-a-uuid"]
        )
        assert not result.is_valid
        assert any("not a valid UUID" in error for error in result.errors)
        
        # Duplicate dependencies
        result = self.validator.validate_task_creation(
            name="Test",
            dependencies=[valid_uuid, valid_uuid]
        )
        assert result.is_valid
        assert len(result.sanitized_data["dependencies"]) == 1
        assert any("Duplicate dependency" in warning for warning in result.warnings)
        
        # Too many dependencies
        many_deps = [f"12345678-1234-1234-1234-{i:012d}" for i in range(60)]
        result = self.validator.validate_task_creation(
            name="Test",
            dependencies=many_deps
        )
        assert not result.is_valid
        assert any("Too many dependencies" in error for error in result.errors)
    
    def test_metadata_validation(self):
        """Test metadata validation."""
        # Valid metadata
        metadata = {"type": "test", "priority": 1, "tags": ["urgent"]}
        result = self.validator.validate_task_creation(
            name="Test",
            metadata=metadata
        )
        assert result.is_valid
        assert result.sanitized_data["metadata"] == metadata
        
        # Too large metadata
        large_metadata = {"data": "x" * 20000}
        result = self.validator.validate_task_creation(
            name="Test",
            metadata=large_metadata
        )
        assert not result.is_valid
        assert any("Metadata too large" in error for error in result.errors)
        
        # Non-serializable metadata
        def non_serializable():
            pass
        
        result = self.validator.validate_task_creation(
            name="Test",
            metadata={"func": non_serializable}
        )
        assert not result.is_valid
        assert any("not serializable" in error for error in result.errors)
    
    def test_quantum_properties_validation(self):
        """Test quantum properties validation."""
        # Valid properties
        result = self.validator.validate_quantum_properties(
            amplitude=0.7,
            phase=1.57,  # π/2
            coherence_time=300.0
        )
        assert result.is_valid
        assert result.sanitized_data["amplitude"] == 0.7
        assert abs(result.sanitized_data["phase"] - 1.57) < 0.01
        
        # Invalid amplitude
        result = self.validator.validate_quantum_properties(
            amplitude=1.5,  # > 1.0
            phase=0.0,
            coherence_time=300.0
        )
        assert not result.is_valid
        assert any("Amplitude must be between" in error for error in result.errors)
        
        # Phase normalization
        result = self.validator.validate_quantum_properties(
            amplitude=0.5,
            phase=8.0,  # > 2π
            coherence_time=300.0
        )
        assert result.is_valid
        normalized_phase = result.sanitized_data["phase"]
        assert 0 <= normalized_phase <= 2 * 3.14159
        assert any("Phase normalized" in warning for warning in result.warnings)
        
        # Invalid coherence time
        result = self.validator.validate_quantum_properties(
            amplitude=0.5,
            phase=0.0,
            coherence_time=0.5  # Too short
        )
        assert not result.is_valid
        assert any("Coherence time must be between" in error for error in result.errors)
    
    def test_state_transition_validation(self):
        """Test state transition validation."""
        # Valid transitions
        valid_transitions = [
            (TaskState.SUPERPOSITION, TaskState.PENDING),
            (TaskState.PENDING, TaskState.RUNNING),
            (TaskState.RUNNING, TaskState.COMPLETED),
            (TaskState.FAILED, TaskState.PENDING),  # Retry
        ]
        
        for current, target in valid_transitions:
            result = self.validator.validate_task_state_transition(
                current, target, "test-task"
            )
            assert result.is_valid
        
        # Invalid transitions
        invalid_transitions = [
            (TaskState.COMPLETED, TaskState.RUNNING),  # Can't go back from completed
            (TaskState.SUPERPOSITION, TaskState.COMPLETED),  # Can't skip states
        ]
        
        for current, target in invalid_transitions:
            result = self.validator.validate_task_state_transition(
                current, target, "test-task"
            )
            assert not result.is_valid
            assert any("Invalid state transition" in error for error in result.errors)
    
    def test_entanglement_validation(self):
        """Test entanglement validation."""
        task1 = Task(name="Task 1", priority=QuantumPriority.HIGH)
        task2 = Task(name="Task 2", priority=QuantumPriority.HIGH)
        
        # Valid entanglement
        result = self.validator.validate_entanglement(task1, task2)
        assert result.is_valid
        
        # Self-entanglement
        result = self.validator.validate_entanglement(task1, task1)
        assert not result.is_valid
        assert any("Cannot entangle task with itself" in error for error in result.errors)
        
        # Already entangled
        task1.entangle_with(task2)
        result = self.validator.validate_entanglement(task1, task2)
        assert not result.is_valid
        assert any("already entangled" in error for error in result.errors)
        
        # Completed task
        task3 = Task(name="Task 3", state=TaskState.COMPLETED)
        task4 = Task(name="Task 4")
        result = self.validator.validate_entanglement(task3, task4)
        assert not result.is_valid
        assert any("Cannot entangle completed tasks" in error for error in result.errors)


class TestDependencyValidator:
    """Test DependencyValidator functionality."""
    
    def test_valid_dependency_graph(self):
        """Test validation of valid dependency graph."""
        # Create tasks with valid dependencies
        task1 = Task(name="Task 1")
        task2 = Task(name="Task 2", dependencies={task1.id})
        task3 = Task(name="Task 3", dependencies={task2.id})
        
        tasks = {
            task1.id: task1,
            task2.id: task2,
            task3.id: task3
        }
        
        validator = DependencyValidator(tasks)
        result = validator.validate_dependency_graph()
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        # Create circular dependency
        task1 = Task(name="Task 1")
        task2 = Task(name="Task 2", dependencies={task1.id})
        task3 = Task(name="Task 3", dependencies={task2.id})
        
        # Create circular dependency
        task1.dependencies.add(task3.id)
        
        tasks = {
            task1.id: task1,
            task2.id: task2,
            task3.id: task3
        }
        
        validator = DependencyValidator(tasks)
        result = validator.validate_dependency_graph()
        
        assert not result.is_valid
        assert any("Circular dependency" in error for error in result.errors)
    
    def test_missing_dependency_detection(self):
        """Test missing dependency detection."""
        task1 = Task(name="Task 1", dependencies={"non-existent-id"})
        
        tasks = {task1.id: task1}
        
        validator = DependencyValidator(tasks)
        result = validator.validate_dependency_graph()
        
        assert not result.is_valid
        assert any("missing dependencies" in error for error in result.errors)
    
    def test_orphaned_task_detection(self):
        """Test orphaned task detection."""
        task1 = Task(name="Task 1")  # No dependencies, not depended on
        task2 = Task(name="Task 2")
        task3 = Task(name="Task 3", dependencies={task2.id})  # Task2 is depended on
        
        tasks = {
            task1.id: task1,
            task2.id: task2,
            task3.id: task3
        }
        
        validator = DependencyValidator(tasks)
        result = validator.validate_dependency_graph()
        
        assert result.is_valid  # Orphaned tasks are warnings, not errors
        assert any("Orphaned tasks" in warning for warning in result.warnings)
    
    def test_specific_task_validation(self):
        """Test validation of specific task dependencies."""
        task1 = Task(name="Task 1")
        task2 = Task(name="Task 2", dependencies={task1.id})
        
        tasks = {
            task1.id: task1,
            task2.id: task2
        }
        
        validator = DependencyValidator(tasks)
        
        # Valid task
        result = validator.validate_task_dependencies(task2.id)
        assert result.is_valid
        
        # Add missing dependency
        task2.dependencies.add("missing-id")
        result = validator.validate_task_dependencies(task2.id)
        assert not result.is_valid
        assert any("does not exist" in error for error in result.errors)


class TestInputSanitization:
    """Test input sanitization functions."""
    
    def test_string_sanitization(self):
        """Test string sanitization."""
        # Basic sanitization
        result = sanitize_input("  test string  ", str)
        assert result == "test string"
        
        # Non-string input
        result = sanitize_input(123, str)
        assert result == "123"
        
        # None input
        result = sanitize_input(None, str)
        assert result is None
    
    def test_numeric_sanitization(self):
        """Test numeric sanitization."""
        # String to int
        result = sanitize_input("42", int)
        assert result == 42
        
        # String to float
        result = sanitize_input("3.14", float)
        assert abs(result - 3.14) < 0.01
        
        # Invalid conversion
        with pytest.raises(ValidationError):
            sanitize_input("not-a-number", int)
    
    def test_boolean_sanitization(self):
        """Test boolean sanitization."""
        # String representations
        true_values = ["true", "1", "yes", "on", "True", "YES"]
        for val in true_values:
            result = sanitize_input(val, bool)
            assert result is True
        
        false_values = ["false", "0", "no", "off", "False", "NO"]
        for val in false_values:
            result = sanitize_input(val, bool)
            assert result is False
        
        # Actual boolean
        result = sanitize_input(True, bool)
        assert result is True
    
    def test_list_sanitization(self):
        """Test list sanitization."""
        # Already a list
        result = sanitize_input([1, 2, 3], list)
        assert result == [1, 2, 3]
        
        # Tuple to list
        result = sanitize_input((1, 2, 3), list)
        assert result == [1, 2, 3]
        
        # Single item to list
        result = sanitize_input("item", list)
        assert result == ["item"]


class TestValidationResult:
    """Test ValidationResult functionality."""
    
    def test_validation_result_creation(self):
        """Test creating validation results."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["A warning"]
        )
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert bool(result) is True
    
    def test_adding_errors_and_warnings(self):
        """Test adding errors and warnings."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Add error
        result.add_error("An error occurred")
        assert not result.is_valid
        assert len(result.errors) == 1
        
        # Add warning
        result.add_warning("A warning")
        assert len(result.warnings) == 1
    
    def test_boolean_evaluation(self):
        """Test boolean evaluation of validation results."""
        valid_result = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert bool(valid_result) is True
        
        invalid_result = ValidationResult(is_valid=False, errors=["Error"], warnings=[])
        assert bool(invalid_result) is False


@pytest.mark.parametrize("name,description,priority,duration,should_pass", [
    ("Valid Task", "Valid description", "medium", 1.0, True),
    ("", "Valid description", "medium", 1.0, False),  # Empty name
    ("Valid Task", "", "medium", 1.0, True),  # Empty description OK
    ("Valid Task", "Valid description", "invalid", 1.0, False),  # Invalid priority
    ("Valid Task", "Valid description", "medium", -1.0, False),  # Negative duration
    ("Valid Task", "Valid description", "medium", 0.0, True),  # Zero duration (warning)
])
def test_validation_scenarios(name, description, priority, duration, should_pass):
    """Parameterized test for various validation scenarios."""
    validator = TaskValidator()
    
    result = validator.validate_task_creation(
        name=name,
        description=description,
        priority=priority,
        estimated_duration=duration
    )
    
    assert result.is_valid == should_pass


class TestEdgeCases:
    """Test edge cases and corner conditions."""
    
    def test_unicode_handling(self):
        """Test Unicode character handling."""
        validator = TaskValidator()
        
        unicode_name = "测试任务 🚀"
        result = validator.validate_task_creation(name=unicode_name)
        
        # Should handle Unicode gracefully
        assert result.is_valid or any("invalid characters" in error for error in result.errors)
    
    def test_very_large_metadata(self):
        """Test handling of very large metadata."""
        validator = TaskValidator()
        
        # Create metadata just under the limit
        large_metadata = {"data": "x" * 9000}
        result = validator.validate_task_creation(
            name="Test",
            metadata=large_metadata
        )
        assert result.is_valid
        
        # Create metadata over the limit
        oversized_metadata = {"data": "x" * 20000}
        result = validator.validate_task_creation(
            name="Test", 
            metadata=oversized_metadata
        )
        assert not result.is_valid
    
    def test_boundary_values(self):
        """Test boundary value conditions."""
        validator = TaskValidator()
        
        # Test minimum valid name length
        result = validator.validate_task_creation(name="a")
        assert result.is_valid
        
        # Test maximum valid name length
        max_name = "a" * validator.MAX_NAME_LENGTH
        result = validator.validate_task_creation(name=max_name)
        assert result.is_valid
        
        # Test just over maximum
        too_long_name = "a" * (validator.MAX_NAME_LENGTH + 1)
        result = validator.validate_task_creation(name=too_long_name)
        assert not result.is_valid
    
    def test_none_and_empty_values(self):
        """Test handling of None and empty values."""
        validator = TaskValidator()
        
        # Test with None values
        result = validator.validate_task_creation(
            name="Test",
            dependencies=None,
            metadata=None
        )
        assert result.is_valid
        
        # Test with empty values
        result = validator.validate_task_creation(
            name="Test",
            dependencies=[],
            metadata={}
        )
        assert result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])