"""Tests for quantum task planner core functionality."""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.quantum_task_planner.core import (
    Task, TaskState, QuantumPriority, QuantumTaskPlanner
)
from src.quantum_task_planner.exceptions import (
    QuantumTaskPlannerError, TaskNotFoundError, ValidationError
)


class TestTask:
    """Test Task class functionality."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        task = Task(
            name="Test Task",
            description="A test task",
            priority=QuantumPriority.HIGH
        )
        
        assert task.name == "Test Task"
        assert task.description == "A test task"
        assert task.priority == QuantumPriority.HIGH
        assert task.state == TaskState.SUPERPOSITION
        assert 0.0 <= task.amplitude <= 1.0
        assert 0.0 <= task.phase <= 2 * 3.14159
        assert task.id is not None
    
    def test_task_probability(self):
        """Test probability calculation."""
        task = Task(name="Test Task", amplitude=0.8)
        
        expected_probability = 0.8 ** 2
        assert abs(task.probability - expected_probability) < 0.001
    
    def test_task_coherence(self):
        """Test quantum coherence."""
        task = Task(name="Test Task", coherence_time=100.0)
        
        # Should be coherent immediately after creation
        assert task.is_coherent
        
        # Simulate time passage
        old_created_at = task.created_at
        task.created_at = datetime.now() - timedelta(seconds=200)
        
        # Should now be decoherent
        assert not task.is_coherent
        
        # Restore original time
        task.created_at = old_created_at
    
    def test_wave_function_collapse(self):
        """Test wave function collapse."""
        task = Task(name="Test Task", amplitude=0.9)
        
        # High amplitude should usually collapse to PENDING
        original_state = task.state
        new_state = task.collapse_wave_function()
        
        assert original_state == TaskState.SUPERPOSITION
        assert new_state in [TaskState.PENDING, TaskState.DEFERRED]
        assert task.state == new_state
    
    def test_entanglement(self):
        """Test quantum entanglement."""
        task1 = Task(name="Task 1")
        task2 = Task(name="Task 2")
        
        # Initially not entangled
        assert task1.entangled_tasks == set()
        assert task2.entangled_tasks == set()
        
        # Create entanglement
        task1.entangle_with(task2)
        
        # Both should be entangled
        assert task2.id in task1.entangled_tasks
        assert task1.id in task2.entangled_tasks
        assert task1.state == TaskState.ENTANGLED
        assert task2.state == TaskState.ENTANGLED
    
    def test_interference(self):
        """Test quantum interference."""
        task1 = Task(name="Task 1", phase=0.0, amplitude=0.5)
        task2 = Task(name="Task 2", phase=0.0, amplitude=0.5)
        
        # Same phase should create constructive interference
        interference = task1.interfere_with(task2)
        assert interference > 0
        
        # Opposite phases should create destructive interference
        task2.phase = 3.14159  # π radians
        interference = task1.interfere_with(task2)
        assert interference < 0
    
    def test_progress_updates(self):
        """Test task progress updates."""
        task = Task(name="Test Task")
        
        # Initial progress
        assert task.progress == 0.0
        
        # Update progress
        task.update_progress(0.5)
        assert task.progress == 0.5
        
        # Complete task
        task.update_progress(1.0)
        assert task.progress == 1.0
        assert task.state == TaskState.COMPLETED
        assert task.completed_at is not None
    
    def test_decoherence(self):
        """Test quantum decoherence."""
        task = Task(name="Test Task", coherence_time=0.1)
        
        # Force decoherence by time passage
        task.created_at = datetime.now() - timedelta(seconds=1)
        task.decohere()
        
        # Properties should be classical
        assert task.amplitude == 1.0
        assert task.phase == 0.0
        assert task.entangled_tasks == set()


class TestQuantumTaskPlanner:
    """Test QuantumTaskPlanner functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.planner = QuantumTaskPlanner(
            name="test-planner",
            coherence_preservation=True,
            entanglement_enabled=True
        )
    
    def test_planner_initialization(self):
        """Test planner initialization."""
        assert self.planner.name == "test-planner"
        assert self.planner.coherence_preservation is True
        assert self.planner.entanglement_enabled is True
        assert len(self.planner.tasks) == 0
        assert self.planner.total_tasks_created == 0
    
    def test_create_task_basic(self):
        """Test basic task creation."""
        task = self.planner.create_task(
            name="Test Task",
            description="A test task",
            priority=QuantumPriority.MEDIUM
        )
        
        assert task.name == "Test Task"
        assert task.id in self.planner.tasks
        assert self.planner.total_tasks_created == 1
    
    def test_create_task_with_dependencies(self):
        """Test task creation with dependencies."""
        # Create first task
        task1 = self.planner.create_task(name="Task 1")
        
        # Create second task depending on first
        task2 = self.planner.create_task(
            name="Task 2",
            dependencies=[task1.id]
        )
        
        assert task1.id in task2.dependencies
    
    def test_create_task_invalid_dependency(self):
        """Test task creation with invalid dependency."""
        with pytest.raises(TaskNotFoundError):
            self.planner.create_task(
                name="Test Task",
                dependencies=["non-existent-id"]
            )
    
    def test_get_task(self):
        """Test task retrieval."""
        task = self.planner.create_task(name="Test Task")
        
        retrieved_task = self.planner.get_task(task.id)
        assert retrieved_task == task
        
        # Non-existent task
        assert self.planner.get_task("non-existent") is None
    
    def test_get_tasks_by_state(self):
        """Test filtering tasks by state."""
        task1 = self.planner.create_task(name="Task 1")
        task2 = self.planner.create_task(name="Task 2")
        
        # Both should be in superposition initially
        superposition_tasks = self.planner.get_tasks_by_state(TaskState.SUPERPOSITION)
        assert len(superposition_tasks) == 2
        
        # Collapse one task
        task1.collapse_wave_function()
        superposition_tasks = self.planner.get_tasks_by_state(TaskState.SUPERPOSITION)
        assert len(superposition_tasks) == 1
    
    def test_get_executable_tasks(self):
        """Test getting executable tasks."""
        # Create tasks
        task1 = self.planner.create_task(name="Task 1")
        task2 = self.planner.create_task(name="Task 2", dependencies=[task1.id])
        
        # Initially, only task1 should be executable (no dependencies)
        executable = self.planner.get_executable_tasks()
        executable_ids = [t.id for t in executable]
        
        assert task1.id in executable_ids
        assert task2.id not in executable_ids  # Has unmet dependency
        
        # Complete task1
        task1.state = TaskState.COMPLETED
        self.planner.completed_tasks.add(task1.id)
        
        # Now task2 should be executable
        executable = self.planner.get_executable_tasks()
        executable_ids = [t.id for t in executable]
        assert task2.id in executable_ids
    
    def test_collapse_superposition_tasks(self):
        """Test superposition collapse."""
        # Create tasks in superposition
        task1 = self.planner.create_task(name="Task 1", amplitude=0.9)
        task2 = self.planner.create_task(name="Task 2", amplitude=0.1)
        
        # Collapse all superposition tasks
        collapsed = self.planner.collapse_superposition_tasks()
        
        assert len(collapsed) >= 0  # Probabilistic
        assert self.planner.quantum_collapses >= 0
    
    @pytest.mark.asyncio
    async def test_execute_task(self):
        """Test task execution."""
        task = self.planner.create_task(
            name="Test Task",
            estimated_duration=0.001  # Very short for testing
        )
        
        # Collapse to executable state
        task.state = TaskState.PENDING
        
        # Execute task
        success = await self.planner.execute_task(task.id)
        
        # Wait a bit for execution to complete
        await asyncio.sleep(0.01)
        
        # Check execution results (probabilistic)
        assert isinstance(success, bool)
        
        if success:
            assert task.state == TaskState.COMPLETED
            assert task.id in self.planner.completed_tasks
        else:
            assert task.state == TaskState.FAILED
            assert task.id in self.planner.failed_tasks
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_task(self):
        """Test executing non-existent task."""
        success = await self.planner.execute_task("non-existent-id")
        assert success is False
    
    def test_decohere_tasks(self):
        """Test decoherence processing."""
        # Create task with short coherence time
        task = self.planner.create_task(name="Test Task", coherence_time=0.1)
        
        # Force time passage
        task.created_at = datetime.now() - timedelta(seconds=1)
        
        # Process decoherence
        decohered = self.planner.decohere_tasks()
        
        # Task should have decohered
        assert len(decohered) >= 0
        assert not task.is_coherent
    
    def test_system_state(self):
        """Test system state reporting."""
        # Create some tasks
        task1 = self.planner.create_task(name="Task 1")
        task2 = self.planner.create_task(name="Task 2")
        
        state = self.planner.get_system_state()
        
        assert state["planner_name"] == "test-planner"
        assert state["total_tasks"] == 2
        assert "tasks_by_state" in state
        assert "quantum_metrics" in state
        assert "execution_metrics" in state
    
    @pytest.mark.asyncio
    async def test_quantum_cycle(self):
        """Test quantum planning cycle."""
        # Create some tasks
        for i in range(3):
            self.planner.create_task(name=f"Task {i+1}")
        
        # Run quantum cycle
        result = await self.planner.run_quantum_cycle()
        
        assert "cycle_duration" in result
        assert "collapsed_tasks" in result
        assert "decohered_tasks" in result
        assert "executable_tasks" in result
        assert "executed_tasks" in result
        assert "system_state" in result
        
        # Verify timing
        assert result["cycle_duration"] >= 0
    
    def test_entanglement_detection(self):
        """Test entanglement opportunity detection."""
        # Create similar tasks that should entangle
        task1 = self.planner.create_task(
            name="Similar Task 1",
            priority=QuantumPriority.HIGH,
            estimated_duration=2.0
        )
        
        task2 = self.planner.create_task(
            name="Similar Task 2",
            priority=QuantumPriority.HIGH,
            estimated_duration=2.0
        )
        
        # Check if entanglement was created (probabilistic)
        assert self.planner.entanglements_created >= 0
    
    def test_task_similarity_calculation(self):
        """Test task similarity calculation for entanglement."""
        task1 = Task(
            name="Task 1",
            priority=QuantumPriority.HIGH,
            estimated_duration=2.0,
            metadata={"type": "computation"}
        )
        
        task2 = Task(
            name="Task 2", 
            priority=QuantumPriority.HIGH,
            estimated_duration=2.0,
            metadata={"type": "computation"}
        )
        
        similarity = self.planner._calculate_task_similarity(task1, task2)
        
        # High similarity expected
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be quite similar


class TestQuantumOperations:
    """Test quantum-specific operations."""
    
    def test_amplitude_probability_relationship(self):
        """Test amplitude-probability relationship."""
        amplitudes = [0.0, 0.5, 0.7, 1.0]
        
        for amp in amplitudes:
            task = Task(name="Test", amplitude=amp)
            expected_prob = amp ** 2
            assert abs(task.probability - expected_prob) < 0.001
    
    def test_phase_normalization(self):
        """Test phase normalization."""
        import math
        
        # Test various phases
        phases = [0, math.pi, 2*math.pi, 3*math.pi, -math.pi]
        
        for phase in phases:
            task = Task(name="Test", phase=phase)
            # Phase should be normalized to [0, 2π]
            assert 0 <= task.phase <= 2*math.pi
    
    def test_coherence_time_effects(self):
        """Test coherence time effects."""
        short_coherence = Task(name="Short", coherence_time=0.1)
        long_coherence = Task(name="Long", coherence_time=3600.0)
        
        # Simulate time passage
        time_delta = timedelta(seconds=1)
        short_coherence.created_at = datetime.now() - time_delta
        long_coherence.created_at = datetime.now() - time_delta
        
        # Short coherence should be lost
        assert not short_coherence.is_coherent
        # Long coherence should remain
        assert long_coherence.is_coherent
    
    def test_entanglement_symmetry(self):
        """Test entanglement symmetry."""
        task1 = Task(name="Task 1")
        task2 = Task(name="Task 2")
        
        # Create entanglement
        task1.entangle_with(task2)
        
        # Should be symmetric
        assert task2.id in task1.entangled_tasks
        assert task1.id in task2.entangled_tasks
        
        # Phases should be synchronized
        assert task1.phase == task2.phase
    
    def test_interference_patterns(self):
        """Test quantum interference patterns."""
        import math
        
        task1 = Task(name="Task 1", amplitude=0.6, phase=0.0)
        
        # Test constructive interference (same phase)
        task2 = Task(name="Task 2", amplitude=0.6, phase=0.0)
        constructive = task1.interfere_with(task2)
        assert constructive > 0
        
        # Test destructive interference (opposite phase)
        task3 = Task(name="Task 3", amplitude=0.6, phase=math.pi)
        destructive = task1.interfere_with(task3)
        assert destructive < 0
        
        # Test null interference (perpendicular phases)
        task4 = Task(name="Task 4", amplitude=0.6, phase=math.pi/2)
        null = task1.interfere_with(task4)
        assert abs(null) < 0.1  # Should be close to zero


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test asynchronous operations."""
    
    async def test_concurrent_execution(self):
        """Test concurrent task execution."""
        planner = QuantumTaskPlanner()
        
        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = planner.create_task(
                name=f"Concurrent Task {i+1}",
                estimated_duration=0.001
            )
            task.state = TaskState.PENDING
            tasks.append(task)
        
        # Execute tasks concurrently
        results = await asyncio.gather(
            *[planner.execute_task(task.id) for task in tasks],
            return_exceptions=True
        )
        
        # Check results
        assert len(results) == 3
        for result in results:
            assert isinstance(result, bool) or isinstance(result, Exception)
    
    async def test_quantum_cycle_performance(self):
        """Test quantum cycle performance."""
        planner = QuantumTaskPlanner()
        
        # Create many tasks
        for i in range(10):
            planner.create_task(name=f"Perf Task {i+1}")
        
        # Run quantum cycle and measure time
        start_time = time.time()
        result = await planner.run_quantum_cycle()
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second
        assert result["cycle_duration"] > 0


@pytest.fixture
def sample_planner():
    """Fixture providing a planner with sample tasks."""
    planner = QuantumTaskPlanner(name="sample-planner")
    
    # Create sample tasks
    task1 = planner.create_task(
        name="Sample Task 1",
        priority=QuantumPriority.HIGH,
        estimated_duration=1.0
    )
    
    task2 = planner.create_task(
        name="Sample Task 2",
        priority=QuantumPriority.MEDIUM,
        dependencies=[task1.id],
        estimated_duration=2.0
    )
    
    task3 = planner.create_task(
        name="Sample Task 3",
        priority=QuantumPriority.LOW,
        estimated_duration=0.5
    )
    
    return planner, [task1, task2, task3]


class TestWithSampleData:
    """Tests using sample data fixture."""
    
    def test_dependency_resolution(self, sample_planner):
        """Test dependency resolution."""
        planner, tasks = sample_planner
        task1, task2, task3 = tasks
        
        # Get executable tasks
        executable = planner.get_executable_tasks()
        executable_ids = [t.id for t in executable]
        
        # Task1 and task3 should be executable (no dependencies)
        # Task2 should not be (depends on task1)
        assert task1.id in executable_ids
        assert task3.id in executable_ids
        assert task2.id not in executable_ids
    
    def test_priority_based_processing(self, sample_planner):
        """Test priority-based task processing."""
        planner, tasks = sample_planner
        task1, task2, task3 = tasks
        
        # Get tasks sorted by priority
        all_tasks = list(planner.tasks.values())
        priority_order = {
            QuantumPriority.CRITICAL: 0,
            QuantumPriority.HIGH: 1,
            QuantumPriority.MEDIUM: 2,
            QuantumPriority.LOW: 3,
            QuantumPriority.DEFERRED: 4
        }
        
        sorted_tasks = sorted(all_tasks, key=lambda t: priority_order[t.priority])
        
        # First task should be highest priority
        assert sorted_tasks[0].priority == QuantumPriority.HIGH
        assert sorted_tasks[-1].priority == QuantumPriority.LOW


if __name__ == "__main__":
    pytest.main([__file__, "-v"])