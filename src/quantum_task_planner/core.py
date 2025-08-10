"""Core quantum-inspired task planning implementation."""

import asyncio
import time
import uuid
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
import logging
from datetime import datetime, timedelta

from .exceptions import QuantumTaskPlannerError, TaskNotFoundError, InvalidTaskStateError
from .exceptions import QuantumCoherenceError, EntanglementError, handle_async_exception
from .validation import TaskValidator, ValidationResult


class TaskState(Enum):
    """Quantum-inspired task states."""
    SUPERPOSITION = "superposition"  # Task exists in multiple potential states
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ENTANGLED = "entangled"  # Task coupled with other tasks


class QuantumPriority(Enum):
    """Quantum priority levels using wave function collapse."""
    CRITICAL = "critical"  # High probability amplitude
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"
    DEFERRED = "deferred"  # Low probability amplitude


@dataclass
class Task:
    """Quantum-inspired task with superposition and entanglement properties."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    priority: QuantumPriority = QuantumPriority.MEDIUM
    state: TaskState = TaskState.SUPERPOSITION
    
    # Quantum properties
    amplitude: float = 0.5  # Probability amplitude (0-1)
    phase: float = 0.0  # Phase angle in radians
    entangled_tasks: Set[str] = field(default_factory=set)
    coherence_time: float = 3600.0  # Seconds before decoherence
    
    # Traditional properties
    created_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    estimated_duration: float = 0.0  # Hours
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    
    def __post_init__(self):
        """Initialize quantum properties."""
        if not self.name:
            self.name = f"Task-{self.id[:8]}"
        
        # Set initial amplitude based on priority
        priority_amplitudes = {
            QuantumPriority.CRITICAL: 0.9,
            QuantumPriority.HIGH: 0.7,
            QuantumPriority.MEDIUM: 0.5,
            QuantumPriority.LOW: 0.3,
            QuantumPriority.DEFERRED: 0.1
        }
        self.amplitude = priority_amplitudes.get(self.priority, 0.5)
        
        # Random initial phase
        self.phase = random.uniform(0, 2 * math.pi)
    
    @property
    def probability(self) -> float:
        """Calculate execution probability from amplitude."""
        return self.amplitude ** 2
    
    @property
    def is_coherent(self) -> bool:
        """Check if task maintains quantum coherence."""
        if self.created_at:
            elapsed = (datetime.now() - self.created_at).total_seconds()
            return elapsed < self.coherence_time
        return True
    
    @property
    def is_executable(self) -> bool:
        """Check if task can be executed (dependencies met)."""
        return self.state in [TaskState.PENDING, TaskState.SUPERPOSITION]
    
    def collapse_wave_function(self) -> TaskState:
        """Collapse superposition to definite state."""
        if self.state != TaskState.SUPERPOSITION:
            return self.state
        
        # Probability-based collapse
        if random.random() < self.probability:
            self.state = TaskState.PENDING
            logging.info(f"Task {self.name} collapsed to PENDING (p={self.probability:.2f})")
        else:
            self.state = TaskState.FAILED  # Use FAILED instead of non-existent DEFERRED
            logging.info(f"Task {self.name} collapsed to FAILED (p={self.probability:.2f})")
        
        return self.state
    
    def entangle_with(self, other_task: 'Task') -> None:
        """Create quantum entanglement between tasks."""
        self.entangled_tasks.add(other_task.id)
        other_task.entangled_tasks.add(self.id)
        self.state = TaskState.ENTANGLED
        other_task.state = TaskState.ENTANGLED
        
        # Synchronize phases for entanglement
        shared_phase = (self.phase + other_task.phase) / 2
        self.phase = shared_phase
        other_task.phase = shared_phase
        
        logging.info(f"Tasks {self.name} and {other_task.name} are now entangled")
    
    def decohere(self) -> None:
        """Lose quantum properties due to environmental interaction."""
        if self.is_coherent:
            return
        
        # Transition to classical state
        if self.state == TaskState.SUPERPOSITION:
            self.collapse_wave_function()
        
        # Break entanglements
        self.entangled_tasks.clear()
        self.amplitude = 1.0  # Classical certainty
        self.phase = 0.0
        
        logging.info(f"Task {self.name} has decohered")
    
    def interfere_with(self, other_task: 'Task') -> float:
        """Calculate quantum interference with another task."""
        if not self.is_coherent or not other_task.is_coherent:
            return 0.0
        
        # Constructive/destructive interference based on phase difference
        phase_diff = abs(self.phase - other_task.phase)
        interference = math.cos(phase_diff)
        
        return self.amplitude * other_task.amplitude * interference
    
    def update_progress(self, progress: float) -> None:
        """Update task progress with quantum effects."""
        self.progress = max(0.0, min(1.0, progress))
        
        # Progress affects amplitude (uncertainty principle)
        uncertainty = 1.0 - self.progress
        self.amplitude *= uncertainty
        
        if self.progress >= 1.0:
            self.state = TaskState.COMPLETED
            self.completed_at = datetime.now()


class QuantumTaskPlanner:
    """Main quantum-inspired task planning system."""
    
    def __init__(
        self,
        name: str = "QuantumPlanner",
        coherence_preservation: bool = True,
        entanglement_enabled: bool = True,
        interference_threshold: float = 0.3
    ):
        """Initialize quantum task planner.
        
        Args:
            name: Planner instance name
            coherence_preservation: Enable quantum coherence preservation
            entanglement_enabled: Allow task entanglement
            interference_threshold: Minimum interference for task coupling
        """
        self.name = name
        self.coherence_preservation = coherence_preservation
        self.entanglement_enabled = entanglement_enabled
        self.interference_threshold = interference_threshold
        
        # Task storage
        self.tasks: Dict[str, Task] = {}
        self.entanglement_map: Dict[str, Set[str]] = {}
        
        # Execution state
        self.running_tasks: Set[str] = set()
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
        # Metrics
        self.total_tasks_created = 0
        self.total_tasks_completed = 0
        self.quantum_collapses = 0
        self.entanglements_created = 0
        
        # Validation
        self.validator = TaskValidator()
        
        # Setup logging
        self.logger = logging.getLogger(f"quantum_planner.{name}")
        
    def create_task(
        self,
        name: str,
        description: str = "",
        priority: QuantumPriority = QuantumPriority.MEDIUM,
        estimated_duration: float = 1.0,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a new quantum task with validation.
        
        Args:
            name: Task name
            description: Task description
            priority: Quantum priority level
            estimated_duration: Estimated duration in hours
            dependencies: List of dependency task IDs
            metadata: Additional task metadata
            
        Returns:
            Created Task instance
            
        Raises:
            ValidationError: If task parameters are invalid
            QuantumTaskPlannerError: If task creation fails
        """
        # Validate input parameters
        validation_result = self.validator.validate_task_creation(
            name=name,
            description=description,
            priority=priority,
            estimated_duration=estimated_duration,
            dependencies=dependencies,
            metadata=metadata
        )
        
        if not validation_result.is_valid:
            error_msg = f"Task validation failed: {', '.join(validation_result.errors)}"
            self.logger.error(error_msg)
            raise QuantumTaskPlannerError(error_msg, error_code="VALIDATION_FAILED")
        
        # Log warnings
        for warning in validation_result.warnings:
            self.logger.warning(f"Task creation warning: {warning}")
        
        # Use sanitized data
        sanitized_data = validation_result.sanitized_data
        
        task = Task(
            name=sanitized_data.get('name', name),
            description=sanitized_data.get('description', description),
            priority=sanitized_data.get('priority', priority),
            estimated_duration=sanitized_data.get('estimated_duration', estimated_duration),
            dependencies=set(sanitized_data.get('dependencies', dependencies or [])),
            metadata=sanitized_data.get('metadata', metadata or {})
        )
        
        # Validate dependencies exist
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                raise TaskNotFoundError(dep_id)
        
        self.tasks[task.id] = task
        self.total_tasks_created += 1
        
        # Check for potential entanglements
        if self.entanglement_enabled:
            try:
                self._detect_entanglement_opportunities(task)
            except Exception as e:
                self.logger.warning(f"Entanglement detection failed for task {task.id}: {e}")
        
        self.logger.info(f"Created quantum task: {task.name} (ID: {task.id})")
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_tasks_by_state(self, state: TaskState) -> List[Task]:
        """Get all tasks in specified state."""
        return [task for task in self.tasks.values() if task.state == state]
    
    def get_executable_tasks(self) -> List[Task]:
        """Get tasks ready for execution (dependencies satisfied)."""
        executable = []
        
        for task in self.tasks.values():
            if not task.is_executable:
                continue
                
            # Check dependencies
            if task.dependencies:
                deps_satisfied = all(
                    dep_id in self.completed_tasks 
                    for dep_id in task.dependencies
                )
                if not deps_satisfied:
                    continue
            
            executable.append(task)
        
        return executable
    
    def collapse_superposition_tasks(self) -> List[Task]:
        """Collapse all tasks in superposition to definite states."""
        collapsed = []
        
        for task in self.get_tasks_by_state(TaskState.SUPERPOSITION):
            old_state = task.state
            new_state = task.collapse_wave_function()
            
            if old_state != new_state:
                collapsed.append(task)
                self.quantum_collapses += 1
        
        return collapsed
    
    def _detect_entanglement_opportunities(self, new_task: Task) -> None:
        """Detect opportunities for quantum entanglement."""
        if not self.entanglement_enabled:
            return
        
        for existing_task in self.tasks.values():
            if existing_task.id == new_task.id:
                continue
            
            # Calculate interference
            interference = new_task.interfere_with(existing_task)
            
            if abs(interference) > self.interference_threshold:
                # Strong interference suggests entanglement opportunity
                similarity_score = self._calculate_task_similarity(new_task, existing_task)
                
                if similarity_score > 0.5:  # Threshold for entanglement
                    new_task.entangle_with(existing_task)
                    self.entanglements_created += 1
                    
                    # Update entanglement map
                    if new_task.id not in self.entanglement_map:
                        self.entanglement_map[new_task.id] = set()
                    if existing_task.id not in self.entanglement_map:
                        self.entanglement_map[existing_task.id] = set()
                    
                    self.entanglement_map[new_task.id].add(existing_task.id)
                    self.entanglement_map[existing_task.id].add(new_task.id)
    
    def _calculate_task_similarity(self, task1: Task, task2: Task) -> float:
        """Calculate similarity between two tasks for entanglement detection."""
        similarity = 0.0
        
        # Priority similarity
        priority_values = {p: i for i, p in enumerate(QuantumPriority)}
        priority_diff = abs(priority_values[task1.priority] - priority_values[task2.priority])
        priority_sim = 1.0 - (priority_diff / len(QuantumPriority))
        similarity += 0.3 * priority_sim
        
        # Duration similarity
        if task1.estimated_duration > 0 and task2.estimated_duration > 0:
            duration_ratio = min(task1.estimated_duration, task2.estimated_duration) / \
                           max(task1.estimated_duration, task2.estimated_duration)
            similarity += 0.2 * duration_ratio
        
        # Metadata similarity (simple keyword matching)
        if task1.metadata and task2.metadata:
            common_keys = set(task1.metadata.keys()) & set(task2.metadata.keys())
            if common_keys:
                key_similarity = len(common_keys) / max(len(task1.metadata), len(task2.metadata))
                similarity += 0.3 * key_similarity
        
        # Phase correlation (quantum property)
        phase_correlation = abs(math.cos(task1.phase - task2.phase))
        similarity += 0.2 * phase_correlation
        
        return similarity
    
    async def execute_task(self, task_id: str) -> bool:
        """Execute a quantum task with superposition collapse.
        
        Args:
            task_id: ID of task to execute
            
        Returns:
            True if task completed successfully
        """
        task = self.get_task(task_id)
        if not task or not task.is_executable:
            return False
        
        # Collapse superposition if needed
        if task.state == TaskState.SUPERPOSITION:
            collapsed_state = task.collapse_wave_function()
            if collapsed_state != TaskState.PENDING:
                return False
        
        # Start execution
        task.state = TaskState.RUNNING
        task.started_at = datetime.now()
        self.running_tasks.add(task_id)
        
        self.logger.info(f"Starting execution of task: {task.name}")
        
        try:
            # Simulate task execution with quantum uncertainty
            execution_time = task.estimated_duration * 3600  # Convert to seconds
            uncertainty_factor = 1.0 - task.amplitude  # Higher amplitude = more predictable
            actual_time = execution_time * (1 + random.uniform(-uncertainty_factor, uncertainty_factor))
            
            # Execute with progress updates
            steps = 10
            step_time = actual_time / steps
            
            for step in range(steps):
                await asyncio.sleep(step_time / 10)  # Speed up for demo
                progress = (step + 1) / steps
                task.update_progress(progress)
                
                # Handle entangled tasks
                if task.entangled_tasks:
                    await self._propagate_entanglement_effects(task)
            
            # Complete task
            task.state = TaskState.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 1.0
            
            self.running_tasks.discard(task_id)
            self.completed_tasks.add(task_id)
            self.total_tasks_completed += 1
            
            self.logger.info(f"Completed task: {task.name}")
            return True
            
        except Exception as e:
            # Task failed
            task.state = TaskState.FAILED
            self.running_tasks.discard(task_id)
            self.failed_tasks.add(task_id)
            
            self.logger.error(f"Task {task.name} failed: {str(e)}")
            return False
    
    async def _propagate_entanglement_effects(self, task: Task) -> None:
        """Propagate quantum effects to entangled tasks."""
        for entangled_id in task.entangled_tasks:
            entangled_task = self.get_task(entangled_id)
            if entangled_task and entangled_task.state != TaskState.COMPLETED:
                # Synchronize progress and state changes
                if entangled_task.state == TaskState.PENDING:
                    entangled_task.state = TaskState.RUNNING
                    entangled_task.started_at = datetime.now()
                
                # Correlate progress
                progress_correlation = 0.5  # Partial correlation
                entangled_task.update_progress(task.progress * progress_correlation)
    
    def decohere_tasks(self) -> List[Task]:
        """Process quantum decoherence for all tasks."""
        decohered = []
        
        for task in self.tasks.values():
            if not task.is_coherent:
                task.decohere()
                decohered.append(task)
        
        return decohered
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get complete quantum system state."""
        return {
            "planner_name": self.name,
            "total_tasks": len(self.tasks),
            "tasks_by_state": {
                state.value: len(self.get_tasks_by_state(state))
                for state in TaskState
            },
            "quantum_metrics": {
                "total_collapses": self.quantum_collapses,
                "active_entanglements": len(self.entanglement_map),
                "coherent_tasks": sum(1 for t in self.tasks.values() if t.is_coherent),
                "average_amplitude": sum(t.amplitude for t in self.tasks.values()) / len(self.tasks) if self.tasks else 0
            },
            "execution_metrics": {
                "completed_tasks": self.total_tasks_completed,
                "success_rate": self.total_tasks_completed / self.total_tasks_created if self.total_tasks_created > 0 else 0,
                "currently_running": len(self.running_tasks)
            }
        }
    
    async def run_quantum_cycle(self) -> Dict[str, Any]:
        """Run one complete quantum planning cycle."""
        cycle_start = time.time()
        
        # 1. Collapse superposition tasks
        collapsed = self.collapse_superposition_tasks()
        
        # 2. Process decoherence
        decohered = self.decohere_tasks()
        
        # 3. Get executable tasks
        executable = self.get_executable_tasks()
        
        # 4. Execute highest probability tasks
        executed = []
        for task in sorted(executable, key=lambda t: t.probability, reverse=True)[:3]:
            if task.state == TaskState.PENDING:
                success = await self.execute_task(task.id)
                if success:
                    executed.append(task)
        
        cycle_time = time.time() - cycle_start
        
        return {
            "cycle_duration": cycle_time,
            "collapsed_tasks": len(collapsed),
            "decohered_tasks": len(decohered),
            "executable_tasks": len(executable),
            "executed_tasks": len(executed),
            "system_state": self.get_system_state()
        }