"""Input validation and sanitization for quantum task planner."""

import re
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set, Callable
from dataclasses import dataclass

from .core import Task, TaskState, QuantumPriority
from .exceptions import ValidationError, DependencyError


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Optional[Dict[str, Any]] = None
    
    def __bool__(self) -> bool:
        """Allow boolean evaluation of validation result."""
        return self.is_valid
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)


class TaskValidator:
    """Validator for quantum tasks."""
    
    # Validation constraints
    MIN_NAME_LENGTH = 1
    MAX_NAME_LENGTH = 200
    MAX_DESCRIPTION_LENGTH = 2000
    MIN_DURATION = 0.0
    MAX_DURATION = 8760.0  # 1 year in hours
    MIN_AMPLITUDE = 0.0
    MAX_AMPLITUDE = 1.0
    MIN_PHASE = 0.0
    MAX_PHASE = 2 * math.pi
    MIN_COHERENCE_TIME = 1.0  # 1 second
    MAX_COHERENCE_TIME = 86400.0  # 24 hours
    MAX_DEPENDENCIES = 50
    MAX_METADATA_SIZE = 10000  # characters in JSON
    
    # Regex patterns
    NAME_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-_\.]+$')
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
    
    def validate_task_creation(
        self,
        name: str,
        description: str = "",
        priority: Union[str, QuantumPriority] = QuantumPriority.MEDIUM,
        estimated_duration: float = 1.0,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate task creation parameters."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], sanitized_data={})
        
        # Validate name
        name_result = self._validate_name(name)
        if not name_result.is_valid:
            result.errors.extend(name_result.errors)
            result.is_valid = False
        else:
            result.sanitized_data['name'] = name_result.sanitized_data['name']
        
        # Validate description
        desc_result = self._validate_description(description)
        if not desc_result.is_valid:
            result.errors.extend(desc_result.errors)
            result.is_valid = False
        else:
            result.sanitized_data['description'] = desc_result.sanitized_data['description']
            result.warnings.extend(desc_result.warnings)
        
        # Validate priority
        priority_result = self._validate_priority(priority)
        if not priority_result.is_valid:
            result.errors.extend(priority_result.errors)
            result.is_valid = False
        else:
            result.sanitized_data['priority'] = priority_result.sanitized_data['priority']
        
        # Validate duration
        duration_result = self._validate_duration(estimated_duration)
        if not duration_result.is_valid:
            result.errors.extend(duration_result.errors)
            result.is_valid = False
        else:
            result.sanitized_data['estimated_duration'] = duration_result.sanitized_data['duration']
            result.warnings.extend(duration_result.warnings)
        
        # Validate dependencies
        if dependencies is not None:
            deps_result = self._validate_dependencies(dependencies)
            if not deps_result.is_valid:
                result.errors.extend(deps_result.errors)
                result.is_valid = False
            else:
                result.sanitized_data['dependencies'] = deps_result.sanitized_data['dependencies']
                result.warnings.extend(deps_result.warnings)
        
        # Validate metadata
        if metadata is not None:
            meta_result = self._validate_metadata(metadata)
            if not meta_result.is_valid:
                result.errors.extend(meta_result.errors)
                result.is_valid = False
            else:
                result.sanitized_data['metadata'] = meta_result.sanitized_data['metadata']
                result.warnings.extend(meta_result.warnings)
        
        return result
    
    def validate_quantum_properties(
        self,
        amplitude: float,
        phase: float,
        coherence_time: float
    ) -> ValidationResult:
        """Validate quantum properties of a task."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], sanitized_data={})
        
        # Validate amplitude
        if not isinstance(amplitude, (int, float)):
            result.add_error(f"Amplitude must be numeric, got {type(amplitude).__name__}")
        elif not (self.MIN_AMPLITUDE <= amplitude <= self.MAX_AMPLITUDE):
            result.add_error(f"Amplitude must be between {self.MIN_AMPLITUDE} and {self.MAX_AMPLITUDE}, got {amplitude}")
        else:
            result.sanitized_data['amplitude'] = float(amplitude)
            if amplitude < 0.1:
                result.add_warning("Very low amplitude may result in task being rarely executed")
            elif amplitude > 0.9:
                result.add_warning("Very high amplitude may reduce quantum effects")
        
        # Validate phase
        if not isinstance(phase, (int, float)):
            result.add_error(f"Phase must be numeric, got {type(phase).__name__}")
        else:
            # Normalize phase to [0, 2π] range
            normalized_phase = phase % (2 * math.pi)
            result.sanitized_data['phase'] = normalized_phase
            
            if abs(phase - normalized_phase) > 0.01:
                result.add_warning(f"Phase normalized from {phase} to {normalized_phase}")
        
        # Validate coherence time
        if not isinstance(coherence_time, (int, float)):
            result.add_error(f"Coherence time must be numeric, got {type(coherence_time).__name__}")
        elif not (self.MIN_COHERENCE_TIME <= coherence_time <= self.MAX_COHERENCE_TIME):
            result.add_error(f"Coherence time must be between {self.MIN_COHERENCE_TIME} and {self.MAX_COHERENCE_TIME} seconds")
        else:
            result.sanitized_data['coherence_time'] = float(coherence_time)
            if coherence_time < 60:
                result.add_warning("Very short coherence time may cause frequent decoherence")
            elif coherence_time > 3600:
                result.add_warning("Very long coherence time may reduce quantum dynamics")
        
        return result
    
    def validate_task_state_transition(
        self,
        current_state: TaskState,
        target_state: TaskState,
        task_id: str
    ) -> ValidationResult:
        """Validate task state transitions."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Define valid state transitions
        valid_transitions = {
            TaskState.SUPERPOSITION: [TaskState.PENDING, TaskState.DEFERRED, TaskState.ENTANGLED],
            TaskState.PENDING: [TaskState.RUNNING, TaskState.SUPERPOSITION, TaskState.ENTANGLED, TaskState.FAILED],
            TaskState.RUNNING: [TaskState.COMPLETED, TaskState.FAILED, TaskState.PENDING],
            TaskState.COMPLETED: [],  # Terminal state
            TaskState.FAILED: [TaskState.PENDING, TaskState.SUPERPOSITION],  # Allow retry
            TaskState.ENTANGLED: [TaskState.PENDING, TaskState.SUPERPOSITION, TaskState.RUNNING]
        }
        
        if target_state not in valid_transitions.get(current_state, []):
            result.add_error(
                f"Invalid state transition for task {task_id}: "
                f"{current_state.value} -> {target_state.value}"
            )
        
        # Add warnings for potentially problematic transitions
        if current_state == TaskState.COMPLETED and target_state != TaskState.COMPLETED:
            result.add_warning("Transitioning completed task back to active state")
        
        if current_state == TaskState.FAILED and target_state == TaskState.RUNNING:
            result.add_warning("Task transitioning directly from failed to running without pending state")
        
        return result
    
    def validate_entanglement(
        self,
        task1: Task,
        task2: Task,
        interference_threshold: float = 0.1
    ) -> ValidationResult:
        """Validate task entanglement."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check if tasks can be entangled
        if task1.id == task2.id:
            result.add_error("Cannot entangle task with itself")
        
        # Check current states
        if task1.state == TaskState.COMPLETED or task2.state == TaskState.COMPLETED:
            result.add_error("Cannot entangle completed tasks")
        
        if task1.state == TaskState.FAILED or task2.state == TaskState.FAILED:
            result.add_error("Cannot entangle failed tasks")
        
        # Check coherence
        if not task1.is_coherent:
            result.add_error(f"Task {task1.name} has lost coherence, cannot entangle")
        
        if not task2.is_coherent:
            result.add_error(f"Task {task2.name} has lost coherence, cannot entangle")
        
        # Check interference strength
        interference = task1.interfere_with(task2)
        if abs(interference) < interference_threshold:
            result.add_warning(
                f"Weak interference ({interference:.3f}) may result in unstable entanglement"
            )
        
        # Check for circular entanglements
        if task2.id in task1.entangled_tasks:
            result.add_error("Tasks are already entangled")
        
        # Check priority compatibility
        priority_values = {p: i for i, p in enumerate(QuantumPriority)}
        priority_diff = abs(priority_values[task1.priority] - priority_values[task2.priority])
        if priority_diff > 2:
            result.add_warning("Large priority difference may cause entanglement instability")
        
        return result
    
    def _validate_name(self, name: str) -> ValidationResult:
        """Validate task name."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], sanitized_data={})
        
        if not isinstance(name, str):
            result.add_error(f"Name must be string, got {type(name).__name__}")
            return result
        
        # Trim whitespace
        sanitized_name = name.strip()
        
        if len(sanitized_name) < self.MIN_NAME_LENGTH:
            result.add_error(f"Name too short (minimum {self.MIN_NAME_LENGTH} characters)")
        elif len(sanitized_name) > self.MAX_NAME_LENGTH:
            result.add_error(f"Name too long (maximum {self.MAX_NAME_LENGTH} characters)")
        elif not self.NAME_PATTERN.match(sanitized_name):
            result.add_error("Name contains invalid characters (only letters, numbers, spaces, hyphens, underscores, dots allowed)")
        else:
            result.sanitized_data['name'] = sanitized_name
            
            if name != sanitized_name:
                result.add_warning("Name whitespace has been trimmed")
        
        return result
    
    def _validate_description(self, description: str) -> ValidationResult:
        """Validate task description."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], sanitized_data={})
        
        if not isinstance(description, str):
            result.add_error(f"Description must be string, got {type(description).__name__}")
            return result
        
        sanitized_desc = description.strip()
        
        if len(sanitized_desc) > self.MAX_DESCRIPTION_LENGTH:
            result.add_error(f"Description too long (maximum {self.MAX_DESCRIPTION_LENGTH} characters)")
        else:
            result.sanitized_data['description'] = sanitized_desc
            
            if description != sanitized_desc:
                result.add_warning("Description whitespace has been trimmed")
            
            if len(sanitized_desc) == 0:
                result.add_warning("Empty description may make task harder to understand")
        
        return result
    
    def _validate_priority(self, priority: Union[str, QuantumPriority]) -> ValidationResult:
        """Validate task priority."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], sanitized_data={})
        
        if isinstance(priority, str):
            try:
                priority_enum = QuantumPriority(priority.lower())
                result.sanitized_data['priority'] = priority_enum
            except ValueError:
                valid_values = [p.value for p in QuantumPriority]
                result.add_error(f"Invalid priority '{priority}'. Valid values: {valid_values}")
        elif isinstance(priority, QuantumPriority):
            result.sanitized_data['priority'] = priority
        else:
            result.add_error(f"Priority must be string or QuantumPriority enum, got {type(priority).__name__}")
        
        return result
    
    def _validate_duration(self, duration: float) -> ValidationResult:
        """Validate estimated duration."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], sanitized_data={})
        
        if not isinstance(duration, (int, float)):
            result.add_error(f"Duration must be numeric, got {type(duration).__name__}")
        elif duration < self.MIN_DURATION:
            result.add_error(f"Duration cannot be negative")
        elif duration > self.MAX_DURATION:
            result.add_error(f"Duration too large (maximum {self.MAX_DURATION} hours)")
        else:
            result.sanitized_data['duration'] = float(duration)
            
            if duration == 0:
                result.add_warning("Zero duration may cause scheduling issues")
            elif duration > 40:  # More than a work week
                result.add_warning("Very long duration task may benefit from decomposition")
        
        return result
    
    def _validate_dependencies(self, dependencies: List[str]) -> ValidationResult:
        """Validate task dependencies."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], sanitized_data={})
        
        if not isinstance(dependencies, list):
            result.add_error(f"Dependencies must be list, got {type(dependencies).__name__}")
            return result
        
        if len(dependencies) > self.MAX_DEPENDENCIES:
            result.add_error(f"Too many dependencies (maximum {self.MAX_DEPENDENCIES})")
            return result
        
        sanitized_deps = []
        seen_deps = set()
        
        for i, dep in enumerate(dependencies):
            if not isinstance(dep, str):
                result.add_error(f"Dependency {i} must be string, got {type(dep).__name__}")
                continue
            
            dep = dep.strip()
            
            if not dep:
                result.add_warning(f"Empty dependency at index {i} ignored")
                continue
            
            if not self.UUID_PATTERN.match(dep):
                result.add_error(f"Dependency {i} is not a valid UUID: {dep}")
                continue
            
            if dep in seen_deps:
                result.add_warning(f"Duplicate dependency removed: {dep}")
                continue
            
            sanitized_deps.append(dep)
            seen_deps.add(dep)
        
        result.sanitized_data['dependencies'] = sanitized_deps
        
        if len(sanitized_deps) > 10:
            result.add_warning("Large number of dependencies may complicate scheduling")
        
        return result
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> ValidationResult:
        """Validate task metadata."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], sanitized_data={})
        
        if not isinstance(metadata, dict):
            result.add_error(f"Metadata must be dictionary, got {type(metadata).__name__}")
            return result
        
        # Estimate JSON size
        import json
        try:
            metadata_json = json.dumps(metadata, default=str)
            if len(metadata_json) > self.MAX_METADATA_SIZE:
                result.add_error(f"Metadata too large (maximum {self.MAX_METADATA_SIZE} characters)")
                return result
        except (TypeError, ValueError) as e:
            result.add_error(f"Metadata not serializable: {e}")
            return result
        
        # Sanitize keys and values
        sanitized_metadata = {}
        
        for key, value in metadata.items():
            if not isinstance(key, str):
                result.add_warning(f"Non-string metadata key {key} converted to string")
                key = str(key)
            
            # Sanitize key
            key = key.strip()[:100]  # Limit key length
            
            if not key:
                result.add_warning("Empty metadata key ignored")
                continue
            
            # Basic value sanitization
            if isinstance(value, str) and len(value) > 1000:
                result.add_warning(f"Metadata value for key '{key}' truncated (too long)")
                value = value[:1000] + "..."
            
            sanitized_metadata[key] = value
        
        result.sanitized_data['metadata'] = sanitized_metadata
        
        return result


class DependencyValidator:
    """Validator for task dependencies and scheduling constraints."""
    
    def __init__(self, tasks: Dict[str, Task]):
        """Initialize with current task collection."""
        self.tasks = tasks
    
    def validate_dependency_graph(self) -> ValidationResult:
        """Validate entire dependency graph for cycles and validity."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check for circular dependencies
        cycles = self._find_circular_dependencies()
        for cycle in cycles:
            result.add_error(f"Circular dependency detected: {' -> '.join(cycle)}")
        
        # Check for missing dependencies
        missing_deps = self._find_missing_dependencies()
        for task_id, missing in missing_deps.items():
            task_name = self.tasks[task_id].name
            result.add_error(f"Task '{task_name}' has missing dependencies: {missing}")
        
        # Check for orphaned tasks
        orphaned = self._find_orphaned_tasks()
        if orphaned:
            task_names = [self.tasks[tid].name for tid in orphaned]
            result.add_warning(f"Orphaned tasks (no dependencies, not depended on): {task_names}")
        
        return result
    
    def validate_task_dependencies(self, task_id: str) -> ValidationResult:
        """Validate dependencies for a specific task."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if task_id not in self.tasks:
            raise TaskNotFoundError(task_id)
        
        task = self.tasks[task_id]
        
        # Check each dependency
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                result.add_error(f"Dependency {dep_id} does not exist")
                continue
            
            dep_task = self.tasks[dep_id]
            
            # Check dependency state
            if dep_task.state == TaskState.FAILED:
                result.add_error(f"Dependency '{dep_task.name}' has failed")
            
            # Check for potential circular dependency
            if self._has_path(dep_id, task_id):
                result.add_error(f"Circular dependency through '{dep_task.name}'")
            
            # Check temporal constraints
            if hasattr(task, 'due_date') and hasattr(dep_task, 'due_date'):
                if (task.due_date and dep_task.due_date and 
                    task.due_date <= dep_task.due_date):
                    result.add_warning(
                        f"Task due date conflicts with dependency '{dep_task.name}'"
                    )
        
        return result
    
    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find all circular dependencies in the task graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(task_id: str, path: List[str]):
            if task_id in rec_stack:
                # Found a cycle
                cycle_start = path.index(task_id)
                cycle = path[cycle_start:] + [task_id]
                cycles.append(cycle)
                return
            
            if task_id in visited or task_id not in self.tasks:
                return
            
            visited.add(task_id)
            rec_stack.add(task_id)
            path.append(task_id)
            
            task = self.tasks[task_id]
            for dep_id in task.dependencies:
                dfs(dep_id, path.copy())
            
            rec_stack.remove(task_id)
        
        for task_id in self.tasks:
            if task_id not in visited:
                dfs(task_id, [])
        
        return cycles
    
    def _find_missing_dependencies(self) -> Dict[str, List[str]]:
        """Find tasks with dependencies that don't exist."""
        missing_deps = {}
        
        for task_id, task in self.tasks.items():
            missing = [dep_id for dep_id in task.dependencies if dep_id not in self.tasks]
            if missing:
                missing_deps[task_id] = missing
        
        return missing_deps
    
    def _find_orphaned_tasks(self) -> List[str]:
        """Find tasks with no dependencies and that aren't depended on."""
        depended_on = set()
        
        # Find all tasks that are depended on
        for task in self.tasks.values():
            depended_on.update(task.dependencies)
        
        # Find orphaned tasks
        orphaned = []
        for task_id, task in self.tasks.items():
            if not task.dependencies and task_id not in depended_on:
                orphaned.append(task_id)
        
        return orphaned
    
    def _has_path(self, start_id: str, target_id: str) -> bool:
        """Check if there's a path from start_id to target_id."""
        if start_id == target_id:
            return True
        
        visited = set()
        stack = [start_id]
        
        while stack:
            current_id = stack.pop()
            if current_id in visited or current_id not in self.tasks:
                continue
            
            if current_id == target_id:
                return True
            
            visited.add(current_id)
            task = self.tasks[current_id]
            stack.extend(task.dependencies)
        
        return False


def sanitize_input(input_data: Any, expected_type: type) -> Any:
    """Generic input sanitization function."""
    if input_data is None:
        return None
    
    # String sanitization
    if expected_type == str:
        if isinstance(input_data, str):
            return input_data.strip()
        else:
            return str(input_data).strip()
    
    # Numeric sanitization
    if expected_type in (int, float):
        try:
            return expected_type(input_data)
        except (ValueError, TypeError):
            raise ValidationError(
                field="numeric_input",
                value=input_data,
                constraint=f"Must be convertible to {expected_type.__name__}"
            )
    
    # Boolean sanitization
    if expected_type == bool:
        if isinstance(input_data, bool):
            return input_data
        elif isinstance(input_data, str):
            return input_data.lower() in ('true', '1', 'yes', 'on')
        else:
            return bool(input_data)
    
    # List sanitization
    if expected_type == list:
        if isinstance(input_data, list):
            return input_data
        elif isinstance(input_data, (tuple, set)):
            return list(input_data)
        else:
            return [input_data]
    
    return input_data