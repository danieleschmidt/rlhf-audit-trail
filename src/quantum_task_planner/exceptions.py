"""Quantum Task Planner exceptions and error handling."""

from typing import Optional, List, Dict, Any


class QuantumTaskPlannerError(Exception):
    """Base exception for Quantum Task Planner."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize quantum task planner error.
        
        Args:
            message: Error message
            error_code: Optional error code for programmatic handling
            context: Additional error context
        """
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        """String representation with context."""
        base_msg = super().__str__()
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg = f"{base_msg} (Context: {context_str})"
        return base_msg


class TaskNotFoundError(QuantumTaskPlannerError):
    """Task with specified ID not found."""
    
    def __init__(self, task_id: str):
        super().__init__(
            f"Task not found: {task_id}",
            error_code="TASK_NOT_FOUND",
            context={"task_id": task_id}
        )
        self.task_id = task_id


class InvalidTaskStateError(QuantumTaskPlannerError):
    """Invalid task state transition or operation."""
    
    def __init__(self, task_id: str, current_state: str, attempted_operation: str):
        super().__init__(
            f"Invalid operation '{attempted_operation}' for task {task_id} in state {current_state}",
            error_code="INVALID_TASK_STATE",
            context={
                "task_id": task_id,
                "current_state": current_state,
                "attempted_operation": attempted_operation
            }
        )
        self.task_id = task_id
        self.current_state = current_state
        self.attempted_operation = attempted_operation


class QuantumCoherenceError(QuantumTaskPlannerError):
    """Quantum coherence has been lost or cannot be maintained."""
    
    def __init__(self, task_id: str, reason: str):
        super().__init__(
            f"Quantum coherence lost for task {task_id}: {reason}",
            error_code="QUANTUM_COHERENCE_ERROR",
            context={"task_id": task_id, "reason": reason}
        )
        self.task_id = task_id
        self.reason = reason


class EntanglementError(QuantumTaskPlannerError):
    """Error in quantum entanglement operations."""
    
    def __init__(self, task1_id: str, task2_id: str, reason: str):
        super().__init__(
            f"Entanglement error between tasks {task1_id} and {task2_id}: {reason}",
            error_code="ENTANGLEMENT_ERROR",
            context={"task1_id": task1_id, "task2_id": task2_id, "reason": reason}
        )
        self.task1_id = task1_id
        self.task2_id = task2_id
        self.reason = reason


class SchedulingError(QuantumTaskPlannerError):
    """Error in quantum scheduling algorithms."""
    
    def __init__(self, algorithm: str, reason: str, tasks_count: Optional[int] = None):
        super().__init__(
            f"Scheduling error in {algorithm}: {reason}",
            error_code="SCHEDULING_ERROR",
            context={"algorithm": algorithm, "reason": reason, "tasks_count": tasks_count}
        )
        self.algorithm = algorithm
        self.reason = reason
        self.tasks_count = tasks_count


class OptimizationError(QuantumTaskPlannerError):
    """Error in quantum optimization algorithms."""
    
    def __init__(self, algorithm: str, iteration: int, reason: str):
        super().__init__(
            f"Optimization error in {algorithm} at iteration {iteration}: {reason}",
            error_code="OPTIMIZATION_ERROR",
            context={"algorithm": algorithm, "iteration": iteration, "reason": reason}
        )
        self.algorithm = algorithm
        self.iteration = iteration
        self.reason = reason


class DependencyError(QuantumTaskPlannerError):
    """Task dependency violation or circular dependency."""
    
    def __init__(self, task_id: str, dependency_chain: List[str], error_type: str = "circular"):
        chain_str = " -> ".join(dependency_chain)
        super().__init__(
            f"{error_type.title()} dependency detected for task {task_id}: {chain_str}",
            error_code="DEPENDENCY_ERROR",
            context={
                "task_id": task_id, 
                "dependency_chain": dependency_chain,
                "error_type": error_type
            }
        )
        self.task_id = task_id
        self.dependency_chain = dependency_chain
        self.error_type = error_type


class ResourceError(QuantumTaskPlannerError):
    """Insufficient resources for task execution."""
    
    def __init__(self, required_resources: int, available_resources: int, operation: str):
        super().__init__(
            f"Insufficient resources for {operation}: required {required_resources}, available {available_resources}",
            error_code="RESOURCE_ERROR",
            context={
                "required_resources": required_resources,
                "available_resources": available_resources,
                "operation": operation
            }
        )
        self.required_resources = required_resources
        self.available_resources = available_resources
        self.operation = operation


class ValidationError(QuantumTaskPlannerError):
    """Data validation error."""
    
    def __init__(self, field: str, value: Any, constraint: str):
        super().__init__(
            f"Validation error for field '{field}' with value '{value}': {constraint}",
            error_code="VALIDATION_ERROR",
            context={"field": field, "value": value, "constraint": constraint}
        )
        self.field = field
        self.value = value
        self.constraint = constraint


class ConfigurationError(QuantumTaskPlannerError):
    """Configuration or initialization error."""
    
    def __init__(self, component: str, setting: str, reason: str):
        super().__init__(
            f"Configuration error in {component} for setting '{setting}': {reason}",
            error_code="CONFIGURATION_ERROR",
            context={"component": component, "setting": setting, "reason": reason}
        )
        self.component = component
        self.setting = setting
        self.reason = reason


class ExecutionError(QuantumTaskPlannerError):
    """Task execution error."""
    
    def __init__(self, task_id: str, reason: str, traceback: Optional[str] = None):
        super().__init__(
            f"Execution error for task {task_id}: {reason}",
            error_code="EXECUTION_ERROR",
            context={"task_id": task_id, "reason": reason, "traceback": traceback}
        )
        self.task_id = task_id
        self.reason = reason
        self.traceback = traceback


class TimeoutError(QuantumTaskPlannerError):
    """Operation timeout error."""
    
    def __init__(self, operation: str, timeout_seconds: float, elapsed_seconds: float):
        super().__init__(
            f"Timeout in {operation}: exceeded {timeout_seconds}s (elapsed: {elapsed_seconds:.2f}s)",
            error_code="TIMEOUT_ERROR",
            context={
                "operation": operation,
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds
            }
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds


class ConcurrencyError(QuantumTaskPlannerError):
    """Concurrency or thread safety error."""
    
    def __init__(self, resource: str, conflict_type: str, task_ids: List[str]):
        task_list = ", ".join(task_ids)
        super().__init__(
            f"Concurrency error: {conflict_type} for {resource} involving tasks: {task_list}",
            error_code="CONCURRENCY_ERROR",
            context={
                "resource": resource,
                "conflict_type": conflict_type,
                "task_ids": task_ids
            }
        )
        self.resource = resource
        self.conflict_type = conflict_type
        self.task_ids = task_ids


def handle_exception(func):
    """Decorator for consistent exception handling."""
    
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except QuantumTaskPlannerError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Wrap other exceptions
            raise QuantumTaskPlannerError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                context={"function": func.__name__, "original_error": type(e).__name__}
            ) from e
    
    return wrapper


def handle_async_exception(func):
    """Decorator for consistent async exception handling."""
    
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except QuantumTaskPlannerError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Wrap other exceptions
            raise QuantumTaskPlannerError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                context={"function": func.__name__, "original_error": type(e).__name__}
            ) from e
    
    return wrapper