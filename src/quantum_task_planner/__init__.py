"""Quantum-Inspired Task Planner - Adaptive task scheduling using quantum principles."""

__version__ = "0.1.0"
__author__ = "Terragon Labs"

from .core import QuantumTaskPlanner, Task, TaskState, QuantumPriority
from .scheduler import QuantumScheduler, SuperpositionScheduler
from .optimizer import QuantumOptimizer, EntanglementOptimizer
from .validation import TaskValidator, ValidationResult
from .exceptions import QuantumTaskPlannerError, ValidationError
from .performance import PerformanceManager, PerformanceConfig
from .security import SecurityManager, SecurityConfig

__all__ = [
    "QuantumTaskPlanner",
    "Task",
    "TaskState", 
    "QuantumPriority",
    "QuantumScheduler",
    "SuperpositionScheduler",
    "QuantumOptimizer",
    "EntanglementOptimizer",
    "TaskValidator",
    "ValidationResult",
    "QuantumTaskPlannerError",
    "ValidationError", 
    "PerformanceManager",
    "PerformanceConfig",
    "SecurityManager",
    "SecurityConfig",
]