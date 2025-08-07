"""Quantum-inspired scheduling algorithms."""

import asyncio
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging

from .core import Task, TaskState, QuantumPriority, QuantumTaskPlanner


@dataclass
class SchedulingDecision:
    """Represents a scheduling decision with quantum properties."""
    task_id: str
    execution_time: float
    probability: float
    quantum_advantage: float
    reasoning: str


class QuantumScheduler(ABC):
    """Abstract base class for quantum scheduling algorithms."""
    
    @abstractmethod
    async def schedule_tasks(
        self, 
        tasks: List[Task], 
        available_resources: int = 1
    ) -> List[SchedulingDecision]:
        """Schedule tasks using quantum principles."""
        pass


class SuperpositionScheduler(QuantumScheduler):
    """Scheduler using quantum superposition for optimal task ordering."""
    
    def __init__(
        self,
        coherence_weight: float = 0.3,
        priority_weight: float = 0.4,
        interference_weight: float = 0.3
    ):
        """Initialize superposition scheduler.
        
        Args:
            coherence_weight: Weight for quantum coherence in scheduling
            priority_weight: Weight for task priority
            interference_weight: Weight for quantum interference effects
        """
        self.coherence_weight = coherence_weight
        self.priority_weight = priority_weight
        self.interference_weight = interference_weight
        self.logger = logging.getLogger("quantum_scheduler.superposition")
    
    async def schedule_tasks(
        self, 
        tasks: List[Task], 
        available_resources: int = 1
    ) -> List[SchedulingDecision]:
        """Schedule tasks using superposition-based optimization.
        
        The scheduler creates a superposition of all possible task orderings
        and collapses to the optimal configuration based on quantum metrics.
        """
        if not tasks:
            return []
        
        self.logger.info(f"Scheduling {len(tasks)} tasks with {available_resources} resources")
        
        # Create superposition of all possible task configurations
        superposition_states = await self._create_scheduling_superposition(tasks)
        
        # Evaluate each configuration using quantum metrics
        evaluated_states = await self._evaluate_superposition_states(superposition_states)
        
        # Collapse superposition to optimal scheduling
        optimal_decisions = await self._collapse_to_optimal_schedule(
            evaluated_states, available_resources
        )
        
        self.logger.info(f"Generated {len(optimal_decisions)} scheduling decisions")
        return optimal_decisions
    
    async def _create_scheduling_superposition(
        self, 
        tasks: List[Task]
    ) -> List[List[Task]]:
        """Create superposition of possible task orderings."""
        # For computational tractability, sample from possible orderings
        max_configurations = min(100, math.factorial(min(len(tasks), 5)))
        configurations = []
        
        for _ in range(max_configurations):
            # Create quantum-weighted random ordering
            config = await self._quantum_weighted_shuffle(tasks)
            configurations.append(config)
        
        return configurations
    
    async def _quantum_weighted_shuffle(self, tasks: List[Task]) -> List[Task]:
        """Shuffle tasks using quantum probability amplitudes."""
        weighted_tasks = []
        
        for task in tasks:
            # Weight by quantum properties
            weight = (
                task.amplitude * self.coherence_weight +
                self._priority_to_weight(task.priority) * self.priority_weight +
                task.probability * self.interference_weight
            )
            weighted_tasks.append((task, weight))
        
        # Sort by weights with quantum noise
        quantum_noise = [random.gauss(0, 0.1) for _ in weighted_tasks]
        sorted_tasks = sorted(
            zip(weighted_tasks, quantum_noise),
            key=lambda x: x[0][1] + x[1],
            reverse=True
        )
        
        return [task for (task, _), _ in sorted_tasks]
    
    def _priority_to_weight(self, priority: QuantumPriority) -> float:
        """Convert priority to numerical weight."""
        weights = {
            QuantumPriority.CRITICAL: 1.0,
            QuantumPriority.HIGH: 0.8,
            QuantumPriority.MEDIUM: 0.6,
            QuantumPriority.LOW: 0.4,
            QuantumPriority.DEFERRED: 0.2
        }
        return weights.get(priority, 0.5)
    
    async def _evaluate_superposition_states(
        self, 
        configurations: List[List[Task]]
    ) -> List[Tuple[List[Task], float]]:
        """Evaluate quantum fitness of each configuration."""
        evaluated = []
        
        for config in configurations:
            fitness = await self._calculate_quantum_fitness(config)
            evaluated.append((config, fitness))
        
        return evaluated
    
    async def _calculate_quantum_fitness(self, task_sequence: List[Task]) -> float:
        """Calculate quantum fitness score for a task sequence."""
        if not task_sequence:
            return 0.0
        
        fitness = 0.0
        
        # 1. Coherence preservation bonus
        coherent_tasks = sum(1 for task in task_sequence if task.is_coherent)
        coherence_score = coherent_tasks / len(task_sequence)
        fitness += coherence_score * self.coherence_weight
        
        # 2. Priority optimization
        priority_score = 0.0
        for i, task in enumerate(task_sequence):
            # Earlier execution of high-priority tasks
            position_penalty = i / len(task_sequence)
            task_priority_weight = self._priority_to_weight(task.priority)
            priority_score += task_priority_weight * (1 - position_penalty)
        
        priority_score /= len(task_sequence)
        fitness += priority_score * self.priority_weight
        
        # 3. Quantum interference effects
        interference_score = await self._calculate_interference_score(task_sequence)
        fitness += interference_score * self.interference_weight
        
        return fitness
    
    async def _calculate_interference_score(self, task_sequence: List[Task]) -> float:
        """Calculate quantum interference score for task sequence."""
        if len(task_sequence) < 2:
            return 0.0
        
        total_interference = 0.0
        pair_count = 0
        
        for i in range(len(task_sequence)):
            for j in range(i + 1, len(task_sequence)):
                task1, task2 = task_sequence[i], task_sequence[j]
                interference = task1.interfere_with(task2)
                
                # Positive interference is beneficial for scheduling
                if interference > 0:
                    total_interference += interference
                else:
                    total_interference += 0.5 * interference  # Penalty for destructive interference
                
                pair_count += 1
        
        return total_interference / pair_count if pair_count > 0 else 0.0
    
    async def _collapse_to_optimal_schedule(
        self, 
        evaluated_states: List[Tuple[List[Task], float]],
        available_resources: int
    ) -> List[SchedulingDecision]:
        """Collapse superposition to optimal scheduling decisions."""
        # Sort by fitness score
        sorted_states = sorted(evaluated_states, key=lambda x: x[1], reverse=True)
        
        # Take the highest-fitness configuration
        optimal_sequence, optimal_fitness = sorted_states[0]
        
        self.logger.info(f"Optimal configuration fitness: {optimal_fitness:.3f}")
        
        # Convert to scheduling decisions
        decisions = []
        current_time = time.time()
        
        for i, task in enumerate(optimal_sequence[:available_resources]):
            # Calculate quantum advantage
            quantum_advantage = await self._calculate_quantum_advantage(
                task, i, optimal_sequence
            )
            
            decision = SchedulingDecision(
                task_id=task.id,
                execution_time=current_time + (i * 0.1),  # Staggered execution
                probability=task.probability,
                quantum_advantage=quantum_advantage,
                reasoning=f"Superposition collapsed to optimal sequence (fitness: {optimal_fitness:.3f})"
            )
            
            decisions.append(decision)
        
        return decisions
    
    async def _calculate_quantum_advantage(
        self, 
        task: Task, 
        position: int, 
        sequence: List[Task]
    ) -> float:
        """Calculate quantum advantage of scheduling this task at this position."""
        # Base advantage from quantum properties
        base_advantage = task.amplitude * task.probability
        
        # Position-based advantage
        position_advantage = 1.0 - (position / len(sequence))
        
        # Coherence advantage
        coherence_advantage = 1.0 if task.is_coherent else 0.5
        
        # Entanglement advantage
        entanglement_advantage = 1.0 + (len(task.entangled_tasks) * 0.1)
        
        total_advantage = (
            base_advantage * 0.4 +
            position_advantage * 0.3 +
            coherence_advantage * 0.2 +
            entanglement_advantage * 0.1
        )
        
        return min(1.0, total_advantage)


class EntanglementScheduler(QuantumScheduler):
    """Scheduler optimized for entangled task groups."""
    
    def __init__(
        self,
        entanglement_boost: float = 0.5,
        correlation_threshold: float = 0.3
    ):
        """Initialize entanglement scheduler.
        
        Args:
            entanglement_boost: Boost factor for entangled task groups
            correlation_threshold: Minimum correlation for group scheduling
        """
        self.entanglement_boost = entanglement_boost
        self.correlation_threshold = correlation_threshold
        self.logger = logging.getLogger("quantum_scheduler.entanglement")
    
    async def schedule_tasks(
        self, 
        tasks: List[Task], 
        available_resources: int = 1
    ) -> List[SchedulingDecision]:
        """Schedule tasks prioritizing entangled groups."""
        if not tasks:
            return []
        
        # Group tasks by entanglement
        entangled_groups = await self._identify_entangled_groups(tasks)
        
        # Schedule groups as units
        decisions = []
        current_time = time.time()
        scheduled_count = 0
        
        for group in entangled_groups:
            if scheduled_count >= available_resources:
                break
            
            group_decisions = await self._schedule_entangled_group(
                group, current_time, scheduled_count
            )
            
            decisions.extend(group_decisions)
            scheduled_count += len(group_decisions)
            current_time += 0.1  # Time offset between groups
        
        return decisions
    
    async def _identify_entangled_groups(
        self, 
        tasks: List[Task]
    ) -> List[List[Task]]:
        """Identify groups of entangled tasks."""
        visited = set()
        groups = []
        
        for task in tasks:
            if task.id in visited:
                continue
            
            # Find all tasks entangled with this one
            group = []
            to_visit = [task]
            
            while to_visit:
                current_task = to_visit.pop()
                if current_task.id in visited:
                    continue
                
                visited.add(current_task.id)
                group.append(current_task)
                
                # Add entangled tasks to visit
                for entangled_id in current_task.entangled_tasks:
                    entangled_task = next(
                        (t for t in tasks if t.id == entangled_id), 
                        None
                    )
                    if entangled_task and entangled_task.id not in visited:
                        to_visit.append(entangled_task)
            
            if group:
                groups.append(group)
        
        # Sort groups by quantum correlation strength
        groups.sort(
            key=lambda g: sum(t.amplitude * t.probability for t in g),
            reverse=True
        )
        
        return groups
    
    async def _schedule_entangled_group(
        self, 
        group: List[Task], 
        start_time: float,
        resource_offset: int
    ) -> List[SchedulingDecision]:
        """Schedule an entangled group of tasks."""
        decisions = []
        
        # Sort group by amplitude (strongest first)
        sorted_group = sorted(group, key=lambda t: t.amplitude, reverse=True)
        
        for i, task in enumerate(sorted_group):
            # Calculate entanglement advantage
            entanglement_advantage = len(task.entangled_tasks) * self.entanglement_boost
            base_probability = task.probability
            enhanced_probability = min(1.0, base_probability + entanglement_advantage)
            
            decision = SchedulingDecision(
                task_id=task.id,
                execution_time=start_time + (i * 0.05),  # Rapid succession for entangled tasks
                probability=enhanced_probability,
                quantum_advantage=entanglement_advantage,
                reasoning=f"Entangled group scheduling (group size: {len(group)})"
            )
            
            decisions.append(decision)
        
        return decisions


class AdaptiveQuantumScheduler(QuantumScheduler):
    """Adaptive scheduler that combines multiple quantum scheduling strategies."""
    
    def __init__(self):
        """Initialize adaptive scheduler with multiple strategies."""
        self.superposition_scheduler = SuperpositionScheduler()
        self.entanglement_scheduler = EntanglementScheduler()
        self.logger = logging.getLogger("quantum_scheduler.adaptive")
        
        # Strategy effectiveness tracking
        self.strategy_performance = {
            "superposition": {"executions": 0, "success_rate": 0.5},
            "entanglement": {"executions": 0, "success_rate": 0.5},
        }
    
    async def schedule_tasks(
        self, 
        tasks: List[Task], 
        available_resources: int = 1
    ) -> List[SchedulingDecision]:
        """Adaptively choose and apply optimal scheduling strategy."""
        if not tasks:
            return []
        
        # Analyze task characteristics
        analysis = await self._analyze_task_characteristics(tasks)
        
        # Choose optimal strategy
        strategy = await self._choose_optimal_strategy(analysis)
        
        self.logger.info(f"Selected {strategy} strategy for {len(tasks)} tasks")
        
        # Apply chosen strategy
        if strategy == "superposition":
            decisions = await self.superposition_scheduler.schedule_tasks(
                tasks, available_resources
            )
        elif strategy == "entanglement":
            decisions = await self.entanglement_scheduler.schedule_tasks(
                tasks, available_resources
            )
        else:
            # Hybrid approach
            decisions = await self._hybrid_scheduling(tasks, available_resources)
        
        # Update strategy performance tracking
        self.strategy_performance[strategy]["executions"] += 1
        
        return decisions
    
    async def _analyze_task_characteristics(self, tasks: List[Task]) -> Dict[str, float]:
        """Analyze characteristics of task set to guide strategy selection."""
        total_tasks = len(tasks)
        
        analysis = {
            "entanglement_ratio": 0.0,
            "coherence_ratio": 0.0,
            "priority_distribution": 0.0,
            "complexity_score": 0.0
        }
        
        if total_tasks == 0:
            return analysis
        
        # Calculate entanglement ratio
        entangled_tasks = sum(1 for task in tasks if task.entangled_tasks)
        analysis["entanglement_ratio"] = entangled_tasks / total_tasks
        
        # Calculate coherence ratio
        coherent_tasks = sum(1 for task in tasks if task.is_coherent)
        analysis["coherence_ratio"] = coherent_tasks / total_tasks
        
        # Analyze priority distribution
        high_priority_tasks = sum(
            1 for task in tasks 
            if task.priority in [QuantumPriority.CRITICAL, QuantumPriority.HIGH]
        )
        analysis["priority_distribution"] = high_priority_tasks / total_tasks
        
        # Calculate complexity score
        avg_dependencies = sum(len(task.dependencies) for task in tasks) / total_tasks
        avg_duration = sum(task.estimated_duration for task in tasks) / total_tasks
        analysis["complexity_score"] = (avg_dependencies + avg_duration) / 2
        
        return analysis
    
    async def _choose_optimal_strategy(self, analysis: Dict[str, float]) -> str:
        """Choose optimal scheduling strategy based on analysis."""
        # Weight factors for strategy selection
        entanglement_weight = analysis["entanglement_ratio"]
        coherence_weight = analysis["coherence_ratio"]
        complexity_weight = analysis["complexity_score"]
        
        # Strategy scores
        superposition_score = (
            coherence_weight * 0.6 +
            (1 - entanglement_weight) * 0.4
        )
        
        entanglement_score = (
            entanglement_weight * 0.8 +
            complexity_weight * 0.2
        )
        
        # Factor in historical performance
        superposition_performance = self.strategy_performance["superposition"]["success_rate"]
        entanglement_performance = self.strategy_performance["entanglement"]["success_rate"]
        
        adjusted_superposition_score = superposition_score * superposition_performance
        adjusted_entanglement_score = entanglement_score * entanglement_performance
        
        # Choose strategy
        if adjusted_entanglement_score > adjusted_superposition_score + 0.1:
            return "entanglement"
        elif adjusted_superposition_score > adjusted_entanglement_score + 0.1:
            return "superposition"
        else:
            return "hybrid"
    
    async def _hybrid_scheduling(
        self, 
        tasks: List[Task], 
        available_resources: int
    ) -> List[SchedulingDecision]:
        """Apply hybrid scheduling combining multiple strategies."""
        # Split resources between strategies
        superposition_resources = available_resources // 2
        entanglement_resources = available_resources - superposition_resources
        
        # Get decisions from both strategies
        superposition_decisions = await self.superposition_scheduler.schedule_tasks(
            tasks, superposition_resources
        )
        
        entanglement_decisions = await self.entanglement_scheduler.schedule_tasks(
            tasks, entanglement_resources
        )
        
        # Combine and optimize decisions
        all_decisions = superposition_decisions + entanglement_decisions
        
        # Remove duplicates and sort by quantum advantage
        seen_tasks = set()
        unique_decisions = []
        
        for decision in sorted(all_decisions, key=lambda d: d.quantum_advantage, reverse=True):
            if decision.task_id not in seen_tasks:
                unique_decisions.append(decision)
                seen_tasks.add(decision.task_id)
        
        return unique_decisions[:available_resources]
    
    def update_strategy_performance(self, strategy: str, success_rate: float):
        """Update performance tracking for a strategy."""
        if strategy in self.strategy_performance:
            current_rate = self.strategy_performance[strategy]["success_rate"]
            executions = self.strategy_performance[strategy]["executions"]
            
            # Exponential moving average
            alpha = 0.2
            new_rate = alpha * success_rate + (1 - alpha) * current_rate
            self.strategy_performance[strategy]["success_rate"] = new_rate
            
            self.logger.info(f"Updated {strategy} success rate: {new_rate:.3f}")