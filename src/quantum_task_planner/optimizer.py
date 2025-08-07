"""Quantum optimization algorithms for task planning."""

import asyncio
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
import numpy as np

from .core import Task, TaskState, QuantumPriority


@dataclass
class OptimizationResult:
    """Result of quantum optimization."""
    optimized_tasks: List[Task]
    optimization_score: float
    quantum_advantage: float
    iterations: int
    convergence_time: float


class QuantumOptimizer(ABC):
    """Abstract base for quantum optimization algorithms."""
    
    @abstractmethod
    async def optimize(
        self, 
        tasks: List[Task],
        objective_function: Optional[Callable] = None
    ) -> OptimizationResult:
        """Optimize task configuration using quantum principles."""
        pass


class QuantumAnnealingOptimizer(QuantumOptimizer):
    """Quantum annealing-inspired optimization for task planning."""
    
    def __init__(
        self,
        initial_temperature: float = 10.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        max_iterations: int = 1000
    ):
        """Initialize quantum annealing optimizer.
        
        Args:
            initial_temperature: Starting temperature for annealing
            cooling_rate: Rate of temperature decrease
            min_temperature: Minimum temperature to reach
            max_iterations: Maximum optimization iterations
        """
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.logger = logging.getLogger("quantum_optimizer.annealing")
    
    async def optimize(
        self, 
        tasks: List[Task],
        objective_function: Optional[Callable] = None
    ) -> OptimizationResult:
        """Optimize using quantum annealing simulation."""
        if not tasks:
            return OptimizationResult([], 0.0, 0.0, 0, 0.0)
        
        start_time = time.time()
        
        # Initialize with random configuration
        current_tasks = [self._copy_task(task) for task in tasks]
        current_score = await self._evaluate_configuration(current_tasks, objective_function)
        
        best_tasks = [self._copy_task(task) for task in current_tasks]
        best_score = current_score
        
        # Annealing parameters
        temperature = self.initial_temperature
        iteration = 0
        
        self.logger.info(f"Starting quantum annealing optimization with {len(tasks)} tasks")
        
        while temperature > self.min_temperature and iteration < self.max_iterations:
            # Generate neighbor configuration
            neighbor_tasks = await self._generate_neighbor_configuration(current_tasks, temperature)
            neighbor_score = await self._evaluate_configuration(neighbor_tasks, objective_function)
            
            # Calculate acceptance probability using quantum tunneling
            delta_score = neighbor_score - current_score
            
            if delta_score > 0:
                # Accept improvement
                current_tasks = neighbor_tasks
                current_score = neighbor_score
                
                if neighbor_score > best_score:
                    best_tasks = [self._copy_task(task) for task in neighbor_tasks]
                    best_score = neighbor_score
            else:
                # Quantum tunneling probability
                tunneling_probability = math.exp(delta_score / temperature)
                if random.random() < tunneling_probability:
                    current_tasks = neighbor_tasks
                    current_score = neighbor_score
            
            # Cool down
            temperature *= self.cooling_rate
            iteration += 1
            
            # Log progress periodically
            if iteration % 100 == 0:
                self.logger.debug(f"Iteration {iteration}, Score: {current_score:.3f}, Temp: {temperature:.3f}")
        
        optimization_time = time.time() - start_time
        
        # Calculate quantum advantage
        classical_score = await self._evaluate_configuration(tasks, objective_function)
        quantum_advantage = (best_score - classical_score) / max(classical_score, 0.01)
        
        self.logger.info(f"Annealing completed in {iteration} iterations ({optimization_time:.2f}s)")
        
        return OptimizationResult(
            optimized_tasks=best_tasks,
            optimization_score=best_score,
            quantum_advantage=quantum_advantage,
            iterations=iteration,
            convergence_time=optimization_time
        )
    
    def _copy_task(self, task: Task) -> Task:
        """Create a deep copy of a task."""
        new_task = Task(
            id=task.id,
            name=task.name,
            description=task.description,
            priority=task.priority,
            state=task.state,
            amplitude=task.amplitude,
            phase=task.phase,
            entangled_tasks=task.entangled_tasks.copy(),
            coherence_time=task.coherence_time,
            created_at=task.created_at,
            due_date=task.due_date,
            estimated_duration=task.estimated_duration,
            dependencies=task.dependencies.copy(),
            metadata=task.metadata.copy(),
            started_at=task.started_at,
            completed_at=task.completed_at,
            progress=task.progress
        )
        return new_task
    
    async def _generate_neighbor_configuration(
        self, 
        tasks: List[Task], 
        temperature: float
    ) -> List[Task]:
        """Generate neighbor configuration for annealing."""
        neighbor_tasks = [self._copy_task(task) for task in tasks]
        
        # Number of modifications based on temperature
        num_modifications = max(1, int(temperature * len(tasks) / 10))
        
        for _ in range(num_modifications):
            modification_type = random.choice([
                "adjust_amplitude",
                "shift_phase", 
                "modify_priority",
                "adjust_coherence_time"
            ])
            
            task = random.choice(neighbor_tasks)
            
            if modification_type == "adjust_amplitude":
                # Quantum amplitude adjustment
                noise = random.gauss(0, 0.1 * temperature / self.initial_temperature)
                task.amplitude = max(0.0, min(1.0, task.amplitude + noise))
                
            elif modification_type == "shift_phase":
                # Phase shift
                phase_shift = random.gauss(0, 0.5 * temperature / self.initial_temperature)
                task.phase = (task.phase + phase_shift) % (2 * math.pi)
                
            elif modification_type == "modify_priority":
                # Priority quantum fluctuation
                if random.random() < 0.1 * temperature / self.initial_temperature:
                    priorities = list(QuantumPriority)
                    task.priority = random.choice(priorities)
                    
            elif modification_type == "adjust_coherence_time":
                # Coherence time adjustment
                coherence_noise = random.gauss(0, 100 * temperature / self.initial_temperature)
                task.coherence_time = max(60, task.coherence_time + coherence_noise)
        
        return neighbor_tasks
    
    async def _evaluate_configuration(
        self, 
        tasks: List[Task],
        objective_function: Optional[Callable] = None
    ) -> float:
        """Evaluate configuration quality."""
        if objective_function:
            return await objective_function(tasks)
        
        # Default quantum objective function
        score = 0.0
        
        # 1. Quantum coherence score
        coherent_count = sum(1 for task in tasks if task.is_coherent)
        coherence_score = coherent_count / len(tasks)
        score += coherence_score * 0.3
        
        # 2. Priority alignment score
        priority_weights = {
            QuantumPriority.CRITICAL: 1.0,
            QuantumPriority.HIGH: 0.8,
            QuantumPriority.MEDIUM: 0.6,
            QuantumPriority.LOW: 0.4,
            QuantumPriority.DEFERRED: 0.2
        }
        
        priority_score = sum(
            task.amplitude * priority_weights.get(task.priority, 0.5)
            for task in tasks
        ) / len(tasks)
        score += priority_score * 0.4
        
        # 3. Quantum interference optimization
        total_interference = 0.0
        pair_count = 0
        
        for i, task1 in enumerate(tasks):
            for task2 in tasks[i+1:]:
                interference = task1.interfere_with(task2)
                total_interference += max(0, interference)  # Prefer constructive interference
                pair_count += 1
        
        if pair_count > 0:
            interference_score = total_interference / pair_count
            score += interference_score * 0.3
        
        return score


class GeneticQuantumOptimizer(QuantumOptimizer):
    """Genetic algorithm with quantum crossover and mutation."""
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1
    ):
        """Initialize genetic quantum optimizer.
        
        Args:
            population_size: Size of genetic population
            generations: Number of evolution generations
            mutation_rate: Probability of quantum mutation
            crossover_rate: Probability of quantum crossover
            elite_ratio: Fraction of elite individuals to preserve
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.logger = logging.getLogger("quantum_optimizer.genetic")
    
    async def optimize(
        self, 
        tasks: List[Task],
        objective_function: Optional[Callable] = None
    ) -> OptimizationResult:
        """Optimize using genetic algorithm with quantum operators."""
        if not tasks:
            return OptimizationResult([], 0.0, 0.0, 0, 0.0)
        
        start_time = time.time()
        
        # Initialize population
        population = await self._initialize_population(tasks)
        
        best_individual = None
        best_score = float('-inf')
        
        self.logger.info(f"Starting genetic optimization with population size {self.population_size}")
        
        for generation in range(self.generations):
            # Evaluate population
            scored_population = []
            for individual in population:
                score = await self._evaluate_configuration(individual, objective_function)
                scored_population.append((individual, score))
                
                if score > best_score:
                    best_individual = [self._copy_task(task) for task in individual]
                    best_score = score
            
            # Sort by fitness
            scored_population.sort(key=lambda x: x[1], reverse=True)
            
            # Select elite individuals
            elite_count = int(len(population) * self.elite_ratio)
            elite = [individual for individual, _ in scored_population[:elite_count]]
            
            # Generate new population
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                # Selection
                parent1 = await self._tournament_selection(scored_population)
                parent2 = await self._tournament_selection(scored_population)
                
                # Quantum crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = await self._quantum_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Quantum mutation
                if random.random() < self.mutation_rate:
                    child1 = await self._quantum_mutation(child1)
                if random.random() < self.mutation_rate:
                    child2 = await self._quantum_mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            # Log progress
            if generation % 20 == 0:
                avg_score = sum(score for _, score in scored_population) / len(scored_population)
                self.logger.debug(f"Generation {generation}: Best={best_score:.3f}, Avg={avg_score:.3f}")
        
        optimization_time = time.time() - start_time
        
        # Calculate quantum advantage
        classical_score = await self._evaluate_configuration(tasks, objective_function)
        quantum_advantage = (best_score - classical_score) / max(classical_score, 0.01)
        
        self.logger.info(f"Genetic optimization completed in {self.generations} generations ({optimization_time:.2f}s)")
        
        return OptimizationResult(
            optimized_tasks=best_individual,
            optimization_score=best_score,
            quantum_advantage=quantum_advantage,
            iterations=self.generations,
            convergence_time=optimization_time
        )
    
    def _copy_task(self, task: Task) -> Task:
        """Create a deep copy of a task."""
        new_task = Task(
            id=task.id,
            name=task.name,
            description=task.description,
            priority=task.priority,
            state=task.state,
            amplitude=task.amplitude,
            phase=task.phase,
            entangled_tasks=task.entangled_tasks.copy(),
            coherence_time=task.coherence_time,
            created_at=task.created_at,
            due_date=task.due_date,
            estimated_duration=task.estimated_duration,
            dependencies=task.dependencies.copy(),
            metadata=task.metadata.copy(),
            started_at=task.started_at,
            completed_at=task.completed_at,
            progress=task.progress
        )
        return new_task
    
    async def _initialize_population(self, tasks: List[Task]) -> List[List[Task]]:
        """Initialize genetic population with quantum variations."""
        population = []
        
        for _ in range(self.population_size):
            individual = []
            for task in tasks:
                # Create quantum variation of each task
                variant = self._copy_task(task)
                
                # Add quantum noise to properties
                variant.amplitude += random.gauss(0, 0.1)
                variant.amplitude = max(0.0, min(1.0, variant.amplitude))
                
                variant.phase += random.gauss(0, 0.5)
                variant.phase = variant.phase % (2 * math.pi)
                
                individual.append(variant)
            
            population.append(individual)
        
        return population
    
    async def _tournament_selection(
        self, 
        scored_population: List[Tuple[List[Task], float]]
    ) -> List[Task]:
        """Tournament selection with quantum uncertainty."""
        tournament_size = 3
        tournament = random.sample(scored_population, min(tournament_size, len(scored_population)))
        
        # Add quantum uncertainty to selection
        quantum_scores = []
        for individual, score in tournament:
            quantum_noise = random.gauss(0, 0.1)
            quantum_scores.append((individual, score + quantum_noise))
        
        # Select best from quantum-modified tournament
        winner = max(quantum_scores, key=lambda x: x[1])
        return [self._copy_task(task) for task in winner[0]]
    
    async def _quantum_crossover(
        self, 
        parent1: List[Task], 
        parent2: List[Task]
    ) -> Tuple[List[Task], List[Task]]:
        """Quantum crossover operation."""
        child1 = []
        child2 = []
        
        for task1, task2 in zip(parent1, parent2):
            # Quantum superposition crossover
            alpha = random.random()
            
            # Child 1: quantum combination
            new_task1 = self._copy_task(task1)
            new_task1.amplitude = alpha * task1.amplitude + (1 - alpha) * task2.amplitude
            new_task1.phase = alpha * task1.phase + (1 - alpha) * task2.phase
            new_task1.coherence_time = alpha * task1.coherence_time + (1 - alpha) * task2.coherence_time
            
            # Child 2: complementary combination
            new_task2 = self._copy_task(task2)
            new_task2.amplitude = (1 - alpha) * task1.amplitude + alpha * task2.amplitude
            new_task2.phase = (1 - alpha) * task1.phase + alpha * task2.phase
            new_task2.coherence_time = (1 - alpha) * task1.coherence_time + alpha * task2.coherence_time
            
            child1.append(new_task1)
            child2.append(new_task2)
        
        return child1, child2
    
    async def _quantum_mutation(self, individual: List[Task]) -> List[Task]:
        """Quantum mutation operation."""
        mutated = []
        
        for task in individual:
            new_task = self._copy_task(task)
            
            # Random quantum perturbations
            if random.random() < 0.3:  # Amplitude mutation
                new_task.amplitude += random.gauss(0, 0.2)
                new_task.amplitude = max(0.0, min(1.0, new_task.amplitude))
            
            if random.random() < 0.3:  # Phase mutation
                new_task.phase += random.gauss(0, 1.0)
                new_task.phase = new_task.phase % (2 * math.pi)
            
            if random.random() < 0.1:  # Priority mutation
                priorities = list(QuantumPriority)
                new_task.priority = random.choice(priorities)
            
            if random.random() < 0.2:  # Coherence time mutation
                new_task.coherence_time += random.gauss(0, 300)
                new_task.coherence_time = max(60, new_task.coherence_time)
            
            mutated.append(new_task)
        
        return mutated
    
    async def _evaluate_configuration(
        self, 
        tasks: List[Task],
        objective_function: Optional[Callable] = None
    ) -> float:
        """Evaluate configuration quality."""
        if objective_function:
            return await objective_function(tasks)
        
        # Default quantum fitness function
        score = 0.0
        
        # Quantum coherence
        coherent_ratio = sum(1 for task in tasks if task.is_coherent) / len(tasks)
        score += coherent_ratio * 0.25
        
        # Priority optimization
        priority_weights = {p: (5 - i) * 0.2 for i, p in enumerate(QuantumPriority)}
        priority_score = sum(
            task.amplitude * priority_weights.get(task.priority, 0.5)
            for task in tasks
        ) / len(tasks)
        score += priority_score * 0.35
        
        # Phase coherence (tasks with similar phases work well together)
        if len(tasks) > 1:
            phases = [task.phase for task in tasks]
            phase_variance = np.var(phases)
            phase_coherence = 1.0 / (1.0 + phase_variance)
            score += phase_coherence * 0.2
        
        # Amplitude optimization (balanced amplitudes)
        amplitudes = [task.amplitude for task in tasks]
        amplitude_mean = np.mean(amplitudes)
        amplitude_balance = 1.0 - abs(amplitude_mean - 0.5) * 2
        score += amplitude_balance * 0.2
        
        return score


class EntanglementOptimizer(QuantumOptimizer):
    """Optimizer focused on maximizing beneficial task entanglements."""
    
    def __init__(
        self,
        max_entanglement_distance: float = 0.5,
        entanglement_strength_threshold: float = 0.3,
        optimization_rounds: int = 50
    ):
        """Initialize entanglement optimizer.
        
        Args:
            max_entanglement_distance: Maximum distance for entanglement
            entanglement_strength_threshold: Minimum strength for entanglement
            optimization_rounds: Number of optimization rounds
        """
        self.max_entanglement_distance = max_entanglement_distance
        self.entanglement_strength_threshold = entanglement_strength_threshold
        self.optimization_rounds = optimization_rounds
        self.logger = logging.getLogger("quantum_optimizer.entanglement")
    
    async def optimize(
        self, 
        tasks: List[Task],
        objective_function: Optional[Callable] = None
    ) -> OptimizationResult:
        """Optimize task entanglements for maximum quantum advantage."""
        if not tasks:
            return OptimizationResult([], 0.0, 0.0, 0, 0.0)
        
        start_time = time.time()
        
        optimized_tasks = [self._copy_task(task) for task in tasks]
        
        self.logger.info(f"Optimizing entanglements for {len(tasks)} tasks")
        
        for round_num in range(self.optimization_rounds):
            # Find optimal entanglement pairs
            entanglement_candidates = await self._find_entanglement_candidates(optimized_tasks)
            
            # Create beneficial entanglements
            created_entanglements = 0
            for task1_id, task2_id, strength in entanglement_candidates:
                task1 = next(t for t in optimized_tasks if t.id == task1_id)
                task2 = next(t for t in optimized_tasks if t.id == task2_id)
                
                if strength > self.entanglement_strength_threshold:
                    await self._create_optimized_entanglement(task1, task2, strength)
                    created_entanglements += 1
            
            # Optimize existing entanglements
            await self._optimize_existing_entanglements(optimized_tasks)
            
            if round_num % 10 == 0:
                self.logger.debug(f"Round {round_num}: Created {created_entanglements} entanglements")
        
        optimization_time = time.time() - start_time
        
        # Evaluate optimization
        final_score = await self._evaluate_entanglement_configuration(optimized_tasks)
        initial_score = await self._evaluate_entanglement_configuration(tasks)
        quantum_advantage = (final_score - initial_score) / max(initial_score, 0.01)
        
        self.logger.info(f"Entanglement optimization completed ({optimization_time:.2f}s)")
        
        return OptimizationResult(
            optimized_tasks=optimized_tasks,
            optimization_score=final_score,
            quantum_advantage=quantum_advantage,
            iterations=self.optimization_rounds,
            convergence_time=optimization_time
        )
    
    def _copy_task(self, task: Task) -> Task:
        """Create a deep copy of a task."""
        new_task = Task(
            id=task.id,
            name=task.name,
            description=task.description,
            priority=task.priority,
            state=task.state,
            amplitude=task.amplitude,
            phase=task.phase,
            entangled_tasks=task.entangled_tasks.copy(),
            coherence_time=task.coherence_time,
            created_at=task.created_at,
            due_date=task.due_date,
            estimated_duration=task.estimated_duration,
            dependencies=task.dependencies.copy(),
            metadata=task.metadata.copy(),
            started_at=task.started_at,
            completed_at=task.completed_at,
            progress=task.progress
        )
        return new_task
    
    async def _find_entanglement_candidates(
        self, 
        tasks: List[Task]
    ) -> List[Tuple[str, str, float]]:
        """Find potential entanglement pairs with strength scores."""
        candidates = []
        
        for i, task1 in enumerate(tasks):
            for task2 in tasks[i+1:]:
                if task1.id in task2.entangled_tasks:
                    continue  # Already entangled
                
                strength = await self._calculate_entanglement_strength(task1, task2)
                if strength > 0:
                    candidates.append((task1.id, task2.id, strength))
        
        # Sort by strength
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates
    
    async def _calculate_entanglement_strength(
        self, 
        task1: Task, 
        task2: Task
    ) -> float:
        """Calculate potential entanglement strength between two tasks."""
        strength = 0.0
        
        # 1. Quantum interference
        interference = task1.interfere_with(task2)
        if interference > 0:  # Constructive interference
            strength += interference * 0.4
        
        # 2. Priority compatibility
        priority_values = {p: i for i, p in enumerate(QuantumPriority)}
        priority_distance = abs(priority_values[task1.priority] - priority_values[task2.priority])
        priority_compatibility = 1.0 - (priority_distance / len(QuantumPriority))
        strength += priority_compatibility * 0.3
        
        # 3. Temporal compatibility
        duration_ratio = min(task1.estimated_duration, task2.estimated_duration) / \
                        max(task1.estimated_duration, task2.estimated_duration, 0.1)
        strength += duration_ratio * 0.2
        
        # 4. Coherence alignment
        if task1.is_coherent and task2.is_coherent:
            strength += 0.1
        
        return min(1.0, strength)
    
    async def _create_optimized_entanglement(
        self, 
        task1: Task, 
        task2: Task, 
        strength: float
    ):
        """Create optimized entanglement between tasks."""
        # Traditional entanglement
        task1.entangle_with(task2)
        
        # Optimize entangled properties
        # Synchronize phases for maximum coherence
        avg_phase = (task1.phase + task2.phase) / 2
        phase_offset = strength * 0.1  # Small controlled offset
        
        task1.phase = avg_phase - phase_offset
        task2.phase = avg_phase + phase_offset
        
        # Optimize amplitudes for constructive interference
        amplitude_boost = strength * 0.1
        task1.amplitude = min(1.0, task1.amplitude + amplitude_boost)
        task2.amplitude = min(1.0, task2.amplitude + amplitude_boost)
        
        self.logger.debug(f"Created optimized entanglement: {task1.name} <-> {task2.name} (strength: {strength:.3f})")
    
    async def _optimize_existing_entanglements(self, tasks: List[Task]):
        """Optimize properties of existing entangled task pairs."""
        entangled_pairs = set()
        
        for task in tasks:
            for entangled_id in task.entangled_tasks:
                pair = tuple(sorted([task.id, entangled_id]))
                entangled_pairs.add(pair)
        
        for task1_id, task2_id in entangled_pairs:
            task1 = next(t for t in tasks if t.id == task1_id)
            task2 = next(t for t in tasks if t.id == task2_id)
            
            # Optimize phase synchronization
            phase_diff = abs(task1.phase - task2.phase)
            if phase_diff > math.pi:
                phase_diff = 2 * math.pi - phase_diff
            
            if phase_diff > 0.1:  # Need phase adjustment
                avg_phase = (task1.phase + task2.phase) / 2
                task1.phase = avg_phase
                task2.phase = avg_phase
            
            # Balance amplitudes
            avg_amplitude = (task1.amplitude + task2.amplitude) / 2
            amplitude_diff = abs(task1.amplitude - task2.amplitude)
            
            if amplitude_diff > 0.2:  # Significant imbalance
                task1.amplitude = avg_amplitude * 1.05
                task2.amplitude = avg_amplitude * 0.95
    
    async def _evaluate_entanglement_configuration(self, tasks: List[Task]) -> float:
        """Evaluate quality of entanglement configuration."""
        if not tasks:
            return 0.0
        
        score = 0.0
        total_entanglements = 0
        constructive_interference = 0.0
        
        # Count and evaluate entanglements
        for task in tasks:
            for entangled_id in task.entangled_tasks:
                entangled_task = next((t for t in tasks if t.id == entangled_id), None)
                if entangled_task:
                    total_entanglements += 1
                    
                    # Evaluate entanglement quality
                    interference = task.interfere_with(entangled_task)
                    if interference > 0:
                        constructive_interference += interference
        
        # Avoid double counting
        total_entanglements //= 2
        
        if total_entanglements > 0:
            # Average constructive interference
            avg_interference = constructive_interference / (total_entanglements * 2)
            score += avg_interference * 0.5
            
            # Entanglement density bonus
            max_possible_entanglements = len(tasks) * (len(tasks) - 1) // 2
            entanglement_density = total_entanglements / max(max_possible_entanglements, 1)
            score += entanglement_density * 0.3
        
        # Coherence preservation bonus
        coherent_tasks = sum(1 for task in tasks if task.is_coherent)
        coherence_ratio = coherent_tasks / len(tasks)
        score += coherence_ratio * 0.2
        
        return score