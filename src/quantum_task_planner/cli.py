"""Command-line interface for Quantum Task Planner."""

import asyncio
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import click

from .core import QuantumTaskPlanner, Task, TaskState, QuantumPriority
from .scheduler import SuperpositionScheduler, EntanglementScheduler, AdaptiveQuantumScheduler
from .optimizer import QuantumAnnealingOptimizer, GeneticQuantumOptimizer, EntanglementOptimizer


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=level
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def main(ctx: click.Context, verbose: bool):
    """Quantum-Inspired Task Planner CLI."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    setup_logging(verbose)


@main.command()
@click.option('--name', '-n', default='quantum-planner', help='Planner instance name')
@click.option('--coherence/--no-coherence', default=True, help='Enable coherence preservation')
@click.option('--entanglement/--no-entanglement', default=True, help='Enable task entanglement')
@click.option('--save', '-s', type=click.Path(), help='Save planner state to file')
@click.pass_context
def create(ctx: click.Context, name: str, coherence: bool, entanglement: bool, save: Optional[str]):
    """Create a new quantum task planner."""
    planner = QuantumTaskPlanner(
        name=name,
        coherence_preservation=coherence,
        entanglement_enabled=entanglement
    )
    
    if save:
        state = planner.get_system_state()
        Path(save).write_text(json.dumps(state, indent=2, default=str))
        click.echo(f"Planner state saved to {save}")
    
    click.echo(f"Created quantum task planner: {name}")
    click.echo(f"Coherence preservation: {coherence}")
    click.echo(f"Entanglement enabled: {entanglement}")


@main.command()
@click.option('--name', '-n', required=True, help='Task name')
@click.option('--description', '-d', default='', help='Task description')
@click.option('--priority', '-p', 
              type=click.Choice(['critical', 'high', 'medium', 'low', 'deferred']),
              default='medium', help='Task priority')
@click.option('--duration', type=float, default=1.0, help='Estimated duration in hours')
@click.option('--depends-on', multiple=True, help='Task dependencies (IDs)')
@click.option('--metadata', type=str, help='Task metadata as JSON')
@click.option('--save', '-s', type=click.Path(), help='Save task to file')
def add_task(
    name: str, 
    description: str, 
    priority: str, 
    duration: float, 
    depends_on: List[str],
    metadata: Optional[str],
    save: Optional[str]
):
    """Add a new quantum task."""
    # Parse priority
    priority_map = {
        'critical': QuantumPriority.CRITICAL,
        'high': QuantumPriority.HIGH,
        'medium': QuantumPriority.MEDIUM,
        'low': QuantumPriority.LOW,
        'deferred': QuantumPriority.DEFERRED
    }
    
    # Parse metadata
    task_metadata = {}
    if metadata:
        try:
            task_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            click.echo("Warning: Invalid JSON metadata, using empty dict", err=True)
    
    # Create planner and task
    planner = QuantumTaskPlanner()
    task = planner.create_task(
        name=name,
        description=description,
        priority=priority_map[priority],
        estimated_duration=duration,
        dependencies=list(depends_on),
        metadata=task_metadata
    )
    
    click.echo(f"Created quantum task: {task.name} (ID: {task.id})")
    click.echo(f"Priority: {task.priority.value}")
    click.echo(f"Amplitude: {task.amplitude:.3f}")
    click.echo(f"Probability: {task.probability:.3f}")
    
    if save:
        task_data = {
            'id': task.id,
            'name': task.name,
            'description': task.description,
            'priority': task.priority.value,
            'state': task.state.value,
            'amplitude': task.amplitude,
            'phase': task.phase,
            'coherence_time': task.coherence_time,
            'estimated_duration': task.estimated_duration,
            'dependencies': list(task.dependencies),
            'metadata': task.metadata
        }
        Path(save).write_text(json.dumps(task_data, indent=2, default=str))
        click.echo(f"Task saved to {save}")


@main.command()
@click.option('--algorithm', '-a', 
              type=click.Choice(['superposition', 'entanglement', 'adaptive']),
              default='adaptive', help='Scheduling algorithm')
@click.option('--resources', '-r', type=int, default=1, help='Available resources')
@click.option('--tasks-file', type=click.Path(exists=True), help='Load tasks from file')
@click.option('--output', '-o', type=click.Path(), help='Save schedule to file')
@click.pass_context
def schedule(
    ctx: click.Context,
    algorithm: str, 
    resources: int, 
    tasks_file: Optional[str],
    output: Optional[str]
):
    """Schedule quantum tasks using various algorithms."""
    
    async def run_scheduling():
        # Create planner and demo tasks if no file provided
        planner = QuantumTaskPlanner()
        
        if tasks_file:
            # Load tasks from file
            tasks_data = json.loads(Path(tasks_file).read_text())
            tasks = []
            for task_data in tasks_data:
                task = Task(
                    id=task_data['id'],
                    name=task_data['name'],
                    description=task_data.get('description', ''),
                    priority=QuantumPriority(task_data.get('priority', 'medium')),
                    estimated_duration=task_data.get('estimated_duration', 1.0),
                    dependencies=set(task_data.get('dependencies', [])),
                    metadata=task_data.get('metadata', {})
                )
                tasks.append(task)
                planner.tasks[task.id] = task
        else:
            # Create demo tasks
            tasks = []
            for i in range(5):
                task = planner.create_task(
                    name=f"Demo Task {i+1}",
                    description=f"Sample quantum task {i+1}",
                    priority=QuantumPriority.MEDIUM,
                    estimated_duration=1.0 + i * 0.5
                )
                tasks.append(task)
        
        # Select scheduler
        if algorithm == 'superposition':
            scheduler = SuperpositionScheduler()
        elif algorithm == 'entanglement':
            scheduler = EntanglementScheduler()
        else:
            scheduler = AdaptiveQuantumScheduler()
        
        click.echo(f"Scheduling {len(tasks)} tasks using {algorithm} algorithm...")
        
        # Run scheduling
        start_time = time.time()
        decisions = await scheduler.schedule_tasks(tasks, resources)
        scheduling_time = time.time() - start_time
        
        # Display results
        click.echo(f"\nScheduling completed in {scheduling_time:.3f}s")
        click.echo(f"Generated {len(decisions)} scheduling decisions:")
        
        for i, decision in enumerate(decisions):
            task = next(t for t in tasks if t.id == decision.task_id)
            click.echo(f"\n{i+1}. {task.name}")
            click.echo(f"   Execution Time: {datetime.fromtimestamp(decision.execution_time)}")
            click.echo(f"   Probability: {decision.probability:.3f}")
            click.echo(f"   Quantum Advantage: {decision.quantum_advantage:.3f}")
            click.echo(f"   Reasoning: {decision.reasoning}")
        
        # Save output
        if output:
            output_data = {
                'algorithm': algorithm,
                'scheduling_time': scheduling_time,
                'decisions': [
                    {
                        'task_id': d.task_id,
                        'task_name': next(t for t in tasks if t.id == d.task_id).name,
                        'execution_time': d.execution_time,
                        'probability': d.probability,
                        'quantum_advantage': d.quantum_advantage,
                        'reasoning': d.reasoning
                    }
                    for d in decisions
                ]
            }
            Path(output).write_text(json.dumps(output_data, indent=2, default=str))
            click.echo(f"\nSchedule saved to {output}")
    
    asyncio.run(run_scheduling())


@main.command()
@click.option('--algorithm', '-a',
              type=click.Choice(['annealing', 'genetic', 'entanglement']),
              default='annealing', help='Optimization algorithm')
@click.option('--tasks-file', type=click.Path(exists=True), help='Load tasks from file')
@click.option('--iterations', '-i', type=int, help='Optimization iterations')
@click.option('--output', '-o', type=click.Path(), help='Save results to file')
@click.pass_context
def optimize(
    ctx: click.Context,
    algorithm: str,
    tasks_file: Optional[str],
    iterations: Optional[int],
    output: Optional[str]
):
    """Optimize quantum task configuration."""
    
    async def run_optimization():
        # Create planner and tasks
        planner = QuantumTaskPlanner()
        
        if tasks_file:
            tasks_data = json.loads(Path(tasks_file).read_text())
            tasks = []
            for task_data in tasks_data:
                task = Task(
                    id=task_data['id'],
                    name=task_data['name'],
                    description=task_data.get('description', ''),
                    priority=QuantumPriority(task_data.get('priority', 'medium')),
                    estimated_duration=task_data.get('estimated_duration', 1.0),
                    dependencies=set(task_data.get('dependencies', [])),
                    metadata=task_data.get('metadata', {})
                )
                tasks.append(task)
        else:
            # Create demo tasks with suboptimal configuration
            tasks = []
            for i in range(8):
                task = planner.create_task(
                    name=f"Optimize Task {i+1}",
                    priority=QuantumPriority.MEDIUM,
                    estimated_duration=1.0 + i * 0.3
                )
                # Add some randomness to make optimization worthwhile
                task.amplitude = 0.3 + (i % 3) * 0.2
                task.phase = i * 0.5
                tasks.append(task)
        
        # Select optimizer
        if algorithm == 'annealing':
            optimizer = QuantumAnnealingOptimizer(
                max_iterations=iterations or 500
            )
        elif algorithm == 'genetic':
            optimizer = GeneticQuantumOptimizer(
                generations=iterations or 50
            )
        else:
            optimizer = EntanglementOptimizer(
                optimization_rounds=iterations or 30
            )
        
        click.echo(f"Optimizing {len(tasks)} tasks using {algorithm} algorithm...")
        
        # Run optimization
        result = await optimizer.optimize(tasks)
        
        # Display results
        click.echo(f"\nOptimization completed:")
        click.echo(f"Algorithm: {algorithm}")
        click.echo(f"Iterations: {result.iterations}")
        click.echo(f"Convergence Time: {result.convergence_time:.3f}s")
        click.echo(f"Optimization Score: {result.optimization_score:.3f}")
        click.echo(f"Quantum Advantage: {result.quantum_advantage:.3f}")
        
        click.echo(f"\nOptimized Tasks:")
        for i, task in enumerate(result.optimized_tasks):
            click.echo(f"{i+1}. {task.name}")
            click.echo(f"   Amplitude: {task.amplitude:.3f}")
            click.echo(f"   Phase: {task.phase:.3f}")
            click.echo(f"   Probability: {task.probability:.3f}")
            if task.entangled_tasks:
                entangled_names = [
                    t.name for t in result.optimized_tasks 
                    if t.id in task.entangled_tasks
                ]
                click.echo(f"   Entangled with: {', '.join(entangled_names)}")
        
        # Save results
        if output:
            result_data = {
                'algorithm': algorithm,
                'optimization_score': result.optimization_score,
                'quantum_advantage': result.quantum_advantage,
                'iterations': result.iterations,
                'convergence_time': result.convergence_time,
                'optimized_tasks': [
                    {
                        'id': t.id,
                        'name': t.name,
                        'amplitude': t.amplitude,
                        'phase': t.phase,
                        'probability': t.probability,
                        'entangled_tasks': list(t.entangled_tasks)
                    }
                    for t in result.optimized_tasks
                ]
            }
            Path(output).write_text(json.dumps(result_data, indent=2, default=str))
            click.echo(f"\nResults saved to {output}")
    
    asyncio.run(run_optimization())


@main.command()
@click.option('--planner-file', type=click.Path(exists=True), help='Load planner state')
@click.option('--tasks', '-t', type=int, default=5, help='Number of demo tasks')
@click.option('--cycles', '-c', type=int, default=3, help='Number of quantum cycles')
@click.option('--output', '-o', type=click.Path(), help='Save execution log')
@click.pass_context
def run(
    ctx: click.Context,
    planner_file: Optional[str],
    tasks: int,
    cycles: int,
    output: Optional[str]
):
    """Run quantum task planner simulation."""
    
    async def run_simulation():
        # Create or load planner
        if planner_file:
            state = json.loads(Path(planner_file).read_text())
            planner = QuantumTaskPlanner(name=state.get('planner_name', 'quantum-planner'))
            click.echo(f"Loaded planner state from {planner_file}")
        else:
            planner = QuantumTaskPlanner(name="simulation-planner")
        
        # Create demo tasks
        demo_tasks = []
        for i in range(tasks):
            priority = [QuantumPriority.CRITICAL, QuantumPriority.HIGH, 
                       QuantumPriority.MEDIUM, QuantumPriority.LOW][i % 4]
            
            task = planner.create_task(
                name=f"Simulation Task {i+1}",
                description=f"Demo task for quantum simulation",
                priority=priority,
                estimated_duration=0.5 + (i % 3) * 0.5
            )
            demo_tasks.append(task)
        
        click.echo(f"Created {len(demo_tasks)} demo tasks")
        
        # Run quantum cycles
        execution_log = []
        
        for cycle in range(cycles):
            click.echo(f"\n--- Quantum Cycle {cycle + 1} ---")
            
            cycle_result = await planner.run_quantum_cycle()
            execution_log.append(cycle_result)
            
            # Display cycle results
            click.echo(f"Cycle Duration: {cycle_result['cycle_duration']:.3f}s")
            click.echo(f"Tasks Collapsed: {cycle_result['collapsed_tasks']}")
            click.echo(f"Tasks Decohered: {cycle_result['decohered_tasks']}")
            click.echo(f"Executable Tasks: {cycle_result['executable_tasks']}")
            click.echo(f"Tasks Executed: {cycle_result['executed_tasks']}")
            
            # Show system state
            state = cycle_result['system_state']
            click.echo(f"System State:")
            click.echo(f"  Total Tasks: {state['total_tasks']}")
            click.echo(f"  Completed: {state['tasks_by_state'].get('completed', 0)}")
            click.echo(f"  Running: {state['tasks_by_state'].get('running', 0)}")
            click.echo(f"  Success Rate: {state['execution_metrics']['success_rate']:.2%}")
            
            # Wait between cycles
            if cycle < cycles - 1:
                await asyncio.sleep(0.5)
        
        # Final summary
        final_state = planner.get_system_state()
        click.echo(f"\n--- Simulation Complete ---")
        click.echo(f"Total Cycles: {cycles}")
        click.echo(f"Final Success Rate: {final_state['execution_metrics']['success_rate']:.2%}")
        click.echo(f"Quantum Collapses: {final_state['quantum_metrics']['total_collapses']}")
        click.echo(f"Active Entanglements: {final_state['quantum_metrics']['active_entanglements']}")
        
        # Save execution log
        if output:
            log_data = {
                'simulation_config': {
                    'tasks': tasks,
                    'cycles': cycles,
                    'planner_name': planner.name
                },
                'execution_log': execution_log,
                'final_state': final_state
            }
            Path(output).write_text(json.dumps(log_data, indent=2, default=str))
            click.echo(f"Execution log saved to {output}")
    
    asyncio.run(run_simulation())


@main.command()
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text', help='Output format')
def version(format: str):
    """Show version information."""
    from . import __version__, __author__
    
    if format == 'json':
        version_info = {
            'version': __version__,
            'author': __author__,
            'description': 'Quantum-Inspired Task Planner'
        }
        click.echo(json.dumps(version_info, indent=2))
    else:
        click.echo(f"Quantum Task Planner v{__version__}")
        click.echo(f"Author: {__author__}")
        click.echo("Adaptive task scheduling using quantum-inspired algorithms")


if __name__ == '__main__':
    main()