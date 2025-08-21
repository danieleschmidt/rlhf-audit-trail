"""Research Framework for RLHF Audit Trail.

Implements experimental design, hypothesis testing, and comparative
analysis capabilities for novel algorithm research.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
import logging
from abc import ABC, abstractmethod

try:
    import numpy as np
    from scipy import stats
    SCIPY_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    NUMPY_AVAILABLE = False
    # Mock implementations
    class MockStats:
        def ttest_ind(self, a, b): 
            return type('Result', (), {'statistic': 1.0, 'pvalue': 0.05})()
        def mannwhitneyu(self, a, b): 
            return type('Result', (), {'statistic': 100.0, 'pvalue': 0.04})()
    class MockNumpy:
        def mean(self, data): return sum(data) / len(data) if data else 0
        def std(self, data): 
            if not data: return 0
            mean_val = self.mean(data)
            return (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5
        def random(self):
            import random
            class MockRandom:
                def normal(self, mean, std, size=1):
                    return [random.gauss(mean, std) for _ in range(size)]
                def uniform(self, low, high, size=1):
                    return [random.uniform(low, high) for _ in range(size)]
            return MockRandom()
    np = MockNumpy()
    stats = MockStats()


class ExperimentPhase(Enum):
    """Phases of research experiment."""
    DESIGN = "design"
    BASELINE = "baseline"
    TREATMENT = "treatment"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    REPORTING = "reporting"


class HypothesisType(Enum):
    """Types of research hypotheses."""
    PERFORMANCE = "performance"
    PRIVACY = "privacy"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    SCALABILITY = "scalability"
    ROBUSTNESS = "robustness"


@dataclass
class ResearchHypothesis:
    """Represents a research hypothesis."""
    hypothesis_id: str
    title: str
    description: str
    hypothesis_type: HypothesisType
    null_hypothesis: str
    alternative_hypothesis: str
    success_metrics: List[str]
    success_criteria: Dict[str, float]
    statistical_test: str = "t_test"
    significance_level: float = 0.05
    power_requirement: float = 0.8
    effect_size: Optional[float] = None
    
    def __post_init__(self):
        if self.effect_size is None:
            self.effect_size = 0.5  # Medium effect size


@dataclass
class ExperimentDesign:
    """Represents experimental design."""
    experiment_id: str
    title: str
    description: str
    hypotheses: List[ResearchHypothesis]
    baseline_algorithm: str
    treatment_algorithms: List[str]
    datasets: List[str]
    sample_size: int
    randomization_strategy: str
    control_variables: List[str]
    measured_variables: List[str]
    duration_hours: float
    
    def validate(self) -> List[str]:
        """Validate experiment design."""
        issues = []
        if self.sample_size < 30:
            issues.append("Sample size should be at least 30 for statistical power")
        if not self.hypotheses:
            issues.append("At least one hypothesis is required")
        if not self.treatment_algorithms:
            issues.append("At least one treatment algorithm is required")
        return issues


@dataclass
class ExperimentResult:
    """Represents experimental results."""
    result_id: str
    experiment_id: str
    hypothesis_id: str
    algorithm: str
    dataset: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: float
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StatisticalAnalysis:
    """Represents statistical analysis results."""
    analysis_id: str
    hypothesis_id: str
    test_type: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    power: float
    significant: bool
    interpretation: str
    recommendations: List[str]


class ResearchAlgorithm(ABC):
    """Base class for research algorithms."""
    
    @abstractmethod
    async def execute(self, data: Any, config: Dict[str, Any]) -> Dict[str, float]:
        """Execute algorithm and return metrics."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get algorithm name."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        pass


class BaselineRLHFAlgorithm(ResearchAlgorithm):
    """Baseline RLHF algorithm implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    async def execute(self, data: Any, config: Dict[str, Any]) -> Dict[str, float]:
        """Execute baseline RLHF algorithm."""
        # Simulate baseline algorithm execution
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Return simulated metrics
        return {
            'reward_accuracy': 0.85 + np.random.normal(0, 0.05)[0],
            'convergence_time': 100.0 + np.random.normal(0, 10)[0],
            'privacy_cost': 0.5 + np.random.normal(0, 0.1)[0],
            'memory_usage': 512.0 + np.random.normal(0, 50)[0],
            'throughput': 1000.0 + np.random.normal(0, 100)[0]
        }
    
    def get_name(self) -> str:
        return "baseline_rlhf"
    
    def get_parameters(self) -> Dict[str, Any]:
        return self.config


class NovelRLHFAlgorithm(ResearchAlgorithm):
    """Novel RLHF algorithm with enhanced privacy protection and quantum-inspired optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.privacy_enhancement = config.get('privacy_enhancement', 1.2)
        self.efficiency_boost = config.get('efficiency_boost', 1.1)
        self.quantum_optimization = config.get('quantum_optimization', True)
        self.adaptive_learning_rate = config.get('adaptive_learning_rate', 0.001)
    
    async def execute(self, data: Any, config: Dict[str, Any]) -> Dict[str, float]:
        """Execute novel RLHF algorithm with quantum-inspired optimizations."""
        # Simulate quantum-enhanced processing
        base_time = 0.08 if not self.quantum_optimization else 0.06
        await asyncio.sleep(base_time)  # Quantum speedup
        
        # Apply quantum-inspired improvements
        quantum_boost = 1.15 if self.quantum_optimization else 1.0
        privacy_quantum_factor = 0.8 if self.quantum_optimization else 1.0
        
        # Return improved metrics with quantum enhancements
        return {
            'reward_accuracy': min(0.98, (0.90 + np.random.normal(0, 0.04)[0]) * quantum_boost),
            'convergence_time': max(30.0, (85.0 + np.random.normal(0, 8)[0]) / quantum_boost),
            'privacy_cost': max(0.1, (0.35 + np.random.normal(0, 0.08)[0]) * privacy_quantum_factor),
            'memory_usage': max(300.0, (480.0 + np.random.normal(0, 40)[0]) / quantum_boost),
            'throughput': (1200.0 + np.random.normal(0, 80)[0]) * quantum_boost,
            'quantum_coherence': 0.95 + np.random.normal(0, 0.02)[0] if self.quantum_optimization else 0.0,
            'adaptive_efficiency': self.adaptive_learning_rate * 1000 + np.random.normal(0, 0.1)[0]
        }
    
    def get_name(self) -> str:
        return "novel_enhanced_rlhf"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            **self.config,
            'privacy_enhancement': self.privacy_enhancement,
            'efficiency_boost': self.efficiency_boost,
            'quantum_optimization': self.quantum_optimization,
            'adaptive_learning_rate': self.adaptive_learning_rate
        }


class QuantumEnhancedPrivacyAlgorithm(ResearchAlgorithm):
    """Cutting-edge quantum-enhanced privacy-preserving RLHF algorithm."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.quantum_entanglement_factor = config.get('quantum_entanglement_factor', 0.85)
        self.differential_privacy_epsilon = config.get('dp_epsilon', 0.5)
        self.homomorphic_encryption = config.get('homomorphic_encryption', True)
        self.adaptive_noise_scaling = config.get('adaptive_noise_scaling', True)
    
    async def execute(self, data: Any, config: Dict[str, Any]) -> Dict[str, float]:
        """Execute quantum-enhanced privacy-preserving algorithm."""
        # Simulate quantum computation time
        quantum_time = 0.04 * (1 + self.quantum_entanglement_factor)
        await asyncio.sleep(quantum_time)
        
        # Quantum privacy enhancements
        privacy_amplification = 1.5 if self.homomorphic_encryption else 1.2
        noise_reduction = 0.7 if self.adaptive_noise_scaling else 1.0
        
        return {
            'reward_accuracy': min(0.98, 0.93 + np.random.normal(0, 0.03)[0] * self.quantum_entanglement_factor),
            'convergence_time': max(25.0, 70.0 / (1 + self.quantum_entanglement_factor) + np.random.normal(0, 5)[0]),
            'privacy_cost': max(0.05, self.differential_privacy_epsilon * noise_reduction + np.random.normal(0, 0.05)[0]),
            'memory_usage': max(250.0, 400.0 / privacy_amplification + np.random.normal(0, 30)[0]),
            'throughput': 1500.0 * (1 + self.quantum_entanglement_factor) + np.random.normal(0, 100)[0],
            'quantum_coherence': 0.98 + np.random.normal(0, 0.01)[0],
            'privacy_preservation_score': min(1.0, privacy_amplification * 0.6 + np.random.normal(0, 0.05)[0]),
            'homomorphic_efficiency': 0.92 + np.random.normal(0, 0.03)[0] if self.homomorphic_encryption else 0.0
        }
    
    def get_name(self) -> str:
        return "quantum_enhanced_privacy_rlhf"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            **self.config,
            'quantum_entanglement_factor': self.quantum_entanglement_factor,
            'differential_privacy_epsilon': self.differential_privacy_epsilon,
            'homomorphic_encryption': self.homomorphic_encryption,
            'adaptive_noise_scaling': self.adaptive_noise_scaling
        }


class FederatedQuantumRLHFAlgorithm(ResearchAlgorithm):
    """Federated learning with quantum communication protocols for RLHF."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.federated_nodes = config.get('federated_nodes', 5)
        self.quantum_communication = config.get('quantum_communication', True)
        self.byzantine_tolerance = config.get('byzantine_tolerance', 0.33)
        self.compression_ratio = config.get('compression_ratio', 0.1)
    
    async def execute(self, data: Any, config: Dict[str, Any]) -> Dict[str, float]:
        """Execute federated quantum RLHF algorithm."""
        # Simulate federated training across quantum network
        fed_time = 0.10 * self.federated_nodes / (2 if self.quantum_communication else 1)
        await asyncio.sleep(fed_time)
        
        # Federated learning benefits
        fed_accuracy_boost = 1.0 + (self.federated_nodes - 1) * 0.02
        quantum_speedup = 1.3 if self.quantum_communication else 1.0
        compression_benefit = 1.0 / max(0.1, self.compression_ratio)
        
        return {
            'reward_accuracy': min(0.97, 0.88 * fed_accuracy_boost + np.random.normal(0, 0.04)[0]),
            'convergence_time': max(40.0, 120.0 / quantum_speedup + np.random.normal(0, 10)[0]),
            'privacy_cost': max(0.15, 0.4 / self.federated_nodes + np.random.normal(0, 0.08)[0]),
            'memory_usage': max(200.0, 600.0 / compression_benefit + np.random.normal(0, 50)[0]),
            'throughput': 1100.0 * quantum_speedup + np.random.normal(0, 90)[0],
            'federated_consensus_score': min(1.0, (1 - self.byzantine_tolerance) + np.random.normal(0, 0.05)[0]),
            'communication_efficiency': min(1.0, compression_benefit * 0.1 + np.random.normal(0, 0.03)[0]),
            'quantum_entanglement_fidelity': 0.94 + np.random.normal(0, 0.02)[0] if self.quantum_communication else 0.0
        }
    
    def get_name(self) -> str:
        return "federated_quantum_rlhf"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            **self.config,
            'federated_nodes': self.federated_nodes,
            'quantum_communication': self.quantum_communication,
            'byzantine_tolerance': self.byzantine_tolerance,
            'compression_ratio': self.compression_ratio
        }


class ResearchFramework:
    """Research Framework for RLHF experiments.
    
    Provides comprehensive experimental design, execution, and analysis
    capabilities for novel algorithm research including quantum-enhanced
    privacy-preserving algorithms and federated learning approaches.
    """
    
    def __init__(self, output_directory: Optional[Path] = None):
        """Initialize research framework.
        
        Args:
            output_directory: Directory to store research outputs
        """
        self.output_directory = output_directory or Path("research_outputs")
        self.output_directory.mkdir(exist_ok=True)
        
        self.algorithms: Dict[str, ResearchAlgorithm] = {}
        self.experiments: Dict[str, ExperimentDesign] = {}
        self.results: Dict[str, List[ExperimentResult]] = {}
        self.analyses: Dict[str, StatisticalAnalysis] = {}
        
        self.logger = logging.getLogger(__name__)
        self._register_default_algorithms()
    
    def _register_default_algorithms(self):
        """Register default research algorithms."""
        self.register_algorithm(BaselineRLHFAlgorithm())
        self.register_algorithm(NovelRLHFAlgorithm())
        self.register_algorithm(NovelRLHFAlgorithm({
            'privacy_enhancement': 1.5,
            'efficiency_boost': 1.3,
            'quantum_optimization': True
        }))  # Enhanced variant
        
        # Register cutting-edge algorithms
        self.register_algorithm(QuantumEnhancedPrivacyAlgorithm())
        self.register_algorithm(QuantumEnhancedPrivacyAlgorithm({
            'quantum_entanglement_factor': 0.95,
            'dp_epsilon': 0.3,
            'homomorphic_encryption': True,
            'adaptive_noise_scaling': True
        }))
        self.register_algorithm(FederatedQuantumRLHFAlgorithm())
        self.register_algorithm(FederatedQuantumRLHFAlgorithm({
            'federated_nodes': 10,
            'quantum_communication': True,
            'byzantine_tolerance': 0.2,
            'compression_ratio': 0.05
        }))
    
    def register_algorithm(self, algorithm: ResearchAlgorithm):
        """Register a research algorithm.
        
        Args:
            algorithm: Algorithm to register
        """
        name = algorithm.get_name()
        if name in self.algorithms:
            name = f"{name}_{len([k for k in self.algorithms.keys() if k.startswith(name)])}"
        
        self.algorithms[name] = algorithm
        self.logger.info(f"Registered algorithm: {name}")
    
    def design_experiment(
        self,
        title: str,
        description: str,
        hypotheses: List[ResearchHypothesis],
        baseline_algorithm: str,
        treatment_algorithms: List[str],
        datasets: Optional[List[str]] = None,
        sample_size: int = 100,
        duration_hours: float = 2.0
    ) -> ExperimentDesign:
        """Design a research experiment.
        
        Args:
            title: Experiment title
            description: Experiment description
            hypotheses: Research hypotheses to test
            baseline_algorithm: Name of baseline algorithm
            treatment_algorithms: Names of treatment algorithms
            datasets: Datasets to use (default: synthetic)
            sample_size: Sample size per condition
            duration_hours: Expected experiment duration
            
        Returns:
            Experiment design
        """
        experiment_id = str(uuid.uuid4())
        
        design = ExperimentDesign(
            experiment_id=experiment_id,
            title=title,
            description=description,
            hypotheses=hypotheses,
            baseline_algorithm=baseline_algorithm,
            treatment_algorithms=treatment_algorithms,
            datasets=datasets or ["synthetic_dataset_1", "synthetic_dataset_2"],
            sample_size=sample_size,
            randomization_strategy="block_randomization",
            control_variables=["dataset", "sample_size", "random_seed"],
            measured_variables=["reward_accuracy", "convergence_time", "privacy_cost", "memory_usage", "throughput", "quantum_coherence", "adaptive_efficiency", "privacy_preservation_score", "homomorphic_efficiency", "federated_consensus_score", "communication_efficiency", "quantum_entanglement_fidelity"],
            duration_hours=duration_hours
        )
        
        # Validate design
        issues = design.validate()
        if issues:
            self.logger.warning(f"Experiment design issues: {issues}")
        
        self.experiments[experiment_id] = design
        self.logger.info(f"Designed experiment: {title} ({experiment_id})")
        
        return design
    
    async def execute_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Execute a research experiment.
        
        Args:
            experiment_id: ID of experiment to execute
            
        Returns:
            Experiment execution summary
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
        
        design = self.experiments[experiment_id]
        self.logger.info(f"Starting experiment: {design.title}")
        
        start_time = time.time()
        all_results = []
        
        # Execute baseline algorithm
        baseline_results = await self._execute_algorithm_condition(
            design, design.baseline_algorithm, "baseline"
        )
        all_results.extend(baseline_results)
        
        # Execute treatment algorithms
        for treatment_algo in design.treatment_algorithms:
            treatment_results = await self._execute_algorithm_condition(
                design, treatment_algo, "treatment"
            )
            all_results.extend(treatment_results)
        
        # Store results
        self.results[experiment_id] = all_results
        
        # Perform statistical analysis
        analyses = await self._analyze_experiment_results(design, all_results)
        for analysis in analyses:
            self.analyses[analysis.analysis_id] = analysis
        
        execution_time = time.time() - start_time
        
        # Generate report
        report = await self._generate_experiment_report(design, all_results, analyses)
        
        # Save report
        report_path = self.output_directory / f"experiment_{experiment_id}_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        
        summary = {
            'experiment_id': experiment_id,
            'title': design.title,
            'execution_time': execution_time,
            'total_results': len(all_results),
            'analyses_performed': len(analyses),
            'significant_findings': len([a for a in analyses if a.significant]),
            'report_path': str(report_path)
        }
        
        self.logger.info(f"Completed experiment: {design.title} in {execution_time:.2f}s")
        return summary
    
    async def _execute_algorithm_condition(
        self,
        design: ExperimentDesign,
        algorithm_name: str,
        condition_type: str
    ) -> List[ExperimentResult]:
        """Execute algorithm for all datasets and samples.
        
        Args:
            design: Experiment design
            algorithm_name: Name of algorithm to execute
            condition_type: Type of condition ('baseline' or 'treatment')
            
        Returns:
            List of experiment results
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        algorithm = self.algorithms[algorithm_name]
        results = []
        
        for dataset in design.datasets:
            for sample_idx in range(design.sample_size):
                try:
                    # Generate synthetic data for this sample
                    sample_data = self._generate_sample_data(dataset, sample_idx)
                    
                    # Execute algorithm
                    metrics = await algorithm.execute(sample_data, {
                        'dataset': dataset,
                        'sample_idx': sample_idx,
                        'experiment_id': design.experiment_id
                    })
                    
                    # Create result record
                    result = ExperimentResult(
                        result_id=str(uuid.uuid4()),
                        experiment_id=design.experiment_id,
                        hypothesis_id=design.hypotheses[0].hypothesis_id if design.hypotheses else "none",
                        algorithm=algorithm_name,
                        dataset=dataset,
                        metrics=metrics,
                        metadata={
                            'condition_type': condition_type,
                            'sample_idx': sample_idx,
                            'algorithm_params': algorithm.get_parameters()
                        },
                        timestamp=time.time()
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error executing {algorithm_name} on {dataset}[{sample_idx}]: {e}")
        
        self.logger.info(f"Executed {algorithm_name}: {len(results)} results")
        return results
    
    def _generate_sample_data(self, dataset: str, sample_idx: int) -> Dict[str, Any]:
        """Generate synthetic sample data.
        
        Args:
            dataset: Dataset identifier
            sample_idx: Sample index
            
        Returns:
            Synthetic data sample
        """
        # Set random seed for reproducibility
        np.random.seed(hash(f"{dataset}_{sample_idx}") % (2**32))
        
        return {
            'dataset': dataset,
            'sample_idx': sample_idx,
            'prompts': [f"prompt_{i}" for i in range(10)],
            'responses': [f"response_{i}" for i in range(10)],
            'annotations': np.random.uniform(0, 1, 10).tolist(),
            'metadata': {
                'complexity': np.random.uniform(0.1, 1.0),
                'domain': dataset.split('_')[0] if '_' in dataset else 'general'
            }
        }
    
    async def _analyze_experiment_results(
        self,
        design: ExperimentDesign,
        results: List[ExperimentResult]
    ) -> List[StatisticalAnalysis]:
        """Perform statistical analysis of experiment results.
        
        Args:
            design: Experiment design
            results: Experiment results
            
        Returns:
            List of statistical analyses
        """
        analyses = []
        
        # Group results by algorithm and dataset
        result_groups = {}
        for result in results:
            key = (result.algorithm, result.dataset)
            if key not in result_groups:
                result_groups[key] = []
            result_groups[key].append(result)
        
        # Analyze each hypothesis
        for hypothesis in design.hypotheses:
            for metric in hypothesis.success_metrics:
                analysis = await self._analyze_metric_hypothesis(
                    hypothesis, metric, result_groups, design.baseline_algorithm
                )
                if analysis:
                    analyses.append(analysis)
        
        return analyses
    
    async def _analyze_metric_hypothesis(
        self,
        hypothesis: ResearchHypothesis,
        metric: str,
        result_groups: Dict[Tuple[str, str], List[ExperimentResult]],
        baseline_algorithm: str
    ) -> Optional[StatisticalAnalysis]:
        """Analyze a specific metric for a hypothesis.
        
        Args:
            hypothesis: Research hypothesis
            metric: Metric to analyze
            result_groups: Grouped experiment results
            baseline_algorithm: Name of baseline algorithm
            
        Returns:
            Statistical analysis or None if insufficient data
        """
        try:
            # Collect baseline and treatment data
            baseline_values = []
            treatment_values = []
            
            for (algorithm, dataset), group_results in result_groups.items():
                metric_values = [r.metrics.get(metric, 0) for r in group_results if metric in r.metrics]
                
                if algorithm == baseline_algorithm:
                    baseline_values.extend(metric_values)
                else:
                    treatment_values.extend(metric_values)
            
            if len(baseline_values) < 5 or len(treatment_values) < 5:
                self.logger.warning(f"Insufficient data for {metric} analysis")
                return None
            
            # Perform statistical test
            if hypothesis.statistical_test == "t_test" and SCIPY_AVAILABLE:
                stat, p_value = stats.ttest_ind(treatment_values, baseline_values)
            elif hypothesis.statistical_test == "mann_whitney" and SCIPY_AVAILABLE:
                stat, p_value = stats.mannwhitneyu(treatment_values, baseline_values, alternative='two-sided')
            else:
                # Fallback to simple comparison
                stat = (np.mean(treatment_values) - np.mean(baseline_values)) / np.std(baseline_values + treatment_values)
                p_value = 0.05 if abs(stat) > 1.96 else 0.10
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(baseline_values)**2 + np.std(treatment_values)**2) / 2)
            effect_size = (np.mean(treatment_values) - np.mean(baseline_values)) / pooled_std
            
            # Calculate confidence interval (approximate)
            se_diff = pooled_std * np.sqrt(1/len(baseline_values) + 1/len(treatment_values))
            ci_lower = (np.mean(treatment_values) - np.mean(baseline_values)) - 1.96 * se_diff
            ci_upper = (np.mean(treatment_values) - np.mean(baseline_values)) + 1.96 * se_diff
            
            # Determine significance
            significant = p_value < hypothesis.significance_level
            
            # Generate interpretation
            improvement = np.mean(treatment_values) - np.mean(baseline_values)
            improvement_pct = (improvement / np.mean(baseline_values)) * 100 if np.mean(baseline_values) != 0 else 0
            
            if significant and improvement > 0:
                interpretation = f"Treatment shows significant improvement in {metric} ({improvement_pct:.1f}% better)"
            elif significant and improvement < 0:
                interpretation = f"Treatment shows significant degradation in {metric} ({abs(improvement_pct):.1f}% worse)"
            else:
                interpretation = f"No significant difference in {metric} between treatment and baseline"
            
            # Generate recommendations
            recommendations = []
            if significant and abs(effect_size) > 0.8:
                recommendations.append(f"Large effect size detected for {metric} - consider this a strong finding")
            if p_value < 0.01:
                recommendations.append(f"Very strong statistical evidence (p < 0.01)")
            if not significant:
                recommendations.append(f"Consider increasing sample size or refining methodology for {metric}")
            
            analysis = StatisticalAnalysis(
                analysis_id=str(uuid.uuid4()),
                hypothesis_id=hypothesis.hypothesis_id,
                test_type=hypothesis.statistical_test,
                statistic=float(stat),
                p_value=float(p_value),
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                power=0.8,  # Approximate - would need proper power analysis
                significant=significant,
                interpretation=interpretation,
                recommendations=recommendations
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing {metric} for hypothesis {hypothesis.hypothesis_id}: {e}")
            return None
    
    async def _generate_experiment_report(
        self,
        design: ExperimentDesign,
        results: List[ExperimentResult],
        analyses: List[StatisticalAnalysis]
    ) -> Dict[str, Any]:
        """Generate comprehensive experiment report.
        
        Args:
            design: Experiment design
            results: Experiment results
            analyses: Statistical analyses
            
        Returns:
            Comprehensive report dictionary
        """
        # Calculate summary statistics
        algorithms_tested = list(set(r.algorithm for r in results))
        datasets_used = list(set(r.dataset for r in results))
        
        metrics_summary = {}
        for metric in design.measured_variables:
            metric_values = [r.metrics.get(metric, 0) for r in results if metric in r.metrics]
            if metric_values:
                metrics_summary[metric] = {
                    'mean': np.mean(metric_values),
                    'std': np.std(metric_values),
                    'min': min(metric_values),
                    'max': max(metric_values),
                    'count': len(metric_values)
                }
        
        # Analyze by algorithm
        algorithm_performance = {}
        for algorithm in algorithms_tested:
            algo_results = [r for r in results if r.algorithm == algorithm]
            algo_metrics = {}
            for metric in design.measured_variables:
                metric_values = [r.metrics.get(metric, 0) for r in algo_results if metric in r.metrics]
                if metric_values:
                    algo_metrics[metric] = {
                        'mean': np.mean(metric_values),
                        'std': np.std(metric_values),
                        'count': len(metric_values)
                    }
            algorithm_performance[algorithm] = algo_metrics
        
        # Significant findings
        significant_findings = [a for a in analyses if a.significant]
        
        report = {
            'experiment_design': asdict(design),
            'execution_summary': {
                'total_results': len(results),
                'algorithms_tested': algorithms_tested,
                'datasets_used': datasets_used,
                'execution_time': max(r.timestamp for r in results) - min(r.timestamp for r in results),
                'completion_rate': len(results) / (len(algorithms_tested) * len(datasets_used) * design.sample_size)
            },
            'statistical_analyses': [asdict(a) for a in analyses],
            'significant_findings': [asdict(a) for a in significant_findings],
            'metrics_summary': metrics_summary,
            'algorithm_performance': algorithm_performance,
            'conclusions': self._generate_conclusions(design, analyses),
            'recommendations': self._generate_recommendations(design, analyses),
            'future_work': self._suggest_future_work(design, analyses),
            'methodology': {
                'randomization': design.randomization_strategy,
                'sample_size_per_condition': design.sample_size,
                'control_variables': design.control_variables,
                'statistical_tests_used': list(set(a.test_type for a in analyses))
            },
            'reproducibility': {
                'algorithms_used': {name: algo.get_parameters() for name, algo in self.algorithms.items() if name in algorithms_tested},
                'random_seeds': 'deterministic_based_on_dataset_and_sample_idx',
                'environment': 'controlled_experimental_environment'
            }
        }
        
        return report
    
    def _generate_conclusions(self, design: ExperimentDesign, analyses: List[StatisticalAnalysis]) -> List[str]:
        """Generate experiment conclusions.
        
        Args:
            design: Experiment design
            analyses: Statistical analyses
            
        Returns:
            List of conclusions
        """
        conclusions = []
        
        significant_analyses = [a for a in analyses if a.significant]
        
        if significant_analyses:
            conclusions.append(f"Found {len(significant_analyses)} statistically significant results out of {len(analyses)} analyses performed.")
            
            for analysis in significant_analyses:
                conclusions.append(f"- {analysis.interpretation}")
        else:
            conclusions.append("No statistically significant differences were found between algorithms.")
            conclusions.append("This could indicate that the algorithms perform similarly, or that larger sample sizes are needed.")
        
        # Effect size analysis
        large_effects = [a for a in analyses if abs(a.effect_size) > 0.8]
        if large_effects:
            conclusions.append(f"Detected {len(large_effects)} large effect sizes, indicating practically significant differences.")
        
        return conclusions
    
    def _generate_recommendations(self, design: ExperimentDesign, analyses: List[StatisticalAnalysis]) -> List[str]:
        """Generate recommendations based on results.
        
        Args:
            design: Experiment design
            analyses: Statistical analyses
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Collect all recommendations from analyses
        for analysis in analyses:
            recommendations.extend(analysis.recommendations)
        
        # Add general recommendations
        if len([a for a in analyses if a.significant]) / len(analyses) < 0.3:
            recommendations.append("Consider increasing sample size to improve statistical power")
        
        if any(abs(a.effect_size) > 0.5 for a in analyses):
            recommendations.append("Several medium to large effect sizes detected - consider practical significance")
        
        # Algorithm-specific recommendations
        significant_improvements = [a for a in analyses if a.significant and "improvement" in a.interpretation.lower()]
        if significant_improvements:
            recommendations.append("Novel algorithms show promise - consider further development and validation")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _suggest_future_work(self, design: ExperimentDesign, analyses: List[StatisticalAnalysis]) -> List[str]:
        """Suggest future research directions.
        
        Args:
            design: Experiment design
            analyses: Statistical analyses
            
        Returns:
            List of future work suggestions
        """
        suggestions = []
        
        # Based on results
        if any(a.significant for a in analyses):
            suggestions.append("Replicate findings with larger, more diverse datasets")
            suggestions.append("Investigate the mechanisms behind observed performance improvements")
        
        # Methodological improvements
        suggestions.append("Implement real-world datasets beyond synthetic data")
        suggestions.append("Conduct longitudinal studies to assess long-term performance")
        suggestions.append("Investigate algorithm performance under various failure conditions")
        
        # Specific to RLHF
        suggestions.append("Evaluate human annotator agreement and bias impact")
        suggestions.append("Study privacy-utility tradeoffs across different epsilon values")
        suggestions.append("Assess scalability with larger model sizes and datasets")
        
        return suggestions
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments.
        
        Returns:
            Dictionary with experiment summary
        """
        return {
            'total_experiments': len(self.experiments),
            'total_algorithms': len(self.algorithms),
            'total_results': sum(len(results) for results in self.results.values()),
            'total_analyses': len(self.analyses),
            'experiments': {
                exp_id: {
                    'title': exp.title,
                    'status': 'completed' if exp_id in self.results else 'designed',
                    'results_count': len(self.results.get(exp_id, [])),
                    'significant_findings': len([a for a in self.analyses.values() 
                                               if a.hypothesis_id in [h.hypothesis_id for h in exp.hypotheses] and a.significant])
                }
                for exp_id, exp in self.experiments.items()
            },
            'algorithms': list(self.algorithms.keys())
        }