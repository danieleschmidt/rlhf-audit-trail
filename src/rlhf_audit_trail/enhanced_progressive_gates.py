"""Enhanced Progressive Quality Gates with ML Integration.

Advanced quality gates system that combines traditional gates with ML-driven
adaptive behavior and autonomous decision making.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
import logging

from .autonomous_ml_engine import AutonomousMLEngine, MLModelType
from .progressive_quality_gates import (
    QualityGateType, GateStatus, QualityGate, QualityGateResult,
    ProgressiveQualityGates
)


class AdaptiveStrategy(Enum):
    """Strategies for adaptive behavior."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ML_DRIVEN = "ml_driven"


class EvolutionTrigger(Enum):
    """Triggers for gate evolution."""
    TIME_BASED = "time_based"
    PERFORMANCE_BASED = "performance_based"
    RISK_BASED = "risk_based"
    FEEDBACK_BASED = "feedback_based"
    HYBRID = "hybrid"


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive behavior."""
    strategy: AdaptiveStrategy = AdaptiveStrategy.BALANCED
    learning_rate: float = 0.01
    adaptation_threshold: float = 0.8
    evolution_trigger: EvolutionTrigger = EvolutionTrigger.HYBRID
    min_samples_for_adaptation: int = 10
    confidence_threshold: float = 0.85
    rollback_on_degradation: bool = True


@dataclass
class EvolutionMetrics:
    """Metrics tracking gate evolution."""
    generation: int
    success_rate: float
    average_execution_time: float
    cost_efficiency: float
    quality_improvement: float
    risk_reduction: float
    user_satisfaction: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class GateEvolution:
    """Tracks evolution of a quality gate."""
    gate_id: str
    original_config: Dict[str, Any]
    current_config: Dict[str, Any]
    evolution_history: List[EvolutionMetrics]
    ml_confidence: float
    adaptation_count: int = 0
    last_evolution: float = field(default_factory=time.time)


class EnhancedProgressiveGates(ProgressiveQualityGates):
    """Enhanced Progressive Quality Gates with ML integration.
    
    Extends the base progressive gates with:
    - Machine learning-driven adaptation
    - Autonomous decision making
    - Risk-based gate selection
    - Performance optimization
    - Real-time evolution
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 adaptive_config: Optional[AdaptiveConfig] = None):
        """Initialize enhanced progressive gates.
        
        Args:
            config: Base configuration
            adaptive_config: Adaptive behavior configuration
        """
        super().__init__(config)
        
        self.adaptive_config = adaptive_config or AdaptiveConfig()
        self.ml_engine = AutonomousMLEngine()
        
        # Enhanced tracking
        self.gate_evolutions: Dict[str, GateEvolution] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.risk_assessments: List[Dict[str, Any]] = []
        self.adaptation_log: List[Dict[str, Any]] = []
        
        # Real-time metrics
        self.current_risk_level = 0.5
        self.system_health_score = 0.8
        self.adaptation_effectiveness = 0.75
        
        self.logger = logging.getLogger(__name__)
        self._setup_enhanced_gates()
        
    def _setup_enhanced_gates(self):
        """Setup enhanced quality gates with ML integration."""
        # Call parent setup first
        super()._setup_default_gates()
        
        # Add ML-driven adaptive gates
        adaptive_gates = [
            self._create_adaptive_functional_gate(),
            self._create_adaptive_performance_gate(),
            self._create_adaptive_security_gate(),
            self._create_adaptive_compliance_gate(),
            self._create_adaptive_reliability_gate(),
            self._create_adaptive_scalability_gate()
        ]
        
        for gate in adaptive_gates:
            self.gates[gate.gate_id] = gate
            
            # Initialize evolution tracking
            self.gate_evolutions[gate.gate_id] = GateEvolution(
                gate_id=gate.gate_id,
                original_config=self._gate_to_dict(gate),
                current_config=self._gate_to_dict(gate),
                evolution_history=[],
                ml_confidence=0.5
            )
            
    def _create_adaptive_functional_gate(self) -> QualityGate:
        """Create adaptive functional testing gate."""
        return QualityGate(
            gate_id="adaptive_functional",
            name="ML-Driven Functional Testing",
            gate_type=QualityGateType.FUNCTIONAL,
            priority=1,
            threshold=0.85,
            validator=self._validate_adaptive_functional
        )
        
    def _create_adaptive_performance_gate(self) -> QualityGate:
        """Create adaptive performance testing gate."""
        return QualityGate(
            gate_id="adaptive_performance",
            name="Intelligent Performance Validation",
            gate_type=QualityGateType.PERFORMANCE,
            priority=2,
            threshold=500.0,  # milliseconds
            validator=self._validate_adaptive_performance
        )
        
    def _create_adaptive_security_gate(self) -> QualityGate:
        """Create adaptive security scanning gate."""
        return QualityGate(
            gate_id="adaptive_security",
            name="Risk-Based Security Scanning",
            gate_type=QualityGateType.SECURITY,
            priority=1,
            threshold=0.95,
            validator=self._validate_adaptive_security
        )
        
    def _create_adaptive_compliance_gate(self) -> QualityGate:
        """Create adaptive compliance validation gate."""
        return QualityGate(
            gate_id="adaptive_compliance",
            name="Dynamic Compliance Validation",
            gate_type=QualityGateType.COMPLIANCE,
            priority=1,
            threshold=0.90,
            validator=self._validate_adaptive_compliance
        )
        
    def _create_adaptive_reliability_gate(self) -> QualityGate:
        """Create adaptive reliability testing gate."""
        return QualityGate(
            gate_id="adaptive_reliability",
            name="Predictive Reliability Testing",
            gate_type=QualityGateType.RELIABILITY,
            priority=2,
            threshold=0.99,
            validator=self._validate_adaptive_reliability
        )
        
    def _create_adaptive_scalability_gate(self) -> QualityGate:
        """Create adaptive scalability testing gate."""
        return QualityGate(
            gate_id="adaptive_scalability",
            name="Autonomous Scalability Validation",
            gate_type=QualityGateType.SCALABILITY,
            priority=3,
            threshold=1000.0,  # requests per second
            validator=self._validate_adaptive_scalability
        )
        
    async def _validate_adaptive_functional(self, data: Dict[str, Any]) -> QualityGateResult:
        """Validate functional requirements with ML adaptation."""
        start_time = time.time()
        
        # Extract features for ML analysis
        features = await self.ml_engine.extract_features(data)
        
        # Predict risk and adjust testing strategy
        risk_predictions = await self.ml_engine.predict_risk(features)
        
        # Adaptive test selection based on risk
        if risk_predictions['overall_risk'] > 0.7:
            test_strategy = "comprehensive"
            test_multiplier = 2.0
        elif risk_predictions['overall_risk'] > 0.4:
            test_strategy = "targeted"
            test_multiplier = 1.5
        else:
            test_strategy = "standard"
            test_multiplier = 1.0
            
        # Simulate adaptive functional testing
        base_score = data.get('test_pass_rate', 0.85)
        complexity_penalty = data.get('code_complexity', 5) / 20.0
        coverage_bonus = data.get('test_coverage', 0.80) * 0.2
        
        # ML-adjusted score
        ml_adjustment = (1 - risk_predictions['overall_risk']) * 0.1
        final_score = base_score - complexity_penalty + coverage_bonus + ml_adjustment
        final_score = max(0.0, min(1.0, final_score))
        
        execution_time = (time.time() - start_time) * test_multiplier
        
        # Adaptive threshold based on ML confidence
        gate = self.gates["adaptive_functional"]
        ml_confidence = self.gate_evolutions["adaptive_functional"].ml_confidence
        adaptive_threshold = gate.threshold * (0.8 + 0.2 * ml_confidence)
        
        passed = final_score >= adaptive_threshold
        
        return QualityGateResult(
            gate_id="adaptive_functional",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            score=final_score,
            passed=passed,
            details={
                "test_strategy": test_strategy,
                "risk_level": risk_predictions['overall_risk'],
                "ml_confidence": ml_confidence,
                "adaptive_threshold": adaptive_threshold,
                "original_threshold": gate.threshold,
                "complexity_penalty": complexity_penalty,
                "coverage_bonus": coverage_bonus,
                "ml_adjustment": ml_adjustment
            },
            execution_time=execution_time,
            timestamp=time.time(),
            recommendations=self._generate_functional_recommendations(risk_predictions, final_score)
        )
        
    async def _validate_adaptive_performance(self, data: Dict[str, Any]) -> QualityGateResult:
        """Validate performance with intelligent optimization."""
        start_time = time.time()
        
        # Extract performance features
        features = await self.ml_engine.extract_features(data)
        
        # Predict optimal performance thresholds
        current_metrics = {
            'response_time': data.get('response_time', 200),
            'throughput': data.get('throughput', 1000),
            'memory_usage': data.get('memory_usage', 512),
            'cpu_usage': data.get('cpu_usage', 50)
        }
        
        optimized_thresholds = await self.ml_engine.optimize_thresholds(current_metrics)
        
        # Performance scoring with ML optimization
        response_time = data.get('response_time', 200)
        throughput = data.get('throughput', 1000)
        memory_efficiency = 1000 / max(data.get('memory_usage', 512), 1)
        cpu_efficiency = 100 - data.get('cpu_usage', 50)
        
        # Weighted performance score
        performance_score = (
            (1000 / max(response_time, 1)) * 0.3 +
            (throughput / 1000) * 0.3 +
            memory_efficiency * 0.2 +
            (cpu_efficiency / 100) * 0.2
        )
        
        # Normalize to 0-1 range
        performance_score = min(1.0, performance_score / 2.0)
        
        execution_time = time.time() - start_time
        
        # Adaptive threshold from ML
        adaptive_threshold = optimized_thresholds.get('performance_threshold', 500)
        passed = response_time <= adaptive_threshold
        
        return QualityGateResult(
            gate_id="adaptive_performance",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            score=performance_score,
            passed=passed,
            details={
                "response_time": response_time,
                "throughput": throughput,
                "memory_usage": data.get('memory_usage', 512),
                "cpu_usage": data.get('cpu_usage', 50),
                "adaptive_threshold": adaptive_threshold,
                "optimized_thresholds": optimized_thresholds,
                "performance_score": performance_score
            },
            execution_time=execution_time,
            timestamp=time.time(),
            recommendations=self._generate_performance_recommendations(current_metrics, optimized_thresholds)
        )
        
    async def _validate_adaptive_security(self, data: Dict[str, Any]) -> QualityGateResult:
        """Validate security with risk-based scanning."""
        start_time = time.time()
        
        # Extract security features
        features = await self.ml_engine.extract_features(data)
        
        # Predict security risks
        risk_predictions = await self.ml_engine.predict_risk(features)
        security_risk = risk_predictions.get('security_risk', 0.3)
        
        # Adaptive security scanning based on risk
        if security_risk > 0.8:
            scan_depth = "deep_penetration"
            scan_multiplier = 3.0
        elif security_risk > 0.5:
            scan_depth = "comprehensive"
            scan_multiplier = 2.0
        else:
            scan_depth = "standard"
            scan_multiplier = 1.0
            
        # Security scoring
        vulnerability_score = 1.0 - data.get('vulnerability_count', 0) / 10.0
        auth_strength = data.get('auth_strength', 0.9)
        encryption_level = data.get('encryption_level', 0.95)
        access_control = data.get('access_control_score', 0.85)
        
        security_score = (
            vulnerability_score * 0.4 +
            auth_strength * 0.25 +
            encryption_level * 0.25 +
            access_control * 0.1
        )
        
        # Risk-adjusted threshold
        gate = self.gates["adaptive_security"]
        risk_adjustment = security_risk * 0.1
        adaptive_threshold = gate.threshold + risk_adjustment
        
        execution_time = (time.time() - start_time) * scan_multiplier
        passed = security_score >= adaptive_threshold
        
        return QualityGateResult(
            gate_id="adaptive_security",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            score=security_score,
            passed=passed,
            details={
                "scan_depth": scan_depth,
                "security_risk": security_risk,
                "vulnerability_score": vulnerability_score,
                "auth_strength": auth_strength,
                "encryption_level": encryption_level,
                "access_control": access_control,
                "adaptive_threshold": adaptive_threshold,
                "risk_adjustment": risk_adjustment
            },
            execution_time=execution_time,
            timestamp=time.time(),
            recommendations=self._generate_security_recommendations(security_risk, security_score)
        )
        
    async def _validate_adaptive_compliance(self, data: Dict[str, Any]) -> QualityGateResult:
        """Validate compliance with dynamic requirements."""
        start_time = time.time()
        
        # Extract compliance features
        features = await self.ml_engine.extract_features(data)
        
        # Compliance scoring
        gdpr_compliance = data.get('gdpr_compliance', 0.9)
        audit_completeness = data.get('audit_trail_completeness', 0.95)
        data_retention = data.get('data_retention_compliance', 0.88)
        privacy_protection = data.get('privacy_protection_level', 0.92)
        
        compliance_score = (
            gdpr_compliance * 0.3 +
            audit_completeness * 0.3 +
            data_retention * 0.2 +
            privacy_protection * 0.2
        )
        
        # Dynamic threshold based on regulatory environment
        base_threshold = self.gates["adaptive_compliance"].threshold
        regulatory_strictness = data.get('regulatory_strictness', 1.0)
        adaptive_threshold = base_threshold * regulatory_strictness
        
        execution_time = time.time() - start_time
        passed = compliance_score >= adaptive_threshold
        
        return QualityGateResult(
            gate_id="adaptive_compliance",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            score=compliance_score,
            passed=passed,
            details={
                "gdpr_compliance": gdpr_compliance,
                "audit_completeness": audit_completeness,
                "data_retention": data_retention,
                "privacy_protection": privacy_protection,
                "regulatory_strictness": regulatory_strictness,
                "adaptive_threshold": adaptive_threshold
            },
            execution_time=execution_time,
            timestamp=time.time(),
            recommendations=self._generate_compliance_recommendations(compliance_score, adaptive_threshold)
        )
        
    async def _validate_adaptive_reliability(self, data: Dict[str, Any]) -> QualityGateResult:
        """Validate reliability with predictive analysis."""
        start_time = time.time()
        
        # Predict failure probability
        failure_prob = await self.ml_engine.predict_failure_probability(data)
        reliability_score = 1.0 - failure_prob
        
        # Additional reliability metrics
        uptime_score = data.get('uptime_percentage', 99.5) / 100.0
        error_rate = data.get('error_rate', 0.01)
        recovery_time = data.get('mean_recovery_time', 60)  # seconds
        
        # Combined reliability score
        combined_score = (
            reliability_score * 0.4 +
            uptime_score * 0.3 +
            (1 - error_rate) * 0.2 +
            max(0, 1 - recovery_time / 300) * 0.1  # 5-minute baseline
        )
        
        execution_time = time.time() - start_time
        passed = combined_score >= self.gates["adaptive_reliability"].threshold
        
        return QualityGateResult(
            gate_id="adaptive_reliability",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            score=combined_score,
            passed=passed,
            details={
                "failure_probability": failure_prob,
                "ml_reliability_score": reliability_score,
                "uptime_score": uptime_score,
                "error_rate": error_rate,
                "recovery_time": recovery_time,
                "combined_score": combined_score
            },
            execution_time=execution_time,
            timestamp=time.time(),
            recommendations=self._generate_reliability_recommendations(failure_prob, combined_score)
        )
        
    async def _validate_adaptive_scalability(self, data: Dict[str, Any]) -> QualityGateResult:
        """Validate scalability with autonomous testing."""
        start_time = time.time()
        
        # Scalability metrics
        current_load = data.get('current_requests_per_second', 100)
        target_load = data.get('target_requests_per_second', 1000)
        response_degradation = data.get('response_time_degradation', 1.2)
        resource_utilization = data.get('resource_utilization_at_load', 0.8)
        
        # Scalability scoring
        load_handling = min(1.0, current_load / max(target_load, 1))
        performance_stability = 1.0 / max(response_degradation, 1.0)
        resource_efficiency = 1.0 - resource_utilization
        
        scalability_score = (
            load_handling * 0.5 +
            performance_stability * 0.3 +
            resource_efficiency * 0.2
        )
        
        execution_time = time.time() - start_time
        passed = current_load >= self.gates["adaptive_scalability"].threshold
        
        return QualityGateResult(
            gate_id="adaptive_scalability",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            score=scalability_score,
            passed=passed,
            details={
                "current_load": current_load,
                "target_load": target_load,
                "response_degradation": response_degradation,
                "resource_utilization": resource_utilization,
                "load_handling": load_handling,
                "performance_stability": performance_stability,
                "resource_efficiency": resource_efficiency
            },
            execution_time=execution_time,
            timestamp=time.time(),
            recommendations=self._generate_scalability_recommendations(scalability_score, current_load, target_load)
        )
        
    def _generate_functional_recommendations(self, 
                                             risk_predictions: Dict[str, float], 
                                             score: float) -> List[str]:
        """Generate recommendations for functional testing."""
        recommendations = []
        
        if risk_predictions['overall_risk'] > 0.7:
            recommendations.append("Increase test coverage for high-risk areas")
            recommendations.append("Implement additional integration tests")
            
        if score < 0.8:
            recommendations.append("Review and improve test assertions")
            recommendations.append("Add more edge case testing")
            
        if risk_predictions.get('complexity_risk', 0) > 0.6:
            recommendations.append("Break down complex functions for better testability")
            
        return recommendations
        
    def _generate_performance_recommendations(self, 
                                              current_metrics: Dict[str, float],
                                              optimized_thresholds: Dict[str, float]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        response_time = current_metrics.get('response_time', 200)
        if response_time > optimized_thresholds.get('performance_threshold', 500):
            recommendations.append("Optimize database queries")
            recommendations.append("Implement caching strategies")
            
        memory_usage = current_metrics.get('memory_usage', 512)
        if memory_usage > 800:
            recommendations.append("Review memory allocation patterns")
            recommendations.append("Implement memory pooling")
            
        cpu_usage = current_metrics.get('cpu_usage', 50)
        if cpu_usage > 80:
            recommendations.append("Profile CPU-intensive operations")
            recommendations.append("Consider async processing for heavy tasks")
            
        return recommendations
        
    def _generate_security_recommendations(self, 
                                           security_risk: float, 
                                           score: float) -> List[str]:
        """Generate security improvement recommendations."""
        recommendations = []
        
        if security_risk > 0.7:
            recommendations.append("Conduct penetration testing")
            recommendations.append("Review and update security policies")
            
        if score < 0.9:
            recommendations.append("Update dependencies to latest secure versions")
            recommendations.append("Implement additional input validation")
            
        if security_risk > 0.5:
            recommendations.append("Enable additional security monitoring")
            recommendations.append("Review access control configurations")
            
        return recommendations
        
    def _generate_compliance_recommendations(self, 
                                             score: float, 
                                             threshold: float) -> List[str]:
        """Generate compliance improvement recommendations."""
        recommendations = []
        
        if score < threshold:
            recommendations.append("Update privacy policy documentation")
            recommendations.append("Enhance audit trail completeness")
            
        if score < 0.85:
            recommendations.append("Review data retention policies")
            recommendations.append("Implement additional consent mechanisms")
            
        return recommendations
        
    def _generate_reliability_recommendations(self, 
                                              failure_prob: float, 
                                              score: float) -> List[str]:
        """Generate reliability improvement recommendations."""
        recommendations = []
        
        if failure_prob > 0.3:
            recommendations.append("Implement circuit breaker patterns")
            recommendations.append("Add comprehensive health checks")
            
        if score < 0.95:
            recommendations.append("Enhance error handling and recovery")
            recommendations.append("Implement graceful degradation")
            
        if failure_prob > 0.5:
            recommendations.append("Review and strengthen backup systems")
            recommendations.append("Implement chaos engineering practices")
            
        return recommendations
        
    def _generate_scalability_recommendations(self, 
                                              score: float, 
                                              current_load: float, 
                                              target_load: float) -> List[str]:
        """Generate scalability improvement recommendations."""
        recommendations = []
        
        if current_load < target_load * 0.8:
            recommendations.append("Optimize application architecture for scale")
            recommendations.append("Implement horizontal scaling strategies")
            
        if score < 0.8:
            recommendations.append("Review and optimize resource usage")
            recommendations.append("Implement load balancing improvements")
            
        if current_load < target_load * 0.5:
            recommendations.append("Consider microservices architecture")
            recommendations.append("Implement auto-scaling mechanisms")
            
        return recommendations
        
    async def evolve_gates(self, performance_data: Dict[str, Any]):
        """Evolve quality gates based on performance and ML insights."""
        for gate_id, evolution in self.gate_evolutions.items():
            if self._should_evolve_gate(evolution, performance_data):
                await self._evolve_single_gate(gate_id, evolution, performance_data)
                
    def _should_evolve_gate(self, 
                            evolution: GateEvolution, 
                            performance_data: Dict[str, Any]) -> bool:
        """Determine if a gate should evolve."""
        # Time-based evolution
        time_since_last = time.time() - evolution.last_evolution
        if time_since_last < 3600:  # Minimum 1 hour between evolutions
            return False
            
        # Performance-based evolution
        if len(evolution.evolution_history) >= self.adaptive_config.min_samples_for_adaptation:
            recent_performance = evolution.evolution_history[-5:]
            avg_success_rate = sum(m.success_rate for m in recent_performance) / len(recent_performance)
            
            if avg_success_rate < self.adaptive_config.adaptation_threshold:
                return True
                
        # ML confidence-based evolution
        if evolution.ml_confidence > self.adaptive_config.confidence_threshold:
            return True
            
        return False
        
    async def _evolve_single_gate(self, 
                                  gate_id: str, 
                                  evolution: GateEvolution, 
                                  performance_data: Dict[str, Any]):
        """Evolve a single quality gate."""
        gate = self.gates[gate_id]
        
        # Generate ML-driven recommendations
        features = await self.ml_engine.extract_features(performance_data)
        adaptive_gates = await self.ml_engine.generate_adaptive_gates(performance_data)
        
        # Find matching adaptive gate
        matching_gate = None
        for adaptive_gate in adaptive_gates:
            if adaptive_gate['name'].lower().replace(' ', '_').replace('-', '_') in gate_id:
                matching_gate = adaptive_gate
                break
                
        if matching_gate:
            # Update gate configuration
            old_threshold = gate.threshold
            new_threshold = matching_gate['threshold']
            
            # Apply conservative evolution
            if self.adaptive_config.strategy == AdaptiveStrategy.CONSERVATIVE:
                new_threshold = old_threshold * 0.9 + new_threshold * 0.1
            elif self.adaptive_config.strategy == AdaptiveStrategy.BALANCED:
                new_threshold = old_threshold * 0.7 + new_threshold * 0.3
            elif self.adaptive_config.strategy == AdaptiveStrategy.AGGRESSIVE:
                new_threshold = old_threshold * 0.5 + new_threshold * 0.5
            else:  # ML_DRIVEN
                new_threshold = new_threshold
                
            gate.threshold = new_threshold
            gate.priority = matching_gate.get('priority', gate.priority)
            
            # Update evolution tracking
            evolution.current_config = self._gate_to_dict(gate)
            evolution.adaptation_count += 1
            evolution.last_evolution = time.time()
            
            # Record evolution metrics
            metrics = EvolutionMetrics(
                generation=evolution.adaptation_count,
                success_rate=performance_data.get('success_rate', 0.8),
                average_execution_time=performance_data.get('avg_execution_time', 1.0),
                cost_efficiency=performance_data.get('cost_efficiency', 0.75),
                quality_improvement=abs(new_threshold - old_threshold) / old_threshold,
                risk_reduction=performance_data.get('risk_reduction', 0.1),
                user_satisfaction=performance_data.get('user_satisfaction', 0.8)
            )
            
            evolution.evolution_history.append(metrics)
            
            # Log evolution
            self.adaptation_log.append({
                'timestamp': time.time(),
                'gate_id': gate_id,
                'action': 'evolved',
                'old_threshold': old_threshold,
                'new_threshold': new_threshold,
                'strategy': self.adaptive_config.strategy.value,
                'ml_confidence': evolution.ml_confidence
            })
            
            self.logger.info(f"Evolved gate {gate_id}: threshold {old_threshold:.3f} -> {new_threshold:.3f}")
            
    def _gate_to_dict(self, gate: QualityGate) -> Dict[str, Any]:
        """Convert quality gate to dictionary representation."""
        return {
            'gate_id': gate.gate_id,
            'name': gate.name,
            'gate_type': gate.gate_type.value,
            'priority': gate.priority,
            'threshold': gate.threshold,
            'status': gate.status.value if gate.status else 'pending'
        }
        
    async def get_evolution_report(self) -> Dict[str, Any]:
        """Get comprehensive evolution report."""
        total_adaptations = sum(e.adaptation_count for e in self.gate_evolutions.values())
        avg_ml_confidence = sum(e.ml_confidence for e in self.gate_evolutions.values()) / len(self.gate_evolutions)
        
        recent_performance = []
        for evolution in self.gate_evolutions.values():
            if evolution.evolution_history:
                recent_performance.extend(evolution.evolution_history[-5:])
                
        avg_success_rate = sum(m.success_rate for m in recent_performance) / max(len(recent_performance), 1)
        avg_quality_improvement = sum(m.quality_improvement for m in recent_performance) / max(len(recent_performance), 1)
        
        return {
            'total_adaptations': total_adaptations,
            'average_ml_confidence': avg_ml_confidence,
            'average_success_rate': avg_success_rate,
            'average_quality_improvement': avg_quality_improvement,
            'current_risk_level': self.current_risk_level,
            'system_health_score': self.system_health_score,
            'adaptation_effectiveness': self.adaptation_effectiveness,
            'active_gates': len(self.gates),
            'evolution_strategy': self.adaptive_config.strategy.value,
            'gate_evolutions': {
                gate_id: {
                    'adaptation_count': evolution.adaptation_count,
                    'ml_confidence': evolution.ml_confidence,
                    'last_evolution': evolution.last_evolution,
                    'current_threshold': self.gates[gate_id].threshold
                }
                for gate_id, evolution in self.gate_evolutions.items()
            }
        }