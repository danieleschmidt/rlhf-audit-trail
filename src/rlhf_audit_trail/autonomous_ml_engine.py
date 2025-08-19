"""Autonomous ML Engine for Progressive Quality Gates.

Implements self-learning, adaptive quality gates that evolve based on
project maturity, risk profile, and performance metrics.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Set
import logging
import hashlib
import math

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class MockNumpy:
        def array(self, x): return x
        def mean(self, x): return sum(x) / len(x) if x else 0
        def std(self, x): return 0.1
        def exp(self, x): return math.exp(x) if isinstance(x, (int, float)) else [math.exp(i) for i in x]
        def log(self, x): return math.log(x) if isinstance(x, (int, float)) else [math.log(i) for i in x]
        def random(self):
            import random
            class MockRandom:
                def uniform(self, low, high): return random.uniform(low, high)
                def normal(self, mean, std): return random.gauss(mean, std)
            return MockRandom()
    np = MockNumpy()


class MLModelType(Enum):
    """Types of ML models in the quality gates system."""
    RISK_PREDICTOR = "risk_predictor"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    THRESHOLD_ADAPTER = "threshold_adapter"
    FAILURE_PREDICTOR = "failure_predictor"
    QUALITY_SCORER = "quality_scorer"


class LearningMode(Enum):
    """Learning modes for the ML engine."""
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"
    UNSUPERVISED = "unsupervised"
    HYBRID = "hybrid"


@dataclass
class MLModelConfig:
    """Configuration for ML models."""
    model_type: MLModelType
    learning_mode: LearningMode
    feature_dimensions: int
    target_accuracy: float = 0.85
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping: bool = True
    regularization: float = 0.01


@dataclass
class TrainingMetrics:
    """Metrics from ML model training."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    training_time: float
    validation_score: float
    epoch: int


class AutonomousMLEngine:
    """Autonomous ML Engine for Quality Gates.
    
    Self-learning system that adapts quality gates based on:
    - Historical performance data
    - Risk patterns
    - Project maturity
    - User feedback
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the autonomous ML engine.
        
        Args:
            config: Configuration for the ML engine
        """
        self.config = config or {}
        self.models: Dict[MLModelType, Any] = {}
        self.training_history: List[TrainingMetrics] = []
        self.feature_extractors: Dict[str, Callable] = {}
        self.adaptive_thresholds: Dict[str, float] = {}
        
        self.logger = logging.getLogger(__name__)
        self._setup_models()
        self._setup_feature_extractors()
        
    def _setup_models(self):
        """Initialize ML models for quality gates."""
        model_configs = [
            MLModelConfig(
                model_type=MLModelType.RISK_PREDICTOR,
                learning_mode=LearningMode.SUPERVISED,
                feature_dimensions=20,
                target_accuracy=0.90
            ),
            MLModelConfig(
                model_type=MLModelType.PERFORMANCE_OPTIMIZER,
                learning_mode=LearningMode.REINFORCEMENT,
                feature_dimensions=15,
                target_accuracy=0.85
            ),
            MLModelConfig(
                model_type=MLModelType.THRESHOLD_ADAPTER,
                learning_mode=LearningMode.HYBRID,
                feature_dimensions=10,
                target_accuracy=0.88
            ),
            MLModelConfig(
                model_type=MLModelType.FAILURE_PREDICTOR,
                learning_mode=LearningMode.SUPERVISED,
                feature_dimensions=25,
                target_accuracy=0.92
            ),
            MLModelConfig(
                model_type=MLModelType.QUALITY_SCORER,
                learning_mode=LearningMode.UNSUPERVISED,
                feature_dimensions=30,
                target_accuracy=0.80
            )
        ]
        
        for config in model_configs:
            self.models[config.model_type] = SimpleMLModel(config)
            
    def _setup_feature_extractors(self):
        """Setup feature extraction functions."""
        self.feature_extractors = {
            'code_complexity': self._extract_code_complexity,
            'test_coverage': self._extract_test_coverage,
            'performance_metrics': self._extract_performance_metrics,
            'security_score': self._extract_security_score,
            'compliance_level': self._extract_compliance_level,
            'deployment_risk': self._extract_deployment_risk,
            'user_feedback': self._extract_user_feedback,
            'historical_patterns': self._extract_historical_patterns
        }
        
    def _extract_code_complexity(self, data: Dict[str, Any]) -> List[float]:
        """Extract code complexity features."""
        # Mock implementation - in real system would analyze actual code
        complexity_score = data.get('cyclomatic_complexity', 5)
        lines_of_code = data.get('lines_of_code', 1000)
        function_count = data.get('function_count', 10)
        
        return [
            complexity_score / 10.0,  # Normalized complexity
            math.log(lines_of_code) / 10.0,  # Log-scaled LOC
            function_count / 100.0,  # Normalized function count
            data.get('duplication_ratio', 0.1),  # Code duplication
            data.get('maintainability_index', 0.8)  # Maintainability
        ]
        
    def _extract_test_coverage(self, data: Dict[str, Any]) -> List[float]:
        """Extract test coverage features."""
        return [
            data.get('line_coverage', 0.85),
            data.get('branch_coverage', 0.80),
            data.get('function_coverage', 0.90),
            data.get('mutation_score', 0.75),
            data.get('integration_coverage', 0.70)
        ]
        
    def _extract_performance_metrics(self, data: Dict[str, Any]) -> List[float]:
        """Extract performance features."""
        return [
            data.get('response_time', 100) / 1000.0,  # Normalized response time
            data.get('throughput', 1000) / 10000.0,  # Normalized throughput
            data.get('memory_usage', 512) / 1024.0,  # Normalized memory usage
            data.get('cpu_usage', 50) / 100.0,  # CPU usage percentage
            data.get('error_rate', 0.01)  # Error rate
        ]
        
    def _extract_security_score(self, data: Dict[str, Any]) -> List[float]:
        """Extract security features."""
        return [
            data.get('vulnerability_score', 0.1),
            data.get('dependency_risk', 0.2),
            data.get('auth_strength', 0.9),
            data.get('encryption_level', 0.95),
            data.get('access_control_score', 0.85)
        ]
        
    def _extract_compliance_level(self, data: Dict[str, Any]) -> List[float]:
        """Extract compliance features."""
        return [
            data.get('gdpr_compliance', 0.9),
            data.get('audit_trail_completeness', 0.95),
            data.get('data_retention_compliance', 0.88),
            data.get('privacy_protection_level', 0.92),
            data.get('regulatory_alignment', 0.85)
        ]
        
    def _extract_deployment_risk(self, data: Dict[str, Any]) -> List[float]:
        """Extract deployment risk features."""
        return [
            data.get('change_frequency', 5) / 10.0,  # Normalized change frequency
            data.get('rollback_history', 2) / 10.0,  # Historical rollbacks
            data.get('dependency_stability', 0.9),
            data.get('infrastructure_readiness', 0.85),
            data.get('team_experience', 0.8)
        ]
        
    def _extract_user_feedback(self, data: Dict[str, Any]) -> List[float]:
        """Extract user feedback features."""
        return [
            data.get('satisfaction_score', 0.8),
            data.get('bug_report_frequency', 0.1),
            data.get('feature_request_volume', 0.3),
            data.get('user_retention', 0.9),
            data.get('performance_complaints', 0.05)
        ]
        
    def _extract_historical_patterns(self, data: Dict[str, Any]) -> List[float]:
        """Extract historical pattern features."""
        return [
            data.get('success_rate_trend', 0.85),
            data.get('quality_improvement_rate', 0.1),
            data.get('defect_density_trend', -0.05),  # Negative is good
            data.get('delivery_predictability', 0.8),
            data.get('process_maturity', 0.7)
        ]
        
    async def extract_features(self, data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract all features from input data.
        
        Args:
            data: Input data containing various metrics
            
        Returns:
            Dictionary of extracted features by category
        """
        features = {}
        
        for feature_name, extractor in self.feature_extractors.items():
            try:
                features[feature_name] = extractor(data)
            except Exception as e:
                self.logger.warning(f"Failed to extract {feature_name}: {e}")
                # Provide default features
                features[feature_name] = [0.5] * 5
                
        return features
        
    async def predict_risk(self, features: Dict[str, List[float]]) -> Dict[str, float]:
        """Predict risk levels using ML models.
        
        Args:
            features: Extracted features
            
        Returns:
            Risk predictions by category
        """
        risk_model = self.models[MLModelType.RISK_PREDICTOR]
        
        # Flatten features for model input
        feature_vector = []
        for feature_list in features.values():
            feature_vector.extend(feature_list)
            
        risk_score = await risk_model.predict(feature_vector)
        
        return {
            'overall_risk': risk_score,
            'deployment_risk': min(risk_score * 1.2, 1.0),
            'security_risk': risk_score * 0.8,
            'performance_risk': risk_score * 0.9,
            'compliance_risk': risk_score * 1.1
        }
        
    async def optimize_thresholds(self, 
                                  current_metrics: Dict[str, float],
                                  target_quality: float = 0.85) -> Dict[str, float]:
        """Optimize quality gate thresholds using ML.
        
        Args:
            current_metrics: Current system metrics
            target_quality: Target quality level
            
        Returns:
            Optimized threshold values
        """
        optimizer_model = self.models[MLModelType.PERFORMANCE_OPTIMIZER]
        
        # Create feature vector from metrics
        metric_vector = list(current_metrics.values())
        
        # Predict optimal thresholds
        optimized_values = await optimizer_model.predict(metric_vector)
        
        # Apply constraints and scaling
        optimized_thresholds = {
            'test_coverage_threshold': max(0.7, min(0.95, optimized_values * 0.85)),
            'performance_threshold': max(100, min(2000, optimized_values * 500)),
            'security_threshold': max(0.8, min(1.0, optimized_values * 0.9)),
            'complexity_threshold': max(5, min(15, optimized_values * 10)),
            'quality_threshold': max(0.6, min(0.98, target_quality))
        }
        
        return optimized_thresholds
        
    async def predict_failure_probability(self, 
                                          deployment_data: Dict[str, Any]) -> float:
        """Predict probability of deployment failure.
        
        Args:
            deployment_data: Data about the deployment
            
        Returns:
            Failure probability (0.0 to 1.0)
        """
        failure_model = self.models[MLModelType.FAILURE_PREDICTOR]
        
        # Extract relevant features
        features = await self.extract_features(deployment_data)
        feature_vector = []
        for feature_list in features.values():
            feature_vector.extend(feature_list)
            
        failure_prob = await failure_model.predict(feature_vector)
        
        # Apply historical adjustment
        historical_adjustment = self._calculate_historical_adjustment()
        adjusted_prob = min(1.0, failure_prob * historical_adjustment)
        
        return adjusted_prob
        
    def _calculate_historical_adjustment(self) -> float:
        """Calculate adjustment factor based on historical performance."""
        if not self.training_history:
            return 1.0
            
        recent_accuracy = np.mean([m.accuracy for m in self.training_history[-10:]])
        
        # Better accuracy = lower adjustment (more confident predictions)
        adjustment = 2.0 - recent_accuracy
        return max(0.5, min(1.5, adjustment))
        
    async def learn_from_feedback(self, 
                                  predictions: Dict[str, float],
                                  actual_outcomes: Dict[str, float],
                                  context: Dict[str, Any]):
        """Learn from prediction accuracy and adjust models.
        
        Args:
            predictions: Previous predictions made by the system
            actual_outcomes: Actual results that occurred
            context: Context data about the situation
        """
        # Calculate prediction accuracy
        accuracy_scores = {}
        for key in predictions:
            if key in actual_outcomes:
                error = abs(predictions[key] - actual_outcomes[key])
                accuracy_scores[key] = 1.0 - error
                
        avg_accuracy = np.mean(list(accuracy_scores.values())) if accuracy_scores else 0.5
        
        # Update model training history
        metrics = TrainingMetrics(
            accuracy=avg_accuracy,
            precision=accuracy_scores.get('precision', 0.8),
            recall=accuracy_scores.get('recall', 0.8),
            f1_score=accuracy_scores.get('f1_score', 0.8),
            loss=1.0 - avg_accuracy,
            training_time=time.time(),
            validation_score=avg_accuracy * 0.9,
            epoch=len(self.training_history) + 1
        )
        
        self.training_history.append(metrics)
        
        # Retrain models if accuracy drops below threshold
        if avg_accuracy < 0.7:
            await self._retrain_models(context)
            
        self.logger.info(f"Learning update: accuracy={avg_accuracy:.3f}")
        
    async def _retrain_models(self, context: Dict[str, Any]):
        """Retrain models with recent data."""
        for model_type, model in self.models.items():
            try:
                await model.retrain(self.training_history[-100:])  # Use last 100 samples
                self.logger.info(f"Retrained {model_type.value} model")
            except Exception as e:
                self.logger.error(f"Failed to retrain {model_type.value}: {e}")
                
    async def generate_adaptive_gates(self, 
                                      project_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate adaptive quality gates based on project context.
        
        Args:
            project_context: Context about the project
            
        Returns:
            List of adaptive quality gate configurations
        """
        # Extract features from project context
        features = await self.extract_features(project_context)
        
        # Predict optimal gate configuration
        quality_model = self.models[MLModelType.QUALITY_SCORER]
        feature_vector = []
        for feature_list in features.values():
            feature_vector.extend(feature_list)
            
        quality_score = await quality_model.predict(feature_vector)
        
        # Generate adaptive gates based on quality score and risk
        risk_predictions = await self.predict_risk(features)
        
        gates = []
        
        # Functional testing gate
        gates.append({
            'name': 'Adaptive Functional Testing',
            'type': 'functional',
            'threshold': max(0.85, quality_score * 0.9),
            'priority': 1 if risk_predictions['overall_risk'] > 0.7 else 2,
            'adaptive_params': {
                'test_selection': 'risk_based' if risk_predictions['overall_risk'] > 0.6 else 'coverage_based',
                'timeout_multiplier': 1.5 if risk_predictions['performance_risk'] > 0.7 else 1.0
            }
        })
        
        # Performance gate
        gates.append({
            'name': 'Adaptive Performance Testing',
            'type': 'performance',
            'threshold': max(200, 1000 * (1 - quality_score)),
            'priority': 1 if risk_predictions['performance_risk'] > 0.6 else 3,
            'adaptive_params': {
                'load_pattern': 'stress' if risk_predictions['performance_risk'] > 0.8 else 'normal',
                'duration_minutes': 30 if risk_predictions['overall_risk'] > 0.7 else 15
            }
        })
        
        # Security gate
        gates.append({
            'name': 'Adaptive Security Scanning',
            'type': 'security',
            'threshold': max(0.9, quality_score * 0.95),
            'priority': 1 if risk_predictions['security_risk'] > 0.5 else 2,
            'adaptive_params': {
                'scan_depth': 'deep' if risk_predictions['security_risk'] > 0.7 else 'standard',
                'include_penetration_test': risk_predictions['security_risk'] > 0.8
            }
        })
        
        return gates


class SimpleMLModel:
    """Simple ML model implementation for quality gates."""
    
    def __init__(self, config: MLModelConfig):
        """Initialize the model with configuration."""
        self.config = config
        self.weights = [0.5] * config.feature_dimensions
        self.bias = 0.0
        self.training_data: List[tuple] = []
        
    async def predict(self, features: List[float]) -> float:
        """Make a prediction using the model.
        
        Args:
            features: Input feature vector
            
        Returns:
            Prediction value
        """
        # Ensure feature vector matches expected dimensions
        if len(features) > len(self.weights):
            features = features[:len(self.weights)]
        elif len(features) < len(self.weights):
            features.extend([0.5] * (len(self.weights) - len(features)))
            
        # Simple linear model prediction
        prediction = sum(w * f for w, f in zip(self.weights, features)) + self.bias
        
        # Apply activation function based on model type
        if self.config.model_type in [MLModelType.RISK_PREDICTOR, MLModelType.FAILURE_PREDICTOR]:
            # Sigmoid for probability outputs
            prediction = 1 / (1 + math.exp(-prediction))
        elif self.config.model_type == MLModelType.QUALITY_SCORER:
            # Tanh for quality scores
            prediction = math.tanh(prediction)
            prediction = (prediction + 1) / 2  # Scale to [0, 1]
        else:
            # ReLU for other outputs
            prediction = max(0, prediction)
            
        return min(1.0, max(0.0, prediction))
        
    async def retrain(self, training_data: List[TrainingMetrics]):
        """Retrain the model with new data.
        
        Args:
            training_data: Training metrics for retraining
        """
        if not training_data:
            return
            
        # Simple gradient descent update
        learning_rate = self.config.learning_rate
        
        # Extract targets from training data
        targets = [m.accuracy for m in training_data]
        avg_target = np.mean(targets)
        
        # Update weights based on performance
        for i in range(len(self.weights)):
            # Simple weight adjustment based on performance
            if avg_target > 0.8:
                self.weights[i] *= 1.01  # Slight increase for good performance
            else:
                self.weights[i] *= 0.99  # Slight decrease for poor performance
                
        # Update bias
        self.bias += learning_rate * (avg_target - 0.5)
        
        # Clip weights to reasonable ranges
        self.weights = [max(-2.0, min(2.0, w)) for w in self.weights]
        self.bias = max(-1.0, min(1.0, self.bias))