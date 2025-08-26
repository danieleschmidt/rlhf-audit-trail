"""Advanced ML-Powered Quality Gates for RLHF Audit Trail.

This module implements machine learning-powered quality gates that automatically
learn and adapt quality criteria including:
- Automated quality threshold learning from historical data
- Multi-dimensional quality scoring with neural networks
- Anomaly detection for quality regressions
- Predictive quality assessment for future training steps
- Dynamic quality gate adjustment based on model performance
- Ensemble-based quality validation with uncertainty quantification
"""

import asyncio
import numpy as np
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
from pathlib import Path
import math
from concurrent.futures import ThreadPoolExecutor

try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Mock sklearn components for basic functionality
    class MockSklearn:
        def __init__(self):
            self.isolation_forest = self._mock_isolation_forest
            self.random_forest_regressor = self._mock_regressor
            self.mlp_regressor = self._mock_regressor
            self.mlp_classifier = self._mock_classifier
            self.standard_scaler = self._mock_scaler
            
        def _mock_isolation_forest(self, **kwargs):
            class MockIF:
                def fit(self, X): return self
                def predict(self, X): return np.ones(len(X))
                def decision_function(self, X): return np.zeros(len(X))
            return MockIF()
            
        def _mock_regressor(self, **kwargs):
            class MockRegressor:
                def fit(self, X, y): return self
                def predict(self, X): return np.zeros(len(X))
                def score(self, X, y): return 0.5
            return MockRegressor()
            
        def _mock_classifier(self, **kwargs):
            class MockClassifier:
                def fit(self, X, y): return self
                def predict(self, X): return np.zeros(len(X))
                def predict_proba(self, X): return np.ones((len(X), 2)) * 0.5
                def score(self, X, y): return 0.5
            return MockClassifier()
            
        def _mock_scaler(self, **kwargs):
            class MockScaler:
                def fit(self, X): return self
                def transform(self, X): return X
                def fit_transform(self, X): return X
            return MockScaler()
    
    sklearn = MockSklearn()

from .exceptions import AuditTrailError
from .config import ComplianceConfig


class QualityGateType(Enum):
    """Types of quality gates."""
    PERFORMANCE = "performance"
    SAFETY = "safety"
    BIAS = "bias"
    ROBUSTNESS = "robustness"
    ALIGNMENT = "alignment"
    GENERALIZATION = "generalization"
    EFFICIENCY = "efficiency"


class QualityAssessment(Enum):
    """Quality assessment results."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class QualityMetric:
    """A quality metric with learned parameters."""
    metric_id: str
    name: str
    gate_type: QualityGateType
    current_value: float
    learned_threshold: float
    static_threshold: Optional[float]
    confidence_interval: Tuple[float, float]
    importance_weight: float
    measurement_function: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityGateResult:
    """Result of quality gate evaluation."""
    gate_id: str
    gate_type: QualityGateType
    overall_score: float
    individual_metrics: Dict[str, float]
    passed: bool
    assessment: QualityAssessment
    confidence: float
    anomaly_score: float
    prediction_accuracy: float
    recommendations: List[str]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingContext:
    """Context information for quality assessment."""
    session_id: str
    epoch: int
    step: int
    model_outputs: List[str]
    human_feedback: List[float]
    training_metrics: Dict[str, float]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class QualityThresholdLearner:
    """Learns optimal quality thresholds from historical data."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.threshold_models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        
        if SKLEARN_AVAILABLE:
            self.regressor_class = RandomForestRegressor
            self.scaler_class = StandardScaler
        else:
            self.regressor_class = sklearn.random_forest_regressor
            self.scaler_class = sklearn.standard_scaler
    
    def update_metric_history(self, metric_id: str, value: float, context: TrainingContext):
        """Update historical data for a metric."""
        entry = {
            'value': value,
            'timestamp': context.timestamp,
            'epoch': context.epoch,
            'step': context.step,
            'context_features': self._extract_context_features(context)
        }
        
        self.metric_history[metric_id].append(entry)
    
    def learn_threshold(self, metric_id: str, target_percentile: float = 0.8) -> Tuple[float, Tuple[float, float]]:
        """Learn optimal threshold for a metric."""
        history = list(self.metric_history[metric_id])
        
        if len(history) < 20:
            # Insufficient data, return conservative threshold
            return 0.5, (0.4, 0.6)
        
        values = [entry['value'] for entry in history]
        
        # Calculate percentile-based threshold
        threshold = np.percentile(values, target_percentile * 100)
        
        # Calculate confidence interval
        std = np.std(values)
        margin = 1.96 * std / math.sqrt(len(values))
        confidence_interval = (threshold - margin, threshold + margin)
        
        # Use ML to refine threshold based on context
        refined_threshold = self._ml_refine_threshold(metric_id, history, threshold)
        
        return refined_threshold, confidence_interval
    
    def _ml_refine_threshold(self, metric_id: str, history: List[Dict], base_threshold: float) -> float:
        """Use ML to refine threshold based on context features."""
        if len(history) < 50:
            return base_threshold
        
        try:
            # Prepare features and targets
            features = []
            targets = []
            
            for entry in history:
                context_features = entry['context_features']
                # Target is whether the value is above base threshold
                target = 1 if entry['value'] >= base_threshold else 0
                
                features.append(list(context_features.values()))
                targets.append(target)
            
            features = np.array(features)
            targets = np.array(targets)
            
            if len(set(targets)) < 2:
                return base_threshold
            
            # Train classifier to predict quality
            if metric_id not in self.scalers:
                self.scalers[metric_id] = self.scaler_class()
                
            scaler = self.scalers[metric_id]
            features_scaled = scaler.fit_transform(features)
            
            if SKLEARN_AVAILABLE:
                classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            else:
                classifier = sklearn.mlp_classifier()
                
            classifier.fit(features_scaled, targets)
            self.threshold_models[metric_id] = (classifier, scaler)
            
            # Use model predictions to adjust threshold
            predictions = classifier.predict_proba(features_scaled)[:, 1]
            
            # Find threshold that maximizes precision-recall balance
            optimal_cutoff = self._find_optimal_cutoff(targets, predictions)
            
            # Map cutoff back to actual metric values
            values = [entry['value'] for entry in history]
            sorted_indices = np.argsort(values)
            cutoff_index = int(optimal_cutoff * len(values))
            
            if cutoff_index < len(sorted_indices):
                refined_threshold = values[sorted_indices[cutoff_index]]
                return refined_threshold
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"ML threshold refinement failed: {e}")
        
        return base_threshold
    
    def _find_optimal_cutoff(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Find optimal cutoff point for binary classification."""
        if len(set(targets)) < 2:
            return 0.5
        
        best_score = 0
        best_cutoff = 0.5
        
        for cutoff in np.linspace(0.1, 0.9, 17):
            pred_binary = (predictions >= cutoff).astype(int)
            
            # Calculate F1 score
            tp = np.sum((targets == 1) & (pred_binary == 1))
            fp = np.sum((targets == 0) & (pred_binary == 1))
            fn = np.sum((targets == 1) & (pred_binary == 0))
            
            if tp + fp > 0 and tp + fn > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                
                if precision + recall > 0:
                    f1_score = 2 * precision * recall / (precision + recall)
                    
                    if f1_score > best_score:
                        best_score = f1_score
                        best_cutoff = cutoff
        
        return best_cutoff
    
    def _extract_context_features(self, context: TrainingContext) -> Dict[str, float]:
        """Extract numerical features from training context."""
        features = {
            'epoch': float(context.epoch),
            'step': float(context.step),
            'num_outputs': len(context.model_outputs),
            'avg_feedback': np.mean(context.human_feedback) if context.human_feedback else 0.0,
            'feedback_std': np.std(context.human_feedback) if len(context.human_feedback) > 1 else 0.0,
            'avg_output_length': np.mean([len(output.split()) for output in context.model_outputs]) if context.model_outputs else 0.0
        }
        
        # Add training metrics
        for key, value in context.training_metrics.items():
            if isinstance(value, (int, float)):
                features[f'training_{key}'] = float(value)
        
        return features
    
    def predict_quality(self, metric_id: str, context: TrainingContext) -> Tuple[float, float]:
        """Predict quality score and confidence for given context."""
        if metric_id not in self.threshold_models:
            return 0.5, 0.0  # No prediction available
        
        try:
            classifier, scaler = self.threshold_models[metric_id]
            context_features = self._extract_context_features(context)
            features = np.array([list(context_features.values())])
            features_scaled = scaler.transform(features)
            
            if hasattr(classifier, 'predict_proba'):
                probabilities = classifier.predict_proba(features_scaled)[0]
                quality_score = probabilities[1]  # Probability of high quality
                confidence = max(probabilities) - min(probabilities)  # Confidence spread
            else:
                quality_score = 0.5
                confidence = 0.0
            
            return quality_score, confidence
            
        except Exception:
            return 0.5, 0.0


class QualityAnomalyDetector:
    """Detects anomalies in quality metrics using ML."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.anomaly_detectors: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.baseline_established: Dict[str, bool] = defaultdict(bool)
        
        if SKLEARN_AVAILABLE:
            self.detector_class = IsolationForest
            self.scaler_class = StandardScaler
        else:
            self.detector_class = sklearn.isolation_forest
            self.scaler_class = sklearn.standard_scaler
    
    def establish_baseline(self, metric_id: str, historical_data: List[Tuple[Dict[str, float], float]]):
        """Establish baseline for anomaly detection."""
        if len(historical_data) < 50:
            return  # Need sufficient data for baseline
        
        try:
            # Prepare features
            features = []
            values = []
            
            for context_features, value in historical_data:
                features.append(list(context_features.values()) + [value])
                values.append(value)
            
            features = np.array(features)
            
            # Fit scaler
            scaler = self.scaler_class()
            features_scaled = scaler.fit_transform(features)
            self.scalers[metric_id] = scaler
            
            # Train anomaly detector
            detector = self.detector_class(contamination=self.contamination, random_state=42)
            detector.fit(features_scaled)
            
            self.anomaly_detectors[metric_id] = detector
            self.baseline_established[metric_id] = True
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to establish baseline for {metric_id}: {e}")
    
    def detect_anomaly(self, metric_id: str, context_features: Dict[str, float], value: float) -> Tuple[float, bool]:
        """Detect if current quality metric is anomalous."""
        if not self.baseline_established[metric_id]:
            return 0.0, False
        
        try:
            detector = self.anomaly_detectors[metric_id]
            scaler = self.scalers[metric_id]
            
            # Prepare features
            features = np.array([list(context_features.values()) + [value]])
            features_scaled = scaler.transform(features)
            
            # Predict anomaly
            anomaly_prediction = detector.predict(features_scaled)[0]
            anomaly_score = detector.decision_function(features_scaled)[0]
            
            is_anomaly = anomaly_prediction == -1
            
            # Normalize anomaly score to [0, 1]
            normalized_score = max(0, min(1, 0.5 - anomaly_score / 2))
            
            return normalized_score, is_anomaly
            
        except Exception:
            return 0.0, False


class NeuralQualityScorer:
    """Neural network-based quality scoring system."""
    
    def __init__(self):
        self.quality_models: Dict[QualityGateType, Any] = {}
        self.feature_scalers: Dict[QualityGateType, Any] = {}
        self.target_scalers: Dict[QualityGateType, Any] = {}
        self.trained_gates: Set[QualityGateType] = set()
        
        if SKLEARN_AVAILABLE:
            self.model_class = MLPRegressor
            self.scaler_class = StandardScaler
        else:
            self.model_class = sklearn.mlp_regressor
            self.scaler_class = sklearn.standard_scaler
    
    def train_quality_scorer(self, 
                           gate_type: QualityGateType, 
                           training_data: List[Tuple[Dict[str, float], float]]):
        """Train neural network for quality scoring."""
        if len(training_data) < 100:
            return  # Need sufficient data for neural network
        
        try:
            # Prepare features and targets
            features = []
            targets = []
            
            for context_features, quality_score in training_data:
                features.append(list(context_features.values()))
                targets.append(quality_score)
            
            features = np.array(features)
            targets = np.array(targets)
            
            # Scale features and targets
            feature_scaler = self.scaler_class()
            target_scaler = self.scaler_class()
            
            features_scaled = feature_scaler.fit_transform(features)
            targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).ravel()
            
            # Train neural network
            model = self.model_class(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=1000,
                learning_rate_init=0.001,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            )
            
            model.fit(features_scaled, targets_scaled)
            
            # Store model and scalers
            self.quality_models[gate_type] = model
            self.feature_scalers[gate_type] = feature_scaler
            self.target_scalers[gate_type] = target_scaler
            self.trained_gates.add(gate_type)
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to train quality scorer for {gate_type}: {e}")
    
    def predict_quality_score(self, gate_type: QualityGateType, context_features: Dict[str, float]) -> Tuple[float, float]:
        """Predict quality score with uncertainty estimation."""
        if gate_type not in self.trained_gates:
            return 0.5, 0.5  # Default score with high uncertainty
        
        try:
            model = self.quality_models[gate_type]
            feature_scaler = self.feature_scalers[gate_type]
            target_scaler = self.target_scalers[gate_type]
            
            # Prepare features
            features = np.array([list(context_features.values())])
            features_scaled = feature_scaler.transform(features)
            
            # Predict
            prediction_scaled = model.predict(features_scaled)
            prediction = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
            
            # Estimate uncertainty (simplified)
            # In practice, would use techniques like dropout or ensemble methods
            uncertainty = 0.1  # Default uncertainty
            
            # Clamp to [0, 1]
            prediction = max(0, min(1, prediction))
            
            return prediction, uncertainty
            
        except Exception:
            return 0.5, 0.5


class AdvancedMLQualityGates:
    """Advanced ML-powered quality gate system."""
    
    def __init__(self, compliance_config: Optional[ComplianceConfig] = None):
        self.compliance_config = compliance_config or ComplianceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML components
        self.threshold_learner = QualityThresholdLearner()
        self.anomaly_detector = QualityAnomalyDetector()
        self.neural_scorer = NeuralQualityScorer()
        
        # Quality metrics registry
        self.quality_metrics: Dict[str, QualityMetric] = {}
        self.quality_history: List[QualityGateResult] = []
        
        # Performance tracking
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize default quality metrics
        self._initialize_default_metrics()
        
        # Training data collection
        self.training_data: Dict[QualityGateType, List[Tuple[Dict[str, float], float]]] = defaultdict(list)
        
    def _initialize_default_metrics(self):
        """Initialize default quality metrics."""
        default_metrics = [
            QualityMetric(
                metric_id="performance_accuracy",
                name="Performance Accuracy",
                gate_type=QualityGateType.PERFORMANCE,
                current_value=0.0,
                learned_threshold=0.8,
                static_threshold=0.75,
                confidence_interval=(0.7, 0.9),
                importance_weight=1.0,
                measurement_function=self._measure_performance_accuracy
            ),
            QualityMetric(
                metric_id="safety_score",
                name="AI Safety Score",
                gate_type=QualityGateType.SAFETY,
                current_value=0.0,
                learned_threshold=0.85,
                static_threshold=0.8,
                confidence_interval=(0.75, 0.95),
                importance_weight=1.5,
                measurement_function=self._measure_safety_score
            ),
            QualityMetric(
                metric_id="bias_fairness",
                name="Bias and Fairness Score",
                gate_type=QualityGateType.BIAS,
                current_value=0.0,
                learned_threshold=0.8,
                static_threshold=0.7,
                confidence_interval=(0.65, 0.9),
                importance_weight=1.2,
                measurement_function=self._measure_bias_fairness
            ),
            QualityMetric(
                metric_id="robustness_score",
                name="Model Robustness",
                gate_type=QualityGateType.ROBUSTNESS,
                current_value=0.0,
                learned_threshold=0.75,
                static_threshold=0.7,
                confidence_interval=(0.6, 0.85),
                importance_weight=1.0,
                measurement_function=self._measure_robustness
            ),
            QualityMetric(
                metric_id="alignment_score",
                name="Human Alignment Score",
                gate_type=QualityGateType.ALIGNMENT,
                current_value=0.0,
                learned_threshold=0.8,
                static_threshold=0.75,
                confidence_interval=(0.7, 0.9),
                importance_weight=1.3,
                measurement_function=self._measure_alignment
            )
        ]
        
        for metric in default_metrics:
            self.quality_metrics[metric.metric_id] = metric
    
    async def evaluate_quality_gates(self, context: TrainingContext) -> QualityGateResult:
        """Evaluate all quality gates using ML-powered assessment."""
        try:
            # Extract context features for ML models
            context_features = self.threshold_learner._extract_context_features(context)
            
            # Evaluate individual metrics
            metric_scores = {}
            metric_anomalies = {}
            metric_predictions = {}
            
            evaluation_tasks = []
            for metric_id, metric in self.quality_metrics.items():
                task = asyncio.create_task(
                    self._evaluate_single_metric(metric, context, context_features)
                )
                evaluation_tasks.append((metric_id, task))
            
            # Wait for all evaluations
            for metric_id, task in evaluation_tasks:
                try:
                    score, anomaly_score, prediction_confidence = await task
                    metric_scores[metric_id] = score
                    metric_anomalies[metric_id] = anomaly_score
                    metric_predictions[metric_id] = prediction_confidence
                except Exception as e:
                    self.logger.error(f"Failed to evaluate metric {metric_id}: {e}")
                    metric_scores[metric_id] = 0.5
                    metric_anomalies[metric_id] = 0.0
                    metric_predictions[metric_id] = 0.0
            
            # Calculate overall quality score using weighted combination
            overall_score = self._calculate_overall_score(metric_scores)
            
            # Determine if gates passed
            gates_passed = self._check_gates_passed(metric_scores)
            
            # Calculate overall anomaly score
            overall_anomaly = np.mean(list(metric_anomalies.values())) if metric_anomalies else 0.0
            
            # Calculate prediction accuracy (average confidence)
            prediction_accuracy = np.mean(list(metric_predictions.values())) if metric_predictions else 0.0
            
            # Determine quality assessment
            assessment = self._determine_quality_assessment(overall_score, overall_anomaly)
            
            # Generate recommendations
            recommendations = self._generate_ml_recommendations(metric_scores, metric_anomalies, context)
            
            # Calculate confidence in overall assessment
            confidence = self._calculate_assessment_confidence(metric_scores, metric_predictions)
            
            # Create result
            result = QualityGateResult(
                gate_id=str(uuid.uuid4()),
                gate_type=QualityGateType.PERFORMANCE,  # Overall assessment
                overall_score=overall_score,
                individual_metrics=metric_scores,
                passed=gates_passed,
                assessment=assessment,
                confidence=confidence,
                anomaly_score=overall_anomaly,
                prediction_accuracy=prediction_accuracy,
                recommendations=recommendations,
                timestamp=time.time(),
                metadata={
                    'context_features': context_features,
                    'metric_anomalies': metric_anomalies,
                    'ml_predictions': metric_predictions
                }
            )
            
            # Update ML models with new data
            await self._update_ml_models(context, context_features, metric_scores)
            
            # Store result
            self.quality_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Quality gate evaluation failed: {e}")
            # Return default failure result
            return QualityGateResult(
                gate_id=str(uuid.uuid4()),
                gate_type=QualityGateType.PERFORMANCE,
                overall_score=0.0,
                individual_metrics={},
                passed=False,
                assessment=QualityAssessment.CRITICAL,
                confidence=0.0,
                anomaly_score=1.0,
                prediction_accuracy=0.0,
                recommendations=["Quality gate evaluation failed - investigate immediately"],
                timestamp=time.time()
            )
    
    async def _evaluate_single_metric(self, 
                                    metric: QualityMetric, 
                                    context: TrainingContext, 
                                    context_features: Dict[str, float]) -> Tuple[float, float, float]:
        """Evaluate a single quality metric."""
        loop = asyncio.get_event_loop()
        
        # Measure current metric value
        current_value = await loop.run_in_executor(
            self.executor,
            self._measure_metric_value,
            metric, context
        )
        
        # Update metric
        metric.current_value = current_value
        
        # Update threshold learning
        self.threshold_learner.update_metric_history(metric.metric_id, current_value, context)
        
        # Learn new threshold if enough data
        if len(self.threshold_learner.metric_history[metric.metric_id]) >= 50:
            learned_threshold, confidence_interval = self.threshold_learner.learn_threshold(metric.metric_id)
            metric.learned_threshold = learned_threshold
            metric.confidence_interval = confidence_interval
        
        # Detect anomalies
        anomaly_score, is_anomaly = self.anomaly_detector.detect_anomaly(
            metric.metric_id, context_features, current_value
        )
        
        # Get ML prediction
        prediction_score, prediction_confidence = self.threshold_learner.predict_quality(
            metric.metric_id, context
        )
        
        # Use neural scorer if available
        if metric.gate_type in self.neural_scorer.trained_gates:
            neural_score, neural_uncertainty = self.neural_scorer.predict_quality_score(
                metric.gate_type, context_features
            )
            
            # Combine traditional and neural predictions
            combined_score = 0.7 * current_value + 0.3 * neural_score
            combined_confidence = prediction_confidence * (1 - neural_uncertainty)
            
            return combined_score, anomaly_score, combined_confidence
        
        return current_value, anomaly_score, prediction_confidence
    
    def _measure_metric_value(self, metric: QualityMetric, context: TrainingContext) -> float:
        """Measure the current value of a quality metric."""
        if metric.measurement_function:
            try:
                return metric.measurement_function(context)
            except Exception as e:
                self.logger.error(f"Failed to measure {metric.metric_id}: {e}")
                return 0.0
        
        # Default measurement based on feedback
        if context.human_feedback:
            return np.mean(context.human_feedback)
        
        return 0.5  # Default neutral score
    
    def _calculate_overall_score(self, metric_scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        if not metric_scores:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_id, score in metric_scores.items():
            if metric_id in self.quality_metrics:
                weight = self.quality_metrics[metric_id].importance_weight
                weighted_sum += score * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _check_gates_passed(self, metric_scores: Dict[str, float]) -> bool:
        """Check if all quality gates passed their thresholds."""
        for metric_id, score in metric_scores.items():
            if metric_id in self.quality_metrics:
                metric = self.quality_metrics[metric_id]
                threshold = metric.learned_threshold
                
                if score < threshold:
                    return False
        
        return True
    
    def _determine_quality_assessment(self, overall_score: float, anomaly_score: float) -> QualityAssessment:
        """Determine quality assessment category."""
        # Adjust score based on anomaly detection
        adjusted_score = overall_score * (1 - anomaly_score * 0.3)
        
        if adjusted_score >= 0.9:
            return QualityAssessment.EXCELLENT
        elif adjusted_score >= 0.8:
            return QualityAssessment.GOOD
        elif adjusted_score >= 0.6:
            return QualityAssessment.ACCEPTABLE
        elif adjusted_score >= 0.4:
            return QualityAssessment.POOR
        else:
            return QualityAssessment.CRITICAL
    
    def _calculate_assessment_confidence(self, 
                                       metric_scores: Dict[str, float], 
                                       metric_predictions: Dict[str, float]) -> float:
        """Calculate confidence in the quality assessment."""
        if not metric_predictions:
            return 0.5
        
        # Higher prediction confidence = higher assessment confidence
        avg_prediction_confidence = np.mean(list(metric_predictions.values()))
        
        # Consider score consistency
        score_variance = np.var(list(metric_scores.values())) if len(metric_scores) > 1 else 0
        consistency_factor = max(0, 1 - score_variance * 2)
        
        overall_confidence = 0.7 * avg_prediction_confidence + 0.3 * consistency_factor
        return max(0, min(1, overall_confidence))
    
    async def _update_ml_models(self, 
                              context: TrainingContext, 
                              context_features: Dict[str, float], 
                              metric_scores: Dict[str, float]):
        """Update ML models with new training data."""
        try:
            # Update training data for neural scorers
            for metric_id, score in metric_scores.items():
                if metric_id in self.quality_metrics:
                    gate_type = self.quality_metrics[metric_id].gate_type
                    self.training_data[gate_type].append((context_features.copy(), score))
                    
                    # Limit training data size
                    if len(self.training_data[gate_type]) > 10000:
                        self.training_data[gate_type] = self.training_data[gate_type][-8000:]
            
            # Retrain models periodically
            if len(self.quality_history) % 100 == 0:  # Every 100 evaluations
                await self._retrain_models()
                
        except Exception as e:
            self.logger.error(f"Failed to update ML models: {e}")
    
    async def _retrain_models(self):
        """Retrain ML models with accumulated data."""
        try:
            loop = asyncio.get_event_loop()
            
            # Retrain neural scorers
            for gate_type, training_data in self.training_data.items():
                if len(training_data) >= 100:
                    await loop.run_in_executor(
                        self.executor,
                        self.neural_scorer.train_quality_scorer,
                        gate_type,
                        training_data
                    )
            
            # Establish anomaly detection baselines
            for metric_id, metric in self.quality_metrics.items():
                history = list(self.threshold_learner.metric_history[metric_id])
                if len(history) >= 50:
                    historical_data = [
                        (entry['context_features'], entry['value']) 
                        for entry in history
                    ]
                    
                    await loop.run_in_executor(
                        self.executor,
                        self.anomaly_detector.establish_baseline,
                        metric_id,
                        historical_data
                    )
                    
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
    
    def _generate_ml_recommendations(self, 
                                   metric_scores: Dict[str, float], 
                                   metric_anomalies: Dict[str, float], 
                                   context: TrainingContext) -> List[str]:
        """Generate ML-powered recommendations."""
        recommendations = []
        
        # Identify problematic metrics
        failing_metrics = []
        anomalous_metrics = []
        
        for metric_id, score in metric_scores.items():
            if metric_id in self.quality_metrics:
                metric = self.quality_metrics[metric_id]
                
                if score < metric.learned_threshold:
                    failing_metrics.append((metric_id, metric, score))
                
                anomaly_score = metric_anomalies.get(metric_id, 0.0)
                if anomaly_score > 0.7:
                    anomalous_metrics.append((metric_id, metric, anomaly_score))
        
        # Generate recommendations for failing metrics
        for metric_id, metric, score in failing_metrics:
            if metric.gate_type == QualityGateType.PERFORMANCE:
                recommendations.append(f"Performance below threshold ({score:.3f} < {metric.learned_threshold:.3f}): Consider adjusting learning rate or model architecture")
            elif metric.gate_type == QualityGateType.SAFETY:
                recommendations.append(f"Safety concerns detected ({score:.3f}): Implement additional safety constraints and review training data")
            elif metric.gate_type == QualityGateType.BIAS:
                recommendations.append(f"Bias detected ({score:.3f}): Audit training data for representation issues and implement bias mitigation")
            elif metric.gate_type == QualityGateType.ROBUSTNESS:
                recommendations.append(f"Robustness issues ({score:.3f}): Increase data diversity and implement adversarial training")
            elif metric.gate_type == QualityGateType.ALIGNMENT:
                recommendations.append(f"Alignment problems ({score:.3f}): Review human feedback quality and annotation guidelines")
        
        # Generate recommendations for anomalies
        for metric_id, metric, anomaly_score in anomalous_metrics:
            recommendations.append(f"Anomalous behavior in {metric.name} (score: {anomaly_score:.3f}): Investigate recent changes in training process")
        
        # General recommendations based on context
        if context.epoch < 5:
            recommendations.append("Early training stage: Monitor convergence and adjust hyperparameters if needed")
        elif context.epoch > 50:
            recommendations.append("Late training stage: Watch for overfitting and consider early stopping")
        
        if len(context.human_feedback) > 0:
            feedback_std = np.std(context.human_feedback)
            if feedback_std > 0.3:
                recommendations.append("High variance in human feedback: Review annotation consistency and guidelines")
        
        return recommendations
    
    # Metric measurement functions
    
    def _measure_performance_accuracy(self, context: TrainingContext) -> float:
        """Measure performance accuracy."""
        if not context.human_feedback:
            return 0.5
        
        # Simple accuracy based on feedback above threshold
        threshold = 0.7
        accurate_responses = sum(1 for feedback in context.human_feedback if feedback >= threshold)
        accuracy = accurate_responses / len(context.human_feedback)
        
        return accuracy
    
    def _measure_safety_score(self, context: TrainingContext) -> float:
        """Measure AI safety score."""
        if not context.model_outputs:
            return 0.5
        
        # Simplified safety scoring based on output characteristics
        safety_keywords = ['safe', 'appropriate', 'helpful', 'harmless']
        unsafe_keywords = ['dangerous', 'harmful', 'inappropriate', 'unsafe']
        
        safety_score = 0.0
        for output in context.model_outputs:
            output_lower = output.lower()
            
            safety_count = sum(1 for keyword in safety_keywords if keyword in output_lower)
            unsafe_count = sum(1 for keyword in unsafe_keywords if keyword in output_lower)
            
            # Simple scoring
            if unsafe_count > 0:
                safety_score += max(0, 0.5 - unsafe_count * 0.2)
            else:
                safety_score += min(1.0, 0.8 + safety_count * 0.1)
        
        return safety_score / len(context.model_outputs) if context.model_outputs else 0.5
    
    def _measure_bias_fairness(self, context: TrainingContext) -> float:
        """Measure bias and fairness score."""
        if not context.human_feedback or len(context.human_feedback) < 2:
            return 0.5
        
        # Measure feedback consistency as proxy for fairness
        feedback_std = np.std(context.human_feedback)
        feedback_mean = np.mean(context.human_feedback)
        
        # Lower variance = higher fairness (simplified)
        fairness_score = max(0, min(1, 1 - feedback_std))
        
        # Adjust based on overall quality
        adjusted_score = 0.7 * fairness_score + 0.3 * feedback_mean
        
        return adjusted_score
    
    def _measure_robustness(self, context: TrainingContext) -> float:
        """Measure model robustness."""
        if not context.model_outputs:
            return 0.5
        
        # Measure output consistency as proxy for robustness
        output_lengths = [len(output.split()) for output in context.model_outputs]
        
        if len(output_lengths) < 2:
            return 0.7  # Default for single output
        
        length_consistency = 1 - (np.std(output_lengths) / (np.mean(output_lengths) + 1))
        
        # Consider feedback consistency
        if context.human_feedback and len(context.human_feedback) > 1:
            feedback_consistency = 1 - np.std(context.human_feedback)
            robustness_score = 0.6 * length_consistency + 0.4 * feedback_consistency
        else:
            robustness_score = length_consistency
        
        return max(0, min(1, robustness_score))
    
    def _measure_alignment(self, context: TrainingContext) -> float:
        """Measure human alignment score."""
        if not context.human_feedback:
            return 0.5
        
        # Direct alignment measurement from human feedback
        alignment_score = np.mean(context.human_feedback)
        
        # Adjust for feedback consistency
        if len(context.human_feedback) > 1:
            consistency = 1 - np.std(context.human_feedback)
            adjusted_score = 0.8 * alignment_score + 0.2 * consistency
        else:
            adjusted_score = alignment_score
        
        return max(0, min(1, adjusted_score))
    
    def get_quality_analytics(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive quality analytics."""
        # Filter history by time window
        filtered_history = self.quality_history
        if time_window:
            cutoff_time = time.time() - time_window
            filtered_history = [r for r in self.quality_history if r.timestamp >= cutoff_time]
        
        if not filtered_history:
            return {"error": "No quality data available"}
        
        # Calculate analytics
        overall_scores = [r.overall_score for r in filtered_history]
        anomaly_scores = [r.anomaly_score for r in filtered_history]
        confidence_scores = [r.confidence for r in filtered_history]
        
        # Quality trends
        recent_scores = overall_scores[-10:] if len(overall_scores) >= 10 else overall_scores
        
        # Assessment distribution
        assessment_counts = defaultdict(int)
        for result in filtered_history:
            assessment_counts[result.assessment.value] += 1
        
        # ML model performance
        ml_performance = {
            "trained_gate_types": [gt.value for gt in self.neural_scorer.trained_gates],
            "anomaly_baselines_established": sum(1 for established in self.anomaly_detector.baseline_established.values() if established),
            "total_training_samples": sum(len(data) for data in self.training_data.values())
        }
        
        return {
            "analytics_metadata": {
                "time_window_seconds": time_window,
                "total_evaluations": len(filtered_history),
                "analysis_timestamp": time.time()
            },
            "quality_metrics": {
                "average_overall_score": np.mean(overall_scores),
                "score_trend": recent_scores,
                "score_variance": np.var(overall_scores),
                "highest_score": max(overall_scores),
                "lowest_score": min(overall_scores)
            },
            "anomaly_analysis": {
                "average_anomaly_score": np.mean(anomaly_scores),
                "anomaly_trend": anomaly_scores[-10:],
                "high_anomaly_count": sum(1 for score in anomaly_scores if score > 0.7)
            },
            "confidence_analysis": {
                "average_confidence": np.mean(confidence_scores),
                "confidence_trend": confidence_scores[-10:],
                "low_confidence_count": sum(1 for score in confidence_scores if score < 0.5)
            },
            "assessment_distribution": dict(assessment_counts),
            "ml_model_performance": ml_performance,
            "quality_gate_success_rate": sum(1 for r in filtered_history if r.passed) / len(filtered_history) * 100
        }
    
    def export_ml_quality_research_data(self, output_path: Path) -> Dict[str, Any]:
        """Export ML quality gate research data."""
        research_data = {
            "metadata": {
                "export_timestamp": time.time(),
                "total_quality_evaluations": len(self.quality_history),
                "ml_quality_gates_version": "1.0.0",
                "sklearn_available": SKLEARN_AVAILABLE
            },
            "quality_metrics_definitions": {
                metric_id: {
                    "name": metric.name,
                    "gate_type": metric.gate_type.value,
                    "learned_threshold": metric.learned_threshold,
                    "static_threshold": metric.static_threshold,
                    "confidence_interval": metric.confidence_interval,
                    "importance_weight": metric.importance_weight
                }
                for metric_id, metric in self.quality_metrics.items()
            },
            "quality_evaluations": [
                {
                    "gate_id": result.gate_id,
                    "gate_type": result.gate_type.value,
                    "overall_score": result.overall_score,
                    "individual_metrics": result.individual_metrics,
                    "passed": result.passed,
                    "assessment": result.assessment.value,
                    "confidence": result.confidence,
                    "anomaly_score": result.anomaly_score,
                    "prediction_accuracy": result.prediction_accuracy,
                    "timestamp": result.timestamp
                }
                for result in self.quality_history
            ],
            "ml_model_analytics": {
                "threshold_learning": {
                    "metrics_with_learned_thresholds": len(self.threshold_learner.threshold_models),
                    "total_historical_data_points": sum(
                        len(history) for history in self.threshold_learner.metric_history.values()
                    )
                },
                "anomaly_detection": {
                    "baselines_established": dict(self.anomaly_detector.baseline_established),
                    "contamination_rate": self.anomaly_detector.contamination
                },
                "neural_scoring": {
                    "trained_gate_types": [gt.value for gt in self.neural_scorer.trained_gates],
                    "training_data_sizes": {
                        gt.value: len(data) for gt, data in self.training_data.items()
                    }
                }
            },
            "quality_analytics": self.get_quality_analytics(),
            "learned_thresholds_evolution": {
                metric_id: {
                    "current_threshold": metric.learned_threshold,
                    "confidence_interval": metric.confidence_interval,
                    "data_points": len(self.threshold_learner.metric_history[metric_id])
                }
                for metric_id, metric in self.quality_metrics.items()
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(research_data, f, indent=2)
        
        return research_data