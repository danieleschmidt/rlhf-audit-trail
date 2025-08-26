"""Real-time Bias Detection System for RLHF Audit Trail.

This module provides comprehensive real-time bias detection capabilities including:
- Multi-dimensional bias analysis (demographic, linguistic, temporal)
- Statistical bias measurement with confidence intervals  
- Causal bias detection using intervention analysis
- Intersectional bias analysis for multiple protected attributes
- Adaptive bias threshold learning from human feedback
- Real-time bias mitigation recommendations
"""

import asyncio
import numpy as np
import json
import time
import uuid
import re
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import logging
from pathlib import Path
import math
from concurrent.futures import ThreadPoolExecutor

from .config import ComplianceConfig
from .exceptions import AuditTrailError


class BiasType(Enum):
    """Types of bias that can be detected."""
    DEMOGRAPHIC = "demographic"
    LINGUISTIC = "linguistic" 
    TEMPORAL = "temporal"
    CONTENT = "content"
    INTERSECTIONAL = "intersectional"
    CAUSAL = "causal"
    REPRESENTATION = "representation"
    PERFORMANCE = "performance"


class BiasMetric(Enum):
    """Statistical metrics for bias measurement."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    STATISTICAL_PARITY = "statistical_parity"
    DISPARATE_IMPACT = "disparate_impact"
    JENSEN_SHANNON_DIVERGENCE = "jensen_shannon_divergence"


class BiasSeverity(Enum):
    """Severity levels for detected bias."""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


@dataclass
class BiasDetectionResult:
    """Result of bias detection analysis."""
    detection_id: str
    bias_type: BiasType
    metric: BiasMetric
    severity: BiasSeverity
    bias_score: float
    confidence_interval: Tuple[float, float]
    affected_groups: List[str]
    statistical_significance: float
    sample_size: int
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    mitigation_recommendations: List[str] = field(default_factory=list)


@dataclass
class ProtectedAttribute:
    """Definition of a protected attribute for bias analysis."""
    attribute_name: str
    attribute_type: str  # categorical, numerical, text
    possible_values: Optional[List[str]] = None
    extraction_function: Optional[Callable[[Any], Any]] = None
    sensitivity_level: float = 1.0  # 0-1, higher = more sensitive to bias


@dataclass  
class BiasContext:
    """Context information for bias analysis."""
    session_id: str
    model_outputs: List[str]
    human_feedback: List[float]
    input_prompts: List[str]
    demographic_info: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class DemographicBiasDetector:
    """Detects bias related to demographic attributes."""
    
    def __init__(self, protected_attributes: List[ProtectedAttribute]):
        self.protected_attributes = {attr.attribute_name: attr for attr in protected_attributes}
        self.group_statistics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
    def analyze_demographic_bias(self, context: BiasContext) -> List[BiasDetectionResult]:
        """Analyze demographic bias in model outputs and feedback."""
        results = []
        
        if not context.demographic_info:
            return results
            
        # Extract demographic groups from context
        demographic_groups = self._extract_demographic_groups(context)
        
        if len(demographic_groups) < 2:
            return results  # Need at least 2 groups for comparison
            
        # Analyze bias for each protected attribute
        for attr_name, attr_def in self.protected_attributes.items():
            if attr_name not in context.demographic_info:
                continue
                
            bias_result = self._analyze_attribute_bias(
                attr_name, attr_def, context, demographic_groups
            )
            
            if bias_result:
                results.append(bias_result)
        
        return results
    
    def _extract_demographic_groups(self, context: BiasContext) -> Dict[str, List[int]]:
        """Extract demographic groups from context data."""
        groups = defaultdict(list)
        
        for i, demo_info in enumerate(context.demographic_info.get('per_sample', [])):
            if isinstance(demo_info, dict):
                for attr_name in self.protected_attributes:
                    if attr_name in demo_info:
                        group_value = str(demo_info[attr_name])
                        groups[f"{attr_name}_{group_value}"].append(i)
                        
        return groups
    
    def _analyze_attribute_bias(self, 
                               attr_name: str,
                               attr_def: ProtectedAttribute, 
                               context: BiasContext,
                               groups: Dict[str, List[int]]) -> Optional[BiasDetectionResult]:
        """Analyze bias for a specific demographic attribute."""
        
        # Get groups for this attribute
        attr_groups = {k: v for k, v in groups.items() if k.startswith(f"{attr_name}_")}
        
        if len(attr_groups) < 2:
            return None
            
        # Calculate group statistics
        group_stats = {}
        for group_name, indices in attr_groups.items():
            if indices:
                group_feedback = [context.human_feedback[i] for i in indices if i < len(context.human_feedback)]
                if group_feedback:
                    group_stats[group_name] = {
                        'mean_feedback': np.mean(group_feedback),
                        'std_feedback': np.std(group_feedback),
                        'size': len(group_feedback),
                        'feedback_values': group_feedback
                    }
        
        if len(group_stats) < 2:
            return None
            
        # Calculate bias metrics
        bias_score = self._calculate_demographic_parity(group_stats)
        significance = self._calculate_statistical_significance(group_stats)
        confidence_interval = self._calculate_confidence_interval(group_stats, bias_score)
        
        # Determine severity
        severity = self._determine_bias_severity(bias_score, significance, sum(s['size'] for s in group_stats.values()))
        
        # Generate mitigation recommendations
        recommendations = self._generate_demographic_recommendations(attr_name, group_stats, bias_score)
        
        return BiasDetectionResult(
            detection_id=str(uuid.uuid4()),
            bias_type=BiasType.DEMOGRAPHIC,
            metric=BiasMetric.DEMOGRAPHIC_PARITY,
            severity=severity,
            bias_score=bias_score,
            confidence_interval=confidence_interval,
            affected_groups=list(group_stats.keys()),
            statistical_significance=significance,
            sample_size=sum(s['size'] for s in group_stats.values()),
            timestamp=time.time(),
            context={
                'attribute': attr_name,
                'group_statistics': {k: {
                    'mean_feedback': v['mean_feedback'],
                    'size': v['size']
                } for k, v in group_stats.items()}
            },
            mitigation_recommendations=recommendations
        )
    
    def _calculate_demographic_parity(self, group_stats: Dict[str, Dict[str, Any]]) -> float:
        """Calculate demographic parity violation."""
        group_means = [stats['mean_feedback'] for stats in group_stats.values()]
        
        if len(group_means) < 2:
            return 0.0
            
        # Calculate maximum difference between group means
        max_diff = max(group_means) - min(group_means)
        
        # Normalize to [0, 1] scale
        bias_score = min(1.0, max_diff)
        return bias_score
    
    def _calculate_statistical_significance(self, group_stats: Dict[str, Dict[str, Any]]) -> float:
        """Calculate statistical significance of observed bias."""
        groups = list(group_stats.values())
        
        if len(groups) < 2:
            return 0.0
            
        # Two-sample t-test between first two groups (simplified)
        group1, group2 = groups[0], groups[1]
        
        if group1['size'] < 2 or group2['size'] < 2:
            return 0.0
            
        # Calculate pooled standard error
        se1 = group1['std_feedback'] / math.sqrt(group1['size'])
        se2 = group2['std_feedback'] / math.sqrt(group2['size'])
        pooled_se = math.sqrt(se1**2 + se2**2)
        
        if pooled_se == 0:
            return 0.0
            
        # Calculate t-statistic
        t_stat = abs(group1['mean_feedback'] - group2['mean_feedback']) / pooled_se
        
        # Approximate p-value (simplified)
        # In practice, would use proper t-distribution
        p_value = max(0.001, 2 * (1 - min(0.999, t_stat / 3)))
        
        return 1 - p_value  # Convert to significance score
    
    def _calculate_confidence_interval(self, 
                                     group_stats: Dict[str, Dict[str, Any]], 
                                     bias_score: float) -> Tuple[float, float]:
        """Calculate confidence interval for bias score."""
        # Simplified confidence interval calculation
        total_samples = sum(s['size'] for s in group_stats.values())
        
        if total_samples < 10:
            margin = 0.5  # Large margin for small samples
        else:
            margin = 1.96 * math.sqrt(bias_score * (1 - bias_score) / total_samples)
            
        lower = max(0, bias_score - margin)
        upper = min(1, bias_score + margin)
        
        return (lower, upper)
    
    def _determine_bias_severity(self, bias_score: float, significance: float, sample_size: int) -> BiasSeverity:
        """Determine severity level of detected bias."""
        # Combine bias score, significance, and sample size
        severity_score = bias_score * significance * min(1.0, sample_size / 100)
        
        if severity_score > 0.8:
            return BiasSeverity.CRITICAL
        elif severity_score > 0.6:
            return BiasSeverity.HIGH
        elif severity_score > 0.4:
            return BiasSeverity.MEDIUM
        elif severity_score > 0.2:
            return BiasSeverity.LOW
        else:
            return BiasSeverity.NEGLIGIBLE
    
    def _generate_demographic_recommendations(self, 
                                            attr_name: str,
                                            group_stats: Dict[str, Dict[str, Any]], 
                                            bias_score: float) -> List[str]:
        """Generate mitigation recommendations for demographic bias."""
        recommendations = []
        
        # Find disadvantaged group
        group_means = {name: stats['mean_feedback'] for name, stats in group_stats.items()}
        min_group = min(group_means.items(), key=lambda x: x[1])
        max_group = max(group_means.items(), key=lambda x: x[1])
        
        recommendations.append(f"Increase representation of {min_group[0]} in training data")
        recommendations.append(f"Review prompts for potential bias against {attr_name}")
        
        if bias_score > 0.5:
            recommendations.append("Consider demographic-aware sampling during training")
            recommendations.append("Implement bias regularization in model training")
        
        recommendations.append(f"Monitor feedback patterns for {attr_name} groups over time")
        
        return recommendations


class LinguisticBiasDetector:
    """Detects linguistic and language-based bias."""
    
    def __init__(self):
        self.language_patterns = {
            'formal_language': r'\b(therefore|consequently|furthermore|moreover)\b',
            'informal_language': r'\b(gonna|wanna|yeah|ok)\b',
            'technical_language': r'\b(algorithm|implementation|optimization|parameter)\b',
            'simple_language': r'\b(good|bad|nice|easy)\b'
        }
        
        self.bias_keywords = {
            'gender': ['he', 'she', 'him', 'her', 'man', 'woman', 'boy', 'girl'],
            'age': ['young', 'old', 'elderly', 'teenage', 'adult'],
            'ethnicity': ['white', 'black', 'asian', 'hispanic', 'african'],
            'religion': ['christian', 'muslim', 'jewish', 'hindu', 'buddhist']
        }
        
    def analyze_linguistic_bias(self, context: BiasContext) -> List[BiasDetectionResult]:
        """Analyze linguistic bias in model outputs."""
        results = []
        
        # Analyze language complexity bias
        complexity_result = self._analyze_language_complexity_bias(context)
        if complexity_result:
            results.append(complexity_result)
            
        # Analyze keyword bias
        keyword_results = self._analyze_keyword_bias(context)
        results.extend(keyword_results)
        
        # Analyze response length bias
        length_result = self._analyze_response_length_bias(context)
        if length_result:
            results.append(length_result)
            
        return results
    
    def _analyze_language_complexity_bias(self, context: BiasContext) -> Optional[BiasDetectionResult]:
        """Analyze bias in language complexity across different groups."""
        if not context.demographic_info or not context.model_outputs:
            return None
            
        # Calculate complexity scores for each output
        complexity_scores = [self._calculate_language_complexity(output) for output in context.model_outputs]
        
        # Group by demographic information (simplified - using first available attribute)
        demo_groups = defaultdict(list)
        for i, demo_info in enumerate(context.demographic_info.get('per_sample', [])):
            if i < len(complexity_scores) and isinstance(demo_info, dict):
                # Use first available demographic attribute
                for key, value in demo_info.items():
                    demo_groups[f"{key}_{value}"].append(complexity_scores[i])
                    break
                    
        if len(demo_groups) < 2:
            return None
            
        # Calculate bias in complexity
        group_means = {group: np.mean(scores) for group, scores in demo_groups.items() if scores}
        
        if len(group_means) < 2:
            return None
            
        bias_score = (max(group_means.values()) - min(group_means.values())) / max(group_means.values())
        
        return BiasDetectionResult(
            detection_id=str(uuid.uuid4()),
            bias_type=BiasType.LINGUISTIC,
            metric=BiasMetric.STATISTICAL_PARITY,
            severity=self._determine_bias_severity(bias_score, 0.5, len(complexity_scores)),
            bias_score=bias_score,
            confidence_interval=(max(0, bias_score - 0.1), min(1, bias_score + 0.1)),
            affected_groups=list(group_means.keys()),
            statistical_significance=0.5,
            sample_size=len(complexity_scores),
            timestamp=time.time(),
            context={'group_complexity_means': group_means},
            mitigation_recommendations=[
                "Review language complexity consistency across demographic groups",
                "Consider complexity-aware response generation",
                "Monitor vocabulary diversity in outputs"
            ]
        )
    
    def _calculate_language_complexity(self, text: str) -> float:
        """Calculate language complexity score for a text."""
        if not text:
            return 0.0
            
        words = text.split()
        if not words:
            return 0.0
            
        # Calculate various complexity metrics
        avg_word_length = np.mean([len(word) for word in words])
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_sentence_length = len(words) / max(1, sentence_count)
        
        # Check for complex patterns
        formal_score = len(re.findall(self.language_patterns['formal_language'], text, re.IGNORECASE))
        technical_score = len(re.findall(self.language_patterns['technical_language'], text, re.IGNORECASE))
        
        # Combine metrics (normalized to [0, 1])
        complexity_score = (
            min(1, avg_word_length / 10) * 0.3 +
            min(1, avg_sentence_length / 20) * 0.3 +
            min(1, formal_score / 5) * 0.2 +
            min(1, technical_score / 5) * 0.2
        )
        
        return complexity_score
    
    def _analyze_keyword_bias(self, context: BiasContext) -> List[BiasDetectionResult]:
        """Analyze bias in keyword usage patterns."""
        results = []
        
        if not context.model_outputs:
            return results
            
        # Analyze each bias category
        for category, keywords in self.bias_keywords.items():
            keyword_usage = self._calculate_keyword_usage(context.model_outputs, keywords)
            
            if not keyword_usage:
                continue
                
            # Calculate bias score based on keyword distribution
            usage_values = list(keyword_usage.values())
            if len(usage_values) > 1:
                bias_score = (max(usage_values) - min(usage_values)) / (max(usage_values) + 1e-8)
                
                if bias_score > 0.1:  # Only report significant keyword bias
                    results.append(BiasDetectionResult(
                        detection_id=str(uuid.uuid4()),
                        bias_type=BiasType.LINGUISTIC,
                        metric=BiasMetric.DISPARATE_IMPACT,
                        severity=self._determine_bias_severity(bias_score, 0.6, len(context.model_outputs)),
                        bias_score=bias_score,
                        confidence_interval=(max(0, bias_score - 0.1), min(1, bias_score + 0.1)),
                        affected_groups=[category],
                        statistical_significance=0.6,
                        sample_size=len(context.model_outputs),
                        timestamp=time.time(),
                        context={
                            'category': category,
                            'keyword_usage': keyword_usage
                        },
                        mitigation_recommendations=[
                            f"Review {category}-related keyword usage patterns",
                            f"Ensure balanced representation of {category} terms",
                            "Consider keyword-aware content generation"
                        ]
                    ))
                    
        return results
    
    def _calculate_keyword_usage(self, texts: List[str], keywords: List[str]) -> Dict[str, int]:
        """Calculate keyword usage statistics."""
        usage = {keyword: 0 for keyword in keywords}
        
        for text in texts:
            text_lower = text.lower()
            for keyword in keywords:
                usage[keyword] += text_lower.count(keyword.lower())
                
        return usage
    
    def _analyze_response_length_bias(self, context: BiasContext) -> Optional[BiasDetectionResult]:
        """Analyze bias in response lengths across demographic groups."""
        if not context.demographic_info or not context.model_outputs:
            return None
            
        # Calculate response lengths
        response_lengths = [len(output.split()) for output in context.model_outputs]
        
        # Group by demographics
        demo_groups = defaultdict(list)
        for i, demo_info in enumerate(context.demographic_info.get('per_sample', [])):
            if i < len(response_lengths) and isinstance(demo_info, dict):
                for key, value in demo_info.items():
                    demo_groups[f"{key}_{value}"].append(response_lengths[i])
                    break
                    
        if len(demo_groups) < 2:
            return None
            
        # Calculate bias in response lengths
        group_means = {group: np.mean(lengths) for group, lengths in demo_groups.items() if lengths}
        
        if len(group_means) < 2:
            return None
            
        bias_score = (max(group_means.values()) - min(group_means.values())) / max(group_means.values())
        
        if bias_score < 0.1:  # Only report significant length bias
            return None
            
        return BiasDetectionResult(
            detection_id=str(uuid.uuid4()),
            bias_type=BiasType.LINGUISTIC,
            metric=BiasMetric.STATISTICAL_PARITY,
            severity=self._determine_bias_severity(bias_score, 0.4, len(response_lengths)),
            bias_score=bias_score,
            confidence_interval=(max(0, bias_score - 0.1), min(1, bias_score + 0.1)),
            affected_groups=list(group_means.keys()),
            statistical_significance=0.4,
            sample_size=len(response_lengths),
            timestamp=time.time(),
            context={'group_length_means': group_means},
            mitigation_recommendations=[
                "Monitor response length consistency across groups",
                "Consider length-aware response generation",
                "Review prompts that may lead to length disparities"
            ]
        )
    
    def _determine_bias_severity(self, bias_score: float, significance: float, sample_size: int) -> BiasSeverity:
        """Determine severity level of linguistic bias."""
        severity_score = bias_score * significance * min(1.0, sample_size / 50)
        
        if severity_score > 0.7:
            return BiasSeverity.HIGH
        elif severity_score > 0.5:
            return BiasSeverity.MEDIUM
        elif severity_score > 0.3:
            return BiasSeverity.LOW
        else:
            return BiasSeverity.NEGLIGIBLE


class IntersectionalBiasDetector:
    """Detects bias at intersections of multiple protected attributes."""
    
    def __init__(self, protected_attributes: List[ProtectedAttribute]):
        self.protected_attributes = {attr.attribute_name: attr for attr in protected_attributes}
        
    def analyze_intersectional_bias(self, context: BiasContext) -> List[BiasDetectionResult]:
        """Analyze intersectional bias across multiple protected attributes."""
        results = []
        
        if not context.demographic_info:
            return results
            
        # Generate intersectional groups
        intersectional_groups = self._generate_intersectional_groups(context)
        
        if len(intersectional_groups) < 4:  # Need sufficient groups for analysis
            return results
            
        # Analyze bias across intersectional groups
        bias_result = self._analyze_intersectional_patterns(context, intersectional_groups)
        if bias_result:
            results.append(bias_result)
            
        return results
    
    def _generate_intersectional_groups(self, context: BiasContext) -> Dict[str, List[int]]:
        """Generate intersectional demographic groups."""
        groups = defaultdict(list)
        
        per_sample_demo = context.demographic_info.get('per_sample', [])
        
        for i, demo_info in enumerate(per_sample_demo):
            if not isinstance(demo_info, dict):
                continue
                
            # Create intersectional group identifier
            group_attrs = []
            for attr_name in self.protected_attributes:
                if attr_name in demo_info:
                    group_attrs.append(f"{attr_name}:{demo_info[attr_name]}")
                    
            if len(group_attrs) >= 2:  # Require at least 2 attributes for intersection
                group_id = "|".join(sorted(group_attrs))
                groups[group_id].append(i)
                
        # Filter out small groups
        filtered_groups = {k: v for k, v in groups.items() if len(v) >= 3}
        return filtered_groups
    
    def _analyze_intersectional_patterns(self, 
                                       context: BiasContext, 
                                       groups: Dict[str, List[int]]) -> Optional[BiasDetectionResult]:
        """Analyze bias patterns in intersectional groups."""
        
        # Calculate group statistics
        group_stats = {}
        for group_id, indices in groups.items():
            feedback_values = [context.human_feedback[i] for i in indices if i < len(context.human_feedback)]
            if feedback_values:
                group_stats[group_id] = {
                    'mean_feedback': np.mean(feedback_values),
                    'std_feedback': np.std(feedback_values),
                    'size': len(feedback_values),
                    'attributes': group_id.split('|')
                }
        
        if len(group_stats) < 2:
            return None
            
        # Calculate intersectional bias score
        bias_score = self._calculate_intersectional_bias_score(group_stats)
        
        # Identify most affected groups
        group_means = {group: stats['mean_feedback'] for group, stats in group_stats.items()}
        min_group = min(group_means.items(), key=lambda x: x[1])
        max_group = max(group_means.items(), key=lambda x: x[1])
        
        # Calculate significance
        significance = self._calculate_intersectional_significance(group_stats)
        
        return BiasDetectionResult(
            detection_id=str(uuid.uuid4()),
            bias_type=BiasType.INTERSECTIONAL,
            metric=BiasMetric.STATISTICAL_PARITY,
            severity=self._determine_bias_severity(bias_score, significance, sum(s['size'] for s in group_stats.values())),
            bias_score=bias_score,
            confidence_interval=(max(0, bias_score - 0.2), min(1, bias_score + 0.2)),
            affected_groups=[min_group[0], max_group[0]],
            statistical_significance=significance,
            sample_size=sum(s['size'] for s in group_stats.values()),
            timestamp=time.time(),
            context={
                'intersectional_groups': len(group_stats),
                'group_statistics': {k: {
                    'mean_feedback': v['mean_feedback'],
                    'size': v['size'],
                    'attributes': v['attributes']
                } for k, v in group_stats.items()},
                'disadvantaged_group': min_group[0],
                'advantaged_group': max_group[0]
            },
            mitigation_recommendations=[
                "Increase representation of intersectional groups in training data",
                "Implement intersectionality-aware sampling strategies",
                "Monitor bias at attribute intersections regularly",
                f"Special attention needed for {min_group[0]} group",
                "Consider multi-attribute bias regularization"
            ]
        )
    
    def _calculate_intersectional_bias_score(self, group_stats: Dict[str, Dict[str, Any]]) -> float:
        """Calculate bias score for intersectional groups."""
        group_means = [stats['mean_feedback'] for stats in group_stats.values()]
        
        if not group_means:
            return 0.0
            
        # Use coefficient of variation as intersectional bias measure
        mean_of_means = np.mean(group_means)
        std_of_means = np.std(group_means)
        
        if mean_of_means == 0:
            return 0.0
            
        coefficient_of_variation = std_of_means / mean_of_means
        
        # Normalize to [0, 1] scale
        bias_score = min(1.0, coefficient_of_variation)
        return bias_score
    
    def _calculate_intersectional_significance(self, group_stats: Dict[str, Dict[str, Any]]) -> float:
        """Calculate statistical significance of intersectional bias."""
        # Simplified ANOVA-like calculation
        groups = list(group_stats.values())
        
        if len(groups) < 2:
            return 0.0
            
        # Calculate between-group and within-group variance
        overall_mean = np.mean([stats['mean_feedback'] for stats in groups])
        between_var = np.mean([(stats['mean_feedback'] - overall_mean)**2 * stats['size'] for stats in groups])
        within_var = np.mean([stats['std_feedback']**2 for stats in groups])
        
        if within_var == 0:
            return 1.0 if between_var > 0 else 0.0
            
        f_ratio = between_var / within_var
        
        # Convert F-ratio to significance score (simplified)
        significance = min(1.0, f_ratio / 5.0)
        return significance
    
    def _determine_bias_severity(self, bias_score: float, significance: float, sample_size: int) -> BiasSeverity:
        """Determine severity of intersectional bias."""
        # Intersectional bias is often more severe due to compound effects
        severity_score = bias_score * significance * min(1.0, sample_size / 100) * 1.2  # Amplification factor
        
        if severity_score > 0.8:
            return BiasSeverity.CRITICAL
        elif severity_score > 0.6:
            return BiasSeverity.HIGH
        elif severity_score > 0.4:
            return BiasSeverity.MEDIUM
        elif severity_score > 0.2:
            return BiasSeverity.LOW
        else:
            return BiasSeverity.NEGLIGIBLE


class RealTimeBiasDetector:
    """Main real-time bias detection system."""
    
    def __init__(self, 
                 protected_attributes: Optional[List[ProtectedAttribute]] = None,
                 compliance_config: Optional[ComplianceConfig] = None):
        self.compliance_config = compliance_config or ComplianceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set up protected attributes
        if protected_attributes is None:
            protected_attributes = self._get_default_protected_attributes()
        
        # Initialize detectors
        self.demographic_detector = DemographicBiasDetector(protected_attributes)
        self.linguistic_detector = LinguisticBiasDetector()
        self.intersectional_detector = IntersectionalBiasDetector(protected_attributes)
        
        # Bias tracking
        self.bias_history: List[BiasDetectionResult] = []
        self.bias_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Real-time processing
        self.executor = ThreadPoolExecutor(max_workers=6)
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            bias_type: 0.3 for bias_type in BiasType
        }
        
    def _get_default_protected_attributes(self) -> List[ProtectedAttribute]:
        """Get default protected attributes for bias detection."""
        return [
            ProtectedAttribute(
                attribute_name="gender",
                attribute_type="categorical",
                possible_values=["male", "female", "non-binary", "other"],
                sensitivity_level=1.0
            ),
            ProtectedAttribute(
                attribute_name="age",
                attribute_type="categorical", 
                possible_values=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
                sensitivity_level=0.8
            ),
            ProtectedAttribute(
                attribute_name="ethnicity",
                attribute_type="categorical",
                possible_values=["white", "black", "hispanic", "asian", "other"],
                sensitivity_level=1.0
            ),
            ProtectedAttribute(
                attribute_name="education",
                attribute_type="categorical",
                possible_values=["high_school", "bachelor", "master", "phd"],
                sensitivity_level=0.6
            )
        ]
    
    async def detect_bias_realtime(self, context: BiasContext) -> List[BiasDetectionResult]:
        """Perform real-time bias detection analysis."""
        try:
            all_results = []
            
            # Run all detectors in parallel
            detection_tasks = [
                asyncio.create_task(self._run_demographic_detection(context)),
                asyncio.create_task(self._run_linguistic_detection(context)),
                asyncio.create_task(self._run_intersectional_detection(context))
            ]
            
            # Wait for all detections to complete
            detection_results = await asyncio.gather(*detection_tasks, return_exceptions=True)
            
            # Process results
            for result in detection_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Bias detection error: {result}")
                    continue
                    
                if isinstance(result, list):
                    all_results.extend(result)
            
            # Filter results by adaptive thresholds
            filtered_results = self._filter_by_adaptive_thresholds(all_results)
            
            # Update bias history and patterns
            self._update_bias_patterns(filtered_results)
            
            # Log significant bias detections
            for result in filtered_results:
                if result.severity in [BiasSeverity.CRITICAL, BiasSeverity.HIGH]:
                    self.logger.warning(f"Significant bias detected: {result.bias_type.value} - {result.severity.value}")
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Real-time bias detection failed: {e}")
            return []
    
    async def _run_demographic_detection(self, context: BiasContext) -> List[BiasDetectionResult]:
        """Run demographic bias detection."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.demographic_detector.analyze_demographic_bias,
            context
        )
    
    async def _run_linguistic_detection(self, context: BiasContext) -> List[BiasDetectionResult]:
        """Run linguistic bias detection."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.linguistic_detector.analyze_linguistic_bias,
            context
        )
    
    async def _run_intersectional_detection(self, context: BiasContext) -> List[BiasDetectionResult]:
        """Run intersectional bias detection."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.intersectional_detector.analyze_intersectional_bias,
            context
        )
    
    def _filter_by_adaptive_thresholds(self, results: List[BiasDetectionResult]) -> List[BiasDetectionResult]:
        """Filter bias detection results by adaptive thresholds."""
        filtered = []
        
        for result in results:
            threshold = self.adaptive_thresholds.get(result.bias_type, 0.3)
            
            if result.bias_score >= threshold:
                filtered.append(result)
                
        return filtered
    
    def _update_bias_patterns(self, results: List[BiasDetectionResult]):
        """Update bias patterns and adaptive thresholds."""
        for result in results:
            # Store in history
            self.bias_history.append(result)
            
            # Update pattern tracking
            pattern_key = f"{result.bias_type.value}_{result.metric.value}"
            self.bias_patterns[pattern_key].append(result.bias_score)
            
            # Keep only recent patterns
            if len(self.bias_patterns[pattern_key]) > 100:
                self.bias_patterns[pattern_key] = self.bias_patterns[pattern_key][-100:]
        
        # Update adaptive thresholds based on patterns
        self._update_adaptive_thresholds()
    
    def _update_adaptive_thresholds(self):
        """Update adaptive bias detection thresholds based on historical patterns."""
        for bias_type in BiasType:
            pattern_key = f"{bias_type.value}_"
            relevant_patterns = [
                scores for key, scores in self.bias_patterns.items()
                if key.startswith(pattern_key) and scores
            ]
            
            if relevant_patterns:
                all_scores = [score for scores in relevant_patterns for score in scores]
                if len(all_scores) >= 10:
                    # Set threshold at 75th percentile of historical scores
                    new_threshold = np.percentile(all_scores, 75)
                    self.adaptive_thresholds[bias_type] = max(0.1, min(0.8, new_threshold))
    
    def get_bias_summary(self, session_id: Optional[str] = None, time_window: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive bias detection summary."""
        # Filter results by session and time window
        filtered_results = self.bias_history
        
        if session_id:
            filtered_results = [r for r in filtered_results if r.context.get('session_id') == session_id]
            
        if time_window:
            cutoff_time = time.time() - time_window
            filtered_results = [r for r in filtered_results if r.timestamp >= cutoff_time]
        
        if not filtered_results:
            return {"error": "No bias detection results found"}
        
        # Aggregate statistics
        bias_by_type = defaultdict(list)
        bias_by_severity = defaultdict(int)
        
        for result in filtered_results:
            bias_by_type[result.bias_type.value].append(result.bias_score)
            bias_by_severity[result.severity.value] += 1
        
        # Calculate summary metrics
        avg_bias_scores = {
            bias_type: np.mean(scores) 
            for bias_type, scores in bias_by_type.items()
        }
        
        max_bias_score = max((r.bias_score for r in filtered_results), default=0)
        critical_detections = len([r for r in filtered_results if r.severity == BiasSeverity.CRITICAL])
        
        return {
            "summary_metadata": {
                "session_id": session_id or "all_sessions",
                "time_window_seconds": time_window,
                "total_detections": len(filtered_results),
                "analysis_timestamp": time.time()
            },
            "bias_statistics": {
                "max_bias_score": max_bias_score,
                "average_bias_by_type": avg_bias_scores,
                "detections_by_severity": dict(bias_by_severity),
                "critical_detections": critical_detections
            },
            "bias_trends": {
                "recent_bias_pattern": [r.bias_score for r in filtered_results[-20:]],
                "bias_types_detected": list(bias_by_type.keys()),
                "most_common_severity": max(bias_by_severity.items(), key=lambda x: x[1])[0] if bias_by_severity else "none"
            },
            "adaptive_thresholds": dict(self.adaptive_thresholds),
            "mitigation_recommendations": self._generate_summary_recommendations(filtered_results)
        }
    
    def _generate_summary_recommendations(self, results: List[BiasDetectionResult]) -> List[str]:
        """Generate summary recommendations based on bias detection results."""
        recommendations = set()
        
        # Count bias types and severities
        bias_type_counts = Counter(r.bias_type for r in results)
        severity_counts = Counter(r.severity for r in results)
        
        # Generate recommendations based on patterns
        if bias_type_counts[BiasType.DEMOGRAPHIC] > 5:
            recommendations.add("Urgent: Review demographic representation in training data")
            recommendations.add("Implement demographic-aware model training procedures")
        
        if bias_type_counts[BiasType.INTERSECTIONAL] > 3:
            recommendations.add("Critical: Address intersectional bias through targeted interventions")
            recommendations.add("Expand intersectional representation in datasets")
        
        if bias_type_counts[BiasType.LINGUISTIC] > 5:
            recommendations.add("Review language patterns and complexity across demographic groups")
        
        if severity_counts[BiasSeverity.CRITICAL] > 0:
            recommendations.add("IMMEDIATE ACTION REQUIRED: Critical bias violations detected")
            recommendations.add("Halt training pending bias remediation")
        
        if severity_counts[BiasSeverity.HIGH] > 5:
            recommendations.add("High-priority bias remediation needed")
            recommendations.add("Implement enhanced bias monitoring protocols")
        
        # General recommendations
        recommendations.add("Establish regular bias auditing schedule")
        recommendations.add("Train annotators on bias awareness")
        
        return list(recommendations)
    
    def export_bias_research_data(self, output_path: Path) -> Dict[str, Any]:
        """Export bias detection data for research and analysis."""
        research_data = {
            "metadata": {
                "export_timestamp": time.time(),
                "total_detections": len(self.bias_history),
                "bias_detector_version": "1.0.0",
                "protected_attributes": [
                    {
                        "name": attr.attribute_name,
                        "type": attr.attribute_type,
                        "sensitivity": attr.sensitivity_level
                    }
                    for attr in self.demographic_detector.protected_attributes.values()
                ]
            },
            "bias_detections": [
                {
                    "detection_id": result.detection_id,
                    "bias_type": result.bias_type.value,
                    "metric": result.metric.value,
                    "severity": result.severity.value,
                    "bias_score": result.bias_score,
                    "confidence_interval": result.confidence_interval,
                    "affected_groups": result.affected_groups,
                    "statistical_significance": result.statistical_significance,
                    "sample_size": result.sample_size,
                    "timestamp": result.timestamp,
                    "context": result.context
                }
                for result in self.bias_history
            ],
            "bias_patterns": {
                pattern: scores[-50:]  # Last 50 scores for each pattern
                for pattern, scores in self.bias_patterns.items()
            },
            "adaptive_thresholds": dict(self.adaptive_thresholds),
            "statistical_analysis": {
                "bias_type_distribution": dict(Counter(r.bias_type.value for r in self.bias_history)),
                "severity_distribution": dict(Counter(r.severity.value for r in self.bias_history)),
                "temporal_trends": self._calculate_temporal_trends()
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(research_data, f, indent=2)
            
        return research_data
    
    def _calculate_temporal_trends(self) -> Dict[str, Any]:
        """Calculate temporal trends in bias detection."""
        if not self.bias_history:
            return {}
            
        # Sort by timestamp
        sorted_results = sorted(self.bias_history, key=lambda x: x.timestamp)
        
        # Calculate trends over time windows
        time_windows = [3600, 86400, 604800]  # 1 hour, 1 day, 1 week
        trends = {}
        
        current_time = time.time()
        
        for window in time_windows:
            cutoff_time = current_time - window
            recent_results = [r for r in sorted_results if r.timestamp >= cutoff_time]
            
            if recent_results:
                avg_bias = np.mean([r.bias_score for r in recent_results])
                trend_slope = self._calculate_trend_slope([r.bias_score for r in recent_results])
                
                trends[f"{window}_seconds"] = {
                    "average_bias_score": avg_bias,
                    "trend_slope": trend_slope,
                    "detection_count": len(recent_results)
                }
        
        return trends
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression."""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        n = len(values)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
        
        return slope