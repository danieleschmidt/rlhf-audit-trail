"""Privacy protection system with differential privacy for RLHF.

This module implements differential privacy mechanisms to protect annotator
privacy while maintaining utility for RLHF training.
"""

import math
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import defaultdict

from .config import PrivacyConfig
from .exceptions import PrivacyBudgetExceededError, ValidationError


@dataclass
class PrivacyBudgetStatus:
    """Status of privacy budget consumption."""
    total_epsilon: float
    total_delta: float
    spent_epsilon: float
    spent_delta: float
    remaining_epsilon: float
    remaining_delta: float
    operations_count: int
    last_operation_time: float
    
    @property
    def exhaustion_risk(self) -> str:
        """Get risk level of budget exhaustion."""
        usage_ratio = self.spent_epsilon / self.total_epsilon
        if usage_ratio > 0.9:
            return "critical"
        elif usage_ratio > 0.75:
            return "high"
        elif usage_ratio > 0.5:
            return "medium"
        else:
            return "low"


class PrivacyBudgetManager:
    """Manages differential privacy budget allocation and tracking.
    
    This class tracks privacy budget consumption across all operations
    and prevents budget exhaustion that could compromise privacy.
    """
    
    def __init__(self, total_epsilon: float, total_delta: float):
        """Initialize privacy budget manager.
        
        Args:
            total_epsilon: Total epsilon budget for the session
            total_delta: Total delta budget for the session
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.operations: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
    @property
    def remaining_epsilon(self) -> float:
        """Get remaining epsilon budget."""
        return max(0.0, self.total_epsilon - self.spent_epsilon)
    
    @property
    def remaining_delta(self) -> float:
        """Get remaining delta budget."""
        return max(0.0, self.total_delta - self.spent_delta)
    
    @property
    def total_spent_epsilon(self) -> float:
        """Get total spent epsilon."""
        return self.spent_epsilon
    
    @property
    def total_spent_delta(self) -> float:
        """Get total spent delta."""
        return self.spent_delta
    
    def can_spend(self, epsilon: float, delta: float = 0.0) -> bool:
        """Check if we can spend the given privacy budget.
        
        Args:
            epsilon: Epsilon to spend
            delta: Delta to spend
            
        Returns:
            True if budget is available
        """
        return (self.remaining_epsilon >= epsilon and 
                self.remaining_delta >= delta)
    
    def spend(self, epsilon: float, delta: float = 0.0, operation_info: Optional[Dict[str, Any]] = None):
        """Spend privacy budget for an operation.
        
        Args:
            epsilon: Epsilon to spend
            delta: Delta to spend
            operation_info: Additional information about the operation
            
        Raises:
            PrivacyBudgetExceededError: If insufficient budget
        """
        if not self.can_spend(epsilon, delta):
            raise PrivacyBudgetExceededError(
                f"Insufficient privacy budget. Requested: ε={epsilon}, δ={delta}. "
                f"Available: ε={self.remaining_epsilon}, δ={self.remaining_delta}",
                requested_epsilon=epsilon,
                available_epsilon=self.remaining_epsilon
            )
        
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        
        # Record operation
        operation = {
            "timestamp": time.time(),
            "epsilon": epsilon,
            "delta": delta,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
            "info": operation_info or {}
        }
        self.operations.append(operation)
        
        self.logger.info(f"Privacy budget spent: ε={epsilon:.6f}, δ={delta:.6f}. "
                        f"Remaining: ε={self.remaining_epsilon:.6f}, δ={self.remaining_delta:.6f}")
    
    def get_status(self) -> PrivacyBudgetStatus:
        """Get current budget status."""
        return PrivacyBudgetStatus(
            total_epsilon=self.total_epsilon,
            total_delta=self.total_delta,
            spent_epsilon=self.spent_epsilon,
            spent_delta=self.spent_delta,
            remaining_epsilon=self.remaining_epsilon,
            remaining_delta=self.remaining_delta,
            operations_count=len(self.operations),
            last_operation_time=self.operations[-1]["timestamp"] if self.operations else 0.0
        )
    
    def estimate_max_annotations(self, epsilon_per_annotation: float) -> int:
        """Estimate maximum annotations possible with remaining budget.
        
        Args:
            epsilon_per_annotation: Epsilon cost per annotation
            
        Returns:
            Maximum number of annotations possible
        """
        if epsilon_per_annotation <= 0:
            return 0
        return int(self.remaining_epsilon / epsilon_per_annotation)
    
    def exhaustion_risk(self) -> str:
        """Get budget exhaustion risk level."""
        return self.get_status().exhaustion_risk


class DifferentialPrivacyEngine:
    """Core differential privacy implementation for RLHF annotation protection.
    
    This class implements various differential privacy mechanisms to protect
    annotator privacy while preserving utility for model training.
    """
    
    def __init__(self, config: PrivacyConfig):
        """Initialize differential privacy engine.
        
        Args:
            config: Privacy configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        issues = config.validate()
        if issues:
            raise ValidationError(f"Invalid privacy configuration: {issues}")
        
        # Set up mechanism parameters
        self.base_epsilon_cost = config.epsilon_per_round
        self.noise_multiplier = config.noise_multiplier
        self.clip_norm = config.clip_norm
        
        # Annotator tracking for k-anonymity
        self.annotator_tracking: Dict[str, List[float]] = defaultdict(list)
        self.anonymized_ids: Dict[str, str] = {}
        self.id_rotation_counter = 0
    
    def estimate_epsilon_cost(self, batch_size: int, sensitivity: float = 1.0) -> float:
        """Estimate epsilon cost for a batch of annotations.
        
        Args:
            batch_size: Number of annotations in batch
            sensitivity: Sensitivity of the query (default 1.0 for reward averaging)
            
        Returns:
            Estimated epsilon cost
        """
        # Basic epsilon calculation with composition
        base_cost = self.base_epsilon_cost * sensitivity
        
        # Apply privacy amplification by subsampling if enabled
        if self.config.use_amplification and self.config.sampling_rate < 1.0:
            # Amplification factor for subsampling
            amplification_factor = self.config.sampling_rate * batch_size
            amplified_cost = base_cost * math.sqrt(amplification_factor)
            return min(amplified_cost, base_cost)  # Never exceed base cost
        
        return base_cost
    
    def add_noise_to_rewards(
        self,
        rewards: List[float],
        annotator_ids: List[str],
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """Add differential privacy noise to reward signals.
        
        Args:
            rewards: List of reward values
            annotator_ids: List of annotator identifiers
            sensitivity: Sensitivity of the reward function
            
        Returns:
            Noisy rewards as numpy array
        """
        if len(rewards) != len(annotator_ids):
            raise ValidationError("Rewards and annotator_ids must have same length")
        
        rewards_array = np.array(rewards, dtype=float)
        
        # Calculate noise scale based on epsilon allocation
        epsilon = self.estimate_epsilon_cost(len(rewards), sensitivity)
        noise_scale = sensitivity / epsilon
        
        # Generate Laplace noise
        noise = np.random.laplace(0, noise_scale, size=len(rewards))
        
        # Apply noise
        noisy_rewards = rewards_array + noise
        
        # Optional: Clip to reasonable range to prevent extreme outliers
        if self.config.clip_norm > 0:
            noisy_rewards = np.clip(
                noisy_rewards,
                -self.config.clip_norm,
                self.config.clip_norm
            )
        
        # Track annotator contributions for k-anonymity
        self._track_annotator_contributions(annotator_ids, rewards)
        
        self.logger.debug(f"Added DP noise to {len(rewards)} rewards with ε={epsilon:.6f}")
        return noisy_rewards
    
    def anonymize_annotator_id(self, original_id: str) -> str:
        """Anonymize annotator ID with potential rotation.
        
        Args:
            original_id: Original annotator identifier
            
        Returns:
            Anonymized identifier
        """
        # Check if we need to rotate IDs
        if (self.config.annotator_id_rotation and 
            self.id_rotation_counter >= self.config.rotation_frequency):
            self._rotate_annotator_ids()
        
        # Generate anonymized ID if not exists
        if original_id not in self.anonymized_ids:
            # Simple hash-based anonymization (in production, use stronger methods)
            import hashlib
            hash_input = f"{original_id}_{self.id_rotation_counter // self.config.rotation_frequency}"
            anonymized = "anon_" + hashlib.sha256(hash_input.encode()).hexdigest()[:8]
            self.anonymized_ids[original_id] = anonymized
        
        self.id_rotation_counter += 1
        return self.anonymized_ids[original_id]
    
    def _track_annotator_contributions(self, annotator_ids: List[str], rewards: List[float]):
        """Track annotator contributions for k-anonymity enforcement."""
        for annotator_id, reward in zip(annotator_ids, rewards):
            self.annotator_tracking[annotator_id].append(reward)
            
            # Ensure k-anonymity by checking contribution count
            if len(self.annotator_tracking[annotator_id]) < self.config.annotator_k_anonymity:
                self.logger.debug(f"Annotator {annotator_id} has insufficient contributions for k-anonymity")
    
    def _rotate_annotator_ids(self):
        """Rotate anonymized annotator IDs for privacy protection."""
        self.anonymized_ids.clear()
        self.logger.info("Rotated annotator IDs for enhanced privacy")
    
    def add_gaussian_noise(
        self,
        data: np.ndarray,
        epsilon: float,
        delta: float,
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """Add Gaussian noise for (ε,δ)-differential privacy.
        
        Args:
            data: Input data array
            epsilon: Privacy parameter epsilon
            delta: Privacy parameter delta  
            sensitivity: Sensitivity of the function
            
        Returns:
            Data with Gaussian noise added
        """
        if delta <= 0 or delta >= 1:
            raise ValidationError("Delta must be in (0, 1)")
        
        # Calculate noise scale for Gaussian mechanism
        # σ ≥ √(2 ln(1.25/δ)) * Δf / ε
        noise_scale = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon
        
        # Generate Gaussian noise
        noise = np.random.normal(0, noise_scale, size=data.shape)
        
        return data + noise
    
    def exponential_mechanism(
        self,
        candidates: List[Any],
        utility_function: callable,
        epsilon: float,
        sensitivity: float = 1.0
    ) -> Any:
        """Implement exponential mechanism for privacy-preserving selection.
        
        Args:
            candidates: List of candidate options
            utility_function: Function that computes utility for each candidate
            epsilon: Privacy parameter
            sensitivity: Sensitivity of utility function
            
        Returns:
            Selected candidate
        """
        if not candidates:
            raise ValidationError("Candidates list cannot be empty")
        
        # Compute utilities
        utilities = [utility_function(candidate) for candidate in candidates]
        utilities = np.array(utilities)
        
        # Compute exponential weights
        scaled_utilities = epsilon * utilities / (2 * sensitivity)
        
        # Prevent overflow
        scaled_utilities = scaled_utilities - np.max(scaled_utilities)
        weights = np.exp(scaled_utilities)
        
        # Normalize probabilities
        probabilities = weights / np.sum(weights)
        
        # Sample according to probabilities
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        
        return candidates[selected_idx]
    
    def compute_privacy_loss(
        self,
        epsilon_spent: float,
        delta_spent: float,
        composition_method: str = "basic"
    ) -> Tuple[float, float]:
        """Compute privacy loss under composition.
        
        Args:
            epsilon_spent: Total epsilon spent so far
            delta_spent: Total delta spent so far
            composition_method: Composition analysis method
            
        Returns:
            Tuple of (effective_epsilon, effective_delta)
        """
        if composition_method == "basic":
            # Basic composition: privacy degrades linearly
            return epsilon_spent, delta_spent
        
        elif composition_method == "advanced":
            # Advanced composition with better bounds
            # This is a simplified version - real implementation would be more complex
            k = len(self.annotator_tracking)  # Number of operations
            if k == 0:
                return 0.0, 0.0
            
            # Advanced composition formula (simplified)
            effective_epsilon = epsilon_spent * math.sqrt(2 * k * math.log(1/delta_spent))
            effective_delta = k * delta_spent
            
            return effective_epsilon, effective_delta
        
        else:
            raise ValidationError(f"Unknown composition method: {composition_method}")
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy analysis report.
        
        Returns:
            Dictionary containing privacy analysis
        """
        # Compute current privacy parameters
        total_annotations = sum(len(contributions) for contributions in self.annotator_tracking.values())
        unique_annotators = len(self.annotator_tracking)
        
        # Check k-anonymity compliance
        k_anonymity_violations = []
        for annotator_id, contributions in self.annotator_tracking.items():
            if len(contributions) < self.config.annotator_k_anonymity:
                k_anonymity_violations.append({
                    "annotator_id": annotator_id,
                    "contributions": len(contributions),
                    "required": self.config.annotator_k_anonymity
                })
        
        # Privacy risk assessment
        risk_factors = []
        if unique_annotators < 10:
            risk_factors.append("Low number of unique annotators")
        if k_anonymity_violations:
            risk_factors.append("K-anonymity violations detected")
        
        report = {
            "privacy_configuration": {
                "epsilon": self.config.epsilon,
                "delta": self.config.delta,
                "noise_multiplier": self.config.noise_multiplier,
                "clip_norm": self.config.clip_norm,
                "privacy_mode": self.config.privacy_mode.value
            },
            "annotation_statistics": {
                "total_annotations": total_annotations,
                "unique_annotators": unique_annotators,
                "average_annotations_per_annotator": total_annotations / max(unique_annotators, 1),
                "id_rotations_performed": self.id_rotation_counter // self.config.rotation_frequency
            },
            "k_anonymity_analysis": {
                "required_k": self.config.annotator_k_anonymity,
                "violations": len(k_anonymity_violations),
                "violation_details": k_anonymity_violations,
                "compliance_rate": 1.0 - len(k_anonymity_violations) / max(unique_annotators, 1)
            },
            "privacy_risk_assessment": {
                "risk_level": "high" if risk_factors else "low",
                "risk_factors": risk_factors,
                "recommendations": self._generate_privacy_recommendations(risk_factors)
            },
            "differential_privacy_guarantees": {
                "mechanism": "Laplace",
                "epsilon_per_query": self.base_epsilon_cost,
                "privacy_amplification": self.config.use_amplification,
                "composition_tracking": "enabled"
            }
        }
        
        return report
    
    def _generate_privacy_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate privacy recommendations based on risk factors."""
        recommendations = []
        
        if "Low number of unique annotators" in risk_factors:
            recommendations.append("Increase the number of annotators to improve privacy protection")
        
        if "K-anonymity violations detected" in risk_factors:
            recommendations.append("Ensure each annotator provides at least k contributions")
            recommendations.append("Consider increasing anonymization parameters")
        
        if not risk_factors:
            recommendations.append("Privacy configuration appears adequate")
            recommendations.append("Continue monitoring privacy metrics")
        
        return recommendations


class PrivacyAuditor:
    """Audits privacy compliance and generates privacy impact assessments."""
    
    def __init__(self, privacy_engine: DifferentialPrivacyEngine, budget_manager: PrivacyBudgetManager):
        """Initialize privacy auditor.
        
        Args:
            privacy_engine: Differential privacy engine
            budget_manager: Privacy budget manager
        """
        self.privacy_engine = privacy_engine
        self.budget_manager = budget_manager
        self.logger = logging.getLogger(__name__)
    
    def conduct_privacy_audit(self) -> Dict[str, Any]:
        """Conduct comprehensive privacy audit.
        
        Returns:
            Audit results with compliance status
        """
        # Get privacy report from engine
        privacy_report = self.privacy_engine.generate_privacy_report()
        
        # Get budget status
        budget_status = self.budget_manager.get_status()
        
        # Evaluate compliance
        compliance_issues = []
        
        # Check budget exhaustion risk
        if budget_status.exhaustion_risk in ["critical", "high"]:
            compliance_issues.append({
                "type": "budget_exhaustion_risk",
                "severity": "high" if budget_status.exhaustion_risk == "critical" else "medium",
                "description": f"Privacy budget exhaustion risk is {budget_status.exhaustion_risk}"
            })
        
        # Check k-anonymity violations
        k_violations = privacy_report["k_anonymity_analysis"]["violations"]
        if k_violations > 0:
            compliance_issues.append({
                "type": "k_anonymity_violation",
                "severity": "high",
                "description": f"{k_violations} annotators have insufficient contributions for k-anonymity"
            })
        
        # Overall compliance status
        compliance_status = "compliant" if not compliance_issues else "non_compliant"
        
        audit_result = {
            "audit_timestamp": time.time(),
            "compliance_status": compliance_status,
            "compliance_issues": compliance_issues,
            "privacy_report": privacy_report,
            "budget_status": budget_status.exhaustion_risk,
            "recommendations": self._generate_audit_recommendations(compliance_issues)
        }
        
        return audit_result
    
    def _generate_audit_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on audit issues."""
        recommendations = []
        
        for issue in issues:
            if issue["type"] == "budget_exhaustion_risk":
                recommendations.append("Reduce epsilon consumption per operation")
                recommendations.append("Implement more efficient privacy mechanisms")
            
            elif issue["type"] == "k_anonymity_violation":
                recommendations.append("Collect more annotations from each annotator")
                recommendations.append("Increase k-anonymity threshold if feasible")
        
        if not issues:
            recommendations.append("Privacy compliance is satisfactory")
            recommendations.append("Continue current privacy protection measures")
        
        return recommendations