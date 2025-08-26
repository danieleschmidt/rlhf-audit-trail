"""Federated Learning Audit Trail System.

This module provides comprehensive auditing capabilities for federated learning
scenarios in RLHF, including:
- Cross-institutional audit coordination
- Privacy-preserving aggregation verification
- Byzantine fault detection in federated settings
- Differential privacy budget tracking across participants
- Model contribution fairness analysis
"""

import asyncio
import json
import time
import uuid
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from collections import defaultdict
import socket
from concurrent.futures import ThreadPoolExecutor

from .config import PrivacyConfig, SecurityConfig
from .exceptions import AuditTrailError
from .crypto import CryptographicEngine


class ParticipantRole(Enum):
    """Roles in federated learning system."""
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant" 
    VALIDATOR = "validator"
    AUDITOR = "auditor"


class AggregationMethod(Enum):
    """Federated aggregation methods."""
    FEDAVG = "federated_averaging"
    FEDPROX = "federated_proximal"
    SCAFFOLD = "scaffold"
    FEDNOVA = "fed_nova"
    CUSTOM = "custom"


class FederatedEventType(Enum):
    """Types of federated learning events."""
    ROUND_START = "round_start"
    MODEL_UPDATE = "model_update"
    AGGREGATION = "aggregation"
    VALIDATION = "validation"
    BYZANTINE_DETECTION = "byzantine_detection"
    PRIVACY_BUDGET_UPDATE = "privacy_budget_update"
    FAIRNESS_ANALYSIS = "fairness_analysis"


@dataclass
class ParticipantInfo:
    """Information about a federated learning participant."""
    participant_id: str
    institution_name: str
    role: ParticipantRole
    public_key: str
    ip_address: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    reputation_score: float = 1.0
    privacy_budget_remaining: float = 10.0
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelUpdate:
    """A model update from a federated participant."""
    update_id: str
    participant_id: str
    round_number: int
    model_delta: Dict[str, Any]  # Serialized model parameters
    gradient_norm: float
    training_samples: int
    privacy_cost: float
    timestamp: float
    signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_hash(self) -> str:
        """Compute cryptographic hash of the model update."""
        update_dict = {
            "participant_id": self.participant_id,
            "round_number": self.round_number,
            "gradient_norm": self.gradient_norm,
            "training_samples": self.training_samples,
            "privacy_cost": self.privacy_cost,
            "timestamp": self.timestamp
        }
        # Note: Excluding model_delta from hash for efficiency
        # In production, would hash compressed/summarized model representation
        content = json.dumps(update_dict, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class AggregationResult:
    """Result of federated model aggregation."""
    aggregation_id: str
    round_number: int
    method: AggregationMethod
    participating_updates: List[str]  # Update IDs
    aggregated_model_hash: str
    convergence_metric: float
    fairness_score: float
    byzantine_participants: List[str]
    privacy_cost_total: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FederatedAuditEvent:
    """Audit event in federated learning system."""
    event_id: str
    event_type: FederatedEventType
    round_number: int
    participant_id: Optional[str]
    timestamp: float
    data: Dict[str, Any]
    signature: str
    verified: bool = False


class ByzantineDetector:
    """Detects Byzantine (malicious) participants in federated learning."""
    
    def __init__(self, threshold_factor: float = 2.0, history_window: int = 10):
        self.threshold_factor = threshold_factor
        self.history_window = history_window
        self.update_history: Dict[str, List[ModelUpdate]] = defaultdict(list)
        self.reputation_scores: Dict[str, float] = defaultdict(lambda: 1.0)
        
    def analyze_updates(self, updates: List[ModelUpdate]) -> Tuple[List[str], Dict[str, float]]:
        """Analyze model updates to detect Byzantine participants."""
        if len(updates) < 3:
            return [], {}  # Need minimum participants for detection
            
        byzantine_participants = []
        anomaly_scores = {}
        
        # Analyze gradient norms
        gradient_norms = [update.gradient_norm for update in updates]
        norm_mean = np.mean(gradient_norms)
        norm_std = np.std(gradient_norms)
        
        for update in updates:
            # Store update in history
            self.update_history[update.participant_id].append(update)
            if len(self.update_history[update.participant_id]) > self.history_window:
                self.update_history[update.participant_id].pop(0)
            
            # Detect anomalous gradient norms
            z_score = abs(update.gradient_norm - norm_mean) / (norm_std + 1e-8)
            anomaly_scores[update.participant_id] = z_score
            
            # Check for Byzantine behavior
            is_byzantine = False
            
            # 1. Gradient norm anomaly
            if z_score > self.threshold_factor:
                is_byzantine = True
                
            # 2. Inconsistent training samples
            if update.training_samples <= 0 or update.training_samples > 1e6:
                is_byzantine = True
                
            # 3. Historical consistency check
            participant_history = self.update_history[update.participant_id]
            if len(participant_history) >= 3:
                recent_norms = [u.gradient_norm for u in participant_history[-3:]]
                if np.std(recent_norms) > norm_mean:  # High variance in norms
                    is_byzantine = True
                    
            # 4. Privacy cost anomalies
            if update.privacy_cost < 0 or update.privacy_cost > 10:
                is_byzantine = True
            
            if is_byzantine:
                byzantine_participants.append(update.participant_id)
                # Reduce reputation
                self.reputation_scores[update.participant_id] *= 0.8
            else:
                # Improve reputation slightly
                self.reputation_scores[update.participant_id] = min(
                    1.0, self.reputation_scores[update.participant_id] * 1.01
                )
                
        return byzantine_participants, anomaly_scores
    
    def get_participant_reputation(self, participant_id: str) -> float:
        """Get reputation score for a participant."""
        return self.reputation_scores[participant_id]
    
    def update_reputation(self, participant_id: str, delta: float):
        """Manually update participant reputation."""
        current = self.reputation_scores[participant_id]
        self.reputation_scores[participant_id] = max(0, min(1.0, current + delta))


class FederatedPrivacyTracker:
    """Tracks differential privacy budgets across federated participants."""
    
    def __init__(self):
        self.participant_budgets: Dict[str, float] = defaultdict(lambda: 10.0)  # Initial budget
        self.global_budget: float = 100.0
        self.budget_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)  # (timestamp, cost)
        
    def consume_budget(self, participant_id: str, privacy_cost: float) -> bool:
        """Consume privacy budget for a participant."""
        current_budget = self.participant_budgets[participant_id]
        
        if privacy_cost > current_budget or privacy_cost > self.global_budget:
            return False  # Budget exceeded
            
        # Consume budget
        self.participant_budgets[participant_id] -= privacy_cost
        self.global_budget -= privacy_cost
        
        # Record history
        self.budget_history[participant_id].append((time.time(), privacy_cost))
        
        return True
    
    def get_remaining_budget(self, participant_id: str) -> float:
        """Get remaining privacy budget for participant."""
        return self.participant_budgets[participant_id]
    
    def get_global_budget_remaining(self) -> float:
        """Get remaining global privacy budget."""
        return self.global_budget
    
    def reset_participant_budget(self, participant_id: str, new_budget: float = 10.0):
        """Reset privacy budget for a participant."""
        old_budget = self.participant_budgets[participant_id]
        self.participant_budgets[participant_id] = new_budget
        
        # Adjust global budget
        self.global_budget += (new_budget - old_budget)
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy budget usage report."""
        total_consumed = sum(
            sum(cost for _, cost in history) 
            for history in self.budget_history.values()
        )
        
        participant_usage = {
            pid: {
                "remaining": self.participant_budgets[pid],
                "consumed": sum(cost for _, cost in self.budget_history[pid]),
                "last_usage": self.budget_history[pid][-1][0] if self.budget_history[pid] else 0
            }
            for pid in self.participant_budgets.keys()
        }
        
        return {
            "global_budget_remaining": self.global_budget,
            "total_consumed": total_consumed,
            "participant_usage": participant_usage,
            "active_participants": len(self.participant_budgets)
        }


class FederatedFairnessAnalyzer:
    """Analyzes fairness in federated learning contributions and benefits."""
    
    def __init__(self):
        self.contribution_history: Dict[str, List[float]] = defaultdict(list)
        self.benefit_history: Dict[str, List[float]] = defaultdict(list)
        
    def record_contribution(self, participant_id: str, contribution_score: float):
        """Record a participant's contribution to the round."""
        self.contribution_history[participant_id].append(contribution_score)
        
    def record_benefit(self, participant_id: str, benefit_score: float):
        """Record benefit received by participant."""
        self.benefit_history[participant_id].append(benefit_score)
    
    def compute_fairness_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive fairness metrics."""
        all_participants = set(self.contribution_history.keys()) | set(self.benefit_history.keys())
        
        if not all_participants:
            return {"error": "No participants found"}
        
        # Contribution fairness (Gini coefficient)
        contributions = []
        for pid in all_participants:
            contrib_sum = sum(self.contribution_history.get(pid, [0]))
            contributions.append(contrib_sum)
            
        contribution_gini = self._compute_gini_coefficient(contributions)
        
        # Benefit fairness
        benefits = []
        for pid in all_participants:
            benefit_sum = sum(self.benefit_history.get(pid, [0]))
            benefits.append(benefit_sum)
            
        benefit_gini = self._compute_gini_coefficient(benefits)
        
        # Contribution-benefit correlation
        contrib_benefit_pairs = []
        for pid in all_participants:
            contrib = sum(self.contribution_history.get(pid, [0]))
            benefit = sum(self.benefit_history.get(pid, [0]))
            contrib_benefit_pairs.append((contrib, benefit))
            
        fairness_correlation = self._compute_fairness_correlation(contrib_benefit_pairs)
        
        # Overall fairness score (lower is more fair)
        overall_fairness = 1.0 - (contribution_gini + benefit_gini) / 2.0
        
        return {
            "overall_fairness_score": overall_fairness,
            "contribution_gini": contribution_gini,
            "benefit_gini": benefit_gini, 
            "fairness_correlation": fairness_correlation,
            "participant_count": len(all_participants),
            "timestamp": time.time()
        }
    
    def _compute_gini_coefficient(self, values: List[float]) -> float:
        """Compute Gini coefficient for fairness measurement."""
        if not values or all(v == 0 for v in values):
            return 0.0
            
        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)
        
        gini = (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(values))) / (n * sum(values))
        return max(0, min(1, gini))
    
    def _compute_fairness_correlation(self, pairs: List[Tuple[float, float]]) -> float:
        """Compute correlation between contribution and benefit."""
        if len(pairs) < 2:
            return 0.0
            
        contributions = [p[0] for p in pairs]
        benefits = [p[1] for p in pairs]
        
        # Pearson correlation coefficient
        try:
            correlation = np.corrcoef(contributions, benefits)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0


class FederatedLearningAuditor:
    """Main auditor for federated RLHF learning systems."""
    
    def __init__(self, 
                 coordinator_id: str,
                 privacy_config: Optional[PrivacyConfig] = None,
                 security_config: Optional[SecurityConfig] = None):
        self.coordinator_id = coordinator_id
        self.privacy_config = privacy_config or PrivacyConfig()
        self.security_config = security_config or SecurityConfig()
        
        # Initialize components
        self.participants: Dict[str, ParticipantInfo] = {}
        self.audit_events: List[FederatedAuditEvent] = []
        self.model_updates: Dict[int, List[ModelUpdate]] = defaultdict(list)  # Round -> Updates
        self.aggregation_results: Dict[int, AggregationResult] = {}
        
        # Analysis components
        self.byzantine_detector = ByzantineDetector()
        self.privacy_tracker = FederatedPrivacyTracker()
        self.fairness_analyzer = FederatedFairnessAnalyzer()
        
        # Cryptographic engine
        self.crypto_engine = CryptographicEngine(self.security_config)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        self.logger = logging.getLogger(__name__)
        self.current_round = 0
        
    def register_participant(self, participant_info: ParticipantInfo) -> bool:
        """Register a new federated learning participant."""
        try:
            # Validate participant
            if not self._validate_participant(participant_info):
                return False
                
            # Store participant
            self.participants[participant_info.participant_id] = participant_info
            
            # Log registration event
            event = FederatedAuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=FederatedEventType.ROUND_START,  # Using as registration event
                round_number=self.current_round,
                participant_id=participant_info.participant_id,
                timestamp=time.time(),
                data={
                    "action": "participant_registration",
                    "institution": participant_info.institution_name,
                    "role": participant_info.role.value,
                    "capabilities": participant_info.capabilities
                },
                signature=""  # Would be computed with crypto_engine
            )
            self.audit_events.append(event)
            
            self.logger.info(f"Registered participant: {participant_info.participant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register participant: {e}")
            return False
    
    def _validate_participant(self, participant_info: ParticipantInfo) -> bool:
        """Validate participant information."""
        # Check for required fields
        required_fields = [participant_info.participant_id, participant_info.institution_name,
                          participant_info.public_key, participant_info.ip_address]
        if not all(required_fields):
            return False
            
        # Validate IP address format (basic check)
        try:
            socket.inet_aton(participant_info.ip_address)
        except socket.error:
            return False
            
        # Check for duplicate participants
        if participant_info.participant_id in self.participants:
            return False
            
        return True
    
    async def process_model_update(self, update: ModelUpdate) -> Dict[str, Any]:
        """Process and audit a model update from a participant."""
        try:
            # Validate update
            if not self._validate_model_update(update):
                return {"success": False, "error": "Invalid model update"}
            
            # Check privacy budget
            if not self.privacy_tracker.consume_budget(update.participant_id, update.privacy_cost):
                return {"success": False, "error": "Privacy budget exceeded"}
            
            # Store update
            self.model_updates[update.round_number].append(update)
            
            # Update participant's last seen time
            if update.participant_id in self.participants:
                self.participants[update.participant_id].last_seen = time.time()
            
            # Log audit event
            event = FederatedAuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=FederatedEventType.MODEL_UPDATE,
                round_number=update.round_number,
                participant_id=update.participant_id,
                timestamp=time.time(),
                data={
                    "update_id": update.update_id,
                    "gradient_norm": update.gradient_norm,
                    "training_samples": update.training_samples,
                    "privacy_cost": update.privacy_cost,
                    "update_hash": update.compute_hash()
                },
                signature=update.signature
            )
            self.audit_events.append(event)
            
            # Analyze contribution for fairness
            contribution_score = self._compute_contribution_score(update)
            self.fairness_analyzer.record_contribution(update.participant_id, contribution_score)
            
            self.logger.info(f"Processed model update from {update.participant_id}")
            
            return {
                "success": True,
                "update_id": update.update_id,
                "privacy_budget_remaining": self.privacy_tracker.get_remaining_budget(update.participant_id)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process model update: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_model_update(self, update: ModelUpdate) -> bool:
        """Validate model update integrity and format."""
        # Check participant exists
        if update.participant_id not in self.participants:
            return False
            
        # Validate basic fields
        if update.gradient_norm < 0 or update.training_samples <= 0:
            return False
            
        # Validate privacy cost
        if update.privacy_cost < 0:
            return False
            
        # Validate timestamp (not too old or in future)
        current_time = time.time()
        if abs(update.timestamp - current_time) > 3600:  # 1 hour tolerance
            return False
            
        return True
    
    def _compute_contribution_score(self, update: ModelUpdate) -> float:
        """Compute contribution score for a model update."""
        # Simple contribution scoring based on gradient norm and training samples
        # More sophisticated methods would analyze model quality improvements
        
        base_score = min(1.0, update.gradient_norm / 10.0)  # Normalize gradient norm
        sample_weight = min(1.0, update.training_samples / 1000.0)  # Normalize sample count
        
        contribution_score = 0.7 * base_score + 0.3 * sample_weight
        return contribution_score
    
    async def aggregate_round(self, round_number: int, method: AggregationMethod) -> AggregationResult:
        """Aggregate model updates for a federated learning round."""
        try:
            updates = self.model_updates[round_number]
            
            if not updates:
                raise AuditTrailError(f"No updates found for round {round_number}")
            
            # Byzantine detection
            byzantine_participants, anomaly_scores = self.byzantine_detector.analyze_updates(updates)
            
            # Filter out Byzantine updates
            valid_updates = [u for u in updates if u.participant_id not in byzantine_participants]
            
            if not valid_updates:
                raise AuditTrailError("No valid updates after Byzantine filtering")
            
            # Perform aggregation (simplified)
            aggregated_model_hash = self._aggregate_models(valid_updates, method)
            
            # Compute fairness metrics
            fairness_metrics = self.fairness_analyzer.compute_fairness_metrics()
            fairness_score = fairness_metrics.get("overall_fairness_score", 0.5)
            
            # Compute convergence metric (simplified)
            convergence_metric = self._compute_convergence_metric(valid_updates)
            
            # Calculate total privacy cost
            total_privacy_cost = sum(update.privacy_cost for update in valid_updates)
            
            # Create aggregation result
            result = AggregationResult(
                aggregation_id=str(uuid.uuid4()),
                round_number=round_number,
                method=method,
                participating_updates=[u.update_id for u in valid_updates],
                aggregated_model_hash=aggregated_model_hash,
                convergence_metric=convergence_metric,
                fairness_score=fairness_score,
                byzantine_participants=byzantine_participants,
                privacy_cost_total=total_privacy_cost,
                timestamp=time.time(),
                metadata={
                    "total_updates": len(updates),
                    "valid_updates": len(valid_updates),
                    "byzantine_count": len(byzantine_participants),
                    "anomaly_scores": anomaly_scores
                }
            )
            
            # Store result
            self.aggregation_results[round_number] = result
            
            # Log aggregation event
            event = FederatedAuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=FederatedEventType.AGGREGATION,
                round_number=round_number,
                participant_id=None,  # Coordinator action
                timestamp=time.time(),
                data={
                    "aggregation_id": result.aggregation_id,
                    "method": method.value,
                    "participating_updates": len(valid_updates),
                    "byzantine_detected": len(byzantine_participants),
                    "fairness_score": fairness_score,
                    "convergence_metric": convergence_metric
                },
                signature=""  # Would be signed by coordinator
            )
            self.audit_events.append(event)
            
            # Record benefits for participants (simplified)
            benefit_score = convergence_metric  # Use convergence as benefit proxy
            for update in valid_updates:
                self.fairness_analyzer.record_benefit(update.participant_id, benefit_score)
            
            self.logger.info(f"Completed aggregation for round {round_number}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to aggregate round {round_number}: {e}")
            raise AuditTrailError(f"Aggregation failed: {e}")
    
    def _aggregate_models(self, updates: List[ModelUpdate], method: AggregationMethod) -> str:
        """Aggregate model updates using specified method."""
        # Simplified aggregation - in practice would aggregate actual model parameters
        
        if method == AggregationMethod.FEDAVG:
            # Weighted average by training samples
            total_samples = sum(u.training_samples for u in updates)
            weights = [u.training_samples / total_samples for u in updates]
            
        elif method == AggregationMethod.FEDPROX:
            # Equal weights with proximity regularization
            weights = [1.0 / len(updates)] * len(updates)
            
        else:
            # Default to equal weights
            weights = [1.0 / len(updates)] * len(updates)
        
        # Create aggregated model hash (simplified)
        aggregation_data = {
            "method": method.value,
            "weights": weights,
            "update_hashes": [u.compute_hash() for u in updates],
            "timestamp": time.time()
        }
        
        content = json.dumps(aggregation_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _compute_convergence_metric(self, updates: List[ModelUpdate]) -> float:
        """Compute convergence metric for the round."""
        if not updates:
            return 0.0
            
        # Use gradient norm variance as convergence proxy
        # Lower variance indicates better convergence
        gradient_norms = [u.gradient_norm for u in updates]
        norm_variance = np.var(gradient_norms)
        
        # Convert to convergence score (higher is better)
        convergence_score = max(0, 1.0 - norm_variance / 10.0)
        return min(1.0, convergence_score)
    
    def get_round_summary(self, round_number: int) -> Dict[str, Any]:
        """Get comprehensive summary for a federated learning round."""
        updates = self.model_updates.get(round_number, [])
        aggregation = self.aggregation_results.get(round_number)
        
        if not updates:
            return {"error": f"No data found for round {round_number}"}
        
        # Participant statistics
        participant_stats = {}
        for update in updates:
            pid = update.participant_id
            if pid not in participant_stats:
                participant_stats[pid] = {
                    "updates": 0,
                    "total_samples": 0,
                    "total_privacy_cost": 0.0,
                    "avg_gradient_norm": 0.0,
                    "reputation": self.byzantine_detector.get_participant_reputation(pid)
                }
            
            stats = participant_stats[pid]
            stats["updates"] += 1
            stats["total_samples"] += update.training_samples
            stats["total_privacy_cost"] += update.privacy_cost
            stats["avg_gradient_norm"] = (stats["avg_gradient_norm"] * (stats["updates"] - 1) + 
                                        update.gradient_norm) / stats["updates"]
        
        summary = {
            "round_number": round_number,
            "total_participants": len(participant_stats),
            "total_updates": len(updates),
            "participant_statistics": participant_stats,
            "privacy_budget_usage": {
                "total_cost": sum(u.privacy_cost for u in updates),
                "global_remaining": self.privacy_tracker.get_global_budget_remaining()
            },
            "fairness_metrics": self.fairness_analyzer.compute_fairness_metrics()
        }
        
        if aggregation:
            summary["aggregation_result"] = {
                "method": aggregation.method.value,
                "convergence_metric": aggregation.convergence_metric,
                "fairness_score": aggregation.fairness_score,
                "byzantine_participants": aggregation.byzantine_participants,
                "aggregated_model_hash": aggregation.aggregated_model_hash
            }
        
        return summary
    
    def get_federated_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive federated learning audit report."""
        total_rounds = max(self.model_updates.keys()) if self.model_updates else 0
        total_updates = sum(len(updates) for updates in self.model_updates.values())
        
        # Byzantine analysis
        all_byzantine = set()
        for result in self.aggregation_results.values():
            all_byzantine.update(result.byzantine_participants)
        
        # Privacy analysis
        privacy_report = self.privacy_tracker.get_privacy_report()
        
        # Fairness analysis
        fairness_metrics = self.fairness_analyzer.compute_fairness_metrics()
        
        # Participant analysis
        active_participants = [
            pid for pid, info in self.participants.items()
            if time.time() - info.last_seen < 3600  # Active in last hour
        ]
        
        return {
            "audit_metadata": {
                "coordinator_id": self.coordinator_id,
                "report_timestamp": time.time(),
                "total_audit_events": len(self.audit_events)
            },
            "federated_learning_metrics": {
                "total_rounds_completed": total_rounds,
                "total_model_updates": total_updates,
                "registered_participants": len(self.participants),
                "active_participants": len(active_participants),
                "byzantine_participants_detected": len(all_byzantine)
            },
            "privacy_analysis": privacy_report,
            "fairness_analysis": fairness_metrics,
            "security_analysis": {
                "byzantine_participants": list(all_byzantine),
                "average_reputation_score": np.mean([
                    self.byzantine_detector.get_participant_reputation(pid) 
                    for pid in self.participants.keys()
                ]) if self.participants else 0
            },
            "convergence_analysis": {
                "average_convergence": np.mean([
                    result.convergence_metric for result in self.aggregation_results.values()
                ]) if self.aggregation_results else 0,
                "convergence_trend": [
                    result.convergence_metric 
                    for result in sorted(self.aggregation_results.values(), 
                                       key=lambda x: x.round_number)
                ]
            }
        }
    
    def export_federated_research_data(self, output_path: Path) -> Dict[str, Any]:
        """Export federated learning research data for analysis."""
        research_data = {
            "metadata": {
                "export_timestamp": time.time(),
                "coordinator_id": self.coordinator_id,
                "total_rounds": max(self.model_updates.keys()) if self.model_updates else 0,
                "total_participants": len(self.participants),
                "total_audit_events": len(self.audit_events)
            },
            "participants": {
                pid: {
                    "institution": info.institution_name,
                    "role": info.role.value,
                    "reputation_score": self.byzantine_detector.get_participant_reputation(pid),
                    "privacy_budget_remaining": self.privacy_tracker.get_remaining_budget(pid),
                    "capabilities": info.capabilities
                }
                for pid, info in self.participants.items()
            },
            "round_summaries": {
                str(round_num): self.get_round_summary(round_num)
                for round_num in self.model_updates.keys()
            },
            "audit_events": [
                {
                    "event_type": event.event_type.value,
                    "round_number": event.round_number,
                    "participant_id": event.participant_id,
                    "timestamp": event.timestamp,
                    "data": event.data
                }
                for event in self.audit_events
            ],
            "privacy_analysis": self.privacy_tracker.get_privacy_report(),
            "fairness_analysis": self.fairness_analyzer.compute_fairness_metrics(),
            "security_analysis": {
                "byzantine_detection_results": {
                    str(round_num): result.byzantine_participants
                    for round_num, result in self.aggregation_results.items()
                }
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(research_data, f, indent=2)
        
        return research_data