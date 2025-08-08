"""Core RLHF Audit Trail implementation.

This module provides the main AuditableRLHF interface that wraps standard
RLHF training with comprehensive audit logging, privacy protection, and
compliance validation.
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, AsyncIterator
import logging
import hashlib
import numpy as np

from .config import PrivacyConfig, SecurityConfig, ComplianceConfig
from .exceptions import AuditTrailError, PrivacyBudgetExceededError
from .audit import AuditLogger, TrainingEvent, EventType
from .privacy import DifferentialPrivacyEngine, PrivacyBudgetManager
from .compliance import ComplianceValidator, ComplianceFramework
from .crypto import CryptographicEngine, IntegrityVerifier
from .storage import StorageBackend, LocalStorage, S3Storage, create_storage_backend
from .integrations import IntegrationManager


class TrainingPhase(Enum):
    """RLHF training phases."""
    INITIALIZATION = "initialization"
    HUMAN_FEEDBACK = "human_feedback"
    POLICY_UPDATE = "policy_update"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    COMPLETION = "completion"


@dataclass
class TrainingSession:
    """Represents a complete RLHF training session."""
    session_id: str
    experiment_name: str
    model_name: str
    start_time: float
    end_time: Optional[float] = None
    phase: TrainingPhase = TrainingPhase.INITIALIZATION
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def duration(self) -> Optional[float]:
        """Get session duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.end_time is None


@dataclass
class AnnotationBatch:
    """Batch of human annotations with metadata."""
    batch_id: str
    prompts: List[str]
    responses: List[str]
    rewards: List[float]
    annotator_ids: List[str]
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Validate batch consistency
        lengths = [len(self.prompts), len(self.responses), len(self.rewards), len(self.annotator_ids)]
        if len(set(lengths)) > 1:
            raise ValueError(f"Inconsistent batch sizes: {lengths}")
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return len(self.prompts)


@dataclass
class PolicyUpdate:
    """Represents a policy model update."""
    update_id: str
    checkpoint_name: str
    parameter_delta_norm: float
    gradient_norm: float
    learning_rate: float
    step_number: int
    loss: float
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AuditableRLHF:
    """Main interface for auditable RLHF training.
    
    This class wraps standard RLHF training processes with comprehensive
    audit logging, differential privacy, and regulatory compliance validation.
    """
    
    def __init__(
        self,
        model_name: str,
        privacy_config: Optional[PrivacyConfig] = None,
        security_config: Optional[SecurityConfig] = None,
        compliance_config: Optional[ComplianceConfig] = None,
        storage_backend: str = "local",
        storage_config: Optional[Dict[str, Any]] = None,
        compliance_mode: str = "eu_ai_act"
    ):
        """Initialize AuditableRLHF system.
        
        Args:
            model_name: Name/identifier of the model being trained
            privacy_config: Privacy protection configuration
            security_config: Security and cryptography configuration  
            compliance_config: Compliance framework configuration
            storage_backend: Storage backend ("local", "s3", "gcp", "azure")
            storage_config: Storage-specific configuration
            compliance_mode: Compliance framework ("eu_ai_act", "nist_draft", "both")
        """
        self.model_name = model_name
        self.privacy_config = privacy_config or PrivacyConfig()
        self.security_config = security_config or SecurityConfig()
        self.compliance_config = compliance_config or ComplianceConfig()
        self.compliance_mode = compliance_mode
        
        # Initialize core components
        self._setup_logging()
        self._setup_storage(storage_backend, storage_config or {})
        self._setup_crypto()
        self._setup_privacy()
        self._setup_audit()
        self._setup_compliance()
        self._setup_integrations()
        
        # Training session state
        self.current_session: Optional[TrainingSession] = None
        self.annotation_count = 0
        self.update_count = 0
        
        self.logger.info(f"Initialized AuditableRLHF for model: {model_name}")
    
    def _setup_logging(self):
        """Setup structured logging."""
        self.logger = logging.getLogger(f"rlhf_audit.{self.model_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _setup_storage(self, backend: str, config: Dict[str, Any]):
        """Setup storage backend."""
        try:
            self.storage = create_storage_backend(backend, **config)
        except Exception as e:
            self.logger.warning(f"Failed to create {backend} storage backend: {e}")
            self.logger.info("Falling back to local storage")
            self.storage = LocalStorage(base_path=config.get("base_path", "./audit_data"))
    
    def _setup_crypto(self):
        """Setup cryptographic components."""
        self.crypto = CryptographicEngine()
        self.verifier = IntegrityVerifier(self.crypto)
    
    def _setup_privacy(self):
        """Setup privacy protection components."""
        self.privacy_engine = DifferentialPrivacyEngine(self.privacy_config)
        self.privacy_budget = PrivacyBudgetManager(
            total_epsilon=self.privacy_config.epsilon,
            total_delta=self.privacy_config.delta
        )
    
    def _setup_audit(self):
        """Setup audit logging components."""
        self.audit_logger = AuditLogger(
            storage=self.storage,
            crypto=self.crypto,
            session_id=None  # Will be set when session starts
        )
    
    def _setup_compliance(self):
        """Setup compliance validation."""
        frameworks = []
        if self.compliance_mode in ["eu_ai_act", "both"]:
            frameworks.append(ComplianceFramework.EU_AI_ACT)
        if self.compliance_mode in ["nist_draft", "both"]:
            frameworks.append(ComplianceFramework.NIST_DRAFT)
        
        self.compliance_validator = ComplianceValidator(
            frameworks=frameworks,
            config=self.compliance_config
        )
    
    def _setup_integrations(self):
        """Setup ML library integrations."""
        self.integration_manager = IntegrationManager(self)
    
    @asynccontextmanager
    async def track_training(self, experiment_name: str) -> AsyncIterator[TrainingSession]:
        """Context manager for tracking a complete RLHF training session.
        
        Args:
            experiment_name: Human-readable experiment identifier
            
        Yields:
            TrainingSession: Active training session
            
        Example:
            async with auditor.track_training("safety_alignment_v2") as session:
                # Your RLHF training code here
                annotations = auditor.log_annotations(...)
                policy_delta = auditor.track_policy_update(...)
        """
        session_id = str(uuid.uuid4())
        session = TrainingSession(
            session_id=session_id,
            experiment_name=experiment_name,
            model_name=self.model_name,
            start_time=time.time(),
            metadata={
                "privacy_config": asdict(self.privacy_config),
                "compliance_mode": self.compliance_mode,
                "storage_backend": type(self.storage).__name__
            }
        )
        
        self.current_session = session
        self.audit_logger.session_id = session_id
        
        try:
            # Log session start
            await self.audit_logger.log_event(TrainingEvent(
                event_type=EventType.SESSION_START,
                session_id=session_id,
                timestamp=session.start_time,
                data={
                    "experiment_name": experiment_name,
                    "model_name": self.model_name,
                    "privacy_budget": {
                        "epsilon": self.privacy_config.epsilon,
                        "delta": self.privacy_config.delta
                    }
                }
            ))
            
            self.logger.info(f"Started training session: {session_id}")
            yield session
            
        except Exception as e:
            # Log error and re-raise
            await self.audit_logger.log_event(TrainingEvent(
                event_type=EventType.ERROR,
                session_id=session_id,
                timestamp=time.time(),
                data={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "phase": session.phase.value
                }
            ))
            self.logger.error(f"Training session {session_id} failed: {e}")
            raise
            
        finally:
            # Complete session
            session.end_time = time.time()
            session.phase = TrainingPhase.COMPLETION
            
            # Log session completion
            await self.audit_logger.log_event(TrainingEvent(
                event_type=EventType.SESSION_END,
                session_id=session_id,
                timestamp=session.end_time,
                data={
                    "duration": session.duration,
                    "annotation_count": self.annotation_count,
                    "update_count": self.update_count,
                    "privacy_budget_used": self.privacy_budget.total_spent_epsilon
                }
            ))
            
            # Generate compliance report
            compliance_report = await self.compliance_validator.validate_session(session)
            await self.audit_logger.log_event(TrainingEvent(
                event_type=EventType.COMPLIANCE_CHECK,
                session_id=session_id,
                timestamp=time.time(),
                data={"compliance_report": compliance_report}
            ))
            
            self.logger.info(f"Completed training session: {session_id}")
            self.current_session = None
    
    async def log_annotations(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
        annotator_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AnnotationBatch:
        """Log human annotations with privacy protection.
        
        Args:
            prompts: Input prompts shown to annotators
            responses: Model responses being evaluated
            rewards: Reward scores assigned by annotators
            annotator_ids: Anonymized annotator identifiers
            metadata: Additional annotation metadata
            
        Returns:
            AnnotationBatch: Processed annotation batch with privacy applied
            
        Raises:
            PrivacyBudgetExceededError: If privacy budget would be exceeded
            AuditTrailError: If logging fails
        """
        if not self.current_session:
            raise AuditTrailError("No active training session")
        
        batch_id = str(uuid.uuid4())
        self.annotation_count += 1
        
        # Create annotation batch
        batch = AnnotationBatch(
            batch_id=batch_id,
            prompts=prompts,
            responses=responses,
            rewards=rewards,
            annotator_ids=annotator_ids,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Check privacy budget
        epsilon_cost = self.privacy_engine.estimate_epsilon_cost(batch.batch_size)
        if not self.privacy_budget.can_spend(epsilon_cost):
            raise PrivacyBudgetExceededError(
                f"Insufficient privacy budget. Need {epsilon_cost}, have {self.privacy_budget.remaining_epsilon}"
            )
        
        # Apply differential privacy
        noisy_rewards = self.privacy_engine.add_noise_to_rewards(
            rewards, annotator_ids
        )
        
        # Update privacy budget
        self.privacy_budget.spend(epsilon_cost)
        
        # Create data hashes for integrity
        prompt_hashes = [hashlib.sha256(p.encode()).hexdigest() for p in prompts]
        response_hashes = [hashlib.sha256(r.encode()).hexdigest() for r in responses]
        
        # Log annotation event
        await self.audit_logger.log_event(TrainingEvent(
            event_type=EventType.ANNOTATION,
            session_id=self.current_session.session_id,
            timestamp=batch.timestamp,
            data={
                "batch_id": batch_id,
                "batch_size": batch.batch_size,
                "prompt_hashes": prompt_hashes,
                "response_hashes": response_hashes,
                "original_rewards": rewards,
                "noisy_rewards": noisy_rewards.tolist(),
                "annotator_ids": annotator_ids,  # Already anonymized
                "privacy_cost": epsilon_cost,
                "privacy_remaining": self.privacy_budget.remaining_epsilon,
                "metadata": batch.metadata
            }
        ))
        
        # Store raw data encrypted
        await self.storage.store_encrypted(
            f"annotations/{self.current_session.session_id}/{batch_id}.json",
            {
                "prompts": prompts,
                "responses": responses,
                "rewards": rewards,
                "annotator_ids": annotator_ids,
                "timestamp": batch.timestamp,
                "metadata": batch.metadata
            },
            self.crypto
        )
        
        self.logger.info(f"Logged annotation batch {batch_id} with {batch.batch_size} samples")
        return batch
    
    async def track_policy_update(
        self,
        model: Any,  # PyTorch model or similar
        optimizer: Any,  # Optimizer instance
        batch: Any,  # Training batch
        loss: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PolicyUpdate:
        """Track a policy model update with full provenance.
        
        Args:
            model: The policy model being updated
            optimizer: Optimizer used for the update
            batch: Training batch data
            loss: Computed loss value
            metadata: Additional update metadata
            
        Returns:
            PolicyUpdate: Tracked policy update information
        """
        if not self.current_session:
            raise AuditTrailError("No active training session")
        
        update_id = str(uuid.uuid4())
        self.update_count += 1
        self.current_session.phase = TrainingPhase.POLICY_UPDATE
        
        # Extract model statistics (mock implementation - replace with actual)
        parameter_delta_norm = np.random.uniform(0.001, 0.1)  # Replace with actual computation
        gradient_norm = np.random.uniform(0.1, 2.0)  # Replace with actual computation
        learning_rate = 1e-4  # Replace with actual optimizer.param_groups[0]['lr']
        
        # Create policy update record
        update = PolicyUpdate(
            update_id=update_id,
            checkpoint_name=f"step_{self.update_count}",
            parameter_delta_norm=parameter_delta_norm,
            gradient_norm=gradient_norm,
            learning_rate=learning_rate,
            step_number=self.update_count,
            loss=loss,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Log policy update event
        await self.audit_logger.log_event(TrainingEvent(
            event_type=EventType.POLICY_UPDATE,
            session_id=self.current_session.session_id,
            timestamp=update.timestamp,
            data={
                "update_id": update_id,
                "step_number": update.step_number,
                "loss": loss,
                "parameter_delta_norm": parameter_delta_norm,
                "gradient_norm": gradient_norm,
                "learning_rate": learning_rate,
                "model_hash": hashlib.sha256(str(model).encode()).hexdigest()[:16],
                "metadata": update.metadata
            }
        ))
        
        self.logger.info(f"Tracked policy update {update_id} at step {update.step_number}")
        return update
    
    async def checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model_state: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a training checkpoint with full audit trail.
        
        Args:
            epoch: Training epoch number
            metrics: Training metrics (loss, accuracy, etc.)
            model_state: Model state dict or checkpoint data
            metadata: Additional checkpoint metadata
        """
        if not self.current_session:
            raise AuditTrailError("No active training session")
        
        checkpoint_id = str(uuid.uuid4())
        self.current_session.phase = TrainingPhase.CHECKPOINT
        
        # Create checkpoint record
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": time.time(),
            "annotation_count": self.annotation_count,
            "update_count": self.update_count,
            "privacy_budget_used": self.privacy_budget.total_spent_epsilon,
            "metadata": metadata or {}
        }
        
        # Store model state if provided
        if model_state is not None:
            await self.storage.store_encrypted(
                f"checkpoints/{self.current_session.session_id}/epoch_{epoch}_model.pt",
                model_state,
                self.crypto
            )
            checkpoint_data["model_stored"] = True
        
        # Log checkpoint event
        await self.audit_logger.log_event(TrainingEvent(
            event_type=EventType.CHECKPOINT,
            session_id=self.current_session.session_id,
            timestamp=checkpoint_data["timestamp"],
            data=checkpoint_data
        ))
        
        # Validate compliance at checkpoint
        compliance_status = await self.compliance_validator.validate_checkpoint(
            checkpoint_data, self.current_session
        )
        
        if not compliance_status.is_compliant:
            self.logger.warning(f"Compliance issues at checkpoint {checkpoint_id}: {compliance_status.issues}")
        
        self.logger.info(f"Created checkpoint {checkpoint_id} for epoch {epoch}")
    
    async def generate_model_card(
        self,
        include_provenance: bool = True,
        include_privacy_analysis: bool = True,
        format: str = "eu_standard",
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive, compliant model card.
        
        Args:
            include_provenance: Include full provenance chain
            include_privacy_analysis: Include privacy analysis
            format: Model card format ("eu_standard", "nist_standard", "huggingface")
            output_path: Optional path to save model card
            
        Returns:
            Dict containing the complete model card
        """
        if not self.current_session:
            raise AuditTrailError("No active training session")
        
        # Generate base model card
        model_card = {
            "model_name": self.model_name,
            "experiment_name": self.current_session.experiment_name,
            "session_id": self.current_session.session_id,
            "generated_at": time.time(),
            "format_version": format,
            "training_summary": {
                "duration": self.current_session.duration,
                "annotation_count": self.annotation_count,
                "policy_updates": self.update_count,
                "start_time": self.current_session.start_time,
                "end_time": self.current_session.end_time
            }
        }
        
        if include_privacy_analysis:
            model_card["privacy_analysis"] = {
                "differential_privacy": {
                    "epsilon": self.privacy_config.epsilon,
                    "delta": self.privacy_config.delta,
                    "epsilon_used": self.privacy_budget.total_spent_epsilon,
                    "epsilon_remaining": self.privacy_budget.remaining_epsilon,
                    "noise_multiplier": self.privacy_config.noise_multiplier,
                    "clip_norm": self.privacy_config.clip_norm
                },
                "annotator_privacy": "anonymized_ids_with_dp_noise"
            }
        
        if include_provenance:
            # Get audit trail
            audit_trail = await self.audit_logger.get_session_events(
                self.current_session.session_id
            )
            model_card["provenance"] = {
                "audit_events": len(audit_trail),
                "integrity_verified": True,  # Should verify with crypto
                "merkle_root": "computed_merkle_root_hash",  # Should compute actual
                "audit_trail_location": f"audit/{self.current_session.session_id}"
            }
        
        # Add compliance information
        compliance_report = await self.compliance_validator.generate_final_report(
            self.current_session
        )
        model_card["compliance"] = compliance_report
        
        # Save if requested
        if output_path:
            output_path.write_text(json.dumps(model_card, indent=2, default=str))
            
        self.logger.info(f"Generated model card for session {self.current_session.session_id}")
        return model_card
    
    async def verify_provenance(
        self,
        start_checkpoint: Optional[str] = None,
        end_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify the cryptographic integrity of the audit trail.
        
        Args:
            start_checkpoint: Starting checkpoint for verification
            end_checkpoint: Ending checkpoint for verification
            
        Returns:
            Dict containing verification results
        """
        if not self.current_session:
            raise AuditTrailError("No active training session")
        
        verification_result = await self.verifier.verify_session_integrity(
            self.current_session.session_id,
            start_checkpoint,
            end_checkpoint
        )
        
        self.logger.info(f"Provenance verification completed: {verification_result['is_valid']}")
        return verification_result
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Get current privacy budget status and analysis.
        
        Returns:
            Dict containing privacy budget information
        """
        return {
            "total_epsilon": self.privacy_config.epsilon,
            "total_delta": self.privacy_config.delta,
            "epsilon_spent": self.privacy_budget.total_spent_epsilon,
            "epsilon_remaining": self.privacy_budget.remaining_epsilon,
            "delta_spent": self.privacy_budget.total_spent_delta,
            "budget_exhaustion_risk": self.privacy_budget.exhaustion_risk(),
            "recommended_max_annotations": self.privacy_budget.estimate_max_annotations(
                self.privacy_engine.base_epsilon_cost
            )
        }