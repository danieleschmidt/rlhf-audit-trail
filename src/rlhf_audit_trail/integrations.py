"""Integration layer for popular RLHF and ML libraries."""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

try:
    import torch
    import torch.nn as nn
    from torch.optim import Optimizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from trl import PPOTrainer, PPOConfig
    from trl.core import respond_to_batch
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

try:
    import trlx
    TRLX_AVAILABLE = True
except ImportError:
    TRLX_AVAILABLE = False

from .exceptions import AuditTrailError

logger = logging.getLogger(__name__)


class ModelExtractor:
    """Utilities for extracting information from ML models."""
    
    @staticmethod
    def extract_model_metadata(model: Any) -> Dict[str, Any]:
        """Extract metadata from various model types."""
        metadata = {
            "model_type": type(model).__name__,
            "timestamp": time.time()
        }
        
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            metadata.update({
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "device": str(next(model.parameters()).device) if list(model.parameters()) else "cpu"
            })
            
        if TRANSFORMERS_AVAILABLE and hasattr(model, 'config'):
            config = model.config
            metadata.update({
                "model_name": getattr(config, 'name_or_path', 'unknown'),
                "architecture": getattr(config, 'architectures', ['unknown'])[0] if getattr(config, 'architectures', None) else 'unknown',
                "vocab_size": getattr(config, 'vocab_size', None),
                "hidden_size": getattr(config, 'hidden_size', None),
                "num_layers": getattr(config, 'num_hidden_layers', None)
            })
            
        return metadata
    
    @staticmethod
    def compute_parameter_hash(model: Any) -> str:
        """Compute hash of model parameters."""
        if not TORCH_AVAILABLE:
            return "torch_not_available"
            
        if isinstance(model, nn.Module):
            import hashlib
            hasher = hashlib.sha256()
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_bytes = param.data.cpu().numpy().tobytes()
                    hasher.update(name.encode('utf-8'))
                    hasher.update(param_bytes)
                    
            return hasher.hexdigest()
        else:
            return "unsupported_model_type"
    
    @staticmethod
    def compute_gradient_statistics(model: Any) -> Dict[str, float]:
        """Compute statistics about model gradients."""
        if not TORCH_AVAILABLE or not isinstance(model, nn.Module):
            return {}
            
        grad_stats = {
            "gradient_norm": 0.0,
            "max_gradient": 0.0,
            "min_gradient": 0.0,
            "gradient_count": 0
        }
        
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                grad_data = param.grad.data.cpu().numpy().flatten()
                gradients.extend(grad_data)
                
        if gradients:
            import numpy as np
            gradients = np.array(gradients)
            grad_stats.update({
                "gradient_norm": float(np.linalg.norm(gradients)),
                "max_gradient": float(np.max(gradients)),
                "min_gradient": float(np.min(gradients)),
                "gradient_count": len(gradients),
                "gradient_mean": float(np.mean(gradients)),
                "gradient_std": float(np.std(gradients))
            })
            
        return grad_stats


class BaseIntegration(ABC):
    """Base class for ML library integrations."""
    
    def __init__(self, audit_logger):
        self.audit_logger = audit_logger
        self.model_extractor = ModelExtractor()
        
    @abstractmethod
    def track_training_step(self, **kwargs) -> Dict[str, Any]:
        """Track a single training step."""
        pass
        
    @abstractmethod
    def track_policy_update(self, **kwargs) -> Dict[str, Any]:
        """Track a policy update step."""
        pass


class TorchIntegration(BaseIntegration):
    """Integration for PyTorch models and training."""
    
    def __init__(self, audit_logger):
        if not TORCH_AVAILABLE:
            raise AuditTrailError("PyTorch not available for integration")
        super().__init__(audit_logger)
        
    def track_training_step(self, model: nn.Module, optimizer: Optimizer, 
                           batch: Dict[str, Any], loss: float,
                           step: int, **kwargs) -> Dict[str, Any]:
        """Track a PyTorch training step."""
        try:
            step_data = {
                "step": step,
                "loss": float(loss),
                "learning_rate": optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0,
                "batch_size": len(batch.get('input_ids', batch.get('inputs', []))),
                "model_metadata": self.model_extractor.extract_model_metadata(model),
                "gradient_stats": self.model_extractor.compute_gradient_statistics(model)
            }
            
            # Log additional kwargs
            for key, value in kwargs.items():
                if isinstance(value, (int, float, str, bool)):
                    step_data[key] = value
                    
            return step_data
            
        except Exception as e:
            logger.error(f"Failed to track PyTorch training step: {e}")
            return {"error": str(e)}
            
    def track_policy_update(self, model: nn.Module, optimizer: Optimizer,
                           old_model_state: Optional[Dict[str, Any]] = None,
                           **kwargs) -> Dict[str, Any]:
        """Track a policy update in PyTorch."""
        try:
            current_state = {
                "parameter_hash": self.model_extractor.compute_parameter_hash(model),
                "gradient_stats": self.model_extractor.compute_gradient_statistics(model),
                "optimizer_state": {
                    "learning_rate": optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0,
                    "optimizer_type": type(optimizer).__name__
                }
            }
            
            policy_delta = {"current_state": current_state}
            
            if old_model_state:
                policy_delta["previous_state"] = old_model_state
                # Compute parameter delta if possible
                if "parameter_hash" in old_model_state:
                    policy_delta["parameters_changed"] = (
                        current_state["parameter_hash"] != old_model_state["parameter_hash"]
                    )
                    
            return policy_delta
            
        except Exception as e:
            logger.error(f"Failed to track PyTorch policy update: {e}")
            return {"error": str(e)}


class TRLIntegration(BaseIntegration):
    """Integration for TRL (Transformers Reinforcement Learning) library."""
    
    def __init__(self, audit_logger):
        if not TRL_AVAILABLE:
            raise AuditTrailError("TRL not available for integration")
        super().__init__(audit_logger)
        
    def track_training_step(self, trainer: Any, queries: List[str], 
                           responses: List[str], rewards: List[float],
                           step: int, **kwargs) -> Dict[str, Any]:
        """Track a TRL training step."""
        try:
            step_data = {
                "step": step,
                "batch_size": len(queries),
                "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
                "reward_stats": {
                    "min": min(rewards) if rewards else 0.0,
                    "max": max(rewards) if rewards else 0.0,
                    "std": self._compute_std(rewards) if rewards else 0.0
                },
                "query_lengths": [len(q.split()) for q in queries],
                "response_lengths": [len(r.split()) for r in responses]
            }
            
            # Extract model information if available
            if hasattr(trainer, 'model'):
                step_data["model_metadata"] = self.model_extractor.extract_model_metadata(trainer.model)
                
            return step_data
            
        except Exception as e:
            logger.error(f"Failed to track TRL training step: {e}")
            return {"error": str(e)}
            
    def track_policy_update(self, trainer: Any, stats: Dict[str, float],
                           **kwargs) -> Dict[str, Any]:
        """Track a TRL policy update."""
        try:
            policy_update = {
                "ppo_stats": stats,
                "timestamp": time.time()
            }
            
            if hasattr(trainer, 'model'):
                policy_update["model_metadata"] = self.model_extractor.extract_model_metadata(trainer.model)
                policy_update["parameter_hash"] = self.model_extractor.compute_parameter_hash(trainer.model)
                
            return policy_update
            
        except Exception as e:
            logger.error(f"Failed to track TRL policy update: {e}")
            return {"error": str(e)}
    
    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


class AuditablePPOTrainer:
    """Wrapper around TRL PPOTrainer with built-in audit trail."""
    
    def __init__(self, model, ref_model, tokenizer, auditor, **ppo_config):
        if not TRL_AVAILABLE:
            raise AuditTrailError("TRL not available for AuditablePPOTrainer")
            
        self.auditor = auditor
        self.integration = TRLIntegration(auditor)
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            **ppo_config
        )
        
        self.step_count = 0
        self._training_active = False
        
    def step(self, queries: List[str], responses: List[str], rewards: List[float]):
        """Perform a PPO step with audit trail."""
        self._training_active = True
        self.step_count += 1
        
        try:
            # Track the training step
            step_data = self.integration.track_training_step(
                trainer=self.ppo_trainer,
                queries=queries,
                responses=responses,
                rewards=rewards,
                step=self.step_count
            )
            
            # Log annotations
            self.auditor.log_annotations(
                prompts=queries,
                responses=responses,
                rewards=rewards,
                annotator_ids=[f"step_{self.step_count}_{i}" for i in range(len(queries))],
                additional_data=step_data
            )
            
            # Perform the actual PPO step
            stats = self.ppo_trainer.step(queries, responses, rewards)
            
            # Track policy update
            policy_data = self.integration.track_policy_update(
                trainer=self.ppo_trainer,
                stats=stats
            )
            
            self.auditor.track_policy_update(
                model=self.ppo_trainer.model,
                optimizer=self.ppo_trainer.optimizer,
                additional_data=policy_data
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to perform auditable PPO step: {e}")
            raise
        finally:
            self._training_active = False
            
    def train(self, **kwargs):
        """Train with full audit trail support."""
        logger.info("Starting auditable PPO training")
        
        with self.auditor.track_training("ppo_training"):
            # This would typically involve a training loop
            # For now, we just set up the context
            pass
            
    def __getattr__(self, name):
        """Delegate to underlying PPO trainer for compatibility."""
        return getattr(self.ppo_trainer, name)


class HuggingFaceIntegration(BaseIntegration):
    """Integration for Hugging Face transformers."""
    
    def __init__(self, audit_logger):
        if not TRANSFORMERS_AVAILABLE:
            raise AuditTrailError("Transformers not available for integration")
        super().__init__(audit_logger)
        
    def track_model_loading(self, model: PreTrainedModel, 
                           model_name: str) -> Dict[str, Any]:
        """Track model loading from Hugging Face."""
        try:
            metadata = self.model_extractor.extract_model_metadata(model)
            metadata.update({
                "source": "huggingface",
                "model_name": model_name,
                "loaded_at": time.time()
            })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to track HF model loading: {e}")
            return {"error": str(e)}
            
    def track_training_step(self, model: PreTrainedModel, inputs: Dict[str, Any],
                           loss: float, step: int, **kwargs) -> Dict[str, Any]:
        """Track a Hugging Face training step."""
        try:
            step_data = {
                "step": step,
                "loss": float(loss),
                "input_shape": inputs.get('input_ids', torch.tensor([])).shape if TORCH_AVAILABLE else [],
                "model_metadata": self.model_extractor.extract_model_metadata(model)
            }
            
            return step_data
            
        except Exception as e:
            logger.error(f"Failed to track HF training step: {e}")
            return {"error": str(e)}
            
    def track_policy_update(self, **kwargs) -> Dict[str, Any]:
        """Track policy updates for HF models."""
        # Implementation similar to TorchIntegration
        return {"timestamp": time.time()}


class IntegrationManager:
    """Manages integrations with different ML libraries."""
    
    def __init__(self, audit_logger):
        self.audit_logger = audit_logger
        self.integrations = {}
        
        # Initialize available integrations
        if TORCH_AVAILABLE:
            self.integrations['torch'] = TorchIntegration(audit_logger)
            
        if TRL_AVAILABLE:
            self.integrations['trl'] = TRLIntegration(audit_logger)
            
        if TRANSFORMERS_AVAILABLE:
            self.integrations['huggingface'] = HuggingFaceIntegration(audit_logger)
            
    def get_integration(self, library: str) -> Optional[BaseIntegration]:
        """Get integration for a specific library."""
        return self.integrations.get(library)
        
    def create_auditable_trainer(self, trainer_type: str, **kwargs):
        """Factory method for creating auditable trainers."""
        if trainer_type == 'ppo' and TRL_AVAILABLE:
            return AuditablePPOTrainer(auditor=self.audit_logger, **kwargs)
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")
            
    def auto_detect_framework(self, model: Any) -> Optional[str]:
        """Automatically detect the ML framework being used."""
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            return 'torch'
        elif TRANSFORMERS_AVAILABLE and hasattr(model, 'config'):
            return 'huggingface'
        else:
            return None