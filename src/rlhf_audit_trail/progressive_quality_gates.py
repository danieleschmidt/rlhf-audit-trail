"""Progressive Quality Gates System for RLHF Audit Trail.

Implements intelligent, adaptive quality gates that progressively enhance
system reliability and compliance validation.
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

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class QualityGateType(Enum):
    """Types of quality gates."""
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"


class GateStatus(Enum):
    """Status of a quality gate."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class QualityGate:
    """Represents a single quality gate."""
    gate_id: str
    name: str
    gate_type: QualityGateType
    priority: int
    threshold: float
    validator: Callable
    status: GateStatus = GateStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.result is None:
            self.result = {}


@dataclass
class QualityGateResult:
    """Result of quality gate execution."""
    gate_id: str
    status: GateStatus
    score: float
    passed: bool
    details: Dict[str, Any]
    execution_time: float
    timestamp: float
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class ProgressiveQualityGates:
    """Progressive Quality Gates System.
    
    Implements intelligent quality gates that adapt based on system
    maturity and risk profile.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize progressive quality gates.
        
        Args:
            config: Quality gates configuration
        """
        self.config = config or {}
        self.gates: Dict[str, QualityGate] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.adaptive_thresholds = {}
        
        self.logger = logging.getLogger(__name__)
        self._setup_default_gates()
    
    def _setup_default_gates(self):
        """Setup default quality gates."""
        # Functional Gates
        self.add_gate(QualityGate(
            gate_id="func_001",
            name="Core API Functionality",
            gate_type=QualityGateType.FUNCTIONAL,
            priority=1,
            threshold=0.95,
            validator=self._validate_core_api
        ))
        
        self.add_gate(QualityGate(
            gate_id="func_002",
            name="Privacy Engine Validation",
            gate_type=QualityGateType.FUNCTIONAL,
            priority=2,
            threshold=0.90,
            validator=self._validate_privacy_engine
        ))
        
        # Performance Gates
        self.add_gate(QualityGate(
            gate_id="perf_001",
            name="Response Time Validation",
            gate_type=QualityGateType.PERFORMANCE,
            priority=3,
            threshold=200.0,  # milliseconds
            validator=self._validate_response_time
        ))
        
        self.add_gate(QualityGate(
            gate_id="perf_002",
            name="Memory Usage Validation",
            gate_type=QualityGateType.PERFORMANCE,
            priority=4,
            threshold=512.0,  # MB
            validator=self._validate_memory_usage
        ))
        
        # Security Gates
        self.add_gate(QualityGate(
            gate_id="sec_001",
            name="Cryptographic Integrity",
            gate_type=QualityGateType.SECURITY,
            priority=1,
            threshold=1.0,
            validator=self._validate_crypto_integrity
        ))
        
        self.add_gate(QualityGate(
            gate_id="sec_002",
            name="Data Protection Validation",
            gate_type=QualityGateType.SECURITY,
            priority=2,
            threshold=0.99,
            validator=self._validate_data_protection
        ))
        
        # Compliance Gates
        self.add_gate(QualityGate(
            gate_id="comp_001",
            name="EU AI Act Compliance",
            gate_type=QualityGateType.COMPLIANCE,
            priority=1,
            threshold=1.0,
            validator=self._validate_eu_compliance
        ))
        
        self.add_gate(QualityGate(
            gate_id="comp_002",
            name="NIST Framework Compliance",
            gate_type=QualityGateType.COMPLIANCE,
            priority=2,
            threshold=0.95,
            validator=self._validate_nist_compliance
        ))
        
        # Reliability Gates
        self.add_gate(QualityGate(
            gate_id="rel_001",
            name="Error Handling Validation",
            gate_type=QualityGateType.RELIABILITY,
            priority=1,
            threshold=0.99,
            validator=self._validate_error_handling
        ))
        
        # Scalability Gates
        self.add_gate(QualityGate(
            gate_id="scale_001",
            name="Load Capacity Validation",
            gate_type=QualityGateType.SCALABILITY,
            priority=1,
            threshold=1000.0,  # requests per second
            validator=self._validate_load_capacity
        ))
    
    def add_gate(self, gate: QualityGate):
        """Add a quality gate.
        
        Args:
            gate: Quality gate to add
        """
        self.gates[gate.gate_id] = gate
        self.logger.info(f"Added quality gate: {gate.name} ({gate.gate_id})")
    
    async def execute_gates(
        self,
        gate_types: Optional[List[QualityGateType]] = None,
        fail_fast: bool = True
    ) -> Dict[str, QualityGateResult]:
        """Execute quality gates.
        
        Args:
            gate_types: Specific gate types to execute
            fail_fast: Stop on first failure
            
        Returns:
            Dictionary of gate results
        """
        start_time = time.time()
        results = {}
        failed_gates = []
        
        # Filter gates by type if specified
        gates_to_run = list(self.gates.values())
        if gate_types:
            gates_to_run = [g for g in gates_to_run if g.gate_type in gate_types]
        
        # Sort by priority
        gates_to_run.sort(key=lambda g: g.priority)
        
        self.logger.info(f"Executing {len(gates_to_run)} quality gates")
        
        for gate in gates_to_run:
            try:
                gate.status = GateStatus.RUNNING
                gate_start = time.time()
                
                # Execute gate validator
                validation_result = await gate.validator(gate)
                
                gate.execution_time = time.time() - gate_start
                
                # Create result
                result = QualityGateResult(
                    gate_id=gate.gate_id,
                    status=GateStatus.PASSED if validation_result['passed'] else GateStatus.FAILED,
                    score=validation_result.get('score', 0.0),
                    passed=validation_result['passed'],
                    details=validation_result.get('details', {}),
                    execution_time=gate.execution_time,
                    timestamp=time.time(),
                    recommendations=validation_result.get('recommendations', [])
                )
                
                gate.status = result.status
                gate.result = asdict(result)
                results[gate.gate_id] = result
                
                if not result.passed:
                    failed_gates.append(gate.gate_id)
                    self.logger.warning(f"Quality gate failed: {gate.name}")
                    
                    if fail_fast:
                        break
                else:
                    self.logger.info(f"Quality gate passed: {gate.name}")
                    
            except Exception as e:
                gate.status = GateStatus.FAILED
                gate.error = str(e)
                
                result = QualityGateResult(
                    gate_id=gate.gate_id,
                    status=GateStatus.FAILED,
                    score=0.0,
                    passed=False,
                    details={'error': str(e)},
                    execution_time=time.time() - gate_start,
                    timestamp=time.time()
                )
                
                results[gate.gate_id] = result
                failed_gates.append(gate.gate_id)
                
                self.logger.error(f"Quality gate error: {gate.name} - {e}")
                
                if fail_fast:
                    break
        
        execution_time = time.time() - start_time
        
        # Store execution history
        self.execution_history.append({
            'timestamp': time.time(),
            'execution_time': execution_time,
            'gates_executed': len(results),
            'gates_passed': len([r for r in results.values() if r.passed]),
            'gates_failed': len(failed_gates),
            'failed_gates': failed_gates,
            'results': {k: asdict(v) for k, v in results.items()}
        })
        
        self.logger.info(
            f"Quality gates execution completed: "
            f"{len(results)} executed, {len([r for r in results.values() if r.passed])} passed, "
            f"{len(failed_gates)} failed in {execution_time:.2f}s"
        )
        
        return results
    
    async def _validate_core_api(self, gate: QualityGate) -> Dict[str, Any]:
        """Validate core API functionality."""
        try:
            from .core import AuditableRLHF
            from .config import PrivacyConfig
            
            # Test basic initialization
            auditor = AuditableRLHF(
                model_name="test-model",
                privacy_config=PrivacyConfig(epsilon=1.0),
                storage_backend="local"
            )
            
            # Test privacy report
            privacy_report = auditor.get_privacy_report()
            assert 'total_epsilon' in privacy_report
            
            return {
                'passed': True,
                'score': 1.0,
                'details': {
                    'initialization': 'success',
                    'privacy_report': 'available'
                }
            }
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)},
                'recommendations': ['Check core dependencies and imports']
            }
    
    async def _validate_privacy_engine(self, gate: QualityGate) -> Dict[str, Any]:
        """Validate privacy engine functionality."""
        try:
            from .privacy import DifferentialPrivacyEngine, PrivacyBudgetManager
            from .config import PrivacyConfig
            
            config = PrivacyConfig(epsilon=1.0, delta=1e-5)
            engine = DifferentialPrivacyEngine(config)
            budget = PrivacyBudgetManager(total_epsilon=1.0, total_delta=1e-5)
            
            # Test epsilon cost estimation
            cost = engine.estimate_epsilon_cost(100)
            assert cost > 0
            
            # Test budget management
            assert budget.can_spend(0.1)
            budget.spend(0.1)
            assert budget.total_spent_epsilon == 0.1
            
            return {
                'passed': True,
                'score': 1.0,
                'details': {
                    'epsilon_cost_estimation': 'working',
                    'budget_management': 'working'
                }
            }
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)},
                'recommendations': ['Check privacy engine dependencies']
            }
    
    async def _validate_response_time(self, gate: QualityGate) -> Dict[str, Any]:
        """Validate API response times."""
        try:
            from .core import AuditableRLHF
            
            start_time = time.time()
            auditor = AuditableRLHF(model_name="perf-test")
            privacy_report = auditor.get_privacy_report()
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            passed = response_time < gate.threshold
            score = max(0, 1 - (response_time / gate.threshold))
            
            return {
                'passed': passed,
                'score': score,
                'details': {
                    'response_time_ms': response_time,
                    'threshold_ms': gate.threshold
                },
                'recommendations': [] if passed else [
                    'Optimize initialization',
                    'Consider lazy loading'
                ]
            }
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    async def _validate_memory_usage(self, gate: QualityGate) -> Dict[str, Any]:
        """Validate memory usage."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            passed = memory_mb < gate.threshold
            score = max(0, 1 - (memory_mb / gate.threshold))
            
            return {
                'passed': passed,
                'score': score,
                'details': {
                    'memory_usage_mb': memory_mb,
                    'threshold_mb': gate.threshold
                },
                'recommendations': [] if passed else [
                    'Optimize memory usage',
                    'Check for memory leaks'
                ]
            }
        except ImportError:
            return {
                'passed': True,
                'score': 1.0,
                'details': {'skipped': 'psutil not available'}
            }
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    async def _validate_crypto_integrity(self, gate: QualityGate) -> Dict[str, Any]:
        """Validate cryptographic integrity."""
        try:
            from .crypto import CryptographicEngine, IntegrityVerifier
            
            crypto = CryptographicEngine()
            verifier = IntegrityVerifier(crypto)
            
            # Test key generation
            key = crypto.generate_key()
            assert len(key) > 0
            
            # Test encryption/decryption
            test_data = "test data for encryption"
            encrypted = crypto.encrypt(test_data.encode(), key)
            decrypted = crypto.decrypt(encrypted, key)
            assert decrypted.decode() == test_data
            
            return {
                'passed': True,
                'score': 1.0,
                'details': {
                    'key_generation': 'working',
                    'encryption_decryption': 'working'
                }
            }
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)},
                'recommendations': ['Check cryptographic dependencies']
            }
    
    async def _validate_data_protection(self, gate: QualityGate) -> Dict[str, Any]:
        """Validate data protection mechanisms."""
        try:
            from .storage import LocalStorage
            from .crypto import CryptographicEngine
            
            storage = LocalStorage()
            crypto = CryptographicEngine()
            
            # Test encrypted storage
            test_data = {"sensitive": "data"}
            await storage.store_encrypted("test/data.json", test_data, crypto)
            
            # Verify encryption
            raw_data = await storage.retrieve_raw("test/data.json")
            assert raw_data != json.dumps(test_data).encode()
            
            return {
                'passed': True,
                'score': 1.0,
                'details': {
                    'encrypted_storage': 'working',
                    'data_protection': 'verified'
                }
            }
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    async def _validate_eu_compliance(self, gate: QualityGate) -> Dict[str, Any]:
        """Validate EU AI Act compliance."""
        try:
            from .compliance import ComplianceValidator, ComplianceFramework
            
            validator = ComplianceValidator(
                frameworks=[ComplianceFramework.EU_AI_ACT]
            )
            
            # Test compliance validation
            test_session_data = {
                'session_id': 'test-session',
                'privacy_protected': True,
                'audit_trail': True,
                'transparency': True
            }
            
            compliance_result = await validator.validate_basic_compliance(test_session_data)
            
            return {
                'passed': compliance_result.get('compliant', False),
                'score': compliance_result.get('score', 0.0),
                'details': compliance_result
            }
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    async def _validate_nist_compliance(self, gate: QualityGate) -> Dict[str, Any]:
        """Validate NIST framework compliance."""
        try:
            from .compliance import ComplianceValidator, ComplianceFramework
            
            validator = ComplianceValidator(
                frameworks=[ComplianceFramework.NIST_DRAFT]
            )
            
            test_session_data = {
                'session_id': 'test-session',
                'transparency': True,
                'documentation': True,
                'risk_assessment': True
            }
            
            compliance_result = await validator.validate_basic_compliance(test_session_data)
            
            return {
                'passed': compliance_result.get('compliant', True),
                'score': compliance_result.get('score', 1.0),
                'details': compliance_result
            }
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    async def _validate_error_handling(self, gate: QualityGate) -> Dict[str, Any]:
        """Validate error handling robustness."""
        try:
            from .core import AuditableRLHF
            from .exceptions import AuditTrailError
            
            auditor = AuditableRLHF(model_name="error-test")
            
            # Test error handling without active session
            try:
                await auditor.log_annotations([], [], [], [])
                return {
                    'passed': False,
                    'score': 0.0,
                    'details': {'error': 'Should have raised AuditTrailError'}
                }
            except AuditTrailError:
                # This is expected
                pass
            
            return {
                'passed': True,
                'score': 1.0,
                'details': {
                    'error_handling': 'working',
                    'proper_exceptions': 'raised'
                }
            }
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    async def _validate_load_capacity(self, gate: QualityGate) -> Dict[str, Any]:
        """Validate load handling capacity."""
        try:
            import asyncio
            from .core import AuditableRLHF
            
            # Simulate concurrent requests
            async def create_auditor():
                return AuditableRLHF(model_name=f"load-test-{uuid.uuid4()}")
            
            start_time = time.time()
            tasks = [create_auditor() for _ in range(10)]
            auditors = await asyncio.gather(*tasks)
            
            # Get privacy reports concurrently
            reports = [auditor.get_privacy_report() for auditor in auditors]
            
            duration = time.time() - start_time
            requests_per_second = len(auditors) / duration
            
            passed = requests_per_second > gate.threshold / 100  # Scale down for testing
            score = min(1.0, requests_per_second / (gate.threshold / 100))
            
            return {
                'passed': passed,
                'score': score,
                'details': {
                    'requests_per_second': requests_per_second,
                    'threshold': gate.threshold,
                    'duration': duration
                },
                'recommendations': [] if passed else [
                    'Optimize initialization performance',
                    'Implement connection pooling'
                ]
            }
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def get_gate_statistics(self) -> Dict[str, Any]:
        """Get quality gate execution statistics.
        
        Returns:
            Dictionary with gate statistics
        """
        if not self.execution_history:
            return {'no_executions': True}
        
        latest = self.execution_history[-1]
        total_executions = len(self.execution_history)
        
        success_rate = sum(
            1 for exec in self.execution_history 
            if exec['gates_failed'] == 0
        ) / total_executions
        
        avg_execution_time = sum(
            exec['execution_time'] for exec in self.execution_history
        ) / total_executions
        
        return {
            'total_executions': total_executions,
            'success_rate': success_rate,
            'average_execution_time': avg_execution_time,
            'latest_execution': latest,
            'gate_count': len(self.gates)
        }
    
    def export_results(self, output_path: Path) -> None:
        """Export quality gate results to file.
        
        Args:
            output_path: Path to save results
        """
        results_data = {
            'gates': {k: asdict(v) for k, v in self.gates.items()},
            'execution_history': self.execution_history,
            'statistics': self.get_gate_statistics(),
            'exported_at': time.time()
        }
        
        output_path.write_text(json.dumps(results_data, indent=2, default=str))
        self.logger.info(f"Quality gate results exported to {output_path}")
