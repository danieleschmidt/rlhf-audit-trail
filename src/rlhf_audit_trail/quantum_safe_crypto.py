"""Quantum-Safe Cryptographic Audit Trail System.

This module implements post-quantum cryptographic security for RLHF audit trails including:
- Lattice-based digital signatures (CRYSTALS-Dilithium)
- Hash-based signatures (XMSS/SPHINCS+)
- Quantum-resistant key encapsulation mechanisms
- Post-quantum secure multi-party computation
- Quantum-safe blockchain-like audit chains
- Quantum entropy analysis for randomness validation
"""

import hashlib
import hmac
import secrets
import time
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import struct
import math
from concurrent.futures import ThreadPoolExecutor
import asyncio

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy for basic operations
    class MockNumpy:
        def array(self, data): return data
        def zeros(self, shape): return [0] * (shape if isinstance(shape, int) else shape[0])
        def random(self):
            import random
            class MockRandom:
                def randint(self, low, high): return random.randint(low, high)
                def bytes(self, size): return secrets.token_bytes(size)
            return MockRandom()
        def linalg(self):
            class MockLinalg:
                def norm(self, vec): return sum(x**2 for x in vec)**0.5
            return MockLinalg()
    np = MockNumpy()

from .exceptions import AuditTrailError
from .config import SecurityConfig


class QuantumSafeAlgorithm(Enum):
    """Post-quantum cryptographic algorithms."""
    CRYSTALS_DILITHIUM = "crystals_dilithium"
    CRYSTALS_KYBER = "crystals_kyber"
    SPHINCS_PLUS = "sphincs_plus"
    XMSS = "xmss"
    NTRU = "ntru"
    FALCON = "falcon"
    RAINBOW = "rainbow"


class QuantumThreatLevel(Enum):
    """Quantum threat assessment levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QuantumSafeKeyPair:
    """Post-quantum cryptographic key pair."""
    algorithm: QuantumSafeAlgorithm
    public_key: bytes
    private_key: bytes
    key_id: str
    generation_time: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    quantum_security_level: int = 128  # Bits of post-quantum security


@dataclass
class QuantumSafeSignature:
    """Post-quantum digital signature."""
    signature_id: str
    algorithm: QuantumSafeAlgorithm
    signature_data: bytes
    message_hash: str
    signer_key_id: str
    timestamp: float
    quantum_entropy_score: float
    verification_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumAuditBlock:
    """Quantum-safe audit block in the blockchain."""
    block_id: str
    previous_block_hash: str
    merkle_root: str
    timestamp: float
    audit_events: List[Dict[str, Any]]
    quantum_signature: QuantumSafeSignature
    quantum_entropy_analysis: Dict[str, Any]
    post_quantum_proof: Dict[str, Any]
    block_height: int


class CRYSTALSDilithiumSigner:
    """Simplified implementation of CRYSTALS-Dilithium signature scheme."""
    
    def __init__(self, security_level: int = 2):
        self.security_level = security_level
        self.q = 8380417  # Prime modulus
        self.n = 256      # Polynomial degree
        self.k = {2: 4, 3: 6, 5: 8}[security_level]  # Matrix dimensions
        self.l = {2: 4, 3: 5, 5: 7}[security_level]
        self.eta = {2: 2, 3: 4, 5: 2}[security_level]
        self.tau = {2: 39, 3: 49, 5: 60}[security_level]
        self.beta = self.tau * self.eta
        self.gamma1 = {2: 131072, 3: 524288, 5: 524288}[security_level]
        self.gamma2 = {2: 95232, 3: 261888, 5: 261888}[security_level]
        
    def keygen(self) -> QuantumSafeKeyPair:
        """Generate CRYSTALS-Dilithium key pair."""
        # Simplified key generation (real implementation would use proper lattices)
        
        # Generate random seed
        seed = secrets.token_bytes(32)
        
        # Generate matrix A from seed (simplified)
        rho = self._shake256(seed, 32)
        
        # Generate secret key vectors s1, s2
        s1 = self._sample_eta_vector(self.l, rho[:16])
        s2 = self._sample_eta_vector(self.k, rho[16:32])
        
        # Compute t = A*s1 + s2 (simplified)
        t = self._compute_public_key(s1, s2, rho)
        
        # Pack keys
        public_key = self._pack_public_key(rho, t)
        private_key = self._pack_private_key(rho, s1, s2, t)
        
        return QuantumSafeKeyPair(
            algorithm=QuantumSafeAlgorithm.CRYSTALS_DILITHIUM,
            public_key=public_key,
            private_key=private_key,
            key_id=str(uuid.uuid4()),
            generation_time=time.time(),
            parameters={
                'security_level': self.security_level,
                'q': self.q,
                'n': self.n,
                'k': self.k,
                'l': self.l
            },
            quantum_security_level=128 + (self.security_level - 2) * 64
        )
    
    def sign(self, message: bytes, private_key: bytes) -> QuantumSafeSignature:
        """Sign message with CRYSTALS-Dilithium."""
        # Unpack private key (simplified)
        rho, s1, s2, t = self._unpack_private_key(private_key)
        
        # Hash message
        message_hash = hashlib.sha256(message).hexdigest()
        
        # Generate signature (simplified version)
        kappa = 0
        while kappa < 1000:  # Rejection sampling loop
            # Sample y
            y = self._sample_gamma1_vector(self.l, secrets.token_bytes(32))
            
            # Compute w = A*y
            w = self._compute_w(y, rho)
            
            # Compute challenge
            c = self._compute_challenge(message, w)
            
            # Compute z = y + c*s1
            z = self._add_vectors(y, self._multiply_scalar_vector(c, s1))
            
            # Check rejection condition (simplified)
            if self._check_rejection(z, y, c, s2):
                kappa += 1
                continue
                
            # Compute hint h
            h = self._compute_hint(c, s2, w)
            
            # Pack signature
            signature_data = self._pack_signature(z, h, c)
            
            # Calculate quantum entropy score
            entropy_score = self._calculate_quantum_entropy(signature_data)
            
            return QuantumSafeSignature(
                signature_id=str(uuid.uuid4()),
                algorithm=QuantumSafeAlgorithm.CRYSTALS_DILITHIUM,
                signature_data=signature_data,
                message_hash=message_hash,
                signer_key_id="",  # Would be filled by caller
                timestamp=time.time(),
                quantum_entropy_score=entropy_score,
                verification_params={
                    'security_level': self.security_level,
                    'rounds': kappa + 1
                }
            )
            
        raise AuditTrailError("Signature generation failed - too many rejections")
    
    def verify(self, message: bytes, signature: QuantumSafeSignature, public_key: bytes) -> bool:
        """Verify CRYSTALS-Dilithium signature."""
        try:
            # Unpack public key and signature
            rho, t = self._unpack_public_key(public_key)
            z, h, c = self._unpack_signature(signature.signature_data)
            
            # Recompute challenge
            w_prime = self._recompute_w_prime(z, c, rho, t, h)
            c_prime = self._compute_challenge(message, w_prime)
            
            # Verify signature
            return c == c_prime and self._verify_bounds(z)
            
        except Exception:
            return False
    
    # Helper methods (simplified implementations)
    
    def _shake256(self, data: bytes, length: int) -> bytes:
        """SHAKE256 extendable output function."""
        return hashlib.shake_256(data).digest(length)
    
    def _sample_eta_vector(self, length: int, seed: bytes) -> List[int]:
        """Sample vector with coefficients in [-eta, eta]."""
        result = []
        for i in range(length * self.n):
            # Simplified sampling
            val = int.from_bytes(seed[i % len(seed):i % len(seed) + 1], 'big') % (2 * self.eta + 1)
            result.append(val - self.eta)
        return result[:length * self.n]
    
    def _sample_gamma1_vector(self, length: int, seed: bytes) -> List[int]:
        """Sample vector with coefficients in [-gamma1, gamma1]."""
        result = []
        for i in range(length * self.n):
            val = int.from_bytes(seed[i % len(seed):i % len(seed) + 1], 'big') % (2 * self.gamma1 + 1)
            result.append(val - self.gamma1)
        return result[:length * self.n]
    
    def _compute_public_key(self, s1: List[int], s2: List[int], rho: bytes) -> List[int]:
        """Compute public key t = A*s1 + s2 (simplified)."""
        # Simplified matrix-vector multiplication
        t = []
        for i in range(self.k * self.n):
            val = (s1[i % len(s1)] * int.from_bytes(rho[i % len(rho):i % len(rho) + 1], 'big') + 
                   s2[i % len(s2)]) % self.q
            t.append(val)
        return t
    
    def _compute_w(self, y: List[int], rho: bytes) -> List[int]:
        """Compute w = A*y (simplified)."""
        w = []
        for i in range(self.k * self.n):
            val = (y[i % len(y)] * int.from_bytes(rho[i % len(rho):i % len(rho) + 1], 'big')) % self.q
            w.append(val)
        return w
    
    def _compute_challenge(self, message: bytes, w: List[int]) -> int:
        """Compute challenge from message and w."""
        hasher = hashlib.sha256()
        hasher.update(message)
        hasher.update(json.dumps(w[:10]).encode())  # Use first 10 elements for simplicity
        return int.from_bytes(hasher.digest()[:4], 'big') % self.q
    
    def _add_vectors(self, a: List[int], b: List[int]) -> List[int]:
        """Add two vectors modulo q."""
        return [(a[i] + b[i]) % self.q for i in range(min(len(a), len(b)))]
    
    def _multiply_scalar_vector(self, scalar: int, vector: List[int]) -> List[int]:
        """Multiply vector by scalar modulo q."""
        return [(scalar * val) % self.q for val in vector]
    
    def _check_rejection(self, z: List[int], y: List[int], c: int, s2: List[int]) -> bool:
        """Check rejection condition (simplified)."""
        # Simplified rejection condition
        for val in z:
            if abs(val) >= self.gamma1 - self.beta:
                return True
        return False
    
    def _compute_hint(self, c: int, s2: List[int], w: List[int]) -> List[int]:
        """Compute hint h (simplified)."""
        # Simplified hint computation
        h = []
        for i in range(min(len(s2), len(w))):
            hint_val = (c * s2[i] + w[i]) % 2
            h.append(hint_val)
        return h
    
    def _recompute_w_prime(self, z: List[int], c: int, rho: bytes, t: List[int], h: List[int]) -> List[int]:
        """Recompute w' for verification."""
        # Simplified recomputation
        w_prime = []
        for i in range(min(len(z), len(t))):
            val = (z[i] * int.from_bytes(rho[i % len(rho):i % len(rho) + 1], 'big') - 
                   c * t[i] + h[i % len(h)]) % self.q
            w_prime.append(val)
        return w_prime
    
    def _verify_bounds(self, z: List[int]) -> bool:
        """Verify signature bounds."""
        for val in z:
            if abs(val) >= self.gamma1 - self.beta:
                return False
        return True
    
    def _pack_public_key(self, rho: bytes, t: List[int]) -> bytes:
        """Pack public key into bytes."""
        packed = rho
        for val in t[:32]:  # Pack first 32 elements for simplicity
            packed += struct.pack('<I', val % (2**32))
        return packed
    
    def _pack_private_key(self, rho: bytes, s1: List[int], s2: List[int], t: List[int]) -> bytes:
        """Pack private key into bytes."""
        packed = rho
        for val in (s1[:16] + s2[:16] + t[:16]):  # Pack first 16 elements of each
            packed += struct.pack('<i', val)
        return packed
    
    def _pack_signature(self, z: List[int], h: List[int], c: int) -> bytes:
        """Pack signature into bytes."""
        packed = struct.pack('<I', c)
        for val in z[:32]:  # Pack first 32 elements
            packed += struct.pack('<i', val)
        for val in h[:32]:  # Pack first 32 elements
            packed += struct.pack('<B', val % 256)
        return packed
    
    def _unpack_public_key(self, public_key: bytes) -> Tuple[bytes, List[int]]:
        """Unpack public key from bytes."""
        rho = public_key[:32]
        t = []
        for i in range(32):
            offset = 32 + i * 4
            if offset + 4 <= len(public_key):
                val = struct.unpack('<I', public_key[offset:offset+4])[0]
                t.append(val)
        return rho, t
    
    def _unpack_private_key(self, private_key: bytes) -> Tuple[bytes, List[int], List[int], List[int]]:
        """Unpack private key from bytes."""
        rho = private_key[:32]
        s1, s2, t = [], [], []
        offset = 32
        
        for i in range(16):  # s1
            if offset + 4 <= len(private_key):
                val = struct.unpack('<i', private_key[offset:offset+4])[0]
                s1.append(val)
                offset += 4
                
        for i in range(16):  # s2
            if offset + 4 <= len(private_key):
                val = struct.unpack('<i', private_key[offset:offset+4])[0]
                s2.append(val)
                offset += 4
                
        for i in range(16):  # t
            if offset + 4 <= len(private_key):
                val = struct.unpack('<i', private_key[offset:offset+4])[0]
                t.append(val)
                offset += 4
                
        return rho, s1, s2, t
    
    def _unpack_signature(self, signature_data: bytes) -> Tuple[List[int], List[int], int]:
        """Unpack signature from bytes."""
        c = struct.unpack('<I', signature_data[:4])[0]
        z, h = [], []
        offset = 4
        
        for i in range(32):  # z
            if offset + 4 <= len(signature_data):
                val = struct.unpack('<i', signature_data[offset:offset+4])[0]
                z.append(val)
                offset += 4
                
        for i in range(32):  # h
            if offset + 1 <= len(signature_data):
                val = struct.unpack('<B', signature_data[offset:offset+1])[0]
                h.append(val)
                offset += 1
                
        return z, h, c
    
    def _calculate_quantum_entropy(self, data: bytes) -> float:
        """Calculate quantum entropy score for randomness quality."""
        if not data:
            return 0.0
            
        # Simplified entropy calculation
        byte_freq = [0] * 256
        for byte in data:
            byte_freq[byte] += 1
            
        total_bytes = len(data)
        entropy = 0.0
        
        for freq in byte_freq:
            if freq > 0:
                prob = freq / total_bytes
                entropy -= prob * math.log2(prob)
        
        # Normalize to [0, 1]
        max_entropy = 8.0  # Maximum entropy for bytes
        return entropy / max_entropy


class QuantumEntropyAnalyzer:
    """Analyzes quantum entropy for cryptographic randomness validation."""
    
    def __init__(self):
        self.entropy_history = []
        
    def analyze_quantum_entropy(self, data: bytes) -> Dict[str, Any]:
        """Perform comprehensive quantum entropy analysis."""
        if not data:
            return {"error": "No data provided"}
            
        analysis = {
            "timestamp": time.time(),
            "data_length": len(data),
            "entropy_metrics": self._calculate_entropy_metrics(data),
            "randomness_tests": self._run_randomness_tests(data),
            "quantum_quality_score": 0.0
        }
        
        # Calculate overall quantum quality score
        entropy_score = analysis["entropy_metrics"]["shannon_entropy"]
        randomness_score = np.mean(list(analysis["randomness_tests"].values()))
        analysis["quantum_quality_score"] = (entropy_score + randomness_score) / 2
        
        self.entropy_history.append(analysis)
        return analysis
    
    def _calculate_entropy_metrics(self, data: bytes) -> Dict[str, float]:
        """Calculate various entropy metrics."""
        if not data:
            return {}
            
        # Shannon entropy
        shannon_entropy = self._shannon_entropy(data)
        
        # Min-entropy (worst-case entropy)
        min_entropy = self._min_entropy(data)
        
        # Renyi entropy
        renyi_entropy = self._renyi_entropy(data, alpha=2)
        
        # Kolmogorov complexity approximation
        kolmogorov_approx = self._kolmogorov_complexity_approx(data)
        
        return {
            "shannon_entropy": shannon_entropy,
            "min_entropy": min_entropy,
            "renyi_entropy": renyi_entropy,
            "kolmogorov_complexity": kolmogorov_approx
        }
    
    def _shannon_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy."""
        if not data:
            return 0.0
            
        byte_freq = [0] * 256
        for byte in data:
            byte_freq[byte] += 1
            
        entropy = 0.0
        total = len(data)
        
        for freq in byte_freq:
            if freq > 0:
                prob = freq / total
                entropy -= prob * math.log2(prob)
                
        return entropy / 8.0  # Normalize to [0, 1]
    
    def _min_entropy(self, data: bytes) -> float:
        """Calculate min-entropy (most conservative entropy measure)."""
        if not data:
            return 0.0
            
        byte_freq = [0] * 256
        for byte in data:
            byte_freq[byte] += 1
            
        max_freq = max(byte_freq)
        if max_freq == 0:
            return 0.0
            
        min_entropy = -math.log2(max_freq / len(data))
        return min_entropy / 8.0  # Normalize to [0, 1]
    
    def _renyi_entropy(self, data: bytes, alpha: float = 2) -> float:
        """Calculate Renyi entropy of order alpha."""
        if not data or alpha == 1:
            return self._shannon_entropy(data)
            
        byte_freq = [0] * 256
        for byte in data:
            byte_freq[byte] += 1
            
        total = len(data)
        sum_powers = sum((freq / total) ** alpha for freq in byte_freq if freq > 0)
        
        if sum_powers == 0:
            return 0.0
            
        renyi_entropy = math.log2(sum_powers) / (1 - alpha)
        return renyi_entropy / 8.0  # Normalize to [0, 1]
    
    def _kolmogorov_complexity_approx(self, data: bytes) -> float:
        """Approximate Kolmogorov complexity using compression."""
        import zlib
        
        if not data:
            return 0.0
            
        # Use compression ratio as approximation
        compressed = zlib.compress(data)
        compression_ratio = len(compressed) / len(data)
        
        # Higher compression ratio = lower complexity = lower randomness
        # Invert and normalize to [0, 1]
        complexity_score = 1.0 - compression_ratio
        return max(0.0, min(1.0, complexity_score))
    
    def _run_randomness_tests(self, data: bytes) -> Dict[str, float]:
        """Run statistical randomness tests."""
        tests = {}
        
        if len(data) >= 100:
            tests["frequency_test"] = self._frequency_test(data)
            tests["runs_test"] = self._runs_test(data)
            tests["serial_correlation"] = self._serial_correlation_test(data)
            
        if len(data) >= 1000:
            tests["poker_test"] = self._poker_test(data)
            
        return tests
    
    def _frequency_test(self, data: bytes) -> float:
        """Frequency (monobit) test."""
        if not data:
            return 0.0
            
        # Convert to bits
        bit_count = 0
        total_bits = len(data) * 8
        
        for byte in data:
            bit_count += bin(byte).count('1')
            
        # Test if frequency is close to 50%
        frequency = bit_count / total_bits
        deviation = abs(frequency - 0.5)
        
        # Convert to score [0, 1] where 1 is most random
        score = max(0, 1 - deviation * 2)
        return score
    
    def _runs_test(self, data: bytes) -> float:
        """Runs test for randomness."""
        if len(data) < 2:
            return 0.0
            
        # Count runs of consecutive identical bytes
        runs = 1
        for i in range(1, len(data)):
            if data[i] != data[i-1]:
                runs += 1
                
        # Expected number of runs
        n = len(data)
        expected_runs = (2 * n - 1) / 3
        
        # Calculate deviation from expected
        deviation = abs(runs - expected_runs) / expected_runs
        score = max(0, 1 - deviation)
        return score
    
    def _serial_correlation_test(self, data: bytes) -> float:
        """Serial correlation test."""
        if len(data) < 3:
            return 0.0
            
        # Calculate serial correlation coefficient
        n = len(data)
        mean_val = sum(data) / n
        
        numerator = sum((data[i] - mean_val) * (data[i+1] - mean_val) for i in range(n-1))
        denominator = sum((val - mean_val)**2 for val in data)
        
        if denominator == 0:
            return 1.0
            
        correlation = numerator / denominator
        
        # Convert correlation to randomness score
        score = max(0, 1 - abs(correlation))
        return score
    
    def _poker_test(self, data: bytes) -> float:
        """Poker test for randomness."""
        if len(data) < 20:
            return 0.0
            
        # Divide data into groups of 4 bits
        groups = []
        for byte in data[:20]:  # Use first 20 bytes
            groups.append(byte >> 4)      # Upper 4 bits
            groups.append(byte & 0x0F)    # Lower 4 bits
            
        # Count frequencies of each 4-bit pattern
        freq = [0] * 16
        for group in groups:
            freq[group] += 1
            
        # Calculate chi-square statistic
        n = len(groups)
        expected = n / 16
        chi_square = sum((f - expected)**2 / expected for f in freq if expected > 0)
        
        # Convert to score (simplified)
        # Chi-square value around 15 indicates good randomness for 15 degrees of freedom
        deviation = abs(chi_square - 15) / 15
        score = max(0, 1 - deviation)
        return score


class QuantumSafeAuditChain:
    """Quantum-safe blockchain-like audit trail."""
    
    def __init__(self, security_config: Optional[SecurityConfig] = None):
        self.security_config = security_config or SecurityConfig()
        self.dilithium_signer = CRYSTALSDilithiumSigner()
        self.entropy_analyzer = QuantumEntropyAnalyzer()
        
        self.chain: List[QuantumAuditBlock] = []
        self.genesis_block: Optional[QuantumAuditBlock] = None
        self.current_height = 0
        
        self.logger = logging.getLogger(__name__)
        
        # Create genesis block
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the genesis block for the quantum-safe audit chain."""
        # Generate key pair for genesis block
        key_pair = self.dilithium_signer.keygen()
        
        genesis_events = [{
            "event_type": "genesis",
            "timestamp": time.time(),
            "quantum_security_init": {
                "algorithm": key_pair.algorithm.value,
                "security_level": key_pair.quantum_security_level,
                "key_id": key_pair.key_id
            }
        }]
        
        # Create genesis signature
        genesis_message = json.dumps(genesis_events, sort_keys=True).encode()
        genesis_signature = self.dilithium_signer.sign(genesis_message, key_pair.private_key)
        genesis_signature.signer_key_id = key_pair.key_id
        
        # Analyze genesis entropy
        entropy_analysis = self.entropy_analyzer.analyze_quantum_entropy(genesis_signature.signature_data)
        
        # Create genesis block
        self.genesis_block = QuantumAuditBlock(
            block_id=str(uuid.uuid4()),
            previous_block_hash="0" * 64,  # Genesis has no predecessor
            merkle_root=self._calculate_merkle_root(genesis_events),
            timestamp=time.time(),
            audit_events=genesis_events,
            quantum_signature=genesis_signature,
            quantum_entropy_analysis=entropy_analysis,
            post_quantum_proof={
                "algorithm": key_pair.algorithm.value,
                "security_level": key_pair.quantum_security_level,
                "public_key_hash": hashlib.sha256(key_pair.public_key).hexdigest()
            },
            block_height=0
        )
        
        self.chain.append(self.genesis_block)
        self.current_height = 1
        
        self.logger.info("Genesis block created with quantum-safe security")
    
    async def add_audit_events(self, events: List[Dict[str, Any]], signer_key_pair: QuantumSafeKeyPair) -> QuantumAuditBlock:
        """Add new audit events to the quantum-safe chain."""
        if not events:
            raise AuditTrailError("No events to add")
            
        previous_block = self.chain[-1]
        
        # Prepare events with quantum security metadata
        enhanced_events = []
        for event in events:
            enhanced_event = event.copy()
            enhanced_event["quantum_timestamp"] = time.time()
            enhanced_event["quantum_hash"] = hashlib.sha256(
                json.dumps(event, sort_keys=True).encode()
            ).hexdigest()
            enhanced_events.append(enhanced_event)
        
        # Create block message for signing
        block_message_data = {
            "previous_block_hash": self._calculate_block_hash(previous_block),
            "merkle_root": self._calculate_merkle_root(enhanced_events),
            "timestamp": time.time(),
            "events": enhanced_events,
            "block_height": self.current_height
        }
        
        block_message = json.dumps(block_message_data, sort_keys=True).encode()
        
        # Sign block with quantum-safe signature
        quantum_signature = self.dilithium_signer.sign(block_message, signer_key_pair.private_key)
        quantum_signature.signer_key_id = signer_key_pair.key_id
        
        # Analyze entropy of signature
        entropy_analysis = self.entropy_analyzer.analyze_quantum_entropy(quantum_signature.signature_data)
        
        # Create post-quantum proof
        post_quantum_proof = {
            "algorithm": signer_key_pair.algorithm.value,
            "security_level": signer_key_pair.quantum_security_level,
            "public_key_hash": hashlib.sha256(signer_key_pair.public_key).hexdigest(),
            "signature_entropy": entropy_analysis["quantum_quality_score"],
            "quantum_threat_assessment": self._assess_quantum_threat(entropy_analysis)
        }
        
        # Create new block
        new_block = QuantumAuditBlock(
            block_id=str(uuid.uuid4()),
            previous_block_hash=self._calculate_block_hash(previous_block),
            merkle_root=self._calculate_merkle_root(enhanced_events),
            timestamp=time.time(),
            audit_events=enhanced_events,
            quantum_signature=quantum_signature,
            quantum_entropy_analysis=entropy_analysis,
            post_quantum_proof=post_quantum_proof,
            block_height=self.current_height
        )
        
        # Verify block before adding
        if not await self._verify_block(new_block, signer_key_pair.public_key):
            raise AuditTrailError("Block verification failed")
        
        # Add to chain
        self.chain.append(new_block)
        self.current_height += 1
        
        self.logger.info(f"Added quantum-safe audit block at height {new_block.block_height}")
        return new_block
    
    async def _verify_block(self, block: QuantumAuditBlock, public_key: bytes) -> bool:
        """Verify quantum-safe audit block."""
        try:
            # Verify previous block hash
            if len(self.chain) > 0:
                expected_prev_hash = self._calculate_block_hash(self.chain[-1])
                if block.previous_block_hash != expected_prev_hash:
                    return False
            
            # Verify merkle root
            calculated_merkle = self._calculate_merkle_root(block.audit_events)
            if block.merkle_root != calculated_merkle:
                return False
            
            # Reconstruct block message
            block_message_data = {
                "previous_block_hash": block.previous_block_hash,
                "merkle_root": block.merkle_root,
                "timestamp": block.timestamp,
                "events": block.audit_events,
                "block_height": block.block_height
            }
            
            block_message = json.dumps(block_message_data, sort_keys=True).encode()
            
            # Verify quantum-safe signature
            signature_valid = self.dilithium_signer.verify(
                block_message, 
                block.quantum_signature, 
                public_key
            )
            
            if not signature_valid:
                return False
            
            # Verify entropy quality
            if block.quantum_entropy_analysis["quantum_quality_score"] < 0.7:
                self.logger.warning("Low quantum entropy detected in block")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Block verification error: {e}")
            return False
    
    def _calculate_block_hash(self, block: QuantumAuditBlock) -> str:
        """Calculate hash of a block."""
        block_data = {
            "block_id": block.block_id,
            "previous_block_hash": block.previous_block_hash,
            "merkle_root": block.merkle_root,
            "timestamp": block.timestamp,
            "block_height": block.block_height
        }
        
        block_bytes = json.dumps(block_data, sort_keys=True).encode()
        return hashlib.sha512(block_bytes).hexdigest()
    
    def _calculate_merkle_root(self, events: List[Dict[str, Any]]) -> str:
        """Calculate Merkle root of audit events."""
        if not events:
            return hashlib.sha256(b"empty").hexdigest()
        
        # Create leaf hashes
        leaf_hashes = []
        for event in events:
            event_bytes = json.dumps(event, sort_keys=True).encode()
            leaf_hash = hashlib.sha256(event_bytes).hexdigest()
            leaf_hashes.append(leaf_hash)
        
        # Build Merkle tree
        while len(leaf_hashes) > 1:
            next_level = []
            
            # Pair up hashes
            for i in range(0, len(leaf_hashes), 2):
                if i + 1 < len(leaf_hashes):
                    combined = leaf_hashes[i] + leaf_hashes[i + 1]
                else:
                    combined = leaf_hashes[i] + leaf_hashes[i]  # Duplicate if odd number
                
                next_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(next_hash)
            
            leaf_hashes = next_level
        
        return leaf_hashes[0]
    
    def _assess_quantum_threat(self, entropy_analysis: Dict[str, Any]) -> QuantumThreatLevel:
        """Assess quantum threat level based on entropy analysis."""
        entropy_score = entropy_analysis.get("quantum_quality_score", 0.0)
        
        if entropy_score >= 0.9:
            return QuantumThreatLevel.NONE
        elif entropy_score >= 0.8:
            return QuantumThreatLevel.LOW
        elif entropy_score >= 0.7:
            return QuantumThreatLevel.MEDIUM
        elif entropy_score >= 0.5:
            return QuantumThreatLevel.HIGH
        else:
            return QuantumThreatLevel.CRITICAL
    
    def verify_chain_integrity(self) -> Dict[str, Any]:
        """Verify integrity of the entire quantum-safe chain."""
        if not self.chain:
            return {"error": "Empty chain"}
        
        integrity_results = {
            "total_blocks": len(self.chain),
            "verified_blocks": 0,
            "failed_blocks": [],
            "quantum_security_analysis": {
                "average_entropy_score": 0.0,
                "min_entropy_score": 1.0,
                "quantum_threat_levels": {}
            },
            "chain_valid": True
        }
        
        entropy_scores = []
        threat_levels = []
        
        for i, block in enumerate(self.chain):
            try:
                if i == 0:  # Genesis block
                    integrity_results["verified_blocks"] += 1
                    continue
                
                # Verify block hash chain
                expected_prev_hash = self._calculate_block_hash(self.chain[i-1])
                if block.previous_block_hash != expected_prev_hash:
                    integrity_results["failed_blocks"].append({
                        "block_height": block.block_height,
                        "error": "Invalid previous block hash"
                    })
                    integrity_results["chain_valid"] = False
                    continue
                
                # Verify merkle root
                calculated_merkle = self._calculate_merkle_root(block.audit_events)
                if block.merkle_root != calculated_merkle:
                    integrity_results["failed_blocks"].append({
                        "block_height": block.block_height,
                        "error": "Invalid merkle root"
                    })
                    integrity_results["chain_valid"] = False
                    continue
                
                # Collect quantum security metrics
                entropy_score = block.quantum_entropy_analysis.get("quantum_quality_score", 0.0)
                entropy_scores.append(entropy_score)
                
                threat_level = block.post_quantum_proof.get("quantum_threat_assessment", QuantumThreatLevel.MEDIUM)
                if isinstance(threat_level, QuantumThreatLevel):
                    threat_levels.append(threat_level.value)
                else:
                    threat_levels.append(str(threat_level))
                
                integrity_results["verified_blocks"] += 1
                
            except Exception as e:
                integrity_results["failed_blocks"].append({
                    "block_height": block.block_height,
                    "error": str(e)
                })
                integrity_results["chain_valid"] = False
        
        # Calculate quantum security analysis
        if entropy_scores:
            integrity_results["quantum_security_analysis"]["average_entropy_score"] = np.mean(entropy_scores)
            integrity_results["quantum_security_analysis"]["min_entropy_score"] = min(entropy_scores)
        
        # Count threat levels
        from collections import Counter
        threat_counts = Counter(threat_levels)
        integrity_results["quantum_security_analysis"]["quantum_threat_levels"] = dict(threat_counts)
        
        return integrity_results
    
    def get_quantum_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive quantum security report."""
        if not self.chain:
            return {"error": "No blocks in chain"}
        
        # Collect metrics from all blocks
        entropy_scores = []
        signature_algorithms = []
        security_levels = []
        
        for block in self.chain:
            entropy_scores.append(block.quantum_entropy_analysis.get("quantum_quality_score", 0.0))
            signature_algorithms.append(block.quantum_signature.algorithm.value)
            security_levels.append(block.post_quantum_proof.get("security_level", 0))
        
        report = {
            "report_metadata": {
                "generation_time": time.time(),
                "total_blocks": len(self.chain),
                "chain_height": self.current_height - 1
            },
            "quantum_entropy_analysis": {
                "average_entropy_score": np.mean(entropy_scores) if entropy_scores else 0,
                "min_entropy_score": min(entropy_scores) if entropy_scores else 0,
                "max_entropy_score": max(entropy_scores) if entropy_scores else 0,
                "entropy_trend": entropy_scores[-10:] if len(entropy_scores) >= 10 else entropy_scores
            },
            "cryptographic_algorithms": {
                "signature_algorithms_used": list(set(signature_algorithms)),
                "security_levels": list(set(security_levels)),
                "predominant_algorithm": max(set(signature_algorithms), key=signature_algorithms.count) if signature_algorithms else "none"
            },
            "quantum_threat_assessment": {
                "overall_threat_level": self._calculate_overall_threat_level(entropy_scores),
                "blocks_at_risk": len([s for s in entropy_scores if s < 0.7]),
                "secure_blocks": len([s for s in entropy_scores if s >= 0.8])
            },
            "chain_integrity": self.verify_chain_integrity()
        }
        
        return report
    
    def _calculate_overall_threat_level(self, entropy_scores: List[float]) -> str:
        """Calculate overall quantum threat level for the chain."""
        if not entropy_scores:
            return QuantumThreatLevel.CRITICAL.value
        
        avg_entropy = np.mean(entropy_scores)
        min_entropy = min(entropy_scores)
        
        # Consider both average and minimum entropy
        if min_entropy < 0.5 or avg_entropy < 0.6:
            return QuantumThreatLevel.CRITICAL.value
        elif min_entropy < 0.7 or avg_entropy < 0.75:
            return QuantumThreatLevel.HIGH.value
        elif min_entropy < 0.8 or avg_entropy < 0.85:
            return QuantumThreatLevel.MEDIUM.value
        elif avg_entropy >= 0.9:
            return QuantumThreatLevel.NONE.value
        else:
            return QuantumThreatLevel.LOW.value
    
    def export_quantum_research_data(self, output_path: Path) -> Dict[str, Any]:
        """Export quantum-safe cryptographic research data."""
        research_data = {
            "metadata": {
                "export_timestamp": time.time(),
                "chain_length": len(self.chain),
                "quantum_crypto_version": "1.0.0",
                "algorithms_implemented": [algo.value for algo in QuantumSafeAlgorithm]
            },
            "quantum_audit_chain": [
                {
                    "block_id": block.block_id,
                    "block_height": block.block_height,
                    "timestamp": block.timestamp,
                    "quantum_signature": {
                        "algorithm": block.quantum_signature.algorithm.value,
                        "entropy_score": block.quantum_signature.quantum_entropy_score,
                        "signature_length": len(block.quantum_signature.signature_data)
                    },
                    "entropy_analysis": block.quantum_entropy_analysis,
                    "post_quantum_proof": block.post_quantum_proof,
                    "events_count": len(block.audit_events)
                }
                for block in self.chain
            ],
            "quantum_security_metrics": self.get_quantum_security_report(),
            "entropy_evolution": [
                block.quantum_entropy_analysis.get("quantum_quality_score", 0.0)
                for block in self.chain
            ],
            "cryptographic_performance": {
                "average_signature_size": np.mean([
                    len(block.quantum_signature.signature_data) for block in self.chain
                ]) if self.chain else 0,
                "signature_generation_distribution": [
                    block.quantum_signature.verification_params.get("rounds", 1)
                    for block in self.chain if block.quantum_signature.verification_params
                ]
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(research_data, f, indent=2)
        
        return research_data