"""Cryptographic engine for audit trail integrity and verification."""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel
import secrets
import os

logger = logging.getLogger(__name__)


class MerkleNode(BaseModel):
    """Node in a Merkle tree structure."""
    hash: str
    left_child: Optional["MerkleNode"] = None
    right_child: Optional["MerkleNode"] = None
    data: Optional[str] = None


class MerkleProof(BaseModel):
    """Proof of inclusion in a Merkle tree."""
    root_hash: str
    leaf_hash: str
    proof_path: List[Dict[str, str]]
    verified: bool = False


class CryptographicEngine:
    """Core cryptographic operations for audit trail integrity."""
    
    def __init__(self, key_size: int = 4096):
        self.key_size = key_size
        self._private_key: Optional[rsa.RSAPrivateKey] = None
        self._public_key: Optional[rsa.RSAPublicKey] = None
        self._setup_keys()
        
    def _setup_keys(self) -> None:
        """Generate or load RSA key pair."""
        try:
            # Try to load existing keys first
            if os.path.exists('.audit_private_key.pem'):
                with open('.audit_private_key.pem', 'rb') as f:
                    self._private_key = serialization.load_pem_private_key(
                        f.read(), password=None
                    )
                self._public_key = self._private_key.public_key()
                logger.info("Loaded existing cryptographic keys")
            else:
                # Generate new key pair
                self._generate_keys()
                self._save_keys()
                logger.info("Generated new cryptographic keys")
        except Exception as e:
            logger.warning(f"Failed to load keys, generating new ones: {e}")
            self._generate_keys()
            
    def _generate_keys(self) -> None:
        """Generate new RSA key pair."""
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size
        )
        self._public_key = self._private_key.public_key()
        
    def _save_keys(self) -> None:
        """Save keys to disk."""
        try:
            # Save private key
            private_pem = self._private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            with open('.audit_private_key.pem', 'wb') as f:
                f.write(private_pem)
            os.chmod('.audit_private_key.pem', 0o600)  # Restrict permissions
            
            # Save public key
            public_pem = self._public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            with open('.audit_public_key.pem', 'wb') as f:
                f.write(public_pem)
                
        except Exception as e:
            logger.error(f"Failed to save keys: {e}")
            
    def hash_data(self, data: Union[str, bytes, Dict[str, Any]]) -> str:
        """Generate SHA-256 hash of data."""
        if isinstance(data, dict):
            # Ensure consistent JSON serialization
            data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
            data_bytes = data_str.encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
            
        return hashlib.sha256(data_bytes).hexdigest()
    
    def sign_data(self, data: Union[str, bytes, Dict[str, Any]]) -> str:
        """Generate digital signature for data."""
        if not self._private_key:
            raise ValueError("Private key not available for signing")
            
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
            data_bytes = data_str.encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
            
        try:
            signature = self._private_key.sign(
                data_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature.hex()
        except Exception as e:
            logger.error(f"Failed to sign data: {e}")
            raise
    
    def verify_signature(self, data: Union[str, bytes, Dict[str, Any]], 
                        signature: str) -> bool:
        """Verify digital signature."""
        if not self._public_key:
            raise ValueError("Public key not available for verification")
            
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
            data_bytes = data_str.encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
            
        try:
            signature_bytes = bytes.fromhex(signature)
            self._public_key.verify(
                signature_bytes,
                data_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"Failed to verify signature: {e}")
            return False
    
    def encrypt_data(self, data: Union[str, bytes], password: str) -> Tuple[bytes, bytes]:
        """Encrypt data using AES-256 with PBKDF2 key derivation."""
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
            
        # Generate random salt and IV
        salt = secrets.token_bytes(16)
        iv = secrets.token_bytes(16)
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode('utf-8'))
        
        # Encrypt data
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padding_length = 16 - (len(data_bytes) % 16)
        padded_data = data_bytes + bytes([padding_length]) * padding_length
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Combine salt, IV, and encrypted data
        encrypted_bundle = salt + iv + encrypted_data
        
        return encrypted_bundle, self.hash_data(encrypted_bundle).encode('utf-8')
    
    def decrypt_data(self, encrypted_bundle: bytes, password: str) -> bytes:
        """Decrypt AES-256 encrypted data."""
        # Extract salt, IV, and encrypted data
        salt = encrypted_bundle[:16]
        iv = encrypted_bundle[16:32]
        encrypted_data = encrypted_bundle[32:]
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode('utf-8'))
        
        # Decrypt data
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Remove padding
        padding_length = padded_data[-1]
        data = padded_data[:-padding_length]
        
        return data


class MerkleTree:
    """Merkle tree implementation for audit trail verification."""
    
    def __init__(self, crypto_engine: CryptographicEngine):
        self.crypto = crypto_engine
        self.leaves: List[str] = []
        self.root: Optional[MerkleNode] = None
        
    def add_leaf(self, data: Union[str, Dict[str, Any]]) -> str:
        """Add a leaf to the tree and return its hash."""
        leaf_hash = self.crypto.hash_data(data)
        self.leaves.append(leaf_hash)
        self._rebuild_tree()
        return leaf_hash
        
    def _rebuild_tree(self) -> None:
        """Rebuild the Merkle tree from current leaves."""
        if not self.leaves:
            self.root = None
            return
            
        # Create leaf nodes
        nodes = [MerkleNode(hash=leaf_hash, data=leaf_hash) for leaf_hash in self.leaves]
        
        # Build tree bottom-up
        while len(nodes) > 1:
            next_level = []
            
            # Process nodes in pairs
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else left
                
                # Create parent node
                combined_hash = self.crypto.hash_data(left.hash + right.hash)
                parent = MerkleNode(
                    hash=combined_hash,
                    left_child=left,
                    right_child=right if right != left else None
                )
                next_level.append(parent)
                
            nodes = next_level
            
        self.root = nodes[0]
        
    def get_root_hash(self) -> Optional[str]:
        """Get the root hash of the tree."""
        return self.root.hash if self.root else None
        
    def generate_proof(self, leaf_hash: str) -> Optional[MerkleProof]:
        """Generate a proof of inclusion for a leaf."""
        if leaf_hash not in self.leaves or not self.root:
            return None
            
        proof_path = []
        self._collect_proof_path(self.root, leaf_hash, proof_path)
        
        return MerkleProof(
            root_hash=self.root.hash,
            leaf_hash=leaf_hash,
            proof_path=proof_path
        )
        
    def _collect_proof_path(self, node: MerkleNode, target_hash: str, 
                          proof_path: List[Dict[str, str]]) -> bool:
        """Recursively collect the proof path for a target hash."""
        if not node:
            return False
            
        # If this is a leaf node
        if node.data == target_hash:
            return True
            
        # Check left subtree
        if node.left_child and self._collect_proof_path(node.left_child, target_hash, proof_path):
            if node.right_child:
                proof_path.append({
                    "direction": "right",
                    "hash": node.right_child.hash
                })
            return True
            
        # Check right subtree
        if node.right_child and self._collect_proof_path(node.right_child, target_hash, proof_path):
            if node.left_child:
                proof_path.append({
                    "direction": "left", 
                    "hash": node.left_child.hash
                })
            return True
            
        return False
        
    def verify_proof(self, proof: MerkleProof) -> bool:
        """Verify a Merkle proof."""
        current_hash = proof.leaf_hash
        
        # Reconstruct the path to root
        for step in proof.proof_path:
            if step["direction"] == "left":
                current_hash = self.crypto.hash_data(step["hash"] + current_hash)
            else:
                current_hash = self.crypto.hash_data(current_hash + step["hash"])
                
        # Verify against root hash
        verified = current_hash == proof.root_hash
        proof.verified = verified
        return verified


class IntegrityVerifier:
    """Verifies the integrity of audit trails."""
    
    def __init__(self, crypto_engine: CryptographicEngine):
        self.crypto = crypto_engine
        
    def verify_chain_integrity(self, audit_records: List[Dict[str, Any]]) -> bool:
        """Verify the integrity of a chain of audit records."""
        if not audit_records:
            return True
            
        # Verify signatures
        for record in audit_records:
            if "signature" in record and "event_data" in record:
                signature = record.pop("signature")  # Remove for verification
                if not self.crypto.verify_signature(record, signature):
                    logger.error(f"Signature verification failed for record: {record.get('timestamp')}")
                    return False
                record["signature"] = signature  # Restore
                
        # Verify hash chain
        for i, record in enumerate(audit_records):
            if i > 0 and "previous_hash" in record:
                expected_hash = self.crypto.hash_data(audit_records[i-1])
                if record["previous_hash"] != expected_hash:
                    logger.error(f"Hash chain broken at record: {record.get('timestamp')}")
                    return False
                    
        logger.info(f"Verified integrity of {len(audit_records)} audit records")
        return True
        
    def verify_merkle_tree_integrity(self, tree: MerkleTree, 
                                   sample_leaves: List[str]) -> bool:
        """Verify Merkle tree integrity using sample leaves."""
        for leaf_hash in sample_leaves:
            proof = tree.generate_proof(leaf_hash)
            if not proof or not tree.verify_proof(proof):
                logger.error(f"Merkle proof verification failed for leaf: {leaf_hash}")
                return False
                
        logger.info(f"Verified Merkle tree integrity for {len(sample_leaves)} samples")
        return True