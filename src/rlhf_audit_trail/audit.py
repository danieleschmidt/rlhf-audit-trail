"""Audit logging system for RLHF training provenance.

This module provides comprehensive audit logging capabilities with
cryptographic integrity verification and immutable storage.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import logging
import hashlib

from .exceptions import AuditTrailError, StorageError, CryptographicError
from .crypto import CryptographicEngine
from .storage import StorageBackend


class EventType(Enum):
    """Types of events in the audit trail."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ANNOTATION = "annotation"
    POLICY_UPDATE = "policy_update"
    CHECKPOINT = "checkpoint"
    COMPLIANCE_CHECK = "compliance_check"
    PRIVACY_BUDGET_UPDATE = "privacy_budget_update"
    ERROR = "error"
    SYSTEM_EVENT = "system_event"
    VALIDATION = "validation"


@dataclass
class TrainingEvent:
    """Represents a single event in the RLHF training audit trail."""
    event_type: EventType
    session_id: str
    timestamp: float
    data: Dict[str, Any]
    event_id: Optional[str] = None
    hash: Optional[str] = None
    signature: Optional[str] = None
    merkle_proof: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
    
    def compute_hash(self, hash_algorithm: str = "SHA-256") -> str:
        """Compute cryptographic hash of the event."""
        event_data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "data": self.data
        }
        
        # Create deterministic JSON representation with enum handling
        try:
            json_bytes = json.dumps(event_data, sort_keys=True, separators=(',', ':'), default=lambda x: x.value if hasattr(x, 'value') else str(x)).encode('utf-8')
        except (TypeError, AttributeError):
            # Fallback: convert complex objects to string
            json_bytes = json.dumps(event_data, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
        
        if hash_algorithm == "SHA-256":
            return hashlib.sha256(json_bytes).hexdigest()
        elif hash_algorithm == "SHA-512":
            return hashlib.sha512(json_bytes).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "hash": self.hash,
            "signature": self.signature,
            "merkle_proof": self.merkle_proof
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingEvent":
        """Create event from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            data=data["data"],
            hash=data.get("hash"),
            signature=data.get("signature"),
            merkle_proof=data.get("merkle_proof")
        )


@dataclass
class AuditChain:
    """Represents a chain of audit events with cryptographic linking."""
    events: List[TrainingEvent]
    chain_hash: str
    merkle_root: str
    signatures_valid: bool
    chain_intact: bool
    
    def __init__(self, events: List[TrainingEvent]):
        self.events = events
        self.chain_hash = self._compute_chain_hash()
        self.merkle_root = self._compute_merkle_root()
        self.signatures_valid = False  # Will be verified separately
        self.chain_intact = True  # Will be verified separately
    
    def _compute_chain_hash(self) -> str:
        """Compute hash of the entire chain."""
        if not self.events:
            return ""
        
        # Hash all event hashes together
        combined_hash = ""
        for event in self.events:
            if event.hash:
                combined_hash += event.hash
        
        return hashlib.sha256(combined_hash.encode()).hexdigest()
    
    def _compute_merkle_root(self) -> str:
        """Compute Merkle tree root for the event chain."""
        if not self.events:
            return ""
        
        # Simple merkle tree implementation
        hashes = [event.hash for event in self.events if event.hash]
        
        while len(hashes) > 1:
            next_level = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined = hashes[i] + hashes[i + 1]
                else:
                    combined = hashes[i] + hashes[i]  # Duplicate if odd number
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = next_level
        
        return hashes[0] if hashes else ""


class AuditLogger:
    """Central audit logging system for RLHF training events.
    
    This class provides comprehensive audit logging with cryptographic
    integrity verification and immutable storage capabilities.
    """
    
    def __init__(
        self,
        storage: StorageBackend,
        crypto: CryptographicEngine,
        session_id: Optional[str] = None,
        batch_size: int = 100,
        auto_flush: bool = True
    ):
        """Initialize audit logger.
        
        Args:
            storage: Storage backend for audit logs
            crypto: Cryptographic engine for signing/verification
            session_id: Current training session ID
            batch_size: Number of events to batch before flushing
            auto_flush: Automatically flush events to storage
        """
        self.storage = storage
        self.crypto = crypto
        self.session_id = session_id
        self.batch_size = batch_size
        self.auto_flush = auto_flush
        
        self.event_buffer: List[TrainingEvent] = []
        self.event_count = 0
        self.logger = logging.getLogger(__name__)
        
        # Chain tracking for integrity
        self.previous_hash: Optional[str] = None
        
    async def log_event(self, event: TrainingEvent) -> str:
        """Log a training event to the audit trail.
        
        Args:
            event: Training event to log
            
        Returns:
            Event ID of the logged event
            
        Raises:
            AuditTrailError: If logging fails
        """
        try:
            # Ensure event has required fields
            if event.event_id is None:
                event.event_id = str(uuid.uuid4())
            
            # Compute cryptographic hash
            event.hash = event.compute_hash()
            
            # Chain events together
            if self.previous_hash:
                # Include previous hash in data for chaining
                event.data["_previous_hash"] = self.previous_hash
                # Recompute hash with chaining data
                event.hash = event.compute_hash()
            
            # Sign the event (crypto.sign_data is synchronous)
            event.signature = self.crypto.sign_data(event.hash)
            
            # Add to buffer
            self.event_buffer.append(event)
            self.event_count += 1
            self.previous_hash = event.hash
            
            # Auto-flush if needed
            if self.auto_flush and len(self.event_buffer) >= self.batch_size:
                await self.flush_events()
            
            self.logger.debug(f"Logged event {event.event_id} of type {event.event_type.value}")
            return event.event_id
            
        except Exception as e:
            raise AuditTrailError(f"Failed to log audit event: {str(e)}") from e
    
    async def flush_events(self) -> int:
        """Flush buffered events to storage.
        
        Returns:
            Number of events flushed
        """
        if not self.event_buffer:
            return 0
        
        try:
            # Create audit chain
            chain = AuditChain(self.event_buffer.copy())
            
            # Store events individually
            for event in self.event_buffer:
                await self._store_event(event)
            
            # Store chain metadata
            await self._store_chain_metadata(chain)
            
            events_flushed = len(self.event_buffer)
            self.event_buffer.clear()
            
            self.logger.info(f"Flushed {events_flushed} events to storage")
            return events_flushed
            
        except Exception as e:
            raise StorageError(f"Failed to flush events to storage: {str(e)}") from e
    
    async def _store_event(self, event: TrainingEvent):
        """Store individual event to storage."""
        if not self.session_id:
            raise AuditTrailError("No session ID set for audit logger")
        
        # Create storage path
        path = f"audit_logs/{self.session_id}/events/{event.event_id}.json"
        
        # Serialize event
        event_data = event.to_dict()
        
        # Store encrypted
        await self.storage.store_encrypted(path, event_data, self.crypto)
    
    async def _store_chain_metadata(self, chain: AuditChain):
        """Store chain metadata for integrity verification."""
        if not self.session_id:
            raise AuditTrailError("No session ID set for audit logger")
        
        chain_metadata = {
            "session_id": self.session_id,
            "event_count": len(chain.events),
            "chain_hash": chain.chain_hash,
            "merkle_root": chain.merkle_root,
            "timestamp": time.time(),
            "event_range": {
                "start": chain.events[0].timestamp if chain.events else None,
                "end": chain.events[-1].timestamp if chain.events else None
            }
        }
        
        # Create storage path with timestamp to avoid conflicts
        timestamp_str = str(int(time.time() * 1000))  # milliseconds
        path = f"audit_logs/{self.session_id}/chains/chain_{timestamp_str}.json"
        
        # Store encrypted
        await self.storage.store_encrypted(path, chain_metadata, self.crypto)
    
    async def get_session_events(
        self,
        session_id: str,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[TrainingEvent]:
        """Retrieve audit events for a training session.
        
        Args:
            session_id: Training session ID
            event_types: Filter by event types
            start_time: Filter events after this timestamp
            end_time: Filter events before this timestamp
            limit: Maximum number of events to return
            
        Returns:
            List of training events
        """
        try:
            # List all event files for the session
            event_files = await self.storage.list_files(f"audit_logs/{session_id}/events/")
            
            events = []
            for file_path in event_files:
                if file_path.endswith('.json'):
                    # Load and decrypt event
                    event_data = await self.storage.load_encrypted(file_path, self.crypto)
                    event = TrainingEvent.from_dict(event_data)
                    
                    # Apply filters
                    if event_types and event.event_type not in event_types:
                        continue
                    if start_time and event.timestamp < start_time:
                        continue
                    if end_time and event.timestamp > end_time:
                        continue
                    
                    events.append(event)
                    
                    # Apply limit
                    if limit and len(events) >= limit:
                        break
            
            # Sort by timestamp
            events.sort(key=lambda e: e.timestamp)
            
            return events
            
        except Exception as e:
            raise StorageError(f"Failed to retrieve session events: {str(e)}") from e
    
    async def verify_event_integrity(self, event: TrainingEvent) -> bool:
        """Verify the cryptographic integrity of an event.
        
        Args:
            event: Event to verify
            
        Returns:
            True if event integrity is valid
        """
        try:
            # Verify hash
            expected_hash = event.compute_hash()
            if event.hash != expected_hash:
                self.logger.warning(f"Hash mismatch for event {event.event_id}")
                return False
            
            # Verify signature
            if event.signature:
                signature_valid = await self.crypto.verify_signature(
                    event.hash, event.signature
                )
                if not signature_valid:
                    self.logger.warning(f"Invalid signature for event {event.event_id}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying event integrity: {str(e)}")
            return False
    
    async def verify_chain_integrity(
        self, 
        session_id: str,
        start_event_id: Optional[str] = None,
        end_event_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify the integrity of an event chain.
        
        Args:
            session_id: Training session ID
            start_event_id: Starting event ID for verification
            end_event_id: Ending event ID for verification
            
        Returns:
            Dictionary containing verification results
        """
        try:
            # Get all events for the session
            events = await self.get_session_events(session_id)
            
            # Filter by event IDs if specified
            if start_event_id or end_event_id:
                start_idx = 0
                end_idx = len(events)
                
                for i, event in enumerate(events):
                    if start_event_id and event.event_id == start_event_id:
                        start_idx = i
                    if end_event_id and event.event_id == end_event_id:
                        end_idx = i + 1
                        break
                
                events = events[start_idx:end_idx]
            
            # Verify individual event integrity
            individual_verification = []
            for event in events:
                is_valid = await self.verify_event_integrity(event)
                individual_verification.append({
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp,
                    "is_valid": is_valid
                })
            
            # Verify chain linking
            chain_intact = True
            for i in range(1, len(events)):
                current_event = events[i]
                previous_hash = current_event.data.get("_previous_hash")
                expected_previous_hash = events[i-1].hash
                
                if previous_hash != expected_previous_hash:
                    chain_intact = False
                    break
            
            # Create chain and verify merkle root
            chain = AuditChain(events)
            
            return {
                "session_id": session_id,
                "event_count": len(events),
                "individual_events_valid": all(v["is_valid"] for v in individual_verification),
                "chain_intact": chain_intact,
                "merkle_root": chain.merkle_root,
                "chain_hash": chain.chain_hash,
                "verification_timestamp": time.time(),
                "event_details": individual_verification
            }
            
        except Exception as e:
            raise CryptographicError(f"Failed to verify chain integrity: {str(e)}") from e
    
    async def generate_audit_report(
        self,
        session_id: str,
        include_raw_events: bool = False,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report for a session.
        
        Args:
            session_id: Training session ID
            include_raw_events: Include full event data
            format: Report format ("json", "html", "pdf")
            
        Returns:
            Audit report data
        """
        try:
            # Get all events
            events = await self.get_session_events(session_id)
            
            # Verify integrity
            integrity_result = await self.verify_chain_integrity(session_id)
            
            # Generate statistics
            event_stats = {}
            for event in events:
                event_type = event.event_type.value
                event_stats[event_type] = event_stats.get(event_type, 0) + 1
            
            # Calculate session duration
            session_duration = None
            if events:
                start_time = min(e.timestamp for e in events)
                end_time = max(e.timestamp for e in events)
                session_duration = end_time - start_time
            
            # Create report
            report = {
                "session_id": session_id,
                "report_generated_at": time.time(),
                "session_summary": {
                    "total_events": len(events),
                    "session_duration": session_duration,
                    "event_statistics": event_stats,
                    "integrity_verified": integrity_result["individual_events_valid"],
                    "chain_intact": integrity_result["chain_intact"]
                },
                "integrity_verification": integrity_result
            }
            
            if include_raw_events:
                report["raw_events"] = [event.to_dict() for event in events]
            
            return report
            
        except Exception as e:
            raise AuditTrailError(f"Failed to generate audit report: {str(e)}") from e
    
    async def close(self):
        """Close the audit logger and flush any remaining events."""
        if self.event_buffer:
            await self.flush_events()
        
        self.logger.info("Audit logger closed")


class AuditMetrics:
    """Metrics collection for audit trail system."""
    
    def __init__(self):
        self.events_logged = 0
        self.events_verified = 0
        self.integrity_checks_passed = 0
        self.integrity_checks_failed = 0
        self.storage_operations = 0
        self.storage_errors = 0
        
    def record_event_logged(self):
        """Record that an event was logged."""
        self.events_logged += 1
    
    def record_event_verified(self, passed: bool):
        """Record an event verification result."""
        self.events_verified += 1
        if passed:
            self.integrity_checks_passed += 1
        else:
            self.integrity_checks_failed += 1
    
    def record_storage_operation(self, success: bool):
        """Record a storage operation result."""
        self.storage_operations += 1
        if not success:
            self.storage_errors += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            "events_logged": self.events_logged,
            "events_verified": self.events_verified,
            "integrity_checks_passed": self.integrity_checks_passed,
            "integrity_checks_failed": self.integrity_checks_failed,
            "storage_operations": self.storage_operations,
            "storage_errors": self.storage_errors,
            "integrity_success_rate": (
                self.integrity_checks_passed / max(self.events_verified, 1)
            ),
            "storage_success_rate": (
                (self.storage_operations - self.storage_errors) / max(self.storage_operations, 1)
            )
        }