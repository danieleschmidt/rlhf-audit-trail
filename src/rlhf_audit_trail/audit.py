"""
RLHF Audit Trail — immutable provenance tracker for RLHF training pipelines.

Logs annotator decisions, policy updates, and differential-privacy noise into a
blockchain-style chain where every entry's hash covers the previous entry's hash.
Tamper-evident by construction; verifiable without trusted third parties.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class RLHFEventType(str, Enum):
    """Taxonomy of events that can appear in an RLHF audit trail."""
    ANNOTATION   = "annotation"     # Human annotator labels a prompt
    POLICY_UPDATE = "policy_update" # Gradient step changes policy weights
    DP_NOISE     = "dp_noise"       # Differential-privacy noise application


# ---------------------------------------------------------------------------
# Immutable audit entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditEntry:
    """
    A single immutable record in the audit chain.

    Fields
    ------
    entry_id      : Unique identifier (UUID4).
    timestamp     : Unix timestamp (float) of when the event was recorded.
    event_type    : One of the RLHFEventType values.
    data_hash     : SHA-256 of the event payload (hex string).
    metadata      : Arbitrary key/value context (annotator id, step number, …).
    prev_hash     : Hash of the immediately preceding entry (empty string for genesis).
    entry_hash    : Self-hash that covers all other fields including prev_hash.
    """
    entry_id  : str
    timestamp : float
    event_type: RLHFEventType
    data_hash : str
    metadata  : Dict[str, Any]
    prev_hash : str
    entry_hash: str

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        event_type: RLHFEventType,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        prev_hash: str = "",
    ) -> "AuditEntry":
        """
        Build a new AuditEntry.  The payload is hashed; the entry_hash
        covers every field so the record is tamper-evident.
        """
        entry_id  = str(uuid.uuid4())
        timestamp = time.time()
        data_hash = _sha256(payload)
        meta      = dict(metadata or {})

        entry_hash = _sha256({
            "entry_id":   entry_id,
            "timestamp":  timestamp,
            "event_type": event_type.value,
            "data_hash":  data_hash,
            "metadata":   meta,
            "prev_hash":  prev_hash,
        })

        return cls(
            entry_id=entry_id,
            timestamp=timestamp,
            event_type=event_type,
            data_hash=data_hash,
            metadata=meta,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


# ---------------------------------------------------------------------------
# Append-only chain
# ---------------------------------------------------------------------------

class AuditChain:
    """
    Append-only audit log.

    Each entry's hash includes the previous entry's hash so the entire
    history is tamper-evident (like a blockchain without the proof-of-work).
    """

    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        event_type: RLHFEventType,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Append a new event and return the resulting entry."""
        prev_hash = self._entries[-1].entry_hash if self._entries else ""
        entry = AuditEntry.create(
            event_type=event_type,
            payload=payload,
            metadata=metadata or {},
            prev_hash=prev_hash,
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> List[AuditEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> AuditEntry:
        return self._entries[idx]


# ---------------------------------------------------------------------------
# Typed event helpers
# ---------------------------------------------------------------------------

class RLHFEvents:
    """Convenience methods for logging typed RLHF events onto a chain."""

    @staticmethod
    def annotation(
        chain: AuditChain,
        annotator_id: str,
        prompt: str,
        label: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Record a human annotator's label for a prompt."""
        payload = {"annotator_id": annotator_id, "prompt": prompt, "label": label}
        meta    = {"annotator_id": annotator_id, **(extra or {})}
        return chain.append(RLHFEventType.ANNOTATION, payload, meta)

    @staticmethod
    def policy_update(
        chain: AuditChain,
        delta_norm: float,
        step: int,
        optimizer_state_hash: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Record a gradient step that changes the policy."""
        payload = {
            "delta_norm": delta_norm,
            "step": step,
            "optimizer_state_hash": optimizer_state_hash,
        }
        meta = {"step": step, **(extra or {})}
        return chain.append(RLHFEventType.POLICY_UPDATE, payload, meta)

    @staticmethod
    def dp_noise(
        chain: AuditChain,
        sigma: float,
        clipping_norm: float,
        epsilon_spent: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Record a differential-privacy noise application."""
        payload = {
            "sigma": sigma,
            "clipping_norm": clipping_norm,
            "epsilon_spent": epsilon_spent,
        }
        meta = {"sigma": sigma, "epsilon_spent": epsilon_spent, **(extra or {})}
        return chain.append(RLHFEventType.DP_NOISE, payload, meta)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class AuditVerifier:
    """
    Verifies the integrity of an AuditChain by recomputing every hash.

    Detects:
    * Tampered entry content (data_hash or metadata changed).
    * Broken chain links (prev_hash mismatch).
    * Recomputed entry_hash drift.
    """

    @staticmethod
    def verify(chain: AuditChain) -> Dict[str, Any]:
        """
        Returns a report dict with keys:
            valid       : bool — True if chain is untampered.
            length      : int  — number of entries checked.
            errors      : list — description of each violation found.
        """
        errors: List[str] = []
        entries = chain.entries
        expected_prev = ""

        for idx, entry in enumerate(entries):
            # 1. Recompute data_hash from stored metadata (we store payload hash, not
            #    the original payload, so we verify the entry_hash covers what it claims)
            recomputed_entry_hash = _sha256({
                "entry_id":   entry.entry_id,
                "timestamp":  entry.timestamp,
                "event_type": entry.event_type.value,
                "data_hash":  entry.data_hash,
                "metadata":   entry.metadata,
                "prev_hash":  entry.prev_hash,
            })

            if recomputed_entry_hash != entry.entry_hash:
                errors.append(
                    f"Entry {idx} ({entry.entry_id[:8]}…): entry_hash mismatch — "
                    f"stored={entry.entry_hash[:16]}… computed={recomputed_entry_hash[:16]}…"
                )

            # 2. Check chain linkage
            if entry.prev_hash != expected_prev:
                errors.append(
                    f"Entry {idx} ({entry.entry_id[:8]}…): prev_hash broken — "
                    f"expected={expected_prev[:16] or '(genesis)'}… "
                    f"found={entry.prev_hash[:16] or '(genesis)'}…"
                )

            expected_prev = entry.entry_hash

        return {
            "valid":  len(errors) == 0,
            "length": len(entries),
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# Provenance report
# ---------------------------------------------------------------------------

class ProvenanceReport:
    """
    Generates a human-readable lineage report for a trained RLHF policy.

    Shows the complete audit trail grouped by event type, with a summary
    of annotation coverage, policy update history, and DP budget consumed.
    """

    @staticmethod
    def generate(chain: AuditChain, policy_name: str = "policy") -> str:
        entries    = chain.entries
        result     = AuditVerifier.verify(chain)
        integrity  = "✅ INTACT" if result["valid"] else f"❌ COMPROMISED ({len(result['errors'])} errors)"

        annotations    = [e for e in entries if e.event_type == RLHFEventType.ANNOTATION]
        policy_updates = [e for e in entries if e.event_type == RLHFEventType.POLICY_UPDATE]
        dp_events      = [e for e in entries if e.event_type == RLHFEventType.DP_NOISE]

        lines = [
            "=" * 70,
            f"  RLHF PROVENANCE REPORT — {policy_name.upper()}",
            "=" * 70,
            f"  Chain integrity : {integrity}",
            f"  Total entries   : {len(entries)}",
            f"  Generated at    : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
            "",
        ]

        # ---- Annotation summary ----
        lines.append("─" * 70)
        lines.append(f"  ANNOTATIONS ({len(annotations)} total)")
        lines.append("─" * 70)
        if annotations:
            annotators: Dict[str, int] = {}
            for e in annotations:
                aid = e.metadata.get("annotator_id", "unknown")
                annotators[aid] = annotators.get(aid, 0) + 1
            for aid, count in sorted(annotators.items()):
                lines.append(f"  • {aid}: {count} label(s)")
            lines.append(f"  First: {_fmt_time(annotations[0].timestamp)}")
            lines.append(f"  Last : {_fmt_time(annotations[-1].timestamp)}")
        else:
            lines.append("  (none)")
        lines.append("")

        # ---- Policy update summary ----
        lines.append("─" * 70)
        lines.append(f"  POLICY UPDATES ({len(policy_updates)} total)")
        lines.append("─" * 70)
        if policy_updates:
            for e in policy_updates:
                step  = e.metadata.get("step", "?")
                lines.append(
                    f"  • Step {step:>4}  entry={e.entry_id[:8]}…  "
                    f"hash={e.entry_hash[:16]}…"
                )
        else:
            lines.append("  (none)")
        lines.append("")

        # ---- DP budget summary ----
        lines.append("─" * 70)
        lines.append(f"  DIFFERENTIAL PRIVACY ({len(dp_events)} noise events)")
        lines.append("─" * 70)
        if dp_events:
            total_eps = sum(e.metadata.get("epsilon_spent", 0.0) for e in dp_events)
            lines.append(f"  Total ε spent : {total_eps:.6f}")
            for e in dp_events:
                lines.append(
                    f"  • σ={e.metadata.get('sigma','?'):<6}  "
                    f"ε={e.metadata.get('epsilon_spent','?'):.6f}  "
                    f"entry={e.entry_id[:8]}…"
                )
        else:
            lines.append("  (none)")
        lines.append("")

        # ---- Chain tail ----
        if entries:
            lines.append("─" * 70)
            lines.append("  CHAIN TAIL")
            lines.append("─" * 70)
            tail = entries[-1]
            lines.append(f"  Last entry ID : {tail.entry_id}")
            lines.append(f"  Last entry hash: {tail.entry_hash}")
            lines.append("")

        if not result["valid"]:
            lines.append("─" * 70)
            lines.append("  INTEGRITY ERRORS")
            lines.append("─" * 70)
            for err in result["errors"]:
                lines.append(f"  ⚠  {err}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256(obj: Any) -> str:
    """Deterministic SHA-256 of a JSON-serialisable object."""
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fmt_time(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
