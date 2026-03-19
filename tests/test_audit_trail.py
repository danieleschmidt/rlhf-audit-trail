"""Tests for the RLHF audit trail core implementation."""

import sys
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlhf_audit_trail import (
    AuditChain,
    AuditEntry,
    AuditVerifier,
    ProvenanceReport,
    RLHFEventType,
    RLHFEvents,
)
from rlhf_audit_trail.audit import _sha256


# ---------------------------------------------------------------------------
# AuditEntry
# ---------------------------------------------------------------------------

class TestAuditEntry:
    def test_create_genesis(self):
        e = AuditEntry.create(
            event_type=RLHFEventType.ANNOTATION,
            payload={"annotator_id": "alice", "prompt": "hello", "label": "good"},
        )
        assert e.prev_hash == ""
        assert len(e.entry_hash) == 64
        assert e.entry_id != ""

    def test_immutable(self):
        e = AuditEntry.create(RLHFEventType.ANNOTATION, {"x": 1})
        with pytest.raises((AttributeError, TypeError)):
            e.entry_hash = "tampered"  # type: ignore[misc]

    def test_entry_hash_covers_prev_hash(self):
        e1 = AuditEntry.create(RLHFEventType.ANNOTATION, {"x": 1})
        e2 = AuditEntry.create(RLHFEventType.POLICY_UPDATE, {"y": 2}, prev_hash=e1.entry_hash)
        assert e2.prev_hash == e1.entry_hash
        assert e2.entry_hash != e1.entry_hash

    def test_to_dict_serialisable(self):
        e = AuditEntry.create(RLHFEventType.DP_NOISE, {"sigma": 1.0})
        d = e.to_dict()
        assert d["event_type"] == "dp_noise"
        assert isinstance(d["metadata"], dict)


# ---------------------------------------------------------------------------
# AuditChain
# ---------------------------------------------------------------------------

class TestAuditChain:
    def test_empty_chain(self):
        chain = AuditChain()
        assert len(chain) == 0

    def test_append_links_prev_hash(self):
        chain = AuditChain()
        e1 = chain.append(RLHFEventType.ANNOTATION, {"a": 1})
        e2 = chain.append(RLHFEventType.ANNOTATION, {"b": 2})
        assert e1.prev_hash == ""
        assert e2.prev_hash == e1.entry_hash

    def test_entries_are_copies(self):
        chain = AuditChain()
        chain.append(RLHFEventType.ANNOTATION, {"x": 1})
        entries = chain.entries
        # Mutating the list should not affect the chain
        entries.clear()
        assert len(chain) == 1

    def test_multiple_event_types(self):
        chain = AuditChain()
        chain.append(RLHFEventType.ANNOTATION, {})
        chain.append(RLHFEventType.POLICY_UPDATE, {})
        chain.append(RLHFEventType.DP_NOISE, {})
        assert len(chain) == 3


# ---------------------------------------------------------------------------
# RLHFEvents helpers
# ---------------------------------------------------------------------------

class TestRLHFEvents:
    def test_annotation_event(self):
        chain = AuditChain()
        e = RLHFEvents.annotation(chain, "ann_01", "What is AI?", "helpful")
        assert e.event_type == RLHFEventType.ANNOTATION
        assert e.metadata["annotator_id"] == "ann_01"

    def test_policy_update_event(self):
        chain = AuditChain()
        e = RLHFEvents.policy_update(chain, delta_norm=0.05, step=1, optimizer_state_hash="abc123")
        assert e.event_type == RLHFEventType.POLICY_UPDATE
        assert e.metadata["step"] == 1

    def test_dp_noise_event(self):
        chain = AuditChain()
        e = RLHFEvents.dp_noise(chain, sigma=1.1, clipping_norm=1.0, epsilon_spent=0.05)
        assert e.event_type == RLHFEventType.DP_NOISE
        assert e.metadata["sigma"] == 1.1
        assert e.metadata["epsilon_spent"] == 0.05


# ---------------------------------------------------------------------------
# AuditVerifier
# ---------------------------------------------------------------------------

class TestAuditVerifier:
    def _build_chain(self) -> AuditChain:
        chain = AuditChain()
        RLHFEvents.annotation(chain, "alice", "p1", "good")
        RLHFEvents.annotation(chain, "bob",   "p2", "neutral")
        RLHFEvents.dp_noise(chain, 1.0, 1.0, 0.05)
        RLHFEvents.policy_update(chain, 0.1, 1, "hash0")
        return chain

    def test_valid_chain(self):
        chain = self._build_chain()
        result = AuditVerifier.verify(chain)
        assert result["valid"] is True
        assert result["errors"] == []
        assert result["length"] == 4

    def test_tampered_entry_hash_detected(self):
        chain = self._build_chain()
        entry = chain[1]
        object.__setattr__(entry, "entry_hash", "0" * 64)
        result = AuditVerifier.verify(chain)
        assert result["valid"] is False
        # Two errors: entry_hash mismatch + broken link in next entry
        assert len(result["errors"]) >= 1

    def test_tampered_data_hash_detected(self):
        chain = self._build_chain()
        entry = chain[0]
        object.__setattr__(entry, "data_hash", "deadbeef" * 8)
        result = AuditVerifier.verify(chain)
        assert result["valid"] is False

    def test_tampered_prev_hash_detected(self):
        chain = self._build_chain()
        entry = chain[2]
        object.__setattr__(entry, "prev_hash", "0" * 64)
        result = AuditVerifier.verify(chain)
        assert result["valid"] is False

    def test_empty_chain_is_valid(self):
        chain = AuditChain()
        result = AuditVerifier.verify(chain)
        assert result["valid"] is True
        assert result["length"] == 0

    def test_single_entry_valid(self):
        chain = AuditChain()
        RLHFEvents.annotation(chain, "a", "prompt", "label")
        result = AuditVerifier.verify(chain)
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# ProvenanceReport
# ---------------------------------------------------------------------------

class TestProvenanceReport:
    def _full_chain(self) -> AuditChain:
        chain = AuditChain()
        for i in range(5):
            RLHFEvents.annotation(chain, f"ann_{i}", f"prompt_{i}", "good", {"round": i+1})
        for i in range(3):
            RLHFEvents.dp_noise(chain, sigma=1.0+i*0.1, clipping_norm=1.0, epsilon_spent=0.05*(i+1))
            RLHFEvents.policy_update(chain, delta_norm=0.1*(i+1), step=i+1, optimizer_state_hash=f"hash{i}")
        return chain

    def test_report_is_string(self):
        chain = self._full_chain()
        report = ProvenanceReport.generate(chain, "test-policy")
        assert isinstance(report, str)
        assert "RLHF PROVENANCE REPORT" in report

    def test_report_shows_integrity_intact(self):
        chain = self._full_chain()
        report = ProvenanceReport.generate(chain)
        assert "INTACT" in report

    def test_report_shows_annotation_count(self):
        chain = self._full_chain()
        report = ProvenanceReport.generate(chain)
        assert "ANNOTATIONS (5 total)" in report

    def test_report_shows_dp_budget(self):
        chain = self._full_chain()
        report = ProvenanceReport.generate(chain)
        assert "DIFFERENTIAL PRIVACY" in report
        assert "ε spent" in report

    def test_report_shows_compromised_on_tamper(self):
        chain = self._full_chain()
        object.__setattr__(chain[0], "data_hash", "bad" * 21)
        report = ProvenanceReport.generate(chain)
        assert "COMPROMISED" in report
