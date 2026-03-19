# rlhf-audit-trail

**Immutable provenance tracking for RLHF training pipelines.**

When you fine-tune a language model with reinforcement learning from human feedback (RLHF), three things happen that regulators, auditors, and safety researchers will eventually ask you to prove:

1. **Who labelled what, and how?** — annotator identity, prompts seen, labels given.
2. **How did the policy change?** — gradient step size, optimizer state, update sequence.
3. **What privacy guarantees were applied?** — DP noise parameters, clipping norms, epsilon budget consumed.

`rlhf-audit-trail` logs all three into a blockchain-style append-only chain. Every entry's SHA-256 hash covers the previous entry's hash, so any post-hoc tampering is detectable without a trusted third party.

---

## Motivation

AI governance frameworks (EU AI Act, NIST AI RMF, CDEI guidelines) increasingly require training-time audit logs for high-risk models. For RLHF specifically:

- **Annotation provenance** matters for detecting bias, workforce-compliance claims, and data-sourcing disputes.
- **Policy-update lineage** is needed to reconstruct which gradient steps produced a given checkpoint.
- **DP accounting** must be auditable to support privacy claims (ε-differential privacy budgets drift silently if you don't track them).

This library gives you a minimal, dependency-free implementation suitable for embedding into existing RLHF training loops.

---

## Installation

```bash
pip install -e .
```

No external dependencies — pure Python stdlib (`hashlib`, `dataclasses`, `enum`, `json`, `uuid`).

---

## Core concepts

```
AuditEntry   — frozen dataclass; one event + its SHA-256 chain hash
AuditChain   — append-only list; each entry's hash covers prev_hash
RLHFEvents   — typed helpers: annotation(), policy_update(), dp_noise()
AuditVerifier — recomputes all hashes; surfaces any chain breaks
ProvenanceReport — human-readable lineage report for a trained policy
```

---

## Quick start

```python
from rlhf_audit_trail import AuditChain, AuditVerifier, ProvenanceReport, RLHFEvents

chain = AuditChain()

# Log annotator decisions
RLHFEvents.annotation(chain, annotator_id="ann_01", prompt="Explain RLHF", label="helpful")

# Log DP noise before each update
RLHFEvents.dp_noise(chain, sigma=1.1, clipping_norm=1.0, epsilon_spent=0.05)

# Log policy gradient steps
RLHFEvents.policy_update(chain, delta_norm=0.034, step=1, optimizer_state_hash="a3f1c8e2…")

# Verify integrity at any time
result = AuditVerifier.verify(chain)
assert result["valid"]

# Generate a full provenance report
print(ProvenanceReport.generate(chain, policy_name="my-policy-v1"))
```

---

## Demo

```bash
python demo.py
```

Simulates 5 annotation rounds + 3 policy updates with DP noise, generates a provenance report, then demonstrates tamper detection.

---

## Tests

```bash
python -m pytest tests/test_audit_trail.py -v
```

22 tests covering: entry immutability, chain linkage, tamper detection (entry hash, data hash, prev hash), empty/single-entry chains, provenance report content.

---

## Architecture

```
src/rlhf_audit_trail/
└── audit.py          ← entire implementation (~280 loc, no external deps)
```

### Hash scheme

```
entry_hash = SHA256({
    entry_id,
    timestamp,
    event_type,
    data_hash,      ← SHA256(payload JSON)
    metadata,
    prev_hash       ← previous entry's entry_hash
})
```

Changing *any* field — including the payload itself (via `data_hash`) or the chain linkage (via `prev_hash`) — invalidates the entry's hash and breaks the chain at that point.

---

## Limitations & extensions

- **In-memory only** — wire in a storage backend (SQLite, append-only file, write-once S3) for production use.
- **No signatures** — add asymmetric signing per annotator if non-repudiation matters.
- **No Merkle proofs** — useful for partial audit disclosure; straightforward to layer on.
- **DP accounting is log-only** — integrate with [Google DP library](https://github.com/google/differential-privacy) or [Opacus](https://opacus.ai/) for actual noise application.

---

## License

MIT
