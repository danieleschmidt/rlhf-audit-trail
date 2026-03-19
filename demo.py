#!/usr/bin/env python3
"""
demo.py — Simulate an RLHF training run and print a provenance report.

Run with:
    python demo.py
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rlhf_audit_trail import (
    AuditChain,
    AuditVerifier,
    ProvenanceReport,
    RLHFEvents,
)


def simulate_rlhf_run() -> AuditChain:
    chain = AuditChain()
    rng   = random.Random(42)

    annotators = ["ann_alice", "ann_bob", "ann_carol"]
    prompts    = [
        "Explain gradient descent simply.",
        "Write a haiku about neural nets.",
        "What is RLHF and why does it matter?",
        "Summarise the EU AI Act in two sentences.",
        "Describe a responsible use of reinforcement learning.",
    ]

    print("Simulating 5 annotation rounds…")
    for i, prompt in enumerate(prompts):
        annotator = rng.choice(annotators)
        label     = rng.choice(["helpful", "neutral", "harmful"])
        RLHFEvents.annotation(
            chain,
            annotator_id=annotator,
            prompt=prompt,
            label=label,
            extra={"round": i + 1},
        )
        print(f"  Round {i+1}: {annotator} → '{label}'")

    print("\nApplying DP noise before each policy update…")
    dp_configs = [
        (1.1, 1.0, 0.05),
        (0.9, 0.8, 0.07),
        (1.2, 1.0, 0.04),
    ]
    for sigma, clip, eps in dp_configs:
        RLHFEvents.dp_noise(chain, sigma=sigma, clipping_norm=clip, epsilon_spent=eps)

    print("\nSimulating 3 policy update steps…")
    opt_hashes = [
        "a3f1c8e2d0b74591",
        "b7d3e9f1a2c54678",
        "c1e5f7a9b3d26890",
    ]
    for i, opt_hash in enumerate(opt_hashes):
        delta_norm = rng.uniform(0.01, 0.5)
        RLHFEvents.policy_update(
            chain,
            delta_norm=round(delta_norm, 4),
            step=i + 1,
            optimizer_state_hash=opt_hash,
            extra={"batch_size": 32, "lr": 1e-5},
        )
        print(f"  Step {i+1}: delta_norm={delta_norm:.4f}")

    return chain


def main() -> None:
    chain = simulate_rlhf_run()

    print(f"\nChain length: {len(chain)} entries")

    # Verify integrity
    result = AuditVerifier.verify(chain)
    print(f"Integrity check: {'PASS' if result['valid'] else 'FAIL'}")

    # Generate and print provenance report
    print()
    report = ProvenanceReport.generate(chain, policy_name="rlhf-demo-v1")
    print(report)

    # Demonstrate tamper detection
    print("\n--- Tamper detection demo ---")
    entry = chain[2]
    # Bypass frozen dataclass to simulate tampering
    object.__setattr__(entry, "data_hash", "deadbeef" * 8)
    bad = AuditVerifier.verify(chain)
    print(f"After tampering with entry 2: valid={bad['valid']}")
    for err in bad["errors"]:
        print(f"  ERROR: {err}")


if __name__ == "__main__":
    main()
