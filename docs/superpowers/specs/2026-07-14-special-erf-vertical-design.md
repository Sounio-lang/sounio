<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-14-special-erf-vertical-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-14-special-erf-vertical-design
-->

# Design — Harden the special::erf vertical (coordinated, disjoint)
**Date:** 2026-07-14 · no compiler changes · `special/` disjoint from active lanes (no open PR touching it);
scalar API is cross-module-safe (no arrays/HOF). This lane reads the module + adds run-proof/gate/docs; no source edits. EN-UK.

## State
`stdlib/special/erf.sio` — self-contained, green, native-compiles under **default Madaros**. Scalar API:
`erf`, `erfc`, `erfinv`, `normal_cdf`, `normal_quantile`. Verified externally: erf(1)=0.8427, Phi(1.96)=0.975002, z(0.975)=1.95996.

## Run-proof (known values / identities)
- erf(1)=0.8427008; erf(0)=0; odd: erf(−1)=−erf(1); erfc(0)=1; erfc(1)=1−erf(1).
- normal_cdf(0)=0.5; normal_cdf(1)=0.8413447; normal_cdf(1.96)=0.975 (±1e-3).
- normal_quantile(0.5)=0; normal_quantile(0.975)=1.96 (±1e-2); round-trip Phi(z(0.9))=0.9.
- Honest bound: `normal_quantile` is accurate near centre; deep-tail (p≥0.99) is approximate — not asserted; the example avoids implying tail precision.

## Layout / verification
`tests/stdlib/special/test_erf_stdlib.sio`, `examples/special/erf_report.sio`, `scripts/special_erf_gate.sh` (default Madaros) → `SPECIAL_ERF_GATE_OK`. Math-review logged. No source/compiler edits.
