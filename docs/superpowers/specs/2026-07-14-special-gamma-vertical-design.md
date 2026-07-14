<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-14-special-gamma-vertical-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-14-special-gamma-vertical-design
-->

# Design — Harden the special::gamma vertical (coordinated, disjoint)
**Date:** 2026-07-14 · no compiler changes · `special/gamma.sio` cold (no open PR). Scalar API is
cross-module-safe. Reads the module + adds run-proof/example/gate; no source edits. EN-UK.

## State
`stdlib/special/gamma.sio` — self-contained, green, native under **default Madaros**. `gamma`, `lgamma`,
`digamma` (scalar). Verified externally: gamma(5)=24, gamma(0.5)=√π, lgamma(5)=ln24, digamma(2)=1−γ.

## Run-proof (known values / identities)
- gamma(n)=(n−1)!: gamma(1)=1, gamma(5)=24, gamma(6)=120; recurrence gamma(6)=5·gamma(5).
- gamma(0.5)=√π=1.7724539; gamma(1.5)=√π/2.
- lgamma(5)=ln24=3.1780538; lgamma(1)=0.
- digamma(1)=−γ=−0.5772157 (asserted by f64 value — print(f64) mis-renders negatives, #890);
  digamma(2)=1−γ; recurrence digamma(2)=digamma(1)+1.

## Layout / verification
`tests/stdlib/special/test_gamma_stdlib.sio`, `examples/special/gamma_report.sio` (negative-safe digamma
print), `scripts/special_gamma_gate.sh` (default Madaros) → `SPECIAL_GAMMA_GATE_OK`. Math-review logged. No source/compiler edits.
