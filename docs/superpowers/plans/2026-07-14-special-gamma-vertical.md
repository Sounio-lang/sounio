<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-14-special-gamma-vertical
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-14-special-gamma-vertical
-->

# special::gamma Run-Proof — Plan
> Default Madaros (self-contained, scalar). No source edits (disjoint). Spec: docs/superpowers/specs/2026-07-14-special-gamma-vertical-design.md.
1. Run-proof `tests/stdlib/special/test_gamma_stdlib.sio`: gamma factorials + √π + recurrence; lgamma; digamma (=−γ, by value) + recurrence. `GAMMA_STDLIB_OK`.
2. Example `examples/special/gamma_report.sio` (negative-safe digamma print) + gate `scripts/special_gamma_gate.sh` → `SPECIAL_GAMMA_GATE_OK`.
3. math-review; governance sync; PR to main (rebase-on-conflict).
