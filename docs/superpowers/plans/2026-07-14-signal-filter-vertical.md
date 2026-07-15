<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-14-signal-filter-vertical
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-14-signal-filter-vertical
-->

# signal::filter Run-Proof — Plan
> Default Madaros (self-contained, by-reference API). No source edits (disjoint). Spec: docs/superpowers/specs/2026-07-14-signal-filter-vertical-design.md.
1. Run-proof `tests/stdlib/signal/test_filter_stdlib.sio`: MA DC gain 1 + impulse 0.25×4; IIR1 lowpass DC gain 1; IIR1 highpass DC gain 0. `FILTER_STDLIB_OK`.
2. Example `examples/signal/filter_report.sio` + gate `scripts/signal_filter_gate.sh` → `SIGNAL_FILTER_GATE_OK`.
3. math-review; governance sync; PR to main (rebase-on-conflict).
