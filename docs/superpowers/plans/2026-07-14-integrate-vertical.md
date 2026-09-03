<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-14-integrate-vertical
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-14-integrate-vertical
-->

# integrate::epistemic_ode Run-Proof — Plan
> Default Madaros (self-contained). No source edits (disjoint from active lanes). Spec: docs/superpowers/specs/2026-07-14-integrate-vertical-design.md.
## Task 1 — run-proof driver
- [ ] `tests/stdlib/integrate/test_integrate_stdlib.sio`: convergence to e⁻¹ (err@2000 < err@200, <2e-4); half-life y(ln2)=0.5; decay y(1)<y0; y(2)=2e⁻¹; uncertainty ∈[0.049,0.051]. `INTEGRATE_STDLIB_OK`. Compile+run. Commit.
## Task 2 — example + gate
- [ ] `examples/integrate/decay_report.sio` (elimination report). `scripts/integrate_gate.sh` → `INTEGRATE_GATE_OK`. Commit.
## Task 3 — math-review + PR
- [ ] math-review; governance sync; push; PR to main with `--auto` merge.
