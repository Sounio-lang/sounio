<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-15-batched-verticals-index
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-15-batched-verticals-index
-->

# Index — grouped stdlib run-proof verticals (batches #941–#967)

**Date:** 2026-07-15 · No compiler changes. Coordination: all modules cold/disjoint from active lanes.

This is the consolidated record for the seven **grouped** run-proof PRs. Each PR was kept lean (only
`tests/stdlib/**` + `scripts/*_gate.sh`, no shared-file edits) so it merged cleanly against a hyperactive
`main`; this index (plus the batched offload-log entry in `.claude/llm_offload_log.md`) is the housekeeping
record. Every module is imported and run as native ELF (default Madaros unless noted), asserting outputs
against known / first-principles values; each PR ships a combined gate.

| PR | Modules | Gate | Known-value anchors |
|---|---|---|---|
| #941 | special::beta, math::hyperbolic, math::rational | `math_verticals_batch_gate.sh` | beta(2,3)=1/12, β(½,½)=π; cosh²−sinh²=1; exact fractions |
| #948 | encoding::hex, math::dd64, math::combinatorics_perm | `verticals_batch2_gate.sh` | hex round-trip; dd64 recovers 1e-20; n! + next_permutation |
| #950 | autodiff::epistemic_dual, collections::vec, collections::stack | `verticals_batch3_gate.sh` | d/dx x²=2x; vec sum/mean/reverse; LIFO |
| #954 | collections::hashmap, core::result, math::qd128 | `verticals_batch4_gate.sh` | map semantics; Result ok/err; qd128 recovers 1e-40 (lean_single) |
| #957 | viz::colormap, geo::pure::types, queue::pure::types | `verticals_batch5_gate.sh` | viridis knots=matplotlib; Euclidean 3-4-5; FIFO |
| #963 | data::json, data::csv, math::approx | `verticals_batch6_gate.sh` | RFC-8259 JSON; CSV; Newton sqrt, Taylor sin |
| #967 | epistemic::covariance, cybernetic::variety, audio::pure::types | `verticals_batch7_gate.sh` | corr coeff 0.5; Ashby log₂ variety; audio buffer (lean_single) |

## Method (playbook)
Discover importable module → probe for cross-module-safe API (scalar / by-reference / small-struct;
avoid `[T;N]`-by-value #913, HOF, large-struct SRET) → run-proof vs known values → combined gate →
math-review (xAI/Grok, logged) → lean PR → merge. `check:OK` never trusted alone — every claim is
compile-and-run.

## Also delivered this campaign (individual PRs, with their own specs/plans on main)
GUM #860, units #873, linalg::matnm #892, negative-display fix #900, prob::distributions #902,
stats::validation #909, signal::fft #917, integrate::epistemic_ode #924, special::erf #926,
signal::filter #934, special::gamma #938. Plus compiler-blocker dispatches #859.

**Total: 32 stdlib verticals run-proven, no compiler changes.**
