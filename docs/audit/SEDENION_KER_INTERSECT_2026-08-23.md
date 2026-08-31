<!-- docs:meta
topic_id: repo.docs.audit.sedenion-ker-intersect-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sedenion-ker-intersect-2026-08-23
-->

# Two-sided annihilator of canonical z — intersection dim 4

```text
Semantic-Lane-ID: sedenion-ker-intersect-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: equal left/right kernels for this z is a
  measurement, not a classification of zero-divisors
Transformation: map the named ker L_z basis through R_z; rank of
  those four images is 0; intersection dim = 4 − 0 = 4
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: Sounio computes v*z=0 for each of the four
  pair-type generators; R-image rank 0; dim(ker L_z ∩ ker R_z)=4;
  dim ker R_z=4, so the two kernels coincide for this z
Claims-Forbidden: new physics; every sedenion ZD has ker L = ker R;
  Moreno; ZD solves g-2; Madaros is fixed-point-verified
Assumptions: Convention X; named basis from #2096; GE tolerance 1e-9
Write-Set: stdlib/algebra/sedenion_annihilator.sio,
  examples/physics/sedenion_ker_intersect.sio,
  tests/run-pass/sedenion_ker_intersect.sio, this file
Read-Set: stdlib/algebra/sedenion_kernel.sio,
  docs/audit/SEDENION_KER_BASIS_2026-08-23.md
Positive-Witness: souc run prints INTERSECT_DIM 4, R_IMAGE_RANK 0,
  each K L 0 R 0 RE1 2
Negative-Witness: L_{e1} / R_{e1} do not annihilate; g-2 is not
  this lane
Acceptance-Gate: tests/run-pass/sedenion_ker_intersect.sio exit 0
  under Madaros
Integration-Target: origin/main after #2096
Authoritative-Only-If: the nsqs and ranks are produced by Madaros
```

## What this is

#2093 showed the *pair* `(z,w)` is two-sided. #2095 showed
`dim ker L_z = dim ker R_z = 4`. #2096 named a basis of `ker L_z`.
Those facts do not decide whether the two 4-spaces are the same.

A vector `Σ a_k g_k` in `ker L_z` lies in `ker R_z` iff
`Σ a_k R_z(g_k) = 0`. Each column `R_z(g_k)` has nsq **0** (the
16×4 matrix is the zero matrix, not merely rank-deficient). Rank
**0**. Intersection dimension is **4**. Combined with
`dim ker R_z = 4`, `ker L_z = ker R_z` **for this z**.

Not a statement about every zero-divisor.

## Receipts (2026-08-23)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B).

LLM-offload math-review (`/tmp/llm-offload-L7pkqp`):

- xAI grok-4.3: three `[OK]`; one `[TIGHTENABLE]` on reporting only rank 0
  — text now states each column nsq is 0 (zero matrix).
- Z.AI: weekly quota exhausted until 2026-08-25 06:34:36.
- Outcome: **PASS_SINGLE_PROVIDER_DEGRADED**.
