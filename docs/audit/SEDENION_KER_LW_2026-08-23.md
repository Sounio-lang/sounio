<!-- docs:meta
topic_id: repo.docs.audit.sedenion-ker-lw-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sedenion-ker-lw-2026-08-23
-->

# ker L_z ∩ ker L_w is {0}

```text
Semantic-Lane-ID: sedenion-ker-lw-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: two 4-dimensional kernels of a pair are not assumed
  equal; intersection is measured
Transformation: restrict L_w to the named ker L_z basis; rank 4
  implies intersection dim 0
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: Sounio computes L_w(g_k) nsq in {4,8} on the four
  generators (none zero); L_w(z)=0; rank(L_w on ker L_z)=4;
  dim(ker L_z ∩ ker L_w)=0
Claims-Forbidden: new physics; complementary in R^16; every ZD pair
  has disjoint kernels; Moreno; ZD solves g-2; Madaros is
  fixed-point-verified
Assumptions: Convention X; named basis from #2096; GE tolerance 1e-9
Write-Set: stdlib/algebra/sedenion_annihilator.sio,
  examples/physics/sedenion_ker_lw.sio,
  tests/run-pass/sedenion_ker_lw.sio, this file
Read-Set: docs/audit/SEDENION_KER_INTERSECT_2026-08-23.md
Positive-Witness: souc run prints INTERSECT_LZ_LW_DIM 0,
  LW_ON_LZ_RANK 4, LW_ON_Z 0, WW_NSQ 4
Negative-Witness: g-2 is not this lane
Acceptance-Gate: tests/run-pass/sedenion_ker_lw.sio exit 0 under Madaros
Integration-Target: origin/main
Authoritative-Only-If: the nsqs and ranks are produced by Madaros
```

## What this is

#2097 showed `ker L_z = ker R_z` for canonical `z`. The same question
for `w` is a different 4-space: `w*w` has nsq 4, so `w ∉ ker L_w`,
while `w ∈ ker L_z`. The spaces cannot coincide.

The remaining number is the intersection. Restriction of `L_w` to the
named `ker L_z` basis has GE rank **4** (the four images are linearly
independent at tolerance 1e-9, not merely each nonzero), so
`dim(ker L_z ∩ ker L_w) = 0`. Computationally witnessed, not a Lean
identity.

`z` itself lives in `ker L_w` (and `ker R_w`), not in `ker L_z`.

## Receipts (2026-08-23)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B).

| k | `L_w(g_k)` nsq |
|---|---:|
| 0 (`w`) | **4** |
| 1 | **8** |
| 2 | **8** |
| 3 | **4** |

LLM-offload math-review (`/tmp/llm-offload-AF1P4G`):

- xAI grok-4.3: three `[OK]`; `[TIGHTENABLE]` that nonzero nsqs are not
  full rank — rank 4 is GE of the four images; `[OVERREACH]` if treated
  as a Lean identity — text now says computationally witnessed.
- Z.AI: weekly quota exhausted until 2026-08-25 06:34:36.
- Outcome: **PASS_SINGLE_PROVIDER_DEGRADED**.
