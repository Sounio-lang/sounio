<!-- docs:meta
topic_id: repo.docs.audit.sedenion-ker-lz-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sedenion-ker-lz-2026-08-23
-->

# Sedenion dim ker L_z — rank-nullity of the canonical pair

```text
Semantic-Lane-ID: sedenion-ker-lz-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: a kernel dimension is a linear-algebra measurement
  of L_a, not a physical ontology and not a classification of
  zero-divisors
Transformation: build the 16×16 matrix of L_a via sed_mul; rank by
  Gaussian elimination with partial pivoting; dim ker = 16 − rank;
  report basis-vanish separately
Types-Changed: none
Effects-Changed: none (sed_mul remains Mut-only)
IR-Changed: none
Claims-Introduced: Sounio computes rank(L_z)=12 and dim ker L_z=4 on
  the canonical pair; the same for L_w, R_z, R_w; embed(e1) and e1
  have rank 16; no standard-basis vector is in ker L_z
Claims-Forbidden: new physics; Sounio proved Moreno; ZD solves g-2;
  dim ker is always half the dimension; Madaros is fixed-point-verified;
  this is the CDElement tower
Assumptions: Convention X via algebra::sedenion; GE tolerance 1e-9;
  the CD-tower L4 row (ker 4/16) is a comparison, not this oracle
Write-Set: stdlib/algebra/sedenion_kernel.sio,
  examples/physics/sedenion_ker_lz.sio,
  tests/run-pass/sedenion_ker_lz.sio, this file
Read-Set: stdlib/algebra/sedenion_action.sio,
  examples/sedeniontrip_projective_measurement.sio
Positive-Witness: souc run prints KER_LZ_DIM 4, KER_LZ_RANK 12,
  KER_LZ_BASIS_VANISH 0, KER_EMBED_E1_DIM 0
Negative-Witness: counting vanishing L_z(e_i) is not dim ker;
  g-2 is not this lane
Acceptance-Gate: tests/run-pass/sedenion_ker_lz.sio exit 0 under Madaros
Integration-Target: origin/main
Authoritative-Only-If: the ranks are produced by Madaros running Sounio
```

## What this is

#2086 showed one vector in the kernel (`z*w=0`). This lane measures
the dimension. New module: does not touch `sedenion_action.sio`
(#2093 holds that file).

## Receipts (2026-08-23)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B).

| map | rank | dim ker | basis vanish |
|---|---:|---:|---:|
| `L_z` | **12** | **4** | **0** |
| `R_z` | **12** | **4** | — |
| `L_w` | **12** | **4** | — |
| `L_{e1}` | **16** | **0** | **0** |
| `L_{embed(e1)}` | **16** | **0** | — |

The kernel is 4-dimensional and contains **no** coordinate vector
`e_i`. Counting `z*e_i=0` would have reported 0 and lied. Other
sparse generators (sums of two basis elements, etc.) are not claimed.

This **agrees** with the L4 row of the CD-tower projective ladder
(`ker 4 / 16 = 25%`, `e3+e10`). That ladder uses
`algebra::cayley_dickson`. This lane uses `sed_mul`. Same integer.

This **disagrees** with the slogan that a sedenion zero-divisor has
an 8-dimensional annihilator. The slogan is not this measurement.

## Claims-forbidden

New physics; Moreno classification; dim ker always n/2; ZD solves
g-2; Madaros is fixed-point-verified.

LLM-offload math-review (`/tmp/llm-offload-dOO3O6`):

- xAI grok-4.3: five `[OK]`; one `[TIGHTENABLE]` on "basis vanish"
  over-reading as "no sparse kernel basis" — text now says no
  coordinate vector, and does not claim other sparse generators.
- Z.AI: weekly quota exhausted until 2026-08-25 06:34:36.
- Outcome: **PASS_SINGLE_PROVIDER_DEGRADED**.
