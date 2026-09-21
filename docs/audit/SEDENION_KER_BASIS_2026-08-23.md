<!-- docs:meta
topic_id: repo.docs.audit.sedenion-ker-basis-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sedenion-ker-basis-2026-08-23
-->

# Named 4-space of ker L_z

```text
Semantic-Lane-ID: sedenion-ker-basis-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: a named spanning set is a measurement of L_z, not a
  classification of zero-divisors and not Moreno
Transformation: scan e_i ± e_j; pin the four pair-type hits as
  generators; rank of the 16×4 matrix is 4; generator 0 is canonical w
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: Sounio computes L_z(g_k)=0 for
  g = {e6−e15, e4+e13, e5−e12, e7+e14}; span rank 4; L_{e1} nsq 2;
  the e_i±e_j scan has exactly four hits
Claims-Forbidden: new physics; unique orthonormal kernel basis;
  Sounio proved Moreno; ZD solves g-2; Madaros is fixed-point-verified
Assumptions: Convention X; GE tolerance 1e-9; pair-type scan is not
  an enumeration of all of ℝ¹⁶
Write-Set: stdlib/algebra/sedenion_kernel.sio,
  examples/physics/sedenion_ker_basis.sio,
  tests/run-pass/sedenion_ker_basis.sio, this file
Read-Set: docs/audit/SEDENION_KER_LZ_2026-08-23.md
Positive-Witness: souc run prints span rank 4, all L_z nsq 0,
  G0 equals w, L_e1 nsq 2
Negative-Witness: coordinate vectors are not in the kernel; g-2 is
  not this lane
Acceptance-Gate: tests/run-pass/sedenion_ker_basis.sio exit 0 under Madaros
Integration-Target: origin/main
Authoritative-Only-If: the nsqs and rank are produced by Madaros
```

## What this is

#2095 measured `dim ker L_z = 4` and `basis-vanish = 0`. This names
the 4-space. A scan of `e_i ± e_j` (`i < j`) finds **exactly four**
hits. They are independent (span rank 4). Generator 0 is canonical `w`.

| k | generator | notes |
|---|---|---|
| 0 | `e₆ − e₁₅` | canonical `w` |
| 1 | `e₄ + e₁₃` | |
| 2 | `e₅ − e₁₂` | |
| 3 | `e₇ + e₁₄` | |

Each has `‖g‖² = 2`, `L_z(g) = 0`, `L_{e1}(g)` nsq **2** (the unit
does not annihilate). **Exactly four** is the `e_i ± e_j` (`i < j`)
scan, not an enumeration of ℝ¹⁶. This is **a** basis of pair-type
vectors, not uniqueness in ℝ¹⁶.

## Receipts (2026-08-23)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B).

Scan: 4 hits, greedy keep 4, final rank 4.

LLM-offload math-review (`/tmp/llm-offload-zA3jCY`):

- xAI grok-4.3: four `[OK]`; one `[TIGHTENABLE]` on "exactly four hits"
  — already scoped to the pair scan; sentence now says so explicitly.
- Z.AI: weekly quota exhausted until 2026-08-25 06:34:36.
- Outcome: **PASS_SINGLE_PROVIDER_DEGRADED**.
