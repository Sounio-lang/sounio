<!-- docs:meta
topic_id: repo.docs.audit.sedenion-ker-lw-basis-2026-08-24
authority: repo_only
audience: users
last_validated: 2026-08-24
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sedenion-ker-lw-basis-2026-08-24
-->

# Named 4-space of ker L_w

```text
Semantic-Lane-ID: sedenion-ker-lw-basis-20260824
Owner: grok-cli2
Concept-IDs: SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: a named spanning set of ker L_w is a measurement,
  not a classification of zero-divisors
Transformation: scan e_i ± e_j under L_w; pin four pair-type hits;
  generator 0 is canonical z; span rank 4
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: Sounio computes L_w(h_k)=0 for
  h = {e3+e10, e2−e11, e4−e13, e5+e12}; span rank 4; L_{e1} nsq 2;
  pair scan has exactly four hits
Claims-Forbidden: new physics; unique orthonormal basis; Moreno;
  complementary in R^16; ZD solves g-2; Madaros is
  fixed-point-verified
Assumptions: Convention X; GE tolerance 1e-9; pair-type scan is not
  an enumeration of all of ℝ¹⁶
Write-Set: stdlib/algebra/sedenion_kernel.sio,
  examples/physics/sedenion_ker_lw_basis.sio,
  tests/run-pass/sedenion_ker_lw_basis.sio, this file
Read-Set: docs/audit/SEDENION_KER_BASIS_2026-08-23.md,
  docs/audit/SEDENION_KER_LW_2026-08-23.md
Positive-Witness: souc run prints span rank 4, all L_w nsq 0,
  H0 equals z, L_e1 nsq 2
Negative-Witness: g-2 is not this lane
Acceptance-Gate: tests/run-pass/sedenion_ker_lw_basis.sio exit 0
  under Madaros
Integration-Target: origin/main
Authoritative-Only-If: the nsqs and rank are produced by Madaros
```

## What this is

#2096 named `ker L_z`. #2100 showed `ker L_z ∩ ker L_w = {0}`.
This names `ker L_w`. The scan of all 240 vectors `e_i ± e_j`
(`i < j`, both signs) finds **exactly four** hits. That family is
exhaustive; ℝ¹⁶ is not. Span GE rank **4** at tolerance 1e-9.
Generator 0 is canonical `z`. Computationally witnessed.

| k | generator |
|---|---|
| 0 | `e₃ + e₁₀` (`z`) |
| 1 | `e₂ − e₁₁` |
| 2 | `e₄ − e₁₃` |
| 3 | `e₅ + e₁₂` |

The L_z basis used `e₄+e₁₃` and `e₅−e₁₂` — the **other sign** of
the same pairs. Pair-type basis, not uniqueness in ℝ¹⁶.
`L_z(h_k)` is nonzero on each generator (consistent with
`ker L_z ∩ ker L_w = {0}` from #2100).

## Receipts (2026-08-24)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B).

LLM-offload math-review (`/tmp/llm-offload-gbAIi1`):

- xAI grok-4.3: `[OVERREACH]` on treating numerical L_w=0 / rank 4 as
  algebraic — text now says computationally witnessed, GE 1e-9;
  `[OVERREACH]` on "exactly four" — scoped to the 240 pair-type
  vectors; `[TIGHTENABLE]` intersection — example/test require
  `L_z(h_k)` nonzero.
- Z.AI: weekly quota exhausted until 2026-08-25 06:34:36.
- Outcome: **PASS_SINGLE_PROVIDER_DEGRADED**.
