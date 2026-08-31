<!-- docs:meta
topic_id: repo.docs.audit.sedenion-zd-twosided-moufang-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sedenion-zd-twosided-moufang-2026-08-23
-->

# Sedenion two-sided ZD pair and Moufang parenthesization

```text
Semantic-Lane-ID: sedenion-zd-moufang-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: parenthesization remains explicit; a two-sided zero
  divisor pair is not a physical ontology; Moufang is an algebraic
  identity of O, not a new theorem on S
Transformation: expose R_a(v)=v*a beside L_a; pin z*w=w*z=0 with
  sequential survival on both sides; name the three Moufang
  parenthesizations as action discrepancies with a witness that is
  not the unit-in-x Artin reduction
Types-Changed: none
Effects-Changed: none (sed_mul remains Mut-only; no ZD effect)
IR-Changed: none
Claims-Introduced: Sounio computes z*w=w*z=0 and z*z, w*w nsq 4 on
  the canonical pair; left and right sequential nsq 8 on v=e1;
  Moufang (e1+e10, e1, e4) nsq 4/4/8; octonion embedding divides
  both orders and Moufang-vanishes
Claims-Forbidden: new physics; sedenion consciousness; ZD solves g-2;
  Sounio proved Moufang independently of Artin; classification of
  alternative subalgebras; dim ker L_z; Madaros is fixed-point-verified
Assumptions: Convention X; canonical pair from algebra::sedenion;
  unital alternative iff Moufang (standard, not a theorem number
  claimed here); x=1 left-Moufang is Artin
  and is not the pinned witness
Write-Set: stdlib/algebra/sedenion_action.sio,
  examples/physics/sedenion_zd_twosided.sio,
  examples/physics/sedenion_moufang.sio,
  tests/run-pass/sedenion_zd_twosided.sio,
  tests/run-pass/sedenion_moufang.sio, this file
Read-Set: stdlib/algebra/sedenion.sio,
  docs/audit/SEDENION_ZD_ACTION_2026-08-23.md,
  docs/audit/SEDENION_ARTIN_2026-08-23.md
Positive-Witness: souc run prints ZD_ZW_NSQ 0, ZD_WZ_NSQ 0,
  sequential 8 both sides, Moufang 4/4/8, oct embed 0
Negative-Witness: octonion embedding does not annihilate; g-2 is not
  this lane; ker L_z is not this lane
Acceptance-Gate: both run-pass tests exit 0 under Madaros
Integration-Target: origin/main
Authoritative-Only-If: the nsqs are produced by Madaros running Sounio
```

## What this is

A one-sided pair (`w*z ≠ 0`) was the working hypothesis. Madaros
says otherwise: both products vanish. The observable that remains is
**sequential survival on both sides**. Moufang is parenthesization,
honestly a corollary of Artin in a unital algebra.

## Two-sided pair (not dim ker)

Canonical `z = e₃+e₁₀`, `w = e₆−e₁₅`. Left action was #2086.
Right action `R_a(v)=v*a`:

| quantity | nsq |
|---|---:|
| `z*w` | **0** |
| `w*z` | **0** |
| `z*z` | **4** |
| `w*w` | **4** |
| left composed `(zw)e₁` | **0** |
| left sequential `z(w e₁)` | **8** |
| right composed `e₁(zw)` | **0** |
| right sequential `(e₁ z)w` | **8** |
| embed `e₁*e₂` and `e₂*e₁` | **1** and **1** |

Not nilpotent. Not dim ker `L_z`. The pair is two-sided as *products*;
composed annihilation is the product being zero; sequential is the
associator.

## Moufang (not a new theorem)

Identities:

- left: `a(x(ay)) = ((ax)a)y`
- middle: `(ax)(ya) = a((xy)a)`
- right: `((xa)y)a = x(a(ya))`

Witness found by Madaros scan over `a=e₁+e₁₀`, `x=e_i`, `y=e_j`.
The first left hit `x=1`, `y=e₄` is Artin and is **not** used.
Pinned triple: `a=e₁+e₁₀`, `x=e₁`, `y=e₄`.

| probe | nsq |
|---|---:|
| left | **4** |
| middle | **4** |
| right | **8** |
| octonion embed `(e₁,e₂,e₄)` all three | **0** |

## Receipts (2026-08-23)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B).

Example and test **rc=0** (`//@ requires: madaros` on the tests).

LLM-offload math-review (`/tmp/llm-offload-JR0SpP`):

- xAI grok-4.3: five `[OK]`; one `[TIGHTENABLE]` on naming Schafer without a
  theorem number — parenthetical now says "standard", no invented citation.
- Z.AI: error (quota/plan); weekly quota historically until 2026-08-25.
- Outcome: **PASS_SINGLE_PROVIDER_DEGRADED**.
