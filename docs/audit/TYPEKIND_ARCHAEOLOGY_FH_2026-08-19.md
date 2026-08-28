<!-- docs:meta
topic_id: repo.docs.audit.typekind-archaeology-fh-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.typekind-archaeology-fh-2026-08-19
-->

# TypeKind archaeology — families F + H (+ ladder control)

**Date:** 2026-08-19  
**Engine (canonical):** Madaros v0.80.0 (`./bin/souc`, default)  
**Protocol:** v3 — position is **calculated**, never asserted  
**Ladder law:** `docs/internal/concepts/MATURITY_LADDER.md` (#1943 Reserved + monotonicity)

A handwritten census rots. The deliverable is no longer a verdict column in this
file. It is:

1. two fixtures per kind under `tests/typekind/<slug>/` — `pass.sio` must run,
   `refuse.sio` must be refused with a named diagnostic; **or**
2. an empty index row (no paths) — declared **Garden**

Index: `tests/typekind/index.tsv`  
Gate (derives the table): `bash scripts/ci/typekind_archaeology_gate.sh`

```
kind | pass | refuse | expected_diagnostic | deepest_named_layer
```

**The index does not store position.** Re-run the gate.

---

## Derived table (re-run; do not hand-edit)

```bash
bash scripts/ci/typekind_archaeology_gate.sh
```

Measured on this lane (Madaros v0.80.0, tip at commit time):

| kind | derived position | pass | refuse | diag | deepest |
|---|---|---|---|---|---|
| TyI64 | Claim-ready | run 0 | check≠0 | E001 | checker |
| TyBool | Claim-ready | run 0 | check≠0 | E001 | checker |
| TyArray | Claim-ready | run 0 | check≠0 | E001 | checker |
| TyI128 | Claim-ready | run 0 | check≠0 | E001 | checker |
| TyU128 | Claim-ready | run 0 | check≠0 | E001 | checker |
| TyRawPtr | Claim-ready | run 0 | check≠0 | E001 | checker |
| TySliceMut | Claim-ready | run 0 | check≠0 | E009 | checker |
| TyF128 | **Reserved** | check≠0 + E218 | check≠0 + E218 | E218 | checker |
| TyF256 | **Reserved** | check≠0 + E218 | check≠0 + E218 | E218 | checker |
| TyVecShaped | Garden | — | — | — | checker |
| TyMatrixShaped | Garden | — | — | — | checker |
| TyBroadcastable | Garden | — | — | — | checker |
| TyDifferentiable | Garden | — | — | — | checker |
| TyGradient | Garden | — | — | — | checker |
| TyJacobian | Garden | — | — | — | checker |
| TyBigO | Garden | — | — | — | checker |
| TyAmortized | Garden | — | — | — | checker |

Gate fail modes (same pattern as XPAS):

- **XPASS** — `refuse.sio` starts passing → kind no longer discriminates
- **PASS_REGRESSION** — `pass.sio` stops running
- **WRONG_DIAGNOSTIC** — refuse fails but expected code is missing

---

## Conversion map (v1/v2 prose → v3 fixtures)

Prior handwritten verdicts told us exactly which fixtures were missing.
Nothing was thrown away.

| v1/v2 prose | what it meant | v3 artefact |
|---|---|---|
| Claim-ready (I128, U128, RawPtr, SliceMut, controls) | found construct + refuse | `pass.sio` + `refuse.sio` → derived Claim-ready |
| Claim-ready (F128/F256) under refuse-only rule | always E218 | both fixtures refuse → derived **Reserved** (#1943) |
| Hypothesis (VecShaped…Amortized) | no constructing user program | **no fixtures** → derived **Garden** (declared) |
| internal `ty_bigO` / epistemic mint notes | dig evidence, not ladder | kept below as archaeology notes |

v2 already corrected F128/F256 off Claim-ready onto Reserva/Reserved via the
monotone ladder. v3 makes that correction **executable**: both programs exist
in-repo and both fail with E218; the gate, not a human, prints Reserved.

---

## Archaeology notes (why F-family rows are empty)

These notes are **not** ladder positions. They explain why no honest
`pass.sio` was written. Inventing a program that only binds a TyNamed label
(`let v: Vec = …`) would spoof Executable without ever touching the TypeKind.

| kind | dig | source |
|---|---|---|
| VecShaped | `ty_vec_shaped` defined; **zero** callers outside `types.sio`. Surface `Vec` → TyNamed E001 | `self-hosted/check/types.sio` enum + ctor |
| MatrixShaped | `ty_matrix_shaped` only from `epistemic.sio` internal | types + epistemic |
| Broadcastable | `ty_broadcastable` never called | types |
| Differentiable / Gradient / Jacobian | ctors only from epistemic bridges; surface names TyNamed | types + epistemic |
| BigO | `ty_bigO` **is** minted inside epistemic complexity bridges; surface `BigO` is still TyNamed — **large seed**, not a user type | types + epistemic |
| Amortized | `ty_amortized` never called; sister seed of BigO | types |

When a real constructor reaches the surface, add `pass.sio` + `refuse.sio` and
delete the empty row's blank paths. The gate will move the kind without editing
this prose.

---

## Fixture contracts

**Claim-ready pair** (example `tests/typekind/i64/`):

- `pass.sio` — constructs the kind, `souc run` exits 0
- `refuse.sio` — wrong program, `souc check` fails containing `error[E001]` (or the kind's diag)

**Reserved pair** (example `tests/typekind/f128/`):

- both programs must fail with the **same** named diagnostic (E218)
- the "pass" file is the would-be correct program; under Reserved it is refused too

**Garden** (example `tests/typekind/bigo/README.md`):

- no pair; index paths empty; directory README declares absence

---

## Scope

This tree covers families **F** (shape / gradient / complexity) and **H**
(wide float/int, raw pointer, exclusive slice) plus ladder controls
TyI64/TyBool/TyArray. Other families own their own indices (family A causal,
type-census epistemic, …). Do not merge foreign kinds here without a coord claim.

---

## Counts (derived, not asserted)

| position | n | kinds |
|---|---:|---|
| Garden | 8 | VecShaped MatrixShaped Broadcastable Differentiable Gradient Jacobian BigO Amortized |
| Reserved | 2 | F128 F256 |
| Claim-ready | 7 | I128 U128 RawPtr SliceMut + TyI64 TyBool TyArray |
| Executable | 0 | (would be XPASS if refuse went green) |
| Hypothesis | 0 | (v3: no fixtures ⇒ Garden; Hypothesis returns only if a partial pair appears) |

*The gate is the table. This document explains the dig and points at the ruler.*
