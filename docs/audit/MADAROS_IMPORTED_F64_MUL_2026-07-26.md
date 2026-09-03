<!-- docs:meta
topic_id: repo.docs.audit.madaros-imported-f64-mul-2026-07-26
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-imported-f64-mul-2026-07-26
-->

# Madaros multimodule: imported f64 prints OK, raw f64*f64 wrong

**Date:** 2026-07-26  
**Status:** **CLOSED** — `lower.sio` marks all-f64 TypeTuple as `returns_float=2`  
  and float-element FieldGet for tuple unpack (`.0`/`.1`).  
  Verified: `MADAROS_IMPORTED_F64_MUL_FIXED` (ga*ga=0.25, gv*gv≈0.001412).  
**Witness:** `tests/multimodule/madaros_imported_f64_mul_{leaf,main}.sio`  
**Gate:** `scripts/ci/madaros_imported_f64_mul_gate.sh`

## Measured (refined)

| Expression | lean | Madaros multimodule |
|---|---|---|
| Scalar import `ret_neg_half()` then `a*a` | 0.25 | **0.25 OK** |
| Tuple import `let (gv, ga) = ret_pair()` then `ga*ga` | 0.25 | **0 WRONG** |
| Tuple `gv*gv` | ~0.0014 | **~0.69 WRONG** |
| `0.0 -` scalar import | 0.5 | 0.5 OK |

So the residual is **not** every imported f64 — it is **tuple-unpack of `(f64, f64)`** from an imported callee. Vertex `nc_coupling_*` returns tuples → amplitude broke.

## Likely fix surface (Claude-1 / native)

Tuple / multi-value return from imported modules: each element used as f64 must be float-typed at the call site (destructure / field-of-tuple).

Candidates:
- `self-hosted/ir/lower.sio` — lower of `let (a,b) = f()` for f64 elements
- `self-hosted/native/codegen_x86_linux.sio` — multi-return / pair ABI float marks
- `module_frontend` finalize of imported multi-return calls

## Product mitigation

- #1514: first amp workaround via `ep_square`
- #1516: `nonunitary_amp` avoids tuple couplings; uses `neutral_current_gv_ep` + exact `g_A²=0.25`
- Follow-up: `vertex` exports scalar `neutral_current_gv` / `neutral_current_ga` and
  `nc_gv_*` / `nc_ga_*`; `amplitude` uses them (no imported tuple unpack)

## True lower fix (Claude)

Recipe: `docs/audit/MADAROS_TUPLE_F64_FLOAT_FIX_RECIPE_2026-07-26.md`  
3-site patch in `self-hosted/ir/lower.sio` (returns_float=2 for all-f64 tuples + FieldGet float mark).

## Acceptance

```text
MADAROS_IMPORTED_F64_MUL_FIXED
# a*a=0.25, b*b=0.25, a*b=-0.25, ga*ga=0.25
```
