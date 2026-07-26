# Madaros multimodule: imported f64 prints OK, raw f64*f64 wrong

**Date:** 2026-07-26  
**Status:** residual open (stdlib workarounds in nonunitary_amp via ep_square)  
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
- Follow-up: `nonunitary_amp` avoids tuple couplings; uses `neutral_current_gv_ep` + exact `g_A²=0.25`

## Acceptance

```text
MADAROS_IMPORTED_F64_MUL_FIXED
# a*a=0.25, b*b=0.25, a*b=-0.25, ga*ga=0.25
```
