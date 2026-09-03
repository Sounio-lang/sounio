<!-- docs:meta
topic_id: repo.docs.audit.madaros-bare-float-arith-return-2026-07-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-bare-float-arith-return-2026-07-21
-->

# Madaros — bare float intrinsic results break subsequent f64 arithmetic (Wave14e)

**Date:** 2026-07-21
**Toolchain tip measure:** stock `bin/madaros-linux-x86_64` on `origin/main` post-#1392
(`cd_exact` green; Wave12 tip-green full pass)
**Status:** **FIXED** (source) — empty float-builtin stubs advertise `returns_float=1`
and call sites receive `IR_FLOAT_REG_MARKER_FLAG`.
**Severity:** high for science path — any bare `exp`/`cos`/`sin`/`sqrt` used in
arithmetic (not merely printed) silently corrupts.

## Residual not owned by Wave13/14 sibling lanes

| Owned elsewhere | This residual |
|---|---|
| Wave13 D bare crossmod f64 **Ident** of imported global | bare **call** of float intrinsic → f64 arith |
| Wave13 A into-acc, B spec DCE, C showcase, E list args | orthogonal |
| Wave14 B #913 f64 array by-value, C Root-2, D thinlink | orthogonal |
| exp_cos **import collision** (#1287 / Wave6 C) | user bodies keep IR; bare intrinsic still wrong for arith |

## Symptom (measured stock tip)

```
cos(0.0)  → f64_to_bits = 0x3FF0000000000000 (1.0)   OK
print_f64(exp(1.0)) → 2.718281                        OK
cos(0.0) * 1000.0 → bits ≈ 4.607e21                   WRONG (want 1000.0)
(exp(1.0) * 1000.0) as i64 → INT64_MIN                WRONG
```

`cvtsi2sd` of the IEEE bit pattern of `1.0` (`4607182418800017408`) × 1000
reproduces the observed `4.607e21` exactly.

## Root cause

1. Bare float names lower to `IrCall` against injected empty stubs
   (`ir_module_ensure_builtin_call_targets` → `ir_empty_function()`,
   `returns_float=0`).
2. Call sites only stamp `IR_FLOAT_REG_MARKER_FLAG` when
   `functions[fn_id].returns_float == 1`.
3. Native core (`nc_emit_core_binop` / `nc_compute_float_types`) marks the
   call result **INT**.
4. Float binop path then does `cvtsi2sd` on the bit pattern instead of
   `movq xmm, rax`.

`print_f64` / `f64_to_bits` stay correct because they use the same integer-bits
ABI as the builtins (identity / re-interpret), never SSE mul on a mis-typed local.

## Fix

- `native_v2_builtin_returns_float(id)` — true for sqrt/exp/log/sin/cos/bits_to_f64
- `ir_module_ensure_builtin_call_targets` sets `returns_float=1` on empty
  float-intrinsic stubs and stamps the call-site float marker
- **Single-file path** (`loaded == 1` / specialized collapse) never entered
  multi-mod `finalize_merged_calls`; bare `cos` lived only on that path. The
  ensure pass is now invoked after single-module resolve as well.

## Gate

`scripts/madaros_bare_float_arith_gate.sh`
→ `MADAROS_BARE_FLOAT_ARITH_GATE_OK`

Witness: `tests/compiler/bare_float_arith/main.sio`

## Claim boundary

**Claims:** bare `cos`/`sin`/`sqrt`/`exp(1)` results participate in subsequent
f64 mul/add under default Madaros; scaled integer sentinels stable.

**Does not claim:** `exp(0)` series soft spot (gate already scopes bare_exp to
`exp(1)`); bare cross-module `use m::{CONST}` Ident; full D1 ungated cast matrix;
language-level `Knowledge<T>` generic import; prebuilt lag outside this rebuild.

## AI disclosure

Localisation and fix by AI agent under human direction. GAIDeT-ICMJE 2025.
