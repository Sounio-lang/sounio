<!-- docs:meta
topic_id: repo.docs.audit.madaros-wave15a-global-list-multistmt-paramful-2026-07-22
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-wave15a-global-list-multistmt-paramful-2026-07-22
-->

# Madaros Wave15a — multi-stmt pure paramful fold in global element-list init

**Date:** 2026-07-22  
**Role:** Wave15 Agent A (XAI ousadia)  
**Branch:** `fix/madaros-wave15a-multistmt-paramful`  
**Tip measured:** `origin/main` @ `3e7ed9f52` (post-#1405 Wave13e)  
**Engine:** default `bin/souc` → Madaros (rebuilt `artifacts/self-hosted/madaros-w15a`)

## Mission

Close the **honest residual** left by #1405 Wave13e: multi-stmt pure paramful
callees in global element-list init still fail-closed (BSS zeros). KIND 3 was
intentionally a no-op because a body-pointer registry poisoned same-module
KIND 1/2.

## Measurement (stock tip, before fix)

```sounio
fn f(a: i64) -> i64 {
  let x = a + 1
  x
}
var A: [i64; 2] = [f(9), 2]
// stock tip: 0 0  (fail-closed entire array)
// expect:    10 2
```

Measured on stock `3e7ed9f52` with committed Madaros: `0 0`.  
KIND 1 baseline `add2(10,20)` → `30 1 2` stayed green.

## Root cause (Wave13e residual design)

`pure_fn_reg_record` classified multi-stmt / general pure bodies as KIND 3 but
**did not register** them. The prior attempt stored `*mut StmtList` in a fixed
`[*mut StmtList; 128]` table plus magic `-900003` multi-word descriptors; that
path failed to fold **and** poisoned same-module KIND 1/2 when a multi-stmt
paramful was merely *defined*:

| Program shape | Result |
|---|---|
| `add2` alone | `30 1 2` GREEN |
| `add2` + multi-stmt `f` defined, only `add2` used | `0 0 0` RED (poison) |
| multi-stmt `f` alone | `0 0` RED |

## Fix (`self-hosted/parser/items.sio`)

1. **KIND 3 registration** for pure return-chain bodies (zero-or-more immutable
   `let`, final `StmtExpr`), ≤4 params, effect-free.
2. **Body pointer as i64** stored inside the proven `GLOBAL_VAR_INIT_*`
   multi-word table (magic `-900003`: `magic, pc, body_bits, ph0..ph3`).
   Same seed-safe pattern as `check/specializer.sio` `SPEC_*_PTRS: [i64; N]`.
3. **Do not write** a fixed `[*mut StmtList; N]` body table (removed the
   `PURE_FN_REG_BODY` array entirely).
4. **Call-site walk** of the StmtList (lets bind into `PURE_FN_LOCAL_*`, final
   expr folds with params bound via `PURE_FN_ACTIVE_SLOT` / `ARG_V*`).
5. Structural accept via `pure_fn_is_pure_return_chain_shape` before register.

## Gates (measured 2026-07-22)

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE
ulimit -s unlimited 2>/dev/null || true

./bin/souc run tests/run-pass/global_array_element_list_call_args_multistmt.sio
# 30 1 2
# 10 2
# 15 1 2

bash scripts/dev/madaros_global_array_init_gate.sh
# GLOBAL_ARRAY_INIT_GATE_OK
#   (call_list_args, call_list_args_multistmt, kind3_no_poison,
#    call_list_args_multistmt_chain)

bash scripts/madaros_dual_import_gate.sh
# MADAROS_DUAL_IMPORT_GATE_OK
```

## Claims

- Pure paramful **multi-stmt pure return chains** with const args fold at parse
  time into BSS words for global element-list init when expression shapes are
  Wave12-proven (`param + lit`, `local + lit`, dependent lets of those):
  - `f(9)` with `{ let x = a + 1; x }` → `10`
  - `g(9)` with `{ let x = a + 1; let y = x + 5; y }` → `15`
- KIND 1/2 single-stmt paramful still green (`add2(10,20)` → 30).
- KIND 3 registration **does not poison** same-module KIND 1/2 (gate
  `call_list_args_kind3_no_poison`).
- Zero-arg pure (Wave10/12) and fail-closed effectful residual still hold.
- No left-shift miscompile of remaining const words.

## Explicit non-claims

- effectful multi-stmt callees (still fail-closed zeros by design)
- arbitrary recursion / mutual recursion in pure global-init fold
- >4-parameter pure callees
- runtime (non-const) global element-list init
- **Ident+Ident binary pure fold** (`a + b`, `x + b`, etc.): pre-existing
  Wave12 residual. Both sides evaluate as the **RHS** Ident under
  `items_eval_global_init_word` pure-local lookup. Reproduced on stock tip
  Madaros for zero-arg `let a = 9; let b = 5; a + b` → `10` (not `14`).
  KIND 1 single-stmt `{ a + b }` still works (positional ARG_V0/ARG_V1, no
  Ident re-resolution). Out of Wave15a claim boundary.

## Rebuild

```bash
# build_modular_madaros.sh acquires souc-build-lock internally (do not nest)
export SOUNIO_BUILD_LOCK=/tmp/sounio-w15a-multistmt-paramful.lock
bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros-w15a
cp -f artifacts/self-hosted/madaros-w15a artifacts/self-hosted/madaros
cp -f artifacts/self-hosted/madaros-w15a bin/madaros-linux-x86_64
```
