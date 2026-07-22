<!-- docs:meta
topic_id: repo.docs.audit.madaros-wave13e-global-list-args-2026-07-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-wave13e-global-list-args-2026-07-21
-->

# Madaros Wave13e — pure paramful call fold in global element-list init

**Date:** 2026-07-21 / measured 2026-07-22  
**Role:** Wave13 Agent E (implementer, resume)  
**Branch:** `fix/madaros-wave13e-global-list-args`  
**Tip measured:** `origin/main` @ `c48dbe1d2` (post-#1396–#1402, #1400 bare Ident)  
**Engine:** default `bin/souc` → Madaros (rebuilt `artifacts/self-hosted/madaros-w13e`)

## Mission

Ship the **second residual** after Wave12 multi-stmt pure (PR #1387), non-overlapping with:

| Agent | Ownership |
|-------|-----------|
| A | into-acc dep lower (#1402 merged — do not fight) |
| B | specialized-list DCE |
| C | showcase (require cd_exact) |
| D | bare cross-mod f64 Ident (#1400 merged — do not fight) |

## Measurement (stock tip, before fix)

```sounio
fn add2(a: i64, b: i64) -> i64 { a + b }
var A: [i64; 3] = [add2(10, 20), 1, 2]
// stock: 0 0 0  (fail-closed entire array)
// expect: 30 1 2
```

Wave10 folded zero-arg pure (`ten()`). Wave12 folded multi-stmt zero-arg
(`let x = 10; x`). **Paramful pure** stayed residual fail-closed BSS zero.

Stock tip re-measured 2026-07-22 on `c48dbe1d2`: still `0 0 0` / `0 0`.

## Root cause

`items_eval_global_init_word` rejected any `ExprCall` with args.  
`items_maybe_record_pure_fn_const` only accepted zero-param functions and
pre-folded them into `GLOBAL_VAR_INIT`.

## Fix (`self-hosted/parser/items.sio`)

1. **Call-site fold for pure paramful SINGLE-STMT bodies** via multi-word
   descriptors under the function name in `GLOBAL_VAR_INIT_*`:
   - **KIND 1** (`MAGIC = -900001`): body is `a OP b` (binary of two Idents) →
     fold `ARG_V0 OP ARG_V1` at the call site.
   - **KIND 2** (`MAGIC = -900002`): body is bare Ident 1-param → return
     `ARG_V0`.
2. **Record** at pure-fn definition (`items_maybe_record_pure_fn_const` →
   `pure_fn_reg_record`) for ≤4 params, effect-free.
3. **Reset** helpers: `items_reset_pure_fn_reg()` paired with
   `ast_reset_global_var_inits` in `parse_items_preloaded`.
4. Effectful / non-const / >4-param / multi-stmt paramful callees remain
   fail-closed zeros (no left-shift of remaining const words).

### Multi-stmt paramful residual (honest)

A KIND 3 path that stored `*mut StmtList` body pointers in
`PURE_FN_REG_BODY` **failed to fold** multi-stmt bodies and, worse,
**poisoned same-module KIND 1/2** when a multi-stmt pure paramful was merely
*defined* (even unused):

| Program shape | Result |
|---|---|
| `add2` alone | `30 1 2` GREEN |
| `add2` + multi-stmt `f` defined, only `add2` used | `0 0 0` RED (poison) |
| multi-stmt `f` alone | `0 0` RED |

KIND 3 registration is therefore a **no-op** (residual fail-closed). Do not
re-enable body-pointer tables without a non-pointer, seed-safe body registry
and a poison regression case.

## Gates (measured 2026-07-22)

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE
ulimit -s unlimited 2>/dev/null || true

./bin/souc run tests/run-pass/global_array_element_list_call_args.sio
# 30 1 2
# 21 9 5

bash scripts/dev/madaros_global_array_init_gate.sh
# GLOBAL_ARRAY_INIT_GATE_OK  (includes call_list_args + multistmt residual)

bash scripts/madaros_dual_import_gate.sh
# MADAROS_DUAL_IMPORT_GATE_OK
```

## Claims

- Pure paramful **single-stmt** callees with const args fold at parse time into
  BSS words for global element-list init (`add2(10,20)` → 30, `id1(9)` → 9).
- Zero-arg pure (Wave10/12) and fail-closed effectful residual still hold.
- No left-shift miscompile of remaining const words.
- Multi-stmt pure paramful in global element-list init remains residual
  fail-closed (documented residual case in the gate).

## Explicit non-claims

- bare cross-module `use m::{CONST}` Ident from main (Agent D / #1400)
- into-acc / specialized-list DCE (Agents A/B / #1402)
- multi-stmt pure paramful call fold (`{ let x = a + 1; x }`)
- effectful callees in global lists (still fail-closed zeros by design)
- >4-parameter pure callees
- runtime (non-const) global element-list init

## Rebuild

```bash
# build_modular_madaros.sh acquires souc-build-lock internally (do not nest)
export SOUNIO_BUILD_LOCK=/tmp/sounio-w13e-global-list-args.lock
bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros-w13e
cp -f artifacts/self-hosted/madaros-w13e artifacts/self-hosted/madaros
cp -f artifacts/self-hosted/madaros-w13e bin/madaros-linux-x86_64
```
