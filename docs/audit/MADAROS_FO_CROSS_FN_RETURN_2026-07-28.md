<!-- docs:meta
topic_id: repo.docs.audit.madaros-fo-cross-fn-return-2026-07-28
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-fo-cross-fn-return-2026-07-28
-->

# Madaros FO: cross-function variance lost on `return` / `let`

**Date:** 2026-07-28  
**Status:** CLOSED FIXED 2026-07-28  
**Witness:** `tests/run-pass/gum_cross_function.sio`  
**Gate:** `scripts/ci/madaros_gum_cross_function_gate.sh`

## Symptom

| Engine | var(sum) | var(scaled) |
|---|---:|---:|
| lean_single | 5.0 | 16.0 |
| Madaros (pre-fix) | **0** | **0** |

Helpers:

```sounio
fn add_values(x: f64, y: f64) -> f64 with Mut {
    let result = x + y
    return result
}
```

## Root cause

`fo_classify_block_transfer` aborted on any `StmtLet`, and never unwrapped
`ExprReturn`. Pure-fn FO transfers for `let r = x+y; return r` were never
registered, so call-site FO stayed zero.

## Fix

1. Remember last pure `let` name + RHS.
2. On trailing `return <ident>` matching that let, classify the RHS binary.
3. Unwrap bare `ExprReturn` in `fo_classify_expr_transfer`.

## Acceptance

```text
var(sum)=5.000000
var(scaled)=16.000000
PASS
```

Madaros stdout matches lean_single on this witness.


## Evidence (2026-07-28)

Pre-fix `bin/madaros-linux-x86_64` on witness:

```text
var(sum)=0.000000
var(scaled)=0.000000
FAIL
```

Post-fix `artifacts/self-hosted/madaros-fo-cross`:

```text
var(a)=4.000000
var(b)=1.000000
var(sum)=5.000000
var(scaled)=16.000000
PASS
```

Gate: `MADAROS_RAW_BIN=.../madaros-fo-cross bash scripts/ci/madaros_gum_cross_function_gate.sh` → `[madaros-gum-cross] PASS: FIXED`.

lean_single control: PASS with same numbers.
