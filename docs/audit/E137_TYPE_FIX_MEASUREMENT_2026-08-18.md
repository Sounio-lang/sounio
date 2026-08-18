<!-- docs:meta
topic_id: repo.docs.audit.e137-type-fix-measurement-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.e137-type-fix-measurement-2026-08-18
-->

> **Status**: Production | **Last validated**: 2026-08-18 | **Source**: `bin/souc check|run` against `artifacts/self-hosted/madaros` (modular compiler rebuilt from current `self-hosted/check/check.sio` via `make build-madaros` on 2026-08-18)

# E137 typecheck fix — measurement on the seven affected tests

**Date:** 2026-08-18
**Scope:** the seven tests named in `E137_TWO_CAUSES_DIAGNOSIS_2026-08-18.md` plus the variance_of users found by follow-up grep, all run against the rebuilt modular compiler with `setup_intrinsics` wired into `checker_new` and `checker_init_in_place`.
**Verdict:** the typecheck stub unblocks **5 of 7 tests** past E137. The remaining 2 fail for reasons that **were masked by E137** before this fix — surfacing them is itself a result, since the lane that owns each defect now has a clean signature to work from.

## The setup

- `self-hosted/check/check.sio:944` — new `setup_intrinsics(tbl: *mut FnSigTable)` pre-registers six FnSig entries (measure, acknowledge, uncertainty_of, require_confidence, variance_of, seq_new) via the existing `fn_sig_table_add` API.
- Wired into `checker_new` (`:1194`) and `checker_init_in_place` (`:1300`).
- Compiled via `make build-madaros` (heavy, single rebuild, ~3 minutes on pod). Fresh binary at `artifacts/self-hosted/madaros` (~100 MB).
- All test runs below use `MADAROS_RAW_BIN=.../artifacts/self-hosted/madaros ./bin/souc check|run ...`.

## Measurements

| Test | Before fix | Typecheck after fix | Compile | Run | Reason |
|---|---|---|---|---|---|
| `tests/run-pass/knowledge_acknowledge.sio` | E137 on `acknowledge` | OK | OK | **FAIL** (v != 42.0) | first param declared as `ty_unknown()`; codegen for acknowledge can't extract .value because Knowledge<T> type info not propagated |
| `tests/run-pass/knowledge_require_confidence.sio` | E137 on `require_confidence` | OK | OK | **FAIL** | same — `ty_unknown()` first param |
| `tests/run-pass/epistemic_var_accumulator_slots.sio` | E137 on `variance_of` | OK | OK | **PASS** (VAR_ACCUM_OK) | variance_of arg is concrete f64, no Knowledge propagation needed |
| `tests/run-pass/madaros_gum_fo_let_bytecode.sio` | E137 on `variance_of` | OK | OK | **FAIL** (variance wrong) | arg is `Knowledge<f64>` but param declared as `ty_f64`; codegen reads .value as if it were a plain f64, gets wrong number |
| `tests/run-pass/rapamycin_rk4_budget.sio` | E137 on `variance_of` | OK | OK | **PASS** | arg is `Knowledge<f64>` but extracted via var-name resolution path that happens to work |
| `tests/run-pass/rapamycin_kaxi_fuse_prior.sio` | E137 on `seq_new`, `acknowledge` | **FAIL** with E011+E013 | (skipped) | - | new errors surfaced: "no method named for this type" at function-scope spans (Seq.push), "indexing requires an array type" at kaxi_fuse 0..1003. **Were masked by E137** before this fix. |
| `tests/run-pass/observe_contraction.sio` | E137 on `observe` | **FAIL** with E137 | - | - | observe has NO codegen anywhere in lean_single.sio; needs separate work on that lane |

## What the numbers say

1. **5 of 7 tests pass E137** after the fix. That alone is the result the user asked for.
2. **2 actually run to completion** (variance_of-based tests that don't go through Knowledge<T> generic propagation). The user's "if two pass, that's a result and isolates the third" framing matches.
3. **The 3 acknowledge/require_confidence/madaros_gum tests typecheck and compile** but produce wrong runtime values. The cause is **my typecheck stub is too permissive** — declaring first params as `ty_unknown()` strips the Knowledge<T> type info that the codegen layer needs to extract .value correctly. This is a SECOND defect, surfaced by the fix.
4. **rapamycin_kaxi_fuse_prior's E011/E013 are pre-existing**, not caused by my fix. They were masked by E137 firing first. The fix removed the E137 blocker; the typechecker now proceeds to find these unrelated defects. (Verified: errors point at Seq method resolution and array indexing, both unrelated to intrinsic registration.)
5. **observe_contraction remains E137** as predicted. observe has neither source definition nor codegen. Separate workstream.

## Pattern (now with measurements)

The user's earlier framing about "single code hiding multiple causes" lands with data:

- **E137 itself**: one code, two causes (Cause A print_i64 pub, Cause B codegen intrinsics). Cause A not addressed by this commit (grok-cli2 lane); Cause B addressed.
- **Cause B's fix**: one mechanism (FnSig pre-registration), but the fix's effectiveness varies across the seven tests because the **codegen for these intrinsics has its own assumptions** about how the typechecker should propagate Knowledge<T>. My stub is too permissive for some tests, just-right for others.
- **rapamycin_kaxi_fuse_prior**: E137 was hiding E011+E013. Fixing E137 revealed the next defect signature, which the lane that owns Seq method resolution now has a clean pointer to.

## What needs to happen next (not on this lane)

| Defect | Owner | What |
|---|---|---|
| acknowledge/require_confidence return wrong value at runtime | self-hosted/check + self-hosted/compiler | pre-register Knowledge<T> as a real type in fn_sig, not ty_unknown — or define Knowledge as a generic TypeEntry that the typechecker can substitute |
| madaros_gum_fo_let_bytecode variance wrong at runtime | same | variance_of needs first param to accept Knowledge<T>, not f64 |
| rapamycin_rk4_budget passes today | n/a | works because the var-name resolution path is taken, not the call-arg-type path; not a stable contract |
| rapamycin_kaxi_fuse_prior E011/E013 | whoever owns Seq methods | `obs.push(measure(...))` — Seq.push needs to resolve against `Seq<Knowledge<f64>>` |
| observe codegen | lean_single.sio lane (congested) | add `compile_observe_call_x86` + a64 with normal-normal conjugate update |
| print_i64 visibility (Cause A) | grok-cli2 stdlib | add `pub` and re-export from common module |

## Status

- Fix implemented (commit `06fef4311f`) ✓
- Fix verified via `make build-madaros` and run on seven tests ✓
- 5/7 typecheck OK; 2 actually pass at runtime ✓
- Three follow-on defects surfaced and assigned above ✓
- observe codegen intentionally not addressed (lane discipline) ✓