# Issue #2143 — measurement: which `begin_function_body_lowering` resets matter

**Verdict:** three of them matter and are a **live cross-function leak** — `fo_nchan`,
`fo_sigma2`, and `fo_bind_count`. The rest are either already reset in the live path or
masked per-seed. The task's working premise ("`fo_nchan` demonstrably IS zeroed per
function, so one of the four Lowerer constructions does it") is **false**: the four
"constructions" are per-**module** constructors, and `fo_nchan` is zeroed once per module,
never per function.

## The setup (measured, `self-hosted/ir/lower.sio`)

`begin_function_body_lowering` (line 5013) is dead — its own definition is the only
occurrence in the whole self-hosted tree (0 callers); `flush_current_func` has ~10. It is
the **pre-`*mut`-refactor by-value original** of the per-function setup. Its live
replacement is `lowerer_lower_fn_item_mut` (line 3598, `lo: &! Lowerer`, in-place),
called per function by the module driver (1542/1562/1613) over one **long-lived** Lowerer.
That the live path explicitly re-zeroes `array_elem_float_reg_count` etc. per function is
itself proof the Lowerer persists across functions (nothing to reset otherwise).

The four Lowerer constructions are the four **module** constructors, each called once:

| line | constructor |
|---|---|
| 784 | `lowerer_new` |
| 1143 | `lowerer_new_from_program_summary` |
| 1201 | `lowerer_new_from_program_summary_owned` |
| 1262 | `lowerer_from_acc_module` |

## The diff between the dead per-function setup and the live one

| Reset in dead `begin_function_body_lowering` | kind | reset in live `lowerer_lower_fn_item_mut`? | leaks across functions? |
|---|---|---|---|
| `current_fn/func`, `env`, `locals.count`, `scope_depth`, `loop_depth` | fields | ✅ yes | no |
| `array_elem_float_reg_count`, `variance_base_reg_count`, `pending_variance_reg`, `pending_closure_fn_id` (the D5 #986 block) | fields | ✅ yes | no |
| **`fo_nchan = 0`** | field | ❌ **no** | **YES** |
| **`fo_sigma2[] = -1`** | field | ❌ **no** | **YES** |
| **`fo_bind_count = 0`** | field | ❌ **no** | **YES** |
| `fo_expr_sens[] = -1` | field | cleared per-SEED (`fo_clear_expr_sens`) | no (masked) |
| `fo_pending_sens[] = -1` | field | cleared per-SEED (`fo_clear_pending_sens`) | no (masked) |
| `fo_cov_rho_clear()` (`FO_COV_RHO` global) | module global | ✅ **#2145** + `fo_xfer_global_reset` + per-seed | no (already fixed) |
| `fo_hoff_clear()` (`FO_HOFF` global) | module global | ✅ per-seed (`fo_clear_expr_sens`) + global | no |

Exhaustive assignment audit (the decisive evidence):
- `fo_nchan` is assigned in exactly **two** places: `= 0` at **5041 (inside the dead fn)**
  and `= ch + 1` at **6489 (the seed increment)**. No other zeroing exists.
- `fo_bind_count` likewise: `= 0` at **5044 (dead)**, `= idx + 1` at **6564 (increment)**.
- `fo_sigma2[ch] = variance_reg` at 6488 (seed) — only ever overwritten when a *new* channel
  is allocated; a stale entry is never cleared.
- `lowerer_lower_fn_item_mut` (3598–3775) contains **zero** references to
  `fo_nchan`/`fo_sigma2`/`fo_bind_count`/`fo_seed`/`fo_clear`.

So `fo_nchan`, `fo_sigma2`, `fo_bind_count` are zeroed once per module at construction and
then only grow, persisting across every function in the module. There is **no** per-function
reset in live code — the only one ever written sits in the never-called dead function.

## Why it is a real bug, and why it stayed latent

These tables are **keyed by vreg number, which restarts at 0 in each function** — the exact
leak class the D5 #986 comment describes for the fields that *are* reset. `fo_seed_from_variance`
(6466) reuses channel `k` when `fo_sigma2[k] == variance_reg` (6472). Across functions,
a later function's `variance_reg` (a fresh vreg) can collide with a **stale** `fo_sigma2[k]`
seeded in an earlier function → the seeder reuses the earlier function's channel instead of
allocating a fresh one, and the multi-channel covariance/Hessian loops (7093–7229, iterating
`k < fo_nchan` over `fo_sigma2[k]`) then walk stale channels that reference vregs belonging to
the earlier function → wrong first-order/Hessian variance codegen, or stale-vreg references.

It stayed latent because the *values* that would be most visibly wrong — `fo_expr_sens`,
`fo_pending_sens`, `FO_HOFF` — are cleared per **seed**, not per function, so only the channel
**count** (`fo_nchan`), the σ²-reg→channel **map** (`fo_sigma2`), and the bind count leak. The
misfire needs two functions in one module both using multi-channel FO variance/Hessian *and* a
vreg collision — rare in the test suite. (The task's "FO_SIGMA_REG … overwritten each seed"
refers to `fo_sigma2`; there is no symbol named `FO_SIGMA_REG`.)

## The fix (mirrors #2145 exactly)

`#2145` fixed the sibling `FO_COV_RHO` global by moving its reset into
`lowerer_lower_fn_item_mut`. The same move closes this leak. In `lowerer_lower_fn_item_mut`,
alongside the existing D5 block, add:

```
// D5/#2143: the FO channel table is keyed by vreg, which restarts per function.
// The *mut port dropped this reset (it lived only in the dead
// begin_function_body_lowering); without it fo_seed_from_variance reuses a stale
// channel when a later function's variance_reg collides with an earlier function's.
(*lo).fo_nchan = 0
(*lo).fo_bind_count = 0
var fci: i64 = 0
while fci < 32 {
    (*lo).fo_sigma2[fci as usize] = -1
    fci = fci + 1
}
```

(`fo_expr_sens`/`fo_pending_sens` are already cleared per-seed, so they need not be added
here, but including them is harmless and matches the dead function verbatim.) Then **delete
`begin_function_body_lowering`** (5013–5052): it is dead, and leaving it standing is exactly
what let the reset rot unnoticed through the `*mut` port.

**Acceptance / repro to add:** a single module with two `with Epistemic` functions, each doing
a multi-operand variance/Hessian computation (e.g. `a*b + c*d` over `Knowledge<f64>`), arranged
so the second function's σ² vreg collides with the first's channel-0 σ² reg. Pre-fix: the second
function's variance lowering reuses the stale channel (observable as wrong variance bits, or a
vreg out of range in the second function). Post-fix: fresh channel, correct lowering. A
same-source-built sabotage control (skip the new reset) should reproduce the pre-fix miscompile.

## Status / handoff

Measurement only — **not** applied. `lower.sio` is compiler-owned (codex-2); a codegen
correctness fix needs the build + zero-regression gate + owner review, the same discipline the
NS wire followed. The fix above is specified to drop in; executing it is a build/verify/coordinate
cycle. Evidence lines are current as of 2026-08-26 HEAD of `lane/fable-1/p0f-ffi-takeover`.
