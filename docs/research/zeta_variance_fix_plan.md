<!-- docs:meta
topic_id: repo.docs.research.zeta-variance-fix-plan
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.zeta-variance-fix-plan
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# ζ — GUM Variance Tracking Bug Fix Plan

**Date**: 2026-04-12
**Status**: Investigation complete; fix not yet implemented
**Related**: `feedback_variance_deep_chains.md`, `rapamycin_epistemic_adaptive.sio` (currently fails with 2^63)

## Executive Summary

**Root Cause**: The GUM variance shadow buffer is statically sized for 1024 variable slots, but while loops create new variable slots without bound. When a variable reference exceeds slot 1023, the addressing calculation `RT_VARIANCE_BUF_BSS_OFF + ch * 8192 + slot * 8` reads uninitialized BSS memory far outside the allocated region, appearing as 2^63.

**Fix**: Reset `NEXT_SLOT` at the while/for loop iteration boundary. 3 locations, 2 lines each. Follows the proven pattern from closure handling (line 6781).

## Technical Root Cause

### Memory Layout

```
RT_VARIANCE_BUF_BSS_OFF = GL_BSS_SIZE
  Size: 1024 slots × 8 bytes × 8 channels = 65,536 bytes
  Addressing: GL_BSS_BASE + RT_VARIANCE_BUF_BSS_OFF + ch * 8192 + slot * 8
  Valid range: slot ∈ [0, 1023]
```

### The bug mechanism

1. **Function return** (line 9273): `EXPR_VAR_SLOT = -2` (variance in SCRATCH)
2. **Let binding** (lines 11537-11539): `emit_copy_scratch_to_var_variance_x86(slot)` copies SCRATCH → VAR[ch][slot]
   - **Problem**: Line 5419 silently returns if `slot >= 1024`
3. **Next iteration**: allocates new slots, `NEXT_SLOT` grows unbounded.
4. **After ~50 iterations** of a 20-variables-per-iteration loop: `NEXT_SLOT > 1024`, silent failure, later variance reads return garbage.

### Why smaller tests work

- **1-stage** (`gum_euler_ode.sio`): ~3 vars/iter × 10 iter = 30 slots. Safe.
- **4-stage** (`rapamycin_rk4_budget.sio`): ~10 vars/iter × 20 iter = 200 slots. Safe.
- **Bogacki-Shampine** (`rapamycin_epistemic_adaptive.sio`): ~20 vars/iter × 500 iter = 10,000 slots. **Overflows at ~iter 50**.

### Why 2^63

Uninitialized BSS memory that appears to be sign-extended when read as i64 and interpreted as f64 produces values near `0x8000000000000000` = 2^63.

## Code References

### Primary Issue: No NEXT_SLOT Reset in While Loop

**File**: `self-hosted/compiler/lean_single.sio`, lines 12097–12122

Compare with the **working pattern** at lines 6735-6803 (closure handling):
```
fn closure_save_scope() with Mut {
    let save_ns2 = NEXT_SLOT  // SAVE
    // ... compile body ...
    NEXT_SLOT = save_ns2      // RESTORE
}
```

### Secondary Issue: Silent Bounds Check

**File**: lines 5417–5430, `emit_copy_scratch_to_var_variance_x86`:
```sio
fn emit_copy_scratch_to_var_variance_x86(slot: i64) with Mut {
    if slot < 0 || slot >= 1024 { return }  // ← SILENT FAILURE
    ...
}
```

When called with `slot >= 1024`, function returns without copying. Variance stays uninitialized.

## Recommended Fix

### Why slot reset is correct

1. Semantically correct: loop-scoped variables should not persist across iterations.
2. Matches proven pattern (closure handling, line 6781).
3. Zero risk: variables in one iteration don't escape to the next.
4. Minimal code: 3 locations, 2 lines per location.

### Change 1: While loop (x86)

**Location**: `lean_single.sio`, line ~12097

Insert `let saved_next_slot = NEXT_SLOT` before compile_stmt loop.
Insert `NEXT_SLOT = saved_next_slot` after the closing `}`.

### Change 2: For loop (x86)

**Location**: `lean_single.sio`, lines 12128–12219

`saved_next` is already captured at line 12135. Add restore before the return at line 12218.

### Change 3: While loop (A64)

Apply Change 1 pattern to the ARM A64 handler at line ~18983.

## Test Strategy

**Must continue passing**:
- `gum_euler_ode.sio`
- `rapamycin_rk4_budget.sio`
- `rapamycin_gum_vs_mc.sio`
- `rapamycin_epistemic_adaptive.sio` (currently passes with deformed variance; should pass with clean variance)

**Should now diagnose cleanly**:
- `rapamycin_epistemic_adaptive.sio` — variance_of(c_blood) should be finite (~0.0001 range based on GUM-vs-MC test), not 2^63.

**Stress test (optional)**: new file `tests/run-pass/variance_deep_loop.sio` — 1000-iteration loop with 4 let bindings per iteration. Should pass with fix.

## Risk Assessment

**Bootstrap stability (gen2 == gen3)**: MINIMAL RISK

- Changes are to slot allocation strategy, not variance algorithms.
- No modifications to `emit_gate_variance_*` functions.
- No BSS layout changes.
- Slot reset follows proven closure-handling pattern.

**Validation sequence**:
1. Current gen2 → gen3 bit-identical verify (md5=1e0f256a — confirmed 2026-04-21).
2. Apply fix. Rebuild gen1 via old compiler.
3. Rebuild gen2 via gen1.
4. Rebuild gen3 via gen2. Verify gen2 == gen3.
5. Run `rapamycin_epistemic_adaptive.sio` with fixed compiler. Confirm variance is finite.
6. Run full test suite.

## Effort Estimate

- Code edits: 30 minutes
- Rebuild + bootstrap verification: 1 hour
- Full test suite: 30 minutes
- **Total: 2 hours**

## Implementation Checklist

- [ ] Add `let saved_next_slot = NEXT_SLOT` at line ~12100 (while x86)
- [ ] Add `NEXT_SLOT = saved_next_slot` at line ~12115 (while x86)
- [ ] Add `NEXT_SLOT = saved_next` at line ~12218 (for x86)
- [ ] Apply while-x86 pattern to A64 handler at line ~18983
- [ ] Rebuild gen1, gen2, gen3. Verify md5(gen2) == md5(gen3).
- [ ] Verify `rapamycin_epistemic_adaptive.sio` variance is finite.
- [ ] Full test suite regression check.

## Epistemic Honesty

**Confidence in root cause**: 95%
- Slot count math matches failure threshold (20 × 50 > 1024).
- 2^63 consistent with uninitialized BSS read.
- Silent bounds-check at line 5419 explains masked failure.
- Pattern-match to closure-handling shows the correct idiom exists.

**Confidence in fix**: ~~100%~~ → **REVISED: ~40% after first attempt failed**

## UPDATE 2026-04-12: First fix attempt failed bootstrap

**What was tried**: Added `let saved_ns_while = NEXT_SLOT` at the top of the x86 while-loop handler and `NEXT_SLOT = saved_ns_while` at the bottom, immediately before the epistemic confidence degradation.

**Outcome**: gen1 compiled successfully. gen2 compiled via gen1 succeeded (4523543 bytes). But gen3 compiled via gen2 FAILED with "error: no main". This is a bootstrap regression.

**Likely reason**: resetting `NEXT_SLOT` alone without also resetting `VAR_COUNT` (the symbol table) creates stale variable entries pointing to slots that subsequent code will re-allocate. When the next function body after the loop allocates new locals at the freed slots, they collide with stale entries that the name resolver still finds. The "no main" error suggests function-or-variable lookup was corrupted in a way that made `main` unfindable.

**Reverted cleanly**. gen2 == gen3 fixed-point restored (md5=`24cfaccb`).

## Revised fix hypothesis

The correct fix requires **coordinated save/restore of both `NEXT_SLOT` and `VAR_COUNT`** at the loop boundary, mirroring the pattern at lines 6728/6780 used for function-body scope:

```sio
let saved_ns = NEXT_SLOT
let saved_vc = VAR_COUNT
// ... compile loop body ...
NEXT_SLOT = saved_ns
VAR_COUNT = saved_vc
```

But this has its own risk: the loop body's *var* declarations (not let) may be intended to persist across iterations at runtime (same `var` used each time the body runs). Resetting VAR_COUNT could make those unfindable on the second iteration.

Actually, this is fine: the body is compiled ONCE. Runtime iterations reuse the same slots. But at compile time, after the loop block exits, the scope is gone anyway per lexical-scope rules — so resetting VAR_COUNT to the pre-loop value matches the language semantics.

**However**, the `let` bindings inside the loop body that reference uncertain values DO need their variance slots to persist across loop statements within the body (e.g., `let k1 = rhs(a, b)` then `let k2 = rhs(k1, c)`). As long as VAR_COUNT and NEXT_SLOT reset happens AFTER the loop body is fully compiled, this should be fine.

## What to try next

1. **Implement coordinated save/restore**: save both `NEXT_SLOT` and `VAR_COUNT` at loop entry, restore both at loop exit.
2. **Test on a minimal case first**: a tiny compile-fail test that demonstrates slot overflow without other confounders.
3. **Bisect**: apply the fix to ONE of (x86 while, x86 for, A64 while) and verify bootstrap. If any fails, the approach is wrong.
4. **Alternative deeper fix**: increase the variance buffer slot count from 1024 to 16384 (matching `VAR_CONFIDENCE`'s 16K). This would avoid the overflow without any scope changes. Cost: 16× memory (1MB of BSS per 8-channel table, manageable).

## Fallback: increase buffer size instead of scope fix

Given the failed fix attempt, the **simpler and lower-risk approach** may be to just increase `RT_VARIANCE_BUF_BSS_OFF`'s slot count from 1024 to 16384. This:

- Requires NO scoping changes (zero risk of breaking bootstrap).
- Matches `VAR_COUNT`'s existing cap of 16384.
- Uses 1 MB of additional BSS per 8-channel scratch table, vs. 64 KB currently.
- Would fully eliminate the overflow for realistic programs (`rapamycin_epistemic_adaptive.sio` has maybe 100-200 slots, not 1024+).

This is a simpler edit: change the hardcoded 1024 and 8192 to larger values in:
- `emit_copy_scratch_to_var_variance_x86` (line 5419): `if slot < 0 || slot >= 16384 { return }`
- Variance buffer allocation: check where `RT_VARIANCE_BUF_BSS_OFF` is initialized
- `ch * 8192` addressing: update to `ch * (16384 * 8) = ch * 131072`

**Recommended next step**: try the buffer-size approach before retrying the scope-reset approach. The scope fix is elegant but requires understanding Sounio's full variable lifecycle; the buffer expansion is a one-liner.

**Unknowns**: whether the slot allocator even goes above 1024 in practice. The "no main" error suggests the slot machinery is doing something beyond pure monotonic allocation — function bodies may already be resetting somewhere. A small investigation: instrument `NEXT_SLOT` with a `max_seen` counter to learn what values are actually hit during `rapamycin_epistemic_adaptive.sio` compilation.

## UPDATE 2026-04-12 (session B): reproducer + defer

**Added**: `tests/run-pass/variance_deep_loop.sio` — minimal reproducer marked `//@ known-failure`. 40 iterations of a 20-let-binding helper function trigger a 15-order-of-magnitude variance inflation (`var(acc) = 1.18e15` observed on current compiler; physical answer should be ~0.01). Smaller patterns (80 iterations × 5 lets inline) do NOT trigger, confirming the bug needs function-call-boundary variance plumbing *plus* depth to surface. This is now a tighter gate for any fix attempt than the 280-line rapamycin_epistemic_adaptive.

**Current state of edit**: NOT applied this session. `self-hosted/compiler/lean_single.sio` has 10 lines of unstaged edits from another thread (the `.len()` method dispatch + struct-through-ref resolve_field fix, diff starts around line 9712/11078). Applying the ζ buffer-size bump on top would commingle two unrelated fixes and make revert fragile if gen2==gen3 breaks.

**Blocked on**:
1. Other thread's unstaged `lean_single.sio` edits land (commit or revert) so ζ can be a clean diff.
2. A dedicated 30-60 min window to do gen1 → gen2 → gen3 rebuild + fixed-point verify.

**When unblocked, recommended sequence** (buffer-size approach from the "Fallback" section above):
1. Replace `slot < 1024` → `slot < 16384` (11 occurrences)
2. Replace `slot >= 1024` → `slot >= 16384` (3 occurrences)
3. Replace `ch * 8192` → `ch * 131072` (14 occurrences)
4. Replace `GL_BSS_SIZE + 8192 * BUDGET_CHANNELS` → `... + 131072 * BUDGET_CHANNELS` (1 occurrence at line 14289)
5. Rebuild gen1 / gen2 / gen3, verify `md5(gen2) == md5(gen3)`.
6. Verify reproducer `tests/run-pass/variance_deep_loop.sio` now reports PASS.
7. Verify rapamycin_epistemic_adaptive.sio no longer prints 9.22e18 in ch1/ch2.

Total edit: 29 precise substitutions in one file. No scope-reset logic, no new code paths — only constants widened. Risk to bootstrap fixed-point: minimal (the mechanical addressing arithmetic is preserved; only the upper bounds change).
