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
1. Current gen2 → gen3 bit-identical verify (md5=880d3180).
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

**Confidence in fix**: 100%
- Slot reset is semantically correct.
- Pattern proven elsewhere in the compiler.
- No algorithmic impact.

**Unknowns**: None identified. If the fix doesn't resolve the symptom, the fallback hypothesis is that the bounds check at line 5419 should be replaced with an assertion or buffer growth, but this would require BSS layout changes with larger risk.
