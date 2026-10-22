# Gen3 Self-Compile Measurement Results for Issue #2966

Branch: `cursor/consolidate-2966-gen3-fixes-3f54`

Baseline commit: `8c4717178` (last known green, PR #3000 before stray commits)

## Environment

- Stack limit tested: 8192 KB (8 MB, default) and unlimited
- CI environment variables: `SOUNIO_MADAROS_FP_EXPECT=run`, `SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1`
- Gate script: `scripts/ci/madaros_current_source_f64_lowering_gate.sh`

## Measurement Results

| Step | Commit | Description | Stack | Exit Code | Wall Time | Peak RSS | Result |
|------|--------|-------------|-------|-----------|-----------|----------|--------|
| Baseline | 8c4717178 | Last known green (PR #3000) | 8 MB | 75 | 350s | 730 MB | FAIL: Madaros requires 512 MB stack |
| Baseline | 8c4717178 | Last known green (PR #3000) | unlimited | 0 | 162s | 1615 MB (1.58 GB) | **PASS** |
| Fix 1 | 1d94abac0 | &Lowerer for field_idx lookups | unlimited | 0 | 491s | 1617 MB (1.58 GB) | **PASS** (3x slower!) |
| Fix 2 | 78abeaaf1 | ir_fn_get_ref in hot loops | unlimited | 0 | 503s | 6614 MB (6.46 GB) | **PASS** (4x memory!) |

## Analysis

### Baseline (8c4717178)
- ✅ Passes all gates with unlimited stack
- ❌ Fails with 8 MB stack due to Madaros stack requirement (524288 KB = 512 MB)
- Wall time: **162s** (reference)
- Peak RSS: **1.58 GB** (reference)

### Fix 1: &Lowerer References (1d94abac0)
**Problem**: 3× slower than baseline (491s vs 162s)

Changes:
- `field_idx_from_name`, `field_idx_from_name_simple`, `field_idx_from_struct_literal_name` take `self: &Lowerer` instead of by-value
- All call sites updated to pass `&self` or `&lo`

**Verdict**: PASS but severe performance regression. The fix is correct but may be trading stack copies for cache misses or preventing optimizations.

### Fix 2: ir_fn_get_ref in Hot Loops (78abeaaf1)
**Problem**: 4× memory usage (6.46 GB vs 1.58 GB) and still 3× slower (503s vs 162s)

Changes:
- Added `ir_fn_get_ref()` returning `&IrFunction` instead of copying ~1.9 KB
- Updated loops in `main.sio`, `check/mod.sio`, `profile.sio` to use `ir_fn_get_ref()`
- Replaced `ir_fn_get() + ir_name_eq()` with `ir_fn_name_eq_at()`

**Verdict**: PASS but massive memory regression. The reference lifetime may be preventing cleanup, or the cache is accumulating artifacts.

## Hypotheses

1. **Cache pollution**: The `.cache/madaros/` artifacts are accumulating and not being cleaned between fixes
2. **Reference lifetime issues**: Keeping references alive longer prevents intermediate cleanup
3. **Wrong fix direction**: The original problem was arena exhaustion, not general memory usage - these fixes may be addressing symptoms rather than root cause

## Next Steps

1. Clean cache and re-test baseline to establish clean reference
2. Test Fix 1 alone with clean cache
3. Test Fix 2 alone (without Fix 1) with clean cache
4. Investigate the third fix (intern slabs/renumber) from commits 538b77dc2 and 720fb69a3
5. Consider that the original #2966 issue was "arena full" at 181 exit code - none of the current fixes have encountered that specific failure mode

## Notes

- All tests passed the gate (exit 0), so the compiler produces valid output
- The performance regressions may be acceptable if they fix the original arena exhaustion
- Need to verify if arena exhaustion still occurs with default stack (8 MB) after applying fixes
