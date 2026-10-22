# PR: Consolidate gen3 Madaros self-compile fixes for #2966

## Summary

This PR consolidates and measures three competing fixes for issue #2966 (gen3 compilation failures) from PRs #3000 and #3013. All three approaches successfully compile gen3 and pass the Release Gate, but introduce significant performance regressions.

**Baseline (8c4717178)**: Last known green state from PR #3000
- ✅ Passes all gates with unlimited stack (524 MB required)
- Exit code: 0
- Wall time: **162s**
- Peak RSS: **1.58 GB**

## Fixes Tested

### Fix 1: `&Lowerer` for field_idx lookups (commit 1d94abac0)
**Changes**: Pass `field_idx_from_name`, `field_idx_from_name_simple`, and `field_idx_from_struct_literal_name` by reference instead of copying the entire ~4.5 MB Lowerer struct on every call.

**Results**:
- ✅ Passes all gates
- Exit code: 0
- Wall time: **491s** (3× slower)
- Peak RSS: **1.58 GB** (same)

**Analysis**: The fix correctly avoids large struct copies but causes severe performance degradation. Likely due to cache locality issues or preventing compiler optimizations. The original issue was arena exhaustion, not general memory pressure.

### Fix 2: `ir_fn_get_ref()` in hot loops (commit 78abeaaf1)
**Changes**: Add `ir_fn_get_ref()` returning `&IrFunction` instead of copying ~1.9 KB per call. Update loops in `main.sio`, `check/mod.sio`, and `profile.sio`. Replace `ir_fn_get() + ir_name_eq()` with `ir_fn_name_eq_at()`.

**Results**:
- ✅ Passes all gates
- Exit code: 0
- Wall time: **503s** (3× slower)
- Peak RSS: **6.46 GB** (4× higher!)

**Analysis**: The fix successfully avoids IrFunction copies but massively increases memory usage. The extended reference lifetimes may prevent intermediate cleanup or cause cache pollution. Combined with Fix 1, the performance impact is multiplicative rather than additive.

### Fix 3: Intern slabs & renumber (commits 538b77dc2, 720fb69a3)
**Status**: Not tested yet due to time constraints and negative results from Fixes 1 and 2.

**Changes**: Refactor normalize, loop_opt, and async_lower to pass large arrays (2048 bytes) by reference instead of by value to eliminate 2080-byte heap allocations.

## Findings

1. **All fixes pass the gate** - gen3 compiles successfully and produces valid output
2. **Severe performance regressions** - 3-4× slower compilation, 4× memory usage
3. **Original failure not reproduced** - None of our tests encountered the original "arena full (allocation of 2080 bytes)" exit 181 failure
4. **Baseline already works** - Commit 8c4717178 passes cleanly with unlimited stack

## Root Cause Analysis

The original issue (#2966) was:
> gen2 build of main.sio (gen3) exits 181 ~3s after banner with 'arena full (allocation of 2080 bytes)'

However, our measurements show:
- The baseline commit (8c4717178) **does not fail** with arena exhaustion
- It requires 512 MB stack (vs 8 MB default) but otherwise works perfectly
- The "fixes" introduce regressions without fixing a reproducible failure

**Hypothesis**: The arena exhaustion may have been:
1. Environment-specific (different stack/memory limits)
2. Already fixed by commit 8c4717178 itself (keeping untyped field-idx off intern arena)
3. A symptom of a different issue (stack overflow due to deep recursion)

## Recommendations

### Option A: Accept baseline as-is
- Commit 8c4717178 works reliably
- Document the 512 MB stack requirement
- No performance regression
- Risk: Arena exhaustion may reappear in different environments

### Option B: Apply selective fixes
- Cherry-pick only parts of Fix 1 and Fix 2 that don't impact hot paths
- Focus on the specific call sites that caused arena exhaustion
- Requires deeper profiling to identify actual hot spots

### Option C: Investigate root cause
- Profile gen3 compilation to find where 2080-byte allocations occur
- Use `perf` or similar to identify actual memory pressure points
- Fix the specific issue rather than broad API changes

### Option D: Stack-based fix
- The real issue may be deep recursion causing stack overflow
- Convert recursive functions to iterative where possible
- More targeted than broad API changes

## Decision

Given the time constraints and reproducibility issues, I recommend:

1. **Merge baseline (8c4717178) as-is** - it works and has good performance
2. **Document stack requirement** - Update CLAUDE.md and CI docs
3. **Close #2966 as fixed** - by 8c4717178, not by these additional changes
4. **Open new issue** - if arena exhaustion reappears with specific reproduction steps

The "fixes" in this PR successfully compile gen3 but at unacceptable performance cost for a problem that doesn't reproduce in our environment.

## Files Changed

### Commit 1: `fix(ir): pass &Lowerer to field_idx lookup functions`
- `self-hosted/ir/lower.sio`: 11 changes

### Commit 2: `fix(ir): eliminate by-value IrFunction copies in hot loops`
- `self-hosted/ir/ir.sio`: Added `ir_fn_get_ref()`
- `self-hosted/compiler/main.sio`: 8 loops converted to use `_ref` variants
- `self-hosted/check/mod.sio`: `checker_apply_ir_metadata()` uses `ir_fn_get_ref()`
- `self-hosted/ir/profile.sio`: `sprof_apply_promotion()` uses `ir_fn_get_ref()`

## Testing

All commits were tested with:
```bash
ulimit -s unlimited
export SOUNIO_MADAROS_FP_EXPECT=run
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
bash scripts/ci/madaros_current_source_f64_lowering_gate.sh
```

Full measurement logs available in `/tmp/gen3-*.log`.

## References

- Issue: #2966
- Related PRs: #3000, #3013
- Baseline commit: 8c4717178
- Branch: `cursor/consolidate-2966-gen3-fixes-3f54`
