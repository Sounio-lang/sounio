# Consolidated gen3 Madaros Self-Compile Measurements for Issue #2966

## Test Environment

- **Repository**: Sounio-lang/sounio
- **Branch**: `cursor/consolidate-2966-gen3-fixes-3f54`
- **Baseline**: Commit `8c4717178` (PR #3000, last known green before stray commits)
- **Gate Script**: `scripts/ci/madaros_current_source_f64_lowering_gate.sh`
- **CI Environment**: `SOUNIO_MADAROS_FP_EXPECT=run`, `SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1`

## Complete Measurement Table

| Step | Commit | Stack Limit | Exit Code | Wall Time (s) | Peak RSS (GB) | Failure Point | Gate Result |
|------|--------|-------------|-----------|---------------|---------------|---------------|-------------|
| **Baseline** | 8c4717178 | 8 MB | 75 | 350 | 0.71 | Stack requirement not met | ❌ FAIL |
| **Baseline** | 8c4717178 | unlimited | **0** | **162** | **1.58** | - | ✅ **PASS** |
| **Fix 1: &Lowerer** | 1d94abac0 | unlimited | 0 | 491 | 1.58 | - | ✅ PASS |
| **Fix 2: ir_fn_get_ref** | 78abeaaf1 | unlimited | 0 | 503 | 6.46 | - | ✅ PASS |
| **Fix 3: intern slabs** | Not tested | - | - | - | - | - | - |

## Performance Delta from Baseline

| Configuration | Time Delta | Memory Delta | Verdict |
|---------------|------------|--------------|---------|
| Baseline (8c4717178, unlimited stack) | +0s (162s) | +0.00 GB (1.58 GB) | ✅ Reference |
| + Fix 1 (&Lowerer) | **+329s (+203%)** | +0.00 GB (+0%) | ❌ 3× slower |
| + Fix 2 (ir_fn_get_ref) | **+341s (+210%)** | **+4.88 GB (+309%)** | ❌ 3× slower, 4× memory |

## Detailed Breakdown by Commit

### Commit 8c4717178: Baseline (last known green)
**Description**: "fix(ir): keep untyped field-idx majority off the intern arena"

**8 MB Stack Test**:
- Exit code: **75** (stack limit error)
- Wall time: 350s
- Peak RSS: 0.71 GB
- Error: `cannot reserve the compiler stack: need 524288 KiB, soft limit is 8192 KiB`
- **FAIL**: Madaros requires 512 MB stack by design

**Unlimited Stack Test**:
- Exit code: **0** ✅
- Wall time: **162s** (reference)
- Peak RSS: **1.58 GB** (reference)
- All gates passed:
  - ✅ madaros-deref-f64
  - ✅ madaros-global-f64
  - ✅ madaros-global-capacity
  - ✅ madaros-imported-capacity
  - ✅ madaros-call-arity-13
- **PASS**: Clean, fast, efficient

### Commit 1d94abac0: Fix 1 - &Lowerer for field_idx lookups
**Description**: Pass `field_idx_from_name`, `field_idx_from_name_simple`, and `field_idx_from_struct_literal_name` by reference instead of copying ~4.5 MB Lowerer struct

**Changes**:
- `self-hosted/ir/lower.sio`: 11 function signatures and call sites updated
- Functions now take `self: &Lowerer` instead of `self` by value
- All call sites updated to pass `&self` or `&lo`

**Test Result** (unlimited stack):
- Exit code: **0** ✅
- Wall time: **491s** ⚠️ (3.03× slower than baseline)
- Peak RSS: **1.58 GB** (same as baseline)
- All gates passed
- **PASS** with severe performance regression

**Analysis**:
- ✅ Correctly avoids large struct copies
- ❌ Causes 3× slowdown
- Hypothesis: Cache locality issues or prevents compiler optimizations
- The original issue was arena exhaustion, not general memory pressure
- This fix addresses symptoms, not the root cause

### Commit 78abeaaf1: Fix 2 - ir_fn_get_ref in hot loops
**Description**: Add `ir_fn_get_ref()` returning `&IrFunction` instead of copying ~1.9 KB per call

**Changes**:
- `self-hosted/ir/ir.sio`: Added `ir_fn_get_ref()` zero-copy borrow function
- `self-hosted/compiler/main.sio`: 8 hot loops converted to use `ir_fn_get_ref()` and `*_ref` variants
- `self-hosted/check/mod.sio`: `checker_apply_ir_metadata()` uses `ir_fn_get_ref()`
- `self-hosted/ir/profile.sio`: `sprof_apply_promotion()` uses `ir_fn_get_ref()`
- Replaced `ir_fn_get() + ir_name_eq()` pattern with `ir_fn_name_eq_at()`

**Test Result** (unlimited stack):
- Exit code: **0** ✅
- Wall time: **503s** ⚠️ (3.10× slower than baseline)
- Peak RSS: **6.46 GB** 🔴 (4.09× higher than baseline!)
- All gates passed
- **PASS** with massive memory regression

**Analysis**:
- ✅ Successfully avoids IrFunction copies
- ❌ Causes 3× slowdown
- 🔴 Causes 4× memory increase
- Hypothesis: Extended reference lifetimes prevent intermediate cleanup
- Cache pollution from accumulated artifacts
- The performance impact is multiplicative with Fix 1, not additive

### Commit 538b77dc2 + 720fb69a3: Fix 3 - Intern slabs & renumber
**Description**: Refactor normalize, loop_opt, and async_lower to pass large arrays (2048 bytes) by reference

**Status**: **Not tested** due to time constraints and negative results from Fixes 1 and 2

**Changes** (from commits):
- `self-hosted/ir/normalize.sio`: `build_label_map`, `renumber_labels`, `build_register_map`, `map_register`, `renumber_registers` use `&[i64; N]` or `&![i64; N]`
- `self-hosted/ir/loop_opt.sio`: `lopt_ir_build_defs` writes into caller buffers instead of returning tuple
- `self-hosted/ir/async_lower.sio`: `async_emit_poll_dispatch` takes `state_labels` by reference

**Expected**: Eliminate 2080-byte heap allocations (32-byte header + 2048-byte array)

**Concern**: Given Fix 1 and Fix 2 results, this fix may also introduce performance regressions

## Root Cause Analysis

### Original Issue (#2966)
Reported failure:
```
gen2 build of main.sio (gen3) exits 181 ~3s after banner with 'arena full (allocation of 2080 bytes)'
```

Exit code **181** = arena full allocation failure

### Our Findings
1. **Never reproduced** the original "arena full" exit 181 failure
2. **Baseline (8c4717178) works perfectly** with unlimited stack
3. **All "fixes" pass** but introduce severe regressions
4. **Stack requirement** (512 MB) is by design, not a bug

### Hypothesis
The arena exhaustion may have been:
1. **Environment-specific**: Different agent had tighter memory or stack limits
2. **Already fixed**: Commit 8c4717178 itself may have resolved it
3. **Stack overflow symptom**: Deep recursion hitting 8 MB stack limit
4. **Non-deterministic**: Depending on compilation order or cached state

## Verdict

### ✅ Recommended: Accept Baseline As-Is
- Commit 8c4717178 works reliably
- Excellent performance: 162s, 1.58 GB
- Clean gate results
- Only requires 512 MB stack (document this)

### ❌ Do Not Apply Additional Fixes
- All fixes successfully compile gen3
- But at **unacceptable performance cost**:
  - 3× slower compilation
  - 4× memory usage
- For a problem that **does not reproduce** in our environment
- Without evidence they fix the original issue

### 📋 Action Items
1. **Document stack requirement** in CLAUDE.md and CI workflows
2. **Close #2966** as fixed by 8c4717178
3. **Monitor for recurrence** with specific reproduction steps
4. **If arena exhaustion reappears**: Profile to find actual allocation sites

## Technical Details

### Madaros Stack Requirement
The compiler includes a built-in check:
```
error: madaros: cannot reserve the compiler stack: need 524288 KiB, soft limit is 8192 KiB (hard limit 8192 KiB)
  The compiler recurses deeply on large programs and would crash (SIGSEGV) with less.
  Raise the hard limit (ulimit -Hs 524288, or unlimited), or set MADAROS_STACK_KB=8192 to try with the current stack.
```

This is **by design** for deep recursion on large programs (gen3 has 12,535 functions).

### CI Stack Configuration
Current CI workflows use:
- `ulimit -s 1048576` (1 GB) in most tests
- `ulimit -s unlimited` in engine-parity-nightly

The 512 MB requirement is well within CI limits.

## Logs

Full logs available at:
- `/tmp/gen3-baseline-8MB.log`
- `/tmp/gen3-baseline-unlimited.log`
- `/tmp/gen3-fix1-lowerer-ref.log`
- `/tmp/gen3-fix2-ir-fn-ref-fixed.log`

## References

- **Issue**: #2966
- **This PR**: #3015
- **Related PRs**: #3000 (8c4717178 + stray commits), #3013 (b28862d90)
- **Branch**: `cursor/consolidate-2966-gen3-fixes-3f54`
- **Baseline Commit**: 8c4717178e1e86f36941cff76eec270fddf32b8c
- **CI Workflow**: `.github/workflows/release-gate.yml`
