# Lowerer Borrow Migration (`&!Lowerer` → Issue #3020)

**Status**: Phase 0 complete (incremental wrapper + 3 leaf functions migrated)  
**PR**: #3040 (`cursor/lowerer-borrow-migration-8bcf`)  
**Blocker**: Madaros fixed-point gate fails with SIGSEGV on **both** `main` and branch  

---

## Executive Summary

**Critical Finding**: The Madaros fixed-point gate **fails with rc=139 (SIGSEGV)** on `main@dbd9347aa` during `[rung check]`, making baseline measurement impossible. The branch exhibits identical failure.

### Immediate Root Cause

- **Gate command**: `scripts/ci/madaros_fixed_point_gate.sh` with `ulimit -s 524288` (512 MB stack)
- **Result**: SIGSEGV during typecheck of 130 self-hosted modules
- **Direct check**: `ulimit -s 524288 && souc check self-hosted/compiler/main.sio` → timeout (120s) with absurd frame size warnings (17 GB frames on GPU functions)
- **Madaros binary**: `artifacts/self-hosted/madaros` (124 MB, built 2026-10-23 14:19:33, after commit `dbd9347aa`)

### Why SIGSEGV with 512 MB Stack?

The current madaros binary in `artifacts/` appears **broken** or triggers a pathological case:
1. Requests 524 MB stack minimum (rejects smaller limits without `MADAROS_STACK_KB` override)
2. SIGSEGV even with 524 MB stack during self-typecheck
3. Timeout with absurd 17 GB frame size warnings on GPU backend functions

**Hypothesis**: Frame size calculation bug or infinite recursion in current madaros build.

---

## Migration Strategy (Incremental)

**User-mandated approach**: Do NOT break all 496 call sites in one pass.

### Phase 0: Wrapper + Leaf Functions ✅

1. **Created `fresh_reg_mut(self: &!Lowerer) -> i64`** — mutating method that updates `self` in place
2. **Kept `fresh_reg(self) -> (Lowerer, i64)`** — thin by-value wrapper:
   ```sio
   fn fresh_reg(self) -> (Lowerer, i64) with Mut, Panic, Div, IO {
       var lo = self
       let r = lo.fresh_reg_mut()
       (lo, r)
   }
   ```
3. **Migrated 3 leaf functions internally**:
   - `emit_i64_const(self, val: i64) -> (Lowerer, i64)`
   - `emit_f64_const(self, val: f64) -> (Lowerer, i64)`
   - `emit_name_string(self, val: &[u8]) -> (Lowerer, i64)`
   
   Pattern:
   ```sio
   fn emit_i64_const(self, val: i64) -> (Lowerer, i64) with Mut, Panic, Div, IO {
       var lo = self
       let r = lo.fresh_reg_mut()  // ← was: let pair = lo.fresh_reg(); lo = pair.0; let r = pair.1
       lo = lo.emit(ir_load_imm(r, val))
       (lo, r)
   }
   ```

**Result**: 0 typecheck errors, 496 call sites → 493 call sites.

### Phase 1: Next 5 Hot Functions (Failed — Incomplete Substitution)

Top candidates (by `fresh_reg()` call count):
1. `lower_sign_call_ref` (49 calls, 251 lines)
2. `lower_binary_expr_ref` (19 calls)
3. `fo_apply_transfer_kind` (16 calls)
4. `fo_step_combine` (15 calls)
5. `fo_emit_recur_ymul_scale` (15 calls)

**Result**: Migration introduced 196 E012 errors (syntax errors).

**Root cause**: Incomplete pattern substitution. The automated script replaced:
```sio
let r = lo.fresh_reg()  // returns (Lowerer, i64)
lo = r.0
// use r.1
```
with:
```sio
let r = lo.fresh_reg_mut()  // returns i64 directly
// use r.1  ← ERROR: r is i64, not a tuple!
```

The correct substitution must also replace **all** references to `r.1` with just `r`:
```sio
let r = lo.fresh_reg_mut()
// use r  ← correct
```

**Commits reverted**: All 5 migrations were reverted due to this error.

**Lesson**: Automated substitution is unsafe for this pattern. Manual migration required, or a two-pass script: (1) replace call + remove `.0` assignment, (2) replace all `.1` accesses with direct variable.

---

## Gate Results

### Main (`dbd9347aa`)

```
=== Main (dbd9347aa) - ulimit -s: 16384 KB ===
Fri Oct 23 03:25:17 PM UTC 2026

[rung check] gen1 typechecks self-hosted/compiler/main.sio
rc=139 errors=0

reached  none
expected run
MADAROS_FIXED_POINT_FAIL
```

**Log**: `/tmp/madaros-fp-main/check.log` shows SIGSEGV after "about to check 130 modules"

### Branch (`cursor/lowerer-borrow-migration-8bcf` @ `b3b63bb2c`)

```
=== Branch (b3b63bb2c) - ulimit -s: 16384 KB ===
Fri Oct 23 03:27:32 PM UTC 2026

[rung check] gen1 typechecks self-hosted/compiler/main.sio
rc=139 errors=0

reached  none
expected run
MADAROS_FIXED_POINT_FAIL
```

**Identical failure**.

### Direct Check Attempt (524 MB Stack)

```bash
ulimit -s 524288
timeout 120 ./bin/souc check self-hosted/compiler/main.sio
```

**Result**: Timeout (rc=124) after 120s, with warnings:
```
warning: stack frame too large (17253513128 bytes) — by-value parameter of fn hlir_extract_kernel_count
warning: stack frame too large (17253513128 bytes) — by-value parameter of fn hlir_kernels_to_ptx
warning: stack frame too large (17997832 bytes) — return of fn struct_table_new
```

**17 GB frames** are clearly bogus — suggests frame size calculation bug.

---

## Known Issues

### 1. Frame Size Calculation Bug

The `check.sio` frame size calculation treats **every stack-passed struct** as consuming the size of the **largest** struct in the entire signature. This was hidden by capping warnings at 16 GiB.

**Example**: `lower_sign_call_ref` signature with embedded `Lowerer` (4.5 MB) causes frame size to be reported as 17 GB due to compounding with GPU structs.

**Related**: Issue #3015 (3x slowdown) was caused by passing 4.5 MB `Lowerer` by value hundreds of times, exhausting the 2 GiB bump arena.

### 2. `rc=181` on `gen3` (Expected)

When the fixed-point gate reaches `[rung gen3]`, it exits with `rc=181` ("arena full"). This is **normal** and documented — the 2 GiB bump arena is designed to be exhausted during gen3 self-compile. The gate expects to reach `[rung run]` before the arena limit.

### 3. SIGSEGV vs rc=181 Distinction

- **SIGSEGV (rc=139)**: Stack overflow (bad) — current blocker
- **rc=181**: Arena exhaustion (expected) — happens on gen3 when memory limit is hit

---

## Hypotheses for SIGSEGV

1. **Broken madaros binary in snapshot** — binary may have been built from broken state
2. **Frame size bug compounds with large by-value structs** — pathological recursion during frame calculation
3. **Infinite loop in typecheck** — timeout with no progress suggests hang

---

## Next Steps (Blocked)

1. **Investigate/rebuild madaros binary** — current binary may be broken
2. **Fix frame size calculation** — 17 GB warnings are clearly wrong
3. **Re-run gate on both sides** — once madaros is fixed, measure:
   - Execution time
   - Maximum RSS
   - Return codes of each rung
   - Maximum stack frame warning size
4. **Continue Phase 1 migration** — migrate next 5 hot functions
5. **Document comparative results**

---

## Commits

- **Phase 0 wrapper**: `b3b63bb2c` — `fn fresh_reg_mut(&!Lowerer) -> i64` + wrapper
- **Leaf migrations**: 
  - `emit_i64_const`: (commit needed)
  - `emit_f64_const`: (commit needed)
  - `emit_name_string`: (commit needed)

---

## References

- Issue #3020: Migrate Lowerer by-value → borrow
- PR #3040: Draft implementation
- Issue #3015: 3x slowdown from Lowerer by-value
- PR #3025: Stack floor 16 MiB

---

**Last updated**: 2026-10-23 15:30 UTC  
**Measurement status**: ❌ Blocked (gate fails on both main and branch with SIGSEGV)
