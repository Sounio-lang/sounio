# Lowerer Borrow Migration (`&!Lowerer` → Issue #3020)

**Status**: Phase 0 complete, Phase 1 failed (incomplete substitution)  
**PR**: #3040 (`cursor/lowerer-borrow-migration-8bcf`)  

---

## Executive Summary

**Phase 0**: Successful incremental wrapper + 3 leaf functions migrated (0 typecheck errors).

**Phase 1**: Failed with 196 E012 syntax errors due to incomplete automated substitution pattern. All 5 function migrations reverted.

**Root cause**: Automated script replaced `let r = lo.fresh_reg(); lo = r.0` with `let r = lo.fresh_reg_mut()` but **did NOT** replace subsequent `r.1` tuple member accesses with `r`, causing syntax errors (accessing `.1` on a non-tuple `i64` value).

**Baseline measurement**: Not obtained. Main CI run is a REPLAY (cache hit), no fresh timing/RSS data available.

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

### Phase 1: Next 5 Hot Functions ❌ (Failed — Incomplete Substitution)

Top candidates (by `fresh_reg()` call count):
1. `lower_sign_call_ref` (49 calls, 251 lines) — f61d00fc9, reverted in 819a8e7da
2. `lower_binary_expr_ref` (19 calls) — fc2ab7be8, reverted in c88df2716
3. `fo_apply_transfer_kind` (16 calls) — da7ce003b, reverted in c16347a54
4. `fo_step_combine` (15 calls) — 7dbcf1247, reverted in bf15e2bd7
5. `fo_emit_recur_ymul_scale` (15 calls) — 1bfa871bb, reverted in 60fdeb2b5

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

**CI Evidence**:
- **Run**: [#39980205128](https://github.com/Sounio-lang/sounio/actions/runs/39980205128)
- **Gate result**: `MADAROS_FIXED_POINT_FAIL` with 196 `error[E012]` in `[rung check]`
- **Message**: "gen1 cannot typecheck the compiler's own entry point: 196 errors, rc=1"

Additionally, the first CI run ([#39980205128](https://github.com/Sounio-lang/sounio/actions/runs/39980205128)) **before reverts** failed with 5 SIGSEGV crashes in FO/Hessian witness tests:
- `epistemic_hessian_transcendentals.sio` — CLEAN-BUILD-FAIL
- `kl11_pow_fo_hessian.sio` — SIGSEGV during build
- `kl6_hessian_composite.sio` — SIGSEGV during build
- `madaros_gum_fo_correlate_latent.sio` — SIGSEGV during build
- `madaros_hessian_transcendental.sio` — SIGSEGV during build

This suggests the FO functions (3, 4, 5) may have had **semantic** issues beyond syntax, causing crashes in code that computes first-order variance and Hessians.

**Lesson**: Automated substitution is **unsafe** for this pattern. A correct two-pass approach would:

1. **Pass 1**: Replace `let r = lo.fresh_reg(); lo = r.0` → `let r = lo.fresh_reg_mut()`
2. **Pass 2**: Replace all `r.1` → `r` in the same scope

Without Pass 2, the code is syntactically invalid.

Alternatively, **manual migration** of each function with careful review of every `.1` access.

---

## Baseline Measurement: Not Obtained

The Madaros fixed-point gate on `main` is cached (REPLAY from prior green run). CI does not execute the gate with fresh timing/RSS data on every green run — only when the gate key changes (gen1 SHA, source tree, gate scripts, ratchet).

**Main run**: [#39971064449](https://github.com/Sounio-lang/sounio/actions/runs/39971064449) (`dbd9347aa`)  
**Gate result**: `MADAROS_FIXED_POINT_OK` (REPLAY, no fresh timing data)

Cannot provide comparative table (Branch vs Main) without forcing a fresh gate run by changing gate scripts or source tree.

---

## Known Issues

### 1. Frame Size Calculation Bug

The `check.sio` frame size calculation treats **every stack-passed struct** as consuming the size of the **largest** struct in the entire signature. This was hidden by capping warnings at 16 GiB.

**Example**: `lower_sign_call_ref` signature with embedded `Lowerer` (4.5 MB) causes frame size to be reported as 17 GB due to compounding with GPU structs.

**Related**: Issue #3015 (3x slowdown) was caused by passing 4.5 MB `Lowerer` by value hundreds of times, exhausting the 2 GiB bump arena.

### 2. `rc=181` on `gen3` (Expected)

When the fixed-point gate reaches `[rung gen3]`, it exits with `rc=181` ("arena full"). This is **normal** and documented — the 2 GiB bump arena is designed to be exhausted during gen3 self-compile. The gate expects to reach `[rung run]` before the arena limit.

### 3. SIGSEGV vs rc=181 Distinction

- **SIGSEGV (rc=139)**: Stack overflow (bad) — local environment issue in this case
- **rc=181**: Arena exhaustion (expected) — happens on gen3 when memory limit is hit

---

## Next Steps

1. **Fix the substitution pattern** — implement two-pass replacement:
   - Pass 1: `let r = lo.fresh_reg(); lo = r.0` → `let r = lo.fresh_reg_mut()`
   - Pass 2: `r.1` → `r` (all tuple member accesses in scope)
   
   Or migrate manually with careful review.

2. **Investigate FO/Hessian SIGSEGV** — why did `fo_apply_transfer_kind`, `fo_step_combine`, and `fo_emit_recur_ymul_scale` cause witness test crashes even with correct syntax?

3. **Force fresh gate run** — invalidate cache by changing a gate script line, or wait for a source tree change that affects the gate key

4. **Measure with CI instrumentation** — extract time/RSS from `/usr/bin/time -v` wrapper in gate script, or add explicit timing logging

---

## Commits

- **Phase 0 wrapper**: (committed earlier, before Phase 1 attempt)
- **Phase 1 migrations (all reverted)**:
  - `f61d00fc9` — lower_sign_call_ref → `819a8e7da` revert
  - `fc2ab7be8` — lower_binary_expr_ref → `c88df2716` revert
  - `da7ce003b` — fo_apply_transfer_kind → `c16347a54` revert
  - `7dbcf1247` — fo_step_combine → `bf15e2bd7` revert
  - `1bfa871bb` — fo_emit_recur_ymul_scale → `60fdeb2b5` revert
- **Documentation of failure**: `8ce266ade`

---

## References

- Issue #3020: Migrate Lowerer by-value → borrow
- PR #3040: Draft implementation
- Issue #3015: 3x slowdown from Lowerer by-value
- PR #3025: Stack floor 16 MiB

---

**Last updated**: 2026-10-23 16:30 UTC  
**Measurement status**: ❌ Not obtained (main CI run is REPLAY, branch migrations reverted due to syntax errors)
