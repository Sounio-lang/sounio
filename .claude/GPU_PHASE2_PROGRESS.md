# GPU Kernel Library Expansion - Progress Report

## Summary

Phases 1-4 COMPLETE, Phase 5 infrastructure COMPLETE. All 51 GPU PTX tests pass.
Remaining: Phase 5 stencil kernels, Phase 6 validation.

---

## Completed Work

### Phase 1: Type System Extension (COMMITTED: a2e0c88)

**Files Modified:**
- `self-hosted/gpu/kernel_ir.sio` (+32 lines)
- `self-hosted/gpu/lower_to_ptx.sio` (+65 lines)
- `self-hosted/gpu/ptx.sio` (+27 lines)

**Changes:**
1. Extended `GpuType` enum: GpuF64, GpuI32, GpuI64
2. Added `gpu_type_sizeof(ty: GpuType) -> i64` helper
3. Extended `GpuKernelIr` struct: max_reg_f64, max_reg_i32, max_reg_i64
4. Added register name functions: gpu_reg_f64, gpu_reg_i32, gpu_reg_i64
5. Extended `gpu_emit_reg_decls()` for .reg .f64 declarations
6. Added 9 PTX opcode functions: add/ld_global/st_global for f64/s32/s64

---

### Phase 2: PTX Opcodes + Element-Wise Kernels (COMMITTED: 799400a + uncommitted)

**Files Modified:**
- `self-hosted/gpu/ptx.sio` (+42 lines) — PTX opcode functions
- `self-hosted/gpu/kernel_ir.sio` (~+600 lines) — Generic builder + 15 wrappers
- `self-hosted/gpu/lower_to_ptx.sio` (~+120 lines) — GpuSub/Mul/Div + type coverage
- `self-hosted/test_gpu_ptx.sio` (~+200 lines) — Tests T11-T26

**Phase 2.1 — PTX Opcode Functions (committed 799400a):**
- Subtraction: `ptx_opcode_sub_f32/f64/s32/s64/u64()`
- Multiplication: `ptx_opcode_mul_lo_f32/f64/s32/s64()`
- Division: `ptx_opcode_div_f32/f64/s32/s64/u64()`

**Phase 2.2 — GpuOpcode Enum Extension:**
- Added: GpuSub, GpuMul, GpuDiv to enum
- (Linter also added Phase 3 opcodes: GpuMax, GpuMin, GpuBarrierSync, GpuShflDown, GpuAtomicAdd, GpuLoadShared, GpuStoreShared)

**Phase 2.3 — gpu_lower_op() Extensions:**
- GpuSub: match on ty for sub.f32/f64/s32/s64 + correct register dispatch
- GpuMul: match on ty for mul.rn.f32/f64, mul.lo.s32/s64 + correct register dispatch
- GpuDiv: match on ty for div.rn.f32/f64, div.s32/s64 + correct register dispatch
- GpuLoadGlobal: extended for f64→ld.global.f64, i32→ld.global.s32, i64→ld.global.s64
- GpuStoreGlobal: extended for f64→st.global.f64, i32→st.global.s32, i64→st.global.s64

**Phase 2.4 — Generic Builder + 15 Wrappers:**
- `gpu_build_vec_binop_ir(kernel_name: Name, binop: GpuOpcode, elem_ty: GpuType)` — generic builder (~150 lines)
  - Parameterizes: kernel name, operation opcode, element type
  - Register allocation: f32→%f1-3, f64→%fd1-3, i32→%r2-4 (tid shares %r1), i64→%rd1-3
  - Sizeof: 4 for f32/i32, 8 for f64/i64
- 15 name builders: `gpu_name_vec_add_f64()`, `gpu_name_vec_sub_f32()`, etc.
- 15 thin wrappers: `gpu_build_vec_add_f64_ir()`, `gpu_build_vec_sub_f32_ir()`, etc.
- Original `gpu_build_vec_add_ir()` kept for backward compat with T10

**Phase 2.5 — Tests T11-T26:**
- T11-T13: vec_add f64/i32/i64
- T14: vec_add f32 backward compat (generic builder)
- T15-T18: vec_sub f32/f64/i32/i64
- T19-T22: vec_mul f32/f64/i32/i64
- T23-T26: vec_div f32/f64/i32/i64
- Each test: build IR → lower to PTX → check entry name, opcode, register prefix, ld/st opcodes

**PTX Float Multiply Fix:**
- Changed `mul.lo.f32` → `mul.rn.f32` and `mul.lo.f64` → `mul.rn.f64` (correct PTX ISA: round-to-nearest for float)

**Result:** All 26 GPU PTX tests pass:
```
=== GPU PTX Tests ===
  T01-T10 OK (Phase 0 + Phase 1)
  T11 OK (add f64)
  T12 OK (add i32)
  T13 OK (add i64)
  T14 OK (add f32 compat)
  T15 OK (sub f32)
  T16 OK (sub f64)
  T17 OK (sub i32)
  T18 OK (sub i64)
  T19 OK (mul f32)
  T20 OK (mul f64)
  T21 OK (mul i32)
  T22 OK (mul i64)
  T23 OK (div f32)
  T24 OK (div f64)
  T25 OK (div i32)
  T26 OK (div i64)
GPU PTX tests: all passed
```

---

### Phase 3: Reduction Kernels (COMPLETE)

**Files Modified:**

- `self-hosted/gpu/kernel_ir.sio` (~+200 lines) — Generic reduction builder + 12 wrappers
- `self-hosted/gpu/lower_to_ptx.sio` (~+10 lines) — Register declaration fix
- `self-hosted/test_gpu_ptx.sio` (~+100 lines) — Tests T27-T34
- `self-hosted/main.sio` (+5 lines) — write_elf_to_file stub for directory mode

**Changes:**

1. Fixed parse error: `GpuType:: GpuU32` → `GpuType::GpuU32`
2. Added 4 missing name helpers: gpu_name_vec_sum_i32/i64, gpu_name_vec_min_f64/i64
3. Replaced all linter-generated reduction builders with generic `gpu_build_reduce_warp_ir(kernel_name, reduce_op, elem_ty)`:
   - Warp shuffle-down pattern: load → 5x (shfl.sync.down + reduce_op at offsets 16,8,4,2,1) → write result
   - Register allocation: f32→%f1-3, f64→%fd1-3, i32→%r2-4 (r1=tid), i64→%rd6-8 (rd1-5=addr calc)
   - Atomic add for sum f32/f64/i32; st.global for sum i64 and all max/min (no atomic max/min)
4. 12 thin wrappers: vec_sum/max/min × f32/f64/i32/i64
5. Fixed `gpu_emit_reg_decls()`: b32 count = max(max_reg_u32, max_reg_i32), b64 count = max(max_reg_u64, max_reg_i64)
6. Added write_elf_to_file stub in main.sio (real impl in disabled io/file_write.sio)

**Result:** All 34 GPU PTX tests pass:

```text
T01-T26 OK (Phase 0-2)
T27 OK (sum f32)  T28 OK (sum f64)  T29 OK (sum i32)  T30 OK (sum i64)
T31 OK (max f32)  T32 OK (max i32)  T33 OK (min f32)  T34 OK (min i32)
GPU PTX tests: all passed
Suites: all passed
```

---

### Phase 4: Matrix Operations (COMPLETE)

**Files Modified:**

- `self-hosted/gpu/kernel_ir.sio` (~+400 lines) — FMA/GEMM/Grid builders + 6 wrappers
- `self-hosted/gpu/ptx.sio` (~+70 lines) — FMA/MAD opcodes, U32 arithmetic, predicate ops
- `self-hosted/gpu/lower_to_ptx.sio` (~+300 lines) — GpuFma/GetBid/GetNtid/AddImm + predicates + i64_to_string rewrite
- `self-hosted/test_gpu_ptx.sio` (~+80 lines) — Tests T35-T42

**Changes:**

1. Extended GpuOpcode enum: +4 opcodes (GpuFma, GpuGetBid, GpuGetNtid, GpuAddImm)
2. Added PTX opcode functions: fma.rn.f32/f64, mad.lo.s32/s64, add.u32, mul.lo.u32
3. Extended gpu_lower_op() match arms for GpuFma, GpuGetBid, GpuGetNtid, GpuAddImm
4. Extended GpuAdd/GpuMul lowering for GpuU32 and GpuU64 types
5. Three generic kernel builders:
   - `gpu_build_fma_ir(name, ty)` — element-wise c[i] = a[i]*b[i]+c[i] (demonstrates fma.rn)
   - `gpu_build_gemm_ir(name, ty)` — blocked FMA: row = bid.x*16+tid.x (demonstrates ctaid.x)
   - `gpu_build_grid_ir(name, ty)` — grid-stride: uses ntid.x for block dimension (demonstrates ntid.x)
6. 6 name helpers + 6 thin wrappers: fma/gemm/grid × f32/f64
7. Rewrote i64_to_string as full algorithmic implementation (modulo/division)

**Phase 5 Infrastructure (also completed):**

- Added predicate opcodes: GpuSetpLt, GpuSetpLe, GpuSetpEq, GpuSelp
- Added predicated memory ops: GpuLoadGlobalPred, GpuStoreGlobalPred, GpuLoadSharedPred, GpuStoreSharedPred
- Added shared memory address computation and declaration
- Tests T43-T51 covering predicates, shared memory, barriers, buffer operations

**Result:** All 51 GPU PTX tests pass:

```text
T01-T34 OK (Phases 0-3)
T35 OK (fma f32)   T36 OK (fma f64)
T37 OK (gemm f32)  T38 OK (gemm f64)
T39 OK (grid f32)  T40 OK (grid f64)
T41 OK (gemm 3 params)  T42 OK (fma loads)
T43 OK (shared bytes declared)
T44-T47 OK (predicated global/shared load/store)
T48 OK (selp present)  T49 OK (barrier present)
T50-T51 OK (op buffer append/overflow)
GPU PTX tests: all passed
Suites: all passed
```

---

## Remaining Work

### Phase 5: Stencil Kernels
- Add conv2d builders using existing predicate/shared memory infrastructure
- 2D indexing with tid.x/tid.y, halo loading

### Phase 6: Testing & Validation
- Cross-kernel consistency tests
- Instruction budget validation

---

## Verification Commands

```bash
# Run all self-hosted tests including GPU
cargo run --bin souc -- run self-hosted/ -- test

# Expected: "GPU PTX tests: all passed" with T01-T51
# Expected: "Suites: all passed"
```

---

## References

- Plan file: `/home/demetrios/.claude/plans/rippling-stirring-naur.md`
- Commits:
  - Phase 1: `a2e0c88` - Type system infrastructure
  - Phase 2 partial: `799400a` - PTX opcode functions
  - Phase 2 remainder + Phase 3 + Phase 4: uncommitted
