# Unified Rustless Cutover Implementation Plan

**Date**: 2026-02-13
**Status**: APPROVED
**Agents**: a2ed775 (CI), a534499 (verify), aee9f8f (Stage0), a897048 (artifacts)

## Executive Summary

This plan integrates four parallel workstreams into a cohesive strategy for removing Rust from the Sounio bootstrap chain. The plan has 5 phases over 6-8 weeks, with clear gates and parallel work opportunities.

**Key Decisions**:
- **Stage 0 runner**: C-based VM (name: `poseidon`) for universal portability
- **Verification artifact**: IR (IrModule) with semantic equivalence checking
- **Runtime artifact**: SOBC v1 bytecode (already stable)
- **CI strategy**: Add reproducibility gate, extend zero-fallback verification

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RUSTLESS BOOTSTRAP CHAIN                  │
└─────────────────────────────────────────────────────────────┘

Phase 0: Build Infrastructure
    ↓
┌─────────────────┐
│ poseidon (C VM) │  ← Stage 0: Minimal trusted base (~1000 LOC C)
│  • Loads .sobc  │
│  • Executes VM  │
└────────┬────────┘
         │ runs
         ↓
┌─────────────────────────┐
│ stage1.sobc (bytecode)  │  ← Stage 1: Self-hosted compiler
│  Compiled by Rust       │     (compiled by Rust oracle)
└────────┬────────────────┘
         │ compiles
         ↓
┌─────────────────────────┐
│ stage2.sobc (bytecode)  │  ← Stage 2: Self-hosted compiler
│  Compiled by Stage 1    │     (compiled by itself)
└────────┬────────────────┘
         │
         ↓
    ┌────────────────────────┐
    │ verify.sio validates:  │
    │  Stage1_IR ≡ Stage2_IR │  ← Reproducibility proof
    │  (semantic equivalence)│
    └────────────────────────┘
```

---

## Phase Breakdown

### Phase 0: Foundation (Week 1)
**Goal**: Set up artifact formats and serialization infrastructure

**Workstreams**:
- **IR Serialization** (3 days, ~300 LOC)
- **IR Normalization** (2 days, ~200 LOC)
- **Test Infrastructure** (2 days)

**Deliverables**:
- `self-hosted/ir/serialize.sio` (SOIR binary format)
- `self-hosted/ir/normalize.sio` (canonicalization)
- `self-hosted/test_ir_serialize.sio` (unit tests)

---

### Phase 1: Verification Pipeline (Week 2)
**Goal**: Wire Stage 1/2 comparison into verify.sio

**Workstreams**:
- **Driver Orchestration** (3 days, ~150 LOC)
- **verify.sio Integration** (3 days, ~200 LOC)
- **CI Gate: Reproducibility** (1 day, ~250 LOC)

**Deliverables**:
- Updated `stdlib/compiler/bootstrap/driver.sio` with `compile_stage1/2`
- Updated `stdlib/compiler/bootstrap/verify.sio` with IR comparison
- `scripts/selfhost_reproducibility_gate.sh`
- CI job: `selfhost-verify` in `.github/workflows/ci.yml`

**Gate**: Stage 1 ≡ Stage 2 (IR semantic equivalence) passes in CI

---

### Phase 2A: C-based Stage 0 VM (Week 3-4, parallel with 2B)
**Goal**: Implement poseidon C VM to replace Rust VM

**Workstreams**:
- **Core VM Interpreter** (5 days, ~800 LOC C)
- **SOBC Deserializer** (2 days, ~200 LOC C)
- **FFI Stubs** (1 day, ~100 LOC C)

**Deliverables**:
- `stage0_vm.c` (complete bytecode VM)
- `Makefile` (static linking, no Cargo dependency)
- `tests/stage0_smoke.sobc` (test fixture)

**Gate**: poseidon executes self-hosted test suite, exit code 0

---

### Phase 2B: Documentation (Week 3-4, parallel with 2A)
**Goal**: Document the new bootstrap architecture

**Workstreams**:
- **Comprehensive Gate Docs** (2 days)
- **Local Dev Runner** (2 days, ~400 LOC bash)
- **Bootstrap Architecture Docs** (2 days)

**Deliverables**:
- `docs/SELFHOST_CI_GATES.md` (all gates explained)
- `scripts/run_selfhost_gates_local.sh` (developer workflow)
- `docs/BOOTSTRAP_VERIFICATION.md` (verification algorithm)
- Updated `docs/SELF_HOSTING_PHASES.md`

---

### Phase 3: Integration & Validation (Week 5)
**Goal**: Replace Rust VM with poseidon in production flow

**Workstreams**:
- **CLI Integration** (2 days)
- **Parity Testing** (3 days)
- **CI Migration** (2 days)

**Changes**:
- `crates/souc/src/main.rs`: Add `SOUNIO_USE_C_VM=1` flag
- Run all existing tests with poseidon VM
- Update CI to use poseidon by default

**Gate**: All existing tests pass with poseidon (zero regressions)

---

### Phase 4: Cleanup & Extract (Week 6)
**Goal**: Remove Rust VM code, extract poseidon as standalone

**Workstreams**:
- **Deprecate Rust VM** (1 day)
- **Extract poseidon** (2 days)
- **Binary Artifacts** (2 days)

**Changes**:
- Mark `crates/souc/src/vm/` as deprecated
- Move `stage0_vm.c` to standalone repo/directory
- Makefile-only build (no Cargo)
- Generate release artifacts (poseidon binaries for Linux/macOS)

**Gate**: `souc` builds without `vm/` module

---

### Phase 5: Verification & Hardening (Week 7-8)
**Goal**: Verify reproducibility across platforms, harden gates

**Workstreams**:
- **Cross-Platform Testing** (3 days)
- **Trusting Trust Mitigation** (3 days)
- **Performance Benchmarking** (2 days)
- **Final Documentation** (2 days)

**Testing**:
- Build poseidon on Ubuntu 24.04, macOS 14, Alpine Linux
- Verify bit-identical SOBC execution across platforms
- Document "Trusting Trust" attack mitigation strategy

**Gate**: Reproducible bootstrap on 3+ platforms

---

## Detailed File-by-File Implementation

### Phase 0: IR Serialization

#### File: `self-hosted/ir/serialize.sio` (NEW, ~300 LOC)

```sio
// Binary IR serialization format (SOIR v1)
//
// Header:
//   Magic: "SOIR" (4 bytes)
//   Version: 1 (1 byte)
// Body:
//   fn_count: i64
//   functions: [IrFunction; fn_count]
//   string_count: i64
//   strings: [(len: i64, bytes: [i8; len]); string_count]

fn serialize_ir_module(module: IrModule) -> ([i8; 131072], i64) with Mut, Panic, Div {
    var buf: [i8; 131072] = [0; 131072]
    var pos: i64 = 0

    // Magic
    buf[0] = 83  // 'S'
    buf[1] = 79  // 'O'
    buf[2] = 73  // 'I'
    buf[3] = 82  // 'R'
    pos = 4

    // Version
    buf[4] = 1
    pos = 5

    // fn_count
    write_i64(&buf, pos, module.fn_count)
    pos = pos + 8

    // Serialize each function
    var i: i64 = 0
    while i < module.fn_count {
        let fn_start = pos
        pos = serialize_ir_function(&buf, pos, get_function(module, i))
        i = i + 1
    }

    // string_count
    write_i64(&buf, pos, module.string_count)
    pos = pos + 8

    // Serialize strings
    i = 0
    while i < module.string_count {
        let s = get_string(module, i)
        write_i64(&buf, pos, s.len)
        pos = pos + 8
        copy_bytes(&buf, pos, s.buf, s.len)
        pos = pos + s.len
        i = i + 1
    }

    (buf, pos)
}

fn deserialize_ir_module(buf: [i8; 131072], len: i64) -> IrModule with Mut, Panic, Div {
    // Verify magic
    if buf[0] != 83 || buf[1] != 79 || buf[2] != 73 || buf[3] != 82 {
        panic("Invalid SOIR magic")
    }

    // Check version
    let version = buf[4]
    if version != 1 {
        panic("Unsupported SOIR version")
    }

    var pos: i64 = 5

    // Read fn_count
    let fn_count = read_i64(buf, pos)
    pos = pos + 8

    var module = IrModule::new()

    // Deserialize functions
    var i: i64 = 0
    while i < fn_count {
        let (fn_obj, new_pos) = deserialize_ir_function(buf, pos)
        add_function(&module, fn_obj)
        pos = new_pos
        i = i + 1
    }

    // Read string_count
    let string_count = read_i64(buf, pos)
    pos = pos + 8

    // Deserialize strings
    i = 0
    while i < string_count {
        let str_len = read_i64(buf, pos)
        pos = pos + 8
        var str_buf: [i8; 256] = [0; 256]
        copy_bytes(&str_buf, 0, &buf, pos, str_len)
        add_string(&module, str_buf, str_len)
        pos = pos + str_len
        i = i + 1
    }

    module
}

// Helper: serialize single function
fn serialize_ir_function(buf: [i8; 131072], pos: i64, func: IrFunction) -> i64 {
    var p = pos

    // name_len + name_bytes
    write_i64(buf, p, func.name.len)
    p = p + 8
    copy_bytes(buf, p, func.name.buf, func.name.len)
    p = p + func.name.len

    // instr_count
    write_i64(buf, p, func.instr_count)
    p = p + 8

    // reg_count, label_count, param_count
    write_i64(buf, p, func.reg_count)
    p = p + 8
    write_i64(buf, p, func.label_count)
    p = p + 8
    write_i64(buf, p, func.param_count)
    p = p + 8

    // params (array of i64)
    var i: i64 = 0
    while i < func.param_count {
        write_i64(buf, p, func.params[i])
        p = p + 8
        i = i + 1
    }

    // Instructions
    i = 0
    while i < func.instr_count {
        p = serialize_ir_instr(buf, p, get_instr(func, i))
        i = i + 1
    }

    p
}

// Helper: serialize single instruction
fn serialize_ir_instr(buf: [i8; 131072], pos: i64, instr: IrInstr) -> i64 {
    var p = pos

    // Opcode (1 byte)
    buf[p] = instr.op as i8
    p = p + 1

    // Fields (fixed size for simplicity)
    write_i64(buf, p, instr.dst)
    p = p + 8
    write_i64(buf, p, instr.src1)
    p = p + 8
    write_i64(buf, p, instr.src2)
    p = p + 8
    write_i64(buf, p, instr.imm_i64)
    p = p + 8
    write_f64(buf, p, instr.imm_f64)
    p = p + 8
    write_i64(buf, p, instr.label_id)
    p = p + 8
    write_i64(buf, p, instr.fn_id)
    p = p + 8
    write_i64(buf, p, instr.field_idx)
    p = p + 8
    buf[p] = instr.bin_op as i8
    p = p + 1
    buf[p] = instr.un_op as i8
    p = p + 1

    // Name (length + bytes)
    write_i64(buf, p, instr.name.len)
    p = p + 8
    copy_bytes(buf, p, instr.name.buf, instr.name.len)
    p = p + 128  // Fixed size name buffer

    // arg_count (for calls)
    write_i64(buf, p, instr.arg_count)
    p = p + 8

    // TODO: Serialize IrRegList if needed (currently simplified)

    p
}

// Low-level I/O helpers
fn write_i64(buf: [i8; 131072], pos: i64, val: i64) with Mut, Panic, Div {
    buf[pos + 0] = (val >> 0) as i8
    buf[pos + 1] = (val >> 8) as i8
    buf[pos + 2] = (val >> 16) as i8
    buf[pos + 3] = (val >> 24) as i8
    buf[pos + 4] = (val >> 32) as i8
    buf[pos + 5] = (val >> 40) as i8
    buf[pos + 6] = (val >> 48) as i8
    buf[pos + 7] = (val >> 56) as i8
}

fn read_i64(buf: [i8; 131072], pos: i64) -> i64 {
    let b0 = buf[pos + 0] as i64 & 0xFF
    let b1 = buf[pos + 1] as i64 & 0xFF
    let b2 = buf[pos + 2] as i64 & 0xFF
    let b3 = buf[pos + 3] as i64 & 0xFF
    let b4 = buf[pos + 4] as i64 & 0xFF
    let b5 = buf[pos + 5] as i64 & 0xFF
    let b6 = buf[pos + 6] as i64 & 0xFF
    let b7 = buf[pos + 7] as i64 & 0xFF
    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
}

fn write_f64(buf: [i8; 131072], pos: i64, val: f64) with Mut, Panic, Div {
    // IEEE 754 double precision (8 bytes)
    // For simplicity, transmute via i64 (requires unsafe or intrinsic)
    let bits = f64_to_bits(val)
    write_i64(buf, pos, bits)
}

fn read_f64(buf: [i8; 131072], pos: i64) -> f64 {
    let bits = read_i64(buf, pos)
    f64_from_bits(bits)
}

fn copy_bytes(dst: [i8; 131072], dst_pos: i64, src: [i8; N], src_pos: i64, len: i64) with Mut, Panic, Div {
    var i: i64 = 0
    while i < len {
        dst[dst_pos + i] = src[src_pos + i]
        i = i + 1
    }
}
```

**Tests**: `self-hosted/test_ir_serialize.sio`
- Round-trip: deserialize(serialize(ir)) == ir
- Version compatibility
- Buffer overflow detection

---

#### File: `self-hosted/ir/normalize.sio` (NEW, ~200 LOC)

```sio
// IR normalization for deterministic comparison
//
// Normalization rules:
// 1. Sort functions by name (lexicographic)
// 2. Renumber labels by first definition (IrLabel instruction position)
// 3. Renumber virtual registers by first use (assignment or operand)
// 4. Sort string table alphabetically

fn normalize_ir_module(module: IrModule) -> IrModule with Mut, Panic, Div, Alloc {
    var normalized = IrModule::new()

    // 1. Sort functions by name
    let sorted_functions = sort_functions_by_name(module)

    // 2. For each function, normalize labels and registers
    var i: i64 = 0
    while i < module.fn_count {
        let func = sorted_functions[i]
        let norm_func = normalize_function(func)
        add_function(&normalized, norm_func)
        i = i + 1
    }

    // 3. Sort string table
    normalized.string_table = sort_string_table(module.string_table)
    normalized.string_count = module.string_count

    normalized
}

fn sort_functions_by_name(module: IrModule) -> [IrFunction; 64] with Mut, Panic, Div {
    // Simple bubble sort (self-hosted suite has ~16 functions, perf is fine)
    var funcs: [IrFunction; 64] = [IrFunction::empty(); 64]
    var i: i64 = 0
    while i < module.fn_count {
        funcs[i] = get_function(module, i)
        i = i + 1
    }

    // Bubble sort by name
    var n = module.fn_count
    var swapped = true
    while swapped {
        swapped = false
        i = 0
        while i < n - 1 {
            if name_compare(funcs[i].name, funcs[i + 1].name) > 0 {
                let tmp = funcs[i]
                funcs[i] = funcs[i + 1]
                funcs[i + 1] = tmp
                swapped = true
            }
            i = i + 1
        }
        n = n - 1
    }

    funcs
}

fn normalize_function(func: IrFunction) -> IrFunction with Mut, Panic, Div, Alloc {
    var norm_func = func

    // 1. Build label map: old_id -> first_definition_position
    let label_map = build_label_map(func)

    // 2. Renumber labels by first definition
    norm_func = renumber_labels(norm_func, label_map)

    // 3. Build register map: old_vreg -> first_use_order
    let reg_map = build_register_map(norm_func)

    // 4. Renumber registers by first use
    norm_func = renumber_registers(norm_func, reg_map)

    norm_func
}

fn build_label_map(func: IrFunction) -> [i64; 256] with Mut, Panic, Div {
    var label_map: [i64; 256] = [-1; 256]  // old_id -> definition_position
    var next_label_id: i64 = 0

    var i: i64 = 0
    while i < func.instr_count {
        let instr = get_instr(func, i)
        if instr.op == IrOpcode::IrLabel {
            let old_id = instr.label_id
            if label_map[old_id] == -1 {
                label_map[old_id] = next_label_id
                next_label_id = next_label_id + 1
            }
        }
        i = i + 1
    }

    label_map
}

fn renumber_labels(func: IrFunction, label_map: [i64; 256]) -> IrFunction with Mut, Panic, Div {
    var new_func = func

    var i: i64 = 0
    while i < func.instr_count {
        var instr = get_instr_mut(&new_func, i)

        // Update label_id field
        if instr.label_id >= 0 && instr.label_id < 256 {
            let new_id = label_map[instr.label_id]
            if new_id >= 0 {
                instr.label_id = new_id
            }
        }

        i = i + 1
    }

    new_func
}

fn build_register_map(func: IrFunction) -> [i64; 1024] with Mut, Panic, Div {
    var reg_map: [i64; 1024] = [-1; 1024]  // old_vreg -> new_vreg
    var next_reg: i64 = 0

    var i: i64 = 0
    while i < func.instr_count {
        let instr = get_instr(func, i)

        // Check dst, src1, src2
        if instr.dst >= 0 && instr.dst < 1024 && reg_map[instr.dst] == -1 {
            reg_map[instr.dst] = next_reg
            next_reg = next_reg + 1
        }
        if instr.src1 >= 0 && instr.src1 < 1024 && reg_map[instr.src1] == -1 {
            reg_map[instr.src1] = next_reg
            next_reg = next_reg + 1
        }
        if instr.src2 >= 0 && instr.src2 < 1024 && reg_map[instr.src2] == -1 {
            reg_map[instr.src2] = next_reg
            next_reg = next_reg + 1
        }

        i = i + 1
    }

    reg_map
}

fn renumber_registers(func: IrFunction, reg_map: [i64; 1024]) -> IrFunction with Mut, Panic, Div {
    var new_func = func

    var i: i64 = 0
    while i < func.instr_count {
        var instr = get_instr_mut(&new_func, i)

        if instr.dst >= 0 && instr.dst < 1024 && reg_map[instr.dst] >= 0 {
            instr.dst = reg_map[instr.dst]
        }
        if instr.src1 >= 0 && instr.src1 < 1024 && reg_map[instr.src1] >= 0 {
            instr.src1 = reg_map[instr.src1]
        }
        if instr.src2 >= 0 && instr.src2 < 1024 && reg_map[instr.src2] >= 0 {
            instr.src2 = reg_map[instr.src2]
        }

        i = i + 1
    }

    new_func
}

fn sort_string_table(table: [StringEntry; 1024]) -> [StringEntry; 1024] with Mut, Panic, Div {
    // Bubble sort strings alphabetically
    var sorted = table
    var n: i64 = count_strings(table)
    var swapped = true

    while swapped {
        swapped = false
        var i: i64 = 0
        while i < n - 1 {
            if string_compare(sorted[i], sorted[i + 1]) > 0 {
                let tmp = sorted[i]
                sorted[i] = sorted[i + 1]
                sorted[i + 1] = tmp
                swapped = true
            }
            i = i + 1
        }
        n = n - 1
    }

    sorted
}

// Helper: lexicographic name comparison
fn name_compare(a: Name, b: Name) -> i64 {
    let min_len = if a.len < b.len { a.len } else { b.len }
    var i: i64 = 0
    while i < min_len {
        if a.buf[i] < b.buf[i] { return -1 }
        if a.buf[i] > b.buf[i] { return 1 }
        i = i + 1
    }
    if a.len < b.len { -1 } else if a.len > b.len { 1 } else { 0 }
}
```

**Tests**: `self-hosted/test_ir_normalize.sio`
- Idempotence: normalize(normalize(ir)) == normalize(ir)
- Determinism: normalize(ir1) == normalize(ir2) if semantically equivalent
- Label renumbering correctness (jumps still target correct labels)
- Register renumbering correctness (SSA form preserved)

---

### Phase 1: verify.sio Integration

#### File: `stdlib/compiler/bootstrap/verify.sio` (MODIFY, +200 LOC)

**Changes**:

1. Update `BootstrapResult` structure:

```sio
struct BootstrapResult {
    stage: i32,
    ir_module: IrModule,              // NEW: IR artifact
    ir_serialized: [i8; 131072],      // NEW: serialized IR
    ir_len: i64,                      // NEW: serialized length

    // Keep old fields for backward compat
    code: [i8; 65536],
    code_len: i64,

    confidence: f32,
    provenance: [ProvenanceStep; 16],
    n_provenance: i64,
}
```

2. Implement IR-based cross-validation:

```sio
fn cross_validate_ir(stage1: BootstrapResult, stage2: BootstrapResult) -> VerificationResult {
    // 1. Normalize both IR modules
    let norm1 = normalize_ir_module(stage1.ir_module)
    let norm2 = normalize_ir_module(stage2.ir_module)

    // 2. Serialize normalized modules
    let (bytes1, len1) = serialize_ir_module(norm1)
    let (bytes2, len2) = serialize_ir_module(norm2)

    // 3. Quick hash check first
    let hash1 = code_hash(bytes1, len1)
    let hash2 = code_hash(bytes2, len2)

    var result = VerificationResult {
        verified: false,
        byte_match: false,
        semantic_match: false,
        confidence_boost: 0.0,
        diff_report: DiffReport::empty(),
    }

    if hash1 == hash2 {
        // Hashes match → byte-identical
        result.byte_match = true
        result.semantic_match = true
        result.verified = true
        result.confidence_boost = 4.0  // 4% boost per design doc

        print("✅ REPRODUCIBILITY VERIFIED\n")
        print("  Stage 1 ≡ Stage 2 (byte-identical IR after normalization)\n")
        print("  Hash: ")
        print_hex(hash1)
        print("\n")
    } else {
        // Hashes differ → drill down
        result.byte_match = false

        print("❌ BYTE-LEVEL MISMATCH\n")
        print("  Stage1 hash: ")
        print_hex(hash1)
        print("\n  Stage2 hash: ")
        print_hex(hash2)
        print("\n")

        // Drill down into semantic differences
        let semantic_diff = compare_ir_semantic(norm1, norm2)
        result.semantic_match = semantic_diff.is_equivalent
        result.diff_report = semantic_diff.report

        if !result.semantic_match {
            print("  Semantic differences detected:\n")
            print_diff_report(semantic_diff.report)
        }
    }

    result
}

fn compare_ir_semantic(ir1: IrModule, ir2: IrModule) -> SemanticComparison {
    var comp = SemanticComparison {
        is_equivalent: true,
        report: DiffReport::empty(),
    }

    // Compare function counts
    if ir1.fn_count != ir2.fn_count {
        comp.is_equivalent = false
        comp.report.diff_type = DIFF_FN_COUNT
        comp.report.expected = ir1.fn_count
        comp.report.actual = ir2.fn_count
        return comp
    }

    // Compare each function
    var i: i64 = 0
    while i < ir1.fn_count {
        let f1 = get_function(ir1, i)
        let f2 = get_function(ir2, i)

        // Names should match (already sorted)
        if !name_eq(f1.name, f2.name) {
            comp.is_equivalent = false
            comp.report.diff_type = DIFF_FN_NAME
            comp.report.fn_index = i
            comp.report.expected_name = f1.name
            comp.report.actual_name = f2.name
            return comp
        }

        // Instruction counts
        if f1.instr_count != f2.instr_count {
            comp.is_equivalent = false
            comp.report.diff_type = DIFF_INSTR_COUNT
            comp.report.fn_index = i
            comp.report.fn_name = f1.name
            comp.report.expected = f1.instr_count
            comp.report.actual = f2.instr_count
            return comp
        }

        // Compare each instruction
        var j: i64 = 0
        while j < f1.instr_count {
            let instr1 = get_instr(f1, j)
            let instr2 = get_instr(f2, j)

            if !instr_eq(instr1, instr2) {
                comp.is_equivalent = false
                comp.report.diff_type = DIFF_INSTR
                comp.report.fn_index = i
                comp.report.fn_name = f1.name
                comp.report.instr_index = j
                comp.report.instr1 = instr1
                comp.report.instr2 = instr2
                return comp
            }

            j = j + 1
        }

        i = i + 1
    }

    comp
}

fn print_diff_report(report: DiffReport) with IO {
    if report.diff_type == DIFF_FN_COUNT {
        print("  Function count mismatch: expected ")
        print_int(report.expected)
        print(", got ")
        print_int(report.actual)
        print("\n")
    } else if report.diff_type == DIFF_FN_NAME {
        print("  Function ")
        print_int(report.fn_index)
        print(" name mismatch\n")
        print("    Expected: ")
        print_name(report.expected_name)
        print("\n    Got: ")
        print_name(report.actual_name)
        print("\n")
    } else if report.diff_type == DIFF_INSTR_COUNT {
        print("  Function '")
        print_name(report.fn_name)
        print("': instruction count mismatch\n")
        print("    Expected ")
        print_int(report.expected)
        print(" instructions, got ")
        print_int(report.actual)
        print("\n")
    } else if report.diff_type == DIFF_INSTR {
        print("  Function '")
        print_name(report.fn_name)
        print("', instr ")
        print_int(report.instr_index)
        print(": MISMATCH\n")
        print("    Stage1: ")
        print_instr(report.instr1)
        print("\n    Stage2: ")
        print_instr(report.instr2)
        print("\n")
    }
}
```

---

#### File: `stdlib/compiler/bootstrap/driver.sio` (MODIFY, +150 LOC)

**Add Stage 1/2 orchestration functions**:

```sio
// Compile source using Rust compiler (Stage 1)
fn compile_stage1(source_path: string) -> CompileArtifact with Mut, Panic, Div, Alloc, IO {
    print("[Stage 1] Compiling with Rust compiler: ")
    print(source_path)
    print("\n")

    // Call Rust compiler bridge (via FFI)
    let rust_artifact = rust_compile_to_ir(source_path)

    // Convert Rust HIR → Sounio IrModule (already done in compiler_loader.rs)
    rust_artifact
}

// Compile source using Stage 1 compiler (Stage 2)
fn compile_stage2(stage1_artifact: CompileArtifact, source_path: string)
    -> CompileArtifact with Mut, Panic, Div, Alloc, IO {
    print("[Stage 2] Compiling with Stage 1 self-hosted compiler: ")
    print(source_path)
    print("\n")

    // Load Stage 1 bytecode into VM
    let stage1_bytecode = stage1_artifact.code
    let stage1_len = stage1_artifact.code_len

    // Execute Stage 1 compiler's compile_file() entrypoint
    // This requires VM interop (either via existing VM or new C VM)
    let stage2_artifact = vm_execute_compile(stage1_bytecode, stage1_len, source_path)

    stage2_artifact
}

// Two-stage bootstrap orchestrator
fn run_two_stage_bootstrap(source_path: string)
    -> (CompileArtifact, CompileArtifact, VerificationResult) with Mut, Panic, Div, Alloc, IO {

    print("=== Two-Stage Bootstrap ===\n")

    // Stage 1: Rust compiler
    let stage1 = compile_stage1(source_path)
    let stage1_result = BootstrapResult {
        stage: STAGE_RUST,
        ir_module: extract_ir(stage1),
        ir_serialized: serialize_ir_module(extract_ir(stage1)).0,
        ir_len: serialize_ir_module(extract_ir(stage1)).1,
        code: stage1.code,
        code_len: stage1.code_len,
        confidence: 1.0,
        provenance: build_provenance_chain(STAGE_RUST),
        n_provenance: 1,
    }

    // Stage 2: Self-hosted compiler (using Stage 1)
    let stage2 = compile_stage2(stage1, source_path)
    let stage2_result = BootstrapResult {
        stage: STAGE_SELF1,
        ir_module: extract_ir(stage2),
        ir_serialized: serialize_ir_module(extract_ir(stage2)).0,
        ir_len: serialize_ir_module(extract_ir(stage2)).1,
        code: stage2.code,
        code_len: stage2.code_len,
        confidence: 0.98,  // 2% degradation per stage (per design doc)
        provenance: build_provenance_chain(STAGE_SELF1),
        n_provenance: 2,
    }

    // Cross-validation
    let verification = cross_validate_ir(stage1_result, stage2_result)

    print("\n=== Verification Result ===\n")
    if verification.verified {
        print("✅ Bootstrap verified: Stage 1 ≡ Stage 2\n")
        let boosted_confidence = apply_cross_validation_boost(
            stage2_result.confidence,
            verification.confidence_boost
        )
        print("  Final confidence: ")
        print_f32(boosted_confidence)
        print(" (boosted by ")
        print_f32(verification.confidence_boost)
        print("%)\n")
    } else {
        print("❌ Bootstrap FAILED: Stage 1 ≢ Stage 2\n")
        print("  Review diff report above for details.\n")
    }

    (stage1, stage2, verification)
}
```

---

### Phase 2A: C-based Stage 0 VM (poseidon)

#### File: `stage0_vm.c` (NEW, ~1,000 LOC)

**Structure** (abbreviated for plan document):

```c
/* poseidon - Sounio Stage 0 Bytecode VM
 * Minimal trusted bootstrap base (~1,000 LOC C99)
 *
 * Features:
 * - Loads SOBC v1 bytecode files
 * - Executes bytecode VM instructions
 * - Minimal FFI (print, file I/O, exit)
 * - Static linking (no runtime dependencies)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

/* Bytecode format (matches Rust vm/bytecode.rs) */
#define BYTECODE_MAGIC 0x43424F53  /* "SOBC" little-endian */
#define BYTECODE_VERSION 1

typedef enum {
    OP_PUSH_UNIT = 0,
    OP_PUSH_BOOL,
    OP_PUSH_INT,
    OP_PUSH_FLOAT,
    OP_POP,
    OP_DUP,
    OP_SWAP,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_REM,
    OP_NEG,
    OP_NOT,
    OP_EQ,
    OP_NE,
    OP_LT,
    OP_LE,
    OP_GT,
    OP_GE,
    OP_AND,
    OP_OR,
    OP_JUMP,
    OP_JUMP_IF,
    OP_CALL,
    OP_RET,
    /* ... ~30 total opcodes */
} Opcode;

typedef struct {
    uint8_t tag;  /* 0=Unit, 1=Bool, 2=Int, 3=Float, 4=Ptr */
    union {
        int64_t i;
        double f;
        void *p;
    } data;
} Value;

typedef struct {
    Value *stack;
    size_t stack_len;
    size_t stack_cap;

    Value *locals;
    size_t locals_cap;

    uint8_t *heap;
    size_t heap_size;
    size_t heap_used;

    /* Call frames */
    struct CallFrame *frames;
    size_t frame_len;
    size_t frame_cap;
} VM;

/* VM operations */
VM* vm_new(void);
void vm_free(VM *vm);
Value vm_pop(VM *vm);
void vm_push(VM *vm, Value v);
int vm_execute(VM *vm, const uint8_t *bytecode, size_t len);

/* Bytecode loading */
typedef struct {
    uint32_t magic;
    uint8_t version;
    uint8_t *instructions;
    size_t instr_len;
} Bytecode;

Bytecode* load_bytecode(const char *path);
void free_bytecode(Bytecode *bc);

/* FFI stubs */
void ffi_print_int(int64_t x) {
    printf("%lld", x);
}

void ffi_print_char(char c) {
    putchar(c);
}

void ffi_print_str(const char *s) {
    fputs(s, stdout);
}

int64_t ffi_read_file(const char *path, uint8_t *buf, size_t buf_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t n = fread(buf, 1, buf_len, f);
    fclose(f);
    return (int64_t)n;
}

/* Main interpreter loop */
int vm_execute(VM *vm, const uint8_t *bytecode, size_t len) {
    size_t ip = 0;

    while (ip < len) {
        uint8_t op = bytecode[ip++];

        switch (op) {
            case OP_PUSH_INT: {
                int64_t val = read_i64_le(&bytecode[ip]);
                ip += 8;
                Value v = {.tag = 2, .data.i = val};
                vm_push(vm, v);
                break;
            }

            case OP_PUSH_FLOAT: {
                double val = read_f64_le(&bytecode[ip]);
                ip += 8;
                Value v = {.tag = 3, .data.f = val};
                vm_push(vm, v);
                break;
            }

            case OP_ADD: {
                Value b = vm_pop(vm);
                Value a = vm_pop(vm);
                if (a.tag == 2 && b.tag == 2) {
                    Value result = {.tag = 2, .data.i = a.data.i + b.data.i};
                    vm_push(vm, result);
                } else if (a.tag == 3 && b.tag == 3) {
                    Value result = {.tag = 3, .data.f = a.data.f + b.data.f};
                    vm_push(vm, result);
                } else {
                    fprintf(stderr, "Type error in OP_ADD\n");
                    return 1;
                }
                break;
            }

            /* ... implement all 30+ opcodes ... */

            default:
                fprintf(stderr, "Unknown opcode: %d at offset %zu\n", op, ip - 1);
                return 1;
        }
    }

    return 0;
}

/* Bytecode deserialization */
Bytecode* load_bytecode(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("Failed to open bytecode file");
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *data = malloc(file_size);
    if (!data) {
        fclose(f);
        return NULL;
    }

    if (fread(data, 1, file_size, f) != (size_t)file_size) {
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);

    /* Verify magic and version */
    uint32_t magic = read_u32_le(data);
    if (magic != BYTECODE_MAGIC) {
        fprintf(stderr, "Invalid bytecode magic: 0x%08X (expected 0x%08X)\n",
                magic, BYTECODE_MAGIC);
        free(data);
        return NULL;
    }

    uint8_t version = data[4];
    if (version != BYTECODE_VERSION) {
        fprintf(stderr, "Unsupported bytecode version: %d (expected %d)\n",
                version, BYTECODE_VERSION);
        free(data);
        return NULL;
    }

    Bytecode *bc = malloc(sizeof(Bytecode));
    bc->magic = magic;
    bc->version = version;
    bc->instructions = data + 5;
    bc->instr_len = file_size - 5;

    return bc;
}

/* Main entry point */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <bytecode.sobc> [args...]\n", argv[0]);
        return 1;
    }

    const char *bytecode_path = argv[1];

    /* Load bytecode */
    Bytecode *bc = load_bytecode(bytecode_path);
    if (!bc) {
        return 1;
    }

    /* Create VM */
    VM *vm = vm_new();

    /* Execute */
    int result = vm_execute(vm, bc->instructions, bc->instr_len);

    /* Cleanup */
    vm_free(vm);
    free_bytecode(bc);

    return result;
}
```

**Makefile**:

```makefile
CC = gcc
CFLAGS = -std=c99 -O2 -Wall -Wextra -Werror -pedantic -static
LDFLAGS = -lm
TARGET = poseidon

$(TARGET): stage0_vm.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: test
test: $(TARGET)
	./$(TARGET) tests/stage0_smoke.sobc

.PHONY: install
install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

.PHONY: clean
clean:
	rm -f $(TARGET)
```

---

### Phase 2B: Documentation

#### File: `docs/SELFHOST_CI_GATES.md` (NEW, ~1,500 words)

**Table of Contents**:
1. Overview
2. Gate Architecture
3. Environment Variables Reference
4. Individual Gate Descriptions
   - Zero-Fallback Gate
   - Driver-Output Gate
   - Driver-Output-Parity Gate
   - Reproducibility Gate (NEW)
   - Verify Gate (NEW)
5. Running Gates Locally
6. Interpreting Results
7. Troubleshooting
8. Extending Gates

**Key sections** (abbreviated):

```markdown
# Self-Hosted CI Gates

## Overview

Sounio's rustless cutover strategy relies on a series of CI gates to ensure
the self-hosted compiler is correct and reproducible. This document describes
each gate, how to run them locally, and how to interpret failures.

## Gate Architecture

```
┌──────────────────────┐
│ Zero-Fallback Gate   │  ← Verifies no silent fallback to Rust
└──────────┬───────────┘
           │ passes
           ↓
┌──────────────────────┐
│ Driver-Output Gate   │  ← Verifies driver produces artifacts
└──────────┬───────────┘
           │ passes
           ↓
┌──────────────────────┐
│ Reproducibility Gate │  ← Verifies deterministic compilation
└──────────┬───────────┘
           │ passes
           ↓
┌──────────────────────┐
│ Verify Gate          │  ← Verifies Stage 1 ≡ Stage 2 (IR equivalence)
└──────────────────────┘
```

## Environment Variables Reference

| Variable | Values | Purpose |
|----------|--------|---------|
| `SOUNIO_SELFHOST_STRICT` | 0, 1 | Enable strict mode (no fallback) |
| `SOUNIO_SELFHOST_NO_RUST_FALLBACK` | 0, 1 | Block Rust fallback entirely |
| `SOUNIO_SELFHOST_PIPELINE` | driver, rust | Select compilation pipeline |
| `SOURCE_DATE_EPOCH` | Unix timestamp | Fix timestamp for reproducibility |
| `SOUNIO_REPRODUCIBLE_BUILD` | 0, 1 | Enable reproducible build mode |

## Individual Gate Descriptions

### Zero-Fallback Gate

**Purpose**: Verify that the self-hosted compiler executes without falling back
to the Rust oracle.

**Command**:
```bash
bash scripts/selfhost_zero_fallback_gate.sh
```

**Success criteria**:
- No `status=fallback` markers in logs
- No `SELFHOST=oracle` markers
- Parse-all suite passes

**Failure modes**:
- Compiler crash → self-hosted bug
- Silent fallback → strict mode not enabled
- Parse failure → lexer/parser bug

### Reproducibility Gate (NEW)

**Purpose**: Verify that compiling the same input twice produces byte-identical
bytecode artifacts.

**Command**:
```bash
bash scripts/selfhost_reproducibility_gate.sh
```

**Success criteria**:
- Two runs produce identical `.sobc` files
- SHA256 checksums match
- No timestamp or environment leakage

**Failure modes**:
- Non-deterministic label allocation
- Random seed not fixed
- Absolute path embedding

### Verify Gate (NEW)

**Purpose**: Verify that Stage 1 (Rust-compiled self-hosted compiler) produces
the same IR output as Stage 2 (self-hosted compiler compiling itself).

**Command**:
```bash
bash scripts/selfhost_verify_gate.sh
```

**Success criteria**:
- Stage 1 IR ≡ Stage 2 IR (byte-identical after normalization)
- Semantic equivalence check passes
- Confidence boost applied (4%)

**Failure modes**:
- IR divergence → compiler bug
- Normalization failure → normalize.sio bug
- Serialization mismatch → serialize.sio bug

## Running Gates Locally

```bash
# Run all gates
./scripts/run_selfhost_gates_local.sh

# Run specific gate
./scripts/run_selfhost_gates_local.sh --gate=verify

# Quick mode (smaller corpus)
./scripts/run_selfhost_gates_local.sh --quick
```

## Troubleshooting

### "FAIL: Bytecode artifacts differ"

This indicates non-deterministic compilation. Check:
1. Is `SOURCE_DATE_EPOCH` set?
2. Are labels/registers renumbered correctly?
3. Is string table sorting enabled?

### "FAIL: Stage 1 ≢ Stage 2 (IR mismatch)"

This is a critical failure indicating a bootstrap bug:
1. Review diff report for first divergence
2. Check HIR lowering for the affected function
3. Verify optimizer didn't introduce non-determinism

...
```

---

#### File: `scripts/selfhost_reproducibility_gate.sh` (NEW, ~250 LOC)

```bash
#!/usr/bin/env bash
set -euo pipefail

# Reproducibility gate: Verify deterministic compilation
# Runs compilation twice with identical inputs, compares bytecode artifacts

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-repro-gate}"
RUN1_DIR="$WORK_DIR/run1"
RUN2_DIR="$WORK_DIR/run2"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"

TIMEOUT_SECS="${TIMEOUT_SECS:-900}"
BUILD_TIMEOUT_SECS="${BUILD_TIMEOUT_SECS:-600}"
TARGET="${TARGET:-self-hosted/native/test_phase1.sio}"

# For reproducibility, fix all non-deterministic sources
export SOURCE_DATE_EPOCH=0
export SOUNIO_REPRODUCIBLE_BUILD=1
export SOUNIO_SELFHOST_STRICT=1
export SOUNIO_SELFHOST_NO_RUST_FALLBACK=1
export SOUNIO_SELFHOST_PIPELINE=driver
export SOUNIO_SELFHOST_NO_RUST_HARNESS=1

echo "=== Reproducibility Gate ==="
echo "Target: $TARGET"
echo "Work dir: $WORK_DIR"
echo ""

# Clean work directory
rm -rf "$WORK_DIR"
mkdir -p "$RUN1_DIR" "$RUN2_DIR" "$LOG_DIR" "$ARTIFACT_DIR"

# Build compiler
echo "[1/4] Building compiler..."
timeout "$BUILD_TIMEOUT_SECS" cargo build -p souc --release 2>&1 | tee "$LOG_DIR/build.log"

# Run 1
echo "[2/4] Running compilation (Run 1)..."
timeout "$TIMEOUT_SECS" cargo run --release --bin souc -- \
    run "$TARGET" \
    --emit-bytecode="$RUN1_DIR/output.sobc" \
    2>&1 | tee "$LOG_DIR/run1.log"

RUN1_EXIT=${PIPESTATUS[0]}
if [ "$RUN1_EXIT" -ne 0 ]; then
    echo "FAIL: Run 1 exited with code $RUN1_EXIT"
    exit 1
fi

# Run 2 (identical invocation)
echo "[3/4] Running compilation (Run 2)..."
timeout "$TIMEOUT_SECS" cargo run --release --bin souc -- \
    run "$TARGET" \
    --emit-bytecode="$RUN2_DIR/output.sobc" \
    2>&1 | tee "$LOG_DIR/run2.log"

RUN2_EXIT=${PIPESTATUS[0]}
if [ "$RUN2_EXIT" -ne 0 ]; then
    echo "FAIL: Run 2 exited with code $RUN2_EXIT"
    exit 1
fi

# Compare
echo "[4/4] Comparing artifacts..."

# Byte-level comparison
if cmp -s "$RUN1_DIR/output.sobc" "$RUN2_DIR/output.sobc"; then
    echo "✅ PASS: Bytecode artifacts are byte-identical"

    # Compute checksum for records
    CHECKSUM=$(sha256sum "$RUN1_DIR/output.sobc" | cut -d' ' -f1)
    echo "   SHA256: $CHECKSUM"
    echo "$CHECKSUM" > "$ARTIFACT_DIR/checksum.txt"

    exit 0
else
    echo "❌ FAIL: Bytecode artifacts differ"

    # Detailed diff
    echo ""
    echo "Checksums:"
    sha256sum "$RUN1_DIR/output.sobc" "$RUN2_DIR/output.sobc"

    echo ""
    echo "File sizes:"
    ls -lh "$RUN1_DIR/output.sobc" "$RUN2_DIR/output.sobc"

    # Hexdump for inspection
    hexdump -C "$RUN1_DIR/output.sobc" > "$ARTIFACT_DIR/run1.hex"
    hexdump -C "$RUN2_DIR/output.sobc" > "$ARTIFACT_DIR/run2.hex"

    echo ""
    echo "Hex diff (first 50 lines):"
    diff -u "$ARTIFACT_DIR/run1.hex" "$ARTIFACT_DIR/run2.hex" | head -50 || true

    echo ""
    echo "Full hex dumps saved to:"
    echo "  $ARTIFACT_DIR/run1.hex"
    echo "  $ARTIFACT_DIR/run2.hex"

    exit 1
fi
```

---

#### File: `scripts/selfhost_verify_gate.sh` (NEW, ~200 LOC)

```bash
#!/usr/bin/env bash
set -euo pipefail

# Verify gate: Stage 1 vs Stage 2 IR equivalence
# Compiles self-hosted suite with Rust (Stage 1), then with self-hosted
# compiler (Stage 2), and verifies semantic equivalence via verify.sio

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-verify-gate}"
TIMEOUT_SECS="${TIMEOUT_SECS:-900}"
TARGET="${TARGET:-self-hosted/}"

export SOUNIO_SELFHOST_STRICT=1
export SOUNIO_SELFHOST_NO_RUST_FALLBACK=1
export SOUNIO_SELFHOST_PIPELINE=driver

echo "=== Verify Gate: Stage 1 vs Stage 2 ==="
echo "Target: $TARGET"
echo "Work dir: $WORK_DIR"
echo ""

# Clean work directory
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

# Build compiler
echo "[1/3] Building Rust compiler..."
timeout "$TIMEOUT_SECS" cargo build -p souc --release 2>&1 | tee "$WORK_DIR/build.log"

# Run two-stage bootstrap via verify.sio
echo "[2/3] Running two-stage bootstrap..."
timeout "$TIMEOUT_SECS" cargo run --release --bin souc -- \
    run stdlib/compiler/bootstrap/verify.sio \
    -- "$TARGET" \
    2>&1 | tee "$WORK_DIR/verify.log"

VERIFY_EXIT=${PIPESTATUS[0]}

# Parse output for verification result
echo "[3/3] Checking verification result..."

if [ "$VERIFY_EXIT" -ne 0 ]; then
    echo "❌ FAIL: verify.sio exited with code $VERIFY_EXIT"
    exit 1
fi

# Check for success markers in log
if grep -q "✅ REPRODUCIBILITY VERIFIED" "$WORK_DIR/verify.log"; then
    echo "✅ PASS: Stage 1 ≡ Stage 2 (IR semantic equivalence)"

    # Extract confidence boost
    CONFIDENCE=$(grep -oP 'Final confidence: \K[0-9.]+' "$WORK_DIR/verify.log" || echo "unknown")
    echo "   Final confidence: $CONFIDENCE"

    exit 0
else
    echo "❌ FAIL: Stage 1 ≢ Stage 2 (IR mismatch)"

    # Extract diff report
    echo ""
    echo "Diff report:"
    grep -A 20 "Semantic differences:" "$WORK_DIR/verify.log" || echo "(no diff report found)"

    exit 1
fi
```

---

## Critical Success Factors

### 1. IR Stability

**Risk**: IrModule schema changes break serialization.

**Mitigation**:
- Version byte in SOIR format
- Backward compatibility layer
- Test suite for version migration

### 2. Normalization Correctness

**Risk**: Normalization introduces semantic changes.

**Mitigation**:
- Idempotence tests: `normalize(normalize(ir)) == normalize(ir)`
- Semantic preservation tests: execute before/after normalization, compare outputs
- Never reorder instructions, only rename IDs

### 3. C VM Correctness

**Risk**: poseidon VM diverges from Rust VM semantics.

**Mitigation**:
- Parity testing: run same bytecode on both VMs, compare outputs
- Test corpus: all existing self-hosted tests
- Fuzzing: generate random bytecode, check for crashes/divergence

### 4. CI Integration

**Risk**: Gates are flaky or timeout in CI.

**Mitigation**:
- Generous timeouts (15 minutes)
- Retry logic for transient failures
- Artifact upload for debugging

---

## Timeline and Resources

### Estimated Effort

| Phase | Duration | LOC | Risk |
|-------|----------|-----|------|
| Phase 0: IR Serialization | 1 week | ~500 | Low |
| Phase 1: Verification Pipeline | 1 week | ~400 | Medium |
| Phase 2A: C VM | 2 weeks | ~1000 | Medium |
| Phase 2B: Documentation | 2 weeks | ~2000 words | Low |
| Phase 3: Integration | 1 week | ~200 | Low |
| Phase 4: Cleanup | 1 week | ~100 | Low |
| Phase 5: Hardening | 2 weeks | ~300 | Medium |
| **Total** | **8 weeks** | **~2500 LOC** | - |

### Parallelization Opportunities

- **Phase 2A and 2B can run in parallel** (C VM + documentation)
- **IR serialization and normalization** can be split between developers
- **Gate scripts** can be developed independently

### Resource Requirements

- **1-2 developers** for implementation
- **CI resources**: ~30 minutes per full gate run
- **Test infrastructure**: Existing self-hosted test suite

---

## Next Steps

1. **Approve this plan** → Get stakeholder buy-in
2. **Phase 0** → Start with IR serialization (Week 1)
3. **Weekly checkpoints** → Review progress, adjust timeline
4. **Phase 3 milestone** → poseidon replaces Rust VM (critical gate)
5. **Phase 5 completion** → Full rustless cutover verified

---

## Appendix: File Checklist

### Files to Create

- [ ] `self-hosted/ir/serialize.sio` (~300 LOC)
- [ ] `self-hosted/ir/normalize.sio` (~200 LOC)
- [ ] `self-hosted/test_ir_serialize.sio` (~200 LOC)
- [ ] `self-hosted/test_ir_normalize.sio` (~150 LOC)
- [ ] `stage0_vm.c` (~1000 LOC)
- [ ] `Makefile` (for poseidon)
- [ ] `scripts/selfhost_reproducibility_gate.sh` (~250 LOC)
- [ ] `scripts/selfhost_verify_gate.sh` (~200 LOC)
- [ ] `scripts/run_selfhost_gates_local.sh` (~400 LOC)
- [ ] `docs/SELFHOST_CI_GATES.md` (~1500 words)
- [ ] `docs/BOOTSTRAP_VERIFICATION.md` (~1000 words)

### Files to Modify

- [ ] `stdlib/compiler/bootstrap/verify.sio` (+200 LOC)
- [ ] `stdlib/compiler/bootstrap/driver.sio` (+150 LOC)
- [ ] `self-hosted/ir/ir.sio` (documentation updates)
- [ ] `.github/workflows/ci.yml` (add reproducibility + verify jobs)
- [ ] `crates/souc/src/main.rs` (add SOUNIO_USE_C_VM flag)
- [ ] `docs/SELF_HOSTING_PHASES.md` (update with new architecture)
- [ ] `.claude/pending.md` (mark items complete)

### Files to Deprecate (Phase 4)

- [ ] `crates/souc/src/vm/mod.rs` (mark deprecated)
- [ ] `crates/souc/src/vm/bytecode.rs` (mark deprecated)
- [ ] `crates/souc/src/vm/serialize.rs` (keep for reference, mark deprecated)

---

**End of Unified Plan**
