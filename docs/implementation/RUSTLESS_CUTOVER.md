<!-- docs:meta
topic_id: repo.docs.implementation.rustless-cutover
authority: historical
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.rustless-cutover
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Rustless Cutover - Sounio Self-Hosting Documentation

**Last Updated**: 2026-02-13

## Overview

The Rustless Cutover is Sounio's transition from Rust-dependent bootstrap to a fully self-hosted toolchain. This document describes the 3-stage bootstrap process, IR formats, verification pipeline, and workflow for working with the self-hosted compiler.

## Quick Navigation

- [3-Stage Bootstrap Process](#3-stage-bootstrap-process)
- [SOIR v1 Format](#soir-v1-format-specification)
- [IR Normalization](#ir-normalization-and-equivalence)
- [Verification Pipeline](#verification-pipeline-workflow)
- [Adding New Opcodes](#adding-new-opcodes)
- [Troubleshooting](#troubleshooting-guide)

## 3-Stage Bootstrap Process

Sounio uses a classical 3-stage bootstrap to achieve self-hosting:

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Stage 0   │         │   Stage 1   │         │   Stage 2   │
│  (Trusted)  │────────>│ (Bootstrap) │────────>│ (Verified)  │
└─────────────┘         └─────────────┘         └─────────────┘
     │                        │                        │
     │ Compiles               │ Compiles               │ Cross-
     │ Self-Hosted            │ Self-Hosted            │ Validation
     │ Compiler               │ Compiler Again         │
     v                        v                        v
  IrModule v1             IrModule v1             Stage1 ≡ Stage2?
```

### Stage 0: Trusted Bootstrap

**Purpose**: Minimal trusted computing base to compile the self-hosted compiler.

**Current Implementation**:
- Rust compiler (`souc` CLI binary) - temporary
- Future: `poseidon` C/Zig VM or self-hosted native binary

**Artifacts**:
- Input: `self-hosted/**/*.sio` (compiler source)
- Output: `stage1_compiler.soir` (SOIR v1 bytecode)

**Key Constraint**: Stage 0 must be as small as possible and eventually replaceable.

### Stage 1: Bootstrap Compiler

**Purpose**: Self-hosted compiler built by Stage 0.

**Execution**:
- Loaded from `stage1_compiler.soir`
- Runs on VM or as native binary
- Compiles the same self-hosted source code again

**Artifacts**:
- Input: `self-hosted/**/*.sio`
- Output: `stage2_compiler.soir`

**Validation**: Stage 1 must be functionally correct (can compile itself).

### Stage 2: Verified Compiler

**Purpose**: Self-hosted compiler built by Stage 1, proving reproducibility.

**Execution**:
- Loaded from `stage2_compiler.soir`
- Should be semantically identical to Stage 1

**Verification**:
```
if normalize(stage1_compiler.soir) == normalize(stage2_compiler.soir):
    ✓ Self-hosting achieved
    ✓ Stage 1 is now the canonical compiler
else:
    ✗ Bootstrap bug detected
    → Debug IR diff
```

**Gate**: CI must verify Stage 1 ≡ Stage 2 before accepting changes.

## SOIR v1 Format Specification

**SOIR** (Sounio Intermediate Representation) is the serialized bytecode format used for bootstrap artifacts.

### File Format

```
┌──────────────────────────────────────┐
│ Header (8 bytes)                     │
├──────────────────────────────────────┤
│ Magic: "SOIR" (0x534F4952)           │  4 bytes
│ Version: 1                           │  1 byte
│ Reserved: 0x00                       │  3 bytes padding
├──────────────────────────────────────┤
│ Body                                 │
├──────────────────────────────────────┤
│ fn_count: i64                        │  8 bytes
│ functions: [IrFunction; fn_count]    │  variable
│ string_count: i64                    │  8 bytes
│ strings: [Name; string_count]        │  variable
└──────────────────────────────────────┘
```

### IrFunction Encoding

```
┌──────────────────────────────────────┐
│ name: Name                           │  136 bytes (128 buf + 8 len)
│ instr_count: i64                     │  8 bytes
│ reg_count: i64                       │  8 bytes
│ label_count: i64                     │  8 bytes
│ param_count: i64                     │  8 bytes
│ param_regs: [i64; 64]                │  512 bytes
│ instrs: [IrInstr; instr_count]       │  variable (see below)
└──────────────────────────────────────┘
```

### IrInstr Encoding (Fixed 168 bytes per instruction)

```
┌──────────────────────────────────────┐
│ op: IrOpcode (1 byte)                │  1 byte
│ padding                              │  7 bytes
│ dst: i64                             │  8 bytes
│ src1: i64                            │  8 bytes
│ src2: i64                            │  8 bytes
│ imm_i64: i64                         │  8 bytes
│ imm_f64: f64                         │  8 bytes
│ label_id: i64                        │  8 bytes
│ fn_id: i64                           │  8 bytes
│ field_idx: i64                       │  8 bytes
│ bin_op: BinaryOp (1 byte)            │  1 byte
│ padding                              │  7 bytes
│ un_op: UnaryOp (1 byte)              │  1 byte
│ padding                              │  7 bytes
│ name: Name                           │  136 bytes
│ arg_count: i64                       │  8 bytes
└──────────────────────────────────────┘
Total: 8 + (3*7) + (9*8) + 136 + 8 = 237 bytes
```

### Opcode Table

| Code | Opcode | Description |
|------|--------|-------------|
| 0 | IrLoadImm | Load integer immediate |
| 1 | IrLoadFloat | Load float immediate |
| 2 | IrLoadBool | Load boolean immediate |
| 3 | IrLoadString | Load string literal |
| 4 | IrCopy | Register copy |
| 5 | IrBinOp | Binary operation (add, sub, mul, div, cmp, etc.) |
| 6 | IrUnaryOp | Unary operation (neg, not, ref, deref) |
| 7 | IrCall | Function call |
| 8 | IrReturn | Return from function |
| 9 | IrJump | Unconditional jump |
| 10 | IrBranchTrue | Branch if true |
| 11 | IrBranchFalse | Branch if false |
| 12 | IrFieldGet | Get struct field |
| 13 | IrFieldSet | Set struct field |
| 14 | IrIndexGet | Get array element |
| 15 | IrIndexSet | Set array element |
| 16 | IrAlloc | Heap allocation |
| 17 | IrLabel | Control flow label |
| 18 | IrNop | No operation |
| 19 | IrPhi | SSA phi node (future) |

### Binary Encoding Details

- All integers: little-endian i64
- Floats: IEEE 754 double (8 bytes)
- Names: length-prefixed (i64) + 128-byte fixed buffer
- Padding: ensures 8-byte alignment
- call_args: NOT serialized (reconstructed from arg_count + src1/src2)

### Size Limits

```sio
IR_MAX_FUNCS: 64        // Max functions per module
IR_MAX_STRINGS: 256     // Max string literals
IR_MAX_INSTRS: 2048     // Max instructions per function
IR_MAX_PARAMS: 64       // Max parameters per function
SOIR_MAX_SIZE: 131072   // 128KB max module size
```

## IR Normalization and Equivalence

### Why Normalization?

Two semantically equivalent programs may have different IR encodings:

1. Virtual register numbering may differ
2. Label IDs may be reordered
3. Unused instructions may be present (dead code)
4. String table ordering may vary

**Normalization** converts IR to a canonical form so that equivalent programs produce identical bytecode.

### Normalization Rules

#### 1. Virtual Register Renumbering

Map vregs to canonical order based on first use:
```
Before:  v5, v3, v7, v1
After:   v0, v1, v2, v3
```

#### 2. Label Renumbering

Map labels to canonical order based on first definition:
```
Before:  L10, L5, L20
After:   L0, L1, L2
```

#### 3. Dead Code Elimination

Remove unreachable instructions:
```
Before:
  ret v0
  add v1, v2, v3  // dead code (unreachable)

After:
  ret v0
```

#### 4. String Table Deduplication

Merge duplicate string literals:
```
Before:  ["hello", "world", "hello"]
After:   ["hello", "world"]
         (update IrLoadString indices)
```

#### 5. Function Ordering

Sort functions alphabetically by name:
```
Before:  [main, helper, add]
After:   [add, helper, main]
```

### Normalization Algorithm

```python
def normalize_ir(module: IrModule) -> IrModule:
    # 1. Sort functions alphabetically
    sorted_fns = sort_by_name(module.functions)

    # 2. For each function:
    for fn in sorted_fns:
        # Renumber vregs canonically
        vreg_map = build_vreg_map(fn)
        remap_vregs(fn, vreg_map)

        # Renumber labels canonically
        label_map = build_label_map(fn)
        remap_labels(fn, label_map)

        # Remove dead code
        remove_unreachable(fn)

    # 3. Deduplicate and sort string table
    string_map = build_string_map(module)
    remap_string_refs(module, string_map)

    return module
```

### Equivalence Checking

```bash
# CLI tool workflow:
sounio-verify compare stage1.sio stage2.sio

# Under the hood:
1. Compile stage1.sio → ir1
2. Compile stage2.sio → ir2
3. ir1_norm = normalize(ir1)
4. ir2_norm = normalize(ir2)
5. if ir1_norm == ir2_norm:
     print("✓ Equivalent")
   else:
     print_diff(ir1_norm, ir2_norm)
```

## Verification Pipeline Workflow

### CI Gate: Self-Hosted Reproducibility

**File**: `.github/workflows/selfhost.yml`

```yaml
name: Self-Hosted Reproducibility

on: [push, pull_request]

jobs:
  bootstrap:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      # Stage 0: Compile self-hosted compiler with Rust
      - name: Build Stage 0 (Rust bootstrap)
        run: |
          cd crates/souc
          cargo build --release

      - name: Compile Stage 1 (self-hosted compiler via Rust)
        run: |
          ./target/release/souc compile \
            --driver self-hosted/bootstrap/driver.sio \
            --output stage1_compiler.soir

      # Stage 1: Compile self-hosted compiler with Stage 1
      - name: Compile Stage 2 (self-hosted compiler via Stage 1)
        run: |
          ./target/release/souc run stage1_compiler.soir \
            --driver self-hosted/bootstrap/driver.sio \
            --output stage2_compiler.soir

      # Verification: Stage 1 ≡ Stage 2?
      - name: Verify Stage 1 ≡ Stage 2
        run: |
          ./scripts/sounio-verify compare \
            stage1_compiler.soir \
            stage2_compiler.soir

      # Store artifacts
      - name: Upload Stage 1 Compiler
        uses: actions/upload-artifact@v3
        with:
          name: stage1-compiler
          path: stage1_compiler.soir
```

### Local Workflow

```bash
# 1. Make changes to self-hosted compiler
vim self-hosted/check/infer.sio

# 2. Run local verification
make verify

# This runs:
# - Compile Stage 1 via Rust
# - Compile Stage 2 via Stage 1
# - Compare normalized IR
# - Run test suite on both stages

# 3. If verification passes, commit
git add self-hosted/check/infer.sio
git commit -m "[check] Improve type inference for match expressions"

# 4. CI will re-verify on push
git push
```

### Debugging Failed Verification

```bash
# Compare IR and show differences
sounio-verify compare stage1.soir stage2.soir --verbose

# Output:
# ✗ Stage 1 ≠ Stage 2
#
# Differences found in function 'type_infer_expr':
#   Instruction 42:
#     Stage 1: IrBinOp v10 = v5 + v6
#     Stage 2: IrBinOp v10 = v6 + v5
#
#   → This suggests non-deterministic codegen
#   → Check for unordered iteration (HashMap, etc.)

# Inspect individual SOIR files
sounio-verify inspect stage1.soir > stage1.txt
sounio-verify inspect stage2.soir > stage2.txt
diff -u stage1.txt stage2.txt

# Normalize and re-compare
sounio-verify normalize stage1.soir > stage1_norm.txt
sounio-verify normalize stage2.soir > stage2_norm.txt
diff -u stage1_norm.txt stage2_norm.txt
```

## Adding New Opcodes

### Step 1: Define in IR

**File**: `self-hosted/ir/ir.sio`

```sio
enum IrOpcode {
    // ... existing opcodes ...
    IrLoadString,
    IrCopy,
    IrBinOp,
    // Add new opcode:
    IrAtomicLoad,    // NEW: atomic load operation
}

fn ir_atomic_load(dst: i64, ptr: i64, ordering: i64) -> IrInstr {
    let base = ir_instr_new(IrOpcode::IrAtomicLoad)
    IrInstr {
        op: base.op,
        dst: dst,
        src1: ptr,
        src2: ordering,  // memory ordering
        // ... rest of fields ...
    }
}
```

### Step 2: Update Serialization

**File**: `self-hosted/ir/serialize.sio`

```sio
fn write_opcode(buf: [i8; 131072], pos: i64, op: IrOpcode) -> i64 {
    let tag: i8 = match op {
        // ... existing cases ...
        IrOpcode::IrNop => 18,
        IrOpcode::IrPhi => 19,
        IrOpcode::IrAtomicLoad => 20,  // NEW: assign next available code
    }
    write_i8(buf, pos, tag)
}

fn read_opcode(buf: [i8; 131072], pos: i64) -> (IrOpcode, i64) {
    let pair = read_i8(buf, pos)
    let tag = pair.0
    let op = if tag == 0 { IrOpcode::IrLoadImm }
        // ... existing cases ...
        else if tag == 19 { IrOpcode::IrPhi }
        else if tag == 20 { IrOpcode::IrAtomicLoad }  // NEW
        else { IrOpcode::IrNop }  // default fallback
    (op, pair.1)
}
```

### Step 3: Add VM Support

**File**: `self-hosted/vm/vm.sio`

```sio
fn vm_step(vm: VmState, instr: IrInstr) -> VmState {
    match instr.op {
        // ... existing cases ...
        IrOpcode::IrAtomicLoad => {
            let ptr = vm_get_reg(vm, instr.src1)
            let ordering = vm_get_reg(vm, instr.src2)
            // For VM: atomic = regular load (single-threaded)
            let value = vm_heap_load(vm.heap, ptr as i64)
            vm_set_reg(vm, instr.dst, value)
        }
        _ => vm
    }
}
```

### Step 4: Add Native Codegen Support

**File**: `self-hosted/native/lower_ir.sio`

```sio
fn lower_ir_instr(comp: NativeCompiler, instr: IrInstr) -> NativeCompiler {
    match instr.op {
        // ... existing cases ...
        IrOpcode::IrAtomicLoad => {
            // Load pointer into rdi
            var c = load_vreg_to_rdi(comp, instr.src1)
            // Emit x86-64 atomic load: mov rax, [rdi]
            c = emit_mov_rax_mem_rdi(c)
            // Store result to vreg slot
            c = store_rax_to_vreg(c, instr.dst)
            c
        }
        _ => comp
    }
}
```

### Step 5: Update Tests

**File**: `self-hosted/ir/test_serialize.sio` -- this file was never committed;
no IR-serialization test module exists in the tree, and `self-hosted/ir/serialize.sio`
holds the implementation only. Treat this step as unimplemented rather than as a
file to edit.

```sio
fn test_atomic_load_roundtrip() with IO, Mut, Panic, Div {
    let instr = ir_atomic_load(5, 10, 0)  // dst=v5, ptr=v10, ordering=0

    // Serialize
    var buf: [i8; 131072] = [0; 131072]
    let end_pos = serialize_ir_instr(buf, 0, instr)

    // Deserialize
    let pair = deserialize_ir_instr(buf, 0)
    let decoded = pair.0

    // Verify
    assert(decoded.op == IrOpcode::IrAtomicLoad, "opcode mismatch")
    assert(decoded.dst == 5, "dst mismatch")
    assert(decoded.src1 == 10, "src1 mismatch")
    assert(decoded.src2 == 0, "src2 mismatch")

    print("✓ test_atomic_load_roundtrip passed")
}
```

### Step 6: Update Documentation

**File**: `docs/architecture/SOIR_REFERENCE.md`

Update the opcode reference table:

| Code | Opcode | Description |
|------|--------|-------------|
| 20 | IrAtomicLoad | Atomic memory load (dst = *ptr with ordering) |

### Step 7: Verify Backward Compatibility

```bash
# Ensure old SOIR files can still be read (even if they don't use new opcode)
sounio-verify inspect old_stage1.soir  # should not error

# Verify new opcode serializes correctly
sounio-verify serialize test_atomic.sio output.soir
sounio-verify inspect output.soir | grep "IrAtomicLoad"
```

## Troubleshooting Guide

### Problem: Stage 1 ≠ Stage 2 (Non-Determinism)

**Symptoms**:
```
✗ Verification failed: Stage 1 and Stage 2 IR differ
  Function 'compile_expr' instruction 137 differs
```

**Common Causes**:
1. Unordered iteration (HashMap, HashSet)
2. Timestamps or random values in codegen
3. Uninitialized memory reads
4. Race conditions (if parallel compilation)

**Debug Steps**:
```bash
# 1. Inspect both stages
sounio-verify inspect stage1.soir > s1.txt
sounio-verify inspect stage2.soir > s2.txt
diff -u s1.txt s2.txt

# 2. Look for patterns in differences
# If register numbers differ: vreg allocation bug
# If label numbers differ: label allocation bug
# If instruction order differs: iteration order bug

# 3. Normalize and re-compare
sounio-verify normalize stage1.sio > s1_norm.soir
sounio-verify normalize stage2.sio > s2_norm.soir
sounio-verify compare s1_norm.soir s2_norm.soir

# 4. Check for HashMap usage in self-hosted compiler
grep -r "HashMap\|HashSet" self-hosted/
# Replace with deterministic alternatives (BTreeMap, Vec + sort)
```

**Fix Example**:
```sio
// Before (non-deterministic):
var symbol_table: HashMap<Name, i64> = HashMap::new()
for name in symbol_table.keys() {  // iteration order undefined
    emit_symbol(name)
}

// After (deterministic):
var symbol_table: HashMap<Name, i64> = HashMap::new()
var sorted_names = symbol_table.keys().collect_vec()
sorted_names.sort()
for name in sorted_names {  // iteration order defined
    emit_symbol(name)
}
```

### Problem: SOIR Deserialization Error

**Symptoms**:
```
Error: Invalid SOIR magic bytes
  Expected: SOIR (0x534F4952)
  Got: RISO (0x4F534952)
```

**Causes**:
1. Byte order issue (endianness)
2. Corrupted file
3. Wrong file format (not a SOIR file)

**Debug Steps**:
```bash
# 1. Check file header
hexdump -C stage1.soir | head -n 2
# Should show:
# 00000000  53 4f 49 52 01 00 00 00  ...
#           S  O  I  R  v  --  --  --

# 2. Verify file size is reasonable
ls -lh stage1.soir
# Should be < 128KB (SOIR_MAX_SIZE)

# 3. Re-serialize from source
sounio-verify serialize stage1.sio stage1_new.soir
diff stage1.soir stage1_new.soir
```

### Problem: VM Execution Crash

**Symptoms**:
```
VM panic: Invalid register access v127
  Function: type_infer_expr
  Instruction: 89
```

**Causes**:
1. Codegen bug (invalid vreg reference)
2. Corrupted IR
3. Stack overflow

**Debug Steps**:
```bash
# 1. Inspect the function IR
sounio-verify inspect stage1.soir --function type_infer_expr

# 2. Check register allocation
# Max vreg should be < reg_count
# All vreg refs should be < reg_count

# 3. Run with verbose VM tracing
SOUNIO_TRACE=1 ./poseidon stage1.soir 2>&1 | tee vm_trace.log
# Look for vreg access pattern before crash

# 4. Validate IR consistency
sounio-verify validate stage1.soir
# Checks:
# - All vreg refs < reg_count
# - All label refs < label_count
# - All fn refs < fn_count
```

### Problem: Opcode Not Recognized

**Symptoms**:
```
VM error: Unknown opcode 99 at function 'main' instruction 42
```

**Causes**:
1. SOIR version mismatch (old VM, new bytecode)
2. Corrupted bytecode
3. Missing opcode implementation

**Debug Steps**:
```bash
# 1. Check SOIR version
sounio-verify inspect stage1.soir | head -n 5
# Should show: Version: 1

# 2. Check VM/compiler version compatibility
./poseidon --version
./souc --version
# Ensure versions match

# 3. Disassemble around the problematic instruction
sounio-verify inspect stage1.soir --function main --instruction 42
# Look for opcode value in hex

# 4. Check if opcode is in valid range (0-19)
# If opcode > 19: likely corruption or future opcode
```

### Problem: Linker Error (Multi-Module)

**Symptoms**:
```
Linker error: Undefined symbol 'helper_function'
  Referenced in: main.sio
  Available symbols: [add, subtract, multiply]
```

**Causes**:
1. Missing module in link step
2. Symbol name mismatch (typo)
3. Forward reference to undefined function

**Debug Steps**:
```bash
# 1. List all symbols in each module
sounio-verify inspect module1.soir --symbols
sounio-verify inspect module2.soir --symbols

# 2. Check link order
# Modules should be linked in dependency order

# 3. Verify function exists in any module
grep -r "fn helper_function" self-hosted/

# 4. Re-compile with symbol tracing
SOUNIO_DEBUG_SYMBOLS=1 ./souc compile main.sio
```

### Problem: Performance Regression

**Symptoms**:
```
Stage 2 compilation takes 10x longer than Stage 1
```

**Causes**:
1. Optimization pass disabled
2. O(n²) algorithm introduced
3. Memory leak (repeated allocations)

**Debug Steps**:
```bash
# 1. Profile both stages
time ./poseidon stage1.soir  # baseline
time ./poseidon stage2.soir  # regression

# 2. Check VM instruction count
SOUNIO_COUNT_INSTRS=1 ./poseidon stage1.soir
SOUNIO_COUNT_INSTRS=1 ./poseidon stage2.soir
# Compare instruction counts

# 3. Look for algorithmic changes
git diff HEAD~1 self-hosted/check/
# Look for nested loops, repeated traversals

# 4. Check allocation patterns
SOUNIO_TRACE_ALLOC=1 ./poseidon stage2.soir 2>&1 | grep IrAlloc | wc -l
```

## Best Practices

### 1. Always Normalize Before Comparing

```bash
# Wrong:
diff stage1.soir stage2.soir  # byte-level diff, too noisy

# Right:
sounio-verify compare stage1.soir stage2.soir  # semantic comparison
```

### 2. Test Incremental Changes

```bash
# After each small change:
make verify-quick  # fast smoke test

# Before committing:
make verify-full   # complete 3-stage bootstrap
```

### 3. Keep SOIR Files Small

```bash
# Check artifact size:
ls -lh stage1_compiler.soir
# Should be < 50KB for most compilers

# If too large, investigate:
sounio-verify inspect stage1_compiler.soir --stats
# Shows: function count, instruction count, string table size
```

### 4. Version Control for SOIR

```bash
# Don't commit SOIR binaries (too large, binary format)
echo "*.soir" >> .gitignore

# Instead, commit source and reproduce:
./scripts/build_stage1.sh  # reproducible build
```

### 5. Document Breaking Changes

When modifying SOIR format:

1. Bump version number
2. Update compatibility table
3. Provide migration script
4. Keep old reader for N-1 version

## Further Reading

- [DEVELOPER_WORKFLOW.md](DEVELOPER_WORKFLOW.md) - Daily workflow guide
- [SOIR_REFERENCE.md](SOIR_REFERENCE.md) - Complete format specification
- [SELF_HOSTING_PHASES.md](SELF_HOSTING_PHASES.md) - Bootstrap roadmap
- `.claude/decisions/2026-02-13-rustless-cutover.md` - Design decisions
