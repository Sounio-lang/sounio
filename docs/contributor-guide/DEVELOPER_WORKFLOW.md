<!-- docs:meta
topic_id: repo.docs.contributor-guide.developer-workflow
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.contributor-guide.developer-workflow
-->

# Sounio Developer Workflow Guide

> **⚠️ Command reality updated 2026-07-11 (doc-reality audit).** This guide predates the Rust→self-hosted cutover; the `cargo` / `crates/` / `./target/release/souc` commands below do **not** work (no `Cargo.toml`). Real equivalents on this checkout: the compiler is **prebuilt** — use `./bin/souc` directly. `cargo run -- <args>` → `./bin/souc <args>`; `cargo build --release` → `make build`; `cargo test` → `make test` or `bash scripts/run_sio_test_suite.sh`. The Make targets `make verify` / `verify-quick` / `test-selfhost` / `test-all` do **not** exist — real ones include `make build`, `make check`, `make test`, `make test-stdlib`, `make test-madaros-identity`, `make lint`. Project-structure paths have moved: `check/type_check.sio` → `self-hosted/check/check.sio`; `lexer/lexer.sio` → `self-hosted/lexer/mod.sio`; the frontend lives under `self-hosted/`.


**Target Audience**: Contributors working on the self-hosted compiler
**Last Updated**: 2026-02-13

## Daily Workflow

### 1. Make Changes to Self-Hosted Compiler

```bash
# Example: Fix a bug in type inference
vim self-hosted/check/infer.sio

# Or add a new feature
vim self-hosted/parser/patterns.sio
```

### 2. Test Locally (Quick Validation)

```bash
# Option A: Run specific test file
cargo test --test selfhost_tests test_type_inference

# Option B: Use the self-hosted test runner
cargo run -- run self-hosted/test_check.sio

# Option C: Run the full self-hosted test suite
make test-selfhost
```

### 3. Verify IR Equivalence (Gate 1)

```bash
# Compile your changes and check if Stage 1 ≡ Stage 2
./scripts/sounio-verify compare \
    <(cargo run -- compile self-hosted/check/infer.sio --emit-soir) \
    <(cargo run -- run stage1.soir self-hosted/check/infer.sio --emit-soir)

# Shortcut using Makefile:
make verify-quick
```

**Expected Output**:
```
Compiling Stage 1...
Compiling Stage 2...
Normalizing IR...
✓ Stage 1 ≡ Stage 2 (normalized IR match)
```

### 4. Run Complete Bootstrap (Gate 2)

```bash
# Full 3-stage bootstrap with verification
make verify

# This runs:
# 1. Compile Stage 1 (Rust → self-hosted compiler)
# 2. Compile Stage 2 (Stage 1 → self-hosted compiler)
# 3. Verify Stage 1 ≡ Stage 2
# 4. Run test suite on both stages
```

### 5. Debug SOIR Execution with poseidon VM

If tests pass but execution fails:

```bash
# Run with VM tracing
SOUNIO_TRACE=1 cargo run -- run self-hosted/test_check.sio

# Or use the poseidon VM directly (if available)
SOUNIO_TRACE=1 ./scripts/poseidon stage1.soir

# Inspect specific function
./scripts/sounio-verify inspect stage1.soir --function type_infer_expr
```

### 6. Commit Your Changes

```bash
# Standard commit workflow
git add self-hosted/check/infer.sio
git commit -m "[check] Fix type inference for pattern matching"
git push

# CI will automatically verify:
# - Stage 1 ≡ Stage 2 (reproducibility)
# - All tests pass
# - No regressions
```

## Common Tasks

### Adding a New Self-Hosted Module

```bash
# 1. Create the module file
vim self-hosted/my_module/new_feature.sio

# 2. Add tests in the same directory
vim self-hosted/my_module/test_new_feature.sio

# 3. Import in the main driver
vim self-hosted/bootstrap/driver.sio
# Add: import my_module::new_feature

# 4. Verify the module loads
cargo run -- check self-hosted/my_module/new_feature.sio

# 5. Run the full suite
make test-selfhost
```

### Modifying the IR

**Scenario**: You need to add a new IR instruction for a new language feature.

**File**: `self-hosted/ir/ir.sio`

```bash
# 1. Define the new opcode
vim self-hosted/ir/ir.sio
# Add to IrOpcode enum:
#   IrMyNewOp,
# Add constructor function:
#   fn ir_my_new_op(dst: i64, src: i64) -> IrInstr { ... }

# 2. Update serialization
vim self-hosted/ir/serialize.sio
# Add to write_opcode: IrOpcode::IrMyNewOp => 20
# Add to read_opcode: else if tag == 20 { IrOpcode::IrMyNewOp }

# 3. Add VM support
vim self-hosted/vm/vm.sio
# Add case to vm_step:
#   IrOpcode::IrMyNewOp => { ... }

# 4. Add native codegen support (optional, for Phase 3)
vim self-hosted/native/lower_ir.sio
# Add case to lower_ir_instr:
#   IrOpcode::IrMyNewOp => { ... }

# 5. Add tests
vim self-hosted/test_ir.sio
# There is no dedicated serializer test file; add the write_opcode/read_opcode
# roundtrip case for the new opcode to the IR test suite.

# 6. Update documentation
vim docs/architecture/SOIR_REFERENCE.md
# Add new opcode to reference table

# 7. Verify everything works
make verify
```

### Debugging Stage 1 ≠ Stage 2

**Problem**: Verification fails because Stage 1 and Stage 2 produce different IR.

```bash
# 1. Get both SOIR files
cargo run -- compile self-hosted/ --output stage1.soir
cargo run -- run stage1.soir self-hosted/ --output stage2.soir

# 2. Inspect both
./scripts/sounio-verify inspect stage1.soir > stage1.txt
./scripts/sounio-verify inspect stage2.soir > stage2.txt

# 3. Find differences
diff -u stage1.txt stage2.txt | less

# 4. Look for patterns:
# - Different register numbers → non-deterministic vreg allocation
# - Different label numbers → non-deterministic label allocation
# - Different instruction order → HashMap/HashSet iteration
# - Different string indices → non-deterministic string table

# 5. Normalize and retry
./scripts/sounio-verify normalize stage1.soir > stage1_norm.soir
./scripts/sounio-verify normalize stage2.soir > stage2_norm.soir
./scripts/sounio-verify compare stage1_norm.soir stage2_norm.soir

# 6. If still different, find the culprit
# Common causes:
grep -r "HashMap\|HashSet" self-hosted/
# Look for iteration without sorting
```

**Fix Example**:

```sio
// Before (non-deterministic):
for (name, symbol) in symbol_table {
    emit_symbol(name, symbol)
}

// After (deterministic):
var names = symbol_table.keys().collect_vec()
names.sort()
for name in names {
    let symbol = symbol_table.get(name)
    emit_symbol(name, symbol)
}
```

### Testing Native Codegen (Phase 3)

```bash
# 1. Compile to native binary
cargo run -- compile-native examples/hello.sio --output test.elf

# 2. Run the native binary
chmod +x test.elf
./test.elf
echo $?  # Check exit code

# 3. Compare output with VM execution
cargo run -- run examples/hello.sio > vm_output.txt
./test.elf > native_output.txt
diff vm_output.txt native_output.txt

# 4. Disassemble the native binary
objdump -d test.elf | less

# 5. Verify ELF structure
readelf -a test.elf
```

### Benchmarking Performance

```bash
# 1. Benchmark Stage 1 compilation
time cargo run --release -- compile self-hosted/ --output stage1.soir

# 2. Benchmark Stage 2 compilation
time cargo run --release -- run stage1.soir self-hosted/ --output stage2.soir

# 3. Compare instruction counts
SOUNIO_COUNT_INSTRS=1 cargo run -- run stage1.soir > stage1_count.txt
SOUNIO_COUNT_INSTRS=1 cargo run -- run stage2.soir > stage2_count.txt
diff stage1_count.txt stage2_count.txt

# 4. Profile with perf (Linux)
perf record cargo run --release -- run stage1.soir
perf report

# 5. Memory profiling with valgrind
valgrind --tool=massif cargo run -- run stage1.soir
ms_print massif.out.*
```

### Working with the Linker

**Scenario**: You're adding multi-file compilation support.

```bash
# 1. Create multiple modules
cat > module1.sio <<EOF
fn helper() -> i64 {
    42
}
EOF

cat > module2.sio <<EOF
fn main() -> i64 {
    helper()
}
EOF

# 2. Compile each module to IR
cargo run -- compile module1.sio --emit-ir > module1.ir
cargo run -- compile module2.sio --emit-ir > module2.ir

# 3. Test linker (self-hosted)
cargo run -- run self-hosted/linker/test_linker.sio

# 4. Link modules and compile to binary
cargo run -- link module1.ir module2.ir --output program.elf

# 5. Run linked program
./program.elf
echo $?  # Should be 42
```

## Project Structure

Note: the per-module test files live at the top of `self-hosted/`, not inside the
module directories.

```
self-hosted/
├── test_lexer.sio          # Lexer tests
├── test_parser.sio         # Parser tests
├── test_check.sio          # Type checker tests
├── test_ir.sio             # IR tests
├── bootstrap/
│   ├── driver.sio          # Main compilation driver
│   └── bootstrap_v0.sio    # Zero-import single-file bootstrap compiler v0
├── lexer/
│   └── mod.sio             # Lexical analysis
├── parser/
│   ├── parser.sio          # Syntax analysis
│   ├── exprs.sio           # Expression parsing
│   └── patterns.sio        # Pattern parsing
├── check/
│   ├── check.sio           # Type checking
│   └── infer.sio           # Type inference
├── ir/
│   ├── ir.sio              # IR definitions
│   └── serialize.sio       # SOIR serialization
├── vm/
│   ├── vm.sio              # Bytecode interpreter
│   └── test_vm.sio         # VM tests
├── linker/
│   ├── mod.sio             # Multi-module linking
│   └── test_linker.sio     # Linker tests
├── native/
│   ├── codegen.sio         # Native code generation
│   ├── lower_ir.sio        # IR → x86-64 lowering
│   ├── elf.sio             # ELF binary emission
│   ├── encode.sio          # x86-64 instruction encoding
│   └── test_phase1.sio     # Native codegen tests
├── hypercomplex/
│   ├── quat_simd.sio       # Quaternion SIMD kernels
│   ├── octonion.sio        # Octonion/Sedenion algebra
│   └── test_*.sio          # Hypercomplex tests
└── tensor/
    ├── contract.sio        # Tensor contraction compiler
    └── test_contract.sio   # Tensor tests
```

## Testing Strategy

### Test Pyramid

```
         ┌─────────────────┐
         │   Integration   │  ← Full 3-stage bootstrap (CI only)
         │   Tests (E2E)   │
         └─────────────────┘
                ↑
         ┌─────────────────┐
         │  Module Tests   │  ← Each self-hosted module
         │  (test_*.sio)   │
         └─────────────────┘
                ↑
         ┌─────────────────┐
         │   Unit Tests    │  ← Individual functions
         │  (inline tests) │
         └─────────────────┘
```

### Running Tests

```bash
# Unit tests (fast, run frequently)
cargo test --test selfhost_unit

# Module tests (medium speed, run before commit)
cargo run -- run self-hosted/test_check.sio
cargo run -- run self-hosted/test_parser.sio
cargo run -- run self-hosted/test_ir.sio

# Integration tests (slow, run before push)
make verify

# All tests (comprehensive, CI only)
make test-all
```

### Writing Tests

**File**: `self-hosted/my_module/test_my_feature.sio`

```sio
fn test_basic_case() with IO {
    let result = my_feature(42)
    assert(result == 84, "Expected 84")
    print("✓ test_basic_case passed")
}

fn test_edge_case() with IO {
    let result = my_feature(0)
    assert(result == 0, "Expected 0")
    print("✓ test_edge_case passed")
}

fn test_error_case() with IO {
    let result = my_feature(-1)
    assert(result == -2, "Expected -2")
    print("✓ test_error_case passed")
}

fn main() with IO {
    test_basic_case()
    test_edge_case()
    test_error_case()
    print("All tests passed!")
}
```

## Debugging Tips

### VM Tracing

```bash
# Enable VM instruction tracing
export SOUNIO_TRACE=1
cargo run -- run self-hosted/test_check.sio

# Output shows each instruction executed:
# [vm] fn=type_infer_expr instr=0 IrLoadImm dst=v0 imm=0
# [vm] fn=type_infer_expr instr=1 IrCopy dst=v1 src1=v0
# [vm] fn=type_infer_expr instr=2 IrBinOp dst=v2 src1=v0 op=Add src2=v1
# ...
```

### Compiler Tracing

```bash
# Enable compiler phase tracing
export SOUNIO_DEBUG=1
cargo run -- compile examples/hello.sio

# Output shows each compilation phase:
# [souc] Phase: Lexing
# [souc] Phase: Parsing
# [souc] Phase: Type Checking
# [souc] Phase: IR Lowering
# [souc] Phase: Codegen
```

### IR Inspection

```bash
# Dump IR after type checking
cargo run -- compile file.sio --emit-ir > file.ir
less file.ir

# Dump normalized IR
./scripts/sounio-verify normalize file.sio > file_norm.ir
less file_norm.ir

# Compare two IR files
./scripts/sounio-verify compare file1.sio file2.sio --verbose
```

### Binary Inspection

```bash
# Disassemble SOIR bytecode
./scripts/sounio-verify inspect stage1.soir > stage1_disasm.txt
less stage1_disasm.txt

# Validate SOIR constraints
./scripts/sounio-verify validate stage1.soir

# Check file size
ls -lh stage1.soir
# Should be < 50KB for typical compiler
```

## Performance Optimization

### Profiling Workflow

```bash
# 1. Establish baseline
time make verify > baseline.txt

# 2. Make optimization changes
vim self-hosted/check/infer.sio

# 3. Measure impact
time make verify > optimized.txt

# 4. Compare
diff baseline.txt optimized.txt

# 5. Check for correctness regressions
./scripts/sounio-verify compare baseline.soir optimized.soir
```

### Common Bottlenecks

1. **Unordered iteration**
   - Fix: Sort before iteration
   - Impact: 5-10% speedup + determinism

2. **Repeated allocations**
   - Fix: Reuse buffers, pool allocations
   - Impact: 10-20% speedup

3. **Deep recursion**
   - Fix: Convert to iterative with explicit stack
   - Impact: 20-30% speedup + avoid stack overflow

4. **Linear search**
   - Fix: Use HashMap or sorted Vec + binary search
   - Impact: 50-90% speedup for large inputs

## CI Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/selfhost.yml
name: Self-Hosted Verification

on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Rust
        uses: actions-rs/toolchain@v1
      - name: Build souc
        run: cargo build --release
      - name: Run verification
        run: make verify
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: stage1-compiler
          path: stage1.soir
```

### Required Checks

Before merging a PR:

1. ✓ All unit tests pass
2. ✓ All module tests pass
3. ✓ Stage 1 ≡ Stage 2 (reproducibility)
4. ✓ No performance regressions (>10% slowdown)
5. ✓ Code style checks (make fmt)
6. ✓ No compiler warnings (make clippy)

## FAQ

### Q: How do I know if my changes require a full bootstrap?

**A**: Run `make verify-quick` first. If it passes, you're probably fine. If it fails or you modified core compiler logic (lexer, parser, checker, IR lowering), run `make verify` for the full 3-stage bootstrap.

### Q: What if Stage 1 ≠ Stage 2 but they both work?

**A**: This indicates non-determinism in the compiler. Even if both work, you must fix it because:
1. Reproducible builds are a requirement
2. Non-determinism hides bugs
3. CI will fail

### Q: How do I test just one module without running the entire suite?

**A**: Use the module's test file directly:
```bash
cargo run -- run self-hosted/test_check.sio
```

### Q: Can I modify multiple modules in one commit?

**A**: Yes, but keep commits atomic. One logical change per commit. If you're refactoring across multiple modules, explain it in the commit message.

### Q: How do I know if I broke something in the native codegen?

**A**: Run the native codegen test suite:
```bash
cargo run -- run self-hosted/native/test_phase1.sio
```

If this passes, your changes are safe.

## Further Reading

- [RUSTLESS_CUTOVER.md](../implementation/RUSTLESS_CUTOVER.md) - Complete rustless cutover documentation
- [SOIR_REFERENCE.md](../architecture/SOIR_REFERENCE.md) - SOIR format specification
- [SELF_HOSTING_PHASES.md](../implementation/SELF_HOSTING_PHASES.md) - Bootstrap roadmap
- `.claude/decisions/2026-02-13-rustless-cutover.md` - Design decisions
