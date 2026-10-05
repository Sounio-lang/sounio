<!-- docs:meta
topic_id: repo.docs.internal.implementation.migration-guide
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.implementation.migration-guide
-->

# Rustless Cutover Migration Guide

**Version**: 1.0
**Last Updated**: 2026-02-13
**Audience**: Users, contributors, and maintainers

## Overview

This guide helps you migrate to the new rustless self-hosted Sounio toolchain. The rustless cutover removes Rust from the critical compilation path, replacing it with:

1. **Self-hosted compiler** (written in Sounio, 24,428 LOC)
2. **Poseidon C VM** (bootstrap VM, 3,184 LOC)
3. **SOIR binary format** (deterministic IR serialization)

**Timeline**: The rustless cutover is complete as of 2026-02-13. All CI gates pass.

---

## Table of Contents

1. [For Users](#for-users)
2. [For Developers](#for-developers)
3. [Migration Path](#migration-path)
4. [Breaking Changes](#breaking-changes)
5. [FAQ](#faq)

---

## For Users

### Quick Start with Self-Hosted Compiler

The self-hosted compiler is now the default. No changes required for basic usage:

```bash
# Clone the repository
git clone https://github.com/sounio-lang/sounio.git
cd sounio

# Build the compiler with the historical Stage-0 Rust bootstrap path
# (kept here only for lineage; see current getting-started docs for today's flow)
[historical bootstrap command]

# Run your programs (now uses self-hosted compiler)
./target/release/souc run examples/hello.sio

# No visible difference from user perspective!
```

### New Features Available

#### 1. SOIR Artifact Inspection

You can now inspect compiled IR artifacts:

```bash
# Compile to SOIR
./target/release/souc compile examples/hello.sio --output hello.soir

# Inspect the IR
cargo run -p soir -- inspect hello.soir

# Output:
# Module: hello
# Functions: 1
# Strings: 1
#
# Function: main (4 instructions)
#   0: IrLoadString v0 = "Hello, world!"
#   1: IrCall v1 = print(v0)
#   2: IrLoadImm v2 = 0
#   3: IrReturn v2
```

#### 2. Reproducible Builds

SOIR artifacts are now deterministic:

```bash
# Compile twice
./target/release/souc compile program.sio --output run1.soir
./target/release/souc compile program.sio --output run2.soir

# Compare
sha256sum run1.soir run2.soir

# Output (identical):
# a3f5e2... run1.soir
# a3f5e2... run2.soir
```

#### 3. Cross-Platform Execution

Run compiled programs on any platform using Poseidon VM:

```bash
# Compile on Linux
./target/release/souc compile program.sio --output program.soir

# Transfer to macOS
scp program.soir mac-machine:

# Run on macOS (without recompiling)
./bootstrap/poseidon/poseidon program.soir
```

### What Stays the Same

- ✅ Command-line interface unchanged (`souc run`, `souc compile`, etc.)
- ✅ Language features unchanged (all existing code works)
- ✅ Standard library unchanged (215K+ LOC stdlib still available)
- ✅ Performance within 10% of previous version

### What's New

- ✅ SOIR binary format (optional `--output file.soir`)
- ✅ `soir` CLI tool for inspection and comparison
- ✅ Poseidon VM for portable execution
- ✅ Reproducible builds (deterministic artifacts)

---

## For Developers

### Getting Started with Self-Hosted Development

If you want to contribute to the self-hosted compiler:

#### 1. Build the Compiler

```bash
# Standard build (uses Rust for Stage 0)
cargo build --release -p souc

# Run self-hosted tests
cargo test --test rustless_e2e
```

#### 2. Modify Self-Hosted Code

All self-hosted compiler code is in `self-hosted/`:

```bash
cd self-hosted

# Directory structure:
# lexer/       - Lexical analysis
# parser/      - Syntax analysis
# check/       - Type checking
# ir/          - IR generation and serialization
# vm/          - Bytecode VM
# native/      - Native code generator (x86-64)
# linker/      - Multi-module linking
```

Example change:

```bash
# Edit self-hosted code
vim self-hosted/check/infer.sio

# Test your changes
cargo run --release -- run self-hosted/test_check.sio

# If tests pass, verify bootstrap reproducibility
cargo test --test rustless_e2e
```

#### 3. Understand the Bootstrap Process

The self-hosted compiler uses a 3-stage bootstrap:

```
Stage 0 (Rust/C)
    ↓ compiles
Stage 1 (Self-hosted compiler built by Rust)
    ↓ compiles itself
Stage 2 (Self-hosted compiler built by Stage 1)
    ↓ verify
Stage 1 ≡ Stage 2? (If yes, reproducible!)
```

**Key Insight**: When you modify self-hosted code, both Stage 1 and Stage 2 must produce identical output after normalization.

#### 4. Test Your Changes

```bash
# Unit tests (fast, run frequently)
cargo run -- run self-hosted/test_check.sio

# Integration tests (medium speed)
cargo test --test rustless_e2e

# Full bootstrap verification (slow, run before commit)
./scripts/verify_bootstrap.sh  # (if available)
```

### Common Development Tasks

#### Task: Add a New Language Feature

**Example**: Add support for `match` expressions.

**Steps**:

1. **Add to parser** (`self-hosted/parser/exprs.sio`):
```sio
fn parse_match_expr(parser: Parser) -> Expr {
    // Parse "match" keyword
    expect_keyword(parser, "match")

    // Parse scrutinee
    let scrutinee = parse_expr(parser)

    // Parse arms
    let arms = parse_match_arms(parser)

    Expr::Match { scrutinee, arms }
}
```

2. **Add to type checker** (`self-hosted/check/check.sio`):
```sio
fn check_match_expr(ctx: CheckContext, expr: Expr) -> Type {
    // Check scrutinee type
    let scrutinee_ty = check_expr(ctx, expr.scrutinee)

    // Check each arm
    for arm in expr.arms {
        check_pattern(ctx, arm.pattern, scrutinee_ty)
        check_expr(ctx, arm.body)
    }

    // All arms must return same type
    unify_arm_types(expr.arms)
}
```

3. **Add to IR lowering** (`self-hosted/ir/lower.sio`):
```sio
fn lower_match_expr(lowerer: Lowerer, expr: Expr) -> IrReg {
    // Lower to if-else chain
    let scrutinee_reg = lower_expr(lowerer, expr.scrutinee)

    for arm in expr.arms {
        // Lower pattern matching to conditional
        let cond_reg = lower_pattern_match(lowerer, arm.pattern, scrutinee_reg)

        // If pattern matches, execute body
        emit_branch_true(lowerer, cond_reg, arm.label)
    }

    scrutinee_reg
}
```

4. **Test**:
```bash
# Write test in self-hosted/test_check.sio
fn test_match_expr() with IO {
    let source = "fn main() { match x { 1 => true, _ => false } }"
    let ast = parse(source)
    let ty = check(ast)
    assert(ty == Type::Bool, "Expected bool type")
    print("✓ test_match_expr passed")
}

# Run test
cargo run -- run self-hosted/test_check.sio
```

5. **Verify bootstrap**:
```bash
# Ensure Stage 1 ≡ Stage 2 after your changes
cargo test --test rustless_e2e
```

#### Task: Fix a Bug in Self-Hosted Code

**Example**: Type inference bug in `self-hosted/check/infer.sio`

**Steps**:

1. **Write failing test**:
```sio
fn test_bug_fix() with IO {
    let source = "fn main() { let x = [1, 2, 3]; x[0] }"
    let result = compile(source)
    assert(result.type == Type::Int, "Expected int type")
    print("✓ test_bug_fix passed")
}
```

2. **Fix the bug**:
```sio
// Before (incorrect):
fn infer_array_index(ctx: CheckContext, expr: Expr) -> Type {
    let array_ty = infer_expr(ctx, expr.array)
    array_ty  // Bug: returns array type, not element type
}

// After (correct):
fn infer_array_index(ctx: CheckContext, expr: Expr) -> Type {
    let array_ty = infer_expr(ctx, expr.array)
    match array_ty {
        Type::Array { element_ty } => element_ty,  // Fixed
        _ => panic("Expected array type")
    }
}
```

3. **Verify fix**:
```bash
# Test passes now
cargo run -- run self-hosted/test_check.sio

# Verify bootstrap still works
cargo test --test rustless_e2e
```

#### Task: Optimize Self-Hosted Code

**Example**: Speed up type inference.

**Steps**:

1. **Profile**:
```bash
# Measure baseline
time cargo run --release -- compile self-hosted/ --output stage1.soir

# Typical result:
# real    0m0.089s
```

2. **Identify bottleneck** (hypothetical):
```sio
// Slow (O(n²) lookup):
fn find_symbol(ctx: CheckContext, name: Name) -> Symbol {
    for symbol in ctx.symbols {  // Linear search
        if symbol.name == name {
            return symbol
        }
    }
    panic("Symbol not found")
}
```

3. **Optimize**:
```sio
// Fast (O(1) lookup with HashMap):
fn find_symbol(ctx: CheckContext, name: Name) -> Symbol {
    match ctx.symbol_table.get(name) {  // Hash table lookup
        Some(symbol) => symbol,
        None => panic("Symbol not found")
    }
}
```

4. **Measure improvement**:
```bash
time cargo run --release -- compile self-hosted/ --output stage1.soir

# Improved result:
# real    0m0.045s  (2x speedup!)
```

5. **Verify correctness**:
```bash
# Ensure Stage 1 ≡ Stage 2 still passes
cargo test --test rustless_e2e
```

**Important**: Always verify bootstrap reproducibility after optimizations. Non-deterministic changes will break verification.

---

## Migration Path

### From Rust-Based Workflow to Self-Hosted

If you were previously developing the Rust compiler, here's how to transition:

#### Before (Rust-centric workflow):

```bash
# Edit Rust compiler
vim crates/souc/src/check/type_check.rs

# Test
cargo test -p souc

# Run
cargo run -p souc -- run program.sio
```

#### After (Self-hosted workflow):

```bash
# Edit self-hosted compiler
vim self-hosted/check/check.sio

# Test
cargo run -- run self-hosted/test_check.sio

# Run (automatically uses self-hosted compiler)
cargo run -- run program.sio
```

### Migrating Existing Code

**Good News**: All existing Sounio code works without changes!

The language syntax, semantics, and standard library are unchanged. Only the compiler implementation changed.

```sio
// This code works exactly the same before and after rustless cutover:
fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

fn main() -> i64 {
    fibonacci(10)  // Returns 55
}
```

---

## Breaking Changes

### For Users: None

There are **no breaking changes** for users. The compiler interface and language features are unchanged.

### For Contributors: Development Workflow Changes

#### 1. No More Direct Rust Compiler Edits

**Before**:
```bash
vim crates/souc/src/check/type_check.rs  # Edit Rust code
[historical Stage-0 rebuild command]
```

**After**:
```bash
vim self-hosted/check/check.sio  # Edit self-hosted code
cargo run -- run self-hosted/test_check.sio
```

**Why**: The Rust compiler is now a thin launcher. All compiler logic is in self-hosted Sounio code.

#### 2. New Test Requirements

**Before**: Rust unit tests only
```rust
#[test]
fn test_type_inference() {
    let source = "fn main() { 42 }";
    let result = compile(source);
    assert_eq!(result.type, Type::Int);
}
```

**After**: Self-hosted tests + verification
```sio
fn test_type_inference() with IO {
    let source = "fn main() { 42 }"
    let result = compile(source)
    assert(result.type == Type::Int, "Expected int type")
    print("✓ test_type_inference passed")
}
```

**Plus**:
```bash
# Verify bootstrap reproducibility
cargo test --test rustless_e2e
```

#### 3. Debugging Changes

**Before**: Use Rust debugger (lldb, gdb)
```bash
rust-lldb ./target/debug/souc
```

**After**: Use Sounio tracing
```bash
# Enable VM tracing
SOUNIO_TRACE=1 cargo run -- run self-hosted/test_check.sio

# Or inspect IR
cargo run -p soir -- inspect stage1.soir
```

---

## FAQ

### Q1: Why did we do this?

**A**: To achieve true self-hosting. Rust was a dependency in the critical path. Now the compiler is written in Sounio, compiled by itself, and verified through 3-stage bootstrap.

**Benefits**:
- ✅ Dogfooding (we use our own language)
- ✅ Reproducible builds (Stage 1 ≡ Stage 2)
- ✅ Trusting Trust mitigation (verifiable bootstrap)
- ✅ Platform independence (C VM is portable)

### Q2: Is Rust completely gone?

**A**: No, Rust is still used for:
- Stage 0 launcher (temporary, will be replaced by C or self-hosted binary)
- Tooling (soir library, LSP, debugger)
- Runtime libraries (for now)

**Long-term goal**: Minimize Rust to optional tooling only.

### Q3: Will my code break?

**A**: No. All existing Sounio code works without changes. The language is unchanged.

### Q4: Is performance worse?

**A**: Slightly slower during bootstrap (10-30x for VM execution), but production uses native codegen (same performance).

**Measurements**:
- Compilation time: <10% slower (mostly from SOIR serialization overhead)
- Runtime: Same (native codegen unchanged)

### Q5: How do I know if Stage 1 ≡ Stage 2?

**A**: Run the rustless E2E tests:

```bash
cargo test --test rustless_e2e

# Output:
# running 10 tests
# test test_fibonacci ... ok
# test test_arithmetic ... ok
# test test_control_flow ... ok
# test test_functions ... ok
# test test_loops ... ok
# test test_arrays ... ok
# test test_structs ... ok
# test test_strings ... ok
# test test_integration ... ok
# test test_vm_execution ... ok
#
# test result: ok. 10 passed; 0 failed; 0 ignored
```

### Q6: What if verification fails?

**A**: This indicates a compiler bug or non-determinism. Common causes:

1. **HashMap iteration** (non-deterministic order)
2. **Timestamps** (embedded in output)
3. **Random values** (non-deterministic)

**Fix**:
```bash
# Inspect diff
cargo run -p soir -- inspect stage1.soir > s1.txt
cargo run -p soir -- inspect stage2.soir > s2.txt
diff -u s1.txt s2.txt

# Find the divergence and fix it
```

### Q7: Can I still use the Rust compiler?

**A**: Rust-bridge self-host transition toggles are removed in cutover builds.
The following env vars now hard-error with migration guidance:

- `SOUNIO_SELFHOST_PIPELINE`
- `SOUNIO_RUST_GHOST`
- `SOUNIO_SELFHOST_NO_RUST_FALLBACK`
- `SOUNIO_SELFHOST_NO_RUST_HARNESS`
- `SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT`

Use signed bundle/state commands instead. They run on the checked artifact
`artifacts/omega/souc-bin/souc-linux-x86_64-gpu`, not on the default `./bin/souc`
(Madaros), which has no `bootstrap` or `opt` subcommand:

```bash
souc bootstrap verify --bundle bootstrap
souc bootstrap init --bundle bootstrap --state .sounio-state
souc bootstrap cycle --state .sounio-state
souc opt policy train --corpus benchmarks --output bootstrap/policies/policy.v1.json
souc opt policy eval --policy bootstrap/policies/policy.v1.json
souc opt policy promote --policy bootstrap/policies/policy.v1.json --output bootstrap/policies/active/policy.v1.json
souc opt policy status --policy bootstrap/policies/active/policy.v1.json
```

### Q8: How do I debug self-hosted code?

**A**: Use VM tracing and IR inspection:

```bash
# Enable VM tracing
SOUNIO_TRACE=1 cargo run -- run self-hosted/test_check.sio

# Output shows each instruction executed:
# [vm] fn=check_expr instr=0 IrLoadImm dst=v0 imm=0
# [vm] fn=check_expr instr=1 IrCopy dst=v1 src1=v0
# [vm] fn=check_expr instr=2 IrBinOp dst=v2 src1=v0 op=Add src2=v1
# ...

# Inspect IR
cargo run -p soir -- inspect stage1.soir --function check_expr
```

### Q9: What's the minimum Rust version required?

**A**: Rust 1.80+ (for Rust Edition 2024 features).

The Rust code is minimal (mostly glue), but we use modern Rust features for clarity.

### Q10: Can I contribute without knowing Sounio?

**A**: Yes! The self-hosted code is well-documented and follows idiomatic Sounio patterns.

Start with small changes:
- Fix typos in documentation
- Add test cases
- Improve error messages

Then gradually work up to larger changes.

**Learning Resources**:
- `docs/guide/tutorial.md` - Language tutorial
- `docs/guide/LLM_PROGRAMMING_GUIDE.md` - Syntax reference
- `self-hosted/test_*.sio` - Example code

### Q11: How do I report bugs?

**A**: Same as before—open a GitHub issue.

**Include**:
- Minimal reproducing example
- Expected vs actual behavior
- Output of `cargo run -- run your_test.sio`
- Output of `cargo test --test rustless_e2e` (if verification fails)

### Q12: What platforms are supported?

**A**: All major platforms:
- ✅ Linux (x86-64, ARM64)
- ✅ macOS (Intel, Apple Silicon)
- ✅ Windows (x86-64 via MinGW)
- ✅ BSDs (FreeBSD, OpenBSD)

Poseidon VM is pure C99 and works anywhere.

---

## Additional Resources

- **Rustless Cutover Documentation**: `docs/implementation/RUSTLESS_CUTOVER.md`
- **SOIR Format Specification**: `docs/architecture/SOIR_REFERENCE.md`
- **Developer Workflow Guide**: `docs/contributor-guide/DEVELOPER_WORKFLOW.md`
- **Poseidon VM Documentation**: `bootstrap/poseidon/README.md`
- **SOIR Library Documentation**: `crates/soir/README.md`
- **Complete Implementation Guide**: `docs/implementation/RUSTLESS_COMPLETE.md`

---

## Summary

**For Users**:
- ✅ No changes required
- ✅ All existing code works
- ✅ New features available (SOIR inspection, reproducible builds)

**For Contributors**:
- ✅ Edit self-hosted code instead of Rust
- ✅ Test with `cargo run -- run self-hosted/test_*.sio`
- ✅ Verify bootstrap with `cargo test --test rustless_e2e`
- ✅ Learn Sounio (idiomatic, well-documented)

**Key Takeaway**: The rustless cutover achieves true self-hosting while maintaining backward compatibility and usability.

---

**End of Migration Guide**
