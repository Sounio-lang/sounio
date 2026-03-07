<!-- docs:meta
topic_id: repo.docs.implementation.tooling-summary
authority: repo_only
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.tooling-summary
-->

# Rustless Cutover Tooling Summary

**Created**: 2026-02-13
**Status**: Complete
**See**: [RUSTLESS_CUTOVER.md](RUSTLESS_CUTOVER.md)

## Overview

The Rustless Cutover tooling provides a complete workflow for developing, testing, and verifying the self-hosted Sounio compiler. This document summarizes the available tools and how they fit together.

## Tools Provided

### 1. sounio-verify CLI

**Location**: `scripts/sounio-verify`
**Purpose**: Swiss-army knife for SOIR verification and inspection

**Commands**:
```bash
# Compare two SOIR files for equivalence
sounio-verify compare stage1.soir stage2.soir [--verbose]

# Show normalized IR for a source file
sounio-verify normalize file.sio

# Compile source to SOIR bytecode
sounio-verify serialize input.sio output.soir

# Disassemble SOIR to human-readable format
sounio-verify inspect file.soir [--function name] [--stats]

# Validate SOIR file structure
sounio-verify validate file.soir
```

**Features**:
- Color-coded output (✓ success, ✗ error, → info)
- Automatic normalization for comparison
- Detailed error messages with debugging hints
- Integration with self-hosted serializer

### 2. Makefile Verification Targets

**Location**: `Makefile.verify`
**Purpose**: Convenient make targets for common workflows

**Quick Commands**:
```bash
# Fast validation (single-stage)
make -f Makefile.verify verify-quick

# Full 3-stage bootstrap
make -f Makefile.verify verify

# Individual stages
make -f Makefile.verify stage1
make -f Makefile.verify stage2
make -f Makefile.verify compare-stages

# Inspection
make -f Makefile.verify inspect-stage1
make -f Makefile.verify stats

# Help
make -f Makefile.verify help-verify
```

**What it does**:
1. Builds souc (Rust compiler) if needed
2. Compiles Stage 1 (Rust → self-hosted compiler)
3. Compiles Stage 2 (Stage 1 → self-hosted compiler)
4. Verifies Stage 1 ≡ Stage 2
5. Runs test suites on both stages

### 3. Self-Hosted Disassembler

**Location**: `self-hosted/ir/disasm.sio`
**Purpose**: Human-readable IR output

**Features**:
- Pretty-prints SOIR bytecode
- Shows function signatures, parameters, registers
- Annotates instructions with operand details
- Prints string table

**Integration**: Used by `sounio-verify inspect`

### 4. Self-Hosted Serializer

**Location**: `self-hosted/ir/serialize.sio`
**Purpose**: Binary SOIR encoding/decoding

**Features**:
- Encode IrModule → SOIR v1 binary format
- Decode SOIR v1 → IrModule
- Header validation (magic bytes, version)
- Roundtrip testing

**Integration**: Used by `sounio-verify serialize` and `compare`

## Workflow Integration

### Daily Development Workflow

```
┌──────────────────────────────────────────┐
│ 1. Make changes to self-hosted compiler │
│    vim self-hosted/check/type_check.sio │
└──────────────┬───────────────────────────┘
               │
               v
┌──────────────────────────────────────────┐
│ 2. Run quick validation                  │
│    make -f Makefile.verify verify-quick  │
│                                           │
│    → Compiles source                     │
│    → Runs test suite                     │
│    → ~30 seconds                          │
└──────────────┬───────────────────────────┘
               │
               v
         ┌─────┴─────┐
         │  Pass?    │
         └─────┬─────┘
         Yes   │   No
               │   │
               │   └──> Debug and retry
               v
┌──────────────────────────────────────────┐
│ 3. Run full verification (before commit)│
│    make -f Makefile.verify verify        │
│                                           │
│    → Stage 1: Rust → self-hosted         │
│    → Stage 2: Stage 1 → self-hosted      │
│    → Compare: Stage 1 ≡ Stage 2?         │
│    → Test both stages                    │
│    → ~2-5 minutes                         │
└──────────────┬───────────────────────────┘
               │
               v
         ┌─────┴─────┐
         │  Pass?    │
         └─────┬─────┘
         Yes   │   No
               │   │
               │   └──> Debug Stage 1 ≠ Stage 2
               v        (see Troubleshooting)
┌──────────────────────────────────────────┐
│ 4. Commit and push                       │
│    git commit -m "[check] Fix type bug"  │
│    git push                              │
└──────────────┬───────────────────────────┘
               │
               v
┌──────────────────────────────────────────┐
│ 5. CI runs full verification             │
│    .github/workflows/selfhost.yml        │
│                                           │
│    → Same as local verify                │
│    → Must pass before merge              │
└──────────────────────────────────────────┘
```

### Debugging Workflow

```
Stage 1 ≠ Stage 2 detected
         │
         v
┌──────────────────────────────────────────┐
│ 1. Get detailed diff                     │
│    sounio-verify compare \               │
│      stage1.soir stage2.soir --verbose   │
└──────────────┬───────────────────────────┘
               │
               v
┌──────────────────────────────────────────┐
│ 2. Inspect both stages                   │
│    sounio-verify inspect stage1.soir \   │
│      > stage1.txt                        │
│    sounio-verify inspect stage2.soir \   │
│      > stage2.txt                        │
│    diff -u stage1.txt stage2.txt         │
└──────────────┬───────────────────────────┘
               │
               v
┌──────────────────────────────────────────┐
│ 3. Look for patterns                     │
│    - Vreg numbers differ?                │
│      → Non-deterministic allocation      │
│    - Instruction order differs?          │
│      → HashMap/HashSet iteration         │
│    - String indices differ?              │
│      → Non-deterministic string table    │
└──────────────┬───────────────────────────┘
               │
               v
┌──────────────────────────────────────────┐
│ 4. Find the culprit                      │
│    grep -r "HashMap\|HashSet" \          │
│      self-hosted/                        │
│                                           │
│    Look for unordered iteration          │
└──────────────┬───────────────────────────┘
               │
               v
┌──────────────────────────────────────────┐
│ 5. Fix: Sort before iteration            │
│    var keys = map.keys().collect_vec()   │
│    keys.sort()                           │
│    for key in keys { ... }               │
└──────────────┬───────────────────────────┘
               │
               v
┌──────────────────────────────────────────┐
│ 6. Verify fix                            │
│    make -f Makefile.verify verify        │
└──────────────────────────────────────────┘
```

## File Organization

```
sounio/
├── docs/
│   ├── RUSTLESS_CUTOVER.md       ← Main documentation (start here)
│   ├── SOIR_REFERENCE.md         ← Format specification
│   ├── DEVELOPER_WORKFLOW.md     ← Daily workflow guide
│   └── TOOLING_SUMMARY.md        ← This file
│
├── scripts/
│   └── sounio-verify             ← CLI tool (executable)
│
├── self-hosted/
│   ├── ir/
│   │   ├── ir.sio                ← IR definitions
│   │   ├── serialize.sio         ← SOIR encoder/decoder
│   │   └── disasm.sio            ← Human-readable output
│   └── bootstrap/
│       ├── driver.sio            ← Main compilation driver
│       └── verify.sio            ← Bootstrap verification
│
├── Makefile.verify               ← Verification make targets
│
└── .github/
    └── workflows/
        └── selfhost.yml          ← CI integration (future)
```

## Integration Points

### With Rust Compiler (souc)

The `sounio-verify` tool wraps the Rust compiler (`target/release/souc`) and uses it to:
- Compile Sounio source to IR
- Run self-hosted serializer/deserializer
- Execute VM for stage comparisons

**Bridge**: The Rust compiler remains as Stage 0 bootstrap until fully replaced.

### With Self-Hosted Compiler

Self-hosted modules are executed by:
1. Rust VM (current): `souc run file.sio`
2. Poseidon VM (future): `poseidon file.soir`
3. Native binary (future): `./sounio file.sio`

**Bridge**: SOIR v1 format is the stable interface between stages.

### With CI/CD

GitHub Actions workflow calls:
```bash
make -f Makefile.verify ci-verify
```

This ensures every PR:
- Builds cleanly
- Passes all tests
- Maintains Stage 1 ≡ Stage 2 reproducibility

**Gate**: PRs cannot merge if verification fails.

## Performance Characteristics

### Quick Verification (~30 seconds)
- Single-stage compilation
- Runs test suite once
- Good for rapid iteration

### Full Verification (~2-5 minutes)
- Three-stage bootstrap
- IR comparison and normalization
- Test suite on both stages
- Good for pre-commit checks

### CI Verification (~5-10 minutes)
- Full verification + additional checks
- Multiple test configurations
- Artifact uploads
- Good for merge gate

## Future Enhancements

### Phase R1: Poseidon VM Integration

Replace Rust VM with C-based poseidon:
```bash
# Current:
souc run stage1.soir

# Future:
poseidon stage1.soir
```

**Benefit**: Removes Rust from execution path

### Phase R2: Native Compilation

Compile directly to native binary:
```bash
# Current: SOIR bytecode
sounio-verify serialize file.sio file.soir

# Future: Native ELF
sounio-verify compile file.sio file.elf
chmod +x file.elf
./file.elf
```

**Benefit**: ~10x faster execution

### Phase R3: Incremental Verification

Cache normalized IR to speed up comparisons:
```bash
# Current: Re-normalize every time
sounio-verify compare stage1.soir stage2.soir

# Future: Cached normalization
sounio-verify compare stage1.soir stage2.soir --cached
```

**Benefit**: ~3x faster verification

## FAQ

### Q: Do I need to run full verification for every change?

**A**: No. Use `verify-quick` during development. Only run `verify` before committing.

### Q: What if I'm only changing tests, not compiler code?

**A**: Still run `verify-quick` to ensure tests pass. Full `verify` is not needed unless you modify core compiler logic.

### Q: Can I use sounio-verify without the Makefile?

**A**: Yes. The `sounio-verify` script is standalone and can be used directly:
```bash
./scripts/sounio-verify compare file1.soir file2.soir
```

### Q: What's the difference between normalize and compare?

**A**:
- `normalize`: Shows normalized IR for a single file (for inspection)
- `compare`: Normalizes two files and checks if they're equivalent (for verification)

### Q: How do I know if my SOIR file is valid?

**A**: Run:
```bash
sounio-verify validate file.soir
```

This checks:
- Magic bytes (SOIR)
- Version (1)
- File size (< 128KB)
- Register/label/function constraints

### Q: Can I inspect a specific function in a SOIR file?

**A**: Yes:
```bash
sounio-verify inspect file.soir --function type_infer_expr
```

## Getting Help

### Documentation
1. [RUSTLESS_CUTOVER.md](RUSTLESS_CUTOVER.md) - Complete documentation
2. [DEVELOPER_WORKFLOW.md](DEVELOPER_WORKFLOW.md) - Daily workflow
3. [SOIR_REFERENCE.md](SOIR_REFERENCE.md) - Format specification

### Tools
```bash
# Help for CLI tool
sounio-verify help

# Help for Makefile
make -f Makefile.verify help-verify
```

### Support
- GitHub Issues: Technical problems
- Discussions: Usage questions
- `.claude/decisions/`: Design rationale

## Summary

The Rustless Cutover tooling provides a complete, integrated workflow for self-hosted compiler development. Key components:

1. **sounio-verify**: CLI tool for SOIR operations
2. **Makefile.verify**: Convenient make targets
3. **Self-hosted modules**: SOIR serialization and disassembly
4. **Documentation**: Comprehensive guides

**Goal**: Make self-hosted development as smooth as possible while maintaining rigorous verification of bootstrap reproducibility.

**Status**: Ready for daily use. All tools integrated and tested.
