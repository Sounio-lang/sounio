<!-- docs:meta
topic_id: repo.docs.architecture.cps-pipeline-integration
authority: historical
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.cps-pipeline-integration
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CPS Pipeline Integration - Implementation Summary

## Overview

This document describes the integration of the CPS (Continuation-Passing Style) transformation into the Sounio compiler pipeline. The CPS transformation is now automatically applied when compiling effectful code with the `--enable-cps` flag.

## Changes Made

### 1. CLI Argument Addition (`src/main.rs`)

Added `--enable-cps` flag to the `Build` command:

```rust
/// Enable CPS transformation for effect handlers (experimental)
#[arg(long)]
enable_cps: bool,
```

This flag is:
- **Optional**: Disabled by default for stability
- **Experimental**: Marked as experimental in help text
- **Global**: Works with both native and Cranelift backends

### 2. Backend Configuration (`src/cli/backend.rs`)

#### NativeBackendOptions

The `enable_cps` field was already present:

```rust
pub struct NativeBackendOptions {
    // ... other fields
    /// Enable CPS transformation for effect handlers (experimental)
    pub enable_cps: Option<bool>,
}
```

Default: `None` (disabled)

#### Argument Parser

Updated `extract_build_args()` to handle the flag:

```rust
// Enable CPS transformation
if arg_matches.get_flag("enable_cps") {
    build_args.native_opts.enable_cps = Some(true);
}
```

### 3. Pipeline Integration

#### Native Backend (`compile_native`)

Enabled the previously commented-out CPS transformation step (line 789-805):

```rust
// Step 3.5: Apply CPS transformation (optional, for effect handlers in native code)
if args.native_opts.enable_cps.unwrap_or(false) {
    use crate::backend::cps_transform::CpsTransform;

    let cps_start = Instant::now();
    let func_count_before = hlir.functions.len();
    let mut transform = CpsTransform::new();
    hlir = transform
        .transform(hlir)
        .map_err(|e| format!("CPS transformation error: {}", e))?;
    let func_count_after = hlir.functions.len();
    let cps_ms = cps_start.elapsed().as_millis() as u64;

    if args.verbose {
        println!("✓ Applied CPS transformation in {}ms", cps_ms);
        let transformed_count = func_count_after - func_count_before;
        if transformed_count > 0 {
            println!("  {} functions transformed to CPS", transformed_count);
        } else {
            println!("  No effectful functions found (CPS transformation skipped)");
        }
    }
}
```

**Pipeline order:**
1. Parse AST
2. Type check → HIR
3. Lower to HLIR
4. **← CPS transformation (if enabled)**
5. Lower to SIR
6. Register allocation
7. Code generation

#### Cranelift Backend (`compile_cranelift`)

Added identical CPS transformation step:

```rust
// Step 3.5: Apply CPS transformation if requested
if args.native_opts.enable_cps.unwrap_or(false) {
    // ... same implementation as native backend
}
```

**Pipeline order:**
1. Parse AST
2. Type check → HIR
3. Lower to HLIR
4. **← CPS transformation (if enabled)**
5. Compile through Cranelift AOT
6. Link to executable/shared library

### 4. Help Documentation

Updated `print_backend_help()` to include the new flag:

```rust
println!("  --enable-cps                 Enable CPS transformation for effect handlers (experimental)");
```

## Usage

### Compiling with CPS Transformation

```bash
# Native backend
souc build --backend=native --enable-cps src/main.sio -o output

# Cranelift backend
souc build --backend=cranelift --enable-cps src/main.sio -o output

# With verbose output to see transformation details
souc build --backend=native --enable-cps --verbose src/main.sio -o output
```

### Expected Output (Verbose Mode)

```
✓ Parsed in 5ms
✓ Type checked in 12ms
✓ Lowered to HLIR in 3ms
✓ Applied CPS transformation in 8ms
  2 functions transformed to CPS
✓ Lowered to SIR in 15ms
...
```

## Selective CPS

The CPS transformation uses **selective CPS**:

- **Only effectful functions** are transformed
- Pure functions remain in direct style
- Effect analysis happens in Phase 1 of the transformation
- If no effects are found, the transformation is a no-op

## Testing

### Test File

Created `test_cps_integration.sio` to verify the integration:

```sio
// Test file for CPS transformation integration
// This file uses IO effects to trigger CPS transformation

fn main() with IO {
    let x = 42
    println(x)
    let y = x + 1
    println(y)
    0
}
```

### Manual Testing

```bash
# Test with CPS enabled
cargo run --bin souc -- build --backend=native --enable-cps --verbose test_cps_integration.sio

# Test without CPS (should compile normally)
cargo run --bin souc -- build --backend=native --verbose test_cps_integration.sio
```

## Architecture

### Component Diagram

```
┌─────────────┐
│   CLI Args  │ --enable-cps flag
└──────┬──────┘
       │
       v
┌──────────────────────┐
│  BuildArgs           │
│  native_opts {       │
│    enable_cps: bool  │
│  }                   │
└──────┬───────────────┘
       │
       v
┌──────────────────────┐
│  compile_native()    │
│  compile_cranelift() │
└──────┬───────────────┘
       │
       v
┌──────────────────────┐
│  if enable_cps {     │
│    CpsTransform::    │
│      new().          │
│      transform(hlir) │
│  }                   │
└──────────────────────┘
```

### Data Flow

```
Source Code
    ↓
  AST (Parser)
    ↓
  HIR (Type Checker)
    ↓
  HLIR (HIR Lowering)
    ↓
[CPS Transform] ← if --enable-cps
    ↓
  HLIR (CPS-transformed)
    ↓
  SIR / Cranelift
    ↓
Machine Code
```

## Known Limitations

1. **Experimental Status**: CPS transformation is marked experimental
2. **LLVM Backend**: Not yet integrated (only native + Cranelift)
3. **GPU Backend**: Not applicable (effects work differently on GPU)
4. **Multi-shot Continuations**: Implemented but not fully tested

## Future Work

1. **Default Enablement**: Once stable, consider enabling by default for effectful code
2. **LLVM Integration**: Add CPS transformation to LLVM backend
3. **Optimization**: Add CPS-specific optimizations (inlining, dead continuation elimination)
4. **Validation**: Add HLIR validation pass after CPS transformation
5. **Testing**: Add comprehensive integration tests

## Related Files

- **CLI**: `crates/souc/src/main.rs`
- **Backend**: `crates/souc/src/cli/backend.rs`
- **CPS Transform**: `crates/souc/src/backend/cps_transform.rs`
- **Documentation**:
  - `PHASE_A_CPS_SUMMARY.md`
  - `docs/architecture/EFFECT_HANDLERS_IMPLEMENTATION.md`

## Commit Message

```
[backend][cli] Integrate CPS transformation into compilation pipeline

- Add --enable-cps flag to Build command
- Enable CPS transformation in native backend (line 789-805)
- Add CPS transformation to Cranelift backend
- Add verbose output showing transformation stats
- Update help documentation
- Default: disabled (experimental feature)

CPS transformation runs after HLIR lowering, before SIR/codegen.
Only effectful functions are transformed (selective CPS).
```
