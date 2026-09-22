<!-- docs:meta
topic_id: repo.docs.compiler.sir-passes
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.sir-passes
-->

# SIR Transformation Passes

> **⚠️ File paths updated 2026-07-11 (doc-reality audit).** This page was written against the retired Rust compiler tree (`crates/`, `compiler/src/*.rs`, `codegen/llvm/`); those files no longer exist — the compiler is self-hosted Sounio (Madaros v0.80.0). The design and concepts below remain accurate, but the SIR/IR passes now live in `self-hosted/ir/` (`const_prop.sio`, `dce.sio`, `inline.sio`, `loop_opt.sio`, `lower.sio`, `normalize.sio`, `optimize.sio`, …) and refinement handling in `self-hosted/check/refinement.sio` — not any `compiler/src/sir/*.rs` or `compiler/src/types/*.rs`. Do not look for the `.rs` paths below.


This document describes the modular SIR transformation pass infrastructure.

## Architecture

Passes are organized in `compiler/src/sir/passes/` as independent modules, replacing the monolithic `passes.rs` file. Each pass:

- Implements its own analysis and transformation logic
- Returns structured results (modifications, statistics)
- Has dedicated unit tests
- Can be composed into optimization pipelines

## Implemented Passes

### Unit Check Insertion Pass

**File**: `unit_check_insertion.rs`  
**Purpose**: Verify dimensional consistency in physical computations

**Strategy**:
1. Collect values with unit metadata (mg, mL, h, etc.)
2. Analyze arithmetic operations (add, sub, mul, div)
3. Check dimensional compatibility:
   - Add/Sub: Units must be identical
   - Mul/Div: Units combine dimensionally (always valid)
4. Insert runtime checks or conversions as needed

**Configuration**:
- `auto_convert`: Enable automatic unit conversion (mg→g)
- `strict`: Panic on mismatch vs degrade confidence

**Example**:
```rust
let mut pass = UnitCheckInsertion::new();
pass.strict = true;
let result = pass.run(&mut module);
println!("Checks inserted: {}", result.checks_inserted);
```

**Limitations**:
- Currently only detects mismatches, conversion not yet implemented
- Relies on metadata being attached to values
- Does not handle user-defined unit types

### Refinement Assertion Pass

**File**: `refine_assert.rs`  
**Purpose**: Insert runtime checks for refinement type predicates

**Strategy**:
1. Scan function signatures for refinement types
2. Extract predicates (e.g., `{ x: i32 | x > 0 }`)
3. Attempt static proof using SMT solver
4. Insert `Assert` instruction where proof fails

**SMT Integration**:
- Handles trivial cases (constant comparisons) locally
- Can invoke Z3 for complex predicates (when `use_smt` enabled)
- Conservative fallback: insert check if uncertain

**Configuration**:
- `use_smt`: Enable Z3 solver for static verification
- `strict`: Panic on violation vs degrade confidence

**Example**:
```rust
let mut pass = RefinementAssertionPass::new();
pass.use_smt = true;
let result = pass.run(&mut module);
println!("Proofs succeeded: {}/{}", 
    result.proofs_succeeded, result.proofs_attempted);
```

**Limitations**:
- Z3 integration not yet wired (uses stub)
- Only supports simple integer bounds (<, >, <=, >=, ==)
- Complex predicates with quantifiers treated conservatively

## Assert Instruction Enhancement

The `SirInst::Assert` instruction was enhanced with a `FailureMode` enum:

```rust
pub enum FailureMode {
    Trap,                    // ud2 instruction (immediate halt)
    Panic,                   // Runtime panic with message
    DegradeConfidence(f64),  // Epistemic fallback (reduce confidence)
}
```

This enables passes to choose appropriate failure handling:
- **Trap**: Hard real-time constraints, undefined behavior
- **Panic**: Development/debugging, clear error messages
- **DegradeConfidence**: Probabilistic inference, graceful degradation

## Testing

**Unit Tests**: Each pass has `#[cfg(test)]` module with logic tests

**Integration Tests**: `compiler/tests/sir_passes.rs` verifies:
- Pass instantiation
- Configuration options
- Physical unit operations
- Result structure correctness

**Running Tests**:
```bash
cargo test sir_passes           # All SIR pass tests
cargo test unit_check           # Unit check pass only
cargo test refinement_pass      # Refinement pass only
```

## Future Work

1. **Unit Conversion**: Implement actual scale conversion (mg→g)
2. **Z3 Integration**: Wire SMT solver for complex predicates
3. **Pass Manager**: Pipeline composition with dependency tracking
4. **More Passes**:
   - Loop invariant hoisting for epistemic values
   - Distribution fusion (combine adjacent sampling ops)
   - ODE step coalescing
   - Automatic differentiation optimization

## Related Files

- `compiler/src/sir/ops.rs` - FailureMode enum and Assert instruction
- `compiler/src/sir/emit.rs` - Native code generation for assertions
- `compiler/src/types/refinement.rs` - Refinement predicate AST
- `compiler/src/sir/values.rs` - PhysicalUnit representation
