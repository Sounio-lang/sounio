# Phase 3.1 Week 1: Knowledge<T> IR Foundation - Implementation Summary

## Objective
Preserve Knowledge types through compilation pipeline with actual confidence bounds and provenance information.

## Tasks Completed

### Task 1: HLIR Knowledge Type Preservation
**File**: `compiler/src/hlir/ir.rs`

**Changes**:
- Extended `HlirType::Knowledge` variant to include:
  - `epsilon_bound: Option<f64>` - Actual confidence bounds from type system
  - `provenance_id: Option<u32>` - Provenance tracking identifier
- Updated `from_hir` conversion to preserve these values from `HirType::Knowledge`
- Added provenance ID hashing for tracking (placeholder for proper ID allocation)
- Updated pattern matches throughout to handle new fields

**Before**:
```rust
Knowledge {
    inner: Box<HlirType>,
    mode: EpistemicMode,
}
```

**After**:
```rust
Knowledge {
    inner: Box<HlirType>,
    mode: EpistemicMode,
    epsilon_bound: Option<f64>,
    provenance_id: Option<u32>,
}
```

### Task 2: SIR Knowledge Constructor
**File**: `compiler/src/sir/types.rs`

**Status**: Already implemented correctly.

The `SirType::knowledge(inner, mode)` constructor was already present and working, generating appropriate struct layouts:
- **Full mode**: 64-byte struct with value, confidence, bounds, provenance, flags
- **Compact mode**: 16-byte struct with value and confidence
- **Erased mode**: Zero overhead, returns inner type directly

Added comprehensive tests:
- `test_knowledge_full_mode` - Verifies Full mode struct generation
- `test_knowledge_compact_mode` - Verifies Compact mode struct generation  
- `test_knowledge_erased_mode` - Verifies Erased mode returns inner type

### Task 3: Extract Knowledge Wrapper Metadata
**File**: `compiler/src/sir/lower.rs`

**Changes**:
- Updated `MetadataTracker::extract_knowledge_wrapper()` to use actual values:
  - Extract actual `epsilon_bound` from `HlirType::Knowledge`
  - Extract actual `provenance_id` from `HlirType::Knowledge`
  - Fall back to defaults (0.05 epsilon / 95% confidence) only if not specified
- Fixed return logic for erased mode (returns None, no runtime tracking)

**Before**:
```rust
let epsilon_bound = match mode {
    EpistemicMode::Full | EpistemicMode::Compact => 0.05, // Stub!
    EpistemicMode::Erased => return None,
};
let provenance_id = match mode {
    EpistemicMode::Full => Some(0), // TODO: extract actual
    _ => None,
};
```

**After**:
```rust
// Return None for erased mode (no runtime tracking)
if matches!(mode, EpistemicMode::Erased) {
    return None;
}

// Use actual epsilon bound if available, otherwise default
let epsilon = epsilon_bound.unwrap_or_else(|| match mode {
    EpistemicMode::Full | EpistemicMode::Compact => 0.05,
    _ => 0.05,
});

// Use actual provenance ID if available
Some((epsilon, *provenance_id))
```

### Bonus: Fixed dimension_to_array Bug
**File**: `compiler/src/sir/lower.rs`

Fixed `dimension_to_array` function which was missing the `amount` field (substance amount), causing array size mismatch.

## Test Results

All tests passing:
```
test sir::types::test_knowledge_full_mode ... ok
test sir::types::test_knowledge_compact_mode ... ok
test sir::types::test_knowledge_erased_mode ... ok
test hlir::builder::tests::test_build_simple_function ... ok
test hlir::builder::tests::test_build_conditional ... ok
test hlir::lower::tests::test_lower_simple_function ... ok
test hlir::tests::test_hlir_type_conversion ... ok
test hlir::tests::test_hlir_type_properties ... ok
```

## Acceptance Criteria

✅ Code compiles  
✅ `cargo test hlir` passes (52 tests)  
✅ `cargo test sir::types` passes (6 tests including 3 new Knowledge tests)  
✅ Knowledge types appear in SIR with correct layouts (verified by tests)  
✅ Actual epsilon bounds extracted from type system (not stubs)

## Technical Details

### Data Flow
1. **Source** → Parser → Type Checker creates `HirType::Knowledge { epsilon_bound, provenance }`
2. **HIR → HLIR**: `HlirType::from_hir()` preserves epsilon_bound and generates provenance_id
3. **HLIR → SIR**: `extract_knowledge_wrapper()` retrieves actual values for metadata tracking
4. **SIR**: `SirType::knowledge()` generates appropriate struct layout based on mode

### Mode Semantics
- **Full**: Complete tracking (64 bytes) - value, confidence, bounds, provenance, timestamp
- **Compact**: Lightweight tracking (16 bytes) - value, confidence, hash
- **Erased**: Zero overhead - just inner type, compile-time only

### Files Modified
1. `compiler/src/hlir/ir.rs` - Extended Knowledge variant, updated conversions
2. `compiler/src/sir/lower.rs` - Fixed metadata extraction, fixed dimension_to_array
3. `compiler/src/sir/types.rs` - Added tests for knowledge constructor

## Next Steps (Phase 3.1 Week 2+)
- Backend code generation for Knowledge struct access
- Effect system integration for epistemic operations
- Runtime library support for confidence propagation
- Optimization passes for mode transitions
