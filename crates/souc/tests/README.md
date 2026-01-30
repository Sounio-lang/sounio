# Integration Tests

This directory contains integration tests for the Sounio compiler.

## Structure

- `*.rs` - Integration test files (each file is compiled as a separate test binary)
- `e2e/` - End-to-end test modules
- `phase2_integration/` - Phase 2 optimization integration tests
- `fixtures/` - Test fixtures and sample code
- `golden/` - Golden test data for snapshot testing

## Running Tests

```bash
# Run all integration tests
cargo test -p souc --tests

# Run a specific integration test
cargo test -p souc --test backend_native_tests

# Run with output
cargo test -p souc --tests -- --nocapture
```

## Test Status

### Working Tests ✅

Most integration tests compile and run successfully, including:
- `backend_native_tests.rs` (8 tests)
- `backend_e2e.rs`
- `epistemic_integration.rs`
- GPU integration tests
- Various other integration tests

### Known Issues ⚠️

Some MIR-related tests have outdated API usage and need updates:

- `mir_pipeline_tests.rs` - Uses old FunctionBuilder API
- `mir_regression_tests.rs` - Uses old FunctionBuilder API
- `mir_optimization_tests.rs` - May have similar issues

**Common API changes:**
- `set_block()` → `switch_to_block()`
- `create_block()` requires a label parameter
- `build_compare()` requires a `MirCompareOp` parameter
- Added `get_param()` method to FunctionBuilder
- Added `is_ok()` method to SSAValidationResult

**Fixes applied:**
- Added `FunctionBuilder::get_param(index)` method
- Added `SSAValidationResult::is_ok()` method
- Fixed import statements to include `MIRPass` trait
- Updated some `set_block` calls to `switch_to_block`
- Updated some `build_compare` calls with correct operators

**Remaining work:**
- Some tests still need mutability fixes (`let mut`)
- Some tests need complete API migration

## Organization

Tests were reorganized from `/home/demetrios/sounio-1/tests/integration/` to
`/home/demetrios/sounio-1/crates/souc/tests/` as part of workspace restructuring.

This follows Rust convention where integration tests live in the `tests/`
directory of each crate.
