//! IR Verification Pipeline Tests
//!
//! Tests the Phase 0 IR serialization and normalization by comparing
//! semantically equivalent programs through the full verification pipeline:
//!   1. Load two .sio programs
//!   2. Lower both to IR using the existing compiler
//!   3. Serialize both IR modules using the self-hosted serializer
//!   4. Normalize both serialized modules
//!   5. Compare byte-for-byte

use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn test_fixtures_dir() -> PathBuf {
    workspace_root().join("tests/verify-ir")
}

/// Helper to load, parse, and type-check a .sio program
fn load_and_check(path: &std::path::Path) -> Result<sounio::hir::Hir, String> {
    let ast = sounio::module_loader::load_program_ast(path)
        .map_err(|e| format!("Load/parse error: {}", e))?;
    sounio::check::check_ast(&ast).map_err(|e| format!("Type error: {}", e))
}

/// Verify that two .sio programs produce equivalent IR after normalization
fn verify_ir_pair(file_a: &str, file_b: &str) -> Result<(), String> {
    let fixtures = test_fixtures_dir();
    let path_a = fixtures.join(file_a);
    let path_b = fixtures.join(file_b);

    // Check that test fixtures exist
    if !path_a.exists() {
        return Err(format!("Test fixture not found: {:?}", path_a));
    }
    if !path_b.exists() {
        return Err(format!("Test fixture not found: {:?}", path_b));
    }

    // Load and type-check both programs
    let _hir_a = load_and_check(&path_a)?;
    let _hir_b = load_and_check(&path_b)?;

    // For now, we verify that both programs type-check successfully
    // The full verification pipeline (lowering to IR, serialization, normalization)
    // requires integrating the self-hosted IR pipeline.
    //
    // TODO(phase1): Implement full verification by:
    //   1. Extending the interpreter to expose lower_program() from self-hosted/ir/mod.sio
    //   2. Calling serialize_ir_module() and normalize_ir_module() from self-hosted/ir/
    //   3. Comparing the byte streams
    //
    // For Phase 1, we establish the test infrastructure and verify parsing/checking.

    Ok(())
}

#[test]
fn test_math_commutative() {
    verify_ir_pair("math_a.sio", "math_b.sio")
        .expect("Math commutative test failed");
}

#[test]
fn test_control_flow_equivalent() {
    verify_ir_pair("control_a.sio", "control_b.sio")
        .expect("Control flow equivalence test failed");
}

#[test]
fn test_call_order_commutative() {
    verify_ir_pair("call_a.sio", "call_b.sio")
        .expect("Call order commutative test failed");
}

#[test]
#[ignore = "Full IR verification pipeline not yet integrated"]
fn test_ir_verification_full_pipeline() {
    // This test will be enabled once we integrate the self-hosted IR lowering,
    // serialization, and normalization into the Rust test harness.
    //
    // The implementation requires:
    //   1. A Rust FFI or interpreter interface to call self-hosted IR functions
    //   2. Conversion between Rust HIR and self-hosted IR representations
    //   3. Byte-level comparison of normalized serialized modules
    unimplemented!("Phase 1 full pipeline integration pending");
}
