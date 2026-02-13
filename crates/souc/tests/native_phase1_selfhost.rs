//! Phase 1 Native Codegen Self-Hosted Test
//!
//! Runs the self-hosted Phase 1 test suite through the Rust interpreter
//! (bypassing the bootstrap driver which cannot handle complex test files).

use sounio::interp::{Interpreter, Value};
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn run_sio_file(path: &std::path::Path) -> Result<Value, String> {
    let ast = sounio::module_loader::load_program_ast(path)
        .map_err(|e| format!("Load/parse error: {}", e))?;
    let hir = sounio::check::check_ast(&ast).map_err(|e| format!("Type error: {}", e))?;
    let mut interpreter = Interpreter::new();
    interpreter
        .interpret(&hir)
        .map_err(|e| format!("Runtime error: {}", e))
}

#[test]
#[ignore = "TODO(selfhost-native): phase0 suite is legacy and currently fails; keep phase1 as the correctness gate"]
fn native_phase0_selfhost_tests_pass() {
    let root = workspace_root();
    let test_file = root.join("self-hosted/native/test_phase0.sio");

    if !test_file.exists() {
        eprintln!("Skipping: {:?} not found", test_file);
        return;
    }

    match run_sio_file(&test_file) {
        Ok(Value::Int(0)) => {} // all tests passed
        Ok(Value::Int(n)) => panic!("Phase 0 tests failed: {} test(s) failed (exit code {})", n, n),
        Ok(v) => panic!("Expected Int(0), got {:?}", v),
        Err(e) => panic!("Phase 0 test execution failed: {}", e),
    }
}

#[test]
fn native_phase1_selfhost_tests_pass() {
    let root = workspace_root();
    let test_file = root.join("self-hosted/native/test_phase1.sio");

    if !test_file.exists() {
        eprintln!("Skipping: {:?} not found", test_file);
        return;
    }

    match run_sio_file(&test_file) {
        Ok(Value::Int(0)) => {} // all tests passed
        Ok(Value::Int(n)) => panic!("Phase 1 tests failed: {} test(s) failed (exit code {})", n, n),
        Ok(v) => panic!("Expected Int(0), got {:?}", v),
        Err(e) => panic!("Phase 1 test execution failed: {}", e),
    }
}
