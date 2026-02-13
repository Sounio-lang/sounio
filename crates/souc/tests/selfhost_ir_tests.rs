//! Self-Hosted IR Tests Runner
//!
//! Runs the self-hosted IR test suite (test_ir.sio) which includes:
//! - IR lowering tests (T01-T16)
//! - Phase 0 serialization/normalization tests (T17-T21)

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
#[ignore = "test_ir.sio requires directory compilation mode - use cargo test native_phase1_selfhost_tests_pass"]
fn selfhost_ir_tests_pass() {
    let root = workspace_root();
    let test_file = root.join("self-hosted/test_ir.sio");

    if !test_file.exists() {
        eprintln!("Skipping: {:?} not found", test_file);
        return;
    }

    match run_sio_file(&test_file) {
        Ok(Value::Int(0)) => {
            println!("✓ Self-hosted IR tests passed!");
            println!("  - IR lowering tests (T01-T16): OK");
            println!("  - Phase 0 serialization tests (T17-T21): OK");
        }
        Ok(Value::Int(n)) => panic!("IR tests failed: {} test(s) failed (exit code {})", n, n),
        Ok(v) => panic!("Expected Int(0), got {:?}", v),
        Err(e) => panic!("IR test execution failed: {}", e),
    }
}
