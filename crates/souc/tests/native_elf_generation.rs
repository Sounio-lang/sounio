//! Native ELF Generation Test
//!
//! Tests that the self-hosted native backend can compile IR to ELF64 binaries.
//! This is a critical milestone for the moonshot: native compilation without Rust runtime.

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
fn native_elf_generation_tests_pass() {
    let root = workspace_root();
    let test_file = root.join("self-hosted/native/test_compile_to_elf.sio");

    if !test_file.exists() {
        eprintln!("Skipping: {:?} not found", test_file);
        return;
    }

    println!("Running ELF generation tests...");

    match run_sio_file(&test_file) {
        Ok(Value::Int(0)) => {
            println!("✅ ELF generation tests PASSED!");
            println!("   T1: Simple function → valid ELF64 magic, headers");
            println!("   T2: Arithmetic operations → code section present");
            println!("   T3: Function calls → relocations applied");
        }
        Ok(Value::Int(n)) => panic!(
            "ELF generation tests FAILED: {} test(s) failed (exit code {})",
            n, n
        ),
        Ok(v) => panic!("Expected Int(0), got {:?}", v),
        Err(e) => {
            eprintln!("⚠️  Test execution failed: {}", e);
            eprintln!("This is expected if the native backend isn't fully wired yet.");
            eprintln!("Missing functions: compile_to_elf(), ir_empty_module(), etc.");
            panic!("ELF generation test failed to execute: {}", e);
        }
    }
}
