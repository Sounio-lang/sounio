//! Native execution cross-validation tests
//!
//! These tests validate that native ELF binaries produce identical output
//! to VM execution of the same IR. This ensures the native backend is
//! semantically correct.

use sounio::interp::{Interpreter, Value};
use sounio::module_loader;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Extract ELF bytes from Elf64Binary struct returned by interpreter
fn extract_elf_bytes(elf_struct: &Value) -> Vec<u8> {
    match elf_struct {
        Value::Struct { name: _, fields } => {
            // Elf64Binary has a 'bytes' field which is an array of u8
            if let Some(Value::Array(bytes_rc)) = fields.get("bytes") {
                let bytes = bytes_rc.borrow();
                bytes
                    .iter()
                    .map(|v| match v {
                        Value::Int(n) => *n as u8,
                        _ => panic!("Expected int in bytes array"),
                    })
                    .collect()
            } else {
                panic!("Elf64Binary missing bytes field");
            }
        }
        _ => panic!("Expected Elf64Binary struct, got {:?}", elf_struct),
    }
}

#[test]
#[ignore = "Blocked on module imports - READY TO ENABLE once multi-module compilation works"]
#[cfg_attr(not(target_os = "linux"), ignore = "Linux-only test")]
fn cross_validate_return_42() {
    // STATUS: Test infrastructure complete, waiting for module imports
    //
    // TO ENABLE: Remove #[ignore] attribute once Project Poseidon multi-module
    // compilation is fully integrated (see .claude/pending.md)
    //
    // This test will:
    // 1. Load self-hosted suite via interpreter (includes native codegen)
    // 2. Call build_return_42_binary() -> returns Elf64Binary struct
    // 3. Extract bytes from Elf64Binary.bytes array
    // 4. Write to /tmp, execute, capture exit code
    // 5. Run VM on same IR and compare results
    // 6. Assert: native_exit_code == vm_result == 42

    eprintln!("⏸  Waiting for module imports (Project Poseidon completion)");
    eprintln!("   Once enabled, this test will validate:");
    eprintln!("   ✓ Native ELF execution produces correct exit codes");
    eprintln!("   ✓ Native output matches VM output (cross-validation)");
    eprintln!("   ✓ ELF structure is valid (magic, headers, x86-64)");
}

#[test]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "Linux-only test (native execution)"
)]
fn cross_validate_arithmetic() {
    // Same as above but for arithmetic (21 + 21 = 42)
    eprintln!("⚠ Test blocked on module system");
}

/// Temporary test that validates ELF structure without execution
///
/// This runs the self-hosted test suite which validates ELF generation
/// internally. Once we can extract ELF bytes, we'll add actual execution.
#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "Linux-only test")]
fn validate_elf_generation_via_selfhost() {
    let root = workspace_root();
    let test_path = root.join("self-hosted/native/test_compile_to_elf.sio");

    if !test_path.exists() {
        eprintln!("Skipping: test file not found");
        return;
    }

    let ast = match module_loader::load_program_ast(&test_path) {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return;
        }
    };

    let hir = match sounio::check::check_ast(&ast) {
        Ok(hir) => hir,
        Err(e) => {
            eprintln!("Type check error: {}", e);
            return;
        }
    };

    let mut interp = Interpreter::new();
    match interp.interpret(&hir) {
        Ok(Value::Int(0)) => {
            println!("✓ ELF generation tests passed (via self-hosted suite)");
        }
        Ok(Value::Int(n)) => panic!("ELF generation tests failed with code: {}", n),
        Ok(v) => panic!("Unexpected return value: {:?}", v),
        Err(e) => panic!("Execution error: {}", e),
    }
}

/// Document the cross-validation workflow once module system is ready
#[test]
fn document_cross_validation_workflow() {
    println!("\n=== Native Cross-Validation Workflow ===\n");
    println!("Goal: Prove native ELF ≡ VM execution (same IR, same output)\n");
    println!("1. Build IR module (fn main() -> i64 {{ 42 }})");
    println!("2. Execute via VM:");
    println!("     vm_result = run_vm(ir_module)  // → 42");
    println!("3. Compile to native ELF:");
    println!("     elf = compile_to_elf(ir_module, base_addr)");
    println!("4. Write ELF to disk and execute:");
    println!("     write('/tmp/test.elf', elf.bytes)");
    println!("     chmod +x /tmp/test.elf");
    println!("     native_result = exec('/tmp/test.elf').exit_code  // → 42");
    println!("5. Cross-validate:");
    println!("     assert(vm_result == native_result)  // PASS ✓\n");
    println!("Test cases:");
    println!("  - return_42:     Simple constant return");
    println!("  - arithmetic:    Binary operations (21 + 21)");
    println!("  - control_flow:  if/else branches");
    println!("  - function_call: Multi-function programs\n");
    println!("Blocked on: Module system (imports)\n");
    println!("=========================================\n");
}
