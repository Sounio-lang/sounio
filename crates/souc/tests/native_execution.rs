//! Native execution validation tests
//!
//! These tests compile Sounio source to native x86-64 ELF executables
//! via the self-hosted compiler, write them to disk, and execute them.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;

#[test]
#[ignore] // Enable when native backend is wired
fn native_compile_and_execute_return_42() {
    // Simple program: fn main() -> i64 { 42 }
    let source = r#"
fn main() -> i64 {
    42
}
"#;

    // Compile to native ELF via self-hosted backend
    let elf_bytes = compile_to_native_elf(source);

    // Write to temp file
    let temp_path = "/tmp/sounio_test_return_42";
    fs::write(temp_path, &elf_bytes).expect("Failed to write ELF");
    fs::set_permissions(temp_path, fs::Permissions::from_mode(0o755))
        .expect("Failed to set executable");

    // Execute
    let output = Command::new(temp_path)
        .output()
        .expect("Failed to execute");

    // Verify exit code
    assert_eq!(output.status.code(), Some(42), "Expected exit code 42");

    // Cleanup
    let _ = fs::remove_file(temp_path);
}

#[test]
#[ignore] // Enable when native backend is wired
fn native_compile_and_execute_addition() {
    let source = r#"
fn main() -> i64 {
    let x = 10
    let y = 32
    x + y
}
"#;

    let elf_bytes = compile_to_native_elf(source);
    let temp_path = "/tmp/sounio_test_addition";

    fs::write(temp_path, &elf_bytes).unwrap();
    fs::set_permissions(temp_path, fs::Permissions::from_mode(0o755)).unwrap();

    let output = Command::new(temp_path).output().unwrap();
    assert_eq!(output.status.code(), Some(42));

    let _ = fs::remove_file(temp_path);
}

#[test]
#[ignore] // Enable when native backend is wired
fn native_compile_multifile_with_imports() {
    // This will test the full pipeline:
    // Multi-module source → parse → resolve → check → lower → link → native ELF

    // For now, placeholder until we wire the pipeline
    todo!("Wire multi-module compilation to native backend");
}

/// Compile Sounio source to native ELF executable
///
/// Pipeline:
/// 1. Parse source to AST
/// 2. Type check to HIR
/// 3. Lower to IR
/// 4. Call self-hosted compile_to_elf() via interpreter
/// 5. Extract ELF bytes from returned struct
fn compile_to_native_elf(_source: &str) -> Vec<u8> {
    // TODO: Wire the full pipeline
    // 1. Use existing Rust frontend (lexer/parser/checker)
    // 2. Lower to IrModule
    // 3. Call self-hosted compile_to_elf() via interpreter
    // 4. Extract Elf64Binary.bytes and return

    todo!("Implement native compilation pipeline")
}
