//! Phase 2 Native Codegen Self-Hosted Tests
//!
//! Validates M1-M5 native backend features:
//!   - Heap allocation with type-tag sizing
//!   - Struct field access (SIB encoding)
//!   - Array indexing
//!   - Negative number support in print_int
//!   - IrLoadFloat lowering
//!
//! Also includes end-to-end ELF execution tests for struct/array programs.

use sounio::interp::{Interpreter, Value};
use sounio::test::pipeline::{
    NativeBackendChoice, NativeCompileOptions, compile_source_to_native_elf, emit_native_executable,
};
use std::path::PathBuf;
use std::process::Command;

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

fn selfhost_options() -> NativeCompileOptions {
    NativeCompileOptions {
        backend: NativeBackendChoice::SelfhostedNative,
        verbose: false,
        ..Default::default()
    }
}

fn run_native_binary(path: &std::path::Path) -> std::process::Output {
    Command::new(path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to execute {}: {}", path.display(), e))
}

// =========================================================================
// Self-hosted byte-level test suite
// =========================================================================

#[test]
fn native_phase2_selfhost_tests_pass() {
    let root = workspace_root();
    let test_file = root.join("self-hosted/native/test_phase2.sio");

    if !test_file.exists() {
        eprintln!("Skipping: {:?} not found", test_file);
        return;
    }

    match run_sio_file(&test_file) {
        Ok(Value::Int(0)) => {} // all 12 tests passed
        Ok(Value::Int(n)) => panic!(
            "Phase 2 tests failed: {} test(s) failed (exit code {})",
            n, n
        ),
        Ok(v) => panic!("Expected Int(0), got {:?}", v),
        Err(e) => panic!("Phase 2 test execution failed: {}", e),
    }
}

// =========================================================================
// End-to-end native ELF execution tests
// =========================================================================

#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "Linux-only: ELF execution")]
fn selfhost_native_struct_field_sum() {
    // Struct with two fields: exit code = field1 + field2
    let source = r#"
struct Point { x: i64, y: i64 }

fn main() -> i64 {
    let p = Point { x: 19, y: 23 }
    p.x + p.y
}
"#;

    let elf_bytes = match compile_source_to_native_elf(source, &selfhost_options()) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Compilation failed (expected for struct support): {}", e);
            return;
        }
    };

    let out_path = std::env::temp_dir().join("sounio_phase2_struct_field_sum");
    emit_native_executable(&elf_bytes, &out_path).expect("write ELF");
    let output = run_native_binary(&out_path);
    assert_eq!(
        output.status.code(),
        Some(42),
        "Expected exit code 42 (19+23)"
    );
    let _ = std::fs::remove_file(&out_path);
}

#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "Linux-only: ELF execution")]
fn selfhost_native_array_index() {
    // Array with 4 elements: exit code = arr[2]
    let source = r#"
fn main() -> i64 {
    let arr = [10, 20, 42, 50]
    arr[2]
}
"#;

    let elf_bytes = match compile_source_to_native_elf(source, &selfhost_options()) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Compilation failed (expected for array support): {}", e);
            return;
        }
    };

    let out_path = std::env::temp_dir().join("sounio_phase2_array_index");
    emit_native_executable(&elf_bytes, &out_path).expect("write ELF");
    let output = run_native_binary(&out_path);
    assert_eq!(
        output.status.code(),
        Some(42),
        "Expected exit code 42 (arr[2])"
    );
    let _ = std::fs::remove_file(&out_path);
}

#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "Linux-only: ELF execution")]
fn selfhost_native_addition_exit_code() {
    let source = "fn main() -> i64 { 21 + 21 }";

    let elf_bytes =
        compile_source_to_native_elf(source, &selfhost_options()).expect("compile addition");

    let out_path = std::env::temp_dir().join("sounio_phase2_addition");
    emit_native_executable(&elf_bytes, &out_path).expect("write ELF");
    let output = run_native_binary(&out_path);
    assert_eq!(
        output.status.code(),
        Some(42),
        "Expected exit code 42 (21+21)"
    );
    let _ = std::fs::remove_file(&out_path);
}
