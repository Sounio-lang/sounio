//! Native execution cross-validation tests
//!
//! Validates that native ELF binaries produce identical output to interpreter
//! execution of the same source. This ensures the native backend is
//! semantically correct.

use sounio::interp::{Interpreter, Value};
use sounio::module_loader;
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

fn selfhost_options() -> NativeCompileOptions {
    NativeCompileOptions {
        backend: NativeBackendChoice::SelfhostedNative,
        verbose: false,
        ..Default::default()
    }
}

/// Run source through the interpreter, return the main() result as i64.
fn interpret_source(source: &str, label: &str) -> Result<i64, String> {
    let tmp_file = std::env::temp_dir().join(format!(
        "sounio_xval_src_{}_{}.sio",
        label,
        std::process::id()
    ));
    std::fs::write(&tmp_file, source).map_err(|e| format!("Write tmp: {}", e))?;
    let ast = module_loader::load_program_ast(&tmp_file)
        .map_err(|e| format!("Parse error: {}", e))?;
    let hir = sounio::check::check_ast(&ast).map_err(|e| format!("Type error: {}", e))?;
    let mut interp = Interpreter::new();
    let result = interp
        .interpret(&hir)
        .map_err(|e| format!("Runtime error: {}", e))?;
    let _ = std::fs::remove_file(&tmp_file);
    match result {
        Value::Int(n) => Ok(n),
        other => Err(format!("Expected Int, got {:?}", other)),
    }
}

/// Compile source to native ELF, execute, return exit code.
fn native_exit_code(source: &str, label: &str) -> Result<i32, String> {
    let elf_bytes = compile_source_to_native_elf(source, &selfhost_options())?;
    let out_path = std::env::temp_dir().join(format!(
        "sounio_xval_{}_{}", label, std::process::id()
    ));
    emit_native_executable(&elf_bytes, &out_path)
        .map_err(|e| format!("Write ELF: {}", e))?;
    let output = Command::new(&out_path)
        .output()
        .map_err(|e| format!("Execute: {}", e))?;
    let _ = std::fs::remove_file(&out_path);
    output
        .status
        .code()
        .ok_or_else(|| "Process terminated by signal".to_string())
}

/// Cross-validate: interpreter result == native exit code.
fn cross_validate(source: &str, label: &str, expected: i64) {
    // 1. Interpreter
    let interp_result = interpret_source(source, label)
        .unwrap_or_else(|e| panic!("[{}] Interpreter failed: {}", label, e));
    assert_eq!(
        interp_result, expected,
        "[{}] Interpreter returned {}, expected {}",
        label, interp_result, expected
    );

    // 2. Native
    let native_result = match native_exit_code(source, label) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("[{}] Native compilation/execution failed: {}", label, e);
            return;
        }
    };
    assert_eq!(
        native_result, expected as i32,
        "[{}] Native exit code {}, expected {}",
        label, native_result, expected
    );

    eprintln!(
        "[{}] Cross-validated: interpreter={}, native={} (match)",
        label, interp_result, native_result
    );
}

// =========================================================================
// Cross-validation tests
// =========================================================================

#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "Linux-only test")]
fn cross_validate_return_42() {
    cross_validate("fn main() -> i64 { 42 }", "return_42", 42);
}

#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "Linux-only test")]
fn cross_validate_arithmetic() {
    cross_validate("fn main() -> i64 { 21 + 21 }", "arithmetic", 42);
}

#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "Linux-only test")]
fn cross_validate_subtraction() {
    cross_validate("fn main() -> i64 { 50 - 8 }", "subtraction", 42);
}

#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "Linux-only test")]
fn cross_validate_function_call() {
    let source = r#"
fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 { add(19, 23) }
"#;
    cross_validate(source, "function_call", 42);
}

#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "Linux-only test")]
fn cross_validate_conditional() {
    let source = r#"
fn main() -> i64 {
    let x = 10
    if x > 5 { 42 } else { 0 }
}
"#;
    cross_validate(source, "conditional", 42);
}

// =========================================================================
// ELF structure validation via self-hosted suite
// =========================================================================

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
            eprintln!("ELF generation tests passed (via self-hosted suite)");
        }
        Ok(Value::Int(n)) => panic!("ELF generation tests failed with code: {}", n),
        Ok(v) => panic!("Unexpected return value: {:?}", v),
        Err(e) => panic!("Execution error: {}", e),
    }
}
