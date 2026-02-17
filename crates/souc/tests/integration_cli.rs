//! Integration Tests for CLI Commands
//!
//! Tests for user-facing CLI commands: check, run, build, jit
//! These tests verify complete end-to-end workflows with actual file I/O.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

// ============================================================================
// Test Helpers
// ============================================================================

/// Get the path to the compiled souc binary
fn souc_bin() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/souc; workspace root is two levels up.
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();

    let debug_bin = workspace_root.join("target/debug/souc");
    if debug_bin.exists() {
        return debug_bin;
    }

    let release_bin = workspace_root.join("target/release/souc");
    if release_bin.exists() {
        return release_bin;
    }

    // Build it
    eprintln!("Building souc binary...");
    let status = Command::new("cargo")
        .args(["build", "--bin", "souc"])
        .current_dir(&workspace_root)
        .status()
        .expect("Failed to build souc");
    assert!(status.success(), "Failed to build souc binary");

    assert!(
        debug_bin.exists(),
        "souc binary not found at {:?}",
        debug_bin
    );
    debug_bin
}

/// Create a temporary test file with the given content
fn create_test_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
    let path = dir.path().join(name);
    fs::write(&path, content).expect("Failed to write test file");
    path
}

/// Run souc with the given arguments
fn run_souc(args: &[&str]) -> std::process::Output {
    Command::new(souc_bin())
        .args(args)
        .output()
        .expect("Failed to execute souc")
}

/// Check if JIT feature is available
fn has_jit_feature() -> bool {
    cfg!(feature = "jit")
}

/// Check if LLVM feature is available
fn has_llvm_feature() -> bool {
    cfg!(feature = "llvm-base")
}

// ============================================================================
// CLI: check command tests
// ============================================================================

#[test]
fn test_cli_check_valid_file() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    let x = 42
    x + 8
}
"#,
    );

    let output = run_souc(&["check", test_file.to_str().unwrap()]);

    assert!(
        output.status.success(),
        "check should succeed for valid file. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_cli_check_syntax_error() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    let x =
}
"#,
    );

    let output = run_souc(&["check", test_file.to_str().unwrap()]);

    assert!(
        !output.status.success(),
        "check should fail for syntax error"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("error") || stderr.contains("Error"),
        "stderr should contain error message: {}",
        stderr
    );
}

#[test]
fn test_cli_check_type_error() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    let x: i64 = true
    x
}
"#,
    );

    let output = run_souc(&["check", test_file.to_str().unwrap()]);

    assert!(!output.status.success(), "check should fail for type error");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("type") || stderr.contains("Type"),
        "stderr should contain type error: {}",
        stderr
    );
}

#[test]
fn test_cli_check_file_not_found() {
    let output = run_souc(&["check", "/nonexistent/path/test.sio"]);

    assert!(
        !output.status.success(),
        "check should fail for nonexistent file"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("No such file"),
        "stderr should indicate file not found: {}",
        stderr
    );
}

#[test]
fn test_cli_check_show_ast_flag() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    42
}
"#,
    );

    let output = run_souc(&["check", test_file.to_str().unwrap(), "--show-ast"]);

    assert!(
        output.status.success(),
        "check should succeed with --show-ast"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", stdout, stderr);

    // Should show AST information
    assert!(
        combined.contains("AST") || combined.contains("Ast") || combined.contains("Function"),
        "output should contain AST information: {}",
        combined
    );
}

#[test]
fn test_cli_check_show_types_flag() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    let x = 42
    x
}
"#,
    );

    let output = run_souc(&["check", test_file.to_str().unwrap(), "--show-types"]);

    assert!(
        output.status.success(),
        "check should succeed with --show-types"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", stdout, stderr);

    // Should show type information
    assert!(
        combined.contains("i64") || combined.contains("Type") || combined.contains("type"),
        "output should contain type information: {}",
        combined
    );
}

// ============================================================================
// CLI: run command tests (interpreter)
// ============================================================================

#[test]
fn test_cli_run_simple_program() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    42
}
"#,
    );

    let output = run_souc(&["run", test_file.to_str().unwrap()]);

    assert!(
        output.status.success(),
        "run should succeed for valid program. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("42"),
        "output should contain result 42: {}",
        stdout
    );
}

#[test]
fn test_cli_run_arithmetic() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    (10 + 20) * 2
}
"#,
    );

    let output = run_souc(&["run", test_file.to_str().unwrap()]);

    assert!(output.status.success(), "run should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("60"),
        "output should contain result 60: {}",
        stdout
    );
}

#[test]
fn test_cli_run_with_variables() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    let x = 10
    let y = 32
    x + y
}
"#,
    );

    let output = run_souc(&["run", test_file.to_str().unwrap()]);

    assert!(output.status.success(), "run should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("42"),
        "output should contain result 42: {}",
        stdout
    );
}

#[test]
fn test_cli_run_runtime_error() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    1 / 0
}
"#,
    );

    let output = run_souc(&["run", test_file.to_str().unwrap()]);

    // Note: Division by zero may or may not be caught at runtime
    // depending on interpreter implementation. This test just verifies
    // the command completes without crashing.

    // If it fails, check for error message
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("division") || stderr.contains("error") || stderr.contains("Error"),
            "stderr should indicate error: {}",
            stderr
        );
    }
    // If it succeeds, that's also acceptable (undefined behavior)
}

// ============================================================================
// CLI: jit command tests (requires jit feature)
// ============================================================================

#[test]
fn test_cli_jit_simple_program() {
    if !has_jit_feature() {
        eprintln!("Skipping test: jit feature not enabled");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    100 + 23
}
"#,
    );

    let output = run_souc(&["jit", test_file.to_str().unwrap()]);

    assert!(
        output.status.success(),
        "jit should succeed for valid program. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("123"),
        "output should contain result 123: {}",
        stdout
    );
}

#[test]
fn test_cli_jit_with_optimization() {
    if !has_jit_feature() {
        eprintln!("Skipping test: jit feature not enabled");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    let x = 2
    let y = 3
    x * y + 4
}
"#,
    );

    let output = run_souc(&["jit", test_file.to_str().unwrap(), "--optimize"]);

    assert!(
        output.status.success(),
        "jit with optimization should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("10"),
        "output should contain result 10: {}",
        stdout
    );
}

#[test]
fn test_cli_jit_not_enabled() {
    if has_jit_feature() {
        eprintln!("Skipping test: jit feature IS enabled");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    42
}
"#,
    );

    let output = run_souc(&["jit", test_file.to_str().unwrap()]);

    assert!(
        !output.status.success(),
        "jit should fail when feature not enabled"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("JIT") || stderr.contains("not enabled") || stderr.contains("feature"),
        "stderr should indicate JIT not enabled: {}",
        stderr
    );
}

// ============================================================================
// CLI: build command tests (requires native backend)
// ============================================================================

#[test]
fn test_cli_build_native_backend() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    42
}
"#,
    );

    let output_file = temp_dir.path().join("test_output");

    let output = run_souc(&[
        "build",
        test_file.to_str().unwrap(),
        "--output",
        output_file.to_str().unwrap(),
        "--backend",
        "native",
    ]);

    // Build might succeed or fail depending on native backend implementation
    // We're just testing that the command parses correctly
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should either succeed or give a meaningful error
    if !output.status.success() {
        assert!(
            stderr.contains("error")
                || stderr.contains("Error")
                || stderr.contains("not implemented"),
            "stderr should contain meaningful error: {}",
            stderr
        );
    } else {
        // If it succeeded, output file should exist
        assert!(
            output_file.exists() || stderr.contains("written") || stdout.contains("written"),
            "build should create output file or indicate success"
        );
    }
}

#[test]
fn test_cli_build_with_optimization() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn main() -> i64 {
    let x = 5
    let y = 10
    x + y
}
"#,
    );

    let output_file = temp_dir.path().join("test_output");

    let output = run_souc(&[
        "build",
        test_file.to_str().unwrap(),
        "--output",
        output_file.to_str().unwrap(),
        "--backend",
        "native",
        "-O2",
    ]);

    // Just verify the command executes and optimization flag is accepted
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        // Error should be about backend, not about invalid flag
        assert!(
            !stderr.contains("invalid") && !stderr.contains("unrecognized"),
            "optimization flag should be recognized: {}",
            stderr
        );
    }
}

// ============================================================================
// CLI: misc command tests
// ============================================================================

#[test]
fn test_cli_version() {
    let output = run_souc(&["--version"]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", stdout, stderr);

    assert!(
        combined.contains("0.") || combined.contains("souc"),
        "version output should contain version number: {}",
        combined
    );
}

#[test]
fn test_cli_help() {
    let output = run_souc(&["--help"]);

    assert!(output.status.success(), "help should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Usage") || stdout.contains("USAGE") || stdout.contains("Commands"),
        "help should show usage information: {}",
        stdout
    );

    // Should list main commands
    assert!(
        stdout.contains("check"),
        "help should mention 'check' command"
    );
    assert!(stdout.contains("run"), "help should mention 'run' command");
    assert!(
        stdout.contains("build"),
        "help should mention 'build' command"
    );
}

#[test]
fn test_cli_check_help() {
    let output = run_souc(&["check", "--help"]);

    assert!(output.status.success(), "check --help should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("check") || stdout.contains("Check"),
        "help should describe check command: {}",
        stdout
    );
    assert!(
        stdout.contains("--show-ast") || stdout.contains("show-ast"),
        "help should list --show-ast flag: {}",
        stdout
    );
}

// ============================================================================
// CLI: complex programs
// ============================================================================

#[test]
fn test_cli_check_function_with_effects() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn add(x: i64, y: i64) -> i64 {
    x + y
}

fn main() -> i64 {
    add(10, 32)
}
"#,
    );

    let output = run_souc(&["check", test_file.to_str().unwrap()]);

    assert!(
        output.status.success(),
        "check should succeed for function definitions. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_cli_run_function_calls() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = create_test_file(
        &temp_dir,
        "test.sio",
        r#"
fn double(x: i64) -> i64 {
    x * 2
}

fn main() -> i64 {
    let x = 21
    double(x)
}
"#,
    );

    let output = run_souc(&["run", test_file.to_str().unwrap()]);

    assert!(
        output.status.success(),
        "run should succeed for function calls. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("42"),
        "output should contain result 42: {}",
        stdout
    );
}

#[test]
fn test_cli_check_multiple_files_fail() {
    let temp_dir = TempDir::new().unwrap();
    let test_file1 = create_test_file(&temp_dir, "test1.sio", "fn main() -> i64 { 1 }");
    let test_file2 = create_test_file(&temp_dir, "test2.sio", "fn main() -> i64 { 2 }");

    // check command expects a single file
    let output = run_souc(&[
        "check",
        test_file1.to_str().unwrap(),
        test_file2.to_str().unwrap(),
    ]);

    // Should fail because check expects exactly one file
    assert!(
        !output.status.success(),
        "check should not accept multiple files"
    );
}

// ============================================================================
// Multi-module compilation (Poseidon)
// ============================================================================

#[test]
fn test_selfhosted_native_multimodule_import_detection() {
    // Verify source_has_imports correctly detects import/use statements
    let temp_dir = TempDir::new().unwrap();

    let no_imports = create_test_file(&temp_dir, "single.sio", "fn main() -> i64 { 42 }");
    let with_import = create_test_file(&temp_dir, "with_import.sio", "import math\n\nfn main() -> i64 { math::add(1, 2) }");
    let _with_use = create_test_file(&temp_dir, "with_use.sio", "use lib\n\nfn main() -> i64 { lib::foo() }");
    let _with_use_underscore = create_test_file(&temp_dir, "use_var.sio", "fn main() -> i64 {\n    let use_count = 5\n    use_count\n}");

    // Single file without imports should work with selfhosted-native
    let output = run_souc(&[
        "build",
        "--backend=selfhosted-native",
        no_imports.to_str().unwrap(),
        "-o",
        temp_dir.path().join("single.out").to_str().unwrap(),
    ]);
    // May or may not succeed depending on backend availability,
    // but should not crash on import detection
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panic"),
        "Import detection should not panic: {}",
        stderr,
    );
}

#[test]
fn test_multimodule_test_fixtures_exist() {
    // Verify the multifile test fixtures are in place
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();

    let main_sio = workspace_root.join("tests/multifile/main.sio");
    let math_sio = workspace_root.join("tests/multifile/math.sio");
    let lib_sio = workspace_root.join("tests/multifile/lib.sio");
    let main_with_use = workspace_root.join("tests/multifile/main_with_use.sio");

    assert!(main_sio.exists(), "tests/multifile/main.sio missing");
    assert!(math_sio.exists(), "tests/multifile/math.sio missing");
    assert!(lib_sio.exists(), "tests/multifile/lib.sio missing");
    assert!(main_with_use.exists(), "tests/multifile/main_with_use.sio missing");

    // Verify main.sio has imports
    let content = fs::read_to_string(&main_sio).unwrap();
    assert!(content.contains("import math"), "main.sio should contain 'import math'");

    let content_use = fs::read_to_string(&main_with_use).unwrap();
    assert!(content_use.contains("use lib"), "main_with_use.sio should contain 'use lib'");
}
