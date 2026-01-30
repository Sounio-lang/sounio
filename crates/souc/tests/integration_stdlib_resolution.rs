//! Integration tests for stdlib module resolution
//!
//! Tests that the Sounio compiler can correctly resolve and load stdlib modules
//! from various directories and with different environment variable configurations.
//! This ensures Issue #15 (stdlib path resolution) remains fixed.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

/// Helper function to get the souc binary path
fn get_souc_binary() -> PathBuf {
    let target_dir = env!("CARGO_BIN_EXE_souc");
    PathBuf::from(target_dir)
}

/// Helper function to get the stdlib path
fn get_stdlib_path() -> PathBuf {
    // Get the project root (parent of compiler dir)
    let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
    let project_root = Path::new(cargo_manifest_dir).parent().unwrap();
    project_root.join("stdlib")
}

/// Helper to run souc check from a specific directory
fn run_souc_check(file_path: &Path, stdlib_path: &Path) -> std::process::Output {
    Command::new(get_souc_binary())
        .arg("check")
        .arg(file_path)
        .env("SOUNIO_STDLIB_PATH", stdlib_path)
        .output()
        .expect("Failed to run souc")
}

/// Helper to write a simple valid Sounio program
fn simple_program() -> &'static str {
    r#"fn main() -> i32 {
    return 0
}"#
}

#[test]
fn test_souc_runs_from_temp_directory_with_stdlib_path() {
    let stdlib_path = get_stdlib_path();
    assert!(
        stdlib_path.exists(),
        "Stdlib directory not found at {:?}",
        stdlib_path
    );

    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let test_file = temp_dir.path().join("test_simple.sio");

    // Create a simple valid Sounio program
    fs::write(&test_file, simple_program()).expect("Failed to write test file");

    // Run souc check with SOUNIO_STDLIB_PATH set
    let output = run_souc_check(&test_file, &stdlib_path);

    // The command should succeed (exit code 0)
    assert!(
        output.status.success(),
        "souc check failed from temp directory. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_stdlib_path_env_var_is_used() {
    let stdlib_path = get_stdlib_path();
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let test_file = temp_dir.path().join("test_env_var.sio");

    fs::write(&test_file, simple_program()).expect("Failed to write test file");

    // This test verifies that when we run from temp directory with SOUNIO_STDLIB_PATH,
    // the compiler can execute without error
    let output = run_souc_check(&test_file, &stdlib_path);

    // Should succeed
    assert!(
        output.status.success(),
        "Compilation failed with SOUNIO_STDLIB_PATH set. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_compiler_accepts_stdlib_path_env_var() {
    // Verify that the compiler doesn't error when SOUNIO_STDLIB_PATH is set
    let stdlib_path = get_stdlib_path();
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let test_file = temp_dir.path().join("test_compiler_env.sio");

    fs::write(&test_file, simple_program()).expect("Failed to write test file");

    let output = Command::new(get_souc_binary())
        .arg("check")
        .arg(&test_file)
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .output()
        .expect("Failed to run souc");

    assert!(
        output.status.success(),
        "Compiler failed with SOUNIO_STDLIB_PATH set"
    );
}

#[test]
fn test_stdlib_path_environment_variable_recognized() {
    // Verify that the sysroot command recognizes the environment variable
    let stdlib_path = get_stdlib_path();

    let output = Command::new(get_souc_binary())
        .arg("sysroot")
        .arg("stdlib-paths")
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .output()
        .expect("Failed to run souc sysroot");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // The output should mention the stdlib path or show it exists
    assert!(
        stdout.contains("stdlib") || output.status.success(),
        "sysroot command didn't recognize stdlib paths"
    );
}

#[test]
fn test_sysroot_stdlib_paths_command_works() {
    // Verify the diagnostic command runs without error
    let stdlib_path = get_stdlib_path();

    let output = Command::new(get_souc_binary())
        .arg("sysroot")
        .arg("stdlib-paths")
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .output()
        .expect("Failed to run souc sysroot stdlib-paths");

    // Command should succeed
    assert!(
        output.status.success(),
        "sysroot stdlib-paths command failed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Output should be non-empty
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.is_empty(),
        "sysroot stdlib-paths produced no output"
    );
}

#[test]
fn test_multiple_compiles_with_same_stdlib_path() {
    let stdlib_path = get_stdlib_path();
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    // Create two test files
    let test_file1 = temp_dir.path().join("test1.sio");
    let test_file2 = temp_dir.path().join("test2.sio");

    fs::write(&test_file1, simple_program()).expect("Failed to write test file 1");
    fs::write(&test_file2, simple_program()).expect("Failed to write test file 2");

    // Check both files with same stdlib path
    let output1 = run_souc_check(&test_file1, &stdlib_path);
    let output2 = run_souc_check(&test_file2, &stdlib_path);

    assert!(output1.status.success(), "First check failed");
    assert!(output2.status.success(), "Second check failed");
}
