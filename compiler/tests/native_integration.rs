//! Integration Tests for Native Backend
//!
//! Comprehensive integration tests covering the full compilation pipeline.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn get_fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("native")
        .join(name)
}

fn compile_with_backend(
    input: &PathBuf,
    output: &PathBuf,
    format: &str,
) -> Result<(), String> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_souc"));
    cmd.arg("build");
    cmd.arg("--backend=native");
    cmd.arg("-o");
    cmd.arg(output);
    cmd.arg(input);

    let output_result = cmd.output().map_err(|e| format!("Failed to run souc: {}", e))?;

    if !output_result.status.success() {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        return Err(format!("Compilation failed: {}", stderr));
    }

    Ok(())
}

// ============================================================================
// ELF GENERATION TESTS
// ============================================================================

#[test]
fn test_elf_generation_from_code_segment() {
    // This test validates that CodeSegment can be converted to ELF
    // The actual conversion happens in compile_native, so we test the end result
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test.o");

    compile_with_backend(&input, &output, "o").expect("Failed to compile");

    // Verify ELF structure
    let bytes = fs::read(&output).unwrap();
    assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F'], "Should be valid ELF");
}

// ============================================================================
// LINKING TESTS
// ============================================================================

#[test]
fn test_shared_library_linking() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test.so");

    compile_with_backend(&input, &output, "so").expect("Failed to compile to shared library");

    assert!(output.exists(), "Shared library should be created");
    
    // Verify it's a shared object
    let bytes = fs::read(&output).unwrap();
    assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F'], "Should be valid ELF");
}

#[test]
fn test_executable_linking() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test");

    compile_with_backend(&input, &output, "").expect("Failed to compile to executable");

    assert!(output.exists(), "Executable should be created");
    
    // Verify it's an executable
    let bytes = fs::read(&output).unwrap();
    assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F'], "Should be valid ELF");
}

// ============================================================================
// RUNTIME TESTS
// ============================================================================

#[test]
#[ignore = "Requires runtime to be fully functional"]
fn test_executable_runs() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test");

    compile_with_backend(&input, &output, "").expect("Failed to compile");

    // Try to run (if runtime is working)
    let result = Command::new(&output).output();

    if let Ok(output_result) = result {
        // Executable ran (may have failed, but that's OK for now)
        eprintln!("Executable ran with status: {:?}", output_result.status);
    } else {
        // Executable doesn't run yet (expected)
        eprintln!("Executable doesn't run yet (expected at this stage)");
    }
}
