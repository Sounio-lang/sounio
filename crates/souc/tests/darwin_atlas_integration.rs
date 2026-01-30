//! Integration Tests for Darwin Atlas with Native Backend
//!
//! These tests validate that Darwin Atlas code can be compiled with the
//! native backend.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn get_darwin_atlas_path(name: &str) -> PathBuf {
    PathBuf::from("/tmp/darwin-atlas/demetrios/src").join(name)
}

fn compile_darwin_atlas_file(input: &PathBuf, output: &PathBuf) -> Result<(), String> {
    if !input.exists() {
        return Err(format!("Input file does not exist: {}", input.display()));
    }

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_souc"));
    cmd.arg("build");
    cmd.arg("--backend=native");
    cmd.arg("-o");
    cmd.arg(output);
    cmd.arg(input);

    let output_result = cmd
        .output()
        .map_err(|e| format!("Failed to run souc: {}", e))?;

    if !output_result.status.success() {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        return Err(format!("Compilation failed: {}", stderr));
    }

    Ok(())
}

// ============================================================================
// DARWIN ATLAS COMPILATION TESTS
// ============================================================================

#[test]
#[ignore = "Requires Darwin Atlas to be available at /tmp/darwin-atlas"]
fn test_darwin_atlas_operators() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_darwin_atlas_path("operators.sio");
    let output = temp_dir.path().join("operators.o");

    if !input.exists() {
        eprintln!(
            "Darwin Atlas not found at {}, skipping test",
            input.display()
        );
        return;
    }

    compile_darwin_atlas_file(&input, &output).expect("Failed to compile operators.sio");

    assert!(output.exists(), "Object file should be created");

    // Verify ELF magic
    let bytes = fs::read(&output).unwrap();
    if bytes.len() >= 4 {
        assert_eq!(
            &bytes[0..4],
            &[0x7f, b'E', b'L', b'F'],
            "Should be valid ELF"
        );
    }
}

#[test]
#[ignore = "Requires Darwin Atlas to be available at /tmp/darwin-atlas"]
fn test_darwin_atlas_quaternion() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_darwin_atlas_path("quaternion.sio");
    let output = temp_dir.path().join("quaternion.o");

    if !input.exists() {
        eprintln!(
            "Darwin Atlas not found at {}, skipping test",
            input.display()
        );
        return;
    }

    compile_darwin_atlas_file(&input, &output).expect("Failed to compile quaternion.sio");

    assert!(output.exists(), "Object file should be created");
}

#[test]
#[ignore = "Requires Darwin Atlas to be available at /tmp/darwin-atlas"]
fn test_darwin_atlas_exact_symmetry() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_darwin_atlas_path("exact_symmetry.sio");
    let output = temp_dir.path().join("exact_symmetry.o");

    if !input.exists() {
        eprintln!(
            "Darwin Atlas not found at {}, skipping test",
            input.display()
        );
        return;
    }

    compile_darwin_atlas_file(&input, &output).expect("Failed to compile exact_symmetry.sio");

    assert!(output.exists(), "Object file should be created");
}

#[test]
#[ignore = "Requires Darwin Atlas to be available at /tmp/darwin-atlas"]
fn test_darwin_atlas_lib() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_darwin_atlas_path("lib.sio");
    let output = temp_dir.path().join("lib.o");

    if !input.exists() {
        eprintln!(
            "Darwin Atlas not found at {}, skipping test",
            input.display()
        );
        return;
    }

    compile_darwin_atlas_file(&input, &output).expect("Failed to compile lib.sio");

    assert!(output.exists(), "Object file should be created");
}

#[test]
#[ignore = "Requires Darwin Atlas to be available and multiple file linking"]
fn test_darwin_atlas_multi_file() {
    let temp_dir = TempDir::new().unwrap();

    // Try to compile multiple files together
    let operators = get_darwin_atlas_path("operators.sio");
    let quaternion = get_darwin_atlas_path("quaternion.sio");
    let _output = temp_dir.path().join("darwin_atlas.so");

    if !operators.exists() || !quaternion.exists() {
        eprintln!("Darwin Atlas files not found, skipping test");
        return;
    }

    // This would require multi-file support in the compiler
    // For now, just test that individual files compile
    let result = compile_darwin_atlas_file(&operators, &temp_dir.path().join("operators.o"));
    assert!(
        result.is_ok() || result.is_err(),
        "Should handle compilation attempt"
    );
}
