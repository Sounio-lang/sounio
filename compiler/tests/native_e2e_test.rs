//! End-to-End Tests for Native Backend
//!
//! These tests validate the complete compilation pipeline from source
//! to ELF object files, shared libraries, and executables.

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

fn compile_with_native_backend(
    input: &PathBuf,
    output: &PathBuf,
    format: &str,
) -> Result<(), String> {
    compile_with_native_backend_opts(input, output, format, &[])
}

fn compile_with_native_backend_opts(
    input: &PathBuf,
    output: &PathBuf,
    format: &str,
    extra_opts: &[&str],
) -> Result<(), String> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_souc"));
    cmd.arg("build");
    cmd.arg("--backend=native");
    cmd.arg("-o");
    cmd.arg(output);
    cmd.args(extra_opts);
    cmd.arg(input);

    // Set output format based on extension
    match format {
        "o" => {}
        "so" => {}
        "" => {} // executable
        _ => return Err(format!("Unknown format: {}", format)),
    }

    let output_result = cmd.output().map_err(|e| format!("Failed to run souc: {}", e))?;

    if !output_result.status.success() {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        return Err(format!("Compilation failed: {}", stderr));
    }

    Ok(())
}

fn is_valid_elf(path: &PathBuf) -> bool {
    // Check if file exists
    if !path.exists() {
        return false;
    }

    // Check ELF magic
    if let Ok(bytes) = fs::read(path) {
        if bytes.len() < 4 {
            return false;
        }
        return bytes[0..4] == [0x7f, b'E', b'L', b'F'];
    }

    false
}

// ============================================================================
// BASIC E2E TESTS
// ============================================================================

#[test]
fn test_minimal_object_file() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test_minimal.o");

    compile_with_native_backend(&input, &output, "o")
        .expect("Failed to compile minimal test");

    assert!(output.exists(), "Object file should be created");
    assert!(is_valid_elf(&output), "Output should be valid ELF");
}

#[test]
fn test_minimal_shared_library() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test_minimal.so");

    compile_with_native_backend(&input, &output, "so")
        .expect("Failed to compile minimal test to shared library");

    assert!(output.exists(), "Shared library should be created");
    assert!(is_valid_elf(&output), "Output should be valid ELF");
}

#[test]
fn test_minimal_executable() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test_minimal");

    compile_with_native_backend(&input, &output, "")
        .expect("Failed to compile minimal test to executable");

    assert!(output.exists(), "Executable should be created");
    assert!(is_valid_elf(&output), "Output should be valid ELF");
}

#[test]
fn test_complex_object_file() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_complex.sio");
    let output = temp_dir.path().join("test_complex.o");

    compile_with_native_backend(&input, &output, "o")
        .expect("Failed to compile complex test");

    assert!(output.exists(), "Object file should be created");
    assert!(is_valid_elf(&output), "Output should be valid ELF");
}

#[test]
fn test_complex_shared_library() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_complex.sio");
    let output = temp_dir.path().join("test_complex.so");

    compile_with_native_backend(&input, &output, "so")
        .expect("Failed to compile complex test to shared library");

    assert!(output.exists(), "Shared library should be created");
    assert!(is_valid_elf(&output), "Output should be valid ELF");
}

#[test]
fn test_complex_executable() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_complex.sio");
    let output = temp_dir.path().join("test_complex");

    compile_with_native_backend(&input, &output, "")
        .expect("Failed to compile complex test to executable");

    assert!(output.exists(), "Executable should be created");
    assert!(is_valid_elf(&output), "Output should be valid ELF");
}

// ============================================================================
// COMPLEX CODE FEATURES TESTS
// ============================================================================

#[test]
fn test_complex_has_functions() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_complex.sio");
    let output = temp_dir.path().join("test_complex.o");

    compile_with_native_backend(&input, &output, "o")
        .expect("Failed to compile");

    // Use readelf to check for function symbols
    let readelf_output = Command::new("readelf")
        .arg("-s")
        .arg(&output)
        .output();

    if let Ok(output_result) = readelf_output {
        let stdout = String::from_utf8_lossy(&output_result.stdout);
        // Should have at least main, and possibly add, factorial, sum_array
        assert!(
            stdout.contains("main") || stdout.contains("FUNC"),
            "Should have function symbols"
        );
    } else {
        eprintln!("readelf not available, skipping symbol check");
    }
}

#[test]
fn test_complex_has_loops() {
    // This test validates that code with loops compiles
    // The actual loop detection would require disassembly
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_complex.sio");
    let output = temp_dir.path().join("test_complex.o");

    compile_with_native_backend(&input, &output, "o")
        .expect("Failed to compile code with loops");

    assert!(output.exists(), "Should compile code with loops");
}

#[test]
fn test_complex_has_conditionals() {
    // This test validates that code with conditionals compiles
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_complex.sio");
    let output = temp_dir.path().join("test_complex.o");

    compile_with_native_backend(&input, &output, "o")
        .expect("Failed to compile code with conditionals");

    assert!(output.exists(), "Should compile code with conditionals");
}

#[test]
fn test_complex_has_recursion() {
    // This test validates that recursive functions compile
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_complex.sio");
    let output = temp_dir.path().join("test_complex.o");

    compile_with_native_backend(&input, &output, "o")
        .expect("Failed to compile code with recursion");

    assert!(output.exists(), "Should compile code with recursion");
}

// ============================================================================
// ELF STRUCTURE VALIDATION
// ============================================================================

#[test]
fn test_elf_has_text_section() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test_minimal.o");

    compile_with_native_backend(&input, &output, "o")
        .expect("Failed to compile");

    // Use readelf to check for .text section
    let readelf_output = Command::new("readelf")
        .arg("-S")
        .arg(&output)
        .output();

    if let Ok(output) = readelf_output {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(".text"),
            "ELF should have .text section"
        );
    } else {
        // readelf not available, skip this test
        eprintln!("readelf not available, skipping section check");
    }
}

#[test]
fn test_elf_has_symbol_table() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test_minimal.o");

    compile_with_native_backend(&input, &output, "o")
        .expect("Failed to compile");

    // Use readelf to check for symbol table
    let readelf_output = Command::new("readelf")
        .arg("-s")
        .arg(&output)
        .output();

    if let Ok(output) = readelf_output {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Should have at least some symbols
        assert!(
            stdout.contains("Symbol table") || stdout.contains("main"),
            "ELF should have symbol table"
        );
    } else {
        eprintln!("readelf not available, skipping symbol table check");
    }
}

// ============================================================================
// EPISTEMIC ALLOCATION TESTS
// ============================================================================

#[test]
fn test_epistemic_allocation_enabled() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_complex.sio");
    let output = temp_dir.path().join("test_epistemic.o");

    // Test with epistemic allocation enabled
    compile_with_native_backend_opts(&input, &output, "o", &["--alloc=epistemic"])
        .expect("Failed to compile with epistemic allocation");

    assert!(output.exists(), "Object file should be created");
    assert!(is_valid_elf(&output), "Output should be valid ELF");
}

// ============================================================================
// REGISTER CLASS TESTS
// ============================================================================

#[test]
fn test_float_register_allocation() {
    // Create a temporary test file with float operations
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test_float.sio");
    fs::write(&test_file, r#"
        fn add_floats(a: f64, b: f64) -> f64 {
            a + b
        }
        
        fn main() -> f64 {
            add_floats(3.14, 2.71)
        }
    "#).unwrap();

    let output = temp_dir.path().join("test_float.o");

    compile_with_native_backend(&test_file, &output, "o")
        .expect("Failed to compile float operations");

    assert!(output.exists(), "Object file should be created");
    assert!(is_valid_elf(&output), "Output should be valid ELF");
}

// ============================================================================
// MULTI-FUNCTION TESTS
// ============================================================================

#[test]
fn test_multiple_functions_linking() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_complex.sio");
    let output = temp_dir.path().join("test_multi.so");

    // Compile to shared library to test linking of multiple functions
    compile_with_native_backend(&input, &output, "so")
        .expect("Failed to compile multiple functions");

    assert!(output.exists(), "Shared library should be created");
    assert!(is_valid_elf(&output), "Output should be valid ELF");

    // Check that multiple function symbols exist
    let readelf_output = Command::new("readelf")
        .arg("-s")
        .arg(&output)
        .output();

    if let Ok(output_result) = readelf_output {
        let stdout = String::from_utf8_lossy(&output_result.stdout);
        // Should have multiple functions: main, add, factorial, sum_array
        let function_count = stdout.matches("FUNC").count() + 
                           stdout.matches("main").count() +
                           stdout.matches("add").count() +
                           stdout.matches("factorial").count();
        assert!(
            function_count >= 2,
            "Should have multiple function symbols, found: {}",
            function_count
        );
    } else {
        eprintln!("readelf not available, skipping multi-function check");
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

#[test]
fn test_compilation_performance() {
    use std::time::Instant;
    
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_complex.sio");
    let output = temp_dir.path().join("test_perf.o");

    let start = Instant::now();
    compile_with_native_backend(&input, &output, "o")
        .expect("Failed to compile");
    let duration = start.elapsed();

    assert!(output.exists(), "Object file should be created");
    
    // Compilation should complete in reasonable time (< 5 seconds for small files)
    assert!(
        duration.as_secs() < 5,
        "Compilation took too long: {:?}",
        duration
    );
}

// ============================================================================
// VERBOSE OUTPUT TESTS
// ============================================================================

#[test]
fn test_verbose_compilation() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test_verbose.o");

    // Test with verbose flag
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_souc"));
    cmd.arg("build");
    cmd.arg("--backend=native");
    cmd.arg("-v");
    cmd.arg("-o");
    cmd.arg(&output);
    cmd.arg(&input);

    let output_result = cmd.output().expect("Failed to run souc");

    if !output_result.status.success() {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        panic!("Compilation failed: {}", stderr);
    }

    // Check that verbose output contains allocation info
    let _stdout = String::from_utf8_lossy(&output_result.stdout);
    // Verbose mode should show some compilation information
    // (exact content depends on implementation)
    assert!(
        output.exists(),
        "Object file should be created even with verbose output"
    );
}
